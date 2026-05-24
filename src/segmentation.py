"""Blind K-means segmentation on behavioral features.

No ground-truth labels exist for personas (synthetic dataset is superseded).
Validation is done via:
  - Silhouette score on a 10k sample (target >= 0.35 per CLAUDE.md)
  - Elbow / inertia curve across k
  - Cross-tabs against model_name and churn rate (label-free quality signals)
  - Manual cluster naming after inspecting feature means

Persona naming follows the project nomenclature (DOC 00):
  fiel / economico / esquecido / abandono
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .features import KMEANS_FEATURES

RANDOM_STATE = 42
SAMPLE_SIZE_FOR_SILHOUETTE = 10_000


@dataclass
class SegmentationArtifacts:
    scaler: StandardScaler
    model: KMeans
    feature_order: list[str]
    cluster_profiles: pd.DataFrame
    cluster_names: dict[int, str]
    metrics: dict


def _prepare_matrix(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    X = df[KMEANS_FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X)
    return scaler.transform(X), scaler


def sweep_k(df: pd.DataFrame, k_range=range(2, 8), random_state=RANDOM_STATE) -> pd.DataFrame:
    """Grid-search k by silhouette + inertia. Returns one row per k."""
    X_scaled, _ = _prepare_matrix(df)
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(len(X_scaled), size=min(SAMPLE_SIZE_FOR_SILHOUETTE, len(X_scaled)), replace=False)
    X_sample = X_scaled[sample_idx]

    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X_scaled)
        labels_sample = km.predict(X_sample)
        sil = silhouette_score(X_sample, labels_sample)
        rows.append({"k": k, "silhouette": sil, "inertia": km.inertia_})
    return pd.DataFrame(rows)


def _name_clusters(profile: pd.DataFrame) -> dict[int, str]:
    """Map cluster ids to persona names by sorting on churn proxy axes.

    Heuristic (label-free, DOC 00 nomenclature):
      - "abandono"  → high days_since_last_service AND single-event
      - "fiel"      → high events_count, low days_since, high dealer share, MULTI-event
      - "esquecido" → multi-event but drifting (high days_since, moderate gap)
      - "economico" → single-event recent or low-engagement remainder

    Single-event clusters are explicitly excluded from "fiel" because dealer
    share = 1.0 is trivially true with only one visit.
    """
    p = profile.copy()

    p["abandono_score"] = p["days_since_last_service"].rank() + p["is_single_event"].rank()
    abandono = p["abandono_score"].idxmax()

    remaining = p.drop(index=abandono)
    multi_event = remaining[remaining["is_single_event"] < 0.5]
    pool = multi_event if len(multi_event) else remaining
    fiel_score = (
        pool["events_count"].rank()
        + pool["primary_dealer_share"].rank()
        - pool["days_since_last_service"].rank()
    )
    fiel = fiel_score.idxmax()

    rest = remaining.drop(index=fiel)
    if len(rest):
        esquecido_score = rest["days_since_last_service"].rank() + rest["gap_last_days"].rank()
        esquecido = esquecido_score.idxmax()
    else:
        esquecido = None

    final = rest.drop(index=esquecido) if esquecido is not None else rest
    economico = final.index[0] if len(final) else None

    names: dict[int, str] = {int(abandono): "abandono", int(fiel): "fiel"}
    if esquecido is not None:
        names[int(esquecido)] = "esquecido"
    if economico is not None:
        names[int(economico)] = "economico"
    for cid in profile.index:
        names.setdefault(int(cid), f"cluster_{cid}")
    return names


def fit(
    df: pd.DataFrame,
    k: int = 4,
    random_state: int = RANDOM_STATE,
) -> SegmentationArtifacts:
    """Fit K-means + name clusters. Expects df already through load_features()."""
    X_scaled, scaler = _prepare_matrix(df)
    model = KMeans(n_clusters=k, random_state=random_state, n_init=20).fit(X_scaled)

    df_local = df.copy()
    df_local["cluster"] = model.labels_

    profile_cols = KMEANS_FEATURES + ["churned", "model_name"]
    numeric_profile = df_local.groupby("cluster")[KMEANS_FEATURES].mean()
    numeric_profile["size"] = df_local["cluster"].value_counts().sort_index()
    numeric_profile["size_pct"] = numeric_profile["size"] / len(df_local)
    numeric_profile["churn_rate"] = df_local.groupby("cluster")["churned"].mean()

    names = _name_clusters(numeric_profile)
    numeric_profile["name"] = numeric_profile.index.map(names)

    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(len(X_scaled), size=min(SAMPLE_SIZE_FOR_SILHOUETTE, len(X_scaled)), replace=False)
    sil = silhouette_score(X_scaled[sample_idx], model.labels_[sample_idx])

    metrics = {
        "k": k,
        "silhouette_10k_sample": float(sil),
        "inertia": float(model.inertia_),
        "n_samples": int(len(df_local)),
    }

    return SegmentationArtifacts(
        scaler=scaler,
        model=model,
        feature_order=list(KMEANS_FEATURES),
        cluster_profiles=numeric_profile,
        cluster_names={int(k_): str(v) for k_, v in names.items()},
        metrics=metrics,
    )


def predict(df: pd.DataFrame, artifacts: SegmentationArtifacts) -> pd.DataFrame:
    """Return df with cluster_id, cluster_name, and distance_to_centroid."""
    X = df[artifacts.feature_order].to_numpy(dtype=float)
    X_scaled = artifacts.scaler.transform(X)
    labels = artifacts.model.predict(X_scaled)
    distances = artifacts.model.transform(X_scaled)
    own_dist = distances[np.arange(len(labels)), labels]

    out = df.copy()
    out["cluster_id"] = labels
    out["cluster_name"] = pd.Series(labels).map(artifacts.cluster_names).to_numpy()
    out["cluster_distance"] = own_dist
    return out


def save(artifacts: SegmentationArtifacts, models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": artifacts.scaler,
            "model": artifacts.model,
            "feature_order": artifacts.feature_order,
            "cluster_names": artifacts.cluster_names,
        },
        models_dir / "kmeans_segmentation.joblib",
    )
    artifacts.cluster_profiles.to_csv(models_dir / "kmeans_cluster_profiles.csv")
    (models_dir / "kmeans_metrics.json").write_text(
        json.dumps(artifacts.metrics, indent=2), encoding="utf-8"
    )
