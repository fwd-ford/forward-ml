"""Churn classifier — predicts churn_probability per VIN.

This is the core of the per-VIN score consumed by the Java backend.
Compares Logistic Regression, Random Forest, and XGBoost; selects the
model with the best PR-AUC (more informative than ROC-AUC for the
moderately imbalanced case) and persists the winner.

Metrics computed:
  - ROC-AUC, PR-AUC, Brier score (calibration), F1, precision, recall
  - Recall@top-k (operational: how many true churners caught if we
    contact the top K% predicted)
  - Per-model breakdown (avoids Ranger-detector pitfall)
  - Reliability/calibration check
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .features import CLASSIFIER_FEATURES, REPO_ROOT, model_inverse_weights

RANDOM_STATE = 42
TEST_SIZE = 0.2
TOP_K_FRACTIONS = (0.05, 0.10, 0.20, 0.30)
TEMPORAL_CUTOFF_QUANTILE = 0.8
SHAP_SAMPLE_SIZE = 5_000

# Leak-safe feature subset.
# Excludes any feature that involves `last_service_date` directly (the second half of the target,
# since `churned = (reference_date - last_service_date) > 365`) and any feature imputed
# with `days_since_last_service` (gap_* for single-event VINs in `_impute_for_modeling`).
LEAK_SAFE_FEATURES = [
    "events_count",
    "tenure_days",
    "km_max",
    "km_per_month",
    "dealers_distinct",
    "primary_dealer_share",
    "service_codes_distinct",
    "service_types_distinct",
    "is_loyal_to_dealer",
    "is_single_event",
    "model_year",
    "model_name_freq",
]

# Stricter still: excludes `tenure_days` (which uses last_service_date even if normalized by sales_date).
LEAK_STRICT_FEATURES = [
    "events_count",
    "dealers_distinct",
    "primary_dealer_share",
    "service_codes_distinct",
    "service_types_distinct",
    "is_loyal_to_dealer",
    "is_single_event",
    "model_year",
    "model_name_freq",
]


@dataclass
class ClassifierArtifacts:
    model_name: str
    model: object
    feature_order: list[str]
    metrics: dict
    threshold: float


def _make_models() -> dict[str, object]:
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            device="cuda",
        ),
    }


def recall_at_top_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float) -> float:
    n_top = max(1, int(len(y_score) * k_frac))
    top_idx = np.argsort(-y_score)[:n_top]
    return float(y_true[top_idx].sum() / max(1, y_true.sum()))


def precision_at_top_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float) -> float:
    n_top = max(1, int(len(y_score) * k_frac))
    top_idx = np.argsort(-y_score)[:n_top]
    return float(y_true[top_idx].mean())


def find_threshold_max_f1(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Return (threshold, f1) that maximizes F1 on the precision-recall curve."""
    p, r, t = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns t of length n-1; align by trimming
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-9, None)
    best_idx = int(np.argmax(f1))
    return float(t[best_idx]), float(f1[best_idx])


def temporal_split(
    df_trainable: pd.DataFrame,
    cutoff_quantile: float = TEMPORAL_CUTOFF_QUANTILE,
    cutoff_column: str = "sales_date",
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series, pd.Timestamp]:
    """Split by purchase cohort: train on older customers, test on newer ones.

    Default partition is `sales_date` (when the VIN entered the official network),
    NOT `last_service_date` — because the target `churned = days_since_last_service > 365`
    is degenerate when split by recency (recent VINs are never churned by definition).

    Returns 9-tuple: (X_train, X_test, y_train, y_test, w_train, w_test, g_train, g_test, cutoff_ts).
    """
    if cutoff_column not in df_trainable.columns:
        raise ValueError(f"temporal_split requires '{cutoff_column}' in df_trainable.")

    df = df_trainable.dropna(subset=[cutoff_column])
    if len(df) < len(df_trainable):
        # carry rows with missing cutoff_column into train as a fallback (no leakage — they predate the cutoff conceptually)
        df_missing = df_trainable[df_trainable[cutoff_column].isna()]
    else:
        df_missing = df_trainable.iloc[0:0]

    cutoff = df[cutoff_column].quantile(cutoff_quantile)
    is_train_main = df[cutoff_column] <= cutoff
    train_df = pd.concat([df.loc[is_train_main], df_missing])
    test_df = df.loc[~is_train_main]

    feats = features if features is not None else CLASSIFIER_FEATURES
    X_train = train_df[feats].copy()
    X_test = test_df[feats].copy()
    y_train = train_df["churned"].to_numpy()
    y_test = test_df["churned"].to_numpy()
    w_train = model_inverse_weights(train_df["model_name"])
    w_test = model_inverse_weights(test_df["model_name"])
    g_train = train_df["model_name"].reset_index(drop=True)
    g_test = test_df["model_name"].reset_index(drop=True)

    return X_train, X_test, y_train, y_test, w_train, w_test, g_train, g_test, pd.Timestamp(cutoff)


def temporal_val_split(
    df_trainable: pd.DataFrame,
    test_quantile: float = 0.80,
    val_quantile: float = 0.64,
    cutoff_column: str = "sales_date",
    features: list[str] | None = None,
) -> dict:
    """Three-way temporal split — train (oldest) / val (middle) / test (newest).

    Sorts by `cutoff_column` (default `sales_date`) and slices:
      - train: rows with cutoff_column <= q(val_quantile)
      - val:   rows with q(val_quantile) < cutoff_column <= q(test_quantile)
      - test:  rows with cutoff_column > q(test_quantile)

    `val` sits between train and test in time — early stopping driven by val
    reflects the kind of drift the model will see when scoring fresh cohorts.

    Returns dict with all relevant arrays plus the cutoff timestamps for audit.
    """
    if cutoff_column not in df_trainable.columns:
        raise ValueError(f"temporal_val_split requires '{cutoff_column}' in df_trainable.")
    if not (0.0 < val_quantile < test_quantile < 1.0):
        raise ValueError("Need 0 < val_quantile < test_quantile < 1.")

    feats = features if features is not None else CLASSIFIER_FEATURES
    df = df_trainable.dropna(subset=[cutoff_column]).copy()

    q_val = df[cutoff_column].quantile(val_quantile)
    q_test = df[cutoff_column].quantile(test_quantile)

    train_mask = df[cutoff_column] <= q_val
    val_mask = (df[cutoff_column] > q_val) & (df[cutoff_column] <= q_test)
    test_mask = df[cutoff_column] > q_test

    out = {}
    for tag, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        sub = df.loc[mask]
        out[f"X_{tag}"] = sub[feats].copy()
        out[f"y_{tag}"] = sub["churned"].to_numpy()
        out[f"w_{tag}"] = model_inverse_weights(sub["model_name"])
        out[f"g_{tag}"] = sub["model_name"].reset_index(drop=True)
    out["cutoff_val"] = pd.Timestamp(q_val)
    out["cutoff_test"] = pd.Timestamp(q_test)
    out["features"] = feats
    return out


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_series: pd.Series | None = None,
) -> dict:
    """Compute the full metric set on a single test fold."""
    threshold, f1_at_best = find_threshold_max_f1(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "best_threshold": threshold,
        "f1_at_best": float(f1_at_best),
        "precision_at_best": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_best": float(recall_score(y_true, y_pred, zero_division=0)),
        "default_threshold_0_5": {
            "precision": float(precision_score(y_true, (y_score >= 0.5).astype(int), zero_division=0)),
            "recall": float(recall_score(y_true, (y_score >= 0.5).astype(int), zero_division=0)),
            "f1": float(f1_score(y_true, (y_score >= 0.5).astype(int), zero_division=0)),
        },
        "confusion_at_best": confusion_matrix(y_true, y_pred).tolist(),
        "recall_at_top_k": {
            f"top_{int(k*100)}pct": recall_at_top_k(y_true, y_score, k) for k in TOP_K_FRACTIONS
        },
        "precision_at_top_k": {
            f"top_{int(k*100)}pct": precision_at_top_k(y_true, y_score, k) for k in TOP_K_FRACTIONS
        },
    }
    if group_series is not None:
        per_group = {}
        for g, idx in group_series.groupby(group_series).groups.items():
            mask = np.zeros(len(y_true), dtype=bool)
            # group_series index is positional after reset; map back via iloc
            pos = group_series.index.get_indexer(idx)
            mask[pos] = True
            if mask.sum() < 50:
                continue
            yt = y_true[mask]
            ys = y_score[mask]
            if len(np.unique(yt)) < 2:
                continue
            per_group[str(g)] = {
                "n": int(mask.sum()),
                "positive_rate": float(yt.mean()),
                "roc_auc": float(roc_auc_score(yt, ys)),
                "pr_auc": float(average_precision_score(yt, ys)),
            }
        metrics["per_group"] = per_group
    return metrics


def fit_and_compare(
    df_trainable: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    use_temporal_split: bool = False,
    cutoff_quantile: float = TEMPORAL_CUTOFF_QUANTILE,
    features: list[str] | None = None,
) -> tuple[dict[str, dict], pd.DataFrame, dict[str, object]]:
    """Train each candidate, evaluate on hold-out, return all metrics.

    When use_temporal_split=True, train/test is split by sales_date quantile
    (older cohorts train, newer cohorts test) — more realistic but typically harder.

    `features` overrides CLASSIFIER_FEATURES (used by the leak audit).
    """
    feats = features if features is not None else CLASSIFIER_FEATURES
    if use_temporal_split:
        (
            X_train, X_test, y_train, y_test, w_train, w_test, g_train, g_test, cutoff_ts,
        ) = temporal_split(df_trainable, cutoff_quantile=cutoff_quantile, features=feats)
        split_info = {
            "strategy": "temporal",
            "cutoff_quantile": cutoff_quantile,
            "cutoff_date": str(cutoff_ts.date()),
        }
    else:
        X = df_trainable[feats].copy()
        y = df_trainable["churned"].to_numpy()
        model_groups = df_trainable["model_name"].reset_index(drop=True)
        sample_w = model_inverse_weights(df_trainable["model_name"])

        X_train, X_test, y_train, y_test, w_train, w_test, g_train, g_test = train_test_split(
            X, y, sample_w, model_groups,
            test_size=test_size, random_state=random_state, stratify=y,
        )
        split_info = {
            "strategy": "stratified_random",
            "test_size": test_size,
            "random_state": random_state,
        }
    split_info["n_features"] = len(feats)

    all_models = _make_models()
    fitted: dict[str, object] = {}
    all_metrics: dict[str, dict] = {}

    for name, mdl in all_models.items():
        if name == "xgboost":
            mdl.fit(X_train, y_train, sample_weight=w_train)
        elif name == "random_forest":
            mdl.fit(X_train, y_train, sample_weight=w_train)
        else:
            mdl.fit(X_train, y_train, lr__sample_weight=w_train)
        y_score = mdl.predict_proba(X_test)[:, 1]
        g_test_reset = g_test.reset_index(drop=True)
        all_metrics[name] = evaluate(y_test, y_score, group_series=g_test_reset)
        all_metrics[name]["n_train"] = int(len(y_train))
        all_metrics[name]["n_test"] = int(len(y_test))
        all_metrics[name]["train_positive_rate"] = float(np.mean(y_train))
        all_metrics[name]["test_positive_rate"] = float(np.mean(y_test))
        all_metrics[name]["split"] = split_info
        fitted[name] = mdl

    summary = pd.DataFrame({
        m: {
            "roc_auc": v["roc_auc"],
            "pr_auc": v["pr_auc"],
            "brier": v["brier"],
            "f1_best": v["f1_at_best"],
            "recall_top10": v["recall_at_top_k"]["top_10pct"],
            "precision_top10": v["precision_at_top_k"]["top_10pct"],
        }
        for m, v in all_metrics.items()
    }).T.round(4)
    return all_metrics, summary, fitted


def fit_and_compare_clean(
    df_trainable: pd.DataFrame,
    feature_subset: list[str] | None = None,
    exclude_single_event: bool = False,
    use_temporal_split: bool = False,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, dict], pd.DataFrame, dict[str, object]]:
    """Leak-audit variant of fit_and_compare.

    feature_subset:        restrict the feature space (e.g., LEAK_SAFE_FEATURES, LEAK_STRICT_FEATURES).
    exclude_single_event:  drop is_single_event==1 VINs (gap_* are imputed with the target for these).
    use_temporal_split:    train on older sales cohorts, test on newer.
    """
    df = df_trainable.copy()
    if exclude_single_event:
        df = df[df["is_single_event"] == 0].copy()
    return fit_and_compare(
        df,
        test_size=test_size,
        random_state=random_state,
        use_temporal_split=use_temporal_split,
        features=feature_subset,
    )


def select_best_model(
    all_metrics: dict[str, dict],
    fitted: dict[str, object],
    primary_metric: str = "pr_auc",
) -> tuple[str, object]:
    """Return (model_name, fitted_model) of the winner by primary_metric (default PR-AUC)."""
    if not all_metrics:
        raise ValueError("all_metrics is empty.")
    ranked = sorted(all_metrics.items(), key=lambda kv: kv[1][primary_metric], reverse=True)
    best_name = ranked[0][0]
    return best_name, fitted[best_name]


def compute_shap(
    model: object,
    X: pd.DataFrame,
    sample_size: int = SHAP_SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP values on a sampled subset using TreeExplainer.

    Returns (shap_values, X_sampled). Designed for tree models (XGBoost, RF).
    For non-tree models (LogReg pipeline), callers should branch before invoking this.
    """
    import shap

    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=random_state)
    else:
        X_sample = X.copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
    return shap_values, X_sample


def cross_validate(df_trainable: pd.DataFrame, model_name: str = "xgboost", n_splits: int = 5) -> pd.DataFrame:
    """Stratified k-fold CV — sanity check that the held-out metric is stable."""
    X = df_trainable[CLASSIFIER_FEATURES]
    y = df_trainable["churned"].to_numpy()
    w = model_inverse_weights(df_trainable["model_name"])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        models = _make_models()
        mdl = models[model_name]
        if model_name in ("xgboost", "random_forest"):
            mdl.fit(X.iloc[tr], y[tr], sample_weight=w[tr])
        else:
            mdl.fit(X.iloc[tr], y[tr], lr__sample_weight=w[tr])
        ys = mdl.predict_proba(X.iloc[te])[:, 1]
        rows.append({
            "fold": fold,
            "roc_auc": roc_auc_score(y[te], ys),
            "pr_auc": average_precision_score(y[te], ys),
            "brier": brier_score_loss(y[te], ys),
        })
    return pd.DataFrame(rows)


def save(artifacts: ClassifierArtifacts, models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": artifacts.model,
            "feature_order": artifacts.feature_order,
            "model_name": artifacts.model_name,
            "threshold": artifacts.threshold,
        },
        models_dir / "churn_classifier.joblib",
    )
    (models_dir / "classifier_metrics.json").write_text(
        json.dumps(artifacts.metrics, indent=2), encoding="utf-8"
    )


def main(use_temporal_split: bool = False) -> ClassifierArtifacts:
    """End-to-end: load features → fit & compare → select winner → persist artifacts."""
    from .features import load_features, split_trainable

    bundle = load_features()
    df_train, _ = split_trainable(bundle.df)

    all_metrics, summary, fitted = fit_and_compare(
        df_train, use_temporal_split=use_temporal_split,
    )
    best_name, best_model = select_best_model(all_metrics, fitted)
    best_threshold = float(all_metrics[best_name]["best_threshold"])

    metrics_payload = {
        "winner": best_name,
        "reference_date": str(bundle.reference_date.date()),
        "n_total": bundle.n_total,
        "n_trainable": bundle.n_trainable,
        "split_strategy": "temporal" if use_temporal_split else "stratified_random",
        "primary_metric": "pr_auc",
        "summary": summary.to_dict(orient="index"),
        "all_models": all_metrics,
    }
    artifacts = ClassifierArtifacts(
        model_name=best_name,
        model=best_model,
        feature_order=CLASSIFIER_FEATURES,
        metrics=metrics_payload,
        threshold=best_threshold,
    )
    save(artifacts, REPO_ROOT / "data" / "models")

    print(summary)
    print(
        f"Winner: {best_name} "
        f"(PR-AUC = {all_metrics[best_name]['pr_auc']:.4f}, "
        f"ROC-AUC = {all_metrics[best_name]['roc_auc']:.4f}, "
        f"threshold = {best_threshold:.3f})"
    )
    return artifacts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and persist the churn classifier.")
    parser.add_argument(
        "--temporal-split",
        action="store_true",
        help="Use temporal split (by last_service_date quantile) instead of stratified random.",
    )
    args = parser.parse_args()
    main(use_temporal_split=args.temporal_split)
