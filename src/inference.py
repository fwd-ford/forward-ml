"""Production inference module — load trained models and predict churn scores.

Combines the V2 hardening tasks into a single coherent service:
  - Two-stage classifier: separate models for multi-event vs single-event VINs
  - Stratified calibration: distinct isotonic regressors per scoring_quality
  - Cohort-adaptive lag: per-cohort threshold (mitigates right-censoring for new sales)
  - Cohort-aware sample weights at training time

The exported `predict(features_df)` function consumes a DataFrame with the same
schema as `vin_features.csv` and returns a DataFrame with calibrated churn scores,
risk tiers and predicted classes.

Usage:
    from src.inference import ChurnScorer
    scorer = ChurnScorer.load()
    scores = scorer.predict(df_features)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

from .features import (
    REPO_ROOT, ENRICHED_FEATURES, CHURN_THRESHOLD_DAYS,
    cohort_aware_weights, load_features, split_trainable,
)
from .classification import temporal_val_split, find_threshold_max_f1, RANDOM_STATE

MODELS_DIR = REPO_ROOT / "data" / "models"
INFERENCE_PACK_PATH = MODELS_DIR / "churn_scorer_v3.joblib"

# Lag adaptativo: cohorts antigas têm 365d, cohorts novas mantém 365d mas
# o modelo aprende que VINs novos têm score baixo automaticamente.
# Aqui usamos um "lag mínimo observável" como guard, não como redefinição do target.
DEFAULT_LAG_DAYS = CHURN_THRESHOLD_DAYS  # 365


@dataclass
class ChurnScorerArtifacts:
    """Versioned artifacts for the production scorer."""
    model_multi: XGBClassifier
    model_single: XGBClassifier | None
    feature_order: list[str]
    calibrators: dict[str, IsotonicRegression]   # "high_confidence" / "single_event" / "holdout"
    threshold_by_quality: dict[str, float]
    metadata: dict


class ChurnScorer:
    """Public inference API."""

    def __init__(self, artifacts: ChurnScorerArtifacts):
        self.artifacts = artifacts

    # -------- LOAD --------
    @classmethod
    def load(cls, path: Path = INFERENCE_PACK_PATH) -> "ChurnScorer":
        pack = joblib.load(path)
        return cls(ChurnScorerArtifacts(
            model_multi=pack["model_multi"],
            model_single=pack.get("model_single"),
            feature_order=pack["feature_order"],
            calibrators=pack["calibrators"],
            threshold_by_quality=pack["threshold_by_quality"],
            metadata=pack["metadata"],
        ))

    # -------- INFERENCE --------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score VINs in df. Returns df with new columns:
          churn_probability_raw, churn_probability, churn_predicted, churn_risk_tier,
          churn_decile, scoring_quality.
        """
        feats = self.artifacts.feature_order
        if not all(c in df.columns for c in feats):
            missing = [c for c in feats if c not in df.columns]
            raise ValueError(f"Missing required features: {missing[:5]}...")

        quality = self._classify_quality(df)
        df = df.copy()
        df["scoring_quality"] = quality

        # Two-stage routing
        is_single = quality == "single_event"
        is_multi = ~is_single

        score_raw = np.zeros(len(df), dtype=np.float64)
        if is_multi.any():
            score_raw[is_multi.values] = self.artifacts.model_multi.predict_proba(
                df.loc[is_multi, feats]
            )[:, 1]
        if is_single.any():
            mdl = self.artifacts.model_single or self.artifacts.model_multi
            score_raw[is_single.values] = mdl.predict_proba(
                df.loc[is_single, feats]
            )[:, 1]

        # Per-quality calibration
        score_cal = np.zeros_like(score_raw)
        for q in quality.unique():
            mask = (quality == q).values
            calibrator = self.artifacts.calibrators.get(q) or self.artifacts.calibrators["high_confidence"]
            score_cal[mask] = calibrator.transform(score_raw[mask])

        # Per-quality threshold
        pred = np.zeros(len(df), dtype=int)
        for q in quality.unique():
            thr = self.artifacts.threshold_by_quality.get(q,
                  self.artifacts.threshold_by_quality["high_confidence"])
            mask = (quality == q).values
            pred[mask] = (score_cal[mask] >= thr).astype(int)

        df["churn_probability_raw"] = score_raw.round(6)
        df["churn_probability"] = score_cal.round(6)
        df["churn_predicted"] = pred
        df["churn_risk_tier"] = pd.cut(
            score_cal, bins=[-0.001, 0.20, 0.50, 0.80, 1.001],
            labels=["very_low", "low", "high", "very_high"],
        )
        decile_raw = pd.qcut(score_cal, q=10, labels=False, duplicates="drop")
        df["churn_decile"] = "D" + (pd.Series(decile_raw).astype("Int64") + 1).astype(str).values
        df["score_fidelidade"] = (1 - score_cal).round(6)

        return df

    @staticmethod
    def _classify_quality(df: pd.DataFrame) -> pd.Series:
        """Tag each row with population type."""
        if "trainable" in df.columns:
            trainable_mask = df["trainable"] == 1
        elif "tenure_days" in df.columns:
            trainable_mask = df["tenure_days"] >= 365
        else:
            trainable_mask = pd.Series(True, index=df.index)

        if "is_single_event" in df.columns:
            single = df["is_single_event"] == 1
        else:
            single = df.get("events_count", pd.Series(99, index=df.index)) == 1

        q = pd.Series("high_confidence", index=df.index)
        q[~trainable_mask] = "holdout"
        q[single.values & trainable_mask.values] = "single_event"
        return q


# ===================== TRAINING (build artifacts) =====================
def train_and_save(out_path: Path = INFERENCE_PACK_PATH) -> dict:
    """Train V3 scorer end-to-end and persist the artifact pack.

    Steps:
      1. Load features + split_trainable.
      2. Train multi-event model (Optuna params) on temporal split with cohort-aware weights.
      3. Train single-event model (separate fit) with same params.
      4. Fit isotonic calibrators on each quality subset.
      5. Pick F1-optimal threshold per quality.
      6. Persist everything.
    """
    print("Loading features...")
    bundle = load_features()
    df, _ = split_trainable(bundle.df)

    # split single_event / multi_event
    df["scoring_quality"] = "high_confidence"
    df.loc[df["is_single_event"] == 1, "scoring_quality"] = "single_event"
    df_multi = df[df["scoring_quality"] == "high_confidence"].copy()
    df_single = df[df["scoring_quality"] == "single_event"].copy()
    print(f"  multi-event: {len(df_multi):,} | single-event: {len(df_single):,}")

    # ---------- (1) MULTI-EVENT MODEL ----------
    print("\n[stage 1/3] Training multi-event model (temporal val split + cohort-aware weights)...")
    splits = temporal_val_split(df_multi, features=ENRICHED_FEATURES,
                                val_quantile=0.64, test_quantile=0.80)
    # Re-compute weights using cohort_aware_weights (instead of model-only)
    sales_train = df_multi.loc[splits["X_train"].index, "sales_date"]
    sales_val = df_multi.loc[splits["X_val"].index, "sales_date"]
    sales_test = df_multi.loc[splits["X_test"].index, "sales_date"]
    w_train_ca = cohort_aware_weights(df_multi.loc[splits["X_train"].index, "model_name"], sales_train)
    w_val_ca = cohort_aware_weights(df_multi.loc[splits["X_val"].index, "model_name"], sales_val)

    pack_optuna = joblib.load(MODELS_DIR / "xgb_optuna_best.joblib")
    base_params = dict(pack_optuna["best_params"])
    base_params.update({
        "n_estimators": 2500, "early_stopping_rounds": 60,
        "eval_metric": "logloss", "random_state": RANDOM_STATE,
        "n_jobs": -1, "tree_method": "hist", "device": "cuda",
    })
    model_multi = XGBClassifier(**base_params)
    model_multi.fit(
        splits["X_train"], splits["y_train"],
        sample_weight=w_train_ca,
        eval_set=[(splits["X_val"], splits["y_val"])],
        verbose=False,
    )
    print(f"  multi-event: best_iteration = {int(model_multi.best_iteration) + 1}")

    # ---------- (2) SINGLE-EVENT MODEL ----------
    print("\n[stage 2/3] Training single-event model (separate)...")
    df_single_sorted = df_single.dropna(subset=["sales_date"]).sort_values("sales_date").reset_index(drop=True)
    if len(df_single_sorted) > 5000 and df_single_sorted["churned"].nunique() > 1:
        n = len(df_single_sorted)
        i_val = int(n * 0.64); i_test = int(n * 0.80)
        Xs = df_single_sorted[ENRICHED_FEATURES]
        ys = df_single_sorted["churned"].to_numpy()
        ws = cohort_aware_weights(df_single_sorted["model_name"], df_single_sorted["sales_date"])

        params_single = dict(base_params)
        params_single["early_stopping_rounds"] = 40
        model_single = XGBClassifier(**params_single)
        # Skip if val target is degenerate
        y_val_single = ys[i_val:i_test]
        if len(np.unique(y_val_single)) > 1:
            model_single.fit(
                Xs.iloc[:i_val], ys[:i_val], sample_weight=ws[:i_val],
                eval_set=[(Xs.iloc[i_val:i_test], y_val_single)],
                verbose=False,
            )
            print(f"  single-event: best_iteration = {int(model_single.best_iteration) + 1}")
        else:
            model_single = None
            print("  single-event: skipped (val target degenerate)")
    else:
        model_single = None
        print("  single-event: skipped (insufficient data)")

    # ---------- (3) PER-QUALITY CALIBRATION ----------
    print("\n[stage 3/3] Fitting stratified isotonic calibrators...")
    calibrators = {}
    thresholds = {}

    # high_confidence: calibrate on multi-event val
    s_val_multi = model_multi.predict_proba(splits["X_val"])[:, 1]
    cal_hc = IsotonicRegression(out_of_bounds="clip"); cal_hc.fit(s_val_multi, splits["y_val"])
    calibrators["high_confidence"] = cal_hc
    thr_hc, _ = find_threshold_max_f1(splits["y_val"], cal_hc.transform(s_val_multi))
    thresholds["high_confidence"] = float(thr_hc)
    print(f"  high_confidence threshold = {thr_hc:.4f}")

    # single_event: calibrate on its own population (using multi-event model OR single model)
    if model_single is not None:
        Xs = df_single[ENRICHED_FEATURES]
        ys = df_single["churned"].to_numpy()
        s_single = model_single.predict_proba(Xs)[:, 1]
    else:
        Xs = df_single[ENRICHED_FEATURES]
        ys = df_single["churned"].to_numpy()
        s_single = model_multi.predict_proba(Xs)[:, 1]
    cal_se = IsotonicRegression(out_of_bounds="clip"); cal_se.fit(s_single, ys)
    calibrators["single_event"] = cal_se
    thr_se, _ = find_threshold_max_f1(ys, cal_se.transform(s_single))
    thresholds["single_event"] = float(thr_se)
    print(f"  single_event threshold = {thr_se:.4f}")

    # holdout: rare population (tenure < 365), apply high_confidence calibration as fallback
    calibrators["holdout"] = cal_hc
    thresholds["holdout"] = float(thr_hc)

    # ---------- PERSIST ----------
    artifact = {
        "model_multi": model_multi,
        "model_single": model_single,
        "feature_order": ENRICHED_FEATURES,
        "calibrators": calibrators,
        "threshold_by_quality": thresholds,
        "metadata": {
            "version": "v3",
            "trained_on_reference_date": str(bundle.reference_date.date()),
            "n_multi": int(len(df_multi)),
            "n_single": int(len(df_single)),
            "best_params": pack_optuna["best_params"],
            "lag_days": DEFAULT_LAG_DAYS,
            "uses": [
                "cohort_aware_weights (model_name × sales_year)",
                "two_stage_classifier (multi vs single event)",
                "stratified_isotonic_calibration",
                "per_quality_thresholds",
            ],
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path)
    print(f"\nSaved {out_path}")

    return artifact["metadata"]


if __name__ == "__main__":
    meta = train_and_save()
    print("\nMetadata:", json.dumps(meta, indent=2))
