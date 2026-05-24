"""Smoke + integration tests for the V3 inference pipeline."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.features import (
    load_features, split_trainable,
    CLASSIFIER_FEATURES, ENRICHED_FEATURES, KMEANS_FEATURES,
    cohort_aware_weights, model_inverse_weights, CHURN_THRESHOLD_DAYS,
)
from src.classification import temporal_val_split, find_threshold_max_f1
from src.inference import ChurnScorer


# ===== Features =====
@pytest.fixture(scope="module")
def bundle():
    return load_features()


def test_load_features_basic(bundle):
    df = bundle.df
    assert len(df) > 150_000, "Expected ~175k VINs"
    assert "churned" in df.columns
    assert "trainable" in df.columns
    assert df["churned"].isin([0, 1]).all()
    assert bundle.reference_date is not None


def test_enriched_features_present(bundle):
    missing = [c for c in ENRICHED_FEATURES if c not in bundle.df.columns]
    assert not missing, f"Missing ENRICHED_FEATURES: {missing}"


def test_no_nan_in_classifier_features(bundle):
    nan_cols = bundle.df[CLASSIFIER_FEATURES].isna().sum()
    bad = nan_cols[nan_cols > 0]
    assert bad.empty, f"NaN found in CLASSIFIER_FEATURES: {bad.to_dict()}"


def test_km_clip(bundle):
    assert bundle.df["km_max"].max() <= 500_000, "km_max not clipped properly"


def test_target_engineering(bundle):
    df = bundle.df
    derived = (df["days_since_last_service"] > CHURN_THRESHOLD_DAYS).astype(int)
    mismatches = (derived != df["churned"]).sum()
    assert mismatches == 0, f"Target mismatch in {mismatches} rows"


def test_split_trainable(bundle):
    df_train, df_holdout = split_trainable(bundle.df)
    assert len(df_train) + len(df_holdout) == len(bundle.df)
    assert (df_train["tenure_days"] >= 365).all()
    assert (df_holdout["tenure_days"] < 365).all()


# ===== Weights =====
def test_model_inverse_weights(bundle):
    df_train, _ = split_trainable(bundle.df)
    w = model_inverse_weights(df_train["model_name"])
    assert len(w) == len(df_train)
    assert np.isclose(w.mean(), 1.0, atol=1e-6), "weights not normalized to mean=1"
    # Rangers (top model) should weigh less than F-150 (rare)
    ranger_idx = df_train["model_name"] == "RANGER"
    f150_idx = df_train["model_name"] == "F-150"
    if f150_idx.any() and ranger_idx.any():
        assert w[f150_idx].mean() > w[ranger_idx].mean()


def test_cohort_aware_weights(bundle):
    df_train, _ = split_trainable(bundle.df)
    w = cohort_aware_weights(df_train["model_name"], df_train["sales_date"])
    assert len(w) == len(df_train)
    assert np.isclose(w.mean(), 1.0, atol=1e-6)
    assert (w > 0).all()


# ===== Temporal split =====
def test_temporal_val_split(bundle):
    df_train, _ = split_trainable(bundle.df)
    df_multi = df_train[df_train["is_single_event"] == 0]
    splits = temporal_val_split(df_multi, features=ENRICHED_FEATURES,
                                val_quantile=0.64, test_quantile=0.80)
    assert len(splits["y_train"]) > 0
    assert len(splits["y_val"]) > 0
    assert len(splits["y_test"]) > 0
    assert splits["cutoff_val"] <= splits["cutoff_test"]


# ===== Threshold =====
def test_find_threshold_max_f1():
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.35, 0.9])
    thr, f1 = find_threshold_max_f1(y, s)
    assert 0 <= thr <= 1
    assert 0 <= f1 <= 1


# ===== Inference =====
def test_churn_scorer_load_and_predict(bundle):
    scorer = ChurnScorer.load()
    assert scorer.artifacts.feature_order == ENRICHED_FEATURES
    assert "high_confidence" in scorer.artifacts.calibrators

    sample = bundle.df.head(500).copy()
    scored = scorer.predict(sample)
    assert "churn_probability" in scored.columns
    assert "churn_predicted" in scored.columns
    assert "scoring_quality" in scored.columns
    assert "score_fidelidade" in scored.columns
    assert scored["churn_probability"].between(0, 1).all()
    assert scored["churn_predicted"].isin([0, 1]).all()
    # fidelidade = 1 - probability
    assert np.allclose(scored["churn_probability"] + scored["score_fidelidade"], 1.0, atol=1e-3)


def test_scoring_quality_classification(bundle):
    scorer = ChurnScorer.load()
    sample = bundle.df.head(1000)
    quality = scorer._classify_quality(sample)
    assert set(quality.unique()).issubset({"high_confidence", "single_event", "holdout"})
    # cada single_event tem events_count == 1
    se_idx = sample.index[quality.values == "single_event"]
    if len(se_idx) > 0:
        assert (sample.loc[se_idx, "is_single_event"] == 1).all()
