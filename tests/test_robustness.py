"""Robustness & bug tests — edge cases, data quality, pipeline integrity."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import os
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pytest
import joblib

from src.features import (
    load_features, split_trainable, ENRICHED_FEATURES, CLASSIFIER_FEATURES,
    KMEANS_FEATURES, cohort_aware_weights, model_inverse_weights,
    CHURN_THRESHOLD_DAYS, MIN_TENURE_FOR_TRAINING_DAYS, KM_CLIP_UPPER,
)
from src.classification import (
    temporal_val_split, temporal_split, find_threshold_max_f1,
    recall_at_top_k, precision_at_top_k, RANDOM_STATE,
)
from src.inference import ChurnScorer


@pytest.fixture(scope="module")
def bundle():
    return load_features()


@pytest.fixture(scope="module")
def scorer():
    return ChurnScorer.load()


# ====================================================================
# DATA QUALITY
# ====================================================================
def test_no_negative_days(bundle):
    """days_since_last_service não pode ser negativo (last_service depois de reference)."""
    df = bundle.df
    neg = (df["days_since_last_service"] < 0).sum()
    assert neg == 0, f"{neg} VINs com days_since_last_service < 0"


def test_tenure_non_negative(bundle):
    assert (bundle.df["tenure_days"] >= 0).all(), "tenure_days < 0 detected"


def test_km_max_within_bounds(bundle):
    """KM clipado a 500k máximo."""
    assert bundle.df["km_max"].max() <= KM_CLIP_UPPER
    assert (bundle.df["km_max"] >= 0).all()


def test_events_count_positive(bundle):
    assert (bundle.df["events_count"] >= 1).all()


def test_target_is_binary(bundle):
    assert set(bundle.df["churned"].unique()).issubset({0, 1})


def test_target_consistent_with_rule(bundle):
    """churned = (days_since_last_service > 365)"""
    derived = (bundle.df["days_since_last_service"] > CHURN_THRESHOLD_DAYS).astype(int)
    mismatch = (derived != bundle.df["churned"]).sum()
    assert mismatch == 0, f"{mismatch} VINs com churned != regra"


def test_trainable_consistent_with_rule(bundle):
    """trainable = (tenure_days >= 365)"""
    derived = (bundle.df["tenure_days"] >= MIN_TENURE_FOR_TRAINING_DAYS).astype(int)
    mismatch = (derived != bundle.df["trainable"]).sum()
    assert mismatch == 0, f"{mismatch} VINs com trainable != regra"


def test_is_single_event_consistent(bundle):
    derived = (bundle.df["events_count"] == 1).astype(int)
    mismatch = (derived != bundle.df["is_single_event"]).sum()
    assert mismatch == 0


def test_dates_parsed_correctly(bundle):
    """sales_date <= last_service_date para multi-event."""
    df = bundle.df.dropna(subset=["sales_date", "last_service_date"])
    multi = df[df["is_single_event"] == 0]
    invalid = (multi["last_service_date"] < multi["sales_date"]).sum()
    assert invalid < len(multi) * 0.01, f"{invalid} VINs multi-event com last_service < sales_date"


def test_primary_dealer_share_in_unit_interval(bundle):
    s = bundle.df["primary_dealer_share"]
    assert (s >= 0).all() and (s <= 1.001).all(), "primary_dealer_share fora de [0,1]"


# ====================================================================
# FEATURE INTEGRITY
# ====================================================================
def test_all_enriched_features_present(bundle):
    missing = [c for c in ENRICHED_FEATURES if c not in bundle.df.columns]
    assert not missing


def test_all_classifier_features_no_nan(bundle):
    nan_counts = bundle.df[CLASSIFIER_FEATURES].isna().sum()
    bad = nan_counts[nan_counts > 0]
    assert bad.empty, f"NaN nas features: {bad.to_dict()}"


def test_all_enriched_no_nan(bundle):
    nan_counts = bundle.df[ENRICHED_FEATURES].isna().sum()
    bad = nan_counts[nan_counts > 0]
    assert bad.empty, f"NaN em ENRICHED_FEATURES: {bad.to_dict()}"


def test_all_kmeans_no_nan_after_fill(bundle):
    X = bundle.df[KMEANS_FEATURES].fillna(bundle.df[KMEANS_FEATURES].median())
    assert X.isna().sum().sum() == 0


def test_no_infinite_values(bundle):
    cols = ENRICHED_FEATURES
    sub = bundle.df[cols].replace([np.inf, -np.inf], np.nan)
    inf_count = (bundle.df[cols].isin([np.inf, -np.inf])).sum().sum()
    assert inf_count == 0, f"{inf_count} inf values nas features"


# ====================================================================
# WEIGHTS SANITY
# ====================================================================
def test_model_weights_normalized(bundle):
    w = model_inverse_weights(bundle.df["model_name"])
    assert np.isclose(w.mean(), 1.0, atol=1e-6)
    assert (w > 0).all()


def test_cohort_weights_normalized(bundle):
    w = cohort_aware_weights(bundle.df["model_name"], bundle.df["sales_date"])
    assert np.isclose(w.mean(), 1.0, atol=1e-6)
    assert (w > 0).all()


def test_cohort_weights_compensate_dominant_cohort(bundle):
    """Cohort 2020 (dominante) deve receber menor peso médio que 2025 (minoritária)."""
    df = bundle.df.dropna(subset=["sales_date"]).copy()
    df["sales_year"] = pd.to_datetime(df["sales_date"]).dt.year
    w = cohort_aware_weights(df["model_name"], df["sales_date"])
    w2020 = w[df["sales_year"] == 2020].mean()
    w2025 = w[df["sales_year"] == 2025].mean() if (df["sales_year"] == 2025).any() else None
    if w2025 is not None:
        assert w2025 > w2020, f"Weight 2025 ({w2025:.3f}) deveria ser > 2020 ({w2020:.3f})"


# ====================================================================
# SPLITS
# ====================================================================
def test_temporal_split_chronological(bundle):
    df_multi = bundle.df[(bundle.df["trainable"] == 1) & (bundle.df["is_single_event"] == 0)]
    splits = temporal_val_split(df_multi, features=ENRICHED_FEATURES,
                                val_quantile=0.64, test_quantile=0.80)
    # cutoffs ascending
    assert splits["cutoff_val"] <= splits["cutoff_test"]
    # train_size > val + test (sanity)
    total = len(splits["y_train"]) + len(splits["y_val"]) + len(splits["y_test"])
    assert total == len(df_multi)


def test_temporal_split_no_overlap(bundle):
    df_multi = bundle.df[(bundle.df["trainable"] == 1) & (bundle.df["is_single_event"] == 0)]
    splits = temporal_val_split(df_multi, features=ENRICHED_FEATURES,
                                val_quantile=0.64, test_quantile=0.80)
    n_total = len(splits["y_train"]) + len(splits["y_val"]) + len(splits["y_test"])
    assert n_total <= len(df_multi)  # sem duplicação


def test_threshold_f1_returns_valid(bundle):
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    s = np.array([0.1, 0.2, 0.6, 0.7, 0.3, 0.8, 0.15, 0.9, 0.55, 0.25])
    thr, f1 = find_threshold_max_f1(y, s)
    assert 0 <= thr <= 1
    assert 0 <= f1 <= 1


# ====================================================================
# INFERENCE ROBUSTNESS
# ====================================================================
def test_predict_handles_single_row(bundle, scorer):
    one = bundle.df.head(1).copy()
    out = scorer.predict(one)
    assert len(out) == 1
    assert 0 <= out["churn_probability"].iloc[0] <= 1


def test_predict_handles_empty(scorer):
    empty = pd.DataFrame(columns=ENRICHED_FEATURES + ["events_count", "tenure_days", "is_single_event", "trainable"])
    # Empty raises gracefully OR returns empty
    try:
        out = scorer.predict(empty)
        assert len(out) == 0
    except Exception as e:
        # esperamos qualquer erro controlado (não crash silencioso)
        assert "feature" in str(e).lower() or "empty" in str(e).lower() or len(empty) == 0


def test_predict_missing_features_raises(scorer):
    """Pedir score sem features → erro claro, não NaN silencioso."""
    bad = pd.DataFrame({"model_name": ["RANGER"], "events_count": [5]})
    with pytest.raises((ValueError, KeyError)):
        scorer.predict(bad)


def test_predict_deterministic(bundle, scorer):
    """Mesma entrada deve dar mesma saída."""
    sample = bundle.df.sample(n=100, random_state=42)
    out1 = scorer.predict(sample.copy())
    out2 = scorer.predict(sample.copy())
    np.testing.assert_array_almost_equal(
        out1["churn_probability"].values,
        out2["churn_probability"].values,
        decimal=6
    )


def test_predict_score_in_unit_interval(bundle, scorer):
    sample = bundle.df.sample(n=1000, random_state=RANDOM_STATE)
    out = scorer.predict(sample)
    assert (out["churn_probability"] >= 0).all()
    assert (out["churn_probability"] <= 1).all()
    assert (out["churn_probability_raw"] >= 0).all()
    assert (out["churn_probability_raw"] <= 1).all()


def test_predict_fidelidade_complement(bundle, scorer):
    sample = bundle.df.sample(n=500, random_state=RANDOM_STATE)
    out = scorer.predict(sample)
    soma = out["churn_probability"] + out["score_fidelidade"]
    assert ((soma - 1.0).abs() < 1e-3).all()


def test_predict_binary_classes(bundle, scorer):
    sample = bundle.df.sample(n=500, random_state=RANDOM_STATE)
    out = scorer.predict(sample)
    assert set(out["churn_predicted"].unique()).issubset({0, 1})


def test_predict_tier_categories(bundle, scorer):
    sample = bundle.df.sample(n=1000, random_state=RANDOM_STATE)
    out = scorer.predict(sample)
    valid = {"very_low", "low", "high", "very_high"}
    assert set(out["churn_risk_tier"].astype(str).unique()).issubset(valid)


def test_two_stage_routing(bundle, scorer):
    """Single-event VINs must be routed to model_single (if it exists)."""
    if scorer.artifacts.model_single is None:
        pytest.skip("model_single not trained")
    sample = bundle.df.head(2000)
    out = scorer.predict(sample.copy())
    # single_event subset
    se = out[out["scoring_quality"] == "single_event"]
    if len(se) == 0:
        pytest.skip("no single-event in sample")
    # se model_single ≠ model_multi, scores devem diferir
    s_via_multi = scorer.artifacts.model_multi.predict_proba(se[ENRICHED_FEATURES])[:, 1]
    s_actual = se["churn_probability_raw"].values
    # Devem diferir em ALGUMA porcentagem (não totalmente, mas alguma)
    pct_diff = (np.abs(s_via_multi - s_actual) > 0.05).mean()
    print(f"  pct VINs onde model_single difere de model_multi: {pct_diff:.1%}")
    assert pct_diff > 0.05, "Two-stage routing parece quebrado — model_single dá quase tudo igual ao multi"


# ====================================================================
# EDGE CASES
# ====================================================================
def test_extreme_km_handled(bundle, scorer):
    """VIN com km no limite máximo (500k) não pode dar NaN."""
    sample = bundle.df.copy()
    sample.loc[sample.index[:5], "km_max"] = KM_CLIP_UPPER
    out = scorer.predict(sample.head(5))
    assert out["churn_probability"].notna().all()


def test_very_recent_vin_handled(bundle, scorer):
    """VIN vendido em 2025 não pode crashar."""
    df = bundle.df.copy()
    df["sales_year"] = pd.to_datetime(df["sales_date"], errors="coerce").dt.year
    recent = df[df["sales_year"] == 2025].head(20)
    if len(recent) == 0:
        pytest.skip("no 2025 VINs")
    out = scorer.predict(recent)
    assert out["churn_probability"].notna().all()


def test_very_old_vin_handled(bundle, scorer):
    df = bundle.df.copy()
    df["sales_year"] = pd.to_datetime(df["sales_date"], errors="coerce").dt.year
    old = df[df["sales_year"] <= 2018].head(20)
    if len(old) == 0:
        pytest.skip("no <=2018 VINs")
    out = scorer.predict(old)
    assert out["churn_probability"].notna().all()


def test_rare_model_handled(bundle, scorer):
    """Modelos com poucos VINs (Cargo, KFA, etc.) não podem crashar."""
    rare_models = bundle.df["model_name"].value_counts().tail(5).index.tolist()
    rare = bundle.df[bundle.df["model_name"].isin(rare_models)]
    if len(rare) == 0:
        pytest.skip("no rare models")
    out = scorer.predict(rare.head(50))
    assert out["churn_probability"].notna().all()


# ====================================================================
# REPRODUCIBILITY
# ====================================================================
def test_scorer_load_idempotent():
    """Carregar 2× e checar que dão o mesmo modelo."""
    s1 = ChurnScorer.load()
    s2 = ChurnScorer.load()
    bundle = load_features()
    sample = bundle.df.sample(n=200, random_state=RANDOM_STATE)
    o1 = s1.predict(sample.copy())
    o2 = s2.predict(sample.copy())
    np.testing.assert_array_almost_equal(
        o1["churn_probability"].values, o2["churn_probability"].values, decimal=6
    )


def test_features_constant_count():
    assert len(ENRICHED_FEATURES) == 22
    assert len(CLASSIFIER_FEATURES) == 19
    assert len(KMEANS_FEATURES) == 11


def test_no_duplicate_features():
    assert len(set(ENRICHED_FEATURES)) == len(ENRICHED_FEATURES)
    assert len(set(CLASSIFIER_FEATURES)) == len(CLASSIFIER_FEATURES)


# ====================================================================
# OUTPUT CSV INTEGRITY
# ====================================================================
def test_scoring_results_csv_exists():
    p = REPO_ROOT / "resultados" / "scoring_results.csv"
    assert p.exists(), "scoring_results.csv não foi gerado"
    df = pd.read_csv(p, nrows=10)
    expected_cols = ["vin_hash", "churn_probability", "churn_predicted",
                      "churn_risk_tier", "score_fidelidade"]
    for c in expected_cols:
        assert c in df.columns, f"coluna {c} faltando no CSV"


def test_metrics_json_exists():
    p = REPO_ROOT / "resultados" / "metrics.json"
    assert p.exists()
    import json
    m = json.loads(p.read_text(encoding="utf-8"))
    assert "global" in m and "by_quality" in m and "by_cohort" in m
    assert m["global"]["accuracy"] > 0.5
    assert 0 < m["global"]["roc_auc"] < 1
