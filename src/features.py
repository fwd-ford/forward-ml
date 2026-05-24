"""Feature loading, cleaning, derivation, and target engineering.

Single source of truth for the ML pipeline. Every notebook and src module
should call `load_features()` instead of reading the CSV directly so that
cleaning rules (km clip, reproducible reference date, NaN handling, target
definition) stay consistent across segmentation, classification, and scoring.

Rules enforced here come from `forward-ml/CLAUDE.md` and `HANDOFF_ML.md`.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES_CSV = REPO_ROOT / "data" / "processed" / "vin_features.csv"

KM_CLIP_UPPER = 500_000
CHURN_THRESHOLD_DAYS = 365
MIN_TENURE_FOR_TRAINING_DAYS = 365

DATE_COLS = [
    "first_service_date",
    "last_service_date",
    "invoice_date",
    "sales_date",
    "warranty_start",
]

KMEANS_FEATURES = [
    "events_count",
    "tenure_days",
    "gap_avg_days",
    "gap_last_days",
    "days_since_last_service",
    "dealers_distinct",
    "primary_dealer_share",
    "service_codes_distinct",
    "km_per_month",
    "is_loyal_to_dealer",
    "is_single_event",
]

CLASSIFIER_FEATURES = [
    "events_count",
    "tenure_days",
    "gap_avg_days",
    "gap_last_days",
    "gap_min_days",
    "gap_max_days",
    "gap_last_vs_avg",
    "km_max",
    "km_per_month",
    "dealers_distinct",
    "primary_dealer_share",
    "service_codes_distinct",
    "service_types_distinct",
    "age_at_last_service_days",
    "is_loyal_to_dealer",
    "is_single_event",
    "is_warranty_active",
    "model_year",
    "model_name_freq",
]

# Enriched feature set: V3 leak-safe + trajectory/seasonality/cohort engineered features.
# These derive ONLY from sales_date, first_service_date, model_year and existing gap stats —
# they DO NOT use last_service_date or any quantity imputed with days_since_last_service.
ENRICHED_FEATURES = [
    # V3 leak-safe core
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
    # sales cohort & seasonality (purchase-anchored, leak-safe)
    "sales_year",
    "sales_month",
    "sales_quarter",
    "years_since_sale",
    "age_when_first_serviced_days",
    "first_service_month",
    "sale_year_minus_model_year",
    # gap shape (multi-event VINs only; single-event get fallback 0)
    "gap_range_ratio",
    "gap_skew",
    "services_per_year",
]


@dataclass
class FeatureBundle:
    df: pd.DataFrame
    reference_date: pd.Timestamp
    n_total: int
    n_trainable: int


def _add_derived(df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()

    df["km_max"] = df["km_max"].clip(upper=KM_CLIP_UPPER)
    df["km_min"] = df["km_min"].clip(upper=KM_CLIP_UPPER)

    df["days_since_last_service"] = (reference_date - df["last_service_date"]).dt.days

    tenure_months = (df["tenure_days"] / 30).clip(lower=1)
    df["km_per_month"] = df["km_max"] / tenure_months

    gap_avg_safe = df["gap_avg_days"].clip(lower=1)
    df["gap_last_vs_avg"] = df["gap_last_days"] / gap_avg_safe

    df["age_at_last_service_days"] = (
        df["last_service_date"] - df["sales_date"]
    ).dt.days

    df["is_loyal_to_dealer"] = (df["primary_dealer_share"] > 0.8).astype(int)
    df["is_single_event"] = (df["events_count"] == 1).astype(int)
    warranty_age = (df["last_service_date"] - df["warranty_start"]).dt.days
    df["is_warranty_active"] = (warranty_age < 3 * 365).astype(int)

    model_freq = df["model_name"].value_counts(normalize=True)
    df["model_name_freq"] = df["model_name"].map(model_freq).astype(float)

    # === Sales cohort & seasonality (purchase-anchored — leak-safe) ===
    df["sales_year"] = df["sales_date"].dt.year.astype("Int64")
    df["sales_month"] = df["sales_date"].dt.month.astype("Int64")
    df["sales_quarter"] = df["sales_date"].dt.quarter.astype("Int64")
    df["years_since_sale"] = ((reference_date - df["sales_date"]).dt.days / 365.25).astype(float)
    df["age_when_first_serviced_days"] = (
        df["first_service_date"] - df["sales_date"]
    ).dt.days.astype(float)
    df["first_service_month"] = df["first_service_date"].dt.month.astype("Int64")
    df["sale_year_minus_model_year"] = (df["sales_year"].astype("Int64") - df["model_year"].astype("Int64")).astype(float)

    # === Gap shape (only informative for multi-event VINs; single-event → 0) ===
    gap_min_safe = df["gap_min_days"].clip(lower=1)
    df["gap_range_ratio"] = (df["gap_max_days"] / gap_min_safe).astype(float)
    df["gap_skew"] = (
        (df["gap_max_days"] - df["gap_avg_days"])
        / (df["gap_avg_days"] - df["gap_min_days"] + 1)
    ).astype(float)
    df["services_per_year"] = (
        df["events_count"] / (df["tenure_days"] / 365.25).clip(lower=0.5)
    ).astype(float)

    return df


def _add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["churned"] = (df["days_since_last_service"] > CHURN_THRESHOLD_DAYS).astype(int)
    df["trainable"] = (df["tenure_days"] >= MIN_TENURE_FOR_TRAINING_DAYS).astype(int)
    return df


def _impute_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    gap_cols = ["gap_avg_days", "gap_min_days", "gap_max_days", "gap_last_days"]
    for c in gap_cols:
        df[c] = df[c].fillna(df["days_since_last_service"])
    df["gap_last_vs_avg"] = df["gap_last_vs_avg"].fillna(1.0)
    df["age_at_last_service_days"] = df["age_at_last_service_days"].fillna(
        df["tenure_days"]
    )

    km_median_by_model = df.groupby("model_name")["km_max"].transform("median")
    df["km_max"] = df["km_max"].fillna(km_median_by_model).fillna(df["km_max"].median())
    df["km_min"] = df["km_min"].fillna(0)
    tenure_months = (df["tenure_days"] / 30).clip(lower=1)
    df["km_per_month"] = df["km_max"] / tenure_months

    # Enriched features fallback for single-event / NaN cases.
    for col, fill in [
        ("gap_range_ratio", 1.0),
        ("gap_skew", 0.0),
        ("services_per_year", df.get("services_per_year", pd.Series(dtype=float)).median() if "services_per_year" in df.columns else 1.0),
        ("age_when_first_serviced_days", 0.0),
        ("sale_year_minus_model_year", 0.0),
        ("years_since_sale", df.get("years_since_sale", pd.Series(dtype=float)).median() if "years_since_sale" in df.columns else 3.0),
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(fill)
    for col in ["sales_year", "sales_month", "sales_quarter", "first_service_month"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64").fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 0).astype(int)

    return df


def _drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["model_name", "model_year"]).reset_index(drop=True)


def load_features(csv_path: Path | str | None = None) -> FeatureBundle:
    """Load vin_features.csv with all cleaning, derivation, and target rules applied.

    The reference date for recency calculations is the max `last_service_date`
    in the dataset, NOT `datetime.now()`, so results are reproducible across
    runs and machines.
    """
    path = Path(csv_path) if csv_path else DEFAULT_FEATURES_CSV
    df = pd.read_csv(path, parse_dates=DATE_COLS)

    reference_date = df["last_service_date"].max()

    df = _drop_invalid_rows(df)
    df = _add_derived(df, reference_date)
    df = _add_target(df)
    df = _impute_for_modeling(df)

    return FeatureBundle(
        df=df,
        reference_date=reference_date,
        n_total=len(df),
        n_trainable=int(df["trainable"].sum()),
    )


def split_trainable(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (trainable, holdout) where holdout = VINs with insufficient tenure."""
    return df[df["trainable"] == 1].copy(), df[df["trainable"] == 0].copy()


def model_inverse_weights(model_series: pd.Series) -> np.ndarray:
    """Inverse-frequency sample weights by model_name to avoid Ranger collapse."""
    freq = model_series.value_counts(normalize=True)
    weights = model_series.map(lambda m: 1.0 / freq[m]).to_numpy()
    return weights / weights.mean()


def cohort_aware_weights(
    model_series: pd.Series,
    sales_date_series: pd.Series,
    cohort_freq: str = "Y",
) -> np.ndarray:
    """Sample weights combining inverse frequency of model_name AND sales cohort.

    Counters two simultaneous biases:
      1. RANGER/KA dominate by model (Ranger-collapse).
      2. Older cohorts dominate by churn target (right-censoring → 2020 has 92% churn,
         2024 has 5% — XGBoost without weighting will collapse into a "2020 detector").

    Final weight = (1/freq_model) * (1/freq_cohort), normalized to mean=1.
    """
    fm = model_series.value_counts(normalize=True)
    cohort = pd.to_datetime(sales_date_series, errors="coerce").dt.to_period(cohort_freq).astype(str)
    cohort = cohort.fillna(cohort.mode().iloc[0] if not cohort.mode().empty else "NaT")
    fc = cohort.value_counts(normalize=True)

    wm = model_series.map(lambda m: 1.0 / fm.get(m, 1.0))
    wc = cohort.map(lambda c: 1.0 / fc.get(c, 1.0))
    w = (wm.to_numpy() * wc.to_numpy())
    return w / w.mean()
