"""Production-grade scoring v2 — Optuna-tuned XGBoost + isotonic calibration on temporal val.

What changed vs scoring v1:
  - Uses ENRICHED_FEATURES (22) instead of CLASSIFIER_FEATURES (19)
  - Trained on multi-event VINs only — single-event scored with caveat flag
  - Hyperparameters from Optuna (data/models/xgb_optuna_best.joblib)
  - Probabilities calibrated isotonically on a held-out val set → less bimodal score
  - 3-way temporal split for validation (train | val | test)

Outputs (under resultados/v2/):
  - scoring_results_v2.csv
  - scoring_results_v2_pt.csv
  - plots/*.png  (hist by class, gradient comparison, calibration, decile)
  - v2_metrics.json
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import os
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
)
from xgboost import XGBClassifier

from src.features import load_features, split_trainable, ENRICHED_FEATURES, KMEANS_FEATURES, model_inverse_weights
from src.classification import (
    temporal_val_split, find_threshold_max_f1, recall_at_top_k, precision_at_top_k,
    RANDOM_STATE,
)
from src.feature_names import FRIENDLY_NAMES

OUT_DIR = REPO_ROOT / "resultados" / "v2"
PLOT_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120


def main():
    print("Loading features + trained artifacts...")
    bundle = load_features()
    df = bundle.df.copy()
    df_train_all, _ = split_trainable(df)
    df_multi = df_train_all[df_train_all["is_single_event"] == 0].copy()

    # Rebuild temporal val split with multi-event-only train+val
    splits = temporal_val_split(df_multi, features=ENRICHED_FEATURES,
                                val_quantile=0.64, test_quantile=0.80)
    print(f"  train: {len(splits['y_train']):,} | val: {len(splits['y_val']):,} | test: {len(splits['y_test']):,}")
    print(f"  positive rate — train: {splits['y_train'].mean():.3f} | val: {splits['y_val'].mean():.3f} | test: {splits['y_test'].mean():.3f}")

    pack = joblib.load(REPO_ROOT / "data" / "models" / "xgb_optuna_best.joblib")
    best_params = pack["best_params"]
    print(f"  loaded Optuna params: {best_params}")

    # Re-fit with early stopping on val (clean fit; ensures internal state matches)
    params = dict(best_params)
    params.update({
        "n_estimators": 2500,
        "early_stopping_rounds": 60,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1, "tree_method": "hist", "device": "cuda",
    })
    print("Fitting XGBoost with Optuna params + early stopping...")
    mdl = XGBClassifier(**params)
    mdl.fit(
        splits["X_train"], splits["y_train"],
        sample_weight=splits["w_train"],
        eval_set=[(splits["X_val"], splits["y_val"])],
        verbose=False,
    )
    print(f"  best_iteration = {int(mdl.best_iteration) + 1}")

    # Isotonic calibration on val
    print("Fitting isotonic calibration on val...")
    val_score_raw = mdl.predict_proba(splits["X_val"])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_score_raw, splits["y_val"])

    # Evaluate test with both raw and calibrated
    test_score_raw = mdl.predict_proba(splits["X_test"])[:, 1]
    test_score_cal = iso.transform(test_score_raw)
    thr_val_raw = find_threshold_max_f1(splits["y_val"], val_score_raw)[0]
    thr_val_cal = find_threshold_max_f1(splits["y_val"], iso.transform(val_score_raw))[0]

    metrics = {
        "model": "xgboost_optuna_v2",
        "features": ENRICHED_FEATURES,
        "n_features": len(ENRICHED_FEATURES),
        "n_train": int(len(splits["y_train"])),
        "n_val": int(len(splits["y_val"])),
        "n_test": int(len(splits["y_test"])),
        "test_positive_rate": float(splits["y_test"].mean()),
        "best_iteration": int(mdl.best_iteration) + 1,
        "best_params": best_params,
        "test_raw": {
            "roc_auc": float(roc_auc_score(splits["y_test"], test_score_raw)),
            "pr_auc": float(average_precision_score(splits["y_test"], test_score_raw)),
            "brier": float(brier_score_loss(splits["y_test"], test_score_raw)),
            "recall_top10pct": float(recall_at_top_k(splits["y_test"], test_score_raw, 0.10)),
            "recall_top20pct": float(recall_at_top_k(splits["y_test"], test_score_raw, 0.20)),
            "precision_top10pct": float(precision_at_top_k(splits["y_test"], test_score_raw, 0.10)),
            "best_threshold_from_val": float(thr_val_raw),
        },
        "test_calibrated": {
            "roc_auc": float(roc_auc_score(splits["y_test"], test_score_cal)),
            "pr_auc": float(average_precision_score(splits["y_test"], test_score_cal)),
            "brier": float(brier_score_loss(splits["y_test"], test_score_cal)),
            "recall_top10pct": float(recall_at_top_k(splits["y_test"], test_score_cal, 0.10)),
            "recall_top20pct": float(recall_at_top_k(splits["y_test"], test_score_cal, 0.20)),
            "precision_top10pct": float(precision_at_top_k(splits["y_test"], test_score_cal, 0.10)),
            "best_threshold_from_val": float(thr_val_cal),
        },
    }
    print("\n=== Test metrics ===")
    print(json.dumps(metrics["test_raw"], indent=2))
    print(json.dumps(metrics["test_calibrated"], indent=2))

    # --- Score ALL VINs (entire dataset, including single-event and holdout) ---
    print("\nScoring all 175k VINs...")
    X_all = df[ENRICHED_FEATURES]
    score_raw = mdl.predict_proba(X_all)[:, 1]
    score_cal = iso.transform(score_raw)

    # K-means cluster
    km_pack = joblib.load(REPO_ROOT / "data" / "models" / "kmeans_segmentation.joblib")
    X_km = df[KMEANS_FEATURES].fillna(df[KMEANS_FEATURES].median())
    cluster_id = km_pack["model"].predict(km_pack["scaler"].transform(X_km))
    cluster_name = pd.Series(cluster_id).map(km_pack["cluster_names"]).values

    out = pd.DataFrame({
        "vin_hash": df["vin_hash"].values,
        "model_name": df["model_name"].values,
        "model_year": df["model_year"].values,
        "sales_date": df["sales_date"].dt.date,
        "last_service_date": df["last_service_date"].dt.date,
        "trainable": df["trainable"].values,
        "is_single_event": df["is_single_event"].values,
        "churned": df["churned"].values,
        "churn_probability_raw": score_raw.round(6),
        "churn_probability": score_cal.round(6),
        "churn_predicted": (score_cal >= thr_val_cal).astype(int),
        "churn_risk_tier": pd.cut(score_cal, bins=[-0.001, 0.20, 0.50, 0.80, 1.001],
                                  labels=["very_low", "low", "high", "very_high"]),
        "churn_decile": pd.qcut(score_cal, q=10, labels=False, duplicates="drop"),
        "cluster_id": cluster_id,
        "cluster_name": cluster_name,
        "events_count": df["events_count"].values,
        "tenure_days": df["tenure_days"].values,
        "days_since_last_service": df["days_since_last_service"].values,
        "km_max": df["km_max"].values,
        "primary_dealer_share": df["primary_dealer_share"].values,
        "scoring_quality": np.where(
            df["is_single_event"] == 1, "low_confidence (single-event, fora do train)",
            np.where(df["trainable"] == 1, "high_confidence", "medium (tenure<365d, fora do train)"),
        ),
    })

    # Format decile as D1..DN
    out["churn_decile"] = "D" + (out["churn_decile"].astype("Int64") + 1).astype(str)
    out.to_csv(OUT_DIR / "scoring_results_v2.csv", index=False)
    out.rename(columns=FRIENDLY_NAMES).to_csv(OUT_DIR / "scoring_results_v2_pt.csv", index=False)
    print(f"  wrote {OUT_DIR / 'scoring_results_v2.csv'}")
    print(f"  wrote {OUT_DIR / 'scoring_results_v2_pt.csv'}")

    (OUT_DIR / "v2_metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(f"  wrote {OUT_DIR / 'v2_metrics.json'}")

    # --- Plots ---
    gen_plots(out, splits, score_raw, score_cal, thr_val_cal)


def gen_plots(out, splits, score_raw, score_cal, thr):
    # 1) Comparação V1 (já em resultados/scoring_results.csv) vs V2 — distribuição
    v1_path = REPO_ROOT / "resultados" / "scoring_results.csv"
    if v1_path.exists():
        v1 = pd.read_csv(v1_path, usecols=["churn_probability"])["churn_probability"]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].hist(v1, bins=50, color="#c44e52", edgecolor="white")
        axes[0].set_title("V1 — baseline (CLASSIFIER_FEATURES, sem calibração)")
        axes[0].set_xlabel("Probabilidade de churn"); axes[0].set_ylabel("# VINs")
        axes[0].axvline(0.5, color="black", linestyle="--", alpha=0.5)
        axes[1].hist(out["churn_probability"], bins=50, color="#4c72b0", edgecolor="white")
        axes[1].set_title("V2 — Optuna + ENRICHED + calibração isotônica")
        axes[1].set_xlabel("Probabilidade de churn"); axes[1].set_ylabel("# VINs")
        axes[1].axvline(thr, color="black", linestyle="--", alpha=0.5, label=f"thr={thr:.3f}")
        axes[1].legend()
        fig.suptitle("Distribuição do score — V1 (bimodal) vs V2 (gradiente)", fontsize=16, y=1.02)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "01_v1_vs_v2_distribution.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {PLOT_DIR / '01_v1_vs_v2_distribution.png'}")

    # 2) V2 distribuição por classe real
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(out[out["churned"] == 0]["churn_probability"], bins=50, alpha=0.65,
            label="Retidos", color="#4c72b0")
    ax.hist(out[out["churned"] == 1]["churn_probability"], bins=50, alpha=0.65,
            label="Churnados", color="#c44e52")
    ax.set_xlabel("Probabilidade de churn (calibrada)"); ax.set_ylabel("# VINs")
    ax.set_title("V2 — score calibrado por classe real")
    ax.axvline(thr, color="black", linestyle="--", alpha=0.5, label=f"threshold={thr:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_v2_hist_by_class.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {PLOT_DIR / '02_v2_hist_by_class.png'}")

    # 3) Calibração: raw vs isotônico no test
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(figsize=(10, 7))
    for s, label, color in [
        (score_raw[:len(splits["y_test"])], "Bruto (sem calibração)", "#c44e52"),
        (score_cal[:len(splits["y_test"])], "Calibrado (isotônico)", "#4c72b0"),
    ]:
        # use the all-VIN scores' restriction is wrong; instead use test only:
        pass
    # The right way: compute calibration_curve on test set scores explicitly
    test_raw = score_raw[: len(out)][out["sales_date"] > splits["cutoff_test"].date()][: len(splits["y_test"])]
    # safer: redo calibration_curve using test arrays from splits, recomputed
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="ideal")
        fp_raw, mp_raw = calibration_curve(splits["y_test"], _score_test(score_raw, out, splits),
                                            n_bins=10, strategy="quantile")
        ax.plot(mp_raw, fp_raw, marker="o", label="Raw (sem calibração)", color="#c44e52")
        fp_cal, mp_cal = calibration_curve(splits["y_test"], _score_test(score_cal, out, splits),
                                            n_bins=10, strategy="quantile")
        ax.plot(mp_cal, fp_cal, marker="s", label="Isotônico", color="#4c72b0")
    ax.set_xlabel("Probabilidade predita média (test)"); ax.set_ylabel("Fração real de positivos")
    ax.set_title("Calibração no test set — raw vs. isotônico")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_calibration_raw_vs_iso.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {PLOT_DIR / '03_calibration_raw_vs_iso.png'}")

    # 4) Decile chart (V2)
    out_train = out[out["trainable"] == 1].copy()
    decile_summary = (out_train.groupby("churn_decile", observed=True)
                       .agg(actual=("churned", "mean"),
                            mean_score=("churn_probability", "mean"),
                            n=("vin_hash", "size"))
                       .reindex([f"D{i+1}" for i in range(10) if f"D{i+1}" in out_train["churn_decile"].unique()]))
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(decile_summary))
    w = 0.4
    ax.bar(x - w/2, decile_summary["actual"], w, label="Churn real", color="#c44e52")
    ax.bar(x + w/2, decile_summary["mean_score"], w, label="Score V2 (calibrado)", color="#4c72b0")
    ax.set_xticks(x); ax.set_xticklabels(decile_summary.index)
    ax.set_xlabel("Decile do score V2 (D1=menor, D10=maior risco)")
    ax.set_ylabel("Taxa de churn")
    ax.set_title("V2 — Calibração por decile (apenas trainable)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_v2_decile.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {PLOT_DIR / '04_v2_decile.png'}")

    # 5) Boxplot do score por cluster
    fig, ax = plt.subplots(figsize=(12, 6))
    order = out.groupby("cluster_name")["churn_probability"].median().sort_values().index
    sns.boxplot(data=out, x="cluster_name", y="churn_probability", order=order, ax=ax,
                hue="cluster_name", palette="RdYlBu_r", legend=False)
    ax.set_title("V2 — score por cluster do K-means")
    ax.set_xlabel("Cluster"); ax.set_ylabel("Probabilidade de churn (calibrada)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "05_v2_by_cluster.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {PLOT_DIR / '05_v2_by_cluster.png'}")


def _score_test(score_all_vins, out_df, splits):
    """Subset score_all_vins to match the test rows in splits (by sales_date filter)."""
    cutoff = splits["cutoff_test"]
    mask = pd.to_datetime(out_df["sales_date"]) > cutoff
    sub = score_all_vins[mask.values]
    # Among those, restrict to multi-event matching the splits test population
    multi_mask = out_df["is_single_event"].values[mask.values] == 0
    return sub[multi_mask][: len(splits["y_test"])]


if __name__ == "__main__":
    main()
