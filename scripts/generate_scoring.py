"""Score every VIN with the trained XGBoost + K-means cluster, then export to CSV.

Outputs (all in resultados/):
  - resultados/scoring_results.csv   (one row per VIN, full info)
  - resultados/plots/scoring_*.png   (score distribution diagnostics)

Run: python -m scripts.generate_scoring
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from src.features import load_features, CLASSIFIER_FEATURES, KMEANS_FEATURES
from src.feature_names import FRIENDLY_NAMES, to_friendly

RESULTS_DIR = REPO_ROOT / "resultados"
CSV_OUT = RESULTS_DIR / "scoring_results.csv"
CSV_OUT_PT = RESULTS_DIR / "scoring_results_pt.csv"
DICT_OUT = RESULTS_DIR / "feature_dictionary_pt.csv"
FIG_DIR = RESULTS_DIR / "plots"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120


def main():
    print("Loading features and trained artifacts...")
    bundle = load_features()
    df = bundle.df.copy()

    clf_pack = joblib.load(REPO_ROOT / "data" / "models" / "churn_classifier.joblib")
    clf_model = clf_pack["model"]
    clf_threshold = clf_pack["threshold"]
    clf_features = clf_pack["feature_order"]
    print(f"  classifier: {clf_pack['model_name']}, threshold={clf_threshold:.3f}, features={len(clf_features)}")

    km_pack = joblib.load(REPO_ROOT / "data" / "models" / "kmeans_segmentation.joblib")
    km_scaler = km_pack["scaler"]
    km_model = km_pack["model"]
    km_names = km_pack["cluster_names"]
    print(f"  kmeans: k={len(km_names)}, names={km_names}")

    # ---------- scoring ----------
    X = df[clf_features]
    proba = clf_model.predict_proba(X)[:, 1]

    X_km = df[KMEANS_FEATURES].fillna(df[KMEANS_FEATURES].median())
    cluster_id = km_model.predict(km_scaler.transform(X_km))

    # ---------- bucket / decile ----------
    df["churn_probability"] = proba
    df["churn_predicted"] = (proba >= clf_threshold).astype(int)
    df["churn_decile"] = pd.qcut(proba, q=10, labels=[f"D{i+1}" for i in range(10)], duplicates="drop").astype(str)
    df["churn_risk_tier"] = pd.cut(
        proba,
        bins=[-0.001, 0.20, 0.50, 0.80, 1.001],
        labels=["very_low", "low", "high", "very_high"],
    )
    df["cluster_id"] = cluster_id
    df["cluster_name"] = pd.Series(cluster_id).map(km_names).values

    # ---------- output schema (compact and useful for the Java backend) ----------
    cols_out = [
        "vin_hash",
        "model_name",
        "model_year",
        "trainable",
        "churned",                # actual (engineered) target — what really happened
        "churn_probability",      # model output
        "churn_predicted",        # 0/1 at the optimized threshold
        "churn_decile",
        "churn_risk_tier",
        "cluster_id",
        "cluster_name",
        # key features the analyst will want side-by-side with the score
        "events_count",
        "tenure_days",
        "days_since_last_service",
        "gap_avg_days",
        "gap_last_days",
        "km_max",
        "dealers_distinct",
        "primary_dealer_share",
        "is_single_event",
        "last_service_date",
        "sales_date",
    ]
    cols_out = [c for c in cols_out if c in df.columns]
    out = df[cols_out].copy()
    out["last_service_date"] = out["last_service_date"].dt.date
    out["sales_date"] = out["sales_date"].dt.date
    out["churn_probability"] = out["churn_probability"].round(6)

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CSV_OUT, index=False)
    print(f"\nWrote {CSV_OUT} ({len(out):,} rows, {len(out.columns)} cols)")

    # Portuguese-headers copy for non-technical reviewers
    out_pt = out.rename(columns=FRIENDLY_NAMES)
    out_pt.to_csv(CSV_OUT_PT, index=False)
    print(f"Wrote {CSV_OUT_PT} (mesmo CSV com cabeçalho em PT)")

    # Companion dictionary so a reader can map back any column
    dict_rows = []
    for col in out.columns:
        dict_rows.append({
            "feature_code": col,
            "feature_pt": FRIENDLY_NAMES.get(col, col),
            "in_classifier_input": col in CLASSIFIER_FEATURES,
            "in_kmeans_input": col in KMEANS_FEATURES,
        })
    pd.DataFrame(dict_rows).to_csv(DICT_OUT, index=False, encoding="utf-8")
    print(f"Wrote {DICT_OUT} (dicionário de features)")

    # ---------- summary report ----------
    print("\n=== Distribuição global do score ===")
    print(out["churn_probability"].describe().round(4).to_string())

    print("\n=== Risk tier × actual churn ===")
    tier_summary = (
        out.groupby("churn_risk_tier", observed=True)
        .agg(
            n=("vin_hash", "size"),
            actual_churn_rate=("churned", "mean"),
            mean_prob=("churn_probability", "mean"),
        )
        .round(3)
    )
    tier_summary["n_pct"] = (tier_summary["n"] / tier_summary["n"].sum() * 100).round(1)
    print(tier_summary.to_string())

    print("\n=== Top 8 modelos: actual churn vs. predicted ===")
    model_summary = (
        out.groupby("model_name")
        .agg(
            n=("vin_hash", "size"),
            actual_churn=("churned", "mean"),
            mean_prob=("churn_probability", "mean"),
        )
        .sort_values("n", ascending=False)
        .head(8)
        .round(3)
    )
    print(model_summary.to_string())

    print("\n=== Cluster × predicted score ===")
    cluster_summary = (
        out.groupby("cluster_name")
        .agg(
            n=("vin_hash", "size"),
            actual_churn=("churned", "mean"),
            mean_prob=("churn_probability", "mean"),
        )
        .round(3)
    )
    print(cluster_summary.to_string())

    # ---------- plots ----------
    gen_plots(out)

    return out


def gen_plots(out):
    print("\nGenerating plots...")

    # 1) Histograma do score por classe real
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(out[out["churned"] == 0]["churn_probability"], bins=50, alpha=0.6, label="Retidos (não churnaram)", color="#4c72b0")
    ax.hist(out[out["churned"] == 1]["churn_probability"], bins=50, alpha=0.6, label="Churnados (saíram da rede)", color="#c44e52")
    ax.set_xlabel("Probabilidade de churn (score do modelo)")
    ax.set_ylabel("Quantidade de VINs")
    ax.set_title("Distribuição do score por classe real")
    ax.legend()
    fig.tight_layout()
    _save(fig, "scoring_01_hist_by_class.png")

    # 2) Score por cluster (boxplot)
    fig, ax = plt.subplots(figsize=(12, 6))
    order = out.groupby("cluster_name")["churn_probability"].median().sort_values().index
    sns.boxplot(data=out, x="cluster_name", y="churn_probability", order=order, ax=ax,
                hue="cluster_name", palette="RdYlBu_r", legend=False)
    ax.set_title("Score por cluster do K-means")
    ax.set_xlabel("Cluster (segmentação K-means)"); ax.set_ylabel("Probabilidade de churn")
    fig.tight_layout()
    _save(fig, "scoring_02_by_cluster.png")

    # 3) Score por modelo de carro (top 10 por volume)
    fig, ax = plt.subplots(figsize=(13, 6))
    top_models = out["model_name"].value_counts().head(10).index.tolist()
    sub = out[out["model_name"].isin(top_models)]
    order = sub.groupby("model_name")["churn_probability"].median().sort_values().index
    sns.boxplot(data=sub, x="model_name", y="churn_probability", order=order, ax=ax,
                hue="model_name", palette="viridis", legend=False)
    ax.set_title("Score por modelo de carro (top 10 por volume)")
    ax.set_xlabel("Modelo do carro"); ax.set_ylabel("Probabilidade de churn")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _save(fig, "scoring_03_by_model.png")

    # 4) Curva de ganho — recall vs. fração contatada
    sorted_ = out.sort_values("churn_probability", ascending=False).reset_index(drop=True)
    cum_pos = sorted_["churned"].cumsum()
    total_pos = sorted_["churned"].sum()
    frac_contacted = (np.arange(len(sorted_)) + 1) / len(sorted_)
    gain = cum_pos / total_pos

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(frac_contacted, gain, color="#4c72b0", lw=2.5, label="modelo")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="aleatório")
    ax.fill_between(frac_contacted, gain, frac_contacted, alpha=0.2, color="#4c72b0")
    for q in [0.10, 0.20, 0.30, 0.50]:
        idx = int(len(sorted_) * q) - 1
        ax.axvline(q, color="gray", alpha=0.3, linestyle=":")
        ax.text(q + 0.005, gain.iloc[idx] - 0.04, f"top {int(q*100)}% → captura {gain.iloc[idx]:.0%} dos churners", fontsize=10)
    ax.set_xlabel("Fração de VINs contatados (ordenados do mais ao menos arriscado)")
    ax.set_ylabel("Cobertura de churners reais")
    ax.set_title("Curva de ganho — ROI por fração contatada")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "scoring_04_gain_curve.png")

    # 5) Decile chart — actual churn rate por decile do score
    decile_summary = (
        out.groupby("churn_decile", observed=True)
        .agg(actual=("churned", "mean"), mean_score=("churn_probability", "mean"), n=("vin_hash", "size"))
        .reindex([f"D{i+1}" for i in range(10) if f"D{i+1}" in out["churn_decile"].unique()])
    )

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(decile_summary))
    w = 0.4
    ax.bar(x - w/2, decile_summary["actual"], w, label="Churn real observado", color="#c44e52")
    ax.bar(x + w/2, decile_summary["mean_score"], w, label="Score médio do modelo", color="#4c72b0")
    ax.set_xticks(x); ax.set_xticklabels(decile_summary.index)
    ax.set_xlabel("Decile do score (D1 = 10% menos arriscados, D10 = 10% mais arriscados)")
    ax.set_ylabel("Taxa de churn")
    ax.set_title("Calibração por decile — score previsto vs. churn observado")
    ax.legend()
    fig.tight_layout()
    _save(fig, "scoring_05_decile_calibration.png")

    # 6) Risk tier breakdown
    tier_summary = (
        out.groupby("churn_risk_tier", observed=True)
        .agg(n=("vin_hash", "size"), actual=("churned", "mean"))
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    tier_labels_pt = {"very_low": "muito baixo", "low": "baixo", "high": "alto", "very_high": "muito alto"}
    tier_summary.index = tier_summary.index.map(lambda x: tier_labels_pt.get(str(x), str(x)))
    tier_summary["n"].plot.bar(ax=axes[0], color="#4c72b0")
    axes[0].set_title("Quantidade de VINs por tier de risco")
    axes[0].set_xlabel("Tier de risco")
    axes[0].set_ylabel("Quantidade de VINs")
    axes[0].tick_params(axis="x", rotation=0)
    for i, v in enumerate(tier_summary["n"]):
        axes[0].text(i, v + 1000, f"{v:,}", ha="center", fontsize=10)

    tier_summary["actual"].plot.bar(ax=axes[1], color="#c44e52")
    axes[1].set_title("Taxa de churn real por tier")
    axes[1].set_xlabel("Tier de risco")
    axes[1].set_ylabel("Taxa de churn real")
    axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis="x", rotation=0)
    for i, v in enumerate(tier_summary["actual"]):
        axes[1].text(i, v + 0.02, f"{v:.2%}", ha="center", fontsize=11)
    fig.tight_layout()
    _save(fig, "scoring_06_risk_tier.png")


def _save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote resultados/plots/{name}")


if __name__ == "__main__":
    main()
