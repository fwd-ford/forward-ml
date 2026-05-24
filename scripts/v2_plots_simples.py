"""Plots simples e diretos do V2 — sem ornamentações pesadas.

10 plots clássicos para análise rápida, salvos em resultados/v2/simples/.
Além disso, gera index_resultados.csv listando todos os artefatos da pasta resultados/.
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

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
)
from xgboost import XGBClassifier

from src.features import load_features, split_trainable, ENRICHED_FEATURES
from src.classification import temporal_val_split, RANDOM_STATE
from src.feature_names import FRIENDLY_NAMES

OUT = REPO_ROOT / "resultados" / "v2" / "simples"
OUT.mkdir(parents=True, exist_ok=True)

# Estilo: limpo e mínimo
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 120, "savefig.bbox": "tight"})
sns.set_style("whitegrid")


def save(fig, name):
    p = OUT / name
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {p.relative_to(REPO_ROOT)}")


def main():
    print("Carregando dados...")
    df = pd.read_csv(REPO_ROOT / "resultados" / "v2" / "scoring_results_v2.csv",
                     parse_dates=["sales_date", "last_service_date"])
    metrics = json.loads((REPO_ROOT / "resultados" / "v2" / "v2_metrics.json").read_text(encoding="utf-8"))
    thr = metrics["test_calibrated"]["best_threshold_from_val"]
    print(f"  {len(df):,} VINs | threshold = {thr:.3f}\n")

    y = df["churned"].to_numpy()
    s = df["churn_probability"].to_numpy()

    # ----- 01: Histograma do score -----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(s, bins=50, color="#4c72b0", edgecolor="white")
    ax.axvline(thr, color="red", linestyle="--", label=f"threshold = {thr:.3f}")
    ax.set_xlabel("Probabilidade de churn")
    ax.set_ylabel("# VINs")
    ax.set_title("01 — Distribuição do score V2")
    ax.legend()
    save(fig, "01_hist_score.png")

    # ----- 02: Histograma por classe (sobreposto) -----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(s[y == 0], bins=50, alpha=0.6, color="#4c72b0", label="Retidos (0)")
    ax.hist(s[y == 1], bins=50, alpha=0.6, color="#c44e52", label="Churnados (1)")
    ax.axvline(thr, color="black", linestyle="--", label=f"thr={thr:.3f}")
    ax.set_xlabel("Probabilidade de churn"); ax.set_ylabel("# VINs")
    ax.set_title("02 — Score por classe real")
    ax.legend()
    save(fig, "02_hist_por_classe.png")

    # ----- 03: ROC -----
    bundle = load_features()
    df_train_all, _ = split_trainable(bundle.df)
    df_multi = df_train_all[df_train_all["is_single_event"] == 0]
    splits = temporal_val_split(df_multi, features=ENRICHED_FEATURES, val_quantile=0.64, test_quantile=0.80)
    cutoff = splits["cutoff_test"]
    mask = (df["sales_date"] > cutoff) & (df["is_single_event"] == 0)
    y_t = df.loc[mask, "churned"].to_numpy()
    s_t = df.loc[mask, "churn_probability"].to_numpy()

    fpr, tpr, _ = roc_curve(y_t, s_t)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#4c72b0", lw=2, label=f"V2 (AUC={roc_auc_score(y_t, s_t):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="aleatório")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("03 — Curva ROC (test temporal)")
    ax.legend()
    save(fig, "03_roc.png")

    # ----- 04: PR -----
    p, r, _ = precision_recall_curve(y_t, s_t)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(r, p, color="#c44e52", lw=2, label=f"V2 (AP={average_precision_score(y_t, s_t):.3f})")
    ax.axhline(y_t.mean(), color="gray", linestyle="--", alpha=0.6, label=f"random (AP={y_t.mean():.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("04 — Precision-Recall (test temporal)")
    ax.legend()
    save(fig, "04_pr.png")

    # ----- 05: Calibração -----
    fp_curve, mp_curve = calibration_curve(y_t, s_t, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfeito")
    ax.plot(mp_curve, fp_curve, marker="o", color="#4c72b0", label="V2 calibrado")
    ax.set_xlabel("Score predito médio"); ax.set_ylabel("Fração real de churners")
    ax.set_title("05 — Calibração (test temporal)")
    ax.legend()
    save(fig, "05_calibracao.png")

    # ----- 06: Confusion matrix no threshold ótimo -----
    pred = (s >= thr).astype(int)
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["Real 0", "Real 1"])
    ax.set_title(f"06 — Matriz de confusão (threshold = {thr:.3f})")
    save(fig, "06_matriz_confusao.png")

    # ----- 07: Boxplot por modelo (top 10) -----
    top_models = df["model_name"].value_counts().head(10).index.tolist()
    sub = df[df["model_name"].isin(top_models)]
    order = sub.groupby("model_name")["churn_probability"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=sub, x="model_name", y="churn_probability", order=order, ax=ax,
                hue="model_name", palette="Set2", legend=False)
    ax.axhline(thr, color="red", linestyle="--", alpha=0.6)
    ax.set_xlabel("Modelo"); ax.set_ylabel("Score")
    ax.set_title("07 — Score por modelo de carro (top 10)")
    ax.tick_params(axis="x", rotation=30)
    save(fig, "07_score_por_modelo.png")

    # ----- 08: Boxplot por cluster -----
    order = df.groupby("cluster_name")["churn_probability"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="cluster_name", y="churn_probability", order=order, ax=ax,
                hue="cluster_name", palette="Set2", legend=False)
    ax.axhline(thr, color="red", linestyle="--", alpha=0.6)
    ax.set_xlabel("Cluster K-means"); ax.set_ylabel("Score")
    ax.set_title("08 — Score por cluster")
    save(fig, "08_score_por_cluster.png")

    # ----- 09: Decile chart -----
    df_q = df.copy()
    df_q["decile"] = pd.qcut(-df_q["churn_probability"], q=10, labels=False, duplicates="drop")
    deciles = df_q.groupby("decile").agg(real=("churned", "mean"),
                                         pred=("churn_probability", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(deciles)); w = 0.4
    ax.bar(x - w/2, deciles["real"], w, label="Churn real", color="#c44e52")
    ax.bar(x + w/2, deciles["pred"], w, label="Score médio", color="#4c72b0")
    ax.set_xticks(x); ax.set_xticklabels([f"D{i+1}" for i in range(len(deciles))])
    ax.set_xlabel("Decile (D1 = mais arriscados)")
    ax.set_ylabel("Taxa")
    ax.set_title("09 — Calibração por decile")
    ax.legend()
    save(fig, "09_decile_chart.png")

    # ----- 10: Threshold sweep -----
    rows = []
    for t in np.linspace(0.05, 0.95, 19):
        pred = (s >= t).astype(int)
        if pred.sum() == 0:
            continue
        rows.append({
            "threshold": t,
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
        })
    sweep = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(sweep["threshold"], sweep["precision"], marker="o", label="Precision", color="#4c72b0")
    ax.plot(sweep["threshold"], sweep["recall"], marker="s", label="Recall", color="#c44e52")
    ax.plot(sweep["threshold"], sweep["f1"], marker="^", label="F1", color="#55a868")
    ax.axvline(thr, color="black", linestyle="--", alpha=0.6, label=f"ótimo={thr:.3f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title("10 — Threshold sweep (dataset completo)")
    ax.legend()
    save(fig, "10_threshold_sweep.png")

    # ----- 11: Feature importance (gain %) -----
    imp = pd.read_csv(REPO_ROOT / "resultados" / "feature_importance_v2_optuna.csv")
    imp = imp.nlargest(15, "gain_pct").sort_values("gain_pct")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(imp["feature_pt"], imp["gain_pct"], color="#4c72b0")
    ax.set_xlabel("gain %"); ax.set_title("11 — Top 15 features por gain (V2)")
    for i, v in enumerate(imp["gain_pct"]):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
    save(fig, "11_feature_importance.png")

    # ----- 12: Tier breakdown -----
    df["tier"] = pd.cut(df["churn_probability"], bins=[-0.001, 0.2, 0.5, 0.8, 1.001],
                       labels=["very_low", "low", "high", "very_high"])
    tier_summary = df.groupby("tier", observed=True).agg(
        n=("vin_hash", "size"), real=("churned", "mean")
    ).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(tier_summary["tier"], tier_summary["n"], color="#4c72b0")
    axes[0].set_title("Quantidade por tier"); axes[0].set_ylabel("# VINs")
    for i, v in enumerate(tier_summary["n"]):
        axes[0].text(i, v, f"{v:,}", ha="center", fontsize=10, va="bottom")
    axes[1].bar(tier_summary["tier"], tier_summary["real"], color="#c44e52")
    axes[1].set_title("Churn real por tier"); axes[1].set_ylabel("Taxa"); axes[1].set_ylim(0, 1.05)
    for i, v in enumerate(tier_summary["real"]):
        axes[1].text(i, v + 0.02, f"{v:.2%}", ha="center", fontsize=10)
    fig.suptitle("12 — Tier de risco", fontsize=14)
    fig.tight_layout()
    save(fig, "12_tier_breakdown.png")

    print(f"\n✓ {len(list(OUT.glob('*.png')))} plots gerados em {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
