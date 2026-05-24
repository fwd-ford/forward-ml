"""Dashboard analítico completo do V2 — 18 plots polidos para análise profunda.

Plots produzidos (em resultados/v2/dashboard/):
  D01 — score_distribution_kde.png         KDE do score por classe real + threshold
  D02 — score_distribution_violin.png      violino com p25/p50/p75 marcados
  D03 — confusion_matrices_grid.png        4 matrizes em 1 figura (thresholds diferentes)
  D04 — roc_pr_zoom.png                    ROC e PR no test temporal com zoom
  D05 — calibration_diagram.png            reliability + histograma de bins
  D06 — gain_lift_combined.png             curva de ganho + lift por decile lado a lado
  D07 — threshold_economic.png             precision/recall/n_alertas vs threshold com zonas
  D08 — heatmap_modelo_x_tier.png          contagem em (modelo × tier)
  D09 — heatmap_cluster_x_tier.png         contagem em (cluster × tier) + churn real
  D10 — per_model_metrics_panel.png        ROC/PR/recall por model_name (top 10)
  D11 — sales_year_curve.png               score predito + churn real por ano de venda
  D12 — shap_summary_panel.png             SHAP beeswarm + bar lado a lado
  D13 — shap_dependence_top6.png           6 dependence plots polidos
  D14 — feature_importance_compare.png     gain V1 vs V2 com nomes PT
  D15 — score_quality_breakdown.png        scoring_quality × distribuição score
  D16 — risk_pyramid.png                   pirâmide de risco (tiers) com tamanho de cada
  D17 — timeseries_fold_summary.png        métricas TimeSeriesSplit (5-fold) em painel
  D18 — top_vins_visual.png                visual das 10 VINs mais arriscados + mais fiéis
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
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import shap

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score,
)
from xgboost import XGBClassifier

from src.features import load_features, split_trainable, ENRICHED_FEATURES, model_inverse_weights
from src.classification import temporal_val_split, RANDOM_STATE
from src.feature_names import FRIENDLY_NAMES, to_friendly

# === Style ===
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})
sns.set_palette("Set2")

# === Cores fixas ===
COLOR_CHURN = "#c44e52"
COLOR_RETIDO = "#4c72b0"
COLOR_THR = "#2c3e50"
COLOR_HIGHLIGHT = "#dd8452"
PALETTE_TIER = ["#55a868", "#ddb142", "#dd8452", "#c44e52"]  # very_low → very_high
TIER_ORDER = ["very_low", "low", "high", "very_high"]
TIER_LABELS = ["Muito baixo\n(fiel)", "Baixo", "Alto", "Muito alto\n(saindo)"]

OUT = REPO_ROOT / "resultados" / "v2" / "dashboard"
OUT.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    p = OUT / name
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote dashboard/{name}")


# ========================= LOAD =========================
def load_all():
    print("Carregando dados e modelo...")
    df = pd.read_csv(REPO_ROOT / "resultados" / "v2" / "scoring_results_v2.csv",
                     parse_dates=["sales_date", "last_service_date"])
    metrics = json.loads((REPO_ROOT / "resultados" / "v2" / "v2_metrics.json").read_text(encoding="utf-8"))
    thr = metrics["test_calibrated"]["best_threshold_from_val"]

    bundle = load_features()
    df_full = bundle.df.copy()
    df_train_all, _ = split_trainable(df_full)
    df_multi = df_train_all[df_train_all["is_single_event"] == 0].copy()
    splits = temporal_val_split(df_multi, features=ENRICHED_FEATURES, val_quantile=0.64, test_quantile=0.80)

    pack = joblib.load(REPO_ROOT / "data" / "models" / "xgb_optuna_best.joblib")
    params = dict(pack["best_params"])
    params.update({
        "n_estimators": 2500, "early_stopping_rounds": 60,
        "eval_metric": "logloss", "random_state": RANDOM_STATE,
        "n_jobs": -1, "tree_method": "hist", "device": "cuda",
    })
    mdl = XGBClassifier(**params)
    mdl.fit(splits["X_train"], splits["y_train"],
            sample_weight=splits["w_train"],
            eval_set=[(splits["X_val"], splits["y_val"])], verbose=False)
    return df, metrics, thr, splits, mdl


# ========================= PLOTS =========================
def D01_kde(df, thr):
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.kdeplot(df.loc[df["churned"] == 0, "churn_probability"],
                ax=ax, fill=True, alpha=0.55, lw=2, label="Retidos (churned=0)", color=COLOR_RETIDO)
    sns.kdeplot(df.loc[df["churned"] == 1, "churn_probability"],
                ax=ax, fill=True, alpha=0.55, lw=2, label="Churnados (churned=1)", color=COLOR_CHURN)
    ax.axvline(thr, color=COLOR_THR, linestyle="--", lw=2, label=f"Threshold ótimo F1 = {thr:.3f}")
    ax.set_xlabel("Probabilidade de churn (V2 calibrado)", fontsize=12)
    ax.set_ylabel("Densidade", fontsize=12)
    ax.set_title("D01 — Distribuição do score por classe real (KDE)", fontsize=15)
    ax.legend(loc="upper center", framealpha=0.95)
    ax.set_xlim(-0.02, 1.02)
    n0, n1 = (df["churned"] == 0).sum(), (df["churned"] == 1).sum()
    ax.text(0.02, 0.98, f"Retidos: n={n0:,}  |  Churnados: n={n1:,}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="gray"))
    save_fig(fig, "D01_score_distribution_kde.png")


def D02_violin(df, thr):
    df = df.copy()
    df["classe"] = df["churned"].map({0: "Retido", 1: "Churnado"})
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.violinplot(data=df, x="classe", y="churn_probability", ax=ax,
                   hue="classe", palette=[COLOR_RETIDO, COLOR_CHURN], inner="quartile",
                   legend=False, cut=0)
    ax.axhline(thr, color=COLOR_THR, linestyle="--", lw=2, label=f"Threshold ótimo = {thr:.3f}")
    ax.set_ylabel("Probabilidade de churn")
    ax.set_xlabel("")
    ax.set_title("D02 — Score por classe (violino com quartis)", fontsize=15)
    ax.legend()
    for i, c in enumerate([0, 1]):
        sub = df.loc[df["churned"] == c, "churn_probability"]
        ax.text(i, 1.07, f"n={len(sub):,}\nmédia={sub.mean():.3f}\nmediana={sub.median():.3f}",
                ha="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))
    ax.set_ylim(-0.05, 1.18)
    save_fig(fig, "D02_score_distribution_violin.png")


def D03_confusion_grid(df, thr):
    y = df["churned"].to_numpy()
    s = df["churn_probability"].to_numpy()
    thresholds = [0.20, round(thr, 3), 0.50, 0.80]
    titles = [f"Threshold = {t:.3f}" for t in thresholds]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, t, title in zip(axes.flatten(), thresholds, titles):
        pred = (s >= t).astype(int)
        cm = confusion_matrix(y, pred)
        cm_pct = cm / cm.sum() * 100
        annot = np.array([[f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)" for j in range(2)] for i in range(2)])
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["Pred. Retido", "Pred. Churnado"],
                    yticklabels=["Real Retido", "Real Churnado"],
                    annot_kws={"size": 12, "weight": "bold"})
        tp, fp, fn, tn = cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        acc = (tp + tn) / cm.sum()
        ax.set_title(f"{title}\nAcc={acc:.3f}  |  Precision={prec:.3f}  |  Recall={rec:.3f}", fontsize=12)
    fig.suptitle("D03 — Matriz de confusão em 4 thresholds (dataset completo)", fontsize=16, y=1.01)
    fig.tight_layout()
    save_fig(fig, "D03_confusion_matrices_grid.png")


def D04_roc_pr_zoom(df, splits, mdl):
    cutoff = splits["cutoff_test"]
    mask = (df["sales_date"] > cutoff) & (df["is_single_event"] == 0)
    y_t = df.loc[mask, "churned"].to_numpy()
    s_t = df.loc[mask, "churn_probability"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fpr, tpr, _ = roc_curve(y_t, s_t)
    auc = roc_auc_score(y_t, s_t)
    axes[0].plot(fpr, tpr, color=COLOR_RETIDO, lw=3, label=f"V2 (AUC={auc:.4f})")
    axes[0].fill_between(fpr, tpr, alpha=0.18, color=COLOR_RETIDO)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Aleatório (AUC=0.5)")
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC — test temporal (n={len(y_t):,}, churn={y_t.mean():.2%})")
    axes[0].legend(loc="lower right"); axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.05)

    p, r, _ = precision_recall_curve(y_t, s_t)
    ap = average_precision_score(y_t, s_t)
    axes[1].plot(r, p, color=COLOR_CHURN, lw=3, label=f"V2 (AP={ap:.4f})")
    axes[1].fill_between(r, p, alpha=0.18, color=COLOR_CHURN)
    axes[1].axhline(y_t.mean(), color="gray", linestyle="--", lw=1.5, alpha=0.7,
                    label=f"Random (AP={y_t.mean():.3f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall — test temporal (base rate {y_t.mean():.2%})")
    axes[1].legend(loc="upper right"); axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1.05)

    fig.suptitle("D04 — ROC e Precision-Recall no test temporal", fontsize=16, y=1.02)
    fig.tight_layout()
    save_fig(fig, "D04_roc_pr_zoom.png")


def D05_calibration(df, splits):
    cutoff = splits["cutoff_test"]
    mask = (df["sales_date"] > cutoff) & (df["is_single_event"] == 0)
    y_t = df.loc[mask, "churned"].to_numpy()
    s_t = df.loc[mask, "churn_probability"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    fp, mp = calibration_curve(y_t, s_t, n_bins=10, strategy="quantile")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfeitamente calibrado")
    axes[0].plot(mp, fp, marker="o", lw=2.5, markersize=10, color=COLOR_RETIDO, label="V2 calibrado")
    for x, y in zip(mp, fp):
        axes[0].annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(8, 5), fontsize=9)
    axes[0].set_ylabel("Fração real de churners")
    axes[0].set_title("D05 — Diagrama de calibração (test temporal)", fontsize=15)
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1); axes[0].legend(loc="upper left")

    axes[1].hist(s_t, bins=30, color=COLOR_HIGHLIGHT, edgecolor="white")
    axes[1].set_xlabel("Probabilidade predita (bins)")
    axes[1].set_ylabel("# VINs no bin")
    axes[1].set_xlim(0, 1)
    save_fig(fig, "D05_calibration_diagram.png")


def D06_gain_lift(df):
    sorted_df = df.sort_values("churn_probability", ascending=False).reset_index(drop=True)
    cum_pos = sorted_df["churned"].cumsum()
    total_pos = sorted_df["churned"].sum()
    frac = (np.arange(len(sorted_df)) + 1) / len(sorted_df)
    gain = cum_pos / total_pos

    df_q = df.copy()
    df_q["decile"] = pd.qcut(-df_q["churn_probability"], q=10, labels=False, duplicates="drop")
    lift = df_q.groupby("decile").agg(rate=("churned", "mean"),
                                       n=("vin_hash", "size")).reset_index()
    base = df_q["churned"].mean()
    lift["lift"] = lift["rate"] / base
    lift["dec_label"] = "D" + (lift["decile"].astype(int) + 1).astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    axes[0].plot(frac, gain, color=COLOR_RETIDO, lw=3, label="Modelo V2")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6, label="Aleatório (baseline)")
    axes[0].fill_between(frac, gain, frac, alpha=0.25, color=COLOR_RETIDO, label="Ganho do modelo")
    for q in [0.10, 0.20, 0.30, 0.50]:
        idx = int(len(sorted_df) * q) - 1
        axes[0].axvline(q, color="gray", alpha=0.3, linestyle=":")
        axes[0].annotate(f"top {int(q*100)}%\n→ {gain.iloc[idx]:.0%}",
                         xy=(q, gain.iloc[idx]),
                         xytext=(q + 0.02, gain.iloc[idx] - 0.10),
                         fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    axes[0].set_xlabel("Fração de VINs contatados (ordenados pelo score)")
    axes[0].set_ylabel("Cobertura de churners reais (recall)")
    axes[0].set_title("Curva de ganho — alcance da campanha vs % contatado")
    axes[0].legend(loc="lower right"); axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.02)

    colors = [COLOR_CHURN if l > 1 else COLOR_RETIDO for l in lift["lift"]]
    bars = axes[1].bar(lift["dec_label"], lift["lift"], color=colors)
    axes[1].axhline(1.0, color="black", linestyle="--", alpha=0.6, label="Baseline aleatório (1.0×)")
    axes[1].set_xlabel("Decile do score (D1 = top 10% mais arriscados)")
    axes[1].set_ylabel("Lift vs aleatório")
    axes[1].set_title("Lift por decile — quantas vezes mais churners que random")
    for b, l in zip(bars, lift["lift"]):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.04, f"{l:.2f}×",
                     ha="center", fontsize=10, fontweight="bold")
    axes[1].legend(loc="upper right")
    axes[1].set_ylim(0, lift["lift"].max() * 1.18)

    fig.suptitle("D06 — Curva de ganho e lift por decile", fontsize=16, y=1.02)
    fig.tight_layout()
    save_fig(fig, "D06_gain_lift_combined.png")


def D07_threshold_economic(df, thr):
    y = df["churned"].to_numpy()
    s = df["churn_probability"].to_numpy()
    thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    for t in thresholds:
        pred = (s >= t).astype(int)
        if pred.sum() == 0:
            continue
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        rows.append({"threshold": t, "precision": prec, "recall": rec, "f1": f1,
                     "n_alertas": pred.sum(), "tp": tp, "fp": fp})
    sweep = pd.DataFrame(rows)

    fig, ax1 = plt.subplots(figsize=(15, 7))
    ax1.plot(sweep["threshold"], sweep["precision"], marker="o", lw=2.5, label="Precision", color=COLOR_RETIDO)
    ax1.plot(sweep["threshold"], sweep["recall"], marker="s", lw=2.5, label="Recall", color=COLOR_CHURN)
    ax1.plot(sweep["threshold"], sweep["f1"], marker="^", lw=2.5, label="F1", color="#55a868")
    ax1.axvline(thr, color=COLOR_THR, linestyle="--", lw=2, label=f"Threshold ótimo = {thr:.3f}")
    ax1.fill_between([0.3, 0.6], 0, 1, alpha=0.07, color="green", label="Zona operacional recomendada")
    ax1.set_xlabel("Threshold de decisão")
    ax1.set_ylabel("Score (precision/recall/F1)")
    ax1.set_xlim(0.05, 0.95); ax1.set_ylim(0, 1.05)
    ax1.legend(loc="center left")

    ax2 = ax1.twinx()
    ax2.bar(sweep["threshold"], sweep["n_alertas"], width=0.04, alpha=0.18,
            color="gray", label="# Alertas gerados")
    ax2.set_ylabel("# Alertas gerados (cinza)", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.grid(False)

    fig.suptitle("D07 — Threshold sweep com zona operacional + volume de alertas", fontsize=15)
    fig.tight_layout()
    save_fig(fig, "D07_threshold_economic.png")


def D08_heatmap_model_tier(df):
    df = df.copy()
    df["tier"] = pd.cut(df["churn_probability"], bins=[-0.001, 0.2, 0.5, 0.8, 1.001],
                       labels=TIER_ORDER)
    top_models = df["model_name"].value_counts().head(10).index.tolist()
    sub = df[df["model_name"].isin(top_models)]
    heat = sub.pivot_table(index="model_name", columns="tier", values="vin_hash",
                            aggfunc="count", fill_value=0, observed=True).reindex(top_models)
    heat_pct = heat.div(heat.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    sns.heatmap(heat, annot=True, fmt=",d", cmap="YlOrRd", ax=axes[0],
                cbar_kws={"label": "# VINs"})
    axes[0].set_title("Volume absoluto"); axes[0].set_xlabel("Tier de risco"); axes[0].set_ylabel("Modelo")
    axes[0].set_xticklabels(TIER_LABELS, rotation=0)

    sns.heatmap(heat_pct, annot=True, fmt=".1f", cmap="RdYlGn_r", ax=axes[1],
                cbar_kws={"label": "% do modelo"})
    axes[1].set_title("% dentro de cada modelo")
    axes[1].set_xlabel("Tier de risco"); axes[1].set_ylabel("Modelo")
    axes[1].set_xticklabels(TIER_LABELS, rotation=0)

    fig.suptitle("D08 — Distribuição de risco por modelo de carro (top 10 por volume)", fontsize=15, y=1.02)
    fig.tight_layout()
    save_fig(fig, "D08_heatmap_modelo_x_tier.png")


def D09_heatmap_cluster_tier(df):
    df = df.copy()
    df["tier"] = pd.cut(df["churn_probability"], bins=[-0.001, 0.2, 0.5, 0.8, 1.001],
                       labels=TIER_ORDER)
    heat = df.pivot_table(index="cluster_name", columns="tier", values="vin_hash",
                           aggfunc="count", fill_value=0, observed=True)
    rate = df.pivot_table(index="cluster_name", columns="tier", values="churned",
                           aggfunc="mean", fill_value=0, observed=True)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    sns.heatmap(heat, annot=True, fmt=",d", cmap="YlOrRd", ax=axes[0],
                cbar_kws={"label": "# VINs"})
    axes[0].set_title("# VINs por (cluster K-means × tier)")
    axes[0].set_xlabel("Tier"); axes[0].set_ylabel("Cluster")
    axes[0].set_xticklabels(TIER_LABELS, rotation=0)

    sns.heatmap(rate * 100, annot=True, fmt=".1f", cmap="RdYlGn_r", ax=axes[1],
                cbar_kws={"label": "% real de churners"})
    axes[1].set_title("Taxa real de churn (%)")
    axes[1].set_xlabel("Tier"); axes[1].set_ylabel("Cluster")
    axes[1].set_xticklabels(TIER_LABELS, rotation=0)

    fig.suptitle("D09 — Cluster K-means × tier V2 (validação entre camadas)", fontsize=15, y=1.02)
    fig.tight_layout()
    save_fig(fig, "D09_heatmap_cluster_x_tier.png")


def D10_per_model_metrics(df):
    top_models = df["model_name"].value_counts().head(10).index.tolist()
    rows = []
    for m in top_models:
        sub = df[df["model_name"] == m]
        if len(sub) < 50 or sub["churned"].nunique() < 2:
            continue
        rows.append({
            "model_name": m,
            "n": len(sub),
            "churn_real": sub["churned"].mean(),
            "score_medio": sub["churn_probability"].mean(),
            "roc_auc": roc_auc_score(sub["churned"], sub["churn_probability"]),
            "pr_auc": average_precision_score(sub["churned"], sub["churn_probability"]),
        })
    per_m = pd.DataFrame(rows).sort_values("n", ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    bars = axes[0, 0].barh(per_m["model_name"], per_m["roc_auc"], color=COLOR_RETIDO)
    axes[0, 0].axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    axes[0, 0].axvline(0.65, color="red", linestyle="--", alpha=0.5, label="floor 0.65")
    axes[0, 0].set_xlabel("ROC-AUC"); axes[0, 0].set_title("ROC-AUC por modelo")
    axes[0, 0].set_xlim(0, 1.05); axes[0, 0].legend()
    for b, v in zip(bars, per_m["roc_auc"]):
        axes[0, 0].text(v + 0.01, b.get_y() + b.get_height()/2, f"{v:.3f}",
                        va="center", fontsize=10)
    axes[0, 0].invert_yaxis()

    bars = axes[0, 1].barh(per_m["model_name"], per_m["pr_auc"], color=COLOR_CHURN)
    axes[0, 1].set_xlabel("PR-AUC"); axes[0, 1].set_title("PR-AUC por modelo")
    axes[0, 1].set_xlim(0, 1.05)
    for b, v in zip(bars, per_m["pr_auc"]):
        axes[0, 1].text(v + 0.01, b.get_y() + b.get_height()/2, f"{v:.3f}",
                        va="center", fontsize=10)
    axes[0, 1].invert_yaxis()

    x = np.arange(len(per_m)); w = 0.4
    axes[1, 0].barh(x - w/2, per_m["churn_real"], w, label="Churn real", color=COLOR_CHURN)
    axes[1, 0].barh(x + w/2, per_m["score_medio"], w, label="Score médio V2", color=COLOR_RETIDO)
    axes[1, 0].set_yticks(x); axes[1, 0].set_yticklabels(per_m["model_name"])
    axes[1, 0].invert_yaxis(); axes[1, 0].set_xlabel("Taxa")
    axes[1, 0].set_title("Churn real vs. score médio por modelo")
    axes[1, 0].set_xlim(0, 1.05); axes[1, 0].legend()
    for i, (cr, sm) in enumerate(zip(per_m["churn_real"], per_m["score_medio"])):
        axes[1, 0].text(cr + 0.01, i - w/2, f"{cr:.2f}", va="center", fontsize=9)
        axes[1, 0].text(sm + 0.01, i + w/2, f"{sm:.2f}", va="center", fontsize=9)

    bars = axes[1, 1].barh(per_m["model_name"], per_m["n"], color="#8172b3")
    axes[1, 1].set_xlabel("# VINs"); axes[1, 1].set_title("Volume por modelo")
    for b, v in zip(bars, per_m["n"]):
        axes[1, 1].text(v + 1000, b.get_y() + b.get_height()/2, f"{v:,}",
                        va="center", fontsize=10)
    axes[1, 1].invert_yaxis()

    fig.suptitle("D10 — Performance V2 quebrada por modelo de carro", fontsize=16, y=1.01)
    fig.tight_layout()
    save_fig(fig, "D10_per_model_metrics_panel.png")


def D11_sales_year(df):
    df["sales_year"] = pd.to_datetime(df["sales_date"], errors="coerce").dt.year
    cohort = (df.dropna(subset=["sales_year"])
                .groupby("sales_year")
                .agg(n=("vin_hash", "size"),
                     churn_real=("churned", "mean"),
                     score_medio=("churn_probability", "mean"))
                .reset_index())
    cohort["sales_year"] = cohort["sales_year"].astype(int)

    fig, ax1 = plt.subplots(figsize=(14, 7))
    bars = ax1.bar(cohort["sales_year"], cohort["n"], alpha=0.5, color="#cccccc",
                   edgecolor="gray", label="# VINs (volume)")
    ax1.set_xlabel("Ano da venda"); ax1.set_ylabel("# VINs (barras cinza)")
    for b, v in zip(bars, cohort["n"]):
        ax1.text(b.get_x() + b.get_width()/2, v + cohort["n"].max() * 0.02,
                 f"{v:,}", ha="center", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(cohort["sales_year"], cohort["churn_real"], marker="o", lw=3,
             color=COLOR_CHURN, label="Churn real")
    ax2.plot(cohort["sales_year"], cohort["score_medio"], marker="s", lw=3,
             color=COLOR_RETIDO, label="Score predito V2")
    ax2.set_ylabel("Taxa (linhas)"); ax2.set_ylim(0, 1.05)
    ax2.grid(False)
    for x, y in zip(cohort["sales_year"], cohort["churn_real"]):
        ax2.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=9, color=COLOR_CHURN)
    ax2.legend(loc="upper right"); ax1.legend(loc="upper left")
    ax1.set_title("D11 — Curva por cohort de venda: volume + churn real + score V2", fontsize=15)
    fig.tight_layout()
    save_fig(fig, "D11_sales_year_curve.png")


def D12_shap_summary(splits, mdl):
    sample = splits["X_val"].sample(n=min(5000, len(splits["X_val"])), random_state=RANDOM_STATE)
    sv = shap.TreeExplainer(mdl).shap_values(sample)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) == 2 else sv[0]
    sample_pt = sample.copy()
    sample_pt.columns = to_friendly(sample.columns)

    fig = plt.figure(figsize=(18, 9))
    plt.subplot(1, 2, 1)
    shap.summary_plot(sv, sample_pt, show=False, max_display=15, color_bar=False)
    plt.title("Beeswarm — impacto SHAP por VIN")

    plt.subplot(1, 2, 2)
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=sample_pt.columns).sort_values(ascending=False)
    total = mean_abs.sum() or 1.0
    pct = (mean_abs / total * 100).head(15)
    bars = plt.barh(pct.index[::-1], pct.values[::-1], color=COLOR_RETIDO)
    plt.xlabel("% da importância total (mean |SHAP|)")
    plt.title("Importância global (top 15)")
    for b, v in zip(bars, pct.values[::-1]):
        plt.text(b.get_width() + max(pct) * 0.01, b.get_y() + b.get_height()/2,
                 f"{v:.1f}%", va="center", fontsize=9)
    plt.suptitle("D12 — SHAP do modelo V2 (top 15 features)", fontsize=16, y=1.02)
    plt.tight_layout()
    save_fig(plt.gcf(), "D12_shap_summary_panel.png")
    return mean_abs


def D13_shap_dependence(splits, mdl, mean_abs):
    sample = splits["X_val"].sample(n=min(5000, len(splits["X_val"])), random_state=RANDOM_STATE)
    sv = shap.TreeExplainer(mdl).shap_values(sample)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) == 2 else sv[0]
    top6 = mean_abs.head(6).index.tolist()
    # mapear nomes PT → code
    pt_to_code = {v: k for k, v in FRIENDLY_NAMES.items()}
    top6_code = [pt_to_code.get(n, n) for n in top6]

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    for ax, feat_code in zip(axes.flatten(), top6_code):
        if feat_code in sample.columns:
            shap.dependence_plot(feat_code, sv, sample, ax=ax, show=False, interaction_index=None)
            ax.set_title(FRIENDLY_NAMES.get(feat_code, feat_code))
            ax.set_xlabel(FRIENDLY_NAMES.get(feat_code, feat_code))
    fig.suptitle("D13 — SHAP dependence dos top 6 features (V2)", fontsize=16, y=1.01)
    fig.tight_layout()
    save_fig(fig, "D13_shap_dependence_top6.png")


def D14_feature_importance_compare():
    v1 = pd.read_csv(REPO_ROOT / "resultados" / "feature_importance_V0_baseline.csv")
    v2 = pd.read_csv(REPO_ROOT / "resultados" / "feature_importance_v2_optuna.csv")
    v1_top = v1.nlargest(15, "gain_pct")[["feature_pt", "gain_pct"]]
    v2_top = v2.nlargest(15, "gain_pct")[["feature_pt", "gain_pct"]]

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    bars = axes[0].barh(v1_top["feature_pt"][::-1], v1_top["gain_pct"][::-1], color=COLOR_CHURN)
    axes[0].set_xlabel("gain % (V1 baseline)")
    axes[0].set_title("V1 — 19 features (com leak via imputação)")
    axes[0].set_xlim(0, 50)
    for b, v in zip(bars, v1_top["gain_pct"][::-1]):
        axes[0].text(v + 0.5, b.get_y() + b.get_height()/2, f"{v:.1f}%",
                     va="center", fontsize=9, fontweight="bold")

    bars = axes[1].barh(v2_top["feature_pt"][::-1], v2_top["gain_pct"][::-1], color=COLOR_RETIDO)
    axes[1].set_xlabel("gain % (V2 Optuna + ENRICHED)")
    axes[1].set_title("V2 — 22 features (sem leak, com cohort+sazonalidade)")
    axes[1].set_xlim(0, 50)
    for b, v in zip(bars, v2_top["gain_pct"][::-1]):
        axes[1].text(v + 0.5, b.get_y() + b.get_height()/2, f"{v:.1f}%",
                     va="center", fontsize=9, fontweight="bold")
    fig.suptitle("D14 — Feature importance (gain %) — V1 vs V2 lado a lado", fontsize=16, y=1.01)
    fig.tight_layout()
    save_fig(fig, "D14_feature_importance_compare.png")


def D15_score_quality(df, thr):
    df = df.copy()
    df["scoring_quality_label"] = df["scoring_quality"].map({
        "high_confidence": "Alta confiança\n(multi-event, treinado)",
    }).fillna(df["scoring_quality"])
    df["scoring_quality_label"] = df["scoring_quality_label"].astype(str)
    df.loc[df["scoring_quality_label"].str.contains("single", na=False), "scoring_quality_label"] = "Baixa confiança\n(single-event)"
    df.loc[df["scoring_quality_label"].str.contains("medium", na=False), "scoring_quality_label"] = "Média confiança\n(holdout)"

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    sns.violinplot(data=df, x="scoring_quality_label", y="churn_probability",
                   ax=axes[0], hue="scoring_quality_label", palette="Set2",
                   inner="quartile", legend=False)
    axes[0].axhline(thr, color=COLOR_THR, linestyle="--", label=f"thr={thr:.3f}")
    axes[0].set_xlabel(""); axes[0].set_ylabel("Probabilidade de churn")
    axes[0].set_title("Distribuição do score por qualidade")
    axes[0].legend()

    calib = df.groupby("scoring_quality_label").agg(
        n=("vin_hash", "size"), real=("churned", "mean"), pred=("churn_probability", "mean")
    ).reset_index()
    x = np.arange(len(calib)); w = 0.35
    axes[1].bar(x - w/2, calib["real"], w, label="Churn real", color=COLOR_CHURN)
    axes[1].bar(x + w/2, calib["pred"], w, label="Score médio", color=COLOR_RETIDO)
    axes[1].set_xticks(x); axes[1].set_xticklabels(calib["scoring_quality_label"])
    axes[1].set_ylabel("Taxa")
    axes[1].set_title("Calibração por qualidade (real vs predito)")
    axes[1].legend()
    for i, (r, p, n) in enumerate(zip(calib["real"], calib["pred"], calib["n"])):
        axes[1].text(i - w/2, r + 0.02, f"{r:.2f}\nn={n:,}", ha="center", fontsize=8)
        axes[1].text(i + w/2, p + 0.02, f"{p:.2f}", ha="center", fontsize=8)
    axes[1].set_ylim(0, 1.1)
    fig.suptitle("D15 — Score V2 por qualidade da população", fontsize=15, y=1.02)
    fig.tight_layout()
    save_fig(fig, "D15_score_quality_breakdown.png")


def D16_pyramid(df):
    df = df.copy()
    df["tier"] = pd.cut(df["churn_probability"], bins=[-0.001, 0.2, 0.5, 0.8, 1.001],
                       labels=TIER_ORDER)
    counts = df["tier"].value_counts().reindex(TIER_ORDER)
    rates = df.groupby("tier", observed=True)["churned"].mean().reindex(TIER_ORDER)

    fig, ax = plt.subplots(figsize=(12, 7))
    y_pos = np.arange(len(counts))[::-1]
    bars = ax.barh(y_pos, counts.values, color=PALETTE_TIER, edgecolor="white", linewidth=2)
    ax.set_yticks(y_pos); ax.set_yticklabels(TIER_LABELS)
    ax.set_xlabel("# VINs")
    ax.set_title("D16 — Pirâmide de risco (tier de score V2)", fontsize=15)
    for b, n, r, t in zip(bars, counts.values, rates.values, TIER_ORDER):
        pct = n / counts.sum() * 100
        ax.text(b.get_width() + counts.max() * 0.01,
                b.get_y() + b.get_height()/2,
                f"{n:,} VINs ({pct:.1f}%)   |   churn real = {r:.1%}",
                va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, counts.max() * 1.45)
    fig.tight_layout()
    save_fig(fig, "D16_risk_pyramid.png")


def D17_timeseries_summary():
    ts_csv = REPO_ROOT / "resultados" / "v2" / "analise" / "T10_timeseries_5fold.csv"
    if not ts_csv.exists():
        print("  skip D17: T10_timeseries_5fold.csv missing")
        return
    ts = pd.read_csv(ts_csv)

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    x = ts["fold"]
    axes[0, 0].plot(x, ts["accuracy_cal"], marker="o", lw=2.5, label="Acurácia", color=COLOR_RETIDO)
    axes[0, 0].plot(x, ts["balanced_acc_cal"], marker="s", lw=2.5, label="Acurácia balanceada", color="#55a868")
    axes[0, 0].set_xlabel("Fold (cohorts ↑ no tempo)"); axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_title("Acurácia ao longo dos folds temporais")
    axes[0, 0].set_xticks(x); axes[0, 0].set_ylim(0, 1.05); axes[0, 0].legend()
    for i, (a, b) in enumerate(zip(ts["accuracy_cal"], ts["balanced_acc_cal"])):
        axes[0, 0].annotate(f"{a:.2f}", (x.iloc[i], a), textcoords="offset points",
                             xytext=(0, 8), ha="center", fontsize=8, color=COLOR_RETIDO)

    axes[0, 1].plot(x, ts["roc_auc_cal"], marker="o", lw=2.5, color=COLOR_RETIDO)
    axes[0, 1].fill_between(x, ts["roc_auc_cal"], alpha=0.18, color=COLOR_RETIDO)
    axes[0, 1].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    axes[0, 1].set_xlabel("Fold"); axes[0, 1].set_ylabel("ROC-AUC")
    axes[0, 1].set_title("ROC-AUC por fold")
    axes[0, 1].set_xticks(x); axes[0, 1].set_ylim(0, 1.05)
    for i, v in enumerate(ts["roc_auc_cal"]):
        axes[0, 1].annotate(f"{v:.3f}", (x.iloc[i], v), textcoords="offset points",
                             xytext=(0, 8), ha="center", fontsize=9)

    w = 0.25
    axes[1, 0].bar(x - w, ts["pos_rate_train"], w, label="Train", color=COLOR_RETIDO)
    axes[1, 0].bar(x, ts["pos_rate_val"], w, label="Val", color="#55a868")
    axes[1, 0].bar(x + w, ts["pos_rate_test"], w, label="Test", color=COLOR_CHURN)
    axes[1, 0].set_xticks(x); axes[1, 0].set_xlabel("Fold"); axes[1, 0].set_ylabel("Base rate (% churners)")
    axes[1, 0].set_title("Distribution shift — base rate por fold")
    axes[1, 0].legend()

    axes[1, 1].plot(x, ts["recall_top10pct_cal"], marker="o", lw=2.5, color=COLOR_HIGHLIGHT)
    axes[1, 1].fill_between(x, ts["recall_top10pct_cal"], alpha=0.18, color=COLOR_HIGHLIGHT)
    axes[1, 1].set_xlabel("Fold"); axes[1, 1].set_ylabel("Recall @ top 10%")
    axes[1, 1].set_title("Cobertura operacional (top 10% mais arriscados)")
    axes[1, 1].set_xticks(x); axes[1, 1].set_ylim(0, 1.05)
    for i, v in enumerate(ts["recall_top10pct_cal"]):
        axes[1, 1].annotate(f"{v:.2%}", (x.iloc[i], v), textcoords="offset points",
                             xytext=(0, 8), ha="center", fontsize=9)
    fig.suptitle("D17 — TimeSeriesSplit 5-fold (validação cruzada temporal)", fontsize=16, y=1.01)
    fig.tight_layout()
    save_fig(fig, "D17_timeseries_fold_summary.png")


def D18_top_vins(df):
    cols = ["vin_hash", "model_name", "model_year", "churn_probability",
            "churned", "events_count", "tenure_days", "cluster_name"]
    cols = [c for c in cols if c in df.columns]
    top_risk = df.nlargest(10, "churn_probability")[cols].reset_index(drop=True)
    top_loyal = df.nsmallest(10, "churn_probability")[cols].reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, table, title, color in [(axes[0], top_risk, "Top 10 mais ARRISCADOS (alto churn_probability)", COLOR_CHURN),
                                      (axes[1], top_loyal, "Top 10 mais FIÉIS (baixo churn_probability)", "#55a868")]:
        ax.axis("off")
        ax.set_title(title, fontsize=14, color=color, fontweight="bold")
        table_data = []
        for _, row in table.iterrows():
            vin_short = row["vin_hash"][:12] + "..."
            real = "🔴 SAIU" if row["churned"] == 1 else "🟢 FICOU"
            table_data.append([
                vin_short,
                row["model_name"],
                int(row["model_year"]) if pd.notna(row["model_year"]) else "?",
                f"{row['churn_probability']:.4f}",
                real,
                int(row["events_count"]),
                int(row["tenure_days"]),
                str(row.get("cluster_name", "?"))[:10],
            ])
        col_labels = ["VIN", "Modelo", "Ano", "Score", "Real", "# serv.", "Tenure", "Cluster"]
        tbl = ax.table(cellText=table_data, colLabels=col_labels,
                       cellLoc="center", loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(10)
        tbl.scale(1.0, 2.0)
        for j in range(len(col_labels)):
            tbl[(0, j)].set_facecolor(color); tbl[(0, j)].set_text_props(color="white", weight="bold")
    fig.suptitle("D18 — Amostra qualitativa: 10 VINs nos extremos do score V2", fontsize=15, y=0.98)
    fig.tight_layout()
    save_fig(fig, "D18_top_vins_visual.png")


def main():
    df, metrics, thr, splits, mdl = load_all()
    print(f"  {len(df):,} VINs | threshold={thr:.4f}")
    print(f"\nGerando 18 plots em {OUT.relative_to(REPO_ROOT)}/...\n")

    D01_kde(df, thr)
    D02_violin(df, thr)
    D03_confusion_grid(df, thr)
    D04_roc_pr_zoom(df, splits, mdl)
    D05_calibration(df, splits)
    D06_gain_lift(df)
    D07_threshold_economic(df, thr)
    D08_heatmap_model_tier(df)
    D09_heatmap_cluster_tier(df)
    D10_per_model_metrics(df)
    D11_sales_year(df)
    mean_abs = D12_shap_summary(splits, mdl)
    D13_shap_dependence(splits, mdl, mean_abs)
    D14_feature_importance_compare()
    D15_score_quality(df, thr)
    D16_pyramid(df)
    D17_timeseries_summary()
    D18_top_vins(df)
    print(f"\n✓ Dashboard completo: {len(list(OUT.glob('*.png')))} plots em {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
