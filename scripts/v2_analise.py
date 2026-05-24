"""Pacote de análise rico do modelo V2 (Optuna + calibração isotônica).

Lê resultados/v2/scoring_results_v2.csv (já gerado pelo generate_scoring_v2.py)
e produz plots + tabelas exploratórias em resultados/v2/analise/.

Tabelas CSV produzidas:
  T01_score_por_modelo.csv           - distribuição por model_name
  T02_score_por_cluster.csv          - distribuição por cluster K-means
  T03_score_por_cohort.csv           - distribuição por sales_year
  T04_threshold_sweep.csv            - precision/recall/F1/n_alertas em vários thresholds
  T05_top100_mais_arriscados.csv     - amostra dos 100 VINs com maior score
  T06_top100_mais_fieis.csv          - amostra dos 100 VINs com menor score
  T07_calibracao_por_populacao.csv   - high_conf vs single_event vs holdout
  T08_confusao_thresholds.csv        - matriz de confusão em 5 thresholds chave
  T09_resumo_geral.csv               - estatísticas-resumo

Plots PNG produzidos:
  P01_distribuicao_score.png         - histograma do score (com tiers marcados)
  P02_score_por_modelo.png           - boxplot por modelo de carro
  P03_score_por_cluster.png          - boxplot por cluster K-means
  P04_score_por_cohort.png           - score médio por ano de venda
  P05_gain_curve.png                 - curva de ganho (recall vs % contatado)
  P06_lift_chart.png                 - lift por decile vs random
  P07_roc_pr_curves.png              - ROC e PR curves no test
  P08_threshold_sweep.png            - precision/recall/F1 vs threshold
  P09_calibracao_por_populacao.png   - calibração estratificada
  P10_heatmap_modelo_x_tier.png      - heatmap quantidade por modelo × tier
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score, average_precision_score,
    confusion_matrix, f1_score, precision_score, recall_score,
)

from src.feature_names import FRIENDLY_NAMES

OUT_DIR = REPO_ROOT / "resultados" / "v2" / "analise"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120


def save_fig(fig, name):
    p = OUT_DIR / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.relative_to(REPO_ROOT)}")


def save_csv(df, name):
    p = OUT_DIR / name
    df.to_csv(p, index=False, encoding="utf-8")
    print(f"  wrote {p.relative_to(REPO_ROOT)}")


def main():
    print("Carregando V2 scoring + métricas...")
    df = pd.read_csv(REPO_ROOT / "resultados" / "v2" / "scoring_results_v2.csv",
                     parse_dates=["sales_date", "last_service_date"])
    metrics = json.loads((REPO_ROOT / "resultados" / "v2" / "v2_metrics.json").read_text(encoding="utf-8"))
    thr_cal = metrics["test_calibrated"]["best_threshold_from_val"]
    print(f"  {len(df):,} VINs | threshold calibrado = {thr_cal:.3f}")

    # ========== TABELAS ==========

    # T01 — por model_name
    t01 = (df.groupby("model_name")
             .agg(n=("vin_hash", "size"),
                  churn_real=("churned", "mean"),
                  score_medio=("churn_probability", "mean"),
                  score_p25=("churn_probability", lambda x: x.quantile(0.25)),
                  score_p75=("churn_probability", lambda x: x.quantile(0.75)))
             .round(4)
             .sort_values("n", ascending=False)
             .reset_index())
    t01["pct_total"] = (t01["n"] / t01["n"].sum() * 100).round(1)
    save_csv(t01, "T01_score_por_modelo.csv")

    # T02 — por cluster
    t02 = (df.groupby("cluster_name")
             .agg(n=("vin_hash", "size"),
                  churn_real=("churned", "mean"),
                  score_medio=("churn_probability", "mean"),
                  score_p25=("churn_probability", lambda x: x.quantile(0.25)),
                  score_p75=("churn_probability", lambda x: x.quantile(0.75)))
             .round(4)
             .reset_index())
    t02["pct_total"] = (t02["n"] / t02["n"].sum() * 100).round(1)
    save_csv(t02, "T02_score_por_cluster.csv")

    # T03 — por cohort de venda (sales_year)
    df["sales_year"] = pd.to_datetime(df["sales_date"], errors="coerce").dt.year
    t03 = (df.dropna(subset=["sales_year"])
             .groupby("sales_year")
             .agg(n=("vin_hash", "size"),
                  churn_real=("churned", "mean"),
                  score_medio=("churn_probability", "mean"))
             .round(4)
             .reset_index())
    t03["sales_year"] = t03["sales_year"].astype(int)
    save_csv(t03, "T03_score_por_cohort.csv")

    # T04 — threshold sweep
    y_true = df["churned"].to_numpy()
    y_score = df["churn_probability"].to_numpy()
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, round(thr_cal, 3), 0.70, 0.80, 0.90]
    rows = []
    for t in sorted(set(thresholds)):
        pred = (y_score >= t).astype(int)
        n_alertas = int(pred.sum())
        if n_alertas == 0:
            continue
        rows.append({
            "threshold": round(t, 3),
            "n_alertas": n_alertas,
            "pct_alertados": round(n_alertas / len(pred) * 100, 1),
            "precision": round(precision_score(y_true, pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, pred, zero_division=0), 4),
            "f1": round(f1_score(y_true, pred, zero_division=0), 4),
            "tp": int(((pred == 1) & (y_true == 1)).sum()),
            "fp": int(((pred == 1) & (y_true == 0)).sum()),
            "fn": int(((pred == 0) & (y_true == 1)).sum()),
            "tn": int(((pred == 0) & (y_true == 0)).sum()),
        })
    t04 = pd.DataFrame(rows)
    save_csv(t04, "T04_threshold_sweep.csv")

    # T05 — top 100 mais arriscados
    cols_pick = ["vin_hash", "model_name", "model_year", "churn_probability", "score_fidelidade",
                 "churned", "events_count", "tenure_days", "days_since_last_service",
                 "primary_dealer_share", "cluster_name", "scoring_quality"]
    cols_pick = [c for c in cols_pick if c in df.columns]
    t05 = df.nlargest(100, "churn_probability")[cols_pick].reset_index(drop=True)
    save_csv(t05, "T05_top100_mais_arriscados.csv")

    # T06 — top 100 mais fiéis (menor risco)
    t06 = df.nsmallest(100, "churn_probability")[cols_pick].reset_index(drop=True)
    save_csv(t06, "T06_top100_mais_fieis.csv")

    # T07 — calibração por população (scoring_quality)
    t07 = (df.groupby("scoring_quality")
             .agg(n=("vin_hash", "size"),
                  churn_real=("churned", "mean"),
                  score_medio=("churn_probability", "mean"),
                  diff_calibracao=("churn_probability", lambda x: 0))
             .round(4)
             .reset_index())
    t07["diff_calibracao"] = (t07["score_medio"] - t07["churn_real"]).round(4)
    save_csv(t07, "T07_calibracao_por_populacao.csv")

    # T08 — confusão em 5 thresholds chave
    cm_rows = []
    for t in [0.20, 0.30, 0.50, round(thr_cal, 3), 0.80]:
        pred = (y_score >= t).astype(int)
        cm = confusion_matrix(y_true, pred)
        cm_rows.append({
            "threshold": round(t, 3),
            "TN_retidos_corretos": int(cm[0, 0]),
            "FP_retidos_marcados_saindo": int(cm[0, 1]),
            "FN_saindo_nao_alertados": int(cm[1, 0]),
            "TP_saindo_alertados": int(cm[1, 1]),
            "precision": round(precision_score(y_true, pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        })
    t08 = pd.DataFrame(cm_rows)
    save_csv(t08, "T08_confusao_thresholds.csv")

    # T09 — resumo geral
    t09_data = {
        "métrica": [
            "n_total_VINs", "churn_real_global",
            "score_medio", "score_mediana",
            "n_high_confidence", "n_low_confidence_single_event", "n_medium_holdout",
            "n_very_low_0_a_0.2", "n_low_0.2_a_0.5", "n_high_0.5_a_0.8", "n_very_high_0.8_a_1",
            "vins_zona_cinza_0.2_a_0.8",
            "threshold_otimizado_F1",
            "test_roc_auc", "test_pr_auc", "test_brier", "test_recall_top10pct",
        ],
        "valor": [
            len(df), round(df["churned"].mean(), 4),
            round(df["churn_probability"].mean(), 4), round(df["churn_probability"].median(), 4),
            int((df["scoring_quality"] == "high_confidence").sum()),
            int(df["scoring_quality"].astype(str).str.contains("single").sum()),
            int(df["scoring_quality"].astype(str).str.contains("medium").sum()),
            int(((df["churn_probability"] <= 0.2)).sum()),
            int(((df["churn_probability"] > 0.2) & (df["churn_probability"] <= 0.5)).sum()),
            int(((df["churn_probability"] > 0.5) & (df["churn_probability"] <= 0.8)).sum()),
            int(((df["churn_probability"] > 0.8)).sum()),
            int(((df["churn_probability"] > 0.2) & (df["churn_probability"] < 0.8)).sum()),
            round(thr_cal, 4),
            round(metrics["test_calibrated"]["roc_auc"], 4),
            round(metrics["test_calibrated"]["pr_auc"], 4),
            round(metrics["test_calibrated"]["brier"], 4),
            round(metrics["test_calibrated"]["recall_top10pct"], 4),
        ],
    }
    t09 = pd.DataFrame(t09_data)
    save_csv(t09, "T09_resumo_geral.csv")

    # ========== PLOTS ==========

    # P01 — distribuição do score com tiers marcados
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.hist(df["churn_probability"], bins=60, color="#4c72b0", edgecolor="white")
    for v, lbl, c in [(0.2, "very_low|low", "#55a868"),
                       (0.5, "low|high", "#ddb142"),
                       (0.8, "high|very_high", "#c44e52"),
                       (thr_cal, f"threshold ótimo ({thr_cal:.3f})", "black")]:
        ax.axvline(v, color=c, linestyle="--", alpha=0.7, label=lbl)
    ax.set_xlabel("Probabilidade de churn (V2 calibrado)")
    ax.set_ylabel("# VINs")
    ax.set_title("V2 — Distribuição global do score")
    ax.legend(loc="upper center")
    fig.tight_layout()
    save_fig(fig, "P01_distribuicao_score.png")

    # P02 — boxplot por modelo (top 10)
    top_models = df["model_name"].value_counts().head(10).index.tolist()
    sub = df[df["model_name"].isin(top_models)]
    order = sub.groupby("model_name")["churn_probability"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=sub, x="model_name", y="churn_probability", order=order,
                ax=ax, hue="model_name", palette="RdYlBu_r", legend=False)
    ax.axhline(thr_cal, color="black", linestyle="--", alpha=0.6)
    ax.set_title("V2 — Score por modelo de carro (top 10 por volume)")
    ax.set_xlabel("Modelo"); ax.set_ylabel("Probabilidade de churn")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    save_fig(fig, "P02_score_por_modelo.png")

    # P03 — boxplot por cluster
    fig, ax = plt.subplots(figsize=(12, 6))
    order = df.groupby("cluster_name")["churn_probability"].median().sort_values().index
    sns.boxplot(data=df, x="cluster_name", y="churn_probability", order=order,
                ax=ax, hue="cluster_name", palette="RdYlBu_r", legend=False)
    ax.axhline(thr_cal, color="black", linestyle="--", alpha=0.6, label=f"thr={thr_cal:.3f}")
    ax.set_title("V2 — Score por cluster K-means")
    ax.set_xlabel("Cluster"); ax.set_ylabel("Probabilidade de churn")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "P03_score_por_cluster.png")

    # P04 — cohort por sales_year
    cohort_df = df.dropna(subset=["sales_year"]).groupby("sales_year").agg(
        n=("vin_hash", "size"), churn_real=("churned", "mean"),
        score_medio=("churn_probability", "mean")
    ).reset_index()
    cohort_df["sales_year"] = cohort_df["sales_year"].astype(int)
    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax2 = ax1.twinx()
    ax1.bar(cohort_df["sales_year"], cohort_df["n"], color="#cccccc", alpha=0.5, label="# VINs")
    ax2.plot(cohort_df["sales_year"], cohort_df["churn_real"], marker="o", color="#c44e52", lw=2.5, label="Churn real")
    ax2.plot(cohort_df["sales_year"], cohort_df["score_medio"], marker="s", color="#4c72b0", lw=2.5, label="Score V2")
    ax1.set_xlabel("Ano da venda"); ax1.set_ylabel("# VINs (barras)")
    ax2.set_ylabel("Taxa (linhas)"); ax2.set_ylim(0, 1)
    ax1.set_title("V2 — Por cohort de venda: volume + churn real + score predito")
    ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
    fig.tight_layout()
    save_fig(fig, "P04_score_por_cohort.png")

    # P05 — gain curve
    sorted_df = df.sort_values("churn_probability", ascending=False).reset_index(drop=True)
    cum_pos = sorted_df["churned"].cumsum()
    total_pos = sorted_df["churned"].sum()
    frac_contacted = (np.arange(len(sorted_df)) + 1) / len(sorted_df)
    gain = cum_pos / total_pos

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(frac_contacted, gain, color="#4c72b0", lw=2.5, label="Modelo V2")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Aleatório (baseline)")
    ax.fill_between(frac_contacted, gain, frac_contacted, alpha=0.2, color="#4c72b0")
    for q in [0.05, 0.10, 0.20, 0.30, 0.50]:
        idx = int(len(sorted_df) * q) - 1
        ax.axvline(q, color="gray", alpha=0.3, linestyle=":")
        ax.text(q + 0.005, gain.iloc[idx] - 0.04, f"top {int(q*100)}% → captura {gain.iloc[idx]:.0%}", fontsize=10)
    ax.set_xlabel("Fração de VINs contatados (ordenados pelo score)")
    ax.set_ylabel("Recall (% dos churners reais cobertos)")
    ax.set_title("V2 — Curva de ganho")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_fig(fig, "P05_gain_curve.png")

    # P06 — lift por decile
    df["decile_int"] = pd.qcut(-df["churn_probability"], q=10, labels=False, duplicates="drop")
    df["decile"] = "D" + (df["decile_int"] + 1).astype(int).astype(str)
    base_rate = df["churned"].mean()
    lift_df = df.groupby("decile_int").agg(
        churn_real=("churned", "mean"), n=("vin_hash", "size")
    ).reset_index()
    lift_df["lift"] = lift_df["churn_real"] / base_rate
    lift_df["decile_label"] = "D" + (lift_df["decile_int"] + 1).astype(int).astype(str)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#c44e52" if i < 3 else "#ddb142" if i < 7 else "#55a868"
              for i in lift_df["decile_int"]]
    ax.bar(lift_df["decile_label"], lift_df["lift"], color=colors)
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="baseline (random = 1.0)")
    ax.set_xlabel("Decile do score (D1=maior risco, D10=menor risco)")
    ax.set_ylabel("Lift vs. random")
    ax.set_title("V2 — Lift por decile (quantas vezes mais churners que o baseline aleatório)")
    for i, (l, n) in enumerate(zip(lift_df["lift"], lift_df["n"])):
        ax.text(i, l + 0.03, f"{l:.2f}×\nn={n:,}", ha="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "P06_lift_chart.png")

    # P07 — ROC + PR (no test apenas)
    # Carregar test via re-split
    from src.features import load_features, split_trainable, ENRICHED_FEATURES
    from src.classification import temporal_val_split
    bundle = load_features()
    df_train_all, _ = split_trainable(bundle.df)
    df_multi = df_train_all[df_train_all["is_single_event"] == 0]
    splits = temporal_val_split(df_multi, features=ENRICHED_FEATURES, val_quantile=0.64, test_quantile=0.80)
    # Reconstruir scoring no test: vamos usar o score que está no df de scoring v2
    # casado pelas datas do split (sales_date > cutoff_test)
    cutoff_test = splits["cutoff_test"]
    test_mask = (df["sales_date"] > cutoff_test) & (df["is_single_event"] == 0)
    y_t = df.loc[test_mask, "churned"].to_numpy()
    s_t = df.loc[test_mask, "churn_probability"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fpr, tpr, _ = roc_curve(y_t, s_t)
    axes[0].plot(fpr, tpr, color="#4c72b0", lw=2.5,
                 label=f"V2 (AUC={roc_auc_score(y_t, s_t):.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="aleatório")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].set_title("V2 — ROC (test temporal)")
    axes[0].legend(loc="lower right")

    p, r, _ = precision_recall_curve(y_t, s_t)
    axes[1].plot(r, p, color="#c44e52", lw=2.5,
                 label=f"V2 (AP={average_precision_score(y_t, s_t):.4f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title(f"V2 — Precision-Recall (test, pos_rate={y_t.mean():.3f})")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    save_fig(fig, "P07_roc_pr_curves.png")

    # P08 — threshold sweep
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(t04["threshold"], t04["precision"], marker="o", label="Precision", color="#4c72b0")
    ax.plot(t04["threshold"], t04["recall"], marker="s", label="Recall", color="#c44e52")
    ax.plot(t04["threshold"], t04["f1"], marker="^", label="F1", color="#55a868")
    ax.axvline(thr_cal, color="black", linestyle="--", alpha=0.5, label=f"threshold ótimo F1={thr_cal:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("V2 — Precision / Recall / F1 vs threshold (dataset todo)")
    ax.legend(loc="lower center")
    fig.tight_layout()
    save_fig(fig, "P08_threshold_sweep.png")

    # P09 — calibração por população
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(figsize=(10, 8))
    pops = [
        ("high_confidence", "high_confidence", "#4c72b0"),
        ("single-event", "single_event", "#dd8452"),
        ("medium", "holdout (tenure<365d)", "#c44e52"),
    ]
    for substr, label, color in pops:
        mask = df["scoring_quality"].astype(str).str.contains(substr)
        if mask.sum() < 100:
            continue
        try:
            fp, mp = calibration_curve(
                df.loc[mask, "churned"], df.loc[mask, "churn_probability"],
                n_bins=10, strategy="quantile"
            )
            ax.plot(mp, fp, marker="o", label=f"{label} (n={mask.sum():,})", color=color)
        except Exception as e:
            print(f"  calibration_curve skipped for {label}: {e}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfeito")
    ax.set_xlabel("Score médio predito (em cada bin)")
    ax.set_ylabel("Fração real de churners no bin")
    ax.set_title("V2 — Calibração estratificada por população")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_fig(fig, "P09_calibracao_por_populacao.png")

    # P10 — heatmap modelo × tier
    df["tier"] = pd.cut(df["churn_probability"], bins=[-0.001, 0.2, 0.5, 0.8, 1.001],
                       labels=["very_low", "low", "high", "very_high"])
    heat = df[df["model_name"].isin(top_models)].pivot_table(
        index="model_name", columns="tier", values="vin_hash", aggfunc="count", fill_value=0, observed=True
    )
    heat = heat.reindex(top_models)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(heat, annot=True, fmt="d", cmap="YlOrRd", ax=ax, cbar_kws={"label": "# VINs"})
    ax.set_title("V2 — Distribuição de VINs por modelo × tier de risco")
    ax.set_xlabel("Tier de risco"); ax.set_ylabel("Modelo")
    fig.tight_layout()
    save_fig(fig, "P10_heatmap_modelo_x_tier.png")

    print(f"\n✓ Análise V2 completa em {OUT_DIR.relative_to(REPO_ROOT)}")
    print(f"  {len(list(OUT_DIR.glob('T*.csv')))} tabelas, {len(list(OUT_DIR.glob('P*.png')))} plots")


if __name__ == "__main__":
    main()
