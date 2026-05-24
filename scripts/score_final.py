"""Score all VINs using the V3 production scorer + write final consolidated CSV +
error analysis report.

Outputs (everything in resultados/):
  scoring_results.csv         - 175k VINs com score V3 (replaces V2)
  metrics.json                - V3 metrics
  analise_erro.csv            - per-segment error breakdown
  RESUMO_FINAL.md             - executive summary
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
import joblib
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, brier_score_loss,
)

from src.features import load_features, KMEANS_FEATURES, CHURN_THRESHOLD_DAYS
from src.feature_names import FRIENDLY_NAMES
from src.inference import ChurnScorer

RESULTS = REPO_ROOT / "resultados"


def main():
    print("Loading features + V3 scorer...")
    bundle = load_features()
    df = bundle.df.copy()
    scorer = ChurnScorer.load()
    print(f"  scorer metadata: {scorer.artifacts.metadata}")

    print("\nScoring 175k VINs (two-stage + stratified calibration)...")
    scored = scorer.predict(df)

    # K-means cluster
    km_pack = joblib.load(REPO_ROOT / "data" / "models" / "kmeans_segmentation.joblib")
    X_km = df[KMEANS_FEATURES].fillna(df[KMEANS_FEATURES].median())
    cluster_id = km_pack["model"].predict(km_pack["scaler"].transform(X_km))
    cluster_name = pd.Series(cluster_id).map(km_pack["cluster_names"]).values
    scored["cluster_id"] = cluster_id
    scored["cluster_name"] = cluster_name

    # ---- Output CSV ----
    out_cols = [
        "vin_hash", "model_name", "model_year", "sales_date", "last_service_date",
        "trainable", "is_single_event", "churned",
        "churn_probability_raw", "churn_probability", "score_fidelidade",
        "churn_predicted", "churn_risk_tier", "churn_decile",
        "cluster_id", "cluster_name",
        "events_count", "tenure_days", "days_since_last_service",
        "km_max", "primary_dealer_share", "scoring_quality",
    ]
    out_cols = [c for c in out_cols if c in scored.columns]
    out = scored[out_cols].copy()
    for c in ("sales_date", "last_service_date"):
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.date

    csv_path = RESULTS / "scoring_results.csv"
    out.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path} ({len(out):,} rows)")

    # PT mirror
    pt_path = RESULTS / "scoring_results_pt.csv"
    out.rename(columns=FRIENDLY_NAMES).rename(columns={
        "score_fidelidade": "Score de fidelidade (1 - churn_prob)"
    }).to_csv(pt_path, index=False)
    print(f"Wrote {pt_path}")

    # ---- Metrics ----
    metrics = {"model": "churn_scorer_v3", "scorer_metadata": scorer.artifacts.metadata,
               "by_quality": {}, "by_cohort": {}, "global": {}, "lag_days": CHURN_THRESHOLD_DAYS}

    for q, sub in scored.groupby("scoring_quality"):
        if sub["churned"].nunique() < 2:
            continue
        y = sub["churned"].to_numpy()
        s = sub["churn_probability"].to_numpy()
        pred = sub["churn_predicted"].to_numpy()
        metrics["by_quality"][q] = {
            "n": int(len(sub)),
            "churn_real": round(float(y.mean()), 4),
            "score_medio": round(float(s.mean()), 4),
            "accuracy": round(accuracy_score(y, pred), 4),
            "balanced_accuracy": round(balanced_accuracy_score(y, pred), 4),
            "precision": round(precision_score(y, pred, zero_division=0), 4),
            "recall": round(recall_score(y, pred, zero_division=0), 4),
            "f1": round(f1_score(y, pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y, s), 4),
            "pr_auc": round(average_precision_score(y, s), 4),
            "brier": round(brier_score_loss(y, s), 4),
        }

    scored["sales_year"] = pd.to_datetime(scored["sales_date"], errors="coerce").dt.year
    for yr, sub in scored.groupby("sales_year"):
        if pd.isna(yr) or sub["churned"].nunique() < 2:
            continue
        y = sub["churned"].to_numpy(); s = sub["churn_probability"].to_numpy()
        metrics["by_cohort"][str(int(yr))] = {
            "n": int(len(sub)),
            "churn_real": round(float(y.mean()), 4),
            "score_medio": round(float(s.mean()), 4),
            "roc_auc": round(roc_auc_score(y, s), 4) if y.mean() not in (0, 1) else None,
            "balanced_accuracy": round(balanced_accuracy_score(y, sub["churn_predicted"].to_numpy()), 4),
        }

    y = scored["churned"].to_numpy(); s = scored["churn_probability"].to_numpy()
    pred = scored["churn_predicted"].to_numpy()
    metrics["global"] = {
        "n": int(len(scored)),
        "churn_real": round(float(y.mean()), 4),
        "accuracy": round(accuracy_score(y, pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, s), 4),
        "pr_auc": round(average_precision_score(y, s), 4),
        "brier": round(brier_score_loss(y, s), 4),
    }

    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {RESULTS / 'metrics.json'}")

    # ---- Error Analysis ----
    print("\nBuilding error analysis...")
    scored["acertou"] = (scored["churn_predicted"] == scored["churned"]).astype(int)
    scored["erro_tipo"] = np.where(
        scored["acertou"] == 1, "acerto",
        np.where(
            (scored["churn_predicted"] == 1) & (scored["churned"] == 0),
            "FP (falso alarme)",
            "FN (perdeu churner)",
        ),
    )

    erro_por_modelo = (scored.groupby("model_name")
                       .agg(n=("vin_hash", "size"),
                            acuracia=("acertou", "mean"),
                            pct_FP=("erro_tipo", lambda x: (x == "FP (falso alarme)").mean()),
                            pct_FN=("erro_tipo", lambda x: (x == "FN (perdeu churner)").mean()))
                       .sort_values("n", ascending=False).head(15).round(4).reset_index())

    erro_por_cohort = (scored.dropna(subset=["sales_year"])
                       .groupby("sales_year")
                       .agg(n=("vin_hash", "size"),
                            acuracia=("acertou", "mean"),
                            pct_FP=("erro_tipo", lambda x: (x == "FP (falso alarme)").mean()),
                            pct_FN=("erro_tipo", lambda x: (x == "FN (perdeu churner)").mean()))
                       .round(4).reset_index())
    erro_por_cohort["sales_year"] = erro_por_cohort["sales_year"].astype(int)

    erro_por_quality = (scored.groupby("scoring_quality")
                        .agg(n=("vin_hash", "size"),
                             acuracia=("acertou", "mean"),
                             pct_FP=("erro_tipo", lambda x: (x == "FP (falso alarme)").mean()),
                             pct_FN=("erro_tipo", lambda x: (x == "FN (perdeu churner)").mean()))
                        .round(4).reset_index())

    with open(RESULTS / "analise_erro.csv", "w", encoding="utf-8") as f:
        f.write("# Erro por MODELO DE CARRO (top 15 por volume)\n")
        erro_por_modelo.to_csv(f, index=False)
        f.write("\n# Erro por COHORT DE VENDA (sales_year)\n")
        erro_por_cohort.to_csv(f, index=False)
        f.write("\n# Erro por SCORING QUALITY\n")
        erro_por_quality.to_csv(f, index=False)
    print(f"Wrote {RESULTS / 'analise_erro.csv'}")

    # ---- Summary ----
    md = ["# Resultado final — V3 (multi-stage + cohort-aware + stratified calibration)\n"]
    md.append(f"## Métricas globais (175k VINs)\n")
    for k, v in metrics["global"].items():
        md.append(f"- **{k}**: {v}")
    md.append("\n## Métricas por scoring_quality\n")
    md.append("| Quality | n | acc | balanced_acc | F1 | ROC-AUC | Brier |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for q, m in metrics["by_quality"].items():
        md.append(f"| {q} | {m['n']:,} | {m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} | {m['f1']:.3f} | {m['roc_auc']:.3f} | {m['brier']:.3f} |")
    md.append("\n## Métricas por cohort de venda\n")
    md.append("| Ano | n | churn_real | score | ROC-AUC | balanced_acc |")
    md.append("|---:|---:|---:|---:|---:|---:|")
    for yr, m in metrics["by_cohort"].items():
        md.append(f"| {yr} | {m['n']:,} | {m['churn_real']:.3f} | {m['score_medio']:.3f} | {m.get('roc_auc') or 'n/a'} | {m['balanced_accuracy']:.3f} |")
    md.append("\n## Resumo dos erros (top 5 modelos com pior acc)\n")
    md.append("| Modelo | n | Acc | %FP | %FN |")
    md.append("|---|---:|---:|---:|---:|")
    worst = erro_por_modelo.nsmallest(5, "acuracia")
    for _, r in worst.iterrows():
        md.append(f"| {r['model_name']} | {int(r['n']):,} | {r['acuracia']:.3f} | {r['pct_FP']:.3f} | {r['pct_FN']:.3f} |")
    (RESULTS / "RESUMO_FINAL.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {RESULTS / 'RESUMO_FINAL.md'}")

    print("\n=== RESULTADO GLOBAL ===")
    for k, v in metrics["global"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
