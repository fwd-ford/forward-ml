"""TimeSeriesSplit 5-fold cross-validation honesta — rolling-origin sobre sales_date.

Ao contrário do split temporal único do V2 (1 train | 1 val | 1 test), aqui rodamos
5 folds onde cada um:
  1. Treina nas cohorts MAIS ANTIGAS conhecidas até ali
  2. Avalia no PRÓXIMO bloco de cohorts (mais recente que o train)

Dentro de cada fold, o train_fold ainda é dividido em (train_inner | val_inner) por
ordem temporal — o val_inner é usado pro early stopping do XGBoost.

População: apenas multi-event trainable (mesma do V2). Features: ENRICHED_FEATURES (22).
Hyperparâmetros: vindos do Optuna (data/models/xgb_optuna_best.joblib).

Outputs:
  resultados/v2/analise/T10_timeseries_5fold.csv
  resultados/v2/analise/T11_timeseries_aggregate.csv
  resultados/v2/analise/P11_timeseries_metrics_por_fold.png
  resultados/v2/analise/P12_timeseries_base_rate.png
  resultados/v2/analise/P13_timeseries_acuracia_por_fold.png
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

from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
)
from xgboost import XGBClassifier

from src.features import load_features, split_trainable, ENRICHED_FEATURES, model_inverse_weights
from src.classification import find_threshold_max_f1, recall_at_top_k, precision_at_top_k, RANDOM_STATE

N_SPLITS = 5
OUT_DIR = REPO_ROOT / "resultados" / "v2" / "analise"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120


def main():
    print("Carregando features...")
    bundle = load_features()
    df, _ = split_trainable(bundle.df)
    df = df[df["is_single_event"] == 0].copy()
    df = df.dropna(subset=["sales_date"])

    # Ordenar por sales_date — TimeSeriesSplit depende da ordem dos índices
    df = df.sort_values("sales_date").reset_index(drop=True)
    print(f"  população: {len(df):,} multi-event VINs (ordenados por sales_date)")
    print(f"  período: {df['sales_date'].min().date()} a {df['sales_date'].max().date()}")

    X = df[ENRICHED_FEATURES]
    y = df["churned"].to_numpy()
    w = model_inverse_weights(df["model_name"])

    # Hyperparâmetros do Optuna
    pack = joblib.load(REPO_ROOT / "data" / "models" / "xgb_optuna_best.joblib")
    base_params = dict(pack["best_params"])
    base_params.update({
        "n_estimators": 2500,
        "early_stopping_rounds": 60,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
        "device": "cuda",
    })
    print(f"  hyperparams (Optuna): max_depth={base_params['max_depth']}, lr={base_params['learning_rate']:.4f}, "
          f"reg_alpha={base_params['reg_alpha']:.3f}, gamma={base_params['gamma']:.3f}")

    tss = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_rows = []

    for fold_idx, (train_outer_idx, test_idx) in enumerate(tss.split(X)):
        # Dentro do train_outer, separar últimos 20% (ordem temporal) como val para early stopping
        cutoff = int(len(train_outer_idx) * 0.80)
        train_inner_idx = train_outer_idx[:cutoff]
        val_inner_idx = train_outer_idx[cutoff:]

        X_tr = X.iloc[train_inner_idx]; y_tr = y[train_inner_idx]; w_tr = w[train_inner_idx]
        X_va = X.iloc[val_inner_idx];   y_va = y[val_inner_idx];   w_va = w[val_inner_idx]
        X_te = X.iloc[test_idx];        y_te = y[test_idx]

        date_train_start = df.iloc[train_inner_idx[0]]["sales_date"]
        date_train_end = df.iloc[train_inner_idx[-1]]["sales_date"]
        date_val_start = df.iloc[val_inner_idx[0]]["sales_date"]
        date_val_end = df.iloc[val_inner_idx[-1]]["sales_date"]
        date_test_start = df.iloc[test_idx[0]]["sales_date"]
        date_test_end = df.iloc[test_idx[-1]]["sales_date"]

        print(f"\n--- Fold {fold_idx + 1}/{N_SPLITS} ---")
        print(f"  train: n={len(y_tr):,}  {date_train_start.date()} → {date_train_end.date()}")
        print(f"  val:   n={len(y_va):,}  {date_val_start.date()} → {date_val_end.date()}")
        print(f"  test:  n={len(y_te):,}  {date_test_start.date()} → {date_test_end.date()}")
        print(f"  positive_rate — train: {y_tr.mean():.3f} | val: {y_va.mean():.3f} | test: {y_te.mean():.3f}")

        # Treinar XGBoost com Optuna params + early stopping no val_inner
        mdl = XGBClassifier(**base_params)
        mdl.fit(X_tr, y_tr, sample_weight=w_tr,
                eval_set=[(X_va, y_va)], verbose=False)
        best_iter = int(mdl.best_iteration) + 1

        # Calibração isotônica no val_inner
        s_val_raw = mdl.predict_proba(X_va)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(s_val_raw, y_va)
        s_val_cal = iso.transform(s_val_raw)

        # Threshold ótimo F1 no val (calibrado)
        thr_cal, _ = find_threshold_max_f1(y_va, s_val_cal)
        thr_raw, _ = find_threshold_max_f1(y_va, s_val_raw)

        # Avaliação no fold's test
        s_test_raw = mdl.predict_proba(X_te)[:, 1]
        s_test_cal = iso.transform(s_test_raw)
        pred_raw = (s_test_raw >= thr_raw).astype(int)
        pred_cal = (s_test_cal >= thr_cal).astype(int)

        row = {
            "fold": fold_idx + 1,
            "n_train": int(len(y_tr)),
            "n_val": int(len(y_va)),
            "n_test": int(len(y_te)),
            "train_period": f"{date_train_start.date()} → {date_train_end.date()}",
            "val_period": f"{date_val_start.date()} → {date_val_end.date()}",
            "test_period": f"{date_test_start.date()} → {date_test_end.date()}",
            "pos_rate_train": round(float(y_tr.mean()), 4),
            "pos_rate_val": round(float(y_va.mean()), 4),
            "pos_rate_test": round(float(y_te.mean()), 4),
            "best_iteration": best_iter,
            "threshold_raw": round(float(thr_raw), 4),
            "threshold_cal": round(float(thr_cal), 4),

            # raw metrics
            "accuracy_raw": round(accuracy_score(y_te, pred_raw), 4),
            "balanced_acc_raw": round(balanced_accuracy_score(y_te, pred_raw), 4),
            "f1_raw": round(f1_score(y_te, pred_raw, zero_division=0), 4),
            "precision_raw": round(precision_score(y_te, pred_raw, zero_division=0), 4),
            "recall_raw": round(recall_score(y_te, pred_raw, zero_division=0), 4),
            "roc_auc_raw": round(roc_auc_score(y_te, s_test_raw) if len(np.unique(y_te)) > 1 else float("nan"), 4),
            "pr_auc_raw": round(average_precision_score(y_te, s_test_raw) if len(np.unique(y_te)) > 1 else float("nan"), 4),
            "brier_raw": round(brier_score_loss(y_te, s_test_raw), 4),
            "recall_top10pct_raw": round(recall_at_top_k(y_te, s_test_raw, 0.10), 4) if y_te.sum() > 0 else float("nan"),
            "precision_top10pct_raw": round(precision_at_top_k(y_te, s_test_raw, 0.10), 4),

            # calibrated metrics
            "accuracy_cal": round(accuracy_score(y_te, pred_cal), 4),
            "balanced_acc_cal": round(balanced_accuracy_score(y_te, pred_cal), 4),
            "f1_cal": round(f1_score(y_te, pred_cal, zero_division=0), 4),
            "precision_cal": round(precision_score(y_te, pred_cal, zero_division=0), 4),
            "recall_cal": round(recall_score(y_te, pred_cal, zero_division=0), 4),
            "roc_auc_cal": round(roc_auc_score(y_te, s_test_cal) if len(np.unique(y_te)) > 1 else float("nan"), 4),
            "pr_auc_cal": round(average_precision_score(y_te, s_test_cal) if len(np.unique(y_te)) > 1 else float("nan"), 4),
            "brier_cal": round(brier_score_loss(y_te, s_test_cal), 4),
            "recall_top10pct_cal": round(recall_at_top_k(y_te, s_test_cal, 0.10), 4) if y_te.sum() > 0 else float("nan"),
            "precision_top10pct_cal": round(precision_at_top_k(y_te, s_test_cal, 0.10), 4),
        }
        fold_rows.append(row)
        print(f"  ↪ accuracy_cal={row['accuracy_cal']:.4f} | balanced={row['balanced_acc_cal']:.4f} | "
              f"f1={row['f1_cal']:.4f} | roc_auc={row['roc_auc_cal']:.4f} | recall@top10={row['recall_top10pct_cal']:.4f}")

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(OUT_DIR / "T10_timeseries_5fold.csv", index=False)
    print(f"\n✓ Escrito {OUT_DIR / 'T10_timeseries_5fold.csv'}")

    # ===== Agregado =====
    metric_cols = [c for c in folds_df.columns if c not in (
        "fold", "n_train", "n_val", "n_test", "train_period", "val_period", "test_period",
        "best_iteration", "threshold_raw", "threshold_cal"
    )]
    agg_rows = []
    for col in metric_cols:
        vals = folds_df[col].dropna()
        if len(vals) == 0:
            continue
        agg_rows.append({
            "metrica": col,
            "mean": round(vals.mean(), 4),
            "std": round(vals.std(), 4),
            "min": round(vals.min(), 4),
            "max": round(vals.max(), 4),
        })
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(OUT_DIR / "T11_timeseries_aggregate.csv", index=False)
    print(f"✓ Escrito {OUT_DIR / 'T11_timeseries_aggregate.csv'}")

    # ===== Plots =====
    # P11 — métricas-chave por fold (acurácia, F1, ROC-AUC)
    fig, ax = plt.subplots(figsize=(14, 7))
    x = folds_df["fold"]
    ax.plot(x, folds_df["accuracy_cal"], marker="o", lw=2.5, label="Acurácia", color="#4c72b0")
    ax.plot(x, folds_df["balanced_acc_cal"], marker="s", lw=2.5, label="Acurácia balanceada", color="#55a868")
    ax.plot(x, folds_df["f1_cal"], marker="^", lw=2.5, label="F1", color="#c44e52")
    ax.plot(x, folds_df["roc_auc_cal"], marker="D", lw=2.5, label="ROC-AUC", color="#dd8452")
    ax.plot(x, folds_df["recall_top10pct_cal"], marker="v", lw=2.5, label="Recall @ top 10%", color="#8172b3")
    ax.set_xlabel("Fold (cohorts mais antigas → mais recentes)")
    ax.set_ylabel("Score")
    ax.set_title("TimeSeriesSplit 5-fold — métricas por fold (calibrado)")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower center", ncol=3)
    for i, (a, ba) in enumerate(zip(folds_df["accuracy_cal"], folds_df["balanced_acc_cal"])):
        ax.text(x.iloc[i], a + 0.02, f"{a:.2f}", ha="center", fontsize=9, color="#4c72b0")
        ax.text(x.iloc[i], ba - 0.04, f"{ba:.2f}", ha="center", fontsize=9, color="#55a868")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "P11_timeseries_metrics_por_fold.png", bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Escrito {OUT_DIR / 'P11_timeseries_metrics_por_fold.png'}")

    # P12 — base rate por fold (positive rate de train/val/test)
    fig, ax = plt.subplots(figsize=(14, 6))
    x = folds_df["fold"]
    w = 0.25
    ax.bar(x - w, folds_df["pos_rate_train"], w, label="Train", color="#4c72b0")
    ax.bar(x, folds_df["pos_rate_val"], w, label="Val", color="#55a868")
    ax.bar(x + w, folds_df["pos_rate_test"], w, label="Test", color="#c44e52")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Base rate (% de churners no subset)")
    ax.set_title("Distribution shift entre folds — taxa de churn por subset")
    ax.set_xticks(x)
    ax.legend()
    for i in range(len(folds_df)):
        ax.text(x.iloc[i] - w, folds_df["pos_rate_train"].iloc[i] + 0.01,
                f"{folds_df['pos_rate_train'].iloc[i]:.1%}", ha="center", fontsize=8)
        ax.text(x.iloc[i], folds_df["pos_rate_val"].iloc[i] + 0.01,
                f"{folds_df['pos_rate_val'].iloc[i]:.1%}", ha="center", fontsize=8)
        ax.text(x.iloc[i] + w, folds_df["pos_rate_test"].iloc[i] + 0.01,
                f"{folds_df['pos_rate_test'].iloc[i]:.1%}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "P12_timeseries_base_rate.png", bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Escrito {OUT_DIR / 'P12_timeseries_base_rate.png'}")

    # P13 — accuracy raw vs cal por fold
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].plot(x, folds_df["accuracy_raw"], marker="o", lw=2.5, label="Raw", color="#c44e52")
    axes[0].plot(x, folds_df["accuracy_cal"], marker="s", lw=2.5, label="Calibrado", color="#4c72b0")
    axes[0].set_xticks(x); axes[0].set_xlabel("Fold"); axes[0].set_ylabel("Acurácia")
    axes[0].set_title("Acurácia por fold — raw vs calibrado")
    axes[0].legend(); axes[0].set_ylim(0, 1.05)
    for i in range(len(folds_df)):
        axes[0].text(x.iloc[i], folds_df["accuracy_raw"].iloc[i] - 0.04,
                     f"{folds_df['accuracy_raw'].iloc[i]:.2f}", ha="center", fontsize=9, color="#c44e52")
        axes[0].text(x.iloc[i], folds_df["accuracy_cal"].iloc[i] + 0.02,
                     f"{folds_df['accuracy_cal'].iloc[i]:.2f}", ha="center", fontsize=9, color="#4c72b0")

    axes[1].plot(x, folds_df["balanced_acc_raw"], marker="o", lw=2.5, label="Raw", color="#c44e52")
    axes[1].plot(x, folds_df["balanced_acc_cal"], marker="s", lw=2.5, label="Calibrado", color="#4c72b0")
    axes[1].set_xticks(x); axes[1].set_xlabel("Fold"); axes[1].set_ylabel("Acurácia balanceada")
    axes[1].set_title("Acurácia balanceada por fold — raw vs calibrado")
    axes[1].legend(); axes[1].set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "P13_timeseries_acuracia_por_fold.png", bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Escrito {OUT_DIR / 'P13_timeseries_acuracia_por_fold.png'}")

    print("\n=== Resumo dos folds ===")
    summary_cols = ["fold", "n_train", "n_test", "pos_rate_train", "pos_rate_test",
                    "accuracy_cal", "balanced_acc_cal", "f1_cal", "roc_auc_cal", "recall_top10pct_cal"]
    print(folds_df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
