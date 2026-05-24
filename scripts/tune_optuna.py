"""Bayesian hyperparameter tuning of XGBoost focused on temporal PR-AUC.

Uses ENRICHED_FEATURES (trajectory + seasonality + cohort, leak-safe) and the proper
3-way temporal split (train | val | test) so that:
  - Optuna optimizes PR-AUC on the val set (middle cohort)
  - Final reported metric is on the test set (newest cohort) — never seen by the tuner

Excludes single-event VINs from train+val (target leak via imputation in gaps);
test still includes them — that mirrors production scoring.

Outputs:
  - data/models/xgb_optuna_best.joblib
  - resultados/optuna_trials.csv
  - resultados/plots/optuna_history.png, optuna_param_importance.png
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
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    f1_score, precision_score, recall_score,
)
from xgboost import XGBClassifier

from src.features import load_features, split_trainable, ENRICHED_FEATURES
from src.classification import (
    temporal_val_split, find_threshold_max_f1, recall_at_top_k, precision_at_top_k,
    RANDOM_STATE,
)

N_TRIALS = 40
OUT_MODEL = REPO_ROOT / "data" / "models" / "xgb_optuna_best.joblib"
OUT_TRIALS = REPO_ROOT / "resultados" / "optuna_trials.csv"
OUT_HISTORY = REPO_ROOT / "resultados" / "plots" / "optuna_history.png"
OUT_IMPORT = REPO_ROOT / "resultados" / "plots" / "optuna_param_importance.png"
OUT_HISTORY.parent.mkdir(parents=True, exist_ok=True)


def make_splits():
    """Controlled population — multi-event VINs only across train/val/test, so the
    distribution shift between splits comes from time (cohort), not from population mix."""
    bundle = load_features()
    df, _ = split_trainable(bundle.df)
    df_multi = df[df["is_single_event"] == 0].copy()
    splits = temporal_val_split(
        df_multi, features=ENRICHED_FEATURES,
        val_quantile=0.64, test_quantile=0.80,
    )
    splits["bundle"] = bundle
    return splits


def objective(trial: optuna.Trial, splits: dict) -> float:
    params = {
        "n_estimators": 2500,
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "early_stopping_rounds": 60,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
        "device": "cuda",
    }
    mdl = XGBClassifier(**params)
    mdl.fit(
        splits["X_train"], splits["y_train"],
        sample_weight=splits["w_train"],
        eval_set=[(splits["X_val"], splits["y_val"])],
        verbose=False,
    )
    y_val_score = mdl.predict_proba(splits["X_val"])[:, 1]
    pr_auc_val = average_precision_score(splits["y_val"], y_val_score)
    trial.set_user_attr("best_iteration", int(mdl.best_iteration) + 1)
    trial.set_user_attr("roc_auc_val", float(roc_auc_score(splits["y_val"], y_val_score)))
    return pr_auc_val


def main():
    print("Loading features + building temporal val split...")
    splits = make_splits()
    print(f"  features: {len(ENRICHED_FEATURES)}")
    print(f"  train: {len(splits['y_train']):,} | val: {len(splits['y_val']):,} | test: {len(splits['y_test']):,}")
    print(f"  cutoffs: val<={splits['cutoff_val'].date()}  test>{splits['cutoff_test'].date()}")
    print(f"  positive rate — train: {splits['y_train'].mean():.3f} | val: {splits['y_val'].mean():.3f} | test: {splits['y_test'].mean():.3f}")

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="xgb_temporal_pr_auc")
    study.optimize(lambda t: objective(t, splits), n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"  PR-AUC (val): {study.best_value:.4f}")
    print(f"  best params: {json.dumps(study.best_params, indent=2)}")

    # Refit best params with early stopping on val, then evaluate on test
    final_params = dict(study.best_params)
    final_params.update({
        "n_estimators": 2500,
        "early_stopping_rounds": 60,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1, "tree_method": "hist", "device": "cuda",
    })
    final = XGBClassifier(**final_params)
    final.fit(
        splits["X_train"], splits["y_train"],
        sample_weight=splits["w_train"],
        eval_set=[(splits["X_val"], splits["y_val"])],
        verbose=False,
    )
    best_iter = int(final.best_iteration) + 1

    y_test_score = final.predict_proba(splits["X_test"])[:, 1]
    thr, _ = find_threshold_max_f1(splits["y_val"], final.predict_proba(splits["X_val"])[:, 1])  # threshold from val, not test
    y_test_pred = (y_test_score >= thr).astype(int)

    test_metrics = {
        "n_test": len(splits["y_test"]),
        "test_positive_rate": float(splits["y_test"].mean()),
        "best_iteration": best_iter,
        "best_threshold_from_val": float(thr),
        "roc_auc": float(roc_auc_score(splits["y_test"], y_test_score)),
        "pr_auc": float(average_precision_score(splits["y_test"], y_test_score)),
        "brier": float(brier_score_loss(splits["y_test"], y_test_score)),
        "f1": float(f1_score(splits["y_test"], y_test_pred, zero_division=0)),
        "precision": float(precision_score(splits["y_test"], y_test_pred, zero_division=0)),
        "recall": float(recall_score(splits["y_test"], y_test_pred, zero_division=0)),
        "recall_top10pct": float(recall_at_top_k(splits["y_test"], y_test_score, 0.10)),
        "precision_top10pct": float(precision_at_top_k(splits["y_test"], y_test_score, 0.10)),
        "recall_top20pct": float(recall_at_top_k(splits["y_test"], y_test_score, 0.20)),
    }
    print("\n=== Test metrics (with tuned model) ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    # Persist
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": final,
        "feature_order": ENRICHED_FEATURES,
        "best_params": study.best_params,
        "best_iteration": best_iter,
        "best_threshold_from_val": float(thr),
        "test_metrics": test_metrics,
        "model_name": "xgboost_optuna_v1",
    }, OUT_MODEL)
    print(f"\nWrote {OUT_MODEL}")

    # Trials CSV
    trial_rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        trial_rows.append({
            "trial": t.number,
            "pr_auc_val": t.value,
            "roc_auc_val": t.user_attrs.get("roc_auc_val"),
            "best_iteration": t.user_attrs.get("best_iteration"),
            **t.params,
        })
    pd.DataFrame(trial_rows).to_csv(OUT_TRIALS, index=False)
    print(f"Wrote {OUT_TRIALS}")

    # Plots
    fig, ax = plt.subplots(figsize=(11, 5))
    values = [t.value for t in study.trials if t.value is not None]
    ax.plot(range(len(values)), values, marker="o", color="#4c72b0", label="PR-AUC (val) por trial")
    best = np.maximum.accumulate(values)
    ax.plot(range(len(best)), best, color="#c44e52", lw=2.5, label="Melhor até agora")
    ax.set_xlabel("Trial"); ax.set_ylabel("PR-AUC (val)")
    ax.set_title("Otimização Optuna — histórico")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_HISTORY, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_HISTORY}")

    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(11, 6))
        keys = list(imp.keys())
        vals = [imp[k] for k in keys]
        ax.barh(keys, vals, color="#4c72b0")
        ax.invert_yaxis(); ax.set_xlabel("Importância")
        ax.set_title("Importância dos hiperparâmetros (Optuna)")
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.01, i, f"{v:.3f}", va="center")
        fig.tight_layout()
        fig.savefig(OUT_IMPORT, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {OUT_IMPORT}")
    except Exception as e:
        print(f"  param importance skipped: {e}")


if __name__ == "__main__":
    main()
