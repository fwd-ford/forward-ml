"""Compare three XGBoost configurations to test if early stopping + lower LR helps.

Splits used in every config:
  1. 64% train  / 16% validation / 20% test   (stratified random)
  2. 64% train  / 16% validation / 20% test   (temporal by sales_date)

Configs:
  - baseline:          current (n_estimators=300, lr=0.08, no early stop)
  - early_stop_v1:     n_estimators=2000, lr=0.03, early_stopping=50, slightly regularized
  - early_stop_v2:     n_estimators=2500, lr=0.02, early_stopping=80, more regularization

Outputs:
  - resultados/tuning_comparison.csv     side-by-side metrics
  - resultados/plots/tuning_comparison.png

Run: python -m scripts.tune_xgboost
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, f1_score,
    precision_score, recall_score,
)
from xgboost import XGBClassifier

from src.features import load_features, split_trainable, CLASSIFIER_FEATURES, model_inverse_weights
from src.classification import (
    temporal_split, find_threshold_max_f1, recall_at_top_k, precision_at_top_k,
    RANDOM_STATE,
)

OUT_CSV = REPO_ROOT / "resultados" / "tuning_comparison.csv"
OUT_FIG = REPO_ROOT / "resultados" / "plots" / "tuning_comparison.png"
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="talk")


def build_baseline():
    return XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", random_state=RANDOM_STATE,
        n_jobs=-1, tree_method="hist", device="cuda",
    )


def build_es_v1():
    return XGBClassifier(
        n_estimators=2000, max_depth=5, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.1, reg_lambda=1.0,
        early_stopping_rounds=50,
        eval_metric="logloss", random_state=RANDOM_STATE,
        n_jobs=-1, tree_method="hist", device="cuda",
    )


def build_es_v2():
    return XGBClassifier(
        n_estimators=2500, max_depth=6, learning_rate=0.02,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.5, reg_lambda=2.0, min_child_weight=5,
        early_stopping_rounds=80,
        eval_metric="logloss", random_state=RANDOM_STATE,
        n_jobs=-1, tree_method="hist", device="cuda",
    )


CONFIGS = {
    "baseline":     build_baseline,
    "early_stop_v1": build_es_v1,
    "early_stop_v2": build_es_v2,
}


def make_split(df, kind):
    X = df[CLASSIFIER_FEATURES].copy()
    y = df["churned"].to_numpy()
    w = model_inverse_weights(df["model_name"])
    g = df["model_name"].reset_index(drop=True)

    if kind == "random":
        # train(64%) + val(16%) + test(20%), stratified
        X_tmp, X_te, y_tmp, y_te, w_tmp, w_te = train_test_split(
            X, y, w, test_size=0.20, random_state=RANDOM_STATE, stratify=y,
        )
        X_tr, X_va, y_tr, y_va, w_tr, w_va = train_test_split(
            X_tmp, y_tmp, w_tmp, test_size=0.20, random_state=RANDOM_STATE, stratify=y_tmp,
        )
        return X_tr, X_va, X_te, y_tr, y_va, y_te, w_tr, w_va, w_te
    elif kind == "temporal":
        # 80% train+val (older sales) / 20% test (newest sales)
        X_tr_all, X_te, y_tr_all, y_te, w_tr_all, w_te, _, _, _ = temporal_split(
            df, cutoff_quantile=0.8, features=CLASSIFIER_FEATURES,
        )
        # 80/20 inside the older block for train/val (also temporally, by index order)
        n = len(X_tr_all)
        cutoff = int(n * 0.80)
        X_tr, X_va = X_tr_all.iloc[:cutoff], X_tr_all.iloc[cutoff:]
        y_tr, y_va = y_tr_all[:cutoff], y_tr_all[cutoff:]
        w_tr, w_va = w_tr_all[:cutoff], w_tr_all[cutoff:]
        return X_tr, X_va, X_te, y_tr, y_va, y_te, w_tr, w_va, w_te
    raise ValueError(kind)


def fit_one(config_name, splits):
    X_tr, X_va, X_te, y_tr, y_va, y_te, w_tr, w_va, w_te = splits
    mdl = CONFIGS[config_name]()
    if config_name == "baseline":
        mdl.fit(X_tr, y_tr, sample_weight=w_tr)
        best_iter = mdl.n_estimators
    else:
        mdl.fit(
            X_tr, y_tr, sample_weight=w_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        best_iter = mdl.best_iteration + 1
    y_score = mdl.predict_proba(X_te)[:, 1]
    thr, _ = find_threshold_max_f1(y_te, y_score)
    y_pred = (y_score >= thr).astype(int)
    return {
        "config": config_name,
        "n_estimators_max": mdl.n_estimators,
        "best_iteration": best_iter,
        "learning_rate": mdl.learning_rate,
        "n_train": len(y_tr),
        "n_val": len(y_va),
        "n_test": len(y_te),
        "test_pos_rate": float(np.mean(y_te)),
        "roc_auc": float(roc_auc_score(y_te, y_score)),
        "pr_auc": float(average_precision_score(y_te, y_score)),
        "brier": float(brier_score_loss(y_te, y_score)),
        "best_threshold": float(thr),
        "f1": float(f1_score(y_te, y_pred, zero_division=0)),
        "precision": float(precision_score(y_te, y_pred, zero_division=0)),
        "recall": float(recall_score(y_te, y_pred, zero_division=0)),
        "recall_top10pct": float(recall_at_top_k(y_te, y_score, 0.10)),
        "precision_top10pct": float(precision_at_top_k(y_te, y_score, 0.10)),
    }


def main():
    print("Loading features...")
    bundle = load_features()
    df, _ = split_trainable(bundle.df)
    print(f"  trainable VINs: {len(df):,}\n")

    all_rows = []
    for split_kind in ["random", "temporal"]:
        print(f">>> Split: {split_kind}")
        splits = make_split(df, split_kind)
        for cfg in CONFIGS:
            print(f"  fitting {cfg}...")
            row = fit_one(cfg, splits)
            row["split"] = split_kind
            all_rows.append(row)
            print(f"    best_iter={row['best_iteration']} | "
                  f"PR-AUC={row['pr_auc']:.4f} | ROC-AUC={row['roc_auc']:.4f} | "
                  f"recall@top10={row['recall_top10pct']:.4f}")

    cols_order = ["split", "config", "n_estimators_max", "best_iteration", "learning_rate",
                  "n_train", "n_val", "n_test", "test_pos_rate",
                  "roc_auc", "pr_auc", "brier", "best_threshold",
                  "f1", "precision", "recall", "recall_top10pct", "precision_top10pct"]
    audit = pd.DataFrame(all_rows)[cols_order]
    audit.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")
    print(audit.round(4).to_string(index=False))

    # ----- plot -----
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    metrics = [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("recall_top10pct", "Recall @ top 10%")]
    pivot_data = {}
    for metric, _ in metrics:
        pivot_data[metric] = audit.pivot(index="config", columns="split", values=metric)
    for ax, (metric, title) in zip(axes, metrics):
        d = pivot_data[metric]
        d = d.reindex(["baseline", "early_stop_v1", "early_stop_v2"])
        x = np.arange(len(d.index))
        ax.bar(x - 0.2, d["random"], 0.4, label="random split", color="#4c72b0")
        ax.bar(x + 0.2, d["temporal"], 0.4, label="temporal split", color="#c44e52")
        ax.set_xticks(x); ax.set_xticklabels(d.index, rotation=0)
        ax.set_title(title)
        ax.set_ylim(0, max(1.0, d.max().max() + 0.1))
        for i, (a, b) in enumerate(zip(d["random"], d["temporal"])):
            ax.text(i - 0.2, a + 0.01, f"{a:.3f}", ha="center", fontsize=9)
            ax.text(i + 0.2, b + 0.01, f"{b:.3f}", ha="center", fontsize=9)
        ax.legend(loc="upper right")
    fig.suptitle("Tuning do XGBoost — baseline vs. early-stopping", fontsize=17, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
