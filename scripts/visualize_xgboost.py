"""Generate a complete pack of XGBoost diagnostic visualizations.

Outputs PNGs to forward-ml/reports/figures/. Designed to be opened by a human reviewer.

Each figure is built from the V3 (leak-safe + no single-event) model AND the V0 (baseline)
model, so the reader sees what production looks like with and without the contaminated features.

Run: python -m scripts.visualize_xgboost
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
matplotlib.use("Agg")  # headless backend — avoids Tk errors on Windows when run from CLI
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, roc_curve,
    roc_auc_score, average_precision_score,
)
from sklearn.model_selection import train_test_split

from src.features import load_features, split_trainable, CLASSIFIER_FEATURES, model_inverse_weights
from src.classification import (
    fit_and_compare_clean, _make_models, find_threshold_max_f1,
    LEAK_SAFE_FEATURES, RANDOM_STATE,
)
from src.feature_names import FRIENDLY_NAMES, to_friendly, rename_df

OUT_DIR = REPO_ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


def train_variant(df_train, features, exclude_single_event):
    df = df_train.copy()
    if exclude_single_event:
        df = df[df["is_single_event"] == 0].copy()
    X = df[features].copy()
    y = df["churned"].to_numpy()
    w = model_inverse_weights(df["model_name"])
    g = df["model_name"].reset_index(drop=True)
    X_tr, X_te, y_tr, y_te, w_tr, w_te, g_tr, g_te = train_test_split(
        X, y, w, g, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )
    mdl = _make_models()["xgboost"]
    mdl.fit(X_tr, y_tr, sample_weight=w_tr)
    return mdl, X_tr, X_te, y_tr, y_te, g_te


def gen_feature_importance(model, features, name):
    """XGBoost-native importance (gain / weight / cover) — shown as % of total."""
    booster = model.get_booster()
    types = [
        ("gain", "Gain — quanto cada feature ajudou a reduzir o erro"),
        ("weight", "Weight — em quantos splits a feature foi usada"),
        ("cover", "Cover — quantos VINs passaram pelos splits da feature"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    for ax, (t, title_pt) in zip(axes, types):
        raw = booster.get_score(importance_type=t)
        items = [(f, raw.get(f, 0)) for f in features]
        items.sort(key=lambda kv: kv[1], reverse=True)
        feats, vals = zip(*items)
        total = sum(vals) or 1.0
        pct = [v / total * 100 for v in vals]
        feats_pt = to_friendly(feats)
        bars = ax.barh(feats_pt, pct, color="#4c72b0")
        ax.invert_yaxis()
        ax.set_title(title_pt, fontsize=13)
        ax.set_xlabel(f"% do total de {t}")
        ax.set_xlim(0, max(pct) * 1.18 if max(pct) > 0 else 1)
        for bar, p in zip(bars, pct):
            ax.text(bar.get_width() + max(pct) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{p:.1f}%", va="center", fontsize=10)
    fig.suptitle(f"Importância das features no XGBoost — {name}", fontsize=17, y=1.02)
    fig.tight_layout()
    save(fig, f"01_feature_importance_{name}.png")


def gen_shap(model, X_te, name):
    """SHAP beeswarm + bar (with Portuguese labels in the plots)."""
    sample = X_te.sample(n=min(5000, len(X_te)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(sample)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) == 2 else shap_vals[0]

    # SHAP uses the DataFrame columns as labels — rename copy for display
    sample_pt = sample.copy()
    sample_pt.columns = to_friendly(sample.columns)

    fig = plt.figure(figsize=(12, 9))
    shap.summary_plot(shap_vals, sample_pt, show=False, max_display=20)
    plt.title(f"SHAP — impacto por VIN ({name})")
    save(plt.gcf(), f"02_shap_beeswarm_{name}.png")

    # SHAP bar with explicit % labels
    mean_abs_pt = pd.Series(np.abs(shap_vals).mean(axis=0), index=sample_pt.columns).sort_values(ascending=False)
    total = mean_abs_pt.sum() or 1.0
    pct_pt = (mean_abs_pt / total * 100)

    fig, ax = plt.subplots(figsize=(13, 9))
    bars = ax.barh(pct_pt.index, pct_pt.values, color="#4c72b0")
    ax.invert_yaxis()
    ax.set_xlabel("% da importância total (mean |SHAP|)")
    ax.set_title(f"SHAP — importância global ({name})")
    ax.set_xlim(0, max(pct_pt.values) * 1.18 if len(pct_pt) else 1)
    for bar, p in zip(bars, pct_pt.values):
        ax.text(bar.get_width() + max(pct_pt.values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{p:.1f}%", va="center", fontsize=10)
    fig.tight_layout()
    save(fig, f"03_shap_bar_{name}.png")

    mean_abs = pd.Series(np.abs(shap_vals).mean(axis=0), index=sample.columns).sort_values(ascending=False)
    top6 = mean_abs.head(6).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    for ax, feat in zip(axes.flatten(), top6):
        shap.dependence_plot(feat, shap_vals, sample, ax=ax, show=False, interaction_index=None)
        ax.set_title(FRIENDLY_NAMES.get(feat, feat), fontsize=13)
        ax.set_xlabel(FRIENDLY_NAMES.get(feat, feat))
    fig.suptitle(f"SHAP dependence (top 6) — {name}", fontsize=16, y=1.01)
    fig.tight_layout()
    save(fig, f"04_shap_dependence_{name}.png")

    return mean_abs


def gen_feature_distributions(df_train, top_features, name):
    """Violin: feature distribution by target class."""
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    for ax, feat in zip(axes.flatten(), top_features[:6]):
        sub = df_train[[feat, "churned"]].copy()
        if sub[feat].dtype != bool and sub[feat].nunique() > 2:
            lo, hi = sub[feat].quantile([0.01, 0.99])
            sub = sub[(sub[feat] >= lo) & (sub[feat] <= hi)]
        sns.violinplot(data=sub, x="churned", y=feat, ax=ax, inner="quartile", palette=["#4c72b0", "#c44e52"])
        ax.set_title(FRIENDLY_NAMES.get(feat, feat), fontsize=13)
        ax.set_xlabel("Churnou (0=retido, 1=churnado)")
        ax.set_ylabel(FRIENDLY_NAMES.get(feat, feat))
    fig.suptitle(f"Distribuição da feature por classe — {name}", fontsize=16, y=1.01)
    fig.tight_layout()
    save(fig, f"05_feature_dist_{name}.png")


def gen_eval_plots(model, X_te, y_te, g_te, name):
    """ROC, PR, calibration, confusion."""
    y_score = model.predict_proba(X_te)[:, 1]
    thr, _ = find_threshold_max_f1(y_te, y_score)
    y_pred = (y_score >= thr).astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ROC
    fpr, tpr, _ = roc_curve(y_te, y_score)
    axes[0, 0].plot(fpr, tpr, color="#4c72b0", lw=2.2, label=f"AUC={roc_auc_score(y_te, y_score):.4f}")
    axes[0, 0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0, 0].set_xlabel("FPR"); axes[0, 0].set_ylabel("TPR"); axes[0, 0].set_title("ROC")
    axes[0, 0].legend(loc="lower right")

    # PR
    p, r, _ = precision_recall_curve(y_te, y_score)
    axes[0, 1].plot(r, p, color="#c44e52", lw=2.2, label=f"AP={average_precision_score(y_te, y_score):.4f}")
    axes[0, 1].set_xlabel("Recall"); axes[0, 1].set_ylabel("Precision"); axes[0, 1].set_title("Precision-Recall")
    axes[0, 1].legend(loc="lower left")

    # Calibration
    frac_pos, mean_pred = calibration_curve(y_te, y_score, n_bins=10, strategy="quantile")
    axes[1, 0].plot(mean_pred, frac_pos, marker="o", color="#55a868", lw=2.2)
    axes[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    axes[1, 0].set_xlabel("Mean predicted prob"); axes[1, 0].set_ylabel("Frac of positives")
    axes[1, 0].set_title("Calibration"); axes[1, 0].legend(loc="upper left")

    # Confusion matrix at best F1 threshold
    cm = confusion_matrix(y_te, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 1],
                xticklabels=["pred 0", "pred 1"], yticklabels=["real 0", "real 1"], cbar=False)
    axes[1, 1].set_title(f"Confusion @ best F1 thr={thr:.3f}")

    fig.suptitle(f"Avaliação no test set — {name}", fontsize=16, y=1.00)
    fig.tight_layout()
    save(fig, f"06_eval_{name}.png")


def gen_per_group(model, X_te, y_te, g_te, name):
    """ROC-AUC per model_name."""
    y_score = model.predict_proba(X_te)[:, 1]
    rows = []
    for m in g_te.unique():
        mask = (g_te.values == m)
        if mask.sum() < 50:
            continue
        yt = y_te[mask]; ys = y_score[mask]
        if len(np.unique(yt)) < 2:
            continue
        rows.append({"model_name": m, "n": int(mask.sum()),
                     "pos_rate": yt.mean(), "roc_auc": roc_auc_score(yt, ys),
                     "pr_auc": average_precision_score(yt, ys)})
    if not rows:
        return
    pg = pd.DataFrame(rows).sort_values("n", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=pg, y="model_name", x="roc_auc", ax=ax, palette="Blues_r", order=pg["model_name"])
    ax.axvline(0.65, color="red", linestyle="--", label="floor 0.65")
    for i, (auc, n) in enumerate(zip(pg["roc_auc"], pg["n"])):
        ax.text(auc - 0.02, i, f"n={n} | AUC={auc:.3f}", va="center", ha="right", color="white", fontsize=10)
    ax.set_xlim(0.5, 1.0)
    ax.set_title(f"ROC-AUC por model_name — {name}")
    ax.legend(loc="lower right")
    save(fig, f"07_per_group_{name}.png")


def gen_compare_audit():
    """Bar chart comparing V0..V4 PR-AUC and recall@top10."""
    bundle = load_features()
    df_train, _ = split_trainable(bundle.df)

    from src.classification import LEAK_STRICT_FEATURES
    variants = {
        "V0_baseline":         dict(feature_subset=None,                  exclude_single_event=False, use_temporal_split=False),
        "V1_no_single_event":  dict(feature_subset=None,                  exclude_single_event=True,  use_temporal_split=False),
        "V2_leak_safe_feats":  dict(feature_subset=LEAK_SAFE_FEATURES,    exclude_single_event=False, use_temporal_split=False),
        "V3_safe_no_single":   dict(feature_subset=LEAK_SAFE_FEATURES,    exclude_single_event=True,  use_temporal_split=False),
        "V4_strict_temporal":  dict(feature_subset=LEAK_STRICT_FEATURES,  exclude_single_event=True,  use_temporal_split=True),
    }
    rows = []
    for name, cfg in variants.items():
        all_metrics, _, _ = fit_and_compare_clean(df_train, **cfg)
        x = all_metrics["xgboost"]
        rows.append({"variant": name,
                     "pr_auc": x["pr_auc"],
                     "roc_auc": x["roc_auc"],
                     "recall_top10": x["recall_at_top_k"]["top_10pct"]})
    audit = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    for ax, col, color, title in zip(
        axes,
        ["pr_auc", "roc_auc", "recall_top10"],
        ["#4c72b0", "#55a868", "#dd8452"],
        ["PR-AUC", "ROC-AUC", "Recall @ top 10%"],
    ):
        sns.barplot(data=audit, y="variant", x=col, ax=ax, color=color, order=audit["variant"])
        for i, v in enumerate(audit[col]):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=11)
        ax.set_xlim(0, max(1.0, audit[col].max() + 0.1))
        ax.set_title(title)
    fig.suptitle("Leak audit — XGBoost por variante", fontsize=18, y=1.02)
    fig.tight_layout()
    save(fig, "08_leak_audit_comparison.png")


def export_importance_csv(model, features, name, mean_abs_shap):
    """Save importance table as CSV to resultados/."""
    booster = model.get_booster()
    raw = {t: booster.get_score(importance_type=t) for t in ["gain", "weight", "cover"]}
    rows = []
    for f in features:
        rows.append({
            "feature_code": f,
            "feature_pt": FRIENDLY_NAMES.get(f, f),
            "gain": raw["gain"].get(f, 0),
            "weight": raw["weight"].get(f, 0),
            "cover": raw["cover"].get(f, 0),
            "mean_abs_shap": float(mean_abs_shap.get(f, 0)),
        })
    df = pd.DataFrame(rows)
    for col in ["gain", "weight", "cover", "mean_abs_shap"]:
        tot = df[col].sum() or 1.0
        df[f"{col}_pct"] = (df[col] / tot * 100).round(2)
    df = df.sort_values("gain_pct", ascending=False)
    out = REPO_ROOT / "resultados" / f"feature_importance_{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"  wrote resultados/feature_importance_{name}.csv")
    return df


def main():
    print(f"Output dir: {OUT_DIR}")
    bundle = load_features()
    df_train, _ = split_trainable(bundle.df)

    for tag, features, excl in [
        ("V0_baseline", CLASSIFIER_FEATURES, False),
        ("V3_safe_no_single", LEAK_SAFE_FEATURES, True),
    ]:
        print(f"\n>>> Training {tag} ({len(features)} features, exclude_single_event={excl})")
        model, X_tr, X_te, y_tr, y_te, g_te = train_variant(df_train, features, excl)
        gen_feature_importance(model, features, tag)
        mean_abs = gen_shap(model, X_te, tag)
        export_importance_csv(model, features, tag, mean_abs)
        gen_feature_distributions(df_train, mean_abs.index.tolist(), tag)
        gen_eval_plots(model, X_te, y_te, g_te, tag)
        gen_per_group(model, X_te, y_te, g_te, tag)

    print("\n>>> Audit comparison")
    gen_compare_audit()

    print(f"\nDone. {len(list(OUT_DIR.glob('*.png')))} PNG files in {OUT_DIR}")


if __name__ == "__main__":
    main()
