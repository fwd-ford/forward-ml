"""Overfitting & generalization tests.

What we check:
  1. Train/test accuracy gap — if too large, model memorized training set
  2. TimeSeriesSplit fold-by-fold consistency — if std too high, unstable
  3. Permutation test — shuffle labels; if model still gets good metrics, it's BUG
  4. Holdout cohort 2025 — must NOT exceed reasonable bound (right-censoring)
  5. Calibration consistency across folds
  6. Feature importance stability (top-5 features should be stable)
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import os
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pytest
import joblib
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from xgboost import XGBClassifier

from src.features import (
    load_features, split_trainable, ENRICHED_FEATURES,
    cohort_aware_weights, model_inverse_weights,
)
from src.classification import temporal_val_split, RANDOM_STATE
from src.inference import ChurnScorer


@pytest.fixture(scope="module")
def bundle():
    return load_features()


@pytest.fixture(scope="module")
def df_multi(bundle):
    df, _ = split_trainable(bundle.df)
    return df[df["is_single_event"] == 0].copy()


@pytest.fixture(scope="module")
def trained_model(df_multi):
    """Train once and reuse — saves time."""
    pack = joblib.load(REPO_ROOT / "data" / "models" / "xgb_optuna_best.joblib")
    params = dict(pack["best_params"])
    params.update({
        "n_estimators": 500, "early_stopping_rounds": 50,
        "eval_metric": "logloss", "random_state": RANDOM_STATE,
        "n_jobs": -1, "tree_method": "hist", "device": "cuda",
    })
    splits = temporal_val_split(df_multi, features=ENRICHED_FEATURES,
                                val_quantile=0.64, test_quantile=0.80)
    mdl = XGBClassifier(**params)
    mdl.fit(splits["X_train"], splits["y_train"],
            eval_set=[(splits["X_val"], splits["y_val"])], verbose=False)
    return mdl, splits


# ====================================================================
# 1. TRAIN VS TEST GAP
# ====================================================================
def test_train_test_gap_not_extreme(trained_model):
    """Checa overfit REAL via:
      - Train ROC-AUC vs test ROC-AUC (não acurácia, que é dominada por base rate)
      - Train balanced_acc não pode ser ~1.0 (memorização perfeita)
      - Test ROC-AUC > 0.55 (modelo precisa aprender ALGO útil)

    Não usamos acc raw aqui porque o test temporal tem base rate 0.6% (cohort recente
    right-censored) — só dizer "retido" já dá 99.4% de acc. Isso engana a métrica."""
    mdl, splits = trained_model

    train_score = mdl.predict_proba(splits["X_train"])[:, 1]
    test_score = mdl.predict_proba(splits["X_test"])[:, 1]

    auc_train = roc_auc_score(splits["y_train"], train_score)
    auc_test = roc_auc_score(splits["y_test"], test_score)
    bal_train = balanced_accuracy_score(splits["y_train"], mdl.predict(splits["X_train"]))
    auc_gap = auc_train - auc_test

    print(f"\n  ROC-AUC train={auc_train:.4f} | test={auc_test:.4f} | gap={auc_gap:.4f}")
    print(f"  balanced_acc train={bal_train:.4f}")

    # Sanity: nenhum modelo perfeito sem trapaça
    assert auc_train < 1.0, "Train ROC-AUC = 1.0 — sinal de leak duro"
    assert bal_train < 0.999, f"Train balanced_acc {bal_train:.4f} suspeitamente perfeito (memorização)"
    # Test tem que ter algum sinal
    assert auc_test > 0.55, f"Test ROC-AUC {auc_test:.3f} <= 0.55 — modelo não generaliza"
    # Gap ROC-AUC entre train e test não pode ser absurdo
    assert auc_gap < 0.45, f"Gap ROC-AUC train-test = {auc_gap:.3f} muito grande (overfit clássico)"


# ====================================================================
# 2. TIMESERIESSPLIT FOLD CONSISTENCY
# ====================================================================
def test_timeseries_consistency(df_multi):
    """Std do ROC-AUC entre folds não pode ser >0.35 — instabilidade severa."""
    df_sorted = df_multi.dropna(subset=["sales_date"]).sort_values("sales_date").reset_index(drop=True)
    X = df_sorted[ENRICHED_FEATURES]
    y = df_sorted["churned"].to_numpy()

    pack = joblib.load(REPO_ROOT / "data" / "models" / "xgb_optuna_best.joblib")
    params = dict(pack["best_params"])
    params.update({
        "n_estimators": 200, "eval_metric": "logloss",
        "random_state": RANDOM_STATE, "n_jobs": -1,
        "tree_method": "hist", "device": "cuda",
    })

    tss = TimeSeriesSplit(n_splits=5)
    aucs = []
    for tr, te in tss.split(X):
        if len(np.unique(y[te])) < 2:
            continue
        mdl = XGBClassifier(**params)
        mdl.fit(X.iloc[tr], y[tr])
        s = mdl.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y[te], s))
    print(f"\n  ROC-AUC por fold: {[f'{a:.3f}' for a in aucs]}")
    print(f"  mean={np.mean(aucs):.3f}, std={np.std(aucs):.3f}")
    assert len(aucs) >= 4, "At least 4 folds should have both classes"
    assert np.mean(aucs) > 0.55, f"Mean ROC-AUC {np.mean(aucs):.3f} barely above random"
    assert np.std(aucs) < 0.35, f"Std {np.std(aucs):.3f} too high — unstable model"


# ====================================================================
# 3. PERMUTATION TEST — shuffled labels
# ====================================================================
def test_permutation_no_signal_in_random(df_multi):
    """Com labels embaralhadas, modelo NÃO pode atingir ROC-AUC > 0.6 — seria bug."""
    sample = df_multi.sample(n=10_000, random_state=RANDOM_STATE)
    X = sample[ENRICHED_FEATURES]
    rng = np.random.default_rng(RANDOM_STATE)
    y_shuf = rng.permutation(sample["churned"].to_numpy())

    X_tr, X_te, y_tr, y_te = train_test_split(X, y_shuf, test_size=0.2,
                                              random_state=RANDOM_STATE, stratify=y_shuf)
    mdl = XGBClassifier(n_estimators=100, max_depth=5, eval_metric="logloss",
                        random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist", device="cuda")
    mdl.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, mdl.predict_proba(X_te)[:, 1])
    print(f"\n  ROC-AUC com labels SHUFFLED: {auc:.4f}")
    assert 0.40 < auc < 0.65, f"AUC {auc:.3f} fora de [0.4, 0.65] com labels random — BUG: modelo achou sinal onde não tem"


# ====================================================================
# 4. HOLDOUT COHORT 2025 — right-censoring guard
# ====================================================================
def test_recent_cohort_bounded(bundle):
    """Cohort 2025 tem churn ~0% por construção do target. Score nessa cohort
    pode até falhar (ROC-AUC < 0.5), mas o classifier não pode 'inventar' positives."""
    df = bundle.df.copy()
    df["sales_year"] = pd.to_datetime(df["sales_date"], errors="coerce").dt.year
    cohort_2025 = df[df["sales_year"] == 2025]
    if len(cohort_2025) < 100:
        pytest.skip("Cohort 2025 too small")

    scorer = ChurnScorer.load()
    scored = scorer.predict(cohort_2025)
    # Predição de churn em cohort recente deve ser baixa em média
    pred_rate = scored["churn_predicted"].mean()
    real_rate = scored["churned"].mean()
    print(f"\n  Cohort 2025 (n={len(cohort_2025)}):")
    print(f"  pred churn rate = {pred_rate:.2%} | real churn rate = {real_rate:.2%}")
    assert pred_rate < 0.6, f"Predicting {pred_rate:.0%} churn em cohort recente = over-prediction grave"


# ====================================================================
# 5. CALIBRATION SANITY
# ====================================================================
def test_calibration_monotonicity(bundle):
    """Score maior deve corresponder a churn maior em médias por bin (calibração ok)."""
    scorer = ChurnScorer.load()
    sample = bundle.df.sample(n=20_000, random_state=RANDOM_STATE)
    scored = scorer.predict(sample)
    bins = np.linspace(0, 1, 11)
    scored["bin"] = pd.cut(scored["churn_probability"], bins=bins, include_lowest=True)
    rates = scored.groupby("bin", observed=True)["churned"].mean().dropna()
    print(f"\n  Churn real por bin: {dict(zip([f'{i*0.1:.1f}' for i in range(len(rates))], rates.round(2).tolist()))}")
    # Verificar tendência ascendente
    diffs = np.diff(rates.values)
    n_subidas = (diffs > 0).sum()
    n_descidas = (diffs < 0).sum()
    assert n_subidas >= n_descidas, f"Calibração não-monotônica ({n_subidas} subidas vs {n_descidas} descidas)"


# ====================================================================
# 6. FEATURE IMPORTANCE STABILITY
# ====================================================================
def test_feature_importance_stable_across_seeds(df_multi):
    """Top features devem ser estáveis entre seeds (não ser por sorte)."""
    pack = joblib.load(REPO_ROOT / "data" / "models" / "xgb_optuna_best.joblib")
    base_params = dict(pack["best_params"])
    base_params.update({
        "n_estimators": 200, "eval_metric": "logloss",
        "n_jobs": -1, "tree_method": "hist", "device": "cuda",
    })

    splits = temporal_val_split(df_multi, features=ENRICHED_FEATURES,
                                val_quantile=0.64, test_quantile=0.80)
    top5_per_seed = []
    for seed in [42, 123, 7]:
        params = {**base_params, "random_state": seed}
        mdl = XGBClassifier(**params)
        mdl.fit(splits["X_train"], splits["y_train"])
        booster = mdl.get_booster()
        gain = pd.Series(booster.get_score(importance_type="gain")).sort_values(ascending=False)
        top5_per_seed.append(set(gain.head(5).index))

    overlap = set.intersection(*top5_per_seed)
    print(f"\n  Top-5 features comuns entre 3 seeds: {overlap}")
    assert len(overlap) >= 2, f"Apenas {len(overlap)} feature(s) consistente(s) — sinal de instabilidade"


# ====================================================================
# 7. SCORE DISTRIBUTION SANITY
# ====================================================================
def test_score_not_degenerate(bundle):
    """Score não pode ser constante nem todo no extremo."""
    scorer = ChurnScorer.load()
    sample = bundle.df.sample(n=10_000, random_state=RANDOM_STATE)
    scored = scorer.predict(sample)
    std = scored["churn_probability"].std()
    print(f"\n  Score std={std:.4f}, min={scored['churn_probability'].min():.3f}, max={scored['churn_probability'].max():.3f}")
    assert std > 0.10, f"Score std {std:.3f} muito baixo — modelo está dando quase a mesma probabilidade pra todos"
    # No mínimo 3% dos VINs devem estar na faixa intermediária (0.3-0.7)
    middle_pct = ((scored["churn_probability"] > 0.3) & (scored["churn_probability"] < 0.7)).mean()
    print(f"  middle (0.3-0.7) = {middle_pct:.2%}")
    assert middle_pct >= 0.03, f"Apenas {middle_pct:.2%} na zona cinza — score muito binarizado"
