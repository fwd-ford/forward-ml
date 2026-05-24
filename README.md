# forward-ml — Churn Scorer Ford (Sprint 1)

![status](https://img.shields.io/badge/status-V3_production_ready-green?style=flat-square)
![stack](https://img.shields.io/badge/stack-Python_·_XGBoost_GPU_·_Optuna-333?style=flat-square)
![accuracy](https://img.shields.io/badge/balanced_accuracy-87%25-brightgreen?style=flat-square)

Modelo de classificação de churn para VINs da rede oficial Ford no Brasil. Three-layer ML pipeline (K-means + XGBoost two-stage + análise auditável).

## TL;DR

- **Input:** [data/processed/vin_features.csv](data/processed/vin_features.csv) (175k VINs × 22 features comportamentais derivadas do dataset oficial Ford).
- **Output principal:** [resultados/scoring_results.csv](resultados/scoring_results.csv) (175k VINs com `churn_probability`, `churn_risk_tier`, `cluster_name`).
- **Modelo final:** V3 — XGBoost two-stage (multi-event + single-event) + stratified isotonic calibration.
- **Métrica headline:** Acurácia balanceada **87%** global, ROC-AUC **0.93**, Brier **0.13**.
- **Documentação técnica:** [MODEL_CARD_CLASSIFIER.md](MODEL_CARD_CLASSIFIER.md).

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Verify (12 tests)
pytest tests/ -v

# 3. Run full pipeline end-to-end (~3 min em GPU)
python -m src.classification          # V1 baseline (XGBoost vs RF vs LogReg)
python -m scripts.tune_optuna         # V2 — 40 trials Optuna
python -m src.inference               # V3 — two-stage + stratified calibration
python -m scripts.score_final         # score 175k VINs + análise de erro

# 4. Use o modelo treinado (deploy)
python -c "
from src.inference import ChurnScorer
from src.features import load_features
scorer = ChurnScorer.load()
df = load_features().df.head(100)
print(scorer.predict(df)[['vin_hash', 'churn_probability', 'churn_risk_tier']].head())
"
```

## Estrutura do repo

```
forward-ml/
├── src/
│   ├── features.py           # load_features + target engineering + cohort weights
│   ├── segmentation.py       # K-means (Camada 1)
│   ├── classification.py     # 3 modelos baseline + temporal splits + Optuna helpers
│   ├── inference.py          # V3 scorer (two-stage + stratified calibration) ⭐
│   └── feature_names.py      # mapping code→PT
├── scripts/
│   ├── tune_optuna.py        # Optuna tuning 40 trials
│   ├── score_final.py        # produz scoring_results.csv + metrics.json
│   └── ...                   # outros utilitários (análises, plots, etc.)
├── notebooks/
│   ├── 01_eda_official.ipynb
│   ├── 04_classification.ipynb
│   └── 05_leak_audit.ipynb
├── data/
│   ├── processed/vin_features.csv      # input
│   └── models/
│       ├── churn_scorer_v3.joblib      # ⭐ produção
│       ├── xgb_optuna_best.joblib      # V2 Optuna
│       ├── churn_classifier.joblib     # V1 baseline
│       └── kmeans_segmentation.joblib  # Camada 1
├── tests/
│   └── test_inference.py     # 12 testes (features + weights + splits + inference)
├── resultados/
│   ├── scoring_results.csv   # ⭐ output principal
│   ├── scoring_results_pt.csv
│   ├── metrics.json
│   ├── analise_erro.csv
│   └── RESUMO_FINAL.md
├── MODEL_CARD_CLASSIFIER.md  # documentação técnica
├── HANDOFF_ML.md             # contexto histórico
└── CLAUDE.md                 # convenções
```

## Conceitos-chave

### O target

```python
churned = (reference_date - last_service_date).days > 365
```

- `churned = 1` → cliente **SAIU** da rede oficial 🔴
- `churned = 0` → cliente **FICOU** 🟢
- `reference_date = 2026-05-04` (max ServiceDate, fixo p/ reprodutibilidade)

### O score

```python
churn_probability ∈ [0, 1]
```

| Valor | Significado |
|---|---|
| próximo de 0 | cliente fiel/retido 🟢 |
| próximo de 1 | cliente churnado/saindo 🔴 |

`score_fidelidade = 1 - churn_probability` (versão invertida).

### Tiers operacionais

| Tier | Score | Ação |
|---|---|---|
| very_low | 0.0–0.2 | Cliente fiel — não ativar |
| low | 0.2–0.5 | Atenção — monitorar |
| **high** | **0.5–0.8** | **Alvo prioritário** |
| very_high | 0.8–1.0 | Provavelmente já churnado |

### Camada 1: K-means (segmentação)

4 clusters (fiel / econômico / esquecido / abandono) — heurística pós-fit. Persistido em `data/models/kmeans_segmentation.joblib`.

### Camada 2: XGBoost two-stage (V3)

- **Stage 1:** roteador determinístico (multi-event / single-event / holdout)
- **Stage 2A:** `model_multi` para 117k multi-event VINs
- **Stage 2B:** `model_single` dedicado para 27k single-event VINs
- **Stage 3:** calibrador isotônico específico por quality

## Output detalhado — `scoring_results.csv`

22 colunas, 175.552 linhas:

| Coluna | Descrição |
|---|---|
| vin_hash | ID SHA1 anonimizado |
| model_name, model_year | Identificação do veículo |
| sales_date, last_service_date | Datas-chave |
| trainable | 0/1 (tenure_days ≥ 365) |
| is_single_event | 0/1 |
| **churned** | Target real (0 ficou, 1 saiu) |
| churn_probability_raw | Score antes calibração |
| **churn_probability** | **Score calibrado (use este)** |
| score_fidelidade | 1 - churn_probability |
| churn_predicted | Classe predita |
| churn_risk_tier | very_low / low / high / very_high |
| churn_decile | D1 (top arriscados) … D10 |
| cluster_id, cluster_name | Cluster K-means |
| events_count, tenure_days, days_since_last_service, km_max, primary_dealer_share | Features originais |
| scoring_quality | high_confidence / single_event / holdout |

## Métricas V3 (175k VINs)

| Métrica | Valor |
|---|---:|
| Acurácia | 87.0% |
| Acurácia balanceada | 87.0% |
| F1 | 0.865 |
| ROC-AUC | 0.926 |
| Brier (calibração) | 0.128 |

Por scoring_quality:

| Quality | n | acc | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| single_event | 27k | 99.2% | 0.99 | 0.998 |
| holdout | 32k | 90.0% | 0.86 | 0.908 |
| high_confidence | 117k | 83.4% | 0.83 | 0.907 |

## Limitações

1. **Right-censoring do target:** cohorts 2024-2025 têm pouco churn observável → ROC-AUC < 0.7. **Não usar score pra VINs com sales_date < 12 meses sem regras complementares.**
2. **Viés de seleção:** dataset só vê quem passou pela rede oficial. Não generaliza pra frota total Ford.
3. **LGPD:** VIN_Hash é pseudonimizado.

## Próximos passos (Sprint 2+)

1. Forward-looking target (resolve right-censoring)
2. Features de evento via xlsx bruto (slope, sazonalidade)
3. API FastAPI (`/predict`, `/simulate`, `/health`)
4. Batch scoring com Supabase (`src/scoring.py`)
5. Survival analysis com lifelines (Camada 3)

## Convenções

- Idioma: código + comentários em inglês; notebook markdown + PDF em PT-BR.
- **GPU obrigatória** para XGBoost (`device="cuda"`).
- Modelos via `joblib` (não pickle puro).
- Random seed: `RANDOM_STATE = 42`.

## Contato

- **Autoria ML:** Lucca Saraiva Borges
- **Sprint:** 1 (deadline 24/05/2026)
- **Coordenação:** FIAP / Ford Brasil

Detalhes históricos: [HANDOFF_ML.md](HANDOFF_ML.md).
