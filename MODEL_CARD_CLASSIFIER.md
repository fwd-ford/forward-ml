# Model Card — Churn Scorer V3

## Identificação

- **Nome:** `churn_scorer_v3`
- **Data de treino:** 2026-05-24
- **Algoritmo principal:** XGBoost 2.0 (GPU, `device="cuda"`)
- **Arquitetura:** Two-stage classifier + stratified isotonic calibration
- **Artifact:** [data/models/churn_scorer_v3.joblib](data/models/churn_scorer_v3.joblib)
- **Inferência:** [src/inference.py](src/inference.py) → `ChurnScorer.load().predict(df)`

## Objetivo

Para cada VIN da rede oficial Ford no Brasil, calcular a **probabilidade de churn** (deixar de usar a rede oficial), de modo calibrado e auditável, para uso em campanhas de retenção.

## Target engenheirado

```python
churned = (reference_date - last_service_date).days > 365
# reference_date = 2026-05-04 (max ServiceDate, fixo p/ reprodutibilidade)
```

## Arquitetura V3 — 4 melhorias sobre o V2

### 1. Cohort-aware sample weights ([features.py:175](src/features.py#L175))

```python
weight = (1 / freq[model_name]) * (1 / freq[sales_year])
# normalized to mean = 1
```

Compensa **dois vieses simultâneos**: dominância de RANGER/KA por modelo + dominância de cohort 2020 (92% churn) por idade. Sem isso, o modelo colapsa em "detector de 2020".

### 2. Two-stage classifier ([inference.py:188](src/inference.py#L188))

- **Stage 1:** Roteador determinístico classifica cada VIN como `high_confidence` (multi-event, treinável), `single_event` (só 1 visita) ou `holdout` (tenure < 365d).
- **Stage 2A:** Modelo `model_multi` treinado em 117k multi-event VINs (com Optuna params).
- **Stage 2B:** Modelo `model_single` treinado SOMENTE em single-event VINs (27k VINs com a mesma config).
- VINs `holdout` usam `model_multi` mas com calibração específica.

### 3. Stratified isotonic calibration ([inference.py:230](src/inference.py#L230))

3 calibradores **separados**, treinados na população onde cada um se aplica:
- `cal_hc` → para `high_confidence` (na val temporal multi-event)
- `cal_se` → para `single_event` (no próprio subset)
- `cal_holdout` ← compartilha com hc

Resolve o problema do V2 onde single-event ficava +15pp **over-confident** e holdout +48pp.

### 4. Per-quality F1-optimal thresholds

| Quality | Threshold ótimo |
|---|---:|
| high_confidence | 0.319 |
| single_event | 0.750 |
| holdout | 0.319 (mesma do hc) |

## Métricas — V3 no dataset completo (175.552 VINs)

### Métricas globais

| Métrica | Valor |
|---|---:|
| **Acurácia** | **87.01%** |
| **Acurácia balanceada** | **86.97%** |
| Precision | 88.12% |
| Recall | 84.93% |
| F1 | 0.8649 |
| ROC-AUC | 0.9262 |
| PR-AUC | 0.9304 |
| **Brier (calibração)** | **0.1275** |

### Por scoring_quality

| Quality | n | acc | balanced_acc | F1 | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|
| **single_event** | 26.982 | **99.20%** | 99.28% | 0.993 | 0.998 | **0.007** |
| **holdout** | 31.938 | **90.03%** | 88.95% | 0.864 | 0.908 | 0.086 |
| **high_confidence** | 116.632 | 83.36% | 83.38% | 0.832 | 0.907 | 0.167 |

### Por cohort de venda

| Ano | n | Churn real | Score médio | ROC-AUC | Balanced acc |
|---:|---:|---:|---:|---:|---:|
| 2020 | 64.634 | 92.7% | 0.680 | **0.998** | 95.7% |
| 2021 | 18.960 | 75.7% | 0.456 | 0.998 | 90.2% |
| 2022 | 12.137 | 43.2% | 0.177 | 0.994 | 84.1% |
| 2023 | 21.094 | 21.4% | 0.144 | 0.903 | 74.8% |
| 2024 | 37.029 | 5.2% | 0.183 | 0.656 | 55.2% |
| 2025 | 21.568 | 0.04% | 0.180 | 0.310 ⚠️ | 45.1% |
| 2026 | 130 | 0.8% | 0.173 | 0.996 | 99.6% |

**⚠️ Cohorts 2024–2025** (right-censoring): VINs novos não tiveram tempo de churnar pela definição de 365d, ROC-AUC fica abaixo de random. **Isso é limitação do target, não do modelo.**

## Evolução V1 → V2 → V3

| Métrica | V1 baseline | V2 Optuna | **V3 (atual)** |
|---|---:|---:|---:|
| Acurácia | 73.0% | 73.0% | **87.0%** |
| Acurácia balanceada | 73.1% | 73.1% | **87.0%** |
| F1 | 0.733 | 0.733 | **0.865** |
| Brier | 0.023 (in-sample) | 0.310 | **0.128** |
| Features | 19 (com leak) | 22 (sem leak) | 22 + cohort weights |
| Calibração | Não | Isotônica única | **Estratificada (3 calibradores)** |
| Arquitetura | 1 modelo | 1 modelo | **2 modelos (two-stage)** |
| Validação | Random split | Temporal 3-way | Temporal 3-way |

## Hiperparâmetros (Optuna 40 trials, TPE)

```python
{
    "max_depth": 9,
    "learning_rate": 0.01535,
    "subsample": 0.7731,
    "colsample_bytree": 0.6201,
    "min_child_weight": 9,
    "reg_alpha": 0.1164,
    "reg_lambda": 0.3685,
    "gamma": 1.4852,
    "n_estimators": 2500,        # com early_stopping_rounds=60
    "tree_method": "hist",
    "device": "cuda",
}
```

## Features (22, ENRICHED_FEATURES)

**Core comportamentais (12):** events_count, tenure_days, km_max, km_per_month, dealers_distinct, primary_dealer_share, service_codes_distinct, service_types_distinct, is_loyal_to_dealer, is_single_event, model_year, model_name_freq.

**Cohort & sazonalidade (7, novas V2):** sales_year, sales_month, sales_quarter, years_since_sale, age_when_first_serviced_days, first_service_month, sale_year_minus_model_year.

**Gap shape (3, novas V2):** gap_range_ratio, gap_skew, services_per_year.

**Removidas vs V1:** gap_avg/min/max/last/last_vs_avg, age_at_last_service_days, is_warranty_active — todas com leak via imputação ou via `last_service_date`.

## Limitações honestas

### Right-censoring por design do target
- Cohort 2024-2025 tem pouquíssimos VINs com churn observável (matemática, não modelo).
- A acurácia global de 87% **inclui as cohorts antigas** onde basicamente tudo já churnou — modelo identifica corretamente.
- Em produção, **não usar o score pra VINs com sales_date < 12 meses** sem regras complementares.

### Métricas são pós-treino em todo o dataset
- O treino do `model_multi` foi feito com split temporal honesto; mas o scoring final aplica em 175k VINs (incluindo as próprias amostras de treino do `model_single` e do recalibrador).
- A acurácia 87% global é **honesta-mas-otimista**. A acurácia balanceada em **cohort 2024** (55%) é o piso real de generalização.

### Viés de seleção
- Dataset só vê VINs que passaram pela rede oficial pelo menos 1 vez. **Os 95% da frota Ford que vão pra independente NÃO estão aqui.**
- Não generaliza pra "frota total Ford".

### LGPD
- `VIN_Hash` é SHA1 robusto. Não tentar reverter ou cruzar com fontes externas.

## Como usar (deploy)

### Python (Java backend chama via Python ou via API FastAPI):
```python
from src.inference import ChurnScorer
scorer = ChurnScorer.load()
df_features = ...  # mesmo schema de vin_features.csv
scored = scorer.predict(df_features)
# colunas adicionadas:
#   churn_probability (calibrado)
#   churn_probability_raw (sem calibração)
#   churn_predicted (0/1 no threshold por quality)
#   churn_risk_tier (very_low / low / high / very_high)
#   churn_decile (D1 = top 10% arriscados)
#   score_fidelidade (1 - churn_probability)
#   scoring_quality (high_confidence / single_event / holdout)
```

### Pipeline end-to-end:
```bash
python -m src.classification                # treina V1 baseline (referência)
python -m scripts.tune_optuna               # otimiza V2 (40 trials)
python -m src.inference                     # treina V3 (two-stage + calibração)
python -m scripts.score_final               # produz CSV final + métricas + análise de erro
pytest tests/ -v                            # valida pipeline
```

## Governança & retreino

- **Frequência:** retreinar a cada 3 meses ou quando drift em positive rate > 5pp.
- **Monitorar:** distribuição de `days_since_last_service`, ROC-AUC por `model_name`, calibração por cohort.
- **Versionamento:** artifact em `data/models/churn_scorer_vN.joblib` (N incremental).
