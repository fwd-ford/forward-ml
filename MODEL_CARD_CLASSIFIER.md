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

### LGPD — Proteção de dados pessoais

O modelo processa identificadores veiculares pseudonimizados e features comportamentais derivadas de eventos de serviço. Esta seção documenta o tratamento conforme Lei 13.709/2018 (LGPD).

#### Base legal (art. 7)
- **Hipótese aplicada:** `legítimo interesse` (art. 7, IX) — retenção pós-venda na rede oficial Ford, com benefício direto para o titular (lembretes de manutenção, ofertas personalizadas, segurança veicular via recall).
- **Teste de proporcionalidade:** finalidade legítima ✓, necessidade (não há substituto menos invasivo do que features agregadas) ✓, balanceamento (titular tem direito ao opt-out via `anonymize_customer()` — vide forward-infra migration 013) ✓.
- **NÃO aplicável:** consentimento explícito (não foi coletado pelo desafio acadêmico) e execução de contrato (modelo não é cláusula contratual).

#### Finalidade (art. 6, I — adequação)
- **Única finalidade declarada:** estimar `churn_probability` por VIN para campanhas de retenção (CRM, ofertas de revisão, agendamento proativo).
- **Vedado:** uso para precificação de seguro, scoring de crédito, transferência a terceiros não-Ford, ou enriquecimento de cadastro fora do escopo pós-venda.

#### Dados tratados (art. 5, I e II)
- **Pseudonimização (art. 13, IV):** `VIN_Hash` = SHA1(VIN). Em 5M tentativas de reversão dirigida, 0 colisões/recoveries (vide `forward-infra/SECURITY.md`, STRIDE-S2). Considera-se **pseudonimização robusta, não anonimização** — re-identificação continua tecnicamente possível mediante cruzamento autorizado com tabela mestre (que NÃO está neste repositório).
- **Features comportamentais (22):** contagens e médias agregadas — `events_count`, `tenure_days`, `km_per_month`, `dealers_distinct`, `primary_dealer_share`, etc. Nenhuma feature identifica diretamente o titular.
- **Dados removidos por design:** nome, CPF, telefone, endereço, e-mail, CEP, gênero, renda — nada disso entra no pipeline (não existe no dataset oficial v2).
- **NÃO há dados sensíveis (art. 11):** sem origem racial, religião, saúde, biometria, sindical, política.

#### Retenção (art. 16)
- **Artefato do modelo (`churn_scorer_v3.joblib`, ~25 MB):** retenção de 12 meses ou até retreino subsequente (a cada 3 meses por governança). Versões antigas (`v1`, `v2`) podem ser preservadas em arquivo morto para auditoria, sem reuso operacional.
- **Scores produzidos (`client_scores` em Supabase):** retenção alinhada à política de CRM da Ford — recomendação: 24 meses, com purge automático via job agendado.
- **Features intermediárias (`vin_features.csv`):** regeneráveis on-demand a partir dos eventos brutos — não há motivo para reter por mais de 30 dias após o treino.
- **Dataset bruto (`vin_share_Desafio_02.xlsx`):** sob custódia da coordenação FIAP; este repositório não redistribui.

#### Direitos do titular (art. 18)
- **Confirmação e acesso (art. 18, I e II):** atendidos via endpoint `GET /clientes/{vin}/scores` (forward-api-java), que retorna o último score do VIN e a versão do modelo.
- **Correção (art. 18, III):** se eventos de serviço estiverem incorretos, a correção é feita no sistema-fonte (DMS dealer); o próximo retreino propaga.
- **Anonimização/eliminação (art. 18, IV e VI):** procedimento operacional via `anonymize_customer(vin_hash)` em `forward-infra/migrations/013_anonymize_customer.sql` — apaga o registro do titular em todas as tabelas e re-hash o VIN com salt rotacionado, tornando o histórico no modelo permanentemente desvinculado.
- **Portabilidade (art. 18, V):** features agregadas exportáveis em CSV mediante solicitação ao DPO.
- **Revisão de decisão automatizada (art. 20):** este modelo **não toma decisão isolada com efeitos jurídicos relevantes** — saída é insumo para campanha de marketing/retenção, sempre revisada por operador humano antes de contato com o cliente. Ainda assim, garante-se direito a explicação via SHAP values disponíveis por VIN.

#### Transferência internacional (art. 33)
- **Treino:** 100% local (GPU CUDA on-premise / workstation do desenvolvedor, Brasil). Sem transferência internacional durante a fase de modelagem.
- **Inferência:** prevista em infraestrutura nacional (Supabase região São Paulo conforme `forward-infra/README.md`). Caso futura migração para cloud em outra região seja avaliada, exigir **DPIA específica** e cláusulas-padrão de transferência internacional.

#### Encarregado / DPO
- **Encarregado pelo tratamento (art. 41):** Lucca Saraiva Borges (`webbersaraivaborges@gmail.com`) — interlocutor do projeto acadêmico com a ANPD e titulares.
- Em deploy real Ford, este papel migraria para o DPO corporativo Ford Brasil.

#### Riscos residuais
- **Re-identificação por cross-reference:** mesmo com VIN pseudonimizado, cruzamento com FENABRAVE (emplacamentos), FIPE (valores), DETRAN (proprietário) **poderia** desanonimizar. Cláusula contratual com qualquer integrador externo deve proibir esse cruzamento.
- **Inferência de atributos sensíveis:** `model_name` + `dealer_state` podem servir de proxy para perfil socioeconômico. Mitigação: não usar o score para decisões com impacto adverso (negação de crédito, seguro, garantia).
- **Vazamento do artefato joblib:** o modelo serializado **não contém PII**, mas contém os feature splits do XGBoost que poderiam, em ataque inverso, ajudar a inferir distribuição de comportamento da base. Mitigação: storage em bucket privado com IAM restrito, sem URL pública.
- **Pseudonimização não é anonimização (parecer ANPD 2022):** este repositório trata `VIN_Hash` como **dado pessoal** para todos os efeitos da LGPD, mesmo após hash.

#### Referências cruzadas
- [`forward-infra/SECURITY.md`](../forward-infra/SECURITY.md) — STRIDE completo, threat S2 (spoofing por VIN reversal)
- [`forward-infra/migrations/013_anonymize_customer.sql`](../forward-infra/migrations/013_anonymize_customer.sql) — implementação do direito de exclusão
- [LGPD art. 7](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm) (bases legais), art. 18 (direitos do titular), art. 50 (boas práticas)

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
