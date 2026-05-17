# data/processed/

Dados derivados do dataset oficial Ford (`vin_share_Desafio_02.xlsx`), gerados pelos scripts em `forward-ml/scripts/`.

## Arquivos

### `vin_features.csv`
- **175.554 linhas × 22 colunas** — 1 linha por VIN
- Features comportamentais derivadas de 602.788 eventos de serviço
- **Input principal** para os notebooks de Segmentação (K-means), Classificação (XGBoost com target engenheirado) e Survival Analysis
- Gerado por: `scripts/internal_analysis.py`
- Colunas-chave: `vin_hash, events_count, days_since_last_service, gap_avg_days, gap_last_days, km_max, dealers_distinct, primary_dealer, primary_dealer_share, model_name, model_year, service_types_distinct, service_codes_distinct`

### `profile.json`
- Perfil quantitativo do dataset oficial (distribuições por coluna, contagens, top valores)
- Gerado por: `scripts/profile_dataset.py`
- Útil pra: validar premissas, gerar slides de EDA, alimentar dashboards

### `feature_dictionary.json`
- **v2.0 (17/05/2026)** — dicionário das 22 colunas de `vin_features.csv` com agrupamentos lógicos (identity, frequency, recency, usage, loyalty, service_mix, lifecycle).
- Define também targets engenheirados (`churned`, `at_risk`) com critérios formais
- Lista features derivadas a criar no notebook (km_per_month, gap_last_vs_avg, etc.)
- Documenta data quality issues e selection bias
- Aponta junções possíveis com `data/external/`

## Reprodutibilidade

```bash
# Pré-requisito: o arquivo bruto vin_share_Desafio_02.xlsx (não versionado por tamanho)
python scripts/profile_dataset.py
python scripts/internal_analysis.py
```

## Histórico de limpeza (17/05/2026)

Os arquivos do trabalho prévio sobre o dataset sintético foram **removidos** nesta data:
- `ford_clientes_clean.parquet` (21 MB — sintético limpo)
- `learnings_experiment_01.json`, `learnings_experiment_02.json` (métricas de experimentos no sintético)

Razão: pós dataset oficial, o sintético deixou de ser ativo. Histórico preservado no git log. Caso seja necessário reativá-lo (notebook 06 — propensão socioeconômica paralela), o CSV bruto `ford_clientes_historico_completo.csv` continua disponível no workspace local (não versionado).

## Contexto

Dataset oficial recebido em 11/05/2026 via coordenação FIAP. Schema documentado em [forward-docs/project/02e_DATASET_OFICIAL_E_FONTES.md](../../../forward-docs/project/02e_DATASET_OFICIAL_E_FONTES.md).
