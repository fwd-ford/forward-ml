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

## Reprodutibilidade

```bash
# Pré-requisito: o arquivo bruto vin_share_Desafio_02.xlsx (não versionado por tamanho)
python scripts/profile_dataset.py
python scripts/internal_analysis.py
```

## Contexto

Dataset oficial recebido em 11/05/2026 via coordenação FIAP. Schema documentado em [forward-docs/project/02e_DATASET_OFICIAL_E_FONTES.md](../../../forward-docs/project/02e_DATASET_OFICIAL_E_FONTES.md).
