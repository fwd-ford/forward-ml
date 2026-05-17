# scripts/

Scripts reproduzíveis pra processar o dataset oficial Ford (`vin_share_Desafio_02.xlsx`) e gerar os artefatos em `data/processed/`.

## Arquivos

### `inspect_xlsx.py`
Inspeção estrutural rápida de um arquivo `.xlsx` — lista sheets, colunas, primeiras linhas. Util pra primeira leitura de qualquer planilha nova.

**Uso:** `python scripts/inspect_xlsx.py /caminho/para/arquivo.xlsx`

### `profile_dataset.py`
Perfil quantitativo completo do dataset (contagens por coluna, top valores, distribuição de VIN, multi-dealer, etc.). Saída: `data/processed/profile.json`.

**Uso:** `python scripts/profile_dataset.py` (caminho do xlsx hardcoded — ajustar se necessário)

**Saída:** `data/processed/profile.json`

### `internal_analysis.py`
Análise interna em 5 blocos:
1. Inventário (per-column stats, missing rates, distributions)
2. Geography (Country breakdown)
3. Codebook inference (cross-tabs Service*)
4. **Feature engineering comportamental** (gera `vin_features.csv` — 175k linhas × 20+ features)
5. SHA1 reversal test (sanity check de pseudonimização)

**Uso:** `python scripts/internal_analysis.py`

**Saídas:**
- `forward-docs/research/official/01_internal_analysis/*.md` (5 markdowns de análise)
- `data/processed/vin_features.csv` (input principal do ML)

## Dependências

```bash
pip install openpyxl
```

(O Python 3.10+ stdlib cobre o resto. Sem pandas — escolha intencional pra manter scripts leves e portáveis.)

## Contexto

Esses scripts foram criados em 11-17/05/2026 durante a análise inicial do dataset oficial recém-recebido da Ford. Relatório consolidado em [forward-docs/project/02e_DATASET_OFICIAL_E_FONTES.md](../../forward-docs/project/02e_DATASET_OFICIAL_E_FONTES.md).
