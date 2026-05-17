# data/external/

Dados públicos coletados de 4 fontes externas pra enriquecer o pipeline de ML. Coleta em 11/05/2026 — fontes detalhadas em [forward-docs/project/02e_REFERENCIAS.md](../../../forward-docs/project/02e_REFERENCIAS.md).

## Arquivos

### `fipe_values.csv` (95 linhas)
Tabela FIPE Brasil — valor médio de mercado por modelo×ano, ref maio/2026.
- **Fonte:** API pública Parallelum FIPE (`parallelum.com.br/fipe/api/v1/`)
- **Colunas:** `model, year, fipe_mean_brl, fipe_min_brl, fipe_max_brl, n_versions, reference_month, source`
- **Uso:** input do cálculo de LSV (Lifetime Service Value) — Pilar 1, LN1

### `ford_dealers.csv` (80 linhas)
Concessionárias Ford Brasil com geolocalização (lat/long) e endereço.
- **Fonte:** ABRADIF (Associação Brasileira dos Distribuidores Ford)
- **Colunas:** `dealer_name, city, uf, address, cep, phone, latitude, longitude, source_url`
- **Cobertura:** 67% da rede estimada (~120 dealers); 25 de 27 UFs cobertas
- **Limitação conhecida:** `DealerCode` numérico do dataset oficial (435 códigos) NÃO mapeia diretamente pra estes nomes — esse mapping é proprietário Ford (DMS interno). Cruzamento possível por cidade/UF.
- **Uso:** Service Share Map, Rede Invertida (LN3) — Pilar 1 e LN3

### `ford_recalls.csv` (12 linhas)
Campanhas de recall Ford Brasil 2015-2026.
- **Fonte:** Senacon, Procon-SP, Ford BR oficial, imprensa especializada
- **Colunas:** `campaign_code, announce_date, model, years_affected, defect_brief, universe_count, adhesion_rate, status, source_url`
- **Limitação:** códigos oficiais Senacon não públicos (portal exige login); 9 de 12 são abertas/recentes (2024-2026)
- **Alavanca regulatória:** novo CTB 2024-25 bloqueia licenciamento de veículo com recall pendente >1 ano
- **Uso:** Recall Gateway (LN4) — Pilar 2

### `maintenance_schedules.csv` (65 linhas)
Cronograma oficial de manutenção Ford BR por modelo×revisão.
- **Fonte:** ford.com.br/servico-ao-cliente/* (oficial), media.ford.com (press releases), imprensa especializada
- **Colunas:** `model, revision_number, km, months, main_items, price_indicative_brl, source_url`
- **Padrão validado:** linha 2015+ = **10.000 km ou 12 meses**, o que vier primeiro
- **Exceções:** Transit 20k/12m; Ranger 3.0 V6 e Maverick FHEV 2023+ 16k/12m
- **Uso:** flag "atrasado pra revisão" como regra de negócio — Pilar 1 e Pilar 2

## Reprodutibilidade

Cada CSV tem coluna `source` ou `source_url` com URL específica por linha — qualquer número pode ser revisitado em segundos. Bibliografia completa em [02e_REFERENCIAS.md](../../../forward-docs/project/02e_REFERENCIAS.md).
