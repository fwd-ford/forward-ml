# Model Card — K-Means Segmentation (Personas Comportamentais)

## Identificação

- **Nome:** `kmeans_segmentation`
- **Data de treino:** 2026-05-24
- **Algoritmo:** `sklearn.cluster.KMeans` (k=4, `n_init=20`, `random_state=42`)
- **Tipo:** segmentação não supervisionada (clustering cego)
- **Artifact:** [data/models/kmeans_segmentation.joblib](data/models/kmeans_segmentation.joblib)
- **Profiles:** [data/models/kmeans_cluster_profiles.csv](data/models/kmeans_cluster_profiles.csv)
- **Métricas:** [data/models/kmeans_metrics.json](data/models/kmeans_metrics.json)
- **Implementação:** [src/segmentation.py](src/segmentation.py)
- **Notebook:** [notebooks/03_segmentation_kmeans.ipynb](notebooks/03_segmentation_kmeans.ipynb)

## Objetivo

Agrupar os 175.552 VINs da rede oficial Ford em **personas comportamentais** com base em padrões de uso da rede (frequência, recência, dispersão entre dealers, dimensão da relação), permitindo:
- Desenhar estratégias de retenção diferenciadas por perfil (fiel × esquecido × abandono × econômico).
- Validar cruzadamente o classificador de churn — clusters com churn rate ≈ 1 (`abandono`) ou ≈ 0 (`fiel`) servem como sanity check independente.
- Fornecer um eixo de leitura humano para os scores do XGBoost (`churn_scorer_v3`), via SHAP médio por cluster.

## Intended use & out-of-scope

### Uso pretendido
- Insumo para campanhas de CRM segmentadas (uma mensagem por persona, não por score puro).
- Análise gerencial: "quantos clientes estão em risco de abandono", "qual o tamanho do segmento fiel".
- Validação cruzada do modelo supervisionado de churn (overlap de top-decile com cluster `abandono`).

### Fora do escopo
- **Não substitui** o classificador para score individual — silhouette 0.40 indica grupos *moderadamente* separados, então cluster ≠ rótulo definitivo.
- **Não inferir socioeconômico, gênero, idade ou intenção de compra** a partir do cluster — features são puramente comportamentais.
- **Não usar como gatilho de decisão automatizada com efeito adverso** (negar garantia, recusar serviço): sempre como insumo a operador humano.

## Training data

- **Fonte:** `data/processed/vin_features.csv` (gerado de `data/raw/vin_share_Desafio_02.xlsx`, dataset oficial Ford 2026-05-11).
- **Granularidade:** 1 linha por `VIN_Hash`. **175.552 VINs**.
- **Features usadas (11):** `events_count, tenure_days, gap_avg_days, gap_last_days, days_since_last_service, dealers_distinct, primary_dealer_share, service_codes_distinct, km_per_month, is_loyal_to_dealer, is_single_event` (vide `KMEANS_FEATURES` em [src/features.py](src/features.py)).
- **Pré-processamento:** `StandardScaler` (z-score) em todas as features — necessário porque K-means é sensível a escala (gap em dias vai a 1700; share em [0,1]).
- **NÃO usado:** `model_name`, `model_year`, `vin_hash`, `churned` (target supervisionado fica de fora — segmentação é cega).

### Viés de seleção
- O dataset só vê **VINs que passaram pela rede oficial Ford pelo menos uma vez**. Os ~95% da frota Ford no Brasil que vão para oficinas independentes **não aparecem aqui** — qualquer afirmação sobre "fidelidade" se refere a *fidelidade visível na rede*, não fidelidade absoluta à marca.
- Distribuição enviesada por modelo: RANGER (56,7%) + KA (22,2%) dominam o dataset. Como a segmentação roda **sem `model_name`**, esse viés afeta os perfis indiretamente (RANGER tem padrão de manutenção pesado, KA tem padrão leve → influencia médias de `km_per_month`, `events_count`).
- Cohort effect: VINs vendidos em 2020 têm muito mais tempo de observação que VINs 2025 → naturalmente caem em clusters de gap alto. Mitigação: leitura dos perfis sempre cruzada com `sales_year` no relatório.

## Evaluation

### Escolha de k (sweep silhouette + inertia)

`sweep_k` em [src/segmentation.py:46](src/segmentation.py#L46) varre `k=2..7` em sample de 10k para silhouette (full dataset para inertia). Decisão final: **k=4** — silhouette aceitável (≥0,35 por regra do CLAUDE.md), elbow visível e 4 personas mapeiam diretamente para a nomenclatura de negócio do DOC 00 (fiel/econômico/esquecido/abandono).

### Métricas (k=4 no dataset completo)

| Métrica | Valor |
|---|---:|
| **k** | 4 |
| **Silhouette (sample 10k)** | **0,4007** |
| Inertia | 773.615 |
| n_samples | 175.552 |

Silhouette ≥ 0,35 é o piso aceitável para clustering comportamental (vide `HANDOFF_ML.md`). Comportamento humano não produz fronteiras tão limpas quanto dados sintéticos.

### Perfis nomeados (heurística label-free, `_name_clusters` em [src/segmentation.py:62](src/segmentation.py#L62))

| Cluster | Nome | n | % | churn_rate | events | tenure_d | gap_last_d | dealers | dealer_share | single_event |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | **econômico** | 30.252 | 17,2% | 10,7% | 1,0 | 352 | 197 | 1,0 | 1,00 | 1,00 |
| 1 | **esquecido** | 44.286 | 25,2% | 58,5% | 4,65 | 1.011 | 273 | 2,29 | 0,59 | 0,00 |
| 2 | **fiel** | 80.796 | 46,0% | 45,3% | 4,29 | 862 | 235 | 1,04 | 0,99 | 0,00 |
| 3 | **abandono** | 20.218 | 11,5% | 99,98% | 1,0 | 417 | 1.727 | 1,0 | 1,00 | 1,00 |

### Cross-validações (sanity checks label-free)

1. **Cluster × `MaintenanceNumber` médio** — `fiel` e `esquecido` concentram revisões 4-5; `abandono` e `econômico` concentram revisão 1. ✓ esperado.
2. **Cluster × `model_name`** — `fiel` e `econômico` concentram RANGER (alta fidelidade da picape). `abandono` espalha entre KA, ECOSPORT, FIESTA (modelos descontinuados). ✓ consistente com retirada de portfólio.
3. **Cluster × `churned`** — `abandono` colide com churn=1 em 99,98% (era esperado: gap_last>365). `fiel` ainda assim tem 45% churn — alerta interno: parcela do "fiel" é fiel-histórico-mas-saiu.
4. **SHAP médio por cluster (validação cruzada com classificador V3, célula 16 do `04_classification.ipynb`):** `abandono` domina top features de risco; `fiel` domina top features de proteção. ✓ os dois modelos enxergam o mesmo fenômeno.

## Persona naming methodology

A nomeação é **heurística determinística**, não LLM. Funciona em 3 etapas (vide `_name_clusters`):

1. `abandono` = `argmax(rank(days_since_last_service) + rank(is_single_event))` — único + parado há muito tempo.
2. `fiel` = `argmax(rank(events_count) + rank(primary_dealer_share) - rank(days_since_last_service))` **restrito a clusters multi-event** — alta frequência + concentrado num dealer + ativo. Restrição multi-event evita que single-event vire "fiel" trivialmente (dealer_share=1 com 1 visita não é fidelidade).
3. `esquecido` = restante com maior `rank(days_since_last_service) + rank(gap_last_days)` — múltiplos serviços mas drifting.
4. `econômico` = sobra.

**Por que heurística e não rótulo manual:** reprodutibilidade. Se rodar com `k` diferente ou em dataset futuro, os nomes recalibram sem revisão humana. O custo é que a nomenclatura pode escorregar em datasets muito diferentes — alerta de validação no notebook 03.

## Limitations

### Métricas
- **Silhouette 0,40** = grupos moderadamente separados, não perfeitamente. Esperar overlap nas fronteiras.
- **K-means assume clusters esféricos isotrópicos** — comportamento humano raramente é assim. DBSCAN/HDBSCAN não foram tentados (escopo Sprint 1).

### Conceituais
- **Single-event domina 2 clusters** (`econômico` e `abandono`) — em datasets futuros com mais coverage isso pode mudar, e o número ótimo de k pode subir para 5-6.
- **Sem features financeiras** (valor do serviço, ticket médio) — `econômico` é nomeado por *padrão de uso leve*, não por ticket. Se ticket entrasse, parte do "fiel" viraria "premium".
- **Cohort confound:** VINs 2020 têm naturalmente mais eventos. A heurística parcialmente compensa (usa gap_last_days e days_since), mas resíduo existe.

### Viés de seleção (repetido de `Training Data`)
- Apenas VINs visíveis na rede oficial — ~5% da frota Ford BR. **NÃO generaliza para "frota total Ford"**.
- VINs vendidos por concessionária mas que nunca voltaram para serviço **não existem no dataset**, então o segmento "comprou e sumiu de cara" é invisível.

### Estabilidade
- `random_state=42` fixo em treino — resultados reprodutíveis. Mas `KMeans` é sensível a init: rodar em outro dataset (mesmo gerado pelo mesmo schema) pode produzir cluster ids permutados — confiar nos **nomes**, não nos ids numéricos.

## LGPD — proteção de dados

A segmentação processa as mesmas features comportamentais agregadas do classificador. Esta seção é resumida — a versão completa está em [MODEL_CARD_CLASSIFIER.md](MODEL_CARD_CLASSIFIER.md#lgpd--proteção-de-dados) e aplica-se integralmente aqui.

### Pontos específicos do segmentador
- **Base legal:** legítimo interesse (art. 7, IX) — finalidade idêntica à do classificador (retenção pós-venda).
- **Finalidade:** desenho de campanhas segmentadas, não tomada de decisão individual. **Cluster sozinho NÃO pode ser usado para decisão automatizada com efeito jurídico (art. 20)**.
- **Pseudonimização:** `VIN_Hash` (SHA1) não entra no clustering — é dropado antes do `StandardScaler`. O artefato `kmeans_segmentation.joblib` contém apenas centroides e scaler, **não armazena nenhum VIN**.
- **Retenção:** mesmo critério do classificador (12 meses ou até retreino). Cluster atribuído por VIN é regenerável sob demanda.
- **Direitos do titular:** atendidos pelo mesmo procedimento `anonymize_customer()` da forward-infra migration 013 — apagar o VIN da tabela `vin_features` automaticamente impede que o cluster seja recalculado para esse titular.
- **Inferência de atributos sensíveis:** o cluster `econômico` poderia, em uso indevido, ser proxy para perfil socioeconômico. **Vedado** o cruzamento do cluster com campanhas de crédito/seguro/preço.
- **Risco residual de re-identificação:** centroides + perfis (n=20k+) não permitem identificar indivíduo. Risco real surge ao cruzar `vin_clusters.csv` (que tem `vin_hash`) com tabelas externas — proibido contratualmente.

## Versionamento & retreino

- **Versionamento:** artefatos em `data/models/kmeans_segmentation.joblib`. Próximas versões: anexar suffix (`kmeans_segmentation_vN.joblib`).
- **Frequência de retreino:** alinhada ao classificador (a cada 3 meses ou quando drift em `events_count` médio > 10%).
- **Monitorar:** distribuição de tamanho dos 4 clusters (se `abandono` cresce > 15% sem mudança no modelo, é drift real do negócio, não bug). Silhouette score em retreino — alerta se cair abaixo de 0,30.

## Referências

- [HANDOFF_ML.md](HANDOFF_ML.md) — seção 4 (estratégia de segmentação)
- [CLAUDE.md](CLAUDE.md) — regras críticas (silhouette ≥ 0,35, nomes de negócio)
- [forward-docs/project/02e_DATASET_OFICIAL_E_FONTES.md](../forward-docs/project/02e_DATASET_OFICIAL_E_FONTES.md) — contexto do dataset oficial
- [MODEL_CARD_CLASSIFIER.md](MODEL_CARD_CLASSIFIER.md) — model card do classificador supervisionado (par)
