# Handoff ML — Pós-dataset oficial Ford

![data](https://img.shields.io/badge/atualizado-17%2F05%2F2026-brightgreen?style=flat-square)
![target](https://img.shields.io/badge/audience-ML_dev-blue?style=flat-square)
![deadline](https://img.shields.io/badge/Sprint_1-24%2F05%2F2026-red?style=flat-square)

> Documento de transferência de contexto técnico pra quem vai codar os notebooks de ML do Sprint 1. Pressupõe familiaridade com Python/pandas/sklearn/XGBoost. Tempo de leitura: ~15 min. Tempo até estar codando: ~30 min (incluindo setup).

---

## TL;DR — 3 bullets

1. **O dataset oficial Ford não é tabela de cliente — é log de eventos de serviço por VIN.** 602k linhas × 25 colunas, 175k VINs únicos, sem labels prontos, sem socioeconômico. Você vai **engenheirar features e target**, não treinar em cima de algo pré-rotulado.
2. **As features comportamentais já estão prontas em `data/processed/vin_features.csv`** (175k VINs × 22 colunas) — output do `internal_analysis.py`. Você consome esse arquivo, não o xlsx bruto.
3. **3 camadas de modelagem**, em ordem de prioridade: **K-means cego (segmentação) → Classificação binária com target engenheirado (`churned = days_since_last_service > 365`) → Kaplan-Meier por modelo×ano (survival, opcional)**. Eu cubro técnica + bibliotecas + armadilhas abaixo.

---

## Sumário

1. [O que mudou em 1 minuto](#1-o-que-mudou-em-1-minuto)
2. [O dataset oficial — fatos](#2-o-dataset-oficial--fatos)
3. [Engenharia de features (o que já está pronto)](#3-engenharia-de-features-o-que-já-está-pronto)
4. [Estratégia de modelagem — 3 camadas](#4-estratégia-de-modelagem--3-camadas)
5. [Stack técnica recomendada](#5-stack-técnica-recomendada)
6. [O que precisa de você](#6-o-que-precisa-de-você-notebooks-a-criar)
7. [Pontos de atenção (armadilhas)](#7-pontos-de-atenção-armadilhas-técnicas)
8. [Atualizações no repo que precisam ser feitas](#8-atualizações-no-repo-que-precisam-ser-feitas)
9. [Como começar](#9-como-começar-30-min)

---

## 1. O que mudou em 1 minuto

| Aspecto | Antes (sintético) | Agora (oficial) |
|---|---|---|
| Fonte | `ford_clientes_historico_completo.csv` (500k clientes × 37 colunas) | `vin_share_Desafio_02.xlsx` (602k eventos × 25 colunas, 175k VINs) |
| Granularidade | 1 linha = 1 cliente | 1 linha = 1 evento de serviço |
| Features socioeconômicas | ✅ tinha (idade, renda, score) | ❌ não tem |
| Labels prontos | ✅ tinha (`perfil_latente`, `churn_rede_24m`) | ❌ não tem |
| Trim/versão | n/a | ❌ não tem |
| Anti-data-leakage | regra crítica (13 colunas pós-compra) | **inaplicável** (sem labels) |
| Target de classificação | usava `churn_rede_24m` direto | **engenheirado** (`days_since_last_service > 365`) |
| Curva da Morte | hipótese a validar | **fato medido** via `MaintenanceNumber` |

O sintético **não foi descartado** (ADR-012) — vira input do modelo paralelo de propensão socioeconômica, se sobrar tempo no D-4/D-5.

**Doc completo do contexto:** [forward-docs/project/02e_DATASET_OFICIAL_E_FONTES.md](../forward-docs/project/02e_DATASET_OFICIAL_E_FONTES.md). Leia se quiser profundidade. Aqui o foco é prático.

---

## 2. O dataset oficial — fatos

### Schema (25 colunas)

| Coluna | Tipo | Uso provável |
|---|---|---|
| `Country` | string constante = `'BRA'` | **dropar** (1 valor único) |
| `ScheduleID` | int (18% missing) | identificador interno, não-feature |
| `MaintenanceID` | int (quase único, ~89%) | quase PK do evento, não-feature |
| `ServiceOrder` | int | identificador, não-feature |
| `ServiceDate` | string formato **misto BR/US** | feature temporal (parsear cuidadosamente) |
| `ServiceOpenDate`, `ServiceClosedDate` | string mista | derivar duração do serviço |
| `ServiceDeptCode` | char (77% missing): S/Q/U/B | provável dept; pouca usabilidade direta |
| `ServiceRepairTypeCode` | char (77% missing): C/W/I/A/U | provável C=Customer, W=Warranty, I=Internal — interessante |
| `ServiceType` | string constante = `'Maintenance'` | **dropar** |
| `ServiceCode` | int (139 valores únicos) | tipo de serviço numérico (sem codebook oficial) |
| `MaintenanceNumber` | int 1-52, dominância em 1-6 | **número da revisão programada** — **feature crítica** |
| `DealerCode` | int (435 únicos) | concessionária; agrupar/encodar |
| `MainSource` | 4 valores, com duplicata por trailing space | **strip antes de usar** |
| `IsAgendaSchedule` | bool, 94% True | feature fraca |
| `StatusUSA` | string constante = `'(60) Concluded'` | **dropar** |
| `VIN_Hash` | SHA1 robusto (40 chars) | chave de agrupamento |
| `ModelYear` | int 2017-2026 | feature |
| `ModelName` | string (21 valores) | feature categórica |
| `KM` | int (94k valores únicos) | **outlier 955M — clip a 500k** |
| `InvoiceDate`, `SalesDate`, `DeliveryDate`, `RegistrationDate`, `WarrantyStartDate` | string mista | derivar idade do veículo, tempo desde compra |

### Estatísticas-chave

- **175.554 VINs únicos** com **3,43 eventos médios** (mediana = 3)
- 28,77% dos VINs com **só 1 evento** — provável churned ou recém-comprado
- **72,7% dos VINs visitam um único `DealerCode`** — feature de lealdade forte
- Tenure (dias entre compra e último serviço): **mediana 669 dias**, p75 1.034
- Gap médio entre serviços: **mediana 221 dias**, p75 342
- KM máximo registrado: mediana 29.508, **mas tem outlier 955M** — clip a 500k
- Distribuição por modelo: **RANGER 56,7% + KA 22,2% = 79%** — fortemente desbalanceada

### Curva da Morte — direto no dado (`MaintenanceNumber`)

| Revisão # | % | Perde da anterior |
|---:|---:|---:|
| 1ª | 31,3% | — |
| 2ª | 22,3% | -29% |
| 3ª | 15,0% | -33% |
| 4ª | 9,5% | -37% |
| 5ª | 6,5% | -32% |

Use isso pra **validar** seus clusters do K-means (clusters "fiéis" devem concentrar quem chegou na 5ª+ revisão; "abandono" deve concentrar quem ficou na 1ª).

---

## 3. Engenharia de features (o que já está pronto)

`data/processed/vin_features.csv` (175.554 linhas × 22 colunas, ~31 MB) foi gerado pelo `scripts/internal_analysis.py`. **Você não precisa derivar do zero.**

### Colunas disponíveis

| Família | Colunas | Tipo |
|---|---|---|
| **Identidade** | `vin_hash`, `model_name`, `model_year` | string, string, int |
| **Frequência** | `events_count`, `tenure_days`, `gap_avg_days`, `gap_min_days`, `gap_max_days`, `gap_last_days` | int / float |
| **Recência** | `first_service_date`, `last_service_date`, `days_since_last_service` | date / int |
| **Uso** | `km_max`, `km_min` | int |
| **Lealdade** | `dealers_distinct`, `primary_dealer`, `primary_dealer_share` | int / string / float |
| **Mix** | `service_types_distinct`, `service_codes_distinct` | int |
| **Ciclo de vida** | `invoice_date`, `sales_date`, `warranty_start` | date |

### Carregando

```python
import pandas as pd

df = pd.read_csv("data/processed/vin_features.csv", parse_dates=[
    "first_service_date", "last_service_date",
    "invoice_date", "sales_date", "warranty_start"
])
print(df.shape)  # (175554, 22)
print(df.dtypes)
```

### Limpezas pendentes que você ainda precisa aplicar

- `km_max`: clip a 500.000 (alguns têm valores físicos impossíveis tipo 955M)
- `days_since_last_service`: o script usa `datetime.now()` como referência — **substituir pela data máxima de `ServiceDate` no dataset** pra evitar drift (a "data atual" é dinâmica, mas você quer reprodutibilidade)
- `gap_avg_days`: pode ser NaN se VIN tem só 1 evento — decidir imputação (median ou 0) ou flag separada
- `primary_dealer`: encoda como número (target encoding ou one-hot)

### Features derivadas que valem criar no notebook

```python
df["age_at_last_service_days"] = (df["last_service_date"] - df["sales_date"]).dt.days
df["km_per_month"] = df["km_max"] / (df["tenure_days"] / 30).clip(lower=1)
df["gap_last_vs_avg"] = df["gap_last_days"] / df["gap_avg_days"].clip(lower=1)
df["is_loyal_to_dealer"] = (df["primary_dealer_share"] > 0.8).astype(int)
df["is_warranty_active"] = (df["last_service_date"] - df["warranty_start"]).dt.days < 3*365
df["is_single_event"] = (df["events_count"] == 1).astype(int)
```

---

## 4. Estratégia de modelagem — 3 camadas

### Camada 1 — Segmentador (K-means cego)

**Objetivo:** descobrir personas espontaneamente, sem ground truth, e batizar com os nomes do DOC 00 (fiel/econômico/esquecido/abandono) baseado nas estatísticas dos clusters.

**Features sugeridas (escala obrigatória):**
- `events_count`, `tenure_days`, `gap_avg_days`, `gap_last_days`, `days_since_last_service`
- `dealers_distinct`, `primary_dealer_share`
- `service_codes_distinct`, `km_per_month`
- `is_loyal_to_dealer`, `is_single_event`

**Pipeline mínimo:**
```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

features_seg = [...]  # lista acima
X = df[features_seg].fillna(df[features_seg].median())
X_scaled = StandardScaler().fit_transform(X)

# Elbow + silhouette pra escolher k
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    score = silhouette_score(X_scaled[:10000], km.labels_[:10000])  # sample pra velocidade
    print(f"k={k}: silhouette={score:.3f}, inertia={km.inertia_:.0f}")
```

**Validação dos clusters (sem ground truth):**
1. Silhouette > 0,35 (real-world, com features comportamentais é difícil chegar a 0,5)
2. **Cross-tab com `MaintenanceNumber`**: clusters "abandono" devem concentrar revisão 1ª/2ª; clusters "fiel" devem concentrar 5ª+
3. **Cross-tab com `model_name`**: distribuição faz sentido por modelo? (Ka deve concentrar mais em abandono; Ranger em fiel)
4. **Nomeação manual** dos clusters após inspeção das médias por feature

**O que NÃO fazer:**
- Não use `ARI` ou `NMI` contra `perfil_latente` — esse campo era do sintético, não existe aqui
- Não rode K-means em 175k linhas sem `n_init=10` (resultado vai oscilar entre runs)

### Camada 2 — Classificador (target engenheirado)

**Objetivo:** prever se um VIN está churnado, dadas as features comportamentais. **Target é construído**, não vem pronto.

**Definição de churn (proposta — discutir):**
```python
# Use a data máxima de ServiceDate como "hoje" pra reprodutibilidade
reference_date = df["last_service_date"].max()
df["days_since_last_for_target"] = (reference_date - df["last_service_date"]).dt.days

# Definição: VIN não voltou em >365 dias → churned
df["churned"] = (df["days_since_last_for_target"] > 365).astype(int)
print(df["churned"].value_counts(normalize=True))
# Esperado: ~30-40% churned (cauda longa de VINs com 1 evento antigo)
```

**Validação da definição:**
- Cruzar com `events_count`: VINs com 1 evento e `last_service_date` antiga devem ser quase todos churned
- Cruzar com `tenure_days`: VINs com tenure curto e churned = compraram, fizeram 1 revisão e sumiram
- Cuidado: VINs com `last_service_date` muito recente NÃO podem ser usados (não dá tempo de observar 365 dias)

**Modelos pra comparar (Prof. Carlos pede 3+):**

| Modelo | Lib | Quando usar |
|---|---|---|
| Logistic Regression | `sklearn.linear_model.LogisticRegression` | baseline simples + interpretável |
| Random Forest | `sklearn.ensemble.RandomForestClassifier` | robusto, baseline médio |
| XGBoost | `xgboost.XGBClassifier` | provavelmente o melhor; SHAP funciona limpinho |
| LightGBM | `lightgbm.LGBMClassifier` | alternativa rápida ao XGBoost |
| Gradient Boosting sklearn | `GradientBoostingClassifier` | conservador |

**Pipeline com SHAP:**
```python
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df[features_class], df["churned"],
    test_size=0.2, random_state=42, stratify=df["churned"]
)

clf = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
clf.fit(X_train, y_train)

# Métricas
from sklearn.metrics import classification_report, confusion_matrix
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))

# SHAP
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test.sample(5000, random_state=42))
shap.summary_plot(shap_values, X_test.sample(5000, random_state=42))
```

**Cuidados:**
- **Split temporal, não random:** prefira `train` em VINs com `last_service_date` antes de uma data de corte e `test` depois (ou stratified por target se temporal não fizer sentido pro caso)
- **Cost-sensitive**: false negative (não pegar quem vai churnar) custa mais que false positive — usar `scale_pos_weight` ou threshold otimizado
- **Probabilidade calibrada**: se for usar score 0-100 no produto, calibrar com `CalibratedClassifierCV`

### Camada 3 — Survival Analysis (opcional, se sobrar tempo)

**Objetivo:** Kaplan-Meier por `model_name` × `model_year` pra responder "qual a curva típica de retorno por modelo?". É a **Curva da Morte estatística**, complemento da que já temos via `MaintenanceNumber`.

**Lib:** `lifelines` (`pip install lifelines`)

```python
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# Definir "evento" = visitou de novo (churn = não voltou ainda)
# "Duração" = gap_last_days (tempo desde o último serviço)
# "Evento observado" = se houve evento de retorno (gap_last_days vs algum gap futuro)

# Caso mais simples — observar tempo até último serviço por modelo
kmf = KaplanMeierFitter()
fig, ax = plt.subplots(figsize=(10, 6))
for model in ["RANGER", "KA", "ECOSPORT", "TERRITORY"]:
    mask = df["model_name"] == model
    kmf.fit(durations=df.loc[mask, "days_since_last_service"],
            event_observed=(1 - df.loc[mask, "churned"]).astype(int),  # voltou = evento
            label=model)
    kmf.plot_survival_function(ax=ax)
plt.title("Curva de sobrevivência (tempo até retornar à rede oficial) por modelo")
plt.show()
```

> **Atenção:** essa formulação é simplificada. Pra Sprint 1 dá pro pitch ("olha como a Curva da Morte é diferente por modelo"). Pra produção precisaria de censura adequada.

### Camada 4 (futura) — Recomendador

Sprint 1 = **regras Java** (não ML). Combina:
- Segmento (camada 1)
- Score de churn (camada 2)
- Flag "atrasado pra revisão" (do cronograma oficial em `data/external/maintenance_schedules.csv`)
- LSV (FIPE + frequência típica em `data/external/fipe_values.csv`)

Você não precisa codar isso — é endpoint Java. Mas o **score** que você produzir vai alimentar a regra.

---

## 5. Stack técnica recomendada

| Lib | Versão | Pra quê |
|---|---|---|
| `pandas` | >=2.0 | manipulação |
| `numpy` | >=1.24 | numérico |
| `scikit-learn` | >=1.3 | K-means, classificação básica, métricas, scaling |
| `xgboost` | >=2.0 | classificador principal |
| `shap` | >=0.44 | feature importance interpretável |
| `lifelines` | >=0.27 | survival (opcional) |
| `matplotlib` + `seaborn` | recentes | gráficos |
| `joblib` | >=1.3 | salvar modelos (formato seguro, conforme regra do `CLAUDE.md` do repo) |

`requirements.txt` do `forward-ml` já tem maioria. Adicionar `xgboost`, `shap`, `lifelines` se faltarem.

---

## 6. O que precisa de você (notebooks a criar)

Conforme [DOC 03 §4.1 v2.1](../forward-docs/project/03_SOLUTION_DESIGN.md#parte-4--sprint-1-o-que-entrega-até-2405):

| # | Notebook | Output | Prioridade | Tempo |
|---|---|---|---|---|
| 1 | `01_eda_official.ipynb` | distribuições por coluna, gráficos, validação de tipos | 🔴 alta | 2-3h |
| 2 | `02_feature_engineering.ipynb` | (já tem `vin_features.csv` — esse notebook é o "como gerei") | 🟡 média (mais doc que código) | 1h |
| 3 | `03_segmentation_kmeans.ipynb` | clusters validados + nomes (fiel/econômico/esquecido/abandono) + perfil de cada cluster | 🔴 alta | 3-4h |
| 4 | `04_classification.ipynb` | 3+ classificadores comparados + métricas + SHAP + decisão de threshold | 🔴 alta | 3-4h |
| 5 | `05_survival_analysis.ipynb` | Kaplan-Meier por modelo | 🟡 opcional | 2h |
| 6 | `06_propensity_synthetic.ipynb` | modelo paralelo no sintético (simulação socioeconômica) | 🟢 só se sobrar tempo | 2h |

### Entregas adicionais (não-notebook)

- `MODEL_CARD.md` — pra cada modelo final, documentar: target, features, métricas, dataset, limitações, política de governança
- `feature_dictionary.json` — atualizar pra refletir features comportamentais (não as antigas socioeconômicas)
- **Relatório PDF** (entrega obrigatória ML, conforme Prof. Carlos)

---

## 7. Pontos de atenção (armadilhas técnicas)

### 7.1 Datas em formato misto

`ServiceDate` tem `dd/mm/yyyy` (BR) e `m/d/yyyy` (US) **misturados na mesma coluna**. O `internal_analysis.py` tenta os 2 formatos. Pra você, na hora de carregar dados crus:

```python
def safe_parse_date(s):
    if pd.isna(s):
        return pd.NaT
    for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT
```

### 7.2 KM com outlier de 955 milhões

Sempre filtrar antes de plotar/calcular médias:
```python
df = df[df["km_max"].between(0, 500_000)]  # ou clip
```

### 7.3 MainSource com duplicata por trailing space

Se for re-derivar features do xlsx bruto:
```python
df["MainSource"] = df["MainSource"].str.strip()
```

### 7.4 Viés de seleção crítico no dataset

**O dataset só vê quem passou pela rede oficial.** Os 95% que vão pra oficinas independentes **não estão aqui**. Isso quer dizer:
- Modelo de churn aprende "quem está saindo entre os que ainda usam a rede"
- Modelo NÃO consegue prever "quem nunca vai pra rede" (não tem como saber sem dado externo)
- Sempre mencionar isso no MODEL_CARD e no relatório

### 7.5 Desbalanceamento por modelo

RANGER 56,7% + KA 22,2% = 79% dos eventos. Se você treinar sem cuidado:
- Modelo pode ficar "modelo Ranger" e generalizar mal pra Ka/EcoSport
- Considere `sample_weight` inversamente proporcional à frequência do modelo, ou stratify por modelo

### 7.6 SHA1 do VIN não é reversível

Não tente cruzar com dados externos via VIN. O hash é robusto (provavelmente com salt). Tratar como pseudonimizado (LGPD).

### 7.7 ServiceCode tem 139 valores únicos sem codebook

Top 6 códigos somam ~60% dos eventos. **Não use one-hot direto** (139 colunas vai explodir). Sugestões:
- Frequency encoding (mantém só count por código)
- Agrupar códigos raros em "OUTROS"
- Target encoding (mas cuidado com leakage temporal)

### 7.8 Nem todo VIN tem 365 dias de observação

Se `tenure_days < 365`, o VIN ainda não teve chance de churnar pela definição. **Excluir do treino** (ou usar definição condicional). Faça `df_train = df[df["tenure_days"] > 365]`.

### 7.9 days_since_last_service usa datetime.now()

O script `internal_analysis.py` calcula `days_since_last_service` usando a data de execução. **Recompute usando `last_service_date.max()` como referência** pra reprodutibilidade entre runs.

---

## 8. Atualizações no repo que precisam ser feitas

### 8.1 `forward-ml/CLAUDE.md` está desatualizado

O arquivo atual ainda fala de "Base 1/Base 2", anti-data-leakage, variáveis pós-compra etc — tudo regra do dataset **sintético**. Não se aplica ao oficial (não existe Base 1/Base 2, sem labels pós-compra, anti-leakage virou non-issue). Sugestão de update:

```markdown
## Critical ML Rules (v2 — pós-dataset oficial)

- Dataset oficial: `data/raw/vin_share_Desafio_02.xlsx` (log de eventos, NÃO tabela de cliente).
- Features comportamentais derivadas em `data/processed/vin_features.csv`.
- Target de classificação é **engenheirado** (`churned = days_since_last_service > 365`), não vem pronto.
- Excluir VINs com `tenure_days < 365` do treino (sem tempo de observação).
- Validar K-means via cross-tab com `MaintenanceNumber` (sem ground truth).
- Usar XGBoost + SHAP. Salvar com joblib (formato seguro recomendado).
- Split temporal preferível, NÃO random.
- Documentar viés de seleção (dataset só vê quem passou pela rede oficial).
```

### 8.2 `requirements.txt` provavelmente precisa de:
```
xgboost>=2.0
shap>=0.44
lifelines>=0.27  # opcional
```

### 8.3 `feature_dictionary.json` desatualizado

Atualmente descreve features socioeconômicas (idade, renda, score) do sintético. Reescrever pra descrever as 22 colunas do `vin_features.csv`.

### 8.4 `data/raw/vin_share_Desafio_02.xlsx`

O xlsx bruto **não está no repo** (85 MB, fora do gitignore). Está em `Ford_oficial/` no workspace local. Decisão do grupo: subir pra Git LFS ou manter fora do repo e baixar via instrução no README?

---

## 9. Como começar (30 min)

```bash
# 1. Entrar no repo
cd forward-repos/forward-ml

# 2. Setup do venv (se ainda não fez)
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate no Windows
pip install -r requirements.txt
pip install xgboost shap lifelines  # se faltarem

# 3. Confirmar que o input está lá
ls -lh data/processed/vin_features.csv  # ~31 MB
ls -lh data/external/                   # 4 CSVs (FIPE, dealers, recalls, manutenção)

# 4. Abrir Jupyter
jupyter lab
```

```python
# 5. Smoke test em notebook novo
import pandas as pd
df = pd.read_csv("data/processed/vin_features.csv", parse_dates=[
    "first_service_date", "last_service_date",
    "invoice_date", "sales_date", "warranty_start"
])
print(df.shape)  # (175554, 22)
print(df["model_name"].value_counts().head(10))
print(df["events_count"].describe())
```

Se isso roda, você está pronto. Comece pelo notebook `01_eda_official.ipynb`.

### Ordem de execução sugerida

1. **D-6 (dom 18/05)**: Notebook 01 (EDA) + 02 (feature eng documentation) — 4-5h
2. **D-5 (seg 19/05)**: Notebook 03 (K-means) — 3-4h
3. **D-4 (ter 20/05)**: Notebook 04 (classificação) — 3-4h
4. **D-3 (qua 21/05)**: Notebook 05 (survival, opcional) OU MODEL_CARD + feature_dictionary — 2-3h
5. **D-2 (qui 22/05)**: Relatório PDF — 3-4h

### Onde pedir socorro

- **02e** (`forward-docs/project/02e_DATASET_OFICIAL_E_FONTES.md`) — sempre o doc-fonte de qualquer dúvida sobre o dataset
- **CLAUDE.md** deste repo — convenções de código e nomenclatura
- **DOC 03 §4.1** — escopo exato das entregas Sprint 1
- **FOLLOW_UP_2026-05-17.md** — visão geral do grupo

---

> *Boa codada. Se algo aqui não estiver claro ou você discordar de uma decisão técnica, levante antes de codar — vale 30 min de conversa pra evitar 3h de retrabalho.*
