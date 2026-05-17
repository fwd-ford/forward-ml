# forward-ml — Repository Instructions

## Language Policy

- Code and comments: English.
- Notebook markdown cells: Portuguese (pt-BR) for academic deliverable.
- Report (PDF): Portuguese.
- Variable names and function names: always English.

## Stack

- Python 3.11+
- pandas, numpy, scikit-learn
- XGBoost, SHAP
- lifelines (survival analysis — Kaplan-Meier)
- matplotlib, seaborn
- joblib (model persistence — safe format)
- FastAPI (future API service — out of Sprint 1 scope)
- psycopg2 or supabase-py (database connection)

## Dataset (v2 — official Ford, 2026-05-11)

The official Ford dataset replaced the synthetic warm-up CSVs. Key facts:

- **Source:** `data/raw/vin_share_Desafio_02.xlsx` (provided by FIAP coordination, 11/05/2026)
- **Shape:** 602,788 service event rows × 25 columns, 175,554 unique `VIN_Hash`
- **Granularity:** 1 row = 1 service event (NOT 1 row = 1 customer)
- **Scope:** 100% Brazil (`Country` column constant), 435 dealers, 21 models, model years 2017-2026
- **No labels, no socioeconomic features.** Target must be engineered (see Critical ML Rules below)
- **Behavioral features already derived:** `data/processed/vin_features.csv` (175k rows × 22 columns) — main ML input
- **External enrichment in `data/external/`:** FIPE values, Ford dealers (ABRADIF), Senacon recalls, official maintenance schedules

Full context: [forward-docs/project/02e_DATASET_OFICIAL_E_FONTES.md](../forward-docs/project/02e_DATASET_OFICIAL_E_FONTES.md).
Onboarding for ML developers: [HANDOFF_ML.md](./HANDOFF_ML.md).

## Critical ML Rules (v2 — post official dataset)

### Data preparation

- Drop constant columns: `Country`, `ServiceType`, `StatusUSA` (all single-valued).
- Clip `KM` to 500,000 (outliers up to 955M exist in raw data — physically impossible).
- `MainSource` has a duplicate value with trailing whitespace — apply `.str.strip()`.
- `ServiceDate` and related date columns have mixed BR (`dd/mm/yyyy`) and US (`m/d/yyyy`) formats in the same column. Parse defensively, trying both formats.
- `VIN_Hash` is SHA1 with high robustness (5M reversal attempts produced 0 matches). Treat as **pseudonymized** per LGPD; do NOT attempt to join with external identifying sources.

### Modeling strategy

- **Three layers**, in priority order:
  1. **Blind K-means segmentation** on behavioral features. NO ground truth available — validate via:
     - Silhouette ≥ 0.35 (real-world acceptable for behavioral data)
     - Cross-tab with `MaintenanceNumber` (loyal clusters should concentrate in revisions 5+; abandoned in 1-2)
     - Cross-tab with `model_name`
     - Manual cluster naming after inspecting feature means (fiel/econômico/esquecido/abandono per DOC 00 nomenclature)
  2. **Classification with engineered target.** Target: `churned = days_since_last_service > 365`. Reference date should be `max(ServiceDate)` from dataset, NOT `datetime.now()` (reproducibility).
  3. **Survival analysis (optional).** Kaplan-Meier per `model_name × model_year` using `lifelines`.

### Critical constraints

- **Exclude VINs with `tenure_days < 365` from classification training** — insufficient observation time to label as churned.
- **Prefer temporal split** over random `train_test_split` when feasible.
- **Cost-sensitive learning:** false negatives (missing real churners) cost more than false positives. Use `scale_pos_weight` (XGBoost) or threshold optimization.
- **Handle model imbalance:** RANGER (56.7%) + KA (22.2%) dominate the dataset. Use `sample_weight` inversely proportional to model frequency, or stratify by `model_name`, to prevent the classifier from collapsing into a "Ranger detector".
- **Document selection bias in every model card:** the dataset only sees VINs that visited the official network. The ~95% of Ford vehicles that go to independent shops are invisible. Models predict churn *within* the visible population, not from total fleet.

### Tooling

- Persist trained models with `joblib` (safe serialization). Do NOT use raw stdlib serialization for cross-process model artifacts.
- Use XGBoost + SHAP for the main classifier. SHAP values feed both the report and the LSV reasoning.
- For categorical encoding of high-cardinality columns (`ServiceCode` with 139 unique values, `DealerCode` with 435), prefer frequency encoding or target encoding over one-hot.

### Historical note (synthetic dataset rules — superseded)

The previous version of these rules required anti-data-leakage separation between "Base 1" (segmentation) and "Base 2" (classification), with 13 columns flagged as post-purchase. **Those rules applied to the synthetic warm-up CSVs (`ford_clientes_*.csv`) and are no longer enforceable on the official dataset** — there are no pre-built labels and no post-purchase variables in the same sense. The synthetic CSVs are retained in `tmp_research/` (workspace) as input for an optional parallel propensity model documented in the final report (ADR-012).

## Batch Scoring

The scoring pipeline (`src/scoring.py`, to be implemented) connects directly to Supabase PostgreSQL,
reads vehicle behavioral features (joined from `vin_features` table), runs inference, and writes scores
to the `client_scores` table. This runs as a scheduled job (cron, 1x/day), not in real-time.

## API (FastAPI — future, out of Sprint 1)

Minimal API with 3-4 endpoints (Sprint 1 calls scoring directly from Java backend; FastAPI is planned for Sprint 2+):

- POST /predict — predict churn probability and segment for a given VIN (using its behavioral features)
- POST /simulate — run strategy simulation, return projected ROI
- GET /health — health check
