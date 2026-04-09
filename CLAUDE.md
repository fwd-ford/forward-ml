# forward-ml — Repository Instructions

## Language Policy

- Code and comments: English.
- Notebook markdown cells: Portuguese (pt-BR) for academic deliverable.
- Report (PDF): Portuguese.
- Variable names and function names: always English.

## Stack

- Python 3.11+
- scikit-learn, XGBoost, SHAP
- pandas, numpy, matplotlib, seaborn
- FastAPI (API service)
- psycopg2 or supabase-py (database connection)

## Critical ML Rules

- NEVER use post-purchase variables in the classification model (Base 2).
  This includes: service count, time to maintenance, spending, any post-purchase indicator.
  Using these constitutes DATA LEAKAGE and invalidates the model.
- Base 1 = segmentation (unsupervised, full history). Base 2 = classification (supervised, purchase-time only).
- Validate clusters with Silhouette Score (target > 0.4).
- Use XGBoost + SHAP for classification. Prioritize recall over precision.
- Export models with joblib, not pickle.

## Batch Scoring

The scoring pipeline (`src/scoring.py`) connects directly to Supabase PostgreSQL,
reads vehicle/customer data, runs inference, and writes scores to the `churn_scores` table.
This runs as a scheduled job (cron), not in real-time.

## API (FastAPI)

Minimal API with 3-4 endpoints:
- POST /predict — predict churn profile for a new customer
- POST /simulate — run strategy simulation, return projected ROI
- GET /health — health check
