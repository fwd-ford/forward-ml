# forward-ml

![org](https://img.shields.io/badge/org-fwd--ford-blue?style=flat-square)
![stack](https://img.shields.io/badge/stack-Python_·_scikit--learn_·_XGBoost_·_FastAPI-333?style=flat-square)

Python ML service for **ForwardService** — customer segmentation, churn prediction, strategy simulation.

## Stack

- **Python 3.11+**
- **scikit-learn** + **XGBoost** for ML
- **SHAP** for model interpretability
- **FastAPI** for API endpoints (simulator)
- **pandas** + **matplotlib** + **seaborn** for analysis

## Structure

```
notebooks/                          # Jupyter notebooks (academic deliverable)
├── 01_exploratory_analysis.ipynb   # EDA on both bases
├── 02_segmentation.ipynb           # K-Means + RFM on Base 1
├── 03_classification.ipynb         # XGBoost on Base 2
└── 04_executive_report.ipynb       # Business insights summary
data/
├── raw/                            # Original datasets (Base 1, Base 2)
├── processed/                      # Cleaned/transformed data
└── models/                         # Exported models (.joblib)
src/
├── api.py                          # FastAPI endpoints (simulator)
├── segmentation.py                 # Segmentation pipeline
├── classification.py               # Classification pipeline
├── simulator.py                    # Strategy ROI simulator
├── scoring.py                      # Batch: calculate scores → write to Supabase
└── db.py                           # Supabase/PostgreSQL connection
reports/
└── relatorio_final.pdf             # Academic report deliverable
```

## Critical Rules

1. **NO DATA LEAKAGE**: Classification (Base 2) must NEVER use post-purchase behavioral variables.
   Base 2 contains only purchase-time features. Using future behavior invalidates the model entirely.

2. **Base separation**: Base 1 (full history) for segmentation ONLY. Base 2 (purchase-time) for classification ONLY.

3. **Cluster naming**: Always use business names (e.g., "loyal", "economic", "forgotten", "abandoned"), never generic labels ("Cluster 0", "Cluster 1").
