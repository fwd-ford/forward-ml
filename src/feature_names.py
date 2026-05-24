"""Mapping from machine-readable feature names to human-readable Portuguese labels.

Centralizes the translations used in plots, CSVs and reports.
Keep the code-side names (snake_case English) — only the display layer changes.
"""
from __future__ import annotations
from typing import Iterable

FRIENDLY_NAMES: dict[str, str] = {
    # ----- identity -----
    "vin_hash": "ID do veículo (hash)",
    "model_name": "Modelo do carro",
    "model_year": "Ano do modelo",
    "model_name_freq": "Popularidade do modelo na frota",
    "primary_dealer": "Concessionária preferida (código)",

    # ----- frequency -----
    "events_count": "Total de serviços",
    "tenure_days": "Vida do carro na rede (dias)",
    "gap_avg_days": "Intervalo médio entre serviços (dias)",
    "gap_min_days": "Menor intervalo entre serviços (dias)",
    "gap_max_days": "Maior intervalo entre serviços (dias)",
    "gap_last_days": "Intervalo entre os 2 últimos serviços (dias)",
    "gap_last_vs_avg": "Último intervalo / intervalo médio",

    # ----- recency -----
    "first_service_date": "Data do 1º serviço",
    "last_service_date": "Data do último serviço",
    "days_since_last_service": "Dias desde o último serviço",

    # ----- usage -----
    "km_max": "KM máximo registrado",
    "km_min": "KM mínimo registrado",
    "km_per_month": "KM rodados por mês",

    # ----- loyalty -----
    "dealers_distinct": "Concessionárias distintas visitadas",
    "primary_dealer_share": "% serviços na concessionária preferida",
    "is_loyal_to_dealer": "Cliente fiel à concessionária (>80%)",

    # ----- service mix -----
    "service_codes_distinct": "Tipos de serviço distintos (códigos)",
    "service_types_distinct": "Categorias de serviço distintas",

    # ----- lifecycle -----
    "invoice_date": "Data de emissão da nota",
    "sales_date": "Data da venda",
    "warranty_start": "Início da garantia",
    "age_at_last_service_days": "Idade do carro no último serviço (dias)",
    "is_warranty_active": "Última visita ainda na garantia",
    "is_single_event": "Cliente fez só 1 serviço",

    # ----- enriched cohort / seasonality (v2) -----
    "sales_year": "Ano da venda",
    "sales_month": "Mês da venda",
    "sales_quarter": "Trimestre da venda",
    "years_since_sale": "Anos desde a venda",
    "age_when_first_serviced_days": "Idade do carro no 1º serviço (dias)",
    "first_service_month": "Mês do 1º serviço",
    "sale_year_minus_model_year": "Diferença ano venda − ano modelo",
    "gap_range_ratio": "Razão maior/menor intervalo entre serviços",
    "gap_skew": "Assimetria dos intervalos entre serviços",
    "services_per_year": "Serviços por ano (na rede)",

    # ----- target / output -----
    "churned": "Churnou (1) / Retido (0)",
    "trainable": "Apto para treino",
    "churn_probability": "Probabilidade de churn",
    "churn_predicted": "Predição (0/1)",
    "churn_decile": "Decile do score",
    "churn_risk_tier": "Tier de risco",
    "cluster_id": "ID do cluster K-means",
    "cluster_name": "Nome do cluster K-means",
}


def to_friendly(names: Iterable[str]) -> list[str]:
    """Map a sequence of code-side names to friendly Portuguese labels.

    Names absent from the mapping are returned unchanged.
    """
    return [FRIENDLY_NAMES.get(n, n) for n in names]


def rename_df(df, axis: str = "columns"):
    """Return a copy of df with axis labels translated to friendly names."""
    return df.rename(columns=FRIENDLY_NAMES) if axis == "columns" else df.rename(index=FRIENDLY_NAMES)
