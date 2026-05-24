"""Análise do score V2 quebrada em 10 bandas de 0.1.

Para cada banda [0.0-0.1, 0.1-0.2, ..., 0.9-1.0]:
  - 1 tabela CSV detalhada (top modelos, top clusters, features médias)
  - 1 plot resumo (composição da banda)

Mais:
  - 1 tabela mestre com agregado das 10 bandas
  - 5 plots comparativos entre bandas

Outputs em resultados/bandas/
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUT = REPO_ROOT / "resultados" / "bandas"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "savefig.bbox": "tight"})
sns.set_style("whitegrid")

# Cores: gradiente verde→amarelo→vermelho ao longo das 10 bandas
BAND_COLORS = sns.color_palette("RdYlGn_r", n_colors=10)
BAND_EDGES = [round(i * 0.1, 1) for i in range(11)]   # 0.0, 0.1, ..., 1.0
BAND_LABELS = [f"B{i+1:02d} [{BAND_EDGES[i]:.1f}-{BAND_EDGES[i+1]:.1f})" for i in range(10)]


def load_data():
    df = pd.read_csv(REPO_ROOT / "resultados" / "scoring_results.csv",
                     parse_dates=["Data da venda", "Data do último serviço"])
    # apelidos curtos para facilitar
    df = df.rename(columns={
        "ID do veículo (hash)": "vin",
        "Modelo do carro": "modelo",
        "Ano do modelo": "ano",
        "Churnou (1) / Retido (0)": "churned",
        "Probabilidade de churn": "score",
        "churn_probability_raw": "score_raw",
        "Predição (0/1)": "pred",
        "Tier de risco": "tier",
        "Nome do cluster K-means": "cluster",
        "Total de serviços": "n_servicos",
        "Vida do carro na rede (dias)": "tenure",
        "Dias desde o último serviço": "dias_sem_servico",
        "KM máximo registrado": "km",
        "% serviços na concessionária preferida": "dealer_share",
        "Cliente fez só 1 serviço": "single_event",
    })

    bins = BAND_EDGES.copy()
    bins[0] = -0.001
    bins[-1] = 1.001
    df["banda"] = pd.cut(df["score"], bins=bins, labels=BAND_LABELS, include_lowest=True)
    return df


def tabela_mestre(df):
    rows = []
    for label in BAND_LABELS:
        sub = df[df["banda"] == label]
        if len(sub) == 0:
            rows.append({"banda": label, "n": 0})
            continue
        rows.append({
            "banda": label,
            "n": len(sub),
            "pct_total": round(len(sub) / len(df) * 100, 2),
            "churn_real_pct": round(sub["churned"].mean() * 100, 2),
            "score_medio": round(sub["score"].mean(), 4),
            "score_mediano": round(sub["score"].median(), 4),
            "n_servicos_medio": round(sub["n_servicos"].mean(), 2),
            "tenure_medio_dias": round(sub["tenure"].mean(), 1),
            "km_medio": round(sub["km"].mean(), 0),
            "dealer_share_medio": round(sub["dealer_share"].mean(), 3),
            "single_event_pct": round(sub["single_event"].mean() * 100, 1),
            "top_modelo_1": sub["modelo"].value_counts().head(1).index.tolist()[0] if len(sub) else "",
            "top_modelo_2": sub["modelo"].value_counts().head(2).index.tolist()[1] if len(sub) > 1 and sub["modelo"].nunique() > 1 else "",
            "top_cluster": sub["cluster"].value_counts().head(1).index.tolist()[0] if len(sub) else "",
        })
    master = pd.DataFrame(rows)
    master.to_csv(OUT / "T00_resumo_bandas.csv", index=False, encoding="utf-8")
    print(f"  wrote bandas/T00_resumo_bandas.csv")
    return master


def tabela_por_banda(df, label, idx):
    sub = df[df["banda"] == label]
    if len(sub) == 0:
        return

    # composição por modelo
    modelo_dist = (sub["modelo"].value_counts()
                    .reset_index().rename(columns={"index": "modelo", "modelo": "n"}))
    modelo_dist["pct_da_banda"] = (modelo_dist.iloc[:, 1] / len(sub) * 100).round(2)
    modelo_dist["churn_real_pct"] = sub.groupby("modelo")["churned"].mean().reindex(modelo_dist.iloc[:, 0]).values
    modelo_dist["churn_real_pct"] = (modelo_dist["churn_real_pct"] * 100).round(2)
    modelo_dist.columns = ["modelo", "n", "pct_da_banda", "churn_real_pct"]

    # composição por cluster
    cluster_dist = (sub["cluster"].value_counts()
                    .reset_index())
    cluster_dist.columns = ["cluster", "n"]
    cluster_dist["pct_da_banda"] = (cluster_dist["n"] / len(sub) * 100).round(2)
    cluster_dist["churn_real_pct"] = (sub.groupby("cluster")["churned"].mean().reindex(cluster_dist["cluster"]).values * 100).round(2)

    # features medias
    feature_means = pd.DataFrame([{
        "feature": "n_servicos_medio",  "valor": round(sub["n_servicos"].mean(), 2),
    }, {
        "feature": "tenure_medio_dias", "valor": round(sub["tenure"].mean(), 1),
    }, {
        "feature": "dias_sem_servico_medio", "valor": round(sub["dias_sem_servico"].mean(), 1),
    }, {
        "feature": "km_medio", "valor": round(sub["km"].mean(), 0),
    }, {
        "feature": "dealer_share_medio", "valor": round(sub["dealer_share"].mean(), 3),
    }, {
        "feature": "single_event_pct", "valor": round(sub["single_event"].mean() * 100, 2),
    }])

    # combinar em 1 CSV com seções (modelo, cluster, features)
    name = f"T{idx+1:02d}_banda_{BAND_EDGES[idx]:.1f}-{BAND_EDGES[idx+1]:.1f}.csv"
    with open(OUT / name, "w", encoding="utf-8") as f:
        f.write(f"# Análise da banda {label}\n")
        f.write(f"# n_VINs = {len(sub):,}  |  churn_real = {sub['churned'].mean():.2%}  |  score_medio = {sub['score'].mean():.4f}\n\n")
        f.write("# === Composição por MODELO de carro ===\n")
        modelo_dist.head(15).to_csv(f, index=False)
        f.write("\n# === Composição por CLUSTER K-means ===\n")
        cluster_dist.to_csv(f, index=False)
        f.write("\n# === Médias de features ===\n")
        feature_means.to_csv(f, index=False)
    print(f"  wrote bandas/{name}")


def plot_overview_bandas(master, df):
    """5 plots comparativos."""
    # P01 — Quantidade por banda
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(master)), master["n"], color=BAND_COLORS)
    ax.set_xticks(range(len(master)))
    ax.set_xticklabels([l.split()[0] for l in master["banda"]], rotation=0)
    ax.set_xlabel("Banda de score (0.1 cada)")
    ax.set_ylabel("# VINs")
    ax.set_title("Quantidade de VINs por banda de score")
    for b, n, p in zip(bars, master["n"], master["pct_total"]):
        ax.text(b.get_x() + b.get_width()/2, n + master["n"].max() * 0.01,
                f"{n:,}\n({p:.1f}%)", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "P01_quantidade_por_banda.png")
    plt.close(fig)
    print("  wrote bandas/P01_quantidade_por_banda.png")

    # P02 — Churn real por banda (deve ser quase monotonic crescente se modelo é bom)
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(master)), master["churn_real_pct"], color=BAND_COLORS)
    ax.set_xticks(range(len(master)))
    ax.set_xticklabels([l.split()[0] for l in master["banda"]], rotation=0)
    ax.set_xlabel("Banda de score")
    ax.set_ylabel("Churn real (%)")
    ax.set_title("Calibração — churn real por banda (esperamos curva ascendente)")
    for b, v in zip(bars, master["churn_real_pct"]):
        ax.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.1f}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(OUT / "P02_churn_real_por_banda.png")
    plt.close(fig)
    print("  wrote bandas/P02_churn_real_por_banda.png")

    # P03 — Composição por modelo (stacked bar)
    top_models = df["modelo"].value_counts().head(8).index.tolist()
    df_top = df[df["modelo"].isin(top_models)].copy()
    comp = (df_top.groupby(["banda", "modelo"], observed=True).size()
              .unstack(fill_value=0))
    comp = comp.reindex(BAND_LABELS).reindex(columns=top_models)
    comp_pct = comp.div(comp.sum(axis=1), axis=0).fillna(0) * 100

    fig, ax = plt.subplots(figsize=(14, 7))
    comp_pct.plot(kind="bar", stacked=True, ax=ax, colormap="tab10", width=0.85)
    ax.set_xticklabels([l.split()[0] for l in comp_pct.index], rotation=0)
    ax.set_xlabel("Banda de score")
    ax.set_ylabel("% da banda")
    ax.set_title("Composição por modelo de carro (top 8) dentro de cada banda")
    ax.legend(title="Modelo", loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(OUT / "P03_composicao_modelo.png")
    plt.close(fig)
    print("  wrote bandas/P03_composicao_modelo.png")

    # P04 — Composição por cluster
    comp_c = (df.groupby(["banda", "cluster"], observed=True).size().unstack(fill_value=0))
    comp_c = comp_c.reindex(BAND_LABELS)
    comp_c_pct = comp_c.div(comp_c.sum(axis=1), axis=0).fillna(0) * 100
    fig, ax = plt.subplots(figsize=(14, 7))
    comp_c_pct.plot(kind="bar", stacked=True, ax=ax,
                     color=["#55a868", "#dd8452", "#c44e52", "#4c72b0"], width=0.85)
    ax.set_xticklabels([l.split()[0] for l in comp_c_pct.index], rotation=0)
    ax.set_xlabel("Banda de score")
    ax.set_ylabel("% da banda")
    ax.set_title("Composição por cluster K-means dentro de cada banda")
    ax.legend(title="Cluster", loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(OUT / "P04_composicao_cluster.png")
    plt.close(fig)
    print("  wrote bandas/P04_composicao_cluster.png")

    # P05 — Heatmap de features médias por banda
    feature_cols = ["n_servicos", "tenure", "dias_sem_servico", "km", "dealer_share"]
    means = df.groupby("banda", observed=True)[feature_cols].mean().reindex(BAND_LABELS)
    # normalizar coluna a coluna pra visualizar relativa (max=1)
    means_norm = means.div(means.max(axis=0), axis=1)
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.heatmap(means_norm.T, annot=means.T, fmt=".1f", cmap="YlOrRd",
                xticklabels=[l.split()[0] for l in means_norm.index],
                yticklabels=["Total de serviços", "Vida na rede (dias)",
                              "Dias sem serviço", "KM máximo", "% dealer preferido"],
                cbar_kws={"label": "Normalizado (0-1)"}, ax=ax)
    ax.set_xlabel("Banda de score")
    ax.set_title("Médias das features por banda (cor=normalizada; texto=valor real)")
    fig.tight_layout()
    fig.savefig(OUT / "P05_features_heatmap.png")
    plt.close(fig)
    print("  wrote bandas/P05_features_heatmap.png")


def main():
    print("Carregando scoring_results.csv...")
    df = load_data()
    print(f"  {len(df):,} VINs\n")

    print("Gerando tabela mestre...")
    master = tabela_mestre(df)

    print("\nGerando 10 tabelas por banda...")
    for idx, label in enumerate(BAND_LABELS):
        tabela_por_banda(df, label, idx)

    print("\nGerando 5 plots comparativos...")
    plot_overview_bandas(master, df)

    print(f"\n=== Resumo das 10 bandas ===")
    print(master[["banda", "n", "pct_total", "churn_real_pct", "score_medio"]].to_string(index=False))
    print(f"\n✓ {len(list(OUT.glob('*.csv')))} CSVs + {len(list(OUT.glob('*.png')))} plots em {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
