"""Generate consolidated Sprint 1 PDF report.

Consolidates MODEL_CARD_CLASSIFIER + MODEL_CARD_SEGMENTATION + RESUMO_FINAL
into a single PDF for academic delivery (Prof. Carlos / FIAP).

Closes issues #5 [ML-4] and #16 [Sprint 1 ML].

Usage:
    python -m scripts.build_relatorio_pdf
"""
from __future__ import annotations
import re
from datetime import date
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, black
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Image,
    Table,
    TableStyle,
    KeepTogether,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = ROOT / "reports" / "relatorio-ml.pdf"
FIG_DIR = ROOT / "reports" / "figures"

PROJECT_TITLE = "Forward — Segmentação e Classificação de Churn da Rede Oficial Ford"
SUBTITLE = "Sprint 1 — Pipeline ML (Segmentação + Classificação + Calibração)"
AUTHORS = "Lucca Saraiva Borges"
DISCIPLINE = "Inteligência Artificial e Machine Learning — Prof. Carlos Fontoura (FIAP)"


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

BRAND = HexColor("#003478")  # Ford blue
ACCENT = HexColor("#0a6bcb")
GRAY = HexColor("#555555")


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title", parent=base["Title"], fontSize=22, leading=26,
            textColor=BRAND, spaceAfter=18, alignment=1,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Heading2"], fontSize=14, leading=18,
            textColor=GRAY, spaceAfter=24, alignment=1,
        ),
        "cover_meta": ParagraphStyle(
            "CoverMeta", parent=base["Normal"], fontSize=11, leading=15,
            textColor=black, alignment=1, spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontSize=18, leading=22,
            textColor=BRAND, spaceBefore=18, spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontSize=14, leading=18,
            textColor=ACCENT, spaceBefore=12, spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "H3", parent=base["Heading3"], fontSize=12, leading=16,
            textColor=BRAND, spaceBefore=8, spaceAfter=4,
        ),
        "h4": ParagraphStyle(
            "H4", parent=base["Heading4"], fontSize=11, leading=14,
            textColor=GRAY, spaceBefore=6, spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontSize=10, leading=14,
            spaceAfter=6, alignment=4,  # justified
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=base["BodyText"], fontSize=10, leading=14,
            leftIndent=14, bulletIndent=4, spaceAfter=2,
        ),
        "code": ParagraphStyle(
            "Code", parent=base["Code"], fontSize=8.5, leading=11,
            backColor=HexColor("#f4f4f4"), leftIndent=8, rightIndent=8,
            spaceAfter=8, spaceBefore=6, textColor=HexColor("#222222"),
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["Normal"], fontSize=8.5, leading=11,
            textColor=GRAY, alignment=1, spaceAfter=10,
        ),
    }


# ---------------------------------------------------------------------------
# Markdown → flowables (subset that covers our model cards)
# ---------------------------------------------------------------------------

INLINE_BOLD = re.compile(r"\*\*([^*]+)\*\*")
INLINE_CODE = re.compile(r"`([^`]+)`")
INLINE_ITALIC = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
INLINE_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inline_md(text: str) -> str:
    text = _xml_escape(text)
    text = INLINE_LINK.sub(r"<font color='#0a6bcb'>\1</font>", text)
    text = INLINE_BOLD.sub(r"<b>\1</b>", text)
    text = INLINE_ITALIC.sub(r"<i>\1</i>", text)
    text = INLINE_CODE.sub(r"<font face='Courier' size='9'>\1</font>", text)
    return text


def _md_table_to_flowable(lines: list[str], styles: dict) -> Table | None:
    rows = []
    for ln in lines:
        ln = ln.strip()
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        rows.append(cells)
    if len(rows) < 2:
        return None
    # remove separator row (---|---|---)
    rows = [r for r in rows if not all(set(c.replace(":", "").replace("-", "").strip()) <= {""} for c in r)]
    rendered = []
    for i, r in enumerate(rows):
        style = styles["body"] if i > 0 else styles["body"]
        rendered.append([Paragraph(f"<b>{_inline_md(c)}</b>" if i == 0 else _inline_md(c), style) for c in r])
    tbl = Table(rendered, repeatRows=1, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#e6eef9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), BRAND),
        ("LINEBELOW", (0, 0), (-1, 0), 1.0, BRAND),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, HexColor("#dddddd")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return tbl


def md_to_flowables(md_text: str, styles: dict, drop_top_h1: bool = False) -> list:
    """Render a small subset of Markdown into reportlab flowables.

    Supports: # / ## / ### / #### headings, paragraphs, - / * bullets,
    fenced ```code``` blocks, GitHub-style | tables |.
    """
    flow: list = []
    lines = md_text.splitlines()
    i = 0
    top_h1_seen = False
    while i < len(lines):
        ln = lines[i]
        stripped = ln.rstrip()

        # Fenced code block
        if stripped.startswith("```"):
            j = i + 1
            buf = []
            while j < len(lines) and not lines[j].lstrip().startswith("```"):
                buf.append(lines[j])
                j += 1
            code_text = "\n".join(buf)
            # XPreformatted preserves whitespace and respects font
            from reportlab.platypus import Preformatted
            flow.append(Preformatted(code_text, styles["code"]))
            i = j + 1
            continue

        # Table
        if stripped.startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            tbl = _md_table_to_flowable(block, styles)
            if tbl is not None:
                flow.append(KeepTogether([tbl, Spacer(1, 6)]))
            continue

        # Headings
        if stripped.startswith("#### "):
            flow.append(Paragraph(_inline_md(stripped[5:]), styles["h4"]))
        elif stripped.startswith("### "):
            flow.append(Paragraph(_inline_md(stripped[4:]), styles["h3"]))
        elif stripped.startswith("## "):
            flow.append(Paragraph(_inline_md(stripped[3:]), styles["h2"]))
        elif stripped.startswith("# "):
            if drop_top_h1 and not top_h1_seen:
                top_h1_seen = True
            else:
                flow.append(Paragraph(_inline_md(stripped[2:]), styles["h1"]))
        elif stripped.startswith("- ") or stripped.startswith("* "):
            flow.append(Paragraph(_inline_md(stripped[2:]), styles["bullet"], bulletText="•"))
        elif stripped == "":
            flow.append(Spacer(1, 4))
        else:
            flow.append(Paragraph(_inline_md(stripped), styles["body"]))

        i += 1

    return flow


# ---------------------------------------------------------------------------
# Document sections
# ---------------------------------------------------------------------------

def cover_page(styles: dict) -> list:
    today = date.today().strftime("%d/%m/%Y")
    return [
        Spacer(1, 4 * cm),
        Paragraph(PROJECT_TITLE, styles["title"]),
        Paragraph(SUBTITLE, styles["subtitle"]),
        Spacer(1, 4 * cm),
        Paragraph(f"<b>Autor:</b> {AUTHORS}", styles["cover_meta"]),
        Paragraph(f"<b>Disciplina:</b> {DISCIPLINE}", styles["cover_meta"]),
        Paragraph(f"<b>Data:</b> {today}", styles["cover_meta"]),
        Paragraph(
            "<b>Repositório:</b> github.com/fwd-ford/forward-ml — branch sprint-1-ml-pipeline",
            styles["cover_meta"],
        ),
        Spacer(1, 2 * cm),
        Paragraph(
            "Pipeline reprodutível de segmentação não-supervisionada (K-means, k=4) "
            "e classificação binária de churn (XGBoost two-stage com calibração "
            "estratificada) sobre o dataset oficial Ford Brasil — 175.552 VINs × "
            "22 features comportamentais. Inclui documentação LGPD completa, model "
            "cards por modelo e análise de viés de seleção.",
            styles["body"],
        ),
        PageBreak(),
    ]


def executive_summary(styles: dict) -> list:
    body = """
## Sumário Executivo

**Problema.** A Ford Brasil precisa identificar quais clientes da rede oficial estão em risco de
abandonar a manutenção autorizada (churn) para direcionar campanhas de retenção com ROI mensurável.
O dataset oficial fornecido pela coordenação FIAP contém **602.788 eventos de serviço** de
**175.554 VINs únicos** ao longo de 2017–2026, sem rótulo de churn e sem qualquer feature
socioeconômica.

**Abordagem.** Pipeline em três camadas:

- **Camada 1 — Segmentação K-means cega** sobre 11 features comportamentais → 4 personas com
nomes de negócio (`fiel`, `econômico`, `esquecido`, `abandono`). Silhouette **0,40** (acima do
piso 0,35 aceito para clustering comportamental).
- **Camada 2 — Classificação supervisionada** com target engenheirado (`churned =
days_since_last_service &gt; 365`, com reference_date fixa para reprodutibilidade) usando XGBoost
two-stage + calibração isotônica estratificada por *scoring quality*. **Acurácia balanceada
87,0%**, **F1 0,865**, **Brier 0,128**.
- **Camada 3 (opcional, Sprint 2)** — Análise de sobrevivência Kaplan-Meier por modelo × ano para
projeção de curvas de retenção.

**Principais resultados.**

- O classificador final V3 atinge ROC-AUC **0,93** no dataset completo e **0,998** em cohorts
maduras (2020–2022).
- Cohorts 2024–2025 têm ROC-AUC degradado por *right-censoring* (definição de churn de 365 dias
não é observável em VINs jovens) — **limitação do target, não do modelo**.
- A persona `abandono` colide com `churned=1` em 99,98%, validação cruzada cega que confirma o
alinhamento entre o classificador e o segmentador.
- O *cohort-aware sample weight* removeu o colapso do V2 em "detector da cohort 2020" e produziu
métricas honestas em todos os anos de venda.

**Limitação principal.** O dataset só vê os ~5% da frota Ford que passou pela rede oficial. O
modelo prediz churn *dentro da população visível*, não da frota total. Todo Model Card documenta
esse viés explicitamente, conforme exigência do CLAUDE.md.

**Próximos passos.** Modelo paralelo de propensão (ADR-012), análise de sobrevivência com
`lifelines`, deploy via FastAPI + scoring batch para Supabase (Sprint 2).
"""
    return md_to_flowables(body, styles)


def insert_figure(fig_path: Path, caption: str, styles: dict, width_cm: float = 14.0) -> list:
    if not fig_path.exists():
        return [
            Paragraph(f"<i>[figura {fig_path.name} indisponível]</i>", styles["caption"]),
            Spacer(1, 6),
        ]
    img = Image(str(fig_path), width=width_cm * cm, height=width_cm * cm * 0.62, kind="proportional")
    return [KeepTogether([img, Paragraph(caption, styles["caption"])])]


def appendix_figures(styles: dict) -> list:
    figs = [
        ("02_shap_beeswarm_V3_safe_no_single.png", "Figura A1 — SHAP beeswarm (V3 sem single-event)."),
        ("03_shap_bar_V3_safe_no_single.png", "Figura A2 — SHAP bar plot (importância média absoluta)."),
        ("06_eval_V3_safe_no_single.png", "Figura A3 — Curvas ROC / PR / Calibração (V3)."),
        ("07_per_group_V3_safe_no_single.png", "Figura A4 — Métricas por grupo (modelo × cohort)."),
        ("08_leak_audit_comparison.png", "Figura A5 — Auditoria anti-data-leakage (comparativo V1 → V3)."),
    ]
    flow = [Paragraph("Apêndice A — Figuras", styles["h1"])]
    for fname, caption in figs:
        flow.extend(insert_figure(FIG_DIR / fname, caption, styles))
        flow.append(Spacer(1, 6))
    return flow


# ---------------------------------------------------------------------------
# Page header / footer
# ---------------------------------------------------------------------------

def _on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GRAY)
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Forward / Sprint 1 ML — pág. {doc.page}")
    canvas.drawString(2 * cm, 1.2 * cm, "forward-ml — Lucca Saraiva Borges")
    canvas.setStrokeColor(HexColor("#dddddd"))
    canvas.line(2 * cm, 1.6 * cm, A4[0] - 2 * cm, 1.6 * cm)
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build():
    styles = build_styles()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title=PROJECT_TITLE,
        author=AUTHORS,
    )

    story: list = []
    story.extend(cover_page(styles))
    story.extend(executive_summary(styles))
    story.append(PageBreak())

    story.append(Paragraph("Contexto do dataset oficial", styles["h1"]))
    story.extend(md_to_flowables("""
**Fonte.** `data/raw/vin_share_Desafio_02.xlsx`, entregue pela coordenação FIAP em 11/05/2026.
Substituiu os datasets sintéticos de aquecimento (`ford_clientes_*.csv`) usados no início da
sprint. Detalhes completos em `forward-docs/project/02e_DATASET_OFICIAL_E_FONTES.md`.

**Forma.** 602.788 linhas × 25 colunas. Granularidade: 1 linha = 1 evento de serviço. **175.554
VINs únicos**, 21 modelos, 435 dealers, model years 2017–2026, 100% Brasil.

**Engenharia de features.** O script `src/features.py` converte os eventos brutos em uma matriz
`vin_features.csv` de 175k × 22 colunas, agregando por `VIN_Hash`. Regras críticas aplicadas:

- `km_max` truncado em 500.000 (raw chega a 955M, fisicamente impossível).
- `ServiceDate` parseada defensivamente em formato BR e US (mistos no mesmo coluna).
- `MainSource` com `.str.strip()` (duplicidade por whitespace).
- `VIN_Hash` (SHA1) tratado como **pseudonimização robusta**, não anonimização.
- `reference_date` = `max(ServiceDate)` = **2026-05-04** (fixa, reprodutível — não usar
`datetime.now()`).

**Enriquecimento externo** (em `data/external/`): FIPE values, dealers ABRADIF, recalls Senacon,
cronogramas oficiais de manutenção. Usado para validação cruzada, não como feature direta
(privacidade + governança).
""", styles))
    story.append(PageBreak())

    # Model card — Segmentação
    story.append(Paragraph("Modelo 1 — Segmentação K-means", styles["h1"]))
    mc_seg = (ROOT / "MODEL_CARD_SEGMENTATION.md").read_text(encoding="utf-8")
    story.extend(md_to_flowables(mc_seg, styles, drop_top_h1=True))
    story.append(PageBreak())

    # Model card — Classifier
    story.append(Paragraph("Modelo 2 — Classificador Churn V3", styles["h1"]))
    mc_cls = (ROOT / "MODEL_CARD_CLASSIFIER.md").read_text(encoding="utf-8")
    story.extend(md_to_flowables(mc_cls, styles, drop_top_h1=True))
    story.append(PageBreak())

    # Results summary
    story.append(Paragraph("Análise por decil operacional (resumo)", styles["h1"]))
    resumo = (ROOT / "resultados" / "RESUMO_FINAL.md").read_text(encoding="utf-8")
    story.extend(md_to_flowables(resumo, styles, drop_top_h1=True))
    story.append(PageBreak())

    # Conclusions
    story.append(Paragraph("Conclusões e roadmap Sprint 2", styles["h1"]))
    story.extend(md_to_flowables("""
**Status Sprint 1.**

- ✅ Pipeline reprodutível com `load_features() → segmentation.fit() → ChurnScorer.train() → score_final`.
- ✅ Dois modelos com Model Card completo (LGPD, viés, limitações documentadas).
- ✅ Notebooks acadêmicos em `notebooks/` para os 4 estágios (EDA, segmentação, classificação, leak audit).
- ✅ Anti-data-leakage auditado em `05_leak_audit.ipynb` — features pós-evento foram removidas do conjunto final V3.
- ✅ Calibração estratificada por *scoring quality* — resolveu over-confidence de +15pp (single-event) e +48pp (holdout) do V2.

**Roadmap Sprint 2.**

1. **Survival analysis** (Kaplan-Meier) por `model_name × model_year` com `lifelines` — projeção de curvas de retenção complementar ao classificador binário.
2. **Modelo paralelo de propensão** sobre o subset sintético antigo (ADR-012) — comparativo metodológico para o relatório acadêmico.
3. **API FastAPI** com `POST /predict`, `POST /simulate`, `GET /health` — desacoplar o Java backend do `joblib` direto.
4. **Scoring batch** para Supabase via cron (1×/dia) — escrever em `client_scores` consumido pelo CRM Ford.
5. **Monitoramento de drift**: alertar quando `positive_rate` se mover &gt; 5pp ou silhouette do segmentador cair &lt; 0,30 no retreino.

**Riscos abertos.**

- Cohort 2024–2025 sem rótulo confiável (right-censoring) — usar score *com cautela*, idealmente combinar com regra de negócio (\"se sales_date &lt; 12 meses, suprimir\").
- Modelo treina em dataset filtrado (`tenure_days ≥ 365`) mas scoring final aplica em todos — métricas globais são *honestas-mas-otimistas*. O piso real está nos 55% de balanced accuracy em cohort 2024.
- Dataset cobre apenas a rede oficial — modelos NÃO devem ser usados para inferir comportamento da frota total Ford.
""", styles))
    story.append(PageBreak())

    # Appendix
    story.extend(appendix_figures(styles))

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    size_kb = OUT_PDF.stat().st_size / 1024
    print(f"OK — {OUT_PDF} ({size_kb:,.1f} KB)")


if __name__ == "__main__":
    build()
