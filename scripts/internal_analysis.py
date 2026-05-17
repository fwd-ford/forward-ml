"""Internal analysis of the official vin_share dataset.

Five blocks:
  1. Inventory (per-column stats, missing rates, distributions)
  2. Geography (Country breakdown, multi-country check)
  3. Codebook inference (cross-tabs of Service* fields)
  4. Behavioral feature engineering (per-VIN features)
  5. SHA1 reversal test (try a small subset of likely Ford BR VINs)

Outputs:
  - tmp_research/official/01_internal_analysis/inventory.md
  - tmp_research/official/01_internal_analysis/geography.md
  - tmp_research/official/01_internal_analysis/codebook_inferred.md
  - tmp_research/official/01_internal_analysis/behavioral_features.md
  - tmp_research/official/01_internal_analysis/sha1_reversal_test.md
  - tmp_research/official/data/vin_features.parquet (or .csv)
"""
from __future__ import annotations
import json
import os
import re
import hashlib
import csv
from collections import Counter, defaultdict
from datetime import datetime
from openpyxl import load_workbook

ROOT = r"c:/Users/jotin/Documents/Ford"
DATA = rf"{ROOT}/Ford_oficial/vin_share_Desafio_02.xlsx"
OUT = rf"{ROOT}/tmp_research/official/01_internal_analysis"
DATAOUT = rf"{ROOT}/tmp_research/official/data"
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATAOUT, exist_ok=True)


def parse_date(s):
    """Try multiple formats. Returns datetime or None."""
    if s is None or s == "":
        return None
    if isinstance(s, datetime):
        return s
    s = str(s).strip()
    for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def write_md(name: str, content: str) -> None:
    with open(f"{OUT}/{name}", "w", encoding="utf-8") as f:
        f.write(content)


print("Loading workbook (read_only)...")
wb = load_workbook(DATA, read_only=True, data_only=True)
ws = wb["vin_share"]
rows_iter = ws.iter_rows(values_only=True)
header = next(rows_iter)
idx = {h: i for i, h in enumerate(header)}
print(f"Columns: {list(idx.keys())}")

# Counters per column for inventory
col_counters: dict[str, Counter] = {h: Counter() for h in idx}
col_missing: dict[str, int] = {h: 0 for h in idx}
col_examples: dict[str, list] = {h: [] for h in idx}

# Codebook cross-tabs
xt_servicetype_repairtype: dict = defaultdict(Counter)
xt_servicetype_servicecode: dict = defaultdict(Counter)
xt_servicedept_servicetype: dict = defaultdict(Counter)
xt_mainsource_servicetype: dict = defaultdict(Counter)

# Per-VIN aggregations for behavioral features
vin_events: dict[str, list[dict]] = defaultdict(list)
vin_first_seen: dict[str, dict] = {}

# Geography
country_count: Counter = Counter()
country_status: dict = defaultdict(Counter)

# SHA1 sample collection
sample_hashes: list[str] = []

n = 0
print("\nFirst pass: scan rows...")
for r in rows_iter:
    n += 1
    row = {h: r[i] for h, i in idx.items()}
    for h, v in row.items():
        if v is None or (isinstance(v, str) and v.strip() == ""):
            col_missing[h] += 1
        else:
            col_counters[h][v] += 1
            if len(col_examples[h]) < 5 and v not in col_examples[h]:
                col_examples[h].append(v)

    st = row.get("ServiceType")
    rt = row.get("ServiceRepairTypeCode")
    sc = row.get("ServiceCode")
    sd = row.get("ServiceDeptCode")
    ms = row.get("MainSource")
    xt_servicetype_repairtype[st][rt] += 1
    xt_servicetype_servicecode[st][sc] += 1
    xt_servicedept_servicetype[sd][st] += 1
    xt_mainsource_servicetype[ms][st] += 1

    country = row.get("Country")
    status = row.get("StatusUSA")
    country_count[country] += 1
    country_status[country][status] += 1

    vh = row.get("VIN_Hash")
    if vh:
        sd_parsed = parse_date(row.get("ServiceDate"))
        invoice = parse_date(row.get("InvoiceDate"))
        sales = parse_date(row.get("SalesDate"))
        delivery = parse_date(row.get("DeliveryDate"))
        warranty = parse_date(row.get("WarrantyStartDate"))
        try:
            km = float(row.get("KM")) if row.get("KM") not in (None, "") else None
        except (ValueError, TypeError):
            km = None
        vin_events[vh].append({
            "service_date": sd_parsed,
            "dealer": row.get("DealerCode"),
            "service_type": st,
            "service_code": sc,
            "repair_type": rt,
            "main_source": ms,
            "km": km,
            "model_name": row.get("ModelName"),
            "model_year": row.get("ModelYear"),
        })
        if vh not in vin_first_seen:
            vin_first_seen[vh] = {
                "invoice_date": invoice,
                "sales_date": sales,
                "delivery_date": delivery,
                "warranty_start": warranty,
                "model_name": row.get("ModelName"),
                "model_year": row.get("ModelYear"),
            }
        if len(sample_hashes) < 50:
            sample_hashes.append(vh)
    if n % 100000 == 0:
        print(f"  ... {n:,} rows  /  vins so far: {len(vin_events):,}")

wb.close()
print(f"\nTotal rows: {n:,}")
print(f"Unique VINs: {len(vin_events):,}")

# --- BLOCK 1: INVENTORY ---
print("\nBlock 1: inventory.md")
lines = ["# Inventario do dataset oficial (vin_share)", ""]
lines.append(f"**Total de linhas:** {n:,}")
lines.append(f"**VIN_Hash unicos:** {len(vin_events):,}")
lines.append(f"**Eventos medios por VIN:** {n / max(len(vin_events),1):.2f}")
lines.append("")
lines.append("## Resumo por coluna")
lines.append("")
lines.append("| Coluna | Valores unicos | % missing | Top 5 valores (com %) | Exemplo |")
lines.append("|---|---:|---:|---|---|")
for h in idx:
    uniq = len(col_counters[h])
    miss_pct = col_missing[h] / n * 100
    top5 = col_counters[h].most_common(5)
    top5_str = "; ".join(f"`{str(v)[:30]}` ({c/n*100:.1f}%)" for v, c in top5)
    examples = "; ".join(f"`{str(e)[:30]}`" for e in col_examples[h][:2])
    lines.append(f"| {h} | {uniq:,} | {miss_pct:.2f}% | {top5_str} | {examples} |")
write_md("inventory.md", "\n".join(lines))

# --- BLOCK 2: GEOGRAPHY ---
print("Block 2: geography.md")
lines = ["# Cobertura geografica", ""]
lines.append("## Distribuicao por Country")
lines.append("")
lines.append("| Country | Linhas | % |")
lines.append("|---|---:|---:|")
for c, cnt in country_count.most_common():
    lines.append(f"| {c} | {cnt:,} | {cnt/n*100:.2f}% |")
lines.append("")
lines.append("## StatusUSA por Country (top 10 por pais)")
lines.append("")
for c in country_count:
    lines.append(f"### Country = `{c}`")
    lines.append("| StatusUSA | Linhas | % |")
    lines.append("|---|---:|---:|")
    total_c = country_count[c]
    for st, cnt in country_status[c].most_common(10):
        lines.append(f"| {st} | {cnt:,} | {cnt/total_c*100:.2f}% |")
    lines.append("")
write_md("geography.md", "\n".join(lines))

# --- BLOCK 3: CODEBOOK INFERRED ---
print("Block 3: codebook_inferred.md")
lines = ["# Codebook inferido (Service* fields)", ""]
lines.append("> Inferencia por co-ocorrencia. Nao substitui o codebook oficial da Ford, mas da pistas fortes.\n")

def render_xt(title: str, xt: dict, top_outer: int = 10, top_inner: int = 6) -> list[str]:
    out = [f"## {title}", ""]
    total = sum(sum(inner.values()) for inner in xt.values())
    outer_counts = Counter({k: sum(v.values()) for k, v in xt.items()})
    out.append("| Outer | Top Inner (com %) | Total |")
    out.append("|---|---|---:|")
    for outer, _ in outer_counts.most_common(top_outer):
        inners = xt[outer].most_common(top_inner)
        outer_total = sum(xt[outer].values())
        inners_str = "; ".join(f"`{str(k)[:30]}` ({c/outer_total*100:.1f}%)" for k, c in inners)
        out.append(f"| `{outer}` | {inners_str} | {outer_total:,} |")
    out.append("")
    return out

lines += render_xt("ServiceType x ServiceRepairTypeCode", xt_servicetype_repairtype)
lines += render_xt("ServiceType x ServiceCode", xt_servicetype_servicecode)
lines += render_xt("ServiceDeptCode x ServiceType", xt_servicedept_servicetype)
lines += render_xt("MainSource x ServiceType", xt_mainsource_servicetype)
write_md("codebook_inferred.md", "\n".join(lines))

# --- BLOCK 4: BEHAVIORAL FEATURES ---
print("Block 4: behavioral_features.md  (this may take a minute...)")

# Derive features per VIN
features: list[dict] = []
for vh, events in vin_events.items():
    events_sorted = [e for e in events if e["service_date"] is not None]
    events_sorted.sort(key=lambda e: e["service_date"])
    if not events_sorted:
        events_sorted = events  # keep at least something
    first = vin_first_seen.get(vh, {})

    dates = [e["service_date"] for e in events_sorted if e["service_date"]]
    gaps_days = []
    for a, b in zip(dates, dates[1:]):
        gaps_days.append((b - a).days)

    kms = [e["km"] for e in events_sorted if e["km"] is not None and e["km"] > 0]

    dealers_set = set(e["dealer"] for e in events_sorted if e["dealer"])
    dealer_counter = Counter(e["dealer"] for e in events_sorted if e["dealer"])
    primary_dealer, primary_dealer_count = dealer_counter.most_common(1)[0] if dealer_counter else (None, 0)
    primary_dealer_share = primary_dealer_count / len(events_sorted) if events_sorted else 0

    last_event_date = dates[-1] if dates else None
    first_event_date = dates[0] if dates else None
    invoice = first.get("invoice_date")
    sales = first.get("sales_date")
    warranty = first.get("warranty_start")

    days_since_last_service = None
    if last_event_date:
        # Reference: max ServiceDate in dataset (computed later) - placeholder uses today
        days_since_last_service = (datetime.now() - last_event_date).days

    feature = {
        "vin_hash": vh,
        "events_count": len(events_sorted),
        "first_service_date": first_event_date.strftime("%Y-%m-%d") if first_event_date else None,
        "last_service_date": last_event_date.strftime("%Y-%m-%d") if last_event_date else None,
        "days_since_last_service": days_since_last_service,
        "tenure_days": (last_event_date - (invoice or sales or first_event_date)).days if last_event_date and (invoice or sales or first_event_date) else None,
        "gap_avg_days": (sum(gaps_days) / len(gaps_days)) if gaps_days else None,
        "gap_min_days": min(gaps_days) if gaps_days else None,
        "gap_max_days": max(gaps_days) if gaps_days else None,
        "gap_last_days": gaps_days[-1] if gaps_days else None,
        "km_max": max(kms) if kms else None,
        "km_min": min(kms) if kms else None,
        "dealers_distinct": len(dealers_set),
        "primary_dealer": primary_dealer,
        "primary_dealer_share": primary_dealer_share,
        "model_name": first.get("model_name"),
        "model_year": first.get("model_year"),
        "invoice_date": invoice.strftime("%Y-%m-%d") if invoice else None,
        "sales_date": sales.strftime("%Y-%m-%d") if sales else None,
        "warranty_start": warranty.strftime("%Y-%m-%d") if warranty else None,
        "service_types_distinct": len(set(e["service_type"] for e in events_sorted if e["service_type"])),
        "service_codes_distinct": len(set(e["service_code"] for e in events_sorted if e["service_code"])),
    }
    features.append(feature)

# Save features as CSV (parquet would be ideal but avoids pandas dependency)
features_csv = f"{DATAOUT}/vin_features.csv"
if features:
    fieldnames = list(features[0].keys())
    with open(features_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(features)
print(f"  Saved {len(features):,} rows of features to {features_csv}")

# Distributions
events_dist = Counter(f["events_count"] for f in features)
tenure_values = [f["tenure_days"] for f in features if f["tenure_days"] is not None]
gap_avg_values = [f["gap_avg_days"] for f in features if f["gap_avg_days"] is not None]
dealers_dist = Counter(f["dealers_distinct"] for f in features)
km_max_values = [f["km_max"] for f in features if f["km_max"] is not None and f["km_max"] > 0]

def stats(arr: list[float]) -> str:
    if not arr:
        return "n/a"
    arr = sorted(arr)
    n = len(arr)
    mean = sum(arr) / n
    p50 = arr[n // 2]
    p25 = arr[n // 4]
    p75 = arr[3 * n // 4]
    return f"mean={mean:.1f} | p25={p25:.0f} | median={p50:.0f} | p75={p75:.0f} | min={arr[0]:.0f} | max={arr[-1]:.0f}"

lines = ["# Features comportamentais por VIN", ""]
lines.append(f"**VINs com features derivadas:** {len(features):,}")
lines.append(f"**Total de eventos:** {n:,}")
lines.append(f"**CSV completo de features:** `tmp_research/official/data/vin_features.csv`")
lines.append("")
lines.append("## Distribuicoes-chave")
lines.append("")
lines.append("### Eventos por VIN")
lines.append("| Eventos | VINs | % |")
lines.append("|---:|---:|---:|")
for ev, cnt in sorted(events_dist.items())[:15]:
    lines.append(f"| {ev} | {cnt:,} | {cnt/len(features)*100:.2f}% |")
lines.append("")
lines.append("### Tenure (dias entre InvoiceDate/SalesDate e ultimo servico)")
lines.append(f"  {stats(tenure_values)}")
lines.append("")
lines.append("### Gap medio entre servicos (dias)")
lines.append(f"  {stats(gap_avg_values)}")
lines.append("")
lines.append("### KM maximo registrado")
lines.append(f"  {stats(km_max_values)}")
lines.append("")
lines.append("### Diversidade de dealers visitados")
lines.append("| Dealers distintos | VINs | % |")
lines.append("|---:|---:|---:|")
for d, cnt in sorted(dealers_dist.items())[:10]:
    lines.append(f"| {d} | {cnt:,} | {cnt/len(features)*100:.2f}% |")
lines.append("")
lines.append("## Implicacoes para ML")
lines.append("")
lines.append("Cada VIN tem ate **20+ features derivadas** sem precisar de fonte externa.")
lines.append("As mais importantes para deteccao de churn:")
lines.append("- `days_since_last_service` (recencia)")
lines.append("- `gap_avg_days` x `gap_last_days` (acelerando ou retardando?)")
lines.append("- `dealers_distinct` (lealdade ao dealer)")
lines.append("- `service_types_distinct` (rico em comportamento ou unidimensional)")
lines.append("")
write_md("behavioral_features.md", "\n".join(lines))

# --- BLOCK 5: SHA1 REVERSAL TEST ---
print("Block 5: sha1_reversal_test.md")
# Generate a small subset of likely Ford BR VIN prefixes to test
# Ford BR WMI: starts with 9BF (third gen) or 8AF (some imports)
# Format: WMI(3) + VDS(6) + VIS(8) = 17 chars
# We'll test a small brute-force on a subset

# Strategy: take 100 sample hashes from dataset, try to reverse with synthetic candidates
import string
sample_subset = sample_hashes[:100]
matched: list[tuple[str, str]] = []
attempts = 0
max_attempts = 5_000_000  # cap

# Build candidate VINs: WMI=9BF, then random VDS+VIS from common chars
# This is a sanity test, not a real cracking attempt
prefixes = ["9BF", "8AF", "9BB"]
chars = string.ascii_uppercase + string.digits
chars = chars.replace("I", "").replace("O", "").replace("Q", "")  # VIN excludes these

# Just try first few thousand combinations to see if salt is obvious
import itertools
for prefix in prefixes:
    for suffix in itertools.product(chars, repeat=6):  # 32^6 ~ 1B - too much. cap below.
        candidate = prefix + "".join(suffix) + ("A" * 8)  # placeholder VIS
        h = hashlib.sha1(candidate.encode()).hexdigest()
        attempts += 1
        if h in sample_subset:
            matched.append((candidate, h))
        if attempts >= max_attempts:
            break
    if attempts >= max_attempts:
        break

lines = ["# Teste de reversao SHA1 do VIN_Hash", ""]
lines.append("## Objetivo")
lines.append("Confirmar se o `VIN_Hash` (SHA1) usa salt ou e SHA1 puro do VIN.")
lines.append("Se for puro, e teoricamente reversivel por rainbow table (VIN tem estrutura fixa de 17 chars).")
lines.append("")
lines.append("## Metodo")
lines.append(f"- Amostra: {len(sample_subset)} hashes do dataset")
lines.append(f"- Espaco testado: prefixes Ford BR (9BF, 8AF, 9BB) + 6 chars de VDS aleatorio + 8 chars placeholder")
lines.append(f"- Limite: {max_attempts:,} tentativas")
lines.append("")
lines.append("## Resultado")
lines.append(f"- Tentativas executadas: {attempts:,}")
lines.append(f"- Hashes encontrados (SHA1 puro): **{len(matched)}**")
if matched:
    lines.append("")
    lines.append("**MATCH ENCONTRADO** -> SHA1 sem salt.")
    for candidate, h in matched[:5]:
        lines.append(f"- candidato: `{candidate}` -> hash: `{h}`")
else:
    lines.append("")
    lines.append("**Nenhum match** com ate {:,} tentativas.".format(attempts))
    lines.append("")
    lines.append("Interpretacao:")
    lines.append("- E provavel que tenha salt (ou seja outro algoritmo).")
    lines.append("- MAS espaco real do VIN e ~17^32 (excluindo IOQ), inviavel exaustivamente.")
    lines.append("- Recomendacao: se assumir 'sem salt' como pior caso, e tratar como pseudonimizado (LGPD).")
write_md("sha1_reversal_test.md", "\n".join(lines))

print("\nDone. Files in:", OUT)
