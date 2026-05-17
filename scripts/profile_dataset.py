"""Full profile of vin_share - ASCII only output, saves to JSON for reuse."""
import json
from collections import Counter
from openpyxl import load_workbook

PATH = r"c:/Users/jotin/Documents/Ford/Ford_oficial/vin_share_Desafio_02.xlsx"
OUT = r"c:/Users/jotin/Documents/Ford/tmp_research/official/data/profile.json"

import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)

wb = load_workbook(PATH, read_only=True, data_only=True)
ws = wb["vin_share"]

counters = {
    "Country": Counter(), "ServiceType": Counter(), "ServiceDeptCode": Counter(),
    "StatusUSA": Counter(), "MainSource": Counter(), "ModelName": Counter(),
    "ModelYear": Counter(), "ServiceCode": Counter(), "ServiceRepairTypeCode": Counter(),
    "IsAgendaSchedule": Counter(), "DealerCode_top": Counter(),
}
vins: set[str] = set()
vin_event_count: Counter = Counter()
vin_dealers: dict[str, set[str]] = {}
vin_models: dict[str, set[str]] = {}
n = 0
empty_vin = 0

rows = ws.iter_rows(values_only=True)
header = next(rows)
idx = {h: i for i, h in enumerate(header)}

for r in rows:
    n += 1
    for col in ("Country", "ServiceType", "ServiceDeptCode", "StatusUSA", "MainSource",
                "ModelName", "ModelYear", "ServiceCode", "ServiceRepairTypeCode", "IsAgendaSchedule"):
        counters[col][r[idx[col]]] += 1
    dc = r[idx["DealerCode"]]
    counters["DealerCode_top"][dc] += 1
    vh = r[idx["VIN_Hash"]]
    if vh:
        vins.add(vh)
        vin_event_count[vh] += 1
        if dc:
            vin_dealers.setdefault(vh, set()).add(str(dc))
        mn = r[idx["ModelName"]]
        if mn:
            vin_models.setdefault(vh, set()).add(str(mn))
    else:
        empty_vin += 1
    if n % 100000 == 0:
        print(f"  ... {n:,} rows")

events_dist = Counter(vin_event_count.values())
multi_dealer = sum(1 for s in vin_dealers.values() if len(s) > 1)
multi_model = sum(1 for s in vin_models.values() if len(s) > 1)

out = {
    "rows": n, "unique_vins": len(vins), "empty_vin": empty_vin,
    "avg_events_per_vin": n / max(len(vins), 1),
    "vins_with_multiple_dealers": multi_dealer,
    "vins_with_multiple_models": multi_model,
    "events_per_vin_distribution": dict(events_dist.most_common(20)),
    "by_column": {k: dict(Counter(v).most_common(30)) for k, v in counters.items()},
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, default=str, ensure_ascii=False)
print(f"\nROWS: {n:,}  VINs: {len(vins):,}  empty: {empty_vin}")
print(f"Models: {list(counters['ModelName'].keys())[:30]}")
print(f"Years: {sorted([y for y in counters['ModelYear'].keys() if y], reverse=True)[:15]}")
print(f"Countries: {dict(counters['Country'])}")
print(f"Dealers count: {len(counters['DealerCode_top'])}")
print(f"VINs with multi-dealer: {multi_dealer:,}")
print(f"Saved profile to {OUT}")
wb.close()
