"""
Enriches existing stock_data_json/stock_N.json files with a 'forecasts' key
containing per-time_id pred/actual arrays for GARCH, HAR-RV, and LightGBM.

Run from the FINANCE8APP directory:
    python generate_forecast_json.py
"""
import json, csv, os, collections

BASE = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(BASE, "m4_outputs", "stock_data_json")
M4_DIR = os.path.join(BASE, "m4_outputs")
LGBM_CSV = os.path.join(M4_DIR, "lgbm_outputs", "lgbm_eval_results.csv")

# Load LGBM results (all stocks in one file)
lgbm_by_stock = collections.defaultdict(list)
with open(LGBM_CSV, newline="") as f:
    for row in csv.DictReader(f):
        sid = int(row["stock_id"])
        lgbm_by_stock[sid].append({
            "t":      int(row["time_id"]),
            "pred":   float(row["pred_vol"]),
            "actual": float(row["target_rv"]),
        })

for sid in lgbm_by_stock:
    lgbm_by_stock[sid].sort(key=lambda x: x["t"])

json_files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]
print(f"Found {len(json_files)} stock JSON files to enrich.")

for fname in sorted(json_files):
    json_path = os.path.join(JSON_DIR, fname)
    stock_id = int(fname.replace("stock_", "").replace(".json", ""))
    stock_dir = os.path.join(M4_DIR, f"stock_{stock_id}")

    with open(json_path) as f:
        data = json.load(f)

    forecasts = {}

    # GARCH (4 buckets per time_id; average actual_RV across buckets, pred_RV is constant)
    garch_csv = os.path.join(stock_dir, "garch_eval_results.csv")
    if os.path.exists(garch_csv):
        buckets = collections.defaultdict(list)
        pred_per_tid = {}
        with open(garch_csv, newline="") as f:
            for row in csv.DictReader(f):
                tid = int(row["time_id"])
                buckets[tid].append(float(row["actual_RV"]))
                pred_per_tid[tid] = float(row["pred_RV"])
        garch_rows = sorted([
            {"t": tid, "pred": pred_per_tid[tid],
             "actual": sum(v) / len(v)}
            for tid, v in buckets.items()
        ], key=lambda x: x["t"])
        forecasts["garch"] = garch_rows

    # HAR-RV
    har_csv = os.path.join(stock_dir, "rosa_har_rv_forecasts.csv")
    if os.path.exists(har_csv):
        har_rows = []
        with open(har_csv, newline="") as f:
            for row in csv.DictReader(f):
                har_rows.append({
                    "t":      int(row["time_id"]),
                    "pred":   float(row["pred_RV"]),
                    "actual": float(row["actual_RV"]),
                })
        har_rows.sort(key=lambda x: x["t"])
        forecasts["har"] = har_rows

    # LightGBM
    if stock_id in lgbm_by_stock:
        forecasts["lgbm"] = lgbm_by_stock[stock_id]

    if forecasts:
        data["forecasts"] = forecasts
        with open(json_path, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"  stock_{stock_id}: GARCH={len(forecasts.get('garch',[]))}, "
              f"HAR={len(forecasts.get('har',[]))}, "
              f"LGBM={len(forecasts.get('lgbm',[]))}")
    else:
        print(f"  stock_{stock_id}: no forecast CSVs found, skipped.")

print("Done.")
