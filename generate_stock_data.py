"""
generate_stock_data.py
Patches existing stock JSON files to add/refresh all forecast models:
  garch (GJR-GARCH), egarchx, har, lgbm

The BAS/WAP histograms are preserved from the existing JSONs.
Run from the newproj directory: python generate_stock_data.py
"""
import os, json
import pandas as pd

SELECTED_STOCKS = [
    2, 3, 5, 6, 7, 9, 14, 15, 17, 18, 19, 26, 27, 29, 31, 32, 33, 37,
    39, 40, 41, 43, 44, 46, 47, 48, 50, 56, 59, 62, 64, 69, 70, 72, 75,
    76, 82, 88, 89, 90, 93, 96, 97, 98, 99, 101, 103, 108, 109, 111,
    112, 113, 116, 119, 120, 122, 123, 124, 125, 126
]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "m4_outputs", "stock_data_json")
STOCK_DIR = os.path.join(HERE, "m4_outputs")
LGBM_CSV = os.path.join(HERE, "m4_outputs", "lgbm_outputs", "lgbm_eval_results.csv")


def load_bucketed(csv_path):
    """CSV with (time_id, bucket_idx, pred_RV, actual_RV) -> one row per time_id."""
    if not os.path.exists(csv_path):
        return None
    d = pd.read_csv(csv_path, usecols=["time_id", "pred_RV", "actual_RV"])
    agg = d.groupby("time_id").agg(pred=("pred_RV","mean"), actual=("actual_RV","mean")).reset_index()
    return [{"t": int(r.time_id), "pred": round(float(r.pred), 10), "actual": round(float(r.actual), 10)}
            for r in agg.itertuples(index=False)]


def load_simple(csv_path, pred_col="pred_RV", actual_col="actual_RV"):
    """CSV with one row per time_id."""
    if not os.path.exists(csv_path):
        return None
    d = pd.read_csv(csv_path, usecols=["time_id", pred_col, actual_col])
    return [{"t": int(r[0]), "pred": round(float(r[1]), 10), "actual": round(float(r[2]), 10)}
            for r in zip(d["time_id"], d[pred_col], d[actual_col])]


# Pre-load LGBM
print("Loading LGBM results...")
lgbm_all = pd.read_csv(LGBM_CSV) if os.path.exists(LGBM_CSV) else None
if lgbm_all is None:
    print("  WARNING: lgbm_eval_results.csv not found")

print(f"Patching {len(SELECTED_STOCKS)} stock JSONs...\n")

ok = 0
for sid in SELECTED_STOCKS:
    json_path = os.path.join(OUT_DIR, f"stock_{sid}.json")
    if not os.path.exists(json_path):
        print(f" [SKIP] stock {sid}: JSON not found")
        continue

    with open(json_path) as f:
        existing = json.load(f)

    folder = os.path.join(STOCK_DIR, f"stock_{sid}")

    gjr = load_bucketed(os.path.join(folder, "garch_eval_results.csv"))
    egx = load_bucketed(os.path.join(folder, "egarchx_eval_results.csv"))
    har = load_simple(os.path.join(folder, "rosa_har_rv_eval_results.csv"))

    lgbm = None
    if lgbm_all is not None:
        ldf = lgbm_all[lgbm_all["stock_id"] == sid]
        if not ldf.empty:
            lgbm = [{"t": int(r.time_id), "pred": round(float(r.pred_vol), 10), "actual": round(float(r.target_rv), 10)}
                    for r in ldf.itertuples(index=False)]

    forecasts = {}
    if gjr: forecasts["garch"] = gjr
    if egx: forecasts["egarchx"] = egx
    if har: forecasts["har"] = har
    if lgbm: forecasts["lgbm"] = lgbm

    existing["forecasts"] = forecasts

    with open(json_path, "w") as f:
        json.dump(existing, f, separators=(",",":"))

    summary = " | ".join(f"{k}:{len(v)}" for k, v in forecasts.items())
    print(f" stock {sid:3d}: [{summary}]")
    ok += 1

print(f"\nPatched {ok}/{len(SELECTED_STOCKS)} stocks. Done!")