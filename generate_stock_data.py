"""
generate_stock_data.py
Generates compact histogram JSON files for the HTML dashboard.

Run once from the FINANCE8APP directory:
    python generate_stock_data.py

Output: m4_outputs/stock_data_json/stock_<ID>.json  (~4 KB each, 60 files total)
Each JSON contains BAS and WAP histogram distributions across all time_ids for that stock.
"""
import os, json
import numpy as np
import pandas as pd

# 60 stocks from config.py
SELECTED_STOCKS = [
    2, 3, 5, 6, 7, 9, 14, 15, 17, 18, 19, 26, 27, 29, 31, 32, 33, 37,
    39, 40, 41, 43, 44, 46, 47, 48, 50, 56, 59, 62, 64, 69, 70, 72, 75,
    76, 82, 88, 89, 90, 93, 96, 97, 98, 99, 101, 103, 108, 109, 111,
    112, 113, 116, 119, 120, 122, 123, 124, 125, 126
]

HERE       = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(HERE, "m4_outputs", "stock_data_json")
CSV_PATH   = os.path.join(HERE, "..", "optiver_aggregated.csv")

os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading {CSV_PATH}  (this may take a minute)...")
df = pd.read_csv(
    CSV_PATH,
    usecols=["stock_id", "time_id", "WAP_mean", "BidAskSpread_mean"]
)
df["BAS_bps"] = df["BidAskSpread_mean"] * 10000
print(f"  Loaded {len(df):,} rows. Processing {len(SELECTED_STOCKS)} stocks...\n")


def make_hist(vals, n_bins=60):
    counts, edges = np.histogram(vals, bins=n_bins)
    centers = ((edges[:-1] + edges[1:]) / 2).round(6)
    return {
        "bins":   [round(float(c), 5) for c in centers],
        "counts": counts.tolist(),
        "mean":   round(float(vals.mean()), 5),
        "median": round(float(np.median(vals)), 5),
        "p10":    round(float(np.percentile(vals, 10)), 5),
        "p25":    round(float(np.percentile(vals, 25)), 5),
        "p75":    round(float(np.percentile(vals, 75)), 5),
        "p90":    round(float(np.percentile(vals, 90)), 5),
        "min":    round(float(vals.min()), 5),
        "max":    round(float(vals.max()), 5),
    }


for sid in SELECTED_STOCKS:
    sdf = df[df["stock_id"] == sid]
    if sdf.empty:
        print(f"  [SKIP] stock {sid}: no data")
        continue

    # Aggregate across the 20 buckets → one value per time_id
    tid = sdf.groupby("time_id").agg(
        bas=("BAS_bps",  "mean"),
        wap=("WAP_mean", "mean"),
    )
    # WAP deviation from 1.0 scaled ×10000 (Optiver normalises WAP ~1.0)
    wap_dev = (tid["wap"].values - 1.0) * 10000

    out = {
        "stock_id":   int(sid),
        "n_time_ids": len(tid),
        "bas":        make_hist(tid["bas"].values),
        "wap":        make_hist(wap_dev),
    }

    fpath = os.path.join(OUT_DIR, f"stock_{sid}.json")
    with open(fpath, "w") as f:
        json.dump(out, f, separators=(",", ":"))

    print(f"  stock {sid:3d}: {len(tid):,} sessions  "
          f"BAS mean={out['bas']['mean']:.2f} bps  "
          f"WAP mean={out['wap']['mean']:.4f}  -> {os.path.basename(fpath)}")

print(f"\nDone! Files in: {OUT_DIR}")
