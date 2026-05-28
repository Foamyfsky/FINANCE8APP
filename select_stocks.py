"""
select_stocks.py 

Quickly generates the 60-stock selection (20 liquid + 20 illiquid + 20 mixed)
WITHOUT running any models.

Run once, share config.py with groupmates after copying the printed ids.

Usage:
    py select_stocks.py
"""

import os
import sys
import numpy as np
import pandas as pd

INPUT_CSV = r"C:\Users\songj\OneDrive\UNI\Y3SEM1\DATA3888\Project Folder\DATA3888G08\optiver_aggregated.csv"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, ROSA_LIQUIDITY_CSV, RANDOM_SEED, N_STOCKS_PER_REGIME

os.makedirs(OUTPUT_DIR, exist_ok=True)
LIQUIDITY_PROFILE_PATH = os.path.join(OUTPUT_DIR, "har_rv_stock_liquidity_profile.csv")


def build_liquidity_profile(df):
    liq = df[["stock_id", "time_id", "time_bucket", "BidAskSpread_mean", "RV"]].copy()
    liq = liq.rename(columns={"BidAskSpread_mean": "bas", "RV": "rv"})
    liq["time_bucket"] = liq["time_bucket"].astype(int)
    liq = liq[liq["time_bucket"].between(1, 16)].copy()

    liq["inv_spread"] = 1.0 / (liq["bas"] + 1e-6)
    liq["log_activity"] = np.log1p(liq["inv_spread"])

    bas_q33 = liq["bas"].quantile(0.33)
    bas_q66 = liq["bas"].quantile(0.66)
    act_q33 = liq["log_activity"].quantile(0.33)
    act_q66 = liq["log_activity"].quantile(0.66)

    conditions = [
        (liq["bas"] <= bas_q33) & (liq["log_activity"] >= act_q66),
        (liq["bas"] >= bas_q66) & (liq["log_activity"] <= act_q33),
    ]
    liq["bucket_liquidity"] = np.select(conditions, ["liquid", "illiquid"], default="mixed")

    profile = (
        liq.groupby("stock_id")
        .agg(
            median_bas = ("bas", "median"),
            median_log_activity = ("log_activity", "median"),
            liquid_pct = ("bucket_liquidity", lambda x: (x == "liquid").mean()),
            illiquid_pct = ("bucket_liquidity", lambda x: (x == "illiquid").mean()),
            mixed_pct = ("bucket_liquidity", lambda x: (x == "mixed").mean()),
        )
        .reset_index()
    )

    def _regime(row):
        if row["liquid_pct"] >= 0.40: return "liquid"
        if row["illiquid_pct"] >= 0.40: return "illiquid"
        return "mixed"

    profile["stock_regime"] = profile.apply(_regime, axis=1)
    profile.to_csv(LIQUIDITY_PROFILE_PATH, index=False)
    print(f"Liquidity profile saved -> {LIQUIDITY_PROFILE_PATH}")
    print(f"Regime counts: {profile['stock_regime'].value_counts().to_dict()}")
    return profile


def select_stocks(df, profile):
    stock_bas = df.groupby("stock_id")["BidAskSpread_mean"].median()

    liquid_pool = profile[profile["stock_regime"] == "liquid"]["stock_id"].tolist()
    illiquid_pool = profile[profile["stock_regime"] == "illiquid"]["stock_id"].tolist()
    mixed_pool = profile[profile["stock_regime"] == "mixed"]["stock_id"].tolist()

    # Liquid: tightest BAS (most liquid)
    liq_bas = stock_bas[stock_bas.index.isin(liquid_pool)].sort_values(ascending=True)
    liquid_sel = liq_bas.head(N_STOCKS_PER_REGIME).index.tolist()

    # Illiquid: widest BAS (most illiquid)
    ilq_bas = stock_bas[stock_bas.index.isin(illiquid_pool)].sort_values(ascending=False)
    illiquid_sel = ilq_bas.head(N_STOCKS_PER_REGIME).index.tolist()

    # Mixed: random sample from remaining pool
    already = set(liquid_sel + illiquid_sel)
    mix_cands = [s for s in mixed_pool if s not in already]
    rng = np.random.default_rng(RANDOM_SEED)
    n_mix = min(N_STOCKS_PER_REGIME, len(mix_cands))
    mixed_sel = sorted(rng.choice(mix_cands, size=n_mix, replace=False).tolist())

    return liquid_sel, illiquid_sel, mixed_sel


print("Loading data ...")
df = pd.read_csv(INPUT_CSV)
print(f"  {df['stock_id'].nunique()} stocks, {df['time_id'].nunique()} time_ids\n")

if os.path.exists(LIQUIDITY_PROFILE_PATH):
    print(f"Reusing existing profile: {LIQUIDITY_PROFILE_PATH}")
    profile = pd.read_csv(LIQUIDITY_PROFILE_PATH)
else:
    profile = build_liquidity_profile(df)

liquid_sel, illiquid_sel, mixed_sel = select_stocks(df, profile)

print("\n" + "="*60)
print(f"  STOCK SELECTION  (seed={RANDOM_SEED}, {N_STOCKS_PER_REGIME} per regime)")
print("="*60)
print(f"\n  LIQUID ({len(liquid_sel)} stocks):")
print(f"    {sorted(liquid_sel)}")
print(f"\n  ILLIQUID ({len(illiquid_sel)} stocks):")
print(f"    {sorted(illiquid_sel)}")
print(f"\n  MIXED ({len(mixed_sel)} stocks):")
print(f"    {sorted(mixed_sel)}")
print(f"\n  ALL 60:  {sorted(liquid_sel + illiquid_sel + mixed_sel)}")
print("="*60)

stock_bas = df.groupby("stock_id")["BidAskSpread_mean"].median()
rows = (
    [{"stock_id": s, "regime": "liquid",   "median_BAS": stock_bas.get(s)} for s in liquid_sel] +
    [{"stock_id": s, "regime": "illiquid", "median_BAS": stock_bas.get(s)} for s in illiquid_sel] +
    [{"stock_id": s, "regime": "mixed",    "median_BAS": stock_bas.get(s)} for s in mixed_sel]
)
out_csv = os.path.join(OUTPUT_DIR, "selected_stocks.csv")
pd.DataFrame(rows).sort_values("stock_id").to_csv(out_csv, index=False)
print(f"\n  Saved -> {out_csv}")

print("""
Next steps:
  1. Copy the ids above into config.py:
       LIQUID_STOCKS   = [...]
       ILLIQUID_STOCKS = [...]
       MIXED_STOCKS    = [...]
  2. Share config.py with your groupmates.
  3. Run:  py rosa.py  then  py arma_jisu.py
""")
