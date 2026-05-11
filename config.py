"""
M4 — Shared Configuration
DATA3888 Group 8

Import this in any script to get consistent stock selection,
paths, and model parameters. This is the single source of truth —
change it here and every script picks it up automatically.

Usage:
    from config import get_selected_stocks, OUTPUT_DIR, JAMIE_LIQUIDITY_CSV, N_TRAIN, N_VAL
"""

import os
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_HERE, "m4_outputs")

JAMIE_LIQUIDITY_CSV = os.path.join(OUTPUT_DIR, "jamie_liquidity.csv")
SELECTED_STOCKS_CSV = os.path.join(OUTPUT_DIR, "selected_stocks.csv")  # written once, read by all

# ── Model parameters (shared across all scripts) ──────────────────────────────
N_TRAIN             = 16      # training buckets per time_id
N_VAL               = 4       # validation buckets per time_id
N_STOCKS_PER_REGIME = 56      # 56 liquid + 56 illiquid = 112 total (all stocks)
TIME_ID_KEEP_PCT    = 0.5     # keep the most regime-extreme 50% of time_ids per stock

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Core function: get_selected_stocks
#
#  Returns the same 40 stocks every time, for every script.
#  On first call it computes and saves selected_stocks.csv.
#  On subsequent calls it loads from that file — no recomputation.
#
#  Override: if selected_stocks.csv already exists, all scripts use it as-is.
#  To rerun selection (e.g. after Jamie updates her CSV), delete it.
# ══════════════════════════════════════════════════════════════════════════════

def get_selected_stocks(df=None):
    """
    Returns a dict:
      {
        "liquid":   [stock_id, ...],   # 20 most liquid
        "illiquid": [stock_id, ...],   # 20 most illiquid
        "all":      [stock_id, ...],   # combined, sorted
      }

    If selected_stocks.csv exists, loads from it (ignores df argument).
    Otherwise requires df (the raw optiver DataFrame) to compute selection.
    """
    if os.path.exists(SELECTED_STOCKS_CSV):
        sel = pd.read_csv(SELECTED_STOCKS_CSV)
        liquid   = sel[sel["regime"] == "liquid"]["stock_id"].tolist()
        illiquid = sel[sel["regime"] == "illiquid"]["stock_id"].tolist()
        return {
            "liquid":   liquid,
            "illiquid": illiquid,
            "all":      sorted(liquid + illiquid),
        }

    if df is None:
        raise ValueError(
            "selected_stocks.csv not found and no DataFrame provided.\n"
            "Pass the raw optiver DataFrame: get_selected_stocks(df=df)"
        )

    return _compute_and_save(df)


def _compute_and_save(df):
    """Compute stock selection from raw data and save to CSV."""

    # Load liquidity labels
    if not os.path.exists(JAMIE_LIQUIDITY_CSV):
        raise FileNotFoundError(
            f"jamie_liquidity.csv not found at {JAMIE_LIQUIDITY_CSV}.\n"
            "Run jamie_eda.py first."
        )
    liq_df        = pd.read_csv(JAMIE_LIQUIDITY_CSV)
    liquidity_map = dict(zip(liq_df["stock_id"], liq_df["liquidity_regime"]))

    # Rank stocks within each regime by median BAS
    stock_bas       = df.groupby("stock_id")["BidAskSpread_mean"].median()
    liquid_stocks   = [sid for sid, reg in liquidity_map.items() if reg == "liquid"]
    illiquid_stocks = [sid for sid, reg in liquidity_map.items() if reg == "illiquid"]

    # Liquid: tightest spread (ascending) — most stable for EGARCH-X
    liquid_selected = (
        stock_bas[stock_bas.index.isin(liquid_stocks)]
        .sort_values(ascending=True)
        .head(N_STOCKS_PER_REGIME)
        .index.tolist()
    )

    # Illiquid: widest spread (descending) — clearest case for HAR-RV
    illiquid_selected = (
        stock_bas[stock_bas.index.isin(illiquid_stocks)]
        .sort_values(ascending=False)
        .head(N_STOCKS_PER_REGIME)
        .index.tolist()
    )

    # Save so every subsequent script loads the same list
    rows = (
        [{"stock_id": sid, "regime": "liquid",   "median_BAS": stock_bas[sid]} for sid in liquid_selected] +
        [{"stock_id": sid, "regime": "illiquid", "median_BAS": stock_bas[sid]} for sid in illiquid_selected]
    )
    pd.DataFrame(rows).sort_values("stock_id").to_csv(SELECTED_STOCKS_CSV, index=False)
    print(f"  Stock selection saved to: {SELECTED_STOCKS_CSV}")
    print(f"  Liquid:   {len(liquid_selected)} stocks — {liquid_selected}")
    print(f"  Illiquid: {len(illiquid_selected)} stocks — {illiquid_selected}")

    return {
        "liquid":   liquid_selected,
        "illiquid": illiquid_selected,
        "all":      sorted(liquid_selected + illiquid_selected),
    }


def get_liquidity_map():
    """Returns {stock_id: regime} from jamie_liquidity.csv."""
    if not os.path.exists(JAMIE_LIQUIDITY_CSV):
        raise FileNotFoundError(
            f"jamie_liquidity.csv not found. Run jamie_eda.py first."
        )
    liq_df = pd.read_csv(JAMIE_LIQUIDITY_CSV)
    return dict(zip(liq_df["stock_id"], liq_df["liquidity_regime"]))


def filter_time_ids(vol_train, time_ids, regime):
    """
    Given a vol_train dict and list of time_ids, returns the subset that
    are most representative of the given regime:
      liquid   → tightest BAS time_ids (bottom TIME_ID_KEEP_PCT)
      illiquid → widest BAS time_ids   (top TIME_ID_KEEP_PCT)
    """
    tid_bas   = {tid: vol_train[tid]["BidAskSpread_mean"].median() for tid in time_ids}
    tid_bas_s = pd.Series(tid_bas).sort_values(ascending=(regime == "liquid"))
    n_keep    = max(1, int(len(tid_bas_s) * TIME_ID_KEEP_PCT))
    return tid_bas_s.head(n_keep).index.tolist()