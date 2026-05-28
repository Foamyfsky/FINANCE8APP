"""
M4 - Shared Configuration
DATA3888 Group 8

Single source of truth for paths, model parameters, and stock selection.
Import this in arma_jisu.py, rosa.py, and any other script.

Liquidity source priority:
  1. Rosa's har_rv_stock_liquidity_profile.csv  - richer, 3-way regime
     (bucket-level BAS + inverse-spread activity, 40% threshold)
  2. Jamie's jamie_liquidity.csv                - fallback binary classification

Usage:
    from config import get_selected_stocks, get_liquidity_map, filter_time_ids,
                       OUTPUT_DIR, N_TRAIN, N_VAL
"""

import os
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_HERE, "m4_outputs")

JAMIE_LIQUIDITY_CSV = os.path.join(OUTPUT_DIR, "jamie_liquidity.csv")
SELECTED_STOCKS_CSV = os.path.join(OUTPUT_DIR, "selected_stocks.csv")

# Rosa's liquidity profile - searched in order; first found wins.
_ROSA_SEARCH_PATHS = [
    os.path.join(OUTPUT_DIR, "har_rv_stock_liquidity_profile.csv"),
    r"C:\Users\songj\OneDrive\UNI\Y3SEM1\DATA3888\Project Folder\DATA3888G08\har_rv_stock_liquidity_profile.csv",
    r"/Users/rosakwak/Desktop/DATA3888/DATA3888G08/har_rv_stock_liquidity_profile.csv",
]
ROSA_LIQUIDITY_CSV = next((p for p in _ROSA_SEARCH_PATHS if os.path.exists(p)), None)

N_TRAIN = 16     # training buckets per time_id
N_VAL = 4        # validation buckets per time_id
N_STOCKS_PER_REGIME = 20     # 20 liquid + 20 illiquid + 20 mixed = 60 total
TIME_ID_KEEP_PCT = 0.5       # keep the most regime-extreme 50% of time_ids

# Mixed regime: Jamie liquidity scores between these bounds -> mixed candidate
MIXED_SCORE_LO = 0.3
MIXED_SCORE_HI = 0.7
RANDOM_SEED = 42   # reproducible mixed-stock selection

# Manual stock overrides.
# If any list is non-empty, all three are used as-is and auto-selection is skipped.
# To use auto-selection, leave all three as empty lists ([]).
#
# How to use:
#   1. Run rosa.py once -> prints "Stock selection from Rosa's profile"
#                          with the exact stock_ids chosen.
#   2. Copy those ids into the lists below and share config.py with groupmates.
#
LIQUID_STOCKS   = [2, 14, 29, 39, 41, 43, 44, 46, 47, 50, 64, 69, 93, 99, 111, 119, 120, 123, 124, 125]
ILLIQUID_STOCKS = [3, 5, 6, 9, 18, 27, 31, 33, 37, 40, 62, 75, 88, 90, 97, 98, 103, 112, 116, 126]
MIXED_STOCKS    = [7, 15, 17, 19, 26, 32, 48, 56, 59, 70, 72, 76, 82, 89, 96, 101, 108, 109, 113, 122]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load_rosa_profile():
    """Load Rosa's liquidity profile CSV. Returns DataFrame or None."""
    if ROSA_LIQUIDITY_CSV and os.path.exists(ROSA_LIQUIDITY_CSV):
        return pd.read_csv(ROSA_LIQUIDITY_CSV)
    return None


def _load_jamie_profile():
    """Load Jamie's liquidity CSV. Returns DataFrame or raises FileNotFoundError."""
    if not os.path.exists(JAMIE_LIQUIDITY_CSV):
        raise FileNotFoundError(
            f"Neither Rosa's nor Jamie's liquidity CSV was found.\n"
            f"Run rosa.py first (generates har_rv_stock_liquidity_profile.csv),\n"
            f"or run jamie_eda.py (generates jamie_liquidity.csv)."
        )
    return pd.read_csv(JAMIE_LIQUIDITY_CSV)


def get_liquidity_map():
    """
    Returns {stock_id: regime} where regime in {'liquid', 'illiquid', 'mixed'}.

    Source priority:
      1. Rosa's har_rv_stock_liquidity_profile.csv  - uses 'stock_regime' column
      2. Jamie's jamie_liquidity.csv                - 'liquidity_regime' column
    """
    rosa = _load_rosa_profile()
    if rosa is not None and "stock_regime" in rosa.columns:
        regime_map = dict(zip(rosa["stock_id"], rosa["stock_regime"]))
        print(f"  [config] Liquidity map loaded from Rosa's profile "
              f"({ROSA_LIQUIDITY_CSV}) -- {len(regime_map)} stocks")
        return regime_map

    jamie = _load_jamie_profile()
    regime_map = dict(zip(jamie["stock_id"], jamie["liquidity_regime"]))
    if "liquidity_score" in jamie.columns:
        for sid, score in zip(jamie["stock_id"], jamie["liquidity_score"]):
            if MIXED_SCORE_LO <= score <= MIXED_SCORE_HI:
                regime_map[sid] = "mixed"
    print(f"  [config] Liquidity map loaded from Jamie's CSV -- {len(regime_map)} stocks")
    return regime_map


def get_selected_stocks(df=None):
    """
    Returns a dict with keys: 'liquid', 'illiquid', 'mixed', 'all'.

    Priority:
      1. Manual overrides (LIQUID_STOCKS / ILLIQUID_STOCKS / MIXED_STOCKS above)
         -- if any list is non-empty, all three are used as-is.
      2. selected_stocks.csv cache -- loaded if it exists and is non-empty.
      3. Auto-selection from Rosa's or Jamie's liquidity profile + raw data.
    """
    # Manual overrides take top priority
    if LIQUID_STOCKS or ILLIQUID_STOCKS or MIXED_STOCKS:
        liquid = [int(s) for s in LIQUID_STOCKS]
        illiquid = [int(s) for s in ILLIQUID_STOCKS]
        mixed = [int(s) for s in MIXED_STOCKS]
        print(f"  [config] Using manual stock overrides:")
        print(f"    Liquid   ({len(liquid)}):   {liquid}")
        print(f"    Illiquid ({len(illiquid)}): {illiquid}")
        print(f"    Mixed    ({len(mixed)}):    {mixed}")
        return {
            "liquid":   liquid,
            "illiquid": illiquid,
            "mixed":    mixed,
            "all":      sorted(liquid + illiquid + mixed),
        }

    # CSV cache
    if os.path.exists(SELECTED_STOCKS_CSV):
        sel = pd.read_csv(SELECTED_STOCKS_CSV)
        if not sel.empty:
            liquid = sel[sel["regime"] == "liquid"]["stock_id"].tolist()
            illiquid = sel[sel["regime"] == "illiquid"]["stock_id"].tolist()
            mixed = sel[sel["regime"] == "mixed"]["stock_id"].tolist()
            if liquid or illiquid:
                return {
                    "liquid":   liquid,
                    "illiquid": illiquid,
                    "mixed":    mixed,
                    "all":      sorted(liquid + illiquid + mixed),
                }

    # Auto-select
    if df is None:
        raise ValueError(
            "selected_stocks.csv not found/empty and no DataFrame provided.\n"
            "Call get_selected_stocks(df=df) with the raw optiver DataFrame."
        )
    return _compute_and_save(df)


def _compute_and_save(df):
    """Compute stock selection and persist to selected_stocks.csv."""
    stock_bas = df.groupby("stock_id")["BidAskSpread_mean"].median()

    rosa = _load_rosa_profile()
    if rosa is not None and "stock_regime" in rosa.columns:
        liquid_pool = rosa[rosa["stock_regime"] == "liquid"]["stock_id"].tolist()
        illiquid_pool = rosa[rosa["stock_regime"] == "illiquid"]["stock_id"].tolist()
        mixed_pool = rosa[rosa["stock_regime"] == "mixed"]["stock_id"].tolist()

        liq_bas = stock_bas[stock_bas.index.isin(liquid_pool)]
        ilq_bas = stock_bas[stock_bas.index.isin(illiquid_pool)]
        mix_bas = stock_bas[stock_bas.index.isin(mixed_pool)]

        liquid_sel = liq_bas.sort_values(ascending=True).head(N_STOCKS_PER_REGIME).index.tolist()
        illiquid_sel = ilq_bas.sort_values(ascending=False).head(N_STOCKS_PER_REGIME).index.tolist()

        already = set(liquid_sel + illiquid_sel)
        mix_cands = [s for s in mix_bas.index if s not in already]
        rng = np.random.default_rng(RANDOM_SEED)
        n_mix = min(N_STOCKS_PER_REGIME, len(mix_cands))
        mixed_sel = sorted(rng.choice(mix_cands, size=n_mix, replace=False).tolist())

        source = "Rosa's profile"

    else:
        jamie = _load_jamie_profile()
        liq_df = jamie.copy()
        liquidity_map = dict(zip(liq_df["stock_id"], liq_df["liquidity_regime"]))
        score_map = dict(zip(liq_df["stock_id"],
                             liq_df["liquidity_score"] if "liquidity_score" in liq_df.columns
                             else pd.Series(dtype=float)))

        liquid_pool = [s for s, r in liquidity_map.items() if r == "liquid"]
        illiquid_pool = [s for s, r in liquidity_map.items() if r == "illiquid"]

        liquid_sel = (stock_bas[stock_bas.index.isin(liquid_pool)]
                      .sort_values(ascending=True).head(N_STOCKS_PER_REGIME).index.tolist())
        illiquid_sel = (stock_bas[stock_bas.index.isin(illiquid_pool)]
                        .sort_values(ascending=False).head(N_STOCKS_PER_REGIME).index.tolist())

        already = set(liquid_sel + illiquid_sel)
        mix_cands = [s for s in liq_df["stock_id"]
                     if MIXED_SCORE_LO <= score_map.get(s, 0) <= MIXED_SCORE_HI
                     and s not in already]
        rng = np.random.default_rng(RANDOM_SEED)
        n_mix = min(N_STOCKS_PER_REGIME, len(mix_cands))
        mixed_sel = sorted(rng.choice(mix_cands, size=n_mix, replace=False).tolist()
                           if mix_cands else [])
        source = "Jamie's CSV (fallback)"

    rows = (
        [{"stock_id": s, "regime": "liquid",
          "median_BAS": stock_bas.get(s, np.nan)} for s in liquid_sel] +
        [{"stock_id": s, "regime": "illiquid",
          "median_BAS": stock_bas.get(s, np.nan)} for s in illiquid_sel] +
        [{"stock_id": s, "regime": "mixed",
          "median_BAS": stock_bas.get(s, np.nan)} for s in mixed_sel]
    )
    pd.DataFrame(rows).sort_values("stock_id").to_csv(SELECTED_STOCKS_CSV, index=False)

    print(f"  [config] Stock selection from {source}:")
    print(f"    Liquid   ({len(liquid_sel)}):   {liquid_sel}")
    print(f"    Illiquid ({len(illiquid_sel)}): {illiquid_sel}")
    print(f"    Mixed    ({len(mixed_sel)}):    {mixed_sel}")

    return {
        "liquid":   liquid_sel,
        "illiquid": illiquid_sel,
        "mixed":    mixed_sel,
        "all":      sorted(liquid_sel + illiquid_sel + mixed_sel),
    }


def filter_time_ids(vol_train, time_ids, regime):
    """
    Return the subset of time_ids most representative of the given regime:
      liquid   -> tightest BAS (bottom TIME_ID_KEEP_PCT)
      illiquid -> widest BAS   (top TIME_ID_KEEP_PCT)
      mixed    -> keep ALL     (no filtering)
    """
    if regime == "mixed":
        return time_ids

    tid_bas = {tid: vol_train[tid]["BidAskSpread_mean"].median() for tid in time_ids}
    tid_s = pd.Series(tid_bas).sort_values(ascending=(regime == "liquid"))
    n_keep = max(1, int(len(tid_s) * TIME_ID_KEEP_PCT))
    return tid_s.head(n_keep).index.tolist()
