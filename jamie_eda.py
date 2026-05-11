"""
M4 — Phase 1 & 2: Liquidity EDA and Stock Classification
Jamie's deliverable

What this does:
  1. Explores BAS and order volume distributions across all 127 stocks
  2. Classifies each stock as liquid or illiquid
  3. Flags the most extreme time_ids within each stock for the models to use
  4. Outputs jamie_liquidity.csv — the shared input every other script depends on

Classification logic:
  A stock is LIQUID if:
    - Median BAS is in the bottom 50% across all stocks  (tight spread)
    - Median order volume is in the top 50%              (deep book)
  Stocks that conflict (tight spread but low volume, or vice versa) are
  classified by BAS alone as the primary signal.

Outputs:
  jamie_liquidity.csv         — stock-level regime labels (shared input)
  jamie_timeid_flags.csv      — per time_id liquidity scores for model filtering
  jamie_eda_summary.png       — overview plots
  jamie_eda_per_stock.png     — BAS vs volume scatter coloured by regime
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────

# ── Paths ────────────────────────────────────────────────────────────────────
INPUT_CSV  = r"C:\Users\songj\OneDrive\UNI\Y3SEM1\DATA3888\Project Folder\DATA3888G08\optiver_aggregated.csv"

_HERE      = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_HERE, "m4_outputs")
# ─────────────────────────────────────────────────────────────────────────────

# Percentile cutoffs for classification
BAS_LIQUID_PERCENTILE    = 50   # stocks below this BAS percentile = liquid
VOL_LIQUID_PERCENTILE    = 50   # stocks above this volume percentile = liquid

# Within-stock time_id flagging: keep this % as most extreme per regime
TIMEID_EXTREME_PCT = 0.5

# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Load data
# ══════════════════════════════════════════════════════════════════════════════

print("Loading data ...")
df = pd.read_csv(INPUT_CSV)
print(f"  {len(df)} rows, {df['stock_id'].nunique()} stocks, "
      f"{df['time_id'].nunique()} unique time_ids")
print(f"  Columns: {list(df.columns)}")

# Detect volume column (may vary by dataset version)
vol_col = None
for candidate in ["order_count", "volume", "total_volume", "bid_size", "ask_size"]:
    if candidate in df.columns:
        vol_col = candidate
        break
if vol_col is None:
    # Fallback: use inverse BAS as proxy for volume
    print("  ⚠ No volume column found — using 1/BAS as volume proxy.")
    df["_vol_proxy"] = 1.0 / df["BidAskSpread_mean"].replace(0, np.nan)
    vol_col = "_vol_proxy"
print(f"  Volume column: {vol_col}")


# ══════════════════════════════════════════════════════════════════════════════
#  Stock-level aggregation
# ══════════════════════════════════════════════════════════════════════════════

print("\nAggregating stock-level statistics ...")

stock_stats = df.groupby("stock_id").agg(
    median_BAS    = ("BidAskSpread_mean", "median"),
    mean_BAS      = ("BidAskSpread_mean", "mean"),
    std_BAS       = ("BidAskSpread_mean", "std"),
    p90_BAS       = ("BidAskSpread_mean", lambda x: np.percentile(x, 90)),
    median_vol    = (vol_col, "median"),
    mean_vol      = (vol_col, "mean"),
    n_time_ids    = ("time_id", "nunique"),
    n_buckets     = ("time_id", "count"),
    median_RV     = ("RV", "median"),
).reset_index()

# Normalise for comparability
stock_stats["BAS_rank"]    = stock_stats["median_BAS"].rank(pct=True)  # low = liquid
stock_stats["vol_rank"]    = stock_stats["median_vol"].rank(pct=True)  # high = liquid

print(f"  BAS range:    [{stock_stats['median_BAS'].min():.6f}, "
      f"{stock_stats['median_BAS'].max():.6f}]")
print(f"  Volume range: [{stock_stats['median_vol'].min():.2f}, "
      f"{stock_stats['median_vol'].max():.2f}]")


# ══════════════════════════════════════════════════════════════════════════════
#  Liquidity classification
# ══════════════════════════════════════════════════════════════════════════════

print("\nClassifying stocks into liquidity regimes ...")

bas_threshold = np.percentile(stock_stats["median_BAS"], BAS_LIQUID_PERCENTILE)
vol_threshold = np.percentile(stock_stats["median_vol"], VOL_LIQUID_PERCENTILE)

def classify(row):
    bas_liquid = row["median_BAS"] <= bas_threshold
    vol_liquid = row["median_vol"] >= vol_threshold
    if bas_liquid and vol_liquid:
        return "liquid"
    elif not bas_liquid and not vol_liquid:
        return "illiquid"
    else:
        # Conflicting signals — BAS is primary
        return "liquid" if bas_liquid else "illiquid"

stock_stats["liquidity_regime"] = stock_stats.apply(classify, axis=1)

n_liquid   = (stock_stats["liquidity_regime"] == "liquid").sum()
n_illiquid = (stock_stats["liquidity_regime"] == "illiquid").sum()
print(f"  BAS threshold:    {bas_threshold:.6f}")
print(f"  Volume threshold: {vol_threshold:.2f}")
print(f"  Liquid:   {n_liquid} stocks")
print(f"  Illiquid: {n_illiquid} stocks")

# Liquidity score (0–1, higher = more liquid) for ranking
stock_stats["liquidity_score"] = (
    (1 - stock_stats["BAS_rank"]) * 0.6 +   # BAS weighted more heavily
    stock_stats["vol_rank"]       * 0.4
)


# ══════════════════════════════════════════════════════════════════════════════
#  Per time_id liquidity flagging (within each stock)
# ══════════════════════════════════════════════════════════════════════════════

print("\nFlagging most extreme time_ids within each stock ...")

timeid_stats = df.groupby(["stock_id", "time_id"]).agg(
    median_BAS = ("BidAskSpread_mean", "median"),
    median_vol = (vol_col, "median"),
    median_RV  = ("RV", "median"),
    n_buckets  = ("time_bucket", "count"),
).reset_index()

# Merge regime
timeid_stats = timeid_stats.merge(
    stock_stats[["stock_id", "liquidity_regime", "liquidity_score"]],
    on="stock_id", how="left"
)

# Within each stock, rank time_ids by BAS (ascending = most liquid)
timeid_stats["bas_rank_within_stock"] = (
    timeid_stats.groupby("stock_id")["median_BAS"]
    .rank(pct=True, ascending=True)
)

def flag_timeid(row):
    """Flag whether this time_id is in the extreme half for its stock's regime."""
    if row["liquidity_regime"] == "liquid":
        return row["bas_rank_within_stock"] <= TIMEID_EXTREME_PCT
    else:
        return row["bas_rank_within_stock"] >= (1 - TIMEID_EXTREME_PCT)

timeid_stats["regime_representative"] = timeid_stats.apply(flag_timeid, axis=1)
n_flagged = timeid_stats["regime_representative"].sum()
print(f"  {n_flagged} / {len(timeid_stats)} time_ids flagged as most regime-representative")


# ══════════════════════════════════════════════════════════════════════════════
#  Save outputs
# ══════════════════════════════════════════════════════════════════════════════

# Primary output — used by all other scripts
liquidity_out = stock_stats[[
    "stock_id", "liquidity_regime", "liquidity_score",
    "median_BAS", "median_vol", "n_time_ids"
]].sort_values("stock_id").copy()
liquidity_out["model_recommended"] = liquidity_out["liquidity_regime"].map(
    {"liquid": "EGARCH-X", "illiquid": "HAR-RV"}
)
liquidity_out["baseline_model"] = "GARCH(1,1)"
liquidity_out.to_csv(os.path.join(OUTPUT_DIR, "jamie_liquidity.csv"), index=False)
print(f"\nSaved: jamie_liquidity.csv")

# Time_id flags — used by arma_models.py for within-stock filtering
timeid_out = timeid_stats[[
    "stock_id", "time_id", "liquidity_regime",
    "median_BAS", "median_vol", "bas_rank_within_stock", "regime_representative"
]].sort_values(["stock_id", "time_id"])
timeid_out.to_csv(os.path.join(OUTPUT_DIR, "jamie_timeid_flags.csv"), index=False)
print(f"Saved: jamie_timeid_flags.csv")

# Full stock stats for the team
stock_stats.to_csv(os.path.join(OUTPUT_DIR, "jamie_stock_stats.csv"), index=False)
print(f"Saved: jamie_stock_stats.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  EDA Plots
# ══════════════════════════════════════════════════════════════════════════════

print("\nGenerating EDA plots ...")

C_LIQUID   = "#1D9E75"
C_ILLIQUID = "#D85A30"
C_NEUTRAL  = "#888780"
SPINE_COL  = "#D3D1C7"

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_color(SPINE_COL)
        sp.set_linewidth(0.6)
    ax.tick_params(labelsize=9, color=SPINE_COL)
    ax.set_title(title, fontsize=11, fontweight="500", pad=8)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9, color="#5F5E5A")
    if ylabel: ax.set_ylabel(ylabel, fontsize=9, color="#5F5E5A")
    ax.grid(axis="y", color=SPINE_COL, linewidth=0.5, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

liquid_df   = stock_stats[stock_stats["liquidity_regime"] == "liquid"].copy()
illiquid_df = stock_stats[stock_stats["liquidity_regime"] == "illiquid"].copy()

# ── Plot 1: Liquidity map — BAS vs Volume (the key client chart) ─────────────
fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor("white")
style_ax(ax,
         title="Stock Liquidity Map — Bid-Ask Spread vs Order Volume",
         xlabel="Median Bid-Ask Spread  (← tighter = more liquid)",
         ylabel="Median Order Volume  (higher = more liquid →)")

ax.scatter(liquid_df["median_BAS"], liquid_df["median_vol"],
           c=C_LIQUID, label=f"Liquid ({n_liquid} stocks)",
           alpha=0.80, s=70, edgecolors="white", linewidth=0.6, zorder=3)
ax.scatter(illiquid_df["median_BAS"], illiquid_df["median_vol"],
           c=C_ILLIQUID, label=f"Illiquid ({n_illiquid} stocks)",
           alpha=0.80, s=70, edgecolors="white", linewidth=0.6, zorder=3)

# Quadrant shading
ylim = ax.get_ylim(); xlim = ax.get_xlim()
ax.axvline(bas_threshold, color=C_NEUTRAL, linestyle="--", linewidth=0.9, alpha=0.6, zorder=2)
ax.axhline(vol_threshold, color=C_NEUTRAL, linestyle="--", linewidth=0.9, alpha=0.6, zorder=2)
ax.fill_betweenx([vol_threshold, ylim[1]], xlim[0], bas_threshold,
                 color=C_LIQUID, alpha=0.05, zorder=1)
ax.fill_betweenx([ylim[0], vol_threshold], bas_threshold, xlim[1],
                 color=C_ILLIQUID, alpha=0.05, zorder=1)
ax.set_xlim(xlim); ax.set_ylim(ylim)
ax.text(xlim[0]*1.01, ylim[1]*0.97, "LIQUID\n(EGARCH-X)",
        fontsize=8, color=C_LIQUID, va="top", fontweight="500")
ax.text(xlim[1]*0.99, ylim[0]*1.03, "ILLIQUID\n(HAR-RV)",
        fontsize=8, color=C_ILLIQUID, va="bottom", ha="right", fontweight="500")

# Annotate the 5 most extreme stocks each side
for grp, col, n in [(liquid_df, C_LIQUID, 5), (illiquid_df, C_ILLIQUID, 5)]:
    extremes = grp.nlargest(n, "liquidity_score") if col == C_LIQUID \
               else grp.nsmallest(n, "liquidity_score")
    for _, row in extremes.iterrows():
        ax.annotate(f"  {int(row['stock_id'])}",
                    (row["median_BAS"], row["median_vol"]),
                    fontsize=7.5, color=col, alpha=0.9)

ax.legend(fontsize=10, framealpha=0.9, edgecolor=SPINE_COL)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "jamie_01_liquidity_map.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: jamie_01_liquidity_map.png")

# ── Plot 2: Ranked liquidity score — all 127 stocks ──────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor("white")
ranked = stock_stats.sort_values("liquidity_score", ascending=False).reset_index(drop=True)
bar_colors = [C_LIQUID if r == "liquid" else C_ILLIQUID
              for r in ranked["liquidity_regime"]]
ax.bar(range(len(ranked)), ranked["liquidity_score"], color=bar_colors,
       width=0.85, alpha=0.85)
ax.axhline(0.5, color=C_NEUTRAL, linestyle="--", linewidth=0.9, alpha=0.6)
style_ax(ax, title="All Stocks Ranked by Liquidity Score",
         xlabel="Stock rank (most liquid → least liquid)",
         ylabel="Composite liquidity score")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_LIQUID, label=f"Liquid ({n_liquid})"),
                   Patch(color=C_ILLIQUID, label=f"Illiquid ({n_illiquid})")],
          fontsize=9, framealpha=0.9)
# Label top 5 and bottom 5
for i in list(range(5)) + list(range(len(ranked)-5, len(ranked))):
    sid = int(ranked.iloc[i]["stock_id"])
    ax.text(i, ranked.iloc[i]["liquidity_score"] + 0.01, str(sid),
            ha="center", fontsize=6.5, color="#444441", rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "jamie_02_stocks_ranked.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: jamie_02_stocks_ranked.png")

# ── Plot 3: BAS and RV distribution by regime (box plots) ────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.patch.set_facecolor("white")
fig.suptitle("Market Characteristics by Liquidity Regime", fontsize=12, fontweight="500")

for ax, col, label, unit in zip(
    axes,
    ["median_BAS", "median_vol", "median_RV"],
    ["Bid-Ask Spread", "Order Volume", "Realised Volatility"],
    ["BAS", "units", "RV"]
):
    data = [liquid_df[col].dropna().values, illiquid_df[col].dropna().values]
    bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                    labels=["Liquid", "Illiquid"],
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(color=C_NEUTRAL, linewidth=0.9),
                    capprops=dict(color=C_NEUTRAL, linewidth=0.9),
                    flierprops=dict(marker="o", markersize=3,
                                   markerfacecolor=C_NEUTRAL, alpha=0.5))
    bp["boxes"][0].set_facecolor(C_LIQUID)
    bp["boxes"][1].set_facecolor(C_ILLIQUID)
    bp["boxes"][0].set_alpha(0.8)
    bp["boxes"][1].set_alpha(0.8)
    style_ax(ax, title=label, ylabel=unit)
    # Annotate medians
    for i, vals in enumerate(data):
        med = np.median(vals)
        ax.text(i+1.28, med, f"{med:.4f}" if col == "median_BAS" else f"{med:.2f}",
                fontsize=7.5, va="center", color="#444441")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "jamie_03_regime_characteristics.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: jamie_03_regime_characteristics.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Summary
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  EDA complete — {len(stock_stats)} stocks classified")
print(f"  Liquid:   {n_liquid:3d} stocks  (EGARCH-X + GARCH)")
print(f"  Illiquid: {n_illiquid:3d} stocks  (HAR-RV + GARCH baseline)")
print(f"{'='*60}")
print(f"\n  ✓ jamie_liquidity.csv is ready.")
print(f"    Other scripts will load this automatically.")
print(f"    To update, just re-run this script.")