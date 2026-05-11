"""
M4 — Phase 4 PLACEHOLDER: GARCH / EGARCH-X Models
Jisu's deliverable

⚠ THIS IS A STUB. Generates synthetic CSVs and plots so the pipeline
  and app can run end-to-end while the real models are being tuned.

When ready, run arma_models.py instead — it writes to the same paths.
Delete jisu_is_placeholder.json once real outputs exist.

Outputs (written to OUTPUT_DIR/stock_{id}/):
  garch_eval_results.csv
  egarchx_eval_results.csv  (liquid stocks only)
  garch_blowups.csv         (illiquid stocks only)

Outputs (written to OUTPUT_DIR/):
  all_stocks_garch_summary.csv
  all_stocks_egarchx_summary.csv
  global_blowup_report.csv
  jisu_01_egarchx_vs_garch_liquid.png
  jisu_02_garch_blowups_illiquid.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from config import (get_selected_stocks, get_liquidity_map,
                    OUTPUT_DIR, JAMIE_LIQUIDITY_CSV, N_STOCKS_PER_REGIME)

N_FAKE_TIME_IDS = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(JAMIE_LIQUIDITY_CSV):
    raise FileNotFoundError("jamie_liquidity.csv not found. Run jamie_eda.py first.")

liquidity_map   = get_liquidity_map()
selected        = get_selected_stocks()
liquid_sel      = selected["liquid"]
illiquid_sel    = selected["illiquid"]
selected_stocks = selected["all"]

print(f"Generating placeholder outputs for {len(selected_stocks)} stocks ...")
print(f"  Liquid   ({len(liquid_sel)}):  {liquid_sel}")
print(f"  Illiquid ({len(illiquid_sel)}): {illiquid_sel}")
print("(Replace with real arma_models.py outputs when ready)\n")

np.random.seed(42)

all_garch_summary   = []
all_egarchx_summary = []
all_blowups         = []

C_GARCH = "#378ADD"; C_EX = "#1D9E75"; C_HAR = "#D85A30"; SPINE = "#D3D1C7"

for STOCK_ID in selected_stocks:
    regime    = liquidity_map.get(STOCK_ID, "liquid")
    stock_out = os.path.join(OUTPUT_DIR, f"stock_{STOCK_ID}")
    os.makedirs(stock_out, exist_ok=True)

    time_ids  = list(range(1, N_FAKE_TIME_IDS + 1))
    actual_rv = np.random.lognormal(mean=-4, sigma=0.5, size=len(time_ids))

    # GARCH — all stocks
    base_q  = 0.85 if regime == "liquid" else 1.25
    qlike_g = np.random.gamma(shape=2.0, scale=base_q / 2.0, size=len(time_ids))
    blowup_mask = np.zeros(len(time_ids), dtype=bool)
    if regime == "illiquid":
        blowup_mask = np.random.random(len(time_ids)) < 0.058
        qlike_g[blowup_mask] = np.random.uniform(8, 25, blowup_mask.sum())

    pred_g = np.clip(actual_rv * np.random.lognormal(0, 0.15, len(time_ids)), 1e-6, None)
    garch_eval = pd.DataFrame({"time_id": time_ids, "pred_RV": pred_g,
                                "actual_RV": actual_rv, "QLIKE": qlike_g,
                                "MSE": (actual_rv - pred_g) ** 2, "stock_id": STOCK_ID})
    garch_eval.to_csv(os.path.join(stock_out, "garch_eval_results.csv"), index=False)

    if regime == "illiquid" and blowup_mask.any():
        blowup_df = garch_eval[blowup_mask].copy()
        blowup_df["model"] = "GARCH"
        blowup_df.to_csv(os.path.join(stock_out, "garch_blowups.csv"), index=False)
        all_blowups.append(blowup_df)

    per_g = garch_eval.groupby("time_id")[["QLIKE", "MSE"]].mean()
    all_garch_summary.append({"stock_id": STOCK_ID, "liquidity_regime": regime,
                               "n_time_ids": len(per_g), "n_forecasts": len(per_g),
                               "median_QLIKE": per_g["QLIKE"].median(),
                               "median_MSE": per_g["MSE"].median(),
                               "blowup_pct": round(blowup_mask.mean(), 4),
                               "is_placeholder": True})

    # EGARCH-X — liquid only
    if regime == "liquid":
        qlike_ex = qlike_g * np.random.uniform(0.72, 0.92, len(time_ids))
        pred_ex  = np.clip(actual_rv * np.random.lognormal(0, 0.09, len(time_ids)), 1e-6, None)
        egarchx_eval = pd.DataFrame({"time_id": time_ids, "pred_RV": pred_ex,
                                      "actual_RV": actual_rv, "QLIKE": qlike_ex,
                                      "MSE": (actual_rv - pred_ex) ** 2, "stock_id": STOCK_ID})
        egarchx_eval.to_csv(os.path.join(stock_out, "egarchx_eval_results.csv"), index=False)
        per_ex = egarchx_eval.groupby("time_id")[["QLIKE", "MSE"]].mean()
        all_egarchx_summary.append({"stock_id": STOCK_ID, "liquidity_regime": regime,
                                     "n_time_ids": len(per_ex), "n_forecasts": len(per_ex),
                                     "median_QLIKE": per_ex["QLIKE"].median(),
                                     "median_MSE": per_ex["MSE"].median(),
                                     "blowup_pct": float("nan"), "egarchx_ran": True,
                                     "is_placeholder": True})
    else:
        all_egarchx_summary.append({"stock_id": STOCK_ID, "liquidity_regime": regime,
                                     "n_time_ids": 0, "n_forecasts": 0,
                                     "median_QLIKE": float("nan"), "median_MSE": float("nan"),
                                     "blowup_pct": float("nan"), "egarchx_ran": False,
                                     "is_placeholder": True})

    tag = f"  blowups: {blowup_mask.sum()}" if regime == "illiquid" else ""
    print(f"  Stock {STOCK_ID:4d} ({regime:9s})  GARCH QLIKE: {per_g['QLIKE'].median():.3f}{tag}")

# Aggregate CSVs
garch_df   = pd.DataFrame(all_garch_summary).sort_values("stock_id")
egarchx_df = pd.DataFrame(all_egarchx_summary).sort_values("stock_id")
garch_df.to_csv(  os.path.join(OUTPUT_DIR, "all_stocks_garch_summary.csv"),   index=False)
egarchx_df.to_csv(os.path.join(OUTPUT_DIR, "all_stocks_egarchx_summary.csv"), index=False)
print("\nSaved: all_stocks_garch_summary.csv")
print("Saved: all_stocks_egarchx_summary.csv")

if all_blowups:
    blowup_report = pd.concat(all_blowups, ignore_index=True)
    blowup_report["liquidity_regime"] = blowup_report["stock_id"].map(liquidity_map)
    blowup_report.to_csv(os.path.join(OUTPUT_DIR, "global_blowup_report.csv"), index=False)
    print(f"Saved: global_blowup_report.csv  ({len(blowup_report)} blowups on illiquid stocks)")

# Plot 1: EGARCH-X vs GARCH on liquid stocks
liquid_g  = garch_df[garch_df["liquidity_regime"] == "liquid"].copy()
liquid_ex = egarchx_df[egarchx_df["egarchx_ran"] == True].copy()

if not liquid_g.empty and not liquid_ex.empty:
    merged = liquid_g.merge(liquid_ex[["stock_id","median_QLIKE","median_MSE"]],
                            on="stock_id", suffixes=("_garch","_egarchx")
                            ).sort_values("median_QLIKE_garch")
    x = np.arange(len(merged)); w = 0.38
    xlabels = [f"Stock {int(s)}" for s in merged["stock_id"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle("EGARCH-X vs GARCH — Liquid Stocks  (lower = better)", fontsize=12, fontweight="500")

    for ax, metric, title in zip(axes,
                                  ["median_QLIKE","median_MSE"],
                                  ["Median QLIKE per stock","Median MSE per stock"]):
        ax.set_facecolor("white")
        ax.bar(x - w/2, merged[f"{metric}_garch"],   width=w, label="GARCH(1,1)", color=C_GARCH, alpha=0.8)
        ax.bar(x + w/2, merged[f"{metric}_egarchx"],  width=w, label="EGARCH-X",   color=C_EX,    alpha=0.8)
        for i, (g, ex) in enumerate(zip(merged[f"{metric}_garch"], merged[f"{metric}_egarchx"])):
            ax.annotate("*", xy=(i + (w/2 if ex < g else -w/2), min(g, ex)),
                        ha="center", va="top", fontsize=10,
                        color=C_EX if ex < g else C_GARCH)
        for sp in ax.spines.values(): sp.set_color(SPINE); sp.set_linewidth(0.6)
        ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7.5)
        ax.tick_params(labelsize=8, color=SPINE)
        ax.set_title(title, fontsize=10, fontweight="500")
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(axis="y", color=SPINE, linewidth=0.4, linestyle="--"); ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "jisu_01_egarchx_vs_garch_liquid.png"), dpi=150, bbox_inches="tight")
    plt.close()
    ex_wins = (merged["median_QLIKE_egarchx"] < merged["median_QLIKE_garch"]).sum()
    print(f"Saved: jisu_01_egarchx_vs_garch_liquid.png  (EGARCH-X wins on {ex_wins}/{len(merged)} liquid stocks)")

# Plot 2: GARCH blowups on illiquid stocks
illiq_g = garch_df[garch_df["liquidity_regime"] == "illiquid"].sort_values("blowup_pct", ascending=False)

if not illiq_g.empty:
    xlabels2 = [f"Stock {int(s)}" for s in illiq_g["stock_id"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle("GARCH Blowups — Illiquid Stocks  (same stocks as Rosa's HAR-RV)", fontsize=12, fontweight="500")

    ax = axes[0]; ax.set_facecolor("white")
    bar_colors = [C_HAR if p > 0.05 else C_GARCH for p in illiq_g["blowup_pct"]]
    ax.bar(range(len(illiq_g)), illiq_g["blowup_pct"] * 100, color=bar_colors, alpha=0.8, width=0.7)
    ax.axhline(5.8, color="#888780", linestyle="--", linewidth=0.9)
    for sp in ax.spines.values(): sp.set_color(SPINE); sp.set_linewidth(0.6)
    ax.set_xticks(range(len(xlabels2))); ax.set_xticklabels(xlabels2, rotation=45, ha="right", fontsize=7.5)
    ax.tick_params(labelsize=8, color=SPINE)
    ax.set_ylabel("Blowup rate (%)", fontsize=9)
    ax.set_title("GARCH blowup rate per illiquid stock", fontsize=10, fontweight="500")
    ax.legend(handles=[Patch(color=C_HAR, label="> 5% — HAR-RV strongly preferred"),
                       Patch(color=C_GARCH, label="< 5% — GARCH borderline"),
                       plt.Line2D([0],[0], color="#888780", linestyle="--", label="5.8% ref")],
              fontsize=7.5, framealpha=0.9)
    ax.grid(axis="y", color=SPINE, linewidth=0.4, linestyle="--"); ax.set_axisbelow(True)

    ax2 = axes[1]; ax2.set_facecolor("white")
    liq_q   = garch_df[garch_df["liquidity_regime"] == "liquid"]["median_QLIKE"].dropna()
    illiq_q = illiq_g["median_QLIKE"].dropna()
    bp = ax2.boxplot([liq_q.values, illiq_q.values], patch_artist=True,
                     labels=["Liquid stocks","Illiquid stocks"], widths=0.45,
                     medianprops=dict(color="white", linewidth=2),
                     whiskerprops=dict(color=SPINE, linewidth=0.8),
                     capprops=dict(color=SPINE, linewidth=0.8),
                     flierprops=dict(marker="o", markersize=3, markerfacecolor=SPINE, alpha=0.5))
    bp["boxes"][0].set_facecolor(C_GARCH); bp["boxes"][0].set_alpha(0.8)
    bp["boxes"][1].set_facecolor(C_HAR);   bp["boxes"][1].set_alpha(0.8)
    for sp in ax2.spines.values(): sp.set_color(SPINE); sp.set_linewidth(0.6)
    ax2.tick_params(labelsize=9, color=SPINE)
    ax2.set_ylabel("Median QLIKE", fontsize=9)
    ax2.set_title("GARCH QLIKE: liquid vs illiquid\n(why GARCH fails on illiquid)", fontsize=10, fontweight="500")
    ax2.grid(axis="y", color=SPINE, linewidth=0.4, linestyle="--"); ax2.set_axisbelow(True)
    for i, vals in enumerate([liq_q, illiq_q]):
        ax2.annotate(f"Median\n{vals.median():.3f}", xy=(i+1, vals.median()),
                     xytext=(8,0), textcoords="offset points", fontsize=8, va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "jisu_02_garch_blowups_illiquid.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: jisu_02_garch_blowups_illiquid.png")

with open(os.path.join(OUTPUT_DIR, "jisu_is_placeholder.json"), "w") as f:
    json.dump({"placeholder": True, "message": "Replace with real arma_models.py outputs"}, f)

print(f"\n Placeholder complete — {len(selected_stocks)} stocks.")
print(f"  Delete jisu_is_placeholder.json once real arma_models.py outputs exist.")