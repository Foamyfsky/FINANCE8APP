"""
final_figures.py 
Research question:
  "Instead of asking which forecasting model is best for every stock,
   we ask: which model is APPROPRIATE under the current liquidity regime?"
Generates 6 publication-ready figures + a text summary.
Run:
    python final_figures.py
Output:
    m4_outputs/final_figures/fig1_regime_model_heatmap.png
    m4_outputs/final_figures/fig2_per_stock_qlike.png
    m4_outputs/final_figures/fig3_boxplots_by_regime.png
    m4_outputs/final_figures/fig4_routing_outcomes.png
    m4_outputs/final_figures/fig5_bas_vs_improvement.png
    m4_outputs/final_figures/fig6_regime_summary_panel.png
    m4_outputs/final_figures/conclusion.txt
"""

import os, sys, io, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from contextlib import redirect_stdout

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
M4 = os.path.join(HERE, "m4_outputs")
OUT = os.path.join(M4, "final_figures")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, HERE)
with redirect_stdout(io.StringIO()):
    from config import get_selected_stocks, get_liquidity_map
    sel = get_selected_stocks()
    lmap = get_liquidity_map()

C = {
    "garch":"#378ADD",
    "egarchx": "#1D9E75",
    "har": "#D85A30",
    "lgbm":"#8B5CF6",
    "liquid":"#1D9E75",
    "illiquid": "#D85A30",
    "mixed":"#8B5CF6",
    "spine":"#D3D1C7",
    "bg": "white",
}
MODEL_LABELS = {
    "garch":"GARCH(1,1)",
    "egarchx":"EGARCH-X",
    "har":"HAR-RV",
    "lgbm":"LightGBM",
}
REGIME_ORDER = ["liquid", "mixed", "illiquid"]


def load(fname):
    p = os.path.join(M4, fname)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

garch_s = load("all_stocks_garch_summary.csv")
egarchx_s = load("all_stocks_egarchx_summary.csv")
har_s = load("all_stocks_har_rv_summary.csv")
lgbm_s = load("lgbm_outputs/lgbm_per_stock.csv")
profile = load("har_rv_stock_liquidity_profile.csv")
blowups = load("global_blowup_report.csv")

# Build master per-stock dataframe
rows = []
for sid in sel["all"]:
    regime = lmap.get(sid, "mixed")
    r = {"stock_id": sid, "regime": regime}

    g = garch_s[garch_s["stock_id"] == sid]
    r["garch"] = g["median_QLIKE"].values[0] if not g.empty else np.nan

    e = egarchx_s[egarchx_s["stock_id"] == sid]
    r["egarchx"] = e["median_QLIKE"].values[0] if not e.empty else np.nan

    h = har_s[har_s["stock_id"] == sid]
    r["har"] = h["median_QLIKE"].values[0] if not h.empty else np.nan

    l = lgbm_s[lgbm_s["stock_id"] == sid]
    r["lgbm"] = l["median_QLIKE"].values[0] if not l.empty else np.nan

    p = profile[profile["stock_id"] == sid]
    r["median_bas"] = p["median_bas"].values[0] if not p.empty else np.nan
    r["liquid_pct"] = p["liquid_pct"].values[0] if not p.empty else np.nan

    # which model wins for this stock? (lowest QLIKE)
    candidates = {k: r[k] for k in ["garch","egarchx","har","lgbm"] if np.isfinite(r.get(k, np.nan))}
    r["winner"] = min(candidates, key=candidates.get) if candidates else "garch"

    # regime-theory recommended model
    r["regime_rec"] = {"liquid":"egarchx", "illiquid":"har", "mixed":"lgbm"}.get(regime, "garch")

    # did the routing agree with data?
    r["routing_correct"] = r["winner"] == r["regime_rec"]

    rows.append(r)

df = pd.DataFrame(rows)


def style_ax(ax, title="", xlabel="", ylabel="", legend=True):
    ax.set_facecolor(C["bg"])
    for sp in ax.spines.values():
        sp.set_color(C["spine"]); sp.set_linewidth(0.6)
    ax.tick_params(labelsize=8.5, color=C["spine"])
    ax.grid(axis="y", color=C["spine"], linewidth=0.4, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    if title:  ax.set_title(title, fontsize=10, fontweight="600", pad=7)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8.5, color="#5F5E5A")
    if ylabel: ax.set_ylabel(ylabel, fontsize=8.5, color="#5F5E5A")
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7.5, framealpha=0.85, edgecolor=C["spine"])


# Fig 1 - Regime x Model QLIKE heatmap (the "routing logic" figure)
print("Building Fig 1 - regime x model heatmap ...")

med = df.groupby("regime")[["garch","egarchx","har","lgbm"]].median()
med = med.reindex(REGIME_ORDER)
med.columns = [MODEL_LABELS[c] for c in med.columns]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                          gridspec_kw={"width_ratios":[2, 1]})
fig.patch.set_facecolor(C["bg"])
fig.suptitle(
    "Regime x Model Performance  (median QLIKE - lower is better)",
    fontsize=12, fontweight="600", y=1.01
)

# left: grouped bar
ax = axes[0]
x = np.arange(len(REGIME_ORDER))
w = 0.19
colors = [C["garch"], C["egarchx"], C["har"], C["lgbm"]]
for i, (col, col_c) in enumerate(zip(med.columns, colors)):
    vals = med[col].values
    bars = ax.bar(x + (i-1.5)*w, vals, width=w, color=col_c, alpha=0.82, label=col)
    for bar, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width()/2, v - 0.03,
                    f"{v:.3f}", ha="center", va="top", fontsize=6.5, color="white", fontweight="600")

ax.set_xticks(x)
ax.set_xticklabels([r.capitalize() for r in REGIME_ORDER], fontsize=9)
style_ax(ax, ylabel="Median QLIKE", title="All 60 stocks - median per model x regime")

# annotate the winner per regime
regime_winners = {"liquid":"GARCH(1,1)", "illiquid":"LightGBM", "mixed":"LightGBM"}
for xi, regime in enumerate(REGIME_ORDER):
    ax.annotate(f"<- {regime_winners[regime]} wins",
                xy=(xi, med.loc[regime].min()),
                xytext=(xi, med.loc[regime].min() - 0.25),
                ha="center", fontsize=7, color="#333", fontstyle="italic")

# right: heatmap
ax2 = axes[1]
heat = med.values.copy()
masked = np.ma.masked_invalid(heat)
im = ax2.imshow(masked, cmap="RdYlGn", aspect="auto",
                vmin=np.nanmin(heat)-0.1, vmax=np.nanmax(heat)+0.1)
ax2.set_xticks(range(len(med.columns)))
ax2.set_xticklabels(med.columns, rotation=30, ha="right", fontsize=8)
ax2.set_yticks(range(len(REGIME_ORDER)))
ax2.set_yticklabels([r.capitalize() for r in REGIME_ORDER], fontsize=9)
for i in range(len(REGIME_ORDER)):
    for j in range(len(med.columns)):
        v = heat[i, j]
        if np.isfinite(v):
            ax2.text(j, i, f"{v:.3f}", ha="center", va="center",
                     fontsize=8.5, fontweight="600",
                     color="white" if abs(v) > 6.5 else "black")
        else:
            ax2.text(j, i, "N/A", ha="center", va="center", fontsize=8, color="#aaa")
ax2.set_title("Heatmap (green = better)", fontsize=9, fontweight="600")
plt.colorbar(im, ax=ax2, shrink=0.8, label="Median QLIKE")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig1_regime_model_heatmap.png"), dpi=160, bbox_inches="tight")
plt.close()
print("  done fig1")


# Fig 2 - Per-stock QLIKE bars, grouped by regime
print("Building Fig 2 - per-stock QLIKE ...")

fig, axes = plt.subplots(3, 1, figsize=(18, 14))
fig.patch.set_facecolor(C["bg"])
fig.suptitle(
    "Per-Stock Model Comparison - QLIKE by Liquidity Regime\n"
    "(lower = better forecast accuracy)",
    fontsize=12, fontweight="600", y=1.005
)

model_keys = ["garch","egarchx","har","lgbm"]
model_colors = [C["garch"], C["egarchx"], C["har"], C["lgbm"]]
model_names = [MODEL_LABELS[k] for k in model_keys]

for ax, regime in zip(axes, REGIME_ORDER):
    sub = df[df["regime"] == regime].sort_values("stock_id")
    stocks = sub["stock_id"].astype(str).values
    x = np.arange(len(stocks))
    w = 0.19

    for i, (key, col, name) in enumerate(zip(model_keys, model_colors, model_names)):
        vals = sub[key].values
        ax.bar(x + (i-1.5)*w, vals, width=w, color=col, alpha=0.8, label=name)

    # highlight winner per stock with a star
    for xi, (_, row) in enumerate(sub.iterrows()):
        winner_q = row[row["winner"]]
        ax.annotate("*", xy=(xi + (model_keys.index(row["winner"])-1.5)*w, winner_q),
                    ha="center", va="bottom", fontsize=7,
                    color=model_colors[model_keys.index(row["winner"])])

    ax.set_xticks(x)
    ax.set_xticklabels([f"#{s}" for s in stocks], rotation=45, ha="right", fontsize=7.5)
    regime_c = C[regime]
    ax.set_title(f"{regime.upper()} stocks  (n={len(sub)})",
                 fontsize=10, fontweight="600", color=regime_c)
    style_ax(ax, ylabel="Median QLIKE", legend=(regime=="liquid"))
    if regime == "liquid":
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

# shared legend at bottom
patches = [mpatches.Patch(color=model_colors[i], label=model_names[i], alpha=0.8)
           for i in range(4)]
patches.append(mpatches.Patch(color="none", label="* = stock winner"))
fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=9,
           framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_per_stock_qlike.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  done fig2")


# Fig 3 - Box plots: QLIKE distribution per model x regime
print("Building Fig 3 - boxplots ...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
fig.patch.set_facecolor(C["bg"])
fig.suptitle(
    "QLIKE Distribution by Model and Liquidity Regime\n"
    "(boxes = interquartile range across stocks in that regime)",
    fontsize=11, fontweight="600"
)

for ax, regime in zip(axes, REGIME_ORDER):
    sub = df[df["regime"] == regime]
    data, labels, cols = [], [], []
    for key in model_keys:
        vals = sub[key].dropna().values
        if len(vals) > 0:
            data.append(vals)
            labels.append(MODEL_LABELS[key])
            cols.append(C[key])

    bp = ax.boxplot(data, patch_artist=True, labels=labels, widths=0.5,
                    medianprops=dict(color="white", linewidth=2.5),
                    whiskerprops=dict(color=C["spine"], linewidth=0.9),
                    capprops=dict(color=C["spine"], linewidth=0.9),
                    flierprops=dict(marker="o", markersize=3.5,
                                    markerfacecolor=C["spine"], alpha=0.5))
    for patch, col in zip(bp["boxes"], cols):
        patch.set_facecolor(col); patch.set_alpha(0.78)

    # annotate medians
    for i, (vals, lbl) in enumerate(zip(data, labels)):
        med_v = np.median(vals)
        ax.text(i+1.28, med_v, f"{med_v:.3f}", va="center", fontsize=7.5,
                color=cols[i], fontweight="600")

    style_ax(ax, title=f"{regime.upper()} stocks (n={len(sub)})",
             ylabel="Median QLIKE per stock", legend=False)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig3_boxplots_by_regime.png"), dpi=160, bbox_inches="tight")
plt.close()
print("  done fig3")


# Fig 4 - Routing outcomes: theory vs data-driven winner
print("Building Fig 4 - routing outcomes ...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor(C["bg"])
fig.suptitle(
    "Dynamic Routing Outcomes - Which Model Actually Wins Per Stock?\n"
    "(regime theory vs. data-driven recommendation)",
    fontsize=11, fontweight="600"
)

for ax, regime in zip(axes, REGIME_ORDER):
    sub = df[df["regime"] == regime]
    win_counts = sub["winner"].value_counts()

    # show all 4 possible models even if 0
    bars_data = [(MODEL_LABELS[k], win_counts.get(k, 0), C[k]) for k in model_keys]
    bars_data = [(lbl, cnt, col) for lbl, cnt, col in bars_data if cnt > 0]
    labels_b = [b[0] for b in bars_data]
    counts_b = [b[1] for b in bars_data]
    colors_b = [b[2] for b in bars_data]

    bars = ax.bar(labels_b, counts_b, color=colors_b, alpha=0.82, width=0.5)
    for bar, cnt in zip(bars, counts_b):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                str(cnt), ha="center", fontsize=11, fontweight="700")

    # mark the regime-theory recommendation
    rec_name = MODEL_LABELS[{"liquid":"egarchx","illiquid":"har","mixed":"lgbm"}[regime]]
    if rec_name in labels_b:
        idx = labels_b.index(rec_name)
        bars[idx].set_edgecolor("#222")
        bars[idx].set_linewidth(2.5)
        ax.text(idx, counts_b[idx] + 0.6, "theory\nrec v", ha="center",
                fontsize=7, color="#222", fontstyle="italic")

    n_correct = sub["routing_correct"].sum()
    style_ax(ax, title=f"{regime.upper()} (n={len(sub)})\n"
             f"Theory-rec wins: {n_correct}/{len(sub)} stocks",
             ylabel="# stocks where model wins", legend=False)
    ax.set_xticklabels(labels_b, rotation=15, ha="right", fontsize=8.5)
    ax.set_ylim(0, max(counts_b) * 1.4 if counts_b else 5)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4_routing_outcomes.png"), dpi=160, bbox_inches="tight")
plt.close()
print("  done fig4")


# Fig 5 - BAS vs LGBM gain over GARCH (why liquidity regime matters)
print("Building Fig 5 - BAS vs LGBM improvement ...")

df["lgbm_vs_garch"] = df["lgbm"] - df["garch"]  # negative = LGBM better

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(C["bg"])
fig.suptitle(
    "Bid-Ask Spread as a Regime Signal\n"
    "Left: BAS distribution by regime, Right: BAS vs LGBM gain over GARCH",
    fontsize=11, fontweight="600"
)

# left: BAS boxplot by regime
ax = axes[0]
bas_data = [df[df["regime"]==r]["median_bas"].dropna().values * 1e4
            for r in REGIME_ORDER]
bp = ax.boxplot(bas_data, patch_artist=True,
                labels=[r.capitalize() for r in REGIME_ORDER],
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(color=C["spine"], linewidth=0.9),
                capprops=dict(color=C["spine"], linewidth=0.9),
                flierprops=dict(marker="o", markersize=3, markerfacecolor=C["spine"], alpha=0.4))
for patch, regime in zip(bp["boxes"], REGIME_ORDER):
    patch.set_facecolor(C[regime]); patch.set_alpha(0.75)
style_ax(ax, title="Bid-Ask Spread by Regime", ylabel="Median BAS (x10^-4)", legend=False)

# right: scatter BAS vs LGBM improvement
ax2 = axes[1]
for regime in REGIME_ORDER:
    sub = df[df["regime"] == regime]
    ax2.scatter(sub["median_bas"] * 1e4, sub["lgbm_vs_garch"],
                color=C[regime], alpha=0.75, s=55, label=regime.capitalize(),
                edgecolors="white", linewidths=0.4)

# trend line
valid = df.dropna(subset=["median_bas","lgbm_vs_garch"])
z = np.polyfit(valid["median_bas"]*1e4, valid["lgbm_vs_garch"], 1)
xfit = np.linspace(valid["median_bas"].min()*1e4, valid["median_bas"].max()*1e4, 100)
ax2.plot(xfit, np.polyval(z, xfit), "--", color="#555", linewidth=1.2, alpha=0.7,
         label="Trend")
ax2.axhline(0, color=C["spine"], linewidth=0.8, linestyle=":")
ax2.text(valid["median_bas"].max()*1e4*0.55, 0.02,
         "<- GARCH better above this line\nLGBM better below ->",
         fontsize=7.5, color="#555", va="bottom")
style_ax(ax2, title="Higher BAS -> Larger LGBM advantage",
         xlabel="Median BAS (x10^-4)", ylabel="LGBM QLIKE - GARCH QLIKE\n(negative = LGBM wins)")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig5_bas_vs_improvement.png"), dpi=160, bbox_inches="tight")
plt.close()
print("  done fig5")


# Fig 6 - Summary panel (the "poster figure")
print("Building Fig 6 - summary panel ...")

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor(C["bg"])
fig.suptitle(
    "Regime-Based Volatility Forecasting - Group 8 Summary\n"
    "\"Which model is appropriate under the current liquidity regime?\"",
    fontsize=13, fontweight="700", y=1.01
)
gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.48, wspace=0.35)

# top-left: median QLIKE by regime (bar)
ax1 = fig.add_subplot(gs[0, :4])
med_re = df.groupby("regime")[["garch","egarchx","har","lgbm"]].median().reindex(REGIME_ORDER)
x = np.arange(len(REGIME_ORDER))
w = 0.19
for i, (key, col) in enumerate(zip(model_keys, model_colors)):
    vals = med_re[key].values
    bars = ax1.bar(x + (i-1.5)*w, vals, width=w, color=col, alpha=0.82,
                   label=MODEL_LABELS[key])
    for bar, v in zip(bars, vals):
        if np.isfinite(v):
            ax1.text(bar.get_x()+bar.get_width()/2, v-0.04,
                     f"{v:.2f}", ha="center", va="top",
                     fontsize=6.5, color="white", fontweight="600")

ax1.set_xticks(x)
ax1.set_xticklabels([r.capitalize() for r in REGIME_ORDER], fontsize=10)
style_ax(ax1, title="Median QLIKE by Regime and Model  (lower = better)",
         ylabel="Median QLIKE")

# top-right: pie - how often does regime routing agree with data?
ax2 = fig.add_subplot(gs[0, 4:])
correct = df["routing_correct"].sum()
incorrect = len(df) - correct
ax2.pie([correct, incorrect],
        labels=[f"Theory-rec wins\n({correct} stocks)",
                f"Data overrides\n({incorrect} stocks)"],
        colors=[C["liquid"], C["spine"]],
        autopct="%1.0f%%", startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=9))
ax2.set_title("Regime routing: theory vs. data\n(across all 60 stocks)",
              fontsize=9, fontweight="600")

# bottom-left: winner counts per regime (stacked)
ax3 = fig.add_subplot(gs[1, :3])
winner_mat = pd.crosstab(df["regime"], df["winner"]).reindex(REGIME_ORDER)
winner_mat = winner_mat.reindex(columns=model_keys, fill_value=0)
bottom = np.zeros(3)
for key, col in zip(model_keys, model_colors):
    vals = winner_mat[key].values.astype(float)
    ax3.bar(range(3), vals, bottom=bottom, color=col, alpha=0.82,
            label=MODEL_LABELS[key], width=0.5)
    for xi, (v, b) in enumerate(zip(vals, bottom)):
        if v > 0:
            ax3.text(xi, b + v/2, str(int(v)), ha="center", va="center",
                     fontsize=9, fontweight="700", color="white")
    bottom += vals
ax3.set_xticks(range(3))
ax3.set_xticklabels([r.capitalize() for r in REGIME_ORDER], fontsize=9)
style_ax(ax3, title="Which model wins per stock?",
         ylabel="# stocks", legend=False)
ax3.legend(fontsize=7.5, loc="upper right", framealpha=0.9)

# bottom-mid: LGBM vs GARCH improvement by regime
ax4 = fig.add_subplot(gs[1, 3:])
for regime in REGIME_ORDER:
    sub = df[df["regime"] == regime]["lgbm_vs_garch"].dropna()
    ax4.boxplot(sub.values, positions=[REGIME_ORDER.index(regime)],
                patch_artist=True,
                boxprops=dict(facecolor=C[regime], alpha=0.75),
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(color=C["spine"]),
                capprops=dict(color=C["spine"]),
                flierprops=dict(marker="o", markersize=3,
                                markerfacecolor=C["spine"], alpha=0.4),
                widths=0.4)

ax4.axhline(0, color="#888", linewidth=0.8, linestyle="--")
ax4.set_xticks(range(3))
ax4.set_xticklabels([r.capitalize() for r in REGIME_ORDER], fontsize=9)
ax4.text(0.5, 0.04, "^ GARCH wins", transform=ax4.transAxes,
         ha="center", fontsize=7.5, color="#555", fontstyle="italic")
ax4.text(0.5, 0.88, "v LGBM wins", transform=ax4.transAxes,
         ha="center", fontsize=7.5, color=C["lgbm"], fontstyle="italic")
style_ax(ax4, title="LightGBM advantage over GARCH",
         ylabel="LGBM QLIKE - GARCH QLIKE", legend=False)


plt.savefig(os.path.join(OUT, "fig6_regime_summary_panel.png"),
            dpi=160, bbox_inches="tight")
plt.close()
print("  done fig6")
