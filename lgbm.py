"""
LightGBM Volatility Forecasting — Regime-Based (20 liquid + 20 illiquid)
DATA3888 Group 8

Uses the same 40 stocks selected by config.py so results are directly
comparable to GARCH, EGARCH-X, and HAR-RV.

Runs LightGBM separately per regime:
  liquid   -> compared against GARCH + EGARCH-X
  illiquid -> compared against GARCH + HAR-RV

Outputs -> OUTPUT_DIR/lgbm_outputs/
  lgbm_eval_results.csv
  lgbm_per_stock.csv
  lgbm_feature_importance.csv
  lgbm_01_vs_garch_egarchx.png
  lgbm_02_vs_garch_har.png
  lgbm_03_feature_importance.png
  lgbm_04_pred_vs_actual.png
"""

import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from config import (get_selected_stocks, get_liquidity_map,
                    OUTPUT_DIR, JAMIE_LIQUIDITY_CSV)

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_CSV = r"C:\Users\songj\OneDrive\UNI\Y3SEM1\DATA3888\Project Folder\DATA3888G08\optiver_aggregated.csv"
LGBM_OUT  = os.path.join(OUTPUT_DIR, "lgbm_outputs")

N_TRAIN      = 16
N_VAL        = 4
N_FOLDS      = 5
RANDOM_STATE = 42

C_LGBM  = "#8B5CF6"
C_GARCH = "#378ADD"
C_EX    = "#1D9E75"
C_HAR   = "#D85A30"
SPINE   = "#D3D1C7"

os.makedirs(LGBM_OUT, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def qlike(pred, actual):
    pred   = np.maximum(pred,   1e-10)
    actual = np.maximum(actual, 1e-10)
    return np.log(pred ** 2) + actual ** 2 / pred ** 2


def engineer_features(grp):
    vol    = grp["RV"].values
    spread = grp["BidAskSpread_mean"].values
    wap    = grp["WAP_mean"].values if "WAP_mean" in grp.columns else np.zeros(len(grp))
    return pd.Series({
        "rv_mean":          vol.mean(),
        "rv_std":           vol.std(),
        "rv_first":         vol[0],
        "rv_last":          vol[-1],
        "rv_trend":         vol[-1] - vol[0],
        "rv_max":           vol.max(),
        "rv_min":           vol.min(),
        "rv_first3_mean":   vol[:3].mean(),
        "rv_last3_mean":    vol[-3:].mean(),
        "rv_mid_mean":      vol[6:10].mean(),
        "spread_mean":      spread.mean(),
        "spread_std":       spread.std(),
        "spread_last":      spread[-1],
        "spread_max":       spread.max(),
        "spread_trend":     spread[-1] - spread[0],
        "wap_mean":         wap.mean(),
        "wap_std":          wap.std(),
        "wap_trend":        wap[-1] - wap[0],
        "spread_rv_ratio":  spread.mean() / max(vol.mean(), 1e-10),
    })


def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("white")
    for sp in ax.spines.values(): sp.set_color(SPINE); sp.set_linewidth(0.6)
    ax.tick_params(labelsize=8.5, color=SPINE)
    ax.set_title(title, fontsize=10, fontweight="500", pad=6)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8.5, color="#5F5E5A")
    if ylabel: ax.set_ylabel(ylabel, fontsize=8.5, color="#5F5E5A")
    ax.grid(axis="y", color=SPINE, linewidth=0.4, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)


def load_csv(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading data ...")
df = pd.read_csv(INPUT_CSV)
df = df.sort_values(["stock_id","time_id","time_bucket"]).reset_index(drop=True)
print(f"  {len(df):,} rows | {df['stock_id'].nunique()} stocks")

counts   = df.groupby(["stock_id","time_id"])["time_bucket"].count()
complete = counts[counts == N_TRAIN + N_VAL].index
df = df.set_index(["stock_id","time_id"]).loc[complete].reset_index()

liquidity_map = get_liquidity_map()
selected      = get_selected_stocks(df=df)
liquid_ids    = selected["liquid"]
illiquid_ids  = selected["illiquid"]
all_ids       = selected["all"]

df = df[df["stock_id"].isin(all_ids)].copy()
print(f"  Using {len(all_ids)} stocks: {len(liquid_ids)} liquid + {len(illiquid_ids)} illiquid")


# ── Feature engineering ───────────────────────────────────────────────────────

print("\nEngineering features ...")
train_raw = df[df["time_bucket"] <= N_TRAIN]
val_raw   = df[df["time_bucket"] >  N_TRAIN]

# Use WAP_mean if available, otherwise create a zero column as placeholder
_feat_cols = ["RV", "BidAskSpread_mean"]
if "WAP_mean" in train_raw.columns:
    _feat_cols.append("WAP_mean")
else:
    train_raw = train_raw.copy()
    train_raw["WAP_mean"] = 0.0

features = (train_raw
            .groupby(["stock_id","time_id"])[["RV","BidAskSpread_mean","WAP_mean"]]
            .apply(engineer_features).reset_index())

targets = (val_raw.groupby(["stock_id","time_id"])["RV"]
           .mean().reset_index().rename(columns={"RV":"target_rv"}))

session_spread = (train_raw.groupby(["stock_id","time_id"])["BidAskSpread_mean"]
                  .mean().reset_index().rename(columns={"BidAskSpread_mean":"mean_spread"}))

stock_global = (train_raw.groupby("stock_id")
                .agg(stock_mean_rv    =("RV","mean"),
                     stock_std_rv     =("RV","std"),
                     stock_mean_spread=("BidAskSpread_mean","mean"))
                .reset_index())

data = (features
        .merge(targets,        on=["stock_id","time_id"])
        .merge(session_spread, on=["stock_id","time_id"])
        .merge(stock_global,   on="stock_id"))
data["regime"] = data["stock_id"].map(liquidity_map)

FEATURE_COLS = [c for c in data.columns
                if c not in ["stock_id","time_id","target_rv","mean_spread","fold","regime"]]

print(f"  {data.shape[0]:,} sessions x {len(FEATURE_COLS)} features")


# ── Cross-validation per regime ───────────────────────────────────────────────

all_preds = []
feat_imps = []

for regime_label, regime_ids in [("liquid", liquid_ids), ("illiquid", illiquid_ids)]:
    rdata    = data[data["stock_id"].isin(regime_ids)].copy()
    tids     = np.array(sorted(rdata["time_id"].unique()))

    # ── Time-ordered folds ────────────────────────────────────────────────────
    # Split time_ids into N_FOLDS strictly chronological blocks.
    # Fold k trains on blocks 0..k-1 and validates on block k.
    # This prevents the model from "seeing the future" during training,
    # which was causing artificially good scores in the round-robin approach.
    fold_size = len(tids) // N_FOLDS
    rdata["fold"] = -1
    for k in range(N_FOLDS):
        start = k * fold_size
        end   = (k + 1) * fold_size if k < N_FOLDS - 1 else len(tids)
        block_tids = tids[start:end]
        rdata.loc[rdata["time_id"].isin(block_tids), "fold"] = k

    print(f"\n{N_FOLDS}-fold time-ordered CV — {regime_label.upper()} "
          f"({len(regime_ids)} stocks, {len(rdata):,} sessions)")
    print(f"  time_id range: {tids[0]} → {tids[-1]}  |  ~{fold_size} time_ids per fold")

    for fold in range(N_FOLDS):
        # Train = all earlier folds, val = current fold
        tr = rdata[rdata["fold"] < fold]
        vl = rdata[rdata["fold"] == fold]

        if len(tr) == 0:
            print(f"  Fold {fold+1}/{N_FOLDS} — skipped (no training data yet)")
            continue
        tr = rdata[rdata["fold"] != fold]
        vl = rdata[rdata["fold"] == fold]

        model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
        model.fit(tr[FEATURE_COLS], tr["target_rv"])
        res = vl[["stock_id","time_id","mean_spread","target_rv","regime"]].copy()
        res["pred_vol"] = model.predict(vl[FEATURE_COLS])
        all_preds.append(res)
        feat_imps.append(model.feature_importances_)
        print(f"  Fold {fold+1}/{N_FOLDS} — {len(vl):,} sessions")


# ── Evaluate and save ─────────────────────────────────────────────────────────

results = pd.concat(all_preds, ignore_index=True)
results["QLIKE"] = qlike(results["pred_vol"].values, results["target_rv"].values)
results["MSE"]   = (results["pred_vol"] - results["target_rv"]) ** 2

per_stock = (results
             .groupby(["stock_id","regime"])
             .agg(median_QLIKE=("QLIKE","median"),
                  mean_QLIKE  =("QLIKE","mean"),
                  median_MSE  =("MSE","median"),
                  mean_spread =("mean_spread","mean"),
                  n_sessions  =("time_id","count"))
             .reset_index().sort_values("median_QLIKE"))

feat_imp = (pd.DataFrame({"feature": FEATURE_COLS,
                           "importance": np.mean(feat_imps, axis=0)})
            .sort_values("importance", ascending=False))

results[["stock_id","time_id","regime","mean_spread",
         "pred_vol","target_rv","QLIKE","MSE"]].to_csv(
    os.path.join(LGBM_OUT, "lgbm_eval_results.csv"), index=False)
per_stock.to_csv(os.path.join(LGBM_OUT, "lgbm_per_stock.csv"), index=False)
feat_imp.to_csv( os.path.join(LGBM_OUT, "lgbm_feature_importance.csv"), index=False)
print("\nSaved CSVs.")

print(f"\n{'='*55}")
for regime_label in ["liquid","illiquid"]:
    r = results[results["regime"] == regime_label]
    print(f"  {regime_label.upper():10s}  median QLIKE: {r['QLIKE'].median():.4f} "
          f" median MSE: {r['MSE'].median():.2e}  n: {len(r):,}")
print(f"{'='*55}")


# ── Load other model summaries ────────────────────────────────────────────────

garch_sum   = load_csv("all_stocks_garch_summary.csv")
egarchx_sum = load_csv("all_stocks_egarchx_summary.csv")
har_sum     = load_csv("all_stocks_har_rv_summary.csv")


def make_comparison_plot(lgbm_ps, other_models, regime_label, filename, suptitle):
    """
    Generic grouped bar chart: LGBM vs 2 other models, QLIKE and MSE side by side.
    other_models: list of (label, color, summary_df)
    """
    # Build merged table
    merged = lgbm_ps.copy()
    for label, col, sdf in other_models:
        if not sdf.empty and "stock_id" in sdf.columns and "median_QLIKE" in sdf.columns:
            merged = merged.merge(
                sdf[["stock_id","median_QLIKE","median_MSE"]].rename(
                    columns={"median_QLIKE": f"{label}_QLIKE",
                             "median_MSE":   f"{label}_MSE"}),
                on="stock_id", how="left"
            )

    merged = merged.sort_values("median_QLIKE").reset_index(drop=True)
    x = np.arange(len(merged))
    w = 0.22
    xlabels = [f"Stk {int(s)}" for s in merged["stock_id"]]
    n_models = 1 + len(other_models)
    offsets  = np.linspace(-(n_models-1)*w/2, (n_models-1)*w/2, n_models)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle(suptitle, fontsize=12, fontweight="500")

    for ax, metric, title in [(axes[0],"QLIKE","Median QLIKE  (lower = better)"),
                               (axes[1],"MSE",  "Median MSE   (lower = better)")]:
        ax.set_facecolor("white")

        # LightGBM
        lgbm_col = "median_QLIKE" if metric == "QLIKE" else "median_MSE"
        ax.bar(x + offsets[0], merged[lgbm_col], width=w,
               label="LightGBM", color=C_LGBM, alpha=0.85)

        for i, (label, col, _) in enumerate(other_models):
            colname = f"{label}_{metric}"
            if colname in merged.columns:
                ax.bar(x + offsets[i+1], merged[colname], width=w,
                       label=label, color=col, alpha=0.85)

        for sp in ax.spines.values(): sp.set_color(SPINE); sp.set_linewidth(0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7.5)
        ax.tick_params(labelsize=8, color=SPINE)
        ax.set_title(title, fontsize=10, fontweight="500")
        ax.legend(fontsize=8.5, framealpha=0.9)
        ax.grid(axis="y", color=SPINE, linewidth=0.4, linestyle="--")
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(LGBM_OUT, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


# ── Plot 1: Liquid — LightGBM vs GARCH vs EGARCH-X ───────────────────────────
liq_lgbm = per_stock[per_stock["regime"] == "liquid"].copy()

if not liq_lgbm.empty:
    liq_garch   = garch_sum[garch_sum["liquidity_regime"] == "liquid"]   if not garch_sum.empty   else pd.DataFrame()
    liq_egarchx = egarchx_sum[egarchx_sum["egarchx_ran"] == True]        if not egarchx_sum.empty else pd.DataFrame()

    make_comparison_plot(
        lgbm_ps      = liq_lgbm,
        other_models = [("GARCH(1,1)", C_GARCH, liq_garch),
                        ("EGARCH-X",   C_EX,    liq_egarchx)],
        regime_label = "liquid",
        filename     = "lgbm_01_vs_garch_egarchx.png",
        suptitle     = "LightGBM vs GARCH vs EGARCH-X — Liquid Stocks",
    )


# ── Plot 2: Illiquid — LightGBM vs GARCH vs HAR-RV ───────────────────────────
illiq_lgbm = per_stock[per_stock["regime"] == "illiquid"].copy()

if not illiq_lgbm.empty:
    illiq_garch = garch_sum[garch_sum["liquidity_regime"] == "illiquid"] if not garch_sum.empty else pd.DataFrame()

    make_comparison_plot(
        lgbm_ps      = illiq_lgbm,
        other_models = [("GARCH(1,1)", C_GARCH, illiq_garch),
                        ("HAR-RV",     C_HAR,   har_sum)],
        regime_label = "illiquid",
        filename     = "lgbm_02_vs_garch_har.png",
        suptitle     = "LightGBM vs GARCH vs HAR-RV — Illiquid Stocks",
    )


# ── Plot 3: Feature importance ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("white")
top12 = feat_imp.head(12)
ax.barh(top12["feature"][::-1], top12["importance"][::-1],
        color=C_LGBM, alpha=0.8, edgecolor="white")
style_ax(ax, title="LightGBM Feature Importance — Top 12  (averaged across folds)",
         xlabel="Importance (split count)")
ax.grid(axis="x", color=SPINE, linewidth=0.4, linestyle="--")
ax.grid(axis="y", linestyle="none")
plt.tight_layout()
plt.savefig(os.path.join(LGBM_OUT, "lgbm_03_feature_importance.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved: lgbm_03_feature_importance.png")


# ── Plot 4: Predicted vs Actual — liquid and illiquid side by side ────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("white")
fig.suptitle("LightGBM — Predicted vs Actual Volatility", fontsize=12, fontweight="500")

for ax, regime_label, col in [(axes[0],"liquid",C_EX), (axes[1],"illiquid",C_HAR)]:
    rdf = results[results["regime"] == regime_label].copy()
    p99 = np.percentile(np.concatenate([rdf["target_rv"].values,
                                         rdf["pred_vol"].values]), 99)
    rdf = rdf[(rdf["target_rv"] <= p99) & (rdf["pred_vol"] <= p99)]

    ax.scatter(rdf["target_rv"], rdf["pred_vol"],
               alpha=0.2, s=6, color=col, edgecolors="none")
    lim = max(rdf["target_rv"].max(), rdf["pred_vol"].max()) * 1.06
    ax.plot([0,lim],[0,lim],"--",color="#888780",linewidth=0.9,label="Perfect forecast")
    ax.set_xlim(0,lim); ax.set_ylim(0,lim)
    style_ax(ax,
             title=f"{regime_label.capitalize()}  (median QLIKE: {rdf['QLIKE'].median():.4f})",
             xlabel="Actual volatility", ylabel="Predicted volatility")
    ax.grid(axis="both", color=SPINE, linewidth=0.3, linestyle="--")
    ax.legend(fontsize=8.5)

plt.tight_layout()
plt.savefig(os.path.join(LGBM_OUT, "lgbm_04_pred_vs_actual.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved: lgbm_04_pred_vs_actual.png")

print(f"\n  LightGBM complete. All outputs in: {LGBM_OUT}")