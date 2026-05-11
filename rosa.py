"""
M4 — Phase 3: HAR-RV Model (Heterogeneous Autoregressive Realised Volatility)
Rosa's deliverable

Why HAR-RV for illiquid stocks:
  EGARCH-X blows up when the order book is thin — the log-spread term
  diverges when BAS is wide or zero. HAR-RV has no such term. It models
  RV as a linear combination of daily, weekly, and monthly RV averages,
  which is numerically stable regardless of liquidity conditions.

HAR-RV equation:
  RV(t+1) = c + β_d * RV_d(t) + β_w * RV_w(t) + β_m * RV_m(t) + ε(t)

  Where:
    RV_d = previous bucket's RV           (daily component)
    RV_w = mean RV over last 5 buckets    (weekly component)
    RV_m = mean RV over last 16 buckets   (monthly component = full training window)

This is fit via OLS per time_id — fast, stable, interpretable.

Outputs (into OUTPUT_DIR/stock_{id}/):
  rosa_har_rv_eval_results.csv   — per time_id QLIKE and MSE
  rosa_har_rv_params.csv         — HAR coefficients per time_id
  rosa_har_rv_forecasts.csv      — raw predictions vs actuals
  rosa_har_rv_summary.png        — evaluation plots

Cross-stock:
  all_stocks_har_rv_summary.csv  — aggregated metrics across all illiquid stocks
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_CSV  = r"C:\Users\songj\OneDrive\UNI\Y3SEM1\DATA3888\Project Folder\DATA3888G08\optiver_aggregated.csv"

from config import (get_selected_stocks, get_liquidity_map, filter_time_ids,
                    OUTPUT_DIR, JAMIE_LIQUIDITY_CSV,
                    N_TRAIN, N_VAL, N_STOCKS_PER_REGIME, TIME_ID_KEEP_PCT)

# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Loss functions
# ══════════════════════════════════════════════════════════════════════════════

def qlike(pred, actual):
    if pred <= 0 or actual < 0 or not np.isfinite(pred):
        return np.nan
    pred = max(pred, 1e-10)
    return np.log(pred) + actual / pred


# ══════════════════════════════════════════════════════════════════════════════
#  HAR-RV feature builder
# ══════════════════════════════════════════════════════════════════════════════

def build_har_features(rv_series):
    """
    Given an array of RV values (training window), returns the HAR features
    for forecasting the next period:
      rv_d = last value
      rv_w = mean of last 5
      rv_m = mean of all (up to 16)
    """
    rv   = np.asarray(rv_series, dtype=np.float64)
    rv_d = rv[-1]
    rv_w = rv[-min(5,  len(rv)):].mean()
    rv_m = rv[-min(16, len(rv)):].mean()
    return rv_d, rv_w, rv_m


def fit_har_rv(vol_train_dict, vol_val_dict, time_ids):
    """
    Fit one HAR-RV model per time_id via OLS.
    Returns eval_records, param_records, forecast_records.
    """
    eval_records     = []
    param_records    = []
    forecast_records = []

    for tid in time_ids:
        rv_train = vol_train_dict[tid]["RV"].values
        rv_val   = vol_val_dict[tid]["RV"].values

        if len(rv_train) < 6:   # need enough history for weekly component
            continue

        # Build one feature row (we forecast the val mean from train features)
        rv_d, rv_w, rv_m = build_har_features(rv_train)
        X_pred = np.array([[rv_d, rv_w, rv_m]])

        # For fitting: use rolling windows within training data
        # Each row predicts the next bucket, giving N_TRAIN-1 training pairs
        X_fit, y_fit = [], []
        for t in range(1, len(rv_train)):
            window    = rv_train[:t]
            d = window[-1]
            w = window[-min(5,  len(window)):].mean()
            m = window[-min(16, len(window)):].mean()
            X_fit.append([d, w, m])
            y_fit.append(rv_train[t])

        if len(X_fit) < 3:
            continue

        X_fit = np.array(X_fit)
        y_fit = np.array(y_fit)

        # OLS fit
        try:
            model = LinearRegression(fit_intercept=True)
            model.fit(X_fit, y_fit)
        except Exception:
            continue

        # Predict: target is mean of val buckets
        pred_rv  = float(model.predict(X_pred)[0])
        pred_rv  = max(pred_rv, 1e-8)   # floor to avoid QLIKE blow-up
        actual   = float(rv_val.mean())

        q   = qlike(pred_rv, actual)
        mse = (actual - pred_rv) ** 2

        eval_records.append({
            "time_id":   tid,
            "pred_RV":   pred_rv,
            "actual_RV": actual,
            "QLIKE":     q,
            "MSE":       mse,
        })
        param_records.append({
            "time_id":    tid,
            "intercept":  model.intercept_,
            "beta_daily":   model.coef_[0],
            "beta_weekly":  model.coef_[1],
            "beta_monthly": model.coef_[2],
            "r2":           model.score(X_fit, y_fit),
        })
        forecast_records.append({
            "time_id":    tid,
            "pred_RV":    pred_rv,
            "actual_RV":  actual,
            "rv_d":       rv_d,
            "rv_w":       rv_w,
            "rv_m":       rv_m,
        })

    return (pd.DataFrame(eval_records),
            pd.DataFrame(param_records),
            pd.DataFrame(forecast_records))


# ══════════════════════════════════════════════════════════════════════════════
#  Load data and liquidity labels
# ══════════════════════════════════════════════════════════════════════════════

print("Loading data ...")
df = pd.read_csv(INPUT_CSV)
print(f"  {df['stock_id'].nunique()} stocks found.")

# Stock selection — same 20 illiquid stocks as arma_models.py (via config.py)
selected          = get_selected_stocks(df=df)
illiquid_selected = selected["illiquid"]
liquidity_map     = get_liquidity_map()

print(f"\nSelected {len(illiquid_selected)} illiquid stocks: {illiquid_selected}")
print("HAR-RV runs on these. GARCH baseline runs on the same stocks (Jisu).")


# ══════════════════════════════════════════════════════════════════════════════
#  Main loop
# ══════════════════════════════════════════════════════════════════════════════

all_har_summary = []

for STOCK_ID in illiquid_selected:
    print(f"\n{'='*65}")
    print(f"  HAR-RV: Stock {STOCK_ID}  "
          f"({illiquid_selected.index(STOCK_ID)+1}/{len(illiquid_selected)})")
    print(f"{'='*65}")

    stock_out = os.path.join(OUTPUT_DIR, f"stock_{STOCK_ID}")
    os.makedirs(stock_out, exist_ok=True)

    stock_df = df[df["stock_id"] == STOCK_ID].copy()
    stock_df = stock_df.sort_values(["time_id", "time_bucket"]).reset_index(drop=True)

    vol_train, vol_val = {}, {}
    for tid in sorted(stock_df["time_id"].unique()):
        buckets = (stock_df[stock_df["time_id"] == tid]
                   .sort_values("time_bucket").reset_index(drop=True))
        if len(buckets) < N_TRAIN + N_VAL:
            continue
        vol_train[tid] = buckets.iloc[:N_TRAIN].copy()
        vol_val[tid]   = buckets.iloc[N_TRAIN : N_TRAIN + N_VAL].copy()

    time_IDs = list(vol_train.keys())
    if not time_IDs:
        print(f"  No complete time_ids — skipping.")
        continue

    # Keep most illiquid time_ids (via config.py — same logic as arma_models.py)
    time_IDs  = filter_time_ids(vol_train, time_IDs, "illiquid")
    vol_train = {tid: vol_train[tid] for tid in time_IDs}
    vol_val   = {tid: vol_val[tid]   for tid in time_IDs if tid in vol_val}
    print(f"  {len(time_IDs)} most illiquid time_ids selected")

    # Fit HAR-RV
    eval_df, param_df, forecast_df = fit_har_rv(vol_train, vol_val, time_IDs)

    if eval_df.empty:
        print(f"  No valid HAR-RV forecasts — skipping.")
        continue

    # Save per-stock outputs
    eval_df.to_csv(    os.path.join(stock_out, "rosa_har_rv_eval_results.csv"),  index=False)
    param_df.to_csv(   os.path.join(stock_out, "rosa_har_rv_params.csv"),         index=False)
    forecast_df.to_csv(os.path.join(stock_out, "rosa_har_rv_forecasts.csv"),      index=False)

    per_tid = eval_df.groupby("time_id")[["QLIKE", "MSE"]].mean()
    med_q   = per_tid["QLIKE"].median()
    med_mse = per_tid["MSE"].median()

    print(f"  n time_ids evaluated: {len(per_tid)}")
    print(f"  Median QLIKE:  {med_q:.4f}")
    print(f"  Median MSE:    {med_mse:.2e}")
    print(f"  Mean beta_d:   {param_df['beta_daily'].mean():.4f}")
    print(f"  Mean beta_w:   {param_df['beta_weekly'].mean():.4f}")
    print(f"  Mean beta_m:   {param_df['beta_monthly'].mean():.4f}")

    # ── Per-stock plot (lightweight) ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"HAR-RV — Stock {STOCK_ID} (illiquid)", fontsize=11, fontweight="500")

    C = "#D85A30"
    for ax in axes:
        ax.set_facecolor("white")
        for sp in ax.spines.values(): sp.set_color("#D3D1C7"); sp.set_linewidth(0.6)

    # Predicted vs actual
    axes[0].scatter(forecast_df["actual_RV"], forecast_df["pred_RV"],
                    alpha=0.45, s=14, color=C, edgecolors="none")
    lim = max(forecast_df["actual_RV"].max(), forecast_df["pred_RV"].max()) * 1.06
    axes[0].plot([0, lim], [0, lim], "k--", linewidth=0.9, alpha=0.6)
    axes[0].set_xlabel("Actual RV", fontsize=9); axes[0].set_ylabel("Predicted RV", fontsize=9)
    axes[0].set_title("Predicted vs Actual RV", fontsize=10)
    axes[0].grid(color="#D3D1C7", linewidth=0.4, linestyle="--"); axes[0].set_axisbelow(True)

    # QLIKE by time_id (sorted)
    qlike_sorted = per_tid["QLIKE"].sort_values().values
    axes[1].bar(range(len(qlike_sorted)), qlike_sorted, color=C, alpha=0.7, width=0.85)
    axes[1].axhline(med_q, color="black", linestyle="--", linewidth=0.9,
                    label=f"Median {med_q:.3f}")
    axes[1].set_xlabel("time_id (sorted by QLIKE)", fontsize=9)
    axes[1].set_ylabel("QLIKE", fontsize=9)
    axes[1].set_title("QLIKE per time_id", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", color="#D3D1C7", linewidth=0.4, linestyle="--"); axes[1].set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(stock_out, "rosa_har_rv_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()

    all_har_summary.append({
        "stock_id":       STOCK_ID,
        "liquidity_regime": "illiquid",
        "n_time_ids":     len(per_tid),
        "median_QLIKE":   med_q,
        "median_MSE":     med_mse,
        "mean_beta_d":    param_df["beta_daily"].mean(),
        "mean_beta_w":    param_df["beta_weekly"].mean(),
        "mean_beta_m":    param_df["beta_monthly"].mean(),
        "mean_r2":        param_df["r2"].mean(),
    })


# ══════════════════════════════════════════════════════════════════════════════
#  Aggregate summary
# ══════════════════════════════════════════════════════════════════════════════

if all_har_summary:
    summary_df = pd.DataFrame(all_har_summary).sort_values("stock_id")
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "all_stocks_har_rv_summary.csv"), index=False)
    print(f"\nSaved: all_stocks_har_rv_summary.csv")
    print(f"\n{'='*60}")
    print(f"  HAR-RV across {len(all_har_summary)} illiquid stocks:")
    print(f"  Overall median QLIKE: {summary_df['median_QLIKE'].median():.4f}")
    print(f"  Overall median MSE:   {summary_df['median_MSE'].median():.2e}")
    print(f"{'='*60}")

    # ── Aggregate plot 1: QLIKE ranking across all illiquid stocks ────────────
    C = "#D85A30"; SPINE = "#D3D1C7"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle("HAR-RV — Performance Across All Illiquid Stocks",
                 fontsize=12, fontweight="500")

    ranked = summary_df.sort_values("median_QLIKE").reset_index(drop=True)
    labels = [f"Stock {int(r['stock_id'])}" for _, r in ranked.iterrows()]

    # QLIKE bar
    ax = axes[0]
    ax.set_facecolor("white")
    bars = ax.barh(labels, ranked["median_QLIKE"], color=C, alpha=0.75, height=0.65)
    ax.axvline(ranked["median_QLIKE"].median(), color="#444441", linestyle="--",
               linewidth=0.9, label=f"Median {ranked['median_QLIKE'].median():.3f}")
    for sp in ax.spines.values(): sp.set_color(SPINE); sp.set_linewidth(0.6)
    ax.tick_params(labelsize=8, color=SPINE)
    ax.set_xlabel("Median QLIKE (lower = better)", fontsize=9)
    ax.set_title("QLIKE by Stock", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="x", color=SPINE, linewidth=0.4, linestyle="--"); ax.set_axisbelow(True)

    # HAR coefficient breakdown — mean β_d, β_w, β_m per stock
    ax2 = axes[1]
    ax2.set_facecolor("white")
    x = np.arange(len(ranked))
    w = 0.25
    ax2.bar(x - w, ranked["mean_beta_d"], width=w, label="β daily",  color="#D85A30", alpha=0.8)
    ax2.bar(x,     ranked["mean_beta_w"], width=w, label="β weekly", color="#F0997B", alpha=0.8)
    ax2.bar(x + w, ranked["mean_beta_m"], width=w, label="β monthly",color="#FAECE7", alpha=0.9,
            edgecolor="#D85A30", linewidth=0.6)
    ax2.axhline(0, color=SPINE, linewidth=0.7)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    for sp in ax2.spines.values(): sp.set_color(SPINE); sp.set_linewidth(0.6)
    ax2.tick_params(labelsize=8, color=SPINE)
    ax2.set_ylabel("Coefficient value", fontsize=9)
    ax2.set_title("HAR Coefficients by Stock", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", color=SPINE, linewidth=0.4, linestyle="--"); ax2.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rosa_01_har_rv_performance.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: rosa_01_har_rv_performance.png")

    # ── Aggregate plot 2: Predicted vs actual — all stocks combined ───────────
    all_forecast_frames = []
    for sid in illiquid_selected:
        p = os.path.join(OUTPUT_DIR, f"stock_{sid}", "rosa_har_rv_forecasts.csv")
        if os.path.exists(p):
            df_ = pd.read_csv(p); df_["stock_id"] = sid
            all_forecast_frames.append(df_)

    if all_forecast_frames:
        all_fc = pd.concat(all_forecast_frames, ignore_index=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor("white")
        fig.suptitle("HAR-RV — Forecast Quality (All Illiquid Stocks Combined)",
                     fontsize=12, fontweight="500")

        # Scatter
        ax = axes[0]; ax.set_facecolor("white")
        ax.scatter(all_fc["actual_RV"], all_fc["pred_RV"],
                   alpha=0.25, s=8, color=C, edgecolors="none")
        lim = np.percentile(np.concatenate([all_fc["actual_RV"], all_fc["pred_RV"]]), 99) * 1.1
        ax.plot([0, lim], [0, lim], "k--", linewidth=0.9, alpha=0.6, label="Perfect forecast")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        for sp in ax.spines.values(): sp.set_color(SPINE); sp.set_linewidth(0.6)
        ax.set_xlabel("Actual RV", fontsize=9); ax.set_ylabel("Predicted RV", fontsize=9)
        ax.set_title("Predicted vs Actual", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(color=SPINE, linewidth=0.3, linestyle="--"); ax.set_axisbelow(True)

        # Residuals
        residuals = all_fc["pred_RV"] - all_fc["actual_RV"]
        ax2 = axes[1]; ax2.set_facecolor("white")
        ax2.hist(residuals.clip(-0.01, 0.01), bins=40, color=C, alpha=0.75, edgecolor="white")
        ax2.axvline(0, color="#444441", linewidth=0.9, linestyle="--", label="Zero error")
        ax2.axvline(residuals.mean(), color="#1D9E75", linewidth=0.9, linestyle="--",
                    label=f"Mean {residuals.mean():.5f}")
        for sp in ax2.spines.values(): sp.set_color(SPINE); sp.set_linewidth(0.6)
        ax2.set_xlabel("Forecast error (pred − actual)", fontsize=9)
        ax2.set_ylabel("Count", fontsize=9)
        ax2.set_title("Residual Distribution", fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", color=SPINE, linewidth=0.3, linestyle="--"); ax2.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "rosa_02_har_rv_residuals.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: rosa_02_har_rv_residuals.png")

print(f"\n✓ rosa_har_rv.py complete.")