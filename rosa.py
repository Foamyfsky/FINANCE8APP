"""
M4 — Phase 3: HAR-RV + WLS Volatility Models (Rosa)
DATA3888 Group 8

Models:
  HAR-RV (Heterogeneous Autoregressive Realised Volatility)
    RV(t+1) = c + β_d·RV_d(t) + β_w·RV_w(t) + β_m·RV_m(t) + ε(t)
    Fit via OLS per time_id. Numerically stable for any liquidity regime.

  WLS-HAR-RV (Weighted Least Squares)
    Same equation, but recent buckets receive higher weight (exponential decay).
    Better than OLS when volatility is heteroskedastic — more weight on recent info.

Why HAR-RV / WLS for illiquid and mixed stocks:
  EGARCH-X can diverge when the order book is thin (wide BAS → large spread term).
  HAR-RV and WLS have no spread term — they are purely autoregressive and
  numerically stable regardless of liquidity conditions.

Liquidity profiling:
  Rosa's bucket-level approach labels each (stock_id, time_id, bucket) as
  liquid/illiquid/mixed using:
    liquid   = BAS ≤ q33 AND log_activity ≥ q66
    illiquid = BAS ≥ q66 AND log_activity ≤ q33
    mixed    = everything else
  A stock is liquid if ≥40% of its buckets are liquid, illiquid if ≥40%
  are illiquid, otherwise mixed.

Coverage:
  Illiquid stocks  → HAR-RV + WLS (primary)
  Mixed stocks     → HAR-RV + WLS (both run; winner reported per stock)
  Liquid stocks    → skipped (EGARCH-X / Jisu handles these)

Outputs:
  Per-stock (m4_outputs/stock_{id}/):
    rosa_har_rv_eval_results.csv    — QLIKE + MSE per time_id (OLS)
    rosa_wls_eval_results.csv       — QLIKE + MSE per time_id (WLS)
    rosa_har_rv_params.csv          — HAR coefficients (OLS)
    rosa_wls_params.csv             — HAR coefficients (WLS)
    rosa_har_rv_forecasts.csv       — raw predictions vs actuals (OLS)
    rosa_model_comparison.png       — OLS vs WLS comparison plot
    rosa_har_rv_summary.png         — per-stock evaluation plot

  Aggregate (m4_outputs/):
    har_rv_stock_liquidity_profile.csv  — Rosa's liquidity profile (all stocks)
    har_rv_top10_liquid_stocks.csv
    har_rv_top10_illiquid_stocks.csv
    all_stocks_har_rv_summary.csv       — HAR-RV + WLS metrics per stock
    rosa_01_har_rv_performance.png
    rosa_02_har_rv_residuals.png
    rosa_03_ols_vs_wls.png              — OLS vs WLS head-to-head across stocks
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_CSV = r"C:\Users\songj\OneDrive\UNI\Y3SEM1\DATA3888\Project Folder\DATA3888G08\optiver_aggregated.csv"

from config import (get_selected_stocks, get_liquidity_map, filter_time_ids,
                    OUTPUT_DIR, N_TRAIN, N_VAL, N_STOCKS_PER_REGIME)

os.makedirs(OUTPUT_DIR, exist_ok=True)

LIQUIDITY_PROFILE_PATH = os.path.join(OUTPUT_DIR, "har_rv_stock_liquidity_profile.csv")

# WLS decay: weight for bucket i from end = exp(-WLS_DECAY * (n - 1 - i))
# λ = 0.05 → recent buckets ~2.2x more weight than oldest; λ = 0.1 → ~4.5x
WLS_DECAY = 0.05


# ══════════════════════════════════════════════════════════════════════════════
#  Step 0 — Generate Rosa's liquidity profile
#
#  Must run BEFORE config.get_liquidity_map() / get_selected_stocks() because
#  config.py looks for this file as its primary liquidity source.
# ══════════════════════════════════════════════════════════════════════════════

def build_liquidity_profile(df):
    """
    Build Rosa's bucket-level liquidity profile and save to OUTPUT_DIR.

    Uses:
      liquid   = BAS ≤ q33 AND log(1 + 1/BAS) ≥ q66
      illiquid = BAS ≥ q66 AND log(1 + 1/BAS) ≤ q33
      mixed    = everything else

    Stock regime determined by 40% threshold on bucket counts.
    Saves:
      har_rv_stock_liquidity_profile.csv
      har_rv_top10_liquid_stocks.csv
      har_rv_top10_illiquid_stocks.csv
    Returns the stock-level profile DataFrame.
    """
    print("\n── Building Rosa's liquidity profile ───────────────────────────────")

    liq = df[["stock_id", "time_id", "time_bucket", "BidAskSpread_mean", "RV"]].copy()
    liq = liq.rename(columns={"BidAskSpread_mean": "bas", "RV": "rv"})
    liq["time_bucket"] = liq["time_bucket"].astype(int)

    # Use buckets 1–16 (training window, same as HAR-RV)
    liq = liq[liq["time_bucket"].between(1, 16)].copy()

    # Inverse-spread as activity proxy (WAP not available in aggregated file)
    liq["inv_spread"]    = 1.0 / (liq["bas"] + 1e-6)
    liq["log_activity"]  = np.log1p(liq["inv_spread"])

    # Global quantile thresholds
    bas_q33  = liq["bas"].quantile(0.33)
    bas_q66  = liq["bas"].quantile(0.66)
    act_q33  = liq["log_activity"].quantile(0.33)
    act_q66  = liq["log_activity"].quantile(0.66)

    print(f"  BAS    q33={bas_q33:.6f}  q66={bas_q66:.6f}")
    print(f"  logAct q33={act_q33:.4f}  q66={act_q66:.4f}")

    # Bucket-level labels
    conditions = [
        (liq["bas"] <= bas_q33) & (liq["log_activity"] >= act_q66),
        (liq["bas"] >= bas_q66) & (liq["log_activity"] <= act_q33),
    ]
    liq["bucket_liquidity"] = np.select(conditions, ["liquid", "illiquid"], default="mixed")

    # Aggregate to stock level
    profile = (
        liq.groupby("stock_id")
        .agg(
            median_bas          = ("bas",              "median"),
            mean_bas            = ("bas",              "mean"),
            median_log_activity = ("log_activity",     "median"),
            median_rv           = ("rv",               "median"),
            mean_rv             = ("rv",               "mean"),
            rv_std              = ("rv",               "std"),
            liquid_pct          = ("bucket_liquidity", lambda x: (x == "liquid").mean()),
            illiquid_pct        = ("bucket_liquidity", lambda x: (x == "illiquid").mean()),
            mixed_pct           = ("bucket_liquidity", lambda x: (x == "mixed").mean()),
            n_buckets           = ("rv",               "count"),
        )
        .reset_index()
    )

    def _stock_regime(row):
        if row["liquid_pct"]   >= 0.40: return "liquid"
        if row["illiquid_pct"] >= 0.40: return "illiquid"
        return "mixed"

    profile["stock_regime"] = profile.apply(_stock_regime, axis=1)
    profile["recommended_model"] = np.where(
        profile["stock_regime"] == "liquid", "EGARCH-X", "HAR-RV / WLS"
    )

    # Sort most liquid first
    profile = profile.sort_values(
        ["liquid_pct", "median_bas"], ascending=[False, True]
    ).reset_index(drop=True)

    profile.to_csv(LIQUIDITY_PROFILE_PATH, index=False)
    print(f"  Saved: {LIQUIDITY_PROFILE_PATH}")
    print(f"  Regime counts: {profile['stock_regime'].value_counts().to_dict()}")

    return profile


# ══════════════════════════════════════════════════════════════════════════════
#  Loss function
# ══════════════════════════════════════════════════════════════════════════════

def qlike(pred, actual):
    """QLIKE = log(pred) + actual/pred  (lower = better). Variance form."""
    if pred <= 0 or actual < 0 or not np.isfinite(pred):
        return np.nan
    pred = max(pred, 1e-10)
    return np.log(pred) + actual / pred


# ══════════════════════════════════════════════════════════════════════════════
#  HAR-RV feature builder
# ══════════════════════════════════════════════════════════════════════════════

def build_har_features(rv_series):
    """
    Returns (rv_d, rv_w, rv_m) from a training RV array:
      rv_d = last value          (daily component)
      rv_w = mean of last 5     (weekly component)
      rv_m = mean of all ≤16    (monthly component)
    """
    rv   = np.asarray(rv_series, dtype=np.float64)
    rv_d = rv[-1]
    rv_w = rv[-min(5,  len(rv)):].mean()
    rv_m = rv[-min(16, len(rv)):].mean()
    return rv_d, rv_w, rv_m


# ══════════════════════════════════════════════════════════════════════════════
#  HAR-RV fitting — OLS and WLS
# ══════════════════════════════════════════════════════════════════════════════

def fit_har_rv(vol_train_dict, vol_val_dict, time_ids, decay=0.0):
    """
    Fit HAR-RV per time_id via OLS (decay=0) or WLS (decay>0).

    WLS: sample weight for training pair at position t (0=oldest) is
         exp(decay * (t - (T-1))), so the most recent pair has weight 1.

    Returns (eval_df, param_df, forecast_df).
    """
    use_wls      = decay > 0
    model_label  = f"WLS-HAR-RV (λ={decay})" if use_wls else "HAR-RV (OLS)"

    eval_records     = []
    param_records    = []
    forecast_records = []

    for tid in time_ids:
        rv_train = vol_train_dict[tid]["RV"].values
        rv_val   = vol_val_dict[tid]["RV"].values

        if len(rv_train) < 6:
            continue

        # Forecast features from end of training window
        rv_d, rv_w, rv_m = build_har_features(rv_train)
        X_pred = np.array([[rv_d, rv_w, rv_m]])

        # Build rolling training pairs
        X_fit, y_fit = [], []
        for t in range(1, len(rv_train)):
            window = rv_train[:t]
            d = window[-1]
            w = window[-min(5,  len(window)):].mean()
            m = window[-min(16, len(window)):].mean()
            X_fit.append([d, w, m])
            y_fit.append(rv_train[t])

        if len(X_fit) < 3:
            continue

        X_fit = np.array(X_fit)
        y_fit = np.array(y_fit)

        # WLS weights: exp(λ * (i - (T-1))) where i is position index
        if use_wls:
            n     = len(y_fit)
            idxs  = np.arange(n)
            w_arr = np.exp(decay * (idxs - (n - 1)))   # most recent → weight 1
        else:
            w_arr = None

        try:
            model = LinearRegression(fit_intercept=True)
            model.fit(X_fit, y_fit, sample_weight=w_arr)
        except Exception:
            continue

        pred_rv = float(model.predict(X_pred)[0])
        pred_rv = max(pred_rv, 1e-8)
        actual  = float(rv_val.mean())

        q   = qlike(pred_rv, actual)
        mse = (actual - pred_rv) ** 2
        r2  = model.score(X_fit, y_fit, sample_weight=w_arr)

        eval_records.append({
            "time_id":   tid,
            "pred_RV":   pred_rv,
            "actual_RV": actual,
            "QLIKE":     q,
            "MSE":       mse,
            "model":     model_label,
        })
        param_records.append({
            "time_id":      tid,
            "intercept":    model.intercept_,
            "beta_daily":   model.coef_[0],
            "beta_weekly":  model.coef_[1],
            "beta_monthly": model.coef_[2],
            "r2":           r2,
            "model":        model_label,
        })
        forecast_records.append({
            "time_id":   tid,
            "pred_RV":   pred_rv,
            "actual_RV": actual,
            "rv_d":      rv_d,
            "rv_w":      rv_w,
            "rv_m":      rv_m,
            "model":     model_label,
        })

    return (pd.DataFrame(eval_records),
            pd.DataFrame(param_records),
            pd.DataFrame(forecast_records))


# ══════════════════════════════════════════════════════════════════════════════
#  Load data + generate liquidity profile
# ══════════════════════════════════════════════════════════════════════════════

print("Loading data ...")
df = pd.read_csv(INPUT_CSV)
print(f"  {df['stock_id'].nunique()} stocks found.")

# Build Rosa's liquidity profile FIRST so config.py can use it
liquidity_profile = build_liquidity_profile(df)

# Read stock selection from CSV — do NOT pass df so auto-selection is blocked.
# selected_stocks.csv is the sole source of truth for which stocks are processed.
selected          = get_selected_stocks()
liquid_selected   = selected["liquid"]
illiquid_selected = selected["illiquid"]
mixed_selected    = selected.get("mixed", [])
rosa_stocks       = selected["all"]   # all 60 — consistent with arma_jisu.py

liquidity_map = get_liquidity_map()

print(f"\n── Stock coverage ───────────────────────────────────────────────────")
print(f"  Liquid   ({len(liquid_selected)}):   HAR-RV + WLS (baseline; EGARCH-X is recommended)")
print(f"  Illiquid ({len(illiquid_selected)}): HAR-RV + WLS (primary regime)")
print(f"  Mixed    ({len(mixed_selected)}): HAR-RV + WLS (both models, winner reported)")
print(f"  Total Rosa stocks: {len(rosa_stocks)} (same 60 as arma_jisu.py)")


# ══════════════════════════════════════════════════════════════════════════════
#  Main loop
# ══════════════════════════════════════════════════════════════════════════════

C_OLS  = "#D85A30"   # orange — HAR-RV OLS
C_WLS  = "#1D9E75"   # green  — WLS
SPINE  = "#D3D1C7"

all_har_summary = []

for STOCK_ID in rosa_stocks:
    regime = liquidity_map.get(STOCK_ID, "illiquid")
    print(f"\n{'='*65}")
    print(f"  HAR-RV + WLS: Stock {STOCK_ID}  "
          f"({rosa_stocks.index(STOCK_ID)+1}/{len(rosa_stocks)})  [{regime}]")
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
        print("  No complete time_ids — skipping.")
        continue

    n_total  = len(time_IDs)
    time_IDs = filter_time_ids(vol_train, time_IDs, regime)
    vol_train = {tid: vol_train[tid] for tid in time_IDs}
    vol_val   = {tid: vol_val[tid]   for tid in time_IDs if tid in vol_val}
    print(f"  time_ids: {n_total} total → {len(time_IDs)} kept ({regime})")

    # ── Fit OLS ───────────────────────────────────────────────────────────────
    ols_eval, ols_params, ols_fc = fit_har_rv(vol_train, vol_val, time_IDs, decay=0.0)

    # ── Fit WLS ───────────────────────────────────────────────────────────────
    wls_eval, wls_params, wls_fc = fit_har_rv(vol_train, vol_val, time_IDs, decay=WLS_DECAY)

    if ols_eval.empty and wls_eval.empty:
        print("  No valid forecasts — skipping.")
        continue

    # ── Save per-stock outputs ────────────────────────────────────────────────
    ols_eval.to_csv(  os.path.join(stock_out, "rosa_har_rv_eval_results.csv"), index=False)
    wls_eval.to_csv(  os.path.join(stock_out, "rosa_wls_eval_results.csv"),    index=False)
    ols_params.to_csv(os.path.join(stock_out, "rosa_har_rv_params.csv"),       index=False)
    wls_params.to_csv(os.path.join(stock_out, "rosa_wls_params.csv"),          index=False)
    ols_fc.to_csv(    os.path.join(stock_out, "rosa_har_rv_forecasts.csv"),    index=False)

    # ── Metrics ───────────────────────────────────────────────────────────────
    def _metrics(eval_df, label):
        if eval_df.empty:
            return np.nan, np.nan, 0
        pt = eval_df.groupby("time_id")[["QLIKE","MSE"]].mean()
        print(f"  [{label}] n={len(pt)}  "
              f"Median QLIKE={pt['QLIKE'].median():.4f}  "
              f"Median MSE={pt['MSE'].median():.2e}")
        return pt["QLIKE"].median(), pt["MSE"].median(), len(pt)

    ols_q, ols_mse, ols_n = _metrics(ols_eval, "OLS HAR-RV")
    wls_q, wls_mse, wls_n = _metrics(wls_eval, "WLS HAR-RV")

    winner = "WLS" if (np.isfinite(wls_q) and np.isfinite(ols_q) and wls_q < ols_q) \
             else "OLS"
    print(f"  → {winner} wins on QLIKE for Stock {STOCK_ID}")

    # ── Per-stock comparison plot ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"HAR-RV vs WLS — Stock {STOCK_ID} ({regime})",
                 fontsize=11, fontweight="500")

    def _ax_style(ax):
        ax.set_facecolor("white")
        for sp in ax.spines.values(): sp.set_color(SPINE); sp.set_linewidth(0.6)
        ax.grid(color=SPINE, linewidth=0.4, linestyle="--")
        ax.set_axisbelow(True)

    # Scatter: OLS predicted vs actual
    if not ols_fc.empty:
        lim = max(ols_fc["actual_RV"].max(), ols_fc["pred_RV"].max()) * 1.06
        axes[0].scatter(ols_fc["actual_RV"], ols_fc["pred_RV"],
                        alpha=0.4, s=10, color=C_OLS, edgecolors="none")
        if not wls_fc.empty:
            axes[0].scatter(wls_fc["actual_RV"], wls_fc["pred_RV"],
                            alpha=0.3, s=8, color=C_WLS, edgecolors="none")
        axes[0].plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.6)
        axes[0].set_xlabel("Actual RV", fontsize=9)
        axes[0].set_ylabel("Predicted RV", fontsize=9)
        axes[0].set_title("Predicted vs Actual", fontsize=10)
        from matplotlib.patches import Patch
        axes[0].legend(handles=[Patch(color=C_OLS, label="OLS"),
                                 Patch(color=C_WLS, label="WLS")], fontsize=8)
    _ax_style(axes[0])

    # QLIKE distribution boxplot
    data_bp, labs_bp = [], []
    if not ols_eval.empty:
        data_bp.append(ols_eval.groupby("time_id")["QLIKE"].mean().values)
        labs_bp.append("OLS")
    if not wls_eval.empty:
        data_bp.append(wls_eval.groupby("time_id")["QLIKE"].mean().values)
        labs_bp.append("WLS")
    if data_bp:
        bp = axes[1].boxplot(data_bp, patch_artist=True, labels=labs_bp, widths=0.45,
                             medianprops=dict(color="white", linewidth=2),
                             whiskerprops=dict(color=SPINE, linewidth=0.8),
                             capprops=dict(color=SPINE, linewidth=0.8),
                             flierprops=dict(marker="o", markersize=3,
                                             markerfacecolor=SPINE, alpha=0.4))
        for box, col in zip(bp["boxes"], [C_OLS, C_WLS]):
            box.set_facecolor(col); box.set_alpha(0.8)
        axes[1].set_title("QLIKE distribution", fontsize=10)
        axes[1].set_ylabel("QLIKE per time_id", fontsize=9)
    _ax_style(axes[1])

    # HAR coefficients comparison
    if not ols_params.empty and not wls_params.empty:
        cats = ["β daily", "β weekly", "β monthly"]
        ols_means = [ols_params["beta_daily"].mean(),
                     ols_params["beta_weekly"].mean(),
                     ols_params["beta_monthly"].mean()]
        wls_means = [wls_params["beta_daily"].mean(),
                     wls_params["beta_weekly"].mean(),
                     wls_params["beta_monthly"].mean()]
        x2 = np.arange(3); w2 = 0.35
        axes[2].bar(x2 - w2/2, ols_means, width=w2, color=C_OLS, alpha=0.8, label="OLS")
        axes[2].bar(x2 + w2/2, wls_means, width=w2, color=C_WLS, alpha=0.8, label="WLS")
        axes[2].axhline(0, color=SPINE, linewidth=0.7)
        axes[2].set_xticks(x2); axes[2].set_xticklabels(cats, fontsize=9)
        axes[2].set_title("HAR Coefficients (mean)", fontsize=10)
        axes[2].legend(fontsize=8)
    _ax_style(axes[2])

    plt.tight_layout()
    plt.savefig(os.path.join(stock_out, "rosa_model_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # Legacy summary plot (simple, for pipeline compatibility)
    if not ols_fc.empty:
        ols_per_tid = ols_eval.groupby("time_id")[["QLIKE","MSE"]].mean()
        fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
        fig2.patch.set_facecolor("white")
        fig2.suptitle(f"HAR-RV — Stock {STOCK_ID} ({regime})", fontsize=11, fontweight="500")
        for ax in axes2: _ax_style(ax)

        axes2[0].scatter(ols_fc["actual_RV"], ols_fc["pred_RV"],
                         alpha=0.45, s=14, color=C_OLS, edgecolors="none")
        lim2 = max(ols_fc["actual_RV"].max(), ols_fc["pred_RV"].max()) * 1.06
        axes2[0].plot([0, lim2], [0, lim2], "k--", linewidth=0.9, alpha=0.6)
        axes2[0].set_xlabel("Actual RV", fontsize=9); axes2[0].set_ylabel("Predicted RV", fontsize=9)
        axes2[0].set_title("Predicted vs Actual RV", fontsize=10)

        qlike_sorted = ols_per_tid["QLIKE"].sort_values().values
        axes2[1].bar(range(len(qlike_sorted)), qlike_sorted, color=C_OLS, alpha=0.7, width=0.85)
        axes2[1].axhline(ols_q, color="black", linestyle="--", linewidth=0.9,
                         label=f"Median {ols_q:.3f}")
        axes2[1].set_xlabel("time_id (sorted)", fontsize=9)
        axes2[1].set_ylabel("QLIKE", fontsize=9)
        axes2[1].set_title("QLIKE per time_id (OLS)", fontsize=10)
        axes2[1].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(stock_out, "rosa_har_rv_summary.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    all_har_summary.append({
        "stock_id":         STOCK_ID,
        "liquidity_regime": regime,
        "n_time_ids":       max(ols_n, wls_n),
        # OLS metrics
        "ols_median_QLIKE": ols_q,
        "ols_median_MSE":   ols_mse,
        "ols_mean_beta_d":  ols_params["beta_daily"].mean()   if not ols_params.empty else np.nan,
        "ols_mean_beta_w":  ols_params["beta_weekly"].mean()  if not ols_params.empty else np.nan,
        "ols_mean_beta_m":  ols_params["beta_monthly"].mean() if not ols_params.empty else np.nan,
        "ols_mean_r2":      ols_params["r2"].mean()           if not ols_params.empty else np.nan,
        # WLS metrics
        "wls_median_QLIKE": wls_q,
        "wls_median_MSE":   wls_mse,
        "wls_mean_beta_d":  wls_params["beta_daily"].mean()   if not wls_params.empty else np.nan,
        "wls_mean_beta_w":  wls_params["beta_weekly"].mean()  if not wls_params.empty else np.nan,
        "wls_mean_beta_m":  wls_params["beta_monthly"].mean() if not wls_params.empty else np.nan,
        "wls_mean_r2":      wls_params["r2"].mean()           if not wls_params.empty else np.nan,
        # Winner
        "winner":           winner,
        # Legacy columns (backward compat with other scripts)
        "median_QLIKE":     min(v for v in [ols_q, wls_q] if np.isfinite(v)) if any(np.isfinite(v) for v in [ols_q, wls_q]) else np.nan,
        "median_MSE":       min(v for v in [ols_mse, wls_mse] if np.isfinite(v)) if any(np.isfinite(v) for v in [ols_mse, wls_mse]) else np.nan,
    })


# ══════════════════════════════════════════════════════════════════════════════
#  Aggregate summary
# ══════════════════════════════════════════════════════════════════════════════

if not all_har_summary:
    print("\nNo HAR-RV results to summarise.")
else:
    summary_df = pd.DataFrame(all_har_summary).sort_values("stock_id")
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "all_stocks_har_rv_summary.csv"), index=False)
    print(f"\nSaved: all_stocks_har_rv_summary.csv")

    illiq_s = summary_df[summary_df["liquidity_regime"] == "illiquid"]
    mixed_s  = summary_df[summary_df["liquidity_regime"] == "mixed"]

    print(f"\n{'='*60}")
    print(f"  HAR-RV + WLS across {len(summary_df)} stocks:")
    if not illiq_s.empty:
        print(f"  Illiquid ({len(illiq_s)}) OLS median QLIKE: {illiq_s['ols_median_QLIKE'].median():.4f}")
        print(f"  Illiquid ({len(illiq_s)}) WLS median QLIKE: {illiq_s['wls_median_QLIKE'].median():.4f}")
    if not mixed_s.empty:
        print(f"  Mixed    ({len(mixed_s)}) OLS median QLIKE: {mixed_s['ols_median_QLIKE'].median():.4f}")
        print(f"  Mixed    ({len(mixed_s)}) WLS median QLIKE: {mixed_s['wls_median_QLIKE'].median():.4f}")

    n_wls_wins = (summary_df["winner"] == "WLS").sum()
    print(f"  WLS wins on QLIKE: {n_wls_wins}/{len(summary_df)} stocks")
    print(f"{'='*60}\n")

    # ── Aggregate plot 1: QLIKE per stock (OLS vs WLS bars) ──────────────────
    fig, ax = plt.subplots(figsize=(max(10, len(summary_df)*0.65), 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for sp in ax.spines.values(): sp.set_color(SPINE); sp.set_linewidth(0.6)
    x = np.arange(len(summary_df))
    ax.bar(x - 0.2, summary_df["ols_median_QLIKE"], 0.38,
           label="OLS", color=C_OLS, alpha=0.85)
    ax.bar(x + 0.2, summary_df["wls_median_QLIKE"], 0.38,
           label="WLS", color=C_WLS, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["stock_id"].astype(str),
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Median QLIKE", fontsize=10)
    ax.set_title("HAR-RV OLS vs WLS — QLIKE per Stock", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", color=SPINE, linewidth=0.4, linestyle="--"); ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rosa_01_qlike_by_stock.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Aggregate plot 2: QLIKE by regime ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle("HAR-RV Performance by Liquidity Regime",
                 fontsize=12, fontweight="bold")
    regime_colors = {"illiquid": "#F44336", "mixed": "#9C27B0"}
    for i, (metric, label) in enumerate([("ols_median_QLIKE","OLS"),
                                          ("wls_median_QLIKE","WLS")]):
        ax = axes[i]
        ax.set_facecolor("white")
        for sp in ax.spines.values(): sp.set_color(SPINE); sp.set_linewidth(0.6)
        groups  = summary_df.groupby("liquidity_regime")[metric].apply(list)
        labels_ = list(groups.index)
        data_   = [groups[l] for l in labels_]
        if data_:
            bp = ax.boxplot(data_, labels=labels_, patch_artist=True, widths=0.4,
                            medianprops=dict(color="white", linewidth=2),
                            whiskerprops=dict(color=SPINE, linewidth=0.8),
                            capprops=dict(color=SPINE, linewidth=0.8))
            for patch, lbl in zip(bp["boxes"], labels_):
                patch.set_facecolor(regime_colors.get(lbl, "#90CAF9"))
                patch.set_alpha(0.85)
        ax.set_title(f"{label} QLIKE by Regime", fontsize=10)
        ax.set_ylabel("Median QLIKE", fontsize=9)
        ax.grid(axis="y", color=SPINE, linewidth=0.4, linestyle="--"); ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rosa_02_regime_qlike.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Aggregate plot 3: OLS vs WLS head-to-head scatter ────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    for sp in ax.spines.values(): sp.set_color(SPINE); sp.set_linewidth(0.6)
    for _, row in summary_df.iterrows():
        color = regime_colors.get(row["liquidity_regime"], "steelblue")
        ax.scatter(row["ols_median_QLIKE"], row["wls_median_QLIKE"],
                   color=color, s=60, alpha=0.85, zorder=3)
        ax.annotate(str(int(row["stock_id"])),
                    (row["ols_median_QLIKE"], row["wls_median_QLIKE"]),
                    fontsize=6, ha="left", va="bottom")
    finite = summary_df[["ols_median_QLIKE","wls_median_QLIKE"]].stack().dropna()
    if not finite.empty:
        lim_ = [finite.min(), finite.max()]
        ax.plot(lim_, lim_, "r--", linewidth=1, label="Equal performance")
    from matplotlib.patches import Patch as _Patch
    leg_els = [_Patch(facecolor=c, label=r)
               for r, c in regime_colors.items()
               if r in summary_df["liquidity_regime"].values]
    leg_els.append(plt.Line2D([0],[0], color="red", linestyle="--",
                               label="OLS = WLS"))
    ax.legend(handles=leg_els, fontsize=8)
    ax.set_xlabel("OLS Median QLIKE", fontsize=10)
    ax.set_ylabel("WLS Median QLIKE", fontsize=10)
    ax.set_title("OLS vs WLS Head-to-Head\n(below diagonal → WLS wins)",
                 fontsize=11, fontweight="bold")
    ax.grid(color=SPINE, linewidth=0.4, linestyle="--"); ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rosa_03_ols_vs_wls.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Aggregate plots saved to: {OUTPUT_DIR}")

print("\nrosa.py complete.")
