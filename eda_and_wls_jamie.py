"""
Member 2 (Jamie): WLS Volatility Forecasting — Complete Three-Phase Pipeline
=============================================================================
  - Phase 3 uses 20 RANDOM liquid + 20 RANDOM mixed + 20 RANDOM illiquid stocks
  - A random subset of time_ids is sampled for speed + reproducibility
  - Fixed 80/20 split CV: first 8 minutes (buckets 1–16) train, last 2 minutes (buckets 17–20) validate
  - All random seeds are fixed (RANDOM_SEED = 42) for reproducibility
  - Stock IDs and time IDs used are printed explicitly at runtime

FEATURE SELECTION is data-driven:
  1. Compute Pearson |r| with RV on stock_id=1 training data
  2. Drop features with |r| < MIN_CORR_THRESHOLD (default 0.05)
  3. Drop one of each highly collinear pair (|r_pair| > COLLINEARITY_THRESHOLD)
  4. FINAL_FEATURES is derived from the data, not hardcoded

PHASE 1 — WLS Baseline + CV on stock_id = 1
PHASE 2 — Liquidity Classification of ALL stocks
PHASE 3 — WLS + CV on 60 Demo Stocks (20 liquid + 20 mixed + 20 illiquid)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────────────────────────────
C_BLUE   = "#1F4E79"
C_ORANGE = "#C55A11"
C_GREEN  = "#375623"
C_PURPLE = "#7030A0"
C_GOLD   = "#BF8F00"
C_GREY   = "#595959"
C_LGREY  = "#D9D9D9"
C_BG     = "#F7F9FC"
C_RED    = "#C00000"
C_TEAL   = "#006A6A"
STOCK_COLORS = {"liquid": C_TEAL, "mixed": C_GOLD, "illiquid": C_RED}
LIQ_COLORS   = {"liquid": C_TEAL, "mixed": C_GOLD, "illiquid": C_RED}

plt.rcParams.update({
    "figure.facecolor":  C_BG,  "axes.facecolor":    C_BG,
    "axes.edgecolor":    C_GREY, "axes.labelcolor":   C_GREY,
    "xtick.color":       C_GREY, "ytick.color":       C_GREY,
    "text.color":        C_GREY, "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,  "axes.spines.right": False,
    "axes.grid":         True,   "grid.alpha":        0.3,
    "grid.color":        C_LGREY,
})

OUT = "/Users/jamiewood/Documents/DATA3888/DATA3888G08/"

# ─────────────────────────────────────────────────────────────────────
# KEY CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
RANDOM_SEED        = 42          # fixed seed for full reproducibility
N_DEMO_PER_REGIME  = 20          # 20 random stocks per regime (liquid/mixed/illiquid)
N_TIME_IDS         = 300         # random subset of time_ids to keep (balances speed vs accuracy)

# CV SPLIT CONFIGURATION
# Each time_id covers a 10-minute window split into 20 × 30-second buckets.
# We use a fixed 80/20 split:  buckets 1–16 = train (8 min), buckets 17–20 = validate (2 min).
CV_TRAIN_BUCKETS = list(range(1, 17))   # buckets 1–16  (8 minutes)
CV_VAL_BUCKETS   = list(range(17, 21))  # buckets 17–20 (2 minutes)
CV_TRAIN_MAX     = 16                   # last training bucket index

# ─────────────────────────────────────────────────────────────────────
# FEATURE SELECTION THRESHOLDS
# ─────────────────────────────────────────────────────────────────────
MIN_CORR_THRESHOLD    = 0.05
COLLINEARITY_THRESHOLD = 0.95

BUCKET_SECONDS     = 30
WINDOW_SECONDS     = 600
N_EXPECTED_BUCKETS = WINDOW_SECONDS // BUCKET_SECONDS
TARGET = "rv"

# ─────────────────────────────────────────────────────────────────────
# CANDIDATE FEATURE POOL
# ─────────────────────────────────────────────────────────────────────
ALL_FEATURES = [
    "rv_lag1", "rv_lag2", "rv_lag3",
    "rv_roll_mean", "rv_roll_sd", "rv_roll_max", "rv_roll_cv",
    "bas", "bas_lag1", "rel_spread", "rel_spread_lag1",
    "bas_change", "bas_pct_change", "bas_roll_mean", "bas_roll_sd",
    "wap", "wap_return", "wap_return2", "wap_dev", "wap_accel",
    "inv_spread", "inv_spread_lag1", "log_activity", "log_activity_lag1",
    "spread_imbalance", "volume_surge",
    "spread_vol_interaction", "rel_spread_vol",
    "spread_change_vol", "activity_vol",
    "rv_lag1_sq", "bas_sq",
]

# ─────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────
def qlike(y, yhat):
    yhat = np.maximum(yhat, 1e-8)
    return float(np.mean(np.log(yhat) + y / yhat))

def mse(y, yhat):  return float(np.mean((y - yhat) ** 2))
def mae(y, yhat):  return float(np.mean(np.abs(y - yhat)))

# ─────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df  = df.sort_values(["stock_id", "time_id", "time_bucket"]).copy()
    grp = df.groupby(["stock_id", "time_id"])
    df["rv_lag1"]      = grp["rv"].shift(1)
    df["rv_lag2"]      = grp["rv"].shift(2)
    df["rv_lag3"]      = grp["rv"].shift(3)
    df["rv_roll_mean"] = grp["rv"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["rv_roll_sd"]   = grp["rv"].transform(lambda x: x.rolling(5, min_periods=1).std())
    df["rv_roll_max"]  = grp["rv"].transform(lambda x: x.rolling(5, min_periods=1).max())
    df["rv_roll_cv"]   = df["rv_roll_sd"] / (df["rv_roll_mean"] + 1e-12)
    df["bas_lag1"]        = grp["bas"].shift(1)
    df["rel_spread"]      = df["bas"] / (df["wap"] + 1e-12)
    df["rel_spread_lag1"] = grp["rel_spread"].shift(1)
    df["bas_change"]      = df["bas"] - df["bas_lag1"]
    df["bas_pct_change"]  = df["bas_change"] / (df["bas_lag1"] + 1e-12)
    df["bas_roll_mean"]   = grp["bas"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["bas_roll_sd"]     = grp["bas"].transform(lambda x: x.rolling(5, min_periods=1).std())
    df["bas_sq"]          = df["bas"] ** 2
    df["wap_lag1"]      = grp["wap"].shift(1)
    df["wap_lag2"]      = grp["wap"].shift(2)
    df["wap_return"]    = np.log(df["wap"] / (df["wap_lag1"] + 1e-12))
    df["wap_return2"]   = np.log(df["wap"] / (df["wap_lag2"] + 1e-12))
    df["wap_roll_mean"] = grp["wap"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["wap_dev"]       = df["wap"] - df["wap_roll_mean"]
    df["wap_accel"]     = df["wap_return"] - grp["wap_return"].shift(1)
    df["inv_spread"]        = 1.0 / (df["bas"] + 1e-6)
    df["inv_spread_lag1"]   = grp["inv_spread"].shift(1)
    df["inv_spread_roll"]   = grp["inv_spread"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["activity_proxy"]    = df["wap"] * df["inv_spread"]
    df["log_activity"]      = np.log1p(df["activity_proxy"])
    df["log_activity_lag1"] = grp["log_activity"].shift(1)
    df["spread_imbalance"]  = df["bas_pct_change"]
    df["volume_surge"]      = df["inv_spread"] / (df["inv_spread_roll"] + 1e-12)
    df["spread_vol_interaction"] = df["bas"]          * df["rv_lag1"]
    df["rel_spread_vol"]         = df["rel_spread"]   * df["rv_lag1"]
    df["spread_change_vol"]      = df["bas_change"]   * df["rv_lag1"]
    df["activity_vol"]           = df["log_activity"] * df["rv_lag1"]
    df["wap_vol_interaction"]    = df["wap_return"]   * df["rv_lag1"]
    df["rv_lag1_sq"] = df["rv_lag1"] ** 2
    return df

# ─────────────────────────────────────────────────────────────────────
# DATA-DRIVEN FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────
def select_features(train_df, candidate_features, target,
                    min_corr=MIN_CORR_THRESHOLD,
                    collinearity_thresh=COLLINEARITY_THRESHOLD,
                    sample_n=50_000, random_state=RANDOM_SEED):
    sample = train_df[candidate_features + [target]].dropna()
    if len(sample) > sample_n:
        sample = sample.sample(sample_n, random_state=random_state)
    rv_corrs = (sample.corr()[target]
                .drop(target)
                .reindex(candidate_features)
                .dropna())
    print(f"\n  [FEATURE SELECTION] Candidate pool : {len(rv_corrs)} features")
    strong = rv_corrs[rv_corrs.abs() >= min_corr]
    dropped_weak = sorted(set(candidate_features) - set(strong.index) -
                          {f for f in candidate_features if f not in rv_corrs.index})
    print(f"  [FEATURE SELECTION] Dropped (|r| < {min_corr}): {dropped_weak}")
    feature_corr = sample[strong.index.tolist()].corr().abs()
    kept = list(strong.index)
    dropped_collinear = []
    for i in range(len(kept)):
        if kept[i] is None:
            continue
        for j in range(i + 1, len(kept)):
            if kept[j] is None:
                continue
            if feature_corr.loc[kept[i], kept[j]] > collinearity_thresh:
                ri = abs(rv_corrs[kept[i]])
                rj = abs(rv_corrs[kept[j]])
                drop = kept[j] if ri >= rj else kept[i]
                dropped_collinear.append(drop)
                if drop == kept[i]:
                    kept[i] = None
                    break
                else:
                    kept[j] = None
    kept = [f for f in kept if f is not None]
    print(f"  [FEATURE SELECTION] Dropped (collinear > {collinearity_thresh}): {sorted(dropped_collinear)}")
    final = sorted(kept, key=lambda f: abs(rv_corrs[f]), reverse=True)
    print(f"  [FEATURE SELECTION] Final features ({len(final)}): {final}")
    return final, rv_corrs

# ─────────────────────────────────────────────────────────────────────
# SHARED PIPELINE HELPERS
# ─────────────────────────────────────────────────────────────────────
def train_val_split(df):
    """80/20 holdout split on time_bucket order (used for final evaluation)."""
    sorted_b  = sorted(df["time_bucket"].unique())
    n_train   = int(len(sorted_b) * 0.8)
    train_max = sorted_b[n_train - 1]
    return (df[df["time_bucket"] <= train_max].copy(),
            df[df["time_bucket"] >  train_max].copy(),
            sorted_b, n_train, train_max)

def tune_alpha(X_tr, y_tr, X_te, y_te, train_max, bv):
    """Grid-search alpha on a single train/val split."""
    alpha_grid = np.round(np.arange(0.05, 1.00, 0.01), 3)
    def _fit(a):
        w    = a ** (train_max - bv)
        pred = np.maximum(LinearRegression().fit(X_tr, y_tr, sample_weight=w).predict(X_te), 1e-8)
        return {"alpha": a, "qlike": qlike(y_te, pred),
                "mse":   mse(y_te, pred), "mae": mae(y_te, pred)}
    results = Parallel(n_jobs=-1)([delayed(_fit)(a) for a in alpha_grid])
    tdf = pd.DataFrame(results)
    return float(tdf.loc[tdf["qlike"].idxmin(), "alpha"]), tdf

def label_bucket_liquidity(frame, bas_q33, bas_q66, act_q33, act_q66):
    def _label(r):
        if r["bas"] <= bas_q33 and r["log_activity"] >= act_q66: return "liquid"
        if r["bas"] >= bas_q66 and r["log_activity"] <= act_q33: return "illiquid"
        return "mixed"
    frame["bucket_liquidity"] = frame.apply(_label, axis=1)
    return frame

def feature_group_color(feat):
    if "rv" in feat and "spread" not in feat and "activity" not in feat: return C_BLUE
    if feat in ("bas","bas_lag1","rel_spread","rel_spread_lag1","bas_change",
                "bas_pct_change","bas_roll_mean","bas_roll_sd","bas_sq"):    return C_ORANGE
    if "activity" in feat or "inv_spread" in feat or "imbalance" in feat \
            or "volume_surge" in feat:                                        return C_GREEN
    if "wap" in feat:                                                         return C_GOLD
    return C_GREY

# ─────────────────────────────────────────────────────────────────────
# CROSS-VALIDATION — FIXED 80/20 BUCKET SPLIT
# ─────────────────────────────────────────────────────────────────────
def wls_cv_on_df(df_feat, features, target):
    """
    Fixed 80/20 time-bucket CV for WLS on a pre-featured DataFrame.

    Split strategy
    --------------
    Each 10-minute window contains 20 × 30-second buckets.
    We fix the split at the 8-minute mark:
      Train : buckets 1–16  (first 8 minutes, 80%)
      Val   : buckets 17–20 (last  2 minutes, 20%)

    This preserves temporal order and prevents look-ahead leakage.
    The split is identical for every stock and every time_id, making
    results directly comparable across stocks and pipeline runs.

    For the single split:
      1. Find the best alpha via grid-search on the train set.
      2. Fit WLS with that alpha.
      3. Record QLIKE, MSE, MAE on the validation set.

    Returns
    -------
    cv_results : list with one dict
    best_alpha : float  (best alpha from this split)
    """
    df_clean = df_feat.dropna(subset=features + [target]).copy()

    # ── Apply the fixed bucket split ──────────────────────────────────
    tr = df_clean[df_clean["time_bucket"].isin(CV_TRAIN_BUCKETS)]
    te = df_clean[df_clean["time_bucket"].isin(CV_VAL_BUCKETS)]

    print(f"    [CV] Fixed 80/20 split — "
          f"Train buckets: {CV_TRAIN_BUCKETS[0]}–{CV_TRAIN_BUCKETS[-1]}  "
          f"({CV_TRAIN_BUCKETS[-1] * BUCKET_SECONDS // 60} min)  |  "
          f"Val buckets: {CV_VAL_BUCKETS[0]}–{CV_VAL_BUCKETS[-1]}  "
          f"({len(CV_VAL_BUCKETS) * BUCKET_SECONDS // 60} min)")
    print(f"    [CV] Train rows: {len(tr):,}   Val rows: {len(te):,}")

    if len(tr) < 50 or len(te) < 10:
        print(f"    [CV] Skipped — too few rows (tr={len(tr)}, te={len(te)})")
        return [], 0.99

    X_tr = tr[features].values;  y_tr = tr[target].values
    X_te = te[features].values;  y_te = te[target].values
    bv   = tr["time_bucket"].values

    # Tune alpha on this fixed split
    best_a, _ = tune_alpha(X_tr, y_tr, X_te, y_te, CV_TRAIN_MAX, bv)
    w         = best_a ** (CV_TRAIN_MAX - bv)
    wls       = LinearRegression().fit(X_tr, y_tr, sample_weight=w)
    pred      = np.maximum(wls.predict(X_te), 1e-8)

    # OLS baseline for comparison
    ols      = LinearRegression().fit(X_tr, y_tr)
    pred_ols = np.maximum(ols.predict(X_te), 1e-8)

    result = {
        "fold":        1,
        "n_train":     len(tr),
        "n_val":       len(te),
        "tr_buckets":  f"{CV_TRAIN_BUCKETS[0]}–{CV_TRAIN_BUCKETS[-1]}",
        "val_buckets": f"{CV_VAL_BUCKETS[0]}–{CV_VAL_BUCKETS[-1]}",
        "best_alpha":  best_a,
        "wls_qlike":   qlike(y_te, pred),
        "wls_mse":     mse(y_te, pred),
        "wls_mae":     mae(y_te, pred),
        "ols_qlike":   qlike(y_te, pred_ols),
    }

    print(f"    [CV] α={best_a:.2f}  "
          f"WLS_QLIKE={result['wls_qlike']:.6f}  "
          f"OLS_QLIKE={result['ols_qlike']:.6f}")

    return [result], float(best_a)


def print_cv_summary(cv_results, label=""):
    """Pretty-print CV metrics (single split)."""
    if not cv_results:
        print("  [CV] No results to summarise.")
        return
    r = cv_results[0]
    print(f"\n  ── CV SUMMARY  {label} ──")
    print(f"  Split   : buckets {r['tr_buckets']} (train)  →  buckets {r['val_buckets']} (val)")
    print(f"  Rows    : train={r['n_train']:,}   val={r['n_val']:,}")
    print(f"  Alpha   : {r['best_alpha']:.3f}")
    print(f"  {'Metric':<20} {'Value':>12}")
    print("  " + "─" * 34)
    for col, lbl in [("wls_qlike","WLS QLIKE"), ("wls_mse","WLS MSE"),
                     ("wls_mae","WLS MAE"), ("ols_qlike","OLS QLIKE")]:
        print(f"  {lbl:<20} {r[col]:>12.6f}")


def plot_cv_results(cv_results, title, save_path):
    """Bar chart comparing WLS vs OLS on the fixed 80/20 split."""
    if not cv_results:
        return
    r = cv_results[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(C_BG)
    fig.suptitle(title, fontsize=12, fontweight="bold", color=C_BLUE)

    # ── Left: WLS vs OLS QLIKE ───────────────────────────────────────
    labels = ["WLS", "OLS"]
    vals   = [r["wls_qlike"], r["ols_qlike"]]
    colors = [C_TEAL, C_ORANGE]
    bars   = axes[0].bar(labels, vals, color=colors, alpha=0.82, width=0.4, edgecolor="none")
    for bar, v in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.01,
                     f"{v:.6f}", ha="center", va="bottom", fontsize=9, color=C_GREY)
    axes[0].set_title(
        f"QLIKE — Fixed 80/20 Split\n"
        f"Train: buckets {r['tr_buckets']}  |  Val: buckets {r['val_buckets']}"
    )
    axes[0].set_ylabel("QLIKE")
    axes[0].set_facecolor(C_BG)

    # ── Right: best alpha display ────────────────────────────────────
    axes[1].bar(["Best α"], [r["best_alpha"]], color=C_BLUE, alpha=0.82,
                width=0.25, edgecolor="none")
    axes[1].axhline(r["best_alpha"], color=C_RED, linestyle="--", lw=1.2,
                    label=f"α = {r['best_alpha']:.3f}")
    axes[1].text(0, r["best_alpha"] + 0.01, f"{r['best_alpha']:.3f}",
                 ha="center", va="bottom", fontsize=10, color=C_GREY)
    axes[1].set_title("Best Alpha (fixed split)")
    axes[1].set_ylabel("α")
    axes[1].set_ylim(0, 1.1)
    axes[1].legend(fontsize=9)
    axes[1].set_facecolor(C_BG)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}  ✓")


# ═════════════════════════════════════════════════════════════════════
# LOAD FULL DATASET
# ═════════════════════════════════════════════════════════════════════
print("=" * 70)
print("LOADING FULL DATASET")
print("=" * 70)
df_full = pd.read_csv(OUT + "optiver_aggregated.csv")
df_full = df_full[df_full["time_bucket"] > 0].copy()
df_full = df_full.rename(columns={
    "WAP_mean": "wap", "BidAskSpread_mean": "bas", "volatility": "rv"})
df_full = df_full.sort_values(["stock_id","time_id","time_bucket"]).reset_index(drop=True)
print(f"  Full dataset (bucket-0 dropped): {df_full.shape}")
print(f"  Unique stocks  : {df_full['stock_id'].nunique()}")
print(f"  Unique time_ids: {df_full['time_id'].nunique()}")

# ─────────────────────────────────────────────────────────────────────
# RANDOM time_id SUBSAMPLE
# ─────────────────────────────────────────────────────────────────────
all_time_ids = sorted(df_full["time_id"].unique())
rng = np.random.default_rng(RANDOM_SEED)

if len(all_time_ids) > N_TIME_IDS:
    sampled_time_ids = sorted(rng.choice(all_time_ids, size=N_TIME_IDS, replace=False).tolist())
    df_full = df_full[df_full["time_id"].isin(sampled_time_ids)].copy()
    print(f"\n  ── TIME_ID SUBSAMPLE ──")
    print(f"  Sampled {N_TIME_IDS} of {len(all_time_ids)} time_ids  (seed={RANDOM_SEED})")
    print(f"  Sampled time_ids: {sampled_time_ids}")
else:
    sampled_time_ids = all_time_ids
    print(f"  Using all {len(all_time_ids)} time_ids (fewer than requested {N_TIME_IDS})")

print(f"  Dataset after time_id filter: {df_full.shape}")


# ═══════════════════════════════════════════════════════════════════════
# ██  PHASE 1 — WLS BASELINE + CV  (stock_id = 1 only)  ██████████████
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "█" * 70)
print("PHASE 1 — WLS BASELINE + CROSS-VALIDATION  (stock_id = 1)")
print(f"  CV strategy: fixed 80/20 split  "
      f"(train buckets {CV_TRAIN_BUCKETS[0]}–{CV_TRAIN_BUCKETS[-1]}, "
      f"val buckets {CV_VAL_BUCKETS[0]}–{CV_VAL_BUCKETS[-1]})")
print("█" * 70)

df1 = df_full[df_full["stock_id"] == 1].copy()
print(f"  stock_id=1 rows: {len(df1):,}")

train1, test1, sorted_b1, n_train1, train_max1 = train_val_split(df1)
print(f"  Holdout split — Train buckets: 1–{train_max1}  |  Test buckets: {train_max1+1}–{sorted_b1[-1]}")
print(f"  Train rows: {len(train1):,}   Test rows: {len(test1):,}")

print("  Engineering features …")
train1 = add_features(train1);  test1 = add_features(test1)

bas_q33_1 = train1["bas"].quantile(0.33); bas_q66_1 = train1["bas"].quantile(0.66)
act_q33_1 = train1["log_activity"].quantile(0.33); act_q66_1 = train1["log_activity"].quantile(0.66)
train1 = label_bucket_liquidity(train1, bas_q33_1, bas_q66_1, act_q33_1, act_q66_1)
test1  = label_bucket_liquidity(test1,  bas_q33_1, bas_q66_1, act_q33_1, act_q66_1)

# ── DATA-DRIVEN FEATURE SELECTION ─────────────────────────────────────
print("\n  Running data-driven feature selection on stock_id=1 training data …")
FINAL_FEATURES, rv_corrs = select_features(
    train1, ALL_FEATURES, TARGET,
    min_corr=MIN_CORR_THRESHOLD,
    collinearity_thresh=COLLINEARITY_THRESHOLD,
)

print(f"\n  ── SELECTED FEATURES ({len(FINAL_FEATURES)}) ──")
print(f"  {'Feature':<30} {'|r with RV|':>12}  {'r':>8}  Group")
print("  " + "─" * 65)
for f in FINAL_FEATURES:
    r = rv_corrs.get(f, np.nan)
    grp = feature_group_color(f)
    grp_name = ("RV lags/rolling" if grp == C_BLUE else
                 "Bid-Ask Spread"  if grp == C_ORANGE else
                 "Order-flow"      if grp == C_GREEN else
                 "Price (WAP)"     if grp == C_GOLD else "Interaction/nonlinear")
    print(f"  {f:<30} {abs(r):>12.4f}  {r:>+8.4f}  {grp_name}")

# ── EDA sample ────────────────────────────────────────────────────────
eda1 = train1.sample(min(50_000, len(train1)), random_state=RANDOM_SEED)
rv_corrs_full = (eda1[ALL_FEATURES + [TARGET]].dropna().corr()[TARGET]
                 .drop(TARGET).sort_values(key=abs, ascending=False))

# ─────────────────────────────────────────────────────────────────────
# EDA PLOTS
# ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.patch.set_facecolor(C_BG)
fig.suptitle("EDA — Covariate Distributions  (stock_id=1, 30-second buckets)",
             fontsize=14, fontweight="bold", color=C_BLUE, y=1.01)
axes[0,0].hist(train1["rv"].dropna(), bins=100, color=C_BLUE, edgecolor="none", alpha=0.85)
axes[0,0].set_title("Realised Volatility  [TARGET]"); axes[0,0].set_xlabel("rv"); axes[0,0].set_ylabel("Count")
axes[0,1].hist(train1["bas"].dropna(), bins=100, color=C_ORANGE, edgecolor="none", alpha=0.85)
axes[0,1].set_title("Bid-Ask Spread  [SPREAD COVARIATE]"); axes[0,1].set_xlabel("bas")
axes[0,2].hist(eda1["rel_spread"].dropna(), bins=80, color=C_PURPLE, edgecolor="none", alpha=0.85)
axes[0,2].set_title("Relative Spread = BAS / WAP  [NORMALISED SPREAD]"); axes[0,2].set_xlabel("rel_spread")
axes[1,0].hist(eda1["log_activity"].dropna(), bins=100, color=C_GREEN, edgecolor="none", alpha=0.85)
axes[1,0].set_title("log(Activity Proxy)  [ORDER-FLOW COVARIATE]"); axes[1,0].set_xlabel("log_activity")
axes[1,1].hist(eda1["inv_spread"].dropna(), bins=80, color=C_GOLD, edgecolor="none", alpha=0.85)
axes[1,1].set_title("Inverse Spread  [NUM-ORDERS PROXY]"); axes[1,1].set_xlabel("inv_spread")
axes[1,2].scatter(eda1["rv_lag1"], eda1["rv"], alpha=0.1, s=3, color=C_BLUE)
axes[1,2].set_title("RV Lag-1 vs RV  (persistence)"); axes[1,2].set_xlabel("rv_lag1"); axes[1,2].set_ylabel("rv")
plt.tight_layout()
plt.savefig(OUT + "eda_distributions.png", dpi=150, bbox_inches="tight"); plt.show()
print("  Saved: eda_distributions.png  ✓")

top_n        = min(len(rv_corrs_full), 28)
rv_corrs_top = rv_corrs_full.head(top_n)
bar_colors   = [feature_group_color(f) if f in FINAL_FEATURES else C_LGREY for f in rv_corrs_top.index]
fig, ax = plt.subplots(figsize=(15, 9)); fig.patch.set_facecolor(C_BG); ax.set_facecolor(C_BG)
ax.barh(range(top_n), rv_corrs_top.values.astype(float), color=bar_colors, alpha=0.88, edgecolor="none")
ax.set_yticks(range(top_n))
ax.set_yticklabels([f"★ {f}" if f in FINAL_FEATURES else f"  {f}" for f in rv_corrs_top.index], fontsize=8)
ax.invert_yaxis()
ax.axvline(0, color=C_GREY, linewidth=0.8)
ax.axvline( MIN_CORR_THRESHOLD, color=C_RED, linestyle=":", lw=1.2)
ax.axvline(-MIN_CORR_THRESHOLD, color=C_RED, linestyle=":", lw=1.2)
ax.set_xlabel("Pearson r with Realised Volatility", fontsize=10)
ax.set_title(f"Feature Selection — Correlation with RV  (stock_id=1, DATA-DRIVEN)\n"
             f"★ = selected ({len(FINAL_FEATURES)} features)  |  Grey = dropped  |  "
             f"Threshold: |r| ≥ {MIN_CORR_THRESHOLD}, collinearity < {COLLINEARITY_THRESHOLD}",
             fontsize=11, fontweight="bold", color=C_BLUE)
legend_items = [
    mpatches.Patch(color=C_BLUE,   label="RV lags / rolling  ★ selected"),
    mpatches.Patch(color=C_ORANGE, label="Bid-Ask Spread      ★ selected"),
    mpatches.Patch(color=C_GREEN,  label="Order-flow proxies  ★ selected"),
    mpatches.Patch(color=C_GOLD,   label="Price (WAP)         ★ selected"),
    mpatches.Patch(color=C_GREY,   label="Interaction/nonlinear ★ selected"),
    mpatches.Patch(color=C_LGREY,  label="Dropped (weak or collinear)"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(OUT + "feature_selection_correlation.png", dpi=150, bbox_inches="tight"); plt.show()
print("  Saved: feature_selection_correlation.png  ✓")

heatmap_cols = [TARGET] + FINAL_FEATURES[:12]
corr_matrix  = eda1[heatmap_cols].dropna().corr()
plt.figure(figsize=(14, 12)); plt.gca().set_facecolor(C_BG)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt=".2f",
            cmap=sns.diverging_palette(220, 20, as_cmap=True),
            center=0, mask=mask, linewidths=0.4, annot_kws={"size": 8})
plt.title(f"Correlation Matrix — Selected Covariates  (stock_id=1, top {len(heatmap_cols)-1} features)\n"
          "(Price  ·  Bid-Ask Spread  ·  Order-Flow  ·  RV Lags)",
          fontsize=12, fontweight="bold", color=C_BLUE)
plt.tight_layout()
plt.savefig(OUT + "correlation_heatmap.png", dpi=150, bbox_inches="tight"); plt.show()
print("  Saved: correlation_heatmap.png  ✓")

fig, axes = plt.subplots(1, 3, figsize=(18, 5)); fig.patch.set_facecolor(C_BG)
fig.suptitle("Bid-Ask Spread Covariate Analysis  (stock_id=1)", fontsize=13, fontweight="bold", color=C_BLUE)
axes[0].scatter(eda1["rel_spread"], eda1["rv"], alpha=0.1, s=3, color=C_PURPLE)
axes[0].set_title("Relative Spread vs RV"); axes[0].set_xlabel("rel_spread"); axes[0].set_ylabel("rv")
axes[1].scatter(eda1["bas_change"], eda1["rv"], alpha=0.1, s=3, color=C_ORANGE)
axes[1].set_title("Spread Change vs RV"); axes[1].set_xlabel("bas_change"); axes[1].set_ylabel("rv")
axes[2].scatter(eda1["spread_vol_interaction"], eda1["rv"], alpha=0.1, s=3, color=C_GREEN)
axes[2].set_title("Spread × Lagged Vol vs RV"); axes[2].set_xlabel("spread_vol_interaction"); axes[2].set_ylabel("rv")
plt.tight_layout()
plt.savefig(OUT + "spread_features_eda.png", dpi=150, bbox_inches="tight"); plt.show()
print("  Saved: spread_features_eda.png  ✓")

fig, axes = plt.subplots(1, 3, figsize=(18, 5)); fig.patch.set_facecolor(C_BG)
fig.suptitle("Order-Flow Proxy & Activity Analysis  (stock_id=1)", fontsize=13, fontweight="bold", color=C_BLUE)
axes[0].scatter(eda1["inv_spread"], eda1["rv"], alpha=0.1, s=3, color=C_PURPLE)
axes[0].set_title("Inv Spread vs RV"); axes[0].set_xlabel("inv_spread"); axes[0].set_ylabel("rv")
axes[1].scatter(eda1["log_activity"], eda1["rv"], alpha=0.1, s=3, color=C_GREEN)
axes[1].set_title("log(Activity) vs RV"); axes[1].set_xlabel("log_activity"); axes[1].set_ylabel("rv")
axes[2].scatter(eda1["volume_surge"], eda1["rv"], alpha=0.1, s=3, color=C_GOLD)
axes[2].set_title("Volume Surge vs RV"); axes[2].set_xlabel("volume_surge"); axes[2].set_ylabel("rv")
plt.tight_layout()
plt.savefig(OUT + "order_flow_features_eda.png", dpi=150, bbox_inches="tight"); plt.show()
print("  Saved: order_flow_features_eda.png  ✓")

q33 = train1["rv"].quantile(0.33); q66 = train1["rv"].quantile(0.66)
def classify_regime(v):
    if v <= q33: return 0
    if v <= q66: return 1
    return 2
train1["regime"] = train1["rv"].apply(classify_regime)
test1["regime"]  = test1["rv"].apply(classify_regime)
REG_COLORS = {0: C_GREEN, 1: C_GOLD, 2: C_ORANGE}
REG_LABELS = {0: "Low Vol", 1: "Med Vol", 2: "High Vol"}
bc1 = train1["bucket_liquidity"].value_counts()
liq_labels = ["liquid", "mixed", "illiquid"]
sample_tids = train1["time_id"].unique()
sample_tid  = sample_tids[len(sample_tids) // 2]
sample_ts   = train1[train1["time_id"] == sample_tid].sort_values("time_bucket")
MAX_LAG  = 15
rv_series = train1.sort_values(["time_id","time_bucket"])["rv"].dropna().values
acf_vals  = np.array([np.corrcoef(rv_series[:-lag], rv_series[lag:])[0,1]
                      if lag > 0 else 1.0 for lag in range(MAX_LAG+1)])
ci = 1.96 / np.sqrt(len(rv_series))

fig = plt.figure(figsize=(22, 18)); fig.patch.set_facecolor(C_BG)
fig.suptitle("Liquidity Regime EDA — Identifying Liquid vs Illiquid Market Conditions\n"
             "Jamie (Member 2) · stock_id=1 · Supports dynamic model selection in trading app",
             fontsize=13, fontweight="bold", color=C_BLUE, y=1.005)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.40)
ax_sc = fig.add_subplot(gs[0, :2])
for regime, color in LIQ_COLORS.items():
    sub = eda1[eda1["bucket_liquidity"] == regime]
    ax_sc.scatter(sub["bas"], sub["log_activity"], alpha=0.20, s=4, color=color,
                  label=f"{regime}  (n={len(sub):,})")
ax_sc.axvline(bas_q33_1, color=C_TEAL, linestyle="--", lw=1.0, alpha=0.8, label=f"BAS q33={bas_q33_1:.5f}")
ax_sc.axvline(bas_q66_1, color=C_RED,  linestyle="--", lw=1.0, alpha=0.8, label=f"BAS q66={bas_q66_1:.5f}")
ax_sc.axhline(act_q33_1, color=C_RED,  linestyle=":",  lw=1.0, alpha=0.8, label=f"Activity q33={act_q33_1:.2f}")
ax_sc.axhline(act_q66_1, color=C_TEAL, linestyle=":",  lw=1.0, alpha=0.8, label=f"Activity q66={act_q66_1:.2f}")
ax_sc.set_title("BAS vs log(Activity) — Bucket Liquidity Classification  (stock_id=1)", fontsize=10)
ax_sc.set_xlabel("Bid-Ask Spread (BAS)"); ax_sc.set_ylabel("log(Activity Proxy)")
ax_sc.legend(fontsize=8, loc="upper right")
ax_reg = fig.add_subplot(gs[0, 2])
liq_cnts = [bc1.get(l, 0) for l in liq_labels]
ax_reg.bar(liq_labels, liq_cnts, color=[LIQ_COLORS[l] for l in liq_labels], alpha=0.85, edgecolor="none")
ax_reg.set_title("Bucket Liquidity\nDistribution (train)", fontsize=9)
ax_reg.set_ylabel("Row count")
for i, (lbl, cnt) in enumerate(zip(liq_labels, liq_cnts)):
    ax_reg.text(i, cnt + max(liq_cnts)*0.01, f"{100*cnt/sum(liq_cnts):.1f}%", ha="center", fontsize=8)
ax_rv = fig.add_subplot(gs[1, 0])
for regime, color in LIQ_COLORS.items():
    vals = eda1.loc[eda1["bucket_liquidity"] == regime, "rv"].dropna()
    ax_rv.hist(vals, bins=60, color=color, alpha=0.55, edgecolor="none", label=regime, density=True)
ax_rv.set_title("RV Distribution\nby Liquidity Regime", fontsize=9)
ax_rv.set_xlabel("Realised Volatility"); ax_rv.set_ylabel("Density"); ax_rv.legend(fontsize=8)
ax_bas = fig.add_subplot(gs[1, 1])
for regime, color in LIQ_COLORS.items():
    vals = eda1.loc[eda1["bucket_liquidity"] == regime, "bas"].dropna()
    ax_bas.hist(vals, bins=60, color=color, alpha=0.55, edgecolor="none", label=regime, density=True)
ax_bas.set_title("BAS Distribution\nby Liquidity Regime", fontsize=9)
ax_bas.set_xlabel("Bid-Ask Spread"); ax_bas.set_ylabel("Density"); ax_bas.legend(fontsize=8)
ax_mrv = fig.add_subplot(gs[1, 2])
mean_rv_by_regime = eda1.groupby("bucket_liquidity")["rv"].mean()
ax_mrv.bar([l for l in liq_labels if l in mean_rv_by_regime.index],
           [mean_rv_by_regime.get(l, 0) for l in liq_labels if l in mean_rv_by_regime.index],
           color=[LIQ_COLORS[l] for l in liq_labels if l in mean_rv_by_regime.index], alpha=0.85, edgecolor="none")
ax_mrv.set_title("Mean RV by Liquidity Regime\n(illiquid = higher vol, GARCH can diverge)", fontsize=9)
ax_mrv.set_ylabel("Mean Realised Volatility")
ax_ts = fig.add_subplot(gs[2, :2])
for _, row in sample_ts.iterrows():
    ax_ts.axvspan(row["time_bucket"]-0.5, row["time_bucket"]+0.5,
                  color=REG_COLORS[row["regime"]], alpha=0.28, linewidth=0)
ax_ts.plot(sample_ts["time_bucket"], sample_ts["rv"], color=C_BLUE, lw=1.8, zorder=3)
ax_ts.axhline(q33, color=C_GREEN,  linestyle="--", lw=0.9, alpha=0.8, label=f"q33={q33:.5f}")
ax_ts.axhline(q66, color=C_ORANGE, linestyle="--", lw=0.9, alpha=0.8, label=f"q66={q66:.5f}")
ax_ts.set_title(f"Volatility Clustering — stock_id=1, time_id={sample_tid}", fontsize=9)
ax_ts.set_xlabel("30-second bucket index"); ax_ts.set_ylabel("Realised Volatility")
regime_patches = [mpatches.Patch(color=REG_COLORS[r], alpha=0.6, label=REG_LABELS[r]) for r in [0,1,2]]
ax_ts.legend(handles=regime_patches, loc="upper right", fontsize=8)
ax_acf = fig.add_subplot(gs[2, 2])
ax_acf.bar(range(MAX_LAG+1), acf_vals, color=C_BLUE, alpha=0.8, edgecolor="none")
ax_acf.axhline( ci, color=C_ORANGE, linestyle="--", lw=0.9, label="95% CI")
ax_acf.axhline(-ci, color=C_ORANGE, linestyle="--", lw=0.9)
ax_acf.axhline(0,   color=C_GREY,   linewidth=0.7)
ax_acf.set_title("RV Autocorrelation  (stock_id=1)\n→ strong persistence justifies rv_lag1, rv_lag2", fontsize=9)
ax_acf.set_xlabel("Lag (×30 sec)"); ax_acf.set_ylabel("ACF"); ax_acf.legend(fontsize=8)
plt.savefig(OUT + "liquidity_regime_eda.png", dpi=150, bbox_inches="tight"); plt.show()
print("  Saved: liquidity_regime_eda.png  ✓")

fig = plt.figure(figsize=(22, 14)); fig.patch.set_facecolor(C_BG)
fig.suptitle(f"Volatility Clustering & Market Regimes  (stock_id=1)\n"
             f"Regime thresholds: Low < {q33:.5f}  ≤  Med < {q66:.5f}  ≤  High",
             fontsize=13, fontweight="bold", color=C_BLUE, y=1.005)
gs2 = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)
ax_ts2 = fig.add_subplot(gs2[0, :])
for _, row in sample_ts.iterrows():
    ax_ts2.axvspan(row["time_bucket"]-0.5, row["time_bucket"]+0.5,
                   color=REG_COLORS[row["regime"]], alpha=0.28, linewidth=0)
ax_ts2.plot(sample_ts["time_bucket"], sample_ts["rv"], color=C_BLUE, lw=1.8, zorder=3)
ax_ts2.axhline(q33, color=C_GREEN,  linestyle="--", lw=0.9, alpha=0.8, label=f"q33={q33:.5f}")
ax_ts2.axhline(q66, color=C_ORANGE, linestyle="--", lw=0.9, alpha=0.8, label=f"q66={q66:.5f}")
ax_ts2.set_title(f"Volatility Clustering — stock_id=1, time_id={sample_tid}", fontsize=10)
ax_ts2.set_xlabel("30-second bucket index"); ax_ts2.set_ylabel("Realised Volatility")
ax_ts2.legend(handles=[mpatches.Patch(color=REG_COLORS[r], alpha=0.6, label=REG_LABELS[r]) for r in [0,1,2]],
              loc="upper right", fontsize=8)
ax_reg2 = fig.add_subplot(gs2[1, 0])
counts = train1["regime"].value_counts().sort_index()
ax_reg2.bar([REG_LABELS[r] for r in counts.index], counts.values,
            color=[REG_COLORS[r] for r in counts.index], alpha=0.85, edgecolor="none")
ax_reg2.set_title("Regime Distribution\n(stock_id=1, train)", fontsize=9); ax_reg2.set_ylabel("Row count")
ax_bas2 = fig.add_subplot(gs2[1, 1])
bas_reg = train1.groupby("regime")["bas"].mean()
ax_bas2.bar([REG_LABELS[r] for r in bas_reg.index], bas_reg.values,
            color=[REG_COLORS[r] for r in bas_reg.index], alpha=0.85, edgecolor="none")
ax_bas2.set_title("Mean BAS by Regime", fontsize=9); ax_bas2.set_ylabel("Mean BAS")
ax_acf2 = fig.add_subplot(gs2[1, 2])
ax_acf2.bar(range(MAX_LAG+1), acf_vals, color=C_BLUE, alpha=0.8, edgecolor="none")
ax_acf2.axhline( ci, color=C_ORANGE, linestyle="--", lw=0.9, label="95% CI")
ax_acf2.axhline(-ci, color=C_ORANGE, linestyle="--", lw=0.9)
ax_acf2.axhline(0,   color=C_GREY,   linewidth=0.7)
ax_acf2.set_title("RV Autocorrelation  (stock_id=1)", fontsize=9)
ax_acf2.set_xlabel("Lag (×30 sec)"); ax_acf2.set_ylabel("ACF"); ax_acf2.legend(fontsize=8)
plt.savefig(OUT + "cluster_plots.png", dpi=150, bbox_inches="tight"); plt.show()
print("  Saved: cluster_plots.png  ✓")

# ── PHASE 1 WLS HOLDOUT MODEL ──────────────────────────────────────────
tr1c = train1.dropna(subset=FINAL_FEATURES + [TARGET])
te1c = test1.dropna(subset=FINAL_FEATURES + [TARGET])
X_tr1, y_tr1 = tr1c[FINAL_FEATURES].values, tr1c[TARGET].values
X_te1, y_te1 = te1c[FINAL_FEATURES].values, te1c[TARGET].values
bv1          = tr1c["time_bucket"].values

ols1      = LinearRegression().fit(X_tr1, y_tr1)
ols1_pred = np.maximum(ols1.predict(X_te1), 1e-8)
print(f"\n  Holdout OLS  MSE={mse(y_te1,ols1_pred):.8f}  QLIKE={qlike(y_te1,ols1_pred):.6f}")

print("  Tuning alpha on holdout split …")
best_alpha1, tune_df1 = tune_alpha(X_tr1, y_tr1, X_te1, y_te1, train_max1, bv1)
print(f"  Holdout best α = {best_alpha1:.2f}")

w1        = best_alpha1 ** (train_max1 - bv1)
wls1      = LinearRegression().fit(X_tr1, y_tr1, sample_weight=w1)
wls1_pred = np.maximum(wls1.predict(X_te1), 1e-8)
print(f"  Holdout WLS  MSE={mse(y_te1,wls1_pred):.8f}  QLIKE={qlike(y_te1,wls1_pred):.6f}")

# ── PHASE 1 CV (fixed 80/20 bucket split) ─────────────────────────────
print(f"\n  Running fixed 80/20 CV on stock_id=1 …")
print(f"  Train: buckets {CV_TRAIN_BUCKETS[0]}–{CV_TRAIN_BUCKETS[-1]} "
      f"({len(CV_TRAIN_BUCKETS) * BUCKET_SECONDS // 60} min)  |  "
      f"Val: buckets {CV_VAL_BUCKETS[0]}–{CV_VAL_BUCKETS[-1]} "
      f"({len(CV_VAL_BUCKETS) * BUCKET_SECONDS // 60} min)")

cv1_results, best_alpha1_cv = wls_cv_on_df(train1, FINAL_FEATURES, TARGET)
print_cv_summary(cv1_results, label="Phase 1 — stock_id=1")
plot_cv_results(cv1_results,
                title="PHASE 1 — Fixed 80/20 CV  (stock_id=1, buckets 1–16 train | 17–20 val)",
                save_path=OUT + "phase1_cv_results.png")

# Phase 1 holdout diagnostic plot
fig, axes = plt.subplots(2, 3, figsize=(22, 12)); fig.patch.set_facecolor(C_BG)
fig.suptitle(f"PHASE 1 — WLS Baseline  (stock_id=1, α={best_alpha1:.2f}, {len(FINAL_FEATURES)} selected features)\n"
             f"Holdout: OLS QLIKE={qlike(y_te1,ols1_pred):.4f}  →  WLS QLIKE={qlike(y_te1,wls1_pred):.4f}  "
             f"|  CV QLIKE={cv1_results[0]['wls_qlike']:.4f}" if cv1_results else "",
             fontsize=12, fontweight="bold", color=C_BLUE)
for ax, metric, color in zip(axes[0,:], ["qlike","mse","mae"], [C_BLUE, C_ORANGE, C_GREEN]):
    ax.plot(tune_df1["alpha"], tune_df1[metric], color=color, lw=1.5)
    ax.axvline(best_alpha1, color="red", linestyle="--", lw=1.3, label=f"Holdout α={best_alpha1:.2f}")
    ax.axvline(best_alpha1_cv, color=C_GOLD, linestyle=":", lw=1.3, label=f"CV best α={best_alpha1_cv:.2f}")
    ax.set_title(f"{metric.upper()} vs α"); ax.set_xlabel("Alpha"); ax.set_ylabel(metric.upper())
    ax.legend(fontsize=8); ax.set_facecolor(C_BG)
for ax, pred, label, color in zip(
    [axes[1,0], axes[1,1]], [ols1_pred, wls1_pred],
    ["OLS", f"WLS (α={best_alpha1:.2f})"], [C_BLUE, C_ORANGE]
):
    ax.set_facecolor(C_BG)
    idx = np.random.choice(len(y_te1), min(5000, len(y_te1)), replace=False)
    ax.scatter(y_te1[idx], pred[idx], alpha=0.15, s=3, color=color)
    lim = max(y_te1.max(), pred.max())
    ax.plot([0,lim],[0,lim], "k--", lw=1, label="Perfect")
    ax.set_title(f"{label}: Predicted vs Actual"); ax.set_xlabel("Actual RV")
    ax.set_ylabel("Predicted RV"); ax.legend(fontsize=8)
ax_w = axes[1,2]; ax_w.set_facecolor(C_BG)
t_v  = np.array(sorted_b1[:n_train1])
ax_w.bar(t_v, best_alpha1**(train_max1-t_v), color=C_BLUE, alpha=0.75, edgecolor="none", width=0.8)
ax_w.set_title(f"WLS Training Weights  (α={best_alpha1:.2f})", fontsize=9)
ax_w.set_xlabel("Bucket"); ax_w.set_ylabel("α^(T-t)")
ax_w.annotate("weight = 1", xy=(train_max1, 1), xytext=(train_max1-5, 0.85),
              arrowprops=dict(arrowstyle="->", color=C_ORANGE), color=C_ORANGE, fontsize=8)
plt.tight_layout()
plt.savefig(OUT + "phase1_wls_stock1.png", dpi=150, bbox_inches="tight"); plt.show()
print("  Saved: phase1_wls_stock1.png  ✓")

print(f"\n  ── PHASE 1 SUMMARY ──")
print(f"  Holdout: OLS QLIKE={qlike(y_te1,ols1_pred):.6f}  WLS QLIKE={qlike(y_te1,wls1_pred):.6f}  α={best_alpha1:.2f}")
if cv1_results:
    print(f"  CV (fixed 80/20): QLIKE={cv1_results[0]['wls_qlike']:.6f}  best α={best_alpha1_cv:.2f}")


# ═══════════════════════════════════════════════════════════════════════
# ██  PHASE 2 — LIQUIDITY CLASSIFICATION  (all stocks)  ████████████████
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "█" * 70)
print("PHASE 2 — LIQUIDITY CLASSIFICATION  (all stocks)")
print("█" * 70)

train_all, _, _, _, _ = train_val_split(df_full)
print(f"  Engineering features on full train ({len(train_all):,} rows) …")
train_all_feat = add_features(train_all)

bas_q33_g = train_all_feat["bas"].quantile(0.33); bas_q66_g = train_all_feat["bas"].quantile(0.66)
act_q33_g = train_all_feat["log_activity"].quantile(0.33); act_q66_g = train_all_feat["log_activity"].quantile(0.66)
print(f"  Global BAS thresholds  : Low < {bas_q33_g:.6f}  ≤  Med < {bas_q66_g:.6f}  ≤  High")
print(f"  Global Activity thresh : Low < {act_q33_g:.4f}   ≤  Med < {act_q66_g:.4f}   ≤  High")

train_all_feat = label_bucket_liquidity(train_all_feat, bas_q33_g, bas_q66_g, act_q33_g, act_q66_g)

stock_liq = (
    train_all_feat.groupby("stock_id").agg(
        median_bas          = ("bas",              "median"),
        median_log_activity = ("log_activity",     "median"),
        median_rv           = ("rv",               "median"),
        mean_rv             = ("rv",               "mean"),
        rv_std              = ("rv",               "std"),
        liquid_pct          = ("bucket_liquidity", lambda x: (x == "liquid").mean()),
        illiquid_pct        = ("bucket_liquidity", lambda x: (x == "illiquid").mean()),
        mixed_pct           = ("bucket_liquidity", lambda x: (x == "mixed").mean()),
        n_buckets           = ("rv",               "count"),
    ).reset_index()
)

def _regime(row):
    if row["liquid_pct"]   >= 0.40: return "liquid"
    if row["illiquid_pct"] >= 0.40: return "illiquid"
    return "mixed"
def _model(regime): return "EGARCH-X" if regime == "liquid" else "WLS / HAR-RV"

stock_liq["stock_regime"]      = stock_liq.apply(_regime, axis=1)
stock_liq["recommended_model"] = stock_liq["stock_regime"].apply(_model)
rc = stock_liq["stock_regime"].value_counts()
print("\n  Stock-level regime counts:")
for r, c in rc.items():
    print(f"    {r:<10}: {c:>4} stocks ({100*c/len(stock_liq):.1f}%)")

def sample_stocks(stock_liq_df, regime, n, seed):
    pool = stock_liq_df[stock_liq_df["stock_regime"] == regime]["stock_id"].tolist()
    n    = min(n, len(pool))
    return sorted(np.random.default_rng(seed).choice(pool, size=n, replace=False).tolist())

liquid_stocks   = sample_stocks(stock_liq, "liquid",   N_DEMO_PER_REGIME, RANDOM_SEED)
mixed_stocks    = sample_stocks(stock_liq, "mixed",    N_DEMO_PER_REGIME, RANDOM_SEED + 1)
illiquid_stocks = sample_stocks(stock_liq, "illiquid", N_DEMO_PER_REGIME, RANDOM_SEED + 2)
demo_stocks     = liquid_stocks + mixed_stocks + illiquid_stocks

print(f"\n  ── DEMO STOCK SELECTION (random, seed={RANDOM_SEED}) ──")
print(f"  20 RANDOM LIQUID   stock_ids : {liquid_stocks}")
print(f"  20 RANDOM MIXED    stock_ids : {mixed_stocks}")
print(f"  20 RANDOM ILLIQUID stock_ids : {illiquid_stocks}")
print(f"  Total demo stocks: {len(demo_stocks)}")

for label, sids in [("LIQUID", liquid_stocks), ("MIXED", mixed_stocks), ("ILLIQUID", illiquid_stocks)]:
    tbl = stock_liq[stock_liq["stock_id"].isin(sids)].sort_values("stock_id")
    print(f"\n  ── {label} (n={len(sids)}) ──")
    print(tbl[["stock_id","liquid_pct","illiquid_pct","mixed_pct",
               "median_bas","median_log_activity","recommended_model"]].to_string(index=False))

stock_liq.to_csv(OUT + "m2_stock_liquidity_profile.csv", index=False)
print(f"\n  Saved: m2_stock_liquidity_profile.csv")

fig, axes = plt.subplots(1, 3, figsize=(22, 7)); fig.patch.set_facecolor(C_BG)
fig.suptitle("PHASE 2 — Per-Stock Liquidity Classification  (all stocks)\n"
             "Each dot = one stock  |  Teal = liquid  |  Gold = mixed  |  Red = illiquid",
             fontsize=12, fontweight="bold", color=C_BLUE)
for regime, color in STOCK_COLORS.items():
    sub = stock_liq[stock_liq["stock_regime"] == regime]
    axes[0].scatter(sub["median_bas"], sub["median_log_activity"], color=color, alpha=0.85,
                    s=60, edgecolors="white", linewidths=0.5, label=f"{regime} (n={len(sub)})", zorder=3)
for sids, color, marker in [(liquid_stocks, C_TEAL, "★"),
                              (mixed_stocks,    C_GOLD, "●"),
                              (illiquid_stocks, C_RED,  "▲")]:
    for sid in sids[:5]:
        row = stock_liq[stock_liq["stock_id"] == sid].iloc[0]
        axes[0].annotate(f"S{int(sid)}", xy=(row["median_bas"], row["median_log_activity"]),
                         xytext=(2,2), textcoords="offset points", fontsize=6, color=color)
axes[0].set_title("All Stocks: Median BAS vs Activity"); axes[0].legend(fontsize=9)
axes[0].set_xlabel("Median Bid-Ask Spread"); axes[0].set_ylabel("Median log(Activity)")
liq_labels_p2 = ["liquid", "mixed", "illiquid"]
axes[1].bar(liq_labels_p2, [rc.get(l,0) for l in liq_labels_p2],
            color=[STOCK_COLORS[l] for l in liq_labels_p2], alpha=0.85, edgecolor="none")
axes[1].set_title("Stock Regime Distribution"); axes[1].set_ylabel("Number of stocks")
for i, l in enumerate(liq_labels_p2):
    cnt = rc.get(l, 0)
    axes[1].text(i, cnt + len(stock_liq)*0.01, f"{100*cnt/len(stock_liq):.1f}%",
                 ha="center", fontsize=9)
ranked = stock_liq.sort_values("liquid_pct", ascending=False).reset_index(drop=True)
bar_c  = [C_TEAL if r["stock_id"] in liquid_stocks else
           C_GOLD if r["stock_id"] in mixed_stocks  else
           C_RED  if r["stock_id"] in illiquid_stocks else C_LGREY
           for _, r in ranked.iterrows()]
axes[2].bar(range(len(ranked)), ranked["liquid_pct"].values, color=bar_c, alpha=0.85, edgecolor="none", width=1.0)
axes[2].set_title("Liquid % per Stock (ranked)"); axes[2].set_xlabel("Stock rank"); axes[2].set_ylabel("liquid_pct")
axes[2].legend(handles=[mpatches.Patch(color=C_TEAL,  label="20 random liquid"),
                         mpatches.Patch(color=C_GOLD,  label="20 random mixed"),
                         mpatches.Patch(color=C_RED,   label="20 random illiquid"),
                         mpatches.Patch(color=C_LGREY, label="Other stocks")], fontsize=8)
plt.tight_layout()
plt.savefig(OUT + "phase2_all_stock_liquidity.png", dpi=150, bbox_inches="tight"); plt.show()
print("  Saved: phase2_all_stock_liquidity.png  ✓")


# ═══════════════════════════════════════════════════════════════════════
# ██  PHASE 3 — WLS + CV ON 60 DEMO STOCKS  ███████████████████████████
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "█" * 70)
print(f"PHASE 3 — WLS + CV ON 60 DEMO STOCKS")
print(f"  {N_DEMO_PER_REGIME} random liquid  +  {N_DEMO_PER_REGIME} random mixed  +  {N_DEMO_PER_REGIME} random illiquid")
print(f"  Using {len(FINAL_FEATURES)} features selected in Phase 1")
print(f"  CV strategy: fixed 80/20 split  "
      f"(train buckets {CV_TRAIN_BUCKETS[0]}–{CV_TRAIN_BUCKETS[-1]}, "
      f"val buckets {CV_VAL_BUCKETS[0]}–{CV_VAL_BUCKETS[-1]})")
print("█" * 70)

df_demo = df_full[df_full["stock_id"].isin(demo_stocks)].copy()
train_d, test_d, sorted_bd, n_train_d, train_max_d = train_val_split(df_demo)
print("  Engineering features …")
train_d = add_features(train_d);  test_d = add_features(test_d)
train_d = label_bucket_liquidity(train_d, bas_q33_g, bas_q66_g, act_q33_g, act_q66_g)
test_d  = label_bucket_liquidity(test_d,  bas_q33_g, bas_q66_g, act_q33_g, act_q66_g)
regime_map = stock_liq.set_index("stock_id")["stock_regime"].to_dict()
train_d["stock_regime"] = train_d["stock_id"].map(regime_map)
test_d["stock_regime"]  = test_d["stock_id"].map(regime_map)

tr_dc = train_d.dropna(subset=FINAL_FEATURES + [TARGET])
te_dc = test_d.dropna(subset=FINAL_FEATURES + [TARGET])
X_trd, y_trd = tr_dc[FINAL_FEATURES].values, tr_dc[TARGET].values
X_ted, y_ted = te_dc[FINAL_FEATURES].values, te_dc[TARGET].values
bv_d         = tr_dc["time_bucket"].values

olsd      = LinearRegression().fit(X_trd, y_trd)
olsd_pred = np.maximum(olsd.predict(X_ted), 1e-8)

print("  Tuning alpha on holdout split …")
best_alpha_d, tune_df_d = tune_alpha(X_trd, y_trd, X_ted, y_ted, train_max_d, bv_d)
print(f"  Holdout best α = {best_alpha_d:.2f}")

wd        = best_alpha_d ** (train_max_d - bv_d)
wlsd      = LinearRegression().fit(X_trd, y_trd, sample_weight=wd)
wlsd_pred = np.maximum(wlsd.predict(X_ted), 1e-8)
print(f"  Holdout: OLS QLIKE={qlike(y_ted,olsd_pred):.6f}  WLS QLIKE={qlike(y_ted,wlsd_pred):.6f}")

# ── PHASE 3 CV (fixed 80/20 bucket split, pooled) ─────────────────────
print(f"\n  Running fixed 80/20 CV on 60 demo stocks (pooled) …")
cv3_results, best_alpha3_cv = wls_cv_on_df(train_d, FINAL_FEATURES, TARGET)
print_cv_summary(cv3_results, label="Phase 3 — 60 demo stocks (pooled)")
plot_cv_results(cv3_results,
                title="PHASE 3 — Fixed 80/20 CV  (60 demo stocks pooled, buckets 1–16 train | 17–20 val)",
                save_path=OUT + "phase3_cv_results.png")

# ── PER-STOCK CV (one fixed split per stock, parallelised) ────────────
print(f"\n  Running per-stock fixed 80/20 CV …")

def _per_stock_cv(sid, df_all_feat, features, target):
    sub = df_all_feat[df_all_feat["stock_id"] == sid].copy()
    if len(sub.dropna(subset=features + [target])) < 10:
        return None
    cv_res, best_a = wls_cv_on_df(sub, features, target)
    if not cv_res:
        return None
    r = cv_res[0]
    return {
        "stock_id":      sid,
        "stock_regime":  regime_map.get(sid, "unknown"),
        "cv_mean_qlike": r["wls_qlike"],
        "cv_std_qlike":  0.0,          # single split → no std
        "cv_mean_alpha": r["best_alpha"],
        "n_folds":       1,
    }

per_stock_cv_raw = Parallel(n_jobs=-1)(
    [delayed(_per_stock_cv)(sid, train_d, FINAL_FEATURES, TARGET)
     for sid in demo_stocks]
)
per_stock_cv = pd.DataFrame([r for r in per_stock_cv_raw if r is not None])

print(f"\n  ── PER-STOCK CV SUMMARY ──")
for regime in ["liquid", "mixed", "illiquid"]:
    sub = per_stock_cv[per_stock_cv["stock_regime"] == regime]
    if len(sub) == 0:
        continue
    print(f"  {regime.upper()} ({len(sub)} stocks):  "
          f"mean CV QLIKE={sub['cv_mean_qlike'].mean():.6f}  "
          f"std={sub['cv_mean_qlike'].std():.6f}  "
          f"mean α={sub['cv_mean_alpha'].mean():.3f}")

# ── HOLDOUT EVALUATION (per stock) ────────────────────────────────────
te_eval = te_dc.copy()
te_eval["ols_pred"]     = olsd_pred
te_eval["wls_pred"]     = wlsd_pred
te_eval["wls_resid"]    = y_ted - wlsd_pred
te_eval["stock_regime"] = te_eval["stock_id"].map(regime_map)

def _stock_metrics(grp):
    y = grp[TARGET].values; yhat = grp["wls_pred"].values
    return pd.Series({
        "n_obs":       len(grp),
        "wls_qlike":   qlike(y, yhat),
        "wls_mse":     mse(y, yhat),
        "wls_mae":     mae(y, yhat),
        "ols_qlike":   qlike(y, grp["ols_pred"].values),
        "mean_rv":     float(np.mean(y)),
        "mean_bas":    float(grp["bas"].mean()),
        "stock_regime": grp["stock_regime"].iloc[0],
    })

per_stock = (te_eval.groupby("stock_id").apply(_stock_metrics)
             .reset_index().sort_values("wls_qlike"))

if len(per_stock_cv) > 0:
    per_stock = per_stock.merge(
        per_stock_cv[["stock_id","cv_mean_qlike","cv_std_qlike","cv_mean_alpha"]],
        on="stock_id", how="left")

print(f"\n  ── PER-STOCK HOLDOUT RESULTS ──")
for _, row in per_stock.iterrows():
    cv_q = f"  CV_QLIKE={row['cv_mean_qlike']:.4f}" if "cv_mean_qlike" in row and pd.notna(row.get("cv_mean_qlike")) else ""
    print(f"  S{int(row['stock_id']):<4} {row['stock_regime']:<10} "
          f"WLS={row['wls_qlike']:.6f}  OLS={row['ols_qlike']:.6f}  "
          f"BAS={row['mean_bas']:.6f}{cv_q}")

stab_rows = []
for regime in ["liquid", "mixed", "illiquid"]:
    sub = per_stock[per_stock["stock_regime"] == regime]
    if len(sub) == 0:
        continue
    stab_rows.append({"regime": regime, "n_stocks": len(sub),
                      "median_QLIKE": sub["wls_qlike"].median(),
                      "std_QLIKE": sub["wls_qlike"].std()})
    print(f"\n  {regime.upper()}: median QLIKE={stab_rows[-1]['median_QLIKE']:.6f}  "
          f"std={stab_rows[-1]['std_QLIKE']:.6f}  n={len(sub)}")

stab_df  = pd.DataFrame(stab_rows)
liq_std  = stab_df.loc[stab_df["regime"]=="liquid",   "std_QLIKE"].values[0] if "liquid"   in stab_df["regime"].values else np.nan
mix_std  = stab_df.loc[stab_df["regime"]=="mixed",    "std_QLIKE"].values[0] if "mixed"    in stab_df["regime"].values else np.nan
ilq_std  = stab_df.loc[stab_df["regime"]=="illiquid", "std_QLIKE"].values[0] if "illiquid" in stab_df["regime"].values else np.nan

# ── PHASE 3 PLOTS ──────────────────────────────────────────────────────
cv3_qlike_str = f"{cv3_results[0]['wls_qlike']:.4f}" if cv3_results else "N/A"
fig, axes = plt.subplots(2, 3, figsize=(24, 14)); fig.patch.set_facecolor(C_BG)
fig.suptitle(f"PHASE 3 — WLS on 60 Demo Stocks  (20 liquid + 20 mixed + 20 illiquid)\n"
             f"{len(FINAL_FEATURES)} data-driven features  |  α={best_alpha_d:.2f}  |  "
             f"Overall WLS QLIKE={qlike(y_ted,wlsd_pred):.4f}  |  "
             f"CV QLIKE={cv3_qlike_str}",
             fontsize=12, fontweight="bold", color=C_BLUE)

sorted_ps = per_stock.sort_values(["stock_regime","wls_qlike"])
x_pos     = np.arange(len(sorted_ps))
axes[0,0].bar(x_pos, sorted_ps["wls_qlike"].values,
              color=[STOCK_COLORS[r] for r in sorted_ps["stock_regime"]],
              alpha=0.82, edgecolor="none", width=0.8)
axes[0,0].set_xticks(x_pos)
axes[0,0].set_xticklabels([f"S{int(s)}" for s in sorted_ps["stock_id"]], rotation=90, fontsize=5)
axes[0,0].set_title("WLS QLIKE per Stock  (grouped by regime)", fontsize=10)
axes[0,0].set_ylabel("QLIKE"); axes[0,0].set_facecolor(C_BG)
axes[0,0].legend(handles=[mpatches.Patch(color=STOCK_COLORS[r], label=r)
                           for r in ["liquid","mixed","illiquid"]], fontsize=8)

bp = axes[0,1].boxplot(
    [per_stock.loc[per_stock["stock_regime"]=="liquid",   "wls_qlike"].values,
     per_stock.loc[per_stock["stock_regime"]=="mixed",    "wls_qlike"].values,
     per_stock.loc[per_stock["stock_regime"]=="illiquid", "wls_qlike"].values],
    labels=["Liquid", "Mixed", "Illiquid"],
    patch_artist=True, widths=0.45, medianprops=dict(color="white", linewidth=2))
for box, color in zip(bp["boxes"], [C_TEAL, C_GOLD, C_RED]):
    box.set_facecolor(color); box.set_alpha(0.75)
axes[0,1].set_title("QLIKE Stability by Regime\n(Narrow box = more stable)", fontsize=10)
axes[0,1].set_ylabel("QLIKE"); axes[0,1].set_facecolor(C_BG)

for regime, color in [("liquid", C_TEAL), ("mixed", C_GOLD), ("illiquid", C_RED)]:
    sub = per_stock[per_stock["stock_regime"] == regime]
    axes[0,2].scatter(sub["ols_qlike"], sub["wls_qlike"], color=color, alpha=0.85,
                      s=60, edgecolors="white", linewidths=0.5, label=regime, zorder=3)
maxq = max(per_stock[["ols_qlike","wls_qlike"]].abs().max())
lim  = (-abs(maxq)*1.1, abs(maxq)*1.1)
axes[0,2].plot([lim[0],lim[1]],[lim[0],lim[1]], "k--", lw=1, label="OLS = WLS")
axes[0,2].set_title("OLS vs WLS QLIKE per Stock", fontsize=10)
axes[0,2].set_xlabel("OLS QLIKE"); axes[0,2].set_ylabel("WLS QLIKE")
axes[0,2].legend(fontsize=8); axes[0,2].set_facecolor(C_BG)

for ax, regime, sids, color in [
    (axes[1,0], "liquid",   liquid_stocks,   C_TEAL),
    (axes[1,1], "illiquid", illiquid_stocks, C_RED),
]:
    mask   = te_eval["stock_id"].isin(sids)
    resids = te_eval.loc[mask, "wls_resid"].dropna()
    ax.hist(resids, bins=100, color=color, alpha=0.75, edgecolor="none", density=True)
    ax.axvline(0, color="black", linestyle="--", lw=1)
    ax.axvline(resids.mean(), color=C_ORANGE, linestyle="-", lw=1.2, label=f"Mean={resids.mean():.2e}")
    ax.set_title(f"WLS Residuals — {regime.capitalize()} Stocks\nStd={resids.std():.4f}  Skew={resids.skew():.2f}",
                 fontsize=10)
    ax.set_xlabel("Residual"); ax.set_ylabel("Density"); ax.legend(fontsize=8); ax.set_facecolor(C_BG)

ax_stab = axes[1,2]; ax_stab.set_facecolor(C_BG)
for i, (regime, color) in enumerate([("liquid", C_TEAL), ("mixed", C_GOLD), ("illiquid", C_RED)]):
    sub = per_stock[per_stock["stock_regime"] == regime]
    if len(sub) == 0:
        continue
    ax_stab.bar(i, sub["wls_qlike"].mean(), color=color, alpha=0.8, edgecolor="none", width=0.5, label=regime)
    ax_stab.errorbar(i, sub["wls_qlike"].mean(), yerr=sub["wls_qlike"].std(),
                     fmt="none", color="black", capsize=5, lw=1.5)
ax_stab.set_xticks([0,1,2]); ax_stab.set_xticklabels(["Liquid", "Mixed", "Illiquid"])
ax_stab.set_title("Mean WLS QLIKE ± Std\nError bars = stability", fontsize=10)
ax_stab.set_ylabel("Mean QLIKE"); ax_stab.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT + "phase3_scenario_comparison.png", dpi=150, bbox_inches="tight"); plt.show()
print("  Saved: phase3_scenario_comparison.png  ✓")

if len(per_stock_cv) > 0 and "cv_mean_qlike" in per_stock.columns:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6)); fig.patch.set_facecolor(C_BG)
    fig.suptitle("PHASE 3 — Holdout vs CV QLIKE per Stock", fontsize=12, fontweight="bold", color=C_BLUE)
    for regime, color in [("liquid", C_TEAL), ("mixed", C_GOLD), ("illiquid", C_RED)]:
        sub = per_stock[per_stock["stock_regime"] == regime].dropna(subset=["cv_mean_qlike"])
        if len(sub) == 0: continue
        axes[0].scatter(sub["wls_qlike"], sub["cv_mean_qlike"], color=color, alpha=0.85,
                        s=70, edgecolors="white", linewidths=0.5, label=regime, zorder=3)
        for _, row in sub.iterrows():
            axes[0].annotate(f"S{int(row['stock_id'])}", xy=(row["wls_qlike"], row["cv_mean_qlike"]),
                             xytext=(2,2), textcoords="offset points", fontsize=5, color=color)
    min_q = per_stock[["wls_qlike","cv_mean_qlike"]].min().min()
    max_q = per_stock[["wls_qlike","cv_mean_qlike"]].max().max()
    axes[0].plot([min_q, max_q],[min_q, max_q], "k--", lw=1, label="Holdout = CV")
    axes[0].set_title("Holdout QLIKE vs CV QLIKE\n(Close to diagonal = CV is a reliable estimate)")
    axes[0].set_xlabel("Holdout WLS QLIKE"); axes[0].set_ylabel("CV WLS QLIKE")
    axes[0].legend(fontsize=8); axes[0].set_facecolor(C_BG)
    for regime, color in [("liquid", C_TEAL), ("mixed", C_GOLD), ("illiquid", C_RED)]:
        sub = per_stock[per_stock["stock_regime"] == regime].dropna(subset=["cv_mean_qlike"])
        if len(sub) == 0: continue
        axes[1].bar(sub.index, sub["cv_mean_qlike"], color=color, alpha=0.75, edgecolor="none", label=regime)
    axes[1].set_title("CV QLIKE per Stock by Regime\n(fixed 80/20 split)")
    axes[1].set_xlabel("Stock (index)"); axes[1].set_ylabel("CV QLIKE")
    axes[1].legend(fontsize=8); axes[1].set_facecolor(C_BG)
    plt.tight_layout()
    plt.savefig(OUT + "phase3_cv_vs_holdout.png", dpi=150, bbox_inches="tight"); plt.show()
    print("  Saved: phase3_cv_vs_holdout.png  ✓")

fig, axes = plt.subplots(1, 3, figsize=(22, 7)); fig.patch.set_facecolor(C_BG)
fig.suptitle("PHASE 3 — BAS & Activity Drive Model Stability", fontsize=12, fontweight="bold", color=C_BLUE)
for regime, color in [("liquid", C_TEAL), ("mixed", C_GOLD), ("illiquid", C_RED)]:
    sub = per_stock[per_stock["stock_regime"] == regime]
    axes[0].scatter(sub["mean_bas"], sub["wls_qlike"], color=color, alpha=0.85,
                    s=70, edgecolors="white", linewidths=0.5, label=regime, zorder=3)
x_all = per_stock["mean_bas"].values; y_all = per_stock["wls_qlike"].values
m, b  = np.polyfit(x_all, y_all, 1)
xr    = np.linspace(x_all.min(), x_all.max(), 100)
axes[0].plot(xr, m*xr+b, color=C_GREY, linestyle="--", lw=1.2,
             label=f"Trend (r={np.corrcoef(x_all,y_all)[0,1]:.2f})")
axes[0].set_title("Mean BAS vs WLS QLIKE\nWider spread → harder to forecast", fontsize=10)
axes[0].set_xlabel("Mean BAS"); axes[0].set_ylabel("WLS QLIKE"); axes[0].legend(fontsize=8); axes[0].set_facecolor(C_BG)
for regime, color in [("liquid", C_TEAL), ("mixed", C_GOLD), ("illiquid", C_RED)]:
    sub = per_stock[per_stock["stock_regime"] == regime]
    axes[1].scatter(sub["mean_rv"], sub["wls_qlike"], color=color, alpha=0.85,
                    s=70, edgecolors="white", linewidths=0.5, label=regime, zorder=3)
axes[1].set_title("Mean RV vs WLS QLIKE\nIlliquid → higher vol → harder to forecast", fontsize=10)
axes[1].set_xlabel("Mean RV"); axes[1].set_ylabel("WLS QLIKE"); axes[1].legend(fontsize=8); axes[1].set_facecolor(C_BG)
for metric, color, label in [("qlike", C_BLUE, "QLIKE"), ("mse", C_ORANGE, "MSE")]:
    axes[2].plot(tune_df_d["alpha"], tune_df_d[metric]/tune_df_d[metric].max(),
                 color=color, lw=1.5, label=f"{label} (normalised)")
axes[2].axvline(best_alpha_d, color="red", linestyle="--", lw=1.3, label=f"Holdout α={best_alpha_d:.2f}")
axes[2].axvline(best_alpha3_cv, color=C_GOLD, linestyle=":", lw=1.3, label=f"CV α={best_alpha3_cv:.2f}")
axes[2].set_title(f"Alpha Tuning  (60-stock demo)", fontsize=10)
axes[2].set_xlabel("Alpha"); axes[2].set_ylabel("Normalised metric"); axes[2].legend(fontsize=8); axes[2].set_facecolor(C_BG)
plt.tight_layout()
plt.savefig(OUT + "phase3_bas_qlike_tradeoff.png", dpi=150, bbox_inches="tight"); plt.show()
print("  Saved: phase3_bas_qlike_tradeoff.png  ✓")

# ── SAVE CSVs ──────────────────────────────────────────────────────────
te_eval[["stock_id","time_id","time_bucket",TARGET,"wls_pred","wls_resid",
          "bucket_liquidity","stock_regime"]].rename(
    columns={TARGET:"actual_rv","wls_pred":"predicted_rv","wls_resid":"residual"}
).assign(model=f"WLS_alpha_{best_alpha_d:.2f}").to_csv(OUT + "m2_wls_predictions_demo60.csv", index=False)
per_stock.to_csv(OUT + "m2_per_stock_eval_demo60.csv", index=False)
if len(per_stock_cv) > 0:
    per_stock_cv.to_csv(OUT + "m2_per_stock_cv_demo60.csv", index=False)
print("  Saved: m2_wls_predictions_demo60.csv  ✓")
print("  Saved: m2_per_stock_eval_demo60.csv   ✓")
if len(per_stock_cv) > 0:
    print("  Saved: m2_per_stock_cv_demo60.csv    ✓")

# ── FINAL SUMMARY ──────────────────────────────────────────────────────
cv1_qlike = cv1_results[0]['wls_qlike'] if cv1_results else float('nan')
cv3_qlike = cv3_results[0]['wls_qlike'] if cv3_results else float('nan')

print(f"""
{"█" * 70}
FINAL SUMMARY
{"█" * 70}

  ── CONFIGURATION ──
  Random seed       : {RANDOM_SEED}
  Time ID subsample : {N_TIME_IDS} randomly sampled time_ids (of {len(all_time_ids)} total)
  Sampled time_ids  : {sampled_time_ids}
  CV strategy       : Fixed 80/20 split — train buckets {CV_TRAIN_BUCKETS[0]}–{CV_TRAIN_BUCKETS[-1]} (8 min)
                      val buckets {CV_VAL_BUCKETS[0]}–{CV_VAL_BUCKETS[-1]} (2 min)
  Stocks per regime : {N_DEMO_PER_REGIME} random

  ── DEMO STOCKS ──
  Liquid   ({len(liquid_stocks):>2}): {liquid_stocks}
  Mixed    ({len(mixed_stocks):>2}): {mixed_stocks}
  Illiquid ({len(illiquid_stocks):>2}): {illiquid_stocks}

  ── FEATURE SELECTION (stock_id=1 training data) ──
  Candidate pool    : {len(ALL_FEATURES)} features
  |r| threshold     : {MIN_CORR_THRESHOLD}
  Collinearity cap  : {COLLINEARITY_THRESHOLD}
  Final features    : {len(FINAL_FEATURES)} → {FINAL_FEATURES}

  ── PHASE 1  (stock_id=1) ──
  Holdout: α={best_alpha1:.2f}  OLS_QLIKE={qlike(y_te1,ols1_pred):.6f}  WLS_QLIKE={qlike(y_te1,wls1_pred):.6f}
  CV (fixed 80/20): QLIKE={cv1_qlike:.6f}  best_α={best_alpha1_cv:.2f}

  ── PHASE 2  (all stocks) ──
  liquid={rc.get('liquid',0)}  illiquid={rc.get('illiquid',0)}  mixed={rc.get('mixed',0)}

  ── PHASE 3  (60 stocks) ──
  Holdout: α={best_alpha_d:.2f}  OLS_QLIKE={qlike(y_ted,olsd_pred):.6f}  WLS_QLIKE={qlike(y_ted,wlsd_pred):.6f}
  CV (fixed 80/20): QLIKE={cv3_qlike:.6f}  best_α={best_alpha3_cv:.2f}
  Liquid std={liq_std:.6f}  Mixed std={mix_std:.6f}  Illiquid std={ilq_std:.6f}
""")
print("Done.")
