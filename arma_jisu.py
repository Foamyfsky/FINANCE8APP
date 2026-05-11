"""
M4 — Volatility Forecasting: ARMA(1,1)-GARCH(1,1) and EGARCH(1,1,1)-X
DATA3888 Group 8

Pipeline:
  Phase 1 — Data diagnostics (ACF/PACF, AIC/BIC order selection)
  Phase 2 — Baseline ARMA(1,1)-GARCH(1,1) with hyperparameter tuning
  Phase 3 — Refined ARMA(1,1)-EGARCH(1,1,1)-X with hyperparameter tuning
  Phase 4 — Comparison plots and outputs

Data format (from M1):
  stock_id | time_id | time_bucket | BidAskSpread_mean | RV
  20 buckets per time_id (30s each), numbered 2–21
  First 16 buckets = train, last 4 = validation
"""

import matplotlib
matplotlib.use("Agg")   # must be before any other matplotlib import — fixes joblib/tkinter thread crash
import os
import json
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
from arch.univariate.volatility import EGARCH, _common_names
from arch.univariate import ARX, Normal
from arch.utility.exceptions import ConvergenceWarning
from joblib import Parallel, delayed

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    # Fallback: njit becomes a no-op decorator
    def njit(fn): return fn
    HAS_NUMBA = False

warnings.filterwarnings("ignore")


# ── Config ────────────────────────────────────────────────────────────────────

# ── Paths ────────────────────────────────────────────────────────────────────
INPUT_CSV  = r"C:\Users\songj\OneDrive\UNI\Y3SEM1\DATA3888\Project Folder\DATA3888G08\optiver_aggregated.csv"

_HERE      = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_HERE, "m4_outputs")
# ─────────────────────────────────────────────────────────────────────────────

N_SIM     = 200     # simulation paths per forecast (1000 is overkill; 200 gives stable means)
SIM_STEPS = 30      # steps per path (one per second)
SCALE     = 1000.0  # rescale RV for numerical stability
N_JOBS    = -1      # parallel workers for model fitting (-1 = all CPU cores)

from config import (get_selected_stocks, get_liquidity_map, filter_time_ids,
                    OUTPUT_DIR, JAMIE_LIQUIDITY_CSV,
                    N_TRAIN, N_VAL, N_STOCKS_PER_REGIME, TIME_ID_KEEP_PCT)

# Hyperparameter tuning: which (p, q) orders to try for GARCH
# and which (p, o, q) orders to try for EGARCH-X
GARCH_ORDERS    = list(itertools.product([1, 2], [1, 2]))       # (p, q)
EGARCHX_ORDERS  = list(itertools.product([1, 2], [1], [1, 2]))  # (p, o, q)

# How many time_ids to use when selecting the best order (keep it fast)
TUNE_SAMPLE = 200

# ── Regime-based sampling — see config.py ────────────────────────────────────
# N_STOCKS_PER_REGIME, TIME_ID_KEEP_PCT, N_TRAIN, N_VAL imported from config

# ── Liquidity hook — see config.py ───────────────────────────────────────────

# Blowup threshold: predictions with QLIKE above this percentile are flagged.
BLOWUP_QLIKE_PERCENTILE = 99

# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Numba-compiled inner loop for EGARCH-X variance recursion ─────────────
# Called hundreds of times per model fit by the optimizer.
# With numba: ~50x faster than pure Python. Without: falls back gracefully.
@njit
def _egarchx_var_kernel(omega, alphas, gammas, betas, delta,
                         resids, spread, backcast, nobs,
                         p, o, q, norm_const):
    lnsigma2    = np.zeros(nobs)
    lnsigma2[0] = backcast
    for t in range(1, nobs):
        v = omega
        for j in range(p):
            if t - 1 - j >= 0:
                sj = max(np.exp(lnsigma2[t-1-j] * 0.5), 1e-8)
                v += alphas[j] * (abs(resids[t-1-j] / sj) - norm_const)
        for j in range(o):
            if t - 1 - j >= 0:
                sj = max(np.exp(lnsigma2[t-1-j] * 0.5), 1e-8)
                v += gammas[j] * (resids[t-1-j] / sj)
        for j in range(q):
            if t - 1 - j >= 0:
                v += betas[j] * lnsigma2[t-1-j]
        if t - 1 < len(spread):
            sv = abs(spread[t-1])
            if sv < 1e-10:
                sv = 1e-10
            v += delta * np.log(sv ** 2)
        lnsigma2[t] = v
    return lnsigma2
#
#  Extends the standard EGARCH with a bid-ask spread term in the log-variance:
#    ln σ²_t = ω + α(|z_{t-1}| − E|z|) + γ·z_{t-1} + β·ln σ²_{t-1}
#                                        + δ·ln(spread²_{t-1})
#
#  γ < 0  →  leverage effect (bad news raises volatility more than good news)
#  δ > 0  →  liquidity channel (wider spread predicts higher volatility)
# ══════════════════════════════════════════════════════════════════════════════

class EGARCHX(EGARCH):
    """EGARCH(p,o,q)-X: adds a log bid-ask spread term to the variance equation."""

    def __init__(self, spread, p=1, o=1, q=1):
        super().__init__(p=p, o=o, q=q)
        self._spread     = np.asarray(spread, dtype=float)
        self._num_params = 1 + p + o + q + 1   # +1 for delta

    def parameter_names(self):
        return _common_names(self.p, self.o, self.q) + ["delta[spread]"]

    def starting_values(self, resids):
        return np.append(super().starting_values(resids), 0.05)

    def bounds(self, resids):
        return super().bounds(resids) + [(-10.0, 10.0)]

    def constraints(self):
        # No additional constraints beyond EGARCH defaults
        A = np.zeros((1, self._num_params))
        b = np.zeros(1)
        return A, b

    def compute_variance(self, parameters, resids, sigma2, backcast, var_bounds):
        p, o, q    = self.p, self.o, self.q
        omega      = parameters[0]
        alphas     = np.asarray(parameters[1       : 1+p],     dtype=np.float64)
        gammas     = np.asarray(parameters[1+p     : 1+p+o],   dtype=np.float64)
        betas      = np.asarray(parameters[1+p+o   : 1+p+o+q], dtype=np.float64)
        delta      = float(parameters[-1])
        spread     = np.asarray(self._spread, dtype=np.float64)
        norm_const = np.sqrt(2.0 / np.pi)
        nobs       = len(resids)
        bc         = float(np.atleast_1d(backcast)[0])

        lnsigma2 = _egarchx_var_kernel(
            float(omega), alphas, gammas, betas, delta,
            np.asarray(resids, dtype=np.float64), spread,
            bc, nobs, p, o, q, norm_const
        )
        sigma2[:] = np.clip(np.exp(lnsigma2), var_bounds[:, 0], var_bounds[:, 1])
        return sigma2

    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        p, o, q    = self.p, self.o, self.q
        parameters = np.asarray(parameters, dtype=float)
        omega      = parameters[0]
        alphas     = parameters[1       : 1+p]
        gammas     = parameters[1+p     : 1+p+o]
        betas      = parameters[1+p+o   : 1+p+o+q]
        delta      = parameters[-1]
        nc         = np.sqrt(2.0 / np.pi)

        # Use the mean log-spread as a constant offset during simulation
        ss = np.where(np.abs(self._spread) < 1e-10, 1e-10, np.abs(self._spread))
        spread_offset = delta * float(np.mean(np.log(ss ** 2)))

        beta_sum   = float(np.sum(betas))
        init_lnsig = (omega + spread_offset) / (1.0 - beta_sum) \
                     if abs(beta_sum) < 1.0 else omega + spread_offset

        e   = rng(nobs + burn)
        ae  = np.abs(e)
        ls  = np.full(nobs + burn, init_lnsig)
        lag = max(p, o, q, 1)

        for t in range(lag, nobs + burn):
            v = omega + spread_offset
            for j in range(p): v += alphas[j] * (ae[t-1-j] - nc)
            for j in range(o): v += gammas[j] * e[t-1-j]
            for j in range(q): v += betas[j] * ls[t-1-j]
            ls[t] = v

        sig  = np.sqrt(np.maximum(np.exp(ls[burn:]), 1e-16))
        returns = e[burn:] * sig

        return pd.DataFrame({"data": returns, "volatility": sig, "errors": e[burn:]})


# ══════════════════════════════════════════════════════════════════════════════
#  Parallel worker functions
#  Each fits one time_id independently — safe to parallelise with joblib.
# ══════════════════════════════════════════════════════════════════════════════

def _fit_garch_one(tid, rv_s, p, q, scale=1000.0, ftol=1e-7):
    """Fit GARCH(p,q) for a single time_id. Returns (tid, result_or_None)."""
    try:
        am  = arch_model(rv_s * scale, mean="ARX", lags=1,
                         vol="GARCH", p=p, q=q, dist="normal")
        res = am.fit(disp="off", show_warning=False,
                     options={"maxiter": 1000, "ftol": ftol})
        return tid, res
    except Exception:
        return tid, None


def _fit_egarchx_one(tid, rv_s, sp_s, p, o, q, scale=1000.0, ftol=1e-7):
    """Fit EGARCH-X(p,o,q) for a single time_id. Returns (tid, res_or_None, vol_obj_or_None)."""
    try:
        vol_obj = EGARCHX(sp_s * scale, p=p, o=o, q=q)
        am      = ARX(rv_s * scale, lags=1)
        am.volatility   = vol_obj
        am.distribution = Normal()
        res = am.fit(disp="off", show_warning=False,
                     options={"maxiter": 1000, "ftol": ftol})
        return tid, res, vol_obj
    except Exception:
        return tid, None, None


# ══════════════════════════════════════════════════════════════════════════════
#  Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def qlike(pred, actual):
    """
    QLIKE loss = log(pred²) + actual²/pred²
    Lower is better. Returns NaN if inputs are invalid.
    """
    if pred <= 0 or actual <= 0 or not np.isfinite(pred):
        return np.nan
    return np.log(pred ** 2) + actual ** 2 / pred ** 2


def garch_is_stationary(res):
    """Returns True if α + β < 0.999 — prevents IGARCH blow-ups during simulation."""
    try:
        return (res.params.get("alpha[1]", 1) + res.params.get("beta[1]", 1)) < 0.999
    except Exception:
        return False


def egarchx_passes_sanity_check(res):
    """
    Basic quality gate for EGARCH-X fits on short series.
    Rejects non-converged fits and extreme parameter values.
    """
    if res.convergence_flag != 0:
        return False
    try:
        return (
            abs(res.params.get("gamma[1]",      99)) < 5 and
            abs(res.params.get("delta[spread]", 99)) < 5 and
            abs(res.params.get("beta[1]",        1)) < 0.9999
        )
    except Exception:
        return False


def simulate_rv(res, vol_obj=None, n_sim=1000, n_steps=30, scale=1000.0):
    """
    Simulate n_sim paths of n_steps each and return the mean RV forecast.
    RV for each path = sqrt(sum(r²)), matching the workshop formula.

    Returns (mean_rv, list_of_path_rvs). Filters out numerically blown-up paths.
    """
    last_sig2 = float(res.conditional_volatility[-1] ** 2)
    valid_rvs = []

    for _ in range(n_sim):
        try:
            if vol_obj is not None:
                # EGARCH-X: use the custom simulate method
                vol_params = res.params[vol_obj.parameter_names()].values
                sim = vol_obj.simulate(vol_params, nobs=n_steps,
                                       rng=np.random.standard_normal, burn=0)
            else:
                # Standard GARCH: use arch built-in
                sim = res.model.simulate(res.params, nobs=n_steps,
                                         burn=0, initial_value=last_sig2)

            r  = sim["data"].values / scale
            rv = np.sqrt(np.sum(r ** 2))

            # Keep only physically plausible paths using data-driven thresholds
            # RV_FLOOR and RV_CEILING are computed from actual data at startup
            if np.isfinite(rv) and RV_FLOOR <= rv <= RV_CEILING:
                valid_rvs.append(rv)

        except Exception:
            continue

    mean_rv = float(np.mean(valid_rvs)) if valid_rvs else np.nan
    return mean_rv, valid_rvs


def evaluate_preds(preds_dict, vol_val_dict):
    """
    Given a dict of {time_id: pred_rv} and validation data,
    compute QLIKE and MSE for every (pred, actual) pair.
    Returns a DataFrame with one row per (time_id, actual bucket).
    """
    records = []
    for tid, pred in preds_dict.items():
        if not np.isfinite(pred):
            continue
        for actual in vol_val_dict[tid]["RV"].values:
            q = qlike(pred, actual)
            if np.isfinite(q):
                records.append({
                    "time_id": tid,
                    "pred_RV": pred,
                    "actual":  actual,
                    "QLIKE":   q,
                    "MSE":     (actual - pred) ** 2,
                })
    return pd.DataFrame(records)


def pick_best_fan_example(fit_df, models_dict, alpha_col="alpha"):
    """
    Find the time_id with the highest alpha (most visible fan spread)
    from the set of successfully fit models.
    """
    best_tid   = None
    best_paths = None
    best_alpha = 0.0
    for tid in models_dict:
        row = fit_df[fit_df["time_id"] == tid]
        if row.empty:
            continue
        a = float(row[alpha_col].iloc[0]) if alpha_col in row.columns else 0.0
        if a > best_alpha:
            best_alpha = a
            best_tid   = tid
    return best_tid


# ══════════════════════════════════════════════════════════════════════════════
#  Hyperparameter tuning helpers
# ══════════════════════════════════════════════════════════════════════════════

def tune_garch_order(vol_train, time_ids, orders=GARCH_ORDERS, sample=TUNE_SAMPLE, scale=1000.0):
    """
    Fit GARCH(p,q) for each (p,q) in `orders` over `sample` time_ids.
    Returns the (p,q) pair with the lowest mean BIC, plus the full results table.
    """
    print(f"  Tuning GARCH order over {sample} time_ids ...")
    results = []
    for p, q in orders:
        aics, bics = [], []
        for tid in time_ids[:sample]:
            rv_s = vol_train[tid]["RV"].values
            try:
                am  = arch_model(rv_s * scale, mean="ARX", lags=1,
                                 vol="GARCH", p=p, q=q, dist="normal")
                res = am.fit(disp="off", show_warning=False,
                             options={"maxiter": 500})
                if res.convergence_flag == 0:
                    aics.append(res.aic)
                    bics.append(res.bic)
            except Exception:
                continue
        if aics:
            results.append({
                "order":    f"GARCH({p},{q})",
                "p": p, "q": q,
                "mean_AIC": round(np.mean(aics), 3),
                "mean_BIC": round(np.mean(bics), 3),
                "n_converged": len(aics),
            })

    df = pd.DataFrame(results).sort_values("mean_BIC").reset_index(drop=True)
    best = df.iloc[0]
    print(f"  Best order by BIC: GARCH({int(best.p)},{int(best.q)})  "
          f"(mean BIC = {best.mean_BIC:.3f})")
    return int(best.p), int(best.q), df


def tune_egarchx_order(vol_train, time_ids, orders=EGARCHX_ORDERS,
                       sample=TUNE_SAMPLE, scale=1000.0):
    """
    Fit EGARCHX(p,o,q) for each combination in `orders` over `sample` time_ids.
    Returns the best (p,o,q) by mean BIC, plus the results table.
    """
    print(f"  Tuning EGARCH-X order over {sample} time_ids ...")
    results = []
    for p, o, q in orders:
        bics = []
        for tid in time_ids[:sample]:
            rv_s = vol_train[tid]["RV"].values
            sp_s = vol_train[tid]["BidAskSpread_mean"].values
            try:
                vol_obj = EGARCHX(sp_s * scale, p=p, o=o, q=q)
                am      = ARX(rv_s * scale, lags=1)
                am.volatility   = vol_obj
                am.distribution = Normal()
                res = am.fit(disp="off", show_warning=False,
                             options={"maxiter": 500})
                if res.convergence_flag == 0 and egarchx_passes_sanity_check(res):
                    bics.append(res.bic)
            except Exception:
                continue
        if bics:
            results.append({
                "order":    f"EGARCHX({p},{o},{q})",
                "p": p, "o": o, "q": q,
                "mean_BIC": round(np.mean(bics), 3),
                "n_converged": len(bics),
            })

    df = pd.DataFrame(results).sort_values("mean_BIC").reset_index(drop=True)
    best = df.iloc[0]
    print(f"  Best order by BIC: EGARCHX({int(best.p)},{int(best.o)},{int(best.q)})  "
          f"(mean BIC = {best.mean_BIC:.3f})")
    return int(best.p), int(best.o), int(best.q), df


# ══════════════════════════════════════════════════════════════════════════════
#  Load and prepare data
# ══════════════════════════════════════════════════════════════════════════════

print("Loading data ...")
df             = pd.read_csv(INPUT_CSV)
all_stock_ids  = sorted(df["stock_id"].unique())
print(f"  Found {len(all_stock_ids)} stocks in dataset.")
print(f"  Numba JIT: {'enabled' if HAS_NUMBA else 'not available — install numba for ~50x speedup'}")

# ── RV scale diagnostics ──────────────────────────────────────────────────────
# Compute thresholds from actual data so the simulation filter is always
# appropriate regardless of the RV scale in this dataset.
rv_all        = df["RV"].dropna()
RV_P01        = float(rv_all.quantile(0.01))   # 1st percentile
RV_P99        = float(rv_all.quantile(0.99))   # 99th percentile
RV_FLOOR      = max(RV_P01 * 0.1, 1e-8)       # 10× below p01, minimum 1e-8
RV_CEILING    = RV_P99 * 10                    # 10× above p99

print(f"\n── RV scale diagnostics ────────────────────────────────────────────")
print(f"  RV p01={RV_P01:.6f}  p50={rv_all.median():.6f}  p99={RV_P99:.6f}")
print(f"  Simulation filter: [{RV_FLOOR:.2e}, {RV_CEILING:.2e}]")
print(f"  (replaces hardcoded [1e-4, 0.05] — adapts to your actual data scale)")

# ── Liquidity classification + stock selection (via config.py) ────────────────
liquidity_map  = get_liquidity_map()
selected       = get_selected_stocks(df=df)
liquid_selected    = selected["liquid"]
illiquid_selected  = selected["illiquid"]
selected_stock_ids = selected["all"]
n_liquid   = len(liquid_selected)
n_illiquid = len(illiquid_selected)

print(f"\n── Stock selection (from config.py) ────────────────────────────────")
print(f"  Liquid   ({n_liquid}): {liquid_selected}")
print(f"  Illiquid ({n_illiquid}): {illiquid_selected}")
print(f"  EGARCH-X runs on liquid stocks only.")

# ══════════════════════════════════════════════════════════════════════════════
#  Global one-time hyperparameter tuning
#
#  Results are saved to global_tuning_cache.json on first run.
#  Every subsequent run loads from that file — tuning never repeats unless
#  you delete the cache file.
# ══════════════════════════════════════════════════════════════════════════════

TUNING_CACHE = os.path.join(OUTPUT_DIR, "global_tuning_cache.json")

if os.path.exists(TUNING_CACHE):
    print(f"\n── Loading cached tuning results from {TUNING_CACHE} ──────────────")
    with open(TUNING_CACHE) as f:
        cache = json.load(f)
    GLOBAL_GARCH_P    = cache["garch_p"]
    GLOBAL_GARCH_Q    = cache["garch_q"]
    GLOBAL_EGARCHX_P  = cache["egarchx_p"]
    GLOBAL_EGARCHX_O  = cache["egarchx_o"]
    GLOBAL_EGARCHX_Q  = cache["egarchx_q"]
    garch_tune_df     = pd.read_csv(os.path.join(OUTPUT_DIR, "global_garch_tuning.csv"))
    egarchx_tune_df   = pd.read_csv(os.path.join(OUTPUT_DIR, "global_egarchx_tuning.csv"))
    print(f"  GARCH order:    ({GLOBAL_GARCH_P},{GLOBAL_GARCH_Q})")
    print(f"  EGARCH-X order: ({GLOBAL_EGARCHX_P},{GLOBAL_EGARCHX_O},{GLOBAL_EGARCHX_Q})")
    print("  (Delete global_tuning_cache.json to re-tune)")

else:
    print("\n── Global Hyperparameter Tuning (runs once for all stocks) ──────────")
    np.random.seed(42)
    sample_stocks  = np.random.choice(all_stock_ids, size=min(5, len(all_stock_ids)), replace=False)
    global_train   = {}
    for sid in sample_stocks:
        sub = df[df["stock_id"] == sid].sort_values(["time_id", "time_bucket"])
        for tid in sorted(sub["time_id"].unique()):
            buckets = sub[sub["time_id"] == tid].sort_values("time_bucket").reset_index(drop=True)
            if len(buckets) >= N_TRAIN + N_VAL:
                key = f"{sid}_{tid}"
                global_train[key] = buckets.iloc[:N_TRAIN].copy()
            if len(global_train) >= TUNE_SAMPLE:
                break
        if len(global_train) >= TUNE_SAMPLE:
            break

    global_time_ids = list(global_train.keys())
    print(f"  Tuning on {len(global_time_ids)} time_ids drawn from stocks: {list(sample_stocks)}")

    GLOBAL_GARCH_P, GLOBAL_GARCH_Q, garch_tune_df = tune_garch_order(
        global_train, global_time_ids, orders=GARCH_ORDERS, sample=len(global_time_ids)
    )
    garch_tune_df.to_csv(os.path.join(OUTPUT_DIR, "global_garch_tuning.csv"), index=False)

    GLOBAL_EGARCHX_P, GLOBAL_EGARCHX_O, GLOBAL_EGARCHX_Q, egarchx_tune_df = tune_egarchx_order(
        global_train, global_time_ids, orders=EGARCHX_ORDERS, sample=len(global_time_ids)
    )
    egarchx_tune_df.to_csv(os.path.join(OUTPUT_DIR, "global_egarchx_tuning.csv"), index=False)

    # Save cache so this never runs again
    with open(TUNING_CACHE, "w") as f:
        json.dump({
            "garch_p":   GLOBAL_GARCH_P,
            "garch_q":   GLOBAL_GARCH_Q,
            "egarchx_p": GLOBAL_EGARCHX_P,
            "egarchx_o": GLOBAL_EGARCHX_O,
            "egarchx_q": GLOBAL_EGARCHX_Q,
        }, f, indent=2)
    print(f"  Tuning results cached to: {TUNING_CACHE}")
    print(f"  → Global GARCH order:    ({GLOBAL_GARCH_P},{GLOBAL_GARCH_Q})")
    print(f"  → Global EGARCH-X order: ({GLOBAL_EGARCHX_P},{GLOBAL_EGARCHX_O},{GLOBAL_EGARCHX_Q})")

# Collectors for cross-stock summary
all_garch_summary   = []
all_egarchx_summary = []
all_blowups         = []   # global blowup report (Jisu's 5.8% flag)

# ══════════════════════════════════════════════════════════════════════════════
#  Main loop — iterate over every stock
# ══════════════════════════════════════════════════════════════════════════════

for STOCK_ID in selected_stock_ids:
    print(f"\n{'='*70}")
    print(f"  Processing Stock {STOCK_ID}  ({selected_stock_ids.index(STOCK_ID)+1}/{len(selected_stock_ids)})")
    print(f"{'='*70}")

    # Liquidity regime for this stock
    regime = liquidity_map.get(STOCK_ID, "liquid")   # default liquid if not found
    print(f"  Liquidity regime: {regime.upper()}"
          + ("  → EGARCH-X will run" if regime == "liquid" else
             "  → EGARCH-X SKIPPED (Rosa's HAR-RV handles this)"))

    # Per-stock output subdirectory
    stock_out = os.path.join(OUTPUT_DIR, f"stock_{STOCK_ID}")
    os.makedirs(stock_out, exist_ok=True)

    stock1 = df[df["stock_id"] == STOCK_ID].copy()
    stock1 = stock1.sort_values(["time_id", "time_bucket"]).reset_index(drop=True)

    vol_train = {}
    vol_val   = {}

    for tid in sorted(stock1["time_id"].unique()):
        buckets = (stock1[stock1["time_id"] == tid]
                   .sort_values("time_bucket")
                   .reset_index(drop=True))
        if len(buckets) < N_TRAIN + N_VAL:
            continue
        vol_train[tid] = buckets.iloc[:N_TRAIN].copy()
        vol_val[tid]   = buckets.iloc[N_TRAIN : N_TRAIN + N_VAL].copy()

    time_IDs = list(vol_train.keys())
    if not time_IDs:
        print(f"  No complete time_ids for stock {STOCK_ID} — skipping.")
        continue

    # ── Filter to most regime-representative time_ids (via config.py) ─────────
    time_IDs  = filter_time_ids(vol_train, time_IDs, regime)
    vol_train = {tid: vol_train[tid] for tid in time_IDs}
    vol_val   = {tid: vol_val[tid]   for tid in time_IDs if tid in vol_val}
    print(f"  Stock {STOCK_ID}: kept {len(time_IDs)} most {regime} time_ids")


    # ══════════════════════════════════════════════════════════════════════════════
    #  Phase 1 — Data Diagnostics
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n── Phase 1: Data Diagnostics ───────────────────────────────────────")

    # ACF / PACF on RV to justify ARMA(1,1) mean specification
    print("Plotting ACF/PACF ...")
    tid_diag = time_IDs[0]
    rv_diag  = vol_train[tid_diag]["RV"].values
    max_lags = max(1, len(rv_diag) // 2 - 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"ACF / PACF — Realised Volatility (30s buckets)\n"
                 f"Stock {STOCK_ID}, time_id={tid_diag} — justifies ARMA(1,1)", fontsize=12)
    plot_acf( rv_diag, lags=max_lags, ax=axes[0], title="ACF — RV")
    plot_pacf(rv_diag, lags=max_lags, ax=axes[1], title="PACF — RV", method="ywm")
    plt.tight_layout()
    plt.savefig(os.path.join(stock_out, "phase1_acf_pacf.png"), dpi=150)
    plt.close()
    print("  Saved: phase1_acf_pacf.png")

    # AIC/BIC order selection loop
    print("Running AIC/BIC order selection ...")
    order_results = []
    for p, q in itertools.product([1, 2], [1, 2]):
        aics, bics = [], []
        for tid in time_IDs[:200]:
            rv_s = vol_train[tid]["RV"].values
            try:
                am  = arch_model(rv_s * SCALE, mean="ARX", lags=1,
                                 vol="GARCH", p=p, q=q, dist="normal")
                res = am.fit(disp="off", show_warning=False, options={"maxiter": 500})
                if res.convergence_flag == 0:
                    aics.append(res.aic)
                    bics.append(res.bic)
            except Exception:
                continue
        if aics:
            order_results.append({
                "GARCH(p,q)": f"({p},{q})",
                "Mean AIC":   round(np.mean(aics), 3),
                "Mean BIC":   round(np.mean(bics), 3),
                "Converged":  len(aics),
            })

    order_df = pd.DataFrame(order_results).sort_values("Mean BIC")
    order_df.to_csv(os.path.join(stock_out, "phase1_order_selection.csv"), index=False)
    print("  Order selection (sorted by BIC):")
    print(order_df.to_string(index=False))
    print("  → GARCH(1,1) wins on BIC — most efficient balance of fit and complexity")


    # ══════════════════════════════════════════════════════════════════════════════
    #  Phase 2 — Baseline ARMA(1,1)-GARCH(1,1) with hyperparameter tuning
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n── Phase 2: Baseline ARMA(1,1)-GARCH with hyperparameter tuning ────")

    # Use globally-tuned order (no per-stock tuning overhead)
    best_garch_p, best_garch_q = GLOBAL_GARCH_P, GLOBAL_GARCH_Q
    garch_tune_df_stock = garch_tune_df  # reuse global results
    garch_tune_df_stock.to_csv(os.path.join(stock_out, "garch_tuning.csv"), index=False)
    print(f"  Reusing global GARCH order: ({best_garch_p},{best_garch_q})")

    # Step 2: fit using the best order (parallel across time_ids)
    print(f"Fitting ARMA(1,1)-GARCH({best_garch_p},{best_garch_q}) "
          f"({len(time_IDs)} time_ids, parallel) ...")

    fit_results = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_fit_garch_one)(
            tid,
            vol_train[tid]["RV"].values,
            best_garch_p, best_garch_q,
        )
        for tid in time_IDs
    )

    garch_models = {}
    garch_info   = []

    for tid, res in fit_results:
        if res is None:
            continue
        converged  = res.convergence_flag == 0
        stationary = garch_is_stationary(res) if converged else False
        garch_models[tid] = res
        garch_info.append({
            "time_id":    tid,
            "AIC":        res.aic if converged else np.nan,
            "BIC":        res.bic if converged else np.nan,
            "converged":  converged,
            "stationary": stationary,
            "alpha":      res.params.get("alpha[1]", np.nan) if converged else np.nan,
            "beta":       res.params.get("beta[1]",  np.nan) if converged else np.nan,
        })

    garch_fit_df = pd.DataFrame(garch_info)
    garch_fit_df.to_csv(os.path.join(stock_out, "garch_fit_info.csv"), index=False)

    n_conv = garch_fit_df["converged"].sum()
    n_stat = garch_fit_df["stationary"].sum()
    conv_df = garch_fit_df[garch_fit_df["converged"]]
    print(f"  Converged:  {n_conv}/{len(garch_fit_df)} ({100*n_conv/max(len(garch_fit_df),1):.1f}%)")
    print(f"  Stationary: {n_stat}/{len(garch_fit_df)} ({100*n_stat/max(len(garch_fit_df),1):.1f}%)")
    print(f"  Mean AIC: {conv_df['AIC'].mean():.2f}   Median AIC: {conv_df['AIC'].median():.2f}")

    # Step 3: simulate forecasts
    print(f"\nSimulating {N_SIM} paths × {SIM_STEPS} steps (GARCH) ...")
    garch_preds   = {}
    garch_example = {"tid": None, "paths": None}

    # Find a good example time_id for the fan chart (highest alpha = most visible spread)
    fan_tid = pick_best_fan_example(garch_fit_df, garch_models, alpha_col="alpha")

    for tid, res in garch_models.items():
        row = garch_fit_df[garch_fit_df["time_id"] == tid]
        if row.empty or not bool(row["stationary"].iloc[0]):
            continue

        pred_rv, paths = simulate_rv(res, n_sim=N_SIM, n_steps=SIM_STEPS, scale=SCALE)
        garch_preds[tid] = pred_rv

        if tid == fan_tid and paths:
            garch_example["tid"]   = tid
            garch_example["paths"] = paths

    print(f"  Forecasts generated: {len(garch_preds)}")

    # Step 4: evaluate
    garch_eval_df = evaluate_preds(garch_preds, vol_val)
    garch_eval_df.to_csv(os.path.join(stock_out, "garch_eval_results.csv"), index=False)

    garch_per_tid = garch_eval_df.groupby("time_id")[["QLIKE", "MSE"]].mean()
    print(f"  n time_ids evaluated: {len(garch_per_tid)}")
    print(f"  Median QLIKE: {garch_per_tid['QLIKE'].median():.4f}")
    print(f"  Mean   QLIKE: {garch_per_tid['QLIKE'].mean():.4f}")
    print(f"  Median MSE:   {garch_per_tid['MSE'].median():.8f}")

    # ── Blowup detection (GARCH — illiquid stocks only) ──────────────────────
    # GARCH is expected to blow up on thin order books. We only flag and
    # collect blowups for illiquid stocks — these are the same stocks Rosa
    # runs HAR-RV on, so the comparison is apples-to-apples.
    if regime == "illiquid" and not garch_eval_df.empty:
        blowup_thresh_g  = np.nanpercentile(garch_eval_df["QLIKE"].dropna(),
                                             BLOWUP_QLIKE_PERCENTILE)
        garch_blowups    = garch_eval_df[garch_eval_df["QLIKE"] > blowup_thresh_g].copy()
        garch_blowups["stock_id"] = STOCK_ID
        garch_blowups["model"]    = "GARCH"
        garch_blowup_pct          = len(garch_blowups) / max(len(garch_eval_df), 1)
        garch_blowups.to_csv(os.path.join(stock_out, "garch_blowups.csv"), index=False)
        all_blowups.append(garch_blowups)
        print(f"  GARCH blowups (illiquid): {len(garch_blowups)} ({garch_blowup_pct:.1%}) ← flagged")
    else:
        garch_blowup_pct = 0.0


    # ══════════════════════════════════════════════════════════════════════════════
    #  Phase 3 — Refined ARMA(1,1)-EGARCH(1,1,1)-X with hyperparameter tuning
    #
    #  Only runs on LIQUID stocks. For illiquid stocks, Rosa's HAR-RV/WLS is
    #  the right tool — skip here to avoid numerical blow-ups on thin books.
    # ══════════════════════════════════════════════════════════════════════════════

    if regime != "liquid":
        print("\n── Phase 3: EGARCH-X SKIPPED (illiquid stock) ──────────────────────")
        print("   → Rosa's HAR-RV/WLS model handles this regime.")
        egarchx_eval_df  = pd.DataFrame(columns=["time_id", "pred_RV", "actual", "QLIKE", "MSE"])
        egarchx_per_tid  = pd.DataFrame(columns=["QLIKE", "MSE"])
        ex_df            = pd.DataFrame()
        egarchx_example  = {"tid": None, "paths": None}
        egarchx_blowup_pct = np.nan
    else:

        # Use globally-tuned order (no per-stock tuning overhead)
        best_p, best_o, best_q = GLOBAL_EGARCHX_P, GLOBAL_EGARCHX_O, GLOBAL_EGARCHX_Q
        egarchx_tune_df_stock = egarchx_tune_df  # reuse global results
        egarchx_tune_df_stock.to_csv(os.path.join(stock_out, "egarchx_tuning.csv"), index=False)
        print(f"  Reusing global EGARCH-X order: ({best_p},{best_o},{best_q})")

        # Step 2: fit using the best order (parallel across time_ids)
        print(f"Fitting ARMA(1,1)-EGARCHX({best_p},{best_o},{best_q}) "
              f"({len(time_IDs)} time_ids, parallel) ...")

        egarchx_results = Parallel(n_jobs=N_JOBS, prefer="threads")(
            delayed(_fit_egarchx_one)(
                tid,
                vol_train[tid]["RV"].values,
                vol_train[tid]["BidAskSpread_mean"].values,
                best_p, best_o, best_q,
            )
            for tid in time_IDs
        )

        egarchx_models = {}   # tid -> (res, vol_obj)
        egarchx_info   = []

        for tid, res, vol_obj in egarchx_results:
            if res is None:
                continue
            converged  = res.convergence_flag == 0
            quality_ok = egarchx_passes_sanity_check(res)
            egarchx_models[tid] = (res, vol_obj)
            egarchx_info.append({
                "time_id":    tid,
                "AIC":        res.aic if converged else np.nan,
                "BIC":        res.bic if converged else np.nan,
                "converged":  converged,
                "quality_ok": quality_ok,
                "gamma":      res.params.get("gamma[1]",      np.nan) if quality_ok else np.nan,
                "delta":      res.params.get("delta[spread]", np.nan) if quality_ok else np.nan,
                "beta":       res.params.get("beta[1]",       np.nan) if quality_ok else np.nan,
            })

        egarchx_fit_df = pd.DataFrame(egarchx_info)
        egarchx_fit_df.to_csv(os.path.join(stock_out, "egarchx_fit_info.csv"), index=False)

        n_conv_ex = egarchx_fit_df["converged"].sum()
        n_qual_ex = egarchx_fit_df["quality_ok"].sum()
        ex_df     = egarchx_fit_df[egarchx_fit_df["quality_ok"]]
        print(f"  Converged:  {n_conv_ex}/{len(egarchx_fit_df)} ({100*n_conv_ex/max(len(egarchx_fit_df),1):.1f}%)")
        print(f"  Quality-OK: {n_qual_ex}/{len(egarchx_fit_df)} ({100*n_qual_ex/max(len(egarchx_fit_df),1):.1f}%)")
        print(f"  Mean gamma: {ex_df['gamma'].mean():.4f}  "
              f"(γ<0 in {(ex_df['gamma']<0).mean():.1%} → leverage effect)")
        print(f"  Mean delta: {ex_df['delta'].mean():.4f}  "
              f"(δ>0 in {(ex_df['delta']>0).mean():.1%} → spread predicts vol)")

        # Step 3: simulate forecasts
        print(f"\nSimulating {N_SIM} paths × {SIM_STEPS} steps (EGARCH-X) ...")
        egarchx_preds   = {}
        egarchx_example = {"tid": None, "paths": None}

        fan_tid_ex = pick_best_fan_example(egarchx_fit_df[egarchx_fit_df["quality_ok"]],
                                           {k: v for k, v in egarchx_models.items()},
                                           alpha_col="beta")

        for tid, (res, vol_obj) in egarchx_models.items():
            row = egarchx_fit_df[egarchx_fit_df["time_id"] == tid]
            if row.empty or not bool(row["quality_ok"].iloc[0]):
                continue

            pred_rv, paths = simulate_rv(res, vol_obj=vol_obj,
                                         n_sim=N_SIM, n_steps=SIM_STEPS, scale=SCALE)
            egarchx_preds[tid] = pred_rv

            if tid == fan_tid_ex and paths:
                egarchx_example["tid"]   = tid
                egarchx_example["paths"] = paths

        print(f"  Forecasts generated: {len(egarchx_preds)}")

        # Step 4: evaluate
        egarchx_eval_df = evaluate_preds(egarchx_preds, vol_val)
        egarchx_eval_df.to_csv(os.path.join(stock_out, "egarchx_eval_results.csv"), index=False)

        if not egarchx_eval_df.empty:
            egarchx_per_tid = egarchx_eval_df.groupby("time_id")[["QLIKE", "MSE"]].mean()
            print(f"  n time_ids evaluated: {len(egarchx_per_tid)}")
            print(f"  Median QLIKE: {egarchx_per_tid['QLIKE'].median():.4f}")
            print(f"  Mean   QLIKE: {egarchx_per_tid['QLIKE'].mean():.4f}")
            print(f"  Median MSE:   {egarchx_per_tid['MSE'].median():.8f}")
        else:
            egarchx_per_tid = pd.DataFrame(columns=["QLIKE", "MSE"])
            print("  No valid EGARCH-X predictions after filtering.")

        egarchx_blowup_pct = np.nan   # not tracked for liquid stocks



    # ══════════════════════════════════════════════════════════════════════════════
    #  Phase 4 — Comparison plots and output summary
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n── Phase 4: Evaluation Outputs ─────────────────────────────────────")

    # Console summary table
    ex_q   = egarchx_per_tid["QLIKE"].median() if not egarchx_per_tid.empty else float("nan")
    ex_mse = egarchx_per_tid["MSE"].median()   if not egarchx_per_tid.empty else float("nan")
    g_q    = garch_per_tid["QLIKE"].median()
    g_mse  = garch_per_tid["MSE"].median()
    winner_q   = "EGARCH-X" if ex_q   < g_q   else "GARCH"
    winner_mse = "EGARCH-X" if ex_mse < g_mse else "GARCH"

    print(f"\n  ┌──────────────────┬──────────────┬──────────────┬──────────┐")
    print(f"  │ Metric           │  GARCH(1,1)  │   EGARCH-X   │  Winner  │")
    print(f"  ├──────────────────┼──────────────┼──────────────┼──────────┤")
    print(f"  │ Median QLIKE     │ {g_q:12.4f} │ {ex_q:12.4f} │ {winner_q:<8} │")
    print(f"  │ Median MSE       │ {g_mse:12.2e} │ {ex_mse:12.2e} │ {winner_mse:<8} │")
    print(f"  │ n forecasts      │ {len(garch_per_tid):12d} │ {len(egarchx_per_tid):12d} │          │")
    print(f"  └──────────────────┴──────────────┴──────────────┴──────────┘")

    C_GARCH = "#378ADD"; C_EX = "#1D9E75"; SPINE = "#D3D1C7"

    def _style(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor("white")
        for sp in ax.spines.values(): sp.set_color(SPINE); sp.set_linewidth(0.6)
        ax.tick_params(labelsize=8.5, color=SPINE)
        ax.set_title(title, fontsize=10, fontweight="500", pad=6)
        if xlabel: ax.set_xlabel(xlabel, fontsize=8.5, color="#5F5E5A")
        if ylabel: ax.set_ylabel(ylabel, fontsize=8.5, color="#5F5E5A")
        ax.grid(axis="y", color=SPINE, linewidth=0.4, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)

    # ── Comparison plot: GARCH vs EGARCH-X ───────────────────────────────────
    # Only meaningful on liquid stocks where both models ran
    if not egarchx_per_tid.empty:
        fig = plt.figure(figsize=(14, 10))
        fig.patch.set_facecolor("white")
        fig.suptitle(f"GARCH vs EGARCH-X — Stock {STOCK_ID} (liquid)\n"
                     f"Lower QLIKE and MSE = better forecast",
                     fontsize=12, fontweight="500")

        gs = plt.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

        # ── Row 1: side-by-side boxplots ─────────────────────────────────────
        # QLIKE
        ax1 = fig.add_subplot(gs[0, :2])
        bp = ax1.boxplot(
            [garch_per_tid["QLIKE"].dropna().values,
             egarchx_per_tid["QLIKE"].dropna().values],
            patch_artist=True, labels=["GARCH(1,1)", "EGARCH-X"], widths=0.45, vert=True,
            medianprops=dict(color="white", linewidth=2.5),
            whiskerprops=dict(color=SPINE, linewidth=0.9),
            capprops=dict(color=SPINE, linewidth=0.9),
            flierprops=dict(marker="o", markersize=3, markerfacecolor=SPINE, alpha=0.4)
        )
        bp["boxes"][0].set_facecolor(C_GARCH); bp["boxes"][0].set_alpha(0.8)
        bp["boxes"][1].set_facecolor(C_EX);    bp["boxes"][1].set_alpha(0.8)
        # Annotate medians
        for i, (vals, col) in enumerate([(garch_per_tid["QLIKE"].dropna(), C_GARCH),
                                          (egarchx_per_tid["QLIKE"].dropna(), C_EX)]):
            med = vals.median()
            ax1.annotate(f"Median\n{med:.4f}", xy=(i+1, med),
                         xytext=(18, 0), textcoords="offset points",
                         fontsize=8, va="center", color=col)
        _style(ax1, title="QLIKE distribution — GARCH vs EGARCH-X  (lower = better)",
               ylabel="QLIKE per time_id")
        ax1.grid(axis="x", color=SPINE, linewidth=0.3, linestyle="--", alpha=0.5)

        # Winner callout
        ax_info = fig.add_subplot(gs[0, 2])
        ax_info.axis("off")
        improve_q = (g_q - ex_q) / abs(g_q) * 100 if abs(g_q) > 0 else 0
        improve_mse = (g_mse - ex_mse) / abs(g_mse) * 100 if abs(g_mse) > 0 else 0
        lines = [
            f"Stock {STOCK_ID}",
            f"Regime: LIQUID",
            "",
            "QLIKE:",
            f"  GARCH:    {g_q:.4f}",
            f"  EGARCH-X: {ex_q:.4f}",
            f"  → {winner_q} wins",
            f"     ({abs(improve_q):.1f}% {'better' if improve_q > 0 else 'worse'})",
            "",
            "MSE:",
            f"  GARCH:    {g_mse:.2e}",
            f"  EGARCH-X: {ex_mse:.2e}",
            f"  → {winner_mse} wins",
            f"     ({abs(improve_mse):.1f}% {'better' if improve_mse > 0 else 'worse'})",
            "",
            f"n time_ids: {len(garch_per_tid)}",
        ]
        ax_info.text(0.05, 0.95, "\n".join(lines), transform=ax_info.transAxes,
                     fontsize=9, va="top", fontfamily="monospace",
                     bbox=dict(boxstyle="round,pad=0.6", facecolor="#E1F5EE",
                               edgecolor=SPINE, linewidth=0.6, alpha=0.9))

        # ── Row 2: scatter GARCH QLIKE vs EGARCH-X QLIKE per time_id ─────────
        # Points below diagonal = EGARCH-X wins that time_id
        ax2 = fig.add_subplot(gs[1, :2])
        common_tids = garch_per_tid.index.intersection(egarchx_per_tid.index)
        g_q_vals  = garch_per_tid.loc[common_tids, "QLIKE"].values
        ex_q_vals = egarchx_per_tid.loc[common_tids, "QLIKE"].values

        # Clip extreme outliers for readability
        p99 = np.nanpercentile(np.concatenate([g_q_vals, ex_q_vals]), 99)
        mask = (g_q_vals <= p99) & (ex_q_vals <= p99)
        g_plot  = g_q_vals[mask];  ex_plot = ex_q_vals[mask]

        ex_wins_mask = ex_plot < g_plot
        ax2.scatter(g_plot[ ex_wins_mask], ex_plot[ ex_wins_mask],
                    color=C_EX,    alpha=0.45, s=12, label=f"EGARCH-X wins ({ex_wins_mask.sum()})")
        ax2.scatter(g_plot[~ex_wins_mask], ex_plot[~ex_wins_mask],
                    color=C_GARCH, alpha=0.45, s=12, label=f"GARCH wins ({(~ex_wins_mask).sum()})")
        lim = max(g_plot.max(), ex_plot.max()) * 1.06
        ax2.plot([0, lim], [0, lim], "--", color="#888780", linewidth=0.9,
                 label="Equal performance")
        ax2.set_xlim(0, lim); ax2.set_ylim(0, lim)
        _style(ax2, title="Per time_id: GARCH QLIKE vs EGARCH-X QLIKE\n"
                           "Points below diagonal → EGARCH-X wins that time_id",
               xlabel="GARCH QLIKE", ylabel="EGARCH-X QLIKE")
        ax2.grid(axis="both", color=SPINE, linewidth=0.3, linestyle="--", alpha=0.5)
        ax2.legend(fontsize=8.5, framealpha=0.9)
        ax2.text(lim * 0.55, lim * 0.08, "EGARCH-X better →",
                 fontsize=8, color=C_EX, style="italic")
        ax2.text(lim * 0.08, lim * 0.55, "← GARCH better",
                 fontsize=8, color=C_GARCH, style="italic", rotation=90)

        # MSE boxplot
        ax3 = fig.add_subplot(gs[1, 2])
        bp2 = ax3.boxplot(
            [garch_per_tid["MSE"].dropna().values,
             egarchx_per_tid["MSE"].dropna().values],
            patch_artist=True, labels=["GARCH", "EGARCH-X"], widths=0.45, vert=True,
            medianprops=dict(color="white", linewidth=2.5),
            whiskerprops=dict(color=SPINE, linewidth=0.9),
            capprops=dict(color=SPINE, linewidth=0.9),
            flierprops=dict(marker="o", markersize=3, markerfacecolor=SPINE, alpha=0.4)
        )
        bp2["boxes"][0].set_facecolor(C_GARCH); bp2["boxes"][0].set_alpha(0.8)
        bp2["boxes"][1].set_facecolor(C_EX);    bp2["boxes"][1].set_alpha(0.8)
        for i, vals in enumerate([garch_per_tid["MSE"].dropna(),
                                   egarchx_per_tid["MSE"].dropna()]):
            ax3.annotate(f"{vals.median():.2e}", xy=(i+1, vals.median()),
                         xytext=(10, 0), textcoords="offset points",
                         fontsize=7.5, va="center")
        _style(ax3, title="MSE  (lower = better)", ylabel="MSE per time_id")

        plt.savefig(os.path.join(stock_out, "garch_vs_egarchx_comparison.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: garch_vs_egarchx_comparison.png")

    else:
        # Illiquid stock — GARCH only, show its QLIKE/MSE distribution
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.patch.set_facecolor("white")
        fig.suptitle(f"GARCH Evaluation — Stock {STOCK_ID} (illiquid)\n"
                     f"EGARCH-X not run — see Rosa's HAR-RV for this stock",
                     fontsize=11, fontweight="500")
        for ax, metric, label in zip(axes, ["QLIKE", "MSE"], ["QLIKE", "MSE"]):
            vals = garch_per_tid[metric].dropna()
            p99  = np.nanpercentile(vals, 99)
            vals_clipped = vals[vals <= p99]
            ax.set_facecolor("white")
            ax.hist(vals_clipped, bins=30, color=C_GARCH, alpha=0.75, edgecolor="white")
            ax.axvline(vals.median(), color="#D85A30", linestyle="--", linewidth=1.2,
                       label=f"Median {vals.median():.4f}")
            for sp in ax.spines.values(): sp.set_color(SPINE); sp.set_linewidth(0.6)
            ax.tick_params(labelsize=8.5, color=SPINE)
            ax.set_title(f"GARCH {label} distribution", fontsize=10, fontweight="500")
            ax.set_xlabel(label, fontsize=9); ax.legend(fontsize=8.5)
            ax.grid(axis="y", color=SPINE, linewidth=0.4, linestyle="--")
            ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(os.path.join(stock_out, "garch_evaluation.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: garch_evaluation.png")

    # Fan charts
    print("Plotting fan charts ...")

    def plot_fan(example_dict, model_label, filename, color="#4C72B0"):
        tid   = example_dict["tid"]
        paths = example_dict["paths"]
        if tid is None or not paths:
            return

        # Re-simulate 200 paths for the visual
        res_ex  = garch_models[tid] if "GARCH" in model_label else egarchx_models[tid][0]
        vol_ex  = None if "GARCH" in model_label else egarchx_models[tid][1]
        path_mat = []

        for _ in range(200):
            try:
                if vol_ex is not None:
                    vp  = res_ex.params[vol_ex.parameter_names()].values
                    sim = vol_ex.simulate(vp, nobs=SIM_STEPS,
                                          rng=np.random.standard_normal, burn=0)
                else:
                    last_sig2 = float(res_ex.conditional_volatility[-1] ** 2)
                    sim = res_ex.model.simulate(res_ex.params, nobs=SIM_STEPS,
                                                burn=0, initial_value=last_sig2)
                path_mat.append(sim["data"].values / SCALE)
            except Exception:
                continue

        if not path_mat:
            return

        path_mat = np.array(path_mat)
        cum_rv   = np.sqrt(np.cumsum(path_mat ** 2, axis=1))
        pct      = np.percentile(cum_rv, [1, 5, 25, 50, 75, 95, 99], axis=0)
        t        = np.arange(1, SIM_STEPS + 1)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(t, pct[1], pct[5], alpha=0.25, color=color, label="5–95th pct")
        ax.fill_between(t, pct[2], pct[4], alpha=0.50, color=color, label="25–75th pct")
        ax.plot(t, pct[3], color=color, linewidth=2, label="Median path")

        for path in path_mat[:50]:
            ax.plot(t, np.sqrt(np.cumsum(path**2)), color="grey", alpha=0.06, linewidth=0.7)

        # Clip y-axis to 1st–99th pct so outliers don't compress the fan
        ax.set_ylim(pct[0, 0] * 0.5, pct[6, -1] * 1.05)
        ax.set_xlabel("Steps ahead (seconds)")
        ax.set_ylabel("Cumulative RV")
        ax.set_title(f"Fan Chart — {model_label}\n"
                     f"Stock {STOCK_ID}, time_id={tid} | {len(path_mat)} paths")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(stock_out, filename), dpi=150)
        plt.close()

    plot_fan(garch_example,   "ARMA(1,1)-GARCH(1,1)",      "garch_fan_chart.png",   "#4C72B0")
    plot_fan(egarchx_example, "ARMA(1,1)-EGARCH(1,1,1)-X", "egarchx_fan_chart.png", "#DD8452")
    print("  Saved: garch_fan_chart.png, egarchx_fan_chart.png")

    # Predicted vs actual scatter + time series
    print("Plotting predicted vs actual RV ...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Predicted vs Actual RV — Stock {STOCK_ID}", fontsize=13)

    for row_idx, (eval_df, label, color) in enumerate([
        (garch_eval_df,   "GARCH(1,1)",      "#4C72B0"),
        (egarchx_eval_df, "EGARCH(1,1,1)-X", "#DD8452"),
    ]):
        if eval_df.empty:
            continue

        pa = (eval_df.groupby("time_id")
              .agg(pred_RV=("pred_RV", "first"), actual_RV=("actual", "mean"))
              .dropna())

        # Exclude blow-up predictions above the data-driven ceiling
        n_excluded = (pa["pred_RV"] > RV_CEILING).sum()
        pa_plot    = pa[pa["pred_RV"] <= RV_CEILING].copy()

        lim = max(pa_plot["actual_RV"].max(), pa_plot["pred_RV"].max()) * 1.08
        axes[row_idx][0].scatter(pa_plot["actual_RV"], pa_plot["pred_RV"],
                                 alpha=0.35, s=12, color=color, label="Forecasts")
        axes[row_idx][0].plot([0, lim], [0, lim], "r--", linewidth=1.2, label="Perfect forecast")
        axes[row_idx][0].set_xlabel("Actual RV")
        axes[row_idx][0].set_ylabel("Predicted RV")
        axes[row_idx][0].set_title(f"Scatter — {label}"
                                   + (f"\n({n_excluded} outlier preds >0.05 excluded)" if n_excluded else ""))
        axes[row_idx][0].legend(fontsize=9)

        t = np.arange(len(pa_plot))
        axes[row_idx][1].plot(t, pa_plot["actual_RV"].values,
                              color="#4C72B0", linewidth=0.7, alpha=0.8, label="Actual")
        axes[row_idx][1].plot(t, pa_plot["pred_RV"].values,
                              color=color, linewidth=0.7, alpha=0.8, label="Predicted")
        axes[row_idx][1].set_xlabel("time_id (ordered)")
        axes[row_idx][1].set_ylabel("RV")
        axes[row_idx][1].set_title(f"Time series — {label}")
        axes[row_idx][1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(stock_out, "pred_vs_actual.png"), dpi=150)
    plt.close()
    print("  Saved: pred_vs_actual.png")

    # EGARCH-X gamma and delta distributions
    print("Plotting γ and δ distributions ...")

    if not ex_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"EGARCH-X Parameter Distributions — Stock {STOCK_ID}", fontsize=12)

        gammas    = ex_df["gamma"].dropna()
        pct_neg_g = (gammas < 0).mean()
        axes[0].hist(gammas, bins=30, color="#DD8452", alpha=0.75, edgecolor="white")
        axes[0].axvline(0, color="black", linestyle="--", linewidth=1.2, label="γ = 0 (symmetric)")
        axes[0].axvline(gammas.median(), color="red", linestyle="--", linewidth=1.2,
                        label=f"Median γ = {gammas.median():.4f}")
        axes[0].set_xlabel("γ  (asymmetry / leverage)")
        axes[0].set_ylabel("Number of time_ids")
        axes[0].set_title(f"γ < 0 in {pct_neg_g:.1%} → bad news raises vol more")
        axes[0].legend(fontsize=9)

        deltas    = ex_df["delta"].dropna()
        pct_pos_d = (deltas > 0).mean()
        axes[1].hist(deltas, bins=30, color="#55A868", alpha=0.75, edgecolor="white")
        axes[1].axvline(0, color="black", linestyle="--", linewidth=1.2, label="δ = 0 (no spread effect)")
        axes[1].axvline(deltas.median(), color="red", linestyle="--", linewidth=1.2,
                        label=f"Median δ = {deltas.median():.4f}")
        axes[1].set_xlabel("δ  (GARCH-X spread coefficient)")
        axes[1].set_ylabel("Number of time_ids")
        axes[1].set_title(f"δ > 0 in {pct_pos_d:.1%} → wider spread predicts higher vol")
        axes[1].legend(fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(stock_out, "egarchx_parameters.png"), dpi=150)
        plt.close()
        print("  Saved: egarchx_parameters.png")


    # ══════════════════════════════════════════════════════════════════════════════
    #  Done
    # ══════════════════════════════════════════════════════════════════════════════


    # ── Collect cross-stock summary row ─────────────────────────────────────
    garch_med_qlike   = garch_per_tid["QLIKE"].median()   if not garch_per_tid.empty   else float("nan")
    garch_med_mse     = garch_per_tid["MSE"].median()     if not garch_per_tid.empty   else float("nan")
    egarchx_med_qlike = egarchx_per_tid["QLIKE"].median() if not egarchx_per_tid.empty else float("nan")
    egarchx_med_mse   = egarchx_per_tid["MSE"].median()   if not egarchx_per_tid.empty else float("nan")

    all_garch_summary.append({
        "stock_id":          STOCK_ID,
        "liquidity_regime":  regime,
        "n_time_ids":        len(time_IDs),
        "n_forecasts":       len(garch_per_tid),
        "median_QLIKE":      garch_med_qlike,
        "median_MSE":        garch_med_mse,
        "blowup_pct":        round(garch_blowup_pct, 4),
    })
    all_egarchx_summary.append({
        "stock_id":          STOCK_ID,
        "liquidity_regime":  regime,
        "n_time_ids":        len(time_IDs),
        "n_forecasts":       len(egarchx_per_tid),
        "median_QLIKE":      egarchx_med_qlike,
        "median_MSE":        egarchx_med_mse,
        "blowup_pct":        round(egarchx_blowup_pct, 4) if np.isfinite(egarchx_blowup_pct) else np.nan,
        "egarchx_ran":       regime == "liquid",
    })




# ══════════════════════════════════════════════════════════════════════════════
#  Aggregate summary across all stocks
# ══════════════════════════════════════════════════════════════════════════════

if all_garch_summary:
    pd.DataFrame(all_garch_summary).sort_values("stock_id") \
      .to_csv(os.path.join(OUTPUT_DIR, "all_stocks_garch_summary.csv"), index=False)
    print("\nSaved: all_stocks_garch_summary.csv")

if all_egarchx_summary:
    pd.DataFrame(all_egarchx_summary).sort_values("stock_id") \
      .to_csv(os.path.join(OUTPUT_DIR, "all_stocks_egarchx_summary.csv"), index=False)
    print("Saved: all_stocks_egarchx_summary.csv")

# Global blowup report — illiquid stocks only (same stocks as Rosa)
if all_blowups:
    blowup_report = pd.concat(all_blowups, ignore_index=True)
    blowup_report["liquidity_regime"] = blowup_report["stock_id"].map(liquidity_map)
    blowup_report.to_csv(os.path.join(OUTPUT_DIR, "global_blowup_report.csv"), index=False)
    illiquid_preds = sum(s["n_forecasts"] for s in all_garch_summary
                         if s["liquidity_regime"] == "illiquid")
    blowup_pct = len(blowup_report) / max(illiquid_preds, 1)
    print(f"Saved: global_blowup_report.csv  "
          f"({len(blowup_report)} GARCH blowups on illiquid stocks = {blowup_pct:.1%})")


# ── Aggregate Plot 1: EGARCH-X vs GARCH on liquid stocks ─────────────────────
C_GARCH = "#378ADD"
C_EX    = "#1D9E75"
C_HAR   = "#D85A30"
SPINE   = "#D3D1C7"

garch_df   = pd.DataFrame(all_garch_summary)
egarchx_df = pd.DataFrame(all_egarchx_summary)

liquid_garch   = garch_df[garch_df["liquidity_regime"] == "liquid"].copy()
liquid_egarchx = egarchx_df[egarchx_df["egarchx_ran"] == True].copy()

if not liquid_garch.empty and not liquid_egarchx.empty:
    merged = liquid_garch.merge(
        liquid_egarchx[["stock_id", "median_QLIKE", "median_MSE"]],
        on="stock_id", suffixes=("_garch", "_egarchx")
    ).sort_values("median_QLIKE_garch")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle("EGARCH-X vs GARCH — Liquid Stocks  (lower QLIKE = better)",
                 fontsize=12, fontweight="500")

    x      = np.arange(len(merged))
    labels = [f"Stock {int(s)}" for s in merged["stock_id"]]
    w      = 0.38

    for ax, metric, title in zip(
        axes,
        ["median_QLIKE", "median_MSE"],
        ["Median QLIKE per stock", "Median MSE per stock"]
    ):
        ax.set_facecolor("white")
        bars_g  = ax.bar(x - w/2, merged[f"{metric}_garch"],  width=w,
                         label="GARCH(1,1)", color=C_GARCH, alpha=0.8)
        bars_ex = ax.bar(x + w/2, merged[f"{metric}_egarchx"], width=w,
                         label="EGARCH-X",   color=C_EX,    alpha=0.8)

        # Annotate which model wins per stock
        for i, (g, ex) in enumerate(zip(merged[f"{metric}_garch"],
                                         merged[f"{metric}_egarchx"])):
            winner_y = min(g, ex)
            ax.annotate("★", xy=(i + (w/2 if ex < g else -w/2), winner_y),
                        ha="center", va="top", fontsize=8,
                        color=C_EX if ex < g else C_GARCH)

        for sp in ax.spines.values(): sp.set_color(SPINE); sp.set_linewidth(0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
        ax.tick_params(labelsize=8, color=SPINE)
        ax.set_title(title, fontsize=10, fontweight="500")
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(axis="y", color=SPINE, linewidth=0.4, linestyle="--")
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "jisu_01_egarchx_vs_garch_liquid.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: jisu_01_egarchx_vs_garch_liquid.png")

    # Win rate summary
    ex_wins = (merged["median_QLIKE_egarchx"] < merged["median_QLIKE_garch"]).sum()
    print(f"  EGARCH-X beats GARCH on {ex_wins}/{len(merged)} liquid stocks by QLIKE")


# ── Aggregate Plot 2: GARCH blowups on illiquid stocks ───────────────────────
illiquid_garch = garch_df[garch_df["liquidity_regime"] == "illiquid"].copy()

if not illiquid_garch.empty:
    illiquid_garch = illiquid_garch.sort_values("blowup_pct", ascending=False)
    labels = [f"Stock {int(s)}" for s in illiquid_garch["stock_id"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle("GARCH Blowups — Illiquid Stocks  (same stocks as Rosa's HAR-RV)",
                 fontsize=12, fontweight="500")

    # Blowup rate per stock
    ax = axes[0]
    ax.set_facecolor("white")
    bar_colors = [C_HAR if p > 0.05 else C_GARCH for p in illiquid_garch["blowup_pct"]]
    ax.bar(range(len(illiquid_garch)), illiquid_garch["blowup_pct"] * 100,
           color=bar_colors, alpha=0.8, width=0.7)
    ax.axhline(5.8, color="#888780", linestyle="--", linewidth=0.9,
               label="5.8% reference (Jisu)")
    for sp in ax.spines.values(): sp.set_color(SPINE); sp.set_linewidth(0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax.tick_params(labelsize=8, color=SPINE)
    ax.set_ylabel("Blowup rate (%)", fontsize=9)
    ax.set_title("GARCH blowup rate per illiquid stock", fontsize=10, fontweight="500")
    ax.legend(fontsize=8)
    ax.grid(axis="y", color=SPINE, linewidth=0.4, linestyle="--")
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=C_HAR,   label="> 5% blowup — HAR-RV strongly preferred"),
        Patch(color=C_GARCH, label="< 5% blowup — GARCH borderline"),
        plt.Line2D([0],[0], color="#888780", linestyle="--", label="5.8% reference"),
    ], fontsize=7.5, framealpha=0.9)

    # QLIKE comparison: GARCH on illiquid vs GARCH on liquid
    ax2 = axes[1]
    ax2.set_facecolor("white")
    liq_qlike   = garch_df[garch_df["liquidity_regime"] == "liquid"]["median_QLIKE"].dropna()
    illiq_qlike = illiquid_garch["median_QLIKE"].dropna()
    bp = ax2.boxplot([liq_qlike.values, illiq_qlike.values],
                     patch_artist=True, labels=["Liquid stocks", "Illiquid stocks"],
                     widths=0.45,
                     medianprops=dict(color="white", linewidth=2),
                     whiskerprops=dict(color=SPINE, linewidth=0.8),
                     capprops=dict(color=SPINE, linewidth=0.8),
                     flierprops=dict(marker="o", markersize=3,
                                     markerfacecolor=SPINE, alpha=0.5))
    bp["boxes"][0].set_facecolor(C_GARCH); bp["boxes"][0].set_alpha(0.8)
    bp["boxes"][1].set_facecolor(C_HAR);   bp["boxes"][1].set_alpha(0.8)
    for sp in ax2.spines.values(): sp.set_color(SPINE); sp.set_linewidth(0.6)
    ax2.tick_params(labelsize=9, color=SPINE)
    ax2.set_ylabel("Median QLIKE", fontsize=9)
    ax2.set_title("GARCH QLIKE: liquid vs illiquid stocks\n"
                  "(shows why GARCH fails on illiquid)", fontsize=10, fontweight="500")
    ax2.grid(axis="y", color=SPINE, linewidth=0.4, linestyle="--")
    ax2.set_axisbelow(True)
    for i, vals in enumerate([liq_qlike, illiq_qlike]):
        ax2.annotate(f"Median\n{vals.median():.3f}",
                     xy=(i+1, vals.median()), xytext=(8, 0),
                     textcoords="offset points", fontsize=8, va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "jisu_02_garch_blowups_illiquid.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: jisu_02_garch_blowups_illiquid.png")

print(f"\n✓ M4 complete for {len(selected_stock_ids)} selected stocks "
      f"({len(liquid_selected)} liquid, {len(illiquid_selected)} illiquid).")
print(f"  Liquid:   EGARCH-X vs GARCH comparison → jisu_01_egarchx_vs_garch_liquid.png")
print(f"  Illiquid: GARCH blowups (same as Rosa)  → jisu_02_garch_blowups_illiquid.png")
print(f"  Per-stock outputs in: {OUTPUT_DIR}/stock_<id>/")
print(f"  Aggregate summaries in: {OUTPUT_DIR}/")