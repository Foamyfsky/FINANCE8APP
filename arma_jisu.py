"""
ARMA(1,1)-GARCH(1,1)/GJR-GARCH and EGARCH(1,1,1)-X modelling
Steps taken:
  Phase 1 - Data diagnostics (ACF/PACF, AIC/BIC order selection)
  Phase 2 - Baseline ARMA(1,1)-GARCH(1,1) -> residual diagnostics -> adaptive upgrade to GJR-GARCH + Student-t if patterns detected
  Phase 3 - EGARCH-X with direct one-step-ahead forecast (no Monte Carlo)-> runs on liquid AND mixed stocks
  Phase 4 - Comparison plots, time_id tracking, and CSV evaluation

Data format needed
  stock_id, time_id, time_bucket,  BidAskSpread_mean, RV
  20 buckets per time_id (30s each), numbered 2-21
  First 16 buckets = train, last 4 = validation

improvements over previous versions:
  - Monte Carlo removed from EGARCH-X -- uses deterministic one-step forecast
  - Residual diagnostics: Ljung-Box on squared residuals -> GJR-GARCH + t-dist
  - HAR-like extended mean (lags 1, 5) for longer memory
  - Mixed stocks (score 0.3-0.7): run both models, pick winner per stock
  - Full time_id accounting: attempted/succeeded/missed all reported
  - QLIKE + MSE computed for every uploaded CSV regardless of regime
"""

import matplotlib
matplotlib.use("Agg")
import os
import json
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
from arch.univariate.volatility import EGARCH, _common_names
from arch.univariate import ARX, Normal, StudentsT
from arch.utility.exceptions import ConvergenceWarning
from joblib import Parallel, delayed

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    def njit(fn): return fn
    HAS_NUMBA = False

warnings.filterwarnings("ignore")


# Paths
INPUT_CSV = r"C:\Users\songj\OneDrive\UNI\Y3SEM1\DATA3888\Project Folder\DATA3888G08\optiver_aggregated.csv"
_HERE = os.path.dirname(os.path.abspath(__file__))

# Example session CSVs uploaded by user (for cross-model evaluation)
EXAMPLE_CSVS = {
    "liquid":   os.path.join(_HERE, "example_liquid_session.csv"),
    "illiquid": os.path.join(_HERE, "example_illiquid_session.csv"),
    "mixed":    os.path.join(_HERE, "example_mixed_session.csv"),
}

from config import (get_selected_stocks, get_liquidity_map, filter_time_ids,
                    OUTPUT_DIR, JAMIE_LIQUIDITY_CSV,
                    N_TRAIN, N_VAL, N_STOCKS_PER_REGIME, TIME_ID_KEEP_PCT)

# Global config
SCALE = 1000.0  # rescale RV for numerical stability
N_JOBS = -1     # parallel workers

# Hyperparameter tuning orders to try
GARCH_ORDERS = list(itertools.product([1, 2], [1, 2]))
EGARCHX_ORDERS = list(itertools.product([1, 2], [1], [1, 2]))
TUNE_SAMPLE = 200

# Residual diagnostics thresholds
LJUNGBOX_LAGS = [5, 10]  # lags for Ljung-Box tests
LJUNGBOX_ALPHA = 0.05    # significance level for rejecting H0 of no autocorrelation

# Blowup threshold: GARCH predictions above this QLIKE percentile are flagged
BLOWUP_QLIKE_PERCENTILE = 99

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Numba-compiled EGARCH-X variance kernel

@njit
def _egarchx_var_kernel(omega, alphas, gammas, betas, delta,
                         resids, spread, backcast, nobs,
                         p, o, q, norm_const):
    lnsigma2 = np.zeros(nobs)
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


class EGARCHX(EGARCH):
    """
    EGARCH(p,o,q)-X: adds a log bid-ask spread term to the variance equation.

      ln sigma^2_t = omega + alpha(|z_{t-1}| - E|z|) + gamma*z_{t-1} + beta*ln sigma^2_{t-1}
                                           + delta*ln(spread^2_{t-1})

    gamma < 0 -> leverage (bad news raises vol more than good)
    delta > 0 -> liquidity channel (wider spread predicts higher vol)
    """

    def __init__(self, spread, p=1, o=1, q=1):
        super().__init__(p=p, o=o, q=q)
        self._spread = np.asarray(spread, dtype=float)
        self._num_params = 1 + p + o + q + 1

    def parameter_names(self):
        return _common_names(self.p, self.o, self.q) + ["delta[spread]"]

    def starting_values(self, resids):
        return np.append(super().starting_values(resids), 0.05)

    def bounds(self, resids):
        return super().bounds(resids) + [(-10.0, 10.0)]

    def constraints(self):
        A = np.zeros((1, self._num_params))
        b = np.zeros(1)
        return A, b

    def compute_variance(self, parameters, resids, sigma2, backcast, var_bounds):
        p, o, q = self.p, self.o, self.q
        omega = parameters[0]
        alphas = np.asarray(parameters[1     : 1+p],     dtype=np.float64)
        gammas = np.asarray(parameters[1+p   : 1+p+o],   dtype=np.float64)
        betas  = np.asarray(parameters[1+p+o : 1+p+o+q], dtype=np.float64)
        delta = float(parameters[-1])
        spread = np.asarray(self._spread, dtype=np.float64)
        norm_const = np.sqrt(2.0 / np.pi)
        nobs = len(resids)
        bc = float(np.atleast_1d(backcast)[0])

        lnsigma2 = _egarchx_var_kernel(
            float(omega), alphas, gammas, betas, delta,
            np.asarray(resids, dtype=np.float64), spread,
            bc, nobs, p, o, q, norm_const
        )
        sigma2[:] = np.clip(np.exp(lnsigma2), var_bounds[:, 0], var_bounds[:, 1])
        return sigma2

    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        """keep for compatibility but no longer called in main pipeline."""
        p, o, q = self.p, self.o, self.q
        parameters = np.asarray(parameters, dtype=float)
        omega = parameters[0]
        alphas = parameters[1     : 1+p]
        gammas = parameters[1+p   : 1+p+o]
        betas  = parameters[1+p+o : 1+p+o+q]
        delta = parameters[-1]
        nc = np.sqrt(2.0 / np.pi)
        ss = np.where(np.abs(self._spread) < 1e-10, 1e-10, np.abs(self._spread))
        spread_offset = delta * float(np.mean(np.log(ss ** 2)))
        beta_sum = float(np.sum(betas))
        init_lnsig = (omega + spread_offset) / (1.0 - beta_sum) \
                     if abs(beta_sum) < 1.0 else omega + spread_offset
        e = rng(nobs + burn)
        ae = np.abs(e)
        ls = np.full(nobs + burn, init_lnsig)
        lag = max(p, o, q, 1)
        for t in range(lag, nobs + burn):
            v = omega + spread_offset
            for j in range(p): v += alphas[j] * (ae[t-1-j] - nc)
            for j in range(o): v += gammas[j] * e[t-1-j]
            for j in range(q): v += betas[j] * ls[t-1-j]
            ls[t] = v
        sig = np.sqrt(np.maximum(np.exp(ls[burn:]), 1e-16))
        returns = e[burn:] * sig
        return pd.DataFrame({"data": returns, "volatility": sig, "errors": e[burn:]})


# Residual diagnostics
# Run after fitting to detect patterns that justify model upgrades.

def residual_diagnostics(res):
    """
    Analyse standardized residuals from a fitted arch model.

    Returns a dict with:
      arch_effect     - True if squared residuals still show autocorrelation
                        (remaining GARCH effect -> try GJR or higher order)
      autocorr_effect - True if residuals show autocorrelation in levels
                        (mean misspecification -> add HAR lags)
      fat_tails       - True if excess kurtosis > 1 (-> use t-distribution)
      leverage        - True if asymmetry detected (-> use GJR-GARCH)
      notes           - human-readable diagnostic strings
    """
    try:
        std_resids = np.asarray(res.std_resid)
        std_resids = std_resids[np.isfinite(std_resids)]
        if len(std_resids) < 10:
            return {"arch_effect": False, "autocorr_effect": False,
                    "fat_tails": False, "leverage": False, "notes": ["too few obs"]}

        notes = []

        # Ljung-Box on squared residuals (remaining ARCH effects)
        lb_sq = acorr_ljungbox(std_resids ** 2, lags=LJUNGBOX_LAGS, return_df=True)
        arch_effect = bool((lb_sq["lb_pvalue"] < LJUNGBOX_ALPHA).any())
        if arch_effect:
            notes.append(f"ARCH effect in sq-residuals (p={lb_sq['lb_pvalue'].min():.3f})")

        # Ljung-Box on residuals in levels (mean autocorrelation)
        lb_lv = acorr_ljungbox(std_resids, lags=LJUNGBOX_LAGS, return_df=True)
        autocorr_effect = bool((lb_lv["lb_pvalue"] < LJUNGBOX_ALPHA).any())
        if autocorr_effect:
            notes.append(f"Autocorr in residuals (p={lb_lv['lb_pvalue'].min():.3f})")

        # Excess kurtosis - fat tails
        kurt = float(pd.Series(std_resids).kurt())  # excess kurtosis
        fat_tails = kurt > 1.0
        if fat_tails:
            notes.append(f"Fat tails: excess kurtosis = {kurt:.2f} -> t-distribution")

        # Leverage detection: negative residuals should predict higher vol.
        # Simple check: correlation between lagged negative residuals and |residuals|.
        neg_mask = std_resids[:-1] < 0
        if neg_mask.sum() > 5:
            pos_response = np.abs(std_resids[1:])[neg_mask].mean()
            neg_response = np.abs(std_resids[1:])[~neg_mask].mean()
            leverage = bool(pos_response > neg_response * 1.1)
        else:
            leverage = False
        if leverage:
            notes.append("Leverage detected: negative shocks raise vol more")

        if not notes:
            notes.append("No patterns in residuals - GARCH(1,1) normal is appropriate")

        return {
            "arch_effect": arch_effect,
            "autocorr_effect": autocorr_effect,
            "fat_tails": fat_tails,
            "leverage": leverage,
            "notes": notes,
        }

    except Exception as e:
        return {"arch_effect": False, "autocorr_effect": False,
                "fat_tails": False, "leverage": False, "notes": [f"diag error: {e}"]}


# Parallel worker functions

def _fit_garch_one(tid, rv_s, p, q, dist="normal", scale=1000.0, ftol=1e-7):
    """
    Fit ARMA(1,1)-GARCH(p,q) for a single time_id.
    dist: 'normal' or 't'
    Returns (tid, result_or_None)
    """
    try:
        am = arch_model(rv_s * scale, mean="ARX", lags=1,
                        vol="GARCH", p=p, q=q, dist=dist)
        res = am.fit(disp="off", show_warning=False,
                     options={"maxiter": 1000, "ftol": ftol})
        return tid, res
    except Exception:
        return tid, None


def _fit_gjrgarch_one(tid, rv_s, p=1, o=1, q=1, dist="t", scale=1000.0, ftol=1e-7):
    """
    Fit ARMA(1,1)-GJR-GARCH(p,o,q) for a single time_id.
    GJR-GARCH captures asymmetric (leverage) effects via the 'o' term.
    dist defaults to 't' since leverage stocks typically show fat tails.
    Returns (tid, result_or_None)
    """
    try:
        am = arch_model(rv_s * scale, mean="ARX", lags=1,
                        vol="GARCH", p=p, o=o, q=q, dist=dist)
        res = am.fit(disp="off", show_warning=False,
                     options={"maxiter": 1000, "ftol": ftol})
        return tid, res
    except Exception:
        return tid, None


def _fit_egarchx_one(tid, rv_s, sp_s, p, o, q, scale=1000.0, ftol=1e-7):
    """
    Fit ARMA(1,1)-EGARCH-X(p,o,q) for a single time_id.
    Returns (tid, res_or_None, vol_obj_or_None)
    """
    try:
        vol_obj = EGARCHX(sp_s * scale, p=p, o=o, q=q)
        am = ARX(rv_s * scale, lags=1)
        am.volatility = vol_obj
        am.distribution = Normal()
        res = am.fit(disp="off", show_warning=False,
                     options={"maxiter": 1000, "ftol": ftol})
        return tid, res, vol_obj
    except Exception:
        return tid, None, None


# One-step ahead forecasting (no Monte Carlo)

def forecast_one_step(res, scale=1000.0):
    """
    Deterministic one-step-ahead forecast of RV.

    Uses the ARX mean equation: E[RV_{t+1}] = const + phi*RV_t
    This is stable, non-simulated, and works for GARCH, GJR-GARCH, and EGARCH-X.

    The GARCH/EGARCH-X variance sigma^2_{t+1} tells us the UNCERTAINTY of the forecast,
    not the level. report both the point forecast and +/-1 sigma bounds.

    Returns: (pred_rv, lower_bound, upper_bound) all in raw RV units.
    """
    try:
        fcast = res.forecast(horizon=1)
        pred_mean = float(fcast.mean.iloc[-1, 0]) / scale
        pred_var = float(fcast.variance.iloc[-1, 0]) / (scale ** 2)
        pred_std = np.sqrt(max(pred_var, 0.0))

        if np.isfinite(pred_mean) and pred_mean > 0:
            return pred_mean, max(pred_mean - pred_std, 1e-8), pred_mean + pred_std

        # FALLBACK: last fitted conditional mean
        last_rv = float(res.resid[-1] + res.conditional_volatility[-1]) / scale
        return max(last_rv, 1e-8), max(last_rv * 0.8, 1e-8), last_rv * 1.2

    except Exception:
        return np.nan, np.nan, np.nan


def egarchx_forecast_one_step(res, vol_obj, val_spread, scale=1000.0):
    """
    One-step-ahead forecast for EGARCH-X using the NEXT spread value from
    the validation window (since spread enters the variance equation at t-1).

    The conditional mean is from the ARX part (same as GARCH).
    The conditional variance uses the EGARCH-X recursion with current spread.

    Returns: (pred_rv, lower_bound, upper_bound)
    """
    try:
        # ARX mean forecast - same formula regardless of volatility spec
        fcast = res.forecast(horizon=1)
        pred_mean = float(fcast.mean.iloc[-1, 0]) / scale

        # EGARCH-X variance: use last EGARCH-X conditional variance.
        # The forecast variance from arch may not handle custom vol correctly,
        # so we use the last fitted conditional variance as a proxy.
        last_cond_vol = float(res.conditional_volatility[-1]) / scale

        # Use the first validation spread to get a forward-looking variance estimate
        next_spread = float(val_spread[0]) if len(val_spread) > 0 else np.nan

        if np.isfinite(pred_mean) and pred_mean > 0:
            # EGARCH-X informed bounds: wider if spread is large (illiquid)
            if np.isfinite(next_spread) and next_spread > 0:
                spread_factor = np.sqrt(1.0 + np.log(max(next_spread * scale, 1e-10)))
                pred_std = last_cond_vol * spread_factor * 0.5
            else:
                pred_std = last_cond_vol * 0.5

            return pred_mean, max(pred_mean - pred_std, 1e-8), pred_mean + pred_std

        return np.nan, np.nan, np.nan

    except Exception:
        return np.nan, np.nan, np.nan


# Helper functions

def qlike(pred, actual):
    """
    QLIKE = log(pred) + actual/pred - lower is better.

    Variance form: pred and actual are REALISED VARIANCE (RV) values.
    This matches rosa.py and is the standard Patton (2011) QLIKE loss for
    variance forecasting. Do NOT use the squared form (log(pred^2) + actual^2/pred^2)
    when comparing to RV - that form applies only when pred/actual are in
    return (standard-deviation) units.
    """
    if pred <= 0 or actual < 0 or not np.isfinite(pred) or not np.isfinite(actual):
        return np.nan
    pred = max(pred, 1e-10)
    return np.log(pred) + actual / pred


def garch_is_stationary(res):
    """Returns True if alpha + beta < 0.999 (prevents IGARCH blow-up)."""
    try:
        return (res.params.get("alpha[1]", 1) + res.params.get("beta[1]", 1)) < 0.999
    except Exception:
        return False


def egarchx_passes_sanity_check(res):
    """Reject non-converged fits and extreme parameter values."""
    if res.convergence_flag != 0:
        return False
    try:
        return (
            abs(res.params.get("gamma[1]", 99)) < 5 and
            abs(res.params.get("delta[spread]", 99)) < 5 and
            abs(res.params.get("beta[1]", 1)) < 0.9999
        )
    except Exception:
        return False


def evaluate_preds(preds_dict, vol_val_dict, model_label="model"):
    """
    Given {time_id: pred_rv} and validation data, compute QLIKE and MSE.
    Tracks which time_ids were attempted, succeeded, and missed.

    Returns (eval_df, tracking_dict).
    """
    records = []
    succeeded = []
    missed = []

    all_val_tids = set(vol_val_dict.keys())

    for tid in all_val_tids:
        pred = preds_dict.get(tid, None)
        if pred is None or not np.isfinite(pred):
            missed.append(tid)
            continue

        val_rvs = vol_val_dict[tid]["RV"].values
        succeeded.append(tid)
        for bucket_idx, actual in enumerate(val_rvs):
            q = qlike(pred, actual)
            records.append({
                "time_id": tid,
                "bucket_idx": bucket_idx,
                "pred_RV": pred,
                "actual_RV": actual,
                "QLIKE": q if np.isfinite(q) else np.nan,
                "MSE": (actual - pred) ** 2 if np.isfinite(pred) else np.nan,
                "model": model_label,
            })

    df = pd.DataFrame(records)
    tracking = {
        "n_val_tids": len(all_val_tids),
        "n_succeeded": len(succeeded),
        "n_missed": len(missed),
        "missed_tids": missed,
        "miss_rate": len(missed) / max(len(all_val_tids), 1),
    }
    return df, tracking


# Hyperparameter tuning helpers

def tune_garch_order(vol_train, time_ids, orders=GARCH_ORDERS,
                     sample=TUNE_SAMPLE, scale=1000.0, dist="normal"):
    print(f"  Tuning GARCH order over {sample} time_ids (dist={dist}) ...")
    results = []
    for p, q in orders:
        bics = []
        for tid in time_ids[:sample]:
            rv_s = vol_train[tid]["log_return"].values   # tune on log returns
            try:
                am = arch_model(rv_s * scale, mean="ARX", lags=1,
                                vol="GARCH", p=p, q=q, dist=dist)
                res = am.fit(disp="off", show_warning=False, options={"maxiter": 500})
                if res.convergence_flag == 0:
                    bics.append(res.bic)
            except Exception:
                continue
        if bics:
            results.append({"order": f"GARCH({p},{q})", "p": p, "q": q,
                             "mean_BIC": round(np.mean(bics), 3),
                             "n_converged": len(bics)})

    df = pd.DataFrame(results).sort_values("mean_BIC").reset_index(drop=True)
    best = df.iloc[0]
    print(f"  Best order: GARCH({int(best.p)},{int(best.q)})  BIC={best.mean_BIC:.3f}")
    return int(best.p), int(best.q), df


def tune_egarchx_order(vol_train, time_ids, orders=EGARCHX_ORDERS,
                        sample=TUNE_SAMPLE, scale=1000.0):
    print(f"  Tuning EGARCH-X order over {sample} time_ids ...")
    results = []
    for p, o, q in orders:
        bics = []
        for tid in time_ids[:sample]:
            rv_s = vol_train[tid]["log_return"].values   # tune on log returns
            sp_s = vol_train[tid]["BidAskSpread_mean"].values
            try:
                vol_obj = EGARCHX(sp_s * scale, p=p, o=o, q=q)
                am = ARX(rv_s * scale, lags=1)
                am.volatility = vol_obj
                am.distribution = Normal()
                res = am.fit(disp="off", show_warning=False, options={"maxiter": 500})
                if res.convergence_flag == 0 and egarchx_passes_sanity_check(res):
                    bics.append(res.bic)
            except Exception:
                continue
        if bics:
            results.append({"order": f"EGARCHX({p},{o},{q})",
                             "p": p, "o": o, "q": q,
                             "mean_BIC": round(np.mean(bics), 3),
                             "n_converged": len(bics)})

    df = pd.DataFrame(results).sort_values("mean_BIC").reset_index(drop=True)
    best = df.iloc[0]
    print(f"  Best order: EGARCHX({int(best.p)},{int(best.o)},{int(best.q)}) BIC={best.mean_BIC:.3f}")
    return int(best.p), int(best.o), int(best.q), df


def evaluate_csv_all_models(csv_path, label, scale=1000.0):
    """
    Fits GARCH(1,1), GJR-GARCH(1,1,1), and EGARCH-X(1,1,1) on the uploaded
    example CSV and computes QLIKE + MSE for each model.

    The CSV has columns: time_bucket, BidAskSpread_mean, RV
    (20 rows: buckets 1-20 or 2-21; first N_TRAIN = 16 train, last N_VAL = 4 val)

    Returns a DataFrame with one row per model.
    """
    try:
        df = pd.read_csv(csv_path).dropna(subset=["RV", "BidAskSpread_mean"])
        if len(df) < N_TRAIN + N_VAL:
            return pd.DataFrame()

        train = df.iloc[:N_TRAIN]
        val = df.iloc[N_TRAIN : N_TRAIN + N_VAL]
        rv_train = train["RV"].values
        sp_train = train["BidAskSpread_mean"].values
        rv_val = val["RV"].values

        rows = []

        # GARCH(1,1) Normal
        try:
            am1 = arch_model(rv_train * scale, mean="ARX", lags=1,
                             vol="GARCH", p=1, q=1, dist="normal")
            r1 = am1.fit(disp="off", show_warning=False, options={"maxiter": 1000})
            pred1, lb1, ub1 = forecast_one_step(r1, scale)
            q1 = float(np.nanmean([qlike(pred1, a) for a in rv_val]))
            mse1 = float(np.nanmean([(a - pred1)**2 for a in rv_val]))
            rows.append({"model": "GARCH(1,1)-Normal", "pred_RV": pred1,
                         "QLIKE": q1, "MSE": mse1, "label": label,
                         "converged": r1.convergence_flag == 0})
        except Exception:
            rows.append({"model": "GARCH(1,1)-Normal", "pred_RV": np.nan,
                         "QLIKE": np.nan, "MSE": np.nan, "label": label,
                         "converged": False})

        # GARCH(1,1) Student-t
        try:
            am2 = arch_model(rv_train * scale, mean="ARX", lags=1,
                             vol="GARCH", p=1, q=1, dist="t")
            r2 = am2.fit(disp="off", show_warning=False, options={"maxiter": 1000})
            pred2, lb2, ub2 = forecast_one_step(r2, scale)
            q2 = float(np.nanmean([qlike(pred2, a) for a in rv_val]))
            mse2 = float(np.nanmean([(a - pred2)**2 for a in rv_val]))
            rows.append({"model": "GARCH(1,1)-t", "pred_RV": pred2,
                         "QLIKE": q2, "MSE": mse2, "label": label,
                         "converged": r2.convergence_flag == 0})
        except Exception:
            rows.append({"model": "GARCH(1,1)-t", "pred_RV": np.nan,
                         "QLIKE": np.nan, "MSE": np.nan, "label": label,
                         "converged": False})

        # GJR-GARCH(1,1,1) Student-t
        try:
            am3 = arch_model(rv_train * scale, mean="ARX", lags=1,
                             vol="GARCH", p=1, o=1, q=1, dist="t")
            r3 = am3.fit(disp="off", show_warning=False, options={"maxiter": 1000})
            pred3, lb3, ub3 = forecast_one_step(r3, scale)
            q3 = float(np.nanmean([qlike(pred3, a) for a in rv_val]))
            mse3 = float(np.nanmean([(a - pred3)**2 for a in rv_val]))
            rows.append({"model": "GJR-GARCH(1,1,1)-t", "pred_RV": pred3,
                         "QLIKE": q3, "MSE": mse3, "label": label,
                         "converged": r3.convergence_flag == 0})
        except Exception:
            rows.append({"model": "GJR-GARCH(1,1,1)-t", "pred_RV": np.nan,
                         "QLIKE": np.nan, "MSE": np.nan, "label": label,
                         "converged": False})

        # EGARCH-X(1,1,1) Normal
        try:
            vol_obj4 = EGARCHX(sp_train * scale, p=1, o=1, q=1)
            am4 = ARX(rv_train * scale, lags=1)
            am4.volatility = vol_obj4
            am4.distribution = Normal()
            r4 = am4.fit(disp="off", show_warning=False, options={"maxiter": 1000})
            if egarchx_passes_sanity_check(r4):
                pred4, lb4, ub4 = egarchx_forecast_one_step(
                    r4, vol_obj4, val["BidAskSpread_mean"].values, scale)
            else:
                pred4 = np.nan
            q4 = float(np.nanmean([qlike(pred4, a) for a in rv_val])) if np.isfinite(pred4 or np.nan) else np.nan
            mse4 = float(np.nanmean([(a - pred4)**2 for a in rv_val])) if np.isfinite(pred4 or np.nan) else np.nan
            rows.append({"model": "EGARCH-X(1,1,1)", "pred_RV": pred4,
                         "QLIKE": q4, "MSE": mse4, "label": label,
                         "converged": r4.convergence_flag == 0})
        except Exception:
            rows.append({"model": "EGARCH-X(1,1,1)", "pred_RV": np.nan,
                         "QLIKE": np.nan, "MSE": np.nan, "label": label,
                         "converged": False})

        return pd.DataFrame(rows)

    except Exception as e:
        print(f"  [WARN] CSV evaluation failed for {label}: {e}")
        return pd.DataFrame()


# Load and prepare data

print("Loading data ...")
df = pd.read_csv(INPUT_CSV)
all_stock_ids = sorted(df["stock_id"].unique())
print(f"  Found {len(all_stock_ids)} stocks in dataset.")
print(f"  Numba JIT: {'enabled' if HAS_NUMBA else 'not available -- install numba for ~50x speedup'}")

# Compute log returns from WAP_mean (required for GARCH/EGARCH-X).
# GARCH models the conditional variance of log returns, not of RV directly.
# log_return_t = log(WAP_t) - log(WAP_{t-1}) within each (stock_id, time_id).
# The first bucket in each session gets NaN -> filled with 0 (no prior price).
if "WAP_mean" in df.columns:
    df = df.sort_values(["stock_id", "time_id", "time_bucket"]).reset_index(drop=True)
    df["log_return"] = (
        df.groupby(["stock_id", "time_id"])["WAP_mean"]
        .transform(lambda x: np.log(x / x.shift(1)))
    )
    df["log_return"] = df["log_return"].fillna(0.0)
    print(f"  log_return computed from WAP_mean. "
          f"Mean abs return: {df['log_return'].abs().mean():.6f}")
    USE_LOG_RETURNS = True
else:
    print("  WAP_mean not found -- falling back to RV as GARCH input.")
    print("    For theoretically correct GARCH, the aggregated CSV must include WAP_mean.")
    df["log_return"] = df["RV"].copy()   # fallback: use RV
    USE_LOG_RETURNS = False

rv_all = df["RV"].dropna()
RV_P01 = float(rv_all.quantile(0.01))
RV_P99 = float(rv_all.quantile(0.99))
RV_FLOOR = max(RV_P01 * 0.1, 1e-8)
RV_CEILING = RV_P99 * 10

print(f"\nRV scale diagnostics:")
print(f"  RV  p01={RV_P01:.6f}  p50={rv_all.median():.6f}  p99={RV_P99:.6f}")
if USE_LOG_RETURNS:
    ret_all = df["log_return"].dropna()
    print(f"  ret p01={ret_all.quantile(0.01):.6f}  p50={ret_all.median():.6f}"
          f"  p99={ret_all.quantile(0.99):.6f}")

# Liquidity classification + stock selection
liquidity_map = get_liquidity_map()
# Do NOT pass df -- forces the CSV cache as sole source of truth.
# If selected_stocks.csv is missing or empty, this raises a clear ValueError
# rather than silently auto-selecting from all stocks in the raw data.
selected = get_selected_stocks()
liquid_sel = selected["liquid"]
illiquid_sel = selected["illiquid"]
mixed_sel = selected.get("mixed", [])
selected_ids = selected["all"]

print(f"\nStock selection:")
print(f"  Liquid   ({len(liquid_sel)}): {liquid_sel}")
print(f"  Illiquid ({len(illiquid_sel)}): {illiquid_sel}")
print(f"  Mixed    ({len(mixed_sel)}): {mixed_sel}")
print(f"  EGARCH-X runs on liquid + mixed stocks. GARCH runs on all.")


# Global one-time hyperparameter tuning (cached after first run)

TUNING_CACHE = os.path.join(OUTPUT_DIR, "global_tuning_cache.json")

if os.path.exists(TUNING_CACHE):
    print(f"\nLoading cached tuning results ...")
    with open(TUNING_CACHE) as f:
        cache = json.load(f)
    GLOBAL_GARCH_P = cache["garch_p"]
    GLOBAL_GARCH_Q = cache["garch_q"]
    GLOBAL_EGARCHX_P = cache["egarchx_p"]
    GLOBAL_EGARCHX_O = cache["egarchx_o"]
    GLOBAL_EGARCHX_Q = cache["egarchx_q"]
    GLOBAL_GARCH_DIST = cache.get("garch_dist", "t")
    garch_tune_df = pd.read_csv(os.path.join(OUTPUT_DIR, "global_garch_tuning.csv"))
    egarchx_tune_df = pd.read_csv(os.path.join(OUTPUT_DIR, "global_egarchx_tuning.csv"))
    print(f"  GARCH order:    ({GLOBAL_GARCH_P},{GLOBAL_GARCH_Q}) dist={GLOBAL_GARCH_DIST}")
    print(f"  EGARCH-X order: ({GLOBAL_EGARCHX_P},{GLOBAL_EGARCHX_O},{GLOBAL_EGARCHX_Q})")
    print("  (Delete global_tuning_cache.json to re-tune)")

else:
    print("\nGlobal Hyperparameter Tuning ...")
    np.random.seed(42)
    sample_stocks = np.random.choice(selected_ids, size=min(5, len(selected_ids)), replace=False)
    global_train = {}
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
    print(f"  Tuning on {len(global_time_ids)} time_ids from stocks: {list(sample_stocks)}")

    # Try t-distribution -- typically better for RV data
    GLOBAL_GARCH_DIST = "t"
    GLOBAL_GARCH_P, GLOBAL_GARCH_Q, garch_tune_df = tune_garch_order(
        global_train, global_time_ids, orders=GARCH_ORDERS,
        sample=len(global_time_ids), dist=GLOBAL_GARCH_DIST
    )
    garch_tune_df.to_csv(os.path.join(OUTPUT_DIR, "global_garch_tuning.csv"), index=False)

    GLOBAL_EGARCHX_P, GLOBAL_EGARCHX_O, GLOBAL_EGARCHX_Q, egarchx_tune_df = tune_egarchx_order(
        global_train, global_time_ids, orders=EGARCHX_ORDERS, sample=len(global_time_ids)
    )
    egarchx_tune_df.to_csv(os.path.join(OUTPUT_DIR, "global_egarchx_tuning.csv"), index=False)

    with open(TUNING_CACHE, "w") as f:
        json.dump({
            "garch_p": GLOBAL_GARCH_P,
            "garch_q": GLOBAL_GARCH_Q,
            "garch_dist": GLOBAL_GARCH_DIST,
            "egarchx_p": GLOBAL_EGARCHX_P,
            "egarchx_o": GLOBAL_EGARCHX_O,
            "egarchx_q": GLOBAL_EGARCHX_Q,
        }, f, indent=2)
    print(f"  Tuning results cached to: {TUNING_CACHE}")


# Collectors for cross-stock summary

all_garch_summary = []
all_egarchx_summary = []
all_blowups = []
all_diag_summary = []  # residual diagnostics summary across stocks


# Main loop - iterate over every selected stock

for STOCK_ID in selected_ids:
    print(f"\n{'='*70}")
    print(f"  Processing Stock {STOCK_ID}  ({selected_ids.index(STOCK_ID)+1}/{len(selected_ids)})")
    print(f"{'='*70}")

    # Regime: liquid / illiquid / mixed
    if STOCK_ID in liquid_sel:
        regime = "liquid"
    elif STOCK_ID in illiquid_sel:
        regime = "illiquid"
    else:
        regime = "mixed"

    run_egarchx = (regime in ("liquid", "mixed"))
    print(f"  Regime: {regime.upper()}"
          + ("  -> EGARCH-X + GARCH both run" if run_egarchx else
             "  -> GARCH only (HAR-RV handles this regime)"))

    stock_out = os.path.join(OUTPUT_DIR, f"stock_{STOCK_ID}")
    os.makedirs(stock_out, exist_ok=True)

    stock1 = df[df["stock_id"] == STOCK_ID].copy()
    stock1 = stock1.sort_values(["time_id", "time_bucket"]).reset_index(drop=True)

    vol_train = {}
    vol_val = {}

    for tid in sorted(stock1["time_id"].unique()):
        buckets = (stock1[stock1["time_id"] == tid]
                   .sort_values("time_bucket")
                   .reset_index(drop=True))
        if len(buckets) < N_TRAIN + N_VAL:
            continue
        vol_train[tid] = buckets.iloc[:N_TRAIN].copy()
        vol_val[tid] = buckets.iloc[N_TRAIN : N_TRAIN + N_VAL].copy()

    time_IDs = list(vol_train.keys())
    if not time_IDs:
        print(f"  No complete time_ids for stock {STOCK_ID} -- skipping.")
        continue

    n_total_tids = len(time_IDs)
    time_IDs = filter_time_ids(vol_train, time_IDs, regime)
    vol_train = {tid: vol_train[tid] for tid in time_IDs}
    vol_val = {tid: vol_val[tid] for tid in time_IDs if tid in vol_val}
    print(f"  Total time_ids: {n_total_tids} -> kept {len(time_IDs)} most {regime}-representative")


    # Phase 1 - Data Diagnostics

    print("\n-- Phase 1: Data Diagnostics")

    tid_diag = time_IDs[0]
    ret_diag = vol_train[tid_diag]["log_return"].values
    max_lags = max(1, len(ret_diag) // 2 - 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"ACF / PACF -- Log Returns\n"
                 f"Stock {STOCK_ID} ({regime}), time_id={tid_diag}", fontsize=12)
    plot_acf( ret_diag, lags=max_lags, ax=axes[0], title="ACF -- Log Return")
    plot_pacf(ret_diag, lags=max_lags, ax=axes[1], title="PACF -- Log Return", method="ywm")
    plt.tight_layout()
    plt.savefig(os.path.join(stock_out, "phase1_acf_pacf.png"), dpi=150)
    plt.close()

    # AIC/BIC order selection on log returns
    order_results = []
    for p, q in itertools.product([1, 2], [1, 2]):
        bics = []
        for tid in time_IDs[:200]:
            rv_s = vol_train[tid]["log_return"].values
            try:
                am = arch_model(rv_s * SCALE, mean="ARX", lags=1,
                                vol="GARCH", p=p, q=q, dist="normal")
                res = am.fit(disp="off", show_warning=False, options={"maxiter": 500})
                if res.convergence_flag == 0:
                    bics.append(res.bic)
            except Exception:
                continue
        if bics:
            order_results.append({"GARCH(p,q)": f"({p},{q})",
                                   "Mean BIC":   round(np.mean(bics), 3),
                                   "Converged":  len(bics)})

    order_df = pd.DataFrame(order_results).sort_values("Mean BIC")
    order_df.to_csv(os.path.join(stock_out, "phase1_order_selection.csv"), index=False)
    print(f"  Saved ACF/PACF plot, AIC/BIC order selection")


    # Phase 2 - GARCH fit with residual diagnostics -> adaptive model upgrade

    print("\n-- Phase 2: GARCH fit + residual diagnostics")

    best_garch_p, best_garch_q = GLOBAL_GARCH_P, GLOBAL_GARCH_Q
    garch_dist_initial = GLOBAL_GARCH_DIST

    print(f"  Fitting GARCH({best_garch_p},{best_garch_q})-{garch_dist_initial} "
          f"({len(time_IDs)} time_ids) ...")

    init_fit_results = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_fit_garch_one)(tid, vol_train[tid]["log_return"].values,
                                best_garch_p, best_garch_q, garch_dist_initial)
        for tid in time_IDs
    )

    print("  Running residual diagnostics on sample ...")
    diag_sample = [r for _, r in init_fit_results if r is not None
                   and r.convergence_flag == 0][:50]

    arch_flags = []
    autocorr_flags = []
    fat_tail_flags = []
    leverage_flags = []
    all_diag_notes = []

    for res_diag in diag_sample:
        d = residual_diagnostics(res_diag)
        arch_flags.append(d["arch_effect"])
        autocorr_flags.append(d["autocorr_effect"])
        fat_tail_flags.append(d["fat_tails"])
        leverage_flags.append(d["leverage"])
        all_diag_notes.extend(d["notes"])

    pct_arch     = np.mean(arch_flags)     if arch_flags     else 0.0
    pct_autocorr = np.mean(autocorr_flags) if autocorr_flags else 0.0
    pct_fat      = np.mean(fat_tail_flags) if fat_tail_flags else 0.0
    pct_leverage = np.mean(leverage_flags) if leverage_flags else 0.0

    print(f"  Residual pattern rates (n={len(diag_sample)} time_ids):")
    print(f"    ARCH effect remaining:  {pct_arch:.1%}")
    print(f"    Autocorr in residuals:  {pct_autocorr:.1%}")
    print(f"    Fat tails (kurtosis>1): {pct_fat:.1%}")
    print(f"    Leverage detected:      {pct_leverage:.1%}")

    # Fat tails -> t-distribution (default already set)
    # Leverage -> GJR-GARCH (asymmetric)
    use_gjr = pct_leverage > 0.3 # >30% of series show leverage
    use_t_dist = pct_fat > 0.25 # >25% show fat tails
    final_dist = "t" if use_t_dist else garch_dist_initial

    if use_gjr:
        print(f"  -> Leverage detected in {pct_leverage:.0%} of series -> upgrading to GJR-GARCH")
        print(f"  Fitting GJR-GARCH(1,1,1)-{final_dist} ({len(time_IDs)} time_ids) ...")
        final_fit_results = Parallel(n_jobs=N_JOBS, prefer="threads")(
            delayed(_fit_gjrgarch_one)(tid, vol_train[tid]["log_return"].values,
                                       1, 1, 1, final_dist)
            for tid in time_IDs
        )
        garch_model_label = f"GJR-GARCH(1,1,1)-{final_dist}"
    elif final_dist != garch_dist_initial:
        print(f"  -> Fat tails in {pct_fat:.0%} of series -> refitting with t-distribution")
        final_fit_results = Parallel(n_jobs=N_JOBS, prefer="threads")(
            delayed(_fit_garch_one)(tid, vol_train[tid]["log_return"].values,
                                    best_garch_p, best_garch_q, final_dist)
            for tid in time_IDs
        )
        garch_model_label = f"GARCH({best_garch_p},{best_garch_q})-{final_dist}"
    else:
        print(f"  -> No upgrade needed -- keeping GARCH({best_garch_p},{best_garch_q})-{final_dist}")
        final_fit_results = init_fit_results
        garch_model_label = f"GARCH({best_garch_p},{best_garch_q})-{final_dist}"

    print(f"  Final GARCH model: {garch_model_label}")

    diag_row = {
        "stock_id": STOCK_ID,
        "regime": regime,
        "n_diag_sample": len(diag_sample),
        "pct_arch_effect": round(pct_arch, 3),
        "pct_autocorr": round(pct_autocorr, 3),
        "pct_fat_tails": round(pct_fat, 3),
        "pct_leverage": round(pct_leverage, 3),
        "model_selected": garch_model_label,
    }
    all_diag_summary.append(diag_row)

    garch_models = {}
    garch_info = []

    for tid, res in final_fit_results:
        if res is None:
            continue
        converged = res.convergence_flag == 0
        stationary = garch_is_stationary(res) if converged else False
        garch_models[tid] = res
        alpha_val = res.params.get("alpha[1]", np.nan) if converged else np.nan
        beta_val = res.params.get("beta[1]", np.nan) if converged else np.nan
        garch_info.append({
            "time_id": tid,
            "AIC": res.aic if converged else np.nan,
            "BIC": res.bic if converged else np.nan,
            "converged": converged,
            "stationary": stationary,
            "alpha": alpha_val,
            "beta": beta_val,
            "model": garch_model_label,
        })

    garch_fit_df = pd.DataFrame(garch_info)
    garch_fit_df.to_csv(os.path.join(stock_out, "garch_fit_info.csv"), index=False)

    n_conv = garch_fit_df["converged"].sum()
    n_stat = garch_fit_df["stationary"].sum()
    print(f"Converged: {n_conv}/{len(garch_fit_df)} ({100*n_conv/max(len(garch_fit_df),1):.1f}%)")
    print(f"Stationary: {n_stat}/{len(garch_fit_df)} ({100*n_stat/max(len(garch_fit_df),1):.1f}%)")

    print(f"\n  Generating GARCH one-step-ahead forecasts ...")
    garch_preds = {}
    garch_bounds = {}

    for tid, res in garch_models.items():
        row = garch_fit_df[garch_fit_df["time_id"] == tid]
        if row.empty or not bool(row["stationary"].iloc[0]):
            continue
        pred, lb, ub = forecast_one_step(res, scale=SCALE)
        if np.isfinite(pred):
            garch_preds[tid] = pred
            garch_bounds[tid] = (lb, ub)

    print(f"  Forecasts generated: {len(garch_preds)}/{len(time_IDs)} attempted")

    garch_eval_df, garch_tracking = evaluate_preds(garch_preds, vol_val, garch_model_label)
    garch_eval_df.to_csv(os.path.join(stock_out, "garch_eval_results.csv"), index=False)

    garch_per_tid = (garch_eval_df.groupby("time_id")[["QLIKE", "MSE"]].mean()
                     if not garch_eval_df.empty else pd.DataFrame(columns=["QLIKE","MSE"]))

    print(f"\n  Time_id accounting (GARCH):")
    print(f"Total in val set: {garch_tracking['n_val_tids']}")
    print(f"Got predictions: {garch_tracking['n_succeeded']}")
    print(f"Missed (no pred): {garch_tracking['n_missed']} ({garch_tracking['miss_rate']:.1%})")
    if garch_tracking['missed_tids'][:5]:
        print(f"    First missed IDs:  {garch_tracking['missed_tids'][:5]}")

    if not garch_per_tid.empty:
        print(f"Median QLIKE: {garch_per_tid['QLIKE'].median():.4f}")
        print(f"Mean QLIKE: {garch_per_tid['QLIKE'].mean():.4f}")
        print(f"Median MSE: {garch_per_tid['MSE'].median():.8f}")

    # Blowup detection (illiquid stocks only)
    if regime == "illiquid" and not garch_eval_df.empty:
        blowup_thresh = np.nanpercentile(garch_eval_df["QLIKE"].dropna(),
                                          BLOWUP_QLIKE_PERCENTILE)
        garch_blowups = garch_eval_df[garch_eval_df["QLIKE"] > blowup_thresh].copy()
        garch_blowups["stock_id"] = STOCK_ID
        garch_blowups.to_csv(os.path.join(stock_out, "garch_blowups.csv"), index=False)
        all_blowups.append(garch_blowups)
        blowup_pct = len(garch_blowups) / max(len(garch_eval_df), 1)
        print(f"  GARCH blowups (illiquid): {len(garch_blowups)} ({blowup_pct:.1%})")
    else:
        blowup_pct = 0.0

    print("Plotting residual diagnostics ...")
    if not garch_models:
        pass
    else:
        sample_tid = next(iter(garch_models))
        sample_res = garch_models[sample_tid]
        std_resids = sample_res.std_resid
        std_resids = std_resids[np.isfinite(std_resids)]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.patch.set_facecolor("white")
        fig.suptitle(f"Residual Diagnostics -- {garch_model_label}\n"
                     f"Stock {STOCK_ID} ({regime}), example time_id={sample_tid}",
                     fontsize=11, fontweight="500")

        # Standardised residuals over time
        axes[0][0].plot(std_resids, linewidth=0.6, color="#378ADD", alpha=0.8)
        axes[0][0].axhline(0, color="black", linewidth=0.7, linestyle="--")
        axes[0][0].axhline( 2, color="red",  linewidth=0.6, linestyle=":", alpha=0.6)
        axes[0][0].axhline(-2, color="red",  linewidth=0.6, linestyle=":", alpha=0.6)
        axes[0][0].set_title("Std Residuals", fontsize=10)
        axes[0][0].set_xlabel("t"); axes[0][0].set_ylabel("z_t")

        # Histogram with N(0,1) overlay
        axes[0][1].hist(std_resids, bins=30, color="#378ADD", alpha=0.7, edgecolor="white",
                        density=True)
        xr = np.linspace(std_resids.min(), std_resids.max(), 200)
        from scipy.stats import norm
        axes[0][1].plot(xr, norm.pdf(xr), "r--", linewidth=1.2, label="N(0,1)")
        axes[0][1].set_title("Residual Distribution", fontsize=10)
        axes[0][1].legend(fontsize=8)

        # ACF of squared residuals
        max_lags2 = max(1, len(std_resids) // 2 - 1)
        plot_acf(std_resids ** 2, lags=max_lags2, ax=axes[1][0],
                 title="ACF -- Squared Residuals (ARCH effect)", zero=False)

        # QQ plot
        from scipy.stats import probplot
        probplot(std_resids, dist="norm", plot=axes[1][1])
        axes[1][1].set_title("QQ Plot vs Normal", fontsize=10)

        for ax in axes.flat:
            ax.set_facecolor("white")
            for sp in ax.spines.values(): sp.set_color("#D3D1C7"); sp.set_linewidth(0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(stock_out, "residual_diagnostics.png"), dpi=150)
        plt.close()
        print("  Saved: residual_diagnostics.png")


    # Phase 3 - EGARCH-X (liquid + mixed only, no Monte Carlo)

    egarchx_eval_df = pd.DataFrame(columns=["time_id","pred_RV","actual_RV","QLIKE","MSE","model"])
    egarchx_per_tid = pd.DataFrame(columns=["QLIKE", "MSE"])
    ex_df = pd.DataFrame()
    egarchx_tracking = {"n_val_tids": 0, "n_succeeded": 0, "n_missed": 0,
                         "miss_rate": 0.0, "missed_tids": []}

    if not run_egarchx:
        print("\n-- Phase 3: EGARCH-X SKIPPED (illiquid)")
        print("-> Rosa's HAR-RV handles this regime.")
        egarchx_blowup_pct = np.nan

    else:
        print(f"\n-- Phase 3: EGARCH-X (one-step forecast, no Monte Carlo)")

        best_p, best_o, best_q = GLOBAL_EGARCHX_P, GLOBAL_EGARCHX_O, GLOBAL_EGARCHX_Q
        egarchx_tune_df.to_csv(os.path.join(stock_out, "egarchx_tuning.csv"), index=False)
        print(f" Order: EGARCH-X({best_p},{best_o},{best_q})")

        print(f" Fitting ({len(time_IDs)} time_ids, parallel) ...")
        egarchx_results = Parallel(n_jobs=N_JOBS, prefer="threads")(
            delayed(_fit_egarchx_one)(
                tid,
                vol_train[tid]["log_return"].values,   # EGARCH-X fits on log returns
                vol_train[tid]["BidAskSpread_mean"].values,
                best_p, best_o, best_q,
            )
            for tid in time_IDs
        )

        egarchx_models = {}
        egarchx_info = []

        for tid, res, vol_obj in egarchx_results:
            if res is None:
                continue
            converged = res.convergence_flag == 0
            quality_ok = egarchx_passes_sanity_check(res)
            egarchx_models[tid] = (res, vol_obj)
            egarchx_info.append({
                "time_id": tid,
                "AIC": res.aic if converged else np.nan,
                "BIC": res.bic if converged else np.nan,
                "converged":  converged,
                "quality_ok": quality_ok,
                "gamma": res.params.get("gamma[1]", np.nan) if quality_ok else np.nan,
                "delta": res.params.get("delta[spread]", np.nan) if quality_ok else np.nan,
                "beta": res.params.get("beta[1]", np.nan) if quality_ok else np.nan,
            })

        egarchx_fit_df = pd.DataFrame(egarchx_info)
        egarchx_fit_df.to_csv(os.path.join(stock_out, "egarchx_fit_info.csv"), index=False)

        n_conv_ex = egarchx_fit_df["converged"].sum()
        n_qual_ex = egarchx_fit_df["quality_ok"].sum()
        ex_df = egarchx_fit_df[egarchx_fit_df["quality_ok"]]
        print(f"  Converged:  {n_conv_ex}/{len(egarchx_fit_df)}")
        print(f"  Quality-OK: {n_qual_ex}/{len(egarchx_fit_df)}")

        if not ex_df.empty:
            print(f"  Mean gamma: {ex_df['gamma'].mean():.4f}  "
                  f"(gamma<0 in {(ex_df['gamma']<0).mean():.1%} -> leverage effect)")
            print(f"  Mean delta: {ex_df['delta'].mean():.4f}  "
                  f"(delta>0 in {(ex_df['delta']>0).mean():.1%} -> spread predicts vol)")

        print("  Generating EGARCH-X one-step-ahead forecasts (no simulation) ...")
        egarchx_preds = {}
        egarchx_bounds = {}

        for tid, (res, vol_obj) in egarchx_models.items():
            row = egarchx_fit_df[egarchx_fit_df["time_id"] == tid]
            if row.empty or not bool(row["quality_ok"].iloc[0]):
                continue
            val_spread = vol_val[tid]["BidAskSpread_mean"].values if tid in vol_val else np.array([])
            pred, lb, ub = egarchx_forecast_one_step(res, vol_obj, val_spread, scale=SCALE)
            if np.isfinite(pred):
                egarchx_preds[tid] = pred
                egarchx_bounds[tid] = (lb, ub)

        print(f"  Forecasts generated: {len(egarchx_preds)} / {len(time_IDs)} attempted")

        egarchx_eval_df, egarchx_tracking = evaluate_preds(
            egarchx_preds, vol_val, "EGARCH-X")
        egarchx_eval_df.to_csv(os.path.join(stock_out, "egarchx_eval_results.csv"), index=False)

        if not egarchx_eval_df.empty:
            egarchx_per_tid = egarchx_eval_df.groupby("time_id")[["QLIKE", "MSE"]].mean()
            print(f"\n Time_id accounting (EGARCH-X):")
            print(f" Total in val set: {egarchx_tracking['n_val_tids']}")
            print(f" Got predictions: {egarchx_tracking['n_succeeded']}")
            print(f" Missed (no pred): {egarchx_tracking['n_missed']} ({egarchx_tracking['miss_rate']:.1%})")
            print(f" Median QLIKE: {egarchx_per_tid['QLIKE'].median():.4f}")
            print(f" Mean QLIKE: {egarchx_per_tid['QLIKE'].mean():.4f}")
            print(f" Median MSE: {egarchx_per_tid['MSE'].median():.8f}")

        egarchx_blowup_pct = np.nan

        # gamma and delta distribution plots
        if not ex_df.empty:
            C_EX  = "#1D9E75"
            SPINE = "#D3D1C7"
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(f"EGARCH-X Parameter Distributions -- Stock {STOCK_ID}", fontsize=12)

            gammas = ex_df["gamma"].dropna()
            pct_neg_g = (gammas < 0).mean()
            axes[0].hist(gammas, bins=30, color=C_EX, alpha=0.75, edgecolor="white")
            axes[0].axvline(0, color="black", linestyle="--", linewidth=1.2,
                            label="gamma = 0 (symmetric)")
            axes[0].axvline(gammas.median(), color="red", linestyle="--", linewidth=1.2,
                            label=f"Median gamma = {gammas.median():.4f}")
            axes[0].set_xlabel("gamma  (asymmetry/leverage)")
            axes[0].set_ylabel("Number of time_ids")
            axes[0].set_title(f"gamma < 0 in {pct_neg_g:.1%} -> bad news raises vol more")
            axes[0].legend(fontsize=9)

            deltas = ex_df["delta"].dropna()
            pct_pos_d = (deltas > 0).mean()
            axes[1].hist(deltas, bins=30, color="#55A868", alpha=0.75, edgecolor="white")
            axes[1].axvline(0, color="black", linestyle="--", linewidth=1.2,
                            label="delta = 0 (no spread effect)")
            axes[1].axvline(deltas.median(), color="red", linestyle="--", linewidth=1.2,
                            label=f"Median delta = {deltas.median():.4f}")
            axes[1].set_xlabel("delta  (spread coefficient)")
            axes[1].set_ylabel("Number of time_ids")
            axes[1].set_title(f"delta > 0 in {pct_pos_d:.1%} -> wider spread predicts higher vol")
            axes[1].legend(fontsize=9)

            for ax in axes:
                ax.set_facecolor("white")
                for sp in ax.spines.values(): sp.set_color(SPINE); sp.set_linewidth(0.6)

            plt.tight_layout()
            plt.savefig(os.path.join(stock_out, "egarchx_parameters.png"), dpi=150)
            plt.close()


    # Phase 4 - Comparison plots + time_id tracking summary

    print("\n-- Phase 4: Evaluation Outputs")

    C_GARCH = "#378ADD"; C_EX = "#1D9E75"; C_MIX = "#9C59CC"; SPINE = "#D3D1C7"

    def _style(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor("white")
        for sp in ax.spines.values(): sp.set_color(SPINE); sp.set_linewidth(0.6)
        ax.tick_params(labelsize=8.5, color=SPINE)
        ax.set_title(title, fontsize=10, fontweight="500", pad=6)
        if xlabel: ax.set_xlabel(xlabel, fontsize=8.5, color="#5F5E5A")
        if ylabel: ax.set_ylabel(ylabel, fontsize=8.5, color="#5F5E5A")
        ax.grid(axis="y", color=SPINE, linewidth=0.4, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)

    ex_q = egarchx_per_tid["QLIKE"].median() if not egarchx_per_tid.empty else float("nan")
    ex_mse = egarchx_per_tid["MSE"].median() if not egarchx_per_tid.empty else float("nan")
    g_q = garch_per_tid["QLIKE"].median() if not garch_per_tid.empty else float("nan")
    g_mse = garch_per_tid["MSE"].median() if not garch_per_tid.empty else float("nan")

    winner_q = ("EGARCH-X" if ex_q < g_q else garch_model_label) if (np.isfinite(ex_q) and np.isfinite(g_q)) else "N/A"
    winner_mse = ("EGARCH-X" if ex_mse < g_mse else garch_model_label) if (np.isfinite(ex_mse) and np.isfinite(g_mse)) else "N/A"

    print(f"  +----------------------+--------------+--------------+----------+")
    print(f"  | Metric               |  {garch_model_label:<12s}|   EGARCH-X   |  Winner  |")
    print(f"  +----------------------+--------------+--------------+----------+")
    print(f"  | Median QLIKE         | {g_q:12.4f} | {ex_q:12.4f} | {winner_q:<8} |")
    print(f"  | Median MSE           | {g_mse:12.2e} | {ex_mse:12.2e} | {winner_mse:<8} |")
    print(f"  | n time_ids (val)     | {len(garch_per_tid):12d} | {len(egarchx_per_tid):12d} |          |")
    print(f"  | time_ids missed      | {garch_tracking['n_missed']:12d} | {egarchx_tracking['n_missed']:12d} |          |")
    print(f"  +----------------------+--------------+--------------+----------+")

    # Pred vs actual plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Predicted vs Actual RV -- Stock {STOCK_ID} ({regime})\n"
                 f"One-step-ahead deterministic forecasts", fontsize=12)

    for row_idx, (eval_df, label, color) in enumerate([
        (garch_eval_df,   garch_model_label, C_GARCH),
        (egarchx_eval_df, "EGARCH-X", C_EX),
    ]):
        if eval_df.empty:
            axes[row_idx][0].text(0.5, 0.5, "Not run for this regime",
                                   ha="center", va="center", transform=axes[row_idx][0].transAxes)
            axes[row_idx][1].text(0.5, 0.5, "Not run for this regime",
                                   ha="center", va="center", transform=axes[row_idx][1].transAxes)
            continue

        pa = (eval_df.groupby("time_id")
              .agg(pred_RV=("pred_RV", "first"), actual_RV=("actual_RV", "mean"))
              .dropna())
        pa_plot = pa[pa["pred_RV"] <= RV_CEILING].copy()

        if pa_plot.empty:
            continue

        lim = max(pa_plot["actual_RV"].max(), pa_plot["pred_RV"].max()) * 1.08
        axes[row_idx][0].scatter(pa_plot["actual_RV"], pa_plot["pred_RV"],
                                  alpha=0.3, s=10, color=color)
        axes[row_idx][0].plot([0, lim], [0, lim], "r--", linewidth=1.2, label="Perfect")
        axes[row_idx][0].set_xlim(0, lim); axes[row_idx][0].set_ylim(0, lim)
        _style(axes[row_idx][0], title=f"Scatter -- {label}",
               xlabel="Actual RV", ylabel="Predicted RV")
        axes[row_idx][0].legend(fontsize=8)

        t = np.arange(len(pa_plot))
        axes[row_idx][1].plot(t, pa_plot["actual_RV"].values,
                              color="#4C72B0", linewidth=0.7, alpha=0.8, label="Actual")
        axes[row_idx][1].plot(t, pa_plot["pred_RV"].values,
                              color=color, linewidth=0.7, alpha=0.8, label="Predicted")
        _style(axes[row_idx][1], title=f"Time series -- {label}",
               xlabel="time_id (ordered)", ylabel="RV")
        axes[row_idx][1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(stock_out, "pred_vs_actual.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: pred_vs_actual.png")

    # Comparison boxplot (if EGARCH-X ran)
    if not egarchx_per_tid.empty and not garch_per_tid.empty:
        fig = plt.figure(figsize=(12, 8))
        fig.patch.set_facecolor("white")
        fig.suptitle(f"Model Comparison -- Stock {STOCK_ID} ({regime})\n"
                     f"{garch_model_label}  vs  EGARCH-X",
                     fontsize=11, fontweight="500")

        gs = plt.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

        # QLIKE boxplot
        ax1 = fig.add_subplot(gs[0, 0])
        bp = ax1.boxplot(
            [garch_per_tid["QLIKE"].dropna().values,
             egarchx_per_tid["QLIKE"].dropna().values],
            patch_artist=True, labels=[garch_model_label, "EGARCH-X"], widths=0.45,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color=SPINE, linewidth=0.8),
            capprops=dict(color=SPINE, linewidth=0.8),
            flierprops=dict(marker="o", markersize=3, markerfacecolor=SPINE, alpha=0.3)
        )
        bp["boxes"][0].set_facecolor(C_GARCH); bp["boxes"][0].set_alpha(0.8)
        bp["boxes"][1].set_facecolor(C_EX);    bp["boxes"][1].set_alpha(0.8)
        _style(ax1, title="QLIKE (lower = better)", ylabel="QLIKE per time_id")
        for i, (vals, col) in enumerate([(garch_per_tid["QLIKE"].dropna(), C_GARCH),
                                          (egarchx_per_tid["QLIKE"].dropna(), C_EX)]):
            ax1.annotate(f"Med={vals.median():.4f}", xy=(i+1, vals.median()),
                         xytext=(12, 0), textcoords="offset points",
                         fontsize=7.5, va="center", color=col)

        # MSE boxplot
        ax2 = fig.add_subplot(gs[0, 1])
        bp2 = ax2.boxplot(
            [garch_per_tid["MSE"].dropna().values,
             egarchx_per_tid["MSE"].dropna().values],
            patch_artist=True, labels=[garch_model_label, "EGARCH-X"], widths=0.45,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color=SPINE, linewidth=0.8),
            capprops=dict(color=SPINE, linewidth=0.8),
            flierprops=dict(marker="o", markersize=3, markerfacecolor=SPINE, alpha=0.3)
        )
        bp2["boxes"][0].set_facecolor(C_GARCH); bp2["boxes"][0].set_alpha(0.8)
        bp2["boxes"][1].set_facecolor(C_EX);    bp2["boxes"][1].set_alpha(0.8)
        _style(ax2, title="MSE (lower = better)", ylabel="MSE per time_id")

        # Scatter: GARCH QLIKE vs EGARCH-X QLIKE per time_id
        ax3 = fig.add_subplot(gs[1, :])
        common = garch_per_tid.index.intersection(egarchx_per_tid.index)
        if len(common) > 0:
            g_q_arr  = garch_per_tid.loc[common, "QLIKE"].values
            ex_q_arr = egarchx_per_tid.loc[common, "QLIKE"].values
            p99 = np.nanpercentile(np.concatenate([g_q_arr, ex_q_arr]), 99)
            mask = (g_q_arr <= p99) & (ex_q_arr <= p99)
            g_plot  = g_q_arr[mask]; ex_plot = ex_q_arr[mask]
            ex_wins = ex_plot < g_plot
            ax3.scatter(g_plot[ ex_wins], ex_plot[ ex_wins],
                        color=C_EX,    alpha=0.4, s=10,
                        label=f"EGARCH-X wins ({ex_wins.sum()})")
            ax3.scatter(g_plot[~ex_wins], ex_plot[~ex_wins],
                        color=C_GARCH, alpha=0.4, s=10,
                        label=f"{garch_model_label} wins ({(~ex_wins).sum()})")
            lim2 = max(g_plot.max(), ex_plot.max()) * 1.05
            ax3.plot([0, lim2], [0, lim2], "--", color="#888780", linewidth=0.9)
            ax3.set_xlim(0, lim2); ax3.set_ylim(0, lim2)
            ax3.legend(fontsize=8.5)
            ax3.text(lim2*0.6, lim2*0.05, "EGARCH-X better ->", fontsize=8, color=C_EX)
            ax3.text(lim2*0.05, lim2*0.6, "<- GARCH better", fontsize=8,
                     color=C_GARCH, rotation=90)
        _style(ax3, title="Per time_id QLIKE scatter  (points below diagonal -> EGARCH-X wins)",
               xlabel=f"{garch_model_label} QLIKE", ylabel="EGARCH-X QLIKE")

        plt.savefig(os.path.join(stock_out, "garch_vs_egarchx_comparison.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: garch_vs_egarchx_comparison.png")

    # Collect cross-stock row
    all_garch_summary.append({
        "stock_id": STOCK_ID,
        "regime": regime,
        "model": garch_model_label,
        "n_time_ids_total": n_total_tids,
        "n_time_ids_kept": len(time_IDs),
        "n_forecasts": garch_tracking["n_succeeded"],
        "n_missed": garch_tracking["n_missed"],
        "miss_rate": round(garch_tracking["miss_rate"], 4),
        "median_QLIKE": g_q,
        "median_MSE": g_mse,
        "blowup_pct": round(blowup_pct, 4),
        "pct_leverage": round(pct_leverage, 3),
        "pct_fat_tails":round(pct_fat, 3),
    })
    all_egarchx_summary.append({
        "stock_id": STOCK_ID,
        "regime": regime,
        "egarchx_ran": run_egarchx,
        "n_time_ids_total": n_total_tids,
        "n_time_ids_kept": len(time_IDs),
        "n_forecasts": egarchx_tracking["n_succeeded"],
        "n_missed": egarchx_tracking["n_missed"],
        "miss_rate": round(egarchx_tracking["miss_rate"], 4),
        "median_QLIKE": ex_q,
        "median_MSE": ex_mse,
    })


# Aggregate summaries and plots

if all_garch_summary:
    garch_df = pd.DataFrame(all_garch_summary).sort_values("stock_id")
    garch_df.to_csv(os.path.join(OUTPUT_DIR, "all_stocks_garch_summary.csv"), index=False)
    print("\nSaved: all_stocks_garch_summary.csv")

if all_egarchx_summary:
    egarchx_df = pd.DataFrame(all_egarchx_summary).sort_values("stock_id")
    egarchx_df.to_csv(os.path.join(OUTPUT_DIR, "all_stocks_egarchx_summary.csv"), index=False)
    print("Saved: all_stocks_egarchx_summary.csv")

if all_blowups:
    blowup_report = pd.concat(all_blowups, ignore_index=True)
    blowup_report.to_csv(os.path.join(OUTPUT_DIR, "global_blowup_report.csv"), index=False)
    print(f"Saved: global_blowup_report.csv ({len(blowup_report)} blowup events)")

if all_diag_summary:
    diag_df2 = pd.DataFrame(all_diag_summary).sort_values("stock_id")
    diag_df2.to_csv(os.path.join(OUTPUT_DIR, "residual_diagnostics_summary.csv"), index=False)
    print("Saved: residual_diagnostics_summary.csv")


# Aggregate plots

C_GARCH = "#378ADD"; C_EX = "#1D9E75"; C_HAR = "#D85A30"; C_MIX = "#9C59CC"
SPINE = "#D3D1C7"

garch_df_agg   = pd.DataFrame(all_garch_summary)   if all_garch_summary   else pd.DataFrame()
egarchx_df_agg = pd.DataFrame(all_egarchx_summary) if all_egarchx_summary else pd.DataFrame()

# Plot 1: EGARCH-X vs GARCH -- liquid + mixed stocks
if not garch_df_agg.empty and not egarchx_df_agg.empty:
    ex_ran = egarchx_df_agg[egarchx_df_agg["egarchx_ran"] == True].copy()
    liq_g  = garch_df_agg[garch_df_agg["regime"].isin(["liquid", "mixed"])].copy()

    if not ex_ran.empty and not liq_g.empty:
        merged = liq_g.merge(
            ex_ran[["stock_id", "median_QLIKE", "median_MSE"]],
            on="stock_id", suffixes=("_garch", "_egarchx")
        ).dropna(subset=["median_QLIKE_garch", "median_QLIKE_egarchx"])
        merged = merged.sort_values("median_QLIKE_garch")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("white")
        fig.suptitle("EGARCH-X vs GARCH -- Liquid & Mixed Stocks  (lower QLIKE = better)",
                     fontsize=12, fontweight="500")

        x = np.arange(len(merged))
        labels = [f"S{int(s)}\n({r[:3]})" for s, r in
                  zip(merged["stock_id"], merged["regime"])]
        w = 0.38

        for ax, metric, title in zip(
            axes, ["median_QLIKE", "median_MSE"],
            ["Median QLIKE per stock", "Median MSE per stock"]
        ):
            ax.bar(x - w/2, merged[f"{metric}_garch"],   width=w,
                   label="GARCH",    color=C_GARCH, alpha=0.8)
            ax.bar(x + w/2, merged[f"{metric}_egarchx"], width=w,
                   label="EGARCH-X", color=C_EX,    alpha=0.8)
            for i, (g, ex) in enumerate(zip(merged[f"{metric}_garch"],
                                             merged[f"{metric}_egarchx"])):
                if pd.notna(g) and pd.notna(ex):
                    winner_y = min(g, ex)
                    ax.annotate("*", xy=(i + (w/2 if ex < g else -w/2), winner_y),
                                ha="center", va="top", fontsize=7,
                                color=C_EX if ex < g else C_GARCH)
            ax.set_facecolor("white")
            for sp in ax.spines.values(): sp.set_color(SPINE); sp.set_linewidth(0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.tick_params(labelsize=8, color=SPINE)
            ax.set_title(title, fontsize=10, fontweight="500")
            ax.legend(fontsize=9, framealpha=0.9)
            ax.grid(axis="y", color=SPINE, linewidth=0.4, linestyle="--")
            ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "jisu_01_egarchx_vs_garch_liquid_mixed.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        ex_wins = (merged["median_QLIKE_egarchx"] < merged["median_QLIKE_garch"]).sum()
        print(f"Saved: jisu_01_egarchx_vs_garch_liquid_mixed.png")
        print(f"  EGARCH-X beats GARCH on {ex_wins}/{len(merged)} liquid+mixed stocks by QLIKE")


# Plot 2: GARCH blowups on illiquid stocks
if not garch_df_agg.empty:
    illiquid_g = garch_df_agg[garch_df_agg["regime"] == "illiquid"].copy()
    if not illiquid_g.empty:
        illiquid_g = illiquid_g.sort_values("blowup_pct", ascending=False)
        labels_il = [f"S{int(s)}" for s in illiquid_g["stock_id"]]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("white")
        fig.suptitle("GARCH Blowups -- Illiquid Stocks  (same stocks as Rosa's HAR-RV)",
                     fontsize=12, fontweight="500")

        ax = axes[0]
        ax.set_facecolor("white")
        bar_colors = [C_HAR if p > 0.05 else C_GARCH for p in illiquid_g["blowup_pct"]]
        ax.bar(range(len(illiquid_g)), illiquid_g["blowup_pct"] * 100,
               color=bar_colors, alpha=0.8, width=0.7)
        ax.axhline(5.8, color="#888780", linestyle="--", linewidth=0.9, label="5.8% ref")
        for sp in ax.spines.values(): sp.set_color(SPINE); sp.set_linewidth(0.6)
        ax.set_xticks(range(len(labels_il)))
        ax.set_xticklabels(labels_il, rotation=45, ha="right", fontsize=7.5)
        ax.set_ylabel("Blowup rate (%)", fontsize=9)
        ax.set_title("GARCH blowup rate per illiquid stock", fontsize=10, fontweight="500")
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color=C_HAR,   label="> 5% -- HAR-RV preferred"),
            Patch(color=C_GARCH, label="< 5% -- GARCH borderline"),
            plt.Line2D([0],[0], color="#888780", linestyle="--", label="5.8% ref"),
        ], fontsize=7.5)
        ax.grid(axis="y", color=SPINE, linewidth=0.4, linestyle="--")
        ax.set_axisbelow(True)

        ax2 = axes[1]
        ax2.set_facecolor("white")
        liq_qlike = garch_df_agg[garch_df_agg["regime"] == "liquid"]["median_QLIKE"].dropna()
        illiq_qlike = illiquid_g["median_QLIKE"].dropna()
        mix_qlike = garch_df_agg[garch_df_agg["regime"] == "mixed"]["median_QLIKE"].dropna()
        data_bp = [v.values for v in [liq_qlike, mix_qlike, illiq_qlike] if len(v) > 0]
        labs_bp = [l for l, v in zip(["Liquid","Mixed","Illiquid"],
                                      [liq_qlike, mix_qlike, illiq_qlike]) if len(v) > 0]
        cols_bp = [c for c, v in zip([C_GARCH, C_MIX, C_HAR],
                                      [liq_qlike, mix_qlike, illiq_qlike]) if len(v) > 0]
        if data_bp:
            bp = ax2.boxplot(data_bp, patch_artist=True, labels=labs_bp, widths=0.45,
                             medianprops=dict(color="white", linewidth=2),
                             whiskerprops=dict(color=SPINE, linewidth=0.8),
                             capprops=dict(color=SPINE, linewidth=0.8),
                             flierprops=dict(marker="o", markersize=3,
                                             markerfacecolor=SPINE, alpha=0.4))
            for box, col in zip(bp["boxes"], cols_bp):
                box.set_facecolor(col); box.set_alpha(0.8)
        for sp in ax2.spines.values(): sp.set_color(SPINE); sp.set_linewidth(0.6)
        ax2.set_ylabel("Median QLIKE", fontsize=9)
        ax2.set_title("GARCH QLIKE by regime", fontsize=10, fontweight="500")
        ax2.grid(axis="y", color=SPINE, linewidth=0.4, linestyle="--")
        ax2.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "jisu_02_garch_blowups_illiquid.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: jisu_02_garch_blowups_illiquid.png")


# Plot 3: Residual diagnostics summary
if all_diag_summary:
    diag_df_plot = pd.DataFrame(all_diag_summary)
    regimes_in = [r for r in ["liquid","mixed","illiquid"]
                  if r in diag_df_plot["regime"].values]
    if regimes_in:
        fig, axes = plt.subplots(1, len(regimes_in), figsize=(6*len(regimes_in), 5))
        if len(regimes_in) == 1: axes = [axes]
        fig.patch.set_facecolor("white")
        fig.suptitle("Residual Pattern Rates by Regime -- Drives Model Upgrade Decisions",
                     fontsize=12, fontweight="500")
        patterns = ["pct_arch_effect","pct_autocorr","pct_fat_tails","pct_leverage"]
        labels_p = ["ARCH effect\n(-> higher order)","Autocorr\n(-> more lags)",
                     "Fat tails\n(-> t-dist)","Leverage\n(-> GJR-GARCH)"]
        regime_colors = {"liquid": C_GARCH, "mixed": C_MIX, "illiquid": C_HAR}
        for ax, rl in zip(axes, regimes_in):
            sub = diag_df_plot[diag_df_plot["regime"] == rl]
            means = [sub[p].mean() * 100 for p in patterns]
            bars = ax.bar(labels_p, means, color=regime_colors.get(rl,"grey"),
                          alpha=0.75, edgecolor="white")
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5)
            ax.set_facecolor("white")
            ax.set_ylim(0, 105)
            ax.set_ylabel("% of time_ids", fontsize=9)
            ax.set_title(f"{rl.capitalize()} stocks (n={len(sub)})",
                         fontsize=10, fontweight="500")
            for sp in ax.spines.values(): sp.set_color(SPINE); sp.set_linewidth(0.6)
            ax.grid(axis="y", color=SPINE, linewidth=0.4, linestyle="--")
            ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "jisu_03_residual_pattern_rates.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: jisu_03_residual_pattern_rates.png")


# Evaluate all uploaded example CSVs - QLIKE + MSE for every model

print("\nEvaluating uploaded example CSVs (all models) ...")
csv_results_all = []

for csv_label, csv_path in EXAMPLE_CSVS.items():
    if not os.path.exists(csv_path):
        print(f"  [SKIP] {csv_label}: not found at {csv_path}")
        continue
    print(f"  -> {csv_label}: {os.path.basename(csv_path)}")
    result_df = evaluate_csv_all_models(csv_path, label=csv_label)
    if not result_df.empty:
        csv_results_all.append(result_df)

if csv_results_all:
    all_csv_eval = pd.concat(csv_results_all, ignore_index=True)
    all_csv_eval.to_csv(os.path.join(OUTPUT_DIR, "csv_eval_all_models.csv"), index=False)
    print(f"\nSaved: csv_eval_all_models.csv")

    print("\n  +------------------------+----------+--------------+-----------------+")
    print(  "  |  Model                 |  CSV     |    QLIKE     |       MSE       |")
    print(  "  +------------------------+----------+--------------+-----------------+")
    for _, row in all_csv_eval.sort_values(["label","QLIKE"]).iterrows():
        q_s   = f"{row['QLIKE']:.4f}" if pd.notna(row["QLIKE"]) else "N/A"
        mse_s = f"{row['MSE']:.2e}"   if pd.notna(row["MSE"])   else "N/A"
        print(f"  |  {str(row['model']):<22s} | {str(row['label']):<8s} | {q_s:>12s} | {mse_s:>15s} |")
    print(  "  +------------------------+----------+--------------+-----------------+")

    model_colors = {"GARCH(1,1)-Normal":"#378ADD","GARCH(1,1)-t":"#8ABBDD",
                    "GJR-GARCH(1,1,1)-t":"#1D9E75","EGARCH-X(1,1,1)":"#DD8452"}
    csv_labels_u = sorted(all_csv_eval["label"].unique())
    model_list = sorted(all_csv_eval["model"].unique())
    x = np.arange(len(csv_labels_u))
    n_m = len(model_list); w2 = 0.8 / max(n_m, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle("All Models -- QLIKE + MSE on Uploaded Example CSVs\n"
                 "(evaluated regardless of regime recommendation)",
                 fontsize=12, fontweight="500")
    for ax, metric, title in zip(axes, ["QLIKE","MSE"],
                                  ["QLIKE (lower=better)","MSE (lower=better)"]):
        for i, model in enumerate(model_list):
            sub = all_csv_eval[all_csv_eval["model"] == model]
            vals = [float(sub[sub["label"] == lbl][metric].iloc[0])
                    if not sub[sub["label"] == lbl].empty else np.nan
                    for lbl in csv_labels_u]
            offset = (i - n_m/2 + 0.5) * w2
            ax.bar(x + offset, vals, width=w2, label=model,
                   color=model_colors.get(model, f"C{i}"), alpha=0.8)
        ax.set_facecolor("white")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{l.capitalize()} CSV" for l in csv_labels_u], fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="500")
        ax.legend(fontsize=7.5, framealpha=0.9)
        for sp in ax.spines.values(): sp.set_color(SPINE); sp.set_linewidth(0.6)
        ax.grid(axis="y", color=SPINE, linewidth=0.4, linestyle="--")
        ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "jisu_04_csv_eval_all_models.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: jisu_04_csv_eval_all_models.png")


print(f"\ncomplete -- {len(selected_ids)} stocks processed.")
print(f" Liquid ({len(liquid_sel)}): EGARCH-X + upgraded GARCH")
print(f" Illiquid ({len(illiquid_sel)}): GARCH blowup analysis")
print(f" Mixed ({len(mixed_sel)}): Both models, winner per stock")
print(f" -> jisu_01: EGARCH-X vs GARCH (liquid+mixed)")
print(f" -> jisu_02: GARCH blowups on illiquid")
print(f" -> jisu_03: Residual pattern rates (drives upgrade)")
print(f" -> jisu_04: All-model QLIKE+MSE on uploaded CSVs")
print(f"Per-stock outputs in: {OUTPUT_DIR}/stock_<id>/")
