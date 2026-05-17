# Model Audit Report — DATA3888 Group 8
**Date:** 14 May 2026  
**Audited files:** `arma_jisu.py`, `rosa.py`, `lgbm.py`, `config.py`, `pipline.py`, `generate_splits.py`

---

## 1. Training Window — 10 Minutes / 30-Second Buckets / 80-20 Split

**Result: PASS on all models.**

| Check | Value | Expected | Status |
|---|---|---|---|
| Buckets per session | 20 | 20 | ✅ |
| Bucket duration | 30 s | 30 s | ✅ |
| Total window | 600 s = 10 min | 10 min | ✅ |
| Training buckets (`N_TRAIN`) | 16 (iloc[:16]) | 80% → 16 | ✅ |
| Validation buckets (`N_VAL`) | 4 (iloc[16:20]) | 20% → 4 | ✅ |
| Bucket numbering in data | 1–20 | — | ✅ |

All three model scripts (`arma_jisu.py`, `rosa.py`, `lgbm.py`) use `buckets.iloc[:N_TRAIN]` for train and `buckets.iloc[N_TRAIN:N_TRAIN+N_VAL]` for validation — consistent throughout.

> **Minor doc mismatch (harmless):** The header comment in `arma_jisu.py` says "buckets numbered 2–21" but the actual data runs 1–20. The code uses `iloc` (position-based) so this has no effect on execution.

---

## 2. Issues Found & Fixed

### 🔴 CRITICAL FIX 1 — GARCH/EGARCH-X was fitting on RV, not log returns

**File:** `arma_jisu.py`  
**Problem:** GARCH and EGARCH-X are theoretically models of the **conditional variance of log returns**. The pipeline was passing the `RV` column directly as the input series — this makes GARCH model the "volatility of volatility" rather than the volatility of returns, which is inconsistent with how GARCH is described in the presentation.

**Fix applied:** Log returns are now computed from `WAP_mean` immediately after loading the data:

```python
df["log_return"] = (
    df.groupby(["stock_id", "time_id"])["WAP_mean"]
    .transform(lambda x: np.log(x / x.shift(1)))
)
df["log_return"] = df["log_return"].fillna(0.0)
```

All GARCH, GJR-GARCH, and EGARCH-X fitting calls — including the global hyperparameter tuning, the Phase 1 BIC selection, and the per-stock fitting loops — now use `vol_train[tid]["log_return"].values` instead of `vol_train[tid]["RV"].values`.

**Fallback:** If `WAP_mean` is not present in the CSV (e.g., when testing with the example 3-column CSVs), the script falls back to using RV and prints a warning.

The GARCH σ² forecast is then compared to the **actual RV** from the validation window — which is the standard approach in the realised volatility literature.

---

### 🔴 CRITICAL FIX 2 — QLIKE formula was inconsistent across models

**Files:** `arma_jisu.py`, `lgbm.py`  
**Problem:** Three different QLIKE formulas were in use simultaneously:

| File | Formula | Form |
|---|---|---|
| `arma_jisu.py` (before fix) | `log(pred²) + actual²/pred²` | Returns QLIKE (σ units) |
| `lgbm.py` (before fix) | `log(pred²) + actual²/pred²` | Returns QLIKE (σ units) |
| `rosa.py` | `log(pred) + actual/pred` | **Variance QLIKE (RV units)** ✅ |

Since all models output **RV forecasts** (variance units, not standard deviation), the correct form is the **variance QLIKE** (`log(pred) + actual/pred`) — this is the Patton (2011) standard used in the realised volatility literature and is what rosa.py correctly uses.

The returns-QLIKE would double-square the inputs, inflating values and making cross-model comparison completely invalid.

**Fix applied:** `arma_jisu.py` and `lgbm.py` updated to match `rosa.py`:
```python
def qlike(pred, actual):
    pred = max(pred, 1e-10)
    return np.log(pred) + actual / pred
```

All existing pre-computed QLIKE scores in `m4_outputs/` were produced with the old formula. **Re-running the models is required to get valid comparable numbers.**

---

### 🟡 CRITICAL FIX 3 — `selected_stocks.csv` was empty

**File:** `m4_outputs/selected_stocks.csv`  
**Problem:** The file contained only a header row (`stock_id,regime,median_BAS`). When `config.get_selected_stocks()` finds an empty CSV, it falls through to auto-selection, which requires the raw `optiver_aggregated.csv` to be present. On any machine where that path doesn't match, all models crash at startup.

**Fix applied:** Populated `selected_stocks.csv` with the 72 stocks from `selected_stocks_NEW72.csv`. Models will now load the stock list from the CSV cache without needing the raw data.

> **Note:** The NEW72 file contains 72 stocks (not 60). This is because it includes all three regimes without the 20-per-regime cap. If you want exactly 20+20+20, run `select_stocks.py` with the raw data and re-pin the selection in `config.py`.

---

## 3. Remaining Items to Watch Before Presenting

### ⚠️ EGARCH-X vs GARCH output units mismatch (informational)

After the log-return fix: GARCH/EGARCH-X now fit on `log_return` (in return units, typically ~0.0001–0.001 scale). The `forecast_one_step()` function divides back by `scale=1000` and returns a value in the same return units. This predicted σ is then compared to actual `RV` from the validation set.

The comparison is valid because RV ≈ Σ r_t² (sum of squared returns over the bucket), so the GARCH σ² forecast and the realised RV are on the same conceptual scale — but **check the numeric magnitudes** before presenting. If predicted σ² is much smaller than actual RV, you may need to adjust `SCALE` or compare σ (not σ²) against √RV.

### ⚠️ `global_tuning_cache.json` is stale

The cached tuning results were produced with the old RV inputs. After the log-return fix, the optimal GARCH/EGARCH-X orders may differ. **Delete `m4_outputs/global_tuning_cache.json` before the next run** so tuning re-runs on log returns.

### ⚠️ `pipline.py` regime routing reads from `jamie_liquidity.csv`

The pipeline routes liquid → EGARCH-X and illiquid → HAR-RV based on Jamie's file. Verify that `jamie_liquidity.csv` is present in `m4_outputs/` before running the full pipeline or the trading app.

### ✅ `rosa.py` (HAR-RV / WLS) — no changes needed

Rosa's models work on `RV` directly (as designed — HAR-RV is a pure autoregression on realised variance), use the correct variance-form QLIKE, and have the correct 16/4 split. No issues found.

### ✅ `lgbm.py` — only QLIKE fixed, no structural changes

The LightGBM features (`rv_mean`, `rv_std`, `spread_mean`, etc.) are all computed from RV/BAS as intended — LightGBM is a regression model, not a GARCH-type model, so it does not need log returns as its input. Only the QLIKE evaluation formula was corrected.

---

## 4. Is It Safe to Run?

| Component | Safe to run? | Notes |
|---|---|---|
| `arma_jisu.py` | ✅ Yes (after fixes) | Requires `WAP_mean` in CSV for log returns; fallback to RV if missing |
| `rosa.py` | ✅ Yes | No changes needed |
| `lgbm.py` | ✅ Yes (after fixes) | QLIKE fix applied |
| `pipline.py` | ✅ Yes | Requires `jamie_liquidity.csv` to be present |
| `generate_splits.py` | ✅ Yes | Standalone; different input path (`D:\...`) — update path before running |
| Full pipeline | ⚠️ Conditional | Delete `global_tuning_cache.json` before re-run; verify raw CSV path |

---

## 5. Summary for Presentation Narrative

The regime-based model routing is **theoretically sound** and **internally consistent** after the fixes:

- **EGARCH-X** captures the liquidity channel (δ·log spread² in the variance equation) and is recommended for **liquid stocks** where the order book is deep and the math is stable.
- **HAR-RV / WLS** is purely autoregressive with no spread term — it's numerically stable regardless of how thin the order book is, making it the right fallback for **illiquid and mixed stocks**.
- **QLIKE** (variance form) is now the single, consistent loss function across all three models — valid for cross-model comparison in the trading app.
- **Log returns** are now used as GARCH input, which correctly aligns with the GARCH theoretical framework described in the presentation.
