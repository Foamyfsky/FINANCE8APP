# DATA3888 Group 8 - Finance Capstone
## Dynamic Volatility Forecasting: A Liquidity-Aware Regime Strategy

This folder contains the reproducible report, cached model outputs, model scripts, report
figures, and interactive dashboard for the DATA3888 Finance Group 8 capstone project.

Public repository:

```
https://github.com/ji1isu/FINANCE8APP
```

---

## 1. Report files

The rendered report is:

```
report.html
```

This is the full written report with all five figures embedded. It can be opened directly
in a web browser. Code blocks are collapsed by default; use "Show code" in the report to
inspect the code linked to each displayed analysis.

The source file used to generate the report is:

```
report.qmd
```

All the figures used in the report are embedded in `report.html`. For re-rendering from
`report.qmd`, the same figures should also be available as image files in:

```
m4_outputs/report_figures/
    fig1_regime_classification.png      (liquidity regime split across 60 stocks)
    fig2_session_distribution.png       (per-session RV and BAS by regime)
    fig3_per_stock_qlike.png            (per-stock model comparison)
    fig4_temporal_stability.png         (LightGBM vs HAR-RV win rate over time)
    fig5_dashboard.png                  (screenshot of the interactive tool)
```

---

## 2. Interactive dashboard

The deployed dashboard is hosted at:

```
https://ji1isu.github.io/FINANCE8APP/final_tool_v4.html
```

To run the dashboard from the submitted folder, use a local web server. This is necessary
because the dashboard reads JSON files using `fetch()`, which browsers block when the HTML
file is opened directly:

```bash
cd FINANCE8APP
python -m http.server 8080
```

Then open:

```
http://localhost:8080/final_tool_v4.html
```

---

## 3. Analysis implementation map

### Data aggregation (the raw order book to our working dataset)

The raw Optiver data is 112 individual per-stock order book CSV files. We aggregated these
into a single working dataset where each row is one 30-second bucket with its WAP, bid-ask
spread, and log return. The raw data and full aggregation notebook are not stored in this
submitted folder because of file size, but the cached model outputs used by the report are
stored under `m4_outputs/`.

The aggregation step reads the raw `individual_book_train/*.csv` files, computes WAP and
bid-ask spread per snapshot, calculates log returns, assigns each row to a 30-second bucket,
and writes the merged working dataset. Every model below reads from that aggregated data.

### The four forecasting models

| Model | Implemented in | What it does |
|---|---|---|
| GJR-GARCH and EGARCH-X | `arma_jisu.py` | Fits the GARCH-family models per stock, including the adaptive choice between GJR-GARCH and plain GARCH based on the leverage test. Writes per-stock evaluation CSVs and `all_stocks_garch_summary.csv` / `all_stocks_egarchx_summary.csv`. |
| HAR-RV and WLS | `rosa.py` | Fits the HAR-RV (OLS) and weighted-least-squares variants. Writes `all_stocks_har_rv_summary.csv` and the per-stock `rosa_har_rv_eval_results.csv`. |
| LightGBM | `lgbm.py` | Trains the gradient-boosted tree model per liquidity regime on the engineered features, using a fixed 80/20 temporal split. Writes `lgbm_outputs/lgbm_eval_results.csv`, `lgbm_per_stock.csv`, and `lgbm_feature_importance.csv`. |

### Liquidity classification, stock selection, and the router

| Purpose | File |
|---|---|
| EDA + bucket-level and stock-level liquidity classification | `eda_and_wls_jamie.py` / `jamie_eda.py`; cached outputs are in `m4_outputs/jamie_*.csv` |
| Stock selection and shared config (regimes, seed = 42, train/val sizes) | `config.py`, `select_stocks.py` |
| Pipeline router that compares all models and picks the winner per stock | `pipline.py` (run with `--all`) |
| Report figures (per-stock comparison panels, summary panels) | `final_figures.py` |
| Builds the per-stock JSON files the dashboard reads | `generate_stock_data.py` |

### Report output source

The headline QLIKE comparison across all 60 stocks is stored in:

```
m4_outputs/pipeline/pipeline_all_stocks.csv
```

This is produced by `pipline.py --all`, which reads each model's summary CSV and records the
median QLIKE per stock plus the recommended model.

---

## 4. Re-rendering the report from source

To rebuild `report.html` from `report.qmd`:

Requirements:
- [Quarto](https://quarto.org/docs/get-started/)
- Python 3.x with `pandas`, `numpy`, `matplotlib`, plus `jupyter` and `nbclient` for Quarto

```bash
cd FINANCE8APP
quarto render report.qmd
```

This command regenerates `report.html`. The report loads pre-computed results from `m4_outputs/`,
so no models are re-trained during rendering.

---

## 5. Re-running the full pipeline

This step is only required if every model must be retrained from raw data. It requires the
raw Optiver dataset, which is not included here because of its size. It is available from
the Optiver competition on Kaggle:
<https://www.kaggle.com/c/optiver-realized-volatility-prediction>.

Step 1 produces the aggregated working dataset from the raw order book files. Use the
aggregation notebook from the full project repository if rebuilding from raw data.

Then run the scripts in this order:

```bash
python config.py             # stock selection
python eda_and_wls_jamie.py  # EDA + liquidity classification
python arma_jisu.py          # GJR-GARCH + EGARCH-X
python rosa.py               # HAR-RV + WLS
python lgbm.py               # LightGBM
python pipline.py --all      # compare models, write pipeline_all_stocks.csv
python final_figures.py      # report figures
python generate_stock_data.py # rebuild dashboard JSON files
```

Every random seed is fixed at 42 for full reproducibility.

---

## 6. Key results at a glance

| Regime | Best model | Stocks won |
|---|---|---|
| Liquid | GJR-GARCH | 19 / 20 |
| Illiquid | LightGBM | 20 / 20 |
| Mixed | LightGBM | 13 / 20 |

Theory and empirical winner agree on 52 of 60 stocks. In mixed stocks, LightGBM beats the
GARCH-family baseline in 16 of 20 cases, but the exact all-model winner count is 13 of 20
because EGARCH-X wins the remaining 7 mixed stocks.
