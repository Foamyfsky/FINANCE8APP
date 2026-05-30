# DATA3888 Group 8 - Finance Capstone
## Dynamic Volatility Forecasting: A Liquidity-Aware Regime Strategy

This folder contains everything needed to read our report, view the figures, run the
interactive dashboard, and inspect the model code.

---

## 1. Just want to read the report and see the plots?

Open this file in any web browser:

```
report.html
```

That is the full written report with every figure already rendered inside it. You do not
need Python, Quarto, or any setup to read it. Just double-click `report.html` and it opens
in your browser. Each code block in the report is collapsed by default; click "Show code"
on any figure to see the code that produced it.

If you prefer to re-render the report yourself from source, the source file is `report.qmd`
(see Section 4 below).

All the figures used in the report are also saved as image files in:

```
m4_outputs/report_figures/
    fig1_regime_classification.png      (liquidity regime split across 60 stocks)
    fig2_session_distribution.png       (per-session RV and BAS by regime)
    fig3_per_stock_qlike.png            (per-stock model comparison)
    fig4_temporal_stability.png         (LightGBM vs HAR-RV win rate over time)
    fig5_dashboard.png                  (screenshot of the interactive tool)
```

---

## 2. Want to try the interactive dashboard?

The live version (no setup needed) is hosted here:

```
https://ji1isu.github.io/FINANCE8APP/final_tool_v4.html
```

To run it locally instead, the dashboard reads its data from JSON files using `fetch()`,
which browsers block when you open the HTML directly. You need a small local web server:

```bash
cd FINANCE8APP
python -m http.server 8080
```

Then open `http://localhost:8080/final_tool_v4.html` in your browser. (The VS Code
"Live Server" extension also works: right-click `final_tool_v4.html` and choose
"Open with Live Server".)

---

## 3. Where is each part of the analysis implemented?

### Data aggregation (the raw order book to our working dataset)

The raw Optiver data is 126 individual per-stock order book CSV files. We aggregated these
into a single dataset (`optiver_aggregated.csv`) where each row is one 30-second bucket with
its WAP, bid-ask spread, and log return. That aggregation code is in:

```
asst.ipynb
```

This notebook reads the 126 raw `individual_book_train/*.csv` files, computes WAP and
bid-ask spread per snapshot, calculates log returns, assigns each row to a 30-second bucket,
and writes the merged `optiver_aggregated.csv`. Every model below reads from that aggregated
file. (When the report says "we aggregated the stock files", this notebook is what did it.)

### The four forecasting models

| Model | Implemented in | What it does |
|---|---|---|
| GJR-GARCH and EGARCH-X | `arma_jisu.py` | Fits the GARCH-family models per stock, including the adaptive choice between GJR-GARCH and plain GARCH based on the leverage test. Writes per-stock evaluation CSVs and `all_stocks_garch_summary.csv` / `all_stocks_egarchx_summary.csv`. |
| HAR-RV and WLS | `rosa.py` | Fits the HAR-RV (OLS) and weighted-least-squares variants. Writes `all_stocks_har_rv_summary.csv` and the per-stock `rosa_har_rv_eval_results.csv`. |
| LightGBM | `lgbm.py` | Trains the gradient-boosted tree model per liquidity regime on the engineered features, using a fixed 80/20 temporal split. Writes `lgbm_outputs/lgbm_eval_results.csv`, `lgbm_per_stock.csv`, and `lgbm_feature_importance.csv`. |

### Liquidity classification, stock selection, and the router

| Purpose | File |
|---|---|
| EDA + bucket-level and stock-level liquidity classification | `eda_and_wls_jamie.py` (and `jamie_eda.py`) |
| Stock selection and shared config (regimes, seed = 42, train/val sizes) | `config.py`, `select_stocks.py` |
| Pipeline router that compares all models and picks the winner per stock | `pipline.py` (run with `--all`) |
| Report figures (per-stock comparison panels, summary panels) | `final_figures.py` |
| Builds the per-stock JSON files the dashboard reads | `generate_stock_data.py` |

### Where the numbers in the report come from

The headline QLIKE comparison across all 60 stocks lives in:

```
m4_outputs/pipeline/pipeline_all_stocks.csv
```

This is produced by `pipline.py --all`, which reads each model's summary CSV and records the
median QLIKE per stock plus the recommended model.

---

## 4. Re-rendering the report from source (optional)

The report source is `report.qmd`. To rebuild `report.html` yourself:

Requirements:
- [Quarto](https://quarto.org/docs/get-started/)
- Python 3.x with `pandas`, `numpy`, `matplotlib`, plus `jupyter` and `nbclient` for Quarto

```bash
cd FINANCE8APP
quarto render report.qmd
```

This regenerates `report.html`. The report loads pre-computed results from `m4_outputs/`,
so no models are re-trained during rendering.

---

## 5. Re-running the whole pipeline from scratch (optional)

Only needed if you want to retrain every model. This requires the raw dataset, which is not
included here because of its size. It is available from the Optiver competition on Kaggle:
<https://www.kaggle.com/c/optiver-realized-volatility-prediction>.

Step 1 produces `optiver_aggregated.csv` from the raw order book files:

```
Run asst.ipynb   (raw individual_book_train/*.csv  ->  optiver_aggregated.csv)
```

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
| Mixed | LightGBM | 16 / 20 |

Theory and empirical winner agree on 52 of 60 stocks.
