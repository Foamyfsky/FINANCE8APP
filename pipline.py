"""
pipeline file

Usage:
  python pipeline.py --stock_id 42
  python pipeline.py --stock_id 42 --time_id 15
  python pipeline.py --all (runs comparison across all selected stocks)

purpose:
  1. Loads Jamie's liquidity classification for the stock
  2. Routes to the correct model's pre-computed forecast:
       liquid -> EGARCH-X (with GARCH as baseline)
       illiquid -> HAR-RV (with GARCH as baseline)
  3. Outputs a side-by-side comparison table + plot
  4. Flags if Jisu's outputs are still placeholder data

The pipeline never re-fits any model. It only reads from CSV outputs.
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

from config import OUTPUT_DIR, JAMIE_LIQUIDITY_CSV, get_liquidity_map, get_selected_stocks
PIPELINE_OUT = os.path.join(OUTPUT_DIR, "pipeline")
LGBM_OUT = os.path.join(OUTPUT_DIR, "lgbm_outputs")

os.makedirs(PIPELINE_OUT, exist_ok=True)


def load_liquidity():
    if not os.path.exists(JAMIE_LIQUIDITY_CSV):
        raise FileNotFoundError(
            "jamie_liquidity.csv not found. Run jamie_eda.py first."
        )
    df = pd.read_csv(JAMIE_LIQUIDITY_CSV)
    return df.set_index("stock_id")


def load_model_eval(stock_id, filename):
    """Load a per-stock eval CSV. Returns empty DataFrame if not found."""
    path = os.path.join(OUTPUT_DIR, f"stock_{stock_id}", filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def load_lgbm_stock(stock_id):
    """Load LightGBM per-stock result row. Returns dict or None."""
    path = os.path.join(LGBM_OUT, "lgbm_per_stock.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    row = df[df["stock_id"] == stock_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def is_placeholder():
    return os.path.exists(os.path.join(OUTPUT_DIR, "jisu_is_placeholder.json"))


def clear_placeholder_if_real():
    """
    Delete jisu_is_placeholder.json automatically if real GARCH outputs exist.
    Called at startup - no manual deletion needed.
    """
    flag = os.path.join(OUTPUT_DIR, "jisu_is_placeholder.json")
    if not os.path.exists(flag):
        return
    real_found = any(
        os.path.exists(os.path.join(OUTPUT_DIR, f"stock_{sid}", "garch_eval_results.csv"))
        for sid in range(1, 200)
    )
    if real_found:
        os.remove(flag)
        print("  Placeholder flag removed - real model outputs detected.")


clear_placeholder_if_real()


def print_banner(text):
    print(f"\n{'='*65}")
    print(f"  {text}")
    print(f"{'='*65}")


def qlike_safe(pred, actual):
    if pred <= 0 or not np.isfinite(pred):
        return np.nan
    pred = max(pred, 1e-10)
    return np.log(pred) + actual / pred


def run_stock(stock_id, time_id=None, save_plot=True):
    """
    Full pipeline for one stock.
    Returns a dict of results for use by the app.
    """
    selected = get_selected_stocks()
    liq_map = get_liquidity_map()

    if stock_id not in selected["all"]:
        print(f"  Stock {stock_id} not in selected 60 stocks.")
        return None

    regime = liq_map.get(stock_id, "mixed")
    liq_score = None
    median_bas = None
    try:
        liq_df = load_liquidity()
        if stock_id in liq_df.index:
            liq_score = liq_df.loc[stock_id].get("liquidity_score", None)
            median_bas = liq_df.loc[stock_id].get("median_BAS", None)
    except FileNotFoundError:
        pass

    print_banner(f"Stock {stock_id}  |  Regime: {regime.upper()}")
    if is_placeholder():
        print("  Warning: using placeholder data (Jisu's real models not run yet)")
    if liq_score is not None:
        print(f"  Liquidity score: {float(liq_score):.3f}  |  Median BAS: {float(median_bas):.6f}")

    # Load model outputs
    garch_eval = load_model_eval(stock_id, "garch_eval_results.csv")
    har_eval = load_model_eval(stock_id, "rosa_har_rv_eval_results.csv")
    egarchx_eval = load_model_eval(stock_id, "egarchx_eval_results.csv")
    blowups = load_model_eval(stock_id, "garch_blowups.csv")

    # Filter to specific time_id if requested
    if time_id is not None:
        for name, df_ in [("GARCH", garch_eval), ("HAR-RV", har_eval), ("EGARCH-X", egarchx_eval)]:
            if not df_.empty and "time_id" in df_.columns:
                filtered = df_[df_["time_id"] == time_id]
                if filtered.empty:
                    print(f"  time_id {time_id} not found in {name} results")

    # Default recommendation by regime
    REGIME_MODEL = {"liquid": "EGARCH-X", "illiquid": "HAR-RV", "mixed": "LightGBM"}
    regime_rec = REGIME_MODEL.get(regime, "HAR-RV")
    fallback = "HAR-RV" if regime in ("liquid", "mixed") else "GARCH"

    results = {}
    lgbm_row = load_lgbm_stock(stock_id)

    model_evals = {
        "GARCH":    garch_eval,
        "EGARCH-X": egarchx_eval,
        "HAR-RV":   har_eval,
    }

    for name, df_ in model_evals.items():
        if df_.empty:
            results[name] = {"available": False}
            continue
        pt = df_.groupby("time_id")[["QLIKE", "MSE"]].mean().dropna()
        results[name] = {
            "available":    True,
            "median_QLIKE": pt["QLIKE"].median(),
            "median_MSE":   pt["MSE"].median(),
            "n":            len(pt),
            "eval_df":      df_,
        }

    # Pick model with lowest median QLIKE
    qlike_scores = {}
    for name, res in results.items():
        if res.get("available") and np.isfinite(res.get("median_QLIKE", float("nan"))):
            qlike_scores[name] = res["median_QLIKE"]
    if lgbm_row and np.isfinite(lgbm_row.get("median_QLIKE", float("nan"))):
        qlike_scores["LightGBM"] = lgbm_row["median_QLIKE"]

    if qlike_scores:
        recommended = min(qlike_scores, key=qlike_scores.get)
        if recommended != regime_rec:
            print(f"  Regime default is {regime_rec}, but {recommended} "
                  f"has lower QLIKE ({qlike_scores[recommended]:.4f} vs "
                  f"{qlike_scores.get(regime_rec, float('nan')):.4f}) - recommending {recommended}")
    else:
        recommended = regime_rec

    recommended_eval = results.get(recommended, {}).get("eval_df", pd.DataFrame())
    recommended_available = not recommended_eval.empty if hasattr(recommended_eval, "empty") else False
    if recommended == "LightGBM" and lgbm_row:
        recommended_available = True

    print(f"\n  {'Model':<14} {'Status':<12} {'Med QLIKE':>10} {'Med MSE':>12} {'N'}")
    print(f"  {'-'*60}")

    for name, res in results.items():
        if not res.get("available"):
            status = "not run" if name != recommended else "missing"
            print(f"  {name:<14} {status:<12} {'':>10} {'':>12}")
            continue
        med_q = res["median_QLIKE"]
        med_mse = res["median_MSE"]
        n = res["n"]
        tag = " <- recommended" if name == recommended else ""
        print(f"  {name:<14} {'ready':<12} {med_q:>10.4f} {med_mse:>12.2e} {n:>4d}{tag}")

    if lgbm_row:
        lgbm_q = lgbm_row.get("median_QLIKE", float("nan"))
        lgbm_mse = lgbm_row.get("median_MSE",   float("nan"))
        lgbm_n = int(lgbm_row.get("n_sessions", 0))
        lgbm_tag = " <- recommended" if recommended == "LightGBM" else "  <- ML baseline"
        print(f"  {'LightGBM':<14} {'ready':<12} {lgbm_q:>10.4f} {lgbm_mse:>12.2e} {lgbm_n:>4d}{lgbm_tag}")
    else:
        print(f"  {'LightGBM':<14} {'not run':<12} {'':>10} {'':>12}")

    if not blowups.empty:
        n_blowups = len(blowups)
        n_total = len(garch_eval) if not garch_eval.empty else 1
        print(f"\n  GARCH blowups: {n_blowups} ({n_blowups/n_total:.1%}) - "
              f"{'expected for illiquid stock' if regime == 'illiquid' else 'low for liquid stock'}")

    print(f"\n  Recommendation:")
    if recommended_available:
        print(f"    Use {recommended} - regime is {regime.upper()}")
        if not blowups.empty and regime == "illiquid":
            print(f"    GARCH has {len(blowups)} blowups on this stock")
            print(f"    HAR-RV avoids this - no spread term to diverge")
        if regime == "mixed":
            print(f"    Mixed liquidity -> LightGBM (stable ML fallback)")
    else:
        print(f"    {recommended} not available yet -> fallback to {fallback}")

    if save_plot:
        _plot_comparison(stock_id, regime, results, blowups, recommended)

    return {
        "stock_id": stock_id,
        "regime": regime,
        "liquidity_score": float(liq_score) if liq_score is not None else None,
        "median_BAS": float(median_bas) if median_bas is not None else None,
        "recommended_model": recommended,
        "recommended_available": recommended_available,
        "blowup_count": len(blowups),
        "models": results,
        "lgbm": lgbm_row,
        "is_placeholder": is_placeholder(),
    }


def _plot_comparison(stock_id, regime, results, blowups, recommended):
    C_GARCH = "#378ADD"
    C_EX = "#1D9E75"
    C_HAR = "#D85A30"
    C_SPINE = "#D3D1C7"
    model_colors = {"GARCH": C_GARCH, "EGARCH-X": C_EX, "HAR-RV": C_HAR}

    available = {k: v for k, v in results.items() if v.get("available")}
    if not available:
        return

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Stock {stock_id} - {regime.upper()} regime, Recommended: {recommended}",
        fontsize=13, fontweight="500", y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

    def _style(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor("white")
        for sp in ax.spines.values(): sp.set_color(C_SPINE); sp.set_linewidth(0.6)
        ax.tick_params(labelsize=8.5, color=C_SPINE)
        ax.set_title(title, fontsize=10, fontweight="500", pad=6)
        if xlabel: ax.set_xlabel(xlabel, fontsize=8.5, color="#5F5E5A")
        if ylabel: ax.set_ylabel(ylabel, fontsize=8.5, color="#5F5E5A")
        ax.grid(axis="y", color=C_SPINE, linewidth=0.4, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)

    # Top-left: QLIKE boxplot - all models
    ax1 = fig.add_subplot(gs[0, :2])
    data_arrays, labels, cols = [], [], []
    for name, res in available.items():
        pt = res["eval_df"].groupby("time_id")["QLIKE"].mean().dropna()
        data_arrays.append(pt.values)
        labels.append(name)
        cols.append(model_colors.get(name, C_SPINE))

    if data_arrays:
        bp = ax1.boxplot(data_arrays, patch_artist=True, labels=labels, widths=0.45,
                         medianprops=dict(color="white", linewidth=2),
                         whiskerprops=dict(color=C_SPINE, linewidth=0.8),
                         capprops=dict(color=C_SPINE, linewidth=0.8),
                         flierprops=dict(marker="o", markersize=3,
                                         markerfacecolor=C_SPINE, alpha=0.4))
        for patch, col in zip(bp["boxes"], cols):
            patch.set_facecolor(col); patch.set_alpha(0.75)
            patch.set_linewidth(1.5 if labels[bp["boxes"].index(patch)] == recommended else 0.5)
        for i, (vals, label) in enumerate(zip(data_arrays, labels)):
            med = np.nanmedian(vals)
            ax1.annotate(f"{med:.3f}", xy=(i+1, med), xytext=(6, 0),
                         textcoords="offset points", fontsize=8, va="center")
        _style(ax1, title="QLIKE comparison - all models  (lower is better)",
               ylabel="QLIKE per time_id")

    # Top-right: model info card
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    lines = [
        f"Stock {stock_id}",
        f"Regime: {regime.upper()}",
        "",
        "Recommended:",
        f"  -> {recommended}",
        "",
        "Baseline:",
        "  -> GARCH(1,1)",
    ]
    if not blowups.empty:
        lines += ["", f"GARCH blowups: {len(blowups)}"]
    bg = "#E1F5EE" if regime == "liquid" else "#FAECE7"
    ax2.text(0.08, 0.92, "\n".join(lines), transform=ax2.transAxes,
             fontsize=9.5, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor=bg, alpha=0.85,
                       edgecolor=C_SPINE, linewidth=0.6))

    # Bottom row: pred vs actual per model
    for col_idx, (name, res) in enumerate(list(available.items())[:3]):
        ax = fig.add_subplot(gs[1, col_idx])
        df_ = res["eval_df"]
        col = model_colors.get(name, C_SPINE)
        if "actual_RV" not in df_.columns:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="#888780", fontsize=9)
            _style(ax, title=name); continue
        pa = df_.groupby("time_id").agg(
            pred_RV=("pred_RV", "mean"), actual_RV=("actual_RV", "mean")
        ).dropna()
        # Clip extreme outliers for readability
        p99 = np.percentile(np.concatenate([pa["actual_RV"], pa["pred_RV"]]), 99)
        pa = pa[(pa["actual_RV"] <= p99) & (pa["pred_RV"] <= p99)]
        ax.scatter(pa["actual_RV"], pa["pred_RV"],
                   alpha=0.4, s=10, color=col, edgecolors="none")
        lim = max(pa["actual_RV"].max(), pa["pred_RV"].max()) * 1.06
        ax.plot([0, lim], [0, lim], "--", color="#888780", linewidth=0.8)
        tag = "  <- recommended" if name == recommended else ""
        _style(ax,
               title=f"{name}{tag}",
               xlabel="Actual RV", ylabel="Predicted RV")

    plt.savefig(os.path.join(PIPELINE_OUT, f"pipeline_stock_{stock_id}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: pipeline/pipeline_stock_{stock_id}.png")


def run_all():
    selected = get_selected_stocks()
    liq_map = get_liquidity_map()
    all_rows = []

    garch_summary = _safe_load_csv("all_stocks_garch_summary.csv")
    egarchx_summary = _safe_load_csv("all_stocks_egarchx_summary.csv")
    har_summary = _safe_load_csv("all_stocks_har_rv_summary.csv")
    lgbm_summary = _safe_load_csv("lgbm_outputs/lgbm_per_stock.csv")

    print_banner("Cross-Stock Pipeline Summary")
    if is_placeholder():
        print("  Warning: using placeholder data")

    print(f"\n  {'Stock':>6}  {'Regime':>9}  {'GARCH QLIKE':>12}  {'Rec. QLIKE':>12}  {'LGBM QLIKE':>12}  {'Recommended'}")
    print(f"  {'-'*80}")

    for stock_id in selected["all"]:
        regime = liq_map.get(stock_id, "mixed")

        garch_q = np.nan
        if not garch_summary.empty and "stock_id" in garch_summary.columns:
            row = garch_summary[garch_summary["stock_id"] == stock_id]
            if not row.empty:
                garch_q = row["median_QLIKE"].values[0]

        REGIME_MODEL = {"liquid": "EGARCH-X", "illiquid": "HAR-RV", "mixed": "LightGBM"}
        regime_default = REGIME_MODEL.get(regime, "HAR-RV")
        rec_q = np.nan

        if regime == "liquid" and not egarchx_summary.empty:
            row = egarchx_summary[egarchx_summary["stock_id"] == stock_id]
            if not row.empty:
                rec_q = row["median_QLIKE"].values[0]
        elif regime == "illiquid" and not har_summary.empty:
            row = har_summary[har_summary["stock_id"] == stock_id]
            if not row.empty:
                rec_q = row["median_QLIKE"].values[0]
        elif regime == "mixed" and not lgbm_summary.empty:
            row = lgbm_summary[lgbm_summary["stock_id"] == stock_id]
            if not row.empty:
                rec_q = row["median_QLIKE"].values[0]

        lgbm_q = np.nan
        if not lgbm_summary.empty and "stock_id" in lgbm_summary.columns:
            lrow = lgbm_summary[lgbm_summary["stock_id"] == stock_id]
            if not lrow.empty:
                lgbm_q = lrow["median_QLIKE"].values[0]

        # Determine recommended model by lowest available QLIKE
        candidates = {}
        if np.isfinite(garch_q): candidates["GARCH"] = garch_q
        if np.isfinite(rec_q): candidates[regime_default] = rec_q
        if np.isfinite(lgbm_q): candidates["LightGBM"] = lgbm_q
        actual_rec = min(candidates, key=candidates.get) if candidates else regime_default

        garch_str = f"{garch_q:.4f}" if np.isfinite(garch_q) else "-"
        rec_str = f"{rec_q:.4f}"     if np.isfinite(rec_q)   else "-"
        lgbm_str = f"{lgbm_q:.4f}"  if np.isfinite(lgbm_q)  else "-"
        # Flag when actual winner differs from regime theory
        flag = "  *" if actual_rec != regime_default and candidates else ""
        print(f"  {stock_id:>6}  {regime:>9}  {garch_str:>12}  {rec_str:>12}  {lgbm_str:>12}  {actual_rec}{flag}")

        all_rows.append({
            "stock_id": stock_id,
            "regime": regime,
            "regime_model": regime_default,
            "actual_rec_model": actual_rec,
            "garch_QLIKE": garch_q,
            "rec_model_QLIKE": rec_q,
            "lgbm_QLIKE": lgbm_q,
        })

    summary = pd.DataFrame(all_rows)
    summary.to_csv(os.path.join(PIPELINE_OUT, "pipeline_all_stocks.csv"), index=False)
    print(f"\n  Saved: pipeline/pipeline_all_stocks.csv")

    liq_rows = summary[summary["regime"] == "liquid"]
    illiq_rows = summary[summary["regime"] == "illiquid"]
    mixed_rows = summary[summary["regime"] == "mixed"]

    def _med(df_, col):
        return df_[col].median() if (col in df_.columns and not df_.empty) else float("nan")
    print(f"\n  Liquid stocks : GARCH={_med(liq_rows,'garch_QLIKE'):.4f} | "
          f"EGARCH-X={_med(liq_rows,'rec_model_QLIKE'):.4f} | "
          f"LGBM={_med(liq_rows,'lgbm_QLIKE'):.4f}")
    print(f"  Illiquid stocks : GARCH={_med(illiq_rows,'garch_QLIKE'):.4f} | "
          f"HAR-RV={_med(illiq_rows,'rec_model_QLIKE'):.4f} | "
          f"LGBM={_med(illiq_rows,'lgbm_QLIKE'):.4f}")
    print(f"  Mixed stocks : LightGBM={_med(mixed_rows,'rec_model_QLIKE'):.4f} | "
          f"LGBM col={_med(mixed_rows,'lgbm_QLIKE'):.4f}")

    C_GARCH = "#378ADD"; C_EX = "#1D9E75"; C_HAR = "#D85A30"; C_LGBM = "#8B5CF6"; SPINE = "#D3D1C7"
    has_data = summary["garch_QLIKE"].notna().any()

    if has_data:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.patch.set_facecolor("white")
        fig.suptitle("Cross-Stock Model Comparison - GARCH vs Recommended Model",
                     fontsize=12, fontweight="500")

        regime_panels = [
            (axes[0], "Liquid stocks", liq_rows,   C_EX,   "EGARCH-X"),
            (axes[1], "Illiquid stocks", illiq_rows, C_HAR,  "HAR-RV"),
            (axes[2], "Mixed stocks", mixed_rows, C_LGBM, "LightGBM"),
        ]
        for ax, regime_label, df_, rec_col, rec_name in regime_panels:
            ax.set_facecolor("white")
            if df_.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="#888780")
                ax.set_title(regime_label, fontsize=10, fontweight="500")
                continue
            stocks = df_["stock_id"].astype(str).values
            x = np.arange(len(stocks))
            w3 = 0.26
            garch_q_c = df_["garch_QLIKE"].values
            rec_q_c = df_["rec_model_QLIKE"].values
            lgbm_q_col = df_["lgbm_QLIKE"].values if "lgbm_QLIKE" in df_.columns else np.full(len(x), np.nan)
            ax.bar(x - w3, garch_q_c, width=w3, label="GARCH(1,1)", color=C_GARCH, alpha=0.75)
            ax.bar(x, rec_q_c,   width=w3, label=rec_name,     color=rec_col, alpha=0.75)
            if not np.all(np.isnan(lgbm_q_col)):
                ax.bar(x + w3, lgbm_q_col, width=w3, label="LightGBM", color=C_LGBM, alpha=0.75)

            for sp in ax.spines.values(): sp.set_color(SPINE); sp.set_linewidth(0.6)
            ax.set_xticks(x); ax.set_xticklabels(stocks, rotation=45, ha="right", fontsize=7.5)
            ax.tick_params(labelsize=8, color=SPINE)
            ax.set_title(regime_label, fontsize=10, fontweight="500")
            ax.set_ylabel("Median QLIKE (lower is better)", fontsize=9)
            ax.legend(fontsize=8, framealpha=0.9)
            ax.grid(axis="y", color=SPINE, linewidth=0.4, linestyle="--")
            ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(os.path.join(PIPELINE_OUT, "pipeline_01_model_comparison.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("\n  Saved: pipeline/pipeline_01_model_comparison.png")

    # Win rate: how often recommended model beats GARCH
    summary["rec_wins"] = summary["rec_model_QLIKE"] < summary["garch_QLIKE"]
    win_by_regime = summary.groupby("regime")["rec_wins"].agg(["sum", "count"])
    win_by_regime["win_rate"] = win_by_regime["sum"] / win_by_regime["count"]

    n_regimes = len(win_by_regime)
    fig, axes = plt.subplots(1, max(n_regimes, 1), figsize=(5 * max(n_regimes, 1), 4))
    if n_regimes == 1: axes = [axes]
    fig.patch.set_facecolor("white")
    fig.suptitle("How often does the recommended model beat GARCH?",
                 fontsize=11, fontweight="500")

    regime_colors = {"liquid": C_EX, "illiquid": C_HAR, "mixed": C_LGBM}
    for ax, (regime_label, row) in zip(axes, win_by_regime.iterrows()):
        ax.set_facecolor("white")
        win_pct = row["win_rate"] * 100
        loss_pct = 100 - win_pct
        col = regime_colors.get(regime_label, SPINE)
        ax.pie(
            [win_pct, loss_pct],
            labels=["Recommended wins", "GARCH wins"],
            colors=[col, C_GARCH],
            autopct="%1.0f%%",
            startangle=90,
            wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.5),
            textprops=dict(fontsize=9),
        )
        ax.set_title(f"{regime_label.capitalize()} stocks\n"
                     f"(n={int(row['count'])})",
                     fontsize=10, fontweight="500")

    plt.tight_layout()
    plt.savefig(os.path.join(PIPELINE_OUT, "pipeline_02_win_rate.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: pipeline/pipeline_02_win_rate.png")


def _safe_load_csv(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="M4 Regime-Based Model Router - DATA3888 Group 8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipline.py --all
  python pipline.py --stock_id 42
  python pipline.py --stock_id 42 --time_id 15
  python pipline.py --stock_id 42 --no_plot
        """
    )
    parser.add_argument("--all", action="store_true",
                        help="Run summary across all 60 selected stocks")
    parser.add_argument("--stock_id", type=int, default=None,
                        help="Run pipeline for a single stock_id")
    parser.add_argument("--time_id", type=int, default=None,
                        help="Filter to a specific time_id within a stock")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip saving plots (faster for batch runs)")
    args = parser.parse_args()

    if args.all:
        run_all()
    elif args.stock_id is not None:
        run_stock(args.stock_id, time_id=args.time_id, save_plot=not args.no_plot)
    else:
        parser.print_help()
        print("\n  Please specify --all or --stock_id <id>")
        import sys; sys.exit(1)
