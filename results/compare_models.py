#!/usr/bin/env python3
"""
Compare CFM and XGBoost validation results side by side.

Reads results.json from each model's output directory and produces:
  1. Bar chart: aggregate metrics by category (both_rain, era5_only, prism_only)
  2. Scatter: per-day RMSE comparison (CFM vs XGBoost)
  3. Box plots: metric distributions by category
  4. Coverage calibration plot
  5. Spread/Skill vs CRPS scatter
  6. Summary table (printed + saved as CSV)

Usage:
    python -u compare_models.py --cfm-results sequential_plots_cfm_ode_v3_no_flow/results.json --xgb-results sequential_plots_xgb/results.json --output-dir results/comparison_v3_no_flow_and_xgb_aef
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

CATEGORIES = ["both_rain", "era5_only", "prism_only", "all"]
CAT_LABELS = {"both_rain": "Both Rain", "era5_only": "ERA5 Only",
              "prism_only": "PRISM Only", "all": "All Days"}
CAT_COLORS = {"both_rain": "#2196F3", "era5_only": "#FF9800",
              "prism_only": "#4CAF50", "all": "#9C27B0"}

METRICS = ["rmse_mean", "rmse_bicubic", "crps", "spread_skill",
           "coverage_50", "coverage_80", "coverage_90"]
METRIC_LABELS = {
    "rmse_mean": "RMSE (Ens Mean)",
    "rmse_bicubic": "RMSE (Bicubic)",
    "crps": "CRPS",
    "spread_skill": "Spread/Skill",
    "coverage_50": "50% Coverage",
    "coverage_80": "80% Coverage",
    "coverage_90": "90% Coverage",
}
METRIC_UNITS = {
    "rmse_mean": "mm/day", "rmse_bicubic": "mm/day", "crps": "mm/day",
    "spread_skill": "ratio", "coverage_50": "fraction",
    "coverage_80": "fraction", "coverage_90": "fraction",
}


def load_results(path):
    with open(path) as f:
        return json.load(f)


def get_per_day_values(results, metric, category="all"):
    days = results.get("per_day", [])
    vals = []
    for d in days:
        if not d.get("metrics_available", True):
            continue
        if category != "all" and d.get("category") != category:
            continue
        if metric in d:
            vals.append(d[metric])
    return vals


# ── Plot 1: Aggregate bar chart ──────────────────────────────────────────────

def plot_aggregate_bars(cfm, xgb, out_dir):
    """Side-by-side bars for each metric, grouped by category."""
    plot_metrics = ["rmse_mean", "crps", "spread_skill"]

    fig, axes = plt.subplots(1, len(plot_metrics), figsize=(5 * len(plot_metrics), 5),
                             constrained_layout=True)

    cats_to_plot = ["both_rain", "era5_only", "prism_only", "all"]
    x = np.arange(len(cats_to_plot))
    width = 0.35

    for ax, metric in zip(axes, plot_metrics):
        cfm_vals = []
        xgb_vals = []
        cfm_errs = []
        xgb_errs = []

        for cat in cats_to_plot:
            cfm_agg = cfm.get("aggregate", {}).get(cat, {})
            xgb_agg = xgb.get("aggregate", {}).get(cat, {})

            cfm_m = cfm_agg.get(metric, {})
            xgb_m = xgb_agg.get(metric, {})

            cfm_vals.append(cfm_m.get("mean", 0))
            xgb_vals.append(xgb_m.get("mean", 0))
            cfm_errs.append(cfm_m.get("std", 0))
            xgb_errs.append(xgb_m.get("std", 0))

        bars1 = ax.bar(x - width/2, cfm_vals, width, yerr=cfm_errs,
                        label="CFM", color="#1976D2", alpha=0.85, capsize=3)
        bars2 = ax.bar(x + width/2, xgb_vals, width, yerr=xgb_errs,
                        label="XGBoost", color="#F57C00", alpha=0.85, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels([CAT_LABELS[c] for c in cats_to_plot], fontsize=8, rotation=15)
        ax.set_ylabel(f"{METRIC_LABELS[metric]} ({METRIC_UNITS[metric]})")
        ax.set_title(METRIC_LABELS[metric], fontweight="bold")
        ax.legend(fontsize=8)

        if metric == "spread_skill":
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_ylim(bottom=0)

    fig.suptitle("CFM vs XGBoost — Aggregate Metrics by Rain Category",
                 fontsize=14, fontweight="bold")
    fig.savefig(out_dir / "aggregate_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved aggregate_bars.png")


# ── Plot 2: Per-day RMSE scatter ─────────────────────────────────────────────

def plot_rmse_scatter(cfm, xgb, out_dir):
    """Scatter plot: CFM RMSE vs XGBoost RMSE for matched dates."""
    cfm_days = {d["date"]: d for d in cfm.get("per_day", [])
                if d.get("metrics_available")}
    xgb_days = {d["date"]: d for d in xgb.get("per_day", [])
                if d.get("metrics_available")}

    common_dates = sorted(set(cfm_days) & set(xgb_days))

    if not common_dates:
        log.warning("No common dates for scatter plot")
        return

    fig, ax = plt.subplots(figsize=(7, 7))

    for cat in ["both_rain", "era5_only", "prism_only"]:
        cfm_rmse = []
        xgb_rmse = []
        for d in common_dates:
            if cfm_days[d].get("category") == cat:
                cfm_rmse.append(cfm_days[d]["rmse_mean"])
                xgb_rmse.append(xgb_days[d]["rmse_mean"])

        if cfm_rmse:
            ax.scatter(xgb_rmse, cfm_rmse, c=CAT_COLORS[cat],
                       label=CAT_LABELS[cat], s=50, alpha=0.7, edgecolors="white",
                       linewidths=0.5)

    # 1:1 line
    all_vals = ([cfm_days[d]["rmse_mean"] for d in common_dates] +
                [xgb_days[d]["rmse_mean"] for d in common_dates])
    if all_vals:
        lo, hi = 0, max(all_vals) * 1.1
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5, label="1:1 line")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    ax.set_xlabel("XGBoost RMSE (mm/day)", fontsize=11)
    ax.set_ylabel("CFM RMSE (mm/day)", fontsize=11)
    ax.set_title("Per-Day RMSE: CFM vs XGBoost", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")

    # Annotate: points below the line = CFM is better
    ax.text(0.05, 0.95, "← CFM better", transform=ax.transAxes,
            fontsize=9, color="gray", va="top")
    ax.text(0.95, 0.05, "XGBoost better →", transform=ax.transAxes,
            fontsize=9, color="gray", ha="right")

    fig.savefig(out_dir / "rmse_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved rmse_scatter.png")


# ── Plot 3: Box plots by category ────────────────────────────────────────────

def plot_boxplots(cfm, xgb, out_dir):
    """Box plots comparing metric distributions."""
    plot_metrics = ["rmse_mean", "crps", "spread_skill"]

    fig, axes = plt.subplots(1, len(plot_metrics),
                             figsize=(5 * len(plot_metrics), 5),
                             constrained_layout=True)

    for ax, metric in zip(axes, plot_metrics):
        data = []
        labels = []
        colors = []

        for cat in ["both_rain", "prism_only", "all"]:
            cfm_vals = get_per_day_values(cfm, metric, cat)
            xgb_vals = get_per_day_values(xgb, metric, cat)

            if cfm_vals:
                data.append(cfm_vals)
                labels.append(f"CFM\n{CAT_LABELS[cat]}")
                colors.append("#1976D2")
            if xgb_vals:
                data.append(xgb_vals)
                labels.append(f"XGB\n{CAT_LABELS[cat]}")
                colors.append("#F57C00")

        if not data:
            continue

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel(f"{METRIC_LABELS[metric]} ({METRIC_UNITS[metric]})")
        ax.set_title(METRIC_LABELS[metric], fontweight="bold")
        ax.tick_params(axis="x", labelsize=7, rotation=20)

        if metric == "spread_skill":
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)

    fig.suptitle("Metric Distributions — CFM vs XGBoost",
                 fontsize=14, fontweight="bold")
    fig.savefig(out_dir / "boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved boxplots.png")


# ── Plot 4: Coverage calibration ─────────────────────────────────────────────

def plot_coverage_calibration(cfm, xgb, out_dir):
    """Coverage calibration: actual vs nominal for each model."""
    fig, ax = plt.subplots(figsize=(6, 6))

    nominal = [0.5, 0.8, 0.9]
    cov_metrics = ["coverage_50", "coverage_80", "coverage_90"]

    for model, results, color, marker, label in [
        ("CFM", cfm, "#1976D2", "o", "CFM"),
        ("XGB", xgb, "#F57C00", "s", "XGBoost"),
    ]:
        for cat in ["both_rain", "all"]:
            actual = []
            for m in cov_metrics:
                vals = get_per_day_values(results, m, cat)
                actual.append(np.mean(vals) if vals else 0)

            linestyle = "-" if cat == "all" else "--"
            cat_label = f"{label} ({CAT_LABELS[cat]})"
            ax.plot(nominal, actual, f"{marker}{linestyle}", color=color,
                    label=cat_label, markersize=8, linewidth=1.5)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4, label="Perfect calibration")
    ax.set_xlabel("Nominal Coverage", fontsize=11)
    ax.set_ylabel("Actual Coverage", fontsize=11)
    ax.set_title("Ensemble Coverage Calibration", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0, 1.05)
    ax.set_aspect("equal")

    fig.savefig(out_dir / "coverage_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved coverage_calibration.png")


# ── Plot 5: Spread/Skill vs CRPS ────────────────────────────────────────────

def plot_spread_skill_vs_crps(cfm, xgb, out_dir):
    """Scatter: spread/skill ratio vs CRPS for each day, colored by model."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for model, results, color, marker, label in [
        ("CFM", cfm, "#1976D2", "o", "CFM"),
        ("XGB", xgb, "#F57C00", "s", "XGBoost"),
    ]:
        ss = get_per_day_values(results, "spread_skill", "all")
        crps = get_per_day_values(results, "crps", "all")
        if ss and crps and len(ss) == len(crps):
            ax.scatter(ss, crps, c=color, marker=marker, label=label,
                       s=50, alpha=0.6, edgecolors="white", linewidths=0.5)

    ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5,
               label="Perfect spread calibration")
    ax.set_xlabel("Spread / Skill Ratio", fontsize=11)
    ax.set_ylabel("CRPS (mm/day)", fontsize=11)
    ax.set_title("Spread/Skill vs CRPS", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)

    fig.savefig(out_dir / "spread_skill_vs_crps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved spread_skill_vs_crps.png")


# ── Plot 6: Per-day time series ──────────────────────────────────────────────

def plot_daily_timeseries(cfm, xgb, out_dir):
    """RMSE and CRPS over time for both models."""
    cfm_days = {d["date"]: d for d in cfm.get("per_day", []) if d.get("metrics_available")}
    xgb_days = {d["date"]: d for d in xgb.get("per_day", []) if d.get("metrics_available")}
    common = sorted(set(cfm_days) & set(xgb_days))

    if len(common) < 3:
        log.warning("Not enough common dates for time series")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True, sharex=True)

    dates_idx = np.arange(len(common))
    date_labels = common

    for ax, metric, title in [
        (axes[0], "rmse_mean", "Ensemble Mean RMSE (mm/day)"),
        (axes[1], "crps", "CRPS (mm/day)"),
    ]:
        cfm_vals = [cfm_days[d][metric] for d in common]
        xgb_vals = [xgb_days[d][metric] for d in common]

        ax.plot(dates_idx, cfm_vals, "o-", color="#1976D2", label="CFM",
                markersize=5, linewidth=1.2)
        ax.plot(dates_idx, xgb_vals, "s-", color="#F57C00", label="XGBoost",
                markersize=5, linewidth=1.2)

        # Color background by category
        for i, d in enumerate(common):
            cat = cfm_days[d].get("category", "")
            if cat in CAT_COLORS:
                ax.axvspan(i - 0.4, i + 0.4, alpha=0.08, color=CAT_COLORS[cat])

        ax.set_ylabel(title, fontsize=10)
        ax.legend(fontsize=9)
        ax.set_yscale("log")

    axes[1].set_xticks(dates_idx)
    axes[1].set_xticklabels(date_labels, rotation=45, ha="right", fontsize=7)
    axes[1].set_xlabel("Date")

    fig.suptitle("Daily Metrics — CFM vs XGBoost", fontsize=13, fontweight="bold")
    fig.savefig(out_dir / "daily_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved daily_timeseries.png")


# ── Summary table ────────────────────────────────────────────────────────────

def print_and_save_table(cfm, xgb, out_dir):
    """Print summary table and save as CSV."""
    header = f"{'Metric':<25} | {'Category':<12} | {'CFM':>20} | {'XGBoost':>20} | {'Δ (CFM-XGB)':>15}"
    sep = "-" * len(header)

    lines = [header, sep]
    csv_lines = ["metric,category,cfm_mean,cfm_std,xgb_mean,xgb_std,delta"]

    for metric in ["rmse_mean", "crps", "spread_skill", "coverage_50", "coverage_80", "coverage_90"]:
        for cat in ["both_rain", "prism_only", "all"]:
            cfm_agg = cfm.get("aggregate", {}).get(cat, {}).get(metric, {})
            xgb_agg = xgb.get("aggregate", {}).get(cat, {}).get(metric, {})

            cfm_mean = cfm_agg.get("mean", float("nan"))
            cfm_std = cfm_agg.get("std", float("nan"))
            xgb_mean = xgb_agg.get("mean", float("nan"))
            xgb_std = xgb_agg.get("std", float("nan"))

            delta = cfm_mean - xgb_mean

            cfm_str = f"{cfm_mean:.4f} ± {cfm_std:.4f}" if not np.isnan(cfm_mean) else "N/A"
            xgb_str = f"{xgb_mean:.4f} ± {xgb_std:.4f}" if not np.isnan(xgb_mean) else "N/A"
            delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "N/A"

            lines.append(
                f"{METRIC_LABELS.get(metric, metric):<25} | {CAT_LABELS[cat]:<12} | "
                f"{cfm_str:>20} | {xgb_str:>20} | {delta_str:>15}"
            )
            csv_lines.append(
                f"{metric},{cat},{cfm_mean},{cfm_std},{xgb_mean},{xgb_std},{delta}"
            )

        lines.append(sep)

    table_str = "\n".join(lines)
    log.info("Summary Table:\n" + table_str)

    with open(out_dir / "comparison_table.txt", "w") as f:
        f.write(table_str)

    with open(out_dir / "comparison_table.csv", "w") as f:
        f.write("\n".join(csv_lines))

    log.info("Saved comparison_table.txt and comparison_table.csv")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare CFM and XGBoost validation results."
    )
    parser.add_argument("--cfm-results", required=True,
                        help="Path to CFM results.json")
    parser.add_argument("--xgb-results", required=True,
                        help="Path to XGBoost results.json")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for comparison plots and tables")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading CFM results: {args.cfm_results}")
    cfm = load_results(args.cfm_results)

    log.info(f"Loading XGBoost results: {args.xgb_results}")
    xgb = load_results(args.xgb_results)

    # Quick summary
    cfm_n = sum(1 for d in cfm.get("per_day", []) if d.get("metrics_available"))
    xgb_n = sum(1 for d in xgb.get("per_day", []) if d.get("metrics_available"))
    log.info(f"CFM: {cfm_n} days with metrics")
    log.info(f"XGBoost: {xgb_n} days with metrics")

    # Generate all plots
    plot_aggregate_bars(cfm, xgb, out_dir)
    plot_rmse_scatter(cfm, xgb, out_dir)
    plot_boxplots(cfm, xgb, out_dir)
    plot_coverage_calibration(cfm, xgb, out_dir)
    plot_spread_skill_vs_crps(cfm, xgb, out_dir)
    plot_daily_timeseries(cfm, xgb, out_dir)
    print_and_save_table(cfm, xgb, out_dir)

    log.info(f"All comparison outputs saved to {out_dir}")


if __name__ == "__main__":
    main()