"""
visualization.py
----------------
All matplotlib (Agg backend) figure generators.
Every function saves to outputs/figures/ and returns the output path.
Never calls plt.show() — non-interactive headless rendering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Must come before pyplot import
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FIGURE_DPI = 120
POLLUTANT_COLORS = {
    "CO": "#e41a1c",
    "NO2": "#377eb8",
    "O3": "#4daf4a",
    "PM2.5": "#984ea3",
    "SO2": "#ff7f00",
}


def _ensure_figures_dir(output_dir: str | Path) -> Path:
    p = Path(output_dir) / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# 1. Missingness heatmap
# ---------------------------------------------------------------------------

def plot_missingness_heatmap(
    availability_df: pd.DataFrame,
    output_dir: str | Path,
) -> Optional[Path]:
    """
    Heatmap of % available data: rows = stations, columns = pollutants.
    Tries missingno first; falls back to seaborn/matplotlib.
    """
    fig_dir = _ensure_figures_dir(output_dir)
    out_path = fig_dir / "missingness_heatmap.png"

    fig, ax = plt.subplots(figsize=(10, 5))
    data = availability_df.values.astype(float)
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(availability_df.columns)))
    ax.set_xticklabels(availability_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(availability_df.index)))
    ax.set_yticklabels([f"Site {s}" for s in availability_df.index])
    plt.colorbar(im, ax=ax, label="% Available")
    ax.set_title("Station × Pollutant Data Availability (%)", fontsize=13, pad=12)

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            color = "black" if 20 < v < 80 else "white"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color=color)

    plt.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved missingness heatmap -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 2. Raw time-series with anomaly flags
# ---------------------------------------------------------------------------

def plot_raw_timeseries_with_anomalies(
    df: pd.DataFrame,
    col: str,
    station_id: int,
    output_dir: str | Path,
) -> Optional[Path]:
    """
    Plot the raw time series and overlay consensus anomaly flags.
    """
    if col not in df.columns:
        return None

    fig_dir = _ensure_figures_dir(output_dir)
    out_path = fig_dir / f"site{station_id}_{col}_anomalies.png"

    fig, ax = plt.subplots(figsize=(14, 4))
    color = POLLUTANT_COLORS.get(col, "#333333")

    ax.plot(df.index, df[col], color=color, linewidth=0.6, alpha=0.8, label=col)

    consensus_col = f"anomaly_consensus_flag_{col}"
    if consensus_col in df.columns:
        anom = df[df[consensus_col] == True]
        ax.scatter(anom.index, anom[col], color="red", s=10, zorder=5,
                   label=f"Consensus anomaly ({len(anom)})", alpha=0.7)

    ax.set_title(f"Site {station_id} – {col} Raw Time Series with Anomaly Flags",
                 fontsize=11)
    ax.set_ylabel(col)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30, ha="right")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved anomaly time series -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 3. STL decomposition
# ---------------------------------------------------------------------------

def plot_stl_decomposition(
    df: pd.DataFrame,
    col: str,
    station_id: int,
    period: int,
    output_dir: str | Path,
) -> Optional[Path]:
    """
    Run STL and plot the four components: observed, trend, seasonal, residual.
    """
    if col not in df.columns:
        return None

    series = df[col].dropna()
    if len(series) < period * 2:
        logger.info("Not enough data for STL plot: %s site %s", col, station_id)
        return None

    fig_dir = _ensure_figures_dir(output_dir)
    out_path = fig_dir / f"site{station_id}_{col}_stl.png"

    try:
        from statsmodels.tsa.seasonal import STL
        filled = series.interpolate(method="time", limit_direction="both")
        result = STL(filled, period=period, robust=True).fit()

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        comps = [
            ("Observed", filled),
            ("Trend", pd.Series(result.trend, index=filled.index)),
            ("Seasonal", pd.Series(result.seasonal, index=filled.index)),
            ("Residual", pd.Series(result.resid, index=filled.index)),
        ]
        color = POLLUTANT_COLORS.get(col, "#333333")
        for ax, (label, comp) in zip(axes, comps):
            ax.plot(comp.index, comp.values, linewidth=0.6, color=color)
            ax.set_ylabel(label, fontsize=9)
            ax.grid(True, alpha=0.3)

        axes[0].set_title(f"Site {station_id} – {col} STL Decomposition (period={period}h)",
                          fontsize=11)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved STL decomposition -> %s", out_path)
        return out_path
    except Exception as exc:
        logger.warning("STL plot failed for site %s %s: %s", station_id, col, exc)
        plt.close("all")
        return None


# ---------------------------------------------------------------------------
# 4. Before / after imputation
# ---------------------------------------------------------------------------

def plot_before_after_imputation(
    df: pd.DataFrame,
    col: str,
    station_id: int,
    output_dir: str | Path,
) -> Optional[Path]:
    """
    Overlay original (blue) and imputed (orange) series; mark imputed points.
    """
    orig_col = f"{col}_original"
    imp_col = f"{col}_imputed"
    method_col = f"{col}_imputation_method"

    if orig_col not in df.columns or imp_col not in df.columns:
        return None

    fig_dir = _ensure_figures_dir(output_dir)
    out_path = fig_dir / f"site{station_id}_{col}_imputation.png"

    fig, ax = plt.subplots(figsize=(14, 4))
    color = POLLUTANT_COLORS.get(col, "#333333")

    ax.plot(df.index, df[orig_col], color=color, linewidth=0.6, alpha=0.9, label="Original")

    if method_col in df.columns:
        for method, mcolor, label in [
            ("short_interp", "#2196F3", "Short interp"),
            ("medium_ridge", "#FF9800", "Medium Ridge"),
            ("long_seasonal", "#9C27B0", "Long seasonal"),
        ]:
            mask = df[method_col] == method
            if mask.any():
                ax.scatter(df.index[mask], df[imp_col][mask], color=mcolor,
                           s=8, zorder=5, label=f"{label} ({mask.sum()})", alpha=0.8)
    else:
        # Just show imputed vs original
        imp_mask = df[orig_col].isna() & df[imp_col].notna()
        ax.scatter(df.index[imp_mask], df[imp_col][imp_mask],
                   color="orange", s=8, zorder=5, label=f"Imputed ({imp_mask.sum()})")

    ax.set_title(f"Site {station_id} – {col} Before/After Imputation", fontsize=11)
    ax.set_ylabel(col)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30, ha="right")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved before/after imputation -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 5. Validation comparison
# ---------------------------------------------------------------------------

def plot_validation_comparison(
    metrics_df: pd.DataFrame,
    output_dir: str | Path,
) -> Optional[Path]:
    """
    Bar chart of mean RMSE by pollutant and mask type.
    """
    if metrics_df is None or metrics_df.empty:
        return None

    fig_dir = _ensure_figures_dir(output_dir)
    out_path = fig_dir / "validation_rmse_comparison.png"

    pivot = metrics_df.pivot_table(index="col", columns="mask", values="mean_RMSE")

    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", ax=ax, colormap="Set2")
    ax.set_title("Imputation Validation – Mean RMSE by Pollutant and Mask Type", fontsize=11)
    ax.set_xlabel("Pollutant")
    ax.set_ylabel("RMSE")
    plt.xticks(rotation=30, ha="right")
    ax.legend(title="Mask scenario", fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved validation RMSE comparison -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 6. Negative value summary
# ---------------------------------------------------------------------------

def plot_negative_value_summary(
    neg_summary: pd.DataFrame,
    output_dir: str | Path,
) -> Optional[Path]:
    """
    Horizontal bar chart showing % negative values per station × pollutant.
    """
    if neg_summary is None or neg_summary.empty:
        return None

    fig_dir = _ensure_figures_dir(output_dir)
    out_path = fig_dir / "negative_value_summary.png"

    plot_data = neg_summary[neg_summary["pct_negative"] > 0].copy()
    if plot_data.empty:
        return None

    plot_data["label"] = "Site " + plot_data["station_id"].astype(str) + " " + plot_data["pollutant"]
    plot_data = plot_data.sort_values("pct_negative", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_data) * 0.4)))
    colors = ["#d32f2f" if pct > 10 else "#ff7043" for pct in plot_data["pct_negative"]]
    ax.barh(plot_data["label"], plot_data["pct_negative"], color=colors)
    ax.set_xlabel("% Negative Values")
    ax.set_title("Physically Impossible Negative Pollutant Values", fontsize=11)
    ax.axvline(10, color="black", linestyle="--", linewidth=0.8, label="10% threshold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved negative value summary -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 7. Anomaly method comparison (vote count distribution)
# ---------------------------------------------------------------------------

def plot_anomaly_method_comparison(
    df: pd.DataFrame,
    col: str,
    station_id: int,
    output_dir: str | Path,
) -> Optional[Path]:
    """
    Bar chart showing how many anomalies each method flagged for one pollutant.
    """
    methods = ["iqr", "hampel", "rolling_zscore", "stl", "isolation_forest"]
    flag_cols = {m: f"flag_{m}_{col}" for m in methods}
    present = {m: c for m, c in flag_cols.items() if c in df.columns}

    if not present:
        return None

    fig_dir = _ensure_figures_dir(output_dir)
    out_path = fig_dir / f"site{station_id}_{col}_method_comparison.png"

    counts = {m: int(df[c].sum()) for m, c in present.items()}
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(list(counts.keys()), list(counts.values()), color="#607D8B")
    ax.set_title(f"Site {station_id} – {col}: Anomaly Count by Detection Method", fontsize=11)
    ax.set_ylabel("# Flagged Points")
    ax.bar_label(bars, padding=2)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path

