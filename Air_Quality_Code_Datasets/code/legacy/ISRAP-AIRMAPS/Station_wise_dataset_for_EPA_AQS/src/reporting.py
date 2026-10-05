"""
reporting.py
------------
Generate CSV audits, availability matrices, gap summaries, anomaly summaries,
imputation summaries, and a final Markdown analysis report.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 1. Dataset audit
# ---------------------------------------------------------------------------

def generate_data_audit_csv(
    all_stations_info: Dict[int, dict],
    output_dir: str | Path,
) -> Path:
    """
    Write outputs/audits/dataset_audit.csv with one row per station.
    """
    audit_dir = _ensure(Path(output_dir) / "audits")
    out_path = audit_dir / "dataset_audit.csv"

    rows = []
    for site_id, info in all_stations_info.items():
        df: pd.DataFrame = info.get("df", pd.DataFrame())
        row: dict = {
            "site_id": site_id,
            "station_file": info.get("station_file", ""),
            "n_rows_raw": info.get("n_rows_raw", len(df)),
            "n_rows_reindexed": len(df),
            "date_start": df.index.min().date() if len(df) > 0 else None,
            "date_end": df.index.max().date() if len(df) > 0 else None,
            "available_pollutants": ",".join(info.get("available_pollutants", [])),
            "structural_missing": ",".join(info.get("structural_missing", [])),
            "n_duplicates_removed": info.get("n_duplicates_removed", 0),
            "n_hours_added_by_reindex": info.get("n_hours_added_by_reindex", 0),
        }
        rows.append(row)

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(out_path, index=False)
    logger.info("Dataset audit -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 2. Availability matrix
# ---------------------------------------------------------------------------

def generate_availability_matrix(
    all_stations: Dict[int, pd.DataFrame],
    pollutants: List[str],
    output_dir: str | Path,
) -> Path:
    """
    Write outputs/audits/station_pollutant_availability.csv
    Values = % non-NaN per station × pollutant.
    """
    from .gap_analysis import availability_matrix
    audit_dir = _ensure(Path(output_dir) / "audits")
    out_path = audit_dir / "station_pollutant_availability.csv"

    avail = availability_matrix(all_stations, pollutants)
    avail.index.name = "site_id"
    avail.to_csv(out_path)
    logger.info("Availability matrix -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 3. Negative value summary
# ---------------------------------------------------------------------------

def generate_negative_summary(
    all_stations: Dict[int, pd.DataFrame],
    pollutants: List[str],
    output_dir: str | Path,
) -> pd.DataFrame:
    """
    Write outputs/audits/negative_value_summary.csv and return the DataFrame.
    """
    audit_dir = _ensure(Path(output_dir) / "audits")
    out_path = audit_dir / "negative_value_summary.csv"

    rows = []
    for site_id, df in all_stations.items():
        for pol in pollutants:
            flag_col = f"physical_invalid_flag_{pol}"
            if flag_col in df.columns:
                # Use the flag column which captured negatives BEFORE they were set to NaN
                flag_series = df[flag_col].astype(bool)
                n_neg = int(flag_series.sum())
                # n_observed = valid readings (including the ones that were negative)
                n_total = int(flag_series.notna().sum())
                # More meaningful: count of non-structural-missing rows
                pol_col_present = pol in df.columns
                if pol_col_present:
                    n_total = int((df[pol].notna() | flag_series).sum())
            elif pol in df.columns:
                series = pd.to_numeric(df[pol], errors="coerce")
                n_total = int(series.notna().sum())
                n_neg = int((series < 0).sum())
            else:
                continue

            rows.append({
                "station_id": site_id,
                "pollutant": pol,
                "n_observed": n_total,
                "n_negative": n_neg,
                "pct_negative": round(n_neg / n_total * 100, 2) if n_total > 0 else 0.0,
            })

    neg_df = pd.DataFrame(rows)
    neg_df.to_csv(out_path, index=False)
    logger.info("Negative value summary -> %s", out_path)
    return neg_df


# ---------------------------------------------------------------------------
# 4. Gap summary
# ---------------------------------------------------------------------------

def generate_gap_summary(
    all_stations: Dict[int, pd.DataFrame],
    pollutants: List[str],
    output_dir: str | Path,
) -> Path:
    """
    Write outputs/audits/missing_gap_summary.csv
    """
    from .gap_analysis import gap_summary_stats
    audit_dir = _ensure(Path(output_dir) / "audits")
    out_path = audit_dir / "missing_gap_summary.csv"

    rows = []
    for site_id, df in all_stations.items():
        for pol in pollutants:
            if pol not in df.columns:
                continue
            stats = gap_summary_stats(df[pol], col_name=pol)
            stats["station_id"] = site_id
            rows.append(stats)

    gap_df = pd.DataFrame(rows)
    gap_df.to_csv(out_path, index=False)
    logger.info("Gap summary -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 5. Anomaly summary
# ---------------------------------------------------------------------------

def generate_anomaly_summary(
    all_anomaly_results: Dict[int, Dict[str, pd.DataFrame]],
    output_dir: str | Path,
) -> Path:
    """
    Write outputs/anomaly_flags/anomaly_summary.csv
    all_anomaly_results: site_id → pollutant → df (with flag columns)
    """
    anom_dir = _ensure(Path(output_dir) / "anomaly_flags")
    out_path = anom_dir / "anomaly_summary.csv"

    rows = []
    methods = ["iqr", "hampel", "rolling_zscore", "stl", "isolation_forest"]

    for site_id, pol_results in all_anomaly_results.items():
        for pol, df in pol_results.items():
            row = {"site_id": site_id, "pollutant": pol, "n_total": len(df)}
            row["n_valid"] = int(df[pol].notna().sum()) if pol in df.columns else 0
            for m in methods:
                flag_col = f"flag_{m}_{pol}"
                row[f"n_flag_{m}"] = int(df[flag_col].sum()) if flag_col in df.columns else 0

            consensus_col = f"anomaly_consensus_flag_{pol}"
            row["n_consensus"] = int(df[consensus_col].sum()) if consensus_col in df.columns else 0
            rows.append(row)

    anom_df = pd.DataFrame(rows)
    anom_df.to_csv(out_path, index=False)
    logger.info("Anomaly summary -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 6. Imputation summary
# ---------------------------------------------------------------------------

def generate_imputation_summary(
    all_imputation_results: Dict[int, Dict[str, pd.DataFrame]],
    output_dir: str | Path,
) -> Path:
    """
    Write outputs/imputed/imputation_summary.csv
    """
    imp_dir = _ensure(Path(output_dir) / "imputed")
    out_path = imp_dir / "imputation_summary.csv"

    rows = []
    for site_id, pol_results in all_imputation_results.items():
        for pol, df in pol_results.items():
            method_col = f"{pol}_imputation_method"
            if method_col not in df.columns:
                continue
            counts = df[method_col].value_counts().to_dict()
            row = {"site_id": site_id, "pollutant": pol}
            row["n_observed"] = counts.get("observed", 0)
            row["n_short_interp"] = counts.get("short_interp", 0)
            row["n_medium_ridge"] = counts.get("medium_ridge", 0)
            row["n_long_seasonal"] = counts.get("long_seasonal", 0)
            row["n_still_missing"] = counts.get("missing", 0)
            row["total_imputed"] = (
                row["n_short_interp"] + row["n_medium_ridge"] + row["n_long_seasonal"]
            )
            rows.append(row)

    imp_df = pd.DataFrame(rows)
    imp_df.to_csv(out_path, index=False)
    logger.info("Imputation summary -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 7. Markdown report
# ---------------------------------------------------------------------------

def generate_markdown_report(
    findings: dict,
    output_dir: str | Path,
) -> Path:
    """
    Write outputs/reports/analysis_report.md
    *findings* is a free-form dict of section → content produced by the pipeline.
    """
    rep_dir = _ensure(Path(output_dir) / "reports")
    out_path = rep_dir / "analysis_report.md"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# ISRAP / AIRMAPS – Air Quality Pipeline Analysis Report",
        f"**Generated:** {now}  \n",
        "---",
        "",
        "## 1. Dataset Overview",
        "",
    ]

    audit = findings.get("audit", {})
    lines.append(f"- **Stations processed:** {audit.get('n_stations', 'N/A')}")
    lines.append(f"- **ERA5 rows loaded:** {audit.get('era5_rows', 'N/A')}")
    lines.append("")

    # Negative values
    lines += [
        "## 2. Physically Impossible Negative Values",
        "",
        "> All pollutant concentrations must be ≥ 0 by physical law.",
        "> Negative readings indicate instrument zero-drift or calibration failure.",
        "",
    ]
    neg_rows = findings.get("negative_values", [])
    if neg_rows:
        lines.append("| Station | Pollutant | n_negative | % Negative |")
        lines.append("|---------|-----------|-----------|------------|")
        for r in neg_rows:
            if r.get("pct_negative", 0) > 0:
                lines.append(
                    f"| Site {r['station_id']} | {r['pollutant']} "
                    f"| {r['n_negative']} | {r['pct_negative']:.1f}% |"
                )
        lines.append("")
    lines.append("**CRITICAL:** Station 3 (Site 59) SO2 has ~33.6% negative values "
                 "– systematic zero-offset drift detected. "
                 "Recommend excluding SO2 at this station from analysis.")
    lines.append("")

    # Anomaly detection
    lines += [
        "## 3. Anomaly Detection",
        "",
        "Five-method tiered pipeline (IQR → Hampel → Rolling z-score → STL → Isolation Forest).",
        "",
    ]
    anom = findings.get("anomaly_summary", pd.DataFrame())
    if isinstance(anom, pd.DataFrame) and not anom.empty:
        lines.append("| Station | Pollutant | IQR | Hampel | RollZ | STL | IsoForest | Consensus |")
        lines.append("|---------|-----------|-----|--------|-------|-----|-----------|-----------|")
        for _, row in anom.iterrows():
            lines.append(
                f"| Site {row.get('site_id','')} | {row.get('pollutant','')} "
                f"| {row.get('n_flag_iqr',0)} | {row.get('n_flag_hampel',0)} "
                f"| {row.get('n_flag_rolling_zscore',0)} | {row.get('n_flag_stl',0)} "
                f"| {row.get('n_flag_isolation_forest',0)} | {row.get('n_consensus',0)} |"
            )
    lines.append("")

    # Imputation
    lines += [
        "## 4. Imputation",
        "",
        "Three-tier strategy: short gaps (≤2 h) → linear interpolation; "
        "medium gaps (3–24 h) → Ridge regression with ERA5 covariates; "
        "long gaps (>24 h) → seasonal median (month × hour).",
        "",
    ]
    imp = findings.get("imputation_summary", pd.DataFrame())
    if isinstance(imp, pd.DataFrame) and not imp.empty:
        lines.append("| Station | Pollutant | Short | Medium | Long | Still Missing |")
        lines.append("|---------|-----------|-------|--------|------|---------------|")
        for _, row in imp.iterrows():
            lines.append(
                f"| Site {row.get('site_id','')} | {row.get('pollutant','')} "
                f"| {row.get('n_short_interp',0)} | {row.get('n_medium_ridge',0)} "
                f"| {row.get('n_long_seasonal',0)} | {row.get('n_still_missing',0)} |"
            )
    lines.append("")

    # Validation
    lines += [
        "## 5. Validation Results",
        "",
        "Artificial masking experiments (5%, 10%, 20% random + block patterns).",
        "",
    ]
    val = findings.get("validation_leaderboard", pd.DataFrame())
    if isinstance(val, pd.DataFrame) and not val.empty:
        lines.append("| Pollutant | Mask | Mean RMSE | Mean R² |")
        lines.append("|-----------|------|-----------|---------|")
        for _, row in val.head(20).iterrows():
            lines.append(
                f"| {row.get('col','')} | {row.get('mask','')} "
                f"| {row.get('mean_RMSE', 'N/A'):.4f} "
                f"| {row.get('mean_R2', 'N/A'):.4f} |"
            )
    lines.append("")

    # Station 3 SO2 investigation
    lines += [
        "## 6. Station 3 SO2 Investigation",
        "",
        "Per ISRAP_AIRMAPS_Research_Analysis.docx:",
        "- **33.6% of SO2 values at Site 59 are negative** (16,416 of ~48,870 recorded values).",
        "- This indicates systematic zero-offset drift or a sign error in the EPA AQS export.",
        "- Action taken: All negative SO2 values at Site 59 set to NaN via physical constraint cleaning.",
        "- Recommendation: Cross-check EPA AQS Data Mart QC flags (CC/NF codes).",
        "  Consider excluding SO2 at Station 3 entirely from comparative analysis.",
        "",
    ]

    lines += [
        "## 7. Methodology References",
        "",
        "- Liu et al. (ICDM 2008) – Isolation Forest",
        "- van Buuren & Groothuis-Oudshoorn (JSS 2011) – MICE",
        "- Khayati et al. (PVLDB 2020) – Mind the Gap (imputation benchmark)",
        "- Schmidl et al. (PVLDB 2022) – Anomaly Detection Comprehensive Evaluation",
        "- Zhang et al. (PVLDB 2017) – Time Series Data Cleaning",
        "",
        "---",
        "*Report generated by ISRAP Air Quality Pipeline*",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Markdown report -> %s", out_path)
    return out_path

