"""Report generator — produces Markdown and HTML audit/analysis reports."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import APP_VERSION, POLLUTANTS, STATIONS


def _fmt(v: Any, decimals: int = 2) -> str:
    """Format a value for display."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _md_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Convert a DataFrame to a Markdown table string."""
    df = df.head(max_rows)
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join([header, sep] + rows)


def generate_markdown_report(
    station_data: dict[str, pd.DataFrame],
    audit_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    availability_df: pd.DataFrame,
    anomaly_summary: dict[str, Any] | None = None,
    imputation_summary: dict[str, Any] | None = None,
) -> str:
    """Generate a full Markdown audit/analysis report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# ISRAP AIRMAPS Intelligence Studio — Research Report",
        "",
        f"**Generated:** {now}  ",
        f"**App Version:** {APP_VERSION}  ",
        f"**Stations:** {', '.join(STATIONS.keys())}  ",
        "",
        "---",
        "",
        "## 1. Dataset Overview",
        "",
    ]

    # Network summary
    n_stations = len(station_data)
    total_rows = sum(len(df) for df in station_data.values())
    all_valid = sum(
        df[p].notna().sum()
        for df in station_data.values()
        for p in POLLUTANTS if p in df.columns
    )
    all_total = sum(
        len(df)
        for df in station_data.values()
        for p in POLLUTANTS if p in df.columns
    )
    completeness = 100 * all_valid / max(all_total, 1)
    negatives = sum(
        int((df[p] < 0).sum())
        for df in station_data.values()
        for p in POLLUTANTS if p in df.columns
    )

    lines += [
        f"| Metric | Value |",
        f"|---|---|",
        f"| Monitoring stations | {n_stations} |",
        f"| Total hourly records | {total_rows:,} |",
        f"| Overall completeness | {completeness:.1f}% |",
        f"| Negative concentration values | {negatives:,} |",
        "",
    ]

    # Per-station summary
    lines += ["### Station Coverage", ""]
    for skey, df in station_data.items():
        meta = STATIONS[skey]
        dt_min = df["datetime"].min() if "datetime" in df.columns else "?"
        dt_max = df["datetime"].max() if "datetime" in df.columns else "?"
        lines += [
            f"**{meta['label']}** (Site {meta['site']})",
            f"- Date range: {dt_min} to {dt_max}",
            f"- Rows: {len(df):,}",
            f"- Monitored: {', '.join(meta['monitored_pollutants'])}",
        ]
        if meta.get("special_case"):
            lines.append(f"- **NOTE:** {meta['special_case']}")
        lines.append("")

    # Availability matrix
    lines += ["## 2. Station–Pollutant Availability", ""]
    if not availability_df.empty:
        p_cols = [p for p in POLLUTANTS if p in availability_df.columns]
        show_cols = ["station", "label"] + p_cols
        show_cols = [c for c in show_cols if c in availability_df.columns]
        lines.append(_md_table(availability_df[show_cols]))
        lines.append("")

    # Audit table
    lines += ["## 3. Data Quality Audit", ""]
    if not audit_df.empty:
        show_cols = ["station", "pollutant", "valid_count", "missing_pct",
                     "negative_count", "structural_missing", "min", "max", "mean"]
        show_cols = [c for c in show_cols if c in audit_df.columns]
        lines.append(_md_table(audit_df[show_cols]))
        lines.append("")

        # Structural missing summary
        structural = audit_df[audit_df.get("structural_missing", False) == True]
        if not structural.empty:
            lines += [
                "### Structurally Missing Station-Pollutant Pairs",
                "",
                "The following station-pollutant pairs have <1% valid data and are treated as "
                "not monitored. They are excluded from anomaly detection and imputation.",
                "",
            ]
            for _, row in structural.iterrows():
                lines.append(f"- **{row.get('station', '?')}** / **{row.get('pollutant', '?')}**")
            lines.append("")

    # Gap summary
    lines += ["## 4. Missing-Data Gaps", ""]
    if not gap_df.empty:
        # gap_summary returns: station, pollutant, gap_category, count, total_missing_hours, longest_gap
        if "gap_category" in gap_df.columns and "count" in gap_df.columns:
            cat_counts = gap_df.groupby("gap_category").agg(
                gap_count=("count", "sum"),
                total_hours=("total_missing_hours", "sum"),
                longest=("longest_gap", "max"),
            ).reset_index()
        else:
            cat_counts = pd.DataFrame()

        if not cat_counts.empty:
            lines.append(_md_table(cat_counts))
            lines.append("")

        lines += [
            "**Gap categories:**",
            "- very_short: 1–2 consecutive missing hours",
            "- medium: 3–24 consecutive missing hours",
            "- long: 25–168 hours (1 week)",
            "- very_long: > 168 hours",
            "",
        ]

    # Station 3 SO2 warning
    lines += [
        "## 5. Station 3 SO₂ — Special Case",
        "",
        "> **WARNING:** Station 3 (Site 59) exhibits systematic SO₂ sensor corruption. "
        "Approximately 33% of observed SO₂ values are negative. This is physically impossible "
        "and indicative of a systematic sensor malfunction. Default recommendation: **exclude "
        "Station 3 SO₂ from all modeling and analysis.**",
        "",
        "Imputation cannot reliably reconstruct a channel when a large proportion of observed "
        "values is itself invalid. See the Station 3 SO₂ Investigation page for full diagnostics.",
        "",
    ]

    # Anomaly summary
    if anomaly_summary:
        lines += ["## 6. Anomaly Detection Summary", ""]
        for k, v in anomaly_summary.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    # Imputation summary
    if imputation_summary:
        lines += ["## 7. Imputation Summary", ""]
        for k, v in imputation_summary.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    # Scientific safeguards
    lines += [
        "## 8. Scientific Safeguards Applied",
        "",
        "1. **Original data preserved:** Raw CSV files are never modified. "
        "All transformations produce separate output columns.",
        "2. **Structural missingness:** 100%-missing station-pollutant channels are marked "
        "as 'Not monitored' and excluded from imputation.",
        "3. **Negative values:** Flagged separately before imputation. "
        "Original negative values are preserved in `original_value` column.",
        "4. **Chronological splits:** All ML model training uses past-only data to avoid "
        "look-ahead bias.",
        "5. **Station 3 SO₂:** Excluded by default due to systematic sensor corruption.",
        "6. **Reproducibility:** Random seeds stored in all experiment records.",
        "",
    ]

    lines += [
        "---",
        "",
        "*Generated by ISRAP AIRMAPS Intelligence Studio*",
        f"*{now}*",
    ]

    return "\n".join(lines)


def generate_html_report(
    station_data: dict[str, pd.DataFrame],
    audit_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    availability_df: pd.DataFrame,
    anomaly_summary: dict[str, Any] | None = None,
    imputation_summary: dict[str, Any] | None = None,
) -> str:
    """Generate a self-contained HTML report."""
    md_content = generate_markdown_report(
        station_data, audit_df, gap_df, availability_df,
        anomaly_summary, imputation_summary,
    )

    # Convert basic Markdown to HTML
    html_lines = []
    for line in md_content.splitlines():
        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("> "):
            html_lines.append(f"<blockquote>{line[2:]}</blockquote>")
        elif line.startswith("- "):
            html_lines.append(f"<li>{line[2:]}</li>")
        elif line.startswith("|"):
            html_lines.append(f"<tr><td>{line.replace('|', '</td><td>')}</td></tr>")
        elif line.strip() == "---":
            html_lines.append("<hr>")
        elif line.strip() == "":
            html_lines.append("<br>")
        else:
            html_lines.append(f"<p>{line}</p>")

    body = "\n".join(html_lines)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ISRAP AIRMAPS Intelligence Studio — Research Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1100px; margin: 40px auto;
          padding: 0 24px; color: #1a1a2e; line-height: 1.6; }}
  h1 {{ color: #0d1b2a; border-bottom: 3px solid #0a9396; padding-bottom: 8px; }}
  h2 {{ color: #0a9396; margin-top: 32px; }}
  h3 {{ color: #4895ef; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  td, th {{ border: 1px solid #dee2e6; padding: 8px 12px; }}
  tr:first-child td {{ background: #0d1b2a; color: #fff; font-weight: 600; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  blockquote {{ border-left: 4px solid #e63946; background: #fdecea;
               padding: 12px 16px; margin: 12px 0; border-radius: 0 8px 8px 0; }}
  li {{ margin: 4px 0; }}
  hr {{ border: none; border-top: 1px solid #dee2e6; margin: 24px 0; }}
  p {{ margin: 8px 0; }}
</style>
</head>
<body>
{body}
</body>
</html>"""

