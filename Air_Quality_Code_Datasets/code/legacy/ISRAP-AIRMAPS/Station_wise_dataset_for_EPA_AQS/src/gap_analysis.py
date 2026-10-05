"""
gap_analysis.py
---------------
Analyse temporal gaps (consecutive NaN runs) in pollutant time series.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def gap_run_lengths(series: pd.Series) -> pd.Series:
    """
    Return a Series of (start_idx, length) for each NaN run.
    Result index = start datetime of the gap; values = gap length in hours.
    """
    is_nan = series.isna()
    runs = {}
    in_gap = False
    start_pos = None

    for i, (ts, v) in enumerate(zip(series.index, is_nan)):
        if v and not in_gap:
            in_gap = True
            start_pos = ts
            run_count = 1
        elif v and in_gap:
            run_count += 1
        elif not v and in_gap:
            runs[start_pos] = run_count
            in_gap = False
    if in_gap:
        runs[start_pos] = run_count  # type: ignore[assignment]

    return pd.Series(runs, name="gap_length_hours")


def gap_summary_stats(series: pd.Series, col_name: str = "value") -> dict:
    """
    Compute summary statistics for gaps in a Series.

    Returns a dict with keys:
      n_gaps, total_missing, pct_missing, min_gap, max_gap,
      mean_gap, median_gap, n_short (≤2), n_medium (3-24), n_long (>24)
    """
    n = len(series)
    n_missing = series.isna().sum()
    pct_missing = n_missing / n * 100 if n > 0 else 0.0

    runs = gap_run_lengths(series)

    if len(runs) == 0:
        return {
            "column": col_name,
            "n_gaps": 0,
            "total_missing": int(n_missing),
            "pct_missing": round(pct_missing, 2),
            "min_gap_h": None,
            "max_gap_h": None,
            "mean_gap_h": None,
            "median_gap_h": None,
            "n_gaps_short_le2h": 0,
            "n_gaps_medium_3_24h": 0,
            "n_gaps_long_gt24h": 0,
        }

    return {
        "column": col_name,
        "n_gaps": int(len(runs)),
        "total_missing": int(n_missing),
        "pct_missing": round(pct_missing, 2),
        "min_gap_h": int(runs.min()),
        "max_gap_h": int(runs.max()),
        "mean_gap_h": round(float(runs.mean()), 1),
        "median_gap_h": round(float(runs.median()), 1),
        "n_gaps_short_le2h": int((runs <= 2).sum()),
        "n_gaps_medium_3_24h": int(((runs > 2) & (runs <= 24)).sum()),
        "n_gaps_long_gt24h": int((runs > 24).sum()),
    }


def analyse_all_gaps(
    station_df: pd.DataFrame,
    pollutants: List[str],
    station_id: int,
) -> List[dict]:
    """
    Run gap analysis for all pollutants in a station DataFrame.
    Returns a list of summary dicts (one per available pollutant).
    """
    results = []
    for pol in pollutants:
        if pol not in station_df.columns:
            continue
        stats = gap_summary_stats(station_df[pol], col_name=pol)
        stats["station_id"] = station_id
        results.append(stats)
    return results


def availability_matrix(
    all_stations: Dict[int, pd.DataFrame],
    pollutants: List[str],
) -> pd.DataFrame:
    """
    Build a station × pollutant availability matrix (% non-NaN).
    Rows = station IDs, Columns = pollutants.
    """
    rows = {}
    for site_id, df in all_stations.items():
        row = {}
        for pol in pollutants:
            if pol not in df.columns:
                row[pol] = 0.0
            else:
                row[pol] = round((1 - df[pol].isna().mean()) * 100, 1)
        rows[site_id] = row
    return pd.DataFrame.from_dict(rows, orient="index", columns=pollutants)

