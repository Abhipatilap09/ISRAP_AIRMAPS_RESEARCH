"""
preprocessing.py
----------------
Standardise datetime, remove duplicates, reindex to complete hourly series,
apply physical-constraint cleaning, and detect structural missingness.
"""

from __future__ import annotations

import logging
from typing import List, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datetime standardisation
# ---------------------------------------------------------------------------

def standardize_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame has a proper DatetimeIndex (UTC-naive) sorted ascending.
    Safe to call if index is already datetime.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df = df.copy()
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.dropna(subset=["datetime"])
            df = df.set_index("datetime")
        else:
            logger.warning("standardize_datetime: no datetime index or column found")
            return df

    df = df.sort_index()
    # Strip timezone if present (EPA data is local time, treat as naive)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "datetime"
    return df


# ---------------------------------------------------------------------------
# Duplicate removal
# ---------------------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate timestamps.
    For each duplicated timestamp keep the row with the most non-NaN values.
    """
    if not df.index.duplicated().any():
        return df

    n_before = len(df)
    # Sort by number of valid (non-NaN) values per row, descending
    df = df.copy()
    df["_valid_count"] = df.notna().sum(axis=1)
    df = df.sort_values("_valid_count", ascending=False)
    df = df[~df.index.duplicated(keep="first")]
    df = df.drop(columns=["_valid_count"])
    df = df.sort_index()
    logger.info("Removed %d duplicate timestamps", n_before - len(df))
    return df


# ---------------------------------------------------------------------------
# Hourly reindexing
# ---------------------------------------------------------------------------

def reindex_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the DataFrame to a complete hourly DatetimeIndex.
    Missing hours become rows of NaN (except metadata like site/county).
    """
    if len(df) == 0:
        logger.warning("reindex_hourly: empty DataFrame")
        return df

    start = df.index.min().floor("h")
    end = df.index.max().ceil("h")
    full_idx = pd.date_range(start=start, end=end, freq="h", name="datetime")

    # Forward-fill metadata columns so they don't become NaN
    meta_cols = [c for c in ["site", "state_code", "county_code"] if c in df.columns]

    df_reindexed = df.reindex(full_idx)

    for col in meta_cols:
        if col in df_reindexed.columns:
            df_reindexed[col] = df_reindexed[col].ffill().bfill()

    n_added = len(full_idx) - len(df)
    if n_added > 0:
        logger.info("reindex_hourly: added %d missing hourly rows", n_added)
    return df_reindexed


# ---------------------------------------------------------------------------
# Physical constraint cleaning
# ---------------------------------------------------------------------------

def physical_constraint_cleaning(
    df: pd.DataFrame,
    pollutants: List[str],
) -> pd.DataFrame:
    """
    Set negative pollutant concentrations to NaN and add binary flag columns.

    Adds column ``physical_invalid_flag_<pollutant>`` (bool) for each pollutant.
    Only operates on columns that are actually present in df.
    """
    df = df.copy()
    for pol in pollutants:
        if pol not in df.columns:
            continue
        mask = df[pol] < 0
        n_neg = mask.sum()
        if n_neg > 0:
            pct = n_neg / df[pol].notna().sum() * 100 if df[pol].notna().sum() > 0 else 0
            logger.warning(
                "Physical constraint: %d negative %s values (%.1f%%) -> set to NaN",
                n_neg, pol, pct,
            )
        df[f"physical_invalid_flag_{pol}"] = mask.astype(bool)
        df.loc[mask, pol] = np.nan
    return df


# ---------------------------------------------------------------------------
# Gap length computation
# ---------------------------------------------------------------------------

def compute_gap_lengths(series: pd.Series) -> pd.Series:
    """
    For each element in *series*, return the length of the consecutive NaN run
    it belongs to. Non-NaN elements get value 0.

    Example: [1, NaN, NaN, 2, NaN] → [0, 2, 2, 0, 1]
    """
    is_nan = series.isna()
    gap_lengths = pd.Series(0, index=series.index, dtype=int)

    start = None
    for i, (idx, val) in enumerate(zip(series.index, is_nan)):
        if val:
            if start is None:
                start = i
        else:
            if start is not None:
                run_len = i - start
                gap_lengths.iloc[start:i] = run_len
                start = None
    # Handle trailing run
    if start is not None:
        run_len = len(series) - start
        gap_lengths.iloc[start:] = run_len

    return gap_lengths


# ---------------------------------------------------------------------------
# Structural missingness detection
# ---------------------------------------------------------------------------

def identify_structural_missingness(
    df: pd.DataFrame,
    pollutants: List[str],
    threshold: float = 0.95,
) -> Set[str]:
    """
    A pollutant is considered *structurally missing* (no monitor installed) if
    the fraction of NaN values exceeds *threshold*.

    Returns the set of structurally missing pollutant names.
    """
    structural = set()
    n = len(df)
    if n == 0:
        return set(pollutants)

    for pol in pollutants:
        if pol not in df.columns:
            structural.add(pol)
            continue
        nan_frac = df[pol].isna().sum() / n
        if nan_frac >= threshold:
            logger.info(
                "Pollutant '%s' is structurally missing (%.1f%% NaN >= threshold %.0f%%)",
                pol, nan_frac * 100, threshold * 100,
            )
            structural.add(pol)
    return structural


