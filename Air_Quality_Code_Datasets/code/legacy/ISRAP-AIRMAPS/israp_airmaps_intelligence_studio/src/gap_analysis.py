"""Detect, classify, and summarise missing-value gaps."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import GAP_CATEGORIES


def classify_gap_length(n: int) -> str:
    for name, (lo, hi) in GAP_CATEGORIES.items():
        if lo <= n <= hi:
            return name
    return "very_long"


def find_gaps(
    df: pd.DataFrame,
    pollutant: str,
    datetime_col: str = "datetime",
) -> pd.DataFrame:
    """Return a DataFrame where each row is one contiguous NaN run.

    Columns: gap_id, gap_start, gap_end, gap_length, gap_category,
             previous_valid_value, next_valid_value
    """
    if pollutant not in df.columns:
        return pd.DataFrame()

    series = df[pollutant].values
    dts = df[datetime_col].values

    gaps: list[dict[str, Any]] = []
    gap_id = 0
    i = 0
    n = len(series)

    while i < n:
        if pd.isna(series[i]):
            start = i
            while i < n and pd.isna(series[i]):
                i += 1
            end = i - 1
            length = end - start + 1

            prev_val = np.nan
            for j in range(start - 1, -1, -1):
                if not pd.isna(series[j]):
                    prev_val = series[j]
                    break

            next_val = np.nan
            for j in range(end + 1, n):
                if not pd.isna(series[j]):
                    next_val = series[j]
                    break

            gaps.append({
                "gap_id": gap_id,
                "gap_start": pd.Timestamp(dts[start]),
                "gap_end": pd.Timestamp(dts[end]),
                "gap_length": length,
                "gap_category": classify_gap_length(length),
                "previous_valid_value": prev_val,
                "next_valid_value": next_val,
            })
            gap_id += 1
        else:
            i += 1

    return pd.DataFrame(gaps)


def gap_summary(
    station_data: dict[str, pd.DataFrame],
    datetime_col: str = "datetime",
) -> pd.DataFrame:
    """Aggregate gap statistics across all stations and pollutants."""
    rows = []
    for skey, df in station_data.items():
        for p in df.columns:
            if p in ("CO", "NO2", "O3", "PM2.5", "SO2"):
                if df[p].isna().all():
                    continue
                gaps = find_gaps(df, p, datetime_col)
                if gaps.empty:
                    continue
                for cat, (lo, hi) in GAP_CATEGORIES.items():
                    sub = gaps[gaps["gap_category"] == cat]
                    rows.append({
                        "station": skey,
                        "pollutant": p,
                        "gap_category": cat,
                        "count": len(sub),
                        "total_missing_hours": int(sub["gap_length"].sum()) if len(sub) > 0 else 0,
                        "longest_gap": int(sub["gap_length"].max()) if len(sub) > 0 else 0,
                    })
    return pd.DataFrame(rows)


def add_gap_flags(df: pd.DataFrame, pollutant: str) -> pd.DataFrame:
    """Add gap_id, gap_category columns for a given pollutant into *df*."""
    gap_id_col = f"{pollutant}_gap_id"
    gap_cat_col = f"{pollutant}_gap_category"
    df[gap_id_col] = np.nan
    df[gap_cat_col] = ""

    if pollutant not in df.columns:
        return df

    gaps = find_gaps(df, pollutant)
    if gaps.empty:
        return df

    for _, row in gaps.iterrows():
        mask = (df["datetime"] >= row["gap_start"]) & (df["datetime"] <= row["gap_end"])
        df.loc[mask, gap_id_col] = row["gap_id"]
        df.loc[mask, gap_cat_col] = row["gap_category"]

    return df

