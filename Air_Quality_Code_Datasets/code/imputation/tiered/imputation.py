"""
imputation.py
-------------
Tiered imputation pipeline for EPA AQS pollutant time series.

Tier 1 – Short gaps (≤ 2 h): linear interpolation
Tier 2 – Medium gaps (3–24 h): Ridge regression with ERA5 + cyclical time features
Tier 3 – Long gaps (> 24 h): seasonal median (month × hour) baseline

All imputed values are clipped to ≥ 0 (physically meaningful minimum).
"""

from __future__ import annotations

import logging
from typing import Optional, Set

import numpy as np
import pandas as pd

from .utils import hour_sin_cos, dayofyear_sin_cos

logger = logging.getLogger(__name__)

ERA5_IMPUTATION_FEATURES = [
    "temp_c", "wind_speed", "blh", "relative_humidity",
    "surface_pressure_hpa", "precip_mm", "u10", "v10",
]


# ---------------------------------------------------------------------------
# Tier 1: Short-gap linear interpolation
# ---------------------------------------------------------------------------

def impute_short_gaps(series: pd.Series, max_gap: int = 2) -> pd.Series:
    """
    Linear interpolation for gaps of *max_gap* hours or fewer.
    Longer gaps remain NaN.
    """
    return series.interpolate(method="time", limit=max_gap, limit_direction="forward")


# ---------------------------------------------------------------------------
# Tier 2: Medium-gap Ridge regression
# ---------------------------------------------------------------------------

def _build_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build cyclical and calendar features from a DatetimeIndex."""
    records = []
    for ts in index:
        h_sin, h_cos = hour_sin_cos(ts.hour)
        d_sin, d_cos = dayofyear_sin_cos(ts.dayofyear)
        records.append({
            "hour_sin": h_sin,
            "hour_cos": h_cos,
            "doy_sin": d_sin,
            "doy_cos": d_cos,
            "month": ts.month,
        })
    return pd.DataFrame(records, index=index)


def impute_medium_gaps(
    station_df: pd.DataFrame,
    col: str,
    era5_df: pd.DataFrame,
    max_gap: int = 24,
    random_seed: int = 42,
) -> pd.Series:
    """
    Ridge regression imputation for gaps of 3–*max_gap* hours.

    Features: ERA5 met variables + cyclical hour/doy encodings.
    The model is trained ONLY on observed (non-NaN) rows in station_df.
    Predictions are clipped to ≥ 0.
    """
    series = station_df[col].copy() if col in station_df.columns else pd.Series(dtype=float)
    if series.empty:
        return series

    # Identify medium-gap positions
    from .preprocessing import compute_gap_lengths
    gap_lens = compute_gap_lengths(series)
    medium_mask = series.isna() & (gap_lens > 0) & (gap_lens <= max_gap)

    if medium_mask.sum() == 0:
        return series  # nothing to impute

    # Build feature matrix
    time_feats = _build_time_features(station_df.index)
    era5_avail = [f for f in ERA5_IMPUTATION_FEATURES if f in era5_df.columns]
    if era5_avail:
        # Deduplicate ERA5 index before reindexing (keep first occurrence)
        era5_clean = era5_df[era5_avail].copy()
        era5_clean = era5_clean[~era5_clean.index.duplicated(keep="first")]
        era5_sub = era5_clean.reindex(station_df.index)
        for c in era5_avail:
            era5_sub[c] = era5_sub[c].ffill().bfill()
        feature_df = pd.concat([time_feats, era5_sub], axis=1)
    else:
        feature_df = time_feats

    X = feature_df.copy()
    # Fill any remaining NaN in features
    X = X.fillna(X.median())

    target = series

    train_mask = target.notna()
    if train_mask.sum() < 10:
        logger.warning("Medium gap imputation for %s: too few training rows (%d)", col, train_mask.sum())
        return series

    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        y_train = target[train_mask].values

        model = Ridge(alpha=1.0, random_state=random_seed)
        model.fit(X_train, y_train)

        X_pred = scaler.transform(X[medium_mask])
        y_pred = model.predict(X_pred)
        y_pred = np.clip(y_pred, 0, None)

        series_out = series.copy()
        series_out[medium_mask] = y_pred
        logger.debug("Medium-gap imputed %d points for %s", medium_mask.sum(), col)
        return series_out
    except ImportError:
        logger.warning("scikit-learn not available; falling back to linear interpolation for medium gaps")
        return series.interpolate(method="time", limit=max_gap, limit_direction="forward")
    except Exception as exc:
        logger.warning("Medium-gap imputation failed for %s: %s", col, exc)
        return series


# ---------------------------------------------------------------------------
# Tier 3: Long-gap seasonal median
# ---------------------------------------------------------------------------

def impute_long_gaps(
    station_df: pd.DataFrame,
    col: str,
    era5_df: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Seasonal median (grouped by month × hour-of-day) as a baseline for long gaps.
    Clipped to ≥ 0.
    """
    series = station_df[col].copy() if col in station_df.columns else pd.Series(dtype=float)
    if series.empty:
        return series

    from .preprocessing import compute_gap_lengths
    gap_lens = compute_gap_lengths(series)
    long_mask = series.isna() & (gap_lens > 24)

    if long_mask.sum() == 0:
        return series

    # Build seasonal median lookup
    temp_df = pd.DataFrame({"value": series}, index=station_df.index)
    temp_df["month"] = temp_df.index.month
    temp_df["hour"] = temp_df.index.hour
    seasonal_median = temp_df.groupby(["month", "hour"])["value"].median()

    series_out = series.copy()
    for idx in series.index[long_mask]:
        key = (idx.month, idx.hour)
        if key in seasonal_median.index:
            val = seasonal_median[key]
            if not np.isnan(val):
                series_out[idx] = max(0.0, float(val))

    n_filled = long_mask.sum() - series_out[long_mask].isna().sum()
    logger.debug("Long-gap seasonal-median imputed %d of %d points for %s",
                 int(n_filled), int(long_mask.sum()), col)
    return series_out


# ---------------------------------------------------------------------------
# Imputation record builder
# ---------------------------------------------------------------------------

def create_imputation_record(
    df: pd.DataFrame,
    col: str,
    series_after_short: pd.Series,
    series_after_medium: pd.Series,
    series_after_long: pd.Series,
) -> pd.DataFrame:
    """
    Add tracking columns to df:
      <col>_original          – raw values before any imputation
      <col>_clean             – after physical constraint removal (same as original here)
      <col>_imputed           – final imputed values
      <col>_imputation_method – 'observed' | 'short_interp' | 'medium_ridge' | 'long_seasonal' | 'missing'
    """
    df = df.copy()
    original = df[col].copy() if col in df.columns else pd.Series(np.nan, index=df.index)

    df[f"{col}_original"] = original
    df[f"{col}_clean"] = original  # post physical-constraint (set in preprocessing)
    df[f"{col}_imputed"] = series_after_long.clip(lower=0)

    method = pd.Series("observed", index=df.index)
    method[original.isna() & series_after_short.notna()] = "short_interp"
    method[original.isna() & series_after_short.isna() & series_after_medium.notna()] = "medium_ridge"
    method[
        original.isna()
        & series_after_short.isna()
        & series_after_medium.isna()
        & series_after_long.notna()
    ] = "long_seasonal"
    method[series_after_long.isna()] = "missing"

    df[f"{col}_imputation_method"] = method
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_imputation_pipeline(
    station_df: pd.DataFrame,
    col: str,
    era5_df: pd.DataFrame,
    structural_missing: Set[str],
    config: dict,
) -> pd.DataFrame:
    """
    Run the three-tier imputation pipeline for one pollutant column.

    Returns station_df with added _original / _imputed / _imputation_method columns.
    """
    if col in structural_missing:
        logger.info("Skipping imputation for structurally missing column: %s", col)
        return station_df

    if col not in station_df.columns or station_df[col].notna().sum() == 0:
        logger.info("Skipping imputation for %s (no valid data)", col)
        return station_df

    short_max = config.get("short_gap_max", 2)
    medium_max = config.get("medium_gap_max", 24)
    seed = config.get("random_seed", 42)

    # Tier 1
    s1 = impute_short_gaps(station_df[col], max_gap=short_max)

    # Tier 2 – work on partially filled series
    tmp_df = station_df.copy()
    tmp_df[col] = s1
    s2 = impute_medium_gaps(tmp_df, col, era5_df, max_gap=medium_max, random_seed=seed)

    # Tier 3
    tmp_df2 = station_df.copy()
    tmp_df2[col] = s2
    s3 = impute_long_gaps(tmp_df2, col, era5_df)

    # Build record
    station_df = create_imputation_record(station_df, col, s1, s2, s3)

    # Overwrite col with the final imputed series
    station_df[col] = station_df[f"{col}_imputed"]

    remaining_nan = station_df[col].isna().sum()
    if remaining_nan:
        logger.info("Imputation for %s: %d observations still missing after 3 tiers", col, remaining_nan)
    return station_df


