"""Leakage-safe feature engineering for anomaly detection and imputation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ERA5_VARS

# ── Time features ─────────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame, dt_col: str = "datetime") -> pd.DataFrame:
    dt = df[dt_col]
    df = df.copy()
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek
    df["dom"] = dt.dt.day
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["season"] = df["month"].map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    )
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    doy = dt.dt.day_of_year
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    return df


# ── Lag features (past-only) ──────────────────────────────────────────────────

DEFAULT_LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]


def add_lag_features(
    df: pd.DataFrame,
    col: str,
    lags: list[int] = DEFAULT_LAGS,
) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


# ── Rolling features (past-only) ──────────────────────────────────────────────

DEFAULT_WINDOWS = [3, 6, 12, 24, 72, 168]


def add_rolling_features(
    df: pd.DataFrame,
    col: str,
    windows: list[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        roll = df[col].shift(1).rolling(window=w, min_periods=max(1, w // 4))
        df[f"{col}_rmean{w}"] = roll.mean()
        df[f"{col}_rmedian{w}"] = roll.median()
        df[f"{col}_rmin{w}"] = roll.min()
        df[f"{col}_rmax{w}"] = roll.max()
        df[f"{col}_rstd{w}"] = roll.std()
    return df


# ── ERA5 derived features ─────────────────────────────────────────────────────

def add_era5_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "u10" in df.columns and "v10" in df.columns:
        u = df["u10"].fillna(0)
        v = df["v10"].fillna(0)
        df["wind_dir_deg"] = (np.degrees(np.arctan2(u, v)) + 360) % 360
    if "temp_c" in df.columns and "dewpoint_c" in df.columns:
        df["temp_dew_diff"] = df["temp_c"] - df["dewpoint_c"]
    if "blh" in df.columns:
        df["low_blh"] = (df["blh"] < 200).astype(int)
    if "precip_mm" in df.columns:
        df["rain_event"] = (df["precip_mm"] > 0.1).astype(int)
    if "relative_humidity" in df.columns:
        df["high_humidity"] = (df["relative_humidity"] > 85).astype(int)
    return df


# ── Build full feature matrix ─────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    target_col: str,
    lags: list[int] | None = None,
    windows: list[int] | None = None,
    include_era5: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (feature_df, feature_names) with all engineered features."""
    lags = lags or DEFAULT_LAGS
    windows = windows or DEFAULT_WINDOWS

    df = add_time_features(df)
    df = add_lag_features(df, target_col, lags)
    df = add_rolling_features(df, target_col, windows)
    if include_era5:
        df = add_era5_derived(df)

    time_feats = ["hour", "dow", "dom", "month", "year", "is_weekend", "season",
                  "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    lag_feats = [f"{target_col}_lag{l}" for l in lags]
    roll_feats = []
    for w in windows:
        roll_feats += [f"{target_col}_{s}{w}" for s in ("rmean", "rmedian", "rmin", "rmax", "rstd")]
    era5_feats = [c for c in ERA5_VARS + ["wind_dir_deg", "temp_dew_diff", "low_blh", "rain_event", "high_humidity"] if c in df.columns]

    feature_names = [f for f in time_feats + lag_feats + roll_feats + era5_feats if f in df.columns]
    return df, feature_names

