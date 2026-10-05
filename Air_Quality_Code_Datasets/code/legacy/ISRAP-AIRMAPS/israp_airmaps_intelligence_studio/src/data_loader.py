"""Data loading, schema detection, hourly reindexing, and ERA5 joining."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    ERA5_FILE,
    ERA5_VARS,
    POLLUTANTS,
    STATION_DIR,
    STATIONS,
    STRUCTURAL_MISSING_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ── Schema helpers ────────────────────────────────────────────────────────────

def _detect_datetime_col(df: pd.DataFrame) -> str:
    """Return the best datetime-like column name in *df*."""
    candidates = ["datetime", "date_time", "timestamp", "time", "date"]
    for c in candidates:
        if c in df.columns:
            return c
    # fall back: first column whose name contains 'date' or 'time'
    for c in df.columns:
        if any(k in c.lower() for k in ("date", "time", "stamp")):
            return c
    raise ValueError(f"No datetime column found among: {list(df.columns)}")


def _parse_datetime(series: pd.Series) -> pd.Series:
    """Try multiple parse strategies; always return UTC-naive hourly datetimes."""
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            result = pd.to_datetime(series, format=fmt, errors="raise", utc=False)
            return result
        except (ValueError, TypeError):
            continue
    # Generic fallback
    return pd.to_datetime(series, infer_datetime_format=True, errors="coerce", utc=False)


def _detect_pollutant_cols(df: pd.DataFrame) -> list[str]:
    """Return pollutant columns present in *df* (normalises common variants)."""
    found = []
    col_map = {
        "co": "CO", "no2": "NO2", "o3": "O3",
        "pm2.5": "PM2.5", "pm25": "PM2.5", "pm_2_5": "PM2.5", "pm_25": "PM2.5",
        "so2": "SO2",
    }
    for col in df.columns:
        norm = col.lower().strip().replace(" ", "")
        if norm in col_map:
            standard = col_map[norm]
            if standard not in found:
                found.append(standard)
    return found


# ── Loader ────────────────────────────────────────────────────────────────────

def load_station(
    station_key: str,
    station_dir: Path = STATION_DIR,
) -> pd.DataFrame:
    """Load one station CSV, reindex to hourly, add metadata columns."""
    meta = STATIONS[station_key]
    path = station_dir / meta["file"]
    logger.info("Loading %s from %s", station_key, path)

    df = pd.read_csv(path, low_memory=False)

    # ── datetime ─────────────────────────────────────────────────────────────
    dt_col = _detect_datetime_col(df)
    if dt_col != "datetime":
        df = df.rename(columns={dt_col: "datetime"})
    df["datetime"] = _parse_datetime(df["datetime"])
    df = df.dropna(subset=["datetime"])
    df["datetime"] = df["datetime"].dt.floor("h")

    # ── dedup ─────────────────────────────────────────────────────────────────
    dupes = df["datetime"].duplicated(keep=False)
    if dupes.any():
        logger.warning("%s: %d duplicate timestamps — keeping mean", station_key, dupes.sum())
        df = df.groupby("datetime", as_index=False).mean(numeric_only=True)

    df = df.sort_values("datetime").reset_index(drop=True)

    # ── reindex to complete hourly grid ───────────────────────────────────────
    full_idx = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
    df = df.set_index("datetime").reindex(full_idx).reset_index()
    df = df.rename(columns={"index": "datetime"})

    # ── add station metadata ──────────────────────────────────────────────────
    df["station_key"] = station_key
    df["station_label"] = meta["label"]
    df["site"] = meta["site"]

    # ── normalise pollutant columns ───────────────────────────────────────────
    rename_map: dict[str, str] = {}
    for col in df.columns:
        norm = col.lower().strip().replace(" ", "")
        mapping = {
            "co": "CO", "no2": "NO2", "o3": "O3",
            "pm2.5": "PM2.5", "pm25": "PM2.5",
            "so2": "SO2",
        }
        if norm in mapping and col != mapping[norm]:
            rename_map[col] = mapping[norm]
    if rename_map:
        df = df.rename(columns=rename_map)

    # ── ensure every standard pollutant column exists ─────────────────────────
    for p in POLLUTANTS:
        if p not in df.columns:
            df[p] = np.nan

    # ── structural missingness flags ──────────────────────────────────────────
    for p in POLLUTANTS:
        valid_frac = df[p].notna().mean()
        df[f"{p}_structural_missing"] = valid_frac < STRUCTURAL_MISSING_THRESHOLD

    return df


def load_all_stations(station_dir: Path = STATION_DIR) -> dict[str, pd.DataFrame]:
    """Load all six station datasets."""
    result = {}
    for key in STATIONS:
        try:
            result[key] = load_station(key, station_dir)
        except Exception as exc:
            logger.error("Failed to load %s: %s", key, exc)
    return result


def load_era5(era5_path: Path = ERA5_FILE) -> pd.DataFrame:
    """Load ERA5 CSV, dedup timestamps, return clean hourly frame."""
    logger.info("Loading ERA5 from %s", era5_path)
    df = pd.read_csv(era5_path, low_memory=False)

    dt_col = _detect_datetime_col(df)
    df["era5_datetime"] = _parse_datetime(df[dt_col])
    df = df.dropna(subset=["era5_datetime"])
    df["era5_datetime"] = df["era5_datetime"].dt.floor("h")

    dupes = df["era5_datetime"].duplicated(keep=False)
    if dupes.any():
        logger.warning("ERA5: %d duplicate timestamps — keeping mean", dupes.sum())
        num_cols = df.select_dtypes(include="number").columns.tolist()
        df = df.groupby("era5_datetime", as_index=False)[num_cols].mean()
    else:
        df = df.set_index("era5_datetime").reset_index()
        df = df.rename(columns={"era5_datetime": "era5_datetime"})

    df = df.sort_values("era5_datetime").reset_index(drop=True)

    # Keep only recognised ERA5 vars plus the datetime
    keep_cols = ["era5_datetime"] + [c for c in ERA5_VARS if c in df.columns]
    df = df[keep_cols]

    return df


def merge_era5(station_df: pd.DataFrame, era5_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join ERA5 meteorology onto a station frame by hourly datetime."""
    merged = station_df.merge(
        era5_df.rename(columns={"era5_datetime": "datetime"}),
        on="datetime",
        how="left",
        suffixes=("", "_era5"),
    )
    era5_matched = merged[[c for c in ERA5_VARS if c in merged.columns]].notna().any(axis=1).mean()
    logger.info("ERA5 join: %.1f%% rows matched", 100 * era5_matched)
    return merged


# ── Audit helpers ─────────────────────────────────────────────────────────────

def audit_station_pollutant(
    df: pd.DataFrame, station_key: str, pollutant: str
) -> dict[str, Any]:
    """Return a dict of quality statistics for one station-pollutant pair."""
    col = df[pollutant] if pollutant in df.columns else pd.Series(dtype=float)
    n_rows = len(df)
    valid = col.notna()
    neg = (col < 0) if valid.any() else pd.Series(False, index=df.index)
    zeros = (col == 0) if valid.any() else pd.Series(False, index=df.index)

    structural = bool(df[f"{pollutant}_structural_missing"].iloc[0]) if f"{pollutant}_structural_missing" in df.columns else (valid.sum() / max(n_rows, 1) < STRUCTURAL_MISSING_THRESHOLD)

    q1 = float(col.quantile(0.25)) if valid.any() else np.nan
    q3 = float(col.quantile(0.75)) if valid.any() else np.nan
    iqr = q3 - q1 if not np.isnan(q1) else np.nan

    return {
        "station": station_key,
        "pollutant": pollutant,
        "total_rows": n_rows,
        "valid_count": int(valid.sum()),
        "missing_count": int((~valid).sum()),
        "missing_pct": round(100 * (~valid).mean(), 2),
        "negative_count": int(neg.sum()),
        "zero_count": int(zeros.sum()),
        "structural_missing": structural,
        "min": round(float(col.min()), 4) if valid.any() else np.nan,
        "max": round(float(col.max()), 4) if valid.any() else np.nan,
        "mean": round(float(col.mean()), 4) if valid.any() else np.nan,
        "median": round(float(col.median()), 4) if valid.any() else np.nan,
        "std": round(float(col.std()), 4) if valid.any() else np.nan,
        "q1": round(q1, 4) if not np.isnan(q1) else np.nan,
        "q3": round(q3, 4) if not np.isnan(q3) else np.nan,
        "iqr": round(iqr, 4) if not np.isnan(iqr) else np.nan,
        "date_start": str(df["datetime"].min().date()) if "datetime" in df.columns else "",
        "date_end": str(df["datetime"].max().date()) if "datetime" in df.columns else "",
    }


def build_availability_matrix(
    station_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return station × pollutant availability summary DataFrame."""
    rows = []
    for skey, df in station_data.items():
        row: dict[str, Any] = {"station": skey, "label": STATIONS[skey]["label"], "site": STATIONS[skey]["site"]}
        for p in POLLUTANTS:
            if p in df.columns:
                frac = df[p].notna().mean()
                structural = frac < STRUCTURAL_MISSING_THRESHOLD
                row[p] = "Not monitored" if structural else f"{100*frac:.1f}%"
                row[f"{p}_pct"] = 0.0 if structural else round(100 * frac, 1)
            else:
                row[p] = "Not monitored"
                row[f"{p}_pct"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)

