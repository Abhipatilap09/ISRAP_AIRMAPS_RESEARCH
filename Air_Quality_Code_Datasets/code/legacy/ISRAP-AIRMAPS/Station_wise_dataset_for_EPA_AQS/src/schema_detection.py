"""
schema_detection.py
-------------------
Detect and validate the schema of incoming CSVs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_STATION_COLS = {"datetime", "site"}
POLLUTANT_COLS = {"CO", "NO2", "O3", "PM2.5", "SO2"}

REQUIRED_ERA5_COLS = {"datetime"}
ERA5_FEATURE_COLS = {
    "temp_c", "wind_speed", "blh", "relative_humidity",
    "surface_pressure_hpa", "precip_mm", "u10", "v10",
}

DATETIME_CANDIDATES = [
    "datetime", "datetime_formatted", "date_time",
    "timestamp", "Datetime", "DateTime", "DATETIME",
]


def detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """Return the first column that looks like a datetime column."""
    for col in DATETIME_CANDIDATES:
        if col in df.columns:
            return col
    # Fall back: look for columns that can be parsed as datetimes
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            return col
    return None


def validate_station_schema(df: pd.DataFrame, filepath: str) -> dict:
    """Return a dict of findings for a station CSV."""
    findings: dict = {
        "filepath": filepath,
        "n_rows": len(df),
        "columns": list(df.columns),
        "missing_required": [],
        "available_pollutants": [],
        "missing_pollutants": [],
    }

    for col in REQUIRED_STATION_COLS:
        if col not in df.columns:
            findings["missing_required"].append(col)
            logger.warning("Station %s missing required column '%s'", filepath, col)

    for pol in POLLUTANT_COLS:
        if pol in df.columns:
            findings["available_pollutants"].append(pol)
        else:
            findings["missing_pollutants"].append(pol)

    return findings


def validate_era5_schema(df: pd.DataFrame, filepath: str) -> dict:
    """Return a dict of findings for the ERA5 CSV."""
    findings: dict = {
        "filepath": filepath,
        "n_rows": len(df),
        "columns": list(df.columns),
        "available_features": [],
        "missing_features": [],
    }

    dt_col = detect_datetime_column(df)
    findings["datetime_column"] = dt_col
    if dt_col is None:
        logger.error("ERA5 file %s has no detectable datetime column", filepath)

    for feat in ERA5_FEATURE_COLS:
        if feat in df.columns:
            findings["available_features"].append(feat)
        else:
            findings["missing_features"].append(feat)
            logger.warning("ERA5 missing expected feature '%s'", feat)

    return findings

