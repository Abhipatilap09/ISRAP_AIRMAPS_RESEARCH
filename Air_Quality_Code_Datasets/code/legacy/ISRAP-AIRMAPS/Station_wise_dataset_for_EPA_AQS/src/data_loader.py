"""
data_loader.py
--------------
Load station CSVs and ERA5 reanalysis data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .schema_detection import validate_station_schema, validate_era5_schema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_station_files(config: dict) -> List[Path]:
    """
    Find all station CSV files in station_dir.
    Returns a sorted list of Paths matching station*_data.csv.
    """
    base_dir = Path(config.get("station_dir", "."))
    if not base_dir.is_absolute():
        # Resolve relative to the script location (pipeline root)
        base_dir = Path(__file__).parent.parent / base_dir

    patterns = ["station*_data.csv", "station*.csv"]
    found = []
    for pat in patterns:
        found.extend(base_dir.glob(pat))
    found = sorted(set(found))

    if not found:
        logger.error("No station files found in %s", base_dir)
    else:
        logger.info("Found %d station files in %s", len(found), base_dir)
    return found


# ---------------------------------------------------------------------------
# Single-station loader
# ---------------------------------------------------------------------------

def load_station_data(filepath: Path) -> Tuple[pd.DataFrame, dict]:
    """
    Load one station CSV with robust datetime parsing.

    Returns (DataFrame, schema_findings).
    The DataFrame has:
      - DatetimeIndex named 'datetime'
      - site column kept
      - Pollutant columns as float
    """
    logger.info("Loading %s", filepath.name)
    df = pd.read_csv(filepath, low_memory=False)

    # Parse datetime
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        # Try to find any date-like column
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                df["datetime"] = pd.to_datetime(df[col], errors="coerce")
                break

    # Drop rows with unparseable datetime
    before = len(df)
    df = df.dropna(subset=["datetime"])
    if len(df) < before:
        logger.warning(
            "%s: dropped %d rows with unparseable datetime",
            filepath.name, before - len(df),
        )

    # Set datetime as index
    df = df.set_index("datetime").sort_index()
    df.index.name = "datetime"

    # Convert pollutant columns to float
    pollutants = ["CO", "NO2", "O3", "PM2.5", "SO2"]
    for col in pollutants:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    findings = validate_station_schema(df.reset_index(), str(filepath))
    return df, findings


# ---------------------------------------------------------------------------
# ERA5 loader
# ---------------------------------------------------------------------------

def load_era5_data(config: dict) -> pd.DataFrame:
    """
    Load the ERA5 CSV. Uses the 'datetime' column directly.
    Returns DataFrame with DatetimeIndex named 'datetime'.
    """
    station_dir = Path(config.get("station_dir", "."))
    if not station_dir.is_absolute():
        station_dir = Path(__file__).parent.parent / station_dir

    era5_path = (station_dir / config["era5_file"]).resolve()

    if not era5_path.exists():
        logger.error("ERA5 file not found: %s", era5_path)
        raise FileNotFoundError(f"ERA5 file not found: {era5_path}")

    logger.info("Loading ERA5 from %s", era5_path)
    df = pd.read_csv(era5_path, low_memory=False)

    # The ERA5 'datetime' column format: "2019-01-01 00:00:00"
    dt_col = None
    for candidate in ["datetime", "datetime_formatted", "date_time", "timestamp"]:
        if candidate in df.columns:
            dt_col = candidate
            break
    if dt_col is None:
        raise ValueError(f"ERA5 file has no recognisable datetime column. Columns: {list(df.columns)}")

    df["datetime"] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df.set_index("datetime").sort_index()
    df.index.name = "datetime"

    # Numeric coercion
    numeric_cols = [
        "temp_c", "wind_speed", "blh", "relative_humidity",
        "surface_pressure_hpa", "precip_mm", "u10", "v10",
        "tp", "d2m", "t2m", "sp", "dewpoint_c",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Deduplicate ERA5 index (safety: keep first occurrence)
    if df.index.duplicated().any():
        n_dups = df.index.duplicated().sum()
        logger.warning("ERA5 has %d duplicate timestamps; keeping first", n_dups)
        df = df[~df.index.duplicated(keep="first")]

    findings = validate_era5_schema(df.reset_index(), str(era5_path))
    logger.info(
        "ERA5 loaded: %d rows, columns: %s",
        len(df), findings["available_features"],
    )
    return df


# ---------------------------------------------------------------------------
# Load all stations
# ---------------------------------------------------------------------------

def load_all_stations(config: dict) -> Dict[int, Tuple[pd.DataFrame, dict]]:
    """
    Load every station CSV.

    Returns dict: site_id -> (DataFrame, schema_findings).
    The site_id is taken from the 'site' column in the data.
    """
    files = discover_station_files(config)
    result: Dict[int, Tuple[pd.DataFrame, dict]] = {}

    for fp in files:
        try:
            df, findings = load_station_data(fp)
            # Infer site from data
            if "site" in df.columns:
                site_id = int(df["site"].iloc[0]) if len(df) > 0 else -1
            else:
                # Fall back to filename-based mapping
                site_map = config.get("station_site_map", {})
                site_id = site_map.get(fp.name, -1)

            findings["station_file"] = fp.name
            findings["site_id"] = site_id
            result[site_id] = (df, findings)
            logger.info(
                "  Site %s: %d rows, pollutants: %s",
                site_id, len(df), findings["available_pollutants"],
            )
        except Exception as exc:
            logger.error("Failed to load %s: %s", fp, exc, exc_info=True)

    return result


# ---------------------------------------------------------------------------
# ERA5 merge
# ---------------------------------------------------------------------------

def merge_era5_with_station(
    station_df: pd.DataFrame,
    era5_df: pd.DataFrame,
    how: str = "left",
) -> pd.DataFrame:
    """
    Merge ERA5 meteorological columns into a station DataFrame on datetime index.

    Uses a left join by default so station rows are preserved even when ERA5
    has no matching hour (very rare given ERA5 completeness).
    """
    era5_cols = [
        c for c in era5_df.columns
        if c not in station_df.columns and c not in ("date", "time", "datetime_formatted")
    ]
    merged = station_df.join(era5_df[era5_cols], how=how, rsuffix="_era5")
    logger.debug(
        "Merged ERA5 into station: %d rows, %d columns",
        len(merged), len(merged.columns),
    )
    return merged

