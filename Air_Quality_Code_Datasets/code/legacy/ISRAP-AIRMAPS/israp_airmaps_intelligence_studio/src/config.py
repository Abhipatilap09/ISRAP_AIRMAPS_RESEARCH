"""Central configuration for ISRAP AIRMAPS Intelligence Studio."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
STATION_DIR = PROJECT_ROOT.parent / "Station_wise_dataset_for_EPA_AQS"
ERA5_FILE = STATION_DIR / "ERA5_hourly_formatted_00_23.csv"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / "cache"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPORTS_DIR = PROJECT_ROOT / "exports"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# ── Station metadata ──────────────────────────────────────────────────────────
STATIONS: dict[str, dict[str, Any]] = {
    "station1": {
        "file": "station1_data.csv",
        "site": 32,
        "label": "Station 1 (Site 32)",
        "short": "S1-32",
        "monitored_pollutants": ["NO2", "O3", "PM2.5"],
    },
    "station2": {
        "file": "station2_data.csv",
        "site": 52,
        "label": "Station 2 (Site 52)",
        "short": "S2-52",
        "monitored_pollutants": ["NO2", "O3"],
    },
    "station3": {
        "file": "station3_data.csv",
        "site": 59,
        "label": "Station 3 (Site 59)",
        "short": "S3-59",
        "monitored_pollutants": ["NO2", "O3", "PM2.5", "SO2"],
        "special_case": "SO2 has severe systematic corruption (~33% negatives). Default: exclude SO2 from modeling.",
    },
    "station4": {
        "file": "station4_data.csv",
        "site": 1069,
        "label": "Station 4 (Site 1069)",
        "short": "S4-1069",
        "monitored_pollutants": ["CO", "NO2", "PM2.5"],
    },
    "station5": {
        "file": "station5_data.csv",
        "site": 1080,
        "label": "Station 5 (Site 1080)",
        "short": "S5-1080",
        "monitored_pollutants": ["SO2"],
    },
    "station6": {
        "file": "station6_data.csv",
        "site": 1087,
        "label": "Station 6 (Site 1087)",
        "short": "S6-1087",
        "monitored_pollutants": ["PM2.5"],
    },
}

# ── Pollutant metadata ────────────────────────────────────────────────────────
POLLUTANTS: dict[str, dict[str, Any]] = {
    "CO": {
        "label": "Carbon Monoxide",
        "unit": "ppm",
        "physical_min": 0.0,
        "physical_max": 50.0,
        "color": "#e07b39",
    },
    "NO2": {
        "label": "Nitrogen Dioxide",
        "unit": "ppb",
        "physical_min": 0.0,
        "physical_max": 500.0,
        "color": "#9b59b6",
    },
    "O3": {
        "label": "Ozone",
        "unit": "ppm",
        "physical_min": 0.0,
        "physical_max": 0.5,
        "color": "#2ecc71",
    },
    "PM2.5": {
        "label": "PM2.5",
        "unit": "µg/m³",
        "physical_min": 0.0,
        "physical_max": 1000.0,
        "color": "#e74c3c",
    },
    "SO2": {
        "label": "Sulfur Dioxide",
        "unit": "ppb",
        "physical_min": 0.0,
        "physical_max": 500.0,
        "color": "#f39c12",
    },
}

# ── ERA5 variables ────────────────────────────────────────────────────────────
ERA5_VARS: list[str] = [
    "temp_c", "dewpoint_c", "surface_pressure_hpa",
    "precip_mm", "wind_speed", "blh", "relative_humidity",
    "u10", "v10", "t2m", "d2m", "sp", "tp",
]

ERA5_VAR_LABELS: dict[str, str] = {
    "temp_c": "Temperature (°C)",
    "dewpoint_c": "Dewpoint (°C)",
    "surface_pressure_hpa": "Surface Pressure (hPa)",
    "precip_mm": "Precipitation (mm)",
    "wind_speed": "Wind Speed (m/s)",
    "blh": "Boundary Layer Height (m)",
    "relative_humidity": "Relative Humidity (%)",
    "u10": "U-Wind 10m (m/s)",
    "v10": "V-Wind 10m (m/s)",
    "t2m": "T2M (K)",
    "d2m": "D2M (K)",
    "sp": "Surface Pressure (Pa)",
    "tp": "Total Precipitation",
}

# ── Structural missing override ───────────────────────────────────────────────
# If observed-value fraction < this threshold → treated as structurally missing
STRUCTURAL_MISSING_THRESHOLD = 0.01

# Special exclusion: Station 3 SO2 is corrupted
EXCLUDED_BY_DEFAULT: set[tuple[str, str]] = {("station3", "SO2")}

# ── Gap categories ────────────────────────────────────────────────────────────
GAP_CATEGORIES: dict[str, tuple[int, int]] = {
    "very_short": (1, 2),
    "medium": (3, 24),
    "long": (25, 168),
    "very_long": (169, 99999),
}

# ── Anomaly default parameters ────────────────────────────────────────────────
ANOMALY_DEFAULTS: dict[str, Any] = {
    "iqr_multiplier": 3.0,
    "hampel_window": 24,
    "hampel_threshold": 3.0,
    "rolling_zscore_window": 72,
    "rolling_zscore_threshold": 3.0,
    "lof_n_neighbors": 20,
    "lof_contamination": 0.01,
    "iforest_contamination": 0.01,
    "iforest_n_estimators": 200,
    "iforest_random_state": 42,
    "stl_seasonal": 24,
    "stl_threshold": 3.0,
    "screen_percentile": 99,
    "lstm_seq_len": 24,
    "lstm_epochs_fast": 10,
    "lstm_epochs_full": 50,
    "usad_seq_len": 12,
    "usad_epochs_fast": 5,
    "usad_epochs_full": 30,
    "threshold_percentile": 99.0,
}

# ── Imputation defaults ───────────────────────────────────────────────────────
IMPUTATION_DEFAULTS: dict[str, Any] = {
    "linear_max_gap": 2,
    "era5_model": "random_forest",
    "era5_n_lags": [1, 2, 3, 6, 12, 24],
    "era5_rolling_windows": [3, 6, 24],
    "mice_max_iter": 10,
    "mice_random_state": 42,
    "brits_hidden": 64,
    "brits_epochs_fast": 5,
    "brits_epochs_full": 20,
    "saits_d_model": 64,
    "saits_epochs_fast": 5,
    "saits_epochs_full": 20,
}

# ── Misc ──────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
DISPLAY_MAX_ROWS = 500
CHART_DOWNSAMPLE = 5000  # max points before downsampling
APP_VERSION = "1.0.0"

