from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable

import streamlit as st
import streamlit.components.v1 as components

IMPORT_ERROR: ModuleNotFoundError | None = None

try:
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError as exc:
    IMPORT_ERROR = exc


BASE_DIR = Path(__file__).resolve().parent

APP_CONFIG = {
    "app_title": "Atmospheric Intelligence Hub",
    "app_subtitle": "Research-grade exploration of EPA AQS station pollutants and ERA5 meteorology.",
    "epa_map_url": "https://epa.maps.arcgis.com/apps/webappviewer/index.html?id=5f239fd3e72f424f98ef3d5def547eb5&extent=-146.2334,13.1913,-46.3896,56.5319",
    "station_aqs_ids": {
        "32": "480290032",
        "52": "480290052",
        "59": "480290059",
        "1069": "480291069",
        "1080": "480291080",
        "1087": "480291087",
    },
    "station_clusters": {
        "32": "Northwest / Central",
        "52": "Northwest / Central",
        "59": "Northwest / Central",
        "1069": "Southeast",
        "1080": "Southeast",
        "1087": "Southeast",
    },
    "distance_pairs_miles": {
        ("32", "52"): 10,
        ("32", "59"): 15,
        ("32", "1069"): 28,
        ("32", "1080"): 35,
        ("32", "1087"): 32,
        ("52", "59"): 12,
        ("52", "1069"): 30,
        ("52", "1080"): 41,
        ("52", "1087"): 37,
        ("59", "1069"): 20,
        ("59", "1080"): 36,
        ("59", "1087"): 30,
        ("1069", "1080"): 12,
        ("1069", "1087"): 8,
        ("1080", "1087"): 10,
    },
    "within_cluster_range_miles": "3-9 miles",
    "across_city_range_miles": "19-25 miles",
    "city_wide_spread_miles": "25 miles",
    "pollutant_dir": BASE_DIR / "Station_wise_dataset_for_EPA_AQS",
    "meteorology_file": BASE_DIR / "ERA5_hourly_formatted_00_23.csv",
    "local_timezone": "America/Chicago",
    "datetime_col": "datetime",
    "site_col": "site",
    "state_col": "state_code",
    "county_col": "county_code",
    "pollutant_candidates": ["PM2.5", "NO2", "O3", "SO2", "CO"],
    "meteorology_candidates": [
        "temp_c",
        "relative_humidity",
        "wind_speed",
        "dewpoint_c",
        "surface_pressure_hpa",
        "precip_mm",
        "tp",
        "u10",
        "v10",
        "t2m",
        "sp",
        "blh",
        "d2m",
    ],
    "meteorology_datetime_candidates": ["datetime_local", "datetime", "datetime_formatted"],
    "display_labels": {
        "PM2.5": "PM2.5",
        "NO2": "NO2",
        "O3": "O3",
        "SO2": "SO2",
        "CO": "CO",
        "temp_c": "Temperature (°C)",
        "relative_humidity": "Relative Humidity (%)",
        "wind_speed": "Wind Speed (m/s)",
        "dewpoint_c": "Dew Point (°C)",
        "surface_pressure_hpa": "Surface Pressure (hPa)",
        "precip_mm": "Precipitation (mm)",
        "tp": "Total Precipitation",
        "u10": "U-Wind 10m",
        "v10": "V-Wind 10m",
        "t2m": "2m Temperature",
        "sp": "Surface Pressure",
        "blh": "Boundary Layer Height",
        "d2m": "2m Dew Point",
    },
    "axis_units": {
        "PM2.5": "µg/m³",
        "NO2": "ppb",
        "O3": "ppm",
        "SO2": "ppb",
        "CO": "ppm",
        "temp_c": "°C",
        "relative_humidity": "%",
        "wind_speed": "m/s",
        "dewpoint_c": "°C",
        "surface_pressure_hpa": "hPa",
        "precip_mm": "mm",
        "tp": "mm",
        "u10": "m/s",
        "v10": "m/s",
        "t2m": "K",
        "sp": "Pa",
        "blh": "m",
        "d2m": "K",
    },
    "colorway": [
        "#0072B2",  # blue
        "#D55E00",  # orange
        "#009E73",  # green
        "#CC79A7",  # pink/magenta
        "#F0E442",  # yellow
        "#56B4E9",  # light blue
        "#E69F00",  # amber
        "#999999",  # grey
    ],
    "default_pollutant_priority": ["PM2.5", "NO2", "O3", "SO2", "CO"],
    "default_meteorology_priority": [
        "temp_c",
        "relative_humidity",
        "wind_speed",
        "dewpoint_c",
        "t2m",
        "u10",
        "v10",
    ],
}

AGGREGATION_OPTIONS = {
    "Hourly": "h",
    "Daily": "D",
    "Monthly": "M",
    "Yearly": "Y",
}


def configure_page() -> None:
    st.set_page_config(
        page_title=APP_CONFIG["app_title"],
        page_icon="🌎",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_theme() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg: #f4f7fb;
                --card: rgba(255, 255, 255, 0.96);
                --card-border: rgba(148, 163, 184, 0.18);
                --ink: #0f172a;
                --muted: #475569;
                --accent: #1d4ed8;
                --shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
            }
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(29, 78, 216, 0.07), transparent 32%),
                    radial-gradient(circle at top right, rgba(15, 118, 110, 0.07), transparent 28%),
                    linear-gradient(180deg, #f8fbff 0%, var(--bg) 70%, #eef4fb 100%);
                color: var(--ink);
            }
            [data-testid="stSidebar"] > div:first-child {
                background: linear-gradient(180deg, #081226 0%, #0f1f3d 100%);
                border-right: 1px solid rgba(148, 163, 184, 0.18);
            }
            [data-testid="stSidebar"] .block-container {
                padding-top: 1rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] .stMarkdown,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] span {
                color: #e2e8f0 !important;
            }
            .hero {
                background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(255,255,255,0.90));
                border: 1px solid var(--card-border);
                border-radius: 28px;
                padding: 1.5rem 1.6rem 1.2rem 1.6rem;
                box-shadow: var(--shadow);
                margin-bottom: 1rem;
            }
            .hero-eyebrow {
                color: var(--accent);
                font-size: 0.85rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            .hero h1 {
                margin: 0.3rem 0 0.45rem 0;
                font-size: 2.2rem;
                line-height: 1.08;
            }
            .hero p {
                margin: 0;
                color: var(--muted);
                font-size: 1rem;
            }
            .hero-badges {
                display: flex;
                gap: 0.55rem;
                flex-wrap: wrap;
                margin-top: 0.95rem;
            }
            .badge {
                padding: 0.38rem 0.75rem;
                border-radius: 999px;
                background: rgba(15, 23, 42, 0.05);
                color: var(--ink);
                font-size: 0.84rem;
                font-weight: 600;
            }
            .kpi-card {
                background: var(--card);
                border: 1px solid var(--card-border);
                border-radius: 24px;
                padding: 1rem 1.1rem;
                box-shadow: var(--shadow);
                min-height: 128px;
            }
            .kpi-label {
                color: var(--muted);
                font-size: 0.82rem;
                font-weight: 700;
                letter-spacing: 0.05em;
                text-transform: uppercase;
            }
            .kpi-value {
                color: var(--ink);
                font-size: 1.65rem;
                font-weight: 800;
                line-height: 1.08;
                margin: 0.55rem 0 0.45rem 0;
            }
            .kpi-caption {
                color: var(--muted);
                font-size: 0.88rem;
                line-height: 1.35;
            }
            .sidebar-card {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 22px;
                padding: 0.95rem 1rem;
                margin-bottom: 0.9rem;
            }
            .tab-note {
                color: var(--muted);
                margin: -0.25rem 0 0.8rem 0;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.45rem;
            }
            .stTabs [data-baseweb="tab"] {
                background: rgba(255, 255, 255, 0.7);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 18px;
                padding: 0.6rem 0.9rem;
                font-weight: 700;
            }
            .stTabs [aria-selected="true"] {
                background: white !important;
                color: var(--accent) !important;
                box-shadow: var(--shadow);
            }
            div[data-testid="stPlotlyChart"] {
                background: rgba(255, 255, 255, 0.90);
                border: 1px solid var(--card-border);
                border-radius: 24px;
                padding: 0.4rem;
                box-shadow: var(--shadow);
            }
            div[data-testid="stDataFrame"] {
                background: rgba(255, 255, 255, 0.90);
                border: 1px solid var(--card-border);
                border-radius: 20px;
                padding: 0.35rem;
                box-shadow: var(--shadow);
            }
            .stAlert {
                border-radius: 20px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def display_label(column_name: str | None) -> str:
    if column_name is None:
        return "None"
    return APP_CONFIG["display_labels"].get(column_name, column_name)


def axis_label(column_name: str | None) -> str:
    if column_name is None:
        return "None"
    unit = APP_CONFIG["axis_units"].get(column_name)
    return f"{display_label(column_name)} ({unit})" if unit else display_label(column_name)


def compact_station_label(label: str) -> str:
    if label.startswith("Site "):
        return label
    return f"Site {label}"


def build_station_label(site_value: str) -> str:
    return f"Site {site_value}"


def normalize_datetime_series(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    try:
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_convert(APP_CONFIG["local_timezone"]).dt.tz_localize(None)
    except Exception:
        pass
    return dt


def build_era5_datetime(frame: pd.DataFrame) -> pd.Series:
    if "datetime_local" in frame.columns:
        return normalize_datetime_series(frame["datetime_local"])

    if {"date", "time"}.issubset(frame.columns):
        time_values = frame["time"].astype(str).str.replace(".", ":", regex=False)
        combined = frame["date"].astype(str) + " " + time_values
        return pd.to_datetime(combined, errors="coerce")

    datetime_column = next(
        (c for c in APP_CONFIG["meteorology_datetime_candidates"] if c in frame.columns),
        None,
    )
    if datetime_column is None:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    return normalize_datetime_series(frame[datetime_column])


def first_available(options: Iterable[str], priority: Iterable[str]) -> str | None:
    option_list = list(options)
    for candidate in priority:
        if candidate in option_list:
            return candidate
    return option_list[0] if option_list else None


def station_files() -> list[Path]:
    primary_files = sorted(APP_CONFIG["pollutant_dir"].glob("*.csv"))
    if primary_files:
        return primary_files
    fallback_files = sorted(BASE_DIR.glob("station*_data.csv"))
    if fallback_files:
        return fallback_files
    return sorted(BASE_DIR.glob("*.csv"))


@st.cache_data(show_spinner="Loading EPA station pollutant files...")
def load_pollutant_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    station_meta: list[dict[str, object]] = []

    files = station_files()
    if not files:
        raise FileNotFoundError(
            f"No station CSV files were found in {APP_CONFIG['pollutant_dir']}."
        )

    for index, csv_path in enumerate(files, start=1):
        frame = pd.read_csv(csv_path)
        if APP_CONFIG["datetime_col"] not in frame.columns:
            continue

        frame[APP_CONFIG["datetime_col"]] = pd.to_datetime(
            frame[APP_CONFIG["datetime_col"]], errors="coerce"
        )
        frame = frame.dropna(subset=[APP_CONFIG["datetime_col"]]).sort_values(
            APP_CONFIG["datetime_col"]
        )
        if frame.empty:
            continue

        pollutant_columns = [
            col for col in APP_CONFIG["pollutant_candidates"] if col in frame.columns
        ]
        for col in pollutant_columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

        site_value = (
            str(frame[APP_CONFIG["site_col"]].dropna().iloc[0])
            if APP_CONFIG["site_col"] in frame.columns and not frame[APP_CONFIG["site_col"]].dropna().empty
            else f"unknown-{index}"
        )
        state_value = (
            str(frame[APP_CONFIG["state_col"]].dropna().iloc[0])
            if APP_CONFIG["state_col"] in frame.columns and not frame[APP_CONFIG["state_col"]].dropna().empty
            else "NA"
        )
        county_value = (
            str(frame[APP_CONFIG["county_col"]].dropna().iloc[0])
            if APP_CONFIG["county_col"] in frame.columns and not frame[APP_CONFIG["county_col"]].dropna().empty
            else "NA"
        )

        station_label = build_station_label(site_value)
        keep_columns = [
            col
            for col in [
                APP_CONFIG["datetime_col"],
                APP_CONFIG["state_col"],
                APP_CONFIG["county_col"],
                APP_CONFIG["site_col"],
                *pollutant_columns,
            ]
            if col in frame.columns
        ]

        trimmed = frame[keep_columns].copy()
        trimmed["station_label"] = station_label
        trimmed["station_file"] = csv_path.stem
        frames.append(trimmed)

        station_meta.append(
            {
                "station_label": station_label,
                "station_file": csv_path.stem,
                "site": site_value,
                "site_sort": int(float(site_value)) if str(site_value).replace(".", "", 1).isdigit() else index,
                "state_code": state_value,
                "county_code": county_value,
                "start": frame[APP_CONFIG["datetime_col"]].min(),
                "end": frame[APP_CONFIG["datetime_col"]].max(),
                "records": int(len(frame)),
            }
        )

    if not frames:
        raise ValueError("Station files were found, but none contained a usable datetime column.")

    pollutant_df = pd.concat(frames, ignore_index=True, sort=False).rename(
        columns={APP_CONFIG["datetime_col"]: "datetime"}
    )
    sort_cols = [APP_CONFIG["site_col"], "datetime"] if APP_CONFIG["site_col"] in pollutant_df.columns else ["station_label", "datetime"]
    pollutant_df = pollutant_df.sort_values(sort_cols).reset_index(drop=True)

    available_pollutants = [
        col
        for col in APP_CONFIG["pollutant_candidates"]
        if col in pollutant_df.columns and pollutant_df[col].notna().any()
    ]
    station_meta_df = pd.DataFrame(station_meta).sort_values("site_sort").reset_index(drop=True)
    return pollutant_df, station_meta_df, available_pollutants


@st.cache_data(show_spinner="Loading ERA5 meteorology...")
def load_meteorology_data() -> tuple[pd.DataFrame, list[str]]:
    csv_path = APP_CONFIG["meteorology_file"]
    if not csv_path.exists():
        raise FileNotFoundError(f"Meteorology file not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    frame["datetime"] = build_era5_datetime(frame)
    if frame["datetime"].isna().all():
        raise ValueError("Meteorology file needs a usable datetime source.")
    frame = frame.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates("datetime")

    available_columns = [
        col for col in APP_CONFIG["meteorology_candidates"] if col in frame.columns
    ]
    for col in available_columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    available_meteorology = [col for col in available_columns if frame[col].notna().any()]
    trimmed = frame[["datetime", *available_meteorology]].copy()
    return trimmed.reset_index(drop=True), available_meteorology


def default_state(
    station_options: list[str],
    available_pollutants: list[str],
    available_meteorology: list[str],
    min_ts: pd.Timestamp,
    max_ts: pd.Timestamp,
) -> dict[str, object]:
    return {
        "station_search": "",
        "selected_stations": station_options[: min(3, len(station_options))],
        "selected_pollutant": first_available(available_pollutants, APP_CONFIG["default_pollutant_priority"]),
        "selected_meteorology": first_available(available_meteorology, APP_CONFIG["default_meteorology_priority"]),
        "aggregation": "Daily",
        "rolling_window": 7,
        "from_date": min_ts.date(),
        "to_date": max_ts.date(),
    }


def initialize_state(defaults: dict[str, object]) -> None:
    if "filters_initialized" in st.session_state:
        return
    for key, value in defaults.items():
        st.session_state[key] = value
    st.session_state["filters_initialized"] = True


def reset_filters(defaults: dict[str, object]) -> None:
    for key, value in defaults.items():
        st.session_state[key] = value


def determine_mode(pollutant: str | None, meteorology: str | None) -> str:
    if pollutant is not None and meteorology is not None:
        return "combined"
    if pollutant is not None:
        return "pollutant"
    if meteorology is not None:
        return "meteorology"
    return "empty"


def filter_pollutant_data(
    pollutant_df: pd.DataFrame,
    selected_stations: list[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    frame = pollutant_df[pollutant_df["datetime"].between(start_ts, end_ts, inclusive="both")].copy()
    if selected_stations:
        frame = frame[frame["station_label"].isin(selected_stations)]
    return frame.sort_values(["station_label", "datetime"]).reset_index(drop=True)


def filter_meteorology_data(
    meteorology_df: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    return (
        meteorology_df[meteorology_df["datetime"].between(start_ts, end_ts, inclusive="both")]
        .copy()
        .sort_values("datetime")
        .reset_index(drop=True)
    )


def expand_meteorology_for_stations(meteorology_df: pd.DataFrame, selected_stations: list[str]) -> pd.DataFrame:
    if meteorology_df.empty:
        return meteorology_df.assign(station_label=pd.Series(dtype="object"))
    labels = selected_stations or ["Regional ERA5"]
    frames = []
    for station in labels:
        clone = meteorology_df.copy()
        clone["station_label"] = station
        frames.append(clone)
    return pd.concat(frames, ignore_index=True).sort_values(["station_label", "datetime"]).reset_index(drop=True)


def build_combined_dataset(pollutant_df: pd.DataFrame, meteorology_df: pd.DataFrame) -> pd.DataFrame:
    if pollutant_df.empty or meteorology_df.empty:
        return pd.DataFrame()
    return pollutant_df.merge(meteorology_df, on="datetime", how="left").sort_values(["station_label", "datetime"]).reset_index(drop=True)


def create_download_dataset(
    mode: str,
    pollutant_df: pd.DataFrame,
    meteorology_df: pd.DataFrame,
    meteorology_expanded_df: pd.DataFrame,
    selected_pollutant: str | None,
    selected_meteorology: str | None,
) -> pd.DataFrame:
    if mode == "combined":
        combined = build_combined_dataset(pollutant_df, meteorology_df)
        keep = [
            c for c in [
                "datetime", "station_label", APP_CONFIG["state_col"], APP_CONFIG["county_col"],
                APP_CONFIG["site_col"], selected_pollutant, selected_meteorology
            ] if c in combined.columns
        ]
        return combined[keep]

    if mode == "pollutant" and selected_pollutant:
        keep = [
            c for c in [
                "datetime", "station_label", APP_CONFIG["state_col"], APP_CONFIG["county_col"],
                APP_CONFIG["site_col"], selected_pollutant
            ] if c in pollutant_df.columns
        ]
        return pollutant_df[keep]

    if mode == "meteorology" and selected_meteorology:
        keep = [c for c in ["datetime", "station_label", selected_meteorology] if c in meteorology_expanded_df.columns]
        return meteorology_expanded_df[keep]

    return pd.DataFrame()


def download_bytes(frame: pd.DataFrame) -> bytes:
    if frame.empty:
        return b""
    buffer = BytesIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue()


def select_value_frame(data: pd.DataFrame, value_column: str) -> pd.DataFrame:
    if value_column not in data.columns:
        return pd.DataFrame(columns=["datetime", "station_label", value_column])
    frame = data[["datetime", "station_label", value_column]].copy()
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    return frame.dropna(subset=[value_column]).sort_values(["station_label", "datetime"])


def selected_day_count(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> int:
    return int((end_ts.floor("D") - start_ts.floor("D")).days) + 1


def is_single_day_selection(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> bool:
    return selected_day_count(start_ts, end_ts) == 1


def is_three_day_selection(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> bool:
    return selected_day_count(start_ts, end_ts) == 3


def is_fifteen_day_selection(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> bool:
    return selected_day_count(start_ts, end_ts) == 15


def is_full_month_selection(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> bool:
    start_day = start_ts.floor("D")
    end_day = end_ts.floor("D")
    return (
        start_day.year == end_day.year
        and start_day.month == end_day.month
        and start_day.day == 1
        and end_day.day == end_day.days_in_month
    )


def is_full_year_selection(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> bool:
    start_day = start_ts.floor("D")
    end_day = end_ts.floor("D")
    return (
        start_day.year == end_day.year
        and start_day.month == 1
        and start_day.day == 1
        and end_day.month == 12
        and end_day.day == 31
    )


def month_bounds(reference_ts: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    month_start = reference_ts.floor("D").replace(day=1)
    month_end = (month_start + pd.offsets.MonthEnd(1)).floor("D") + pd.Timedelta(hours=23)
    return month_start, month_end


def year_bounds(reference_ts: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    year_start = reference_ts.floor("D").replace(month=1, day=1)
    year_end = year_start.replace(month=12, day=31) + pd.Timedelta(hours=23)
    return year_start, year_end


def floor_to_period(datetime_series: pd.Series, aggregation: str) -> pd.Series:
    if aggregation == "Hourly":
        return datetime_series.dt.floor("h")
    if aggregation == "Daily":
        return datetime_series.dt.floor("D")
    if aggregation == "Monthly":
        return datetime_series.dt.to_period("M").dt.to_timestamp()
    return datetime_series.dt.to_period("Y").dt.to_timestamp()


def aggregate_series(data: pd.DataFrame, value_column: str, aggregation: str) -> pd.DataFrame:
    frame = select_value_frame(data, value_column)
    if frame.empty:
        return frame
    frame["period"] = floor_to_period(frame["datetime"], aggregation)
    return (
        frame.groupby(["station_label", "period"], observed=True, as_index=False)[value_column]
        .mean()
        .sort_values(["station_label", "period"])
        .reset_index(drop=True)
    )


def diurnal_profile(data: pd.DataFrame, value_column: str) -> pd.DataFrame:
    frame = select_value_frame(data, value_column)
    if frame.empty:
        return frame
    frame["hour"] = frame["datetime"].dt.hour
    return (
        frame.groupby(["station_label", "hour"], observed=True, as_index=False)[value_column]
        .mean()
        .sort_values(["station_label", "hour"])
        .reset_index(drop=True)
    )


def resample_to_expected_index(
    data: pd.DataFrame,
    value_column: str,
    expected_index: pd.DatetimeIndex,
    freq: str,
    time_column: str,
) -> pd.DataFrame:
    frame = select_value_frame(data, value_column)
    if frame.empty or len(expected_index) == 0:
        return pd.DataFrame(columns=[time_column, "station_label", value_column])

    pieces = []
    for station, station_frame in frame.groupby("station_label", observed=True):
        station_series = (
            station_frame.set_index("datetime")[value_column]
            .sort_index()
            .resample(freq)
            .mean()
            .reindex(expected_index)
        )
        station_resampled = (
            station_series.rename(value_column)
            .rename_axis(time_column)
            .reset_index()
        )
        station_resampled["station_label"] = station
        pieces.append(station_resampled)

    if not pieces:
        return pd.DataFrame(columns=[time_column, "station_label", value_column])

    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["station_label", time_column])
        .reset_index(drop=True)
    )


def quantile_summary(data: pd.DataFrame, value_column: str, station_order: list[str]) -> pd.DataFrame:
    frame = select_value_frame(data, value_column)
    if frame.empty:
        return pd.DataFrame()
    summary = (
        frame.groupby("station_label", observed=True)[value_column]
        .agg(observations="count", mean="mean", std="std", minimum="min", median="median", maximum="max")
        .reset_index()
    )
    q1 = frame.groupby("station_label", observed=True)[value_column].quantile(0.25)
    q3 = frame.groupby("station_label", observed=True)[value_column].quantile(0.75)
    summary["Q1"] = summary["station_label"].map(q1)
    summary["Q3"] = summary["station_label"].map(q3)

    if station_order:
        summary["station_label"] = pd.Categorical(summary["station_label"], categories=station_order, ordered=True)
        summary = summary.sort_values("station_label")
        summary["station_label"] = summary["station_label"].astype(str)

    return summary.reset_index(drop=True)


def extreme_days_table(data: pd.DataFrame, value_column: str) -> pd.DataFrame:
    frame = select_value_frame(data, value_column)
    if frame.empty:
        return pd.DataFrame()
    frame["date"] = frame["datetime"].dt.date
    table = (
        frame.groupby(["station_label", "date"], observed=True)[value_column]
        .agg(daily_mean="mean", daily_max="max", daily_min="min", valid_obs="count")
        .reset_index()
    )
    if table.empty:
        return table
    threshold = table["daily_max"].quantile(0.90)
    return table[table["daily_max"] >= threshold].sort_values("daily_max", ascending=False).head(12).reset_index(drop=True)


def sample_scatter_data(frame: pd.DataFrame, max_points: int = 12000) -> pd.DataFrame:
    if frame.empty or len(frame) <= max_points:
        return frame
    pieces = []
    per_station = max(max_points // max(frame["station_label"].nunique(), 1), 1)
    for _, station_frame in frame.groupby("station_label", observed=True):
        pieces.append(station_frame.sample(min(len(station_frame), per_station), random_state=42))
    return pd.concat(pieces, ignore_index=True).sort_values(["station_label", "datetime"]).reset_index(drop=True)


def compose_title(title: str, subtitle: str | None = None) -> str:
    if not subtitle:
        return title
    return (
        f"{title}<br>"
        f"<span style='font-size:13px;color:#64748b;font-weight:500'>{subtitle}</span>"
    )


def format_time_window(start_ts: pd.Timestamp, end_ts: pd.Timestamp, resolution: str | None = None) -> str:
    start_ts = pd.Timestamp(start_ts)
    end_ts = pd.Timestamp(end_ts)

    if is_full_year_selection(start_ts, end_ts):
        window_label = f"Year {start_ts:%Y}"
    elif is_full_month_selection(start_ts, end_ts):
        window_label = f"{start_ts:%B %Y}"
    elif start_ts.floor("D") == end_ts.floor("D"):
        window_label = f"{start_ts:%Y-%m-%d} | {start_ts:%H:%M}-{end_ts:%H:%M}"
    else:
        window_label = f"{start_ts:%Y-%m-%d %H:%M} to {end_ts:%Y-%m-%d %H:%M}"

    if resolution:
        return f"{window_label} | {resolution}"
    return window_label


def format_window_from_data(data: pd.DataFrame, resolution: str | None = None) -> str | None:
    if "datetime" not in data.columns or data.empty:
        return resolution

    dt = pd.to_datetime(data["datetime"], errors="coerce").dropna()
    if dt.empty:
        return resolution

    return format_time_window(dt.min(), dt.max(), resolution)


def chart_template(
    fig: go.Figure,
    title: str,
    y_title: str,
    x_title: str,
    height: int = 420,
    subtitle: str | None = None,
) -> go.Figure:
    fig.update_layout(
        title={
            "text": compose_title(title, subtitle),
            "x": 0.02,
            "xanchor": "left",
            "y": 0.955,
            "yanchor": "top",
            "pad": {"t": 14, "b": 8},
            "automargin": True,
        },
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "color": "#0f172a"},
        colorway=APP_CONFIG["colorway"],
        height=height,
        margin={"l": 28, "r": 24, "t": 126, "b": 84},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.22,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(255,255,255,0.72)",
        },
    )
    fig.update_xaxes(title=x_title, showgrid=True, gridcolor="rgba(148, 163, 184, 0.18)", zeroline=False)
    fig.update_yaxes(title=y_title, showgrid=True, gridcolor="rgba(148, 163, 184, 0.16)", zeroline=False)
    return fig


def add_stats_overlay(fig: go.Figure, y_values: pd.Series, completeness_pct: float | None = None) -> go.Figure:
    y_clean = pd.Series(y_values).dropna()
    if y_clean.empty:
        return fig

    mean_val = float(y_clean.mean())
    std_val = float(y_clean.std()) if len(y_clean) > 1 else 0.0
    min_val = float(y_clean.min())
    max_val = float(y_clean.max())

    fig.add_hline(y=mean_val, line_dash="dash", line_width=2, opacity=0.8)

    if std_val > 0:
        fig.add_hrect(
            y0=mean_val - std_val,
            y1=mean_val + std_val,
            fillcolor="lightblue",
            opacity=0.18,
            line_width=0,
        )

    stats_text = (
        f"Mean: {mean_val:.2f}<br>"
        f"Std: {std_val:.2f}<br>"
        f"Min: {min_val:.2f}<br>"
        f"Max: {max_val:.2f}"
    )
    if completeness_pct is not None:
        stats_text += f"<br>Data Completeness: {completeness_pct:.1f}%"

    fig.add_annotation(
        x=0.995,
        y=0.98,
        xref="paper",
        yref="paper",
        text=stats_text,
        showarrow=False,
        xanchor="right",
        yanchor="top",
        bordercolor="black",
        borderwidth=1,
        bgcolor="rgba(255,255,255,0.86)",
        font={"size": 12},
    )
    return fig


def add_peak_annotation(fig: go.Figure, x_value, y_value, text: str = "Peak") -> go.Figure:
    if pd.isna(y_value):
        return fig
    fig.add_trace(
        go.Scatter(
            x=[x_value],
            y=[y_value],
            mode="markers",
            marker={"size": 12, "color": "#f97316", "line": {"width": 1, "color": "white"}},
            name=text,
            showlegend=False,
        )
    )
    fig.add_annotation(
        x=x_value,
        y=y_value,
        text=f"{text}<br><b>{y_value:.2f}</b>",
        showarrow=True,
        arrowhead=2,
        ay=-35,
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="rgba(0,0,0,0.15)",
    )
    return fig


def hourly_axis(fig: go.Figure, row: int | None = None, col: int | None = None) -> go.Figure:
    kwargs = {}
    if row is not None and col is not None:
        kwargs = {"row": row, "col": col}
    fig.update_xaxes(
        tickmode="linear",
        dtick=1,
        range=[0, 23],
        tickvals=list(range(24)),
        ticktext=[f"{h:02d}:00" for h in range(24)],
        **kwargs,
    )
    return fig


def datetime_axis(
    fig: go.Figure,
    tickformat: str = "%d %H:%M",
    hoverformat: str = "%Y-%m-%d %H:%M",
    dtick: str | int | None = None,
    tickangle: int = -45,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    kwargs = {}
    if row is not None and col is not None:
        kwargs = {"row": row, "col": col}
    axis_kwargs = {
        "tickformat": tickformat,
        "hoverformat": hoverformat,
        "tickangle": tickangle,
        "automargin": True,
    }
    if dtick is not None:
        axis_kwargs["dtick"] = dtick
    fig.update_xaxes(**axis_kwargs, **kwargs)
    return fig


def axis_config_by_window(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> dict[str, object]:
    span = end_ts - start_ts
    if span <= pd.Timedelta(days=1):
        return {
            "tickformat": "%H:%M",
            "hoverformat": "%Y-%m-%d %H:%M",
            "dtick": 3600 * 1000,
            "tickangle": -45,
        }
    if span <= pd.Timedelta(days=3):
        return {
            "tickformat": "%d %b %H:%M",
            "hoverformat": "%Y-%m-%d %H:%M",
            "dtick": 3 * 3600 * 1000,
            "tickangle": -45,
        }
    if span <= pd.Timedelta(days=31):
        return {
            "tickformat": "%d %b",
            "hoverformat": "%Y-%m-%d",
            "dtick": "D1",
            "tickangle": -45,
        }
    if span <= pd.Timedelta(days=366):
        return {
            "tickformat": "%d %b %Y",
            "hoverformat": "%Y-%m",
            "dtick": "M1",
            "tickangle": -45,
        }
    return {
        "tickformat": "%b %Y",
        "hoverformat": "%Y",
        "dtick": "M3",
        "tickangle": -45,
    }


def apply_temporal_axis(fig: go.Figure, aggregation: str, start_ts: pd.Timestamp | None = None, end_ts: pd.Timestamp | None = None) -> go.Figure:
    aggregation_key = aggregation.lower()
    if start_ts is not None and end_ts is not None:
        config = axis_config_by_window(start_ts, end_ts)
        return datetime_axis(fig, **config)

    if aggregation_key == "hourly":
        return datetime_axis(fig, tickformat="%d %b %H:%M", hoverformat="%Y-%m-%d %H:%M", dtick=3600 * 1000)
    if aggregation_key == "daily":
        return datetime_axis(fig, tickformat="%d %b", hoverformat="%Y-%m-%d", dtick="D1")
    if aggregation_key == "monthly":
        return datetime_axis(fig, tickformat="%b %Y", hoverformat="%Y-%m", dtick="M1")
    if aggregation_key == "yearly":
        return datetime_axis(fig, tickformat="%Y", hoverformat="%Y", dtick="M12")
    return fig


def prefers_webgl(point_count: int) -> bool:
    return point_count >= 1500


def create_diurnal_figure(data: pd.DataFrame, value_column: str) -> go.Figure | None:
    profile = diurnal_profile(data, value_column)
    if profile.empty:
        return None

    fig = px.line(
        profile,
        x="hour",
        y=value_column,
        color="station_label",
        markers=True,
        color_discrete_sequence=APP_CONFIG["colorway"],
    )
    fig.update_traces(line={"width": 3}, marker={"size": 6})
    for trace in fig.data:
        if getattr(trace, "name", None):
            trace.name = compact_station_label(trace.name)

    fig = chart_template(
        fig,
        f"{display_label(value_column)} Diurnal Variation",
        axis_label(value_column),
        "Time (Hour of Day)",
        height=430,
        subtitle=format_window_from_data(data, "Mean by hour of day"),
    )
    fig.update_layout(legend_title_text="", showlegend=True)
    fig = hourly_axis(fig)
    fig.update_xaxes(title_text="Hour of day (00:00–23:00)")
    fig.update_traces(hovertemplate="%{x:02.0f}:00<br>%{y:.2f}<extra>%{fullData.name}</extra>")

    base_station = profile["station_label"].iloc[0]
    station_profile = profile[profile["station_label"] == base_station]
    completeness_pct = (station_profile[value_column].notna().sum() / 24) * 100
    fig = add_stats_overlay(fig, station_profile[value_column], completeness_pct)

    peak_row = station_profile.loc[station_profile[value_column].idxmax()]
    fig = add_peak_annotation(fig, peak_row["hour"], peak_row[value_column], "Peak")
    return fig


def create_single_day_figure(
    data: pd.DataFrame,
    value_column: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> go.Figure | None:
    full_hours = pd.date_range(start=start_ts.floor("D"), end=end_ts.floor("h"), freq="h")
    hourly = resample_to_expected_index(data, value_column, full_hours, "h", "datetime")
    if hourly.empty:
        return None

    hourly["hour"] = hourly["datetime"].dt.hour

    fig = px.line(
        hourly,
        x="hour",
        y=value_column,
        color="station_label",
        markers=True,
        color_discrete_sequence=APP_CONFIG["colorway"],
    )
    fig.update_traces(line={"width": 3}, marker={"size": 6})
    for trace in fig.data:
        if getattr(trace, "name", None):
            trace.name = compact_station_label(trace.name)

    fig = chart_template(
        fig,
        f"{display_label(value_column)} Daily Variation",
        axis_label(value_column),
        "Time (Hour of Day)",
        height=430,
        subtitle=format_time_window(start_ts, end_ts, "Hourly resolution"),
    )
    fig = hourly_axis(fig)

    base_station = hourly["station_label"].dropna().iloc[0]
    station_hourly = hourly[hourly["station_label"] == base_station]
    completeness_pct = (station_hourly[value_column].notna().sum() / max(len(full_hours), 1)) * 100
    fig = add_stats_overlay(fig, station_hourly[value_column], completeness_pct)

    if station_hourly[value_column].notna().any():
        peak_row = station_hourly.loc[station_hourly[value_column].idxmax()]
        fig = add_peak_annotation(fig, peak_row["hour"], peak_row[value_column], "Peak")
    return fig


def create_three_day_figure(data: pd.DataFrame, value_column: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> go.Figure | None:
    full_hours = pd.date_range(start=start_ts.floor("h"), end=end_ts.floor("h"), freq="h")
    hourly = resample_to_expected_index(data, value_column, full_hours, "h", "datetime")
    if hourly.empty:
        return None

    hourly["day"] = hourly["datetime"].dt.floor("D")
    hourly["hour"] = hourly["datetime"].dt.hour
    days = pd.date_range(start=start_ts.floor("D"), end=end_ts.floor("D"), freq="D")
    if len(days) != 3:
        return None

    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        subplot_titles=[f"{d:%Y-%m-%d}" for d in days],
        horizontal_spacing=0.04,
    )

    station_order = list(hourly["station_label"].drop_duplicates())
    if not station_order:
        return None

    for idx, station in enumerate(station_order):
        color = APP_CONFIG["colorway"][idx % len(APP_CONFIG["colorway"])]
        station_frame = hourly[hourly["station_label"] == station]

        for col_idx, day in enumerate(days, start=1):
            day_frame = station_frame[station_frame["day"] == day]

            fig.add_trace(
                go.Scatter(
                    x=day_frame["hour"],
                    y=day_frame[value_column],
                    mode="lines+markers",
                    line={"width": 3, "color": color},
                    marker={"size": 6},
                    name=compact_station_label(station),
                    legendgroup=station,
                    showlegend=(col_idx == 1),
                ),
                row=1,
                col=col_idx,
            )

    fig = chart_template(
        fig,
        f"{display_label(value_column)} Variation Over 3 Days",
        axis_label(value_column),
        "Time (Hour of Day)",
        height=470,
        subtitle=format_time_window(start_ts, end_ts, "Hourly resolution"),
    )

    for col_idx in range(1, 4):
        fig = hourly_axis(fig, row=1, col=col_idx)

    base_station = station_order[0]
    station_data = hourly[hourly["station_label"] == base_station]
    completeness_pct = (station_data[value_column].notna().sum() / max(len(full_hours), 1)) * 100
    fig = add_stats_overlay(fig, station_data[value_column], completeness_pct)

    if station_data[value_column].notna().any():
        peak_row = station_data.loc[station_data[value_column].idxmax()]
        fig.add_annotation(
            x=peak_row["hour"],
            y=peak_row[value_column],
            xref="x2" if peak_row["day"] == days[1] else ("x3" if peak_row["day"] == days[2] else "x"),
            yref="y",
            text=f"Peak<br><b>{peak_row[value_column]:.2f}</b>",
            showarrow=True,
            arrowhead=2,
            ay=-35,
            bgcolor="rgba(255,255,255,0.85)",
        )
    return fig


def create_fifteen_day_figure(data: pd.DataFrame, value_column: str) -> go.Figure | None:
    frame = select_value_frame(data, value_column)
    if frame.empty:
        return None
    frame["date"] = frame["datetime"].dt.floor("D")
    daily = (
        frame.groupby(["station_label", "date"], observed=True, as_index=False)[value_column]
        .mean()
        .sort_values(["station_label", "date"])
    )

    fig = px.line(
        daily,
        x="date",
        y=value_column,
        color="station_label",
        markers=True,
        color_discrete_sequence=APP_CONFIG["colorway"],
    )
    fig.update_traces(line={"width": 3}, marker={"size": 6})
    for trace in fig.data:
        if getattr(trace, "name", None):
            trace.name = compact_station_label(trace.name)
    fig = chart_template(
        fig,
        f"{display_label(value_column)} 15-Day Trend (Daily Average)",
        axis_label(value_column),
        "Date",
        height=430,
        subtitle=format_window_from_data(data, "Daily average"),
    )

    base_station = daily["station_label"].iloc[0]
    station_daily = daily[daily["station_label"] == base_station]
    fig = add_stats_overlay(fig, station_daily[value_column], 100.0)
    peak_row = station_daily.loc[station_daily[value_column].idxmax()]
    fig = add_peak_annotation(fig, peak_row["date"], peak_row[value_column], "Peak")
    return fig


def create_monthly_hourly_figure(
    data: pd.DataFrame,
    value_column: str,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
) -> go.Figure | None:
    full_hours = pd.date_range(start=month_start.floor("D"), end=month_end.floor("h"), freq="h")
    frame = resample_to_expected_index(data, value_column, full_hours, "h", "datetime")
    if frame.empty:
        return None
    fig = px.line(
        frame,
        x="datetime",
        y=value_column,
        color="station_label",
        color_discrete_sequence=APP_CONFIG["colorway"],
    )
    fig.update_traces(line={"width": 2.5})
    for trace in fig.data:
        if getattr(trace, "name", None):
            trace.name = compact_station_label(trace.name)

    fig = chart_template(
        fig,
        f"{display_label(value_column)} Monthly Variation",
        axis_label(value_column),
        "Date",
        height=430,
        subtitle=format_time_window(month_start, month_end, "Hourly resolution"),
    )
    fig = datetime_axis(fig, tickformat="%d %b", hoverformat="%Y-%m-%d %H:%M", dtick="D2", tickangle=-45)
    fig.update_traces(hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}<extra>%{fullData.name}</extra>")

    base_station = frame["station_label"].iloc[0]
    station_frame = frame[frame["station_label"] == base_station]
    completeness_pct = (station_frame[value_column].notna().sum() / max(len(full_hours), 1)) * 100
    fig = add_stats_overlay(fig, station_frame[value_column], completeness_pct)
    if station_frame[value_column].notna().any():
        peak_row = station_frame.loc[station_frame[value_column].idxmax()]
        fig = add_peak_annotation(fig, peak_row["datetime"], peak_row[value_column], "Peak")
    return fig


def create_monthly_daily_average_figure(
    data: pd.DataFrame,
    value_column: str,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
) -> go.Figure | None:
    full_days = pd.date_range(start=month_start.floor("D"), end=month_end.floor("D"), freq="D")
    daily = resample_to_expected_index(data, value_column, full_days, "D", "date")
    if daily.empty:
        return None

    fig = px.line(
        daily,
        x="date",
        y=value_column,
        color="station_label",
        markers=True,
        color_discrete_sequence=APP_CONFIG["colorway"],
    )
    fig.update_traces(line={"width": 3}, marker={"size": 7})
    for trace in fig.data:
        if getattr(trace, "name", None):
            trace.name = compact_station_label(trace.name)

    fig = chart_template(
        fig,
        f"{display_label(value_column)} Monthly Trend (Daily Average)",
        axis_label(value_column),
        "Date",
        height=430,
        subtitle=format_time_window(month_start, month_end, "Daily average"),
    )
    fig = datetime_axis(fig, tickformat="%d %b", hoverformat="%Y-%m-%d", dtick="D1", tickangle=-45)
    fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra>%{fullData.name}</extra>")

    base_station = daily["station_label"].iloc[0]
    station_daily = daily[daily["station_label"] == base_station]
    completeness_pct = (station_daily[value_column].notna().sum() / max(len(full_days), 1)) * 100
    fig = add_stats_overlay(fig, station_daily[value_column], completeness_pct)
    if station_daily[value_column].notna().any():
        peak_row = station_daily.loc[station_daily[value_column].idxmax()]
        fig = add_peak_annotation(fig, peak_row["date"], peak_row[value_column], "Peak")
    return fig


def create_yearly_month_average_figure(
    data: pd.DataFrame,
    value_column: str,
    year_start: pd.Timestamp,
    year_end: pd.Timestamp,
) -> go.Figure | None:
    full_months = pd.date_range(start=year_start.floor("D"), end=year_end.floor("D"), freq="MS")
    monthly = resample_to_expected_index(data, value_column, full_months, "MS", "month")
    if monthly.empty:
        return None

    fig = px.line(
        monthly,
        x="month",
        y=value_column,
        color="station_label",
        markers=True,
        color_discrete_sequence=APP_CONFIG["colorway"],
    )
    fig.update_traces(line={"width": 3}, marker={"size": 8})
    for trace in fig.data:
        if getattr(trace, "name", None):
            trace.name = compact_station_label(trace.name)

    fig = chart_template(
        fig,
        f"{display_label(value_column)} Yearly Variation",
        axis_label(value_column),
        "Month",
        height=430,
        subtitle=format_time_window(year_start, year_end, "Monthly average"),
    )
    fig.update_xaxes(dtick="M1", tickformat="%b %Y", hoverformat="%Y-%m", tickangle=-45)
    fig.update_traces(hovertemplate="%{x|%Y-%m}<br>%{y:.2f}<extra>%{fullData.name}</extra>")

    base_station = monthly["station_label"].iloc[0]
    station_monthly = monthly[monthly["station_label"] == base_station]
    completeness_pct = (station_monthly[value_column].notna().sum() / max(len(full_months), 1)) * 100
    fig = add_stats_overlay(fig, station_monthly[value_column], completeness_pct)
    if station_monthly[value_column].notna().any():
        peak_row = station_monthly.loc[station_monthly[value_column].idxmax()]
        fig = add_peak_annotation(fig, peak_row["month"], peak_row[value_column], "Peak")
    return fig


def create_recent_figure(data: pd.DataFrame, value_column: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> go.Figure | None:
    if is_single_day_selection(start_ts, end_ts):
        return create_single_day_figure(data, value_column, start_ts, end_ts)
    if is_three_day_selection(start_ts, end_ts):
        return create_three_day_figure(data, value_column, start_ts, end_ts)
    if is_fifteen_day_selection(start_ts, end_ts):
        return create_fifteen_day_figure(data, value_column)
    if is_full_month_selection(start_ts, end_ts):
        return create_monthly_hourly_figure(data, value_column, start_ts, end_ts)
    if is_full_year_selection(start_ts, end_ts):
        return create_yearly_month_average_figure(data, value_column, start_ts, end_ts)

    frame = select_value_frame(data, value_column)
    if frame.empty:
        return None

    fig = px.line(
        frame,
        x="datetime",
        y=value_column,
        color="station_label",
        color_discrete_sequence=APP_CONFIG["colorway"],
    )
    fig.update_traces(line={"width": 2.8})
    for trace in fig.data:
        if getattr(trace, "name", None):
            trace.name = compact_station_label(trace.name)

    fig = chart_template(
        fig,
        f"{display_label(value_column)} Selected Window Continuous View",
        axis_label(value_column),
        "Datetime",
        height=420,
        subtitle=format_time_window(start_ts, end_ts, "Continuous timestamps"),
    )
    fig = apply_temporal_axis(fig, "Hourly", start_ts, end_ts)
    fig.update_traces(hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}<extra>%{fullData.name}</extra>")
    fig.update_layout(legend_title_text="")
    return fig


def create_timeseries_figure(data: pd.DataFrame, value_column: str, aggregation: str, rolling_window: int) -> go.Figure | None:
    frame = aggregate_series(data, value_column, aggregation)
    if frame.empty:
        return None

    frame["rolling"] = (
        frame.groupby("station_label", observed=True)[value_column]
        .transform(lambda s: s.rolling(rolling_window, min_periods=1).mean())
    )

    fig = go.Figure()
    station_order = list(frame["station_label"].drop_duplicates())
    use_webgl = prefers_webgl(len(frame))
    trace_cls = go.Scattergl if use_webgl else go.Scatter

    for idx, station in enumerate(station_order):
        station_frame = frame[frame["station_label"] == station]
        color = APP_CONFIG["colorway"][idx % len(APP_CONFIG["colorway"])]
        show_markers = len(station_frame) <= 500
        base_mode = "lines+markers" if show_markers else "lines"

        fig.add_trace(
            trace_cls(
                x=station_frame["period"],
                y=station_frame[value_column],
                mode=base_mode,
                line={"color": color, "width": 2.5},
                marker={"size": 6} if show_markers else None,
                name=compact_station_label(station),
                legendgroup=station,
                opacity=0.30 if rolling_window > 1 else 0.95,
            )
        )

        if rolling_window > 1:
            fig.add_trace(
                trace_cls(
                    x=station_frame["period"],
                    y=station_frame["rolling"],
                    mode="lines",
                    line={"color": color, "width": 3},
                    name=f"{compact_station_label(station)} rolling",
                    legendgroup=station,
                    showlegend=False,
                )
            )

        threshold = station_frame[value_column].quantile(0.90)
        outliers = station_frame[station_frame[value_column] >= threshold]
        if not outliers.empty and len(station_frame) <= 2500:
            fig.add_trace(
                trace_cls(
                    x=outliers["period"],
                    y=outliers[value_column],
                    mode="markers",
                    marker={"size": 7, "color": color, "symbol": "circle-open"},
                    name=f"{compact_station_label(station)} > p90",
                    legendgroup=station,
                    showlegend=False,
                )
            )

    fig = chart_template(
        fig,
        f"{aggregation} Station Comparison - {display_label(value_column)}",
        axis_label(value_column),
        aggregation,
        height=470,
        subtitle=format_window_from_data(data, f"{aggregation} aggregation"),
    )

    period_start = frame["period"].min()
    period_end = frame["period"].max()
    fig = apply_temporal_axis(fig, aggregation, period_start, period_end)
    fig.update_traces(hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}<extra>%{fullData.name}</extra>")

    base_station = station_order[0]
    station_data = frame[frame["station_label"] == base_station]
    fig = add_stats_overlay(fig, station_data[value_column], 100.0)
    return fig


def create_overlay_figure(
    pollutant_df: pd.DataFrame,
    meteorology_df: pd.DataFrame,
    pollutant_column: str,
    meteorology_column: str,
    aggregation: str,
    rolling_window: int,
    title: str,
) -> go.Figure | None:
    pollutant_frame = aggregate_series(pollutant_df, pollutant_column, aggregation)
    meteorology_temp = meteorology_df.copy()
    if "station_label" not in meteorology_temp.columns:
        meteorology_temp["station_label"] = "Regional ERA5"
    meteorology_frame = aggregate_series(meteorology_temp, meteorology_column, aggregation)

    if pollutant_frame.empty or meteorology_frame.empty:
        return None

    pollutant_frame["rolling"] = (
        pollutant_frame.groupby("station_label", observed=True)[pollutant_column]
        .transform(lambda s: s.rolling(rolling_window, min_periods=1).mean())
    )
    meteorology_frame["rolling"] = (
        meteorology_frame.groupby("station_label", observed=True)[meteorology_column]
        .transform(lambda s: s.rolling(rolling_window, min_periods=1).mean())
    )

    use_webgl = prefers_webgl(max(len(pollutant_frame), len(meteorology_frame)))
    trace_cls = go.Scattergl if use_webgl else go.Scatter

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for idx, station in enumerate(pollutant_frame["station_label"].drop_duplicates()):
        station_frame = pollutant_frame[pollutant_frame["station_label"] == station]
        color = APP_CONFIG["colorway"][idx % len(APP_CONFIG["colorway"])]
        fig.add_trace(
            trace_cls(
                x=station_frame["period"],
                y=station_frame["rolling"] if rolling_window > 1 else station_frame[pollutant_column],
                mode="lines",
                line={"color": color, "width": 3},
                fill="tozeroy",
                opacity=0.24,
                name=compact_station_label(station),
            ),
            secondary_y=False,
        )

    met_station = meteorology_frame["station_label"].iloc[0]
    met_frame = meteorology_frame[meteorology_frame["station_label"] == met_station]
    fig.add_trace(
        trace_cls(
            x=met_frame["period"],
            y=met_frame["rolling"] if rolling_window > 1 else met_frame[meteorology_column],
            mode="lines",
            line={"color": "#0f172a", "width": 3, "dash": "dot"},
            name=display_label(meteorology_column),
        ),
        secondary_y=True,
    )

    fig = chart_template(
        fig,
        title,
        axis_label(pollutant_column),
        aggregation,
        height=460,
        subtitle=format_window_from_data(pollutant_df, f"{aggregation} overlay"),
    )
    fig.update_yaxes(title_text=axis_label(pollutant_column), secondary_y=False)
    fig.update_yaxes(title_text=axis_label(meteorology_column), secondary_y=True)

    period_start = pollutant_frame["period"].min()
    period_end = pollutant_frame["period"].max()
    fig = apply_temporal_axis(fig, aggregation, period_start, period_end)
    fig.update_traces(hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}<extra>%{fullData.name}</extra>")
    fig.update_layout(legend_title_text="")
    return fig


def create_boxplot_figure(data: pd.DataFrame, value_column: str) -> go.Figure | None:
    frame = select_value_frame(data, value_column)
    if frame.empty:
        return None
    fig = px.box(
        frame,
        x="station_label",
        y=value_column,
        color="station_label",
        color_discrete_sequence=APP_CONFIG["colorway"],
        points=False,
    )
    for trace in fig.data:
        if getattr(trace, "name", None):
            trace.name = compact_station_label(trace.name)
    return chart_template(
        fig,
        f"Station Distribution Boxplot - {display_label(value_column)}",
        axis_label(value_column),
        "Station",
        height=430,
        subtitle=format_window_from_data(data, "Current filter window"),
    )


def create_histogram_figure(data: pd.DataFrame, value_column: str) -> go.Figure | None:
    frame = select_value_frame(data, value_column)
    if frame.empty:
        return None
    fig = px.histogram(
        frame,
        x=value_column,
        color="station_label",
        nbins=40,
        opacity=0.65,
        marginal="rug",
        barmode="overlay",
        color_discrete_sequence=APP_CONFIG["colorway"],
    )
    for trace in fig.data:
        if getattr(trace, "name", None):
            trace.name = compact_station_label(trace.name)
    return chart_template(
        fig,
        f"Histogram - {display_label(value_column)}",
        axis_label(value_column),
        axis_label(value_column),
        height=430,
        subtitle=format_window_from_data(data, "Current filter window"),
    )


def create_scatter_figure(combined_df: pd.DataFrame, pollutant_column: str, meteorology_column: str) -> go.Figure | None:
    if combined_df.empty:
        return None
    frame = combined_df[["datetime", "station_label", pollutant_column, meteorology_column]].dropna()
    frame = sample_scatter_data(frame)
    if frame.empty:
        return None

    fig = px.scatter(
        frame,
        x=meteorology_column,
        y=pollutant_column,
        color="station_label",
        opacity=0.65,
        hover_data={"datetime": True},
        color_discrete_sequence=APP_CONFIG["colorway"],
    )
    for trace in fig.data:
        if getattr(trace, "name", None):
            trace.name = compact_station_label(trace.name)

    if len(frame) >= 2:
        x_vals = frame[meteorology_column].to_numpy()
        y_vals = frame[pollutant_column].to_numpy()
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        line_x = np.linspace(np.nanmin(x_vals), np.nanmax(x_vals), 200)
        line_y = slope * line_x + intercept
        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                line={"color": "#0f172a", "width": 3, "dash": "dash"},
                name="Regression line",
            )
        )

    return chart_template(
        fig,
        f"Relationship Explorer - {display_label(pollutant_column)} vs {display_label(meteorology_column)}",
        axis_label(pollutant_column),
        axis_label(meteorology_column),
        height=480,
        subtitle=format_window_from_data(combined_df, "Timestamped paired observations"),
    )


def create_correlation_heatmap(
    combined_df: pd.DataFrame,
    selected_pollutant: str,
    selected_meteorology: str,
    available_pollutants: list[str],
    available_meteorology: list[str],
) -> go.Figure | None:
    candidates = [selected_pollutant, selected_meteorology]
    candidates.extend([c for c in available_pollutants if c != selected_pollutant])
    candidates.extend([c for c in available_meteorology if c != selected_meteorology])

    usable = [
        c for c in candidates
        if c in combined_df.columns and combined_df[c].notna().sum() >= 24
    ]
    selected_cols = []
    for c in usable:
        if c not in selected_cols:
            selected_cols.append(c)
        if len(selected_cols) == 8:
            break

    if len(selected_cols) < 2:
        return None

    corr = combined_df[selected_cols].corr(numeric_only=True)
    labels = [display_label(c) for c in corr.columns]
    corr.columns = labels
    corr.index = labels

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(coloraxis_colorbar={"title": "Corr"})
    return chart_template(
        fig,
        "Correlation Heatmap",
        "",
        "",
        height=440,
        subtitle=format_window_from_data(combined_df, "Correlation window"),
    )


def quality_granularity(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> str:
    span_days = max((end_ts - start_ts).days, 1)
    return "Daily" if span_days <= 120 else "Monthly"


def completeness_matrix(data: pd.DataFrame, value_column: str, station_order: list[str], start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> tuple[pd.DataFrame, str]:
    frame = select_value_frame(data, value_column)
    granularity = quality_granularity(start_ts, end_ts)

    full_hours = pd.date_range(start=start_ts.floor("h"), end=end_ts.floor("h"), freq="h")
    if granularity == "Daily":
        full_periods = pd.to_datetime(full_hours.floor("D").unique())
        frame["bucket"] = frame["datetime"].dt.floor("D")
    else:
        full_periods = pd.to_datetime(full_hours.to_period("M").to_timestamp().unique())
        frame["bucket"] = frame["datetime"].dt.to_period("M").dt.to_timestamp()

    presence = (
        frame.groupby(["station_label", "bucket"], observed=True)
        .size()
        .reset_index(name="count")
    )
    presence["available"] = 1

    full_index = pd.MultiIndex.from_product([station_order, full_periods], names=["station_label", "bucket"])
    complete = (
        presence.set_index(["station_label", "bucket"])
        .reindex(full_index)
        .fillna(0)
        .reset_index()
    )
    matrix = complete.pivot(index="station_label", columns="bucket", values="available")
    return matrix, granularity


def create_completeness_figure(data: pd.DataFrame, value_column: str, station_order: list[str], start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> tuple[go.Figure | None, str]:
    matrix, granularity = completeness_matrix(data, value_column, station_order, start_ts, end_ts)
    if matrix.empty:
        return None, granularity

    matrix.index = [compact_station_label(i) for i in matrix.index]
    fig = px.imshow(
        matrix,
        aspect="auto",
        color_continuous_scale=[[0.0, "#e2e8f0"], [1.0, "#1d4ed8"]],
        labels={"x": granularity, "y": "Station", "color": "Available"},
    )
    fig.update_layout(coloraxis_colorbar={"title": "Data"})
    fig = chart_template(
        fig,
        f"Completeness Matrix - {display_label(value_column)}",
        "",
        granularity,
        height=380,
        subtitle=format_time_window(start_ts, end_ts, f"{granularity} availability"),
    )
    return fig, granularity


def expected_points(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> int:
    return int((end_ts.floor("h") - start_ts.floor("h")).total_seconds() / 3600) + 1


def quality_summary_table(data: pd.DataFrame, value_column: str, station_order: list[str], start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    frame = select_value_frame(data, value_column)
    if frame.empty:
        return pd.DataFrame()

    summary = quantile_summary(data, value_column, station_order)
    expected = max(expected_points(start_ts, end_ts), 1)
    summary["valid_obs"] = summary["observations"]
    summary["availability_pct"] = (summary["valid_obs"] / expected) * 100
    summary["missing_pct"] = 100 - summary["availability_pct"]
    summary = summary.rename(columns={"minimum": "min", "maximum": "max"})
    keep = ["station_label", "valid_obs", "availability_pct", "missing_pct", "mean", "std", "min", "Q1", "median", "Q3", "max"]
    return summary[keep]


def render_chart(fig: go.Figure | None, empty_message: str) -> None:
    if fig is None:
        st.info(empty_message)
    else:
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def render_kpi_card(label: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(station_meta_df: pd.DataFrame, min_ts: pd.Timestamp, max_ts: pd.Timestamp) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-eyebrow">Interactive Research Dashboard</div>
            <h1>{APP_CONFIG["app_title"]}</h1>
            <p>{APP_CONFIG["app_subtitle"]}</p>
            <div class="hero-badges">
                <div class="badge">{len(station_meta_df)} stations loaded</div>
                <div class="badge">{min_ts:%Y-%m-%d} to {max_ts:%Y-%m-%d}</div>
                <div class="badge">1 day → hourly | 3 days → 3 panels | 15 days → daily avg | month → monthly | year → yearly</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_date_controls(min_ts: pd.Timestamp, max_ts: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    st.markdown("### 📅 Date Selection")

    from_date = st.date_input(
        "From Date",
        value=st.session_state.get("from_date", min_ts.date()),
        min_value=min_ts.date(),
        max_value=max_ts.date(),
        key="from_date",
    )
    to_date = st.date_input(
        "To Date",
        value=st.session_state.get("to_date", max_ts.date()),
        min_value=min_ts.date(),
        max_value=max_ts.date(),
        key="to_date",
    )

    start_ts = pd.Timestamp(from_date).floor("D")
    end_ts = pd.Timestamp(to_date).floor("D") + pd.Timedelta(hours=23)

    if start_ts > end_ts:
        st.warning("From Date cannot be after To Date. Using the same day for both.")
        start_ts = pd.Timestamp(from_date).floor("D")
        end_ts = start_ts + pd.Timedelta(hours=23)

    return start_ts, end_ts


def sidebar_controls(
    station_options: list[str],
    available_pollutants: list[str],
    available_meteorology: list[str],
    defaults: dict[str, object],
    min_ts: pd.Timestamp,
    max_ts: pd.Timestamp,
) -> dict[str, object]:
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-card"><strong>Control Panel</strong><br><span style="color:#cbd5e1;">Select stations, variables, date window, aggregation, and export data.</span></div>',
            unsafe_allow_html=True,
        )

        station_search = st.text_input("🔍 Search station", key="station_search")
        visible_station_options = (
            [s for s in station_options if station_search.lower() in s.lower()]
            if station_search else station_options
        )

        selected_stations = st.multiselect(
            "✅ Select stations",
            options=visible_station_options,
            default=st.session_state.get("selected_stations", defaults["selected_stations"]),
            key="selected_stations",
        )

        selected_pollutant = st.selectbox(
            "🧪 Pollutant",
            options=[None] + available_pollutants,
            index=([None] + available_pollutants).index(st.session_state.get("selected_pollutant", defaults["selected_pollutant"]))
            if st.session_state.get("selected_pollutant", defaults["selected_pollutant"]) in ([None] + available_pollutants)
            else 0,
            format_func=lambda x: "None" if x is None else display_label(x),
            key="selected_pollutant",
        )

        selected_meteorology = st.selectbox(
            "🌪 Meteorology variable",
            options=[None] + available_meteorology,
            index=([None] + available_meteorology).index(st.session_state.get("selected_meteorology", defaults["selected_meteorology"]))
            if st.session_state.get("selected_meteorology", defaults["selected_meteorology"]) in ([None] + available_meteorology)
            else 0,
            format_func=lambda x: "None" if x is None else display_label(x),
            key="selected_meteorology",
        )

        aggregation = st.selectbox(
            "📊 Aggregation",
            options=list(AGGREGATION_OPTIONS.keys()),
            index=list(AGGREGATION_OPTIONS.keys()).index(st.session_state.get("aggregation", defaults["aggregation"])),
            key="aggregation",
        )

        rolling_window = st.slider(
            "🎚 Rolling average window",
            min_value=1,
            max_value=14,
            value=int(st.session_state.get("rolling_window", defaults["rolling_window"])),
            key="rolling_window",
        )

        st.caption("1 day → hourly 00:00–23:00 | 3 days → 3 panels | 15 days → daily avg | full month → monthly trend | full year → yearly monthly avg")

        start_ts, end_ts = sidebar_date_controls(min_ts, max_ts)

        if st.button("Reset Filters", use_container_width=True):
            reset_filters(defaults)
            st.rerun()

        return {
            "selected_stations": selected_stations,
            "selected_pollutant": selected_pollutant,
            "selected_meteorology": selected_meteorology,
            "aggregation": aggregation,
            "rolling_window": rolling_window,
            "start_ts": start_ts,
            "end_ts": end_ts,
        }


def peak_metric(data: pd.DataFrame, value_column: str | None) -> tuple[str, str]:
    if value_column is None:
        return "None", "No variable selected"
    frame = select_value_frame(data, value_column)
    if frame.empty:
        return "No data", "No valid values in current selection"
    peak_row = frame.loc[frame[value_column].idxmax()]
    return (
        f"{peak_row[value_column]:,.2f}",
        f"{peak_row['datetime']:%Y-%m-%d %H:%M} • {compact_station_label(peak_row['station_label'])}",
    )


def render_top_kpis(
    selected_stations: list[str],
    selected_pollutant: str | None,
    selected_meteorology: str | None,
    peak_value: str,
    peak_caption: str,
) -> None:
    cols = st.columns(4)
    with cols[0]:
        render_kpi_card(
            "Selected Stations",
            ", ".join([compact_station_label(s) for s in selected_stations[:2]]) if selected_stations else "None",
            f"{len(selected_stations)} station(s) active",
        )
    with cols[1]:
        render_kpi_card("Selected Pollutant", display_label(selected_pollutant), "Primary pollutant selection")
    with cols[2]:
        render_kpi_card("Peak Value", peak_value, peak_caption)
    with cols[3]:
        render_kpi_card("Meteorology Variable", display_label(selected_meteorology), "Context or relationship variable")


def render_quantile_table(data: pd.DataFrame, value_column: str, station_order: list[str]) -> None:
    table = quantile_summary(data, value_column, station_order)
    if table.empty:
        st.info("No quantile summary is available for the active filters.")
        return
    styled = table.style.format(
        {
            "observations": "{:,.0f}",
            "mean": "{:,.2f}",
            "std": "{:,.2f}",
            "minimum": "{:,.2f}",
            "Q1": "{:,.2f}",
            "median": "{:,.2f}",
            "Q3": "{:,.2f}",
            "maximum": "{:,.2f}",
        }
    )
    st.dataframe(styled, use_container_width=True)


def render_quality_block(
    title: str,
    data: pd.DataFrame,
    value_column: str,
    station_order: list[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    st.markdown(f"#### {title}")
    summary = quality_summary_table(data, value_column, station_order, start_ts, end_ts)
    availability = float(summary["availability_pct"].mean()) if not summary.empty else 0.0
    missing = float(summary["missing_pct"].mean()) if not summary.empty else 0.0
    valid_obs = int(summary["valid_obs"].sum()) if not summary.empty else 0

    cols = st.columns(3)
    with cols[0]:
        render_kpi_card("Availability", f"{availability:.1f}%", "Mean coverage across selected stations")
    with cols[1]:
        render_kpi_card("Missing", f"{missing:.1f}%", "Average missing share across selected stations")
    with cols[2]:
        render_kpi_card("Valid Observations", f"{valid_obs:,}", f"Usable {display_label(value_column)} records")

    heatmap, granularity = create_completeness_figure(data, value_column, station_order, start_ts, end_ts)
    render_chart(heatmap, f"No completeness matrix is available for {display_label(value_column)}.")
    st.caption(f"Availability matrix is summarized at the {granularity.lower()} level.")

    styled = summary.style.format(
        {
            "valid_obs": "{:,.0f}",
            "availability_pct": "{:,.1f}%",
            "missing_pct": "{:,.1f}%",
            "mean": "{:,.2f}",
            "std": "{:,.2f}",
            "min": "{:,.2f}",
            "Q1": "{:,.2f}",
            "median": "{:,.2f}",
            "Q3": "{:,.2f}",
            "max": "{:,.2f}",
        }
    )
    st.dataframe(styled, width="stretch")
    return summary


def build_quantile_window_summary(data: pd.DataFrame, value_column: str) -> pd.DataFrame:
    frame = select_value_frame(data, value_column)
    if frame.empty:
        return pd.DataFrame()
    q1 = frame[value_column].quantile(0.25)
    median = frame[value_column].quantile(0.50)
    q3 = frame[value_column].quantile(0.75)
    total_count = len(frame)
    below_q1 = (frame[value_column] < q1).sum()
    q1_to_median = ((frame[value_column] >= q1) & (frame[value_column] < median)).sum()
    median_to_q3 = ((frame[value_column] >= median) & (frame[value_column] < q3)).sum()
    above_q3 = (frame[value_column] >= q3).sum()
    return pd.DataFrame({
        "window": ["Below Q1", "Q1 to Median", "Median to Q3", "Above Q3"],
        "count": [below_q1, q1_to_median, median_to_q3, above_q3],
        "percentage": [
            (below_q1 / total_count) * 100 if total_count > 0 else 0,
            (q1_to_median / total_count) * 100 if total_count > 0 else 0,
            (median_to_q3 / total_count) * 100 if total_count > 0 else 0,
            (above_q3 / total_count) * 100 if total_count > 0 else 0,
        ]
    })


def build_station_quantile_comparison(data: pd.DataFrame, value_column: str, selected_stations: list[str]) -> pd.DataFrame:
    if len(selected_stations) <= 1:
        return pd.DataFrame()
    summaries = []
    for station in selected_stations:
        station_data = data[data["station_label"] == station]
        if not station_data.empty:
            summary = build_quantile_window_summary(station_data, value_column)
            if not summary.empty:
                summary["station"] = compact_station_label(station)
                summaries.append(summary)
    if summaries:
        return pd.concat(summaries, ignore_index=True)
    return pd.DataFrame()


def create_quantile_window_bar(data: pd.DataFrame, value_column: str) -> go.Figure | None:
    analysis = build_quantile_window_summary(data, value_column)
    if analysis.empty:
        return None
    fig = px.bar(
        analysis,
        x="window",
        y="percentage",
        color="window",
        color_discrete_sequence=APP_CONFIG["colorway"],
        text="percentage",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(showlegend=False)
    return chart_template(
        fig,
        f"Quantile Window Distribution - {display_label(value_column)}",
        "Percentage (%)",
        "Quantile Window",
        height=400,
        subtitle="Distribution of values across quantile ranges",
    )


def create_station_quantile_chart(data: pd.DataFrame, value_column: str, selected_stations: list[str]) -> go.Figure | None:
    comparison = build_station_quantile_comparison(data, value_column, selected_stations)
    if comparison.empty:
        return None
    fig = px.bar(
        comparison,
        x="window",
        y="percentage",
        color="station",
        barmode="group",
        color_discrete_sequence=APP_CONFIG["colorway"],
        text="percentage",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    return chart_template(
        fig,
        f"Station-wise Quantile Window Comparison - {display_label(value_column)}",
        "Percentage (%)",
        "Quantile Window",
        height=400,
        subtitle="Comparison across selected stations",
    )


def render_quantile_window_section(
    data: pd.DataFrame,
    value_column: str,
    selected_stations: list[str],
) -> None:
    if data.empty or value_column not in data.columns:
        st.warning("Insufficient data for quantile window analysis.")
        return

    frame = select_value_frame(data, value_column)
    if frame.empty:
        st.warning("No valid data available for the selected variable.")
        return

    q1 = frame[value_column].quantile(0.25)
    median = frame[value_column].quantile(0.50)
    q3 = frame[value_column].quantile(0.75)
    valid_count = len(frame)

    cols = st.columns(4)
    with cols[0]:
        render_kpi_card("Q1 (25th Percentile)", f"{q1:.2f}", "Lower quartile")
    with cols[1]:
        render_kpi_card("Median (50th)", f"{median:.2f}", "Middle value")
    with cols[2]:
        render_kpi_card("Q3 (75th Percentile)", f"{q3:.2f}", "Upper quartile")
    with cols[3]:
        render_kpi_card("Valid Count", f"{valid_count:,}", "Total observations")

    analysis = build_quantile_window_summary(data, value_column)
    if not analysis.empty:
        styled = analysis.style.format({
            "count": "{:,.0f}",
            "percentage": "{:.1f}%",
        })
        st.markdown("#### Summary Table")
        st.dataframe(styled, width="stretch")

        render_chart(create_quantile_window_bar(data, value_column), "No bar chart available.")

    if len(selected_stations) > 1:
        st.markdown("#### Station-wise Comparison")
        comparison = build_station_quantile_comparison(data, value_column, selected_stations)
        if not comparison.empty:
            st.dataframe(comparison, width="stretch")
            render_chart(create_station_quantile_chart(data, value_column, selected_stations), "No comparison chart available.")


def render_quantile_tab(
    pollutant_filtered,
    meteorology_expanded,
    selected_pollutant,
    selected_meteorology,
    selected_stations,
    mode,
):
    st.markdown(
        '<p class="tab-note">Analyze the distribution of values within quantile windows for the selected variable.</p>',
        unsafe_allow_html=True,
    )
    if mode == "empty":
        st.warning("Select a pollutant or meteorology variable to analyze quantile windows.")
    if selected_pollutant:
        st.subheader(display_label(selected_pollutant))
        render_quantile_window_section(pollutant_filtered, selected_pollutant, selected_stations)
    if selected_meteorology:
        st.subheader(display_label(selected_meteorology))
        render_quantile_window_section(meteorology_expanded, selected_meteorology, selected_stations)


def render_epa_map_tab() -> None:
    st.markdown(
        '<p class="tab-note">Embedded EPA ArcGIS viewer for side-by-side spatial context with the dashboard analytics.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'[Open the EPA map in a new tab]({APP_CONFIG["epa_map_url"]})',
        unsafe_allow_html=True,
    )
    components.iframe(APP_CONFIG["epa_map_url"], height=900, scrolling=True)


def build_station_network_table(station_meta_df: pd.DataFrame) -> pd.DataFrame:
    network = station_meta_df.copy()
    network["site"] = network["site"].astype(str)
    network["full_aqs_id"] = network["site"].map(APP_CONFIG["station_aqs_ids"]).fillna("Not provided")
    network["cluster"] = network["site"].map(APP_CONFIG["station_clusters"]).fillna("Unclassified")
    keep = ["station_label", "site", "full_aqs_id", "cluster", "state_code", "county_code", "records", "start", "end"]
    network = network[keep].rename(
        columns={
            "station_label": "Station",
            "site": "Site Code",
            "full_aqs_id": "Full AQS ID",
            "cluster": "Cluster",
            "state_code": "State",
            "county_code": "County",
            "records": "Records",
            "start": "Start",
            "end": "End",
        }
    )
    return network


def build_distance_matrix(site_codes: list[str]) -> pd.DataFrame:
    ordered_sites = [site for site in APP_CONFIG["station_aqs_ids"] if site in site_codes]
    for site in site_codes:
        if site not in ordered_sites:
            ordered_sites.append(site)

    matrix = pd.DataFrame(index=ordered_sites, columns=ordered_sites, dtype="object")
    for site in ordered_sites:
        matrix.loc[site, site] = "0 km"

    for (left, right), distance_miles in APP_CONFIG["distance_pairs_miles"].items():
        if left in matrix.index and right in matrix.columns:
            value = f"~{distance_miles} miles"
            matrix.loc[left, right] = value
            matrix.loc[right, left] = value

    matrix = matrix.fillna("-")
    matrix.index = [f"Site {site}" for site in matrix.index]
    matrix.columns = [f"Site {site}" for site in matrix.columns]
    return matrix


def render_station_network_tab(station_meta_df: pd.DataFrame) -> None:
    st.markdown(
        '<p class="tab-note">Manual Bexar County station metadata, AQS IDs, spatial clustering, and approximate inter-station distances for research context.</p>',
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    with cols[0]:
        render_kpi_card("Within Cluster", APP_CONFIG["within_cluster_range_miles"], "Typical spacing for stations in the same cluster")
    with cols[1]:
        render_kpi_card("Across City", APP_CONFIG["across_city_range_miles"], "Typical spacing between the northwest and southeast groups")
    with cols[2]:
        render_kpi_card("City-Wide Spread", APP_CONFIG["city_wide_spread_miles"], "Approximate monitor spread across San Antonio")

    st.markdown(
        "Research insight: stations 32, 52, and 59 form the northwest/central cluster, while 1069, 1080, and 1087 form the southeast cluster.",
    )
    st.markdown(
        "Research insight: nearby stations are roughly 3-9 miles apart, while cross-city comparisons are often 19-25 miles apart.",
    )

    network_table = build_station_network_table(station_meta_df)
    st.markdown("#### Station Metadata")
    st.dataframe(
        network_table.style.format({"Records": "{:,.0f}", "Start": "{:%Y-%m-%d}", "End": "{:%Y-%m-%d}"}),
        width="stretch",
    )

    st.markdown("#### Approximate Distance Matrix")
    distance_matrix = build_distance_matrix(network_table["Site Code"].astype(str).tolist())
    st.dataframe(distance_matrix, width="stretch")





def prepare_anomaly_dataset(data: pd.DataFrame, value_column: str) -> pd.DataFrame:
    frame = select_value_frame(data, value_column)
    if frame.empty:
        return frame
    return frame.sort_values(["station_label", "datetime"]).reset_index(drop=True)


def detect_rolling_zscore_anomalies(
    data: pd.DataFrame,
    value_column: str,
    window: int = 24,
    threshold: float = 3.0,
) -> pd.DataFrame:
    frame = prepare_anomaly_dataset(data, value_column)
    if frame.empty:
        return frame

    pieces = []
    for station, station_frame in frame.groupby("station_label", observed=True):
        station_frame = station_frame.copy().sort_values("datetime")
        min_periods = max(3, min(window, 6))
        rolling_mean = station_frame[value_column].rolling(window=window, min_periods=min_periods).mean()
        rolling_std = station_frame[value_column].rolling(window=window, min_periods=min_periods).std().replace(0, np.nan)
        zscore = (station_frame[value_column] - rolling_mean) / rolling_std
        station_frame["anomaly_score"] = zscore
        station_frame["anomaly_flag"] = zscore.abs() > threshold
        station_frame["method"] = "Rolling Z-Score"
        pieces.append(station_frame)

    return pd.concat(pieces, ignore_index=True)


def detect_iqr_anomalies(
    data: pd.DataFrame,
    value_column: str,
    multiplier: float = 1.5,
) -> pd.DataFrame:
    frame = prepare_anomaly_dataset(data, value_column)
    if frame.empty:
        return frame

    pieces = []
    for station, station_frame in frame.groupby("station_label", observed=True):
        station_frame = station_frame.copy()
        q1 = station_frame[value_column].quantile(0.25)
        q3 = station_frame[value_column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        station_frame["lower_bound"] = lower
        station_frame["upper_bound"] = upper
        station_frame["anomaly_score"] = np.where(
            station_frame[value_column] > upper,
            station_frame[value_column] - upper,
            np.where(station_frame[value_column] < lower, lower - station_frame[value_column], 0.0),
        )
        station_frame["anomaly_flag"] = (station_frame[value_column] < lower) | (station_frame[value_column] > upper)
        station_frame["method"] = "IQR"
        pieces.append(station_frame)

    return pd.concat(pieces, ignore_index=True)


def detect_isolation_forest_anomalies(
    data: pd.DataFrame,
    value_column: str,
    contamination: float = 0.02,
) -> pd.DataFrame:
    frame = prepare_anomaly_dataset(data, value_column)
    if frame.empty:
        return frame

    try:
        from sklearn.ensemble import IsolationForest
    except ModuleNotFoundError:
        frame["anomaly_score"] = np.nan
        frame["anomaly_flag"] = False
        frame["method"] = "Isolation Forest"
        frame["status_message"] = "scikit-learn is not installed. Run: pip install scikit-learn"
        return frame

    excluded = {"datetime", "station_label", APP_CONFIG["state_col"], APP_CONFIG["county_col"], APP_CONFIG["site_col"], "station_file"}
    feature_candidates = [
        c for c in data.columns
        if c not in excluded and c != value_column and pd.api.types.is_numeric_dtype(data[c])
    ]

    pieces = []
    for station, station_frame in frame.groupby("station_label", observed=True):
        station_frame = station_frame.copy().sort_values("datetime")
        if "station_label" in data.columns:
            original_station_frame = data[data["station_label"] == station].copy().sort_values("datetime")
        else:
            original_station_frame = station_frame.copy()

        model_frame = pd.DataFrame({
            value_column: pd.to_numeric(original_station_frame[value_column], errors="coerce"),
            "hour": pd.to_datetime(original_station_frame["datetime"]).dt.hour,
            "dayofweek": pd.to_datetime(original_station_frame["datetime"]).dt.dayofweek,
            "month": pd.to_datetime(original_station_frame["datetime"]).dt.month,
        })
        for col in feature_candidates:
            if col in original_station_frame.columns:
                model_frame[col] = pd.to_numeric(original_station_frame[col], errors="coerce")

        valid_mask = model_frame.notna().all(axis=1)
        scores = pd.Series(np.nan, index=original_station_frame.index, dtype=float)
        flags = pd.Series(False, index=original_station_frame.index, dtype=bool)

        if int(valid_mask.sum()) >= 10:
            model = IsolationForest(
                contamination=min(max(contamination, 0.001), 0.25),
                random_state=42,
                n_estimators=200,
            )
            fitted = model.fit(model_frame.loc[valid_mask])
            decision = fitted.decision_function(model_frame.loc[valid_mask])
            preds = fitted.predict(model_frame.loc[valid_mask])
            scores.loc[valid_mask] = -decision
            flags.loc[valid_mask] = preds == -1

        station_result = original_station_frame[["datetime", "station_label", value_column]].copy()
        station_result["anomaly_score"] = scores.to_numpy()
        station_result["anomaly_flag"] = flags.to_numpy()
        station_result["method"] = "Isolation Forest"
        pieces.append(station_result)

    result = pd.concat(pieces, ignore_index=True)
    result[value_column] = pd.to_numeric(result[value_column], errors="coerce")
    return result.dropna(subset=[value_column]).sort_values(["station_label", "datetime"]).reset_index(drop=True)


def detect_stl_anomalies(
    data: pd.DataFrame,
    value_column: str,
    period: int = 24,
    threshold: float = 3.0,
) -> pd.DataFrame:
    frame = prepare_anomaly_dataset(data, value_column)
    if frame.empty:
        return frame

    try:
        from statsmodels.tsa.seasonal import STL
    except ModuleNotFoundError:
        frame["trend"] = np.nan
        frame["seasonal"] = np.nan
        frame["resid"] = np.nan
        frame["anomaly_score"] = np.nan
        frame["anomaly_flag"] = False
        frame["method"] = "STL"
        frame["status_message"] = "statsmodels is not installed. Run: pip install statsmodels"
        return frame

    pieces = []
    for station, station_frame in frame.groupby("station_label", observed=True):
        station_frame = station_frame.copy().sort_values("datetime")
        indexed = station_frame.set_index("datetime")[[value_column]].resample("h").mean()
        indexed[value_column] = indexed[value_column].interpolate(method="time").ffill().bfill()

        if len(indexed) < max(period * 2, 12):
            station_result = indexed.reset_index()
            station_result["station_label"] = station
            station_result["trend"] = np.nan
            station_result["seasonal"] = np.nan
            station_result["resid"] = np.nan
            station_result["anomaly_score"] = np.nan
            station_result["anomaly_flag"] = False
            station_result["method"] = "STL"
            pieces.append(station_result)
            continue

        seasonal_period = max(2, min(period, max(2, len(indexed) // 2)))
        stl = STL(indexed[value_column], period=seasonal_period, robust=True)
        result = stl.fit()
        resid = pd.Series(result.resid, index=indexed.index)
        median_resid = float(np.nanmedian(resid))
        mad = float(np.nanmedian(np.abs(resid - median_resid)))
        scale = 1.4826 * mad if mad > 0 else np.nan
        robust_score = (resid - median_resid) / scale if pd.notna(scale) else pd.Series(np.nan, index=indexed.index)

        station_result = indexed.reset_index()
        station_result["station_label"] = station
        station_result["trend"] = result.trend
        station_result["seasonal"] = result.seasonal
        station_result["resid"] = result.resid
        station_result["anomaly_score"] = robust_score.to_numpy()
        station_result["anomaly_flag"] = np.abs(station_result["anomaly_score"]) > threshold
        station_result["method"] = "STL"
        pieces.append(station_result)

    return pd.concat(pieces, ignore_index=True).sort_values(["station_label", "datetime"]).reset_index(drop=True)


def anomaly_summary_table(result_df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    if result_df.empty or "anomaly_flag" not in result_df.columns:
        return pd.DataFrame()

    summary = (
        result_df.groupby("station_label", observed=True)
        .agg(
            observations=(value_column, "count"),
            anomalies=("anomaly_flag", "sum"),
            mean_value=(value_column, "mean"),
            max_value=(value_column, "max"),
        )
        .reset_index()
    )
    summary["anomaly_rate_pct"] = np.where(
        summary["observations"] > 0,
        (summary["anomalies"] / summary["observations"]) * 100.0,
        0.0,
    )
    return summary.sort_values(["anomalies", "anomaly_rate_pct"], ascending=False).reset_index(drop=True)


def anomaly_events_table(result_df: pd.DataFrame, value_column: str, max_rows: int = 100) -> pd.DataFrame:
    if result_df.empty or "anomaly_flag" not in result_df.columns:
        return pd.DataFrame()
    events = result_df[result_df["anomaly_flag"]].copy()
    if events.empty:
        return pd.DataFrame()

    sort_col = "anomaly_score" if "anomaly_score" in events.columns else value_column
    events = events.sort_values(sort_col, ascending=False, na_position="last")
    keep = [c for c in ["datetime", "station_label", value_column, "anomaly_score", "trend", "seasonal", "resid"] if c in events.columns]
    return events[keep].head(max_rows).reset_index(drop=True)


def create_anomaly_timeseries_figure(
    result_df: pd.DataFrame,
    value_column: str,
    title: str,
    subtitle: str | None = None,
) -> go.Figure | None:
    if result_df.empty:
        return None

    use_webgl = prefers_webgl(len(result_df)) if "prefers_webgl" in globals() else len(result_df) > 3000
    trace_cls = go.Scattergl if use_webgl else go.Scatter
    fig = go.Figure()

    for idx, station in enumerate(result_df["station_label"].dropna().unique()):
        station_frame = result_df[result_df["station_label"] == station].sort_values("datetime")
        color = APP_CONFIG["colorway"][idx % len(APP_CONFIG["colorway"])]
        fig.add_trace(
            trace_cls(
                x=station_frame["datetime"],
                y=station_frame[value_column],
                mode="lines",
                name=compact_station_label(station),
                line={"width": 2, "color": color},
            )
        )
        flagged = station_frame[station_frame["anomaly_flag"]] if "anomaly_flag" in station_frame.columns else station_frame.iloc[0:0]
        if not flagged.empty:
            fig.add_trace(
                trace_cls(
                    x=flagged["datetime"],
                    y=flagged[value_column],
                    mode="markers",
                    name=f"{compact_station_label(station)} anomalies",
                    marker={"size": 9, "symbol": "x", "color": "#dc2626", "line": {"width": 1}},
                    showlegend=True,
                )
            )

    fig = chart_template(
        fig,
        title,
        axis_label(value_column),
        "Datetime",
        height=460,
        subtitle=subtitle or format_window_from_data(result_df, "Detected anomalies over active filter window"),
    )
    fig.update_traces(hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}<extra>%{fullData.name}</extra>")
    return fig


def create_anomaly_score_figure(result_df: pd.DataFrame, title: str) -> go.Figure | None:
    if result_df.empty or "anomaly_score" not in result_df.columns:
        return None

    plot_frame = result_df.dropna(subset=["anomaly_score"]).copy()
    if plot_frame.empty:
        return None

    fig = px.scatter(
        plot_frame,
        x="datetime",
        y="anomaly_score",
        color="station_label",
        hover_data={"station_label": True},
        color_discrete_sequence=APP_CONFIG["colorway"],
    )
    for trace in fig.data:
        if getattr(trace, "name", None):
            trace.name = compact_station_label(trace.name)
    return chart_template(
        fig,
        title,
        "Anomaly score",
        "Datetime",
        height=420,
        subtitle=format_window_from_data(plot_frame, "Higher magnitude indicates more unusual behavior"),
    )


def create_stl_components_figure(result_df: pd.DataFrame, value_column: str, station_label: str) -> go.Figure | None:
    if result_df.empty or not {"trend", "seasonal", "resid"}.issubset(result_df.columns):
        return None
    station_frame = result_df[result_df["station_label"] == station_label].copy().sort_values("datetime")
    if station_frame.empty:
        return None

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=(
        f"Observed {display_label(value_column)}", "Trend", "Seasonal", "Residual"
    ))
    component_specs = [
        (value_column, "Observed"),
        ("trend", "Trend"),
        ("seasonal", "Seasonal"),
        ("resid", "Residual"),
    ]
    for row_idx, (col, name) in enumerate(component_specs, start=1):
        fig.add_trace(
            go.Scatter(
                x=station_frame["datetime"],
                y=station_frame[col],
                mode="lines",
                name=name,
                showlegend=False,
            ),
            row=row_idx,
            col=1,
        )
    flagged = station_frame[station_frame["anomaly_flag"]] if "anomaly_flag" in station_frame.columns else station_frame.iloc[0:0]
    if not flagged.empty:
        fig.add_trace(
            go.Scatter(
                x=flagged["datetime"],
                y=flagged[value_column],
                mode="markers",
                marker={"size": 8, "symbol": "x", "color": "#dc2626"},
                name="Anomalies",
            ),
            row=1,
            col=1,
        )
    fig.update_layout(
        height=760,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        margin={"l": 28, "r": 24, "t": 90, "b": 40},
        title={
            "text": compose_title(f"STL Components - {compact_station_label(station_label)}", format_window_from_data(station_frame, None)),
            "x": 0.02,
            "xanchor": "left",
        },
    )
    return fig


def render_anomaly_method_tab(
    method_name: str,
    result_df: pd.DataFrame,
    value_column: str,
    description: str,
    station_options: list[str] | None = None,
    show_stl_components: bool = False,
) -> None:
    st.markdown(f'<p class="tab-note">{description}</p>', unsafe_allow_html=True)

    if result_df.empty:
        st.info("No records are available for the current filters.")
        return

    if "status_message" in result_df.columns and result_df["status_message"].dropna().nunique() > 0:
        st.warning(result_df["status_message"].dropna().iloc[0])

    summary = anomaly_summary_table(result_df, value_column)
    events = anomaly_events_table(result_df, value_column)

    top_left, top_right = st.columns([1.5, 1.0])
    with top_left:
        render_chart(
            create_anomaly_timeseries_figure(result_df, value_column, f"{method_name} - {display_label(value_column)}"),
            f"{method_name} chart is unavailable.",
        )
    with top_right:
        render_chart(
            create_anomaly_score_figure(result_df, f"{method_name} Score Profile"),
            f"{method_name} score view is unavailable.",
        )

    metric_cols = st.columns(3)
    anomaly_count = int(result_df["anomaly_flag"].sum()) if "anomaly_flag" in result_df.columns else 0
    total_count = int(result_df[value_column].count()) if value_column in result_df.columns else 0
    anomaly_rate = (anomaly_count / total_count * 100.0) if total_count else 0.0
    metric_cols[0].metric("Anomalies flagged", f"{anomaly_count:,}")
    metric_cols[1].metric("Observations", f"{total_count:,}")
    metric_cols[2].metric("Flag rate", f"{anomaly_rate:.2f}%")

    st.markdown("#### Summary by Station")
    if summary.empty:
        st.info("No anomaly summary is available.")
    else:
        st.dataframe(summary, use_container_width=True)

    st.markdown("#### Top Flagged Events")
    if events.empty:
        st.info("No anomalies were flagged for the active filters.")
    else:
        st.dataframe(events, use_container_width=True)

    if show_stl_components and station_options:
        available_stations = [s for s in station_options if s in set(result_df["station_label"].dropna())]
        if available_stations:
            selected_station = st.selectbox(
                "Inspect STL components for station",
                options=available_stations,
                key=f"stl_station_{value_column}",
            )
            render_chart(
                create_stl_components_figure(result_df, value_column, selected_station),
                "STL component view is unavailable for this station.",
            )

def render_dashboard(
    pollutant_df: pd.DataFrame,
    meteorology_df: pd.DataFrame,
    station_meta_df: pd.DataFrame,
    available_pollutants: list[str],
    available_meteorology: list[str],
) -> None:
    station_options = station_meta_df["station_label"].tolist()
    min_ts = min(pollutant_df["datetime"].min(), meteorology_df["datetime"].min())
    max_ts = max(pollutant_df["datetime"].max(), meteorology_df["datetime"].max())

    defaults = default_state(station_options, available_pollutants, available_meteorology, min_ts, max_ts)
    initialize_state(defaults)

    render_header(station_meta_df, min_ts, max_ts)

    controls = sidebar_controls(
        station_options,
        available_pollutants,
        available_meteorology,
        defaults,
        min_ts,
        max_ts,
    )

    selected_stations = controls["selected_stations"]
    selected_pollutant = controls["selected_pollutant"]
    selected_meteorology = controls["selected_meteorology"]
    aggregation = controls["aggregation"]
    rolling_window = int(controls["rolling_window"])
    start_ts = controls["start_ts"]
    end_ts = controls["end_ts"]

    mode = determine_mode(selected_pollutant, selected_meteorology)

    pollutant_filtered = filter_pollutant_data(pollutant_df, selected_stations, start_ts, end_ts)
    meteorology_filtered = filter_meteorology_data(meteorology_df, start_ts, end_ts)
    meteorology_expanded = expand_meteorology_for_stations(meteorology_filtered, selected_stations)
    combined_df = build_combined_dataset(pollutant_filtered, meteorology_filtered)

    if mode == "pollutant":
        primary_source = pollutant_filtered
        primary_variable = selected_pollutant
    elif mode == "meteorology":
        primary_source = meteorology_expanded
        primary_variable = selected_meteorology
    elif mode == "combined":
        primary_source = pollutant_filtered
        primary_variable = selected_pollutant
    else:
        primary_source = pd.DataFrame()
        primary_variable = None

    peak_value, peak_caption = peak_metric(primary_source, primary_variable)

    render_top_kpis(
        selected_stations,
        selected_pollutant,
        selected_meteorology,
        peak_value,
        peak_caption,
    )

    download_df = create_download_dataset(
        mode,
        pollutant_filtered,
        meteorology_filtered,
        meteorology_expanded,
        selected_pollutant,
        selected_meteorology,
    )
    st.download_button(
        label="Download filtered dataset",
        data=download_bytes(download_df),
        file_name="filtered_air_quality_dashboard.csv",
        mime="text/csv",
        disabled=download_df.empty,
    )

    if mode == "empty":
        st.warning("Select a pollutant, a meteorology variable, or both to start exploring the dashboard.")
        return

    anomaly_base_df = combined_df if mode == "combined" else primary_source
    zscore_results = detect_rolling_zscore_anomalies(primary_source, primary_variable, window=max(rolling_window, 6), threshold=3.0) if primary_variable and not primary_source.empty else pd.DataFrame()
    iqr_results = detect_iqr_anomalies(primary_source, primary_variable, multiplier=1.5) if primary_variable and not primary_source.empty else pd.DataFrame()
    isolation_results = detect_isolation_forest_anomalies(anomaly_base_df, primary_variable, contamination=0.02) if primary_variable and not anomaly_base_df.empty else pd.DataFrame()
    stl_results = detect_stl_anomalies(primary_source, primary_variable, period=24 if aggregation in {"Hourly", "Daily"} else 12, threshold=3.0) if primary_variable and not primary_source.empty else pd.DataFrame()

    overview_tab, network_tab, map_tab, timeseries_tab, distribution_tab, relationships_tab, quality_tab, quantile_tab, zscore_tab, iqr_tab, isolation_tab, stl_tab = st.tabs(
        ["Overview", "Station Network", "EPA Map", "Time Series", "Distribution", "Relationships", "Data Quality", "Quantile Windows", "Z-Score", "IQR", "Isolation Forest", "STL Decomposition"]
    )

    with overview_tab:
        st.markdown(
            '<p class="tab-note">Adaptive logic: 1 day = hourly 00:00-23:00, 3 days = 3 hourly panels, 15 days = daily average, full month = monthly views, full year = yearly monthly average.</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="tab-note"><strong>New:</strong> Open the embedded EPA viewer from the <strong>EPA Map</strong> tab or <a href="{APP_CONFIG["epa_map_url"]}" target="_blank">launch it in a new tab</a>.</p>',
            unsafe_allow_html=True,
        )

        if mode == "pollutant" and selected_pollutant:
            render_chart(
                create_diurnal_figure(pollutant_filtered, selected_pollutant),
                "No pollutant diurnal profile is available.",
            )
            render_chart(
                create_recent_figure(pollutant_filtered, selected_pollutant, start_ts, end_ts),
                "No pollutant recent view is available.",
            )

            if is_full_month_selection(start_ts, end_ts):
                render_chart(
                    create_monthly_daily_average_figure(pollutant_filtered, selected_pollutant, start_ts, end_ts),
                    "No monthly pollutant trend is available.",
                )
            if is_full_year_selection(start_ts, end_ts):
                render_chart(
                    create_yearly_month_average_figure(pollutant_filtered, selected_pollutant, start_ts, end_ts),
                    "No yearly pollutant trend is available.",
                )

        elif mode == "meteorology" and selected_meteorology:
            render_chart(
                create_diurnal_figure(meteorology_expanded, selected_meteorology),
                "No meteorology diurnal profile is available.",
            )
            render_chart(
                create_recent_figure(meteorology_expanded, selected_meteorology, start_ts, end_ts),
                "No meteorology recent view is available.",
            )

            if is_full_month_selection(start_ts, end_ts):
                render_chart(
                    create_monthly_daily_average_figure(meteorology_expanded, selected_meteorology, start_ts, end_ts),
                    "No monthly meteorology trend is available.",
                )
            if is_full_year_selection(start_ts, end_ts):
                render_chart(
                    create_yearly_month_average_figure(meteorology_expanded, selected_meteorology, start_ts, end_ts),
                    "No yearly meteorology trend is available.",
                )

        elif mode == "combined" and selected_pollutant and selected_meteorology:
            left_chart, right_chart = st.columns(2)
            with left_chart:
                render_chart(
                    create_diurnal_figure(pollutant_filtered, selected_pollutant),
                    "No pollutant diurnal profile is available.",
                )
            with right_chart:
                render_chart(
                    create_diurnal_figure(meteorology_expanded, selected_meteorology),
                    "No meteorology diurnal profile is available.",
                )

            render_chart(
                create_overlay_figure(
                    pollutant_filtered,
                    meteorology_filtered.assign(station_label="Regional ERA5"),
                    selected_pollutant,
                    selected_meteorology,
                    "Hourly",
                    1,
                    "Dual-Axis Overlay - Pollutant vs Meteorology",
                ),
                "Overlay plot needs valid pollutant and meteorology records.",
            )

            if is_full_month_selection(start_ts, end_ts):
                col1, col2 = st.columns(2)
                with col1:
                    render_chart(
                        create_monthly_daily_average_figure(pollutant_filtered, selected_pollutant, start_ts, end_ts),
                        "No monthly pollutant trend is available.",
                    )
                with col2:
                    render_chart(
                        create_monthly_daily_average_figure(meteorology_expanded, selected_meteorology, start_ts, end_ts),
                        "No monthly meteorology trend is available.",
                    )

            if is_full_year_selection(start_ts, end_ts):
                col1, col2 = st.columns(2)
                with col1:
                    render_chart(
                        create_yearly_month_average_figure(pollutant_filtered, selected_pollutant, start_ts, end_ts),
                        "No yearly pollutant trend is available.",
                    )
                with col2:
                    render_chart(
                        create_yearly_month_average_figure(meteorology_expanded, selected_meteorology, start_ts, end_ts),
                        "No yearly meteorology trend is available.",
                    )

        if primary_variable is not None and not primary_source.empty:
            st.markdown("#### Top Extreme Days")
            extreme_table = extreme_days_table(primary_source, primary_variable)
            if extreme_table.empty:
                st.info("No extreme-day summary is available for the current filters.")
            else:
                styled = extreme_table.style.format(
                    {
                        "daily_mean": "{:,.2f}",
                        "daily_max": "{:,.2f}",
                        "daily_min": "{:,.2f}",
                        "valid_obs": "{:,.0f}",
                    }
                )
                st.dataframe(styled, use_container_width=True)

    with timeseries_tab:
        st.markdown(
            '<p class="tab-note">Aggregation, rolling averages, peaks, and >90th percentile markers are controlled from the sidebar.</p>',
            unsafe_allow_html=True,
        )

        if mode in {"pollutant", "combined"} and selected_pollutant:
            render_chart(
                create_timeseries_figure(pollutant_filtered, selected_pollutant, aggregation, rolling_window),
                "No pollutant time series is available.",
            )

        if mode in {"meteorology", "combined"} and selected_meteorology:
            render_chart(
                create_timeseries_figure(meteorology_expanded, selected_meteorology, aggregation, rolling_window),
                "No meteorology time series is available.",
            )

        if mode == "combined" and selected_pollutant and selected_meteorology:
            render_chart(
                create_overlay_figure(
                    pollutant_filtered,
                    meteorology_filtered.assign(station_label="Regional ERA5"),
                    selected_pollutant,
                    selected_meteorology,
                    aggregation,
                    rolling_window,
                    "Dual-Axis Overlay",
                ),
                "Overlay time series requires both pollutant and meteorology records.",
            )

    with distribution_tab:
        st.markdown(
            '<p class="tab-note">Boxplots, histograms, and quantile summaries for the active variable selection.</p>',
            unsafe_allow_html=True,
        )

        if mode in {"pollutant", "combined"} and selected_pollutant:
            st.markdown(f"#### {display_label(selected_pollutant)}")
            c1, c2 = st.columns(2)
            with c1:
                render_chart(
                    create_boxplot_figure(pollutant_filtered, selected_pollutant),
                    "No pollutant boxplot is available.",
                )
            with c2:
                render_chart(
                    create_histogram_figure(pollutant_filtered, selected_pollutant),
                    "No pollutant histogram is available.",
                )
            render_quantile_table(pollutant_filtered, selected_pollutant, selected_stations)

        if mode in {"meteorology", "combined"} and selected_meteorology:
            st.markdown(f"#### {display_label(selected_meteorology)}")
            c1, c2 = st.columns(2)
            with c1:
                render_chart(
                    create_boxplot_figure(meteorology_expanded, selected_meteorology),
                    "No meteorology boxplot is available.",
                )
            with c2:
                render_chart(
                    create_histogram_figure(meteorology_expanded, selected_meteorology),
                    "No meteorology histogram is available.",
                )
            render_quantile_table(meteorology_expanded, selected_meteorology, selected_stations)

    with relationships_tab:
        st.markdown(
            '<p class="tab-note">Scatter, regression, and correlation heatmaps help explore pollutant-meteorology relationships.</p>',
            unsafe_allow_html=True,
        )

        if mode != "combined" or not selected_pollutant or not selected_meteorology:
            st.info("Relationship analytics unlock when both a pollutant and a meteorology variable are selected.")
        else:
            left_rel, right_rel = st.columns([1.1, 1.0])
            with left_rel:
                render_chart(
                    create_scatter_figure(combined_df, selected_pollutant, selected_meteorology),
                    "Relationship scatter plot is unavailable.",
                )
            with right_rel:
                render_chart(
                    create_correlation_heatmap(
                        combined_df,
                        selected_pollutant,
                        selected_meteorology,
                        available_pollutants,
                        available_meteorology,
                    ),
                    "Correlation heatmap is unavailable.",
                )

    with quality_tab:
        st.markdown(
            '<p class="tab-note">Completeness, missingness, and summary statistics for the active variables.</p>',
            unsafe_allow_html=True,
        )
        summaries = []

        if mode in {"pollutant", "combined"} and selected_pollutant:
            summaries.append(
                render_quality_block(
                    f"{display_label(selected_pollutant)} Data Quality",
                    pollutant_filtered,
                    selected_pollutant,
                    selected_stations,
                    start_ts,
                    end_ts,
                )
            )

        if mode in {"meteorology", "combined"} and selected_meteorology:
            summaries.append(
                render_quality_block(
                    f"{display_label(selected_meteorology)} Data Quality",
                    meteorology_expanded,
                    selected_meteorology,
                    selected_stations,
                    start_ts,
                    end_ts,
                )
            )

        if summaries:
            combined_summary = pd.concat(summaries, ignore_index=True)
            st.markdown("#### Combined Quality Table")
            st.dataframe(combined_summary, use_container_width=True)

    with map_tab:
        render_epa_map_tab()

    with network_tab:
        render_station_network_tab(station_meta_df)

    with quantile_tab:
        render_quantile_tab(
            pollutant_filtered,
            meteorology_expanded,
            selected_pollutant,
            selected_meteorology,
            selected_stations,
            mode,
        )
    with zscore_tab:
        if primary_variable is None or primary_source.empty:
            st.info("Select an active variable to run rolling Z-score detection.")
        else:
            render_anomaly_method_tab(
                "Rolling Z-Score",
                zscore_results,
                primary_variable,
                "Uses a rolling mean and rolling standard deviation to highlight sudden spikes and drops relative to the recent local window.",
            )

    with iqr_tab:
        if primary_variable is None or primary_source.empty:
            st.info("Select an active variable to run IQR-based outlier detection.")
        else:
            render_anomaly_method_tab(
                "IQR Outlier Detection",
                iqr_results,
                primary_variable,
                "Uses Q1 and Q3 for each station and flags values outside the classic 1.5 × IQR bounds. This is robust to skewed distributions.",
            )

    with isolation_tab:
        if primary_variable is None or anomaly_base_df.empty:
            st.info("Select an active variable to run Isolation Forest detection.")
        else:
            render_anomaly_method_tab(
                "Isolation Forest",
                isolation_results,
                primary_variable,
                "Uses an unsupervised tree-based model with time features and any available numeric covariates to isolate unusual multivariate behavior.",
            )

    with stl_tab:
        if primary_variable is None or primary_source.empty:
            st.info("Select an active variable to run STL decomposition.")
        else:
            render_anomaly_method_tab(
                "STL Decomposition",
                stl_results,
                primary_variable,
                "Decomposes the signal into trend, seasonality, and residuals, then flags unusually large residual deviations.",
                station_options=selected_stations,
                show_stl_components=True,
            )



def render_dependency_error() -> None:
    missing_package = IMPORT_ERROR.name if IMPORT_ERROR is not None else "a required package"
    st.error(f"The dashboard cannot start because `{missing_package}` is not installed.")
    st.code("pip install -r requirements.txt")


def main() -> None:
    configure_page()
    inject_theme()

    if IMPORT_ERROR is not None:
        render_dependency_error()
        st.stop()

    try:
        pollutant_df, station_meta_df, available_pollutants = load_pollutant_data()
        meteorology_df, available_meteorology = load_meteorology_data()
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

    render_dashboard(
        pollutant_df,
        meteorology_df,
        station_meta_df,
        available_pollutants,
        available_meteorology,
    )


if __name__ == "__main__":
    main()

