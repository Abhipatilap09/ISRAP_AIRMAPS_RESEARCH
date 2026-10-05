from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable

import streamlit as st

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
        "temp_c": "Temperature (C)",
        "relative_humidity": "Relative Humidity (%)",
        "wind_speed": "Wind Speed (m/s)",
        "dewpoint_c": "Dew Point (C)",
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
        "PM2.5": "ug/m3",
        "NO2": "ppb",
        "O3": "ppm",
        "SO2": "ppb",
        "CO": "ppm",
        "temp_c": "C",
        "relative_humidity": "%",
        "wind_speed": "m/s",
        "dewpoint_c": "C",
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
        "#1d4ed8",
        "#0f766e",
        "#c2410c",
        "#7c3aed",
        "#0891b2",
        "#be123c",
        "#65a30d",
        "#334155",
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


def display_label(column_name: str | None) -> str:
    if column_name is None:
        return "None"
    return APP_CONFIG["display_labels"].get(column_name, column_name)


def axis_label(column_name: str | None) -> str:
    if column_name is None:
        return "None"
    base_label = display_label(column_name)
    unit = APP_CONFIG["axis_units"].get(column_name)
    return f"{base_label} ({unit})" if unit else base_label


def compact_station_label(label: str) -> str:
    marker = " - Site "
    if marker in label:
        return f"Site {label.split(marker, 1)[1]}"
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


def inject_theme() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg: #f4f7fb;
                --card: rgba(255, 255, 255, 0.92);
                --card-border: rgba(148, 163, 184, 0.18);
                --ink: #0f172a;
                --muted: #475569;
                --accent: #1d4ed8;
                --shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
            }
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(29, 78, 216, 0.08), transparent 32%),
                    radial-gradient(circle at top right, rgba(15, 118, 110, 0.08), transparent 28%),
                    linear-gradient(180deg, #f8fbff 0%, var(--bg) 65%, #eef4fb 100%);
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
            [data-testid="stSidebar"] .stTextInput input,
            [data-testid="stSidebar"] .stDateInput input,
            [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"],
            [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] {
                border-radius: 16px !important;
            }
            .hero {
                background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(255,255,255,0.86));
                border: 1px solid var(--card-border);
                border-radius: 28px;
                padding: 1.5rem 1.6rem 1.25rem 1.6rem;
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
                margin: 0.35rem 0 0.45rem 0;
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
                min-height: 132px;
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
                font-size: 1.7rem;
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
                background: rgba(255, 255, 255, 0.88);
                border: 1px solid var(--card-border);
                border-radius: 24px;
                padding: 0.45rem;
                box-shadow: var(--shadow);
            }
            div[data-testid="stDataFrame"] {
                background: rgba(255, 255, 255, 0.88);
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


def station_files() -> list[Path]:
    return sorted(APP_CONFIG["pollutant_dir"].glob("*.csv"))


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
            column
            for column in APP_CONFIG["pollutant_candidates"]
            if column in frame.columns
        ]
        for column in pollutant_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        site_value = (
            str(frame[APP_CONFIG["site_col"]].dropna().iloc[0])
            if APP_CONFIG["site_col"] in frame.columns
            and not frame[APP_CONFIG["site_col"]].dropna().empty
            else f"unknown-{index}"
        )
        state_value = (
            str(frame[APP_CONFIG["state_col"]].dropna().iloc[0])
            if APP_CONFIG["state_col"] in frame.columns
            and not frame[APP_CONFIG["state_col"]].dropna().empty
            else "NA"
        )
        county_value = (
            str(frame[APP_CONFIG["county_col"]].dropna().iloc[0])
            if APP_CONFIG["county_col"] in frame.columns
            and not frame[APP_CONFIG["county_col"]].dropna().empty
            else "NA"
        )

        station_label = build_station_label(site_value)
        keep_columns = [
            column
            for column in [
                APP_CONFIG["datetime_col"],
                APP_CONFIG["state_col"],
                APP_CONFIG["county_col"],
                APP_CONFIG["site_col"],
                *pollutant_columns,
            ]
            if column in frame.columns
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
                "site_sort": (
                    int(float(site_value))
                    if str(site_value).replace(".", "", 1).isdigit()
                    else index
                ),
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

    if APP_CONFIG["site_col"] in pollutant_df.columns:
        pollutant_df = pollutant_df.sort_values([APP_CONFIG["site_col"], "datetime"]).reset_index(drop=True)
    else:
        pollutant_df = pollutant_df.sort_values(["station_label", "datetime"]).reset_index(drop=True)

    available_pollutants = [
        column
        for column in APP_CONFIG["pollutant_candidates"]
        if column in pollutant_df.columns and pollutant_df[column].notna().any()
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
        raise ValueError(
            "Meteorology file needs a usable datetime source such as date/time, datetime_local, or datetime."
        )

    frame = frame.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates("datetime")

    available_columns = [
        column for column in APP_CONFIG["meteorology_candidates"] if column in frame.columns
    ]
    for column in available_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    available_meteorology = [column for column in available_columns if frame[column].notna().any()]
    trimmed = frame[["datetime", *available_meteorology]].copy()
    return trimmed.reset_index(drop=True), available_meteorology


def default_state(
    station_options: list[str],
    available_pollutants: list[str],
    available_meteorology: list[str],
    min_timestamp: pd.Timestamp,
    max_timestamp: pd.Timestamp,
) -> dict[str, object]:
    return {
        "station_search": "",
        "selected_stations": station_options[: min(3, len(station_options))],
        "selected_pollutant": first_available(
            available_pollutants, APP_CONFIG["default_pollutant_priority"]
        ),
        "selected_meteorology": first_available(
            available_meteorology, APP_CONFIG["default_meteorology_priority"]
        ),
        "aggregation": "Daily",
        "rolling_window": 7,
        "date_range": (min_timestamp.date(), max_timestamp.date()),
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


def parse_date_range(
    date_value: object, min_ts: pd.Timestamp, max_ts: pd.Timestamp
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if isinstance(date_value, (tuple, list)) and len(date_value) == 2:
        start_raw, end_raw = date_value
    else:
        start_raw = end_raw = date_value

    start_ts = pd.Timestamp(start_raw)
    end_ts = pd.Timestamp(end_raw) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return max(start_ts, min_ts), min(end_ts, max_ts)


def determine_mode(pollutant: str | None, meteorology: str | None) -> str:
    if pollutant and meteorology:
        return "combined"
    if pollutant:
        return "pollutant"
    if meteorology:
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


def expand_meteorology_for_stations(
    meteorology_df: pd.DataFrame, selected_stations: list[str]
) -> pd.DataFrame:
    if meteorology_df.empty:
        return meteorology_df.assign(station_label=pd.Series(dtype="object"))

    station_labels = selected_stations or ["Regional ERA5"]
    station_frames = []
    for station in station_labels:
        clone = meteorology_df.copy()
        clone["station_label"] = station
        station_frames.append(clone)
    return pd.concat(station_frames, ignore_index=True).sort_values(
        ["station_label", "datetime"]
    ).reset_index(drop=True)


def build_combined_dataset(
    pollutant_df: pd.DataFrame, meteorology_df: pd.DataFrame
) -> pd.DataFrame:
    if pollutant_df.empty or meteorology_df.empty:
        return pd.DataFrame()
    combined = pollutant_df.merge(meteorology_df, on="datetime", how="left")
    return combined.sort_values(["station_label", "datetime"]).reset_index(drop=True)


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
        keep_columns = [
            column
            for column in [
                "datetime",
                "station_label",
                APP_CONFIG["state_col"],
                APP_CONFIG["county_col"],
                APP_CONFIG["site_col"],
                selected_pollutant,
                selected_meteorology,
            ]
            if column in combined.columns
        ]
        return combined[keep_columns]

    if mode == "pollutant" and selected_pollutant:
        keep_columns = [
            column
            for column in [
                "datetime",
                "station_label",
                APP_CONFIG["state_col"],
                APP_CONFIG["county_col"],
                APP_CONFIG["site_col"],
                selected_pollutant,
            ]
            if column in pollutant_df.columns
        ]
        return pollutant_df[keep_columns]

    if mode == "meteorology" and selected_meteorology:
        keep_columns = [
            column
            for column in ["datetime", "station_label", selected_meteorology]
            if column in meteorology_expanded_df.columns
        ]
        return meteorology_expanded_df[keep_columns]

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
    aggregated = (
        frame.groupby(["station_label", "period"], observed=True, as_index=False)[value_column]
        .mean()
        .sort_values(["station_label", "period"])
        .reset_index(drop=True)
    )
    return aggregated


def monthly_series(data: pd.DataFrame, value_column: str) -> pd.DataFrame:
    return aggregate_series(data, value_column, "Monthly")


def yearly_series(data: pd.DataFrame, value_column: str) -> pd.DataFrame:
    return aggregate_series(data, value_column, "Yearly")


def recent_window(data: pd.DataFrame, value_column: str, hours: int = 72) -> pd.DataFrame:
    frame = select_value_frame(data, value_column)
    if frame.empty:
        return frame
    max_timestamp = frame["datetime"].max()
    lower_bound = max_timestamp - pd.Timedelta(hours=hours)
    windowed = frame[frame["datetime"] >= lower_bound]
    return windowed if not windowed.empty else frame.tail(300)


def recent_source_window(data: pd.DataFrame, value_column: str, hours: int = 72) -> pd.DataFrame:
    if value_column not in data.columns or data.empty:
        return data.iloc[0:0].copy()
    frame = data[["datetime", value_column]].copy()
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    frame = frame.dropna(subset=[value_column])
    if frame.empty:
        return data.iloc[0:0].copy()
    max_timestamp = frame["datetime"].max()
    lower_bound = max_timestamp - pd.Timedelta(hours=hours)
    windowed = data[data["datetime"] >= lower_bound].copy()
    return windowed if not windowed.empty else data.copy()


def monthly_average_series(data: pd.DataFrame, value_column: str) -> pd.DataFrame:
    frame = select_value_frame(data, value_column)
    if frame.empty:
        return frame
    frame["month"] = frame["datetime"].dt.to_period("M").dt.to_timestamp()
    return (
        frame.groupby(["station_label", "month"], observed=True, as_index=False)[value_column]
        .mean()
        .sort_values(["station_label", "month"])
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


def add_rolling_average(frame: pd.DataFrame, value_column: str, window: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["rolling_value"] = (
        frame.groupby("station_label", observed=True)[value_column]
        .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    )
    return frame


def quantile_summary(data: pd.DataFrame, value_column: str, station_order: list[str]) -> pd.DataFrame:
    frame = select_value_frame(data, value_column)
    if frame.empty:
        return pd.DataFrame()

    summary = (
        frame.groupby("station_label", observed=True)[value_column]
        .agg(
            observations="count",
            mean="mean",
            std="std",
            minimum="min",
            median="median",
            maximum="max",
        )
        .reset_index()
    )
    q1 = frame.groupby("station_label", observed=True)[value_column].quantile(0.25)
    q3 = frame.groupby("station_label", observed=True)[value_column].quantile(0.75)
    summary["Q1"] = summary["station_label"].map(q1)
    summary["Q3"] = summary["station_label"].map(q3)

    if station_order:
        summary["station_label"] = pd.Categorical(
            summary["station_label"], categories=station_order, ordered=True
        )
        summary = summary.sort_values("station_label")
        summary["station_label"] = summary["station_label"].astype(str)

    return summary.reset_index(drop=True)


def expected_observations(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> int:
    if end_ts < start_ts:
        return 0
    return int(((end_ts - start_ts).total_seconds() // 3600) + 1)


def quality_summary_table(
    data: pd.DataFrame,
    value_column: str,
    station_order: list[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    frame = select_value_frame(data, value_column)
    total_expected = max(expected_observations(start_ts, end_ts), 1)

    if frame.empty:
        base = pd.DataFrame({"station_label": station_order or []})
        if base.empty:
            return pd.DataFrame()
        base["variable"] = display_label(value_column)
        base["valid_obs"] = 0
        base["availability_pct"] = 0.0
        base["missing_pct"] = 100.0
        for col in ["mean", "std", "min", "Q1", "median", "Q3", "max"]:
            base[col] = np.nan
        return base

    summary = (
        frame.groupby("station_label", observed=True)[value_column]
        .agg(valid_obs="count", mean="mean", std="std", min="min", median="median", max="max")
        .reset_index()
    )
    q1 = frame.groupby("station_label", observed=True)[value_column].quantile(0.25)
    q3 = frame.groupby("station_label", observed=True)[value_column].quantile(0.75)

    summary["Q1"] = summary["station_label"].map(q1)
    summary["Q3"] = summary["station_label"].map(q3)
    summary["availability_pct"] = summary["valid_obs"] / total_expected * 100
    summary["missing_pct"] = 100 - summary["availability_pct"]
    summary["variable"] = display_label(value_column)

    if station_order:
        summary["station_label"] = pd.Categorical(
            summary["station_label"], categories=station_order, ordered=True
        )
        summary = summary.sort_values("station_label")
        summary["station_label"] = summary["station_label"].astype(str)

    return summary[
        [
            "station_label",
            "variable",
            "valid_obs",
            "availability_pct",
            "missing_pct",
            "mean",
            "std",
            "min",
            "Q1",
            "median",
            "Q3",
            "max",
        ]
    ].reset_index(drop=True)


def create_completeness_figure(
    data: pd.DataFrame,
    value_column: str,
    station_order: list[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> tuple[go.Figure | None, str]:
    frame = select_value_frame(data, value_column)
    if frame.empty:
        return None, "Hourly"

    span_days = max((end_ts.floor("D") - start_ts.floor("D")).days + 1, 1)
    granularity = "Hourly" if span_days <= 10 else "Daily"

    expected_range = pd.date_range(
        start=start_ts.floor("h") if granularity == "Hourly" else start_ts.floor("D"),
        end=end_ts.floor("h") if granularity == "Hourly" else end_ts.floor("D"),
        freq="h" if granularity == "Hourly" else "D",
    )

    if granularity == "Hourly":
        frame["bucket"] = frame["datetime"].dt.floor("h")
    else:
        frame["bucket"] = frame["datetime"].dt.floor("D")

    present = (
        frame.groupby(["station_label", "bucket"], observed=True)
        .size()
        .reset_index(name="present")
    )
    present["present"] = 1

    full_index = pd.MultiIndex.from_product(
        [station_order or sorted(frame["station_label"].unique()), expected_range],
        names=["station_label", "bucket"],
    )
    matrix = (
        present.set_index(["station_label", "bucket"])
        .reindex(full_index)
        .fillna(0)
        .reset_index()
    )

    heat = matrix.pivot(index="station_label", columns="bucket", values="present")
    if heat.empty:
        return None, granularity

    fig = px.imshow(
        heat,
        aspect="auto",
        color_continuous_scale=[[0.0, "#e2e8f0"], [1.0, "#1d4ed8"]],
        labels={"x": granularity, "y": "Station", "color": "Present"},
    )
    fig.update_layout(
        title=f"Completeness Matrix - {display_label(value_column)}",
        template="plotly_white",
        height=330,
        margin={"l": 20, "r": 20, "t": 70, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
    )
    fig.update_yaxes(title="")
    return fig, granularity


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
    return table[table["daily_max"] >= threshold].sort_values(
        "daily_max", ascending=False
    ).head(12).reset_index(drop=True)


def sample_scatter_data(frame: pd.DataFrame, max_points: int = 12000) -> pd.DataFrame:
    if frame.empty or len(frame) <= max_points:
        return frame

    pieces = []
    per_station = max(max_points // max(frame["station_label"].nunique(), 1), 1)
    for _, station_frame in frame.groupby("station_label", observed=True):
        pieces.append(
            station_frame.sample(min(len(station_frame), per_station), random_state=42)
        )
    sampled = pd.concat(pieces, ignore_index=True)
    return sampled.sort_values(["station_label", "datetime"]).reset_index(drop=True)


def chart_template(
    fig: go.Figure,
    title: str,
    y_title: str,
    x_title: str,
    height: int = 420,
) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left", "y": 0.98, "yanchor": "top"},
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "color": "#0f172a"},
        colorway=APP_CONFIG["colorway"],
        height=height,
        margin={"l": 28, "r": 24, "t": 88, "b": 84},
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


def line_chart(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    y_title: str,
    x_title: str,
    markers: bool = False,
    height: int = 420,
) -> go.Figure:
    fig = px.line(
        data,
        x=x_column,
        y=y_column,
        color="station_label",
        markers=markers,
        color_discrete_sequence=APP_CONFIG["colorway"],
    )
    fig.update_traces(line={"width": 3}, marker={"size": 6})
    for trace in fig.data:
        if getattr(trace, "name", None):
            trace.name = compact_station_label(trace.name)
    return chart_template(fig, title, y_title, x_title, height=height)


def create_diurnal_figure(data: pd.DataFrame, value_column: str) -> go.Figure | None:
    profile = diurnal_profile(data, value_column)
    if profile.empty:
        return None
    fig = line_chart(
        profile,
        "hour",
        value_column,
        f"Hero Diurnal Profile - {display_label(value_column)}",
        axis_label(value_column),
        "Hour of Day",
        markers=True,
        height=440,
    )
    fig.update_xaxes(tickmode="linear", dtick=1, range=[0, 23])
    return fig


def create_recent_line_figure(
    data: pd.DataFrame, value_column: str, rolling_window: int, title: str
) -> go.Figure | None:
    frame = recent_window(data, value_column, hours=72)
    if frame.empty:
        return None

    frame = add_rolling_average(frame, value_column, rolling_window)

    fig = go.Figure()
    station_order = list(frame["station_label"].drop_duplicates())

    for idx, station in enumerate(station_order):
        station_frame = frame[frame["station_label"] == station]
        color = APP_CONFIG["colorway"][idx % len(APP_CONFIG["colorway"])]

        fig.add_trace(
            go.Scatter(
                x=station_frame["datetime"],
                y=station_frame[value_column],
                mode="lines",
                line={"width": 2.2, "color": color},
                opacity=0.35,
                name=f"{compact_station_label(station)} - Raw",
                legendgroup=station,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=station_frame["datetime"],
                y=station_frame["rolling_value"],
                mode="lines",
                line={"width": 3.2, "color": color},
                name=f"{compact_station_label(station)} - Rolling",
                legendgroup=station,
            )
        )

    # peak marker
    peak_row = frame.loc[frame[value_column].idxmax()]
    fig.add_trace(
        go.Scatter(
            x=[peak_row["datetime"]],
            y=[peak_row[value_column]],
            mode="markers",
            marker={"size": 12, "symbol": "diamond"},
            name="Peak",
        )
    )

    return chart_template(
        fig,
        title,
        axis_label(value_column),
        "Datetime",
        height=430,
    )


def create_monthly_figure(data: pd.DataFrame, value_column: str) -> go.Figure | None:
    series = monthly_series(data, value_column)
    if series.empty:
        return None
    return line_chart(
        series,
        "period",
        value_column,
        f"Monthly Trend - {display_label(value_column)}",
        axis_label(value_column),
        "Month",
        markers=True,
    )


def create_yearly_figure(data: pd.DataFrame, value_column: str) -> go.Figure | None:
    series = yearly_series(data, value_column)
    if series.empty:
        return None
    return line_chart(
        series,
        "period",
        value_column,
        f"Yearly Trend - {display_label(value_column)}",
        axis_label(value_column),
        "Year",
        markers=True,
    )


def create_aggregation_figure(
    data: pd.DataFrame,
    value_column: str,
    aggregation: str,
    rolling_window: int,
) -> go.Figure | None:
    series = aggregate_series(data, value_column, aggregation)
    if series.empty:
        return None

    series = series.rename(columns={"period": "datetime"})
    series = add_rolling_average(series, value_column, rolling_window)

    fig = go.Figure()
    station_order = list(series["station_label"].drop_duplicates())

    for idx, station in enumerate(station_order):
        station_frame = series[series["station_label"] == station]
        color = APP_CONFIG["colorway"][idx % len(APP_CONFIG["colorway"])]

        fig.add_trace(
            go.Scatter(
                x=station_frame["datetime"],
                y=station_frame[value_column],
                mode="lines+markers",
                line={"width": 2.2, "color": color},
                marker={"size": 6},
                opacity=0.35,
                name=f"{compact_station_label(station)} - {aggregation}",
                legendgroup=station,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=station_frame["datetime"],
                y=station_frame["rolling_value"],
                mode="lines",
                line={"width": 3.2, "color": color},
                name=f"{compact_station_label(station)} - Rolling",
                legendgroup=station,
            )
        )

    return chart_template(
        fig,
        f"{aggregation} Time Series - {display_label(value_column)}",
        axis_label(value_column),
        aggregation,
        height=430,
    )


def create_overlay_figure(
    combined: pd.DataFrame,
    pollutant_column: str,
    meteorology_column: str,
) -> go.Figure | None:
    if combined.empty or pollutant_column not in combined.columns or meteorology_column not in combined.columns:
        return None

    frame = combined[
        ["datetime", "station_label", pollutant_column, meteorology_column]
    ].copy().dropna(subset=[pollutant_column, meteorology_column])

    if frame.empty:
        return None

    station_order = list(frame["station_label"].drop_duplicates())
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for idx, station in enumerate(station_order):
        station_frame = frame[frame["station_label"] == station]
        color = APP_CONFIG["colorway"][idx % len(APP_CONFIG["colorway"])]

        fig.add_trace(
            go.Scatter(
                x=station_frame["datetime"],
                y=station_frame[pollutant_column],
                mode="lines",
                line={"width": 3, "color": color},
                fill="tozeroy",
                opacity=0.25,
                name=f"{compact_station_label(station)} - {display_label(pollutant_column)}",
                legendgroup=f"{station}-poll",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=station_frame["datetime"],
                y=station_frame[meteorology_column],
                mode="lines",
                line={"width": 2.4, "dash": "dot", "color": color},
                name=f"{compact_station_label(station)} - {display_label(meteorology_column)}",
                legendgroup=f"{station}-met",
            ),
            secondary_y=True,
        )

    fig.update_yaxes(title_text=axis_label(pollutant_column), secondary_y=False)
    fig.update_yaxes(title_text=axis_label(meteorology_column), secondary_y=True)

    return chart_template(
        fig,
        f"3-Day Overlay - {display_label(pollutant_column)} vs {display_label(meteorology_column)}",
        axis_label(pollutant_column),
        "Datetime",
        height=430,
    )


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
    fig.update_xaxes(title="Station")
    for trace in fig.data:
        if getattr(trace, "name", None):
            trace.name = compact_station_label(trace.name)
    return chart_template(
        fig,
        f"Station-wise Boxplot - {display_label(value_column)}",
        axis_label(value_column),
        "Station",
        height=420,
    )


def create_histogram_figure(data: pd.DataFrame, value_column: str) -> go.Figure | None:
    frame = select_value_frame(data, value_column)
    if frame.empty:
        return None

    fig = px.histogram(
        frame,
        x=value_column,
        color="station_label",
        marginal="rug",
        barmode="overlay",
        nbins=40,
        opacity=0.55,
        color_discrete_sequence=APP_CONFIG["colorway"],
    )
    for trace in fig.data:
        if getattr(trace, "name", None):
            trace.name = compact_station_label(trace.name)
    return chart_template(
        fig,
        f"Distribution Histogram - {display_label(value_column)}",
        axis_label(value_column),
        axis_label(value_column),
        height=420,
    )


def create_scatter_figure(
    combined: pd.DataFrame,
    pollutant_column: str,
    meteorology_column: str,
) -> go.Figure | None:
    if combined.empty:
        return None
    needed = ["station_label", pollutant_column, meteorology_column, "datetime"]
    frame = combined[needed].dropna()
    if frame.empty:
        return None

    frame = sample_scatter_data(frame)
    fig = px.scatter(
        frame,
        x=meteorology_column,
        y=pollutant_column,
        color="station_label",
        trendline="ols" if len(frame) >= 10 else None,
        hover_data=["datetime"],
        color_discrete_sequence=APP_CONFIG["colorway"],
    )
    for trace in fig.data:
        if getattr(trace, "name", None):
            trace.name = compact_station_label(trace.name)
    return chart_template(
        fig,
        f"Relationship - {display_label(pollutant_column)} vs {display_label(meteorology_column)}",
        axis_label(pollutant_column),
        axis_label(meteorology_column),
        height=450,
    )


def create_correlation_heatmap(
    combined: pd.DataFrame,
    selected_pollutant: str,
    selected_meteorology: str,
) -> go.Figure | None:
    cols = [c for c in [selected_pollutant, selected_meteorology] if c in combined.columns]
    if len(cols) < 2:
        return None

    numeric = combined[cols].dropna()
    if numeric.empty or len(numeric) < 2:
        return None

    corr = numeric.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    return chart_template(
        fig,
        "Correlation Heatmap",
        "",
        "",
        height=360,
    )


def render_chart(fig: go.Figure | None, empty_message: str) -> None:
    if fig is None:
        st.info(empty_message)
        return
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


def peak_metric(data: pd.DataFrame, value_column: str | None) -> tuple[str, str]:
    if value_column is None:
        return "No variable", "Select a pollutant or meteorology field"

    frame = select_value_frame(data, value_column)
    if frame.empty:
        return "No data", "No valid observations within the current filters"

    peak_row = frame.loc[frame[value_column].idxmax()]
    peak_value = f"{peak_row[value_column]:,.2f}"
    caption = (
        f"{display_label(value_column)} peaked on "
        f"{peak_row['datetime']:%Y-%m-%d %H:%M} at {compact_station_label(peak_row['station_label'])}"
    )
    return peak_value, caption


def render_header(
    station_meta: pd.DataFrame,
    available_pollutants: list[str],
    available_meteorology: list[str],
    min_ts: pd.Timestamp,
    max_ts: pd.Timestamp,
) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-eyebrow">Atmospheric Analytics</div>
            <h1>{APP_CONFIG["app_title"]}</h1>
            <p>{APP_CONFIG["app_subtitle"]}</p>
            <div class="hero-badges">
                <div class="badge">{len(station_meta)} stations</div>
                <div class="badge">{len(available_pollutants)} pollutants</div>
                <div class="badge">{len(available_meteorology)} meteorology fields</div>
                <div class="badge">{min_ts:%Y-%m-%d} to {max_ts:%Y-%m-%d}</div>
                <div class="badge">Hourly source resolution</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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

    metric_columns = st.columns(3)
    with metric_columns[0]:
        render_kpi_card("Availability", f"{availability:.1f}%", "Mean coverage across selected stations")
    with metric_columns[1]:
        render_kpi_card("Missing", f"{missing:.1f}%", "Average missing share across selected stations")
    with metric_columns[2]:
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
    st.dataframe(styled, use_container_width=True)
    return summary


def render_empty_relationships() -> None:
    st.info("Relationship analytics unlock when both a pollutant and a meteorology variable are selected.")


def render_dependency_error() -> None:
    missing_package = IMPORT_ERROR.name if IMPORT_ERROR is not None else "a required package"
    st.error(
        f"The dashboard cannot start because `{missing_package}` is not installed in this Python environment."
    )
    st.code("pip install -r requirements.txt")
    st.info("After installing the requirements, rerun: `streamlit run app.py`")


def sidebar_controls(
    station_options: list[str],
    available_pollutants: list[str],
    available_meteorology: list[str],
    defaults: dict[str, object],
    min_ts: pd.Timestamp,
    max_ts: pd.Timestamp,
) -> dict[str, object]:
    with st.sidebar:
        st.markdown('<div class="sidebar-card"><h3 style="margin:0;color:#fff;">Control Panel</h3><p style="margin:0.35rem 0 0 0;color:#cbd5e1;">Filter stations, variables, aggregation, and export your active view.</p></div>', unsafe_allow_html=True)

        search_text = st.text_input("🔍 Search station", key="station_search")

        filtered_station_options = [
            s for s in station_options if search_text.lower() in s.lower()
        ] if search_text else station_options

        selected_stations = st.multiselect(
            "✅ Stations",
            options=filtered_station_options,
            default=st.session_state.get("selected_stations", defaults["selected_stations"]),
            key="selected_stations",
        )

        pollutant_options = [None] + available_pollutants
        met_options = [None] + available_meteorology

        current_pollutant = st.session_state.get("selected_pollutant", defaults["selected_pollutant"])
        current_met = st.session_state.get("selected_meteorology", defaults["selected_meteorology"])

        selected_pollutant = st.selectbox(
            "🧪 Pollutant",
            options=pollutant_options,
            index=pollutant_options.index(current_pollutant) if current_pollutant in pollutant_options else 0,
            format_func=lambda x: "None" if x is None else display_label(x),
            key="selected_pollutant",
        )

        selected_meteorology = st.selectbox(
            "🌪 Meteorological Variable",
            options=met_options,
            index=met_options.index(current_met) if current_met in met_options else 0,
            format_func=lambda x: "None" if x is None else display_label(x),
            key="selected_meteorology",
        )

        aggregation = st.selectbox(
            "📊 Aggregation",
            options=list(AGGREGATION_OPTIONS.keys()),
            index=list(AGGREGATION_OPTIONS.keys()).index(st.session_state.get("aggregation", "Daily")),
            key="aggregation",
        )

        rolling_window = st.slider(
            "🎚 Rolling Average Window",
            min_value=1,
            max_value=14,
            value=int(st.session_state.get("rolling_window", 7)),
            key="rolling_window",
        )

        date_range = st.date_input(
            "📅 Date Range",
            value=st.session_state.get("date_range", defaults["date_range"]),
            min_value=min_ts.date(),
            max_value=max_ts.date(),
            key="date_range",
        )

        left, right = st.columns(2)
        with left:
            if st.button("Reset Filters", use_container_width=True):
                reset_filters(defaults)
                st.rerun()

        return {
            "selected_stations": selected_stations,
            "selected_pollutant": selected_pollutant,
            "selected_meteorology": selected_meteorology,
            "aggregation": aggregation,
            "rolling_window": rolling_window,
            "date_range": date_range,
        }


def render_top_kpis(
    selected_stations: list[str],
    selected_pollutant: str | None,
    selected_meteorology: str | None,
    peak_value: str,
    peak_caption: str,
) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        station_text = ", ".join(compact_station_label(s) for s in selected_stations[:3]) if selected_stations else "None"
        suffix = "" if len(selected_stations) <= 3 else f" +{len(selected_stations)-3} more"
        render_kpi_card("Selected Stations", station_text + suffix, f"{len(selected_stations)} station(s) in active comparison")
    with c2:
        render_kpi_card("Selected Pollutant", display_label(selected_pollutant), "Primary pollutant currently active")
    with c3:
        render_kpi_card("Peak Value", peak_value, peak_caption)
    with c4:
        render_kpi_card("Meteorology Variable", display_label(selected_meteorology), "Meteorological context channel")


def render_overview_tab(
    mode: str,
    pollutant_filtered: pd.DataFrame,
    meteorology_expanded: pd.DataFrame,
    combined_filtered: pd.DataFrame,
    selected_pollutant: str | None,
    selected_meteorology: str | None,
    rolling_window: int,
) -> None:
    st.markdown(
        '<p class="tab-note">Hero diurnal view, short-horizon trends, and long-horizon signal evolution.</p>',
        unsafe_allow_html=True,
    )

    if mode in {"pollutant", "combined"} and selected_pollutant:
        render_chart(
            create_diurnal_figure(pollutant_filtered, selected_pollutant),
            f"No diurnal plot is available for {display_label(selected_pollutant)}.",
        )
        left, right = st.columns(2)
        with left:
            render_chart(
                create_recent_line_figure(
                    pollutant_filtered,
                    selected_pollutant,
                    rolling_window,
                    f"3-Day Continuous Trend - {display_label(selected_pollutant)}",
                ),
                f"No recent trend is available for {display_label(selected_pollutant)}.",
            )
        with right:
            render_chart(
                create_monthly_figure(pollutant_filtered, selected_pollutant),
                f"No monthly trend is available for {display_label(selected_pollutant)}.",
            )
        render_chart(
            create_yearly_figure(pollutant_filtered, selected_pollutant),
            f"No yearly trend is available for {display_label(selected_pollutant)}.",
        )

    if mode in {"meteorology", "combined"} and selected_meteorology:
        st.markdown("---")
        render_chart(
            create_diurnal_figure(meteorology_expanded, selected_meteorology),
            f"No diurnal plot is available for {display_label(selected_meteorology)}.",
        )
        left, right = st.columns(2)
        with left:
            render_chart(
                create_recent_line_figure(
                    meteorology_expanded,
                    selected_meteorology,
                    rolling_window,
                    f"3-Day Continuous Trend - {display_label(selected_meteorology)}",
                ),
                f"No recent trend is available for {display_label(selected_meteorology)}.",
            )
        with right:
            render_chart(
                create_monthly_figure(meteorology_expanded, selected_meteorology),
                f"No monthly trend is available for {display_label(selected_meteorology)}.",
            )
        render_chart(
            create_yearly_figure(meteorology_expanded, selected_meteorology),
            f"No yearly trend is available for {display_label(selected_meteorology)}.",
        )

    if mode == "combined" and selected_pollutant and selected_meteorology:
        st.markdown("---")
        render_chart(
            create_overlay_figure(
                recent_source_window(combined_filtered, selected_pollutant, 72),
                selected_pollutant,
                selected_meteorology,
            ),
            "No overlay figure is available for the active filters.",
        )


def render_time_series_tab(
    mode: str,
    pollutant_filtered: pd.DataFrame,
    meteorology_expanded: pd.DataFrame,
    selected_pollutant: str | None,
    selected_meteorology: str | None,
    aggregation: str,
    rolling_window: int,
) -> None:
    st.markdown(
        '<p class="tab-note">Station-vs-station comparison with aggregation-aware trends and rolling smoothing.</p>',
        unsafe_allow_html=True,
    )

    if mode in {"pollutant", "combined"} and selected_pollutant:
        render_chart(
            create_aggregation_figure(pollutant_filtered, selected_pollutant, aggregation, rolling_window),
            f"No time-series chart is available for {display_label(selected_pollutant)}.",
        )

    if mode in {"meteorology", "combined"} and selected_meteorology:
        render_chart(
            create_aggregation_figure(meteorology_expanded, selected_meteorology, aggregation, rolling_window),
            f"No time-series chart is available for {display_label(selected_meteorology)}.",
        )


def render_distribution_tab(
    mode: str,
    pollutant_filtered: pd.DataFrame,
    meteorology_expanded: pd.DataFrame,
    selected_pollutant: str | None,
    selected_meteorology: str | None,
    selected_stations: list[str],
) -> None:
    st.markdown(
        '<p class="tab-note">Boxplots, histograms, and quantile summaries for the active variable selection.</p>',
        unsafe_allow_html=True,
    )

    if mode in {"pollutant", "combined"} and selected_pollutant:
        left, right = st.columns(2)
        with left:
            render_chart(
                create_boxplot_figure(pollutant_filtered, selected_pollutant),
                f"No boxplot is available for {display_label(selected_pollutant)}.",
            )
        with right:
            render_chart(
                create_histogram_figure(pollutant_filtered, selected_pollutant),
                f"No histogram is available for {display_label(selected_pollutant)}.",
            )
        render_quantile_table(pollutant_filtered, selected_pollutant, selected_stations)

        extremes = extreme_days_table(pollutant_filtered, selected_pollutant)
        if not extremes.empty:
            st.markdown("#### Extreme Event Days (>90th percentile of daily max)")
            st.dataframe(
                extremes.style.format(
                    {"daily_mean": "{:,.2f}", "daily_max": "{:,.2f}", "daily_min": "{:,.2f}", "valid_obs": "{:,.0f}"}
                ),
                use_container_width=True,
            )

    if mode in {"meteorology", "combined"} and selected_meteorology:
        st.markdown("---")
        left, right = st.columns(2)
        with left:
            render_chart(
                create_boxplot_figure(meteorology_expanded, selected_meteorology),
                f"No boxplot is available for {display_label(selected_meteorology)}.",
            )
        with right:
            render_chart(
                create_histogram_figure(meteorology_expanded, selected_meteorology),
                f"No histogram is available for {display_label(selected_meteorology)}.",
            )
        render_quantile_table(meteorology_expanded, selected_meteorology, selected_stations)


def render_relationships_tab(
    mode: str,
    combined_filtered: pd.DataFrame,
    selected_pollutant: str | None,
    selected_meteorology: str | None,
) -> None:
    st.markdown(
        '<p class="tab-note">Scatter, correlation, and simple trend diagnostics for pollutant-meteorology interaction.</p>',
        unsafe_allow_html=True,
    )

    if mode != "combined" or not selected_pollutant or not selected_meteorology:
        render_empty_relationships()
        return

    left, right = st.columns([1.4, 1.0])
    with left:
        render_chart(
            create_scatter_figure(combined_filtered, selected_pollutant, selected_meteorology),
            "No scatter plot is available for the active pair.",
        )
    with right:
        render_chart(
            create_correlation_heatmap(combined_filtered, selected_pollutant, selected_meteorology),
            "No correlation heatmap is available for the active pair.",
        )

    paired = combined_filtered[[selected_pollutant, selected_meteorology]].dropna()
    if len(paired) >= 2:
        corr = paired[selected_pollutant].corr(paired[selected_meteorology])
        relationship_cols = st.columns(3)
        with relationship_cols[0]:
            render_kpi_card(
                "Paired Records",
                f"{len(paired):,}",
                "Rows with both pollutant and meteorology values",
            )
        with relationship_cols[1]:
            render_kpi_card(
                "Correlation",
                f"{corr:.2f}" if pd.notna(corr) else "NA",
                "Pearson correlation for the active pair",
            )
        with relationship_cols[2]:
            slope, _ = np.polyfit(
                paired[selected_meteorology].to_numpy(),
                paired[selected_pollutant].to_numpy(),
                1,
            )
            render_kpi_card(
                "Regression Slope",
                f"{slope:.3f}",
                f"Change in {display_label(selected_pollutant)} per unit {display_label(selected_meteorology)}",
            )
    elif len(paired) == 1:
        st.info("Only one paired observation is available, so correlation and regression summaries are skipped.")


def render_quality_tab(
    mode: str,
    pollutant_filtered: pd.DataFrame,
    meteorology_expanded: pd.DataFrame,
    selected_pollutant: str | None,
    selected_meteorology: str | None,
    selected_stations: list[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> None:
    st.markdown(
        '<p class="tab-note">Coverage, missingness, completeness, and descriptive statistics for each selected channel.</p>',
        unsafe_allow_html=True,
    )

    quality_tables = []

    if mode in {"pollutant", "combined"} and selected_pollutant:
        quality_tables.append(
            render_quality_block(
                f"Pollutant Quality - {display_label(selected_pollutant)}",
                pollutant_filtered,
                selected_pollutant,
                selected_stations,
                start_ts,
                end_ts,
            )
        )

    if mode in {"meteorology", "combined"} and selected_meteorology:
        quality_tables.append(
            render_quality_block(
                f"Meteorology Quality - {display_label(selected_meteorology)}",
                meteorology_expanded,
                selected_meteorology,
                selected_stations,
                start_ts,
                end_ts,
            )
        )

    if quality_tables:
        combined_quality = pd.concat(quality_tables, ignore_index=True)
        st.markdown("#### Combined Quality Summary")
        styled = combined_quality.style.format(
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
        st.dataframe(styled, use_container_width=True)
    else:
        st.info("Select a pollutant or meteorology variable to inspect data quality.")


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

    defaults = default_state(
        station_options,
        available_pollutants,
        available_meteorology,
        min_ts,
        max_ts,
    )
    initialize_state(defaults)

    render_header(
        station_meta_df,
        available_pollutants,
        available_meteorology,
        min_ts,
        max_ts,
    )

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
    start_ts, end_ts = parse_date_range(controls["date_range"], min_ts, max_ts)

    mode = determine_mode(selected_pollutant, selected_meteorology)

    pollutant_filtered = filter_pollutant_data(
        pollutant_df, selected_stations, start_ts, end_ts
    )
    meteorology_filtered = filter_meteorology_data(meteorology_df, start_ts, end_ts)
    meteorology_expanded = expand_meteorology_for_stations(meteorology_filtered, selected_stations)
    combined_filtered = build_combined_dataset(pollutant_filtered, meteorology_filtered)

    peak_source = (
        pollutant_filtered if mode in {"pollutant", "combined"} and selected_pollutant else meteorology_expanded
    )
    peak_var = selected_pollutant if mode in {"pollutant", "combined"} and selected_pollutant else selected_meteorology
    peak_value, peak_caption = peak_metric(peak_source, peak_var)

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
        "📥 Download Filtered Dataset",
        data=download_bytes(download_df),
        file_name="filtered_dashboard_data.csv",
        mime="text/csv",
        use_container_width=False,
        disabled=download_df.empty,
    )

    if mode == "empty":
        st.warning("Select at least one pollutant or one meteorology variable to start the dashboard.")
        return

    overview_tab, time_tab, distribution_tab, relationships_tab, quality_tab = st.tabs(
        ["Overview", "Time Series", "Distribution", "Relationships", "Data Quality"]
    )

    with overview_tab:
        render_overview_tab(
            mode,
            pollutant_filtered,
            meteorology_expanded,
            combined_filtered,
            selected_pollutant,
            selected_meteorology,
            rolling_window,
        )

    with time_tab:
        render_time_series_tab(
            mode,
            pollutant_filtered,
            meteorology_expanded,
            selected_pollutant,
            selected_meteorology,
            aggregation,
            rolling_window,
        )

    with distribution_tab:
        render_distribution_tab(
            mode,
            pollutant_filtered,
            meteorology_expanded,
            selected_pollutant,
            selected_meteorology,
            selected_stations,
        )

    with relationships_tab:
        render_relationships_tab(
            mode,
            combined_filtered,
            selected_pollutant,
            selected_meteorology,
        )

    with quality_tab:
        render_quality_tab(
            mode,
            pollutant_filtered,
            meteorology_expanded,
            selected_pollutant,
            selected_meteorology,
            selected_stations,
            start_ts,
            end_ts,
        )


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
        st.error(f"Failed to load dashboard data: {exc}")
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
