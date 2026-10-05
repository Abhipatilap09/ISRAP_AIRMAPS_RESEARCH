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


def chart_template(fig: go.Figure, title: str, y_title: str, x_title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left", "y": 0.98, "yanchor": "top"},
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "color": "#0f172a"},
        colorway=APP_CONFIG["colorway"],
        height=height,
        margin={"l": 28, "r": 24, "t": 90, "b": 84},
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
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    kwargs = {}
    if row is not None and col is not None:
        kwargs = {"row": row, "col": col}
    fig.update_xaxes(tickformat=tickformat, hoverformat=hoverformat, **kwargs)
    return fig


def apply_temporal_axis(fig: go.Figure, aggregation: str) -> go.Figure:
    aggregation_key = aggregation.lower()
    if aggregation_key == "hourly":
        return datetime_axis(fig, tickformat="%d %H:%M", hoverformat="%Y-%m-%d %H:%M")
    if aggregation_key == "daily":
        return datetime_axis(fig, tickformat="%d %b", hoverformat="%Y-%m-%d")
    if aggregation_key == "monthly":
        fig.update_xaxes(dtick="M1", tickformat="%b %Y", hoverformat="%Y-%m")
        return fig
    if aggregation_key == "yearly":
        fig.update_xaxes(dtick="M12", tickformat="%Y", hoverformat="%Y")
        return fig
    return fig


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
    )
    fig = hourly_axis(fig)

    base_station = profile["station_label"].iloc[0]
    station_profile = profile[profile["station_label"] == base_station]
    completeness_pct = (station_profile[value_column].notna().sum() / 24) * 100
    fig = add_stats_overlay(fig, station_profile[value_column], completeness_pct)

    peak_row = station_profile.loc[station_profile[value_column].idxmax()]
    fig = add_peak_annotation(fig, peak_row["hour"], peak_row[value_column], "Peak")
    return fig


def create_single_day_figure(data: pd.DataFrame, value_column: str) -> go.Figure | None:
    frame = select_value_frame(data, value_column)
    if frame.empty:
        return None

    full_hours = pd.DataFrame({"hour": list(range(24))})
    station_frames = []

    for station, station_frame in frame.groupby("station_label", observed=True):
        station_frame = station_frame.copy()
        station_frame["hour"] = station_frame["datetime"].dt.hour
        hourly = (
            station_frame.groupby("hour", as_index=False)[value_column]
            .mean()
            .merge(full_hours, on="hour", how="right")
            .sort_values("hour")
        )
        hourly["station_label"] = station
        station_frames.append(hourly)

    hourly = pd.concat(station_frames, ignore_index=True)

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
    )
    fig = hourly_axis(fig)

    base_station = hourly["station_label"].dropna().iloc[0]
    station_hourly = hourly[hourly["station_label"] == base_station]
    completeness_pct = (station_hourly[value_column].notna().sum() / 24) * 100
    fig = add_stats_overlay(fig, station_hourly[value_column], completeness_pct)

    if station_hourly[value_column].notna().any():
        peak_row = station_hourly.loc[station_hourly[value_column].idxmax()]
        fig = add_peak_annotation(fig, peak_row["hour"], peak_row[value_column], "Peak")
    return fig


def create_three_day_figure(data: pd.DataFrame, value_column: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> go.Figure | None:
    frame = select_value_frame(data, value_column)
    if frame.empty:
        return None

    frame["day"] = frame["datetime"].dt.floor("D")
    frame["hour"] = frame["datetime"].dt.hour
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

    full_hours = pd.DataFrame({"hour": list(range(24))})
    station_order = list(frame["station_label"].drop_duplicates())

    for idx, station in enumerate(station_order):
        color = APP_CONFIG["colorway"][idx % len(APP_CONFIG["colorway"])]
        station_frame = frame[frame["station_label"] == station]

        for col_idx, day in enumerate(days, start=1):
            day_frame = station_frame[station_frame["day"] == day]
            hourly = (
                day_frame.groupby("hour", as_index=False)[value_column]
                .mean()
                .merge(full_hours, on="hour", how="right")
                .sort_values("hour")
            )

            fig.add_trace(
                go.Scatter(
                    x=hourly["hour"],
                    y=hourly[value_column],
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
    )

    for col_idx in range(1, 4):
        fig = hourly_axis(fig, row=1, col=col_idx)

    base_station = station_order[0]
    station_data = frame[frame["station_label"] == base_station]
    completeness_pct = (station_data[value_column].notna().sum() / (24 * 3)) * 100
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
    )

    base_station = daily["station_label"].iloc[0]
    station_daily = daily[daily["station_label"] == base_station]
    fig = add_stats_overlay(fig, station_daily[value_column], 100.0)
    peak_row = station_daily.loc[station_daily[value_column].idxmax()]
    fig = add_peak_annotation(fig, peak_row["date"], peak_row[value_column], "Peak")
    return fig


def create_monthly_hourly_figure(data: pd.DataFrame, value_column: str) -> go.Figure | None:
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
    )
    fig = datetime_axis(fig)

    base_station = frame["station_label"].iloc[0]
    station_frame = frame[frame["station_label"] == base_station]
    fig = add_stats_overlay(fig, station_frame[value_column], 100.0)
    peak_row = station_frame.loc[station_frame[value_column].idxmax()]
    fig = add_peak_annotation(fig, peak_row["datetime"], peak_row[value_column], "Peak")
    return fig


def create_monthly_daily_average_figure(data: pd.DataFrame, value_column: str) -> go.Figure | None:
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
    )

    base_station = daily["station_label"].iloc[0]
    station_daily = daily[daily["station_label"] == base_station]
    fig = add_stats_overlay(fig, station_daily[value_column], 100.0)
    peak_row = station_daily.loc[station_daily[value_column].idxmax()]
    fig = add_peak_annotation(fig, peak_row["date"], peak_row[value_column], "Peak")
    return fig


def create_yearly_month_average_figure(data: pd.DataFrame, value_column: str) -> go.Figure | None:
    frame = select_value_frame(data, value_column)
    if frame.empty:
        return None
    frame["month"] = frame["datetime"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        frame.groupby(["station_label", "month"], observed=True, as_index=False)[value_column]
        .mean()
        .sort_values(["station_label", "month"])
    )

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
    )
    fig.update_xaxes(dtick="M1", tickformat="%b")

    base_station = monthly["station_label"].iloc[0]
    station_monthly = monthly[monthly["station_label"] == base_station]
    fig = add_stats_overlay(fig, station_monthly[value_column], 100.0)
    peak_row = station_monthly.loc[station_monthly[value_column].idxmax()]
    fig = add_peak_annotation(fig, peak_row["month"], peak_row[value_column], "Peak")
    return fig


def create_recent_figure(data: pd.DataFrame, value_column: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> go.Figure | None:
    if is_single_day_selection(start_ts, end_ts):
        return create_single_day_figure(data, value_column)
    if is_three_day_selection(start_ts, end_ts):
        return create_three_day_figure(data, value_column, start_ts, end_ts)
    if is_fifteen_day_selection(start_ts, end_ts):
        return create_fifteen_day_figure(data, value_column)
    if is_full_month_selection(start_ts, end_ts):
        return create_monthly_hourly_figure(data, value_column)
    if is_full_year_selection(start_ts, end_ts):
        return create_yearly_month_average_figure(data, value_column)

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
    )
    fig = datetime_axis(fig)
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

    for idx, station in enumerate(station_order):
        station_frame = frame[frame["station_label"] == station]
        color = APP_CONFIG["colorway"][idx % len(APP_CONFIG["colorway"])]

        fig.add_trace(
            go.Scatter(
                x=station_frame["period"],
                y=station_frame[value_column],
                mode="lines+markers",
                line={"color": color, "width": 2.5},
                marker={"size": 6},
                name=compact_station_label(station),
                legendgroup=station,
                opacity=0.30 if rolling_window > 1 else 0.95,
            )
        )

        if rolling_window > 1:
            fig.add_trace(
                go.Scatter(
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
        if not outliers.empty:
            fig.add_trace(
                go.Scatter(
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
    )
    fig = apply_temporal_axis(fig, aggregation)

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

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for idx, station in enumerate(pollutant_frame["station_label"].drop_duplicates()):
        station_frame = pollutant_frame[pollutant_frame["station_label"] == station]
        color = APP_CONFIG["colorway"][idx % len(APP_CONFIG["colorway"])]
        fig.add_trace(
            go.Scatter(
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
        go.Scatter(
            x=met_frame["period"],
            y=met_frame["rolling"] if rolling_window > 1 else met_frame[meteorology_column],
            mode="lines",
            line={"color": "#0f172a", "width": 3, "dash": "dot"},
            name=display_label(meteorology_column),
        ),
        secondary_y=True,
    )

    fig = chart_template(fig, title, axis_label(pollutant_column), aggregation, height=460)
    fig.update_yaxes(title_text=axis_label(pollutant_column), secondary_y=False)
    fig.update_yaxes(title_text=axis_label(meteorology_column), secondary_y=True)
    fig = apply_temporal_axis(fig, aggregation)
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
    return chart_template(fig, "Correlation Heatmap", "", "", height=440)


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
    fig = chart_template(fig, f"Completeness Matrix - {display_label(value_column)}", "", granularity, height=380)
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
    st.dataframe(styled, use_container_width=True)
    return summary


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
        label="📥 Download filtered dataset",
        data=download_bytes(download_df),
        file_name="filtered_air_quality_dashboard.csv",
        mime="text/csv",
        disabled=download_df.empty,
    )

    overview_tab, timeseries_tab, distribution_tab, relationships_tab, quality_tab = st.tabs(
        ["Overview", "Time Series", "Distribution", "Relationships", "Data Quality"]
    )

    with overview_tab:
        st.markdown(
            '<p class="tab-note">Adaptive logic: 1 day = hourly 00:00–23:00, 3 days = 3 hourly panels, 15 days = daily average, full month = monthly views, full year = yearly monthly average.</p>',
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
            col1, col2 = st.columns(2)
            with col1:
                render_chart(
                    create_monthly_daily_average_figure(pollutant_filtered, selected_pollutant),
                    "No monthly daily-average trend is available.",
                )
            with col2:
                render_chart(
                    create_yearly_month_average_figure(pollutant_filtered, selected_pollutant),
                    "No yearly trend is available.",
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
            col1, col2 = st.columns(2)
            with col1:
                render_chart(
                    create_monthly_daily_average_figure(meteorology_expanded, selected_meteorology),
                    "No monthly daily-average meteorology trend is available.",
                )
            with col2:
                render_chart(
                    create_yearly_month_average_figure(meteorology_expanded, selected_meteorology),
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

            col1, col2 = st.columns(2)
            with col1:
                render_chart(
                    create_monthly_daily_average_figure(pollutant_filtered, selected_pollutant),
                    "No monthly pollutant trend is available.",
                )
            with col2:
                render_chart(
                    create_monthly_daily_average_figure(meteorology_expanded, selected_meteorology),
                    "No monthly meteorology trend is available.",
                )
        else:
            st.info("Select a pollutant, a meteorology variable, or both.")

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
                render_chart(create_boxplot_figure(pollutant_filtered, selected_pollutant), "No pollutant boxplot is available.")
            with c2:
                render_chart(create_histogram_figure(pollutant_filtered, selected_pollutant), "No pollutant histogram is available.")
            render_quantile_table(pollutant_filtered, selected_pollutant, selected_stations)

        if mode in {"meteorology", "combined"} and selected_meteorology:
            st.markdown(f"#### {display_label(selected_meteorology)}")
            c1, c2 = st.columns(2)
            with c1:
                render_chart(create_boxplot_figure(meteorology_expanded, selected_meteorology), "No meteorology boxplot is available.")
            with c2:
                render_chart(create_histogram_figure(meteorology_expanded, selected_meteorology), "No meteorology histogram is available.")
            render_quantile_table(meteorology_expanded, selected_meteorology, selected_stations)

    with relationships_tab:
        st.markdown(
            '<p class="tab-note">Scatter, regression, and correlation heatmaps help explore pollutant–meteorology relationships.</p>',
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

