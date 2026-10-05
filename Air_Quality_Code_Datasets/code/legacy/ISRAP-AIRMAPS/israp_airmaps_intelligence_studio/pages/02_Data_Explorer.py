"""Data Explorer — comprehensive interactive time-series viewer for EPA AQS pollutant data."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.config import (
    CHART_DOWNSAMPLE,
    ERA5_VAR_LABELS,
    ERA5_VARS,
    POLLUTANTS,
    STATIONS,
    STRUCTURAL_MISSING_THRESHOLD,
)
from src.data_loader import load_all_stations, load_era5, merge_era5
from ui.theme import (
    ANOMALY_COLOR,
    COLORS,
    IMPUTED_COLOR,
    MISSING_COLOR,
    OBSERVED_COLOR,
    SERIES_COLORS,
    inject_css,
    info_box,
    metric_card,
)

CHART_H = 500
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ── Cached data loading ───────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading station data…")
def _load() -> dict[str, pd.DataFrame]:
    """Load all stations merged with ERA5; result is cached across reruns."""
    stations = load_all_stations()
    era5 = load_era5()
    return {k: merge_era5(v, era5) for k, v in stations.items()}


# ── Utility functions ─────────────────────────────────────────────────────────

def _downsample(df: pd.DataFrame, max_pts: int = CHART_DOWNSAMPLE) -> pd.DataFrame:
    """Evenly thin a DataFrame to at most *max_pts* rows for Plotly rendering."""
    if len(df) <= max_pts:
        return df
    step = max(1, len(df) // max_pts)
    return df.iloc[::step].copy()


def _detect_zscore_anomalies(
    series: pd.Series, window: int = 72, threshold: float = 3.0
) -> pd.Series:
    """Boolean mask: True where rolling z-score exceeds *threshold*."""
    min_pts = max(1, window // 4)
    if series.notna().sum() < min_pts:
        return pd.Series(False, index=series.index)
    rm = series.rolling(window, center=True, min_periods=min_pts).mean()
    rs = series.rolling(window, center=True, min_periods=min_pts).std()
    z = (series - rm) / rs.replace(0.0, np.nan)
    return z.abs() > threshold


def _missing_spans(
    datetimes: pd.Series, is_nan: pd.Series
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return (start, end) pairs for every contiguous block of NaN rows."""
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    in_span = False
    start: pd.Timestamp | None = None
    for dt, nan in zip(datetimes, is_nan):
        if nan and not in_span:
            in_span, start = True, dt
        elif not nan and in_span:
            in_span = False
            spans.append((start, dt))  # type: ignore[arg-type]
    if in_span and start is not None:
        spans.append((start, datetimes.iloc[-1]))
    return spans


def _aggregate_df(df: pd.DataFrame, pollutant: str, agg: str) -> pd.DataFrame:
    """Resample to Daily or Weekly mean. Hourly returns df unchanged."""
    if agg == "Hourly":
        return df
    freq = "D" if agg == "Daily" else "W"
    era5_num = [c for c in ERA5_VARS if c in df.columns]
    num_cols = [c for c in ([pollutant] + era5_num) if c in df.columns]
    return df.set_index("datetime")[num_cols].resample(freq).mean().reset_index()


# ── Tab 1: Raw Series ─────────────────────────────────────────────────────────

def _tab_raw_series(
    df: pd.DataFrame,
    pollutant: str,
    unit: str,
    station_key: str,
    station_label: str,
    station_short: str,
    show_missing: bool,
    show_anomalies: bool,
    show_imputed: bool,
    compare_df: pd.DataFrame | None,
    compare_short: str | None,
) -> None:
    col = df[pollutant]

    # Inline rolling z-score anomaly detection
    anom_mask = pd.Series(False, index=df.index)
    if show_anomalies and col.notna().sum() >= 72:
        anom_mask = _detect_zscore_anomalies(col)

    ds = _downsample(df)
    fig = go.Figure()

    # ── Primary pollutant series ───────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=ds["datetime"], y=ds[pollutant],
        mode="lines",
        name=f"{pollutant} — {station_short}",
        line=dict(color=OBSERVED_COLOR, width=1.5),
        hovertemplate=(
            f"<b>%{{x|%Y-%m-%d %H:%M}}</b><br>"
            f"{pollutant}: %{{y:.4f}} {unit}<extra></extra>"
        ),
    ))

    # ── Negative value markers (red down-triangles) ────────────────────────────
    neg_mask = col < 0
    if neg_mask.any():
        neg_df = df.loc[neg_mask].head(2000)
        fig.add_trace(go.Scatter(
            x=neg_df["datetime"], y=neg_df[pollutant],
            mode="markers", name="Negative (invalid)",
            marker=dict(color=ANOMALY_COLOR, size=7, symbol="triangle-down"),
            hovertemplate=(
                f"%{{x|%Y-%m-%d %H:%M}}<br>"
                f"{pollutant}: %{{y:.4f}} {unit}<extra>Invalid</extra>"
            ),
        ))

    # ── Anomaly markers (amber ×) ──────────────────────────────────────────────
    if show_anomalies and anom_mask.any():
        anom_df = df.loc[anom_mask & col.notna()].head(2000)
        fig.add_trace(go.Scatter(
            x=anom_df["datetime"], y=anom_df[pollutant],
            mode="markers", name="Anomaly (|z| > 3)",
            marker=dict(
                color=COLORS["amber"], size=7,
                symbol="x-thin-open", line=dict(width=2),
            ),
            hovertemplate=(
                f"%{{x|%Y-%m-%d %H:%M}}<br>"
                f"{pollutant}: %{{y:.4f}} {unit}<extra>Anomaly</extra>"
            ),
        ))

    # ── Imputed values (green circles) ────────────────────────────────────────
    imp_col = f"{pollutant}_imputed"
    if show_imputed:
        if imp_col in df.columns and df[imp_col].notna().any():
            imp_ds = _downsample(df.loc[df[imp_col].notna()])
            fig.add_trace(go.Scatter(
                x=imp_ds["datetime"], y=imp_ds[imp_col],
                mode="markers", name="Imputed",
                marker=dict(color=IMPUTED_COLOR, size=4, symbol="circle"),
                hovertemplate=(
                    f"%{{x|%Y-%m-%d %H:%M}}<br>"
                    f"Imputed: %{{y:.4f}} {unit}<extra></extra>"
                ),
            ))
        else:
            st.caption(
                f"No '{imp_col}' column found. Run the imputation pipeline first "
                f"to see imputed values here."
            )

    # ── Comparison station trace ───────────────────────────────────────────────
    if compare_df is not None and compare_short and pollutant in compare_df.columns:
        comp_ds = _downsample(compare_df)
        fig.add_trace(go.Scatter(
            x=comp_ds["datetime"], y=comp_ds[pollutant],
            mode="lines", name=f"{pollutant} — {compare_short}",
            line=dict(color=SERIES_COLORS[1], width=1.5, dash="dash"),
            hovertemplate=(
                f"%{{x|%Y-%m-%d %H:%M}}<br>"
                f"{pollutant}: %{{y:.4f}} {unit}<extra>{compare_short}</extra>"
            ),
        ))

    # ── Missing period background shading ─────────────────────────────────────
    if show_missing:
        for s0, s1 in _missing_spans(df["datetime"], col.isna())[:200]:
            fig.add_vrect(
                x0=s0, x1=s1,
                fillcolor="rgba(141,153,174,0.15)",
                line_width=0, layer="below",
            )

    fig.update_layout(
        template="plotly_white",
        height=CHART_H,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=70, r=20, t=30, b=70),
        yaxis=dict(title=f"{pollutant} ({unit})"),
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=7,  label="1W", step="day",   stepmode="backward"),
                    dict(count=1,  label="1M", step="month", stepmode="backward"),
                    dict(count=3,  label="3M", step="month", stepmode="backward"),
                    dict(count=1,  label="1Y", step="year",  stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor=COLORS["bg"],
                activecolor=COLORS["teal"],
                font=dict(size=11),
            ),
            rangeslider=dict(visible=True, thickness=0.06),
            type="date",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Quality callouts ───────────────────────────────────────────────────────
    neg_pct  = 100.0 * float(neg_mask.mean())
    miss_pct = 100.0 * float(col.isna().mean())

    if station_key == "station3" and pollutant == "SO2":
        info_box(
            "<b>Station 3 SO2 — Systematic Corruption:</b> "
            "~33% of readings are negative due to sensor malfunction. "
            "This pollutant should be excluded from imputation modeling.",
            kind="danger",
        )
    elif neg_pct > 5:
        info_box(
            f"<b>High negative-value rate ({neg_pct:.1f}%):</b> "
            f"A large proportion of {pollutant} readings are physically invalid (< 0). "
            "Investigate sensor health before using this data for analysis.",
            kind="danger",
        )
    elif miss_pct > 20:
        info_box(
            f"<b>High missingness ({miss_pct:.1f}%):</b> "
            f"Over 20% of {pollutant} readings are absent in this window. "
            "Consider examining gap patterns in the Data Quality page before imputation.",
            kind="warn",
        )
    elif neg_pct < 0.5 and miss_pct < 5:
        info_box(
            f"<b>Good data quality:</b> {pollutant} at {station_label} shows "
            f"minimal negatives ({neg_pct:.2f}%) and low missingness ({miss_pct:.1f}%).",
            kind="info",
        )


# ── Tab 2: ERA5 Context ───────────────────────────────────────────────────────

def _tab_era5_context(
    df: pd.DataFrame, pollutant: str, unit: str, sel_era5: list[str]
) -> None:
    available = [v for v in sel_era5 if v in df.columns and df[v].notna().any()]
    if not available:
        st.info(
            "No ERA5 variables are available for this date range. "
            "Select one or more ERA5 variables in the sidebar."
        )
        return

    ds = _downsample(df)
    n = len(available)
    row_heights = [0.55] + [round(0.45 / n, 3)] * n

    titles = [f"{pollutant} ({unit})"] + [ERA5_VAR_LABELS.get(v, v) for v in available]
    fig = make_subplots(
        rows=1 + n, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.04,
        subplot_titles=titles,
    )

    # Pollutant (row 1)
    fig.add_trace(go.Scatter(
        x=ds["datetime"], y=ds[pollutant],
        mode="lines", name=pollutant,
        line=dict(color=OBSERVED_COLOR, width=1.8),
        hovertemplate=(
            f"%{{x|%Y-%m-%d %H:%M}}<br>{pollutant}: %{{y:.4f}} {unit}<extra></extra>"
        ),
    ), row=1, col=1)
    fig.update_yaxes(title_text=f"{pollutant} ({unit})", row=1, col=1)

    # ERA5 variables (rows 2…)
    for i, var in enumerate(available, start=2):
        label = ERA5_VAR_LABELS.get(var, var)
        fig.add_trace(go.Scatter(
            x=ds["datetime"], y=ds[var],
            mode="lines", name=label,
            line=dict(color=SERIES_COLORS[i % len(SERIES_COLORS)], width=1.2),
            hovertemplate=f"%{{x|%Y-%m-%d %H:%M}}<br>{label}: %{{y:.3f}}<extra></extra>",
        ), row=i, col=1)
        fig.update_yaxes(title_text=label, row=i, col=1)

    fig.update_layout(
        template="plotly_white",
        height=max(CHART_H, 280 + 150 * n),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=80, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Pearson correlation table
    corr_rows = []
    for var in available:
        pair = df[[pollutant, var]].dropna()
        if len(pair) >= 10:
            r = pair[pollutant].corr(pair[var])
            corr_rows.append({
                "ERA5 Variable": ERA5_VAR_LABELS.get(var, var),
                "Pearson r": round(r, 4),
                "N pairs": len(pair),
            })
    if corr_rows:
        st.markdown("**Pearson correlation with ERA5 variables (selected date window)**")
        corr_df = pd.DataFrame(corr_rows).sort_values("Pearson r", key=abs, ascending=False)
        try:
            st.dataframe(
                corr_df.style.background_gradient(
                    subset=["Pearson r"], cmap="RdBu_r", vmin=-1, vmax=1
                ),
                use_container_width=True, hide_index=True,
            )
        except Exception:
            st.dataframe(corr_df, use_container_width=True, hide_index=True)


# ── Tab 3: Station Comparison ─────────────────────────────────────────────────

def _tab_station_comparison(
    station_data: dict[str, pd.DataFrame],
    pollutant: str,
    unit: str,
    d_from: datetime.date,
    d_to: datetime.date,
) -> None:
    eligible = [
        k for k, smeta in STATIONS.items()
        if pollutant in smeta["monitored_pollutants"]
        and k in station_data
        and pollutant in station_data[k].columns
        and station_data[k][pollutant].notna().mean() > STRUCTURAL_MISSING_THRESHOLD
    ]

    if not eligible:
        st.info(f"No stations have valid {pollutant} data available for comparison.")
        return

    fig = go.Figure()
    table_rows = []

    for i, sk in enumerate(eligible):
        smeta = STATIONS[sk]
        sdf = station_data[sk]
        mask = (sdf["datetime"].dt.date >= d_from) & (sdf["datetime"].dt.date <= d_to)
        dfc = sdf.loc[mask]
        if dfc.empty:
            continue
        ds = _downsample(dfc)
        col_c = dfc[pollutant]
        color = SERIES_COLORS[i % len(SERIES_COLORS)]

        fig.add_trace(go.Scatter(
            x=ds["datetime"], y=ds[pollutant],
            mode="lines", name=smeta["short"],
            line=dict(color=color, width=1.5),
            hovertemplate=(
                f"%{{x|%Y-%m-%d %H:%M}}<br>"
                f"{smeta['short']}: %{{y:.4f}} {unit}<extra></extra>"
            ),
        ))

        table_rows.append({
            "Station":    smeta["label"],
            "N Valid":    f"{col_c.notna().sum():,}",
            "Missing %":  f"{100 * col_c.isna().mean():.1f}%",
            "Mean":       f"{col_c.mean():.4f}"   if col_c.notna().any() else "—",
            "Std":        f"{col_c.std():.4f}"    if col_c.notna().any() else "—",
            "Min":        f"{col_c.min():.4f}"    if col_c.notna().any() else "—",
            "Max":        f"{col_c.max():.4f}"    if col_c.notna().any() else "—",
        })

    fig.update_layout(
        template="plotly_white",
        height=CHART_H,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=70, r=20, t=50, b=50),
        yaxis=dict(title=f"{pollutant} ({unit})"),
        xaxis=dict(
            rangeselector=dict(buttons=[
                dict(count=7, label="1W", step="day",   stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(step="all", label="All"),
            ]),
            rangeslider=dict(visible=True, thickness=0.05),
            type="date",
        ),
        title=dict(text=f"{pollutant} — Cross-Station Comparison", x=0.5, font_size=14),
    )
    st.plotly_chart(fig, use_container_width=True)

    if table_rows:
        st.dataframe(
            pd.DataFrame(table_rows), use_container_width=True, hide_index=True
        )


# ── Tab 4: Seasonality ────────────────────────────────────────────────────────

def _tab_seasonality(df: pd.DataFrame, pollutant: str, unit: str) -> None:
    if df[pollutant].notna().sum() < 48:
        st.info("Insufficient valid readings for seasonality analysis (need ≥ 48).")
        return

    ds = df[["datetime", pollutant]].dropna(subset=[pollutant]).copy()
    ds["month"] = ds["datetime"].dt.month
    ds["hour"]  = ds["datetime"].dt.hour

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Monthly Distribution", "Diurnal (Hour-of-Day) Distribution"],
        horizontal_spacing=0.10,
    )

    for m in range(1, 13):
        mdata = ds.loc[ds["month"] == m, pollutant]
        if mdata.notna().any():
            fig.add_trace(go.Box(
                y=mdata, name=MONTH_ABBR[m - 1],
                marker_color=COLORS["teal"], line_color=COLORS["navy"],
                boxmean="sd", showlegend=False,
                hovertemplate=f"{MONTH_ABBR[m - 1]}<br>%{{y:.4f}} {unit}<extra></extra>",
            ), row=1, col=1)

    for h in range(24):
        hdata = ds.loc[ds["hour"] == h, pollutant]
        if hdata.notna().any():
            fig.add_trace(go.Box(
                y=hdata, name=f"{h:02d}:00",
                marker_color=COLORS["blue"], line_color=COLORS["navy"],
                boxmean="sd", showlegend=False,
                hovertemplate=f"{h:02d}:00<br>%{{y:.4f}} {unit}<extra></extra>",
            ), row=1, col=2)

    fig.update_yaxes(title_text=f"{pollutant} ({unit})", row=1, col=1)
    fig.update_yaxes(title_text=f"{pollutant} ({unit})", row=1, col=2)
    fig.update_xaxes(title_text="Month",              row=1, col=1)
    fig.update_xaxes(title_text="Hour of Day (UTC)",  row=1, col=2)
    fig.update_layout(
        template="plotly_white",
        height=CHART_H,
        showlegend=False,
        margin=dict(l=70, r=20, t=60, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Monthly statistics table"):
        mstats = (
            ds.groupby("month")[pollutant]
            .agg(
                Count="count",
                Mean="mean",
                Std="std",
                P25=lambda x: x.quantile(0.25),
                Median="median",
                P75=lambda x: x.quantile(0.75),
            )
            .round(4)
            .reset_index()
        )
        mstats["Month"] = mstats["month"].apply(lambda m: MONTH_ABBR[m - 1])
        st.dataframe(
            mstats[["Month", "Count", "Mean", "Std", "P25", "Median", "P75"]],
            use_container_width=True, hide_index=True,
        )


# ── Tab 5: Distribution ───────────────────────────────────────────────────────

def _tab_distribution(df: pd.DataFrame, pollutant: str, unit: str) -> None:
    col = df[pollutant].dropna()
    if len(col) < 10:
        st.info("Not enough valid readings for distribution analysis.")
        return

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Histogram", "Cumulative Distribution (ECDF)", "Violin"],
        horizontal_spacing=0.10,
    )

    # Histogram
    fig.add_trace(go.Histogram(
        x=col, nbinsx=60,
        name="Count",
        marker_color=COLORS["blue"], opacity=0.8,
        hovertemplate=f"Bin: %{{x}}<br>Count: %{{y}}<extra></extra>",
    ), row=1, col=1)

    # ECDF
    sv   = col.sort_values().values
    ecdf = np.arange(1, len(sv) + 1) / len(sv)
    fig.add_trace(go.Scatter(
        x=sv, y=ecdf,
        mode="lines", name="ECDF",
        line=dict(color=COLORS["teal"], width=2),
        hovertemplate=(
            f"{pollutant}: %{{x:.4f}}<br>P(X ≤ x): %{{y:.3f}}<extra></extra>"
        ),
    ), row=1, col=2)

    # Violin
    fig.add_trace(go.Violin(
        y=col, name=pollutant,
        box_visible=True, meanline_visible=True,
        fillcolor=COLORS["teal_lt"], line_color=COLORS["navy"], opacity=0.75,
        hovertemplate=f"{pollutant}: %{{y:.4f}} {unit}<extra></extra>",
    ), row=1, col=3)

    fig.update_xaxes(title_text=f"{pollutant} ({unit})", row=1, col=1)
    fig.update_yaxes(title_text="Count",                 row=1, col=1)
    fig.update_xaxes(title_text=f"{pollutant} ({unit})", row=1, col=2)
    fig.update_yaxes(title_text="Cumulative Fraction",   row=1, col=2)
    fig.update_yaxes(title_text=f"{pollutant} ({unit})", row=1, col=3)

    fig.update_layout(
        template="plotly_white",
        height=CHART_H,
        showlegend=False,
        margin=dict(l=70, r=20, t=60, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Percentile summary row
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_row = {f"P{p}": round(float(col.quantile(p / 100)), 4) for p in pcts}
    pct_row["Mean"] = round(float(col.mean()), 4)
    pct_row["Std"]  = round(float(col.std()),  4)
    st.dataframe(pd.DataFrame([pct_row]), use_container_width=True, hide_index=True)


# ── Summary stats + monthly completeness heatmap ──────────────────────────────

def _render_summary_stats(df: pd.DataFrame, pollutant: str, unit: str) -> None:
    col      = df[pollutant]
    valid_n  = int(col.notna().sum())
    miss_n   = int(col.isna().sum())
    miss_pct = 100.0 * col.isna().mean()
    neg_n    = int((col < 0).sum())
    dmin = round(float(col.min()), 4) if col.notna().any() else float("nan")
    dmax = round(float(col.max()), 4) if col.notna().any() else float("nan")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card(
        "Valid Readings", f"{valid_n:,}",
        delta=f"{100.0 - miss_pct:.1f}% completeness",
        color=COLORS["teal"], icon="✓",
    ), unsafe_allow_html=True)
    c2.markdown(metric_card(
        "Missing Readings", f"{miss_n:,}",
        delta=f"{miss_pct:.1f}% of total",
        color=COLORS["amber"] if miss_pct > 10 else COLORS["teal"], icon="○",
    ), unsafe_allow_html=True)
    c3.markdown(metric_card(
        "Negative Values", f"{neg_n:,}",
        delta="Physically invalid" if neg_n > 0 else "None detected",
        color=COLORS["red"] if neg_n > 100 else COLORS["teal"], icon="▽",
    ), unsafe_allow_html=True)
    c4.markdown(metric_card(
        "Data Range",
        f"{dmin:.3f} – {dmax:.3f}",
        delta=unit,
        color=COLORS["navy"], icon="↕",
    ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Monthly completeness heatmap
    if "datetime" not in df.columns or not col.notna().any():
        return

    df_h = df[["datetime", pollutant]].copy()
    df_h["year"]  = df_h["datetime"].dt.year
    df_h["month"] = df_h["datetime"].dt.month
    df_h["_one"]  = 1

    grp      = df_h.groupby(["year", "month"])
    totals   = grp["_one"].sum()
    valids   = grp[pollutant].count()
    complete = (100.0 * valids / totals.replace(0, np.nan)).reset_index()
    complete.columns = ["year", "month", "completeness"]

    if complete.empty:
        return

    pivot    = complete.pivot(index="year", columns="month", values="completeness")
    x_labels = [MONTH_ABBR[m - 1] for m in pivot.columns]
    y_labels = [str(y) for y in pivot.index]
    z_vals   = pivot.values
    text_vals = [
        [f"{v:.0f}%" if not np.isnan(v) else "—" for v in row]
        for row in z_vals
    ]

    fig_hm = go.Figure(go.Heatmap(
        z=z_vals,
        x=x_labels, y=y_labels,
        colorscale=[
            [0.00, "#fdecea"],
            [0.40, "#fff3cd"],
            [0.75, "#d4edda"],
            [1.00, "#0a9396"],
        ],
        zmin=0, zmax=100,
        colorbar=dict(title="%", ticksuffix="%", len=0.8),
        hovertemplate="<b>%{y} %{x}</b><br>Completeness: %{z:.1f}%<extra></extra>",
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=9, color="black"),
    ))
    fig_hm.update_layout(
        template="plotly_white",
        title=dict(text="Monthly Data Completeness Heatmap", x=0.5, font_size=13),
        height=max(200, 70 + 50 * len(y_labels)),
        margin=dict(l=60, r=110, t=50, b=40),
        xaxis=dict(side="bottom"),
    )
    st.plotly_chart(fig_hm, use_container_width=True)


# ── Point inspector ───────────────────────────────────────────────────────────

def _render_point_inspector(
    df: pd.DataFrame, pollutant: str, unit: str, sel_era5: list[str]
) -> None:
    if df.empty or "datetime" not in df.columns:
        st.info("No data available for the point inspector.")
        return

    dt_min = df["datetime"].min().to_pydatetime()
    dt_max = df["datetime"].max().to_pydatetime()

    col_a, col_b = st.columns([1, 3])
    with col_a:
        sel_date = st.date_input(
            "Date",
            value=dt_min.date(),
            min_value=dt_min.date(),
            max_value=dt_max.date(),
            key="pi_date",
        )
        sel_hour = st.selectbox("Hour (UTC)", list(range(24)), key="pi_hour")

    target = pd.Timestamp(sel_date) + pd.Timedelta(hours=int(sel_hour))
    idx    = (df["datetime"] - target).abs().idxmin()
    row    = df.loc[idx]
    val    = row[pollutant]

    with col_b:
        if pd.isna(val):
            st.warning(
                f"Reading at **{row['datetime'].strftime('%Y-%m-%d %H:%M')}** is missing."
            )
        else:
            # Local z-score within ±36-hour window
            w = df.loc[
                (df["datetime"] >= target - pd.Timedelta(hours=36)) &
                (df["datetime"] <= target + pd.Timedelta(hours=36)),
                pollutant,
            ].dropna()
            delta_str = None
            if len(w) > 5:
                std = w.std()
                z   = abs(val - w.mean()) / std if std > 0 else 0.0
                delta_str = f"Local z-score: {z:.2f}"
            st.metric(
                label=f"{pollutant} at {row['datetime'].strftime('%Y-%m-%d %H:%M')}",
                value=f"{val:.4f} {unit}",
                delta=delta_str,
            )

    # ±12-hour context window chart
    win = df.loc[
        (df["datetime"] >= target - pd.Timedelta(hours=12)) &
        (df["datetime"] <= target + pd.Timedelta(hours=12))
    ]
    if not win.empty and win[pollutant].notna().any():
        fig24 = go.Figure()
        fig24.add_trace(go.Scatter(
            x=win["datetime"], y=win[pollutant],
            mode="lines+markers", name=pollutant,
            line=dict(color=OBSERVED_COLOR, width=2),
            marker=dict(size=5),
        ))
        fig24.add_vline(
            x=target,
            line_dash="dash", line_color=COLORS["red"],
            annotation_text="Selected",
            annotation_position="top right",
        )
        fig24.update_layout(
            template="plotly_white",
            height=270,
            margin=dict(l=60, r=20, t=40, b=40),
            yaxis=dict(title=f"{pollutant} ({unit})"),
            xaxis=dict(title="Time (UTC)"),
            title=dict(text="±12-Hour Context Window", x=0.5, font_size=12),
        )
        st.plotly_chart(fig24, use_container_width=True)
    else:
        st.info("No valid readings in the ±12-hour window around the selected time.")

    # ERA5 conditions at selected row
    era5_present = [
        v for v in sel_era5
        if v in df.columns and pd.notna(row.get(v, np.nan))
    ]
    if era5_present:
        st.markdown("**ERA5 Conditions at Selected Time**")
        era5_vals = {
            ERA5_VAR_LABELS.get(v, v): f"{row[v]:.3f}"
            for v in era5_present
        }
        era5_tbl = (
            pd.DataFrame.from_dict(era5_vals, orient="index", columns=["Value"])
            .reset_index()
            .rename(columns={"index": "Variable"})
        )
        st.dataframe(era5_tbl, use_container_width=True, hide_index=True)
    elif sel_era5:
        st.caption("ERA5 data not available at the selected timestamp.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Data Explorer | ISRAP AIRMAPS",
        layout="wide",
        page_icon="🔭",
        initial_sidebar_state="expanded",
    )
    inject_css()

    navy  = COLORS["navy"]
    muted = COLORS["muted"]
    st.markdown(
        f"<h1 style='color:{navy};margin-bottom:2px'>🔭 Data Explorer</h1>"
        f"<p style='color:{muted};margin-top:0;margin-bottom:18px'>"
        "Interactive time-series viewer for EPA AQS station pollutant measurements</p>",
        unsafe_allow_html=True,
    )

    station_data = _load()
    if not station_data:
        st.error(
            "No station data could be loaded. "
            "Verify that the Station_wise_dataset_for_EPA_AQS/ directory is present "
            "and contains the expected CSV files."
        )
        return

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Dataset Controls")
        st.markdown("---")

        # Station
        station_options = {smeta["label"]: k for k, smeta in STATIONS.items()}
        sel_station_label = st.selectbox(
            "Station", list(station_options.keys()), key="de_station"
        )
        sel_station = station_options[sel_station_label]
        df_raw = station_data.get(sel_station)
        meta   = STATIONS[sel_station]

        if df_raw is None or df_raw.empty:
            st.error(f"Data unavailable for {sel_station_label}.")
            return

        # Pollutant — only non-structural-missing monitored pollutants
        available_pols = [
            p for p in meta["monitored_pollutants"]
            if p in df_raw.columns
            and df_raw[p].notna().mean() > STRUCTURAL_MISSING_THRESHOLD
        ]
        if not available_pols:
            available_pols = meta["monitored_pollutants"]
        sel_pollutant = st.selectbox(
            "Pollutant", available_pols, key="de_pollutant"
        )

        # Date range
        if "datetime" in df_raw.columns and df_raw["datetime"].notna().any():
            dt_min = df_raw["datetime"].min().date()
            dt_max = df_raw["datetime"].max().date()
        else:
            dt_min = datetime.date(2019, 1, 1)
            dt_max = datetime.date(2024, 12, 31)

        date_range = st.date_input(
            "Date Range",
            value=(dt_min, dt_max),
            min_value=dt_min, max_value=dt_max,
            key="de_dates",
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            d_from, d_to = date_range
        else:
            d_from, d_to = dt_min, dt_max

        st.markdown("---")
        st.markdown("**ERA5 Meteorology**")
        era5_avail = [
            v for v in ERA5_VARS
            if v in df_raw.columns and df_raw[v].notna().any()
        ]
        default_era5 = (
            ["temp_c"] if "temp_c" in era5_avail
            else era5_avail[:1] if era5_avail
            else []
        )
        sel_era5 = st.multiselect(
            "ERA5 Variables", era5_avail, default=default_era5,
            format_func=lambda x: ERA5_VAR_LABELS.get(x, x),
            key="de_era5",
        )

        st.markdown("---")
        st.markdown("**Aggregation**")
        agg = st.radio(
            "Level", ["Hourly", "Daily", "Weekly"],
            horizontal=True, key="de_agg",
        )

        st.markdown("---")
        st.markdown("**Markers & Overlays**")
        show_missing   = st.checkbox(
            "Shade missing periods",          value=True,  key="de_show_miss"
        )
        show_anomalies = st.checkbox(
            "Show anomaly markers (z-score)", value=False, key="de_show_anom"
        )
        show_imputed   = st.checkbox(
            "Show imputed values",            value=False, key="de_show_imp"
        )

        st.markdown("---")
        st.markdown("**Station Comparison**")
        compare_on = st.checkbox(
            "Compare with another station", value=False, key="de_compare"
        )
        compare_station: str | None = None
        compare_short:   str | None = None
        if compare_on:
            compat = {
                STATIONS[k]["label"]: k
                for k in STATIONS
                if k != sel_station
                and sel_pollutant in STATIONS[k]["monitored_pollutants"]
            }
            if compat:
                comp_label      = st.selectbox(
                    "Compare Station", list(compat.keys()), key="de_comp_sel"
                )
                compare_station = compat[comp_label]
                compare_short   = STATIONS[compare_station]["short"]
            else:
                st.caption(f"No other station monitors {sel_pollutant}.")

    # ── Date filter ────────────────────────────────────────────────────────────
    mask    = (df_raw["datetime"].dt.date >= d_from) & (df_raw["datetime"].dt.date <= d_to)
    df_filt = df_raw.loc[mask].copy()

    if df_filt.empty:
        st.warning(
            "No data found for the selected date range. Try widening the range."
        )
        return

    # Check structural missing / empty pollutant
    if (
        sel_pollutant not in df_filt.columns
        or df_filt[sel_pollutant].notna().sum() == 0
    ):
        st.markdown(
            "<div class='empty-state'>"
            "<div class='empty-state-icon'>⚠️</div>"
            f"<div class='empty-state-title'>{sel_pollutant} is not available at "
            f"{meta['label']}</div>"
            "<div>No valid readings exist in the selected window. "
            "This pollutant may be structurally absent — "
            "try a different pollutant or station.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # Aggregated df (may equal df_filt when Hourly)
    df_agg = _aggregate_df(df_filt, sel_pollutant, agg)

    # Comparison station filtered df
    compare_df: pd.DataFrame | None = None
    if compare_on and compare_station and compare_station in station_data:
        cdf = station_data[compare_station]
        cmask = (
            (cdf["datetime"].dt.date >= d_from) &
            (cdf["datetime"].dt.date <= d_to)
        )
        compare_df = cdf.loc[cmask].copy()
        if compare_df.empty:
            compare_df = None

    unit = POLLUTANTS.get(sel_pollutant, {}).get("unit", "")

    # ── Breadcrumb ─────────────────────────────────────────────────────────────
    st.caption(
        f"Home › Data Explorer › {meta['label']} › {sel_pollutant} "
        f"› {d_from} → {d_to}  ({agg})"
    )
    if "special_case" in meta:
        info_box(f"<b>Data note:</b> {meta['special_case']}", kind="warn")

    # ── Chart tabs ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Raw Series",
        "🌤 ERA5 Context",
        "🔀 Station Comparison",
        "📅 Seasonality",
        "📊 Distribution",
    ])

    with tab1:
        st.markdown(
            f"<div class='section-header'>"
            f"{meta['label']} — {sel_pollutant} ({agg})"
            f"</div>",
            unsafe_allow_html=True,
        )
        _tab_raw_series(
            df=df_agg,
            pollutant=sel_pollutant,
            unit=unit,
            station_key=sel_station,
            station_label=meta["label"],
            station_short=meta["short"],
            show_missing=show_missing,
            show_anomalies=show_anomalies,
            show_imputed=show_imputed,
            compare_df=compare_df,
            compare_short=compare_short,
        )

    with tab2:
        st.markdown(
            f"<div class='section-header'>"
            f"{sel_pollutant} vs ERA5 Meteorology"
            f"</div>",
            unsafe_allow_html=True,
        )
        _tab_era5_context(df_agg, sel_pollutant, unit, sel_era5)

    with tab3:
        st.markdown(
            f"<div class='section-header'>"
            f"{sel_pollutant} — All-Station Cross-Comparison"
            f"</div>",
            unsafe_allow_html=True,
        )
        _tab_station_comparison(station_data, sel_pollutant, unit, d_from, d_to)

    with tab4:
        st.markdown(
            f"<div class='section-header'>"
            f"{sel_pollutant} — Monthly & Diurnal Patterns"
            f"</div>",
            unsafe_allow_html=True,
        )
        # Seasonality always uses hourly df for meaningful temporal resolution
        _tab_seasonality(df_filt, sel_pollutant, unit)

    with tab5:
        st.markdown(
            f"<div class='section-header'>"
            f"{sel_pollutant} — Statistical Distribution"
            f"</div>",
            unsafe_allow_html=True,
        )
        _tab_distribution(df_filt, sel_pollutant, unit)

    # ── Summary stats & heatmap ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"<div class='section-header'>"
        f"Summary Statistics — {meta['label']} / {sel_pollutant}"
        f"</div>",
        unsafe_allow_html=True,
    )
    _render_summary_stats(df_filt, sel_pollutant, unit)

    # ── Point inspector ────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander(
        "🔍 Point Inspector — examine a specific date and hour", expanded=False
    ):
        _render_point_inspector(df_filt, sel_pollutant, unit, sel_era5)

    # ── Raw data table + CSV download ──────────────────────────────────────────
    with st.expander("📋 Raw Data Table (first 500 rows)", expanded=False):
        show_cols = (
            ["datetime", "station_key", sel_pollutant]
            + [c for c in sel_era5 if c in df_filt.columns]
        )
        show_cols = [c for c in show_cols if c in df_filt.columns]
        st.dataframe(
            df_filt[show_cols].head(500), use_container_width=True
        )
        csv_bytes = df_filt[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download filtered data as CSV",
            data=csv_bytes,
            file_name=f"{sel_station}_{sel_pollutant}_{d_from}_{d_to}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()

