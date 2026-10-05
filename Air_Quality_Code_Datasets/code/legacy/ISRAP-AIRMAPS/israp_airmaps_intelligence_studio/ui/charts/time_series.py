"""Reusable Plotly time-series chart factory."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.theme import ANOMALY_COLOR, IMPUTED_COLOR, MISSING_COLOR, OBSERVED_COLOR, COLORS


def _downsample(df: pd.DataFrame, max_pts: int = 5000) -> pd.DataFrame:
    if len(df) <= max_pts:
        return df
    step = max(1, len(df) // max_pts)
    return df.iloc[::step]


def raw_series(
    df: pd.DataFrame,
    pollutant: str,
    title: str = "",
    unit: str = "",
    show_missing_shading: bool = True,
) -> go.Figure:
    df = _downsample(df.copy())
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df[pollutant],
        mode="lines", name=pollutant,
        line=dict(color=OBSERVED_COLOR, width=1.5),
        hovertemplate=f"%{{x}}<br>{pollutant}: %{{y:.3f}} {unit}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis_title="Date", yaxis_title=f"{pollutant} ({unit})",
        template="plotly_white",
        hovermode="x unified",
        height=380,
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


def anomaly_overlay(
    df: pd.DataFrame,
    pollutant: str,
    flag_col: str,
    score_col: str | None = None,
    title: str = "",
    unit: str = "",
) -> go.Figure:
    df = _downsample(df.copy())
    rows = 2 if score_col and score_col in df.columns else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3] if rows == 2 else [1.0],
                        vertical_spacing=0.08)

    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df[pollutant],
        mode="lines", name="Observed",
        line=dict(color=OBSERVED_COLOR, width=1.2),
        hovertemplate=f"%{{x}}<br>{pollutant}: %{{y:.3f}}<extra></extra>",
    ), row=1, col=1)

    if flag_col in df.columns:
        anom = df[df[flag_col].astype(bool)]
        if not anom.empty:
            fig.add_trace(go.Scatter(
                x=anom["datetime"], y=anom[pollutant],
                mode="markers", name="Anomaly",
                marker=dict(color=ANOMALY_COLOR, size=7, symbol="x"),
                hovertemplate=f"%{{x}}<br>ANOMALY: %{{y:.3f}}<extra></extra>",
            ), row=1, col=1)

    if rows == 2 and score_col and score_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=df[score_col],
            mode="lines", name="Anomaly Score",
            line=dict(color=COLORS["violet"], width=1),
            fill="tozeroy", fillcolor=f"rgba(123,45,139,0.12)",
        ), row=2, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        template="plotly_white", height=400 if rows == 2 else 350,
        hovermode="x unified",
        margin=dict(l=60, r=20, t=50, b=50),
    )
    fig.update_yaxes(title_text=f"{pollutant} ({unit})", row=1, col=1)
    if rows == 2:
        fig.update_yaxes(title_text="Score", row=2, col=1)
    return fig


def imputation_comparison(
    df: pd.DataFrame,
    pollutant: str,
    imputed_cols: list[str],
    title: str = "",
    unit: str = "",
) -> go.Figure:
    df = _downsample(df.copy())
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df[pollutant],
        mode="lines", name="Observed",
        line=dict(color=OBSERVED_COLOR, width=2),
    ))
    colors = [IMPUTED_COLOR, COLORS["amber"], COLORS["violet"], COLORS["red"]]
    for i, ic in enumerate(imputed_cols):
        if ic in df.columns:
            imp_mask = df[pollutant].isna() & df[ic].notna()
            fig.add_trace(go.Scatter(
                x=df.loc[imp_mask, "datetime"], y=df.loc[imp_mask, ic],
                mode="markers", name=f"Imputed ({ic})",
                marker=dict(color=colors[i % len(colors)], size=5, symbol="circle-open"),
            ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        template="plotly_white", height=380,
        hovermode="x unified",
        xaxis_title="Date", yaxis_title=f"{pollutant} ({unit})",
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


def missingness_heatmap(station_data: dict, pollutants: list[str]) -> go.Figure:
    rows_data = []
    for skey, df in station_data.items():
        row = {"station": skey}
        for p in pollutants:
            if p in df.columns:
                row[p] = round(100 * df[p].isna().mean(), 1)
            else:
                row[p] = 100.0
        rows_data.append(row)
    mat = pd.DataFrame(rows_data).set_index("station")[pollutants]
    fig = go.Figure(go.Heatmap(
        z=mat.values,
        x=mat.columns.tolist(),
        y=mat.index.tolist(),
        colorscale=[[0, "#2dc653"], [0.5, "#f4a261"], [1, "#e63946"]],
        zmin=0, zmax=100,
        text=[[f"{v:.1f}%" for v in row] for row in mat.values],
        texttemplate="%{text}",
        hovertemplate="Station: %{y}<br>Pollutant: %{x}<br>Missing: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="% Missing"),
    ))
    fig.update_layout(
        title="Missing-Value Percentage by Station and Pollutant",
        template="plotly_white",
        height=300,
        margin=dict(l=100, r=20, t=50, b=50),
    )
    return fig

