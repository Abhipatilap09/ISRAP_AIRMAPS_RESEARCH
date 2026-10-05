"""Station 3 SO₂ Investigation — dedicated diagnostic page."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.config import STATIONS, ERA5_VAR_LABELS
from src.data_loader import load_all_stations, load_era5, merge_era5
from ui.theme import COLORS, inject_css, info_box


@st.cache_data(show_spinner="Loading data…")
def _load():
    stations = load_all_stations()
    era5 = load_era5()
    return {k: merge_era5(v, era5) for k, v in stations.items()}


def main():
    st.set_page_config(page_title="Station 3 SO₂ | ISRAP AIRMAPS", layout="wide", page_icon="⚠️")
    inject_css()

    st.markdown(f"<h1 style='color:{COLORS['red']};'>⚠️ Station 3 SO₂ Investigation</h1>", unsafe_allow_html=True)

    info_box(
        "<b>⚠️ CRITICAL WARNING: Station 3 SO₂ contains substantial systematic corruption.</b> "
        "Out of 48,870 observed SO₂ readings, <b>16,416 (33.6%) are negative</b> — "
        "physically impossible for a concentration measurement. "
        "Imputation cannot reliably reconstruct a channel when one-third of observed values are invalid. "
        "<b>Exclusion is the default research recommendation.</b> "
        "Experimental repair is available below but must be clearly labelled and not mixed with valid data.",
        kind="danger",
    )

    station_data = _load()
    df3 = station_data["station3"].copy()
    df5 = station_data.get("station5", pd.DataFrame()).copy()

    if "SO2" not in df3.columns:
        st.error("SO2 column not found in Station 3 data.")
        return

    so2 = df3["SO2"].copy()
    n_obs = int(so2.notna().sum())
    n_neg = int((so2 < 0).sum())
    n_pos = int((so2 > 0).sum())
    n_zero = int((so2 == 0).sum())
    neg_pct = 100 * n_neg / max(n_obs, 1)

    # ── Top metrics ────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Observed", f"{n_obs:,}")
    c2.metric("Negative Values", f"{n_neg:,}", f"{neg_pct:.1f}% of observed", delta_color="inverse")
    c3.metric("Positive Values", f"{n_pos:,}")
    c4.metric("Zero Values", f"{n_zero:,}")
    c5.metric("Min Value", f"{float(so2.min()):.4f}")

    # ── Raw SO2 timeline ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Raw SO₂ Timeline</div>', unsafe_allow_html=True)
    fig_ts = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.7, 0.3], vertical_spacing=0.06)

    fig_ts.add_trace(go.Scatter(
        x=df3["datetime"], y=so2,
        mode="lines", name="SO₂ (all)", line=dict(color=COLORS["blue"], width=0.8),
    ), row=1, col=1)

    neg_mask = so2 < 0
    fig_ts.add_trace(go.Scatter(
        x=df3.loc[neg_mask, "datetime"], y=so2[neg_mask],
        mode="markers", name="Negative (invalid)",
        marker=dict(color=COLORS["red"], size=3, opacity=0.5),
    ), row=1, col=1)
    fig_ts.add_hline(y=0, line_dash="dash", line_color=COLORS["amber"], row=1, col=1,
                     annotation_text="Zero baseline")

    # Monthly negative count
    monthly_neg = df3[neg_mask].groupby(df3.loc[neg_mask, "datetime"].dt.to_period("M")).size()
    monthly_neg.index = monthly_neg.index.astype(str)
    fig_ts.add_trace(go.Bar(
        x=monthly_neg.index, y=monthly_neg.values,
        name="Monthly negatives", marker_color=COLORS["red"], opacity=0.7,
    ), row=2, col=1)

    fig_ts.update_layout(
        template="plotly_white", height=480, hovermode="x unified",
        legend=dict(orientation="h", y=-0.08),
        margin=dict(l=60, r=20, t=30, b=70),
    )
    fig_ts.update_yaxes(title_text="SO₂ (ppb)", row=1, col=1)
    fig_ts.update_yaxes(title_text="Count", row=2, col=1)
    st.plotly_chart(fig_ts, use_container_width=True)

    # ── Distribution ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Value Distribution</div>', unsafe_allow_html=True)
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        fig_dist_neg = go.Figure()
        fig_dist_neg.add_trace(go.Histogram(
            x=so2[neg_mask].clip(-50, 0), nbinsx=50,
            name="Negative values", marker_color=COLORS["red"], opacity=0.7,
        ))
        fig_dist_neg.update_layout(title="Distribution of Negative SO₂",
                                   template="plotly_white", height=280,
                                   margin=dict(l=60, r=20, t=40, b=50))
        st.plotly_chart(fig_dist_neg, use_container_width=True)

    with col_d2:
        fig_dist_pos = go.Figure()
        pos_vals = so2[so2 > 0].clip(0, 50)
        fig_dist_pos.add_trace(go.Histogram(
            x=pos_vals, nbinsx=50,
            name="Positive values", marker_color=COLORS["green"], opacity=0.7,
        ))
        fig_dist_pos.update_layout(title="Distribution of Valid (Positive) SO₂",
                                   template="plotly_white", height=280,
                                   margin=dict(l=60, r=20, t=40, b=50))
        st.plotly_chart(fig_dist_pos, use_container_width=True)

    # ── Comparison with Station 5 ─────────────────────────────────────────────
    if not df5.empty and "SO2" in df5.columns and df5["SO2"].notna().any():
        st.markdown('<div class="section-header">Comparison: Station 3 vs Station 5 SO₂</div>',
                    unsafe_allow_html=True)
        common_range = df3[df3["datetime"].isin(df5["datetime"])]["datetime"]
        if len(common_range) > 0:
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(
                x=df3["datetime"], y=so2,
                mode="lines", name="Station 3 SO₂", line=dict(color=COLORS["red"], width=0.8),
            ))
            fig_comp.add_trace(go.Scatter(
                x=df5["datetime"], y=df5["SO2"],
                mode="lines", name="Station 5 SO₂", line=dict(color=COLORS["blue"], width=0.8),
            ))
            fig_comp.add_hline(y=0, line_dash="dash", line_color=COLORS["amber"])
            fig_comp.update_layout(
                title="Station 3 vs Station 5 SO₂",
                template="plotly_white", height=350, hovermode="x unified",
                margin=dict(l=60, r=20, t=40, b=50),
            )
            st.plotly_chart(fig_comp, use_container_width=True)
    else:
        info_box("Station 5 SO₂ data not available for comparison in the overlapping date range.", kind="info")

    # ── Options ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Analysis Options</div>', unsafe_allow_html=True)
    opt = st.radio("Select analysis option:", [
        "1. Exclude Station 3 SO₂ (Default — recommended)",
        "2. Retain only valid positive observations",
        "3. Experimental repair (clearly labelled — not for modeling)",
    ])

    if opt.startswith("1"):
        info_box(
            "<b>Option 1 selected:</b> Station 3 SO₂ will be excluded from all downstream analysis. "
            "This is the scientifically safe choice given the 33.6% negative rate.",
            kind="info",
        )
    elif opt.startswith("2"):
        pos_only = so2.copy()
        pos_only[pos_only <= 0] = np.nan
        retained = int(pos_only.notna().sum())
        removed = int((so2 <= 0).sum())
        info_box(
            f"<b>Option 2:</b> Retaining {retained:,} positive observations, "
            f"removing {removed:,} non-positive values. "
            "Note: this does not address the underlying cause of negative readings.",
            kind="warn",
        )
        fig_pos = go.Figure(go.Scatter(
            x=df3["datetime"], y=pos_only,
            mode="lines", name="Valid SO₂ only", line=dict(color=COLORS["green"], width=0.8),
        ))
        fig_pos.update_layout(title="Station 3 — Valid (positive) SO₂ only",
                               template="plotly_white", height=300,
                               margin=dict(l=60, r=20, t=40, b=50))
        st.plotly_chart(fig_pos, use_container_width=True)

        csv = pd.DataFrame({"datetime": df3["datetime"], "SO2_valid_only": pos_only}).to_csv(index=False).encode()
        st.download_button("📥 Download valid-only SO₂", csv, "station3_SO2_valid_only.csv")

    elif opt.startswith("3"):
        info_box(
            "⚠️ <b>EXPERIMENTAL — DO NOT USE IN FINAL MODELS.</b> "
            "The experimental repair below applies an absolute-value transformation to negative readings. "
            "This is a simple heuristic, not a validated repair method. "
            "Results are clearly labelled [EXPERIMENTAL] and must not be combined with valid data.",
            kind="danger",
        )
        repaired = so2.abs()
        fig_rep = go.Figure()
        fig_rep.add_trace(go.Scatter(
            x=df3["datetime"], y=so2,
            mode="lines", name="Original", line=dict(color=COLORS["blue"], width=0.8, dash="dot"),
        ))
        fig_rep.add_trace(go.Scatter(
            x=df3["datetime"], y=repaired,
            mode="lines", name="[EXPERIMENTAL] abs() repair", line=dict(color=COLORS["amber"], width=0.8),
        ))
        fig_rep.update_layout(title="Station 3 SO₂ — [EXPERIMENTAL] abs() repair",
                               template="plotly_white", height=300,
                               margin=dict(l=60, r=20, t=40, b=50))
        st.plotly_chart(fig_rep, use_container_width=True)


if __name__ == "__main__":
    main()

