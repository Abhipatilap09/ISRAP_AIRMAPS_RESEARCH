"""Data Quality — audit, negative values, gaps, structural missingness."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import POLLUTANTS, STATIONS
from src.data_loader import audit_station_pollutant, build_availability_matrix, load_all_stations, load_era5, merge_era5
from src.gap_analysis import find_gaps, gap_summary
from ui.theme import COLORS, inject_css, info_box


@st.cache_data(show_spinner="Loading data…")
def _load():
    stations = load_all_stations()
    era5 = load_era5()
    return {k: merge_era5(v, era5) for k, v in stations.items()}


def main():
    st.set_page_config(page_title="Data Quality | ISRAP AIRMAPS", layout="wide", page_icon="🔍")
    inject_css()

    st.markdown(f"<h1 style='color:{COLORS['navy']};'>🔍 Data Quality Audit</h1>", unsafe_allow_html=True)
    st.caption("Comprehensive audit of missing values, negative concentrations, duplicates, and structural availability.")

    station_data = _load()

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Summary", "📉 Missingness", "⚠️ Negative Values", "📋 Gaps", "📥 Downloads"
    ])

    # ── Tab 1: Summary ─────────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">Station–Pollutant Audit</div>', unsafe_allow_html=True)

        audit_rows = []
        for skey, df in station_data.items():
            for p in POLLUTANTS:
                row = audit_station_pollutant(df, skey, p)
                audit_rows.append(row)
        audit_df = pd.DataFrame(audit_rows)

        # Style: highlight structural missing in gray
        def _color_row(row):
            if row["structural_missing"]:
                return ["background-color: #f0f0f0; color: #999"] * len(row)
            if row["negative_count"] > 100:
                return ["background-color: #fde8e8"] * len(row)
            return [""] * len(row)

        display_cols = [
            "station", "pollutant", "valid_count", "missing_pct",
            "negative_count", "structural_missing", "min", "max", "mean", "std",
        ]
        st.dataframe(
            audit_df[display_cols].style.apply(_color_row, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        info_box(
            "<b>Grey rows</b> = structurally missing (pollutant not monitored at that station). "
            "<b>Red rows</b> = high negative-value count (likely sensor issues).",
            kind="info",
        )

    # ── Tab 2: Missingness ─────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">Missingness Heatmap</div>', unsafe_allow_html=True)

        p_cols = list(POLLUTANTS.keys())
        pct_matrix = []
        labels = []
        for skey, df in station_data.items():
            row = []
            for p in p_cols:
                if p in df.columns:
                    pct = round(100 * df[p].isna().mean(), 1)
                else:
                    pct = 100.0
                row.append(pct)
            pct_matrix.append(row)
            labels.append(STATIONS[skey]["label"])

        fig_heat = go.Figure(go.Heatmap(
            z=pct_matrix,
            x=p_cols,
            y=labels,
            text=[[f"{v:.1f}%" for v in row] for row in pct_matrix],
            texttemplate="%{text}",
            colorscale=[[0, "#2dc653"], [0.01, "#94d2bd"], [0.3, "#f4a261"], [1, "#e63946"]],
            zmin=0, zmax=100,
            hovertemplate="Station: %{y}<br>Pollutant: %{x}<br>Missing: %{z:.1f}%<extra></extra>",
            colorbar=dict(title="% Missing"),
        ))
        fig_heat.update_layout(
            template="plotly_white", height=300,
            margin=dict(l=200, r=40, t=30, b=60),
            xaxis=dict(side="top"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Monthly completeness for selected station/pollutant
        st.markdown("#### Monthly Completeness")
        sel_s = st.selectbox("Station", list(station_data.keys()),
                             format_func=lambda k: STATIONS[k]["label"], key="miss_s")
        sel_p = st.selectbox("Pollutant",
                             [p for p in POLLUTANTS if station_data[sel_s][p].notna().mean() > 0.01],
                             key="miss_p")
        df_s = station_data[sel_s]
        monthly = (
            df_s.groupby(df_s["datetime"].dt.to_period("M"))
            .agg(valid=(sel_p, "count"), total=(sel_p, "size"))
            .reset_index()
        )
        monthly["completeness"] = 100 * monthly["valid"] / monthly["total"].clip(lower=1)
        monthly["month_str"] = monthly["datetime"].astype(str)

        fig_monthly = go.Figure(go.Bar(
            x=monthly["month_str"],
            y=monthly["completeness"],
            marker_color=monthly["completeness"].apply(
                lambda v: "#2dc653" if v >= 90 else ("#f4a261" if v >= 70 else "#e63946")
            ),
        ))
        fig_monthly.update_layout(
            title=f"Monthly Completeness — {sel_s} / {sel_p}",
            template="plotly_white", height=300,
            yaxis_title="Completeness (%)", xaxis_title="Month",
            margin=dict(l=60, r=20, t=40, b=60),
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

    # ── Tab 3: Negative Values ─────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">Negative Value Analysis</div>', unsafe_allow_html=True)
        info_box(
            "<b>Negative pollutant concentrations are physically impossible.</b> "
            "They indicate sensor calibration errors, electronic noise, or data transmission issues. "
            "These values must be flagged and excluded before any analysis.",
            kind="danger",
        )

        neg_rows = []
        for skey, df in station_data.items():
            for p in POLLUTANTS:
                if p not in df.columns:
                    continue
                n_neg = int((df[p] < 0).sum())
                if n_neg > 0:
                    neg_rows.append({
                        "Station": STATIONS[skey]["label"],
                        "Pollutant": p,
                        "Negative Count": n_neg,
                        "Total Observed": int(df[p].notna().sum()),
                        "Negative %": round(100 * n_neg / max(df[p].notna().sum(), 1), 2),
                        "Min Value": round(float(df.loc[df[p] < 0, p].min()), 4),
                        "Severity": "CRITICAL" if n_neg / max(df[p].notna().sum(), 1) > 0.1 else "HIGH" if n_neg > 100 else "MEDIUM",
                    })

        if neg_rows:
            neg_df = pd.DataFrame(neg_rows).sort_values("Negative %", ascending=False)
            st.dataframe(neg_df, use_container_width=True, hide_index=True)

            # Bar chart
            fig_neg = go.Figure(go.Bar(
                x=[f"{r['Station']} / {r['Pollutant']}" for _, r in neg_df.iterrows()],
                y=neg_df["Negative %"],
                marker_color=[
                    COLORS["red"] if r["Severity"] == "CRITICAL"
                    else COLORS["amber"] if r["Severity"] == "HIGH"
                    else COLORS["gray"]
                    for _, r in neg_df.iterrows()
                ],
                text=neg_df["Negative Count"].astype(str),
                textposition="outside",
            ))
            fig_neg.update_layout(
                title="Negative Value Rate by Station / Pollutant",
                template="plotly_white", height=350,
                yaxis_title="Negative %", xaxis_tickangle=-30,
                margin=dict(l=60, r=20, t=50, b=100),
            )
            st.plotly_chart(fig_neg, use_container_width=True)
        else:
            st.success("No negative values found.")

    # ── Tab 4: Gaps ─────────────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">Gap Analysis</div>', unsafe_allow_html=True)
        with st.spinner("Analysing gaps…"):
            gap_df = gap_summary(station_data)

        if not gap_df.empty:
            st.dataframe(
                gap_df.sort_values("total_missing_hours", ascending=False),
                use_container_width=True, hide_index=True,
            )

            sel_gs = st.selectbox("View gaps for station", list(station_data.keys()),
                                  format_func=lambda k: STATIONS[k]["label"], key="gap_s")
            sel_gp = st.selectbox("Pollutant",
                                  [p for p in POLLUTANTS if station_data[sel_gs][p].notna().mean() > 0.01],
                                  key="gap_p")
            gaps_detail = find_gaps(station_data[sel_gs], sel_gp)
            if not gaps_detail.empty:
                st.write(f"Found **{len(gaps_detail)}** gaps for {sel_gs} / {sel_gp}")
                st.dataframe(gaps_detail.head(200), use_container_width=True, hide_index=True)
            else:
                st.info("No missing-value gaps found.")

    # ── Tab 5: Downloads ───────────────────────────────────────────────────────
    with tab5:
        st.markdown('<div class="section-header">Download Audit Reports</div>', unsafe_allow_html=True)

        audit_rows_all = []
        for skey, df in station_data.items():
            for p in POLLUTANTS:
                audit_rows_all.append(audit_station_pollutant(df, skey, p))
        audit_all = pd.DataFrame(audit_rows_all)

        c1, c2, c3 = st.columns(3)
        with c1:
            csv = audit_all.to_csv(index=False).encode()
            st.download_button("📥 Full Audit Report (CSV)", csv, "data_audit_summary.csv", "text/csv")

        gap_csv = gap_summary(station_data).to_csv(index=False).encode() if not gap_df.empty else b""
        with c2:
            if gap_csv:
                st.download_button("📥 Gap Summary (CSV)", gap_csv, "missing_gap_summary.csv", "text/csv")

        neg_all = []
        for skey, df in station_data.items():
            for p in POLLUTANTS:
                if p in df.columns:
                    neg = df[df[p] < 0][["datetime", p]].copy()
                    if not neg.empty:
                        neg["station"] = skey
                        neg["pollutant"] = p
                        neg["original_value"] = neg[p]
                        neg_all.append(neg[["datetime", "station", "pollutant", "original_value"]])
        with c3:
            if neg_all:
                neg_csv = pd.concat(neg_all).to_csv(index=False).encode()
                st.download_button("📥 Negative Values (CSV)", neg_csv, "negative_values_summary.csv", "text/csv")


if __name__ == "__main__":
    main()

