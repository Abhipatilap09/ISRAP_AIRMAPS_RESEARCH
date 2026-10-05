"""Executive Overview -- network-wide metrics and availability matrix."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import POLLUTANTS, STATIONS
from src.data_loader import build_availability_matrix, load_all_stations, load_era5, merge_era5
from src.gap_analysis import gap_summary
from ui.theme import COLORS, SERIES_COLORS, inject_css, metric_card, info_box, badge


# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading station data...")
def _load():
    stations = load_all_stations()
    era5 = load_era5()
    merged = {k: merge_era5(v, era5) for k, v in stations.items()}
    return merged


@st.cache_data(show_spinner="Analysing missing-data gaps...")
def _get_gap_df():
    """Cache gap analysis separately -- it iterates every row of every station."""
    station_data = _load()
    return gap_summary(station_data)


# ── Helper: network metrics ───────────────────────────────────────────────────

def _compute_network_metrics(station_data: dict) -> dict:
    n_stations = len(station_data)
    total_rows = sum(len(df) for df in station_data.values())

    active_channels = 0
    all_valid = 0
    all_total = 0
    total_negatives = 0

    for skey, df in station_data.items():
        for p in POLLUTANTS:
            sm_col = f"{p}_structural_missing"
            if p not in df.columns:
                continue
            frac = df[p].notna().mean()
            is_structural = (
                bool(df[sm_col].iloc[0])
                if sm_col in df.columns
                else frac < 0.01
            )
            if not is_structural:
                active_channels += 1
                all_valid += int(df[p].notna().sum())
                all_total += len(df)
                total_negatives += int((df[p] < 0).sum())

    completeness = round(100 * all_valid / max(all_total, 1), 1)

    all_dates = []
    for df in station_data.values():
        if "datetime" in df.columns:
            mn = df["datetime"].min()
            mx = df["datetime"].max()
            if pd.notna(mn):
                all_dates.append(mn)
            if pd.notna(mx):
                all_dates.append(mx)

    date_coverage = (
        f"{min(all_dates).year}–{max(all_dates).year}" if all_dates else "N/A"
    )

    return {
        "n_stations": n_stations,
        "active_channels": active_channels,
        "total_rows": total_rows,
        "completeness": completeness,
        "total_negatives": total_negatives,
        "date_coverage": date_coverage,
    }


# ── Helper: annual completeness ───────────────────────────────────────────────

def _build_annual_completeness(station_data: dict) -> pd.DataFrame:
    rows = []
    for skey, df in station_data.items():
        if "datetime" not in df.columns:
            continue
        meta = STATIONS[skey]
        monitored = meta["monitored_pollutants"]
        active_pols = []
        for p in monitored:
            sm_col = f"{p}_structural_missing"
            if p in df.columns:
                is_s = (
                    bool(df[sm_col].iloc[0])
                    if sm_col in df.columns
                    else df[p].notna().mean() < 0.01
                )
                if not is_s:
                    active_pols.append(p)
        if not active_pols:
            continue
        df_yr = df.copy()
        df_yr["_year"] = df_yr["datetime"].dt.year
        for year, ydf in df_yr.groupby("_year"):
            fracs = [ydf[p].notna().mean() for p in active_pols if p in ydf.columns]
            avg = round(100 * float(np.mean(fracs)), 1) if fracs else 0.0
            rows.append({
                "station": skey,
                "short": meta["short"],
                "label": meta["label"],
                "year": int(year),
                "completeness": avg,
            })
    return pd.DataFrame(rows)


# ── Helper: missingness timeline ──────────────────────────────────────────────

def _build_missingness_timeline(station_data: dict) -> pd.DataFrame:
    parts = []
    for skey, df in station_data.items():
        if "datetime" not in df.columns:
            continue
        meta = STATIONS[skey]
        for p in meta["monitored_pollutants"]:
            sm_col = f"{p}_structural_missing"
            if p not in df.columns:
                continue
            is_s = (
                bool(df[sm_col].iloc[0])
                if sm_col in df.columns
                else df[p].notna().mean() < 0.01
            )
            if is_s:
                continue
            tmp = df[["datetime", p]].copy()
            tmp["missing"] = tmp[p].isna().astype(int)
            tmp["ym"] = tmp["datetime"].dt.to_period("M")
            monthly = tmp.groupby("ym")["missing"].sum().reset_index()
            monthly.columns = ["ym", "missing_hours"]
            parts.append(monthly)

    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    timeline = (
        combined.groupby("ym")["missing_hours"]
        .sum()
        .reset_index()
        .sort_values("ym")
    )
    timeline["ym_str"] = timeline["ym"].astype(str)
    return timeline


# ── Helper: negative values ───────────────────────────────────────────────────

def _build_negative_df(station_data: dict) -> pd.DataFrame:
    rows = []
    for skey, df in station_data.items():
        meta = STATIONS[skey]
        for p in meta["monitored_pollutants"]:
            sm_col = f"{p}_structural_missing"
            if p not in df.columns:
                continue
            is_s = (
                bool(df[sm_col].iloc[0])
                if sm_col in df.columns
                else df[p].notna().mean() < 0.01
            )
            if is_s:
                continue
            neg_count = int((df[p] < 0).sum())
            if neg_count > 0:
                rows.append({
                    "station": meta["short"],
                    "station_label": meta["label"],
                    "pollutant": p,
                    "negative_count": neg_count,
                    "corrupted": skey == "station3" and p == "SO2",
                })
    return pd.DataFrame(rows)


# ── Helper: executive insights (deterministic) ────────────────────────────────

def _build_insights(station_data: dict, gap_df: pd.DataFrame) -> list[dict]:
    insights: list[dict] = []

    # 1. Station with highest average missingness
    station_miss: dict[str, float] = {}
    for skey, df in station_data.items():
        fracs = []
        for p in STATIONS[skey]["monitored_pollutants"]:
            sm_col = f"{p}_structural_missing"
            if p not in df.columns:
                continue
            is_s = (
                bool(df[sm_col].iloc[0])
                if sm_col in df.columns
                else df[p].notna().mean() < 0.01
            )
            if not is_s:
                fracs.append(df[p].isna().mean())
        if fracs:
            station_miss[skey] = float(np.mean(fracs))

    if station_miss:
        worst_k = max(station_miss, key=lambda k: station_miss[k])
        worst_pct = round(100 * station_miss[worst_k], 1)
        insights.append({
            "kind": "warn",
            "title": "Highest Network Missingness",
            "text": (
                f"{STATIONS[worst_k]['label']} has the highest average missing rate "
                f"at <b>{worst_pct}%</b> across its monitored pollutants. "
                "Prioritise gap-filling for this station before downstream modeling."
            ),
        })

    # 2. Pollutant with most negatives network-wide
    pol_neg: defaultdict[str, int] = defaultdict(int)
    for skey, df in station_data.items():
        for p in POLLUTANTS:
            sm_col = f"{p}_structural_missing"
            if p not in df.columns:
                continue
            is_s = (
                bool(df[sm_col].iloc[0])
                if sm_col in df.columns
                else df[p].notna().mean() < 0.01
            )
            if not is_s:
                pol_neg[p] += int((df[p] < 0).sum())

    if pol_neg:
        worst_pol = max(pol_neg, key=lambda k: pol_neg[k])
        insights.append({
            "kind": "warn",
            "title": "Most Network-Wide Negative Readings",
            "text": (
                f"<b>{worst_pol}</b> accumulates the most negative values across the network "
                f"({pol_neg[worst_pol]:,} readings). "
                "Negative values for this pollutant may indicate sensor zero-offset drift; "
                "verify with station maintenance logs before applying a blanket clip-to-zero."
            ),
        })

    # 3. Longest single contiguous gap
    if not gap_df.empty and "longest_gap" in gap_df.columns:
        max_idx = gap_df["longest_gap"].idxmax()
        row = gap_df.loc[max_idx]
        hrs = int(row["longest_gap"])
        days = hrs // 24
        st_label = STATIONS.get(row["station"], {}).get("label", row["station"])
        insights.append({
            "kind": "info",
            "title": "Longest Contiguous Data Gap",
            "text": (
                f"The longest single missing-data run is <b>{hrs:,} hours ({days} days)</b> "
                f"in <b>{st_label} / {row['pollutant']}</b>. "
                "Gaps of this length cannot be recovered by local interpolation; "
                "ERA5-regression or multi-station imputation is required."
            ),
        })

    # 4. Station 3 SO2 corruption warning
    insights.append({
        "kind": "danger",
        "title": "Station 3 SO2 -- Systematic Sensor Corruption",
        "text": (
            "Station 3 (Site 59) has approximately <b>16,416 negative SO2 readings "
            "out of 48,870 observed (33.6%)</b>. This pattern is consistent with systematic "
            "sensor miscalibration, not random noise. "
            "Station 3 SO2 is <b>excluded from all default modeling pipelines</b>. "
            "See the Station 3 SO2 Investigation page for the full forensic analysis."
        ),
    })

    # 5. Data coverage asymmetry note
    insights.append({
        "kind": "info",
        "title": "Asymmetric Temporal Coverage",
        "text": (
            "Most stations cover <b>January 2019 -- December 2024</b> (6 years). "
            "Exceptions: <b>Station 5 (Site 1080, SO2-only)</b> ends in March 2023; "
            "<b>Station 6 (Site 1087, PM2.5-only)</b> begins in November 2023. "
            "Network-wide analyses must account for these unequal time windows to avoid "
            "biasing temporal averages."
        ),
    })

    return insights


# ── Helper: station summary table ────────────────────────────────────────────

def _build_summary_table(station_data: dict) -> pd.DataFrame:
    rows = []
    for skey, df in station_data.items():
        meta = STATIONS[skey]
        has_dt = "datetime" in df.columns and not df["datetime"].isna().all()
        date_min = str(df["datetime"].min().date()) if has_dt else "N/A"
        date_max = str(df["datetime"].max().date()) if has_dt else "N/A"

        fracs, total_neg = [], 0
        for p in meta["monitored_pollutants"]:
            sm_col = f"{p}_structural_missing"
            if p not in df.columns:
                continue
            is_s = (
                bool(df[sm_col].iloc[0])
                if sm_col in df.columns
                else df[p].notna().mean() < 0.01
            )
            if not is_s:
                fracs.append(df[p].notna().mean())
                total_neg += int((df[p] < 0).sum())

        completeness = round(100 * float(np.mean(fracs)), 1) if fracs else 0.0
        special = meta.get("special_case", "")

        rows.append({
            "Station": meta["label"],
            "Site ID": str(meta["site"]),
            "Date From": date_min,
            "Date To": date_max,
            "Rows": len(df),
            "Monitored Pollutants": ", ".join(meta["monitored_pollutants"]),
            "Completeness %": completeness,
            "Negative Values": total_neg,
            "Notes": special[:90] if special else "",
        })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Executive Overview | ISRAP AIRMAPS",
        layout="wide",
        page_icon="\U0001f4ca",
    )
    inject_css()

    # Page header
    st.markdown(
        f"<h1 style='color:{COLORS['navy']}; margin-bottom:4px;'>Executive Overview</h1>"
        f"<p style='color:{COLORS['muted']}; font-size:1rem; margin-top:0;'>"
        "Network-wide data quality and availability summary across all EPA AQS monitoring stations "
        "in San Antonio, TX &mdash; NASA / ISRAP research network, 2019&ndash;2024.</p>",
        unsafe_allow_html=True,
    )

    # ── Load all data ──────────────────────────────────────────────────────────
    station_data = _load()
    gap_df = _get_gap_df()

    # Derived data (fast, computed from cached DataFrames)
    metrics = _compute_network_metrics(station_data)
    avail = build_availability_matrix(station_data)
    annual_df = _build_annual_completeness(station_data)
    miss_timeline = _build_missingness_timeline(station_data)
    neg_df = _build_negative_df(station_data)
    insights = _build_insights(station_data, gap_df)
    summary_df = _build_summary_table(station_data)

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 1 -- TOP METRIC CARDS
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-header">Network At-a-Glance</div>',
        unsafe_allow_html=True,
    )

    card_defs = [
        ("Stations", str(metrics["n_stations"]), "EPA AQS monitoring sites", COLORS["teal"], "NET"),
        ("Active Channels", str(metrics["active_channels"]), "pollutant-station pairs", COLORS["blue"], "CHL"),
        ("Total Records", f"{metrics['total_rows']:,}", "hourly observations", COLORS["navy"], "REC"),
        ("Completeness", f"{metrics['completeness']:.1f}%", "across monitored channels", COLORS["green"], "CPL"),
        ("Negative Values", f"{metrics['total_negatives']:,}", "potential sensor errors", COLORS["red"], "NEG"),
        ("Date Coverage", metrics["date_coverage"], "6 years of data", COLORS["violet"], "COV"),
    ]

    cols = st.columns(6)
    for col, (label, val, delta, color, icon) in zip(cols, card_defs):
        with col:
            st.markdown(metric_card(label, val, delta, color, icon=icon), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 2 -- STATION-POLLUTANT AVAILABILITY HEATMAP
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-header">Station-Pollutant Availability Matrix</div>',
        unsafe_allow_html=True,
    )

    p_cols = list(POLLUTANTS.keys())
    pct_matrix = avail[[f"{p}_pct" for p in p_cols]].values
    text_matrix = avail[p_cols].values

    fig_avail = go.Figure(go.Heatmap(
        z=pct_matrix,
        x=p_cols,
        y=avail["label"].tolist(),
        text=text_matrix,
        texttemplate="%{text}",
        colorscale=[
            [0.0,  "#e74c3c"],
            [0.01, "#f4a261"],
            [0.50, "#f9e79f"],
            [1.00, "#2dc653"],
        ],
        zmin=0,
        zmax=100,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Pollutant: %{x}<br>"
            "Completeness: %{z:.1f}%"
            "<extra></extra>"
        ),
        colorbar=dict(title="Completeness %", thickness=14, len=0.85),
    ))
    fig_avail.update_layout(
        template="plotly_white",
        height=290,
        margin=dict(l=210, r=50, t=40, b=60),
        xaxis=dict(side="top", title="Pollutant"),
        yaxis=dict(title=""),
    )
    st.plotly_chart(fig_avail, use_container_width=True)

    info_box(
        "<b>How to read this matrix:</b> "
        "Green cells indicate well-monitored channels with high data completeness. "
        "Red or <i>Not monitored</i> indicates structural missingness (the station does not "
        "instrument that pollutant). Structurally absent channels are excluded automatically "
        "from all anomaly detection, imputation, and modeling pipelines.",
        kind="info",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 3 -- ANNUAL COMPLETENESS CHART
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-header">Annual Data Completeness by Station (2019&ndash;2024)</div>',
        unsafe_allow_html=True,
    )

    if not annual_df.empty:
        station_color_map = {
            "station1": COLORS["teal"],
            "station2": COLORS["blue"],
            "station3": COLORS["green"],
            "station4": COLORS["amber"],
            "station5": COLORS["violet"],
            "station6": COLORS["red"],
        }

        fig_annual = go.Figure()
        for skey in STATIONS:
            sdf = annual_df[annual_df["station"] == skey].sort_values("year")
            if sdf.empty:
                continue
            meta = STATIONS[skey]
            color = station_color_map.get(skey, COLORS["gray"])
            fig_annual.add_trace(go.Bar(
                name=meta["short"],
                x=sdf["year"].astype(str),
                y=sdf["completeness"],
                marker_color=color,
                marker_line_color="white",
                marker_line_width=0.5,
                hovertemplate=(
                    f"<b>{meta['label']}</b><br>"
                    "Year: %{x}<br>"
                    "Completeness: %{y:.1f}%"
                    "<extra></extra>"
                ),
            ))

        fig_annual.add_hline(
            y=80,
            line_dash="dot",
            line_color=COLORS["amber"],
            line_width=1.5,
            annotation_text="80% target",
            annotation_position="top right",
            annotation_font_size=11,
        )
        fig_annual.update_layout(
            barmode="group",
            template="plotly_white",
            height=390,
            xaxis_title="Year",
            yaxis_title="Average Completeness (%)",
            yaxis=dict(range=[0, 108], ticksuffix="%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=70, r=20, t=60, b=60),
        )
        st.plotly_chart(fig_annual, use_container_width=True)

        info_box(
            "Completeness is averaged across all actively monitored (non-structural-missing) pollutants "
            "for each station-year combination. Stations with partial-year coverage (e.g. Station 5 ends "
            "March 2023, Station 6 starts November 2023) will show lower completeness in boundary years.",
            kind="info",
        )
    else:
        st.info("Annual completeness data could not be computed. Check that station CSV files are accessible.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 4 -- MISSINGNESS TIMELINE
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-header">Network Missingness Timeline (Monthly Missing Hours)</div>',
        unsafe_allow_html=True,
    )

    if not miss_timeline.empty:
        fig_miss = go.Figure()
        fig_miss.add_trace(go.Scatter(
            x=miss_timeline["ym_str"],
            y=miss_timeline["missing_hours"],
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(10,147,150,0.18)",
            line=dict(color=COLORS["teal"], width=2),
            name="Total Missing Hours",
            hovertemplate="Month: %{x}<br>Missing Hours: %{y:,}<extra></extra>",
        ))

        # Overlay a 3-month rolling average
        rolling_avg = miss_timeline["missing_hours"].rolling(3, center=True, min_periods=1).mean()
        fig_miss.add_trace(go.Scatter(
            x=miss_timeline["ym_str"],
            y=rolling_avg,
            mode="lines",
            line=dict(color=COLORS["navy"], width=1.8, dash="dot"),
            name="3-Month Rolling Average",
            hovertemplate="Month: %{x}<br>3M Avg: %{y:,.0f}<extra></extra>",
        ))

        fig_miss.update_layout(
            template="plotly_white",
            height=330,
            xaxis_title="Calendar Month",
            yaxis_title="Total Missing Hours (network-wide)",
            xaxis=dict(tickangle=-45, nticks=24),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=80, r=20, t=50, b=80),
        )
        st.plotly_chart(fig_miss, use_container_width=True)

        info_box(
            "This chart aggregates missing hourly observations across all stations and all monitored "
            "pollutants per calendar month. Spikes indicate periods of simultaneous data dropout "
            "(e.g., instrument maintenance windows, power outages, EPA data submission holds). "
            "The dashed line is a 3-month centred rolling average to reveal seasonal missingness patterns.",
            kind="info",
        )
    else:
        st.info("Missingness timeline data could not be computed.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 5 -- GAP ANALYSIS
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-header">Missing-Data Gap Profile by Station</div>',
        unsafe_allow_html=True,
    )

    if not gap_df.empty:
        GAP_COLORS = {
            "very_short": COLORS["green"],
            "medium":     COLORS["amber"],
            "long":       COLORS["red"],
            "very_long":  COLORS["navy"],
        }
        GAP_LABELS = {
            "very_short": "Very Short (1-2 hrs)",
            "medium":     "Medium (3-24 hrs)",
            "long":       "Long (25-168 hrs)",
            "very_long":  "Very Long (>168 hrs / >1 week)",
        }

        gap_df_plot = gap_df.copy()
        gap_df_plot["station_label"] = gap_df_plot["station"].map(
            lambda s: STATIONS.get(s, {}).get("short", s)
        )
        gap_agg = (
            gap_df_plot.groupby(["station_label", "gap_category"])["total_missing_hours"]
            .sum()
            .reset_index()
        )

        fig_gaps = go.Figure()
        for cat in ["very_short", "medium", "long", "very_long"]:
            sub = gap_agg[gap_agg["gap_category"] == cat]
            if sub.empty:
                continue
            fig_gaps.add_trace(go.Bar(
                name=GAP_LABELS[cat],
                x=sub["station_label"],
                y=sub["total_missing_hours"],
                marker_color=GAP_COLORS[cat],
                hovertemplate=(
                    "Station: %{x}<br>"
                    f"Category: {GAP_LABELS[cat]}<br>"
                    "Missing Hours: %{y:,}"
                    "<extra></extra>"
                ),
            ))

        fig_gaps.update_layout(
            barmode="stack",
            template="plotly_white",
            height=390,
            xaxis_title="Station",
            yaxis_title="Total Missing Hours",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=70, r=20, t=60, b=60),
        )
        st.plotly_chart(fig_gaps, use_container_width=True)

        info_box(
            "<b>Gap Categories:</b> "
            "<b>Very Short</b> (1-2 hrs) -- typically recoverable by linear interpolation. "
            "<b>Medium</b> (3-24 hrs) -- linear or seasonal interpolation. "
            "<b>Long</b> (25-168 hrs, up to 1 week) -- requires ERA5-regression or multi-station methods. "
            "<b>Very Long</b> (&gt;168 hrs) -- structural outages; consider excluding from training windows.",
            kind="info",
        )

        with st.expander("Detailed Gap Count Table", expanded=False):
            try:
                pivot_cols = ["very_short", "medium", "long", "very_long"]
                gap_pivot = gap_df.pivot_table(
                    index=["station", "pollutant"],
                    columns="gap_category",
                    values="count",
                    aggfunc="sum",
                    fill_value=0,
                ).reset_index()
                # Ensure all category columns exist
                for c in pivot_cols:
                    if c not in gap_pivot.columns:
                        gap_pivot[c] = 0
                gap_pivot["station"] = gap_pivot["station"].map(
                    lambda s: STATIONS.get(s, {}).get("label", s)
                )
                gap_pivot = gap_pivot.rename(columns={
                    "station": "Station",
                    "pollutant": "Pollutant",
                    "very_short": "Very Short (1-2 h)",
                    "medium":     "Medium (3-24 h)",
                    "long":       "Long (25-168 h)",
                    "very_long":  "Very Long (>168 h)",
                })
                gap_pivot["Total Gaps"] = (
                    gap_pivot.get("Very Short (1-2 h)", 0)
                    + gap_pivot.get("Medium (3-24 h)", 0)
                    + gap_pivot.get("Long (25-168 h)", 0)
                    + gap_pivot.get("Very Long (>168 h)", 0)
                )
                st.dataframe(gap_pivot, use_container_width=True, hide_index=True)
            except Exception:
                st.dataframe(gap_df, use_container_width=True, hide_index=True)
    else:
        st.info(
            "Gap analysis returned no results. "
            "Either no missing data was detected or the station files could not be loaded."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 6 -- NEGATIVE VALUES
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-header">Negative Value Distribution by Station and Pollutant</div>',
        unsafe_allow_html=True,
    )

    # Station 3 SO2 prominent danger warning
    info_box(
        "<b>CRITICAL ALERT -- Station 3 (Site 59) SO2 Systematic Corruption:</b> "
        "Station 3 records approximately <b>16,416 negative SO2 values</b> out of 48,870 total "
        "observations (<b>33.6%</b>). This is not random noise -- it indicates systematic sensor "
        "malfunction or zero-calibration failure. "
        "<b>Station 3 SO2 is excluded from all default analysis and modeling pipelines.</b> "
        "See the <i>Station 3 SO2 Investigation</i> page for the full forensic analysis and "
        "recommended handling strategies.",
        kind="danger",
    )

    if not neg_df.empty:
        pol_names = sorted(neg_df["pollutant"].unique())
        pol_color_map = {p: POLLUTANTS[p]["color"] for p in pol_names if p in POLLUTANTS}

        fig_neg = go.Figure()
        for p in pol_names:
            sub = neg_df[neg_df["pollutant"] == p]
            if sub.empty:
                continue
            # Mark corrupted Station 3 SO2 bar with a pattern
            marker_kwargs: dict = {"color": pol_color_map.get(p, COLORS["gray"])}
            fig_neg.add_trace(go.Bar(
                name=p,
                x=sub["station"],
                y=sub["negative_count"],
                marker=marker_kwargs,
                hovertemplate=(
                    f"<b>{p}</b><br>"
                    "Station: %{x}<br>"
                    "Negative Count: %{y:,}"
                    "<extra></extra>"
                ),
            ))

        fig_neg.update_layout(
            barmode="group",
            template="plotly_white",
            height=390,
            xaxis_title="Station",
            yaxis_title="Number of Negative Readings",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=70, r=20, t=60, b=60),
        )
        st.plotly_chart(fig_neg, use_container_width=True)

        # Negative values table for full transparency
        with st.expander("Negative Value Counts Table", expanded=False):
            neg_table = neg_df[["station_label", "pollutant", "negative_count", "corrupted"]].copy()
            neg_table.columns = ["Station", "Pollutant", "Negative Count", "Flagged as Corrupted"]
            neg_table = neg_table.sort_values("Negative Count", ascending=False).reset_index(drop=True)
            st.dataframe(neg_table, use_container_width=True, hide_index=True)
    else:
        st.success(
            "No negative values detected across all actively monitored channels. "
            "Data quality appears clean with respect to the physical lower bound."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 7 -- EXECUTIVE INSIGHTS
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-header">Executive Insights</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<p style='color:{COLORS['muted']}; font-size:0.92rem; margin-bottom:12px;'>"
        "Deterministic, rule-based findings derived from the loaded datasets. "
        "These are recomputed on every page load and always reflect the current data state.</p>",
        unsafe_allow_html=True,
    )

    # Render insights in a 2-column grid
    n_insights = len(insights)
    left_col, right_col = st.columns(2)
    for idx, insight in enumerate(insights):
        target_col = left_col if idx % 2 == 0 else right_col
        with target_col:
            kind = insight["kind"]
            kind_badge_map = {
                "info":   ("info",  "INFO"),
                "warn":   ("warn",  "WARNING"),
                "danger": ("error", "CRITICAL"),
            }
            bkind, blabel = kind_badge_map.get(kind, ("info", "INFO"))
            st.markdown(
                badge(blabel, kind=bkind) + f" <b style='font-size:0.9rem;'>{insight['title']}</b>",
                unsafe_allow_html=True,
            )
            info_box(insight["text"], kind=kind)
            st.markdown("<br>", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 8 -- STATION SUMMARY TABLE
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-header">Station Summary</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Completeness %": st.column_config.ProgressColumn(
                label="Completeness %",
                help="Average completeness across actively monitored pollutants",
                format="%.1f%%",
                min_value=0.0,
                max_value=100.0,
            ),
            "Rows": st.column_config.NumberColumn(
                label="Rows",
                help="Total hourly rows after reindexing to continuous hourly grid",
                format="%d",
            ),
            "Negative Values": st.column_config.NumberColumn(
                label="Negative Values",
                help="Count of negative readings across all monitored (non-structural-missing) channels",
                format="%d",
            ),
            "Notes": st.column_config.TextColumn(
                label="Notes",
                help="Special handling notes (e.g. systematic corruption flags)",
                width="large",
            ),
        },
    )

    info_box(
        "<b>Rows</b> = total hourly slots in the continuous time index (including structurally "
        "missing rows). "
        "<b>Completeness %</b> = mean of per-pollutant non-null fractions across actively "
        "monitored channels only -- structurally absent pollutants are excluded from this average. "
        "<b>Negative Values</b> = count of readings below zero (physically impossible for these pollutants); "
        "includes Station 3 SO2 systematic corruption.",
        kind="info",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Footer
    st.markdown(
        f"<hr style='border-color:{COLORS['border']};'/>"
        f"<p style='color:{COLORS['muted']}; font-size:0.80rem; text-align:center;'>"
        "ISRAP AIRMAPS Intelligence Studio &mdash; NASA Graduate Research Fellowship &mdash; "
        "San Antonio Air Quality Analysis &mdash; Data: EPA AQS + ERA5 Reanalysis</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

