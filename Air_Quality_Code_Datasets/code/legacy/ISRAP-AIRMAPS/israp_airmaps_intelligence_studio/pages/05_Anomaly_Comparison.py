"""05_Anomaly_Comparison.py — Multi-method anomaly comparison for ISRAP AIRMAPS.

Run all 8 detectors side-by-side on any station/pollutant, compare their
agreement, inspect a rich time-series overlay, build an ensemble, and download
a flagged-observations CSV.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.config import POLLUTANTS, STATIONS, ANOMALY_DEFAULTS, STRUCTURAL_MISSING_THRESHOLD
from src.data_loader import load_all_stations, load_era5, merge_era5
from src.anomaly_detection.physical_constraints import PhysicalConstraintDetector
from src.anomaly_detection.iqr_detector import IQRDetector
from src.anomaly_detection.hampel_detector import HampelDetector
from src.anomaly_detection.rolling_zscore import RollingZScoreDetector
from src.anomaly_detection.lof_detector import LOFDetector
from src.anomaly_detection.isolation_forest import IsolationForestDetector
from src.anomaly_detection.stl_detector import STLDetector
from src.anomaly_detection.screen_detector import SCREENDetector
from src.anomaly_detection.ensemble import build_ensemble, jaccard_matrix, method_agreement_matrix
from ui.theme import COLORS, inject_css, info_box, badge, metric_card, SERIES_COLORS


# ── Method registry ───────────────────────────────────────────────────────────
METHOD_REGISTRY: dict[str, dict[str, Any]] = {
    "Physical Constraints": {
        "class": PhysicalConstraintDetector,
        "key": "physical_constraints",
        "flag": "is_physical_anomaly",
        "score": "physical_constraint_score",
        "category": "Physical",
        "best_for": "Impossible values & sensor hardware errors",
        "limitation": "Cannot detect elevated-but-valid readings",
        "color": COLORS["red"],
        "symbol": "x",
    },
    "IQR Fence": {
        "class": IQRDetector,
        "key": "iqr",
        "flag": "is_iqr_anomaly",
        "score": "iqr_score",
        "category": "Statistical",
        "best_for": "Isolated point spikes in a stable distribution",
        "limitation": "Insensitive to seasonality; skew can distort fences",
        "color": COLORS["amber"],
        "symbol": "diamond-open",
    },
    "Hampel Filter": {
        "class": HampelDetector,
        "key": "hampel",
        "flag": "is_hampel_anomaly",
        "score": "hampel_score",
        "category": "Robust Temporal",
        "best_for": "Isolated spikes in slowly varying signals",
        "limitation": "Can miss slow drifts; window size matters",
        "color": COLORS["violet"],
        "symbol": "circle-open",
    },
    "Rolling Z-Score": {
        "class": RollingZScoreDetector,
        "key": "rolling_zscore",
        "flag": "is_rolling_z_anomaly",
        "score": "rolling_z_score",
        "category": "Statistical Temporal",
        "best_for": "Local level shifts and transient spikes",
        "limitation": "Assumes local normality; window-size sensitive",
        "color": COLORS["teal"],
        "symbol": "triangle-up",
    },
    "Local Outlier Factor": {
        "class": LOFDetector,
        "key": "lof",
        "flag": "is_lof_anomaly",
        "score": "lof_score",
        "category": "Machine Learning",
        "best_for": "Contextual anomalies using ERA5 meteorological context",
        "limitation": "Computationally heavier; needs sufficient valid rows",
        "color": COLORS["green"],
        "symbol": "star",
    },
    "Isolation Forest": {
        "class": IsolationForestDetector,
        "key": "isolation_forest",
        "flag": "is_iforest_anomaly",
        "score": "iforest_score",
        "category": "Machine Learning",
        "best_for": "High-dimensional outliers with ERA5 features",
        "limitation": "Contamination parameter requires tuning",
        "color": COLORS["blue"],
        "symbol": "square-open",
    },
    "STL Residual": {
        "class": STLDetector,
        "key": "stl",
        "flag": "is_stl_anomaly",
        "score": "stl_score",
        "category": "Seasonal Temporal",
        "best_for": "Anomalies in residuals after seasonal decomposition",
        "limitation": "Requires sufficient data for STL fitting",
        "color": "#f72585",
        "symbol": "cross",
    },
    "SCREEN Rate-of-Change": {
        "class": SCREENDetector,
        "key": "screen",
        "flag": "is_screen_anomaly",
        "score": "screen_score",
        "category": "Rate-of-Change",
        "best_for": "Unrealistic hour-to-hour concentration jumps",
        "limitation": "Misses sustained elevated levels; requires steady-state assumption",
        "color": "#4cc9f0",
        "symbol": "pentagon",
    },
}

_MNAMES = list(METHOD_REGISTRY.keys())
_DEFAULT_METHODS = ["IQR Fence", "Hampel Filter", "Rolling Z-Score", "Isolation Forest", "STL Residual"]
_STATE_KEY = "anc_state"


# ── Cached data loader ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading station data…")
def _load() -> dict[str, pd.DataFrame]:
    stations = load_all_stations()
    era5 = load_era5()
    return {k: merge_era5(v, era5) for k, v in stations.items()}


# ── Helper utilities ──────────────────────────────────────────────────────────
def _status_badge(status: str) -> str:
    kind = {"ok": "ok", "skipped": "skip", "error": "error"}.get(status, "info")
    return badge(status.upper(), kind)


def _method_color(name: str) -> str:
    return METHOD_REGISTRY.get(name, {}).get("color", COLORS["gray"])


def _flag_col(name: str) -> str:
    return METHOD_REGISTRY.get(name, {}).get("flag", "")


def _score_col(name: str) -> str:
    return METHOD_REGISTRY.get(name, {}).get("score", "")


def _ens_key(name: str) -> str:
    return METHOD_REGISTRY.get(name, {}).get("key", "")


def _available_flag_cols(df: pd.DataFrame, methods: list[str]) -> dict[str, str]:
    """Return {method_name: flag_col} for methods whose flag column exists in df."""
    out = {}
    for m in methods:
        fc = _flag_col(m)
        if fc and fc in df.columns:
            out[m] = fc
    return out


def _count_by_n_methods(df: pd.DataFrame, methods: list[str]) -> pd.Series:
    """Return series indexed 0..N of how many rows are flagged by exactly k methods."""
    avail = _available_flag_cols(df, methods)
    if not avail:
        return pd.Series(dtype=int)
    mat = pd.concat(
        [df[fc].fillna(False).astype(int) for fc in avail.values()], axis=1
    )
    mat.columns = list(avail.keys())
    counts = mat.sum(axis=1)
    return counts.value_counts().sort_index()


# ── Sidebar controls ──────────────────────────────────────────────────────────
def _render_sidebar(station_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Render all sidebar controls and return a params dict."""
    with st.sidebar:
        st.markdown(
            f"<h2 style='color:#ffffff; margin-bottom:4px;'>⚖️ Anomaly Comparison</h2>"
            f"<p style='color:#94d2bd; font-size:0.8rem; margin-top:0;'>Compare methods side-by-side</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        # Station & pollutant
        sel_station = st.selectbox(
            "Station",
            list(station_data.keys()),
            format_func=lambda k: STATIONS[k]["label"],
            key="anc_station",
        )
        meta = STATIONS[sel_station]
        available_pollutants = meta["monitored_pollutants"]
        sel_pollutant = st.selectbox(
            "Pollutant",
            available_pollutants,
            key="anc_pollutant",
        )

        # Date range
        df_s = station_data[sel_station]
        dt_min = df_s["datetime"].min().date()
        dt_max = df_s["datetime"].max().date()
        date_range = st.date_input(
            "Date range",
            value=(dt_min, dt_max),
            min_value=dt_min,
            max_value=dt_max,
            key="anc_dates",
        )
        # Normalize to always be a 2-tuple
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            date_start, date_end = date_range[0], date_range[1]
        else:
            date_start, date_end = dt_min, dt_max

        # Method selector
        st.markdown("**Methods to compare**")
        sel_methods = st.multiselect(
            "Select methods",
            _MNAMES,
            default=_DEFAULT_METHODS,
            key="anc_methods",
            label_visibility="collapsed",
        )

        # Advanced parameters per method
        params: dict[str, Any] = {}
        with st.expander("Advanced parameters", expanded=False):

            if "Physical Constraints" in sel_methods:
                st.markdown("**Physical Constraints**")
                params["pc_min"] = st.number_input(
                    "Physical min", value=None, format="%.4f", key="anc_pc_min",
                    help="Override default physical minimum (leave blank for auto)"
                )
                params["pc_max"] = st.number_input(
                    "Physical max", value=None, format="%.4f", key="anc_pc_max",
                    help="Override default physical maximum (leave blank for auto)"
                )

            if "IQR Fence" in sel_methods:
                st.markdown("**IQR Fence**")
                params["iqr_mult"] = st.slider(
                    "IQR multiplier k", 1.0, 6.0,
                    float(ANOMALY_DEFAULTS["iqr_multiplier"]), 0.25, key="anc_iqr_k"
                )

            if "Hampel Filter" in sel_methods:
                st.markdown("**Hampel Filter**")
                params["hampel_win"] = st.slider(
                    "Window (hours)", 4, 168,
                    int(ANOMALY_DEFAULTS["hampel_window"]), 4, key="anc_hamp_win"
                )
                params["hampel_thr"] = st.slider(
                    "Threshold σ", 1.0, 6.0,
                    float(ANOMALY_DEFAULTS["hampel_threshold"]), 0.25, key="anc_hamp_thr"
                )

            if "Rolling Z-Score" in sel_methods:
                st.markdown("**Rolling Z-Score**")
                params["rz_win"] = st.slider(
                    "Window (hours)", 12, 720,
                    int(ANOMALY_DEFAULTS["rolling_zscore_window"]), 12, key="anc_rz_win"
                )
                params["rz_thr"] = st.slider(
                    "Threshold σ", 1.0, 6.0,
                    float(ANOMALY_DEFAULTS["rolling_zscore_threshold"]), 0.25, key="anc_rz_thr"
                )

            if "STL Residual" in sel_methods:
                st.markdown("**STL Residual**")
                params["stl_seas"] = st.slider(
                    "Seasonal period", 13, 169,
                    int(ANOMALY_DEFAULTS["stl_seasonal"]) + 1, 2, key="anc_stl_seas"
                )
                params["stl_thr"] = st.slider(
                    "Threshold", 1.0, 6.0,
                    float(ANOMALY_DEFAULTS["stl_threshold"]), 0.25, key="anc_stl_thr"
                )

            if "Local Outlier Factor" in sel_methods:
                st.markdown("**Local Outlier Factor**")
                params["lof_k"] = st.slider(
                    "n_neighbors", 5, 50,
                    int(ANOMALY_DEFAULTS["lof_n_neighbors"]), 1, key="anc_lof_k"
                )
                params["lof_cont"] = st.slider(
                    "Contamination", 0.001, 0.10,
                    float(ANOMALY_DEFAULTS["lof_contamination"]), 0.001,
                    format="%.3f", key="anc_lof_cont"
                )

            if "Isolation Forest" in sel_methods:
                st.markdown("**Isolation Forest**")
                params["if_cont"] = st.slider(
                    "Contamination", 0.001, 0.10,
                    float(ANOMALY_DEFAULTS["iforest_contamination"]), 0.001,
                    format="%.3f", key="anc_if_cont"
                )
                params["if_trees"] = st.slider(
                    "n_estimators", 50, 500,
                    int(ANOMALY_DEFAULTS["iforest_n_estimators"]), 50, key="anc_if_trees"
                )

            if "SCREEN Rate-of-Change" in sel_methods:
                st.markdown("**SCREEN Rate-of-Change**")
                params["screen_pct"] = st.slider(
                    "Percentile threshold", 90.0, 99.9,
                    float(ANOMALY_DEFAULTS["screen_percentile"]), 0.5,
                    format="%.1f", key="anc_screen_pct"
                )

        st.divider()
        run_btn = st.button(
            "▶ Run Comparison",
            type="primary",
            use_container_width=True,
            disabled=len(sel_methods) == 0,
        )

        if len(sel_methods) == 0:
            st.warning("Select at least one method.")

    return {
        "sel_station": sel_station,
        "sel_pollutant": sel_pollutant,
        "date_start": date_start,
        "date_end": date_end,
        "sel_methods": sel_methods,
        "params": params,
        "run": run_btn,
        "df_s": df_s,
    }


# ── Run detectors ─────────────────────────────────────────────────────────────
def _run_detectors(
    df_filtered: pd.DataFrame,
    sel_methods: list[str],
    params: dict[str, Any],
    pollutant: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute all selected detectors sequentially. Returns updated df + results dict."""
    df_work = df_filtered.copy()
    results: dict[str, Any] = {}

    progress = st.progress(0, text="Initialising…")
    total = len(sel_methods)

    for i, mname in enumerate(sel_methods):
        progress.progress((i) / total, text=f"Running {mname}…")
        meta = METHOD_REGISTRY[mname]
        detector_cls = meta["class"]
        detector = detector_cls()

        detect_kwargs: dict[str, Any] = {}
        if mname == "Physical Constraints":
            if params.get("pc_min") is not None:
                detect_kwargs["physical_min"] = params["pc_min"]
            if params.get("pc_max") is not None:
                detect_kwargs["physical_max"] = params["pc_max"]
        elif mname == "IQR Fence":
            detect_kwargs["multiplier"] = params.get("iqr_mult", ANOMALY_DEFAULTS["iqr_multiplier"])
        elif mname == "Hampel Filter":
            detect_kwargs["window"] = params.get("hampel_win", ANOMALY_DEFAULTS["hampel_window"])
            detect_kwargs["threshold"] = params.get("hampel_thr", ANOMALY_DEFAULTS["hampel_threshold"])
        elif mname == "Rolling Z-Score":
            detect_kwargs["window"] = params.get("rz_win", ANOMALY_DEFAULTS["rolling_zscore_window"])
            detect_kwargs["threshold"] = params.get("rz_thr", ANOMALY_DEFAULTS["rolling_zscore_threshold"])
        elif mname == "STL Residual":
            detect_kwargs["seasonal"] = params.get("stl_seas", ANOMALY_DEFAULTS["stl_seasonal"])
            detect_kwargs["threshold"] = params.get("stl_thr", ANOMALY_DEFAULTS["stl_threshold"])
        elif mname == "Local Outlier Factor":
            detect_kwargs["n_neighbors"] = params.get("lof_k", ANOMALY_DEFAULTS["lof_n_neighbors"])
            detect_kwargs["contamination"] = params.get("lof_cont", ANOMALY_DEFAULTS["lof_contamination"])
        elif mname == "Isolation Forest":
            detect_kwargs["contamination"] = params.get("if_cont", ANOMALY_DEFAULTS["iforest_contamination"])
            detect_kwargs["n_estimators"] = params.get("if_trees", ANOMALY_DEFAULTS["iforest_n_estimators"])
        elif mname == "SCREEN Rate-of-Change":
            detect_kwargs["percentile"] = params.get("screen_pct", ANOMALY_DEFAULTS["screen_percentile"])

        try:
            df_work, res = detector.detect(df_work, pollutant, **detect_kwargs)
            results[mname] = res
        except Exception as exc:
            st.warning(f"{mname} failed: {exc}")

    progress.progress(1.0, text="Done.")
    time.sleep(0.3)
    progress.empty()

    return df_work, results


# ── TAB 1: Method Summary ─────────────────────────────────────────────────────
def _tab_summary(df_work: pd.DataFrame, results: dict, sel_methods: list[str], pollutant: str) -> None:
    unit = POLLUTANTS.get(pollutant, {}).get("unit", "")

    # Top metric row
    total_obs = df_work[pollutant].notna().sum()
    all_flags = pd.concat(
        [df_work[_flag_col(m)].fillna(False) for m in sel_methods if _flag_col(m) in df_work.columns],
        axis=1,
    )
    union_count = int(all_flags.any(axis=1).sum()) if not all_flags.empty else 0
    intersection_count = int(all_flags.all(axis=1).sum()) if not all_flags.empty else 0
    mean_runtime = float(np.mean([r.runtime_sec for r in results.values()])) if results else 0.0

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(metric_card("Valid observations", f"{total_obs:,}", color=COLORS["blue"]), unsafe_allow_html=True)
    with mc2:
        pct_union = round(100 * union_count / max(total_obs, 1), 2)
        st.markdown(metric_card("Union flagged", f"{union_count:,}", delta=f"{pct_union}%", color=COLORS["red"]), unsafe_allow_html=True)
    with mc3:
        pct_inter = round(100 * intersection_count / max(total_obs, 1), 2)
        st.markdown(metric_card("All methods agree", f"{intersection_count:,}", delta=f"{pct_inter}%", color=COLORS["amber"]), unsafe_allow_html=True)
    with mc4:
        st.markdown(metric_card("Mean runtime", f"{mean_runtime:.3f}s", color=COLORS["teal"]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Method Results Table</div>', unsafe_allow_html=True)

    rows = []
    badge_html_parts = []
    for mname in sel_methods:
        if mname not in results:
            continue
        res = results[mname]
        rows.append({
            "Method": mname,
            "Category": METHOD_REGISTRY[mname]["category"],
            "Flagged": res.n_flagged,
            "Flagged %": f"{res.pct_flagged:.3f}%",
            "Runtime (s)": f"{res.runtime_sec:.3f}",
            "Status": res.status,
            "Warning": res.warning or "—",
        })
        bkind = {"ok": "ok", "skipped": "skip", "error": "error"}.get(res.status, "info")
        badge_html_parts.append(
            f"<span style='margin-right:8px;'>"
            f"<b style='font-size:0.85rem;'>{mname}</b>&nbsp;"
            f"{badge(res.status.upper(), bkind)}"
            f"</span>"
        )

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Badges row
    st.markdown("**Status badges:** " + " ".join(badge_html_parts), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Flagged Count by Method</div>', unsafe_allow_html=True)

    # Horizontal bar chart
    if rows:
        sorted_rows = sorted(rows, key=lambda r: int(r["Flagged"]))
        fig_bar = go.Figure(go.Bar(
            x=[r["Flagged"] for r in sorted_rows],
            y=[r["Method"] for r in sorted_rows],
            orientation="h",
            marker_color=[_method_color(r["Method"]) for r in sorted_rows],
            text=[f"{r['Flagged']} ({r['Flagged %']})" for r in sorted_rows],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Flagged: %{x}<extra></extra>",
        ))
        fig_bar.update_layout(
            template="plotly_white",
            height=max(260, len(rows) * 44),
            margin=dict(l=20, r=80, t=20, b=30),
            xaxis_title="Observations flagged",
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ── TAB 2: Overlap Analysis ───────────────────────────────────────────────────
def _tab_overlap(df_work: pd.DataFrame, sel_methods: list[str]) -> None:
    ens_keys = [_ens_key(m) for m in sel_methods if _ens_key(m)]

    if len(ens_keys) < 2:
        info_box("Select at least 2 methods to view overlap analysis.", "warn")
        return

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">Jaccard Similarity</div>', unsafe_allow_html=True)
        jacc = jaccard_matrix(df_work, ens_keys)
        if not jacc.empty:
            # Rename index/cols back to display names
            key_to_name = {_ens_key(m): m for m in sel_methods}
            jacc = jacc.rename(index=key_to_name, columns=key_to_name)
            fig_jacc = go.Figure(go.Heatmap(
                z=jacc.values,
                x=list(jacc.columns),
                y=list(jacc.index),
                colorscale="Teal",
                zmin=0, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in jacc.values],
                texttemplate="%{text}",
                hovertemplate="%{y} ↔ %{x}: %{z:.3f}<extra></extra>",
                colorbar=dict(title="Jaccard", thickness=12),
            ))
            fig_jacc.update_layout(
                template="plotly_white", height=340,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(tickangle=-30),
            )
            st.plotly_chart(fig_jacc, use_container_width=True)
            info_box(
                "Jaccard = intersection / union of flags. 1.0 = perfect agreement, 0.0 = no overlap.",
                "info",
            )
        else:
            st.info("Insufficient flag data to compute Jaccard matrix.")

    with c2:
        st.markdown('<div class="section-header">Co-Flagging Agreement</div>', unsafe_allow_html=True)
        agree = method_agreement_matrix(df_work, ens_keys)
        if not agree.empty:
            key_to_name = {_ens_key(m): m for m in sel_methods}
            agree = agree.rename(index=key_to_name, columns=key_to_name)
            fig_agree = go.Figure(go.Heatmap(
                z=agree.values,
                x=list(agree.columns),
                y=list(agree.index),
                colorscale="Blues",
                text=agree.values.astype(str),
                texttemplate="%{text}",
                hovertemplate="%{y} ∩ %{x}: %{z} shared flags<extra></extra>",
                colorbar=dict(title="Count", thickness=12),
            ))
            fig_agree.update_layout(
                template="plotly_white", height=340,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(tickangle=-30),
            )
            st.plotly_chart(fig_agree, use_container_width=True)
            info_box("Diagonal = self-agreement (total flags per method).", "info")
        else:
            st.info("Insufficient flag data to compute agreement matrix.")

    # Upset-style distribution
    st.markdown('<div class="section-header">Agreement Distribution — Flagged by Exactly N Methods</div>', unsafe_allow_html=True)
    dist = _count_by_n_methods(df_work, sel_methods)
    if not dist.empty:
        dist = dist.sort_index()
        labels = [
            "Unflagged" if k == 0 else (f"Exactly {k} method" if k == 1 else f"Exactly {k} methods")
            for k in dist.index
        ]
        bar_colors = [COLORS["gray"]] + [SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(len(dist) - 1)]
        fig_dist = go.Figure(go.Bar(
            x=labels,
            y=dist.values,
            marker_color=bar_colors,
            text=[f"{v:,}" for v in dist.values],
            textposition="outside",
            hovertemplate="%{x}<br>Count: %{y:,}<extra></extra>",
        ))
        fig_dist.update_layout(
            template="plotly_white", height=300,
            margin=dict(l=20, r=20, t=10, b=60),
            yaxis_title="Observation count",
            xaxis_tickangle=-20,
            showlegend=False,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # Summary sentence
        if len(dist) > 1:
            unanimous = int(dist.get(len(sel_methods), 0))
            any_one = int(dist[dist.index > 0].sum())
            st.markdown(
                f"Of **{any_one:,}** flagged observations, "
                f"**{unanimous:,}** ({round(100*unanimous/max(any_one,1),1)}%) "
                f"are flagged by all {len(sel_methods)} selected methods.",
                unsafe_allow_html=False,
            )


# ── TAB 3: Timeline Comparison ────────────────────────────────────────────────
def _tab_timeline(df_work: pd.DataFrame, sel_methods: list[str], pollutant: str) -> None:
    unit = POLLUTANTS.get(pollutant, {}).get("unit", "")

    avail = _available_flag_cols(df_work, sel_methods)
    if not avail:
        st.info("No anomaly flag columns found in results. Re-run the comparison.")
        return

    # Downsample if very large
    df_plot = df_work.copy()
    if len(df_plot) > 8000:
        step = len(df_plot) // 8000 + 1
        df_plot = df_plot.iloc[::step].copy()

    # Compute per-point agreement count
    flag_mat = pd.concat(
        [df_work[fc].fillna(False).astype(int) for fc in avail.values()], axis=1
    )
    flag_mat.columns = list(avail.keys())
    count_series = flag_mat.sum(axis=1)

    # Subplot: signal on top, agreement bar at bottom
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.72, 0.28],
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[f"{pollutant} ({unit}) with anomaly markers", "Agreement count (methods flagging this point)"],
    )

    # Raw signal
    fig.add_trace(
        go.Scatter(
            x=df_plot["datetime"], y=df_plot[pollutant],
            mode="lines", name="Observed",
            line=dict(color=COLORS["blue"], width=1.0),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>" + f"{pollutant}: %{{y:.4f}} {unit}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Anomaly markers per method
    for mname, fc in avail.items():
        meta = METHOD_REGISTRY[mname]
        anom = df_work[df_work[fc].fillna(False)]
        if anom.empty:
            continue
        # If downsampled, also downsample anomaly points
        if len(df_work) > 8000:
            step = len(df_work) // 8000 + 1
            anom = anom[anom.index % step == 0]
        fig.add_trace(
            go.Scatter(
                x=anom["datetime"], y=anom[pollutant],
                mode="markers", name=mname,
                marker=dict(
                    color=meta["color"], size=7,
                    symbol=meta["symbol"],
                    line=dict(width=1.2, color=meta["color"]),
                ),
                hovertemplate=f"<b>{mname}</b><br>%{{x|%Y-%m-%d %H:%M}}<br>{pollutant}: %{{y:.4f}} {unit}<extra></extra>",
            ),
            row=1, col=1,
        )

    # Agreement count bar (sampled to df_work index)
    cnt_plot = count_series.copy()
    if len(cnt_plot) > 8000:
        step_cnt = len(cnt_plot) // 8000 + 1
        cnt_plot = cnt_plot.iloc[::step_cnt]
        dt_bar = df_work["datetime"].iloc[::step_cnt]
    else:
        dt_bar = df_work["datetime"]

    fig.add_trace(
        go.Bar(
            x=dt_bar, y=cnt_plot.values,
            name="Agreement",
            marker_color=COLORS["teal"],
            opacity=0.7,
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Methods agreeing: %{y}<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        template="plotly_white",
        height=580,
        legend=dict(orientation="h", y=-0.10, x=0, font_size=11),
        margin=dict(l=60, r=20, t=50, b=80),
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=False)),
        xaxis2=dict(rangeslider=dict(visible=True, thickness=0.04)),
    )
    fig.update_yaxes(title_text=f"{pollutant} ({unit})", row=1, col=1)
    fig.update_yaxes(title_text="# methods", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    n_total = int((count_series > 0).sum())
    n_multi = int((count_series >= 2).sum())
    st.caption(
        f"Showing {len(df_plot):,} of {len(df_work):,} points "
        f"(downsampled for performance). "
        f"Total flagged by any method: {n_total:,} | "
        f"Flagged by 2+ methods: {n_multi:,}."
    )


# ── TAB 4: Ensemble ──────────────────────────────────────────────────────────
def _tab_ensemble(df_work: pd.DataFrame, sel_methods: list[str], pollutant: str) -> None:
    unit = POLLUTANTS.get(pollutant, {}).get("unit", "")
    ens_keys = [_ens_key(m) for m in sel_methods if _ens_key(m)]

    st.markdown('<div class="section-header">Ensemble Configuration</div>', unsafe_allow_html=True)

    e1, e2 = st.columns([2, 1])
    with e1:
        ens_mode = st.radio(
            "Ensemble strategy",
            ["Majority Vote (N of M)", "Any method", "All methods (intersection)"],
            horizontal=True,
            key="anc_ens_mode",
        )
    with e2:
        majority_k = st.slider(
            "Majority threshold (N)",
            min_value=1,
            max_value=max(len(sel_methods), 1),
            value=min(2, len(sel_methods)),
            key="anc_majority_k",
        )

    if ens_mode == "Any method":
        majority_k = 1
    elif ens_mode == "All methods (intersection)":
        majority_k = len(sel_methods)

    # Build ensemble
    df_ens = build_ensemble(df_work, ens_keys, majority_threshold=majority_k)

    # Metrics
    total_valid = int(df_ens[pollutant].notna().sum())
    ens_flagged = int(df_ens["combined_anomaly_flag"].sum())
    ens_pct = round(100 * ens_flagged / max(total_valid, 1), 3)
    high_conf = int((df_ens["anomaly_confidence"] == "High").sum() + (df_ens["anomaly_confidence"] == "Very High").sum())
    med_conf = int((df_ens["anomaly_confidence"] == "Medium").sum())
    low_conf = int((df_ens["anomaly_confidence"] == "Low").sum())

    em1, em2, em3, em4 = st.columns(4)
    with em1:
        st.markdown(metric_card("Ensemble flagged", f"{ens_flagged:,}", delta=f"{ens_pct}%", color=COLORS["red"]), unsafe_allow_html=True)
    with em2:
        st.markdown(metric_card("High / Very High confidence", f"{high_conf:,}", color=COLORS["violet"]), unsafe_allow_html=True)
    with em3:
        st.markdown(metric_card("Medium confidence", f"{med_conf:,}", color=COLORS["amber"]), unsafe_allow_html=True)
    with em4:
        st.markdown(metric_card("Low confidence", f"{low_conf:,}", color=COLORS["teal"]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    info_box(
        f"<b>Strategy:</b> {ens_mode} &nbsp;|&nbsp; "
        f"<b>Threshold:</b> {majority_k} of {len(sel_methods)} methods must agree. "
        "Confidence is elevated to Very High if Physical Constraints also fires.",
        "info",
    )

    # Ensemble timeline
    st.markdown('<div class="section-header">Ensemble Flagged Observations</div>', unsafe_allow_html=True)

    df_ens_plot = df_ens.copy()
    if len(df_ens_plot) > 8000:
        step = len(df_ens_plot) // 8000 + 1
        df_ens_plot = df_ens_plot.iloc[::step]

    conf_colors = {"Very High": COLORS["red"], "High": COLORS["violet"],
                   "Medium": COLORS["amber"], "Low": COLORS["teal"], "None": COLORS["gray"]}

    fig_ens = go.Figure()
    fig_ens.add_trace(go.Scatter(
        x=df_ens_plot["datetime"], y=df_ens_plot[pollutant],
        mode="lines", name="Observed",
        line=dict(color=COLORS["blue"], width=1.1),
    ))

    for conf_level, col_hex in conf_colors.items():
        if conf_level == "None":
            continue
        mask = df_ens["combined_anomaly_flag"] & (df_ens["anomaly_confidence"] == conf_level)
        sub = df_ens[mask]
        if sub.empty:
            continue
        fig_ens.add_trace(go.Scatter(
            x=sub["datetime"], y=sub[pollutant],
            mode="markers", name=f"{conf_level} confidence",
            marker=dict(color=col_hex, size=8, symbol="circle",
                        line=dict(width=1.5, color="white")),
            hovertemplate=(
                f"<b>{conf_level} confidence</b><br>"
                "%{x|%Y-%m-%d %H:%M}<br>"
                f"{pollutant}: %{{y:.4f}} {unit}<extra></extra>"
            ),
        ))

    # Method count shading
    fig_ens.add_trace(go.Bar(
        x=df_ens_plot["datetime"],
        y=df_ens_plot.get("anomaly_method_count", pd.Series(0, index=df_ens_plot.index)),
        name="Method count",
        marker_color=COLORS["teal"],
        opacity=0.18,
        yaxis="y2",
        hovertemplate="Methods agreeing: %{y}<extra></extra>",
    ))

    fig_ens.update_layout(
        template="plotly_white", height=440, hovermode="x unified",
        legend=dict(orientation="h", y=-0.15, font_size=11),
        margin=dict(l=60, r=60, t=30, b=90),
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.04)),
        yaxis=dict(title=f"{pollutant} ({unit})"),
        yaxis2=dict(title="Method count", overlaying="y", side="right",
                    showgrid=False, range=[0, len(sel_methods) * 3]),
    )
    st.plotly_chart(fig_ens, use_container_width=True)

    # Recommended action breakdown
    if "recommended_action" in df_ens.columns:
        st.markdown('<div class="section-header">Recommended Actions</div>', unsafe_allow_html=True)
        action_counts = df_ens["recommended_action"].value_counts()
        action_map = {
            "confirm_invalid": ("Confirm invalid", COLORS["red"]),
            "review": ("Review required", COLORS["amber"]),
            "monitor": ("Monitor", COLORS["teal"]),
            "no_action": ("No action", COLORS["gray"]),
        }
        cols_act = st.columns(len(action_counts))
        for ci, (act, cnt) in enumerate(action_counts.items()):
            label, color = action_map.get(act, (act, COLORS["gray"]))
            with cols_act[ci % len(cols_act)]:
                st.markdown(metric_card(label, f"{cnt:,}", color=color), unsafe_allow_html=True)


# ── TAB 5: Method Scorecards ─────────────────────────────────────────────────
def _tab_scorecards(df_work: pd.DataFrame, results: dict, sel_methods: list[str]) -> None:
    if not results:
        st.info("No results available. Run the comparison first.")
        return

    avail = _available_flag_cols(df_work, sel_methods)
    all_flag_series = {m: df_work[fc].fillna(False).astype(bool) for m, fc in avail.items()}

    n_cols = min(len(sel_methods), 3)
    cols = st.columns(n_cols)

    for ci, mname in enumerate(sel_methods):
        if mname not in results:
            continue
        res = results[mname]
        meta = METHOD_REGISTRY[mname]
        col = cols[ci % n_cols]

        with col:
            fc = avail.get(mname)
            my_flags = all_flag_series.get(mname, pd.Series(False, index=df_work.index))

            # Unique (flagged only by this method)
            others = [s for m, s in all_flag_series.items() if m != mname]
            if others:
                others_union = pd.concat(others, axis=1).any(axis=1)
                unique_count = int((my_flags & ~others_union).sum())
            else:
                unique_count = int(my_flags.sum())

            # Common with all selected methods
            if len(all_flag_series) > 1:
                all_intersection = pd.concat(list(all_flag_series.values()), axis=1).all(axis=1)
                common_count = int((my_flags & all_intersection).sum())
            else:
                common_count = int(my_flags.sum())

            bkind = {"ok": "ok", "skipped": "skip", "error": "error"}.get(res.status, "info")
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {meta['color']}; margin-bottom:16px;">
                <div class="metric-label" style="color:{meta['color']};">
                    {meta['category']} &nbsp; {badge(res.status.upper(), bkind)}
                </div>
                <div style="font-size:1.1rem; font-weight:700; color:{COLORS['navy']}; margin:6px 0 2px;">
                    {mname}
                </div>
                <hr style="margin:6px 0; border-color:{COLORS['border']};">
                <table style="width:100%; font-size:0.83rem; color:{COLORS['text']};">
                  <tr><td>Flagged</td><td style="text-align:right;"><b>{res.n_flagged:,}</b></td></tr>
                  <tr><td>Flagged&nbsp;%</td><td style="text-align:right;"><b>{res.pct_flagged:.3f}%</b></td></tr>
                  <tr><td>Unique&nbsp;flags</td><td style="text-align:right;"><b>{unique_count:,}</b></td></tr>
                  <tr><td>Common&nbsp;(all)</td><td style="text-align:right;"><b>{common_count:,}</b></td></tr>
                  <tr><td>Runtime</td><td style="text-align:right;"><b>{res.runtime_sec:.3f}s</b></td></tr>
                </table>
                <hr style="margin:6px 0; border-color:{COLORS['border']};">
                <div style="font-size:0.78rem; color:{COLORS['muted']};">
                    <b>Best for:</b> {meta['best_for']}<br>
                    <b>Limitation:</b> {meta['limitation']}
                </div>
                {f'<div class="warn-box" style="margin-top:6px; font-size:0.76rem;">{res.warning}</div>' if res.warning else ''}
            </div>
            """, unsafe_allow_html=True)

    # Normalised score overlay (all methods on one chart)
    st.markdown('<div class="section-header">Normalised Score Comparison</div>', unsafe_allow_html=True)
    score_available = [(m, _score_col(m)) for m in sel_methods if _score_col(m) in df_work.columns]
    if score_available:
        df_sc = df_work.copy()
        if len(df_sc) > 8000:
            step = len(df_sc) // 8000 + 1
            df_sc = df_sc.iloc[::step]

        fig_sc = go.Figure()
        for mname, sc in score_available:
            sc_ser = df_work[sc].fillna(0)
            q99 = sc_ser.quantile(0.99)
            if q99 > 0:
                sc_norm = (sc_ser / q99).clip(0, 3)
            else:
                sc_norm = sc_ser
            # subsample
            if len(df_work) > 8000:
                step = len(df_work) // 8000 + 1
                dt_sc = df_work["datetime"].iloc[::step]
                sc_plot = sc_norm.iloc[::step]
            else:
                dt_sc = df_work["datetime"]
                sc_plot = sc_norm

            fig_sc.add_trace(go.Scatter(
                x=dt_sc, y=sc_plot,
                mode="lines", name=mname,
                line=dict(color=METHOD_REGISTRY[mname]["color"], width=1.0),
                opacity=0.8,
                hovertemplate=f"<b>{mname}</b> score: %{{y:.3f}}<extra></extra>",
            ))

        fig_sc.update_layout(
            template="plotly_white", height=320, hovermode="x unified",
            legend=dict(orientation="h", y=-0.15, font_size=11),
            margin=dict(l=60, r=20, t=20, b=80),
            yaxis_title="Normalised score (clipped at 3×P99)",
            xaxis=dict(rangeslider=dict(visible=True, thickness=0.04)),
        )
        st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.info("No score columns found in the results dataframe.")


# ── TAB 6: Download ───────────────────────────────────────────────────────────
def _tab_download(
    df_work: pd.DataFrame,
    results: dict,
    sel_methods: list[str],
    sel_station: str,
    sel_pollutant: str,
) -> None:
    st.markdown('<div class="section-header">Export Flagged Observations</div>', unsafe_allow_html=True)

    avail = _available_flag_cols(df_work, sel_methods)

    # Flagged by at least one method
    if avail:
        flag_union = pd.concat(
            [df_work[fc].fillna(False) for fc in avail.values()], axis=1
        ).any(axis=1)
        df_flagged = df_work[flag_union].copy()
    else:
        df_flagged = df_work.copy()

    # Select useful columns
    keep_cols = ["datetime", sel_pollutant]
    for mname in sel_methods:
        fc = _flag_col(mname)
        sc = _score_col(mname)
        if fc in df_work.columns:
            keep_cols.append(fc)
        if sc in df_work.columns:
            keep_cols.append(sc)
    if "anomaly_method_count" in df_work.columns:
        keep_cols.append("anomaly_method_count")
    if "anomaly_confidence" in df_work.columns:
        keep_cols.append("anomaly_confidence")
    if "recommended_action" in df_work.columns:
        keep_cols.append("recommended_action")

    df_export = df_flagged[[c for c in keep_cols if c in df_flagged.columns]].copy()
    df_full_export = df_work[[c for c in keep_cols if c in df_work.columns]].copy()

    info_box(
        f"<b>{len(df_export):,}</b> observations flagged by at least one method "
        f"out of <b>{df_work[sel_pollutant].notna().sum():,}</b> valid observations "
        f"({round(100*len(df_export)/max(df_work[sel_pollutant].notna().sum(),1),2)}%).",
        "info",
    )

    d1, d2 = st.columns(2)

    with d1:
        st.markdown("**Flagged observations only**")
        st.dataframe(df_export.head(200), use_container_width=True, hide_index=True)
        csv_flagged = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download flagged CSV ({len(df_export):,} rows)",
            data=csv_flagged,
            file_name=f"{sel_station}_{sel_pollutant}_anomaly_flags.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d2:
        st.markdown("**All observations (with flag columns)**")
        st.dataframe(df_full_export.head(200), use_container_width=True, hide_index=True)
        csv_full = df_full_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download full annotated CSV ({len(df_full_export):,} rows)",
            data=csv_full,
            file_name=f"{sel_station}_{sel_pollutant}_all_annotated.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Method summary CSV
    st.markdown("**Method summary table**")
    rows = []
    for mname in sel_methods:
        if mname not in results:
            continue
        res = results[mname]
        rows.append({
            "method": mname,
            "category": METHOD_REGISTRY[mname]["category"],
            "n_flagged": res.n_flagged,
            "pct_flagged": res.pct_flagged,
            "runtime_sec": res.runtime_sec,
            "status": res.status,
            "warning": res.warning or "",
        })
    if rows:
        df_summary = pd.DataFrame(rows)
        csv_summary = df_summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download method summary CSV",
            data=csv_summary,
            file_name=f"{sel_station}_{sel_pollutant}_method_summary.csv",
            mime="text/csv",
        )


# ── Main entry point ──────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Anomaly Comparison | ISRAP AIRMAPS",
        layout="wide",
        page_icon="⚖️",
        initial_sidebar_state="expanded",
    )
    inject_css()

    st.markdown(
        f"<h1 style='color:{COLORS['navy']}; margin-bottom:2px;'>⚖️ Anomaly Method Comparison</h1>"
        f"<p style='color:{COLORS['muted']}; font-size:0.95rem; margin-top:0;'>"
        "Run up to 8 detectors simultaneously, compare agreement, build an ensemble, and export results."
        "</p>",
        unsafe_allow_html=True,
    )

    # Load data
    try:
        station_data = _load()
    except Exception as exc:
        st.error(f"Failed to load station data: {exc}")
        st.stop()

    # Render sidebar controls
    ctrl = _render_sidebar(station_data)
    sel_station = ctrl["sel_station"]
    sel_pollutant = ctrl["sel_pollutant"]
    date_start = ctrl["date_start"]
    date_end = ctrl["date_end"]
    sel_methods = ctrl["sel_methods"]
    params = ctrl["params"]
    run_btn = ctrl["run"]
    df_s = ctrl["df_s"]

    # ── Run on button click ───────────────────────────────────────────────────
    if run_btn:
        if not sel_methods:
            st.warning("Please select at least one method before running.")
            st.stop()

        # Filter by date
        mask = (df_s["datetime"].dt.date >= date_start) & (df_s["datetime"].dt.date <= date_end)
        df_filtered = df_s[mask].copy()

        if df_filtered.empty:
            st.error("No data in the selected date range.")
            st.stop()

        # Structural missing check
        valid_frac = df_filtered[sel_pollutant].notna().mean()
        if valid_frac < STRUCTURAL_MISSING_THRESHOLD:
            st.error(
                f"**{sel_pollutant}** at **{STATIONS[sel_station]['label']}** appears structurally "
                f"missing ({valid_frac*100:.1f}% valid). Cannot run anomaly detection."
            )
            st.stop()

        if valid_frac < 0.05:
            info_box(
                f"Warning: Only {valid_frac*100:.1f}% of {sel_pollutant} observations are valid. "
                "Some methods may skip or produce unreliable results.",
                "warn",
            )

        # Execute
        with st.spinner("Running detectors…"):
            df_work, results = _run_detectors(df_filtered, sel_methods, params, sel_pollutant)

        # Persist to session state
        st.session_state[_STATE_KEY] = {
            "df_work": df_work,
            "results": results,
            "sel_methods": sel_methods,
            "sel_station": sel_station,
            "sel_pollutant": sel_pollutant,
        }

    # ── Display results from session state ───────────────────────────────────
    state = st.session_state.get(_STATE_KEY)

    if state is None:
        st.markdown(
            f"""
            <div class="empty-state">
                <div class="empty-state-icon">⚖️</div>
                <div class="empty-state-title">No comparison run yet</div>
                <p>Select a station, pollutant, date range, and methods in the sidebar,
                then click <b>▶ Run Comparison</b>.</p>
                <p style="font-size:0.85rem; color:{COLORS['muted']};">
                Available methods: {" · ".join(_MNAMES)}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Warn if stale state (different station/pollutant)
    if state["sel_station"] != sel_station or state["sel_pollutant"] != sel_pollutant:
        info_box(
            "Results below are from a previous run on a different station/pollutant. "
            "Click <b>▶ Run Comparison</b> to refresh.",
            "warn",
        )

    df_work = state["df_work"]
    results = state["results"]
    run_methods = state["sel_methods"]
    run_station = state["sel_station"]
    run_pollutant = state["sel_pollutant"]

    # Context banner
    st.markdown(
        f"<div class='info-box' style='margin-bottom:12px;'>"
        f"<b>Results for:</b>&nbsp; {STATIONS[run_station]['label']} &nbsp;|&nbsp; "
        f"<b>Pollutant:</b> {run_pollutant} ({POLLUTANTS.get(run_pollutant,{}).get('unit','')}) &nbsp;|&nbsp; "
        f"<b>Methods:</b> {len(run_methods)} &nbsp;|&nbsp; "
        f"<b>Rows:</b> {len(df_work):,} &nbsp;|&nbsp; "
        f"<b>Date span:</b> {df_work['datetime'].min().date()} → {df_work['datetime'].max().date()}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Render 6 tabs
    tab_labels = [
        "📋 Method Summary",
        "🔗 Overlap Analysis",
        "📈 Timeline",
        "🎯 Ensemble",
        "🃏 Scorecards",
        "⬇ Download",
    ]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        _tab_summary(df_work, results, run_methods, run_pollutant)

    with tabs[1]:
        _tab_overlap(df_work, run_methods)

    with tabs[2]:
        _tab_timeline(df_work, run_methods, run_pollutant)

    with tabs[3]:
        _tab_ensemble(df_work, run_methods, run_pollutant)

    with tabs[4]:
        _tab_scorecards(df_work, results, run_methods)

    with tabs[5]:
        _tab_download(df_work, results, run_methods, run_station, run_pollutant)


if __name__ == "__main__":
    main()

