"""Anomaly Lab — run any anomaly detection method with interactive controls."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.config import POLLUTANTS, STATIONS, ANOMALY_DEFAULTS
from src.data_loader import load_all_stations, load_era5, merge_era5
from src.anomaly_detection.physical_constraints import PhysicalConstraintDetector
from src.anomaly_detection.iqr_detector import IQRDetector
from src.anomaly_detection.hampel_detector import HampelDetector
from src.anomaly_detection.rolling_zscore import RollingZScoreDetector
from src.anomaly_detection.lof_detector import LOFDetector
from src.anomaly_detection.isolation_forest import IsolationForestDetector
from src.anomaly_detection.stl_detector import STLDetector
from src.anomaly_detection.screen_detector import SCREENDetector
from services.experiment_store import save_experiment
from services.review_store import save_review, REVIEW_STATUS
from services.interpretation_engine import interpret_anomaly
from ui.theme import COLORS, inject_css, info_box, badge


@st.cache_data(show_spinner="Loading data…")
def _load():
    stations = load_all_stations()
    era5 = load_era5()
    return {k: merge_era5(v, era5) for k, v in stations.items()}


METHOD_META = {
    "Physical Constraints": {
        "class": PhysicalConstraintDetector,
        "category": "Physical",
        "desc": "Flags values below zero or above known physical maxima for each pollutant.",
        "best_for": "First-pass cleaning of sensor hardware errors.",
        "limitation": "Cannot detect elevated but technically valid readings.",
        "cost": "Negligible",
        "badge": "Statistical",
    },
    "IQR Fence": {
        "class": IQRDetector,
        "category": "Statistical",
        "desc": "Flags values outside Q1−k×IQR and Q3+k×IQR for configurable multiplier k.",
        "best_for": "Isolated point spikes in a stable distribution.",
        "limitation": "Sensitive to skewed distributions; does not account for seasonality.",
        "cost": "Very Low",
        "badge": "Statistical",
    },
    "Hampel Filter": {
        "class": HampelDetector,
        "category": "Robust Temporal",
        "desc": "Rolling median ± k×MAD window filter. Robust to up to 50% contamination.",
        "best_for": "Isolated spikes in a slowly varying signal.",
        "limitation": "Can miss slow drifts; window size matters.",
        "cost": "Low",
        "badge": "Robust",
    },
    "Rolling Z-Score": {
        "class": RollingZScoreDetector,
        "category": "Statistical Temporal",
        "desc": "Standardises each observation using a rolling mean and rolling std.",
        "best_for": "Detecting local level shifts and spikes.",
        "limitation": "Assumes local normality; sensitive to window size.",
        "cost": "Low",
        "badge": "Temporal",
    },
    "Local Outlier Factor": {
        "class": LOFDetector,
        "category": "Machine Learning",
        "desc": "Compares local density of each point to its k-nearest neighbours.",
        "best_for": "Contextual anomalies with ERA5 features.",
        "limitation": "Computationally slow on long series; no temporal ordering.",
        "cost": "Medium",
        "badge": "Machine Learning",
    },
    "Isolation Forest": {
        "class": IsolationForestDetector,
        "category": "Machine Learning",
        "desc": "Isolates anomalies using random feature splits. Works well multivariate.",
        "best_for": "Multivariate detection with ERA5 and time features.",
        "limitation": "Contamination parameter must be set; no temporal ordering.",
        "cost": "Medium",
        "badge": "Machine Learning",
    },
    "STL Residual": {
        "class": STLDetector,
        "category": "Seasonal Temporal",
        "desc": "Decomposes into trend + seasonal + residual. Flags anomalous residuals.",
        "best_for": "Detecting anomalies that deviate from seasonal patterns.",
        "limitation": "Requires long series (≥ 2 × seasonal period) and no excessive gaps.",
        "cost": "Medium",
        "badge": "Seasonal",
    },
    "SCREEN Rate-of-Change": {
        "class": SCREENDetector,
        "category": "Rate-of-Change",
        "desc": "Flags unrealistic hour-to-hour changes using data-derived speed limits.",
        "best_for": "Detecting step changes, dropouts, and frozen sensors.",
        "limitation": "Approximation only; real EPA SCREEN algorithm is more complex.",
        "cost": "Low",
        "badge": "Temporal",
    },
}

try:
    from src.anomaly_detection.lstm_ndt import LSTMNDTDetector
    METHOD_META["LSTM NDT"] = {
        "class": LSTMNDTDetector,
        "category": "Deep Learning",
        "desc": "LSTM prediction error + nonparametric dynamic threshold.",
        "best_for": "Long temporal sequences with complex dynamics.",
        "limitation": "Requires PyTorch; slow on CPU; needs retraining when data changes.",
        "cost": "High",
        "badge": "Deep Learning",
    }
except Exception:
    pass

try:
    from src.anomaly_detection.usad import USADDetector
    METHOD_META["USAD"] = {
        "class": USADDetector,
        "category": "Deep Learning",
        "desc": "Two-decoder autoencoder anomaly detection (Audibert et al. KDD 2020).",
        "best_for": "Multivariate deep anomaly detection.",
        "limitation": "Requires PyTorch; long training; sensitive to hyperparameters.",
        "cost": "High",
        "badge": "Deep Learning",
    }
except Exception:
    pass

try:
    from src.anomaly_detection.anomaly_transformer import AnomalyTransformerDetector
    METHOD_META["Anomaly Transformer"] = {
        "class": AnomalyTransformerDetector,
        "category": "Deep Learning",
        "desc": (
            "Detects anomalies via association discrepancy: KL divergence between "
            "learned series-association (attention) and a Gaussian prior-association "
            "(Wu et al. ICLR 2022). High discrepancy indicates the point cannot attract "
            "temporal context consistent with normal patterns."
        ),
        "best_for": "Complex temporal anomalies where context-window disruption is the signal.",
        "limitation": "Requires PyTorch; O(T²) attention (bounded by seq_len); may miss static level shifts.",
        "cost": "High",
        "badge": "Deep Learning",
    }
except Exception:
    pass

try:
    from src.anomaly_detection.dcdetector import DCdetectorDetector
    METHOD_META["DCdetector"] = {
        "class": DCdetectorDetector,
        "category": "Deep Learning",
        "desc": (
            "Dual-channel attention model with patch-level (inter-patch) and point-level "
            "(intra-patch) branches. Anomaly score = distance between the two representations "
            "(Chen et al. KDD 2023). Captures both local and global structural anomalies."
        ),
        "best_for": "Multi-scale anomalies: both isolated spikes and sustained level shifts.",
        "limitation": "Requires PyTorch; seq_len must be a multiple of patch_size.",
        "cost": "High",
        "badge": "Deep Learning",
    }
except Exception:
    pass

try:
    from src.anomaly_detection.omni_anomaly import OmniAnomalyDetector
    METHOD_META["OmniAnomaly"] = {
        "class": OmniAnomalyDetector,
        "category": "Deep Learning",
        "desc": (
            "Stochastic GRU VAE: learns the generative distribution of normal time series "
            "and flags points with high reconstruction error under Monte-Carlo sampling "
            "(Su et al. KDD 2019). Captures complex temporal dependencies and uncertainty."
        ),
        "best_for": "Subtle distributional anomalies; produces uncertainty-aware scores.",
        "limitation": "Requires PyTorch; MC sampling makes it slower; VAE can over-smooth.",
        "cost": "High",
        "badge": "Deep Learning",
    }
except Exception:
    pass


def _param_controls(method_name: str) -> dict:
    params = {}
    d = ANOMALY_DEFAULTS
    if method_name == "IQR Fence":
        params["multiplier"] = st.slider("IQR Multiplier", 1.0, 5.0, float(d["iqr_multiplier"]), 0.5)
    elif method_name == "Hampel Filter":
        params["window"] = st.slider("Window (hours)", 6, 168, int(d["hampel_window"]), 6)
        params["threshold"] = st.slider("MAD Threshold", 1.0, 6.0, float(d["hampel_threshold"]), 0.5)
        params["past_only"] = st.checkbox("Past-only (no leakage)", value=True)
    elif method_name == "Rolling Z-Score":
        params["window"] = st.selectbox("Window (hours)", [24, 48, 72, 168], index=1)
        params["threshold"] = st.slider("Z Threshold", 1.5, 5.0, float(d["rolling_zscore_threshold"]), 0.25)
    elif method_name == "Local Outlier Factor":
        params["n_neighbors"] = st.slider("k Neighbours", 5, 50, int(d["lof_n_neighbors"]), 5)
        params["contamination"] = st.slider("Contamination", 0.001, 0.1, float(d["lof_contamination"]), 0.005,
                                            format="%.3f")
        params["include_era5"] = st.checkbox("Include ERA5 features", value=True)
    elif method_name == "Isolation Forest":
        params["n_estimators"] = st.slider("Trees", 50, 500, int(d["iforest_n_estimators"]), 50)
        params["contamination"] = st.slider("Contamination", 0.001, 0.1, float(d["iforest_contamination"]), 0.005,
                                            format="%.3f")
        params["include_era5"] = st.checkbox("Include ERA5 features", value=True)
        params["include_lags"] = st.checkbox("Include lag features", value=True)
    elif method_name == "STL Residual":
        params["seasonal"] = st.slider("Seasonal Period", 13, 49, int(d["stl_seasonal"]), 2)
        params["threshold"] = st.slider("Residual Threshold", 1.5, 5.0, float(d["stl_threshold"]), 0.25)
        params["threshold_method"] = st.selectbox("Threshold Method", ["iqr", "mad", "zscore"])
    elif method_name == "SCREEN Rate-of-Change":
        params["percentile"] = st.slider("Speed Limit Percentile", 90.0, 99.9, float(d["screen_percentile"]), 0.5)
    elif method_name in ("LSTM NDT", "USAD"):
        params["fast_mode"] = st.checkbox("Fast mode (fewer epochs)", value=True)
        params["seq_len"] = st.slider("Sequence Length", 6, 72, 24, 6)
        params["epochs"] = st.slider("Epochs (full mode)", 5, 100, 20, 5)
        params["threshold_percentile"] = st.slider("Threshold Percentile", 95.0, 99.9, 99.0, 0.5)
    elif method_name == "Anomaly Transformer":
        params["fast_mode"] = st.checkbox("Fast mode (fewer epochs)", value=True)
        params["seq_len"] = st.selectbox("Sequence Length", [32, 64, 128], index=1)
        params["d_model"] = st.selectbox("Model Dimension", [16, 32, 64], index=1)
        params["epochs"] = st.slider("Epochs (full mode)", 5, 50, 10, 5)
        params["sigma"] = st.slider("Prior Sigma (Gaussian kernel)", 0.5, 10.0, 3.0, 0.5)
        params["threshold_percentile"] = st.slider("Threshold Percentile", 95.0, 99.9, 99.0, 0.5)
    elif method_name == "DCdetector":
        params["fast_mode"] = st.checkbox("Fast mode (fewer epochs)", value=True)
        params["patch_size"] = st.selectbox("Patch Size", [4, 8, 16], index=1)
        params["d_model"] = st.selectbox("Model Dimension", [16, 32, 64], index=1)
        params["epochs"] = st.slider("Epochs (full mode)", 5, 50, 10, 5)
        params["threshold_percentile"] = st.slider("Threshold Percentile", 95.0, 99.9, 99.0, 0.5)
    elif method_name == "OmniAnomaly":
        params["fast_mode"] = st.checkbox("Fast mode (fewer epochs)", value=True)
        params["seq_len"] = st.selectbox("Sequence Length", [24, 48, 96], index=1)
        params["hidden_dim"] = st.selectbox("Hidden Dim", [16, 32, 64], index=1)
        params["latent_dim"] = st.selectbox("Latent Dim", [4, 8, 16], index=1)
        params["epochs"] = st.slider("Epochs (full mode)", 5, 50, 10, 5)
        params["threshold_percentile"] = st.slider("Threshold Percentile", 95.0, 99.9, 99.0, 0.5)
    return params


def main():
    st.set_page_config(page_title="Anomaly Lab | ISRAP AIRMAPS", layout="wide", page_icon="⚗️")
    inject_css()

    station_data = _load()

    # ── Layout: left controls | center chart | right explanation ──────────────
    left, center, right = st.columns([1, 3, 1.2])

    with left:
        st.markdown(f"<h3 style='color:{COLORS['navy']};'>⚗️ Anomaly Lab</h3>", unsafe_allow_html=True)
        sel_station = st.selectbox("Station",
                                   list(station_data.keys()),
                                   format_func=lambda k: STATIONS[k]["label"])
        meta = STATIONS[sel_station]
        sel_pollutant = st.selectbox("Pollutant", meta["monitored_pollutants"])

        df_s = station_data[sel_station]
        dt_min, dt_max = df_s["datetime"].min().date(), df_s["datetime"].max().date()
        date_range = st.date_input("Date Range", value=(dt_min, dt_max),
                                   min_value=dt_min, max_value=dt_max)
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            d_from, d_to = date_range
        else:
            d_from, d_to = dt_min, dt_max

        sel_method = st.selectbox("Anomaly Method", list(METHOD_META.keys()))

        st.markdown("---")
        st.markdown("**Parameters**")
        with st.container():
            params = _param_controls(sel_method)

        run_btn = st.button("▶ Run Detection", type="primary", use_container_width=True)

    # ── Filter ─────────────────────────────────────────────────────────────────
    mask = (df_s["datetime"].dt.date >= d_from) & (df_s["datetime"].dt.date <= d_to)
    df_filt = df_s[mask].copy().reset_index(drop=True)

    with center:
        st.caption(f"Home / Anomaly Lab / {meta['label']} / {sel_pollutant} / {sel_method}")
        unit = POLLUTANTS.get(sel_pollutant, {}).get("unit", "")

        if "anomaly_result" not in st.session_state:
            st.session_state["anomaly_result"] = None
            st.session_state["anomaly_df"] = None

        if run_btn:
            # Check structural missing
            if df_filt[sel_pollutant].notna().mean() < 0.01:
                st.error(f"**{sel_pollutant} is structurally missing at {meta['label']}**. "
                         "This pollutant is not monitored here and cannot be analysed.")
            else:
                method_info = METHOD_META[sel_method]
                detector = method_info["class"](**params)

                with st.spinner(f"Running {sel_method}…"):
                    t0 = time.perf_counter()
                    df_out, result = detector.detect(df_filt.copy(), sel_pollutant, **params)
                    elapsed = time.perf_counter() - t0

                st.session_state["anomaly_df"] = df_out
                st.session_state["anomaly_result"] = result

                # Save to experiment history
                save_experiment(
                    station=sel_station,
                    pollutant=sel_pollutant,
                    method=sel_method,
                    category=method_info["category"],
                    params=params,
                    metrics={
                        "n_flagged": result.n_flagged,
                        "pct_flagged": result.pct_flagged,
                        "runtime_sec": result.runtime_sec,
                    },
                    status=result.status,
                    runtime_sec=result.runtime_sec,
                )

        # ── Results ────────────────────────────────────────────────────────────
        result = st.session_state.get("anomaly_result")
        df_out = st.session_state.get("anomaly_df")

        if result is None:
            st.info("Select a station, pollutant, and method, then click **▶ Run Detection**.")
        else:
            # Status + metrics
            status_color = {"ok": "green", "skipped": "warn", "error": "error"}.get(result.status, "info")
            cols_m = st.columns(5)
            cols_m[0].metric("Status", result.status.upper())
            cols_m[1].metric("Flagged", f"{result.n_flagged:,}")
            cols_m[2].metric("Flagged %", f"{result.pct_flagged:.2f}%")
            cols_m[3].metric("Runtime", f"{result.runtime_sec:.2f}s")
            if result.warning:
                cols_m[4].warning(result.warning)

            if result.status == "error":
                st.error(f"Method error: {result.warning}")
            elif result.status == "skipped":
                st.warning(f"Method skipped: {result.warning}")
            else:
                flag_col = result.flag_col
                score_col = result.score_col

                # Main chart
                n_subplot = 2 if score_col and score_col in df_out.columns else 1
                fig = make_subplots(rows=n_subplot, cols=1, shared_xaxes=True,
                                    row_heights=[0.65, 0.35] if n_subplot == 2 else [1.0],
                                    vertical_spacing=0.08)

                fig.add_trace(go.Scatter(
                    x=df_out["datetime"], y=df_out[sel_pollutant],
                    mode="lines", name="Observed",
                    line=dict(color=COLORS["blue"], width=1.3),
                ), row=1, col=1)

                if flag_col in df_out.columns:
                    anom = df_out[df_out[flag_col].astype(bool)]
                    if not anom.empty:
                        fig.add_trace(go.Scatter(
                            x=anom["datetime"], y=anom[sel_pollutant],
                            mode="markers", name=f"Anomaly ({sel_method})",
                            marker=dict(color=COLORS["red"], size=7, symbol="x"),
                        ), row=1, col=1)

                if n_subplot == 2 and score_col in df_out.columns:
                    fig.add_trace(go.Scatter(
                        x=df_out["datetime"], y=df_out[score_col],
                        mode="lines", name="Anomaly Score",
                        line=dict(color=COLORS["violet"], width=1),
                        fill="tozeroy", fillcolor="rgba(123,45,139,0.10)",
                    ), row=2, col=1)
                    # Threshold line
                    if result.threshold is not None and isinstance(result.threshold, (int, float)):
                        fig.add_hline(y=result.threshold, line_dash="dash",
                                      line_color=COLORS["red"], row=2, col=1,
                                      annotation_text=f"Threshold={result.threshold}")

                fig.update_layout(
                    template="plotly_white", height=430, hovermode="x unified",
                    legend=dict(orientation="h", y=-0.08),
                    margin=dict(l=60, r=20, t=30, b=60),
                )
                fig.update_yaxes(title_text=f"{sel_pollutant} ({unit})", row=1, col=1)
                if n_subplot == 2:
                    fig.update_yaxes(title_text="Score", row=2, col=1)
                st.plotly_chart(fig, use_container_width=True)

                # Insights
                insights = interpret_anomaly(
                    df_out, sel_method, result.pct_flagged, sel_station, sel_pollutant
                )
                for ins in insights:
                    info_box(ins, kind="info")

                # Bottom tabs
                btab1, btab2, btab3, btab4 = st.tabs(
                    ["Flagged Rows", "Score Distribution", "ERA5 Context", "Review Actions"]
                )

                with btab1:
                    if flag_col in df_out.columns:
                        anom_rows = df_out[df_out[flag_col].astype(bool)][
                            ["datetime", sel_pollutant, score_col]
                        ].copy() if score_col in df_out.columns else df_out[df_out[flag_col].astype(bool)][["datetime", sel_pollutant]]
                        st.write(f"**{len(anom_rows)} flagged observations** (first 200 shown)")
                        st.dataframe(anom_rows.head(200), use_container_width=True)

                with btab2:
                    if score_col in df_out.columns:
                        fig_dist = go.Figure()
                        fig_dist.add_trace(go.Histogram(
                            x=df_out[score_col].dropna(), nbinsx=60,
                            name="Anomaly Score",
                            marker_color=COLORS["violet"],
                            opacity=0.7,
                        ))
                        if result.threshold is not None and isinstance(result.threshold, (int, float)):
                            fig_dist.add_vline(x=result.threshold, line_dash="dash",
                                               line_color=COLORS["red"],
                                               annotation_text=f"Threshold={result.threshold}")
                        fig_dist.update_layout(
                            title="Anomaly Score Distribution",
                            template="plotly_white", height=300,
                            margin=dict(l=60, r=20, t=40, b=50),
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)

                with btab3:
                    era5_avail = [c for c in ["temp_c", "wind_speed", "blh", "relative_humidity"] if c in df_out.columns]
                    if era5_avail and flag_col in df_out.columns:
                        anom_mask = df_out[flag_col].astype(bool)
                        era5_compare = pd.DataFrame({
                            var: [
                                df_out.loc[~anom_mask, var].mean(),
                                df_out.loc[anom_mask, var].mean(),
                            ]
                            for var in era5_avail
                        }, index=["Normal", "Anomaly"])
                        st.write("**ERA5 conditions: Normal vs Anomaly**")
                        st.dataframe(era5_compare.round(3), use_container_width=True)
                    else:
                        st.info("No ERA5 variables available for context comparison.")

                with btab4:
                    st.markdown("**Review selected anomaly**")
                    if flag_col in df_out.columns and df_out[flag_col].any():
                        anom_idx = df_out[df_out[flag_col].astype(bool)].index
                        sel_idx = st.selectbox(
                            "Select anomaly to review",
                            anom_idx[:50].tolist(),
                            format_func=lambda i: str(df_out.loc[i, "datetime"]),
                        )
                        row = df_out.loc[sel_idx]
                        st.write(f"**Timestamp:** {row['datetime']}")
                        st.write(f"**Value:** {row[sel_pollutant]:.4f} {unit}")
                        if score_col in df_out.columns:
                            st.write(f"**Score:** {row[score_col]:.4f}")
                        review_status = st.selectbox("Review Decision", REVIEW_STATUS)
                        note = st.text_input("Reviewer Note (optional)")
                        if st.button("Save Review"):
                            save_review(
                                station=sel_station,
                                pollutant=sel_pollutant,
                                dt=str(row["datetime"]),
                                original_value=float(row[sel_pollutant]),
                                methods_flagged=[sel_method],
                                anomaly_score=float(row[score_col]) if score_col in df_out.columns else 0.0,
                                review_status=review_status,
                                reviewer_note=note,
                            )
                            st.success("Review saved.")
                    else:
                        st.info("No flagged anomalies to review.")

    # ── Right: method explanation ─────────────────────────────────────────────
    with right:
        minfo = METHOD_META[sel_method]
        st.markdown(f"### {sel_method}")
        st.markdown(f"**{minfo['badge']}**")
        st.caption(minfo["category"])
        st.markdown("**What it does:**")
        st.write(minfo["desc"])
        st.markdown("**Best for:**")
        st.write(minfo["best_for"])
        st.markdown("**Limitation:**")
        st.write(minfo["limitation"])
        st.markdown(f"**Computational cost:** {minfo['cost']}")

        if result and result.status == "ok":
            st.markdown("---")
            st.markdown("**Current Result:**")
            st.markdown(f"- Flagged: **{result.n_flagged}** ({result.pct_flagged:.2f}%)")
            st.markdown(f"- Runtime: **{result.runtime_sec:.2f}s**")
            if result.threshold is not None:
                st.markdown(f"- Threshold: **{result.threshold}**")


if __name__ == "__main__":
    main()

