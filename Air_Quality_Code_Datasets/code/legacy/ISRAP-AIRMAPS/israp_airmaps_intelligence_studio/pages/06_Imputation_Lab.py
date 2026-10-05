"""Imputation Lab — interactive imputation for any station/pollutant/gap."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import POLLUTANTS, STATIONS, IMPUTATION_DEFAULTS
from src.data_loader import load_all_stations, load_era5, merge_era5
from src.gap_analysis import find_gaps
from src.imputation.linear_imputer import LinearImputer
from src.imputation.era5_regression import ERA5RegressionImputer
from src.imputation.mice_imputer import MICEImputer
from src.imputation.brits_imputer import BRITSImputer, _TORCH_AVAILABLE as _TORCH
from src.imputation.saits_imputer import SAITSImputer, _PYPOTS_AVAILABLE, _TORCH_AVAILABLE as _TORCH2
from src.imputation.csdi_imputer import CSDIImputer, _TORCH_AVAILABLE as _TORCH3
from src.imputation.imputeformer_imputer import ImputeFormerImputer, _TORCH_AVAILABLE as _TORCH4
from services.interpretation_engine import recommend_imputation_method, interpret_imputation
from ui.theme import COLORS, inject_css, info_box, OBSERVED_COLOR, IMPUTED_COLOR


@st.cache_data(show_spinner="Loading data…")
def _load():
    stations = load_all_stations()
    era5 = load_era5()
    return {k: merge_era5(v, era5) for k, v in stations.items()}


IMPUTERS = {
    "Linear Interpolation": LinearImputer,
    "ERA5 Ridge Regression": lambda: ERA5RegressionImputer(),
    "ERA5 Random Forest": lambda: ERA5RegressionImputer(),
    "ERA5 HistGradientBoosting": lambda: ERA5RegressionImputer(),
    "MICE": MICEImputer,
    "BRITS": BRITSImputer,
    "SAITS": SAITSImputer,
    "CSDI": CSDIImputer,
    "ImputeFormer": ImputeFormerImputer,
}

IMPUTER_PARAMS = {
    "Linear Interpolation": {"max_gap": 2},
    "ERA5 Ridge Regression": {"model_name": "ridge"},
    "ERA5 Random Forest": {"model_name": "random_forest"},
    "ERA5 HistGradientBoosting": {"model_name": "hist_gradient_boosting"},
    "MICE": {"max_iter": 10},
    "BRITS": {"epochs": 20, "seq_len": 48},
    "SAITS": {"epochs": 20, "seq_len": 48},
    "CSDI": {"epochs": 15, "seq_len": 48, "T_diff": 50},
    "ImputeFormer": {"epochs": 15, "seq_len": 48, "d_model": 32, "rank": 8},
}

IMPUTER_AVAILABILITY = {
    "Linear Interpolation": "Available",
    "ERA5 Ridge Regression": "Available",
    "ERA5 Random Forest": "Available",
    "ERA5 HistGradientBoosting": "Available",
    "MICE": "Available",
    "BRITS": "BRITS-LSTM (PyTorch)" if _TORCH else "BRITS-approx (Bidir. ETS)",
    "SAITS": ("SAITS (PyPOTS)" if _PYPOTS_AVAILABLE else
              "SAITS-lite (PyTorch)" if _TORCH2 else "SAITS-approx (Bidir. Weighted)"),
    "CSDI": "CSDI-lite (PyTorch DDPM)" if _TORCH3 else "CSDI-approx (SavGol)",
    "ImputeFormer": "ImputeFormer-lite (PyTorch low-rank attn)" if _TORCH4 else "ImputeFormer-approx (cubic spline)",
}

try:
    from src.imputation.mice_imputer import MICEImputer as _MICE
    IMPUTERS["MICE (multiple)"] = _MICE
    IMPUTER_PARAMS["MICE (multiple)"] = {"max_iter": 10, "n_imputations": 5}
    IMPUTER_AVAILABILITY["MICE (multiple)"] = "Available"
except Exception:
    pass


def main():
    st.set_page_config(page_title="Imputation Lab | ISRAP AIRMAPS", layout="wide", page_icon="🔧")
    inject_css()

    st.markdown(f"<h1 style='color:{COLORS['navy']};'>🔧 Imputation Lab</h1>", unsafe_allow_html=True)

    station_data = _load()

    left, center = st.columns([1, 3])

    with left:
        sel_station = st.selectbox("Station", list(station_data.keys()),
                                   format_func=lambda k: STATIONS[k]["label"])
        meta = STATIONS[sel_station]
        sel_pollutant = st.selectbox("Pollutant", meta["monitored_pollutants"])
        df_s = station_data[sel_station].copy()

        dt_min, dt_max = df_s["datetime"].min().date(), df_s["datetime"].max().date()
        date_range = st.date_input("Date Range", value=(dt_min, dt_max),
                                   min_value=dt_min, max_value=dt_max)
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            d_from, d_to = date_range
        else:
            d_from, d_to = dt_min, dt_max

        sel_method = st.selectbox("Imputation Method", list(IMPUTERS.keys()))

        st.markdown("---")
        st.markdown("**Method Parameters**")
        d = IMPUTATION_DEFAULTS
        extra_params = {}
        if sel_method == "Linear Interpolation":
            extra_params["max_gap"] = st.slider("Max gap (hours)", 1, 12, 2)
        elif sel_method.startswith("ERA5"):
            extra_params["n_splits"] = st.slider("Cross-validation folds", 2, 5, 3)
        elif sel_method.startswith("MICE"):
            extra_params["max_iter"] = st.slider("MICE iterations", 3, 20, 10)
            if "multiple" in sel_method:
                extra_params["n_imputations"] = st.slider("Number of imputations", 3, 10, 5)
        elif sel_method == "BRITS":
            extra_params["epochs"] = st.slider("Training epochs", 5, 50, 20)
            extra_params["seq_len"] = st.selectbox("Sequence length", [24, 48, 72, 168], index=1)
            avail = IMPUTER_AVAILABILITY.get("BRITS", "")
            st.caption(f"Mode: {avail}")
        elif sel_method == "SAITS":
            extra_params["epochs"] = st.slider("Training epochs", 5, 50, 20)
            extra_params["seq_len"] = st.selectbox("Sequence length", [24, 48, 72], index=1)
            extra_params["d_model"] = st.selectbox("Model dimension", [16, 32, 64], index=1)
            avail = IMPUTER_AVAILABILITY.get("SAITS", "")
            st.caption(f"Mode: {avail}")
        elif sel_method == "CSDI":
            extra_params["epochs"] = st.slider("Training epochs", 5, 30, 15)
            extra_params["seq_len"] = st.selectbox("Sequence length", [24, 48, 96], index=1)
            extra_params["T_diff"] = st.selectbox("Diffusion steps (T)", [20, 50, 100], index=1)
            extra_params["n_samples"] = st.slider("MC samples (inference)", 1, 10, 5)
            avail = IMPUTER_AVAILABILITY.get("CSDI", "")
            st.caption(f"Mode: {avail}")
        elif sel_method == "ImputeFormer":
            extra_params["epochs"] = st.slider("Training epochs", 5, 30, 15)
            extra_params["seq_len"] = st.selectbox("Sequence length", [24, 48, 96], index=1)
            extra_params["d_model"] = st.selectbox("Model dimension", [16, 32, 64], index=1)
            extra_params["rank"] = st.selectbox("Attention rank (r)", [4, 8, 16], index=1)
            avail = IMPUTER_AVAILABILITY.get("ImputeFormer", "")
            st.caption(f"Mode: {avail}")

        run_btn = st.button("▶ Run Imputation", type="primary", use_container_width=True)

    with center:
        unit = POLLUTANTS.get(sel_pollutant, {}).get("unit", "")
        st.caption(f"Home / Imputation Lab / {meta['label']} / {sel_pollutant} / {sel_method}")

        mask = (df_s["datetime"].dt.date >= d_from) & (df_s["datetime"].dt.date <= d_to)
        df_filt = df_s[mask].copy().reset_index(drop=True)

        if df_filt[sel_pollutant].notna().mean() < 0.01:
            info_box(
                f"<b>{sel_pollutant} is structurally missing at {meta['label']}.</b> "
                "This pollutant is not monitored here and must not be imputed.",
                kind="danger",
            )
            return

        # Gap analysis
        gaps = find_gaps(df_filt, sel_pollutant)
        n_missing = int(df_filt[sel_pollutant].isna().sum())
        n_gaps = len(gaps)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Missing values", f"{n_missing:,}")
        c2.metric("Missing %", f"{100*df_filt[sel_pollutant].isna().mean():.1f}%")
        c3.metric("Gaps", str(n_gaps))
        c4.metric("Longest gap", f"{int(gaps['gap_length'].max()):,}h" if n_gaps > 0 else "0h")

        if not gaps.empty:
            # Recommendation
            longest = int(gaps["gap_length"].max())
            rec = recommend_imputation_method(
                longest,
                has_era5=any(c in df_filt.columns for c in ["temp_c", "wind_speed"]),
                has_multi_station=True,
                is_structural=False,
                needs_uncertainty=False,
            )
            info_box(
                f"<b>Recommended method for longest gap ({longest}h):</b> "
                f"<code>{rec['method']}</code> — {rec['reason']} "
                f"(Confidence: {rec['confidence']})",
                kind="info",
            )

        if "imp_result" not in st.session_state:
            st.session_state["imp_result"] = None
            st.session_state["imp_df"] = None

        if run_btn:
            imp_params = {**IMPUTER_PARAMS.get(sel_method, {}), **extra_params}
            ImpClass = IMPUTERS[sel_method]
            imputer = ImpClass() if callable(ImpClass) else ImpClass(**imp_params)

            with st.spinner(f"Running {sel_method}…"):
                df_imp, res = imputer.impute(df_filt.copy(), sel_pollutant, **imp_params)

            st.session_state["imp_df"] = df_imp
            st.session_state["imp_result"] = res

        imp_df = st.session_state.get("imp_df")
        imp_res = st.session_state.get("imp_result")

        if imp_df is None:
            st.info("Select a method and click **▶ Run Imputation**.")
        else:
            # Status metrics
            imp_col = [c for c in imp_df.columns if c.startswith("imputed_")][0] if any(c.startswith("imputed_") for c in imp_df.columns) else None

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Status", imp_res.status.upper())
            mc2.metric("Imputed", f"{imp_res.n_imputed:,}")
            mc3.metric("Runtime", f"{imp_res.runtime_sec:.2f}s")

            if imp_res.warning:
                st.warning(imp_res.warning)

            if imp_col:
                # Chart: observed + imputed
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=imp_df["datetime"], y=imp_df[sel_pollutant],
                    mode="lines", name="Observed",
                    line=dict(color=OBSERVED_COLOR, width=1.5),
                ))
                imp_mask = imp_df[sel_pollutant].isna() & imp_df[imp_col].notna()
                if imp_mask.any():
                    fig.add_trace(go.Scatter(
                        x=imp_df.loc[imp_mask, "datetime"],
                        y=imp_df.loc[imp_mask, imp_col],
                        mode="markers", name=f"Imputed ({sel_method})",
                        marker=dict(color=IMPUTED_COLOR, size=6, symbol="circle"),
                    ))

                # MICE uncertainty interval
                if "mice_std" in imp_df.columns and sel_method.startswith("MICE"):
                    imp_df["upper"] = imp_df[imp_col] + 1.96 * imp_df["mice_std"]
                    imp_df["lower"] = (imp_df[imp_col] - 1.96 * imp_df["mice_std"]).clip(lower=0)
                    fig.add_trace(go.Scatter(
                        x=pd.concat([imp_df.loc[imp_mask, "datetime"],
                                     imp_df.loc[imp_mask, "datetime"].iloc[::-1]]),
                        y=pd.concat([imp_df.loc[imp_mask, "upper"],
                                     imp_df.loc[imp_mask, "lower"].iloc[::-1]]),
                        fill="toself", fillcolor="rgba(45,198,83,0.15)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="95% CI",
                    ))

                fig.update_layout(
                    template="plotly_white", height=420, hovermode="x unified",
                    legend=dict(orientation="h", y=-0.10),
                    margin=dict(l=60, r=20, t=30, b=70),
                    yaxis_title=f"{sel_pollutant} ({unit})",
                    xaxis=dict(rangeslider=dict(visible=True, thickness=0.05)),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Interpretation insights
                avg_gap = int(gaps["gap_length"].mean()) if not gaps.empty else 0
                insights = interpret_imputation(
                    imp_res.n_imputed, sel_method,
                    "very_short" if avg_gap <= 2 else "medium" if avg_gap <= 24 else "long",
                    None, sel_pollutant,
                )
                for ins in insights:
                    info_box(ins, kind="info")

                # Feature importance
                if imp_res.feature_importance:
                    top_fi = sorted(imp_res.feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                    fi_df = pd.DataFrame(top_fi, columns=["Feature", "Importance"])
                    fig_fi = go.Figure(go.Bar(
                        x=fi_df["Importance"].abs(),
                        y=fi_df["Feature"],
                        orientation="h",
                        marker_color=COLORS["teal"],
                    ))
                    fig_fi.update_layout(title="Top Feature Importances",
                                         template="plotly_white", height=280,
                                         margin=dict(l=150, r=20, t=40, b=50))
                    st.plotly_chart(fig_fi, use_container_width=True)

                # Data table
                with st.expander("Show imputed data table"):
                    show_cols = ["datetime", sel_pollutant, imp_col]
                    if "mice_std" in imp_df.columns:
                        show_cols.append("mice_std")
                    st.dataframe(imp_df[show_cols].head(500), use_container_width=True)

                # Download
                csv = imp_df[["datetime", sel_pollutant, imp_col]].to_csv(index=False).encode()
                st.download_button(
                    f"📥 Download imputed data ({sel_method})", csv,
                    f"imputed_{sel_station}_{sel_pollutant}_{sel_method.replace(' ', '_')}.csv",
                )


if __name__ == "__main__":
    main()

