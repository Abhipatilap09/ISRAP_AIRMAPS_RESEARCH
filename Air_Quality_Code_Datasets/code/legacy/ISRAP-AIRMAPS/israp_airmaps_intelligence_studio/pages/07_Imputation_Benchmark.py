"""Imputation Benchmark — masking experiments, gap-length heatmap, and missingness scenarios."""
from __future__ import annotations

import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import POLLUTANTS, STATIONS, RANDOM_SEED
from src.data_loader import load_all_stations, load_era5, merge_era5
from src.imputation.linear_imputer import LinearImputer
from src.imputation.era5_regression import ERA5RegressionImputer
from src.imputation.mice_imputer import MICEImputer
from src.imputation.brits_imputer import BRITSImputer
from src.imputation.saits_imputer import SAITSImputer
from src.imputation.csdi_imputer import CSDIImputer
from src.imputation.imputeformer_imputer import ImputeFormerImputer
from src.evaluation.imputation_metrics import mask_and_evaluate
from ui.theme import COLORS, inject_css, info_box, metric_card


@st.cache_data(show_spinner="Loading station data…")
def _load():
    stations = load_all_stations()
    era5 = load_era5()
    return {k: merge_era5(v, era5) for k, v in stations.items()}


BENCHMARK_METHODS: dict[str, tuple] = {
    "Linear":      (LinearImputer,        {"max_gap": 200}),
    "ERA5 Ridge":  (ERA5RegressionImputer, {"model_name": "ridge"}),
    "ERA5 RF":     (ERA5RegressionImputer, {"model_name": "random_forest"}),
    "MICE":        (MICEImputer,           {"max_iter": 10}),
    "BRITS":       (BRITSImputer,          {"epochs": 10, "seq_len": 48}),
    "SAITS":       (SAITSImputer,          {"epochs": 10, "seq_len": 48}),
    "CSDI":        (CSDIImputer,           {"epochs": 10, "seq_len": 48, "T_diff": 50, "n_samples": 3}),
    "ImputeFormer":(ImputeFormerImputer,   {"epochs": 10, "seq_len": 48, "d_model": 32, "rank": 8}),
}

GAP_LENGTHS = [1, 2, 6, 12, 24, 48, 72, 168]
GAP_LABELS  = ["1h", "2h", "6h", "12h", "24h", "48h", "72h", "168h"]


# ---------------------------------------------------------------------------
# Low-level metric helpers
# ---------------------------------------------------------------------------

def _metrics_from_arrays(true_vals: np.ndarray, pred_vals: np.ndarray) -> dict | None:
    mask = ~np.isnan(pred_vals) & ~np.isnan(true_vals)
    if mask.sum() == 0:
        return None
    t, p = true_vals[mask], pred_vals[mask]
    mae  = float(np.mean(np.abs(t - p)))
    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    bias = float(np.mean(p - t))
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    denom  = (np.abs(t) + np.abs(p)) / 2
    smape  = float(100 * np.mean(np.where(denom > 0, np.abs(t - p) / denom, 0.0)))
    return {"mae": mae, "rmse": rmse, "r2": r2, "smape": smape, "bias": bias, "n": int(mask.sum())}


def _agg_metrics(metric_list: list[dict], keys=("mae", "rmse", "r2", "smape", "bias")) -> dict:
    return {
        k: round(float(np.nanmean([m[k] for m in metric_list if k in m and not np.isnan(m[k])])), 4)
        for k in keys
    }


# ---------------------------------------------------------------------------
# Block masking for gap-length heatmap
# ---------------------------------------------------------------------------

def _mask_block_and_eval(
    df: pd.DataFrame,
    pollutant: str,
    imputer,
    gap_len: int,
    n_trials: int = 3,
    random_seed: int = RANDOM_SEED,
    **params,
) -> dict | None:
    """Mask gap_len consecutive observed hours and measure imputation quality."""
    col = df[pollutant]
    valid_positions = np.where(col.notna().values)[0]
    if len(valid_positions) < gap_len + 30:
        return None

    trial_results = []
    for trial in range(n_trials):
        rng = np.random.default_rng(random_seed + trial * 31)
        max_start = len(valid_positions) - gap_len - 5
        if max_start <= 5:
            break
        start_vi = int(rng.integers(5, max_start))

        # iloc positions in the full DataFrame
        start_iloc = int(valid_positions[start_vi])
        block_iloc  = list(range(start_iloc, start_iloc + gap_len))
        if block_iloc[-1] >= len(df):
            continue

        block_index = df.index[block_iloc]
        if col.iloc[block_iloc].isna().any():
            continue

        true_vals = col.iloc[block_iloc].values.copy()
        df_test = df.copy()
        df_test.loc[block_index, pollutant] = np.nan

        try:
            t0 = time.time()
            imputed_df, _ = imputer.impute(df_test, pollutant, **params)
            runtime = time.time() - t0
            pred_vals = imputed_df.loc[block_index, pollutant].values
            m = _metrics_from_arrays(true_vals, pred_vals)
            if m:
                m["runtime"] = runtime
                trial_results.append(m)
        except Exception:
            continue

    if not trial_results:
        return None
    return {k: float(np.nanmean([r[k] for r in trial_results if k in r])) for k in trial_results[0]}


# ---------------------------------------------------------------------------
# MCAR masking (wraps existing mask_and_evaluate)
# ---------------------------------------------------------------------------

def _run_mcar(df, pollutant, methods_cfg, mask_pct, n_reps):
    """Run MCAR benchmark; returns list of aggregated result dicts."""
    results = []
    total  = len(methods_cfg) * n_reps
    prog   = st.progress(0, text="Running MCAR benchmark…")
    count  = 0

    for mname, (ImpClass, base_params) in methods_cfg.items():
        rep_metrics = []
        for rep in range(n_reps):
            prog.progress(count / max(total, 1), text=f"{mname} — rep {rep+1}/{n_reps}")
            imp = ImpClass()
            m = mask_and_evaluate(
                df.copy(), pollutant, imp,
                mask_fraction=mask_pct / 100,
                random_seed=RANDOM_SEED + rep,
                **base_params,
            )
            if m:
                rep_metrics.append(m)
            count += 1

        if rep_metrics:
            agg = _agg_metrics(rep_metrics)
            agg["method"]  = mname
            agg["n_reps"]  = len(rep_metrics)
            results.append(agg)

    prog.empty()
    return results


# ---------------------------------------------------------------------------
# Gap-length heatmap runner
# ---------------------------------------------------------------------------

def _run_gap_heatmap(df, pollutant, methods_cfg, metric, n_trials):
    """Build method × gap_length performance matrix."""
    matrix: dict[str, dict[int, float]] = {}
    total  = len(methods_cfg) * len(GAP_LENGTHS)
    prog   = st.progress(0, text="Building gap-length heatmap…")
    count  = 0

    for mname, (ImpClass, base_params) in methods_cfg.items():
        matrix[mname] = {}
        imp = ImpClass()
        for gap_len in GAP_LENGTHS:
            prog.progress(count / max(total, 1),
                          text=f"{mname} / {gap_len}h gap — trial {count+1}/{total}")
            result = _mask_block_and_eval(
                df, pollutant, imp, gap_len,
                n_trials=n_trials,
                random_seed=RANDOM_SEED,
                **base_params,
            )
            matrix[mname][gap_len] = result.get(metric, np.nan) if result else np.nan
            count += 1

    prog.empty()
    return matrix


# ---------------------------------------------------------------------------
# Missingness scenario runner (MCAR / MAR / MNAR)
# ---------------------------------------------------------------------------

def _mask_conditional(
    df: pd.DataFrame,
    pollutant: str,
    threshold: float | None,
    mask_pct: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Return (true_vals, df_masked) for MCAR / MAR / MNAR."""
    col = df[pollutant]
    observed = col.notna()

    if threshold is None:
        # MCAR
        obs_idx = col.index[observed]
        n_mask  = int(len(obs_idx) * mask_pct / 100)
        chosen  = obs_idx[rng.choice(len(obs_idx), size=n_mask, replace=False)]
    else:
        # MAR/MNAR: 80% of masked from high-value bucket
        high_idx = col.index[observed & (col >= threshold)]
        low_idx  = col.index[observed & (col <  threshold)]
        n_total  = int(observed.sum() * mask_pct / 100)
        n_high   = min(int(n_total * 0.80), len(high_idx))
        n_low    = max(n_total - n_high, 0)
        n_low    = min(n_low, len(low_idx))
        chosen   = pd.Index(
            list(high_idx[rng.choice(len(high_idx), size=n_high, replace=False)] if n_high else [])
            + list(low_idx[rng.choice(len(low_idx),  size=n_low,  replace=False)] if n_low  else [])
        )

    true_vals = col[chosen].values.copy()
    df_masked = df.copy()
    df_masked.loc[chosen, pollutant] = np.nan
    return true_vals, df_masked, chosen


def _run_scenarios(df, pollutant, methods_cfg, mask_pct, n_reps):
    """Run MCAR / MAR / MNAR scenario comparison."""
    col = df[pollutant].dropna()
    scenarios = {
        "MCAR":              None,
        "MAR (high values)": float(col.median()),
        "MNAR (extreme)":    float(col.quantile(0.90)),
    }

    all_results = []
    total  = len(methods_cfg) * len(scenarios) * n_reps
    prog   = st.progress(0, text="Running scenario benchmark…")
    count  = 0

    for scen_name, threshold in scenarios.items():
        for mname, (ImpClass, base_params) in methods_cfg.items():
            rep_metrics = []
            for rep in range(n_reps):
                prog.progress(count / max(total, 1),
                              text=f"{scen_name} — {mname} rep {rep+1}/{n_reps}")
                rng = np.random.default_rng(RANDOM_SEED + rep + abs(hash(scen_name)) % 10000)
                try:
                    true_vals, df_masked, chosen = _mask_conditional(
                        df, pollutant, threshold, mask_pct, rng
                    )
                    if len(chosen) == 0:
                        count += 1
                        continue
                    imp = ImpClass()
                    t0  = time.time()
                    imp_df, _ = imp.impute(df_masked, pollutant, **base_params)
                    runtime   = time.time() - t0
                    pred_vals = imp_df.loc[chosen, pollutant].values
                    m = _metrics_from_arrays(true_vals, pred_vals)
                    if m:
                        m["runtime"] = runtime
                        rep_metrics.append(m)
                except Exception:
                    pass
                count += 1

            if rep_metrics:
                agg = _agg_metrics(rep_metrics)
                agg["method"]   = mname
                agg["scenario"] = scen_name
                all_results.append(agg)

    prog.empty()
    return all_results


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------

def _render_mcar_tab(station_data: dict):
    st.markdown("### MCAR Masking Leaderboard")
    info_box(
        "Randomly mask a fraction of observed values (MCAR = Missing Completely At Random), "
        "run each imputation method, and compare to known true values. "
        "Metrics are averaged over multiple random repetitions.",
        kind="info",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        station = st.selectbox("Station", list(station_data), key="mcar_stn",
                               format_func=lambda k: STATIONS[k]["label"])
    with c2:
        pollutant = st.selectbox("Pollutant", STATIONS[station]["monitored_pollutants"], key="mcar_poll")
    with c3:
        mask_pct = st.selectbox("Masking %", [5, 10, 20, 30], index=1, key="mcar_pct")
    with c4:
        n_reps = st.slider("Repetitions", 1, 5, 2, key="mcar_reps")

    methods_sel = st.multiselect(
        "Methods to benchmark",
        list(BENCHMARK_METHODS),
        default=["Linear", "ERA5 Ridge", "MICE", "BRITS"],
        key="mcar_meths",
    )

    if st.button("▶ Run MCAR Benchmark", type="primary", key="btn_mcar"):
        df_s = station_data[station].copy()
        if df_s[pollutant].notna().mean() < 0.01:
            st.error(f"{pollutant} is structurally missing at {STATIONS[station]['label']}.")
            return
        cfg = {m: BENCHMARK_METHODS[m] for m in methods_sel if m in BENCHMARK_METHODS}
        results = _run_mcar(df_s, pollutant, cfg, mask_pct, n_reps)
        st.session_state["mcar_res"]  = results
        st.session_state["mcar_meta"] = dict(station=station, pollutant=pollutant,
                                              mask_pct=mask_pct, n_reps=n_reps)

    if not st.session_state.get("mcar_res"):
        return

    results  = st.session_state["mcar_res"]
    rdf      = pd.DataFrame(results).set_index("method").sort_values("mae")
    meta_m   = st.session_state.get("mcar_meta", {})
    best_mth = rdf.index[0]

    # Metric cards
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        metric_card("Best Method", best_mth, subtitle="Lowest MAE")
    with mc2:
        metric_card("Best MAE",  f"{rdf.loc[best_mth,'mae']:.4f}",
                    subtitle=f"{meta_m.get('pollutant','')} units")
    with mc3:
        best_r2_mth = str(rdf["r2"].idxmax())
        metric_card("Best R²",  f"{rdf['r2'].max():.3f}", subtitle=f"by {best_r2_mth}")
    with mc4:
        metric_card("Methods Tested", str(len(rdf)), subtitle=f"{meta_m.get('n_reps','')} reps each")

    # Leaderboard table
    st.markdown("#### Leaderboard")
    display = rdf[["mae", "rmse", "smape", "r2", "bias", "n_reps"]].copy()
    display.columns = ["MAE ↓", "RMSE ↓", "SMAPE% ↓", "R² ↑", "Bias", "Reps"]
    display.insert(0, "Rank", range(1, len(display) + 1))

    def _row_style(row):
        return [
            "background-color: #d4edda; font-weight:bold" if row.name == best_mth else ""
            for _ in display.columns
        ]

    st.dataframe(display.style.apply(_row_style, axis=1), use_container_width=True)

    # MAE / RMSE grouped bar
    fig_bar = go.Figure([
        go.Bar(x=rdf.index, y=rdf["mae"],  name="MAE",  marker_color=COLORS.get("red",  "#e63946"),
               text=rdf["mae"].round(4),  textposition="outside"),
        go.Bar(x=rdf.index, y=rdf["rmse"], name="RMSE", marker_color=COLORS.get("amber","#e9c46a"),
               text=rdf["rmse"].round(4), textposition="outside"),
    ])
    fig_bar.update_layout(
        barmode="group",
        title=f"MAE & RMSE — {meta_m.get('pollutant','')} @ {meta_m.get('mask_pct','')}% MCAR masking",
        template="plotly_white", height=340,
        margin=dict(l=60, r=20, t=55, b=80),
        legend=dict(orientation="h", y=1.10),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # MAE vs R² scatter
    fig_scat = go.Figure()
    palette  = [COLORS.get("teal","#0a9396"), COLORS.get("red","#e63946"),
                COLORS.get("amber","#e9c46a"), "#4895ef", "#7b2d8b", "#06d6a0", "#ffd166", "#ef476f"]
    for i, mth in enumerate(rdf.index):
        fig_scat.add_trace(go.Scatter(
            x=[rdf.loc[mth, "mae"]], y=[rdf.loc[mth, "r2"]],
            mode="markers+text", name=mth, text=[mth], textposition="top center",
            marker=dict(size=14, color=palette[i % len(palette)]),
        ))
    fig_scat.update_layout(
        title="MAE vs R² Trade-off",
        xaxis_title="MAE (lower = better)",
        yaxis_title="R² (higher = better)",
        template="plotly_white", height=320, showlegend=False,
        margin=dict(l=60, r=20, t=50, b=60),
    )
    st.plotly_chart(fig_scat, use_container_width=True)

    csv = display.to_csv().encode("utf-8")
    st.download_button("📥 Download MCAR results", csv,
                       f"mcar_benchmark_{meta_m.get('station','')}_{meta_m.get('pollutant','')}.csv",
                       key="dl_mcar")


def _render_gap_heatmap_tab(station_data: dict):
    st.markdown("### Gap-Length Performance Heatmap")
    info_box(
        "For each (method, gap length) cell: mask a contiguous block of that many consecutive "
        "observed hours, impute, then measure accuracy against the known true values. "
        "The heatmap reveals where each method degrades as gaps grow. "
        "★ marks the best method for each gap length.",
        kind="info",
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        station = st.selectbox("Station", list(station_data), key="gh_stn",
                               format_func=lambda k: STATIONS[k]["label"])
    with c2:
        pollutant = st.selectbox("Pollutant", STATIONS[station]["monitored_pollutants"], key="gh_poll")
    with c3:
        metric_ch = st.selectbox("Metric", ["mae", "rmse", "r2", "smape"], index=0, key="gh_metric")
    with c4:
        n_trials = st.slider("Trials per cell", 1, 5, 3, key="gh_trials")
    with c5:
        lower_is_better = metric_ch != "r2"
        st.metric("Direction", "↓ lower better" if lower_is_better else "↑ higher better")

    methods_sel = st.multiselect(
        "Methods (recommend ≤ 4 for speed)",
        list(BENCHMARK_METHODS),
        default=["Linear", "ERA5 Ridge", "MICE"],
        key="gh_meths",
    )

    if st.button("▶ Run Gap-Length Heatmap", type="primary", key="btn_gh"):
        df_s = station_data[station].copy()
        if df_s[pollutant].notna().mean() < 0.01:
            st.error(f"{pollutant} is structurally missing at {STATIONS[station]['label']}.")
            return
        cfg    = {m: BENCHMARK_METHODS[m] for m in methods_sel if m in BENCHMARK_METHODS}
        matrix = _run_gap_heatmap(df_s, pollutant, cfg, metric_ch, n_trials)
        st.session_state["gh_matrix"] = matrix
        st.session_state["gh_meta"]   = dict(station=station, pollutant=pollutant,
                                              metric=metric_ch, n_trials=n_trials)

    if not st.session_state.get("gh_matrix"):
        return

    matrix   = st.session_state["gh_matrix"]
    meta_gh  = st.session_state.get("gh_meta", {})
    metric_g = meta_gh.get("metric", "mae")
    lib      = metric_g != "r2"
    methods  = list(matrix)

    # Build numpy matrix
    values = np.array(
        [[matrix[m].get(gl, np.nan) for gl in GAP_LENGTHS] for m in methods],
        dtype=float,
    )

    # Best method per column
    best_per_col = []
    for j in range(len(GAP_LENGTHS)):
        col_v = values[:, j]
        valid = ~np.isnan(col_v)
        if not valid.any():
            best_per_col.append(-1)
        elif lib:
            best_per_col.append(int(np.nanargmin(col_v)))
        else:
            best_per_col.append(int(np.nanargmax(col_v)))

    # Annotation text
    text_arr = []
    for i in range(len(methods)):
        row_t = []
        for j in range(len(GAP_LENGTHS)):
            v = values[i, j]
            if np.isnan(v):
                row_t.append("N/A")
            elif best_per_col[j] == i:
                row_t.append(f"★{v:.3f}")
            else:
                row_t.append(f"{v:.3f}")
        text_arr.append(row_t)

    colorscale = "RdYlGn_r" if lib else "RdYlGn"

    fig_hm = go.Figure(go.Heatmap(
        z=values,
        x=GAP_LABELS,
        y=methods,
        text=text_arr,
        texttemplate="%{text}",
        textfont={"size": 11, "color": "black"},
        colorscale=colorscale,
        colorbar=dict(title=metric_g.upper()),
        hoverongaps=False,
    ))
    fig_hm.update_layout(
        title=(
            f"Gap-Length Heatmap — {metric_g.upper()} "
            f"({'↓ lower = better' if lib else '↑ higher = better'})<br>"
            f"<span style='font-size:11px;color:#666'>★ = best method per gap length | "
            f"{meta_gh.get('pollutant','')} @ {STATIONS.get(meta_gh.get('station',''), {}).get('label','')}</span>"
        ),
        xaxis_title="Gap Length",
        yaxis_title="Method",
        template="plotly_white",
        height=max(260, 70 * len(methods) + 120),
        margin=dict(l=130, r=60, t=90, b=60),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # Best-method summary table
    st.markdown("#### Best Method per Gap Length")
    best_tbl = pd.DataFrame({
        "Gap Length": GAP_LABELS,
        "Best Method": [
            methods[best_per_col[j]] if best_per_col[j] >= 0 else "—"
            for j in range(len(GAP_LENGTHS))
        ],
        metric_g.upper(): [
            round(float(values[best_per_col[j], j]), 4) if best_per_col[j] >= 0 else None
            for j in range(len(GAP_LENGTHS))
        ],
    })
    st.dataframe(best_tbl, use_container_width=True, hide_index=True)

    # Degradation line chart
    st.markdown("#### Performance Degradation with Gap Length")
    fig_line = go.Figure()
    palette  = [COLORS.get("teal","#0a9396"), COLORS.get("red","#e63946"),
                COLORS.get("amber","#e9c46a"), "#4895ef", "#7b2d8b", "#06d6a0", "#ffd166", "#ef476f"]
    for i, mth in enumerate(methods):
        fig_line.add_trace(go.Scatter(
            x=GAP_LABELS, y=values[i, :],
            mode="lines+markers", name=mth,
            line=dict(width=2, color=palette[i % len(palette)]),
            marker=dict(size=7),
            connectgaps=False,
        ))
    fig_line.update_layout(
        title=f"{metric_g.upper()} vs. Gap Length — all methods",
        xaxis_title="Gap Length",
        yaxis_title=metric_g.upper(),
        template="plotly_white", height=340,
        margin=dict(l=60, r=20, t=55, b=60),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # Download
    df_exp = pd.DataFrame(values, index=methods, columns=GAP_LABELS)
    df_exp.index.name = "Method"
    st.download_button("📥 Download heatmap data",
                       df_exp.to_csv().encode("utf-8"),
                       f"gap_heatmap_{meta_gh.get('station','')}_{meta_gh.get('pollutant','')}.csv",
                       key="dl_gh")


def _render_scenario_tab(station_data: dict):
    st.markdown("### Missingness Mechanism Scenarios")
    info_box(
        "<b>Why this matters:</b> Sensor failures often correlate with pollution events — "
        "monitors may drop out during concentration spikes (MNAR) or during certain weather conditions (MAR). "
        "A method that performs well on MCAR data may degrade badly on realistic MNAR missingness.<br><br>"
        "<b>MCAR</b> — Missing Completely At Random (uniform random subset)<br>"
        "<b>MAR</b> — Missing At Random conditional on value level "
        "(80% of masked values drawn from above-median observations)<br>"
        "<b>MNAR</b> — Missing Not At Random "
        "(80% of masked values drawn from above-90th-percentile — hardest for imputation)",
        kind="warning",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        station = st.selectbox("Station", list(station_data), key="sc_stn",
                               format_func=lambda k: STATIONS[k]["label"])
    with c2:
        pollutant = st.selectbox("Pollutant", STATIONS[station]["monitored_pollutants"], key="sc_poll")
    with c3:
        mask_pct = st.selectbox("Masking %", [5, 10, 20], index=1, key="sc_pct")
    with c4:
        n_reps = st.slider("Repetitions", 1, 3, 1, key="sc_reps")

    methods_sel = st.multiselect(
        "Methods to evaluate",
        list(BENCHMARK_METHODS),
        default=["Linear", "ERA5 Ridge", "MICE"],
        key="sc_meths",
    )

    if st.button("▶ Run Scenario Benchmark", type="primary", key="btn_sc"):
        df_s = station_data[station].copy()
        if df_s[pollutant].notna().mean() < 0.01:
            st.error(f"{pollutant} is structurally missing at {STATIONS[station]['label']}.")
            return
        cfg     = {m: BENCHMARK_METHODS[m] for m in methods_sel if m in BENCHMARK_METHODS}
        results = _run_scenarios(df_s, pollutant, cfg, mask_pct, n_reps)
        st.session_state["sc_res"]  = results
        st.session_state["sc_meta"] = dict(station=station, pollutant=pollutant, mask_pct=mask_pct)

    if not st.session_state.get("sc_res"):
        return

    results  = st.session_state["sc_res"]
    meta_sc  = st.session_state.get("sc_meta", {})
    scen_df  = pd.DataFrame(results)
    scenarios = ["MCAR", "MAR (high values)", "MNAR (extreme)"]
    methods   = scen_df["method"].unique().tolist()

    scen_colors = {
        "MCAR":             COLORS.get("teal",  "#0a9396"),
        "MAR (high values)":COLORS.get("amber", "#e9c46a"),
        "MNAR (extreme)":   COLORS.get("red",   "#e63946"),
    }

    # MAE grouped bar
    fig_mae = go.Figure()
    for scen in scenarios:
        sub = scen_df[scen_df["scenario"] == scen]
        if sub.empty:
            continue
        fig_mae.add_trace(go.Bar(
            x=sub["method"], y=sub["mae"], name=scen,
            marker_color=scen_colors.get(scen, "#888"),
            text=sub["mae"].round(3), textposition="outside",
        ))
    fig_mae.update_layout(
        barmode="group",
        title=f"MAE by Missingness Mechanism — {meta_sc.get('pollutant','')} "
              f"@ {meta_sc.get('mask_pct','')}% masking",
        xaxis_title="Method", yaxis_title="MAE",
        template="plotly_white", height=380,
        margin=dict(l=60, r=20, t=60, b=80),
        legend=dict(title="Mechanism"),
    )
    st.plotly_chart(fig_mae, use_container_width=True)

    # R² grouped bar
    fig_r2 = go.Figure()
    for scen in scenarios:
        sub = scen_df[scen_df["scenario"] == scen]
        if sub.empty:
            continue
        fig_r2.add_trace(go.Bar(
            x=sub["method"], y=sub["r2"], name=scen,
            marker_color=scen_colors.get(scen, "#888"),
            text=sub["r2"].round(3), textposition="outside",
        ))
    fig_r2.update_layout(
        barmode="group",
        title=f"R² by Missingness Mechanism — {meta_sc.get('pollutant','')}",
        xaxis_title="Method", yaxis_title="R²",
        template="plotly_white", height=360,
        margin=dict(l=60, r=20, t=55, b=80),
        legend=dict(title="Mechanism"),
    )
    st.plotly_chart(fig_r2, use_container_width=True)

    # Degradation by mechanism: for each method, show MCAR→MAR→MNAR MAE change
    st.markdown("#### MAE Degradation: MCAR → MAR → MNAR")
    fig_deg = go.Figure()
    for i, mth in enumerate(methods):
        sub_m = scen_df[scen_df["method"] == mth].set_index("scenario")
        ys = [sub_m.loc[s, "mae"] if s in sub_m.index else np.nan for s in scenarios]
        fig_deg.add_trace(go.Scatter(
            x=scenarios, y=ys,
            mode="lines+markers+text",
            name=mth,
            text=[f"{v:.3f}" if not np.isnan(v) else "" for v in ys],
            textposition="top center",
            line=dict(width=2),
            marker=dict(size=10),
        ))
    fig_deg.update_layout(
        title="MAE degradation across missingness mechanisms",
        xaxis_title="Missingness mechanism",
        yaxis_title="MAE",
        template="plotly_white", height=340,
        margin=dict(l=60, r=20, t=55, b=60),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig_deg, use_container_width=True)

    # Full results table
    st.markdown("#### Full Scenario Results Table")
    display_sc = scen_df[["scenario", "method", "mae", "rmse", "r2", "smape", "bias"]].copy()
    display_sc.columns = ["Scenario", "Method", "MAE ↓", "RMSE ↓", "R² ↑", "SMAPE% ↓", "Bias"]
    st.dataframe(
        display_sc.sort_values(["Scenario", "MAE ↓"]),
        use_container_width=True, hide_index=True,
    )

    csv = display_sc.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download scenario results", csv,
                       f"scenario_benchmark_{meta_sc.get('station','')}_{meta_sc.get('pollutant','')}.csv",
                       key="dl_sc")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Imputation Benchmark | ISRAP AIRMAPS",
        layout="wide", page_icon="📏",
    )
    inject_css()

    st.markdown(
        f"<h1 style='color:{COLORS.get('navy','#0d1b2a')};'>📏 Imputation Benchmark</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Controlled masking experiments to compare imputation methods across gap lengths, "
        "masking percentages, and real-world missingness mechanisms."
    )

    station_data = _load()

    tab1, tab2, tab3 = st.tabs([
        "📊 MCAR Leaderboard",
        "🔥 Gap-Length Heatmap",
        "🎭 Missingness Scenarios",
    ])
    with tab1:
        _render_mcar_tab(station_data)
    with tab2:
        _render_gap_heatmap_tab(station_data)
    with tab3:
        _render_scenario_tab(station_data)


if __name__ == "__main__":
    main()

