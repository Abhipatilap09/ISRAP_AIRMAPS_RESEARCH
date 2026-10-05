"""Final Pipeline — orchestrate and visualise the full data processing workflow."""
from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.config import POLLUTANTS, STATIONS, EXPORTS_DIR
from src.pipeline.pipeline import run_pipeline
from ui.theme import COLORS, inject_css, info_box, metric_card, badge


# ── Stage metadata ─────────────────────────────────────────────────────────────

PIPELINE_STAGES = [
    {
        "id": "ingest",
        "name": "Data Ingestion",
        "icon": "📂",
        "desc": "Load all station CSVs and ERA5 file. Detect schemas and datetime columns.",
        "outputs": "Raw DataFrames per station",
    },
    {
        "id": "align",
        "name": "Hourly Alignment",
        "icon": "🕐",
        "desc": "Reindex each station to a complete hourly DatetimeIndex. Fill newly created rows with NaN.",
        "outputs": "Complete-grid DataFrames",
    },
    {
        "id": "structural",
        "name": "Structural Missingness",
        "icon": "🔍",
        "desc": "Identify station-pollutant pairs with <1% valid data (not monitored). Exclude from further processing.",
        "outputs": "Availability matrix, structural missing flags",
    },
    {
        "id": "era5",
        "name": "ERA5 Merge",
        "icon": "🌤️",
        "desc": "Left-join ERA5 meteorology onto each station by hourly datetime. Report join quality.",
        "outputs": "ERA5-enriched DataFrames",
    },
    {
        "id": "physical",
        "name": "Physical Validation",
        "icon": "⚖️",
        "desc": "Flag values below zero or above physical maxima. Mark Station 3 SO2 for exclusion.",
        "outputs": "is_negative flags, is_physical_anomaly flags",
    },
    {
        "id": "anomaly",
        "name": "Anomaly Detection",
        "icon": "⚗️",
        "desc": "Run selected statistical and ML anomaly detectors. Build ensemble flag and confidence.",
        "outputs": "Anomaly flags and scores per method",
    },
    {
        "id": "gaps",
        "name": "Gap Classification",
        "icon": "📊",
        "desc": "Classify missing runs as very_short (1-2h), medium (3-24h), long (>24h), very_long (>168h).",
        "outputs": "gap_id, gap_length, gap_category columns",
    },
    {
        "id": "impute",
        "name": "Imputation",
        "icon": "🔧",
        "desc": "Apply gap-aware imputation strategy. Short gaps → linear; medium → ERA5 regression; long → MICE.",
        "outputs": "final_imputed_value, was_imputed, imputation_method columns",
    },
    {
        "id": "export",
        "name": "Export",
        "icon": "📦",
        "desc": "Write processed CSVs and Parquet files to exports/. Generate manifest JSON.",
        "outputs": "Processed station CSVs, combined dataset, manifest",
    },
]


def _stage_flow_chart() -> go.Figure:
    """Draw a horizontal pipeline flow diagram."""
    n = len(PIPELINE_STAGES)
    fig = go.Figure()

    # Draw boxes
    for i, s in enumerate(PIPELINE_STAGES):
        x = i / (n - 1)
        fig.add_shape(
            type="rect",
            x0=x - 0.04, y0=0.25, x1=x + 0.04, y1=0.75,
            line=dict(color=COLORS["teal"], width=2),
            fillcolor="#e8f4f8",
        )
        fig.add_annotation(
            x=x, y=0.5,
            text=f"{s['icon']}<br><b>{s['name']}</b>",
            showarrow=False,
            font=dict(size=10, color=COLORS["navy"]),
            align="center",
        )
        # Arrow between boxes
        if i < n - 1:
            x2 = (i + 1) / (n - 1)
            fig.add_annotation(
                x=(x + 0.04 + x2 - 0.04) / 2, y=0.5,
                ax=x + 0.04, ay=0.5,
                axref="x", ayref="y",
                xref="x", yref="y",
                arrowhead=2, arrowsize=1, arrowwidth=2,
                arrowcolor=COLORS["teal"],
                text="",
            )

    fig.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[-0.06, 1.06], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def _stage_result_card(stage_name: str, status: str, detail: str = "",
                        runtime: float = 0.0, metrics: dict | None = None) -> None:
    color = {
        "ok": COLORS["green"],
        "error": COLORS["red"],
        "skipped": COLORS["gray"],
        "running": COLORS["teal"],
    }.get(status, COLORS["muted"])
    label = {
        "ok": "Completed",
        "error": "Failed",
        "skipped": "Skipped",
        "running": "Running",
    }.get(status, status.title())
    rt = f" — {runtime:.1f}s" if runtime > 0 else ""

    m_html = ""
    if metrics:
        pairs = [f"<b>{k}:</b> {v}" for k, v in list(metrics.items())[:4]]
        m_html = f"<br><span style='font-size:0.79rem; color:{COLORS['muted']};'>{' &nbsp;|&nbsp; '.join(pairs)}</span>"

    st.markdown(
        f"""<div style='
            border-left: 4px solid {color};
            padding: 10px 16px;
            margin: 4px 0;
            background: #fff;
            border-radius: 0 8px 8px 0;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        '>
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <span><b style='color:{color};'>{label}</b>
            <span style='color:{COLORS["navy"]}; font-weight:600;'> — {stage_name}{rt}</span></span>
        </div>
        <div style='color:{COLORS["muted"]}; font-size:0.82rem; margin-top:2px;'>{detail}{m_html}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _build_summary_charts(pipeline_result) -> None:
    """Build runtime and anomaly charts from pipeline result."""
    stages = pipeline_result.stages
    if not stages:
        return

    names = [s.stage for s in stages]
    runtimes = [s.runtime_sec for s in stages]
    statuses = [s.status for s in stages]
    colors = [COLORS["green"] if s == "ok" else
              COLORS["red"] if s == "error" else
              COLORS["gray"] for s in statuses]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Stage Runtime (seconds)", "Stage Status"))

    fig.add_trace(
        go.Bar(x=names, y=runtimes, marker_color=colors, name="Runtime"),
        row=1, col=1,
    )

    status_counts = {"ok": 0, "error": 0, "skipped": 0}
    for s in statuses:
        status_counts[s] = status_counts.get(s, 0) + 1

    fig.add_trace(
        go.Pie(
            labels=list(status_counts.keys()),
            values=list(status_counts.values()),
            marker_colors=[COLORS["green"], COLORS["red"], COLORS["gray"]],
            name="Status",
        ),
        row=1, col=2,
    )

    fig.update_layout(
        height=300,
        template="plotly_white",
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(
        page_title="Final Pipeline | ISRAP AIRMAPS",
        layout="wide",
        page_icon="\U0001f504",
    )
    inject_css()

    st.markdown(
        f"<h1 style='color:{COLORS['navy']};'>\U0001f504 Final Processing Pipeline</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Run the complete data processing pipeline from raw files to a cleaned, "
        "anomaly-flagged, imputed research dataset."
    )

    # ── Pipeline flow diagram ──────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">Processing Workflow</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(_stage_flow_chart(), use_container_width=True)

    # ── Stage reference table ──────────────────────────────────────────────────
    with st.expander("Stage descriptions", expanded=False):
        for s in PIPELINE_STAGES:
            st.markdown(
                f"**{s['icon']} {s['name']}** — {s['desc']}  \n"
                f"*Outputs:* {s['outputs']}"
            )
            st.divider()

    # ── Configuration ──────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">Pipeline Configuration</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Scope**")
        sel_stations = st.multiselect(
            "Stations",
            list(STATIONS.keys()),
            default=list(STATIONS.keys()),
            format_func=lambda k: STATIONS[k]["label"],
        )
        anomaly_methods = st.multiselect(
            "Anomaly Methods",
            [
                "Physical Constraints",
                "IQR Fence",
                "Hampel Filter",
                "Rolling Z-Score",
                "Isolation Forest",
                "STL Residual",
            ],
            default=["Physical Constraints", "IQR Fence", "Hampel Filter"],
        )

    with c2:
        st.markdown("**Processing**")
        imputation_strategy = st.selectbox(
            "Imputation Strategy",
            ["Auto (gap-aware)", "Linear only", "ERA5 RF only", "MICE only"],
            help=(
                "Auto: uses linear for 1-2h gaps, ERA5 regression for 3-24h, "
                "MICE for longer gaps."
            ),
        )
        exclude_station3_so2 = st.checkbox(
            "Exclude Station 3 SO2 (recommended)", value=True,
            help="Station 3 SO2 has ~33% negative values indicating systematic corruption."
        )
        ensemble_method = st.selectbox(
            "Anomaly Ensemble",
            ["majority_vote", "union", "weighted"],
            help="How to combine flags from multiple anomaly methods."
        )
        min_votes = st.slider(
            "Minimum votes (majority)", 1, len(anomaly_methods) or 2, 2,
            help="An observation is flagged in ensemble if flagged by at least this many methods."
        )

    with c3:
        st.markdown("**Execution**")
        exec_mode = st.radio(
            "Mode",
            ["Fast", "Full Research"],
            help=(
                "Fast mode: fewer estimators, fewer epochs, quick results. "
                "Full Research mode: production-quality settings."
            ),
        )
        random_seed = st.number_input("Random Seed", value=42, min_value=0,
                                       help="Controls all randomness for reproducibility.")
        save_parquet = st.checkbox("Save Parquet outputs", value=True,
                                    help="Also writes .parquet files alongside CSVs.")

    # ── Pre-run summary ────────────────────────────────────────────────────────
    if sel_stations:
        n_monitored = sum(
            len(STATIONS[s]["monitored_pollutants"]) for s in sel_stations
        )
        st.markdown("**Pre-run summary:**")
        pr_cols = st.columns(5)
        pr_cols[0].metric("Stations", len(sel_stations))
        pr_cols[1].metric("Pollutant channels", n_monitored)
        pr_cols[2].metric("Anomaly methods", len(anomaly_methods))
        pr_cols[3].metric("Mode", exec_mode)
        pr_cols[4].metric("Seed", random_seed)
    else:
        info_box("Select at least one station to configure the pipeline.", kind="warn")

    st.divider()

    # ── Run ────────────────────────────────────────────────────────────────────
    col_run, col_load = st.columns([1, 4])
    with col_run:
        run_btn = st.button(
            "▶ Run Complete Pipeline", type="primary",
            disabled=not bool(sel_stations),
        )
    with col_load:
        saved_runs = sorted(EXPORTS_DIR.glob("pipeline_*.json")) if EXPORTS_DIR.exists() else []
        if saved_runs:
            load_run = st.selectbox(
                "Load previous run",
                ["—"] + [f.stem for f in saved_runs[-10:][::-1]],
                label_visibility="collapsed",
            )
        else:
            load_run = "—"

    # ── Execute pipeline ───────────────────────────────────────────────────────
    if run_btn and sel_stations:
        st.markdown(
            '<div class="section-header">Pipeline Execution</div>',
            unsafe_allow_html=True,
        )

        progress_bar = st.progress(0.0)
        status_ph = st.empty()
        log_ph = st.empty()
        t_start = time.time()

        def _progress(stage_name: str, fraction: float):
            progress_bar.progress(min(fraction, 1.0))
            elapsed = time.time() - t_start
            status_ph.info(f"Running: **{stage_name}** — {elapsed:.0f}s elapsed")

        with st.spinner("Running pipeline — this may take 30-120 seconds…"):
            pipeline_result = run_pipeline(
                station_keys=sel_stations,
                anomaly_methods=anomaly_methods,
                imputation_strategy=imputation_strategy,
                exclude_station3_so2=exclude_station3_so2,
                exec_mode=exec_mode,
                random_seed=int(random_seed),
                progress_callback=_progress,
            )

        progress_bar.progress(1.0)
        status_ph.empty()
        log_ph.empty()

        elapsed_total = time.time() - t_start

        # ── Stage results ──────────────────────────────────────────────────────
        st.markdown(
            '<div class="section-header">Stage Results</div>',
            unsafe_allow_html=True,
        )

        for sr in pipeline_result.stages:
            _stage_result_card(
                sr.stage, sr.status, sr.detail, sr.runtime_sec,
                sr.metrics if hasattr(sr, "metrics") else None,
            )

        # ── Summary metrics ────────────────────────────────────────────────────
        st.markdown(
            '<div class="section-header">Summary</div>',
            unsafe_allow_html=True,
        )

        stat_color = COLORS["green"] if pipeline_result.status == "ok" else COLORS["red"]
        m_cols = st.columns(6)
        m_cols[0].markdown(
            metric_card("Status", pipeline_result.status.upper(), color=stat_color),
            unsafe_allow_html=True,
        )
        m_cols[1].markdown(
            metric_card("Stations", str(pipeline_result.stations_processed)),
            unsafe_allow_html=True,
        )
        m_cols[2].markdown(
            metric_card("Anomalies Detected", f"{pipeline_result.total_anomalies_flagged:,}",
                        color=COLORS["red"]),
            unsafe_allow_html=True,
        )
        m_cols[3].markdown(
            metric_card("Values Imputed", f"{pipeline_result.total_values_imputed:,}",
                        color=COLORS["green"]),
            unsafe_allow_html=True,
        )
        m_cols[4].markdown(
            metric_card("Total Runtime", f"{elapsed_total:.1f}s"),
            unsafe_allow_html=True,
        )
        ok_stages = sum(1 for s in pipeline_result.stages if s.status == "ok")
        m_cols[5].markdown(
            metric_card("Stages OK", f"{ok_stages}/{len(pipeline_result.stages)}"),
            unsafe_allow_html=True,
        )

        # ── Runtime chart ──────────────────────────────────────────────────────
        _build_summary_charts(pipeline_result)

        # ── Alerts ────────────────────────────────────────────────────────────
        if pipeline_result.status == "ok":
            info_box(
                f"Pipeline completed in <b>{elapsed_total:.1f}s</b>. "
                f"<b>{pipeline_result.total_anomalies_flagged:,}</b> anomalies detected and "
                f"<b>{pipeline_result.total_values_imputed:,}</b> values imputed. "
                "Outputs saved to the <b>exports/</b> folder.",
                kind="info",
            )
        else:
            failed = [s.stage for s in pipeline_result.stages if s.status == "error"]
            info_box(
                f"Pipeline finished with errors in stages: {', '.join(failed)}. "
                "Partial results may still be available in exports/.",
                kind="danger",
            )

        # ── Downloads ──────────────────────────────────────────────────────────
        st.markdown(
            '<div class="section-header">Download Outputs</div>',
            unsafe_allow_html=True,
        )

        export_files = []
        if EXPORTS_DIR.exists():
            export_files = sorted(
                EXPORTS_DIR.glob(f"pipeline_{pipeline_result.pipeline_id}_*")
            )

        if export_files:
            # ZIP of all outputs
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for f in export_files:
                    if f.is_file():
                        zf.write(f, f.name)
            buf.seek(0)
            st.download_button(
                "\U0001f4e6 Download All Pipeline Outputs (ZIP)",
                buf.getvalue(),
                f"pipeline_{pipeline_result.pipeline_id}.zip",
                mime="application/zip",
            )

            # Individual file table
            file_rows = []
            for f in export_files:
                if f.is_file():
                    sz = f.stat().st_size
                    file_rows.append({
                        "File": f.name,
                        "Size": f"{sz/1000:.0f} KB" if sz < 1_000_000 else f"{sz/1_000_000:.1f} MB",
                        "Type": f.suffix.upper().lstrip("."),
                    })
            if file_rows:
                st.dataframe(pd.DataFrame(file_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No output files found for this pipeline run.")

        # ── Manifest ──────────────────────────────────────────────────────────
        with st.expander("Pipeline manifest (JSON)", expanded=False):
            st.json({
                "pipeline_id": pipeline_result.pipeline_id,
                "started_at": pipeline_result.started_at,
                "finished_at": pipeline_result.finished_at,
                "status": pipeline_result.status,
                "config": pipeline_result.config,
                "total_anomalies_flagged": pipeline_result.total_anomalies_flagged,
                "total_values_imputed": pipeline_result.total_values_imputed,
                "stations_processed": pipeline_result.stations_processed,
                "stages": [
                    {
                        "stage": s.stage,
                        "status": s.status,
                        "runtime_sec": round(s.runtime_sec, 2),
                        "detail": s.detail,
                    }
                    for s in pipeline_result.stages
                ],
            })

            manifest_bytes = json.dumps({
                "pipeline_id": pipeline_result.pipeline_id,
                "config": pipeline_result.config,
            }, indent=2).encode()
            st.download_button(
                "Download manifest (JSON)",
                manifest_bytes,
                f"manifest_{pipeline_result.pipeline_id}.json",
                mime="application/json",
            )

    elif not run_btn and load_run != "—":
        # ── Load previous run ──────────────────────────────────────────────────
        manifest_path = EXPORTS_DIR / f"{load_run}.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            st.success(f"Loaded run: {load_run}")
            with st.expander("Manifest", expanded=True):
                st.json(manifest)
        else:
            st.warning(f"Manifest file not found: {manifest_path}")

    elif run_btn and not sel_stations:
        st.warning("Select at least one station before running.")

    # ── Information ────────────────────────────────────────────────────────────
    if not run_btn:
        st.markdown(
            '<div class="section-header">How the Pipeline Works</div>',
            unsafe_allow_html=True,
        )
        info_box(
            "<b>Data lineage:</b> Original CSVs are never modified. "
            "The pipeline produces separate processed files in exports/ "
            "with full provenance columns: original_value, was_anomaly, was_imputed, "
            "final_imputed_value, selected_imputation_method.",
            kind="info",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**What gets produced:**")
            st.markdown(
                "- `pipeline_{id}_station*.csv` — processed station files\n"
                "- `pipeline_{id}_combined.csv` — all stations merged\n"
                "- `pipeline_{id}_anomaly_flags.csv` — anomaly flag columns only\n"
                "- `pipeline_{id}_gap_summary.csv` — gap analysis table\n"
                "- `pipeline_{id}_manifest.json` — configuration and provenance"
            )
        with col_b:
            st.markdown("**Scientific safeguards:**")
            st.markdown(
                "- Original values always preserved\n"
                "- Station 3 SO2 excluded by default\n"
                "- Structural missingness never imputed\n"
                "- Anomalies flagged before imputation\n"
                "- Chronological train-test splits for all models\n"
                "- Random seed stored in manifest for reproducibility"
            )


if __name__ == "__main__":
    main()

