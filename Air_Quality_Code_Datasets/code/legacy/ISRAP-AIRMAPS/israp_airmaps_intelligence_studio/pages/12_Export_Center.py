"""Export Center — download processed datasets, reports, and analysis outputs."""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from src.config import POLLUTANTS, STATIONS, EXPORTS_DIR, EXPERIMENTS_DIR
from src.data_loader import (
    audit_station_pollutant,
    build_availability_matrix,
    load_all_stations,
    load_era5,
    merge_era5,
)
from src.gap_analysis import gap_summary
from services.experiment_store import list_experiments
from services.report_generator import generate_markdown_report, generate_html_report
from ui.theme import COLORS, inject_css, info_box, metric_card


@st.cache_data(show_spinner="Loading data…")
def _load():
    stations = load_all_stations()
    era5 = load_era5()
    return {k: merge_era5(v, era5) for k, v in stations.items()}


@st.cache_data(show_spinner="Computing audit…")
def _build_audit(station_data_keys: tuple) -> pd.DataFrame:
    station_data = _load()
    audit_rows = []
    for skey, df in station_data.items():
        for p in POLLUTANTS:
            audit_rows.append(audit_station_pollutant(df, skey, p))
    return pd.DataFrame(audit_rows)


@st.cache_data(show_spinner="Computing gaps…")
def _build_gaps(station_data_keys: tuple) -> pd.DataFrame:
    station_data = _load()
    return gap_summary(station_data)


def _negative_df(station_data: dict) -> pd.DataFrame:
    neg_all = []
    for skey, df in station_data.items():
        for p in POLLUTANTS:
            if p in df.columns:
                neg = df[df[p] < 0][["datetime", p]].copy()
                if not neg.empty:
                    neg["station"] = skey
                    neg["pollutant"] = p
                    neg["original_value"] = neg[p]
                    neg_all.append(
                        neg[["datetime", "station", "pollutant", "original_value"]]
                    )
    return pd.concat(neg_all) if neg_all else pd.DataFrame()


def _station_export_df(df: pd.DataFrame, skey: str) -> pd.DataFrame:
    """Build a clean export frame for one station."""
    pollutant_cols = [p for p in POLLUTANTS if p in df.columns]
    # Include ERA5 vars if present
    era5_cols = [c for c in df.columns if c in [
        "temp_c", "dewpoint_c", "surface_pressure_hpa", "precip_mm",
        "wind_speed", "blh", "relative_humidity"
    ]]
    base_cols = ["datetime", "station_key", "station_label", "site"]
    export_cols = base_cols + pollutant_cols + era5_cols
    export_df = df[[c for c in export_cols if c in df.columns]].copy()
    # Add structural missing flags
    for p in pollutant_cols:
        flag_col = f"{p}_structural_missing"
        if flag_col in df.columns:
            export_df[flag_col] = df[flag_col]
    return export_df


def main():
    st.set_page_config(
        page_title="Export Center | ISRAP AIRMAPS",
        layout="wide",
        page_icon="📦",
    )
    inject_css()

    st.markdown(
        f"<h1 style='color:{COLORS['navy']};'>📦 Export Center</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Download processed datasets, audit reports, gap summaries, and analysis outputs. "
        "Raw files are never modified."
    )

    station_data = _load()
    station_keys_tuple = tuple(sorted(station_data.keys()))

    # ── Quick metrics ──────────────────────────────────────────────────────────
    total_rows = sum(len(df) for df in station_data.values())
    n_exports = len(list(EXPORTS_DIR.glob("*"))) if EXPORTS_DIR.exists() else 0
    exp_count = len(list_experiments(limit=1000))

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Stations", str(len(station_data))), unsafe_allow_html=True)
    c2.markdown(metric_card("Total Records", f"{total_rows:,}"), unsafe_allow_html=True)
    c3.markdown(metric_card("Files in exports/", str(n_exports)), unsafe_allow_html=True)
    c4.markdown(metric_card("Saved Experiments", str(exp_count)), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_data, tab_audit, tab_reports, tab_pipeline, tab_all = st.tabs([
        "Station Data",
        "Audit & Quality",
        "Reports",
        "Pipeline Outputs",
        "Download All",
    ])

    # ── Tab 1: Station Data ────────────────────────────────────────────────────
    with tab_data:
        st.markdown(
            '<div class="section-header">Per-Station Data Downloads</div>',
            unsafe_allow_html=True,
        )
        info_box(
            "Downloads include the hourly-reindexed station data merged with ERA5 variables. "
            "Original values are preserved. Structural missing flags included.",
            kind="info",
        )

        cols = st.columns(3)
        for i, (skey, df) in enumerate(station_data.items()):
            meta = STATIONS[skey]
            with cols[i % 3]:
                export_df = _station_export_df(df, skey)
                st.markdown(
                    f"**{meta['label']}**  \n"
                    f"Site {meta['site']} | {len(export_df):,} rows | "
                    f"{', '.join(meta['monitored_pollutants'])}"
                )
                st.download_button(
                    f"📥 Download {skey}",
                    export_df.to_csv(index=False).encode(),
                    f"{skey}_hourly_processed.csv",
                    mime="text/csv",
                    key=f"dl_station_{skey}",
                )

        st.divider()
        st.markdown("**Combined Multi-Station Dataset**")
        if st.button("Build combined dataset", key="build_combined"):
            with st.spinner("Combining stations…"):
                all_dfs = []
                for skey, df in station_data.items():
                    edf = _station_export_df(df, skey)
                    all_dfs.append(edf)
                combined = pd.concat(all_dfs, ignore_index=True)
            st.download_button(
                "📥 Download Combined (CSV)",
                combined.to_csv(index=False).encode(),
                "israp_airmaps_all_stations_combined.csv",
                mime="text/csv",
                key="dl_combined",
            )
            st.success(f"Combined dataset: {len(combined):,} rows, {len(combined.columns)} columns")

    # ── Tab 2: Audit & Quality ─────────────────────────────────────────────────
    with tab_audit:
        st.markdown(
            '<div class="section-header">Data Quality Downloads</div>',
            unsafe_allow_html=True,
        )

        with st.spinner("Computing audit…"):
            audit_df = _build_audit(station_keys_tuple)
            avail = build_availability_matrix(station_data)
            gap_df = _build_gaps(station_keys_tuple)
            neg_df = _negative_df(station_data)

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**Full Data Audit**")
            st.caption(f"{len(audit_df)} station-pollutant pairs analyzed")
            st.download_button(
                "📥 Data Audit Summary (CSV)",
                audit_df.to_csv(index=False).encode(),
                "data_audit_summary.csv",
                key="dl_audit",
            )

        with col_b:
            st.markdown("**Availability Matrix**")
            st.caption("Station × pollutant completeness percentages")
            st.download_button(
                "📥 Availability Matrix (CSV)",
                avail.to_csv(index=False).encode(),
                "station_pollutant_availability.csv",
                key="dl_avail",
            )

        with col_c:
            st.markdown("**Negative Values Report**")
            st.caption(f"{len(neg_df):,} negative values found" if not neg_df.empty else "No negative values")
            if not neg_df.empty:
                st.download_button(
                    "📥 Negative Values (CSV)",
                    neg_df.to_csv(index=False).encode(),
                    "negative_values_summary.csv",
                    key="dl_neg",
                )

        col_d, col_e = st.columns(2)
        with col_d:
            st.markdown("**Missing-Data Gap Summary**")
            n_gaps = len(gap_df) if not gap_df.empty else 0
            st.caption(f"{n_gaps:,} gaps identified")
            if not gap_df.empty:
                st.download_button(
                    "📥 Gap Summary (CSV)",
                    gap_df.to_csv(index=False).encode(),
                    "missing_gap_summary.csv",
                    key="dl_gaps",
                )

        with col_e:
            st.markdown("**Experiment History**")
            exps = list_experiments(limit=10000)
            if exps:
                exp_df = pd.DataFrame(exps)
                st.caption(f"{len(exps):,} experiments recorded")
                st.download_button(
                    "📥 Experiment History (CSV)",
                    exp_df.to_csv(index=False).encode(),
                    "experiment_history.csv",
                    key="dl_exps",
                )

        # Preview tables
        st.markdown("---")
        preview_choice = st.selectbox(
            "Preview table",
            ["Audit Summary", "Availability Matrix", "Gap Summary", "Negative Values"],
            key="preview_choice",
        )
        if preview_choice == "Audit Summary":
            st.dataframe(audit_df.head(30), use_container_width=True, hide_index=True)
        elif preview_choice == "Availability Matrix":
            disp_cols = ["station", "label"] + [p for p in POLLUTANTS if p in avail.columns]
            st.dataframe(avail[[c for c in disp_cols if c in avail.columns]],
                         use_container_width=True, hide_index=True)
        elif preview_choice == "Gap Summary" and not gap_df.empty:
            st.dataframe(gap_df.head(50), use_container_width=True, hide_index=True)
        elif preview_choice == "Negative Values" and not neg_df.empty:
            st.dataframe(neg_df.head(50), use_container_width=True, hide_index=True)

    # ── Tab 3: Reports ─────────────────────────────────────────────────────────
    with tab_reports:
        st.markdown(
            '<div class="section-header">Research Reports</div>',
            unsafe_allow_html=True,
        )
        info_box(
            "Reports summarise the full dataset audit, missing-data patterns, "
            "Station 3 SO₂ warning, and scientific safeguards applied. "
            "Generated on-demand from current data.",
            kind="info",
        )

        with st.spinner("Building reports (first time may take 30s)…"):
            audit_df = _build_audit(station_keys_tuple)
            avail = build_availability_matrix(station_data)
            gap_df = _build_gaps(station_keys_tuple)

        col_md, col_html = st.columns(2)

        with col_md:
            st.markdown("**Markdown Report**")
            st.caption("Plain Markdown — paste into Jupyter, Obsidian, or GitHub")
            if st.button("Generate Markdown Report", key="gen_md"):
                with st.spinner("Generating…"):
                    md_text = generate_markdown_report(
                        station_data, audit_df, gap_df, avail
                    )
                st.download_button(
                    "📥 Download Report (Markdown)",
                    md_text.encode(),
                    "israp_airmaps_report.md",
                    mime="text/markdown",
                    key="dl_md",
                )
                with st.expander("Preview (first 100 lines)", expanded=False):
                    preview_lines = "\n".join(md_text.splitlines()[:100])
                    st.markdown(preview_lines)

        with col_html:
            st.markdown("**HTML Report**")
            st.caption("Self-contained HTML — open in any browser")
            if st.button("Generate HTML Report", key="gen_html"):
                with st.spinner("Generating…"):
                    html_text = generate_html_report(
                        station_data, audit_df, gap_df, avail
                    )
                st.download_button(
                    "📥 Download Report (HTML)",
                    html_text.encode(),
                    "israp_airmaps_report.html",
                    mime="text/html",
                    key="dl_html",
                )
                st.success("HTML report ready — click button above to download.")

        st.divider()
        st.markdown("**Audit Preview**")
        st.dataframe(audit_df.head(20), use_container_width=True, hide_index=True)

    # ── Tab 4: Pipeline Outputs ────────────────────────────────────────────────
    with tab_pipeline:
        st.markdown(
            '<div class="section-header">Pipeline Output Files</div>',
            unsafe_allow_html=True,
        )
        info_box(
            "Files produced by the Final Pipeline page appear here. "
            "Run the pipeline first to generate outputs.",
            kind="info",
        )

        pipeline_files = sorted(EXPORTS_DIR.glob("pipeline_*")) if EXPORTS_DIR.exists() else []

        if not pipeline_files:
            st.info(
                "No pipeline output files found. "
                "Go to the **Final Pipeline** page and run the pipeline to generate outputs."
            )
        else:
            # Group by pipeline ID
            run_ids = set()
            for f in pipeline_files:
                parts = f.stem.split("_")
                if len(parts) >= 2:
                    run_ids.add(parts[1])

            for run_id in sorted(run_ids, reverse=True):
                files_in_run = [f for f in pipeline_files if f"_{run_id}_" in f.stem or f.stem.endswith(run_id)]
                with st.expander(f"Pipeline run: {run_id} ({len(files_in_run)} files)", expanded=True):
                    file_rows = []
                    for f in files_in_run:
                        if f.is_file():
                            sz = f.stat().st_size
                            size_str = f"{sz/1_000_000:.2f} MB" if sz > 1_000_000 else f"{sz/1000:.0f} KB"
                            file_rows.append({"File": f.name, "Size": size_str, "Type": f.suffix})

                    if file_rows:
                        st.dataframe(pd.DataFrame(file_rows), use_container_width=True, hide_index=True)

                    # ZIP download
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                        for f in files_in_run:
                            if f.is_file():
                                zf.write(f, f.name)
                    buf.seek(0)
                    st.download_button(
                        f"📦 Download run {run_id} (ZIP)",
                        buf.getvalue(),
                        f"pipeline_{run_id}.zip",
                        mime="application/zip",
                        key=f"dl_run_{run_id}",
                    )

    # ── Tab 5: Download All ────────────────────────────────────────────────────
    with tab_all:
        st.markdown(
            '<div class="section-header">Complete Export Package</div>',
            unsafe_allow_html=True,
        )
        info_box(
            "Creates a ZIP file containing all station CSVs, audit tables, gap summaries, "
            "negative-values report, availability matrix, experiment history, and Markdown report.",
            kind="info",
        )

        with st.spinner("Preparing export package…"):
            audit_df = _build_audit(station_keys_tuple)
            avail = build_availability_matrix(station_data)
            gap_df = _build_gaps(station_keys_tuple)
            neg_df = _negative_df(station_data)

        col_opts, col_preview = st.columns([1, 2])
        with col_opts:
            include_era5 = st.checkbox("Include ERA5 columns", value=True)
            include_structural_flags = st.checkbox("Include structural missing flags", value=True)
            include_report = st.checkbox("Include Markdown report", value=True)
            include_experiments = st.checkbox("Include experiment history", value=True)

        with col_preview:
            items = [
                f"{skey}_data.csv" for skey in station_data
            ] + [
                "combined_all_stations.csv",
                "data_audit_summary.csv",
                "station_pollutant_availability.csv",
                "missing_gap_summary.csv",
                "negative_values_summary.csv",
            ]
            if include_report:
                items.append("israp_airmaps_report.md")
            if include_experiments:
                items.append("experiment_history.csv")
            st.markdown("**Contents:**")
            for item in items:
                st.markdown(f"- {item}")

        if st.button("📦 Create & Download Full ZIP", type="primary"):
            with st.spinner("Building ZIP — this may take 30-60 seconds…"):
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    # Station CSVs
                    all_station_dfs = []
                    for skey, df in station_data.items():
                        edf = _station_export_df(df, skey)
                        if not include_era5:
                            era5_c = [c for c in edf.columns if c in [
                                "temp_c", "dewpoint_c", "surface_pressure_hpa",
                                "precip_mm", "wind_speed", "blh", "relative_humidity"
                            ]]
                            edf = edf.drop(columns=era5_c, errors="ignore")
                        if not include_structural_flags:
                            flag_c = [c for c in edf.columns if c.endswith("_structural_missing")]
                            edf = edf.drop(columns=flag_c, errors="ignore")
                        zf.writestr(f"{skey}_data.csv", edf.to_csv(index=False))
                        all_station_dfs.append(edf)

                    # Combined
                    combined = pd.concat(all_station_dfs, ignore_index=True)
                    zf.writestr("combined_all_stations.csv", combined.to_csv(index=False))

                    # Audit tables
                    zf.writestr("data_audit_summary.csv", audit_df.to_csv(index=False))
                    zf.writestr("station_pollutant_availability.csv", avail.to_csv(index=False))
                    if not gap_df.empty:
                        zf.writestr("missing_gap_summary.csv", gap_df.to_csv(index=False))
                    if not neg_df.empty:
                        zf.writestr("negative_values_summary.csv", neg_df.to_csv(index=False))

                    # Report
                    if include_report:
                        md_text = generate_markdown_report(station_data, audit_df, gap_df, avail)
                        zf.writestr("israp_airmaps_report.md", md_text)

                    # Experiments
                    if include_experiments:
                        exps = list_experiments(limit=10000)
                        if exps:
                            exp_csv = pd.DataFrame(exps).to_csv(index=False)
                            zf.writestr("experiment_history.csv", exp_csv)

                buf.seek(0)
                total_size = buf.getbuffer().nbytes

            st.download_button(
                "📥 Download Full Export ZIP",
                buf.getvalue(),
                "israp_airmaps_full_export.zip",
                mime="application/zip",
            )
            st.success(
                f"ZIP ready — {total_size/1_000_000:.1f} MB containing {len(items)} files."
            )


if __name__ == "__main__":
    main()

