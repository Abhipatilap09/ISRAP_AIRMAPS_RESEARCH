"""
pipeline_app.py
---------------
Streamlit dashboard for the ISRAP Air Quality Analysis Pipeline.

7 tabs:
  1. Executive Overview
  2. Data Quality
  3. Anomaly Detection
  4. Imputation
  5. Method Comparison
  6. Methodology Guide
  7. Station SO2 Investigation (Station 3)

Run with:
    streamlit run pipeline_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Bootstrap src/ imports
# ---------------------------------------------------------------------------

PIPELINE_ROOT = Path(__file__).parent.resolve()
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

import yaml

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ISRAP Air Quality Pipeline",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = PIPELINE_ROOT / "config.yaml"


@st.cache_data
def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    return {}


CONFIG = load_config()
OUTPUT_DIR = PIPELINE_ROOT / CONFIG.get("output_dir", "outputs")
POLLUTANTS = CONFIG.get("pollutants", ["CO", "NO2", "O3", "PM2.5", "SO2"])
ERA5_FEATURES = CONFIG.get("era5_features",
                           ["temp_c", "wind_speed", "blh", "relative_humidity",
                            "surface_pressure_hpa", "precip_mm"])

POLLUTANT_COLORS = {
    "CO": "#e41a1c",
    "NO2": "#377eb8",
    "O3": "#4daf4a",
    "PM2.5": "#984ea3",
    "SO2": "#ff7f00",
}

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_era5() -> pd.DataFrame:
    from src.data_loader import load_era5_data
    return load_era5_data(CONFIG)


@st.cache_data(show_spinner=False)
def load_all_raw() -> dict:
    from src.data_loader import load_all_stations
    raw = load_all_stations(CONFIG)
    return {sid: df for sid, (df, _) in raw.items()}


@st.cache_data(show_spinner=False)
def load_imputed_csv(site_id: int) -> pd.DataFrame:
    path = OUTPUT_DIR / "imputed" / f"site{site_id}_imputed.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
        return df
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_audit_csv(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / "audits" / name
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_anomaly_summary() -> pd.DataFrame:
    path = OUTPUT_DIR / "anomaly_flags" / "anomaly_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_imputation_summary() -> pd.DataFrame:
    path = OUTPUT_DIR / "imputed" / "imputation_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_validation_leaderboard() -> pd.DataFrame:
    path = OUTPUT_DIR / "validation" / "validation_leaderboard.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def outputs_exist() -> bool:
    return (OUTPUT_DIR / "audits" / "dataset_audit.csv").exists()


def run_pipeline_on_demand() -> None:
    """Run the pipeline from within Streamlit (fallback)."""
    import subprocess
    python = sys.executable
    cmd = [python, str(PIPELINE_ROOT / "run_pipeline.py")]
    with st.spinner("Running pipeline... This may take several minutes."):
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PIPELINE_ROOT))
    if result.returncode == 0:
        st.success("Pipeline completed successfully!")
        st.cache_data.clear()
        st.rerun()
    else:
        st.error("Pipeline failed. See details below.")
        st.code(result.stderr[-3000:] if result.stderr else "No stderr")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.image("https://www.nasa.gov/wp-content/uploads/2023/03/nasa-logo-web-rgb.png",
                 width=100)
        st.markdown("## ISRAP Air Quality Pipeline")
        st.markdown("**San Antonio, TX** | EPA AQS + ERA5")
        st.divider()

        st.markdown("### Pipeline Status")
        if outputs_exist():
            st.success("✅ Outputs available")
        else:
            st.warning("⚠️ No outputs found")
            if st.button("▶ Run Pipeline Now"):
                run_pipeline_on_demand()

        st.divider()
        audit = load_audit_csv("dataset_audit.csv")
        if not audit.empty:
            st.markdown(f"**Stations:** {len(audit)}")
            for _, row in audit.iterrows():
                avail = row.get("available_pollutants", "")
                st.markdown(f"- Site **{row['site_id']}**: {avail}")
        st.divider()
        st.markdown("*Research pipeline v1.0*")


# ---------------------------------------------------------------------------
# Tab 1: Executive Overview
# ---------------------------------------------------------------------------

def tab_executive_overview():
    st.header("Executive Overview")

    audit = load_audit_csv("dataset_audit.csv")
    avail = load_audit_csv("station_pollutant_availability.csv")
    neg = load_audit_csv("negative_value_summary.csv")
    anom = load_anomaly_summary()
    imp = load_imputation_summary()

    if audit.empty:
        st.info("No pipeline outputs found. Use the sidebar to run the pipeline.")
        return

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stations", len(audit))
    col2.metric("Total Raw Rows", f"{audit['n_rows_raw'].sum():,}" if "n_rows_raw" in audit else "—")
    col3.metric(
        "Consensus Anomalies",
        f"{anom['n_consensus'].sum():,}" if not anom.empty and "n_consensus" in anom else "—",
    )
    col4.metric(
        "Total Imputed Points",
        f"{imp['total_imputed'].sum():,}" if not imp.empty and "total_imputed" in imp else "—",
    )

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Station Overview")
        st.dataframe(audit, use_container_width=True)
    with c2:
        st.subheader("Data Availability (% non-NaN)")
        if not avail.empty:
            idx_col = avail.columns[0]
            avail_display = avail.set_index(idx_col) if idx_col != avail.index.name else avail
            st.dataframe(avail_display.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=100),
                         use_container_width=True)

    st.divider()
    st.subheader("Critical Finding: Station 3 SO2")
    st.error(
        "**CRITICAL:** Site 59 (Station 3) SO2 has ~33.6% negative values "
        "(systematic zero-offset drift or EPA AQS export sign error). "
        "All negative values have been set to NaN. "
        "See Tab 7 for the dedicated investigation."
    )
    st.warning(
        "**HIGH:** Site 52 (Station 2) NO2 has ~8.7% negative values. "
        "Flagged and cleaned by physical constraint step."
    )

    # Negative value chart
    if not neg.empty:
        neg_nonzero = neg[neg["pct_negative"] > 0]
        if not neg_nonzero.empty:
            st.subheader("Negative Value Summary")
            st.dataframe(neg_nonzero.sort_values("pct_negative", ascending=False),
                         use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2: Data Quality
# ---------------------------------------------------------------------------

def tab_data_quality():
    st.header("Data Quality")

    audit = load_audit_csv("dataset_audit.csv")
    if audit.empty:
        st.info("Run the pipeline first to generate outputs.")
        return

    site_options = sorted(audit["site_id"].tolist())
    col1, col2 = st.columns(2)
    selected_site = col1.selectbox("Station (Site ID)", site_options)
    selected_pol = col2.selectbox("Pollutant", POLLUTANTS)

    df = load_imputed_csv(selected_site)

    # Missingness heatmap
    heatmap_path = OUTPUT_DIR / "figures" / "missingness_heatmap.png"
    if heatmap_path.exists():
        st.subheader("Station × Pollutant Data Availability")
        st.image(str(heatmap_path))

    st.divider()
    st.subheader(f"Site {selected_site} – {selected_pol} Raw Time Series")

    if df.empty:
        st.warning("Imputed data not found. Run the pipeline first.")
        return

    orig_col = f"{selected_pol}_original"
    if orig_col in df.columns:
        plot_series = df[orig_col]
        plot_label = f"{selected_pol} (original)"
    elif selected_pol in df.columns:
        plot_series = df[selected_pol]
        plot_label = selected_pol
    else:
        st.info(f"{selected_pol} not available for Site {selected_site}.")
        return

    plot_data = plot_series.dropna()
    if plot_data.empty:
        st.info("No data available.")
        return

    st.line_chart(plot_data.rename(plot_label))

    # Gap summary
    st.subheader("Gap Summary")
    gap_df = load_audit_csv("missing_gap_summary.csv")
    if not gap_df.empty:
        site_gap = gap_df[gap_df.get("station_id", pd.Series(dtype=int)) == selected_site]
        if not site_gap.empty:
            st.dataframe(site_gap[site_gap["column"] == selected_pol], use_container_width=True)

    # Saved anomaly figure
    anom_fig = OUTPUT_DIR / "figures" / f"site{selected_site}_{selected_pol}_anomalies.png"
    if anom_fig.exists():
        st.subheader("Time Series with Anomaly Flags")
        st.image(str(anom_fig))


# ---------------------------------------------------------------------------
# Tab 3: Anomaly Detection
# ---------------------------------------------------------------------------

def tab_anomaly_detection():
    st.header("Anomaly Detection")

    audit = load_audit_csv("dataset_audit.csv")
    if audit.empty:
        st.info("Run the pipeline first.")
        return

    site_options = sorted(audit["site_id"].tolist())
    col1, col2 = st.columns(2)
    selected_site = col1.selectbox("Station", site_options, key="anom_site")
    selected_pol = col2.selectbox("Pollutant", POLLUTANTS, key="anom_pol")

    anom_df = load_anomaly_summary()
    if not anom_df.empty:
        site_anom = anom_df[anom_df["site_id"] == selected_site]
        pol_anom = site_anom[site_anom["pollutant"] == selected_pol]
        if not pol_anom.empty:
            row = pol_anom.iloc[0]
            st.subheader(f"Site {selected_site} – {selected_pol} Anomaly Summary")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("IQR", int(row.get("n_flag_iqr", 0)))
            m2.metric("Hampel", int(row.get("n_flag_hampel", 0)))
            m3.metric("Rolling Z", int(row.get("n_flag_rolling_zscore", 0)))
            m4.metric("STL", int(row.get("n_flag_stl", 0)))
            m5.metric("IsoForest", int(row.get("n_flag_isolation_forest", 0)))
            m6.metric("Consensus (≥2)", int(row.get("n_consensus", 0)))

    # Method comparison figure
    comp_fig = OUTPUT_DIR / "figures" / f"site{selected_site}_{selected_pol}_method_comparison.png"
    if comp_fig.exists():
        st.image(str(comp_fig))

    # Anomaly timeline figure
    anom_ts_fig = OUTPUT_DIR / "figures" / f"site{selected_site}_{selected_pol}_anomalies.png"
    if anom_ts_fig.exists():
        st.subheader("Anomaly Timeline")
        st.image(str(anom_ts_fig))

    # Interactive: load imputed data and show flagged points
    df = load_imputed_csv(selected_site)
    if df.empty:
        return

    consensus_col = f"anomaly_consensus_flag_{selected_pol}"
    if consensus_col not in df.columns:
        st.info("Anomaly flags not found in imputed output – run the pipeline first.")
        return

    flagged = df[df[consensus_col] == True]
    if not flagged.empty and selected_pol in df.columns:
        st.subheader("Flagged Rows (consensus ≥ 2 methods)")
        st.dataframe(
            flagged[[selected_pol, consensus_col,
                      f"anomaly_vote_count_{selected_pol}",
                      f"anomaly_severity_{selected_pol}"]]
            .head(200),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Tab 4: Imputation
# ---------------------------------------------------------------------------

def tab_imputation():
    st.header("Imputation")

    audit = load_audit_csv("dataset_audit.csv")
    if audit.empty:
        st.info("Run the pipeline first.")
        return

    site_options = sorted(audit["site_id"].tolist())
    col1, col2 = st.columns(2)
    selected_site = col1.selectbox("Station", site_options, key="imp_site")
    selected_pol = col2.selectbox("Pollutant", POLLUTANTS, key="imp_pol")

    # Before/after figure
    ba_fig = OUTPUT_DIR / "figures" / f"site{selected_site}_{selected_pol}_imputation.png"
    if ba_fig.exists():
        st.subheader("Before / After Imputation")
        st.image(str(ba_fig))
    else:
        st.info("Imputation figure not found – run the pipeline first.")

    # Imputation summary table
    imp_df = load_imputation_summary()
    if not imp_df.empty:
        site_imp = imp_df[(imp_df["site_id"] == selected_site) &
                          (imp_df["pollutant"] == selected_pol)]
        if not site_imp.empty:
            row = site_imp.iloc[0]
            st.subheader("Imputation Breakdown")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Short (≤2h) interpolated", int(row.get("n_short_interp", 0)))
            m2.metric("Medium (3–24h) Ridge", int(row.get("n_medium_ridge", 0)))
            m3.metric("Long (>24h) seasonal", int(row.get("n_long_seasonal", 0)))
            m4.metric("Still missing", int(row.get("n_still_missing", 0)))

    # Show data
    df = load_imputed_csv(selected_site)
    if df.empty:
        return

    method_col = f"{selected_pol}_imputation_method"
    if method_col in df.columns:
        method_counts = df[method_col].value_counts()
        st.subheader("Imputation Method Distribution")
        st.bar_chart(method_counts)


# ---------------------------------------------------------------------------
# Tab 5: Method Comparison
# ---------------------------------------------------------------------------

def tab_method_comparison():
    st.header("Method Comparison – Validation Leaderboard")

    leaderboard = load_validation_leaderboard()
    if leaderboard.empty:
        st.info("Validation leaderboard not found – run the pipeline first.")
        return

    st.subheader("Overall RMSE Leaderboard (lower = better)")
    st.dataframe(leaderboard.sort_values("mean_RMSE"), use_container_width=True)

    # Validation comparison figure
    val_fig = OUTPUT_DIR / "figures" / "validation_rmse_comparison.png"
    if val_fig.exists():
        st.image(str(val_fig))

    # Per-pollutant breakdown
    st.subheader("Filter by Pollutant")
    pol_filter = st.selectbox("Pollutant", ["All"] + POLLUTANTS, key="val_pol")
    if pol_filter != "All":
        st.dataframe(
            leaderboard[leaderboard["col"] == pol_filter].sort_values("mean_RMSE"),
            use_container_width=True,
        )

    # Error heatmap (mean_RMSE pivot)
    try:
        pivot = leaderboard.pivot_table(index="col", columns="mask", values="mean_RMSE")
        st.subheader("RMSE Heatmap (Pollutant × Mask Scenario)")
        st.dataframe(
            pivot.style.background_gradient(cmap="YlOrRd"),
            use_container_width=True,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tab 6: Methodology Guide
# ---------------------------------------------------------------------------

def tab_methodology_guide():
    st.header("Methodology Guide")

    st.markdown("""
## Pipeline Architecture

The pipeline implements a six-stage tiered approach recommended in the
[ISRAP_AIRMAPS_Research_Analysis.docx](../ISRAP_AIRMAPS_Research_Analysis.docx):

---

### Stage 1 – Physical Constraint Cleaning
All pollutant concentrations (CO, NO2, O3, PM2.5, SO2) are non-negative by physical law.
Negative readings arise from instrument zero-drift or calibration failure.
**Action:** Set all values < 0 → NaN and record in `physical_invalid_flag_<pol>`.

---

### Stage 2 – IQR Fence (Anomaly)
**Multiplier:** 3.0 (deliberately conservative to avoid flagging real pollution episodes).
**Flag:** `flag_iqr_<pol>` — points outside [Q1 − 3·IQR, Q3 + 3·IQR].

---

### Stage 3 – Hampel Filter (Anomaly)
Sliding median with MAD-based sigma estimate (σ̂ = 1.4826 · MAD).
**Window:** 24 hours (daily cycle), **threshold:** 3σ.
**Flag:** `flag_hampel_<pol>`.

---

### Stage 4 – Rolling Z-Score (Anomaly)
Rolling mean ± std with a wide window to capture episodic pollution spikes.
**Window:** 168 hours (7-day), **threshold:** 3.5σ.
**Flag:** `flag_rolling_zscore_<pol>`.

---

### Stage 5 – STL Residual (Anomaly)
Seasonal-Trend decomposition via Loess (statsmodels STL, period=24h, robust=True).
Flag residuals outside the 3× IQR fence of the residual distribution.
**Flag:** `flag_stl_<pol>`.

---

### Stage 6 – Isolation Forest (Anomaly)
Multivariate, unsupervised. Features: pollutant + ERA5 meteorological covariates.
**Contamination:** 5%, **n_estimators:** 100, **seed:** 42.
**Flag:** `flag_isolation_forest_<pol>`, **Score:** `anomaly_score_<pol>`.

---

### Consensus Voting
A row is a *consensus anomaly* when ≥ 2 of the 5 methods agree.
**Columns:** `anomaly_vote_count_<pol>`, `anomaly_consensus_flag_<pol>`,
`anomaly_severity_<pol>` (none / low / medium / high).

---

### Imputation Tiers

| Tier | Gap Length | Method |
|------|-----------|--------|
| 1 | ≤ 2 hours | Linear time interpolation |
| 2 | 3–24 hours | Ridge regression (ERA5 + cyclical time features) |
| 3 | > 24 hours | Seasonal median (month × hour-of-day) |

All imputed values clipped to ≥ 0.

---

### Validation Strategy
Ground-truth values are artificially masked (5%, 10%, 20% random + short/medium/long blocks),
imputation is run on the masked series, and predictions are compared to held-out truth.
**Metrics:** MAE, RMSE, R², Bias.

---

### Key References
- Liu et al. (ICDM 2008) – Isolation Forest
- van Buuren & Groothuis-Oudshoorn (JSS 2011) – MICE
- Khayati et al. (PVLDB 2020) – *Mind the Gap* (imputation benchmark)
- Schmidl et al. (PVLDB 2022) – Anomaly Detection Comprehensive Evaluation
- Zhang et al. (PVLDB 2017) – Time Series Data Cleaning
    """)


# ---------------------------------------------------------------------------
# Tab 7: Station SO2 Investigation
# ---------------------------------------------------------------------------

def tab_so2_investigation():
    st.header("Station 3 (Site 59) SO2 Investigation")

    st.error(
        "**CRITICAL:** Out of ~48,870 recorded SO2 values at Site 59, "
        "approximately **16,416 (33.6%) are negative**. "
        "This indicates systematic zero-offset drift or a sign error in the EPA AQS export."
    )

    st.markdown("""
### What We Know
- **Site 59 (Station 3)** is one of the primary stations in the San Antonio AQS network.
- The negative SO2 values are **not random** — they cluster in specific time windows,
  consistent with a systematic sensor calibration error rather than noise.
- **Station 2 (Site 52) NO2** also has 8.7% negative values (distinct but related issue).

### Actions Taken
1. Physical constraint cleaning set all negative SO2 values at Site 59 → NaN.
2. Flag column `physical_invalid_flag_SO2` records every affected timestamp.
3. SO2 at Site 59 is treated as **unreliable** and excluded from comparative analysis.
4. Imputation is skipped for this station-pollutant pair if the missingness exceeds 95%.

### Recommended Follow-up
1. Download raw QC-flagged data from EPA AQS Data Mart for AQS ID **48-029-0059**.
2. Check `QC Code` columns for CC (Calibration Check) and NF (No Flow) events.
3. Compare against nearby stations (Site 32, Site 52) for temporal correlation.
4. If the negatives are a systematic sign flip, consider multiplying by −1 for that period.
    """)

    # Load and display the SO2 data for Site 59
    df = load_imputed_csv(59)
    if df.empty:
        st.info("No imputed data found for Site 59 – run the pipeline first.")
        return

    flag_col = "physical_invalid_flag_SO2"
    if flag_col in df.columns:
        n_flagged = int(df[flag_col].sum()) if df[flag_col].dtype == bool else int(df[flag_col].astype(bool).sum())
        st.metric("Physical constraint flags (SO2 < 0 at Site 59)", n_flagged)

    so2_orig = f"SO2_original"
    if so2_orig in df.columns:
        st.subheader("Raw SO2 Distribution (including negatives)")
        raw_vals = df[so2_orig].dropna()
        if not raw_vals.empty:
            col1, col2 = st.columns(2)
            col1.metric("Min SO2", f"{raw_vals.min():.2f}")
            col1.metric("Max SO2", f"{raw_vals.max():.2f}")
            col2.metric("Median SO2", f"{raw_vals.median():.2f}")
            col2.metric("% Negative", f"{(raw_vals < 0).mean() * 100:.1f}%")

            # Histogram
            import matplotlib.pyplot as plt
            import io
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.hist(raw_vals.clip(-50, 50), bins=100, color="#ff7f00", alpha=0.7)
            ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero")
            ax.set_xlabel("SO2 value")
            ax.set_ylabel("Count")
            ax.set_title("Site 59 SO2 Value Distribution (clipped to ±50 for display)")
            ax.legend()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            st.image(buf)

    # Saved STL figure for SO2
    stl_fig = OUTPUT_DIR / "figures" / "site59_SO2_stl.png"
    if stl_fig.exists():
        st.subheader("STL Decomposition")
        st.image(str(stl_fig))

    # Timeline of SO2 before/after
    ba_fig = OUTPUT_DIR / "figures" / "site59_SO2_imputation.png"
    if ba_fig.exists():
        st.subheader("Before / After Physical Constraint Cleaning")
        st.image(str(ba_fig))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    render_sidebar()

    tab_labels = [
        "Executive Overview",
        "Data Quality",
        "Anomaly Detection",
        "Imputation",
        "Method Comparison",
        "Methodology Guide",
        "SO2 Investigation",
    ]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        tab_executive_overview()
    with tabs[1]:
        tab_data_quality()
    with tabs[2]:
        tab_anomaly_detection()
    with tabs[3]:
        tab_imputation()
    with tabs[4]:
        tab_method_comparison()
    with tabs[5]:
        tab_methodology_guide()
    with tabs[6]:
        tab_so2_investigation()


if __name__ == "__main__":
    main()

