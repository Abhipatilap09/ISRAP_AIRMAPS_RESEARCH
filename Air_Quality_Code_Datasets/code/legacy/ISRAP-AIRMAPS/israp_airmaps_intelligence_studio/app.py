"""ISRAP AIRMAPS Intelligence Studio — main entry point.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── ensure src/ and project root are importable ───────────────────────────────
_ROOT = Path(__file__).parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from ui.theme import inject_css, COLORS


def _configure_page() -> None:
    st.set_page_config(
        page_title="ISRAP AIRMAPS Intelligence Studio",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": "ISRAP AIRMAPS Intelligence Studio — NASA/ISRAP Air Quality Research",
        },
    )


def _check_deps() -> list[str]:
    missing = []
    try:
        import pandas  # noqa: F401
    except ImportError:
        missing.append("pandas")
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")
    try:
        import plotly  # noqa: F401
    except ImportError:
        missing.append("plotly")
    try:
        import sklearn  # noqa: F401
    except ImportError:
        missing.append("scikit-learn")
    return missing


def main() -> None:
    _configure_page()
    inject_css()

    missing_deps = _check_deps()
    if missing_deps:
        st.error(f"Missing required dependencies: {', '.join(missing_deps)}")
        st.code("pip install -r requirements.txt")
        return

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 16px 0 8px 0;'>
            <div style='font-size:2rem;'>🌍</div>
            <div style='font-size:1rem; font-weight:700; color:#fff;'>ISRAP AIRMAPS</div>
            <div style='font-size:0.72rem; color:#94d2bd; margin-top:4px;'>Intelligence Studio</div>
        </div>
        <hr style='border-color:#1e3a5f; margin:8px 0;'>
        """, unsafe_allow_html=True)

        st.markdown("**Navigation**")
        st.page_link("app.py", label="Home / Overview", icon="🏠")
        st.page_link("pages/01_Executive_Overview.py", label="Executive Overview", icon="📊")
        st.page_link("pages/02_Data_Explorer.py", label="Data Explorer", icon="🔭")
        st.page_link("pages/03_Data_Quality.py", label="Data Quality", icon="🔍")
        st.page_link("pages/04_Anomaly_Lab.py", label="Anomaly Lab", icon="⚗️")
        st.page_link("pages/05_Anomaly_Comparison.py", label="Anomaly Comparison", icon="⚖️")
        st.page_link("pages/06_Imputation_Lab.py", label="Imputation Lab", icon="🔧")
        st.page_link("pages/07_Imputation_Benchmark.py", label="Imputation Benchmark", icon="📏")
        st.page_link("pages/08_Final_Pipeline.py", label="Final Pipeline", icon="🔄")
        st.page_link("pages/09_Station3_SO2.py", label="Station 3 SO₂ Investigation", icon="⚠️")
        st.page_link("pages/10_Method_Library.py", label="Method Library", icon="📚")
        st.page_link("pages/11_Experiment_History.py", label="Experiment History", icon="🕐")
        st.page_link("pages/12_Export_Center.py", label="Export Center", icon="📦")
        st.page_link("pages/13_Settings.py", label="Settings", icon="⚙️")

    # ── Landing page ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='text-align:center; padding: 40px 0 24px 0;'>
        <div style='font-size:3.5rem;'>🌍</div>
        <h1 style='color:{COLORS["navy"]}; margin:8px 0 4px 0;'>ISRAP AIRMAPS Intelligence Studio</h1>
        <p style='color:{COLORS["teal"]}; font-size:1.1rem; font-weight:600;'>
            Anomaly Detection, Data Quality &amp; Intelligent Imputation for Multi-Station Air Quality Data
        </p>
        <p style='color:{COLORS["muted"]}; max-width:700px; margin:12px auto;'>
            An interactive research platform for detecting sensor anomalies, analysing missing-data patterns,
            comparing statistical and deep-learning methods, and producing scientifically defensible imputed
            air-quality datasets using EPA AQS and ERA5 meteorological data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Get Started →** Use the sidebar to navigate to any page.\n\n"
                "Start with **Executive Overview** to see the network summary, "
                "or jump to **Anomaly Lab** to run detection.")
    with col2:
        st.success("**Phase 1 Complete:**\n\n"
                   "Data loaded ✓ | Hourly reindex ✓ | ERA5 merged ✓ | "
                   "Structural missingness detected ✓ | Gap analysis ✓")
    with col3:
        st.warning("**Station 3 SO₂ Warning:**\n\n"
                   "~33% of SO₂ observations at Station 3 (Site 59) are negative. "
                   "See the **Station 3 SO₂ Investigation** page.")


if __name__ == "__main__":
    main()

