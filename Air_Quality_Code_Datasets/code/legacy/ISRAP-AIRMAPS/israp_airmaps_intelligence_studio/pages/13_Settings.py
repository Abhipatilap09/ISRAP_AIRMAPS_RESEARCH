"""Settings — theme, thresholds, paths, dependency status, config save/load."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from src.config import (
    PROJECT_ROOT, ANOMALY_DEFAULTS, IMPUTATION_DEFAULTS,
    RANDOM_SEED, CHART_DOWNSAMPLE, DISPLAY_MAX_ROWS,
    STATION_DIR, ERA5_FILE,
)
from ui.theme import COLORS, inject_css, info_box, badge

SAVED_CONFIGS_DIR = PROJECT_ROOT / "saved_configs"
SAVED_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)


# ── Dependency checker ─────────────────────────────────────────────────────────

_DEPS = {
    "pandas":       ("Core data handling", True),
    "numpy":        ("Numerical computation", True),
    "scipy":        ("STL, statistics", True),
    "sklearn":      ("ML anomaly detection & imputation", True),
    "statsmodels":  ("Time-series decomposition", True),
    "plotly":       ("Interactive charts", True),
    "streamlit":    ("Dashboard framework", True),
    "pyarrow":      ("Parquet I/O", True),
    "joblib":       ("Model serialisation", True),
    "torch":        ("Deep learning (LSTM, USAD, BRITS, SAITS, CSDI)", False),
    "xgboost":      ("XGBoost imputation model", False),
    "pypots":       ("SAITS imputation (PyPOTS)", False),
    "psutil":       ("Memory profiling", False),
    "fpdf":         ("PDF report generation", False),
    "python_pptx":  ("PowerPoint report export", False),
    "openpyxl":     ("Excel export", False),
}


def _check_dep(pkg: str) -> tuple[bool, str]:
    """Return (available, version_string)."""
    real_pkg = pkg.replace("_", "-").replace("-", "_")
    for name in (pkg, real_pkg, pkg.replace("_", "")):
        try:
            mod = importlib.import_module(name)
            v = getattr(mod, "__version__", "?")
            return True, v
        except ImportError:
            pass
    return False, ""


def _load_saved_config(name: str) -> dict:
    p = SAVED_CONFIGS_DIR / f"{name}.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _save_config(name: str, data: dict) -> None:
    p = SAVED_CONFIGS_DIR / f"{name}.json"
    p.write_text(json.dumps(data, indent=2))


def main():
    st.set_page_config(
        page_title="Settings | ISRAP AIRMAPS", layout="wide", page_icon="⚙️"
    )
    inject_css()

    st.markdown(
        f"<h1 style='color:{COLORS['navy']};'>⚙️ Settings</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Configure default parameters, check dependency status, manage saved configurations.")

    tab_anomaly, tab_imputation, tab_display, tab_paths, tab_deps, tab_configs = st.tabs([
        "Anomaly Defaults",
        "Imputation Defaults",
        "Display & Performance",
        "Paths & Data",
        "Dependency Status",
        "Saved Configurations",
    ])

    # ── Tab 1: Anomaly Defaults ────────────────────────────────────────────────
    with tab_anomaly:
        st.markdown(
            '<div class="section-header">Default Anomaly Detection Parameters</div>',
            unsafe_allow_html=True,
        )
        info_box(
            "These are the default parameter values used when a page first loads. "
            "Changes here do not affect already-running experiments. "
            "Save to persist across browser sessions.",
            kind="info",
        )

        saved_ad = _load_saved_config("anomaly_defaults")
        ad = {**ANOMALY_DEFAULTS, **saved_ad}

        c1, c2, c3 = st.columns(3)
        new_ad = {}

        with c1:
            st.markdown("**IQR Fence**")
            new_ad["iqr_multiplier"] = st.slider(
                "IQR Multiplier (k)", 1.0, 5.0, float(ad["iqr_multiplier"]), 0.5,
                help="Flag values outside Q1 - k*IQR and Q3 + k*IQR. Typical: 2.5-3.5"
            )
            st.markdown("**Hampel Filter**")
            new_ad["hampel_window"] = st.slider(
                "Hampel Window (hours)", 6, 168, int(ad["hampel_window"]), 6,
                help="Rolling window for Hampel median filter. 24h is a common choice."
            )
            new_ad["hampel_threshold"] = st.slider(
                "Hampel Threshold (MAD units)", 1.0, 6.0, float(ad["hampel_threshold"]), 0.5,
                help="Number of scaled MAD units to flag. 3.0 = 99.7% of normal distribution."
            )

        with c2:
            st.markdown("**Rolling Z-Score**")
            new_ad["rolling_zscore_window"] = st.slider(
                "Rolling Z Window (hours)", 12, 336, int(ad["rolling_zscore_window"]), 12,
                help="How many past hours to use for rolling mean/std. 72h is robust for daily cycles."
            )
            new_ad["rolling_zscore_threshold"] = st.slider(
                "Rolling Z Threshold", 1.5, 5.0, float(ad["rolling_zscore_threshold"]), 0.25,
                help="Flag if |z-score| > this value. 3.0 = 0.27% false positive rate under normality."
            )
            st.markdown("**STL Residual**")
            new_ad["stl_seasonal"] = st.slider(
                "STL Seasonal Period", 13, 49, int(ad["stl_seasonal"]), 2,
                help="Seasonal period for STL. 24 for diurnal cycle; 25 is the STL minimum odd value."
            )
            new_ad["stl_threshold"] = st.slider(
                "STL Residual Threshold", 1.5, 5.0, float(ad.get("stl_threshold", 3.0)), 0.25,
                help="Sigma multiplier for residual flagging."
            )

        with c3:
            st.markdown("**Isolation Forest**")
            new_ad["iforest_contamination"] = st.slider(
                "IF Contamination", 0.001, 0.10, float(ad["iforest_contamination"]), 0.005,
                format="%.3f",
                help="Expected proportion of anomalies. 0.01 = 1% contamination (reasonable for air quality)."
            )
            new_ad["iforest_n_estimators"] = st.slider(
                "IF Trees", 50, 500, int(ad["iforest_n_estimators"]), 50,
                help="Number of isolation trees. More = more stable but slower."
            )
            new_ad["iforest_random_state"] = st.number_input(
                "IF Random State", 0, 9999, int(ad.get("iforest_random_state", 42)),
                help="Random seed for reproducibility."
            )
            st.markdown("**SCREEN Rate-of-Change**")
            new_ad["screen_percentile"] = st.slider(
                "SCREEN Threshold Percentile", 90, 99, int(ad["screen_percentile"]), 1,
                help="Training-period percentile used to derive the speed limit."
            )

        col_l, col_r = st.columns([1, 3])
        with col_l:
            if st.button("Save Anomaly Defaults", type="primary"):
                _save_config("anomaly_defaults", new_ad)
                st.success("Anomaly defaults saved.")
        with col_r:
            if st.button("Reset to Built-in Defaults", key="reset_ad"):
                (SAVED_CONFIGS_DIR / "anomaly_defaults.json").unlink(missing_ok=True)
                st.success("Reset to built-in defaults.")
                st.rerun()

    # ── Tab 2: Imputation Defaults ─────────────────────────────────────────────
    with tab_imputation:
        st.markdown(
            '<div class="section-header">Default Imputation Parameters</div>',
            unsafe_allow_html=True,
        )

        saved_id = _load_saved_config("imputation_defaults")
        imd = {**IMPUTATION_DEFAULTS, **saved_id}

        c1, c2, c3 = st.columns(3)
        new_id = {}

        with c1:
            st.markdown("**Linear Interpolation**")
            new_id["linear_max_gap"] = st.slider(
                "Linear: Max Gap (hours)", 1, 24, int(imd["linear_max_gap"]),
                help="Maximum gap length to fill with linear interpolation. Beyond this, a smarter method is used."
            )

            st.markdown("**ERA5 Regression**")
            new_id["era5_model"] = st.selectbox(
                "ERA5 Default Model",
                ["random_forest", "ridge", "hist_gradient_boosting"],
                index=["random_forest", "ridge", "hist_gradient_boosting"].index(
                    imd.get("era5_model", "random_forest")
                ),
                help="Model used for ERA5-assisted regression imputation."
            )
            new_id["era5_n_lags"] = st.multiselect(
                "ERA5 Lag Hours",
                [1, 2, 3, 6, 12, 24, 48, 72],
                default=imd.get("era5_n_lags", [1, 2, 3, 6, 12, 24]),
                help="Which past-hour lags to include as regression features."
            )

        with c2:
            st.markdown("**MICE**")
            new_id["mice_max_iter"] = st.slider(
                "MICE Max Iterations", 3, 30, int(imd["mice_max_iter"]),
                help="Convergence iterations for iterative imputation."
            )
            new_id["mice_random_state"] = st.number_input(
                "MICE Random State", 0, 9999, int(imd["mice_random_state"])
            )

            st.markdown("**BRITS (PyTorch)**")
            new_id["brits_hidden"] = st.slider(
                "BRITS Hidden Units", 16, 256, int(imd["brits_hidden"]), 16
            )
            new_id["brits_epochs_fast"] = st.slider(
                "BRITS Epochs (Fast Mode)", 2, 20, int(imd["brits_epochs_fast"])
            )
            new_id["brits_epochs_full"] = st.slider(
                "BRITS Epochs (Full Mode)", 10, 100, int(imd["brits_epochs_full"])
            )

        with c3:
            st.markdown("**SAITS (PyTorch)**")
            new_id["saits_d_model"] = st.slider(
                "SAITS d_model", 16, 256, int(imd["saits_d_model"]), 16
            )
            new_id["saits_epochs_fast"] = st.slider(
                "SAITS Epochs (Fast Mode)", 2, 20, int(imd["saits_epochs_fast"])
            )
            new_id["saits_epochs_full"] = st.slider(
                "SAITS Epochs (Full Mode)", 10, 100, int(imd["saits_epochs_full"])
            )

            st.markdown("**Global**")
            new_id["random_seed"] = st.number_input(
                "Global Random Seed", 0, 99999, RANDOM_SEED
            )
            new_id["gap_aware_strategy"] = st.selectbox(
                "Default Gap-Aware Strategy",
                ["Auto (recommended)", "Conservative (linear-first)", "Aggressive (deep-first)"],
                help="How the automatic strategy picker balances method choice vs. data availability."
            )

        c_l, c_r = st.columns([1, 3])
        with c_l:
            if st.button("Save Imputation Defaults", type="primary"):
                _save_config("imputation_defaults", new_id)
                st.success("Imputation defaults saved.")
        with c_r:
            if st.button("Reset to Built-in Defaults", key="reset_id"):
                (SAVED_CONFIGS_DIR / "imputation_defaults.json").unlink(missing_ok=True)
                st.success("Reset to built-in defaults.")
                st.rerun()

    # ── Tab 3: Display & Performance ───────────────────────────────────────────
    with tab_display:
        st.markdown(
            '<div class="section-header">Display & Performance Settings</div>',
            unsafe_allow_html=True,
        )

        saved_disp = _load_saved_config("display_settings")
        disp = {
            "chart_downsample": CHART_DOWNSAMPLE,
            "table_max_rows": DISPLAY_MAX_ROWS,
            "theme_mode": "light",
            "default_chart_height": 450,
            **saved_disp,
        }

        c1, c2 = st.columns(2)
        new_disp = {}
        with c1:
            st.markdown("**Charts**")
            new_disp["chart_downsample"] = st.slider(
                "Max points before downsampling", 1000, 20000,
                int(disp["chart_downsample"]), 500,
                help="Charts with more than this many points are downsampled for performance."
            )
            new_disp["default_chart_height"] = st.slider(
                "Default chart height (px)", 300, 800,
                int(disp["default_chart_height"]), 50
            )
            new_disp["theme_mode"] = st.selectbox(
                "Theme Mode", ["light", "dark"],
                index=["light", "dark"].index(disp["theme_mode"])
            )

        with c2:
            st.markdown("**Tables**")
            new_disp["table_max_rows"] = st.slider(
                "Max rows in data tables", 50, 2000,
                int(disp["table_max_rows"]), 50,
                help="Larger values may cause slow rendering."
            )
            new_disp["show_debug_info"] = st.checkbox(
                "Show debug information in expanders",
                value=bool(disp.get("show_debug_info", False))
            )
            new_disp["auto_refresh"] = st.checkbox(
                "Auto-refresh data on parameter change",
                value=bool(disp.get("auto_refresh", False)),
                help="When disabled, you must click Run buttons manually. Strongly recommended for deep learning."
            )

        info_box(
            "Tip: Disabling auto-refresh prevents expensive methods from re-running on every widget change. "
            "This is the recommended mode for all deep learning methods.",
            kind="info",
        )

        if st.button("Save Display Settings", type="primary"):
            _save_config("display_settings", new_disp)
            st.success("Display settings saved.")

        st.markdown("---")
        st.markdown("**Cache Management**")
        c_a, c_b, c_c = st.columns(3)
        with c_a:
            if st.button("Clear Streamlit Cache"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Streamlit cache cleared.")
        with c_b:
            cache_dir = PROJECT_ROOT / "cache"
            cache_files = list(cache_dir.glob("**/*")) if cache_dir.exists() else []
            n_cache = len([f for f in cache_files if f.is_file()])
            st.metric("Cached files", n_cache)
        with c_c:
            model_dir = PROJECT_ROOT / "models"
            model_files = list(model_dir.glob("**/*")) if model_dir.exists() else []
            n_models = len([f for f in model_files if f.is_file()])
            st.metric("Saved model checkpoints", n_models)

    # ── Tab 4: Paths & Data ────────────────────────────────────────────────────
    with tab_paths:
        st.markdown(
            '<div class="section-header">Data Paths & File Status</div>',
            unsafe_allow_html=True,
        )

        paths = {
            "Project Root": PROJECT_ROOT,
            "Station CSV Directory": STATION_DIR,
            "ERA5 File": ERA5_FILE,
            "Models Directory": PROJECT_ROOT / "models",
            "Cache Directory": PROJECT_ROOT / "cache",
            "Exports Directory": PROJECT_ROOT / "exports",
            "Logs Directory": PROJECT_ROOT / "logs",
            "Experiments Database": PROJECT_ROOT / "experiments",
            "Saved Configs": SAVED_CONFIGS_DIR,
        }

        rows = []
        for label, p in paths.items():
            exists = p.exists()
            is_file = p.is_file() if exists else False
            size = ""
            if exists and is_file:
                sz = p.stat().st_size
                if sz > 1_000_000:
                    size = f"{sz/1_000_000:.1f} MB"
                elif sz > 1000:
                    size = f"{sz/1000:.0f} KB"
                else:
                    size = f"{sz} B"
            rows.append({
                "Label": label,
                "Path": str(p),
                "Exists": "✅" if exists else "❌",
                "Type": "File" if is_file else ("Dir" if exists else "Missing"),
                "Size": size,
            })

        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("**Station Files**")
        station_rows = []
        for f in sorted(STATION_DIR.glob("*.csv")) if STATION_DIR.exists() else []:
            sz = f.stat().st_size
            station_rows.append({
                "File": f.name,
                "Size": f"{sz/1_000_000:.2f} MB",
                "Status": "✅ Found",
            })
        if station_rows:
            st.dataframe(pd.DataFrame(station_rows), use_container_width=True, hide_index=True)
        else:
            st.error(f"No CSV files found in {STATION_DIR}")

    # ── Tab 5: Dependency Status ───────────────────────────────────────────────
    with tab_deps:
        st.markdown(
            '<div class="section-header">Package Dependency Status</div>',
            unsafe_allow_html=True,
        )

        info_box(
            "Required packages must be installed for core functionality. "
            "Optional packages enable deep learning methods and additional export formats. "
            "Install with: <code>pip install -r requirements.txt</code>",
            kind="info",
        )

        import pandas as pd
        dep_rows = []
        for pkg, (desc, required) in _DEPS.items():
            avail, version = _check_dep(pkg)
            dep_rows.append({
                "Package": pkg,
                "Description": desc,
                "Required": "Yes" if required else "Optional",
                "Status": "✅ Installed" if avail else "❌ Missing",
                "Version": version if avail else "—",
            })

        dep_df = pd.DataFrame(dep_rows)

        st.markdown("**Required Packages**")
        req_df = dep_df[dep_df["Required"] == "Yes"]
        st.dataframe(req_df.drop(columns=["Required"]), use_container_width=True, hide_index=True)

        missing_required = req_df[req_df["Status"].str.startswith("❌")]
        if not missing_required.empty:
            info_box(
                f"<b>Missing required packages:</b> {', '.join(missing_required['Package'].tolist())}. "
                "Run <code>pip install -r requirements.txt</code>",
                kind="danger",
            )

        st.markdown("**Optional Packages**")
        opt_df = dep_df[dep_df["Required"] == "Optional"]
        st.dataframe(opt_df.drop(columns=["Required"]), use_container_width=True, hide_index=True)

        missing_opt = opt_df[opt_df["Status"].str.startswith("❌")]
        if not missing_opt.empty:
            info_box(
                f"Optional packages not installed: {', '.join(missing_opt['Package'].tolist())}. "
                "Install from <code>requirements-optional.txt</code> to enable those features. "
                "The app runs without them.",
                kind="warn",
            )

        st.markdown("**Python Environment**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Python Version", sys.version.split()[0])
        req_installed = int(((dep_df['Required'] == 'Yes') & dep_df['Status'].str.startswith('✅')).sum())
        req_total = int((dep_df['Required'] == 'Yes').sum())
        opt_installed = int(((dep_df['Required'] == 'Optional') & dep_df['Status'].str.startswith('✅')).sum())
        opt_total = int((dep_df['Required'] == 'Optional').sum())
        c2.metric("Required installed", f"{req_installed} / {req_total}")
        c3.metric("Optional installed", f"{opt_installed} / {opt_total}")

    # ── Tab 6: Saved Configurations ────────────────────────────────────────────
    with tab_configs:
        st.markdown(
            '<div class="section-header">Saved Configurations</div>',
            unsafe_allow_html=True,
        )

        config_files = sorted(SAVED_CONFIGS_DIR.glob("*.json"))

        if not config_files:
            st.info("No saved configurations found. Use the other tabs to save default parameters.")
        else:
            st.write(f"{len(config_files)} saved configuration(s):")
            for cf in config_files:
                with st.expander(f"📄 {cf.stem}", expanded=False):
                    try:
                        content = json.loads(cf.read_text())
                        st.json(content)
                        col_a, col_b = st.columns([1, 4])
                        with col_a:
                            if st.button(f"Delete {cf.stem}", key=f"del_{cf.stem}"):
                                cf.unlink()
                                st.success(f"Deleted {cf.stem}.")
                                st.rerun()
                        with col_b:
                            st.download_button(
                                f"Download {cf.stem}.json",
                                cf.read_bytes(),
                                f"{cf.stem}.json",
                                mime="application/json",
                                key=f"dl_{cf.stem}",
                            )
                    except Exception as e:
                        st.error(f"Could not read {cf.name}: {e}")

        st.markdown("---")
        st.markdown("**Import Configuration**")
        uploaded = st.file_uploader("Upload a JSON configuration file", type=["json"])
        if uploaded is not None:
            try:
                content = json.loads(uploaded.read())
                name = Path(uploaded.name).stem
                _save_config(name, content)
                st.success(f"Imported configuration as '{name}'.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not parse uploaded file: {e}")

        st.markdown("---")
        st.markdown("**Export All Settings**")
        all_settings = {}
        for cf in config_files:
            try:
                all_settings[cf.stem] = json.loads(cf.read_text())
            except Exception:
                pass

        if all_settings:
            st.download_button(
                "Download All Settings (JSON)",
                json.dumps(all_settings, indent=2).encode(),
                "israp_airmaps_all_settings.json",
                mime="application/json",
            )

        st.markdown("---")
        st.markdown("**Reset Everything**")
        info_box(
            "This will delete ALL saved configurations and reset to built-in defaults. "
            "It will NOT delete experiment history or model checkpoints.",
            kind="warn",
        )
        confirm_reset = st.checkbox("I confirm: delete all saved configurations")
        if st.button("Reset All Settings", type="secondary") and confirm_reset:
            for cf in SAVED_CONFIGS_DIR.glob("*.json"):
                cf.unlink()
            st.success("All saved configurations deleted. Built-in defaults will be used.")
            st.rerun()


if __name__ == "__main__":
    main()

