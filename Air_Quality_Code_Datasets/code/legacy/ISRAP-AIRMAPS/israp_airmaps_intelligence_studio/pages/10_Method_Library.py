"""Method Library — reference guide for all methods."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from ui.theme import COLORS, inject_css, info_box
from services.interpretation_engine import recommend_anomaly_method, recommend_imputation_method


ANOMALY_METHODS_CATALOG = [
    {
        "name": "Physical Constraints",
        "category": "Physical",
        "badges": ["Statistical"],
        "desc": "Flags values that violate physical bounds (negative concentrations, infinite values, values above known maxima).",
        "math": "Flag if x < physical_min or x > physical_max.",
        "best_for": "First-pass validation — always run this before any other method.",
        "not_for": "Detecting statistically unusual but physically valid values.",
        "inputs": ["Pollutant value", "Physical bounds"],
        "params": ["physical_min", "physical_max"],
        "cost": "Negligible",
        "output": "is_physical_anomaly, is_negative, physical_anomaly_reason",
        "default_recommended": True,
    },
    {
        "name": "IQR Fence",
        "category": "Statistical",
        "badges": ["Statistical", "Robust"],
        "desc": "Computes Q1 and Q3 of the distribution. Flags values below Q1−k×IQR or above Q3+k×IQR.",
        "math": "Lower = Q1 − k×IQR; Upper = Q3 + k×IQR. Flag if x < Lower or x > Upper.",
        "best_for": "Isolated point spikes. Simple, fast, interpretable.",
        "not_for": "Contextual anomalies; seasonal data; slowly drifting sensors.",
        "inputs": ["Pollutant value"],
        "params": ["multiplier (default 3.0)"],
        "cost": "Negligible",
        "output": "is_iqr_anomaly, iqr_score, iqr_lower_bound, iqr_upper_bound",
    },
    {
        "name": "Hampel Filter",
        "category": "Robust Temporal",
        "badges": ["Robust", "Temporal"],
        "desc": "Rolling window of size W. For each point, computes rolling median and rolling MAD. "
                "Flags if |x − median| > k × 1.4826 × MAD.",
        "math": "Score = |x − med(W)| / (1.4826 × MAD(W)). Flag if score > k.",
        "best_for": "Isolated spikes in slowly varying signals. More robust than rolling Z-score.",
        "not_for": "Detecting sustained level shifts; very short series (< 2W).",
        "inputs": ["Pollutant value"],
        "params": ["window (default 24h)", "threshold (default 3.0)"],
        "cost": "Low",
        "output": "is_hampel_anomaly, hampel_score, hampel_rolling_median, hampel_rolling_mad",
    },
    {
        "name": "Rolling Z-Score",
        "category": "Statistical Temporal",
        "badges": ["Statistical", "Temporal"],
        "desc": "Computes rolling mean and rolling std. Flags if |x − mean| / std > threshold.",
        "math": "Z = (x − μ_rolling) / σ_rolling. Flag if |Z| > threshold.",
        "best_for": "Detecting local level shifts and short-lived spikes.",
        "not_for": "Series with heavy tails; near-zero standard deviation periods.",
        "inputs": ["Pollutant value"],
        "params": ["window (24h, 72h, 168h)", "threshold (default 3.0)"],
        "cost": "Low",
        "output": "is_rolling_z_anomaly, rolling_z_score",
    },
    {
        "name": "Local Outlier Factor",
        "category": "Machine Learning",
        "badges": ["Machine Learning"],
        "desc": "Compares the local density of each point to its k nearest neighbours. "
                "LOF > 1 indicates lower density than neighbours → anomaly.",
        "math": "LOF(p) = avg(lrd(N(p))) / lrd(p) where lrd = local reachability density.",
        "best_for": "Contextual multivariate anomalies. Detects clusters of anomalies.",
        "not_for": "Very large datasets (slow); highly skewed features without scaling.",
        "inputs": ["Pollutant value", "Time features", "ERA5 variables (optional)"],
        "params": ["n_neighbors (default 20)", "contamination (default 0.01)"],
        "cost": "Medium",
        "output": "is_lof_anomaly, lof_score",
    },
    {
        "name": "Isolation Forest",
        "category": "Machine Learning",
        "badges": ["Machine Learning", "Multivariate"],
        "desc": "Randomly isolates observations. Anomalies are isolated in fewer splits than normal points.",
        "math": "Anomaly score = 2^(−average_path_length / c(n)). Lower score = more anomalous.",
        "best_for": "Multivariate detection using pollutant + ERA5 + time features.",
        "not_for": "Interpretability; very small datasets.",
        "inputs": ["Pollutant value", "ERA5", "lag features", "time features"],
        "params": ["n_estimators (default 200)", "contamination (default 0.01)"],
        "cost": "Medium",
        "output": "is_iforest_anomaly, iforest_score",
        "default_recommended": True,
    },
    {
        "name": "STL Residual",
        "category": "Seasonal Temporal",
        "badges": ["Seasonal", "Temporal"],
        "desc": "Seasonal-Trend decomposition. Flags anomalous residuals after removing trend and seasonal components.",
        "math": "x = trend + seasonal + residual. Flag if |residual| > threshold × IQR(residual).",
        "best_for": "Detecting anomalies that deviate from the expected seasonal pattern.",
        "not_for": "Short series (< 2 × seasonal period); series with excessive structural missingness.",
        "inputs": ["Pollutant value (≥ 2 × seasonal period)"],
        "params": ["seasonal period (default 25 for hourly data)", "threshold (default 3.0)"],
        "cost": "Medium",
        "output": "is_stl_anomaly, stl_score, stl_trend, stl_seasonal, stl_residual",
    },
    {
        "name": "SCREEN Rate-of-Change",
        "category": "Rate-of-Change",
        "badges": ["Temporal"],
        "desc": "Approximation of SCREEN-style speed constraints. Flags unrealistic hour-to-hour changes.",
        "math": "Δx = |x_t − x_{t-1}|. Limit = percentile(Δx, p%). Flag if Δx > Limit.",
        "best_for": "Detecting frozen sensors, step changes, and dropouts.",
        "not_for": "Seasonal events with legitimately large changes.",
        "inputs": ["Pollutant value"],
        "params": ["percentile (default 99)"],
        "cost": "Low",
        "note": "Practical approximation — not the exact published SCREEN algorithm.",
        "output": "is_screen_anomaly, screen_score, rate_of_change",
    },
    {
        "name": "LSTM NDT",
        "category": "Deep Learning",
        "badges": ["Deep Learning"],
        "desc": "LSTM predicts the next value. Flags observations where prediction error exceeds a dynamic threshold.",
        "math": "Error_t = |x_t − LSTM(x_{t-seq_len:t})|. Threshold = percentile(Error, p%).",
        "best_for": "Long temporal sequences with complex nonlinear dynamics.",
        "not_for": "Short series; when interpretability is required.",
        "inputs": ["Pollutant value (normalised)"],
        "params": ["seq_len", "hidden_dim", "epochs", "threshold_percentile"],
        "cost": "High (CPU/GPU)",
        "note": "Inspired by NASA TELEMANOM. Threshold is a percentile, not the full iterative pruning.",
        "output": "is_lstm_anomaly, lstm_error, lstm_dynamic_threshold",
    },
    {
        "name": "USAD",
        "category": "Deep Learning",
        "badges": ["Deep Learning", "Multivariate"],
        "desc": "Two-decoder autoencoder (Audibert et al. KDD 2020). Adversarially trains two decoders to maximise sensitivity to reconstruction errors.",
        "math": "Score = α×||W1−W2||² + (1−α)×||W1−W3||² where W1,W2,W3 are reconstructions.",
        "best_for": "Multivariate deep learning detection.",
        "not_for": "Small datasets; univariate-only scenarios.",
        "inputs": ["Pollutant + ERA5 features (multivariate window)"],
        "params": ["seq_len", "hidden_dim", "epochs", "alpha", "threshold_percentile"],
        "cost": "High (CPU/GPU)",
        "output": "is_usad_anomaly, usad_score",
    },
    {
        "name": "Anomaly Transformer",
        "category": "Deep Learning",
        "badges": ["Deep Learning", "Attention"],
        "desc": (
            "Association-discrepancy anomaly detection (Wu et al. ICLR 2022). Learns two attention "
            "distributions per timestep: a series-association (actual softmax attention) and a "
            "prior-association (Gaussian kernel over time distances). High symmetric KL divergence "
            "between the two indicates anomalies that cannot attract consistent temporal context."
        ),
        "math": "Score_t = 0.5×(KL(P_prior||P_series) + KL(P_series||P_prior)) row-averaged at position t.",
        "best_for": "Temporal anomalies that disrupt neighbourhood context; works without labels.",
        "not_for": "Static distributional outliers; very short series (< seq_len).",
        "inputs": ["Pollutant value (normalised window)"],
        "params": ["seq_len (64)", "d_model (32)", "sigma (3.0)", "epochs", "threshold_percentile (99%)"],
        "cost": "High (CPU/GPU)",
        "note": "Faithful reimplementation. Full ICLR 2022 uses multi-layer minimax training; this uses KL loss only.",
        "output": "is_atransformer_anomaly, atransformer_score",
    },
    {
        "name": "DCdetector",
        "category": "Deep Learning",
        "badges": ["Deep Learning", "Dual-Scale"],
        "desc": (
            "Dual-channel contrastive anomaly detection (Chen et al. KDD 2023). Two attention branches: "
            "patch-level (inter-patch attention across P patches) and point-level (intra-patch attention "
            "across T positions). Anomaly score = L1 distance between branch reconstructions."
        ),
        "math": "Score_t = |patch_recon_t − point_recon_t|. High divergence between scales → anomaly.",
        "best_for": "Multi-scale anomalies: both isolated spikes (point) and sustained shifts (patch).",
        "not_for": "Series length < 4 × patch_size.",
        "inputs": ["Pollutant value (normalised)"],
        "params": ["patch_size (8)", "d_model (32)", "seq_len (64)", "epochs", "threshold_percentile (99%)"],
        "cost": "High (CPU/GPU)",
        "note": "Faithful reimplementation. seq_len must be divisible by patch_size.",
        "output": "is_dcdetector_anomaly, dcdetector_score",
    },
    {
        "name": "OmniAnomaly",
        "category": "Deep Learning",
        "badges": ["Deep Learning", "Stochastic", "VAE"],
        "desc": (
            "Stochastic recurrent neural network with variational autoencoder (Su et al. KDD 2019). "
            "GRU encoder produces a stochastic latent z from the posterior q(z|x). GRU decoder "
            "reconstructs x from z. Anomaly score = Monte-Carlo mean reconstruction error under "
            "multiple samples from q(z|x)."
        ),
        "math": "Score_t = E_{z~q}[|x_t − x̂_t|]. High MC reconstruction error → anomaly.",
        "best_for": "Subtle distributional anomalies where deterministic models miss rare events.",
        "not_for": "Very short series (< seq_len); interpretability requirements.",
        "inputs": ["Pollutant value (normalised window)"],
        "params": ["hidden_dim (32)", "latent_dim (8)", "seq_len (48)", "epochs", "n_mc_samples (5)"],
        "cost": "High (CPU/GPU)",
        "note": "Simplified: omits planar normalizing flows from the original paper. Uses standard GRU VAE.",
        "output": "is_omnianoaly_anomaly, omnianomaly_score",
    },
]

IMPUTATION_METHODS_CATALOG = [
    {
        "name": "Linear Interpolation",
        "best_for_gap": "Very short (1–2 hours)",
        "desc": "Fits a straight line between the last observed value before the gap and the first observed value after.",
        "pros": ["Zero computational cost", "Always produces non-extrapolated values", "Physically plausible for short gaps"],
        "cons": ["Misses diurnal patterns", "Unreliable for gaps > 2 hours"],
        "requires": "Nothing — always available",
    },
    {
        "name": "ERA5 Regression",
        "best_for_gap": "Medium (3–24 hours)",
        "desc": "Trains a regression model (Ridge, RF, HistGB, or XGBoost) to predict pollutant from ERA5 meteorology, time features, and lag features.",
        "pros": ["Leverages physical relationships", "Cross-validated", "Feature importance available"],
        "cons": ["Requires ERA5 to be matched to station timestamps", "Needs sufficient training data"],
        "requires": "ERA5 meteorological data",
    },
    {
        "name": "MICE",
        "best_for_gap": "Long gaps (> 24 hours)",
        "desc": "Multiple Imputation by Chained Equations. Iteratively imputes each variable as a function of all others.",
        "pros": ["Multivariate", "Can produce uncertainty estimates", "Standard classical baseline"],
        "cons": ["Ignores temporal ordering", "Slow for large datasets"],
        "requires": "scikit-learn IterativeImputer",
    },
    {
        "name": "BRITS",
        "best_for_gap": "Medium to long",
        "desc": "Bidirectional recurrent imputation. Models forward and backward temporal dependencies simultaneously.",
        "pros": ["Temporal awareness", "Bidirectional — uses future context"],
        "cons": ["Requires PyTorch", "Needs training"],
        "requires": "PyTorch",
        "status": "Planned",
    },
    {
        "name": "SAITS",
        "best_for_gap": "Medium to long",
        "desc": "Self-Attention Imputation for Time Series (Du et al. 2023). Transformer-based imputation with diagonally masked multi-head self-attention.",
        "pros": ["State-of-the-art transformer architecture", "Handles irregular patterns", "Falls back to PyTorch attention"],
        "cons": ["Requires PyPOTS for full implementation", "Computationally intensive on CPU"],
        "requires": "PyPOTS (preferred) or PyTorch",
    },
    {
        "name": "CSDI",
        "best_for_gap": "Long gaps with complex patterns",
        "desc": (
            "Conditional Score-based Diffusion Imputation (Tashiro et al. NeurIPS 2021). Trains a DDPM "
            "denoising model conditioned on observed values and their mask. Inference iteratively denoises "
            "from Gaussian noise to the data distribution at missing positions."
        ),
        "pros": [
            "Principled probabilistic framework",
            "Monte-Carlo samples enable uncertainty quantification",
            "Naturally handles irregular missingness patterns",
        ],
        "cons": [
            "Slow: requires T_diff reverse-diffusion steps per sample × n_samples",
            "Requires PyTorch",
            "Falls back to Savitzky-Golay without PyTorch",
        ],
        "requires": "PyTorch",
        "note": "Faithful structural reimplementation (simplified Transformer noise predictor, not full U-Net).",
    },
    {
        "name": "ImputeFormer",
        "best_for_gap": "Medium to long temporal gaps",
        "desc": (
            "Low-rank projected attention Transformer (Nie et al. KDD 2024). Replaces standard O(T²) "
            "self-attention with a factored O(T×r) low-rank attention, enabling efficient global temporal "
            "context modelling. Conditioned on ERA5 features when available."
        ),
        "pros": [
            "Linear attention complexity O(T×r) — faster than full self-attention",
            "Uses ERA5 covariates for better imputation",
            "Falls back to cubic spline without PyTorch",
        ],
        "cons": [
            "Rank r controls expressiveness — low rank may miss fine structure",
            "Single-station mode (no spatial graph without multi-station setup)",
            "Requires PyTorch",
        ],
        "requires": "PyTorch",
        "note": "Single-station simplified mode; original ImputeFormer operates on multi-variate spatial graphs.",
    },
]


def main():
    st.set_page_config(page_title="Method Library | ISRAP AIRMAPS", layout="wide", page_icon="📚")
    inject_css()

    st.markdown(f"<h1 style='color:{COLORS['navy']};'>📚 Method Library</h1>", unsafe_allow_html=True)
    st.caption("Complete reference for all anomaly detection and imputation methods.")

    tab1, tab2, tab3 = st.tabs(["Anomaly Detection", "Imputation", "Recommendation Wizard"])

    with tab1:
        for m in ANOMALY_METHODS_CATALOG:
            with st.expander(f"**{m['name']}** — {m['category']} · {m.get('cost', '?')} cost"):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**Description:** {m['desc']}")
                    st.markdown(f"**Mathematics:** `{m['math']}`")
                    st.markdown(f"**Best for:** {m['best_for']}")
                    st.markdown(f"**Not for:** {m['not_for']}")
                    st.markdown(f"**Inputs:** {', '.join(m['inputs'])}")
                    st.markdown(f"**Key parameters:** {', '.join(m['params'])}")
                    st.markdown(f"**Output columns:** `{m['output']}`")
                    if m.get("note"):
                        st.caption(f"Note: {m['note']}")
                with c2:
                    for b in m.get("badges", []):
                        color = {
                            "Statistical": COLORS["blue"],
                            "Robust": COLORS["teal"],
                            "Temporal": COLORS["violet"],
                            "Seasonal": COLORS["green"],
                            "Machine Learning": COLORS["amber"],
                            "Deep Learning": COLORS["red"],
                            "Multivariate": COLORS["navy"],
                        }.get(b, COLORS["gray"])
                        st.markdown(
                            f'<span style="background:{color}; color:#fff; padding:2px 8px; '
                            f'border-radius:12px; font-size:0.72rem; font-weight:700; margin-right:4px;">{b}</span>',
                            unsafe_allow_html=True,
                        )
                    if m.get("default_recommended"):
                        st.markdown(
                            '<span style="background:#155724; color:#fff; padding:2px 8px; '
                            'border-radius:12px; font-size:0.72rem; font-weight:700;">Recommended</span>',
                            unsafe_allow_html=True,
                        )

    with tab2:
        for m in IMPUTATION_METHODS_CATALOG:
            status = m.get("status", "Available")
            with st.expander(f"**{m['name']}** — Best for: {m['best_for_gap']} gaps" +
                             (" (Planned)" if status == "Planned" else "")):
                st.markdown(f"**Description:** {m['desc']}")
                st.markdown(f"**Requires:** {m['requires']}")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Advantages:**")
                    for p in m.get("pros", []):
                        st.markdown(f"  ✓ {p}")
                with c2:
                    st.markdown("**Limitations:**")
                    for c in m.get("cons", []):
                        st.markdown(f"  ✗ {c}")
                if status == "Planned":
                    st.info("This method is planned for a future phase. Install required dependencies and retrain.")

    with tab3:
        st.markdown("### Anomaly Method Recommendation Wizard")
        w1, w2 = st.columns(2)
        with w1:
            has_neg = st.checkbox("Has negative values?")
            ser_len = st.slider("Series length (observations)", 10, 50000, 5000)
            has_era5 = st.checkbox("ERA5 available?", value=True)
        with w2:
            multi_station = st.checkbox("Multiple stations available?", value=True)
            needs_dl = st.checkbox("Deep learning preferred?")
            needs_interp = st.checkbox("Interpretability required?")

        rec = recommend_anomaly_method(has_neg, ser_len, has_era5, multi_station, needs_dl, needs_interp)
        st.markdown("---")
        st.success(f"**Recommended:** {rec['method']}")
        st.info(rec["reason"])
        st.caption(f"Alternative: {rec['alternative']} | Cost: {rec['cost']}")

        st.markdown("### Imputation Method Recommendation Wizard")
        g1, g2 = st.columns(2)
        with g1:
            gap_len = st.number_input("Gap length (hours)", 1, 5000, 6)
            is_struct = st.checkbox("Is pollutant structurally missing?")
            needs_unc = st.checkbox("Uncertainty estimates needed?")
        with g2:
            era5_avail = st.checkbox("ERA5 available?", value=True, key="imp_era5")
            multi_st = st.checkbox("Multi-station data?", value=True, key="imp_ms")

        irec = recommend_imputation_method(gap_len, era5_avail, multi_st, is_struct, needs_unc)
        st.markdown("---")
        if irec["method"] == "none":
            st.error(f"**{irec['reason']}**")
        else:
            st.success(f"**Recommended:** {irec['method']}")
            st.info(irec["reason"])
            st.caption(f"Expected confidence: {irec['confidence']}")


if __name__ == "__main__":
    main()

