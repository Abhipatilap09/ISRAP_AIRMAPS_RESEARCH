"""Deterministic rule-based interpretation engine."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def interpret_anomaly(
    flagged_df: pd.DataFrame,
    method: str,
    pct_flagged: float,
    station: str,
    pollutant: str,
) -> list[str]:
    """Return a list of plain-English insight strings about the anomaly run."""
    insights = []

    if pct_flagged > 10:
        insights.append(
            f"The {method} method flagged {pct_flagged:.1f}% of observations — "
            "this may indicate an aggressive threshold or systematic data issues. "
            "Consider raising the threshold."
        )
    elif pct_flagged < 0.1:
        insights.append(
            f"Only {pct_flagged:.2f}% of observations were flagged. "
            "The threshold may be too conservative; consider lowering it."
        )
    else:
        insights.append(
            f"{method} flagged {pct_flagged:.1f}% of observations."
        )

    if "is_physical_anomaly" in flagged_df.columns:
        n_phys = int(flagged_df["is_physical_anomaly"].sum())
        if n_phys > 0:
            insights.append(
                f"{n_phys} values violate physical constraints (e.g., negative concentrations). "
                "These are almost certainly sensor errors and should be converted to NaN."
            )

    if "datetime" in flagged_df.columns:
        flag_col = f"is_{method.replace('_ndt', '')}_anomaly"
        flag_cols_available = [c for c in flagged_df.columns if c.startswith("is_") and c.endswith("_anomaly")]
        for fc in flag_cols_available:
            if flagged_df[fc].any():
                flag_series = flagged_df[fc]
                if len(flagged_df) > 0:
                    hour_counts = flagged_df.loc[flag_series, "datetime"].dt.hour.value_counts()
                    if len(hour_counts) > 0:
                        peak_hr = int(hour_counts.idxmax())
                        insights.append(
                            f"Anomalies are most frequent at hour {peak_hr:02d}:00, "
                            "which may indicate a diurnal sensor pattern or real pollution peak."
                        )
                break

    return insights


def interpret_imputation(
    n_imputed: int,
    method: str,
    gap_category: str,
    r2_score: float | None,
    pollutant: str,
) -> list[str]:
    insights = []

    if gap_category == "very_short":
        insights.append(
            f"Linear interpolation is appropriate for {gap_category} gaps (1–2 hours) "
            "and should produce reliable results."
        )
    elif gap_category == "medium" and method in ("era5_regression", "mice"):
        if r2_score is not None and r2_score > 0.7:
            insights.append(
                f"ERA5 regression achieved R²={r2_score:.2f}, indicating a strong meteorological "
                f"association with {pollutant}. Imputed values are likely reliable."
            )
        elif r2_score is not None and r2_score < 0.3:
            insights.append(
                f"ERA5 regression R²={r2_score:.2f} is low — {pollutant} may not be strongly "
                "correlated with available meteorological variables. Treat imputed values as uncertain."
            )
    elif gap_category in ("long", "very_long"):
        insights.append(
            f"This is a {gap_category} gap ({n_imputed} hours). "
            "Imputed values carry high uncertainty. Consider retaining as missing in final modeling."
        )

    if n_imputed == 0:
        insights.append("No values were imputed — either no missing data exists, or the method was unable to impute.")

    return insights


def recommend_anomaly_method(
    has_negatives: bool,
    series_length: int,
    has_era5: bool,
    has_multi_station: bool,
    needs_deep_learning: bool,
    needs_interpretable: bool,
) -> dict[str, str]:
    if has_negatives:
        return {
            "method": "physical_constraints",
            "reason": "Negative concentration values are physically impossible. Use Physical Constraint detection first.",
            "alternative": "iqr",
            "cost": "Very low",
        }
    if series_length < 100:
        return {
            "method": "iqr",
            "reason": "Series is short — IQR fence is the most robust choice with limited data.",
            "alternative": "rolling_zscore",
            "cost": "Very low",
        }
    if needs_interpretable:
        return {
            "method": "hampel",
            "reason": "Hampel filter is interpretable (median-MAD), robust to outliers, and works on short windows.",
            "alternative": "rolling_zscore",
            "cost": "Low",
        }
    if needs_deep_learning and series_length > 500:
        return {
            "method": "isolation_forest",
            "reason": "Isolation Forest handles multivariate features well and is scalable.",
            "alternative": "lstm_ndt" if series_length > 2000 else "stl",
            "cost": "Medium",
        }
    return {
        "method": "isolation_forest",
        "reason": "Isolation Forest combines pollutant values with ERA5 context and time features for balanced detection.",
        "alternative": "stl",
        "cost": "Medium",
    }


def recommend_imputation_method(
    gap_length: int,
    has_era5: bool,
    has_multi_station: bool,
    is_structural: bool,
    needs_uncertainty: bool,
) -> dict[str, str]:
    if is_structural:
        return {
            "method": "none",
            "reason": "This pollutant is structurally missing at this station and must not be imputed.",
            "confidence": "N/A",
        }
    if gap_length <= 2:
        return {
            "method": "linear",
            "reason": "Very short gap (≤ 2 hours). Linear interpolation is reliable and computationally free.",
            "confidence": "High",
        }
    if gap_length <= 24 and has_era5:
        return {
            "method": "era5_regression",
            "reason": "Medium gap with ERA5 available. ERA5 regression leverages meteorological context.",
            "confidence": "Medium–High",
        }
    if gap_length <= 72:
        return {
            "method": "mice",
            "reason": "Long gap. MICE uses multi-variable imputation to fill complex patterns.",
            "confidence": "Medium",
        }
    if needs_uncertainty:
        return {
            "method": "csdi",
            "reason": "Very long gap requiring probabilistic uncertainty estimates. CSDI provides confidence intervals.",
            "confidence": "Low–Medium",
        }
    return {
        "method": "mice",
        "reason": "Default for long gaps. MICE is the most reliable classical baseline.",
        "confidence": "Low–Medium",
    }

