"""
validation.py
-------------
Hold-out validation of the imputation pipeline.

Strategy: artificially mask known-good observations, then run imputation
and compare predictions against ground truth. Prevents data leakage by
fitting models only on non-masked data.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mask creation
# ---------------------------------------------------------------------------

def create_artificial_masks(
    series: pd.Series,
    config: dict,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Create boolean mask arrays (True = hide this value) for several
    missingness scenarios.

    Returns dict: mask_name → boolean ndarray (same length as series).
    Only observed (non-NaN) positions can be masked.
    """
    rng = np.random.default_rng(seed)
    n = len(series)
    observed = ~series.isna()
    obs_idx = np.where(observed)[0]

    if len(obs_idx) < 50:
        logger.warning("Too few observations (%d) to create validation masks", len(obs_idx))
        return {}

    masks: Dict[str, np.ndarray] = {}

    # Random fraction masks
    for frac in config.get("validation_mask_fractions", [0.05, 0.10, 0.20]):
        k = max(1, int(frac * len(obs_idx)))
        chosen = rng.choice(obs_idx, size=k, replace=False)
        m = np.zeros(n, dtype=bool)
        m[chosen] = True
        masks[f"random_{int(frac*100)}pct"] = m

    # Block masks
    block_lengths = config.get("validation_block_lengths", {"short": 3, "medium": 12, "long": 48})
    for name, blen in block_lengths.items():
        if len(obs_idx) < blen + 2:
            continue
        # Find a run of *blen* consecutive observed rows
        start_candidates = []
        for i in range(len(obs_idx) - blen + 1):
            window = obs_idx[i:i + blen]
            if window[-1] - window[0] == blen - 1:   # consecutive
                start_candidates.append(i)
        if not start_candidates:
            continue
        chosen_start = rng.choice(start_candidates)
        m = np.zeros(n, dtype=bool)
        m[obs_idx[chosen_start:chosen_start + blen]] = True
        masks[f"block_{name}_{blen}h"] = m

    return masks


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate_imputation(
    true_values: np.ndarray,
    imputed_values: np.ndarray,
) -> dict:
    """
    Compute MAE, RMSE, R², and bias between true and imputed arrays.
    Ignores positions where either array is NaN.
    """
    valid = ~(np.isnan(true_values) | np.isnan(imputed_values))
    if valid.sum() == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "Bias": np.nan, "n_eval": 0}

    y_true = true_values[valid]
    y_pred = imputed_values[valid]

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    bias = float(np.mean(y_pred - y_true))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Bias": round(bias, 4),
        "n_eval": int(valid.sum()),
    }


# ---------------------------------------------------------------------------
# Validation experiment
# ---------------------------------------------------------------------------

def run_validation_experiment(
    station_df: pd.DataFrame,
    col: str,
    era5_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    For each mask scenario:
      1. Hide true values
      2. Run imputation pipeline on the masked series
      3. Compare predictions to held-out ground truth

    Returns a DataFrame with one row per mask scenario.
    """
    if col not in station_df.columns:
        return pd.DataFrame()

    series = station_df[col]
    if series.notna().sum() < 100:
        logger.info("Skipping validation for %s: too few observations", col)
        return pd.DataFrame()

    masks = create_artificial_masks(series, config, seed=config.get("random_seed", 42))
    if not masks:
        return pd.DataFrame()

    from .imputation import (
        impute_short_gaps,
        impute_medium_gaps,
        impute_long_gaps,
    )

    rows = []
    for mask_name, mask in masks.items():
        # Create masked series (hide ground truth)
        masked_series = series.copy()
        masked_series[mask] = np.nan

        masked_df = station_df.copy()
        masked_df[col] = masked_series

        try:
            # Tier 1
            s1 = impute_short_gaps(masked_series, max_gap=config.get("short_gap_max", 2))
            # Tier 2
            tmp = masked_df.copy()
            tmp[col] = s1
            s2 = impute_medium_gaps(tmp, col, era5_df,
                                    max_gap=config.get("medium_gap_max", 24),
                                    random_seed=config.get("random_seed", 42))
            # Tier 3
            tmp2 = masked_df.copy()
            tmp2[col] = s2
            s3 = impute_long_gaps(tmp2, col, era5_df)

            # Evaluate
            true_vals = series.values[mask]
            imputed_vals = s3.values[mask]

            metrics = evaluate_imputation(true_vals, imputed_vals)
            metrics["mask"] = mask_name
            metrics["col"] = col
            rows.append(metrics)
        except Exception as exc:
            logger.warning("Validation experiment failed for %s / %s: %s", col, mask_name, exc)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

def generate_leaderboard(all_metrics: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate per-station/per-pollutant metrics and produce a ranked summary.
    Ranked by RMSE ascending.
    """
    if not all_metrics:
        return pd.DataFrame()

    combined = pd.concat([m for m in all_metrics if not m.empty], ignore_index=True)
    if combined.empty:
        return pd.DataFrame()

    summary = (
        combined
        .groupby(["col", "mask"])
        .agg(
            mean_MAE=("MAE", "mean"),
            mean_RMSE=("RMSE", "mean"),
            mean_R2=("R2", "mean"),
            mean_Bias=("Bias", "mean"),
            total_n=("n_eval", "sum"),
        )
        .reset_index()
        .sort_values("mean_RMSE")
    )
    return summary

