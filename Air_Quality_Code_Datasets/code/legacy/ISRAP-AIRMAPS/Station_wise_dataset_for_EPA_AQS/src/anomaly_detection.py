"""
anomaly_detection.py
--------------------
Five anomaly-detection methods for EPA AQS pollutant time series.

Recommended pipeline order (from ISRAP_AIRMAPS_Research_Analysis.docx):
  1. Physical cleaning (in preprocessing.py)
  2. IQR fence (3× multiplier)
  3. Hampel filter (window=24 h)
  4. Rolling z-score (window=168 h, threshold=3.5)
  5. STL residual (period=24)
  6. Isolation Forest (multivariate with ERA5 features)
  7. Consensus vote across methods
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. IQR fence
# ---------------------------------------------------------------------------

def detect_iqr_anomalies(
    df: pd.DataFrame,
    col: str,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Flag values outside  [Q1 - k*IQR, Q3 + k*IQR].
    Adds column ``flag_iqr_<col>`` (bool, NaN-safe).
    """
    df = df.copy()
    flag_col = f"flag_iqr_{col}"
    if col not in df.columns:
        df[flag_col] = False
        return df

    series = df[col]
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    df[flag_col] = (series < lower) | (series > upper)
    n = df[flag_col].sum()
    logger.debug("IQR anomalies in %s: %d (lower=%.3f, upper=%.3f)", col, n, lower, upper)
    return df


# ---------------------------------------------------------------------------
# 2. Hampel filter
# ---------------------------------------------------------------------------

def detect_hampel_anomalies(
    df: pd.DataFrame,
    col: str,
    window: int = 24,
    threshold: float = 3.0,
    min_obs: int = 48,
) -> pd.DataFrame:
    """
    Hampel identifier: flag values that deviate from the rolling median
    by more than ``threshold`` times the rolling MAD-based sigma estimate
    (sigma_hat = 1.4826 * MAD).

    Adds column ``flag_hampel_<col>`` (bool).
    """
    df = df.copy()
    flag_col = f"flag_hampel_{col}"
    if col not in df.columns or df[col].notna().sum() < min_obs:
        df[flag_col] = False
        return df

    series = df[col]
    half_w = window // 2
    roll_med = series.rolling(window=window, center=True, min_periods=max(1, half_w)).median()
    roll_mad = (series - roll_med).abs().rolling(
        window=window, center=True, min_periods=max(1, half_w)
    ).median()
    sigma_hat = 1.4826 * roll_mad
    # Avoid division by zero
    sigma_hat = sigma_hat.replace(0, np.nan)
    deviation = (series - roll_med).abs()
    df[flag_col] = deviation > threshold * sigma_hat
    df[flag_col] = df[flag_col].fillna(False)
    n = df[flag_col].sum()
    logger.debug("Hampel anomalies in %s: %d", col, n)
    return df


# ---------------------------------------------------------------------------
# 3. Rolling z-score
# ---------------------------------------------------------------------------

def detect_rolling_zscore_anomalies(
    df: pd.DataFrame,
    col: str,
    window: int = 168,
    threshold: float = 3.5,
    min_obs: int = 48,
) -> pd.DataFrame:
    """
    Flag values where |z| > threshold using a rolling mean/std.
    Adds column ``flag_rolling_zscore_<col>`` (bool).
    """
    df = df.copy()
    flag_col = f"flag_rolling_zscore_{col}"
    if col not in df.columns or df[col].notna().sum() < min_obs:
        df[flag_col] = False
        return df

    series = df[col]
    roll_mean = series.rolling(window=window, center=True, min_periods=max(1, window // 4)).mean()
    roll_std = series.rolling(window=window, center=True, min_periods=max(1, window // 4)).std()
    roll_std = roll_std.replace(0, np.nan)
    z = (series - roll_mean) / roll_std
    df[flag_col] = z.abs() > threshold
    df[flag_col] = df[flag_col].fillna(False)
    n = df[flag_col].sum()
    logger.debug("Rolling z-score anomalies in %s: %d", col, n)
    return df


# ---------------------------------------------------------------------------
# 4. STL decomposition
# ---------------------------------------------------------------------------

def detect_stl_anomalies(
    df: pd.DataFrame,
    col: str,
    period: int = 24,
    min_obs: int = 720,
    iqr_multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Decompose the series with STL (statsmodels) and flag residuals that
    fall outside [Q1 - k*IQR, Q3 + k*IQR] of the residual distribution.

    Adds column ``flag_stl_<col>`` (bool).
    Requires statsmodels >= 0.12.
    """
    df = df.copy()
    flag_col = f"flag_stl_{col}"
    df[flag_col] = False

    if col not in df.columns:
        return df

    series = df[col].copy()
    n_valid = series.notna().sum()
    if n_valid < min_obs:
        logger.info("STL skip %s: only %d valid obs (need %d)", col, n_valid, min_obs)
        return df

    # STL requires no NaN → interpolate gaps temporarily
    series_filled = series.interpolate(method="time", limit_direction="both").ffill().bfill()
    if series_filled.isna().any():
        logger.warning("STL: could not fill all NaN for %s, skipping", col)
        return df

    try:
        from statsmodels.tsa.seasonal import STL
        stl = STL(series_filled, period=period, robust=True)
        result = stl.fit()
        residuals = pd.Series(result.resid, index=series_filled.index)

        q1 = residuals.quantile(0.25)
        q3 = residuals.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr

        # Only flag where original series is not NaN
        valid_mask = series.notna()
        outlier_mask = (residuals < lower) | (residuals > upper)
        df[flag_col] = (valid_mask & outlier_mask)
        n = df[flag_col].sum()
        logger.debug("STL anomalies in %s: %d", col, n)
    except ImportError:
        logger.warning("statsmodels not available – skipping STL for %s", col)
    except Exception as exc:
        logger.warning("STL failed for %s: %s", col, exc)
    return df


# ---------------------------------------------------------------------------
# 5. Isolation Forest
# ---------------------------------------------------------------------------

def detect_isolation_forest_anomalies(
    df: pd.DataFrame,
    col: str,
    era5_features: List[str],
    contamination: float = 0.05,
    n_estimators: int = 100,
    random_seed: int = 42,
    min_obs: int = 200,
) -> pd.DataFrame:
    """
    Multivariate Isolation Forest using the pollutant + available ERA5 features.

    Adds columns:
      ``flag_isolation_forest_<col>`` (bool)
      ``anomaly_score_<col>``         (float, higher = more anomalous)
    """
    df = df.copy()
    flag_col = f"flag_isolation_forest_{col}"
    score_col = f"anomaly_score_{col}"
    df[flag_col] = False
    df[score_col] = np.nan

    if col not in df.columns:
        return df

    # Build feature matrix: pollutant + available ERA5 vars
    feature_cols = [col] + [f for f in era5_features if f in df.columns]
    sub = df[feature_cols].copy()

    # We can only use rows where the pollutant itself is not NaN
    valid_mask = sub[col].notna()
    if valid_mask.sum() < min_obs:
        logger.info("IsolationForest skip %s: only %d valid rows (need %d)",
                    col, valid_mask.sum(), min_obs)
        return df

    sub_valid = sub[valid_mask].copy()

    # Fill ERA5 NaN with column median (ERA5 is ~100% complete so this is rare)
    for c in feature_cols:
        if sub_valid[c].isna().any():
            sub_valid[c] = sub_valid[c].fillna(sub_valid[c].median())

    try:
        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_seed,
            n_jobs=-1,
        )
        X = sub_valid[feature_cols].values
        preds = clf.fit_predict(X)         # -1 = anomaly, 1 = normal
        scores = -clf.score_samples(X)     # higher = more anomalous

        df.loc[valid_mask, flag_col] = preds == -1
        df.loc[valid_mask, score_col] = scores
        n = df[flag_col].sum()
        logger.debug("IsolationForest anomalies in %s: %d", col, n)
    except ImportError:
        logger.warning("scikit-learn not available – skipping IsolationForest for %s", col)
    except Exception as exc:
        logger.warning("IsolationForest failed for %s: %s", col, exc)

    return df


# ---------------------------------------------------------------------------
# Consensus voting
# ---------------------------------------------------------------------------

def compute_anomaly_consensus(
    df: pd.DataFrame,
    col: str,
    methods: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Combine flag columns from multiple methods into a consensus score.

    *methods* defaults to ['iqr', 'hampel', 'rolling_zscore', 'stl', 'isolation_forest'].

    Adds columns:
      ``anomaly_vote_count_<col>``    – how many methods flagged the row
      ``anomaly_consensus_flag_<col>``– True when ≥ 2 methods agree
      ``anomaly_severity_<col>``      – 'none' | 'low' (1) | 'medium' (2-3) | 'high' (≥4)
    """
    df = df.copy()
    if methods is None:
        methods = ["iqr", "hampel", "rolling_zscore", "stl", "isolation_forest"]

    flag_cols = [f"flag_{m}_{col}" for m in methods if f"flag_{m}_{col}" in df.columns]

    if not flag_cols:
        df[f"anomaly_vote_count_{col}"] = 0
        df[f"anomaly_consensus_flag_{col}"] = False
        df[f"anomaly_severity_{col}"] = "none"
        return df

    vote_matrix = df[flag_cols].fillna(False).astype(int)
    df[f"anomaly_vote_count_{col}"] = vote_matrix.sum(axis=1)
    df[f"anomaly_consensus_flag_{col}"] = df[f"anomaly_vote_count_{col}"] >= 2

    def severity(v: int) -> str:
        if v == 0:
            return "none"
        elif v == 1:
            return "low"
        elif v <= 3:
            return "medium"
        else:
            return "high"

    df[f"anomaly_severity_{col}"] = df[f"anomaly_vote_count_{col}"].map(severity)
    n_consensus = df[f"anomaly_consensus_flag_{col}"].sum()
    logger.info("Consensus anomalies in %s: %d (>=2 methods agree)", col, n_consensus)
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_anomaly_detection(
    df: pd.DataFrame,
    col: str,
    era5_features: List[str],
    config: dict,
) -> pd.DataFrame:
    """
    Run the full tiered anomaly detection pipeline for one pollutant column.
    Returns df with all flag / score / consensus columns added.
    """
    if col not in df.columns or df[col].notna().sum() == 0:
        logger.info("Skipping anomaly detection for %s (no valid data)", col)
        return df

    df = detect_iqr_anomalies(df, col, multiplier=config.get("iqr_multiplier", 3.0))
    df = detect_hampel_anomalies(
        df, col,
        window=config.get("hampel_window", 24),
        threshold=config.get("hampel_threshold", 3),
        min_obs=config.get("min_obs_hampel", 48),
    )
    df = detect_rolling_zscore_anomalies(
        df, col,
        window=config.get("rolling_zscore_window", 168),
        threshold=config.get("rolling_zscore_threshold", 3.5),
    )
    df = detect_stl_anomalies(
        df, col,
        period=config.get("stl_period", 24),
        min_obs=config.get("min_obs_stl", 720),
    )
    df = detect_isolation_forest_anomalies(
        df, col,
        era5_features=era5_features,
        contamination=config.get("isolation_forest_contamination", 0.05),
        n_estimators=config.get("isolation_forest_n_estimators", 100),
        random_seed=config.get("random_seed", 42),
        min_obs=config.get("min_obs_if", 200),
    )
    df = compute_anomaly_consensus(df, col)
    return df

