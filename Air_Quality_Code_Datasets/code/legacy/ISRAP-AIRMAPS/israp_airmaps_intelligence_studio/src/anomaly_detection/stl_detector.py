"""STL residual anomaly detector."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.anomaly_detection.base import AnomalyResult, BaseDetector

try:
    from statsmodels.tsa.seasonal import STL
    _STL_AVAILABLE = True
except ImportError:
    _STL_AVAILABLE = False


class STLDetector(BaseDetector):
    name = "stl"
    category = "Seasonal Temporal"

    def detect(
        self,
        df: pd.DataFrame,
        pollutant: str,
        seasonal: int = 25,   # must be odd; 25 ≈ daily cycle
        threshold: float = 3.0,
        threshold_method: str = "iqr",  # "iqr" | "mad" | "zscore"
        **_,
    ) -> tuple[pd.DataFrame, AnomalyResult]:
        df = df.copy()
        flag_col = "is_stl_anomaly"
        score_col = "stl_score"
        t0 = time.perf_counter()

        if not _STL_AVAILABLE:
            df[flag_col] = False
            df[score_col] = 0.0
            runtime = time.perf_counter() - t0
            return df, self._make_result(flag_col, score_col, df, threshold, runtime,
                                         status="error", warning="statsmodels not installed")

        col = df[pollutant].copy() if pollutant in df.columns else pd.Series(np.nan, index=df.index)
        min_points = seasonal * 2 + 1

        if col.notna().sum() < min_points:
            df[flag_col] = False
            df[score_col] = 0.0
            runtime = time.perf_counter() - t0
            return df, self._make_result(flag_col, score_col, df, threshold, runtime,
                                         status="skipped", warning=f"Need ≥{min_points} valid points; got {col.notna().sum()}")

        # Interpolate gaps for decomposition only (not final imputation)
        col_filled = col.interpolate(method="linear", limit=72).ffill().bfill()

        try:
            # seasonal must be odd
            if seasonal % 2 == 0:
                seasonal += 1
            stl = STL(col_filled, seasonal=seasonal, period=24, robust=True)
            res = stl.fit()
            trend = res.trend
            seasonal_comp = res.seasonal
            residual = res.resid
        except Exception as exc:
            df[flag_col] = False
            df[score_col] = 0.0
            runtime = time.perf_counter() - t0
            return df, self._make_result(flag_col, score_col, df, threshold, runtime,
                                         status="error", warning=str(exc))

        df["stl_trend"] = trend.values
        df["stl_seasonal"] = seasonal_comp.values
        df["stl_residual"] = residual.values

        r = pd.Series(residual.values, index=df.index)
        if threshold_method == "iqr":
            q1, q3 = r.quantile(0.25), r.quantile(0.75)
            iqr = q3 - q1
            cutoff = threshold * iqr if iqr > 0 else threshold * r.std()
            score = r.abs() / (iqr + 1e-9)
        elif threshold_method == "mad":
            med = r.median()
            mad = (r - med).abs().median()
            cutoff = threshold * 1.4826 * (mad if mad > 0 else r.std())
            score = (r - med).abs() / (1.4826 * mad + 1e-9)
        else:  # zscore
            mu, sd = r.mean(), r.std()
            cutoff = threshold * sd
            score = (r - mu).abs() / (sd + 1e-9)

        df[score_col] = score.fillna(0.0)
        # Only flag on observed (non-interpolated) positions
        df[flag_col] = (score > threshold) & col.notna()

        runtime = time.perf_counter() - t0
        return df, self._make_result(flag_col, score_col, df, threshold, runtime)

