"""Hampel filter anomaly detector."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.anomaly_detection.base import AnomalyResult, BaseDetector

# MAD → Gaussian σ scale factor
_K = 1.4826


class HampelDetector(BaseDetector):
    name = "hampel"
    category = "Robust Temporal"

    def detect(
        self,
        df: pd.DataFrame,
        pollutant: str,
        window: int = 24,
        threshold: float = 3.0,
        past_only: bool = True,
        **_,
    ) -> tuple[pd.DataFrame, AnomalyResult]:
        df = df.copy()
        flag_col = "is_hampel_anomaly"
        score_col = "hampel_score"
        t0 = time.perf_counter()

        col = df[pollutant].copy() if pollutant in df.columns else pd.Series(np.nan, index=df.index)
        n = len(col)

        if n < 2 * window:
            df[flag_col] = False
            df[score_col] = 0.0
            df["hampel_rolling_median"] = np.nan
            df["hampel_rolling_mad"] = np.nan
            runtime = time.perf_counter() - t0
            return df, self._make_result(flag_col, score_col, df, threshold, runtime,
                                         status="skipped", warning="Series too short for Hampel window")

        half = window // 2
        rolling_med = np.full(n, np.nan)
        rolling_mad = np.full(n, np.nan)
        scores = np.zeros(n)
        flags = np.zeros(n, dtype=bool)

        for i in range(n):
            if past_only:
                lo = max(0, i - window + 1)
                hi = i + 1
            else:
                lo = max(0, i - half)
                hi = min(n, i + half + 1)
            window_vals = col.iloc[lo:hi].dropna().values
            if len(window_vals) == 0:
                continue
            med = float(np.median(window_vals))
            mad = float(np.median(np.abs(window_vals - med)))
            rolling_med[i] = med
            rolling_mad[i] = mad
            sigma = _K * mad if mad > 0 else _K * (np.std(window_vals) or 1.0)
            v = col.iloc[i]
            if not pd.isna(v):
                sc = abs(v - med) / sigma
                scores[i] = sc
                flags[i] = sc > threshold

        df["hampel_rolling_median"] = rolling_med
        df["hampel_rolling_mad"] = rolling_mad
        df[score_col] = scores
        df[flag_col] = flags

        runtime = time.perf_counter() - t0
        return df, self._make_result(flag_col, score_col, df, threshold, runtime)

