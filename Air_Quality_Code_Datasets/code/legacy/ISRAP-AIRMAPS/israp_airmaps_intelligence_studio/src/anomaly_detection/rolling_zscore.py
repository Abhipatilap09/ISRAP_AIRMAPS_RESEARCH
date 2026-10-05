"""Rolling Z-score anomaly detector."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.anomaly_detection.base import AnomalyResult, BaseDetector


class RollingZScoreDetector(BaseDetector):
    name = "rolling_zscore"
    category = "Statistical Temporal"

    def detect(
        self,
        df: pd.DataFrame,
        pollutant: str,
        window: int = 72,
        threshold: float = 3.0,
        past_only: bool = True,
        **_,
    ) -> tuple[pd.DataFrame, AnomalyResult]:
        df = df.copy()
        flag_col = "is_rolling_z_anomaly"
        score_col = "rolling_z_score"
        t0 = time.perf_counter()

        col = df[pollutant].copy() if pollutant in df.columns else pd.Series(np.nan, index=df.index)

        if past_only:
            shift = 1
        else:
            shift = 0

        roll = col.shift(shift).rolling(window=window, min_periods=max(4, window // 6))
        mu = roll.mean()
        sigma = roll.std()

        z = (col - mu) / sigma.replace(0, np.nan)
        df["rolling_mean"] = mu
        df["rolling_std"] = sigma
        df[score_col] = z.abs().fillna(0.0)
        df[flag_col] = (z.abs() > threshold).fillna(False)

        runtime = time.perf_counter() - t0
        return df, self._make_result(flag_col, score_col, df, threshold, runtime)

