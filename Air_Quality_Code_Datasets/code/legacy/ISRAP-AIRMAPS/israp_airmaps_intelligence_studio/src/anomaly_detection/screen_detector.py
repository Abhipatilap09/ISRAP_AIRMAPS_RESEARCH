"""SCREEN-style rate-of-change anomaly detector (practical approximation)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.anomaly_detection.base import AnomalyResult, BaseDetector


class SCREENDetector(BaseDetector):
    """Flags unrealistic hour-to-hour changes.

    This is a practical approximation inspired by SCREEN-style speed constraints.
    It flags observations whose first difference exceeds a data-derived limit.
    """

    name = "screen"
    category = "Rate-of-Change"

    def detect(
        self,
        df: pd.DataFrame,
        pollutant: str,
        percentile: float = 99.0,
        rolling_limits: bool = True,
        **_,
    ) -> tuple[pd.DataFrame, AnomalyResult]:
        df = df.copy()
        flag_col = "is_screen_anomaly"
        score_col = "screen_score"
        t0 = time.perf_counter()

        col = df[pollutant].copy() if pollutant in df.columns else pd.Series(np.nan, index=df.index)
        diff = col.diff().abs()
        df["rate_of_change"] = diff

        valid_diff = diff.dropna()
        if len(valid_diff) < 10:
            df[flag_col] = False
            df[score_col] = 0.0
            runtime = time.perf_counter() - t0
            return df, self._make_result(flag_col, score_col, df, None, runtime,
                                         status="skipped", warning="Too few observations")

        if rolling_limits:
            roll = diff.shift(1).rolling(window=168, min_periods=24)
            upper = roll.quantile(percentile / 100.0).fillna(float(np.nanpercentile(valid_diff, percentile)))
        else:
            upper_val = float(np.nanpercentile(valid_diff, percentile))
            upper = pd.Series(upper_val, index=df.index)

        lower = pd.Series(0.0, index=df.index)
        df["screen_upper_limit"] = upper
        df["screen_lower_limit"] = lower

        df[flag_col] = (diff > upper).fillna(False)
        df[score_col] = (diff / upper.replace(0, np.nan)).fillna(0.0)

        runtime = time.perf_counter() - t0
        return df, self._make_result(flag_col, score_col, df, percentile, runtime)

