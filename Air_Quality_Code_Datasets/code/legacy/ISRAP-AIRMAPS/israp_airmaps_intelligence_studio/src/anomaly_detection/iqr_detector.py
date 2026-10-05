"""IQR fence anomaly detector."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.anomaly_detection.base import AnomalyResult, BaseDetector


class IQRDetector(BaseDetector):
    name = "iqr"
    category = "Statistical"

    def detect(
        self,
        df: pd.DataFrame,
        pollutant: str,
        multiplier: float = 3.0,
        **_,
    ) -> tuple[pd.DataFrame, AnomalyResult]:
        df = df.copy()
        flag_col = "is_iqr_anomaly"
        score_col = "iqr_score"
        t0 = time.perf_counter()

        col = df[pollutant] if pollutant in df.columns else pd.Series(np.nan, index=df.index)
        valid = col.dropna()

        q1 = float(valid.quantile(0.25)) if len(valid) > 4 else np.nan
        q3 = float(valid.quantile(0.75)) if len(valid) > 4 else np.nan
        iqr = q3 - q1 if not np.isnan(q1) else 0.0

        if iqr == 0:
            # Fall back to std-based fence when IQR=0
            mu = float(valid.mean())
            sd = float(valid.std()) or 1.0
            lower = mu - multiplier * sd
            upper = mu + multiplier * sd
        else:
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr

        df["iqr_lower_bound"] = lower
        df["iqr_upper_bound"] = upper

        below = col < lower
        above = col > upper
        df[flag_col] = (below | above).fillna(False).astype(bool)

        score = pd.Series(0.0, index=df.index)
        if iqr > 0:
            score = ((col - col.median()).abs() / (iqr + 1e-9))
        else:
            score = ((col - valid.mean()).abs() / (valid.std() + 1e-9))
        df[score_col] = score.fillna(0.0)

        runtime = time.perf_counter() - t0
        return df, self._make_result(flag_col, score_col, df, multiplier, runtime)

