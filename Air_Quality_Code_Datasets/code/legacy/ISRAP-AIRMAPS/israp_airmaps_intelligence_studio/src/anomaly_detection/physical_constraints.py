"""Physical constraint anomaly detector."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.anomaly_detection.base import AnomalyResult, BaseDetector
from src.config import POLLUTANTS


class PhysicalConstraintDetector(BaseDetector):
    name = "physical_constraints"
    category = "Physical"

    def detect(
        self,
        df: pd.DataFrame,
        pollutant: str,
        physical_min: float | None = None,
        physical_max: float | None = None,
        **_,
    ) -> tuple[pd.DataFrame, AnomalyResult]:
        df = df.copy()
        flag_col = "is_physical_anomaly"
        score_col = "physical_constraint_score"
        reason_col = "physical_anomaly_reason"

        meta = POLLUTANTS.get(pollutant, {})
        p_min = physical_min if physical_min is not None else meta.get("physical_min", -np.inf)
        p_max = physical_max if physical_max is not None else meta.get("physical_max", np.inf)

        t0 = time.perf_counter()
        col = df[pollutant] if pollutant in df.columns else pd.Series(np.nan, index=df.index)

        is_neg = col < 0
        is_above_max = col > p_max
        is_inf = ~np.isfinite(col.fillna(0)) & col.notna()

        df["is_negative"] = is_neg.astype(bool)
        df[flag_col] = (is_neg | is_above_max | is_inf).astype(bool)
        df[score_col] = np.where(
            is_neg, np.abs(col) / max(abs(p_min) if p_min != -np.inf else 1, 1),
            np.where(is_above_max, (col - p_max) / max(p_max, 1), 0.0)
        )

        reasons = []
        for _, row in df.iterrows():
            v = row.get(pollutant, np.nan)
            if pd.isna(v):
                reasons.append("")
            elif v < 0:
                reasons.append(f"Negative value ({v:.4f}); physically impossible for {pollutant}")
            elif v > p_max:
                reasons.append(f"Exceeds physical max ({v:.4f} > {p_max})")
            elif not np.isfinite(float(v)):
                reasons.append("Infinite or NaN value")
            else:
                reasons.append("")
        df[reason_col] = reasons

        runtime = time.perf_counter() - t0
        return df, self._make_result(flag_col, score_col, df, p_min, runtime)

