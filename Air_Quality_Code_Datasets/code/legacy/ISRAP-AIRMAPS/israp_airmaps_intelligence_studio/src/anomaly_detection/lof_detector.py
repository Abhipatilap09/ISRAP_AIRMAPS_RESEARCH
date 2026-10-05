"""Local Outlier Factor anomaly detector."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src.anomaly_detection.base import AnomalyResult, BaseDetector
from src.config import ERA5_VARS


class LOFDetector(BaseDetector):
    name = "lof"
    category = "Machine Learning"

    def detect(
        self,
        df: pd.DataFrame,
        pollutant: str,
        n_neighbors: int = 20,
        contamination: float = 0.01,
        include_era5: bool = True,
        include_time: bool = True,
        **_,
    ) -> tuple[pd.DataFrame, AnomalyResult]:
        df = df.copy()
        flag_col = "is_lof_anomaly"
        score_col = "lof_score"
        t0 = time.perf_counter()

        feature_cols = [pollutant] if pollutant in df.columns else []
        if include_time:
            for tc in ["hour", "dow", "month"]:
                if tc not in df.columns:
                    if tc == "hour":
                        df["hour"] = df["datetime"].dt.hour
                    elif tc == "dow":
                        df["dow"] = df["datetime"].dt.dayofweek
                    elif tc == "month":
                        df["month"] = df["datetime"].dt.month
                feature_cols.append(tc)
        if include_era5:
            feature_cols += [c for c in ERA5_VARS if c in df.columns]

        feature_cols = list(dict.fromkeys(feature_cols))  # deduplicate
        X = df[feature_cols].copy()
        valid_mask = X.notna().all(axis=1)

        if valid_mask.sum() < max(n_neighbors + 1, 10):
            df[flag_col] = False
            df[score_col] = 0.0
            runtime = time.perf_counter() - t0
            return df, self._make_result(flag_col, score_col, df, None, runtime,
                                         status="skipped", warning="Insufficient valid rows for LOF")

        X_valid = X[valid_mask].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)

        lof = LocalOutlierFactor(
            n_neighbors=min(n_neighbors, len(X_valid) - 1),
            contamination=contamination,
            novelty=False,
        )
        preds = lof.fit_predict(X_scaled)
        scores_raw = -lof.negative_outlier_factor_

        flag_series = pd.Series(False, index=df.index)
        score_series = pd.Series(0.0, index=df.index)
        flag_series.loc[X_valid.index] = (preds == -1)
        score_series.loc[X_valid.index] = scores_raw

        df[flag_col] = flag_series
        df[score_col] = score_series

        runtime = time.perf_counter() - t0
        return df, self._make_result(flag_col, score_col, df, None, runtime)

