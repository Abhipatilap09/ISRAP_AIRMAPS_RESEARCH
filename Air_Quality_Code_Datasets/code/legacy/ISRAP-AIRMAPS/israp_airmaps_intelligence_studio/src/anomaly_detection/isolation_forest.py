"""Isolation Forest anomaly detector."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.anomaly_detection.base import AnomalyResult, BaseDetector
from src.config import ERA5_VARS, RANDOM_SEED


class IsolationForestDetector(BaseDetector):
    name = "isolation_forest"
    category = "Machine Learning"

    def detect(
        self,
        df: pd.DataFrame,
        pollutant: str,
        contamination: float = 0.01,
        n_estimators: int = 200,
        max_samples: str | int = "auto",
        random_state: int = RANDOM_SEED,
        include_era5: bool = True,
        include_time: bool = True,
        include_lags: bool = True,
        **_,
    ) -> tuple[pd.DataFrame, AnomalyResult]:
        df = df.copy()
        flag_col = "is_iforest_anomaly"
        score_col = "iforest_score"
        t0 = time.perf_counter()

        feature_cols = [pollutant] if pollutant in df.columns else []

        if include_time:
            for tc, src in [("hour", lambda d: d.dt.hour), ("month", lambda d: d.dt.month),
                             ("dow", lambda d: d.dt.dayofweek)]:
                if tc not in df.columns:
                    df[tc] = src(df["datetime"])
                feature_cols.append(tc)

        if include_era5:
            feature_cols += [c for c in ERA5_VARS if c in df.columns]

        if include_lags:
            for lag in [1, 2, 3, 6, 12, 24]:
                lag_col = f"{pollutant}_lag{lag}"
                if lag_col not in df.columns:
                    df[lag_col] = df[pollutant].shift(lag)
                feature_cols.append(lag_col)

        feature_cols = list(dict.fromkeys([c for c in feature_cols if c in df.columns]))
        X = df[feature_cols].copy()
        valid_mask = X.notna().all(axis=1)

        if valid_mask.sum() < 20:
            df[flag_col] = False
            df[score_col] = 0.0
            runtime = time.perf_counter() - t0
            return df, self._make_result(flag_col, score_col, df, None, runtime,
                                         status="skipped", warning="Insufficient valid rows for Isolation Forest")

        X_valid = X[valid_mask].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)

        clf = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
        )
        preds = clf.fit_predict(X_scaled)
        raw_scores = clf.decision_function(X_scaled)
        # Negate so higher = more anomalous
        anom_scores = -raw_scores

        flag_series = pd.Series(False, index=df.index)
        score_series = pd.Series(0.0, index=df.index)
        flag_series.loc[X_valid.index] = (preds == -1)
        score_series.loc[X_valid.index] = anom_scores

        df[flag_col] = flag_series
        df[score_col] = score_series

        runtime = time.perf_counter() - t0
        return df, self._make_result(flag_col, score_col, df, None, runtime)

