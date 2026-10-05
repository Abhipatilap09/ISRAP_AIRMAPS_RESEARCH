"""ERA5-assisted regression imputer."""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .base import BaseImputer, ImputationResult
from .config import ERA5_VARS, RANDOM_SEED
from .feature_engineering import build_features

logger = logging.getLogger(__name__)

MODELS: dict[str, Any] = {
    "ridge": lambda: Ridge(alpha=1.0),
    "random_forest": lambda: RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
    "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(random_state=RANDOM_SEED),
}

try:
    from xgboost import XGBRegressor
    MODELS["xgboost"] = lambda: XGBRegressor(n_estimators=100, random_state=RANDOM_SEED,
                                              tree_method="hist", verbosity=0)
except ImportError:
    pass


class ERA5RegressionImputer(BaseImputer):
    name = "era5_regression"

    def impute(
        self,
        df: pd.DataFrame,
        pollutant: str,
        model_name: str = "random_forest",
        lags: list[int] | None = None,
        windows: list[int] | None = None,
        n_splits: int = 3,
        **_,
    ) -> tuple[pd.DataFrame, ImputationResult]:
        df = df.copy()
        out_col = "imputed_era5_regression"
        was_col = "was_imputed_era5"
        model_col = "era5_model_name"
        conf_col = "era5_imputation_confidence"
        t0 = time.perf_counter()

        if pollutant not in df.columns:
            df[out_col] = np.nan
            df[was_col] = False
            return df, ImputationResult(self.name, 0, None, None, self._elapsed(t0),
                                        "error", f"{pollutant} not in dataframe")

        lags = lags or [1, 2, 3, 6, 12, 24]
        windows = windows or [3, 6, 24]

        df_feat, feature_names = build_features(df, pollutant, lags=lags, windows=windows)
        feature_names = [c for c in feature_names if c in df_feat.columns]

        observed = df_feat[pollutant].notna()
        missing = df_feat[pollutant].isna()

        X_obs = df_feat.loc[observed, feature_names]
        y_obs = df_feat.loc[observed, pollutant]
        X_obs = X_obs.fillna(X_obs.median())

        if len(y_obs) < 20 or X_obs.shape[1] == 0:
            df[out_col] = np.nan
            df[was_col] = False
            return df, ImputationResult(self.name, 0, None, None, self._elapsed(t0),
                                        "skipped", "Insufficient observed data for ERA5 regression")

        factory = MODELS.get(model_name, MODELS["ridge"])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_obs)

        model = factory()
        model.fit(X_scaled, y_obs)

        if missing.any():
            X_miss = df_feat.loc[missing, feature_names].fillna(df_feat[feature_names].median())
            X_miss_scaled = scaler.transform(X_miss)
            preds = model.predict(X_miss_scaled)
            preds = np.clip(preds, 0.0, None)  # no negative concentrations

            df[out_col] = df[pollutant].copy()
            df.loc[missing, out_col] = preds
            df[was_col] = missing & df[out_col].notna()
        else:
            df[out_col] = df[pollutant].copy()
            df[was_col] = False

        df[model_col] = model_name

        # Confidence: average R² from TimeSeriesSplit
        try:
            tss = TimeSeriesSplit(n_splits=n_splits)
            scores = []
            for tr_idx, val_idx in tss.split(X_obs):
                m = factory()
                m.fit(scaler.transform(X_obs.iloc[tr_idx]), y_obs.iloc[tr_idx])
                scores.append(m.score(scaler.transform(X_obs.iloc[val_idx]), y_obs.iloc[val_idx]))
            r2_mean = float(np.mean(scores))
        except Exception:
            r2_mean = np.nan
        df[conf_col] = round(max(0.0, r2_mean), 3) if not np.isnan(r2_mean) else np.nan

        # Feature importance
        fi: dict[str, float] = {}
        if hasattr(model, "feature_importances_"):
            fi = dict(zip(feature_names, model.feature_importances_.tolist()))
        elif hasattr(model, "coef_"):
            fi = dict(zip(feature_names, model.coef_.tolist()))

        n_imp = int(df[was_col].sum())
        return df, ImputationResult(self.name, n_imp, None, None, self._elapsed(t0),
                                    "ok", feature_importance=fi)


