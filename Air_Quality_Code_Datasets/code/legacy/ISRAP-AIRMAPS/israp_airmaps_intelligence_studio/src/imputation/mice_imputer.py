"""MICE (Multiple Imputation by Chained Equations) imputer."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from src.imputation.base import BaseImputer, ImputationResult
from src.config import ERA5_VARS, RANDOM_SEED


class MICEImputer(BaseImputer):
    name = "mice"

    def impute(
        self,
        df: pd.DataFrame,
        pollutant: str,
        max_iter: int = 10,
        random_state: int = RANDOM_SEED,
        n_imputations: int = 1,
        **_,
    ) -> tuple[pd.DataFrame, ImputationResult]:
        df = df.copy()
        out_col = "imputed_mice"
        was_col = "was_imputed_mice"
        std_col = "mice_std"
        t0 = time.perf_counter()

        if pollutant not in df.columns:
            df[out_col] = np.nan
            df[was_col] = False
            df[std_col] = np.nan
            return df, ImputationResult(self.name, 0, None, None, self._elapsed(t0),
                                        "error", f"{pollutant} not in dataframe")

        from src.config import POLLUTANTS
        pollutant_cols = [c for c in POLLUTANTS if c in df.columns]
        era5_cols = [c for c in ERA5_VARS if c in df.columns]
        time_feats = []
        if "datetime" in df.columns:
            df["_hour"] = df["datetime"].dt.hour
            df["_month"] = df["datetime"].dt.month
            time_feats = ["_hour", "_month"]

        feature_cols = [c for c in pollutant_cols + era5_cols + time_feats if c in df.columns]
        X = df[feature_cols].copy()
        missing_mask = df[pollutant].isna()

        if missing_mask.sum() == 0:
            df[out_col] = df[pollutant]
            df[was_col] = False
            df[std_col] = 0.0
            return df, ImputationResult(self.name, 0, None, None, self._elapsed(t0), "ok")

        if n_imputations > 1:
            imputeds = []
            for seed_offset in range(n_imputations):
                imp = IterativeImputer(max_iter=max_iter, random_state=random_state + seed_offset,
                                       min_value=0.0, skip_complete=True)
                imputed = imp.fit_transform(X)
                p_idx = feature_cols.index(pollutant)
                imputeds.append(imputed[:, p_idx])
            stacked = np.stack(imputeds, axis=1)
            mean_imp = np.mean(stacked, axis=1)
            std_imp = np.std(stacked, axis=1)
        else:
            imp = IterativeImputer(max_iter=max_iter, random_state=random_state,
                                   min_value=0.0, skip_complete=True)
            imputed_X = imp.fit_transform(X)
            p_idx = feature_cols.index(pollutant)
            mean_imp = imputed_X[:, p_idx]
            std_imp = np.zeros(len(mean_imp))

        df[out_col] = df[pollutant].copy()
        df.loc[missing_mask, out_col] = mean_imp[missing_mask.values]
        df[out_col] = df[out_col].clip(lower=0.0)
        df[was_col] = missing_mask & df[out_col].notna()
        df[std_col] = std_imp

        # Cleanup temp cols
        df.drop(columns=[c for c in time_feats if c in df.columns], inplace=True)

        n_imp = int(df[was_col].sum())
        return df, ImputationResult(self.name, n_imp, None, None, self._elapsed(t0), "ok")

