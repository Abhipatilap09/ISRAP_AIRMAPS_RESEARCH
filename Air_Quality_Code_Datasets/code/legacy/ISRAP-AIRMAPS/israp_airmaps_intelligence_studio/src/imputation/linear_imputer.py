"""Linear interpolation imputer (for short gaps ≤ 2 hours by default)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.imputation.base import BaseImputer, ImputationResult


class LinearImputer(BaseImputer):
    name = "linear"

    def impute(
        self,
        df: pd.DataFrame,
        pollutant: str,
        max_gap: int = 2,
        allow_extrapolation: bool = False,
        **_,
    ) -> tuple[pd.DataFrame, ImputationResult]:
        df = df.copy()
        out_col = "imputed_linear"
        was_col = "was_imputed_linear"
        t0 = time.perf_counter()

        if pollutant not in df.columns:
            df[out_col] = np.nan
            df[was_col] = False
            return df, ImputationResult(self.name, 0, None, None, self._elapsed(t0),
                                        "error", f"{pollutant} not in dataframe")

        col = df[pollutant].copy()
        mask = col.isna()

        # Only interpolate within runs ≤ max_gap
        # Use "time" method only if datetime is the index; otherwise use "linear"
        if "datetime" in df.columns:
            col_indexed = col.copy()
            col_indexed.index = pd.to_datetime(df["datetime"])
            interpolated_indexed = col_indexed.interpolate(
                method="time",
                limit=max_gap,
                limit_direction="forward" if not allow_extrapolation else "both",
                limit_area="inside" if not allow_extrapolation else None,
            )
            interpolated = pd.Series(interpolated_indexed.values, index=col.index)
        else:
            interpolated = col.interpolate(
                method="linear",
                limit=max_gap,
                limit_direction="forward" if not allow_extrapolation else "both",
                limit_area="inside" if not allow_extrapolation else None,
            )

        # Clip physically impossible negatives
        interpolated = interpolated.clip(lower=0.0)

        df[out_col] = interpolated
        df[was_col] = mask & interpolated.notna()
        n_imputed = int(df[was_col].sum())

        return df, ImputationResult(self.name, n_imputed, None, None, self._elapsed(t0), "ok")

