"""Base class for all imputation methods."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ImputationResult:
    method: str
    n_imputed: int
    mae: float | None
    rmse: float | None
    runtime_sec: float
    status: str
    warning: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)


class BaseImputer(ABC):
    name: str = "base"

    def __init__(self, **kwargs: Any) -> None:
        self.params = kwargs

    @abstractmethod
    def impute(
        self,
        df: pd.DataFrame,
        pollutant: str,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, ImputationResult]:
        """Impute missing values for *pollutant* in *df*."""

    def _elapsed(self, t0: float) -> float:
        return round(time.perf_counter() - t0, 3)

