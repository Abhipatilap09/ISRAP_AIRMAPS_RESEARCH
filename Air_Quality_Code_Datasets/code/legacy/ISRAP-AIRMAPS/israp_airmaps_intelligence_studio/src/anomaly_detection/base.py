"""Base class for all anomaly detectors."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class AnomalyResult:
    method: str
    category: str
    flag_col: str
    score_col: str
    threshold: float | None
    n_flagged: int
    pct_flagged: float
    runtime_sec: float
    status: str          # "ok" | "skipped" | "error"
    warning: str = ""
    params: dict[str, Any] = field(default_factory=dict)


class BaseDetector(ABC):
    """All detectors must implement detect() and return a consistent dict."""

    name: str = "base"
    category: str = "statistical"

    def __init__(self, **kwargs: Any) -> None:
        self.params = kwargs

    @abstractmethod
    def detect(
        self,
        df: pd.DataFrame,
        pollutant: str,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, AnomalyResult]:
        """Add flag/score columns to *df* and return updated df + result summary."""

    def _time_it(self, func, *args, **kwargs):
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        return out, time.perf_counter() - t0

    def _make_result(
        self,
        flag_col: str,
        score_col: str,
        df: pd.DataFrame,
        threshold: float | None,
        runtime: float,
        status: str = "ok",
        warning: str = "",
    ) -> AnomalyResult:
        n = int(df[flag_col].sum()) if flag_col in df.columns else 0
        pct = round(100 * n / max(len(df), 1), 3)
        return AnomalyResult(
            method=self.name,
            category=self.category,
            flag_col=flag_col,
            score_col=score_col,
            threshold=threshold,
            n_flagged=n,
            pct_flagged=pct,
            runtime_sec=round(runtime, 3),
            status=status,
            warning=warning,
            params=self.params,
        )

