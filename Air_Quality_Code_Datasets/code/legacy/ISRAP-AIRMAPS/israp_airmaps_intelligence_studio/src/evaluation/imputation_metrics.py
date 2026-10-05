"""Imputation evaluation metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe(a: np.ndarray, b: np.ndarray):
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def mae(true: np.ndarray, pred: np.ndarray) -> float:
    t, p = _safe(true, pred)
    return float(np.mean(np.abs(t - p))) if len(t) > 0 else np.nan


def rmse(true: np.ndarray, pred: np.ndarray) -> float:
    t, p = _safe(true, pred)
    return float(np.sqrt(np.mean((t - p) ** 2))) if len(t) > 0 else np.nan


def smape(true: np.ndarray, pred: np.ndarray) -> float:
    t, p = _safe(true, pred)
    if len(t) == 0:
        return np.nan
    denom = (np.abs(t) + np.abs(p)) / 2
    mask = denom > 0
    return float(np.mean(np.abs(t[mask] - p[mask]) / denom[mask]) * 100) if mask.any() else np.nan


def r2(true: np.ndarray, pred: np.ndarray) -> float:
    t, p = _safe(true, pred)
    if len(t) < 2:
        return np.nan
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def bias(true: np.ndarray, pred: np.ndarray) -> float:
    t, p = _safe(true, pred)
    return float(np.mean(p - t)) if len(t) > 0 else np.nan


def evaluate_imputation(
    true: np.ndarray | pd.Series,
    pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    t = np.asarray(true, dtype=float)
    p = np.asarray(pred, dtype=float)
    return {
        "mae": mae(t, p),
        "rmse": rmse(t, p),
        "smape": smape(t, p),
        "r2": r2(t, p),
        "bias": bias(t, p),
        "n_evaluated": int(np.sum(np.isfinite(t) & np.isfinite(p))),
    }


def mask_and_evaluate(
    df: pd.DataFrame,
    pollutant: str,
    imputer,
    mask_fraction: float = 0.1,
    random_seed: int = 42,
    **imputer_kwargs,
) -> dict[str, float]:
    """Artificially mask *mask_fraction* of observed values, then evaluate."""
    rng = np.random.default_rng(random_seed)
    df_work = df.copy()
    observed_idx = df_work[df_work[pollutant].notna()].index
    if len(observed_idx) < 10:
        return {}
    n_mask = max(1, int(len(observed_idx) * mask_fraction))
    mask_idx = rng.choice(observed_idx, size=n_mask, replace=False)
    true_vals = df_work.loc[mask_idx, pollutant].values.copy()
    df_work.loc[mask_idx, pollutant] = np.nan

    df_imp, _ = imputer.impute(df_work, pollutant, **imputer_kwargs)
    imp_col = [c for c in df_imp.columns if c.startswith("imputed_")]
    if not imp_col:
        return {}
    pred_vals = df_imp.loc[mask_idx, imp_col[0]].values
    return evaluate_imputation(true_vals, pred_vals)

