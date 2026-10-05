"""Anomaly ensemble: union, majority vote, weighted score, confidence."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Map method name → flag column
FLAG_COLS: dict[str, str] = {
    "physical_constraints": "is_physical_anomaly",
    "iqr": "is_iqr_anomaly",
    "hampel": "is_hampel_anomaly",
    "rolling_zscore": "is_rolling_z_anomaly",
    "lof": "is_lof_anomaly",
    "isolation_forest": "is_iforest_anomaly",
    "stl": "is_stl_anomaly",
    "screen": "is_screen_anomaly",
    "lstm_ndt": "is_lstm_anomaly",
    "usad": "is_usad_anomaly",
}

SCORE_COLS: dict[str, str] = {
    "physical_constraints": "physical_constraint_score",
    "iqr": "iqr_score",
    "hampel": "hampel_score",
    "rolling_zscore": "rolling_z_score",
    "lof": "lof_score",
    "isolation_forest": "iforest_score",
    "stl": "stl_score",
    "screen": "screen_score",
    "lstm_ndt": "lstm_error",
    "usad": "usad_score",
}


def build_ensemble(
    df: pd.DataFrame,
    methods: list[str],
    weights: dict[str, float] | None = None,
    majority_threshold: int = 2,
) -> pd.DataFrame:
    """Add ensemble columns to *df*.

    Added columns
    -------------
    anomaly_method_count, anomaly_union, anomaly_majority,
    anomaly_weighted_score, combined_anomaly_flag,
    anomaly_confidence, recommended_action
    """
    df = df.copy()
    available = [m for m in methods if FLAG_COLS.get(m, "x") in df.columns]
    if not available:
        df["anomaly_method_count"] = 0
        df["anomaly_union"] = False
        df["anomaly_majority"] = False
        df["anomaly_weighted_score"] = 0.0
        df["combined_anomaly_flag"] = False
        df["anomaly_confidence"] = "None"
        df["recommended_action"] = "no_anomaly"
        return df

    flag_mat = pd.concat(
        [df[FLAG_COLS[m]].fillna(False).astype(int) for m in available], axis=1
    )
    flag_mat.columns = available

    count = flag_mat.sum(axis=1)
    df["anomaly_method_count"] = count
    df["anomaly_union"] = (count > 0)
    df["anomaly_majority"] = (count >= majority_threshold)

    # ── Weighted score ────────────────────────────────────────────────────────
    w = weights or {m: 1.0 for m in available}
    total_w = sum(w.get(m, 1.0) for m in available)
    weighted = pd.Series(0.0, index=df.index)
    for m in available:
        sc_col = SCORE_COLS.get(m)
        if sc_col and sc_col in df.columns:
            sc = df[sc_col].fillna(0.0)
            sc_norm = sc / (sc.quantile(0.99) + 1e-9)
            weighted += sc_norm * w.get(m, 1.0)
    df["anomaly_weighted_score"] = (weighted / total_w).clip(0, 10)

    df["combined_anomaly_flag"] = df["anomaly_majority"]

    # ── Confidence ────────────────────────────────────────────────────────────
    def _confidence(row) -> str:
        c = row["anomaly_method_count"]
        physical = row.get("is_physical_anomaly", False)
        if physical:
            return "Very High"
        if c >= 4:
            return "High"
        if c >= 2:
            return "Medium"
        if c >= 1:
            return "Low"
        return "None"

    df["anomaly_confidence"] = df.apply(_confidence, axis=1)

    # ── Recommended action ────────────────────────────────────────────────────
    def _action(row) -> str:
        if row.get("is_physical_anomaly", False):
            return "confirm_invalid"
        c = row["anomaly_method_count"]
        if c >= 3:
            return "review"
        if c == 2:
            return "review"
        if c == 1:
            return "monitor"
        return "no_action"

    df["recommended_action"] = df.apply(_action, axis=1)

    return df


def method_agreement_matrix(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Return method × method co-flagging counts."""
    available = [m for m in methods if FLAG_COLS.get(m, "x") in df.columns]
    flags = {m: df[FLAG_COLS[m]].fillna(False).astype(bool) for m in available}
    mat = {}
    for a in available:
        mat[a] = {}
        for b in available:
            mat[a][b] = int((flags[a] & flags[b]).sum())
    return pd.DataFrame(mat)


def jaccard_matrix(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Return Jaccard similarity matrix between method flags."""
    available = [m for m in methods if FLAG_COLS.get(m, "x") in df.columns]
    flags = {m: df[FLAG_COLS[m]].fillna(False).astype(bool) for m in available}
    mat = {}
    for a in available:
        mat[a] = {}
        for b in available:
            inter = (flags[a] & flags[b]).sum()
            union = (flags[a] | flags[b]).sum()
            mat[a][b] = round(inter / union, 4) if union > 0 else 0.0
    return pd.DataFrame(mat)

