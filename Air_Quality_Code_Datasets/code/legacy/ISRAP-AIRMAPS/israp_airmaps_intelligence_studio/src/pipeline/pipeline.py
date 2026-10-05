"""Full processing pipeline orchestrator.

Runs all phases in order:
    ingest → clean → detect → classify_gaps → impute → export

Each phase records its status, runtime, and output path.
This module is UI-independent and can be called from Streamlit or CLI.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    EXPORTS_DIR, LOGS_DIR, POLLUTANTS, RANDOM_SEED, STATIONS,
    EXCLUDED_BY_DEFAULT,
)
from src.data_loader import load_all_stations, load_era5, merge_era5
from src.gap_analysis import find_gaps, gap_summary
from src.anomaly_detection.physical_constraints import PhysicalConstraintDetector
from src.anomaly_detection.iqr_detector import IQRDetector
from src.anomaly_detection.hampel_detector import HampelDetector
from src.anomaly_detection.ensemble import build_ensemble
from src.imputation.linear_imputer import LinearImputer
from src.imputation.era5_regression import ERA5RegressionImputer
from src.imputation.mice_imputer import MICEImputer

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    stage: str
    status: str           # "ok" | "skipped" | "error"
    detail: str = ""
    runtime_sec: float = 0.0
    warning: str = ""
    output_path: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    pipeline_id: str
    started_at: str
    finished_at: str
    config: dict[str, Any]
    stages: list[StageResult] = field(default_factory=list)
    total_anomalies_flagged: int = 0
    total_values_imputed: int = 0
    stations_processed: int = 0
    status: str = "ok"


def _gap_aware_imputer(gap_length: int, strategy: str):
    """Select imputation method based on gap length and strategy."""
    if strategy == "Linear only":
        return LinearImputer(), {"max_gap": max(gap_length, 2)}
    if strategy == "ERA5 RF only":
        return ERA5RegressionImputer(), {"model_name": "random_forest"}
    if strategy == "MICE only":
        return MICEImputer(), {"max_iter": 10}
    # "Auto (gap-aware)"
    if gap_length <= 2:
        return LinearImputer(), {"max_gap": 2}
    if gap_length <= 24:
        return ERA5RegressionImputer(), {"model_name": "ridge"}
    return MICEImputer(), {"max_iter": 10}


def run_pipeline(
    station_keys: list[str] | None = None,
    anomaly_methods: list[str] | None = None,
    imputation_strategy: str = "Auto (gap-aware)",
    exclude_station3_so2: bool = True,
    exec_mode: str = "Fast",
    random_seed: int = RANDOM_SEED,
    export_dir: Path | None = None,
    pipeline_id: str | None = None,
    station_dir: Path | None = None,
    progress_callback=None,
) -> PipelineResult:
    """Run the complete ISRAP processing pipeline.

    Parameters
    ----------
    station_keys : list of station keys to process (None = all)
    anomaly_methods : anomaly detection methods to apply
    imputation_strategy : "Auto (gap-aware)" | "Linear only" | "ERA5 RF only" | "MICE only"
    exclude_station3_so2 : whether to exclude Station 3 SO₂ (default True)
    exec_mode : "Fast" or "Full Research"
    random_seed : reproducibility seed
    export_dir : where to write CSV outputs (None = EXPORTS_DIR)
    pipeline_id : unique run identifier (auto-generated if None)
    station_dir : override for station CSV directory
    progress_callback : optional callable(stage_name, fraction_done) for UI updates
    """
    import uuid
    from datetime import datetime

    pid = pipeline_id or str(uuid.uuid4())[:8]
    started = datetime.utcnow().isoformat()
    export_dir = export_dir or EXPORTS_DIR
    export_dir.mkdir(parents=True, exist_ok=True)

    anomaly_methods = anomaly_methods or ["Physical Constraints", "IQR Fence", "Hampel Filter"]
    station_keys = station_keys or list(STATIONS.keys())

    cfg = {
        "pipeline_id": pid,
        "station_keys": station_keys,
        "anomaly_methods": anomaly_methods,
        "imputation_strategy": imputation_strategy,
        "exclude_station3_so2": exclude_station3_so2,
        "exec_mode": exec_mode,
        "random_seed": random_seed,
    }

    result = PipelineResult(
        pipeline_id=pid,
        started_at=started,
        finished_at="",
        config=cfg,
    )

    def _stage(name: str, fraction: float, func) -> StageResult:
        t0 = time.perf_counter()
        logger.info("[Pipeline %s] Starting stage: %s", pid, name)
        if progress_callback:
            progress_callback(name, fraction)
        try:
            sr = func()
            sr.runtime_sec = round(time.perf_counter() - t0, 2)
            logger.info("[Pipeline %s] %s completed in %.1fs", pid, name, sr.runtime_sec)
        except Exception as exc:
            sr = StageResult(name, "error", warning=str(exc),
                             runtime_sec=round(time.perf_counter() - t0, 2))
            logger.exception("[Pipeline %s] %s failed: %s", pid, name, exc)
        result.stages.append(sr)
        return sr

    # ── Stage 1: Load data ────────────────────────────────────────────────────
    station_data: dict[str, pd.DataFrame] = {}
    era5_df: pd.DataFrame | None = None

    def _load():
        nonlocal station_data, era5_df
        _kw = {} if station_dir is None else {"station_dir": station_dir}
        all_stations = load_all_stations(**_kw)
        era5_df = load_era5()
        station_data = {
            k: merge_era5(all_stations[k], era5_df)
            for k in station_keys if k in all_stations
        }
        n = len(station_data)
        result.stations_processed = n
        return StageResult("Data Ingestion", "ok", f"Loaded {n} stations + ERA5")

    _stage("Data Ingestion", 0.05, _load)
    if not station_data:
        result.status = "error"
        result.finished_at = __import__("datetime").datetime.utcnow().isoformat()
        return result

    # ── Stage 2: Validate hourly alignment ────────────────────────────────────
    def _validate():
        total = sum(len(df) for df in station_data.values())
        return StageResult("Hourly Alignment", "ok", f"{total:,} hourly rows")

    _stage("Hourly Alignment", 0.10, _validate)

    # ── Stage 3: Structural missingness ───────────────────────────────────────
    excluded_channels: list[str] = []

    def _structural():
        for skey, df in station_data.items():
            for p in POLLUTANTS:
                if p in df.columns and df[p].notna().mean() < 0.01:
                    excluded_channels.append(f"{skey}/{p}")
        if exclude_station3_so2 and "station3" in station_keys:
            tag = "station3/SO2 (user excluded)"
            if tag not in excluded_channels:
                excluded_channels.append(tag)
        return StageResult("Structural Missingness", "ok",
                           f"{len(excluded_channels)} excluded channels",
                           metrics={"excluded": excluded_channels})

    _stage("Structural Missingness", 0.15, _structural)

    # ── Stage 4: ERA5 coverage check ─────────────────────────────────────────
    def _era5_check():
        era5_cols = [c for c in ["temp_c", "wind_speed", "blh"] if any(c in df.columns for df in station_data.values())]
        return StageResult("ERA5 Merge", "ok", f"{len(era5_cols)} ERA5 vars matched")

    _stage("ERA5 Merge", 0.20, _era5_check)

    # ── Stage 5: Physical cleaning ────────────────────────────────────────────
    phys_total = 0

    def _physical():
        nonlocal phys_total
        det = PhysicalConstraintDetector()
        for skey, df in station_data.items():
            for p in POLLUTANTS:
                if _is_skipped(skey, p, excluded_channels, exclude_station3_so2):
                    continue
                if p in df.columns and df[p].notna().mean() > 0.01:
                    _, r = det.detect(df, p)
                    phys_total += r.n_flagged
        return StageResult("Physical Cleaning", "ok", f"{phys_total:,} values flagged",
                           metrics={"n_flagged": phys_total})

    _stage("Physical Cleaning", 0.30, _physical)
    result.total_anomalies_flagged += phys_total

    # ── Stage 6: Anomaly detection ────────────────────────────────────────────
    anomaly_total = 0

    def _detect():
        nonlocal anomaly_total
        for skey, df in station_data.items():
            for p in POLLUTANTS:
                if _is_skipped(skey, p, excluded_channels, exclude_station3_so2):
                    continue
                if p not in df.columns or df[p].notna().mean() <= 0.01:
                    continue
                if "IQR Fence" in anomaly_methods:
                    _, r = IQRDetector().detect(df, p)
                    anomaly_total += r.n_flagged
                if "Hampel Filter" in anomaly_methods:
                    _, r = HampelDetector().detect(df, p)
                    anomaly_total += r.n_flagged
        return StageResult("Anomaly Detection", "ok", f"{anomaly_total:,} flags",
                           metrics={"n_flagged": anomaly_total})

    _stage("Anomaly Detection", 0.45, _detect)
    result.total_anomalies_flagged += anomaly_total

    # ── Stage 7: Gap classification ───────────────────────────────────────────
    def _gaps():
        gap_df = gap_summary(station_data)
        n = len(gap_df)
        out = export_dir / f"pipeline_{pid}_gaps.csv"
        gap_df.to_csv(out, index=False)
        return StageResult("Gap Classification", "ok", f"{n} gap records", output_path=str(out))

    _stage("Gap Classification", 0.55, _gaps)

    # ── Stage 8: Imputation ───────────────────────────────────────────────────
    imp_total = 0
    imputed_frames: dict[str, pd.DataFrame] = {}

    def _impute():
        nonlocal imp_total
        for skey, df in station_data.items():
            df_imp = df.copy()
            for p in POLLUTANTS:
                if _is_skipped(skey, p, excluded_channels, exclude_station3_so2):
                    continue
                if p not in df.columns or df[p].notna().mean() <= 0.01 or not df[p].isna().any():
                    continue
                try:
                    gaps = find_gaps(df, p)
                    max_gap = int(gaps["gap_length"].max()) if not gaps.empty else 2
                    imp, params = _gap_aware_imputer(max_gap, imputation_strategy)
                    df_imp, r = imp.impute(df_imp, p, **params)
                    imp_total += r.n_imputed
                except Exception as exc:
                    logger.warning("Imputation failed for %s/%s: %s", skey, p, exc)
            imputed_frames[skey] = df_imp
        return StageResult("Imputation", "ok", f"{imp_total:,} values imputed",
                           metrics={"n_imputed": imp_total})

    _stage("Imputation", 0.75, _impute)
    result.total_values_imputed = imp_total

    # ── Stage 9: Export ───────────────────────────────────────────────────────
    def _export():
        paths = []
        frames = imputed_frames if imputed_frames else station_data
        for skey, df in frames.items():
            out = export_dir / f"pipeline_{pid}_{skey}.csv"
            keep_cols = (
                ["datetime", "station_key"]
                + [p for p in POLLUTANTS if p in df.columns]
                + [c for c in df.columns if c.startswith("imputed_") or c.startswith("is_")]
            )
            df[[c for c in keep_cols if c in df.columns]].to_csv(out, index=False)
            paths.append(str(out))
        # Write pipeline manifest
        manifest = {
            "pipeline_id": pid,
            "config": cfg,
            "outputs": paths,
            "total_anomalies": result.total_anomalies_flagged,
            "total_imputed": result.total_values_imputed,
        }
        manifest_path = export_dir / f"pipeline_{pid}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return StageResult("Export", "ok", f"{len(paths)} station files exported",
                           output_path=str(manifest_path))

    _stage("Export", 0.95, _export)

    if progress_callback:
        progress_callback("Complete", 1.0)

    result.finished_at = __import__("datetime").datetime.utcnow().isoformat()
    result.status = "ok"
    return result


def _is_skipped(
    station_key: str,
    pollutant: str,
    excluded_channels: list[str],
    exclude_station3_so2: bool,
) -> bool:
    if f"{station_key}/{pollutant}" in excluded_channels:
        return True
    if exclude_station3_so2 and station_key == "station3" and pollutant == "SO2":
        return True
    return False

