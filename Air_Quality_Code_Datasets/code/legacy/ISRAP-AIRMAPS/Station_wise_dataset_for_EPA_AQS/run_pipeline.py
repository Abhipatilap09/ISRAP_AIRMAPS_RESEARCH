"""
run_pipeline.py
---------------
Main orchestrator for the ISRAP Air Quality Analysis Pipeline.

Usage:
    python run_pipeline.py [--config config.yaml]

Steps:
  1.  Load config
  2.  Setup logging
  3.  Create output directories
  4.  Load all station data + ERA5
  5.  For each station:
       a. Standardize datetime
       b. Remove duplicates
       c. Reindex hourly
       d. Physical constraint cleaning
       e. Identify structural missingness
       f. Merge ERA5
       g. Run anomaly detection (all methods, all pollutants)
       h. Run imputation pipeline
       i. Save cleaned / imputed CSVs
  6.  Generate audit CSVs
  7.  Generate figures
  8.  Run validation experiments
  9.  Generate Markdown report
 10.  Print summary
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Bootstrap: make sure src/ is importable when running from pipeline root
# ---------------------------------------------------------------------------

PIPELINE_ROOT = Path(__file__).parent.resolve()
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_output_dirs(config: dict, base_dir: Path) -> None:
    output_dir = base_dir / config.get("output_dir", "outputs")
    for sub in ["audits", "cleaned", "anomaly_flags", "imputed",
                "validation", "figures", "reports"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
    (base_dir / "logs").mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="ISRAP Air Quality Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config_path = PIPELINE_ROOT / args.config
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)

    # --- Logging ---
    from src.utils import setup_logging, ensure_dir
    log_file = PIPELINE_ROOT / config.get("log_file", "logs/pipeline.log")
    log = setup_logging(log_file=log_file, level=config.get("log_level", "INFO"))
    log.info("=" * 60)
    log.info("ISRAP Air Quality Pipeline starting")
    log.info("Config: %s", config_path)
    log.info("=" * 60)

    # --- Output directories ---
    create_output_dirs(config, PIPELINE_ROOT)
    OUTPUT_DIR = PIPELINE_ROOT / config.get("output_dir", "outputs")

    t0 = time.time()

    # -----------------------------------------------------------------------
    # Step 4: Load data
    # -----------------------------------------------------------------------
    from src.data_loader import load_all_stations, load_era5_data, merge_era5_with_station
    from src.preprocessing import (
        standardize_datetime, remove_duplicates, reindex_hourly,
        physical_constraint_cleaning, identify_structural_missingness,
    )
    from src.anomaly_detection import run_anomaly_detection
    from src.imputation import run_imputation_pipeline

    pollutants = config.get("pollutants", ["CO", "NO2", "O3", "PM2.5", "SO2"])
    era5_features = config.get("era5_features",
                               ["temp_c", "wind_speed", "blh", "relative_humidity",
                                "surface_pressure_hpa", "precip_mm", "u10", "v10"])

    log.info("Loading ERA5 ...")
    era5_df = load_era5_data(config)
    log.info("ERA5 loaded: %d rows", len(era5_df))

    log.info("Loading station CSVs ...")
    raw_stations = load_all_stations(config)
    log.info("Loaded %d stations", len(raw_stations))

    # -----------------------------------------------------------------------
    # Step 5: Per-station pipeline
    # -----------------------------------------------------------------------
    station_info: dict = {}          # site_id -> metadata dict
    all_processed: dict = {}         # site_id -> processed DataFrame
    all_anomaly_results: dict = {}   # site_id -> pollutant -> df
    all_imputation_results: dict = {}

    for site_id, (raw_df, schema_findings) in raw_stations.items():
        log.info("--- Processing Site %s ---", site_id)
        n_raw = len(raw_df)

        # 5a. Standardize datetime
        df = standardize_datetime(raw_df)

        # 5b. Remove duplicates
        n_before_dedup = len(df)
        df = remove_duplicates(df)
        n_dups_removed = n_before_dedup - len(df)

        # 5c. Reindex hourly
        n_before_reindex = len(df)
        df = reindex_hourly(df)
        n_hours_added = len(df) - n_before_reindex

        # 5d. Physical constraint cleaning
        df = physical_constraint_cleaning(df, pollutants)

        # 5e. Structural missingness
        struct_missing = identify_structural_missingness(
            df, pollutants,
            threshold=config.get("structural_missing_threshold", 0.95)
        )

        # 5f. Merge ERA5
        df_merged = merge_era5_with_station(df, era5_df)

        # 5g. Anomaly detection
        avail_pollutants = [p for p in pollutants if p in df.columns
                            and p not in struct_missing
                            and df[p].notna().sum() > 0]
        pol_anom_results: dict = {}
        for pol in avail_pollutants:
            log.info("  Anomaly detection: Site %s / %s", site_id, pol)
            df_merged = run_anomaly_detection(df_merged, pol, era5_features, config)
            pol_anom_results[pol] = df_merged.copy()

        all_anomaly_results[site_id] = pol_anom_results

        # 5h. Imputation
        pol_imp_results: dict = {}
        for pol in avail_pollutants:
            log.info("  Imputation: Site %s / %s", site_id, pol)
            df_merged = run_imputation_pipeline(
                df_merged, pol, era5_df, struct_missing, config
            )
            pol_imp_results[pol] = df_merged.copy()

        all_imputation_results[site_id] = pol_imp_results

        # 5i. Save cleaned CSVs
        cleaned_path = OUTPUT_DIR / "cleaned" / f"site{site_id}_cleaned.csv"
        imputed_path = OUTPUT_DIR / "imputed" / f"site{site_id}_imputed.csv"

        # Cleaned = post physical-constraint, pre-imputation (original values)
        orig_cols = [c for c in df_merged.columns
                     if c.endswith("_original") or c in pollutants
                     or c in ["site", "state_code", "county_code"]]
        cleaned_out = df_merged[[c for c in orig_cols if c in df_merged.columns]].copy()
        # Restore original values before imputation
        for pol in pollutants:
            oc = f"{pol}_original"
            if oc in df_merged.columns:
                cleaned_out[pol] = df_merged[oc]
        cleaned_out.to_csv(cleaned_path)

        # Imputed = final values
        imp_cols = [c for c in df_merged.columns
                    if not c.startswith("flag_") and not c.startswith("anomaly_")]
        df_merged[imp_cols].to_csv(imputed_path)

        log.info(
            "  Site %s: raw=%d, dedup_removed=%d, hours_added=%d, "
            "avail_pollutants=%s, struct_missing=%s",
            site_id, n_raw, n_dups_removed, n_hours_added,
            avail_pollutants, list(struct_missing),
        )

        # Collect metadata
        station_info[site_id] = {
            "df": df_merged,
            "station_file": schema_findings.get("station_file", ""),
            "n_rows_raw": n_raw,
            "available_pollutants": avail_pollutants,
            "structural_missing": list(struct_missing),
            "n_duplicates_removed": n_dups_removed,
            "n_hours_added_by_reindex": n_hours_added,
        }
        all_processed[site_id] = df_merged

    # -----------------------------------------------------------------------
    # Step 6: Audit CSVs
    # -----------------------------------------------------------------------
    log.info("Generating audit CSVs ...")
    from src.reporting import (
        generate_data_audit_csv, generate_availability_matrix,
        generate_negative_summary, generate_gap_summary,
        generate_anomaly_summary, generate_imputation_summary,
        generate_markdown_report,
    )

    # For availability / gap reports we need the processed DFs
    processed_dfs = {sid: info["df"] for sid, info in station_info.items()}

    generate_data_audit_csv(station_info, OUTPUT_DIR)
    generate_availability_matrix(processed_dfs, pollutants, OUTPUT_DIR)
    neg_summary = generate_negative_summary(processed_dfs, pollutants, OUTPUT_DIR)
    generate_gap_summary(processed_dfs, pollutants, OUTPUT_DIR)
    generate_anomaly_summary(all_anomaly_results, OUTPUT_DIR)
    generate_imputation_summary(all_imputation_results, OUTPUT_DIR)

    # -----------------------------------------------------------------------
    # Step 7: Figures
    # -----------------------------------------------------------------------
    log.info("Generating figures ...")
    from src.visualization import (
        plot_missingness_heatmap, plot_raw_timeseries_with_anomalies,
        plot_stl_decomposition, plot_before_after_imputation,
        plot_negative_value_summary, plot_anomaly_method_comparison,
    )
    from src.gap_analysis import availability_matrix as avail_matrix_fn

    avail_df = avail_matrix_fn(processed_dfs, pollutants)
    plot_missingness_heatmap(avail_df, OUTPUT_DIR)
    plot_negative_value_summary(neg_summary, OUTPUT_DIR)

    for site_id, df in all_processed.items():
        for pol in pollutants:
            if pol not in df.columns:
                continue
            if df[pol].notna().sum() == 0:
                continue
            plot_raw_timeseries_with_anomalies(df, pol, site_id, OUTPUT_DIR)
            plot_stl_decomposition(df, pol, site_id, config.get("stl_period", 24), OUTPUT_DIR)
            plot_before_after_imputation(df, pol, site_id, OUTPUT_DIR)
            plot_anomaly_method_comparison(df, pol, site_id, OUTPUT_DIR)

    # -----------------------------------------------------------------------
    # Step 8: Validation
    # -----------------------------------------------------------------------
    log.info("Running validation experiments ...")
    from src.validation import run_validation_experiment, generate_leaderboard
    import pandas as pd

    all_val_metrics = []
    for site_id, df in all_processed.items():
        for pol in pollutants:
            if pol not in df.columns or df[pol].notna().sum() < 100:
                continue
            struct_miss = set(station_info[site_id].get("structural_missing", []))
            if pol in struct_miss:
                continue
            log.info("  Validation: Site %s / %s", site_id, pol)
            try:
                metrics_df = run_validation_experiment(df, pol, era5_df, config)
                if not metrics_df.empty:
                    metrics_df["site_id"] = site_id
                    all_val_metrics.append(metrics_df)
            except Exception as exc:
                log.warning("Validation failed for Site %s / %s: %s", site_id, pol, exc)

    leaderboard = generate_leaderboard(all_val_metrics)
    val_path = OUTPUT_DIR / "validation" / "validation_leaderboard.csv"
    leaderboard.to_csv(val_path, index=False)
    log.info("Validation leaderboard → %s", val_path)

    # Validation figure
    if not leaderboard.empty:
        from src.visualization import plot_validation_comparison
        plot_validation_comparison(leaderboard, OUTPUT_DIR)

    # Save all validation metrics
    if all_val_metrics:
        all_val_df = pd.concat(all_val_metrics, ignore_index=True)
        all_val_df.to_csv(OUTPUT_DIR / "validation" / "validation_all_metrics.csv", index=False)

    # -----------------------------------------------------------------------
    # Step 9: Markdown report
    # -----------------------------------------------------------------------
    log.info("Generating report ...")
    anom_summary_path = OUTPUT_DIR / "anomaly_flags" / "anomaly_summary.csv"
    imp_summary_path = OUTPUT_DIR / "imputed" / "imputation_summary.csv"

    anom_summary_df = pd.read_csv(anom_summary_path) if anom_summary_path.exists() else pd.DataFrame()
    imp_summary_df = pd.read_csv(imp_summary_path) if imp_summary_path.exists() else pd.DataFrame()

    neg_rows = neg_summary.to_dict("records") if isinstance(neg_summary, pd.DataFrame) else []

    findings = {
        "audit": {"n_stations": len(station_info), "era5_rows": len(era5_df)},
        "negative_values": neg_rows,
        "anomaly_summary": anom_summary_df,
        "imputation_summary": imp_summary_df,
        "validation_leaderboard": leaderboard,
    }
    generate_markdown_report(findings, OUTPUT_DIR)

    # -----------------------------------------------------------------------
    # Step 10: Summary
    # -----------------------------------------------------------------------
    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("Pipeline complete in %.1f seconds", elapsed)
    log.info("Output directory: %s", OUTPUT_DIR)
    log.info("=" * 60)

    print("\n" + "=" * 60)
    print("ISRAP Air Quality Pipeline – COMPLETE")
    print(f"  Elapsed:   {elapsed:.1f}s")
    print(f"  Stations:  {len(station_info)}")
    print(f"  Output:    {OUTPUT_DIR}")
    print("=" * 60)
    print("\nKey outputs:")
    print(f"  Audit:      {OUTPUT_DIR / 'audits'}")
    print(f"  Cleaned:    {OUTPUT_DIR / 'cleaned'}")
    print(f"  Imputed:    {OUTPUT_DIR / 'imputed'}")
    print(f"  Anomalies:  {OUTPUT_DIR / 'anomaly_flags'}")
    print(f"  Figures:    {OUTPUT_DIR / 'figures'}")
    print(f"  Validation: {OUTPUT_DIR / 'validation'}")
    print(f"  Report:     {OUTPUT_DIR / 'reports' / 'analysis_report.md'}")
    print(f"  Log:        {log_file}")
    print()


if __name__ == "__main__":
    main()

