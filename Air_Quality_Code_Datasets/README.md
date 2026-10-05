# Air Quality: Code and Datasets

This package contains the available implementation source code and existing datasets. It contains no reports, model checkpoints, training logs, or result plots. No models were trained while preparing it.

## Contents

- `code/imputation/`: separate PM2.5 baseline, Random Forest, station-temporal Transformer, and TimeXer imputation implementations, plus experimental and tiered methods.
- `code/forecasting/`: original TimeXer and temporal Transformer forecasting code.
- `code/source/pkg/timexer_upstream/`: upstream TimeXer architecture and supporting layers.
- `code/offline_legacy/`: earlier multi-pollutant offline imputation experiment, including median, interpolation, Kalman, DeepMVI adaptations, and frozen MPIN adaptation.
- `code/legacy/`: earlier pipeline, application, and model implementations.
- `code/notebooks/`: available source notebooks.
- `datasets/raw/`: original AQS records, station metadata, and ERA5 hourly inputs available for this project.
- `datasets/merged/`: initial, UTC-audited, and quality-labelled 2022 merged panels; the 2024 PM2.5 panel; the existing gap metadata table.
- `datasets/imputed/`: original selected-method offline completed panel and candidate-estimate audit.
- `datasets/imputed/by_method/`: six separate completed datasets: hour median, linear interpolation, Kalman smoothing, DeepMVI random adaptation, DeepMVI outage adaptation, and frozen MPIN adaptation.
- `datasets/imputed/pm25_masked_predictions/`: saved PM2.5 masked-target prediction tables. These are validation predictions, not full completed natural-gap datasets.

## Datasets

The full 2022 imputed panels have 43,800 rows and 142 columns. Each method-specific panel uses the same 2,624 eligible natural missing cells. Original pollutant columns remain unchanged; estimates are in `*_imputed`, with `*_was_imputed` and `*_imputation_method` fields. Structural non-monitoring and ineligible gaps remain missing.

The six method panels were exported from saved estimates without retraining. They belong to the earlier multi-pollutant experiment; they must not be treated as the later PM2.5-only Transformer comparison. Offline interpolation and smoothing can use future context, so these panels are not directly suitable for causal forecasting.

PM2.5 is monitored at sites 0032, 0059, and 1069. The shared ERA5 weather must not be described as station-specific weather. The uploaded ERA5 table preserves its original clock representation; use the audited UTC tables for aligned experiments.

Read compressed CSVs directly with pandas:

```python
import pandas as pd
raw = pd.read_csv('datasets/raw/aqs_2022_original_records.csv.gz', low_memory=False)
merged = pd.read_csv('datasets/merged/aqs_era5_2022_quality_labelled.csv.gz', low_memory=False)
imputed = pd.read_csv('datasets/imputed/by_method/kalman_offline.csv.gz', low_memory=False)
```

## Code setup

From the extracted package directory:

```bash
cd code
python -m pip install -r requirements.txt
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install reformer-pytorch==1.4.4
```

Active PM2.5 and forecasting data loaders point at the bundled merged dataset. Earlier application and evaluation scripts retain their original conventions and may require data-path configuration or outputs produced by a run. Evaluation scripts that replay checkpoints require separately saved checkpoints; checkpoints are intentionally excluded from this code-and-datasets package. Neural runner commands train models when you explicitly run them.

The older DeepMVI/MPIN runner also needs author repositories under `code/methods/` and torch-geometric:

- DeepMVI: https://github.com/pbansal5/DeepMVI (commit `8d1c0eca50ef2f6f14411cc0508c47f242539a2d`)
- MPIN: https://github.com/XLI-2020/MPIN (commit `2ff3f1d92f8cae1ec10d7a0a0326992e7a362083`)
- TimeXer: https://github.com/thuml/TimeXer (commit `76011909357972bd55a27adba2e1be994d81b327`)

These external DeepMVI/MPIN sources are dependencies, not bundled files. Their included project adapters are not claims of reproducing published benchmark results.

## Upload to GitHub

Extract this ZIP and upload its `code/`, `datasets/`, and `README.md` contents. Do not upload the ZIP as a replacement for the repository contents. Gzip files use lossless compression and are readable with pandas without manual extraction.
