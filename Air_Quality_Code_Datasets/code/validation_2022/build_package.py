from pathlib import Path
import json,hashlib,zipfile
import pandas as pd
root=Path(__file__).resolve().parent;base=root.parent
s=pd.read_csv(root/'comparison_2022_nov_dec.csv').set_index('method');old=pd.read_csv(base/'source/pkg/timexer_results/patch_6/comparison_all_methods.csv').set_index('method')
methods=['Hour median','Linear interpolation','Random Forest','PM2.5 station-temporal Transformer','TimeXer imputation adaptation'];rows='\n'.join(f'| {k} | {old.loc[k,"MAE"]:.3f} | {old.loc[k,"RMSE"]:.3f} | {s.loc[k,"MAE"]:.3f} | {s.loc[k,"RMSE"]:.3f} |' for k in methods)
october_rows='\n'.join(f'| {k} | {old.loc[k,"MAE"]:.3f} | {old.loc[k,"RMSE"]:.3f} |' for k in methods)
test_rows='\n'.join(f'| {k} | {s.loc[k,"MAE"]:.3f} | {s.loc[k,"RMSE"]:.3f} |' for k in methods)
e=pd.read_csv(root/'examples.csv');ex='\n'.join(f'| {r.datetime_utc} | {r.gap_hours} | {r.truth:.2f} | {r["Random Forest"]:.2f} | {r["PM2.5 station-temporal Transformer"]:.2f} | {r["TimeXer imputation adaptation"]:.2f} |' for _,r in e.iterrows())
text=f'''# PM2.5 2022 validation and test results

Prepared for Professor Ke Yang by Abhishek Patil.

I evaluated the saved models on November–December 2022, preserving January–September for fitting and October for checkpoint and TimeXer patch selection. I did not retrain or tune any model on the evaluation period. The station-temporal Transformer has the lowest pooled MAE and RMSE, but its advantage over Random Forest is small and uncertain.

## Results

All errors are in µg/m³. October has 886 hidden instances per method; November–December has 1,865, across masking seeds 11, 22 and 33 (637, 640 and 588 instances). Every method is scored on identical seed–timestamp–station–gap keys and labels. Masking seeds can revisit the same observation; the pooled count is not the number of independent measurements.

### Comparison 1 October 2022 validation

| Method | MAE (µg/m³) | RMSE (µg/m³) |
|---|---:|---:|
{october_rows}

October uses 886 hidden instances across seeds 11, 22 and 33 (267, 319 and 300). The station-temporal Transformer has the lowest point errors. October also selected the saved neural checkpoints and TimeXer patch setting, so these are validation results rather than an independent test.

### Comparison 2 November and December 2022 testing

| Method | MAE (µg/m³) | RMSE (µg/m³) |
|---|---:|---:|
{test_rows}

The frozen models are scored on 1,865 hidden instances. The station-temporal Transformer again has the lowest point errors. These two periods provide two separate comparisons, not two independent proofs of superiority. The November–December difference from Random Forest is very small.

In November–December, the Transformer MAE is 0.014 lower than Random Forest and 0.109 lower than interpolation. A paired seven-day block bootstrap, grouping all masking seeds within each UTC day, gives 95% intervals of −0.130 to 0.091 for Transformer minus Random Forest, and −0.259 to 0.028 for Transformer minus interpolation. Both include zero. These results support a similar performance level; they do not establish a reliable superiority over either baseline. The bootstrap does not cover variation across fresh training runs.

## What was used

The source is the unchanged quality-labelled 2022 EPA AQS–ERA5 panel. Targets are PM2.5 at stations 0032, 0059 and 1069 where observed_valid equals 1. Qualified or missing values are unavailable. The evaluation covers 1 November 2022 00:00 CST through 31 December 2022 23:00 CST (1 November 06:00 UTC through 1 January 2023 05:00 UTC), 1,464 hours.

The seven shared ERA5 inputs are u10, v10, t2m, d2m, sp, tp and blh. The verified source grid point is 29.5° N, 98.5° W. The station Transformer receives 24 hourly tokens with 15 fields: three masked PM2.5 values, three visibility flags, seven weather values and two hour encodings. Its saved checkpoint is epoch 16. The official TimeXer embedding and encoder are adapted to same-window imputation, with three station series and a reconstruction head; the frozen patch length is 6 and checkpoint is epoch 9. This is not the unchanged TimeXer forecasting model.

Both neural models were trained on January–September with train-only standardization, masked MAE, fresh masks each epoch, AdamW learning rate 0.001 and batch size 32. Each uses 64 dimensions, four heads and two layers. October selected their checkpoints. Random Forest uses 160 trees and training-fitted feature medians, visible station values at temporal offsets, seven weather fields, calendar encodings and station identity. Hour medians use January–September only. Interpolation uses the visible endpoints around a hidden gap.

Evaluation masks use 1-, 3- and 6-hour gaps wholly inside 24-hour windows, with valid observations immediately on both sides. PM2.5 context for every method is restricted to November–December; October observations cannot enter Random Forest lag features. Artificially hidden values are removed everywhere they would appear in inputs. All scores are in original concentration units.

## Three examples at station 0059

These are the earliest hidden timestamps for each gap length under seed 11, chosen chronologically rather than by prediction accuracy.

| UTC timestamp | Gap hours | True PM2.5 | RF | Station Transformer | TimeXer adaptation |
|---|---:|---:|---:|---:|---:|
{ex}

## Checks and limits

I regenerated the masks, checked exact target and label agreement for all five methods, independently recomputed MAE and RMSE, verified finite predictions and training-only neural scalers, and replayed the October neural predictions from their saved checkpoints. The largest replay discrepancy was 0.0000034 µg/m³, consistent with floating-point variation. Saved model hashes were frozen before evaluation and remained unchanged.

The screening policy retains unqualified negative observations as in the pilot. Earlier source checks identified some below-negative-detection-limit records; scientific quality review remains necessary. November–December appeared in earlier project analyses, so this period is not claimed to have never been inspected. No new architecture or setting was selected from these results. Training uses one initialization seed, and masks represent only short artificial gaps with valid endpoints. This evaluates offline imputation using visible future observations and retrospective weather; it does not validate forecasting or regional generalization.

## Reproduction

The companion ZIP contains the unchanged source panel, frozen models, original training code, evaluation runner, mask-key predictions, metrics by station/gap/seed, bootstrap intervals and SHA256 manifest. Install torch 2.6.0 CPU, numpy, pandas, scikit-learn 1.8.0, joblib, einops, reformer-pytorch and its dependencies; run `python validation_2022/evaluate.py`, then `python validation_2022/check_results.py`. The runner never retrains models. The original source panel SHA256 is 21977e03a09f9391f171f57f46f048dc4c4130e960e4b5f64dad8321bec8070e.
'''
report=base/'PM25_2022_Test_Results.md';report.write_text(text)
files=[report]+[p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[p for p in (base/'source/pkg').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
manifest={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
with zipfile.ZipFile(base/'PM25_2022_Test_Package.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in files:z.write(p,p.relative_to(base))
 z.writestr('test_sha256_manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(base/'PM25_2022_Test_Package.zip') as z:
 assert z.testzip() is None
 for n,sha in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==sha
print('2022 test report and package generated; all archive hashes verified.')
