"""Verify packaged bytes and independently recompute recorded prediction metrics.

Does not load models, retrain, or change recorded files. Model replay is a separate
check using the evaluation scripts and PyTorch.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ['seed', 'datetime_utc', 'site', 'gap_hours']


def verify_period(label, predictions, summary, count):
    d = pd.concat([pd.read_csv(ROOT / p) for p in predictions], ignore_index=True)
    table = pd.read_csv(ROOT / summary).set_index('method')
    assert set(d.method) == set(table.index), label
    reference = None
    for method, group in d.groupby('method'):
        assert len(group) == count and not group.duplicated(KEYS).any(), (label, method)
        assert np.isfinite(group[['truth', 'prediction']]).all().all()
        assert set(group.seed) == {11, 22, 33}
        assert set(group.site) == {32, 59, 1069}
        assert set(group.gap_hours) == {1, 3, 6}
        keys_and_truth = set(group[KEYS + ['truth']].itertuples(index=False, name=None))
        if reference is None:
            reference = keys_and_truth
        assert keys_and_truth == reference, (label, method, 'mismatched targets')
        errors = group.prediction - group.truth
        assert np.isclose(errors.abs().mean(), table.loc[method, 'MAE'], rtol=0, atol=1e-10)
        assert np.isclose(np.sqrt((errors ** 2).mean()), table.loc[method, 'RMSE'], rtol=0, atol=1e-10)
        assert table.loc[method, 'n'] == count
    print(f'{label}: {len(table)} methods, {count} equal scored keys each; MAE/RMSE verified')


def main():
    manifest = json.loads((ROOT / 'MANIFEST.sha256.json').read_text())
    for name, expected in manifest.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        assert actual == expected, f'File changed: {name}'
    print(f'Integrity: {len(manifest)} files verified')
    verify_period('October 2022', [
        'source/pkg/october_masked_predictions.csv',
        'source/pkg/timexer_results/patch_6/october_masked_predictions_timexer.csv'],
        'source/pkg/timexer_results/patch_6/comparison_all_methods.csv', 886)
    verify_period('Nov–Dec 2022', ['validation_2022/predictions_2022_nov_dec.csv'],
                  'validation_2022/comparison_2022_nov_dec.csv', 1865)
    verify_period('2024 transfer', ['validation_2024/predictions_2024.csv'],
                  'validation_2024/comparison_2024.csv', 10305)
    print('Stored-result verification passed. This check does not rerun model inference.')


if __name__ == '__main__':
    main()
