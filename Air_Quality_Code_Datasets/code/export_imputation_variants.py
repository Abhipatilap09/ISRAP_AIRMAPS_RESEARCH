"""Export method-specific panels from saved candidate estimates; never trains models."""
from pathlib import Path
import gzip
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parents[1] / 'datasets/imputed'
METHODS = ['hour_median', 'linear_offline', 'kalman_offline',
           'DeepMVI_random_adapted', 'DeepMVI_outage_adapted', 'MPIN_frozen_adapted']
POL = ['CO', 'NO2', 'O3', 'PM2.5', 'SO2']


def main():
    frame = pd.read_csv(HERE/'EPA_AQS_ERA5_2022_OFFLINE_IMPUTED.csv.gz', low_memory=False)
    audit = pd.read_csv(HERE/'imputation_audit_all_methods.csv')
    assert np.isfinite(audit[METHODS].to_numpy()).all()
    lookup = {(int(s), t):i for i,(s,t) in enumerate(zip(frame.site, frame.datetime_utc))}
    assert len(lookup) == len(frame)
    output = HERE/'by_method'; output.mkdir(exist_ok=True)
    for method in METHODS:
        data = frame.copy()
        for pollutant in POL:
            rows = audit[audit.pollutant == pollutant]
            indexes = [lookup[int(r.site), r.datetime_utc] for r in rows.itertuples()]
            assert data.loc[indexes, pollutant].isna().all()
            assert not data.loc[indexes, pollutant+'_structural_non_monitoring'].astype(bool).any()
            data[pollutant+'_imputed'] = data[pollutant]
            data[pollutant+'_was_imputed'] = 0
            data[pollutant+'_imputation_method'] = 'observed_or_not_filled'
            data.loc[indexes, pollutant+'_imputed'] = rows[method].to_numpy()
            data.loc[indexes, pollutant+'_was_imputed'] = 1
            data.loc[indexes, pollutant+'_imputation_method'] = method
        (output/f'{method}.csv.gz').write_bytes(gzip.compress(data.to_csv(index=False).encode(), compresslevel=9, mtime=0))
        print(method, len(data), 'rows;', len(audit), 'eligible cells filled')


if __name__ == '__main__':
    main()
