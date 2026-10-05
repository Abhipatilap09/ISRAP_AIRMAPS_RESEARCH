"""Original, unmodified TimeXer versus a temporal Transformer: causal PM2.5 forecasting.

Run from the repository root. No future pollutant or weather is an input.
The upstream Model is imported directly, with its original normalization and head.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'source/pkg/timexer_upstream'
SITES = [32, 59, 1069]
WEATHER = ['era5_u10', 'era5_v10', 'era5_t2m', 'era5_d2m', 'era5_sp', 'era5_tp', 'era5_blh']
SOURCE_SHA = 'b334d7869544d0a5de7a35501d0342d7bd0be4ebda06ccc045855a42819728c0'
CONFIG = dict(task_name='short_term_forecast', features='MS', seq_len=24,
              pred_len=6, use_norm=True, patch_len=6, enc_in=18, d_model=64,
              dropout=0.1, embed='timeF', freq='h', factor=3, n_heads=4,
              d_ff=128, activation='gelu', e_layers=2)


def prepare():
    source = ROOT.parent / 'datasets/merged/aqs_era5_2022_quality_labelled.csv.gz'
    if not source.exists():
        source = source.with_suffix('')
    df = pd.read_csv(source, low_memory=False)
    df['datetime_utc'] = pd.to_datetime(df.datetime_utc, utc=True)
    times = pd.DatetimeIndex(sorted(df.datetime_utc.unique()))
    assert len(times) == 8760 and (np.diff(times.asi8) == 3600 * 10**9).all()
    frame = df[df.site == SITES[0]].set_index('datetime_utc').reindex(times)
    weather = frame[WEATHER].to_numpy(float)
    assert np.isfinite(weather).all()
    assert (frame.evaluation_split.iloc[:6552] == 'train').all()
    assert (frame.evaluation_split.iloc[6552:7296] == 'validation').all()
    assert (frame.evaluation_split.iloc[7296:] == 'test').all()
    pm = np.full((8760, 3), np.nan)
    for j, site in enumerate(SITES):
        s = df[df.site == site].set_index('datetime_utc').reindex(times)
        assert not s['PM2.5_structural_non_monitoring'].fillna(0).astype(bool).any()
        pm[:, j] = np.where(s['PM2.5_observed_valid'].fillna(0).eq(1),
                            pd.to_numeric(s['PM2.5'], errors='coerce'), np.nan)
    mu, sd = np.nanmean(pm[:6552], 0), np.nanstd(pm[:6552], 0)
    assert (sd > 0).all()
    wm, ws = weather[:6552].mean(0), weather[:6552].std(0)
    ws[ws == 0] = 1
    z = np.nan_to_num((pm - mu) / sd).astype('float32')
    observed = np.isfinite(pm).astype('float32')
    wz = ((weather - wm) / ws).astype('float32')
    hour = ((times - pd.Timedelta(hours=6)).hour).to_numpy()
    clock = np.column_stack([np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)])
    sets = {}
    for period, lo, hi in [('train', 24, 6552), ('October', 6552, 7296),
                           ('November-December', 7296, 8760)]:
        xs, ys, masks, origins, stations = [], [], [], [], []
        for origin in range(lo, hi - 5, 6):
            assert origin - 24 >= 0 and origin + 6 <= hi
            for j in range(3):
                others = [k for k in range(3) if k != j]
                order = others + [j]
                sl = slice(origin-24, origin)
                identity = np.tile(np.eye(3)[j], (24, 1))
                # Target column MUST be last for the original MS forecast method.
                x = np.column_stack([z[sl][:, others], observed[sl][:, order],
                                      wz[sl], clock[sl], identity, z[sl, j]])
                y = (pm[origin:origin+6, j] - mu[j]) / sd[j]
                mask = np.isfinite(y)
                if period == 'train' and not mask.any():
                    continue
                xs.append(x); ys.append(np.nan_to_num(y)); masks.append(mask)
                origins.append(origin); stations.append(j)
        sets[period] = dict(x=np.array(xs, dtype='float32'), y=np.array(ys, dtype='float32'),
                            mask=np.array(masks), origins=np.array(origins), sites=np.array(stations))
        assert sets[period]['x'].shape[1:] == (24, 18)
        assert np.isfinite(sets[period]['x']).all()
    return times, pm, hour, mu, sd, wm, ws, sets


def records(period, method, pred, data, times, pm):
    rows = []
    for i, (origin, j) in enumerate(zip(data['origins'], data['sites'])):
        for h in range(6):
            if data['mask'][i, h]:
                rows.append(dict(period=period, method=method, origin_utc=times[origin].isoformat(),
                                 last_input_utc=times[origin-1].isoformat(), site=SITES[j],
                                 horizon=h+1, target_utc=times[origin+h].isoformat(),
                                 truth=float(pm[origin+h, j]), prediction=float(pred[i, h])))
    return rows


def verify_and_summarize(predictions):
    keys = ['origin_utc', 'site', 'horizon', 'target_utc']
    summaries = []
    for period, group in predictions.groupby('period'):
        reference = None
        for method, g in group.groupby('method'):
            assert not g.duplicated(keys).any()
            assert np.isfinite(g[['truth', 'prediction']]).all().all()
            assert (pd.to_datetime(g.last_input_utc) < pd.to_datetime(g.origin_utc)).all()
            assert (pd.to_datetime(g.target_utc) >= pd.to_datetime(g.origin_utc)).all()
            current = set(g[keys+['truth']].itertuples(index=False, name=None))
            if reference is None: reference = current
            assert current == reference, (period, method, 'unequal evaluation targets')
            error = g.prediction.to_numpy() - g.truth.to_numpy()
            summaries.append(dict(period=period, method=method, n=len(g),
                                  MAE=float(np.abs(error).mean()), RMSE=float(np.sqrt((error**2).mean()))))
    return pd.DataFrame(summaries)


def run(args):
    times, pm, hour, mu, sd, wm, ws, sets = prepare()
    source = UPSTREAM / 'models/TimeXer.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA, 'Upstream model changed'
    protocol = dict(config=CONFIG, upstream_commit='76011909357972bd55a27adba2e1be994d81b327',
                    upstream_model_sha256=SOURCE_SHA, seed=172, optimizer='AdamW', learning_rate=0.001,
                    batch_size=32, max_epochs=args.max_epochs, patience=18,
                    train_hours=6552, validation_hours=744, test_hours=1464,
                    forecast_stride_hours=6, loss='masked standardized MAE',
                    checkpoint_selection='October pooled MAE in micrograms per cubic metre',
                    train_samples=len(sets['train']['x']),
                    scored_targets={k:int(v['mask'].sum()) for k,v in sets.items()},
                    training_pm_mean=mu.tolist(), training_pm_std=sd.tolist(),
                    training_weather_mean=wm.tolist(), training_weather_std=ws.tolist(),
                    feature_order=['other_station_1_PM25', 'other_station_2_PM25',
                                   'other_station_1_observed', 'other_station_2_observed', 'target_observed']
                                  + WEATHER + ['hour_sin', 'hour_cos', 'target_site_32',
                                               'target_site_59', 'target_site_1069', 'target_PM25_last'],
                    checks=['only observed_valid PM2.5 scored', 'all features strictly before forecast origin',
                            'all transforms fitted on January-September only', 'equal scored keys across methods',
                            'original upstream Model/head/normalization unchanged'],
                    limitations=['single initialization, no hyperparameter search',
                                 'October selects checkpoints and is not independent test evidence',
                                 '2022 periods previously examined in imputation experiments',
                                 'ERA5 is retrospective reanalysis; release-time availability is not audited'])
    print(json.dumps(protocol, indent=2), flush=True)
    if args.check_data: return
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.ensemble import RandomForestRegressor
    import joblib
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(UPSTREAM))
    from models.TimeXer import Model

    class TemporalTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Linear(18, 64)
            self.position = nn.Parameter(torch.zeros(1, 24, 64))
            layer = nn.TransformerEncoderLayer(64, 4, 128, 0.1, activation='gelu', batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.head = nn.Linear(24*64, 6)
        def forward(self, x):
            return self.head(self.encoder(self.embedding(x) + self.position).flatten(1))

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    checkpoints = {}
    train, val = sets['train'], sets['October']
    for name in ['Original TimeXer', 'Temporal Transformer']:
        random.seed(172); np.random.seed(172); torch.manual_seed(172)
        model = Model(SimpleNamespace(**CONFIG)) if name == 'Original TimeXer' else TemporalTransformer()
        def forward(x):
            return model(x, None, None, None).squeeze(-1) if name == 'Original TimeXer' else model(x)
        def predict(d):
            model.eval(); chunks = []
            with torch.no_grad():
                for start in range(0, len(d['x']), 128):
                    chunks.append(forward(torch.from_numpy(d['x'][start:start+128])).numpy())
            p = np.concatenate(chunks)
            return p * sd[d['sites'], None] + mu[d['sites'], None]
        dataset = TensorDataset(torch.from_numpy(train['x']), torch.from_numpy(train['y']),
                                torch.from_numpy(train['mask']))
        loader = DataLoader(dataset, batch_size=32, shuffle=True,
                            generator=torch.Generator().manual_seed(172))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        best = float('inf'); history = []; best_state = None; best_epoch = 0
        for epoch in range(1, args.max_epochs+1):
            model.train(); total = count = 0
            for x, y, mask in loader:
                optimizer.zero_grad(); p = forward(x)
                loss = (p-y).abs()[mask].mean()
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                total += float(loss.detach()) * int(mask.sum()); count += int(mask.sum())
            vp = predict(val)
            truth = val['y']*sd[val['sites'], None] + mu[val['sites'], None]
            mae = float(np.abs(vp-truth)[val['mask']].mean())
            history.append(dict(epoch=epoch, train_normalized_MAE=total/count, validation_MAE=mae))
            print(f'{name}: epoch={epoch} train_MAE={total/count:.6f} October_MAE={mae:.6f}', flush=True)
            if mae < best - 1e-4:
                best, best_epoch = mae, epoch
                best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
            if epoch - best_epoch >= 18: break
        assert best_state is not None
        model.load_state_dict(best_state)
        slug = 'timexer' if name == 'Original TimeXer' else 'temporal_transformer'
        checkpoints[name] = dict(best_epoch=best_epoch, trained_epochs=len(history), October_MAE=best,
                                 parameters=sum(p.numel() for p in model.parameters()))
        torch.save(dict(state_dict=best_state, protocol=protocol, selection=checkpoints[name]), out/f'{slug}.pt')
        pd.DataFrame(history).to_csv(out/f'{slug}_loss_history.csv', index=False)
        for period in ['October', 'November-December']:
            all_rows.extend(records(period, name, predict(sets[period]), sets[period], times, pm))

    # Each horizon forest uses exactly the same past input channels as the neural methods.
    forests = []
    for h in range(6):
        eligible = train['mask'][:, h]
        rf = RandomForestRegressor(n_estimators=160, max_depth=16, min_samples_leaf=3,
                                   max_features=0.8, random_state=702+h, n_jobs=2)
        rf.fit(train['x'][eligible].reshape(eligible.sum(), -1), train['y'][eligible, h])
        forests.append(rf)
    for horizon, forest in enumerate(forests, 1):
        joblib.dump(forest, out/f'random_forest_horizon_{horizon}.joblib', compress=3)
    med = {(j,h): float(np.nanmedian(pm[:6552,j][hour[:6552] == h])) for j in range(3) for h in range(24)}
    for period in ['October', 'November-December']:
        d = sets[period]
        rf = np.column_stack([m.predict(d['x'].reshape(len(d['x']), -1)) for m in forests])
        rf = rf * sd[d['sites'], None] + mu[d['sites'], None]
        persistence, median = [], []
        for origin,j in zip(d['origins'], d['sites']):
            past = pm[origin-24:origin,j]; valid = past[np.isfinite(past)]
            persistence.append(np.repeat(valid[-1] if len(valid) else mu[j], 6))
            median.append([med[j,hour[origin+h]] for h in range(6)])
        for name,p in [('Random Forest', rf), ('Persistence', np.array(persistence)),
                       ('Hour-of-day median', np.array(median))]:
            all_rows.extend(records(period, name, p, d, times, pm))
    pred = pd.DataFrame(all_rows)
    summary = verify_and_summarize(pred)
    pred.to_csv(out/'predictions.csv', index=False)
    summary.to_csv(out/'comparison.csv', index=False)
    for column in ['horizon', 'site']:
        rows = []
        for (period, method, key), g in pred.groupby(['period','method',column]):
            e = g.prediction-g.truth
            rows.append(dict(period=period, method=method, **{column:key}, n=len(g),
                             MAE=float(e.abs().mean()), RMSE=float(np.sqrt((e**2).mean()))))
        pd.DataFrame(rows).to_csv(out/f'comparison_by_{column}.csv', index=False)
    protocol.update(checkpoints=checkpoints, versions=dict(torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__))
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    report = ['# Original TimeXer: 2022 causal forecasting comparison', '',
              'This is a new forecasting experiment, not a rescore of the earlier imputation models.', '',
              'The unchanged upstream Model uses 24 historical hours to predict the next 6 PM2.5 readings. '
              'One pooled single-target model handles all three stations using station identity. '
              'The target is the final input channel, as required by features=MS. '
              'Inputs are two other stations, three observation indicators, seven shared historical ERA5 fields, '
              'two hour encodings, three target-station indicators, and target PM2.5 history.', '',
              'January–September trains models and all scalers. October selects neural checkpoints. '
              'November–December evaluates the frozen models. Targets are naturally observed_valid values; '
              'no artificial gaps are created. Forecasts start every six hours, with identical scored targets for every method. '
              'Historical evaluation-period readings can enter later forecasts after their timestamps, without retraining.', '',
              'MAE and RMSE are in µg/m³. October is validation, not a second independent test.', '',
              '| Period | Method | Targets | MAE | RMSE |', '|---|---|---:|---:|---:|']
    for r in summary.itertuples():
        report.append(f'| {r.period} | {r.method} | {r.n} | {r.MAE:.4f} | {r.RMSE:.4f} |')
    report += ['', '## Training', '', json.dumps(checkpoints, indent=2), '',
               '64 hidden dimensions, 4 heads, 2 layers, 128 feedforward dimensions, dropout 0.1, '
               'TimeXer patch length 6 and original use_norm=True. Both neural methods use AdamW at 0.001, '
               'batch size 32, seed 172, maximum 120 epochs, masked standardized MAE, gradient clipping at 1, '
               'and patience 18. These are dataset-specific settings; this is not a reproduction of paper benchmark tables.', '',
               '## Limits', '', *['- '+x for x in protocol['limitations']], '',
               'Results and equal-target/temporal checks are reproducible from the saved predictions and protocol. '
               'They do not establish superiority across years, regions, forecast horizons, or random initializations.', '',
               'Source: https://github.com/thuml/TimeXer/tree/76011909357972bd55a27adba2e1be994d81b327']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(summary.to_string(index=False), flush=True)
    print('FORECAST_RUN_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-data', action='store_true')
    parser.add_argument('--max-epochs', type=int, default=120)
    parser.add_argument('--output', default=str(ROOT/'forecasting/results/original_timexer_2022'))
    run(parser.parse_args())
