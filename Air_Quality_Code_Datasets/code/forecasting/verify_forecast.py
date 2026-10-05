"""Recompute metrics, audit reference targets, and optionally replay all saved models."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from run_original_timexer import ROOT, UPSTREAM, SOURCE_SHA, SITES, CONFIG, prepare, records, verify_and_summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT/'forecasting/results/original_timexer_2022')
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    out = args.results
    pred = pd.read_csv(out/'predictions.csv')
    actual = verify_and_summarize(pred).sort_values(['period','method']).reset_index(drop=True)
    saved = pd.read_csv(out/'comparison.csv').sort_values(['period','method']).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, saved, check_exact=False, atol=1e-10, rtol=0)
    times, pm, hour, mu, sd, wm, ws, sets = prepare()
    lookup = {t.isoformat():i for i,t in enumerate(times)}
    for r in pred.itertuples():
        t = lookup[r.target_utc]; origin = lookup[r.origin_utc]; j = SITES.index(r.site)
        assert t == origin + r.horizon - 1
        assert lookup[r.last_input_utc] == origin - 1
        assert np.isfinite(pm[t,j]) and np.isclose(r.truth, pm[t,j], rtol=0, atol=1e-10)
        lo,hi = (6552,7296) if r.period == 'October' else (7296,8760)
        assert lo <= t < hi and origin in range(lo, hi-5, 6)
    protocol = json.loads((out/'protocol.json').read_text())
    assert protocol['config'] == CONFIG
    assert hashlib.sha256((UPSTREAM/'models/TimeXer.py').read_bytes()).hexdigest() == SOURCE_SHA
    assert protocol['upstream_model_sha256'] == SOURCE_SHA
    for period in ['October','November-December']:
        assert (saved[saved.period == period]['n'] == int(sets[period]['mask'].sum())).all()
    for name,slug in [('Original TimeXer','timexer'), ('Temporal Transformer','temporal_transformer')]:
        h = pd.read_csv(out/f'{slug}_loss_history.csv')
        best, epoch = float('inf'), 0
        for row in h.itertuples():
            if row.validation_MAE < best - 1e-4: best,epoch = row.validation_MAE,row.epoch
        assert epoch == protocol['checkpoints'][name]['best_epoch']
        assert len(h) == protocol['checkpoints'][name]['trained_epochs']
    if args.replay:
        import torch
        from torch import nn
        import joblib
        torch.set_num_threads(2)
        sys.path.insert(0, str(UPSTREAM))
        from models.TimeXer import Model
        from types import SimpleNamespace
        class TemporalTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Linear(18,64)
                self.position = nn.Parameter(torch.zeros(1,24,64))
                layer = nn.TransformerEncoderLayer(64,4,128,0.1,activation='gelu',batch_first=True)
                self.encoder = nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
                self.head = nn.Linear(24*64,6)
            def forward(self,x):
                return self.head(self.encoder(self.embedding(x)+self.position).flatten(1))
        for name,slug in [('Original TimeXer','timexer'), ('Temporal Transformer','temporal_transformer')]:
            model = Model(SimpleNamespace(**CONFIG)) if slug == 'timexer' else TemporalTransformer()
            checkpoint = torch.load(out/f'{slug}.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['state_dict']); model.eval()
            for period in ['October','November-December']:
                d = sets[period]; chunks=[]
                with torch.no_grad():
                    for start in range(0,len(d['x']),128):
                        x = torch.from_numpy(d['x'][start:start+128])
                        y = model(x,None,None,None).squeeze(-1) if slug == 'timexer' else model(x)
                        chunks.append(y.numpy())
                p = np.concatenate(chunks)*sd[d['sites'],None] + mu[d['sites'],None]
                expected = pd.DataFrame(records(period,name,p,d,times,pm))
                stored = pred[(pred.method == name)&(pred.period == period)]
                keys=['origin_utc','site','horizon','target_utc']
                a=expected.sort_values(keys).prediction.to_numpy(); b=stored.sort_values(keys).prediction.to_numpy()
                np.testing.assert_allclose(a,b,atol=1e-5,rtol=0)
            print(f'{name}: all saved forecasts replayed successfully')
        forests=[joblib.load(out/f'random_forest_horizon_{h}.joblib') for h in range(1,7)]
        for period in ['October','November-December']:
            d=sets[period]
            p=np.column_stack([m.predict(d['x'].reshape(len(d['x']),-1)) for m in forests])
            p=p*sd[d['sites'],None]+mu[d['sites'],None]
            expected=pd.DataFrame(records(period,'Random Forest',p,d,times,pm))
            stored=pred[(pred.method=='Random Forest')&(pred.period==period)]
            keys=['origin_utc','site','horizon','target_utc']
            np.testing.assert_allclose(expected.sort_values(keys).prediction,
                                       stored.sort_values(keys).prediction,atol=1e-10,rtol=0)
        print('Random Forest: all saved forecasts replayed successfully')
    print('FORECAST_VERIFICATION_PASSED: metrics, reference truths, timestamps, common targets, source hash, checkpoints')


if __name__ == '__main__':
    main()
