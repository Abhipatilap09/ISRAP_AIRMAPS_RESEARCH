"""Station-temporal Transformer: complete training and masked reconstruction.

This is the implementation used in the recorded PM2.5 pilot, with 24 time tokens.
"""
import random
import numpy as np
from .common import OUT, SITES, WINDOW, make_block_mask, timestamp_features

def run_transformer(times, pm, weather, masks, max_epochs=100, patience=15):
    import torch
    from torch import nn
    torch.set_num_threads(2)
    np.random.seed(172); random.seed(172); torch.manual_seed(172)
    mu = np.nanmean(pm[:6552], axis=0)
    sd = np.nanstd(pm[:6552], axis=0)
    wm = weather[:6552].mean(0); ws = weather[:6552].std(0); ws[ws == 0] = 1
    z = (pm-mu)/sd
    wz = (weather-wm)/ws
    cyc = timestamp_features(times)[:, :2]

    class StationTemporalTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            # At each of 24 time tokens: 3 station PM2.5, 3 observation masks,
            # 7 shared weather fields, 2 time encodings. Nodes are never pollutants.
            self.project = nn.Linear(15, 64)
            self.pos = nn.Parameter(torch.zeros(1, WINDOW, 64))
            block = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128,
                                                dropout=0.1, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(block, num_layers=2)
            self.out = nn.Linear(64, 3)
        def forward(self, x):
            return self.out(self.encoder(self.project(x) + self.pos))

    def windows(mask, lo, hi):
        xs, targets, hidden = [], [], []
        for start in range(lo, hi-WINDOW+1, WINDOW):
            stop = start+WINDOW
            val = z[start:stop].copy()
            obs = np.isfinite(val) & ~mask[start:stop]
            val[~obs] = 0
            xs.append(np.concatenate([val, obs.astype(float), wz[start:stop], cyc[start:stop]], axis=1))
            targets.append(np.nan_to_num(z[start:stop], nan=0))
            hidden.append(mask[start:stop])
        return (torch.tensor(np.stack(xs), dtype=torch.float32),
                torch.tensor(np.stack(targets), dtype=torch.float32),
                torch.tensor(np.stack(hidden), dtype=torch.bool))

    val_sets = {}
    for seed, (mask, rows) in masks.items():
        val_sets[seed] = windows(mask, 6552, 7296)
    model = StationTemporalTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    best = float("inf"); bad = 0; best_state = None; best_epoch = None
    history = []
    for epoch in range(1, max_epochs+1):
        model.train()
        # New observed-only masks every epoch; validation masks remain fixed.
        epoch_mask, _ = make_block_mask(pm, 0, 6552, 9000+epoch, train=True)
        x, y, hidden = windows(epoch_mask, 0, 6552)
        perm = torch.randperm(len(x))
        train_sum, train_count = 0., 0
        for ids in perm.split(32):
            optimizer.zero_grad()
            pred = model(x[ids])
            loss = torch.abs(pred-y[ids])[hidden[ids]].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_sum += float(loss)*len(ids); train_count += len(ids)
        model.eval()
        errors=[]
        with torch.no_grad():
            for vx, vy, vh in val_sets.values():
                e = (model(vx)-vy).abs()[vh].numpy()
                errors.extend(e.tolist())
        val_mae = float(np.mean(errors))
        history.append(dict(epoch=epoch, train_masked_mae_scaled=train_sum/train_count,
                            validation_masked_mae_scaled=val_mae))
        if val_mae < best-1e-4:
            best = val_mae; bad=0; best_epoch=epoch
            best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            bad += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"Transformer epoch {epoch}: train={train_sum/train_count:.4f}, validation={val_mae:.4f}", flush=True)
        if bad >= patience: break
    model.load_state_dict(best_state); model.eval()
    torch.save({"state_dict":best_state,"pm_mean":mu,"pm_std":sd,
                "weather_mean":wm,"weather_std":ws,"best_epoch":best_epoch},
               OUT/"transformer_checkpoint.pt")
    result=[]
    with torch.no_grad():
        for seed,(mask,rows) in masks.items():
            vx, _, _ = val_sets[seed]
            pred=model(vx).numpy()*sd+mu
            for t,j,gap in rows:
                result.append(dict(seed=seed,method="PM2.5 station-temporal Transformer",
                                   datetime_utc=times[t].isoformat(),site=SITES[j],gap_hours=gap,
                                   truth=pm[t,j],prediction=float(pred[(t-6552)//WINDOW,(t-6552)%WINDOW,j])))
    return result, history, best_epoch, {"d_model":64,"heads":4,"layers":2,
                                         "max_epochs":max_epochs,"patience":patience,
                                         "best_validation_mae_scaled":best}

