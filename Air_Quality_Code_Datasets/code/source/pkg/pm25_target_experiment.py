"""PM2.5-only offline imputation experiment on the quality-labelled 2022 panel.

The same fixed October masks are evaluated by all methods. This script never
fills natural missing values or changes the reference CSV. Transformer is our
station-temporal adaptation, not a reproduction of an external paper.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent
SOURCE = Path(__file__).resolve().parents[3] / "datasets/merged/aqs_era5_2022_quality_labelled.csv.gz"
OUT = ROOT / "outputs" / "professor_ke_pm25_pilot"
WEATHER = ["era5_u10", "era5_v10", "era5_t2m", "era5_d2m", "era5_sp", "era5_tp", "era5_blh"]
SITES = [32, 59, 1069]
SEEDS = [11, 22, 33]
GAPS = [1, 3, 6]
WINDOW = 24


def load_panel():
    df = pd.read_csv(SOURCE, low_memory=False)
    df["datetime_utc"] = pd.to_datetime(df.datetime_utc, utc=True)
    times = pd.DatetimeIndex(sorted(df.datetime_utc.unique()))
    assert len(times) == 8760 and times.is_unique
    assert (times[1:] - times[:-1] == pd.Timedelta(hours=1)).all()
    frame = df[df.site == SITES[0]].set_index("datetime_utc").reindex(times)
    weather = frame[WEATHER].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    assert np.isfinite(weather).all(), "ERA5 missing in PM2.5 study"
    pm = np.full((len(times), len(SITES)), np.nan)
    for j, site in enumerate(SITES):
        sub = df[df.site == site].set_index("datetime_utc").reindex(times)
        valid = sub["PM2.5_observed_valid"].fillna(0).eq(1).to_numpy()
        pm[:, j] = np.where(valid, pd.to_numeric(sub["PM2.5"], errors="coerce"), np.nan)
        assert not sub["PM2.5_structural_non_monitoring"].fillna(0).astype(bool).any()
    split = frame.evaluation_split.to_numpy()
    assert np.array_equal(np.unique(split, return_counts=True)[1], [1464, 6552, 744])
    assert (split[:6552] == "train").all() and (split[6552:7296] == "validation").all()
    assert (split[7296:] == "test").all()
    assert times[0] == pd.Timestamp("2022-01-01 06:00:00Z")
    assert times[6552] == pd.Timestamp("2022-10-01 06:00:00Z")
    return times, pm, weather, split


def make_block_mask(pm, lo, hi, seed, train=False):
    """Select known 1/3/6-hour blocks wholly inside 24-hour windows."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(pm.shape, dtype=bool)
    records = []
    for start in range(lo, hi - WINDOW + 1, WINDOW):
        for j, site in enumerate(SITES):
            # Training covers more examples; October masks cover about 10%.
            lengths = rng.choice(GAPS, size=2 if train else 1, replace=False)
            for gap in lengths:
                options = [t for t in range(start + 1, start + WINDOW - gap)
                           if np.isfinite(pm[t:t + gap, j]).all()
                           and np.isfinite(pm[t-1, j]) and np.isfinite(pm[t+gap, j])
                           and not mask[t-1:t + gap + 1, j].any()]
                if not options:
                    continue
                t = int(rng.choice(options))
                mask[t:t + gap, j] = True
                for k in range(gap):
                    records.append((t + k, j, gap))
    assert mask[:lo].sum() == mask[hi:].sum() == 0
    return mask, records


def timestamp_features(times):
    # CST is fixed UTC-06 for AQS clock time, without DST ambiguity.
    local = times - pd.Timedelta(hours=6)
    hour = local.hour.to_numpy()
    doy = local.dayofyear.to_numpy()
    return np.column_stack([np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                            np.sin(2*np.pi*doy/365), np.cos(2*np.pi*doy/365)])


def tabular_features(pm_masked, weather, cal, rows):
    # Offline imputation may use past AND future observations within a gap.
    lags = [-24, -6, -3, -2, -1, 0, 1, 2, 3, 6, 24]
    features = []
    for t, j, _ in rows:
        x = []
        for station in range(len(SITES)):
            for delta in lags:
                if station == j and delta == 0:
                    continue  # Never pass the target at its own hour.
                q = t + delta
                x.append(pm_masked[q, station] if 0 <= q < len(pm_masked) else np.nan)
        x.extend(weather[t])
        x.extend(cal[t])
        x.extend(float(j == site_index) for site_index in range(len(SITES)))
        features.append(x)
    return np.asarray(features, float)


def interpolate(pm_masked, rows):
    ans = []
    for t, j, gap in rows:
        left, right = t-1, t+1
        while left >= 0 and not np.isfinite(pm_masked[left, j]): left -= 1
        while right < len(pm_masked) and not np.isfinite(pm_masked[right, j]): right += 1
        if left >= 0 and right < len(pm_masked) and right-left <= gap+1:
            ans.append(pm_masked[left, j] + (pm_masked[right, j]-pm_masked[left, j])*(t-left)/(right-left))
        else:
            ans.append(np.nan)
    return np.asarray(ans)


def hour_median(pm, times, rows):
    hour = (times - pd.Timedelta(hours=6)).hour.to_numpy()
    med = {(j, h): np.nanmedian(pm[:6552, j][hour[:6552] == h])
           for j in range(len(SITES)) for h in range(24)}
    return np.array([med[j, hour[t]] for t, j, _ in rows])


def run_rf(times, pm, weather, masks):
    cal = timestamp_features(times)
    train_mask, train_rows = make_block_mask(pm, 0, 6552, 702, train=True)
    train_pm = pm.copy()
    train_pm[train_mask] = np.nan
    # Lags/leads must stay in the fitting split. In particular, the last
    # September windows must not consume October PM2.5 as future features.
    train_pm[6552:] = np.nan
    xtrain = tabular_features(train_pm, weather, cal, train_rows)
    ytrain = np.array([pm[t, j] for t, j, _ in train_rows])
    assert np.isfinite(ytrain).all()
    model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                          RandomForestRegressor(n_estimators=160, max_depth=16,
                                                min_samples_leaf=3, max_features=0.8,
                                                n_jobs=2, random_state=702))
    model.fit(xtrain, ytrain)
    joblib.dump(model, OUT/"random_forest.joblib")
    results = []
    for seed, (mask, rows) in masks.items():
        masked = pm.copy(); masked[mask] = np.nan
        # The RF comparison is evaluated using October PM2.5 context only;
        # no November/December measurements enter its future features.
        rf_context = masked.copy()
        rf_context[:6552] = np.nan
        rf_context[7296:] = np.nan
        pred = model.predict(tabular_features(rf_context, weather, cal, rows))
        baselines = {"Hour median": hour_median(pm, times, rows),
                     "Linear interpolation": interpolate(masked, rows),
                     "Random Forest": pred}
        for method, p in baselines.items():
            for (t, j, gap), yhat in zip(rows, p):
                results.append(dict(seed=seed, method=method, datetime_utc=times[t].isoformat(),
                                    site=SITES[j], gap_hours=gap, truth=pm[t, j], prediction=yhat))
    return results, len(train_rows)


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


def summarize(df):
    x=df.dropna(subset=["prediction"]).copy()
    x["absolute_error"]=(x.truth-x.prediction).abs()
    x["squared_error"]=(x.truth-x.prediction)**2
    grouped=x.groupby(["method","seed","site","gap_hours"],as_index=False).agg(
        n=("truth","size"),MAE=("absolute_error","mean"),MSE=("squared_error","mean"))
    grouped["RMSE"]=np.sqrt(grouped.MSE)
    grouped.drop(columns="MSE",inplace=True)
    return grouped


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--max-epochs",type=int,default=100)
    parser.add_argument("--patience",type=int,default=15)
    parser.add_argument("--skip-transformer",action="store_true")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    times,pm,weather,split=load_panel()
    masks={seed:make_block_mask(pm,6552,7296,seed) for seed in SEEDS}
    base,ntrain=run_rf(times,pm,weather,masks)
    details={"source":str(SOURCE),"task":"PM2.5 offline imputation, 3 monitored sites",
             "train":"2022-01-01 00:00 to 2022-09-30 23:00 CST",
             "validation":"2022-10-01 00:00 to 2022-10-31 23:00 CST",
             "test":"2022-11 and 2022-12 withheld from experiment",
             "seed_list":SEEDS,"mask_block_lengths":GAPS,"rf_train_hidden_values":ntrain,
             "weather":"seven ERA5 fields, same grid point at all stations",
             "forecasting_caveat":"offline interpolation and bidirectional imputation cannot access future observations in a live forecast"}
    if not args.skip_transformer:
        extra,history,best_epoch,settings=run_transformer(times,pm,weather,masks,args.max_epochs,args.patience)
        base.extend(extra)
        pd.DataFrame(history).to_csv(OUT/"transformer_loss_history.csv",index=False)
        details["transformer"]={"best_epoch":best_epoch,**settings}
    df=pd.DataFrame(base)
    df.to_csv(OUT/"october_masked_predictions.csv",index=False)
    score=summarize(df)
    score.to_csv(OUT/"metrics_by_seed_station_gap.csv",index=False)
    pooled=(df.assign(ae=lambda x:abs(x.truth-x.prediction),se=lambda x:(x.truth-x.prediction)**2)
            .groupby("method",as_index=False).agg(n=("prediction","count"),MAE=("ae","mean"),MSE=("se","mean")))
    pooled["RMSE"]=np.sqrt(pooled.MSE); pooled=pooled.drop(columns="MSE")
    pooled.to_csv(OUT/"comparison.csv",index=False)
    details["validation_hidden_values_per_seed"]={str(s):len(rows) for s,(_,rows) in masks.items()}
    (OUT/"protocol.json").write_text(json.dumps(details,indent=2))
    print(pooled.to_string(index=False),flush=True)


if __name__=="__main__": main()
