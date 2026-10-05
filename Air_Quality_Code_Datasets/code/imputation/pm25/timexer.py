"""TimeXer architecture adapted to masked PM2.5 reconstruction.

Uses the official thuml/TimeXer Model's patch embedding, inverted exogenous
embedding, and attention encoder. Its future-forecast head is replaced by a
same-window reconstruction head. See README for the exact upstream revision.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

from .common import (SITES, SEEDS, WINDOW, load_panel,
                                     make_block_mask, summarize,
                                     timestamp_features)

HERE = Path(__file__).resolve().parents[2] / "source" / "pkg"
UPSTREAM = HERE / "timexer_upstream"
sys.path.insert(0, str(UPSTREAM))
from models.TimeXer import EnEmbedding, Model  # noqa: E402


class TimeXerImputer(Model):
    """Official TimeXer encoder, three PM2.5 endogenous series, 15 exogenous tokens.

    The exogenous channels include all *masked* station values and observation
    indicators, allowing each station to attend to other stations' visible
    readings without ever seeing a deliberately hidden target.
    """

    def __init__(self, patch_len=6, d_model=64, heads=4, layers=2, d_ff=128,
                 dropout=0.1):
        cfg = SimpleNamespace(task_name="short_term_forecast", features="MS",
                              seq_len=WINDOW, pred_len=WINDOW, use_norm=False,
                              patch_len=patch_len, enc_in=1, d_model=d_model,
                              dropout=dropout, embed="timeF", freq="h",
                              factor=3, n_heads=heads, e_layers=layers,
                              d_ff=d_ff, activation="gelu")
        super().__init__(cfg)
        # An endogenous station is one variate. All three share the official
        # patch/self-attention encoder parameters, as forecast_multi does.
        self.en_embedding = EnEmbedding(3, d_model, patch_len, dropout)
        self.head = nn.Sequential(nn.Flatten(start_dim=-2),
                                  nn.Linear(self.head_nf, WINDOW),
                                  nn.Dropout(dropout))

    def forward(self, x):
        # x [B, 24, 15]: PM2.5 values (3), masks (3), ERA5 (7), hour (2).
        endogenous, n_vars = self.en_embedding(x[:, :, :3].permute(0, 2, 1))
        exogenous = self.ex_embedding(x, None)
        encoded = self.encoder(endogenous, exogenous)
        encoded = encoded.reshape(-1, n_vars, encoded.shape[-2], encoded.shape[-1])
        return self.head(encoded).permute(0, 2, 1)


def run(max_epochs=120, patience=18, patch_len=6):
    torch.set_num_threads(2)
    np.random.seed(172)
    random.seed(172)
    torch.manual_seed(172)
    times, pm, weather, _ = load_panel()
    original = pd.read_csv(HERE / "october_masked_predictions.csv")
    keys = ["seed", "datetime_utc", "site", "gap_hours"]
    unique = original[keys + ["truth"]].drop_duplicates(keys)
    assert len(unique) == 886
    masks = {seed: make_block_mask(pm, 6552, 7296, seed) for seed in SEEDS}
    # Exact label/key identity is checked against the prior four-method pilot.
    generated = {(seed, times[t].isoformat(), SITES[j], gap, pm[t, j])
                 for seed, (_, rows) in masks.items() for t, j, gap in rows}
    assert generated == set(unique.itertuples(index=False, name=None))

    mu = np.nanmean(pm[:6552], axis=0)
    sd = np.nanstd(pm[:6552], axis=0)
    wm = weather[:6552].mean(0)
    ws = weather[:6552].std(0)
    ws[ws == 0] = 1
    z = (pm - mu) / sd
    wz = (weather - wm) / ws
    cyc = timestamp_features(times)[:, :2]

    def windows(mask, lo, hi):
        xs, ys, hs = [], [], []
        for start in range(lo, hi - WINDOW + 1, WINDOW):
            stop = start + WINDOW
            val = z[start:stop].copy()
            obs = np.isfinite(val) & ~mask[start:stop]
            val[~obs] = 0
            xs.append(np.concatenate([val, obs.astype(float), wz[start:stop],
                                      cyc[start:stop]], axis=1))
            ys.append(np.nan_to_num(z[start:stop], nan=0))
            hs.append(mask[start:stop])
        return (torch.tensor(np.stack(xs), dtype=torch.float32),
                torch.tensor(np.stack(ys), dtype=torch.float32),
                torch.tensor(np.stack(hs), dtype=torch.bool))

    val = {seed: windows(mask, 6552, 7296) for seed, (mask, _) in masks.items()}
    model = TimeXerImputer(patch_len=patch_len)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    best, bad, best_epoch, best_state = float("inf"), 0, 0, None
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_mask, _ = make_block_mask(pm, 0, 6552, 9000 + epoch, train=True)
        x, y, hidden = windows(epoch_mask, 0, 6552)
        total, count = 0., 0
        for ids in torch.randperm(len(x)).split(32):
            opt.zero_grad()
            loss = (model(x[ids]) - y[ids]).abs()[hidden[ids]].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            total += float(loss.detach()) * len(ids)
            count += len(ids)
        model.eval()
        with torch.no_grad():
            errors = np.concatenate([(model(vx) - vy).abs()[vh].numpy()
                                     for vx, vy, vh in val.values()])
        score = float(errors.mean())
        history.append(dict(epoch=epoch, train_masked_mae_scaled=total/count,
                            validation_masked_mae_scaled=score))
        if score < best - 1e-4:
            best, best_epoch, bad = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch == 1 or epoch % 10 == 0 or bad >= patience:
            print(f"TimeXer adaptation epoch {epoch}: train={total/count:.4f}, "
                  f"validation={score:.4f}, best={best_epoch}", flush=True)
        if bad >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    rows = []
    with torch.no_grad():
        for seed, (_, hidden_rows) in masks.items():
            prediction = model(val[seed][0]).numpy() * sd + mu
            for t, j, gap in hidden_rows:
                rows.append(dict(seed=seed, method="TimeXer imputation adaptation",
                                 datetime_utc=times[t].isoformat(), site=SITES[j],
                                 gap_hours=gap, truth=pm[t, j],
                                 prediction=float(prediction[(t-6552)//WINDOW,
                                                             (t-6552)%WINDOW, j])))
    out = HERE / "timexer_results" / f"patch_{patch_len}"
    out.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    assert len(df) == 886 and np.isfinite(df.prediction).all()
    assert set(df[keys + ["truth"]].itertuples(index=False, name=None)) == generated
    df.to_csv(out / "october_masked_predictions_timexer.csv", index=False)
    pd.DataFrame(history).to_csv(out / "timexer_loss_history.csv", index=False)
    summarize(df).to_csv(out / "timexer_by_seed_station_gap.csv", index=False)
    comparison = pd.concat([original, df], ignore_index=True)
    comparison.assign(ae=lambda v: abs(v.truth-v.prediction),
                      se=lambda v: (v.truth-v.prediction)**2).groupby(
        "method", as_index=False).agg(n=("prediction", "count"),
                                      MAE=("ae", "mean"), MSE=("se", "mean")).pipe(
        lambda q: q.assign(RMSE=np.sqrt(q.MSE)).drop(columns="MSE")
    ).to_csv(out / "comparison_all_methods.csv", index=False)
    torch.save(dict(state_dict=best_state, pm_mean=mu, pm_std=sd,
                    weather_mean=wm, weather_std=ws, best_epoch=best_epoch,
                    patch_len=patch_len), out / "timexer_checkpoint.pt")
    (out / "timexer_protocol.json").write_text(json.dumps(dict(
        upstream_commit="76011909357972bd55a27adba2e1be994d81b327",
        adaptation="official embedding and encoder; same-window 3-station reconstruction head",
        patch_len=patch_len, d_model=64, heads=4, layers=2, d_ff=128,
        dropout=.1, max_epochs=max_epochs, patience=patience,
        best_epoch=best_epoch, trained_epochs=len(history),
        best_validation_mae_scaled=best, validation_hidden_values=886,
        split="Jan-Sep train, October validation; repeated seed observations"
    ), indent=2))
    print(pd.read_csv(out / "comparison_all_methods.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--patch-len", type=int, default=6)
    args = parser.parse_args()
    run(args.max_epochs, args.patience, args.patch_len)

