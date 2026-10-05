"""Run one set of existing PM2.5 implementations from the repository root."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["baselines", "station-transformer", "timexer"], required=True)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--patch-len", type=int, default=6)
    args = parser.parse_args()
    if args.method == "timexer":
        from .pm25.timexer import run
        run(args.max_epochs, args.patience, args.patch_len)
        return
    from .pm25.common import OUT, SEEDS, load_panel, make_block_mask, summarize
    times, pm, weather, _ = load_panel()
    masks = {seed: make_block_mask(pm, 6552, 7296, seed) for seed in SEEDS}
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / args.method
    destination.mkdir(exist_ok=True)
    if args.method == "baselines":
        from .pm25.random_forest import run_rf
        records, count = run_rf(times, pm, weather, masks)
        metadata = {"method": args.method, "training_hidden_values": count}
    else:
        from .pm25.station_transformer import run_transformer
        records, history, epoch, settings = run_transformer(times, pm, weather, masks, args.max_epochs, args.patience)
        pd.DataFrame(history).to_csv(destination / "loss_history.csv", index=False)
        metadata = {"method": args.method, "best_epoch": epoch, **settings}
    predictions = pd.DataFrame(records)
    predictions.to_csv(destination / "predictions.csv", index=False)
    summarize(predictions).to_csv(destination / "metrics_by_seed_station_gap.csv", index=False)
    score = predictions.assign(ae=lambda x: abs(x.truth-x.prediction), se=lambda x: (x.truth-x.prediction)**2).groupby("method", as_index=False).agg(n=("prediction", "count"), MAE=("ae", "mean"), MSE=("se", "mean"))
    score["RMSE"] = np.sqrt(score.MSE)
    score.drop(columns="MSE").to_csv(destination / "comparison.csv", index=False)
    (destination / "run_settings.json").write_text(json.dumps(metadata, indent=2))
    print(score.drop(columns="MSE").to_string(index=False))

if __name__ == "__main__":
    main()

