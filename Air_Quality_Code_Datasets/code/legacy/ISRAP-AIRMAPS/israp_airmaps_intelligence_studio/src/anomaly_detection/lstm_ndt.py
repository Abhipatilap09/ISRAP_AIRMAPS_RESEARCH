"""LSTM with non-parametric dynamic thresholding anomaly detector."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.anomaly_detection.base import AnomalyResult, BaseDetector
from src.config import MODELS_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class _LSTMModel(object if not _TORCH_AVAILABLE else object):
    pass


if _TORCH_AVAILABLE:
    import torch
    import torch.nn as nn

    class _LSTMModel(nn.Module):  # type: ignore[no-redef]
        def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, input_dim)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out)


def _dynamic_threshold(errors: np.ndarray, percentile: float = 99.0) -> float:
    return float(np.nanpercentile(errors, percentile))


class LSTMNDTDetector(BaseDetector):
    """LSTM prediction error + non-parametric dynamic threshold.

    Implementation note: This is an original implementation inspired by the
    NASA TELEMANOM approach (Hundman et al. 2018).  The threshold is set via
    a simple percentile rather than the full iterative pruning algorithm.
    """

    name = "lstm_ndt"
    category = "Deep Learning"

    def detect(
        self,
        df: pd.DataFrame,
        pollutant: str,
        seq_len: int = 24,
        hidden_dim: int = 32,
        num_layers: int = 1,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        threshold_percentile: float = 99.0,
        fast_mode: bool = True,
        model_dir: Path = MODELS_DIR,
        **_,
    ) -> tuple[pd.DataFrame, AnomalyResult]:
        df = df.copy()
        flag_col = "is_lstm_anomaly"
        score_col = "lstm_error"
        t0 = time.perf_counter()

        if not _TORCH_AVAILABLE:
            df[flag_col] = False
            df[score_col] = 0.0
            runtime = time.perf_counter() - t0
            return df, self._make_result(flag_col, score_col, df, None, runtime,
                                         status="error", warning="PyTorch not installed")

        import torch

        col = df[pollutant].copy() if pollutant in df.columns else pd.Series(np.nan, index=df.index)
        valid_idx = col.dropna().index
        if len(valid_idx) < seq_len * 4:
            df[flag_col] = False
            df[score_col] = 0.0
            runtime = time.perf_counter() - t0
            return df, self._make_result(flag_col, score_col, df, None, runtime,
                                         status="skipped", warning="Insufficient data for LSTM")

        values = col.ffill().bfill().values.astype(np.float32)
        mu, sd = values.mean(), values.std() + 1e-9
        values_norm = (values - mu) / sd

        split = int(len(values_norm) * 0.8)
        train_vals = values_norm[:split]
        test_vals = values_norm[split:]

        def make_sequences(v, sl):
            X, y = [], []
            for i in range(len(v) - sl):
                X.append(v[i:i+sl])
                y.append(v[i+sl])
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

        X_tr, y_tr = make_sequences(train_vals, seq_len)
        if len(X_tr) < 4:
            df[flag_col] = False
            df[score_col] = 0.0
            runtime = time.perf_counter() - t0
            return df, self._make_result(flag_col, score_col, df, None, runtime,
                                         status="skipped", warning="Training sequences too short")

        torch.manual_seed(RANDOM_SEED)
        device = torch.device("cpu")
        model = _LSTMModel(1, hidden_dim, num_layers).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        X_t = torch.from_numpy(X_tr[:, :, None]).to(device)
        y_t = torch.from_numpy(y_tr[:, None]).to(device)

        actual_epochs = 5 if fast_mode else epochs
        for epoch in range(actual_epochs):
            model.train()
            optim.zero_grad()
            pred = model(X_t)[:, -1, :]
            loss = criterion(pred, y_t)
            loss.backward()
            optim.step()

        # Compute errors on full series
        model.eval()
        errors = np.zeros(len(values_norm))
        with torch.no_grad():
            for i in range(len(values_norm) - seq_len):
                x_in = torch.from_numpy(values_norm[i:i+seq_len][None, :, None]).to(device)
                pred_val = model(x_in)[0, -1, 0].item()
                errors[i + seq_len] = abs(values_norm[i + seq_len] - pred_val)

        thr = _dynamic_threshold(errors[:split], threshold_percentile)
        df["lstm_dynamic_threshold"] = thr
        df[score_col] = errors
        df[flag_col] = (errors > thr) & col.notna().values

        runtime = time.perf_counter() - t0
        return df, self._make_result(flag_col, score_col, df, thr, runtime)

