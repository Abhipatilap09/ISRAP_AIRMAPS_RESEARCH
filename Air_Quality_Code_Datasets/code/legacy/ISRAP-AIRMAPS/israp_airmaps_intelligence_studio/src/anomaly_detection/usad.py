"""USAD (UnSupervised Anomaly Detection) detector.

Reference: Audibert et al. "USAD: UnSupervised Anomaly Detection on Multivariate
Time Series." KDD 2020.

This is an implementation of the core two-decoder architecture.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from src.anomaly_detection.base import AnomalyResult, BaseDetector
from src.config import ERA5_VARS, RANDOM_SEED

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:
    import torch
    import torch.nn as nn

    class _Encoder(nn.Module):
        def __init__(self, in_dim, hidden):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden // 2), nn.ReLU())
        def forward(self, x):
            return self.net(x)

    class _Decoder(nn.Module):
        def __init__(self, out_dim, hidden):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(hidden // 2, hidden), nn.ReLU(),
                                     nn.Linear(hidden, out_dim), nn.Sigmoid())
        def forward(self, z):
            return self.net(z)


class USADDetector(BaseDetector):
    """USAD two-decoder autoencoder anomaly detector."""

    name = "usad"
    category = "Deep Learning"

    def detect(
        self,
        df: pd.DataFrame,
        pollutant: str,
        seq_len: int = 12,
        hidden_dim: int = 32,
        epochs: int = 5,
        batch_size: int = 32,
        lr: float = 1e-3,
        threshold_percentile: float = 99.0,
        alpha: float = 0.5,
        fast_mode: bool = True,
        **_,
    ) -> tuple[pd.DataFrame, AnomalyResult]:
        df = df.copy()
        flag_col = "is_usad_anomaly"
        score_col = "usad_score"
        t0 = time.perf_counter()

        if not _TORCH_AVAILABLE:
            df[flag_col] = False
            df[score_col] = 0.0
            runtime = time.perf_counter() - t0
            return df, self._make_result(flag_col, score_col, df, None, runtime,
                                         status="error", warning="PyTorch not installed")

        import torch

        feat_cols = [pollutant] + [c for c in ERA5_VARS if c in df.columns]
        feat_cols = [c for c in feat_cols if c in df.columns]
        X = df[feat_cols].copy()
        valid_mask = X.notna().all(axis=1)

        if valid_mask.sum() < seq_len * 4:
            df[flag_col] = False
            df[score_col] = 0.0
            runtime = time.perf_counter() - t0
            return df, self._make_result(flag_col, score_col, df, None, runtime,
                                         status="skipped", warning="Insufficient data for USAD")

        X_filled = X.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
        mu = X_filled.mean()
        sd = X_filled.std().replace(0, 1)
        X_norm = ((X_filled - mu) / sd).values.astype(np.float32)

        n = len(X_norm)
        in_dim = seq_len * X_norm.shape[1]
        seqs = np.array([X_norm[i:i+seq_len].flatten() for i in range(n - seq_len)])
        split = int(len(seqs) * 0.8)
        seqs_tr = seqs[:split]

        torch.manual_seed(RANDOM_SEED)
        device = torch.device("cpu")
        enc = _Encoder(in_dim, hidden_dim).to(device)
        dec1 = _Decoder(in_dim, hidden_dim).to(device)
        dec2 = _Decoder(in_dim, hidden_dim).to(device)
        opt = torch.optim.Adam(list(enc.parameters()) + list(dec1.parameters()) + list(dec2.parameters()), lr=lr)

        actual_epochs = 3 if fast_mode else epochs
        tr_t = torch.from_numpy(seqs_tr).to(device)
        for ep in range(1, actual_epochs + 1):
            enc.train(); dec1.train(); dec2.train()
            opt.zero_grad()
            z = enc(tr_t)
            w1 = dec1(z)
            w2 = dec2(enc(w1))
            loss = (1/ep) * torch.mean((tr_t - w1)**2) + (1 - 1/ep) * torch.mean((tr_t - w2)**2)
            loss.backward()
            opt.step()

        # Compute anomaly scores on all sequences
        enc.eval(); dec1.eval(); dec2.eval()
        all_seqs = torch.from_numpy(seqs).to(device)
        with torch.no_grad():
            z = enc(all_seqs)
            w1 = dec1(z)
            w2 = dec2(enc(w1))
            scores = (alpha * torch.mean((all_seqs - w1)**2, dim=1)
                      + (1 - alpha) * torch.mean((all_seqs - w2)**2, dim=1)).cpu().numpy()

        thr_val = float(np.nanpercentile(scores[:split], threshold_percentile))
        full_scores = np.zeros(n)
        full_scores[seq_len:] = scores
        df[score_col] = full_scores
        df["usad_threshold"] = thr_val
        df[flag_col] = (full_scores > thr_val) & valid_mask.values

        runtime = time.perf_counter() - t0
        return df, self._make_result(flag_col, score_col, df, thr_val, runtime)

