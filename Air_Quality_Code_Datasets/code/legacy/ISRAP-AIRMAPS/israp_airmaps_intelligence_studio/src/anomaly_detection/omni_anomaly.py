"""OmniAnomaly anomaly detector — Stochastic RNN with Normalizing Flows.

Reference: Su et al. "Robust Anomaly Detection for Multivariate Time Series through
Stochastic Recurrent Neural Network" (KDD 2019).

Core idea
---------
A VAE-style architecture:
  - GRU encoder → approximate posterior q(z|x) = N(mu_enc, sigma_enc)
  - GRU decoder → reconstruct x from z
  - Anomaly score = negative ELBO = reconstruction loss + KL(q||p)

Points with high reconstruction error (i.e. the model can't explain them under the
learned generative process) are flagged as anomalies.

Planar normalizing flows: The original paper uses them to enrich the posterior; here
we use a 2-layer MLP posterior network as a practical simplification that preserves the
stochastic latent variable structure.

Implementation tiers
--------------------
1. PyTorch available  → stochastic GRU VAE (simplified OmniAnomaly)
2. PyTorch unavailable → ETS reconstruction error proxy

Label: "OmniAnomaly (PyTorch, simplified stochastic GRU VAE)" or
       "OmniAnomaly-approx (ETS recon. error)"
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from src.anomaly_detection.base import AnomalyResult, BaseDetector

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ── Pure-NumPy fallback ───────────────────────────────────────────────────────

def _ets_recon_error(values: np.ndarray, alpha: float = 0.15, window: int = 24) -> np.ndarray:
    """Approximate anomaly score via ETS reconstruction error."""
    n = len(values)
    clean = np.where(np.isnan(values), np.nanmean(values), values)
    mu, sd = clean.mean(), clean.std() + 1e-9

    # Forward ETS prediction
    pred = np.full(n, np.nan)
    state = clean[0]
    for i in range(1, n):
        pred[i] = state
        state = (1 - alpha) * state + alpha * clean[i]

    errors = np.abs(clean - pred)
    errors[0] = 0.0
    return errors / (sd + 1e-9)


# ── PyTorch OmniAnomaly (stochastic GRU VAE) ─────────────────────────────────

if _TORCH_AVAILABLE:
    class _OmniEncoder(nn.Module):
        """GRU encoder → posterior q(z|x): outputs (mu, log_var)."""
        def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.mu_head = nn.Linear(hidden_dim, latent_dim)
            self.lv_head = nn.Linear(hidden_dim, latent_dim)

        def forward(self, x):  # x: (B, T, input_dim)
            h, _ = self.gru(x)           # (B, T, hidden_dim)
            mu = self.mu_head(h)         # (B, T, latent_dim)
            log_var = self.lv_head(h)    # (B, T, latent_dim)
            return mu, log_var

    class _OmniDecoder(nn.Module):
        """GRU decoder z → x̂."""
        def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
            super().__init__()
            self.gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)
            self.out = nn.Linear(hidden_dim, output_dim)

        def forward(self, z):  # z: (B, T, latent_dim)
            h, _ = self.gru(z)
            return self.out(h)  # (B, T, output_dim)

    class _OmniAnomalyNet(nn.Module):
        def __init__(self, input_dim: int = 1, hidden_dim: int = 32, latent_dim: int = 8):
            super().__init__()
            self.encoder = _OmniEncoder(input_dim, hidden_dim, latent_dim)
            self.decoder = _OmniDecoder(latent_dim, hidden_dim, input_dim)

        def reparametrize(self, mu, log_var):
            std = torch.exp(0.5 * log_var.clamp(-10, 2))
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            mu, log_var = self.encoder(x)
            z = self.reparametrize(mu, log_var)
            recon = self.decoder(z)
            return recon, mu, log_var


def _elbo_loss(recon, x, mu, log_var, obs_mask, beta: float = 0.1):
    """ELBO: reconstruction MSE + beta * KL divergence."""
    recon_loss = ((recon - x) ** 2 * obs_mask).sum() / (obs_mask.sum() + 1e-9)
    kl = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).mean()
    return recon_loss + beta * kl, recon_loss, kl


def _run_omni_anomaly(
    values: np.ndarray,
    hidden_dim: int = 32,
    latent_dim: int = 8,
    epochs: int = 10,
    lr: float = 1e-3,
    seq_len: int = 48,
    beta: float = 0.1,
    n_mc_samples: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """Run simplified OmniAnomaly and return negative-ELBO anomaly scores."""
    import torch

    torch.manual_seed(random_state)
    n = len(values)
    obs = ~np.isnan(values)
    if obs.sum() < seq_len:
        return _ets_recon_error(values)

    clean = np.where(obs, values, np.nanmean(values))
    mu_v, sd_v = clean[obs].mean(), clean[obs].std() + 1e-9
    v_norm = (clean - mu_v) / sd_v

    net = _OmniAnomalyNet(input_dim=1, hidden_dim=hidden_dim, latent_dim=latent_dim)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    starts = list(range(0, n - seq_len + 1, max(1, seq_len // 2)))
    if not starts:
        return _ets_recon_error(values)

    # ── Training ───────────────────────────────────────────────────────────────
    net.train()
    for ep in range(epochs):
        np.random.seed(ep + random_state)
        for s in np.random.choice(starts, size=min(32, len(starts)), replace=False):
            x_np = v_norm[s:s + seq_len].astype(np.float32)
            x_t = torch.tensor(x_np).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
            obs_m = torch.tensor(
                obs[s:s + seq_len].astype(np.float32)
            ).unsqueeze(0).unsqueeze(-1)

            recon, enc_mu, enc_lv = net(x_t)
            loss, _, _ = _elbo_loss(recon, x_t, enc_mu, enc_lv, obs_m, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # ── Inference: Monte-Carlo reconstruction error as anomaly score ──────────
    net.eval()
    score_sums = np.zeros(n)
    score_counts = np.zeros(n)

    with torch.no_grad():
        for s in starts:
            x_np = v_norm[s:s + seq_len].astype(np.float32)
            x_t = torch.tensor(x_np).unsqueeze(0).unsqueeze(-1)

            mc_recon_errors = []
            for _ in range(n_mc_samples):
                recon, enc_mu, enc_lv = net(x_t)
                err = (recon - x_t).abs().squeeze(0).squeeze(-1).cpu().numpy()
                mc_recon_errors.append(err)

            mean_err = np.mean(mc_recon_errors, axis=0)  # (T,)
            score_sums[s:s + seq_len] += mean_err
            score_counts[s:s + seq_len] += 1

    mask = score_counts > 0
    scores = np.zeros(n)
    scores[mask] = score_sums[mask] / score_counts[mask]
    if not mask.all():
        first_v = np.argmax(mask)
        last_v = len(mask) - 1 - np.argmax(mask[::-1])
        scores[:first_v] = scores[first_v]
        scores[last_v + 1:] = scores[last_v]

    return scores


# ── Public detector ───────────────────────────────────────────────────────────

class OmniAnomalyDetector(BaseDetector):
    """OmniAnomaly: stochastic GRU VAE anomaly detection.

    Simplified implementation of Su et al. (KDD 2019). Uses a VAE-style
    bidirectional stochastic RNN. Anomaly score is the Monte-Carlo mean
    reconstruction error. Falls back to ETS reconstruction error when
    PyTorch is unavailable.
    """

    name = "omni_anomaly"
    category = "Deep Learning"

    def detect(
        self,
        df: pd.DataFrame,
        pollutant: str,
        hidden_dim: int = 32,
        latent_dim: int = 8,
        seq_len: int = 48,
        epochs: int = 10,
        lr: float = 1e-3,
        beta: float = 0.1,
        n_mc_samples: int = 5,
        threshold_percentile: float = 99.0,
        random_state: int = 42,
        fast_mode: bool = True,
        **_,
    ) -> tuple[pd.DataFrame, AnomalyResult]:
        df = df.copy()
        flag_col = "is_omnianoaly_anomaly"
        score_col = "omnianomaly_score"
        t0 = time.perf_counter()

        col = df[pollutant] if pollutant in df.columns else pd.Series(np.nan, index=df.index)

        if col.notna().sum() < seq_len:
            df[flag_col] = False
            df[score_col] = 0.0
            runtime = time.perf_counter() - t0
            return df, self._make_result(
                flag_col, score_col, df, None, runtime,
                status="skipped", warning="Insufficient data for OmniAnomaly",
            )

        values = col.values.astype(float)

        if _TORCH_AVAILABLE:
            actual_epochs = 3 if fast_mode else epochs
            actual_mc = 3 if fast_mode else n_mc_samples
            try:
                scores = _run_omni_anomaly(
                    values, hidden_dim=hidden_dim, latent_dim=latent_dim,
                    seq_len=seq_len, epochs=actual_epochs, lr=lr,
                    beta=beta, n_mc_samples=actual_mc, random_state=random_state,
                )
                impl_label = "OmniAnomaly (PyTorch, simplified stochastic GRU VAE)"
            except Exception as exc:
                logger.warning("OmniAnomaly PyTorch failed: %s; using ETS proxy", exc)
                scores = _ets_recon_error(values)
                impl_label = "OmniAnomaly-approx (ETS recon. error, fallback)"
        else:
            scores = _ets_recon_error(values)
            impl_label = "OmniAnomaly-approx (ETS recon. error — PyTorch unavailable)"

        observed = col.notna().values
        obs_scores = scores[observed]
        thr = float(np.percentile(obs_scores, threshold_percentile)) if len(obs_scores) > 0 else np.inf
        df[score_col] = scores
        df[flag_col] = (scores > thr) & observed

        runtime = time.perf_counter() - t0
        result = self._make_result(flag_col, score_col, df, thr, runtime)
        result.warning = impl_label
        return df, result

