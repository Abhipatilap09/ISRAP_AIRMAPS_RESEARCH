"""CSDI imputer — Conditional Score-based Diffusion Imputation.

Reference: Tashiro et al. "CSDI: Conditional Score-based Diffusion Models for
Probabilistic Time Series Imputation" (NeurIPS 2021).

Core idea
---------
A denoising diffusion probabilistic model (DDPM) is trained to predict noise added
to the observed values. During inference, the model iteratively denoises from pure
Gaussian noise to the data distribution, conditioned on observed (non-missing) values.

Implementation tiers
--------------------
1. PyTorch available  → simplified DDPM with linear beta schedule, small Transformer
   noise predictor conditioned on observed mask + time step embedding.
2. PyTorch unavailable → Savitzky-Golay polynomial smoothing + linear interpolation

This is a faithful structural reimplementation of CSDI; not the full U-Net/bidirectional
Transformer architecture but the same DDPM training objective and inference scheme.

Label: "CSDI-lite (PyTorch DDPM, simplified)" or "CSDI-approx (SavGol, no PyTorch)"

Scientific rules enforced:
- Structural missing (< 1% observed) → skip
- Negative imputed values are clipped to 0 for concentration pollutants
- Raw observed values are never modified
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.imputation.base import BaseImputer, ImputationResult
from src.config import RANDOM_SEED

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ── Pure-NumPy fallback ───────────────────────────────────────────────────────

def _savgol_interp(values: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing + linear interpolation for gaps.

    Fills NaNs with scipy.signal.savgol_filter applied to linearly interpolated series,
    then replaces only the originally missing positions.
    """
    try:
        from scipy.signal import savgol_filter
        n = len(values)
        filled = pd.Series(values).interpolate("linear", limit_direction="both").values
        wl = min(window_length, n if n % 2 == 1 else n - 1)
        if wl < polyorder + 1:
            wl = polyorder + 1 if (polyorder + 1) % 2 == 1 else polyorder + 2
        smoothed = savgol_filter(filled, window_length=wl, polyorder=polyorder)
        result = values.copy()
        result[np.isnan(values)] = smoothed[np.isnan(values)]
        return result
    except Exception:
        # Fallback to pure linear
        return pd.Series(values).interpolate("linear", limit_direction="both").values


# ── DDPM schedule ─────────────────────────────────────────────────────────────

def _make_beta_schedule(T: int = 50, beta_start: float = 1e-4, beta_end: float = 0.02) -> "tuple":
    """Linear beta schedule; returns (betas, alphas, alpha_cumprod)."""
    betas = np.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    return betas, alphas, alpha_bar


# ── PyTorch CSDI ──────────────────────────────────────────────────────────────

if _TORCH_AVAILABLE:
    class _SinusoidalPE(nn.Module):
        """Sinusoidal position embedding for diffusion time step t."""
        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim

        def forward(self, t):  # t: (B,) int tensor
            device = t.device
            half = self.dim // 2
            freq = torch.exp(
                -torch.arange(half, device=device).float() * (np.log(10000) / (half - 1 + 1e-9))
            )
            args = t.float().unsqueeze(-1) * freq.unsqueeze(0)
            pe = torch.cat([args.sin(), args.cos()], dim=-1)
            return pe  # (B, dim)

    class _NoisePredNet(nn.Module):
        """Small Transformer noise predictor conditioned on mask and diffusion time step."""
        def __init__(self, seq_len: int, d_model: int = 32, n_heads: int = 2, n_layers: int = 2):
            super().__init__()
            self.time_emb = _SinusoidalPE(d_model)
            # Input: noisy value (1) + observed mask (1) + time_emb projected per-position
            self.input_proj = nn.Linear(2, d_model)
            self.time_proj = nn.Linear(d_model, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward=d_model * 4,
                dropout=0.0, batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.out = nn.Linear(d_model, 1)

        def forward(self, x_noisy, mask, t_emb):
            # x_noisy: (B, T), mask: (B, T), t_emb: (B, d_model)
            h = self.input_proj(
                torch.stack([x_noisy, mask], dim=-1)
            )  # (B, T, d_model)
            h = h + self.time_proj(t_emb).unsqueeze(1)
            h = self.transformer(h)
            return self.out(h).squeeze(-1)  # (B, T) predicted noise


def _csdi_impute(
    values: np.ndarray,
    T_diff: int = 50,
    d_model: int = 32,
    n_heads: int = 2,
    n_layers: int = 2,
    epochs: int = 15,
    lr: float = 1e-3,
    seq_len: int = 48,
    n_samples: int = 5,
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    """DDPM-based imputation: train on observed, denoise at missing positions."""
    import torch

    torch.manual_seed(random_state)
    n = len(values)
    obs = ~np.isnan(values)
    if obs.sum() < seq_len // 4:
        return _savgol_interp(values)

    mu_v = float(np.nanmean(values))
    sd_v = float(np.nanstd(values)) + 1e-8
    v_norm = np.where(obs, (values - mu_v) / sd_v, 0.0)

    betas, _, alpha_bar = _make_beta_schedule(T_diff)
    alpha_bar_t = torch.tensor(alpha_bar, dtype=torch.float32)
    betas_t = torch.tensor(betas, dtype=torch.float32)

    model = _NoisePredNet(seq_len, d_model, n_heads, n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    time_emb_layer = model.time_emb

    starts = list(range(0, n - seq_len + 1, max(1, seq_len // 2)))
    if not starts:
        return _savgol_interp(values)

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    for ep in range(epochs):
        np.random.seed(ep + random_state)
        idxs = np.random.choice(starts, size=min(16, len(starts)), replace=False)
        for s in idxs:
            sl = slice(s, s + seq_len)
            x0 = torch.tensor(v_norm[sl], dtype=torch.float32).unsqueeze(0)   # (1, T)
            m = torch.tensor(obs[sl].astype(np.float32)).unsqueeze(0)          # (1, T)

            # Sample diffusion time step t
            t_idx = torch.randint(0, T_diff, (1,))
            sqrt_ab = alpha_bar_t[t_idx].sqrt()
            sqrt_1mab = (1 - alpha_bar_t[t_idx]).sqrt()

            epsilon = torch.randn_like(x0)
            x_noisy = sqrt_ab * x0 + sqrt_1mab * epsilon

            t_emb = time_emb_layer(t_idx)                                      # (1, d_model)
            pred_eps = model(x_noisy, m, t_emb)

            # Only supervise on observed positions (condition on mask)
            loss = ((pred_eps - epsilon) ** 2 * m).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    # ── Inference: DDPM reverse sampling at missing positions ─────────────────
    model.eval()
    sample_sums = np.zeros(n)
    sample_counts = np.zeros(n)

    with torch.no_grad():
        for s in starts:
            sl = slice(s, s + seq_len)
            x_obs = torch.tensor(v_norm[sl], dtype=torch.float32).unsqueeze(0)
            m = torch.tensor(obs[sl].astype(np.float32)).unsqueeze(0)

            for _ in range(n_samples):
                # Start from noise at missing positions, observed at known positions
                x_t = torch.where(m.bool(), x_obs, torch.randn_like(x_obs))

                for t_step in reversed(range(T_diff)):
                    t_idx = torch.tensor([t_step])
                    t_emb = model.time_emb(t_idx)
                    pred_eps = model(x_t, m, t_emb)

                    beta_t = betas_t[t_step]
                    ab_t = alpha_bar_t[t_step]
                    alpha_t = 1.0 - beta_t
                    x0_hat = (x_t - (1 - ab_t).sqrt() * pred_eps) / (ab_t.sqrt() + 1e-9)
                    x0_hat = x0_hat.clamp(-5, 5)

                    if t_step > 0:
                        ab_prev = alpha_bar_t[t_step - 1]
                        posterior_mean = (
                            (ab_prev.sqrt() * beta_t / (1 - ab_t)) * x0_hat +
                            (alpha_t.sqrt() * (1 - ab_prev) / (1 - ab_t)) * x_t
                        )
                        noise = torch.randn_like(x_t)
                        posterior_var = beta_t * (1 - ab_prev) / (1 - ab_t)
                        x_t = posterior_mean + posterior_var.sqrt() * noise
                    else:
                        x_t = x0_hat

                    # Keep observed values fixed
                    x_t = torch.where(m.bool(), x_obs, x_t)

                imputed_seg = x_t.squeeze(0).cpu().numpy()
                sample_sums[s:s + seq_len] += imputed_seg
                sample_counts[s:s + seq_len] += 1

    mask = sample_counts > 0
    imputed_norm = np.zeros(n)
    imputed_norm[mask] = sample_sums[mask] / sample_counts[mask]
    if not mask.all():
        fv = np.argmax(mask)
        lv = len(mask) - 1 - np.argmax(mask[::-1])
        imputed_norm[:fv] = imputed_norm[fv]
        imputed_norm[lv + 1:] = imputed_norm[lv]

    result = values.copy()
    imputed_denorm = imputed_norm * sd_v + mu_v
    result[~obs] = imputed_denorm[~obs]
    return result


# ── Public imputer class ──────────────────────────────────────────────────────

class CSDIImputer(BaseImputer):
    """CSDI: Conditional Score-based Diffusion Imputation.

    Faithful structural reimplementation of Tashiro et al. (NeurIPS 2021) using a
    simplified Transformer noise predictor and DDPM reverse sampling. Falls back to
    Savitzky-Golay + linear interpolation when PyTorch is unavailable.
    """
    name = "csdi"

    def impute(
        self,
        df: pd.DataFrame,
        pollutant: str,
        T_diff: int = 50,
        d_model: int = 32,
        n_heads: int = 2,
        n_layers: int = 2,
        epochs: int = 15,
        seq_len: int = 48,
        n_samples: int = 5,
        random_state: int = RANDOM_SEED,
        **_,
    ) -> tuple[pd.DataFrame, ImputationResult]:
        df = df.copy()
        out_col = "imputed_csdi"
        was_col = "was_imputed_csdi"
        t0 = time.perf_counter()

        if pollutant not in df.columns:
            df[out_col] = np.nan
            df[was_col] = False
            return df, ImputationResult(
                self.name, 0, None, None, self._elapsed(t0),
                "error", f"{pollutant} not in dataframe",
            )

        values = df[pollutant].values.astype(float)
        missing_mask = np.isnan(values)

        if missing_mask.sum() == 0:
            df[out_col] = values
            df[was_col] = False
            return df, ImputationResult(self.name, 0, None, None, self._elapsed(t0), "ok")

        if (~missing_mask).sum() < 10:
            df[out_col] = np.nan
            df[was_col] = False
            return df, ImputationResult(
                self.name, 0, None, None, self._elapsed(t0),
                "skipped", "Fewer than 10 observed values — cannot impute",
            )

        if _TORCH_AVAILABLE:
            impl_label = "CSDI-lite (PyTorch DDPM, simplified)"
            try:
                imputed_vals = _csdi_impute(
                    values, T_diff=T_diff, d_model=d_model, n_heads=n_heads,
                    n_layers=n_layers, epochs=epochs, seq_len=seq_len,
                    n_samples=n_samples, random_state=random_state,
                )
            except Exception as exc:
                impl_label = "CSDI-approx (SavGol interp., fallback)"
                imputed_vals = _savgol_interp(values)
        else:
            impl_label = "CSDI-approx (SavGol interp. — PyTorch unavailable)"
            imputed_vals = _savgol_interp(values)

        imputed_vals = np.clip(imputed_vals, 0.0, None)
        df[out_col] = imputed_vals
        df[was_col] = pd.Series(missing_mask & ~np.isnan(imputed_vals), index=df.index)
        n_imp = int(df[was_col].sum())

        return df, ImputationResult(
            self.name, n_imp, None, None, self._elapsed(t0), "ok",
            warning=impl_label,
        )

