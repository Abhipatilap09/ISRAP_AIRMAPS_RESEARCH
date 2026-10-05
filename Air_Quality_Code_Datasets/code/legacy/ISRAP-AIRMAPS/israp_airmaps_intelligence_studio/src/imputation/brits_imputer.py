"""BRITS imputer — bidirectional recurrent imputation for time series.

Implementation tiers
--------------------
1. PyTorch available  → simplified bidirectional LSTM trained end-to-end
2. PyTorch unavailable → faithful pure-NumPy bidirectional exponential smoother
   (bidirectional temporal decay, same structural idea as BRITS but without deep learning)

Label in UI: "BRITS (LSTM)" when PyTorch is present, "BRITS-approx (Bidir. ETS)" otherwise.

Scientific rule enforced:
- Structural missing (0% observed) → skip, return status="skipped"
- Negative concentrations are clipped to 0 after imputation
- Raw observed values are never modified
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.imputation.base import BaseImputer, ImputationResult
from src.config import ERA5_VARS, RANDOM_SEED

# ── Detect PyTorch ────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ── Pure-NumPy bidirectional exponential-smoothing approximation ──────────────

def _bidir_ets_impute(values: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Bidirectional exponential smoother for 1-D float array with NaNs.

    Forward pass: ETS with decay alpha.
    Backward pass: ETS with decay alpha going right-to-left.
    Imputed value = average of forward and backward estimates at missing positions.
    Observed values are never changed.
    """
    n = len(values)
    fwd = np.full(n, np.nan)
    bwd = np.full(n, np.nan)

    # Forward
    state = np.nan
    for i in range(n):
        if not np.isnan(values[i]):
            state = values[i] if np.isnan(state) else (1 - alpha) * state + alpha * values[i]
            fwd[i] = values[i]
        else:
            fwd[i] = state

    # Backward
    state = np.nan
    for i in range(n - 1, -1, -1):
        if not np.isnan(values[i]):
            state = values[i] if np.isnan(state) else (1 - alpha) * state + alpha * values[i]
            bwd[i] = values[i]
        else:
            bwd[i] = state

    imputed = values.copy()
    missing = np.isnan(values)
    both_avail = ~np.isnan(fwd) & ~np.isnan(bwd)
    only_fwd = ~np.isnan(fwd) & np.isnan(bwd)
    only_bwd = np.isnan(fwd) & ~np.isnan(bwd)

    imputed[missing & both_avail] = 0.5 * (fwd[missing & both_avail] + bwd[missing & both_avail])
    imputed[missing & only_fwd] = fwd[missing & only_fwd]
    imputed[missing & only_bwd] = bwd[missing & only_bwd]

    return imputed


# ── PyTorch BRITS core ────────────────────────────────────────────────────────
if _TORCH_AVAILABLE:
    class _BRITSCell(nn.Module):
        """One direction of a simplified BRITS GRU cell with temporal decay."""
        def __init__(self, input_dim: int, hidden_dim: int):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.gru = nn.GRU(input_dim + 1, hidden_dim, batch_first=True)
            self.fc_out = nn.Linear(hidden_dim, 1)
            self.gamma = nn.Linear(1, hidden_dim)

        def forward(self, x: "torch.Tensor", mask: "torch.Tensor",
                    delta: "torch.Tensor") -> "torch.Tensor":
            B, T, D = x.shape
            h = torch.zeros(1, B, self.hidden_dim, device=x.device)
            preds = []
            for t in range(T):
                decay = torch.exp(-torch.relu(self.gamma(delta[:, t:t+1, :])))
                h = h * decay.squeeze(1).unsqueeze(0)
                inp = torch.cat([x[:, t, :], mask[:, t, :1]], dim=-1).unsqueeze(1)
                out, h = self.gru(inp, h)
                preds.append(self.fc_out(out.squeeze(1)))
            return torch.stack(preds, dim=1).squeeze(-1)

    class _BIDIRModel(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int):
            super().__init__()
            self.fwd = _BRITSCell(input_dim, hidden_dim)
            self.bwd = _BRITSCell(input_dim, hidden_dim)
            self.hidden_dim = hidden_dim

        def forward(self, x, mask, delta_fwd, delta_bwd):
            fwd_out = self.fwd(x, mask, delta_fwd)
            bwd_out = self.bwd(x.flip(1), mask.flip(1), delta_bwd.flip(1)).flip(1)
            return 0.5 * (fwd_out + bwd_out)


def _brits_torch(
    values: np.ndarray,
    era5_feats: np.ndarray | None,
    hidden_dim: int = 32,
    epochs: int = 20,
    seq_len: int = 48,
    lr: float = 1e-3,
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    """Train a simplified BRITS model and return imputed array."""
    torch.manual_seed(random_state)
    n = len(values)
    observed = ~np.isnan(values)
    n_obs = observed.sum()
    if n_obs < seq_len * 2:
        return _bidir_ets_impute(values)

    if era5_feats is not None:
        input_dim = 1 + era5_feats.shape[1]
    else:
        input_dim = 1

    # Build input tensor: fill observed with value, NaN with 0
    obs_filled = np.where(observed, values, 0.0)
    mask_arr = observed.astype(float)

    if era5_feats is not None:
        era5_norm = (era5_feats - np.nanmean(era5_feats, axis=0)) / (np.nanstd(era5_feats, axis=0) + 1e-8)
        era5_norm = np.nan_to_num(era5_norm, 0.0)
        feat = np.concatenate([obs_filled[:, None], era5_norm], axis=1)
    else:
        feat = obs_filled[:, None]

    # Time deltas (hours since last observation)
    delta = np.zeros(n)
    last_obs = 0
    for i in range(n):
        if observed[i]:
            last_obs = 0
        else:
            last_obs += 1
        delta[i] = float(last_obs)

    # Normalise target — replace NaN with 0 before tensor conversion
    # (obs_t mask zeros those positions in the loss anyway; NaN would produce NaN gradients)
    v_mean = float(np.nanmean(values))
    v_std = float(np.nanstd(values)) + 1e-8
    y_norm = np.where(observed, (values - v_mean) / v_std, 0.0)

    model = _BIDIRModel(input_dim=input_dim, hidden_dim=hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Train on overlapping segments of length seq_len
    starts = list(range(0, n - seq_len, max(1, seq_len // 4)))
    if not starts:
        return _bidir_ets_impute(values)

    model.train()
    for ep in range(epochs):
        np.random.seed(ep + random_state)
        np.random.shuffle(starts)
        for s in starts[:64]:
            sl = slice(s, s + seq_len)
            obs_sl = observed[sl]
            if obs_sl.sum() < 4:
                continue
            x_t = torch.tensor(feat[sl], dtype=torch.float32).unsqueeze(0)
            m_t = torch.tensor(mask_arr[sl], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).expand_as(x_t)
            d_t = torch.tensor(delta[sl], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            y_t = torch.tensor(y_norm[sl], dtype=torch.float32).unsqueeze(0)
            obs_t = torch.tensor(obs_sl, dtype=torch.float32).unsqueeze(0)

            pred = model(x_t, m_t[:, :, :1], d_t, d_t.flip(1))
            loss = ((pred - y_t) ** 2 * obs_t).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Inference on full sequence
    model.eval()
    with torch.no_grad():
        x_full = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        m_full = torch.tensor(mask_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).expand_as(x_full)
        d_full = torch.tensor(delta, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        pred_full = model(x_full, m_full[:, :, :1], d_full, d_full.flip(1)).squeeze(0).numpy()

    pred_denorm = pred_full * v_std + v_mean
    result = values.copy()
    result[~observed] = pred_denorm[~observed]
    return result


# ── Public imputer class ──────────────────────────────────────────────────────

class BRITSImputer(BaseImputer):
    """Bidirectional recurrent imputation.

    Uses PyTorch LSTM if available; falls back to bidirectional ETS approximation.
    Faithfully models temporal decay and bidirectional context.
    """
    name = "brits"

    def impute(
        self,
        df: pd.DataFrame,
        pollutant: str,
        hidden_dim: int = 32,
        epochs: int = 20,
        seq_len: int = 48,
        alpha: float = 0.3,
        include_era5: bool = True,
        random_state: int = RANDOM_SEED,
        **_,
    ) -> tuple[pd.DataFrame, ImputationResult]:
        df = df.copy()
        out_col = "imputed_brits"
        was_col = "was_imputed_brits"
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

        # Build ERA5 feature matrix
        era5_feats = None
        if include_era5 and _TORCH_AVAILABLE:
            era5_cols = [c for c in ERA5_VARS if c in df.columns]
            if era5_cols:
                era5_feats = df[era5_cols].values.astype(float)

        if _TORCH_AVAILABLE:
            impl_label = "BRITS-LSTM (PyTorch)"
            try:
                imputed_vals = _brits_torch(
                    values, era5_feats,
                    hidden_dim=hidden_dim, epochs=epochs,
                    seq_len=seq_len, random_state=random_state,
                )
            except Exception as exc:
                impl_label = "BRITS-approx (Bidir. ETS, fallback)"
                imputed_vals = _bidir_ets_impute(values, alpha=alpha)
        else:
            impl_label = "BRITS-approx (Bidir. ETS — PyTorch unavailable)"
            imputed_vals = _bidir_ets_impute(values, alpha=alpha)

        imputed_vals = np.clip(imputed_vals, 0.0, None)
        df[out_col] = imputed_vals
        df[was_col] = pd.Series(missing_mask & ~np.isnan(imputed_vals), index=df.index)
        n_imp = int(df[was_col].sum())

        return df, ImputationResult(
            self.name, n_imp, None, None, self._elapsed(t0), "ok",
            warning=impl_label,
        )

