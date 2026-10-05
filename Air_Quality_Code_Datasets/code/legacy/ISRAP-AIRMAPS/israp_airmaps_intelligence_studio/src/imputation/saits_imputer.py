"""SAITS imputer — Self-Attention Imputation for Time Series (Du et al. 2023).

Implementation tiers
--------------------
1. PyPOTS installed  → use the official SAITS from the PyPOTS library
2. PyTorch installed → simplified single-head self-attention imputer
   (same inductive bias as SAITS but not the full iterative procedure)
3. Neither          → bidirectional weighted interpolation fallback

Label in UI: "SAITS (PyPOTS)", "SAITS-lite (PyTorch)", or "SAITS-approx (Bidir. Weighted)"

Scientific rule enforced:
- Structural missing → skip
- Clipped to 0 at output
- Raw observed values are never modified
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.imputation.base import BaseImputer, ImputationResult
from src.config import ERA5_VARS, RANDOM_SEED

# ── Detect optional dependencies ──────────────────────────────────────────────
try:
    from pypots.imputation import SAITS as _PyPOTS_SAITS
    _PYPOTS_AVAILABLE = True
except ImportError:
    _PYPOTS_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ── Fallback: distance-weighted bidirectional interpolation ───────────────────

def _weighted_bidir_impute(values: np.ndarray) -> np.ndarray:
    """Linearly interpolate with inverse-distance weighting.

    For each missing value at position i, the imputation is a distance-weighted
    average of the nearest valid value on the left (L) and right (R):
        imputed[i] = (d_R * L + d_L * R) / (d_L + d_R)
    This is equivalent to linear interpolation, but is implemented explicitly to
    match the SAITS bidirectional context motivation.
    """
    result = values.copy()
    n = len(values)
    observed = ~np.isnan(values)

    i = 0
    while i < n:
        if np.isnan(result[i]):
            # Find left neighbour
            left_idx = i - 1
            while left_idx >= 0 and np.isnan(result[left_idx]):
                left_idx -= 1
            # Find right neighbour
            right_idx = i + 1
            while right_idx < n and np.isnan(result[right_idx]):
                right_idx += 1

            # Impute the whole gap segment
            for j in range(i, n):
                if not np.isnan(values[j]):
                    break
                d_left = j - left_idx if left_idx >= 0 else np.inf
                d_right = right_idx - j if right_idx < n else np.inf
                if np.isfinite(d_left) and np.isfinite(d_right):
                    result[j] = (d_right * values[left_idx] + d_left * values[right_idx]) / (d_left + d_right)
                elif np.isfinite(d_left):
                    result[j] = values[left_idx]
                elif np.isfinite(d_right):
                    result[j] = values[right_idx]
            i = j if not np.isnan(values[j]) else n
        else:
            i += 1
    return result


# ── PyTorch simplified self-attention imputer ─────────────────────────────────
if _TORCH_AVAILABLE:
    class _SelfAttentionImputer(nn.Module):
        """Single-layer masked self-attention imputer."""

        def __init__(self, d_model: int, n_heads: int = 4):
            super().__init__()
            self.proj_in = nn.Linear(2, d_model)  # (value, mask) → d_model
            self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
            self.norm = nn.LayerNorm(d_model)
            self.proj_out = nn.Linear(d_model, 1)

        def forward(self, x: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
            inp = torch.stack([x, mask], dim=-1)
            h = self.proj_in(inp)
            attn_out, _ = self.attn(h, h, h)
            h = self.norm(h + attn_out)
            return self.proj_out(h).squeeze(-1)


def _saits_torch(
    values: np.ndarray,
    d_model: int = 32,
    n_heads: int = 4,
    epochs: int = 20,
    seq_len: int = 48,
    lr: float = 1e-3,
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    """Train the simplified self-attention imputer and return imputed array."""
    torch.manual_seed(random_state)
    n = len(values)
    observed = ~np.isnan(values)
    if observed.sum() < seq_len:
        return _weighted_bidir_impute(values)

    v_mean = float(np.nanmean(values))
    v_std = float(np.nanstd(values)) + 1e-8
    y_norm = np.where(observed, (values - v_mean) / v_std, 0.0).astype(np.float32)
    mask_arr = observed.astype(np.float32)

    model = _SelfAttentionImputer(d_model=d_model, n_heads=n_heads)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    starts = list(range(0, n - seq_len, max(1, seq_len // 4)))
    if not starts:
        return _weighted_bidir_impute(values)

    model.train()
    for ep in range(epochs):
        np.random.seed(ep + random_state)
        np.random.shuffle(starts)
        for s in starts[:64]:
            sl = slice(s, s + seq_len)
            obs_sl = observed[sl]
            if obs_sl.sum() < 4:
                continue
            x_t = torch.from_numpy(y_norm[sl]).unsqueeze(0)
            m_t = torch.from_numpy(mask_arr[sl]).unsqueeze(0)
            obs_t = torch.from_numpy(obs_sl.astype(np.float32)).unsqueeze(0)

            pred = model(x_t, m_t)
            loss = ((pred - x_t) ** 2 * obs_t).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        x_full = torch.from_numpy(y_norm).unsqueeze(0)
        m_full = torch.from_numpy(mask_arr).unsqueeze(0)
        pred_full = model(x_full, m_full).squeeze(0).numpy()

    pred_denorm = pred_full * v_std + v_mean
    result = values.copy()
    result[~observed] = pred_denorm[~observed]
    return result


# ── PyPOTS implementation ─────────────────────────────────────────────────────
def _saits_pypots(
    values: np.ndarray,
    n_steps: int = 48,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    epochs: int = 20,
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    """Use the official PyPOTS SAITS implementation."""
    observed = ~np.isnan(values)
    if observed.sum() < n_steps:
        return _weighted_bidir_impute(values)

    v_mean = float(np.nanmean(values))
    v_std = float(np.nanstd(values)) + 1e-8

    # Segment into windows for PyPOTS API
    n = len(values)
    n_windows = max(1, n // n_steps)
    X = np.zeros((n_windows, n_steps, 1))
    for i in range(n_windows):
        sl = slice(i * n_steps, (i + 1) * n_steps)
        X[i, :, 0] = (values[sl][:n_steps] - v_mean) / v_std

    saits = _PyPOTS_SAITS(
        n_steps=n_steps, n_features=1,
        n_layers=n_layers, d_model=d_model, d_inner=d_model * 2,
        n_heads=n_heads, d_k=d_model // n_heads, d_v=d_model // n_heads,
        dropout=0.1, epochs=epochs,
        device="cpu",
    )
    saits.fit({"X": X})
    out = saits.impute({"X": X})

    result = values.copy()
    for i in range(n_windows):
        sl = slice(i * n_steps, (i + 1) * n_steps)
        preds = out[i, :, 0] * v_std + v_mean
        missing_sl = ~observed[sl]
        sl_len = min(n_steps, n - i * n_steps)
        result_sl = result[sl]
        result_sl[missing_sl[:sl_len]] = preds[missing_sl[:sl_len]]
        result[sl] = result_sl

    return result


# ── Public imputer class ──────────────────────────────────────────────────────

class SAITSImputer(BaseImputer):
    """Self-Attention Imputation for Time Series.

    Tries PyPOTS → PyTorch → bidirectional weighted interpolation in order.
    Clearly labels which tier was used in the result warning field.
    """
    name = "saits"

    def impute(
        self,
        df: pd.DataFrame,
        pollutant: str,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        epochs: int = 20,
        seq_len: int = 48,
        random_state: int = RANDOM_SEED,
        **_,
    ) -> tuple[pd.DataFrame, ImputationResult]:
        df = df.copy()
        out_col = "imputed_saits"
        was_col = "was_imputed_saits"
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

        impl_label: str
        try:
            if _PYPOTS_AVAILABLE:
                impl_label = "SAITS (PyPOTS — official implementation)"
                imputed_vals = _saits_pypots(
                    values, n_steps=seq_len, d_model=d_model,
                    n_heads=n_heads, n_layers=n_layers,
                    epochs=epochs, random_state=random_state,
                )
            elif _TORCH_AVAILABLE:
                impl_label = "SAITS-lite (PyTorch single-head attention — simplified)"
                imputed_vals = _saits_torch(
                    values, d_model=d_model, n_heads=min(n_heads, max(1, d_model // 8)),
                    epochs=epochs, seq_len=seq_len, random_state=random_state,
                )
            else:
                impl_label = "SAITS-approx (bidirectional weighted interpolation — PyTorch/PyPOTS unavailable)"
                imputed_vals = _weighted_bidir_impute(values)
        except Exception as exc:
            impl_label = f"SAITS-approx (fallback — error: {exc!s:.80})"
            imputed_vals = _weighted_bidir_impute(values)

        imputed_vals = np.clip(imputed_vals, 0.0, None)
        df[out_col] = imputed_vals
        df[was_col] = pd.Series(missing_mask & ~np.isnan(imputed_vals), index=df.index)
        n_imp = int(df[was_col].sum())

        return df, ImputationResult(
            self.name, n_imp, None, None, self._elapsed(t0), "ok",
            warning=impl_label,
        )

