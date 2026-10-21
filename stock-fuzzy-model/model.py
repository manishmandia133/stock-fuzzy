"""Neural network models for ^GSPC regression.

RegressionLSTM  – base model, fixed or fuzzy LR (variants: fixed, M2, M3, M6).
FuzzyMoELSTM    – 3-expert mixture blended by Ruspini vol_rank weights (M4).
FuzzyInputLSTM  – membership degrees appended to each timestep input (M5).

All models output a scalar log-return prediction per sample.
Reconstruct price: pred_close = close_t * exp(pred_log_return).
"""
import torch
from torch import nn

import config as cfg


class RegressionLSTM(nn.Module):
    """Standard LSTM → scalar regression. Used by fixed, M2, M3, M6."""

    def __init__(self, n_features,
                 hidden_size=cfg.HIDDEN_SIZE,
                 num_layers=cfg.NUM_LAYERS,
                 dropout=cfg.DROPOUT):
        super().__init__()
        lstm_drop = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers,
                            batch_first=True, dropout=lstm_drop)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x, vr=None, tr=None):
        # vr, tr accepted but ignored — same API as specialised models
        _, (h, _) = self.lstm(x)
        return self.fc(self.drop(h[-1])).squeeze(-1)


class FuzzyMoELSTM(nn.Module):
    """M4: three LSTM experts (Low / Med / High volatility regime).

    Outputs are blended by Ruspini membership weights of vol_rank,
    so the mixture is end-to-end differentiable.

    Expert hidden size = hidden_size // 2 to keep parameter counts similar
    to the single-expert baseline.
    """

    def __init__(self, n_features,
                 hidden_size=cfg.HIDDEN_SIZE,
                 num_layers=cfg.NUM_LAYERS,
                 dropout=cfg.DROPOUT,
                 n_experts=3):
        super().__init__()
        expert_h  = max(hidden_size // 2, 16)
        lstm_drop = dropout if num_layers > 1 else 0.0
        self.experts = nn.ModuleList([
            nn.LSTM(n_features, expert_h, num_layers,
                    batch_first=True, dropout=lstm_drop)
            for _ in range(n_experts)
        ])
        self.heads = nn.ModuleList([nn.Linear(expert_h, 1)
                                    for _ in range(n_experts)])
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _ruspini_weights(vr: torch.Tensor):
        """Ruspini triangular partition → (B, 3) weight tensor."""
        vr  = vr.clamp(0.0, 1.0)
        w_l = (1.0 - 2.0 * vr).clamp(0.0, 1.0)           # Low
        w_m = (1.0 - (2.0 * vr - 1.0).abs()).clamp(0.0, 1.0)  # Med
        w_h = (2.0 * vr - 1.0).clamp(0.0, 1.0)           # High
        return torch.stack([w_l, w_m, w_h], dim=1)        # (B, 3)

    def forward(self, x, vr=None, tr=None):
        B = x.shape[0]
        if vr is None:
            vr = torch.full((B,), 0.5, device=x.device)
        vr = vr.float().to(x.device)

        weights = self._ruspini_weights(vr)                # (B, 3)

        preds = []
        for expert, head in zip(self.experts, self.heads):
            _, (h, _) = expert(x)
            p = head(self.drop(h[-1])).squeeze(-1)         # (B,)
            preds.append(p)
        preds = torch.stack(preds, dim=1)                  # (B, 3)
        return (preds * weights).sum(dim=1)                # (B,)


class FuzzyInputLSTM(nn.Module):
    """M5: membership degrees appended to every timestep of the input.

    Appends 5 scalars per timestep:
      μ_Low(vol), μ_Med(vol), μ_High(vol), μ_Down(trend), μ_Up(trend)

    All five are fixed during the forward pass (Ruspini triangular).
    """
    N_EXTRA = 5  # extra input features

    def __init__(self, n_features,
                 hidden_size=cfg.HIDDEN_SIZE,
                 num_layers=cfg.NUM_LAYERS,
                 dropout=cfg.DROPOUT):
        super().__init__()
        lstm_drop = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(n_features + self.N_EXTRA, hidden_size,
                            num_layers, batch_first=True, dropout=lstm_drop)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size, 1)

    @staticmethod
    def _memberships(vr: torch.Tensor, tr: torch.Tensor) -> torch.Tensor:
        """Return (B, 5) membership tensor: [μvL, μvM, μvH, μtD, μtU]."""
        vr = vr.clamp(0.0, 1.0)
        tr = tr.clamp(0.0, 1.0)
        mu_vL = (1.0 - 2.0 * vr).clamp(0.0, 1.0)
        mu_vM = (1.0 - (2.0 * vr - 1.0).abs()).clamp(0.0, 1.0)
        mu_vH = (2.0 * vr - 1.0).clamp(0.0, 1.0)
        mu_tD = (1.0 - tr).clamp(0.0, 1.0)   # trend Down
        mu_tU = tr.clamp(0.0, 1.0)            # trend Up
        return torch.stack([mu_vL, mu_vM, mu_vH, mu_tD, mu_tU], dim=1)  # (B, 5)

    def forward(self, x, vr=None, tr=None):
        B, T, _ = x.shape
        if vr is None:
            vr = torch.full((B,), 0.5, device=x.device)
        if tr is None:
            tr = torch.full((B,), 0.5, device=x.device)
        vr = vr.float().to(x.device)
        tr = tr.float().to(x.device)

        m = self._memberships(vr, tr)          # (B, 5)
        m = m.unsqueeze(1).expand(B, T, self.N_EXTRA)   # (B, T, 5)
        x_aug = torch.cat([x, m], dim=-1)      # (B, T, F+5)

        _, (h, _) = self.lstm(x_aug)
        return self.fc(self.drop(h[-1])).squeeze(-1)


def build_model(variant: str, n_features: int,
                hidden_size: int = None,
                num_layers: int = None,
                dropout: float = None) -> nn.Module:
    """Factory: return the correct model for a given variant."""
    hidden_size = cfg.HIDDEN_SIZE if hidden_size is None else hidden_size
    num_layers  = cfg.NUM_LAYERS  if num_layers  is None else num_layers
    dropout     = cfg.DROPOUT     if dropout     is None else dropout

    if variant == "M4":
        return FuzzyMoELSTM(n_features, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout)
    if variant == "M5":
        return FuzzyInputLSTM(n_features, hidden_size=hidden_size,
                              num_layers=num_layers, dropout=dropout)
    return RegressionLSTM(n_features, hidden_size=hidden_size,
                          num_layers=num_layers, dropout=dropout)
