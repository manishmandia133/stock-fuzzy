"""Neural network models for ^GSPC regression.

Fuzzy_MoE          – Ruspini-gated mixture of 3 regime expert LSTMs.
Fuzzy_Input        – membership degrees appended to every input timestep.
Fuzzy_Adaptive_MoE – Fuzzy_MoE; train.py applies fuzzy adaptive grad clipping.
Fuzzy_ANFIS        – hybrid: membership inputs + fuzzy-gated consequent heads.

All models output a scalar log-return prediction per sample.
Reconstruct price: pred_close = close_t * exp(pred_log_return).
"""
import torch
from torch import nn

import config as cfg


def _ruspini_vol_weights(vr: torch.Tensor) -> torch.Tensor:
    """Ruspini triangular partition of volatility → (batch, 3) weights."""
    vr = vr.clamp(0.0, 1.0)
    low = (1.0 - 2.0 * vr).clamp(0.0, 1.0)
    med = (1.0 - torch.abs(2.0 * vr - 1.0)).clamp(0.0, 1.0)
    high = (2.0 * vr - 1.0).clamp(0.0, 1.0)
    return torch.stack([low, med, high], dim=1)


def _prepare_inputs(x, vr, tr):
    """Normalise optional regime inputs to (batch,) tensors on x's device/dtype."""
    batch_size = x.shape[0]
    device, dtype = x.device, x.dtype

    if vr is None:
        vr = torch.full((batch_size,), 0.5, device=device, dtype=dtype)
    else:
        vr = vr.to(device=device, dtype=dtype).reshape(batch_size)

    if tr is None:
        tr = torch.full((batch_size,), 0.5, device=device, dtype=dtype)
    else:
        tr = tr.to(device=device, dtype=dtype).reshape(batch_size)

    return vr, tr


class FuzzyInputLSTM(nn.Module):
    """Fuzzy-membership-augmented LSTM for next-day log-return regression.

    Inputs:
        x:  (batch, sequence_length, n_features)
        vr: (batch,) normalized volatility input in [0, 1]
        tr: (batch,) normalized trend input in [0, 1]

    Output:
        Predicted next-day log return: (batch,)
    """

    N_EXTRA = 5

    def __init__(
        self,
        n_features,
        hidden_size=cfg.HIDDEN_SIZE,
        num_layers=cfg.NUM_LAYERS,
        dropout=cfg.DROPOUT,
    ):
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=n_features + self.N_EXTRA,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, 1)

    @staticmethod
    def fuzzy_memberships(vr, tr):
        # Both inputs must be normalized to [0, 1].
        vr = vr.clamp(0.0, 1.0)
        tr = tr.clamp(0.0, 1.0)

        # Ruspini triangular volatility memberships.
        low_vol = (1.0 - 2.0 * vr).clamp(0.0, 1.0)
        med_vol = (1.0 - torch.abs(2.0 * vr - 1.0)).clamp(0.0, 1.0)
        high_vol = (2.0 * vr - 1.0).clamp(0.0, 1.0)

        # Trend memberships.
        down_trend = 1.0 - tr
        up_trend = tr

        return torch.stack(
            [low_vol, med_vol, high_vol, down_trend, up_trend],
            dim=1,
        )

    def forward(self, x, vr=None, tr=None):
        batch_size, seq_len, _ = x.shape
        vr, tr = _prepare_inputs(x, vr, tr)

        memberships = self.fuzzy_memberships(vr, tr)

        # Add the same sample-level memberships to each timestep.
        memberships = memberships.unsqueeze(1).expand(
            batch_size, seq_len, self.N_EXTRA
        )

        x_augmented = torch.cat([x, memberships], dim=-1)

        _, (hidden, _) = self.lstm(x_augmented)

        last_hidden = self.dropout(hidden[-1])
        predicted_log_return = self.output_layer(last_hidden).squeeze(-1)

        return predicted_log_return


class FuzzyMoELSTM(nn.Module):
    """3-expert LSTM mixture blended by Ruspini volatility membership weights.

    Expert hidden size is ``hidden_size // 2`` (minimum 16) so the parameter
    count stays close to the single-expert baseline.
    """

    def __init__(
        self,
        n_features,
        hidden_size=cfg.HIDDEN_SIZE,
        num_layers=cfg.NUM_LAYERS,
        dropout=cfg.DROPOUT,
        n_experts=3,
    ):
        super().__init__()
        if n_experts != 3:
            raise ValueError("Ruspini gating requires exactly 3 experts")

        expert_hidden = max(hidden_size // 2, 16)
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.experts = nn.ModuleList([
            nn.LSTM(
                input_size=n_features,
                hidden_size=expert_hidden,
                num_layers=num_layers,
                batch_first=True,
                dropout=lstm_dropout,
            )
            for _ in range(n_experts)
        ])
        self.heads = nn.ModuleList([
            nn.Linear(expert_hidden, 1) for _ in range(n_experts)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, vr=None, tr=None):
        vr, _ = _prepare_inputs(x, vr, tr)
        weights = _ruspini_vol_weights(vr)  # (batch, n_experts)

        predictions = []
        for expert, head in zip(self.experts, self.heads):
            _, (hidden, _) = expert(x)
            predictions.append(head(self.dropout(hidden[-1])).squeeze(-1))

        predictions = torch.stack(predictions, dim=1)
        return (predictions * weights).sum(dim=1)


class FuzzyANFISLSTM(nn.Module):
    """Hybrid ANFIS-style model: fuzzy membership inputs + fuzzy-gated heads."""

    N_EXTRA = FuzzyInputLSTM.N_EXTRA

    def __init__(
        self,
        n_features,
        hidden_size=cfg.HIDDEN_SIZE,
        num_layers=cfg.NUM_LAYERS,
        dropout=cfg.DROPOUT,
        n_experts=3,
    ):
        super().__init__()
        if n_experts != 3:
            raise ValueError("Ruspini gating requires exactly 3 experts")

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=n_features + self.N_EXTRA,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(n_experts)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, vr=None, tr=None):
        batch_size, seq_len, _ = x.shape
        vr, tr = _prepare_inputs(x, vr, tr)

        memberships = FuzzyInputLSTM.fuzzy_memberships(vr, tr)
        memberships = memberships.unsqueeze(1).expand(
            batch_size, seq_len, self.N_EXTRA
        )

        x_augmented = torch.cat([x, memberships], dim=-1)
        _, (hidden, _) = self.lstm(x_augmented)

        last_hidden = self.dropout(hidden[-1])
        predictions = torch.cat(
            [head(last_hidden) for head in self.heads], dim=1
        )

        weights = _ruspini_vol_weights(vr)
        return (predictions * weights).sum(dim=1)


def build_model(
    variant: str,
    n_features: int,
    hidden_size: int = None,
    num_layers: int = None,
    dropout: float = None,
) -> nn.Module:
    """Factory returning the model for a configured variant."""
    hidden_size = cfg.HIDDEN_SIZE if hidden_size is None else hidden_size
    num_layers = cfg.NUM_LAYERS if num_layers is None else num_layers
    dropout = cfg.DROPOUT if dropout is None else dropout

    kwargs = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
    }

    if variant in ("Fuzzy_MoE", "Fuzzy_Adaptive_MoE"):
        return FuzzyMoELSTM(n_features, **kwargs)
    if variant == "Fuzzy_Input":
        return FuzzyInputLSTM(n_features, **kwargs)
    if variant == "Fuzzy_ANFIS":
        return FuzzyANFISLSTM(n_features, **kwargs)

    raise ValueError(
        f"Unknown variant '{variant}'. Must be one of {cfg.VARIANTS}"
    )
