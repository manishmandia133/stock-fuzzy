"""Single fuzzy network for ^GSPC regression.

Fuzzy_Input – membership degrees appended to every input timestep.

The model outputs a scalar log-return prediction per sample.
Reconstruct price: pred_close = close_t * exp(pred_log_return).
"""
import torch
from torch import nn

import config as cfg


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


def build_model(
    variant: str,
    n_features: int,
    hidden_size: int | None = None,
    num_layers: int | None = None,
    dropout: float | None = None,
) -> nn.Module:
    """Build the project's only active fuzzy model."""
    if variant != "Fuzzy_Input":
        raise ValueError(
            f"Unknown variant '{variant}'. Must be one of {cfg.VARIANTS}"
        )

    hidden_size = cfg.HIDDEN_SIZE if hidden_size is None else hidden_size
    num_layers = cfg.NUM_LAYERS if num_layers is None else num_layers
    dropout = cfg.DROPOUT if dropout is None else dropout

    return FuzzyInputLSTM(
        n_features=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
