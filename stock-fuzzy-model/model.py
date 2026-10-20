"""Binary direction LSTM. Defaults come from config.py (small 1-layer model)."""

import torch
from torch import nn

import config as cfg


class DirectionLSTM(nn.Module):
    """One-layer LSTM followed by a linear binary classifier."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = cfg.HIDDEN,
        num_layers: int = cfg.NUM_LAYERS,
        dropout: float = cfg.DROPOUT,
    ) -> None:
        super().__init__()
        # nn.LSTM only applies dropout between layers, never for a single layer.
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return positive-class probabilities for a batch of windows."""
        _out, (h_n, _c_n) = self.lstm(x)
        last = h_n[-1]  # top-layer final hidden state
        return self.sigmoid(self.fc(last)).squeeze(-1)
