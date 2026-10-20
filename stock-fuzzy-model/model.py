"""Binary direction LSTM. Defaults come from config.py (small 1-layer model)."""
import torch.nn as nn

import config as cfg


class DirectionLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=cfg.HIDDEN,
                 num_layers=cfg.NUM_LAYERS, dropout=cfg.DROPOUT):
        super().__init__()
        # nn.LSTM only applies dropout between layers, never for a single layer.
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=lstm_dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, (h_n, _c_n) = self.lstm(x)
        last = h_n[-1]  # top-layer final hidden state
        return self.sigmoid(self.fc(last)).squeeze(-1)
