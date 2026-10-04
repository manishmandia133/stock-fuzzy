"""Binary direction LSTM (PLAN: 1 layer, 64 units, dropout 0.2)."""
import torch.nn as nn


class DirectionLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, (h_n, _c_n) = self.lstm(x)
        last = self.drop(h_n[-1])
        return self.sigmoid(self.fc(last)).squeeze(-1)
