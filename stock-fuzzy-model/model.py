"""Binary direction LSTM (report: 2 layers, 64 units, dropout 0.2 between layers)."""
import torch.nn as nn


class DirectionLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        # dropout>0 with num_layers>1 applies dropout BETWEEN LSTM layers
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, (h_n, _c_n) = self.lstm(x)
        last = h_n[-1]  # top-layer final hidden state
        return self.sigmoid(self.fc(last)).squeeze(-1)
