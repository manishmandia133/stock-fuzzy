"""Binary direction LSTM (1 layer, 32 units, dropout 0.3 before the classifier)."""
from torch import nn

import config as cfg


class DirectionLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=cfg.HIDDEN_SIZE,
                 num_layers=cfg.NUM_LAYERS, dropout=cfg.DROPOUT):
        super().__init__()
        # nn.LSTM dropout is ignored unless num_layers > 1.
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=lstm_dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _out, (h_n, _c_n) = self.lstm(x)
        last = self.dropout(h_n[-1])  # final hidden state -> dropout -> classifier
        return self.sigmoid(self.fc(last)).squeeze(-1)
