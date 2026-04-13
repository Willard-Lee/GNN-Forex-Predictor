"""
Baseline LSTM model — same heads, no GAT path.
Used as ablation to isolate graph contribution.
"""

import torch
import torch.nn as nn
from typing import Tuple


class BaselineLSTM(nn.Module):
    """LSTM-only baseline with 3-class direction + return + volatility heads."""

    def __init__(
        self,
        num_features: int,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        shared_dim: int = 128,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        num_direction_classes: int = 3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        self.lstm_ln = nn.LayerNorm(lstm_hidden) if use_layer_norm else nn.Identity()

        self.fc_shared = nn.Sequential(
            nn.Linear(lstm_hidden, shared_dim),
            nn.LayerNorm(shared_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head_direction = nn.Linear(shared_dim, num_direction_classes)
        self.head_return = nn.Linear(shared_dim, 1)
        self.head_volatility = nn.Linear(shared_dim, 1)

    def forward(self, x_seq):
        lstm_out, _ = self.lstm(x_seq)
        h = self.lstm_ln(lstm_out[:, -1, :])
        shared = self.fc_shared(h)

        dir_logits = self.head_direction(shared)
        ret_pred = self.head_return(shared)
        vol_pred = self.head_volatility(shared)
        return dir_logits, ret_pred, vol_pred
