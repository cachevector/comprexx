"""Tiny model fixtures for testing."""

import torch.nn as nn


def tiny_cnn() -> nn.Module:
    """3-layer ConvNet for testing."""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10),
    )


def tiny_transformer() -> nn.Module:
    """2-layer Transformer encoder for testing."""

    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Linear(64, 64)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=64, nhead=2, dim_feedforward=128, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.head = nn.Linear(64, 10)

        def forward(self, x):
            x = self.embedding(x)
            x = self.encoder(x)
            x = x.mean(dim=1)
            return self.head(x)

    return TinyTransformer()


def tiny_rnn() -> nn.Module:
    """Simple LSTM model for testing."""

    class TinyRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
            self.head = nn.Linear(64, 10)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    return TinyRNN()
