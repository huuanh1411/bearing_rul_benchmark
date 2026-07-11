from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


class CNNRegressor(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        padding = max(0, config.kernel_size // 2)
        self.network = nn.Sequential(
            nn.Conv1d(config.input_channels, config.conv_filters, kernel_size=config.kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(config.conv_filters, config.conv_filters, kernel_size=config.kernel_size, padding=padding),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(config.dropout),
            nn.Linear(config.conv_filters, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.network(inputs.transpose(1, 2))
        return self.head(encoded)


class LSTMRegressor(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dropout = config.dropout if config.lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=config.input_channels,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.lstm_hidden, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(inputs)
        return self.head(outputs[:, -1, :])


class CNNLSTMRegressor(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        padding = max(0, config.kernel_size // 2)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(config.input_channels, config.conv_filters, kernel_size=config.kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(config.conv_filters, config.conv_filters, kernel_size=config.kernel_size, padding=padding),
            nn.ReLU(),
        )
        dropout = config.dropout if config.lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=config.conv_filters,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.lstm_hidden, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(inputs.transpose(1, 2)).transpose(1, 2)
        outputs, _ = self.lstm(features)
        return self.head(outputs[:, -1, :])


def build_model(config: ModelConfig) -> nn.Module:
    if config.name == "cnn":
        return CNNRegressor(config)
    if config.name == "lstm":
        return LSTMRegressor(config)
    if config.name in {"cnn_lstm", "firefly_cnn_lstm"}:
        return CNNLSTMRegressor(config)
    raise ValueError(f"Unsupported model name: {config.name}")
