from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class DataConfig:
    data_path: Path
    id_column: str
    time_column: str
    target_column: str
    feature_columns: list[str]
    window_size: int = 64
    stride: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    batch_size: int = 64
    seed: int = 42


@dataclass(slots=True)
class ModelConfig:
    name: str
    input_channels: int
    conv_filters: int = 64
    kernel_size: int = 3
    lstm_hidden: int = 128
    lstm_layers: int = 1
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 0.0


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 40
    patience: int = 8
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    device: str = "auto"
    num_workers: int = 0


@dataclass(slots=True)
class FireflyConfig:
    population_size: int = 10
    iterations: int = 10
    alpha: float = 0.25
    beta0: float = 1.0
    gamma: float = 1.0
    search_epochs: int = 12
    search_patience: int = 4
    final_repeats: int = 3


@dataclass(slots=True)
class ExperimentConfig:
    data: DataConfig
    training: TrainingConfig
    output_path: Path
    include_firefly: bool = False
