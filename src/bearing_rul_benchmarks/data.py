from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig


class WindowedRULDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, target_offset: float, target_scale: float) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
        self.target_offset = float(target_offset)
        self.target_scale = float(target_scale)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


@dataclass(slots=True)
class SplitData:
    train_features: np.ndarray
    train_targets: np.ndarray
    val_features: np.ndarray
    val_targets: np.ndarray
    test_features: np.ndarray
    test_targets: np.ndarray
    feature_columns: list[str]
    target_offset: float
    target_scale: float
    test_bearing_ids: np.ndarray
    test_time_idx: np.ndarray


class MinMaxNormalizer:
    def __init__(self) -> None:
        self.minimum: np.ndarray | None = None
        self.maximum: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> None:
        self.minimum = values.min(axis=0)
        self.maximum = values.max(axis=0)

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.minimum is None or self.maximum is None:
            raise RuntimeError("Normalizer must be fitted before transform.")
        scale = np.where(self.maximum - self.minimum == 0.0, 1.0, self.maximum - self.minimum)
        return (values - self.minimum) / scale


def load_split_data(config: DataConfig) -> SplitData:
    frame = pd.read_csv(config.data_path)
    required = {config.id_column, config.time_column, config.target_column, *config.feature_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    numeric_columns = [config.time_column, config.target_column, *config.feature_columns]
    frame = frame.copy()
    frame.loc[:, numeric_columns] = frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=[config.id_column, *numeric_columns]).reset_index(drop=True)

    split_ids = _split_ids(frame[config.id_column].drop_duplicates().tolist(), config.seed, config.train_ratio, config.val_ratio)
    train_ids, val_ids, test_ids = split_ids

    train_frame = _sort_frame(frame[frame[config.id_column].isin(train_ids)], config)
    val_frame = _sort_frame(frame[frame[config.id_column].isin(val_ids)], config)
    test_frame = _sort_frame(frame[frame[config.id_column].isin(test_ids)], config)

    normalizer = MinMaxNormalizer()
    normalizer.fit(train_frame[config.feature_columns].to_numpy(dtype=np.float32))

    train_frame = train_frame.copy()
    val_frame = val_frame.copy()
    test_frame = test_frame.copy()

    train_frame.loc[:, config.feature_columns] = normalizer.transform(train_frame[config.feature_columns].to_numpy(dtype=np.float32))
    val_frame.loc[:, config.feature_columns] = normalizer.transform(val_frame[config.feature_columns].to_numpy(dtype=np.float32))
    test_frame.loc[:, config.feature_columns] = normalizer.transform(test_frame[config.feature_columns].to_numpy(dtype=np.float32))

    train_x, train_y, _, _ = _build_windows(train_frame, config)
    val_x, val_y, _, _ = _build_windows(val_frame, config)
    test_x, test_y, test_ids, test_time = _build_windows(test_frame, config)

    target_offset, target_scale = _fit_target_scaler(train_y)
    train_y = _transform_targets(train_y, target_offset, target_scale)
    val_y = _transform_targets(val_y, target_offset, target_scale)
    test_y = _transform_targets(test_y, target_offset, target_scale)

    return SplitData(
        train_features=train_x,
        train_targets=train_y,
        val_features=val_x,
        val_targets=val_y,
        test_features=test_x,
        test_targets=test_y,
        feature_columns=list(config.feature_columns),
        target_offset=target_offset,
        target_scale=target_scale,
        test_bearing_ids=test_ids,
        test_time_idx=test_time,
    )


def create_dataloaders(split_data: SplitData, batch_size: int, num_workers: int = 0) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        WindowedRULDataset(split_data.train_features, split_data.train_targets, split_data.target_offset, split_data.target_scale),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        WindowedRULDataset(split_data.val_features, split_data.val_targets, split_data.target_offset, split_data.target_scale),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        WindowedRULDataset(split_data.test_features, split_data.test_targets, split_data.target_offset, split_data.target_scale),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


def _fit_target_scaler(targets: np.ndarray) -> tuple[float, float]:
    target_offset = float(np.min(targets))
    target_scale = float(np.max(targets) - target_offset)
    if target_scale == 0.0:
        target_scale = 1.0
    return target_offset, target_scale


def _transform_targets(targets: np.ndarray, target_offset: float, target_scale: float) -> np.ndarray:
    return ((targets - target_offset) / target_scale).astype(np.float32, copy=False)


def _split_ids(ids: list[str], seed: int, train_ratio: float, val_ratio: float) -> tuple[list[str], list[str], list[str]]:
    rng = np.random.default_rng(seed)
    ids_array = np.array(ids)
    rng.shuffle(ids_array)

    train_end = max(1, int(len(ids_array) * train_ratio))
    val_end = train_end + max(1, int(len(ids_array) * val_ratio))

    train_ids = ids_array[:train_end].tolist()
    val_ids = ids_array[train_end:val_end].tolist()
    test_ids = ids_array[val_end:].tolist()

    if not val_ids or not test_ids:
        raise ValueError("Need enough bearing_id groups to create train, validation, and test splits.")

    return train_ids, val_ids, test_ids


def _sort_frame(frame: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    return frame.sort_values([config.id_column, config.time_column]).reset_index(drop=True)


def _build_windows(frame: pd.DataFrame, config: DataConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    windows: list[np.ndarray] = []
    targets: list[float] = []
    ids: list[object] = []
    time_indices: list[object] = []

    grouped = frame.groupby(config.id_column, sort=False)
    for group_id, group in grouped:
        values = group[config.feature_columns].to_numpy(dtype=np.float32)
        rul_values = group[config.target_column].to_numpy(dtype=np.float32)
        time_values = group[config.time_column].to_numpy()
        if len(group) < config.window_size:
            continue
        for end_index in range(config.window_size - 1, len(group), config.stride):
            start_index = end_index - config.window_size + 1
            windows.append(values[start_index : end_index + 1])
            targets.append(float(rul_values[end_index]))
            ids.append(group_id)
            time_indices.append(time_values[end_index])

    if not windows:
        raise ValueError("No windows were created. Check window_size and the dataset length per bearing_id.")

    return (
        np.stack(windows),
        np.array(targets, dtype=np.float32),
        np.array(ids, dtype=object),
        np.array(time_indices),
    )
