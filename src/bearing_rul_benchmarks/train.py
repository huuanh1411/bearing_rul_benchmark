from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainingConfig
from .metrics import regression_metrics
from .models import build_model


@dataclass(slots=True)
class TrainingResult:
    model: nn.Module
    history: list[dict[str, float]]
    validation_metrics: dict[str, float]
    training_time_seconds: float


@dataclass(slots=True)
class EvaluationResult:
    metrics: dict[str, float]
    predictions: np.ndarray
    targets: np.ndarray


ProgressCallback = Callable[[str], None]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def fit_model(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    seed: int,
    progress_callback: ProgressCallback | None = None,
    report_epoch_progress: bool = True,
    run_label: str | None = None,
) -> TrainingResult:
    set_seed(seed)
    device = resolve_device(training_config.device)
    model = build_model(model_config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config.learning_rate, weight_decay=model_config.weight_decay)
    resolved_label = run_label or f"{model_config.name} seed={seed}"

    best_state: dict[str, torch.Tensor] | None = None
    best_rmse = float("inf")
    patience_left = training_config.patience
    history: list[dict[str, float]] = []
    start = time.perf_counter()

    if progress_callback is not None:
        progress_callback(
            f"[{resolved_label}] Starting on {device.type} for up to {training_config.epochs} epoch(s)"
        )

    for epoch in range(1, training_config.epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        val_result = evaluate_model(model, val_loader, device)
        val_rmse = val_result.metrics["rmse"]
        history.append({"epoch": float(epoch), "train_loss": train_loss, **val_result.metrics})
        improved = val_rmse < best_rmse

        if improved:
            best_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            patience_left = training_config.patience
        else:
            patience_left -= 1

        if progress_callback is not None and report_epoch_progress:
            status = "improved" if improved else f"patience_left={patience_left}"
            progress_callback(
                f"[{resolved_label}] Epoch {epoch}/{training_config.epochs} "
                f"train_loss={train_loss:.6f} val_rmse={val_rmse:.3f} "
                f"best_rmse={best_rmse:.3f} {status}"
            )

        if not improved:
            if patience_left <= 0:
                if progress_callback is not None:
                    progress_callback(f"[{resolved_label}] Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training finished without producing a best model state.")

    model.load_state_dict(best_state)
    final_val_result = evaluate_model(model, val_loader, device)
    elapsed = time.perf_counter() - start
    if progress_callback is not None:
        progress_callback(
            f"[{resolved_label}] Finished in {elapsed:.2f}s final_val_rmse={final_val_result.metrics['rmse']:.3f}"
        )
    return TrainingResult(model=model, history=history, validation_metrics=final_val_result.metrics, training_time_seconds=elapsed)


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device) -> EvaluationResult:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    dataset = data_loader.dataset
    target_offset = float(getattr(dataset, "target_offset", 0.0))
    target_scale = float(getattr(dataset, "target_scale", 1.0))

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            predictions.append(outputs.cpu().numpy())
            targets.append(labels.cpu().numpy())

    prediction_array = np.concatenate(predictions, axis=0) * target_scale + target_offset
    target_array = np.concatenate(targets, axis=0) * target_scale + target_offset
    return EvaluationResult(metrics=regression_metrics(target_array, prediction_array), predictions=prediction_array, targets=target_array)


def _train_epoch(model: nn.Module, data_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    model.train()
    losses: list[float] = []
    for features, labels in data_loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0
