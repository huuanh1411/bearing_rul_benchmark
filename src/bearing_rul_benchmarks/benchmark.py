from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd

from .config import ExperimentConfig, FireflyConfig, ModelConfig, TrainingConfig
from .data import SplitData, create_dataloaders, load_split_data
from .firefly import FireflyOptimizer
from .metrics import summarize_metric_dicts
from .train import evaluate_model, fit_model, resolve_device


BASELINE_MODELS = ("cnn", "lstm", "cnn_lstm")


def _print_progress(message: str) -> None:
    print(message, flush=True)


def run_benchmarks(config: ExperimentConfig) -> dict[str, object]:
    split_data = load_split_data(config.data)
    input_channels = len(split_data.feature_columns)
    train_loader, val_loader, test_loader = create_dataloaders(
        split_data,
        batch_size=config.data.batch_size,
        num_workers=config.training.num_workers,
    )

    results: dict[str, object] = {
        "data": {
            "path": str(config.data.data_path),
            "features": split_data.feature_columns,
            "window_size": config.data.window_size,
        },
        "benchmarks": {},
    }

    prediction_frames: list[pd.DataFrame] = []

    for model_name in BASELINE_MODELS:
        run_metrics = []
        run_times = []
        per_seed_entries = []
        _print_progress(f"[benchmark] Starting {model_name} across {len(config.training.seeds)} seed(s)")
        for seed in config.training.seeds:
            model_config = _default_model_config(model_name, input_channels)
            training_result = fit_model(
                model_config,
                config.training,
                train_loader,
                val_loader,
                seed,
                progress_callback=_print_progress,
                report_epoch_progress=True,
                run_label=f"{model_name} seed={seed}",
            )
            test_result = evaluate_model(training_result.model, test_loader, resolve_device(config.training.device))
            run_metrics.append(test_result.metrics)
            run_times.append(training_result.training_time_seconds)
            per_seed_entries.append({"seed": seed, **test_result.metrics})
            prediction_frames.append(
                _build_prediction_frame(model_name, seed, split_data, test_result)
            )
        results["benchmarks"][model_name] = {
            **summarize_metric_dicts(run_metrics),
            "training_time_mean": float(np.mean(run_times)),
            "training_time_std": float(np.std(run_times, ddof=0)),
            "per_seed": per_seed_entries,
        }
        _print_progress(
            f"[benchmark] Finished {model_name} rmse_mean={results['benchmarks'][model_name]['rmse_mean']:.3f}"
        )

    if config.include_firefly:
        firefly_result, firefly_predictions = _run_firefly_benchmark(
            split_data=split_data,
            input_channels=input_channels,
            training_config=config.training,
        )
        results["benchmarks"]["firefly_cnn_lstm"] = firefly_result
        prediction_frames.append(firefly_predictions)

    config.output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if prediction_frames:
        predictions_path = config.output_path.with_name(config.output_path.stem + "_predictions.csv")
        pd.concat(prediction_frames, ignore_index=True).to_csv(predictions_path, index=False)
        _print_progress(f"[benchmark] Wrote per-window test predictions to {predictions_path}")

    return results


def _build_prediction_frame(model_name: str, seed: int, split_data: SplitData, test_result) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": model_name,
            "seed": seed,
            "bearing_id": split_data.test_bearing_ids,
            "time_idx": split_data.test_time_idx,
            "actual_rul": test_result.targets.reshape(-1),
            "predicted_rul": test_result.predictions.reshape(-1),
        }
    )


def _default_model_config(model_name: str, input_channels: int) -> ModelConfig:
    return ModelConfig(
        name=model_name,
        input_channels=input_channels,
        conv_filters=64,
        kernel_size=3,
        lstm_hidden=128,
        lstm_layers=1,
        dropout=0.2,
        learning_rate=1e-3,
        weight_decay=0.0,
    )


def _run_firefly_benchmark(
    split_data: SplitData,
    input_channels: int,
    training_config: TrainingConfig,
) -> tuple[dict[str, object], pd.DataFrame]:
    search_config = FireflyConfig()
    bounds = {
        "learning_rate": (1e-4, 1e-2),
        "conv_filters": (16.0, 128.0),
        "kernel_size": (3.0, 7.0),
        "lstm_hidden": (32.0, 256.0),
        "batch_size": (16.0, 128.0),
        "dropout": (0.1, 0.5),
    }

    search_training = replace(training_config, epochs=search_config.search_epochs, patience=search_config.search_patience, seeds=[training_config.seeds[0]])
    evaluation_count = 0

    def objective(candidate: dict[str, float]) -> float:
        nonlocal evaluation_count
        evaluation_count += 1
        batch_size = int(round(candidate["batch_size"]))
        candidate_train_loader, candidate_val_loader, _ = create_dataloaders(
            split_data,
            batch_size=batch_size,
            num_workers=training_config.num_workers,
        )
        model_config = ModelConfig(
            name="firefly_cnn_lstm",
            input_channels=input_channels,
            conv_filters=_nearest_multiple(candidate["conv_filters"], 8),
            kernel_size=_nearest_odd(candidate["kernel_size"]),
            lstm_hidden=int(round(candidate["lstm_hidden"])),
            lstm_layers=1,
            dropout=float(candidate["dropout"]),
            learning_rate=float(candidate["learning_rate"]),
            weight_decay=0.0,
        )
        candidate_label = f"firefly eval={evaluation_count} batch={batch_size} lr={model_config.learning_rate:.5f}"
        _print_progress(f"[{candidate_label}] Starting candidate search training")
        result = fit_model(
            model_config,
            search_training,
            candidate_train_loader,
            candidate_val_loader,
            search_training.seeds[0],
            progress_callback=_print_progress,
            report_epoch_progress=False,
            run_label=candidate_label,
        )
        candidate_rmse = result.validation_metrics["rmse"]
        _print_progress(f"[{candidate_label}] Completed candidate with val_rmse={candidate_rmse:.3f}")
        return candidate_rmse

    _print_progress("[firefly] Starting hyperparameter search")
    optimizer = FireflyOptimizer(search_config, bounds, objective, progress_callback=_print_progress)
    search_result = optimizer.optimize()
    _print_progress(f"[firefly] Search complete best_val_rmse={search_result.best_score:.3f}")

    final_config = ModelConfig(
        name="firefly_cnn_lstm",
        input_channels=input_channels,
        conv_filters=_nearest_multiple(search_result.best_position["conv_filters"], 8),
        kernel_size=_nearest_odd(search_result.best_position["kernel_size"]),
        lstm_hidden=int(round(search_result.best_position["lstm_hidden"])),
        lstm_layers=1,
        dropout=float(search_result.best_position["dropout"]),
        learning_rate=float(search_result.best_position["learning_rate"]),
        weight_decay=0.0,
    )
    final_batch_size = int(round(search_result.best_position["batch_size"]))
    train_loader, val_loader, test_loader = create_dataloaders(
        split_data,
        batch_size=final_batch_size,
        num_workers=training_config.num_workers,
    )

    run_metrics = []
    run_times = []
    per_seed_entries = []
    prediction_frames: list[pd.DataFrame] = []
    for seed in training_config.seeds[: search_config.final_repeats]:
        result = fit_model(
            final_config,
            training_config,
            train_loader,
            val_loader,
            seed,
            progress_callback=_print_progress,
            report_epoch_progress=True,
            run_label=f"firefly final seed={seed}",
        )
        test_result = evaluate_model(result.model, test_loader, resolve_device(training_config.device))
        run_metrics.append(test_result.metrics)
        run_times.append(result.training_time_seconds)
        per_seed_entries.append({"seed": seed, **test_result.metrics})
        prediction_frames.append(
            _build_prediction_frame("firefly_cnn_lstm", seed, split_data, test_result)
        )

    firefly_result = {
        **summarize_metric_dicts(run_metrics),
        "training_time_mean": float(np.mean(run_times)),
        "training_time_std": float(np.std(run_times, ddof=0)),
        "per_seed": per_seed_entries,
        "best_validation_rmse": search_result.best_score,
        "best_hyperparameters": {
            **search_result.best_position,
            "conv_filters": final_config.conv_filters,
            "kernel_size": final_config.kernel_size,
            "lstm_hidden": final_config.lstm_hidden,
            "batch_size": final_batch_size,
        },
        "search_history": search_result.history,
    }
    firefly_predictions = (
        pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    )
    return firefly_result, firefly_predictions


def _nearest_multiple(value: float, factor: int) -> int:
    return max(factor, int(round(value / factor)) * factor)


def _nearest_odd(value: float) -> int:
    rounded = int(round(value))
    if rounded % 2 == 0:
        rounded += 1
    return min(max(3, rounded), 7)
