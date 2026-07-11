from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import run_benchmarks
from .config import DataConfig, ExperimentConfig, TrainingConfig
from .plots import generate_all_plots
from .prepare import convert_log_directory


def _print_progress(message: str) -> None:
    print(message, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bearing RUL benchmark runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark", help="Run baseline benchmarking")
    benchmark.add_argument("--data", required=True, type=Path)
    benchmark.add_argument("--id-column", required=True)
    benchmark.add_argument("--time-column", required=True)
    benchmark.add_argument("--target-column", required=True)
    benchmark.add_argument("--feature-columns", nargs="+", required=True)
    benchmark.add_argument("--window-size", type=int, default=64)
    benchmark.add_argument("--stride", type=int, default=1)
    benchmark.add_argument("--batch-size", type=int, default=64)
    benchmark.add_argument("--train-ratio", type=float, default=0.7)
    benchmark.add_argument("--val-ratio", type=float, default=0.15)
    benchmark.add_argument("--epochs", type=int, default=40)
    benchmark.add_argument("--patience", type=int, default=8)
    benchmark.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument("--num-workers", type=int, default=0)
    benchmark.add_argument("--output", required=True, type=Path)
    benchmark.add_argument("--include-firefly", action="store_true")
    benchmark.add_argument("--seed", type=int, default=42)

    prepare_logs = subparsers.add_parser("prepare-logs", help="Convert raw log CSV files into a benchmark-ready dataset")
    prepare_logs.add_argument("--input-dir", required=True, type=Path)
    prepare_logs.add_argument("--output", required=True, type=Path)
    prepare_logs.add_argument("--pattern", default="*.csv")
    prepare_logs.add_argument("--has-header", action="store_true")
    prepare_logs.add_argument("--feature-columns", nargs="+")
    prepare_logs.add_argument("--max-files", type=int)
    prepare_logs.add_argument("--row-step", type=int, default=1)
    prepare_logs.add_argument("--chunk-size", type=int, default=250000)

    plot = subparsers.add_parser("plot", help="Generate charts from a benchmark results JSON file")
    plot.add_argument("--results", required=True, type=Path)
    plot.add_argument("--output-dir", required=True, type=Path)
    plot.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Path to the *_predictions.csv file written by 'benchmark'. "
        "If omitted, the tool looks for '<results-stem>_predictions.csv' next to --results.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "benchmark":
        data_config = DataConfig(
            data_path=args.data,
            id_column=args.id_column,
            time_column=args.time_column,
            target_column=args.target_column,
            feature_columns=args.feature_columns,
            window_size=args.window_size,
            stride=args.stride,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        training_config = TrainingConfig(
            epochs=args.epochs,
            patience=args.patience,
            seeds=args.seeds,
            device=args.device,
            num_workers=args.num_workers,
        )
        experiment_config = ExperimentConfig(
            data=data_config,
            training=training_config,
            output_path=args.output,
            include_firefly=args.include_firefly,
        )
        run_benchmarks(experiment_config)
    elif args.command == "prepare-logs":
        output_path = convert_log_directory(
            input_dir=args.input_dir,
            output_path=args.output,
            pattern=args.pattern,
            has_header=args.has_header,
            feature_columns=args.feature_columns,
            max_files=args.max_files,
            row_step=args.row_step,
            chunk_size=args.chunk_size,
            progress_callback=_print_progress,
        )
        print(f"Wrote prepared dataset to {output_path}")
    elif args.command == "plot":
        written_paths = generate_all_plots(args.results, args.output_dir, predictions_path=args.predictions)
        for path in written_paths:
            print(f"Wrote chart to {path}")


if __name__ == "__main__":
    main()
