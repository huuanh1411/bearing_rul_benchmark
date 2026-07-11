from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_LABELS = {
    "rmse_mean": "RMSE",
    "mae_mean": "MAE",
    "r2_mean": "R\u00b2",
    "training_time_mean": "Training time (s)",
}
METRIC_STD_KEYS = {
    "rmse_mean": "rmse_std",
    "mae_mean": "mae_std",
    "r2_mean": "r2_std",
    "training_time_mean": "training_time_std",
}


def load_results(results_path: Path) -> dict[str, object]:
    return json.loads(Path(results_path).read_text(encoding="utf-8"))


def _model_names(benchmarks: dict[str, object]) -> list[str]:
    return list(benchmarks.keys())


def plot_benchmark_dashboard(results: dict[str, object], output_path: Path) -> Path:
    """Write a single figure that nests several benchmark charts together.

    Layout (2x2 grid, one axes embedded per metric):
      - RMSE bar chart (lower is better)
      - MAE bar chart (lower is better)
      - R2 bar chart (higher is better)
      - Training time bar chart

    If a firefly search history is present, its convergence curve is drawn as
    an inset axes nested inside the RMSE panel, so the figure contains charts
    embedded within charts ("bieu do long ghep").
    """
    benchmarks = results.get("benchmarks", {})
    if not benchmarks:
        raise ValueError("results has no 'benchmarks' section to plot")

    model_names = _model_names(benchmarks)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(model_names), 1)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Bearing RUL Benchmark Comparison", fontsize=15, fontweight="bold")

    metric_keys = ["rmse_mean", "mae_mean", "r2_mean", "training_time_mean"]
    for ax, metric_key in zip(axes.flat, metric_keys):
        std_key = METRIC_STD_KEYS[metric_key]
        values = [benchmarks[name].get(metric_key, np.nan) for name in model_names]
        errors = [benchmarks[name].get(std_key, 0.0) for name in model_names]
        bars = ax.bar(model_names, values, yerr=errors, capsize=4, color=colors)
        ax.set_title(METRIC_LABELS[metric_key])
        ax.set_ylabel(METRIC_LABELS[metric_key])
        ax.tick_params(axis="x", rotation=20)
        for bar, value in zip(bars, values):
            if value == value:  # skip NaN
                ax.annotate(
                    f"{value:,.3g}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )

    firefly_entry = None
    for name, entry in benchmarks.items():
        if isinstance(entry, dict) and entry.get("search_history"):
            firefly_entry = entry
            break

    if firefly_entry is not None:
        rmse_ax = axes.flat[0]
        inset_ax = rmse_ax.inset_axes([0.42, 0.42, 0.55, 0.5])
        history = firefly_entry["search_history"]
        inset_ax.plot(range(1, len(history) + 1), history, marker="o", markersize=3, color="tab:red")
        inset_ax.set_title("Firefly search convergence", fontsize=8)
        inset_ax.set_xlabel("Evaluation", fontsize=7)
        inset_ax.set_ylabel("Val RMSE", fontsize=7)
        inset_ax.tick_params(labelsize=6)
        inset_ax.patch.set_alpha(0.9)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_firefly_convergence(results: dict[str, object], output_path: Path) -> Path | None:
    """Write a standalone convergence chart for the firefly search history."""
    benchmarks = results.get("benchmarks", {})
    firefly_entry = None
    for entry in benchmarks.values():
        if isinstance(entry, dict) and entry.get("search_history"):
            firefly_entry = entry
            break
    if firefly_entry is None:
        return None

    history = firefly_entry["search_history"]
    best_so_far = np.minimum.accumulate(history)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, len(history) + 1), history, marker="o", label="Candidate validation RMSE")
    ax.plot(range(1, len(best_so_far) + 1), best_so_far, linestyle="--", label="Best so far")
    ax.set_title("Firefly Hyperparameter Search Convergence")
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Validation RMSE")
    ax.legend()
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_seed_comparison(results: dict[str, object], output_path: Path) -> Path | None:
    """Grouped bar chart comparing test RMSE per seed, across all models.

    Requires each benchmark entry to have a "per_seed" list of
    {"seed": int, "rmse": float, ...} records (added alongside the
    aggregated mean/std metrics).
    """
    benchmarks = results.get("benchmarks", {})
    model_entries = {
        name: entry["per_seed"]
        for name, entry in benchmarks.items()
        if isinstance(entry, dict) and entry.get("per_seed")
    }
    if not model_entries:
        return None

    model_names = list(model_entries.keys())
    all_seeds = sorted({record["seed"] for records in model_entries.values() for record in records})
    if not all_seeds:
        return None

    x_positions = np.arange(len(model_names))
    bar_width = 0.8 / len(all_seeds)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(all_seeds)))

    fig, ax = plt.subplots(figsize=(max(6, len(model_names) * 2.2), 5))
    for seed_index, seed in enumerate(all_seeds):
        seed_values = []
        for name in model_names:
            match = next((record["rmse"] for record in model_entries[name] if record["seed"] == seed), np.nan)
            seed_values.append(match)
        offsets = x_positions + (seed_index - (len(all_seeds) - 1) / 2) * bar_width
        bars = ax.bar(offsets, seed_values, width=bar_width, label=f"seed {seed}", color=colors[seed_index])
        for bar, value in zip(bars, seed_values):
            if value == value:  # skip NaN
                ax.annotate(
                    f"{value:,.3g}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_names, rotation=15)
    ax.set_ylabel("Test RMSE")
    ax.set_title("Test RMSE by Seed, per Model")
    ax.legend(title="Seed")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_predictions_vs_actual(
    predictions_path: Path,
    output_path: Path,
    model_name: str | None = None,
    seed: int | None = None,
    max_bearings: int = 12,
) -> Path | None:
    """Grid of per-test-bearing charts: actual vs predicted RUL over time_idx.

    Reads the "<output>_predictions.csv" file written by the benchmark
    command (one row per test window, with columns model/seed/bearing_id/
    time_idx/actual_rul/predicted_rul).
    """
    predictions_path = Path(predictions_path)
    if not predictions_path.exists():
        return None

    frame = pd.read_csv(predictions_path)
    if frame.empty:
        return None

    if model_name is None:
        model_name = frame["model"].iloc[0]
    subset = frame[frame["model"] == model_name]
    if subset.empty:
        return None

    if seed is None:
        seed = int(subset["seed"].iloc[0])
    subset = subset[subset["seed"] == seed]
    if subset.empty:
        return None

    bearing_ids = list(subset["bearing_id"].drop_duplicates())[:max_bearings]
    n_bearings = len(bearing_ids)
    n_cols = min(3, n_bearings)
    n_rows = int(np.ceil(n_bearings / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.2 * n_rows), squeeze=False)
    for index, bearing_id in enumerate(bearing_ids):
        ax = axes.flat[index]
        bearing_frame = subset[subset["bearing_id"] == bearing_id].sort_values("time_idx")
        ax.plot(bearing_frame["time_idx"], bearing_frame["actual_rul"], label="Actual RUL", color="tab:blue")
        ax.plot(
            bearing_frame["time_idx"],
            bearing_frame["predicted_rul"],
            label="Predicted RUL",
            color="tab:orange",
            linestyle="--",
        )
        ax.set_title(str(bearing_id), fontsize=9)
        ax.set_xlabel("time_idx")
        ax.set_ylabel("RUL")
        ax.legend(fontsize=7)

    for extra_index in range(n_bearings, n_rows * n_cols):
        axes.flat[extra_index].axis("off")

    fig.suptitle(f"Predicted vs Actual RUL — {model_name} (seed={seed})", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def generate_all_plots(
    results_path: Path,
    output_dir: Path,
    predictions_path: Path | None = None,
) -> list[Path]:
    results = load_results(results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    written.append(plot_benchmark_dashboard(results, output_dir / "benchmark_dashboard.png"))

    convergence_path = plot_firefly_convergence(results, output_dir / "firefly_convergence.png")
    if convergence_path is not None:
        written.append(convergence_path)

    seed_path = plot_seed_comparison(results, output_dir / "seed_comparison.png")
    if seed_path is not None:
        written.append(seed_path)

    if predictions_path is None:
        guessed_path = Path(results_path).with_name(Path(results_path).stem + "_predictions.csv")
        if guessed_path.exists():
            predictions_path = guessed_path

    if predictions_path is not None and Path(predictions_path).exists():
        for model_name in results.get("benchmarks", {}):
            pred_path = plot_predictions_vs_actual(
                predictions_path,
                output_dir / f"predictions_{model_name}.png",
                model_name=model_name,
            )
            if pred_path is not None:
                written.append(pred_path)

    return written
