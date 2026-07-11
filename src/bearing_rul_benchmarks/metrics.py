from __future__ import annotations

import numpy as np


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    targets = targets.reshape(-1)
    predictions = predictions.reshape(-1)
    rmse = float(np.sqrt(np.mean((targets - predictions) ** 2)))
    mae = float(np.mean(np.abs(targets - predictions)))
    denom = float(np.sum((targets - np.mean(targets)) ** 2))
    r2 = 1.0 - float(np.sum((targets - predictions) ** 2)) / denom if denom > 0 else 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2}


def summarize_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    if not metric_dicts:
        return summary
    for key in metric_dicts[0]:
        values = np.array([item[key] for item in metric_dicts], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=0))
    return summary
