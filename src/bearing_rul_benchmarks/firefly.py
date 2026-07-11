from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import FireflyConfig


@dataclass(slots=True)
class FireflySearchResult:
    best_position: dict[str, float]
    best_score: float
    history: list[float]


class FireflyOptimizer:
    def __init__(
        self,
        config: FireflyConfig,
        bounds: dict[str, tuple[float, float]],
        objective: Callable[[dict[str, float]], float],
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.bounds = bounds
        self.objective = objective
        self.keys = list(bounds)
        self.rng = np.random.default_rng()
        self.progress_callback = progress_callback

    def optimize(self) -> FireflySearchResult:
        population = [self._sample_position() for _ in range(self.config.population_size)]
        scores = []
        for index, position in enumerate(population, start=1):
            score = self.objective(position)
            scores.append(score)
            if self.progress_callback is not None:
                self.progress_callback(
                    f"[firefly] Initial candidate {index}/{self.config.population_size} "
                    f"score={score:.3f} best={float(np.min(scores)):.3f}"
                )
        history = [float(np.min(scores))]

        for iteration in range(1, self.config.iterations + 1):
            update_count = 0
            for i in range(len(population)):
                for j in range(len(population)):
                    if scores[j] < scores[i]:
                        population[i] = self._move_towards(population[i], population[j])
                        scores[i] = self.objective(population[i])
                        update_count += 1
            history.append(float(np.min(scores)))
            if self.progress_callback is not None:
                self.progress_callback(
                    f"[firefly] Iteration {iteration}/{self.config.iterations} "
                    f"best_val_rmse={history[-1]:.3f} updates={update_count}"
                )

        best_index = int(np.argmin(scores))
        return FireflySearchResult(best_position=population[best_index], best_score=float(scores[best_index]), history=history)

    def _sample_position(self) -> dict[str, float]:
        return {
            key: float(self.rng.uniform(low, high))
            for key, (low, high) in self.bounds.items()
        }

    def _move_towards(self, current: dict[str, float], other: dict[str, float]) -> dict[str, float]:
        current_vec = np.array([current[key] for key in self.keys], dtype=np.float64)
        other_vec = np.array([other[key] for key in self.keys], dtype=np.float64)
        distance = np.linalg.norm(current_vec - other_vec)
        beta = self.config.beta0 * np.exp(-self.config.gamma * (distance ** 2))
        random_step = self.config.alpha * self.rng.normal(0.0, 1.0, size=len(self.keys))
        moved = current_vec + beta * (other_vec - current_vec) + random_step

        clamped: dict[str, float] = {}
        for index, key in enumerate(self.keys):
            low, high = self.bounds[key]
            clamped[key] = float(np.clip(moved[index], low, high))
        return clamped
