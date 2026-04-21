from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class DriftResult:
    status: str
    z_score: float | None
    recent_mean: float | None


class DriftMonitor:
    def __init__(self, reference_mean: float, reference_std: float, window_size: int = 200) -> None:
        self.reference_mean = reference_mean
        self.reference_std = max(reference_std, 1e-6)
        self.window_size = window_size
        self._scores: deque[float] = deque(maxlen=window_size)

    def add_score(self, score: float) -> None:
        self._scores.append(float(score))

    def evaluate(self) -> DriftResult:
        if len(self._scores) < min(30, self.window_size):
            return DriftResult(status="insufficient_data", z_score=None, recent_mean=None)

        recent_mean = float(np.mean(self._scores))
        z_score = abs(recent_mean - self.reference_mean) / self.reference_std
        status = "detected" if z_score > 2.0 else "ok"
        return DriftResult(status=status, z_score=float(z_score), recent_mean=recent_mean)

    @property
    def total_observations(self) -> int:
        return len(self._scores)
