from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Deque

from config import K_WINDOW, THRESHOLD


scores_buffer: Deque[float] = collections.deque(maxlen=K_WINDOW)


def decide(new_score: float, threshold: float = THRESHOLD) -> bool:
    scores_buffer.append(float(new_score))
    if len(scores_buffer) < K_WINDOW:
        return False
    mean_score = sum(scores_buffer) / K_WINDOW
    return mean_score >= threshold


def reset_decision_buffer() -> None:
    scores_buffer.clear()


@dataclass
class DecisionEngine:
    threshold: float = THRESHOLD
    k_window: int = K_WINDOW
    scores: Deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self.scores = collections.deque(maxlen=self.k_window)

    def update(self, score: float) -> tuple[bool, float]:
        self.scores.append(float(score))
        mean_score = sum(self.scores) / len(self.scores)
        if len(self.scores) < self.k_window:
            return False, mean_score
        return mean_score >= self.threshold, mean_score

    def reset(self) -> None:
        self.scores.clear()

    def configure(self, threshold: float, k_window: int) -> None:
        self.threshold = float(threshold)
        if k_window != self.k_window:
            self.k_window = int(k_window)
            self.scores = collections.deque(maxlen=self.k_window)
