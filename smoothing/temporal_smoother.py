"""수 프레임 예측 누적 후 평균(Smoothing)으로 안정화."""
from __future__ import annotations

from collections import deque

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / (np.sum(exp) + 1e-8)


class TemporalSmoother:
    """
    track_id별로 최근 N프레임의 표정 로짓·나이 예측을 저장하고,
    표정은 softmax 확률의 평균, 나이는 스칼라 평균을 반환한다.
    """

    def __init__(self, window_size: int = 8, num_expressions: int = 7) -> None:
        self.window_size = window_size
        self.num_expressions = num_expressions
        self._history: dict[int, deque[dict[str, np.ndarray | float]]] = {}

    def reset_track(self, track_id: int) -> None:
        self._history.pop(track_id, None)

    def prune(self, active_track_ids: set[int]) -> None:
        stale = [tid for tid in self._history if tid not in active_track_ids]
        for tid in stale:
            del self._history[tid]

    def update(
        self,
        track_id: int,
        expression_logits: np.ndarray,
        age: float,
    ) -> tuple[np.ndarray, float, int]:
        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=self.window_size)

        self._history[track_id].append(
            {
                "expression_logits": np.asarray(expression_logits, dtype=np.float32),
                "age": float(age),
            }
        )
        return self.get_smoothed(track_id)

    def get_smoothed(self, track_id: int) -> tuple[np.ndarray, float, int]:
        history = self._history[track_id]
        probs = np.stack([_softmax(item["expression_logits"]) for item in history], axis=0)
        mean_probs = np.mean(probs, axis=0)
        mean_age = float(np.mean([item["age"] for item in history]))
        return mean_probs, mean_age, len(history)
