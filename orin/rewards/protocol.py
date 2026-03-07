"""Protocol for reward functions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RewardFn(Protocol):
    """Protocol that any reward function must satisfy."""

    def compute(
        self,
        predicted_direction: int,
        actual_return: float,
        confidence: float,
    ) -> float: ...
