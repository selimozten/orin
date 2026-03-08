"""Reward based on directional accuracy of predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DirectionalReward:
    """Reward for correct direction prediction.

    +1 for correct direction, -1 for wrong, 0 for flat predictions.
    Optionally scaled by magnitude of actual move.
    """

    scale_by_magnitude: bool = True
    flat_threshold: float = 0.005
    partial_credit: bool = False

    def compute(
        self,
        predicted_direction: int,
        actual_return: float,
        confidence: float = 1.0,
    ) -> float:
        """Compute directional reward.

        Args:
            predicted_direction: 0=down, 1=flat, 2=up
            actual_return: actual percentage return (e.g. 0.03 for +3%)
            confidence: prediction confidence in [0, 1]

        Returns:
            Reward value.
        """
        if predicted_direction == 1:  # flat
            if abs(actual_return) < self.flat_threshold:
                reward = 1.0
            else:
                reward = -abs(actual_return) / self.flat_threshold
        else:
            actual_dir = (
                1 if abs(actual_return) < self.flat_threshold else (2 if actual_return > 0 else 0)
            )
            if predicted_direction == actual_dir:
                reward = 1.0
                if self.scale_by_magnitude:
                    reward *= 1.0 + abs(actual_return)
            else:
                if self.partial_credit and abs(actual_return) < 2 * self.flat_threshold:
                    reward = -0.3
                else:
                    reward = -1.0
                if self.scale_by_magnitude:
                    reward *= 1.0 + abs(actual_return)

        return float(np.clip(reward, -2.0, 2.0))
