"""Reward that penalizes miscalibrated confidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationReward:
    """Reward for well-calibrated confidence scores.

    High confidence + correct = high reward.
    High confidence + wrong = high penalty.
    Encourages the agent to be confident only when it should be.
    """

    penalty_scale: float = 2.0

    def compute(
        self,
        predicted_direction: int,
        actual_return: float,
        confidence: float,
        flat_threshold: float = 0.005,
    ) -> float:
        """Compute calibration reward.

        Args:
            predicted_direction: 0=down, 1=flat, 2=up
            actual_return: actual percentage return
            confidence: prediction confidence in [0, 1]
            flat_threshold: threshold below which return is considered flat

        Returns:
            Reward value.
        """
        confidence = float(np.clip(confidence, 0.0, 1.0))

        if predicted_direction == 1:
            correct = abs(actual_return) < flat_threshold
        else:
            actual_dir = (
                2
                if actual_return > flat_threshold
                else (0 if actual_return < -flat_threshold else 1)
            )
            correct = predicted_direction == actual_dir

        if correct:
            return confidence
        else:
            return -confidence * self.penalty_scale
