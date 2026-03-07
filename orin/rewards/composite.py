"""Composite reward combining multiple reward signals."""

from __future__ import annotations

from dataclasses import dataclass, field

from orin.rewards.calibration import CalibrationReward
from orin.rewards.directional import DirectionalReward


@dataclass
class CompositeReward:
    """Weighted combination of directional and calibration rewards.

    Default weights: 0.7 directional + 0.3 calibration.
    """

    directional: DirectionalReward = field(default_factory=DirectionalReward)
    calibration: CalibrationReward = field(default_factory=CalibrationReward)
    directional_weight: float = 0.7
    calibration_weight: float = 0.3

    def compute(
        self,
        predicted_direction: int,
        actual_return: float,
        confidence: float,
    ) -> float:
        """Compute composite reward.

        Args:
            predicted_direction: 0=down, 1=flat, 2=up
            actual_return: actual percentage return
            confidence: prediction confidence in [0, 1]

        Returns:
            Weighted sum of component rewards.
        """
        d = self.directional.compute(predicted_direction, actual_return, confidence)
        c = self.calibration.compute(
            predicted_direction,
            actual_return,
            confidence,
            flat_threshold=self.directional.flat_threshold,
        )
        return self.directional_weight * d + self.calibration_weight * c
