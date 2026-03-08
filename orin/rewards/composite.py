"""Composite reward combining multiple reward signals."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

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


@dataclass
class AdaptiveCompositeReward:
    """Composite reward that adapts weights based on training progress.

    Starts with high directional weight. Once accuracy exceeds threshold
    and confidence variance drops below threshold, shifts weight toward
    calibration to encourage better-calibrated confidence.
    """

    directional: DirectionalReward = field(default_factory=DirectionalReward)
    calibration: CalibrationReward = field(default_factory=CalibrationReward)
    initial_dir_weight: float = 0.8
    final_dir_weight: float = 0.4
    accuracy_threshold: float = 0.70
    conf_var_threshold: float = 0.05
    window_size: int = 100
    # internal state
    _history: list = field(default_factory=list, init=False, repr=False)
    _confidences: list = field(default_factory=list, init=False, repr=False)

    def compute(
        self,
        predicted_direction: int,
        actual_return: float,
        confidence: float,
    ) -> float:
        """Compute adaptive composite reward.

        Args:
            predicted_direction: 0=down, 1=flat, 2=up
            actual_return: actual percentage return
            confidence: prediction confidence in [0, 1]

        Returns:
            Adaptively weighted sum of component rewards.
        """
        d = self.directional.compute(predicted_direction, actual_return, confidence)
        c = self.calibration.compute(
            predicted_direction,
            actual_return,
            confidence,
            flat_threshold=self.directional.flat_threshold,
        )

        # Track accuracy and confidence
        flat_threshold = self.directional.flat_threshold
        if predicted_direction == 1:
            correct = abs(actual_return) < flat_threshold
        else:
            actual_dir = (
                2
                if actual_return > flat_threshold
                else (0 if actual_return < -flat_threshold else 1)
            )
            correct = predicted_direction == actual_dir

        self._history.append(float(correct))
        self._confidences.append(confidence)

        # Trim to window
        if len(self._history) > self.window_size:
            self._history = self._history[-self.window_size:]
            self._confidences = self._confidences[-self.window_size:]

        # Compute adaptive weights
        dir_weight = self.initial_dir_weight
        if len(self._history) >= self.window_size:
            acc = sum(self._history) / len(self._history)
            conf_var = float(np.var(self._confidences))
            if acc > self.accuracy_threshold and conf_var < self.conf_var_threshold:
                dir_weight = self.final_dir_weight

        cal_weight = 1.0 - dir_weight
        return dir_weight * d + cal_weight * c
