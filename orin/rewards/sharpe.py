"""Sharpe-ratio based reward function."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SharpeReward:
    """Reward based on risk-adjusted returns using rolling Sharpe ratio.

    Tracks a rolling window of returns and computes reward as
    return / rolling_std, encouraging consistent performance.
    """

    window_size: int = 50
    min_window: int = 5
    _returns: deque = field(default_factory=lambda: deque(maxlen=50), init=False, repr=False)

    def __post_init__(self):
        self._returns = deque(maxlen=self.window_size)

    def compute(
        self,
        predicted_direction: int,
        actual_return: float,
        confidence: float,
    ) -> float:
        """Compute Sharpe-ratio reward.

        Args:
            predicted_direction: 0=down, 1=flat, 2=up
            actual_return: actual percentage return
            confidence: prediction confidence in [0, 1]

        Returns:
            Risk-adjusted reward value.
        """
        # Compute signed return based on prediction
        if predicted_direction == 2:  # up
            pnl = actual_return * confidence
        elif predicted_direction == 0:  # down
            pnl = -actual_return * confidence
        else:  # flat
            pnl = -abs(actual_return) * confidence * 0.5

        self._returns.append(pnl)

        if len(self._returns) < self.min_window:
            return float(pnl)

        returns = np.array(self._returns)
        std = np.std(returns)
        if std < 1e-8:
            return float(np.mean(returns))

        return float(np.mean(returns) / std)

    def reset(self) -> None:
        """Clear the rolling window."""
        self._returns.clear()
