"""Macroeconomic environment."""

from __future__ import annotations

from typing import Any

from orin.envs.base import FinTextEnv


class FinTextMacro(FinTextEnv):
    """Environment for predicting market regime from macroeconomic data.

    Input: economic indicators, Fed speeches, macro reports
    Action: predict market regime (risk-on/risk-off/neutral)
    Reward: based on actual market behavior
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def _load_data(self) -> list[dict[str, Any]]:
        from orin.data.loaders import load_sample_data

        return load_sample_data("macro")
