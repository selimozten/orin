"""Earnings call transcript environment."""

from __future__ import annotations

from typing import Any

from orin.envs.base import FinTextEnv


class FinTextEarnings(FinTextEnv):
    """Environment for predicting stock direction from earnings call transcripts.

    Input: earnings call transcript text
    Action: predict stock price direction after earnings
    Reward: based on actual post-earnings price movement
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def _load_data(self) -> list[dict[str, Any]]:
        from orin.data.loaders import load_sample_data

        return load_sample_data("earnings")
