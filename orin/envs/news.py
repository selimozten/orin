"""Financial news environment."""

from __future__ import annotations

from typing import Any

from orin.envs.base import FinTextEnv


class FinTextNews(FinTextEnv):
    """Environment for predicting market sentiment from financial news.

    Input: financial news headlines/articles
    Action: predict market sentiment and direction
    Reward: based on actual market reaction
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def _load_data(self) -> list[dict[str, Any]]:
        from orin.data.loaders import load_sample_data

        return load_sample_data("news")
