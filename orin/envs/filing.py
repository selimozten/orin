"""SEC filing environment."""

from __future__ import annotations

from typing import Any

from orin.envs.base import FinTextEnv


class FinTextFiling(FinTextEnv):
    """Environment for predicting market impact from SEC filings.

    Input: SEC filing excerpts (10-K, 10-Q, 8-K)
    Action: predict stock direction based on filing content
    Reward: based on actual price movement after filing
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def _load_data(self) -> list[dict[str, Any]]:
        from orin.data.loaders import load_sample_data

        return load_sample_data("filing")
