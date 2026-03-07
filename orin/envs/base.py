"""Base financial text environment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from orin.rewards.composite import CompositeReward
from orin.rewards.protocol import RewardFn


class FinTextEnv(gym.Env):
    """Base gymnasium environment for financial text reasoning.

    The agent reads a financial document and predicts the market direction,
    with rewards based on actual market outcomes.

    Observation space:
        - text: the financial document (string, encoded as bytes)
        - metadata: dict with ticker, date, source

    Action space:
        - direction: 0=down, 1=flat, 2=up
        - confidence: float in [0, 1]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: list[dict[str, Any]] | None = None,
        reward_fn: RewardFn | None = None,
        max_text_length: int = 4096,
        render_mode: str | None = None,
        shuffle: bool = True,
    ) -> None:
        super().__init__()

        self.data = data or []
        self.reward_fn = reward_fn or CompositeReward()
        self.max_text_length = max_text_length
        self.render_mode = render_mode
        self.shuffle = shuffle

        self.observation_space = spaces.Dict(
            {
                "text": spaces.Text(
                    min_length=0,
                    max_length=max_text_length,
                ),
                "metadata": spaces.Dict(
                    {
                        "ticker": spaces.Text(min_length=0, max_length=10),
                        "date": spaces.Text(min_length=0, max_length=10),
                        "source": spaces.Text(min_length=0, max_length=32),
                    }
                ),
            }
        )

        self.action_space = spaces.Dict(
            {
                "direction": spaces.Discrete(3),  # 0=down, 1=flat, 2=up
                "confidence": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            }
        )

        self._current_idx: int = 0
        self._current_record: dict[str, Any] | None = None
        self._indices: list[int] = []
        self._step_count: int = 0
        self._episode_count: int = 0
        self._cumulative_reward: float = 0.0

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"records={len(self.data)}, "
            f"episodes={self._episode_count})"
        )

    def _load_data(self) -> list[dict[str, Any]]:
        """Override in subclasses to load environment-specific data."""
        return self.data

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)

        if not self.data:
            self.data = self._load_data()
            if not self.data:
                raise ValueError(
                    "No data available. Pass data to the constructor or "
                    "ensure sample data is generated."
                )

        if not self._indices or self._current_idx >= len(self._indices):
            self._indices = list(range(len(self.data)))
            if self.shuffle:
                self.np_random.shuffle(self._indices)
            self._current_idx = 0

        idx = self._indices[self._current_idx]
        self._current_record = self.data[idx]
        self._step_count = 0

        obs = self._make_obs(self._current_record)
        info = self._make_info(self._current_record)
        return obs, info

    def step(
        self,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._current_record is None:
            raise RuntimeError("Call reset() before step().")

        direction = int(action["direction"])
        confidence = float(action.get("confidence", 0.5))

        outcome = self._current_record.get("outcome", {})
        actual_return = self._get_actual_return(outcome)

        reward = self.reward_fn.compute(direction, actual_return, confidence)

        self._step_count += 1
        self._current_idx += 1
        self._episode_count += 1
        self._cumulative_reward += reward
        terminated = True  # one prediction per document
        truncated = False

        info = self._make_info(self._current_record)
        info["actual_return"] = actual_return
        info["predicted_direction"] = direction
        info["confidence"] = confidence
        info["episode_count"] = self._episode_count
        info["cumulative_reward"] = self._cumulative_reward

        if self.render_mode == "human":
            self.render()

        # Prepare next observation (for vectorized envs that auto-reset)
        obs = self._make_obs(self._current_record)

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if self._current_record is None:
            return
        record = self._current_record
        text = record.get("text", "")
        ticker = record.get("ticker", "?")
        date = record.get("date", "?")
        outcome = record.get("outcome", {})
        print(f"[{ticker} | {date}]")
        print(text[:500])
        if outcome:
            print(
                f"Outcome: {outcome.get('direction', '?')} "
                f"({outcome.get('magnitude', '?')}) "
                f"over {outcome.get('timeframe', '?')}"
            )
        print()

    def _make_obs(self, record: dict[str, Any]) -> dict[str, Any]:
        text = record.get("text", "")[: self.max_text_length]
        return {
            "text": text,
            "metadata": {
                "ticker": record.get("ticker", ""),
                "date": record.get("date", ""),
                "source": record.get("source", ""),
            },
        }

    def _make_info(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "ticker": record.get("ticker", ""),
            "date": record.get("date", ""),
            "source": record.get("source", ""),
        }

    def _get_actual_return(self, outcome: dict[str, Any]) -> float:
        if "magnitude" in outcome:
            magnitude = float(outcome["magnitude"])
            direction = outcome.get("direction", "flat")
            if direction == "down":
                return -magnitude
            elif direction == "up":
                return magnitude
            else:
                return 0.0
        return 0.0
