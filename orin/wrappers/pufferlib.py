"""PufferLib compatibility wrapper for orin environments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PufferLibWrapper(gym.Wrapper):
    """Wrap a FinTextEnv for PufferLib compatibility.

    PufferLib expects flat observation and action spaces. This wrapper:
    - Converts text observations to tokenized integer arrays
    - Flattens the action space to a single Discrete or MultiDiscrete space
    - Supports vectorized environments

    Args:
        env: A FinTextEnv instance.
        max_tokens: Maximum number of tokens in the text observation.
        vocab_size: Size of the token vocabulary (for byte-level, 256).
    """

    def __init__(
        self,
        env: gym.Env,
        max_tokens: int = 512,
        vocab_size: int = 256,
        tokenizer: Any | None = None,
    ) -> None:
        super().__init__(env)
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size
        self._tokenizer = tokenizer

        # Flat observation: tokenized text as integer array
        self.observation_space = spaces.Box(
            low=0,
            high=vocab_size - 1,
            shape=(max_tokens,),
            dtype=np.int32,
        )

        # Flat action: direction (3) * confidence_bins (10) = 30 discrete actions
        self.confidence_bins = 10
        self.action_space = spaces.Discrete(3 * self.confidence_bins)

    def _tokenize(self, text: str) -> np.ndarray:
        """Convert text to token array."""
        if self._tokenizer is not None:
            tokens = self._tokenizer(text)
        else:
            # Default: byte-level encoding
            tokens = list(text.encode("utf-8", errors="replace"))

        arr = np.zeros(self.max_tokens, dtype=np.int32)
        n = min(len(tokens), self.max_tokens)
        arr[:n] = tokens[:n]
        return arr

    def _decode_action(self, action: int) -> dict[str, Any]:
        """Convert flat discrete action to dict action."""
        direction = action // self.confidence_bins
        confidence_bin = action % self.confidence_bins
        confidence = (confidence_bin + 0.5) / self.confidence_bins
        return {
            "direction": direction,
            "confidence": np.float32(confidence),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self._tokenize(obs["text"]), info

    def step(
        self,
        action: int | np.integer,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        dict_action = self._decode_action(int(action))
        obs, reward, terminated, truncated, info = self.env.step(dict_action)
        return self._tokenize(obs["text"]), reward, terminated, truncated, info


def make_pufferlib_env(
    env_id: str = "orin/FinText-Earnings-v0",
    max_tokens: int = 512,
    **kwargs: Any,
) -> PufferLibWrapper:
    """Create a PufferLib-compatible orin environment.

    Args:
        env_id: Gymnasium environment ID.
        max_tokens: Maximum tokens in observation.
        **kwargs: Additional arguments passed to the environment.

    Returns:
        Wrapped environment ready for PufferLib training.
    """
    env = gym.make(env_id, **kwargs)
    return PufferLibWrapper(env, max_tokens=max_tokens)
