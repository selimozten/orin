"""Stable-Baselines3 compatible wrapper for orin environments."""

from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SB3Wrapper(gym.Wrapper):
    """Wrap a FinTextEnv for Stable-Baselines3 compatibility.

    SB3 requires flat observation and action spaces. This wrapper:
    - Converts text observations to fixed-length float32 feature vectors
    - Maps the Dict action space to a single Discrete action

    Two observation modes:
    - "byte": byte-level encoding normalized to [0, 1] (default, no deps)
    - "tfidf": TF-IDF features (requires scikit-learn)

    Args:
        env: A FinTextEnv instance.
        obs_size: Size of the flat observation vector.
        obs_mode: "byte" or "tfidf".
        confidence_bins: Number of confidence levels per direction.
    """

    def __init__(
        self,
        env: gym.Env,
        obs_size: int = 512,
        obs_mode: str = "byte",
        confidence_bins: int = 5,
    ) -> None:
        super().__init__(env)
        self.obs_size = obs_size
        self.obs_mode = obs_mode
        self.confidence_bins = confidence_bins

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32,
        )

        # 3 directions * confidence_bins = total discrete actions
        self.action_space = spaces.Discrete(3 * confidence_bins)

        self._tfidf = None

    def _encode_obs(self, text: str) -> np.ndarray:
        if self.obs_mode == "tfidf":
            return self._encode_tfidf(text)
        return self._encode_bytes(text)

    def _encode_bytes(self, text: str) -> np.ndarray:
        raw = list(text.encode("utf-8", errors="replace"))
        arr = np.zeros(self.obs_size, dtype=np.float32)
        n = min(len(raw), self.obs_size)
        arr[:n] = np.array(raw[:n], dtype=np.float32) / 255.0
        return arr

    def _encode_tfidf(self, text: str) -> np.ndarray:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            raise ImportError("scikit-learn required for tfidf mode: pip install scikit-learn")

        if self._tfidf is None:
            self._tfidf = TfidfVectorizer(max_features=self.obs_size)
            # Fit on available training data
            data = getattr(self.env.unwrapped, "data", [])
            corpus = [r.get("text", "") for r in data] if data else [text]
            self._tfidf.fit(corpus)

        vec = self._tfidf.transform([text]).toarray()[0]
        arr = np.zeros(self.obs_size, dtype=np.float32)
        n = min(len(vec), self.obs_size)
        arr[:n] = vec[:n]
        return arr

    def _decode_action(self, action: int) -> dict[str, Any]:
        direction = action // self.confidence_bins
        conf_bin = action % self.confidence_bins
        confidence = (conf_bin + 0.5) / self.confidence_bins
        return {
            "direction": direction,
            "confidence": np.float32(confidence),
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self._encode_obs(obs["text"]), info

    def step(
        self, action: int | np.integer,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        dict_action = self._decode_action(int(action))
        obs, reward, terminated, truncated, info = self.env.step(dict_action)
        return self._encode_obs(obs["text"]), reward, terminated, truncated, info


def make_sb3_env(
    env_id: str = "orin/FinText-Earnings-v0",
    obs_size: int = 512,
    obs_mode: str = "byte",
    confidence_bins: int = 5,
    **kwargs: Any,
) -> SB3Wrapper:
    """Create an SB3-compatible orin environment.

    Args:
        env_id: Gymnasium environment ID.
        obs_size: Observation vector size.
        obs_mode: "byte" or "tfidf".
        confidence_bins: Confidence discretization bins.
        **kwargs: Passed to the underlying environment.

    Returns:
        Wrapped environment ready for SB3 training.
    """
    env = gym.make(env_id, **kwargs)
    return SB3Wrapper(env, obs_size=obs_size, obs_mode=obs_mode,
                      confidence_bins=confidence_bins)


def make_sb3_vec_env(
    env_id: str = "orin/FinText-Earnings-v0",
    n_envs: int = 4,
    obs_size: int = 512,
    obs_mode: str = "byte",
    confidence_bins: int = 5,
    **kwargs: Any,
) -> Any:
    """Create a vectorized SB3 environment.

    Requires: pip install stable-baselines3

    Args:
        env_id: Gymnasium environment ID.
        n_envs: Number of parallel environments.
        obs_size: Observation vector size.
        obs_mode: "byte" or "tfidf".
        confidence_bins: Confidence discretization bins.
        **kwargs: Passed to the underlying environment.

    Returns:
        SB3 SubprocVecEnv or DummyVecEnv.
    """
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        raise ImportError("stable-baselines3 not installed: pip install stable-baselines3")

    def _make() -> Callable[[], SB3Wrapper]:
        def _init() -> SB3Wrapper:
            return make_sb3_env(env_id, obs_size, obs_mode, confidence_bins, **kwargs)
        return _init

    return DummyVecEnv([_make() for _ in range(n_envs)])
