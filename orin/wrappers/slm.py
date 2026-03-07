"""Small Language Model (SLM) wrapper for orin environments.

Uses a pretrained sentence-transformer to encode text observations into
dense embeddings, producing far richer features than byte-level encoding.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SLMWrapper(gym.Wrapper):
    """Wrap a FinTextEnv using a small language model for text encoding.

    Encodes text observations using a pretrained sentence-transformer model.
    The resulting dense embeddings capture semantic meaning, enabling the RL
    policy to learn much richer text-outcome associations.

    Default model: all-MiniLM-L6-v2 (22M params, 384-dim embeddings, fast).

    Args:
        env: A FinTextEnv instance.
        model_name: sentence-transformers model name.
        confidence_bins: Number of confidence levels per direction.
        device: "cpu", "cuda", or "mps".
    """

    def __init__(
        self,
        env: gym.Env,
        model_name: str = "all-MiniLM-L6-v2",
        confidence_bins: int = 5,
        device: str | None = None,
    ) -> None:
        super().__init__(env)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

        self.confidence_bins = confidence_bins
        self._model = SentenceTransformer(model_name, device=device)
        self._embed_dim = self._model.get_sentence_embedding_dimension()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._embed_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(3 * confidence_bins)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def _encode(self, text: str) -> np.ndarray:
        """Encode text to a dense embedding vector."""
        embedding = self._model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return embedding.astype(np.float32)

    def _decode_action(self, action: int) -> dict[str, Any]:
        direction = action // self.confidence_bins
        conf_bin = action % self.confidence_bins
        confidence = (conf_bin + 0.5) / self.confidence_bins
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
        return self._encode(obs["text"]), info

    def step(
        self,
        action: int | np.integer,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        dict_action = self._decode_action(int(action))
        obs, reward, terminated, truncated, info = self.env.step(dict_action)
        return self._encode(obs["text"]), reward, terminated, truncated, info


def make_slm_env(
    env_id: str = "orin/FinText-Earnings-v0",
    model_name: str = "all-MiniLM-L6-v2",
    confidence_bins: int = 5,
    device: str | None = None,
    **kwargs: Any,
) -> SLMWrapper:
    """Create an SLM-wrapped orin environment.

    Args:
        env_id: Gymnasium environment ID.
        model_name: sentence-transformers model name.
        confidence_bins: Confidence discretization bins.
        device: Compute device.
        **kwargs: Passed to the underlying environment.

    Returns:
        Wrapped environment with dense text embeddings.
    """
    env = gym.make(env_id, **kwargs)
    return SLMWrapper(
        env,
        model_name=model_name,
        confidence_bins=confidence_bins,
        device=device,
    )
