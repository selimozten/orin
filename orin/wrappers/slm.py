"""Small Language Model (SLM) wrapper for orin environments.

Uses a pretrained sentence-transformer to encode text observations into
dense embeddings, producing far richer features than byte-level encoding.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from orin.wrappers.metadata import MetadataEncoder


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
        cache_size: Maximum number of embeddings to cache.
        normalize: L2-normalize and clip embeddings to [-3, 3].
        include_metadata: Concatenate metadata features to embedding.
    """

    def __init__(
        self,
        env: gym.Env,
        model_name: str = "all-MiniLM-L6-v2",
        confidence_bins: int = 5,
        device: str | None = None,
        cache_size: int = 2048,
        normalize: bool = True,
        include_metadata: bool = False,
    ) -> None:
        super().__init__(env)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

        self.confidence_bins = confidence_bins
        self.normalize = normalize
        self.include_metadata = include_metadata
        self._model = SentenceTransformer(model_name, device=device)
        self._embed_dim = self._model.get_sentence_embedding_dimension()
        self._metadata_encoder = MetadataEncoder() if include_metadata else None

        obs_dim = self._embed_dim
        if include_metadata:
            obs_dim += MetadataEncoder.n_features

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(3 * confidence_bins)

        # Embedding cache
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def cache_stats(self) -> dict[str, int | float]:
        """Return cache performance statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache),
            "hit_rate": hit_rate,
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache and reset statistics."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def _encode(self, text: str) -> np.ndarray:
        """Encode text to a dense embedding vector, using cache when possible."""
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()

        if key in self._cache:
            self._cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        self._cache_misses += 1
        embedding = self._model.encode(
            text, convert_to_numpy=True, show_progress_bar=False,
        )
        embedding = embedding.astype(np.float32)

        if self.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            embedding = np.clip(embedding, -3.0, 3.0)

        self._cache[key] = embedding
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

        return embedding

    def _make_observation(self, obs: dict[str, Any]) -> np.ndarray:
        """Build full observation from env obs dict (embedding + optional metadata)."""
        embedding = self._encode(obs["text"])
        if self.include_metadata and self._metadata_encoder is not None:
            meta_features = self._metadata_encoder.encode(obs.get("metadata", {}))
            return np.concatenate([embedding, meta_features])
        return embedding

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
        return self._make_observation(obs), info

    def step(
        self,
        action: int | np.integer,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        dict_action = self._decode_action(int(action))
        obs, reward, terminated, truncated, info = self.env.step(dict_action)
        return self._make_observation(obs), reward, terminated, truncated, info


def make_slm_env(
    env_id: str = "orin/FinText-Earnings-v0",
    model_name: str = "all-MiniLM-L6-v2",
    confidence_bins: int = 5,
    device: str | None = None,
    normalize: bool = True,
    include_metadata: bool = False,
    **kwargs: Any,
) -> SLMWrapper:
    """Create an SLM-wrapped orin environment.

    Args:
        env_id: Gymnasium environment ID.
        model_name: sentence-transformers model name.
        confidence_bins: Confidence discretization bins.
        device: Compute device.
        normalize: L2-normalize and clip embeddings.
        include_metadata: Concatenate metadata features to embedding.
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
        normalize=normalize,
        include_metadata=include_metadata,
    )
