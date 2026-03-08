"""Tests for SLMWrapper with embedding cache."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

st = pytest.importorskip("sentence_transformers")

# isort: split

from gymnasium import spaces  # noqa: E402

from orin.wrappers.slm import SLMWrapper  # noqa: E402

# ── Helpers ────────────────────────────────────────────────────────────


EMBED_DIM = 8


def _fake_encode(text: str, *, convert_to_numpy: bool = True, show_progress_bar: bool = False):
    """Deterministic fake encoder: hash text into a fixed-dim vector."""
    rng = np.random.default_rng(hash(text) % 2**31)
    return rng.standard_normal(EMBED_DIM).astype(np.float32)


class _StubEnv:
    """Minimal gymnasium-compatible env for testing SLMWrapper."""

    observation_space = spaces.Dict(
        {"text": spaces.Text(min_length=1, max_length=512)}
    )
    action_space = spaces.Dict(
        {
            "direction": spaces.Discrete(3),
            "confidence": spaces.Box(0.0, 1.0, shape=()),
        }
    )
    metadata: dict[str, Any] = {}
    render_mode = None
    spec = None
    np_random = np.random.default_rng(0)

    def __init__(self, texts: list[str] | None = None) -> None:
        self._texts = texts or ["earnings beat expectations", "revenue missed"]
        self._idx = 0

    def reset(self, *, seed=None, options=None):
        self._idx = 0
        return {"text": self._texts[self._idx]}, {}

    def step(self, action):
        self._idx = min(self._idx + 1, len(self._texts) - 1)
        obs = {"text": self._texts[self._idx]}
        terminated = self._idx >= len(self._texts) - 1
        return obs, 1.0, terminated, False, {}


@pytest.fixture()
def wrapper():
    """Create an SLMWrapper with a mocked sentence-transformer model."""
    env = _StubEnv(texts=["hello world", "second text", "hello world"])
    w = SLMWrapper.__new__(SLMWrapper)

    # Manually initialise to avoid loading a real model
    w.env = env
    w.confidence_bins = 5
    w.normalize = False
    w.include_metadata = False
    w._embed_dim = EMBED_DIM
    w._model = MagicMock()
    w._model.encode = MagicMock(side_effect=_fake_encode)
    w._model.get_sentence_embedding_dimension = MagicMock(return_value=EMBED_DIM)
    w._metadata_encoder = None

    w.observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(EMBED_DIM,), dtype=np.float32,
    )
    w.action_space = spaces.Discrete(3 * 5)

    # Cache initialisation
    from collections import OrderedDict

    w._cache = OrderedDict()
    w._cache_size = 2048
    w._cache_hits = 0
    w._cache_misses = 0

    return w


# ── Space tests ────────────────────────────────────────────────────────


def test_observation_space(wrapper: SLMWrapper):
    assert wrapper.observation_space.shape == (EMBED_DIM,)


def test_action_space(wrapper: SLMWrapper):
    assert wrapper.action_space.n == 15  # 3 directions * 5 bins


# ── Reset / step ───────────────────────────────────────────────────────


def test_reset_returns_embedding(wrapper: SLMWrapper):
    obs, info = wrapper.reset()
    assert obs.shape == (EMBED_DIM,)
    assert obs.dtype == np.float32


def test_step_returns_embedding(wrapper: SLMWrapper):
    wrapper.reset()
    obs, reward, terminated, truncated, info = wrapper.step(0)
    assert obs.shape == (EMBED_DIM,)


# ── Action decode ──────────────────────────────────────────────────────


def test_decode_action_direction(wrapper: SLMWrapper):
    a = wrapper._decode_action(0)
    assert a["direction"] == 0
    a = wrapper._decode_action(5)
    assert a["direction"] == 1
    a = wrapper._decode_action(10)
    assert a["direction"] == 2


def test_decode_action_confidence(wrapper: SLMWrapper):
    a = wrapper._decode_action(0)
    assert abs(a["confidence"] - 0.1) < 1e-6
    a = wrapper._decode_action(4)
    assert abs(a["confidence"] - 0.9) < 1e-6


# ── Embed dim ──────────────────────────────────────────────────────────


def test_embed_dim(wrapper: SLMWrapper):
    assert wrapper.embed_dim == EMBED_DIM


# ── Cache behaviour ───────────────────────────────────────────────────


def test_cache_hit_after_same_text(wrapper: SLMWrapper):
    wrapper.reset()  # encodes "hello world" -> miss
    assert wrapper._cache_misses == 1
    assert wrapper._cache_hits == 0

    # Step to "second text" -> miss
    wrapper.step(0)
    assert wrapper._cache_misses == 2

    # Step to "hello world" again -> hit
    wrapper.step(0)
    assert wrapper._cache_hits == 1
    assert wrapper._cache_misses == 2


def test_clear_cache(wrapper: SLMWrapper):
    wrapper.reset()
    assert len(wrapper._cache) == 1

    wrapper.clear_cache()
    assert len(wrapper._cache) == 0
    assert wrapper._cache_hits == 0
    assert wrapper._cache_misses == 0


def test_cache_stats(wrapper: SLMWrapper):
    wrapper.reset()
    wrapper.step(0)
    wrapper.step(0)  # triggers a hit

    stats = wrapper.cache_stats
    assert stats["hits"] == 1
    assert stats["misses"] == 2
    assert stats["size"] == 2
    assert abs(stats["hit_rate"] - 1 / 3) < 1e-6


def test_cache_eviction():
    """When cache is full, oldest entry is evicted."""
    env = _StubEnv(texts=["a", "b", "c", "d"])
    w = SLMWrapper.__new__(SLMWrapper)
    w.env = env
    w.confidence_bins = 5
    w.normalize = False
    w.include_metadata = False
    w._embed_dim = EMBED_DIM
    w._model = MagicMock()
    w._model.encode = MagicMock(side_effect=_fake_encode)
    w._metadata_encoder = None

    w.observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(EMBED_DIM,), dtype=np.float32,
    )
    w.action_space = spaces.Discrete(15)

    from collections import OrderedDict

    w._cache = OrderedDict()
    w._cache_size = 2  # tiny cache
    w._cache_hits = 0
    w._cache_misses = 0

    w._encode("a")
    w._encode("b")
    assert len(w._cache) == 2

    w._encode("c")  # should evict "a"
    assert len(w._cache) == 2
    assert w._cache_misses == 3

    # "a" was evicted, encoding it again should be a miss
    w._encode("a")
    assert w._cache_misses == 4
    assert w._cache_hits == 0


def test_cached_embedding_equals_original(wrapper: SLMWrapper):
    """Cache should return the same array as the original encode."""
    emb1 = wrapper._encode("test sentence")
    emb2 = wrapper._encode("test sentence")
    np.testing.assert_array_equal(emb1, emb2)
    assert wrapper._cache_hits == 1
