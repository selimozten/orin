"""Error handling and edge case tests."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import orin  # noqa: F401


def test_empty_data():
    """Environment with empty data should raise on reset."""
    from orin.envs.base import FinTextEnv

    env = FinTextEnv(data=[])
    with pytest.raises(ValueError, match="No data"):
        env.reset()
    env.close()


def test_step_before_reset():
    """Stepping before reset should raise."""
    data = [
        {
            "text": "test",
            "ticker": "X",
            "date": "2024-01-01",
            "source": "test",
            "outcome": {"direction": "up", "magnitude": 0.01, "timeframe": "1d"},
        }
    ]
    env = gym.make("orin/FinText-Earnings-v0", data=data)
    with pytest.raises((RuntimeError, gym.error.ResetNeeded)):
        env.step({"direction": 1, "confidence": np.float32(0.5)})
    env.close()


def test_out_of_range_confidence_clipped():
    """Confidence outside [0,1] should be clipped, not crash."""
    data = [
        {
            "text": "test",
            "ticker": "X",
            "date": "2024-01-01",
            "source": "test",
            "outcome": {"direction": "up", "magnitude": 0.01, "timeframe": "1d"},
        }
    ]
    env = gym.make("orin/FinText-Earnings-v0", data=data)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(
        {"direction": 2, "confidence": np.float32(5.0)}
    )
    assert 0.0 <= info["confidence"] <= 1.0
    env.close()


def test_malformed_jsonl(tmp_path):
    """Malformed JSONL lines should raise."""
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text("not valid json\n")

    from orin.data.loaders import load_jsonl

    with pytest.raises(Exception):
        load_jsonl(bad_file)


def test_single_record_many_episodes():
    """Single record replayed over many episodes."""
    data = [
        {
            "text": "test record",
            "ticker": "X",
            "date": "2024-01-01",
            "source": "test",
            "outcome": {"direction": "up", "magnitude": 0.05, "timeframe": "1d"},
        }
    ]
    env = gym.make("orin/FinText-Earnings-v0", data=data)

    for ep in range(10):
        obs, info = env.reset(seed=ep)
        action = {"direction": 2, "confidence": np.float32(0.5)}
        obs, reward, terminated, truncated, info = env.step(action)
        assert terminated

    env.close()
