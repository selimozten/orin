"""Tests for orin gym environments."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import orin  # noqa: F401 -- registers environments


@pytest.fixture(autouse=True)
def _generate_sample_data():
    """Ensure sample data exists before tests run."""
    from orin.data.sources import write_sample_data

    write_sample_data()


ENV_IDS = [
    "orin/FinText-Earnings-v0",
    "orin/FinText-News-v0",
    "orin/FinText-Filing-v0",
    "orin/FinText-Macro-v0",
]


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_make_env(env_id: str):
    env = gym.make(env_id)
    assert env is not None
    env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_reset_returns_obs(env_id: str):
    env = gym.make(env_id)
    obs, info = env.reset(seed=42)
    assert "text" in obs
    assert "metadata" in obs
    assert len(obs["text"]) > 0
    assert "ticker" in info
    env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_step_cycle(env_id: str):
    env = gym.make(env_id)
    obs, info = env.reset(seed=42)

    action = {
        "direction": 2,  # up
        "confidence": np.float32(0.8),
    }
    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(reward, float)
    assert terminated is True
    assert truncated is False
    assert "actual_return" in info
    env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_multiple_episodes(env_id: str):
    env = gym.make(env_id)

    for episode in range(5):
        obs, info = env.reset(seed=episode)
        action = {
            "direction": episode % 3,
            "confidence": np.float32(0.5),
        }
        obs, reward, terminated, truncated, info = env.step(action)
        assert terminated is True

    env.close()


def test_custom_data():
    data = [
        {
            "text": "Test earnings call transcript.",
            "ticker": "TEST",
            "date": "2024-01-01",
            "source": "test",
            "outcome": {"direction": "up", "magnitude": 0.05, "timeframe": "1d"},
        },
    ]
    env = gym.make("orin/FinText-Earnings-v0", data=data)
    obs, info = env.reset(seed=0)
    assert obs["text"] == "Test earnings call transcript."
    assert obs["metadata"]["ticker"] == "TEST"
    env.close()


def test_render(capsys):
    env = gym.make("orin/FinText-Earnings-v0", render_mode="human")
    obs, info = env.reset(seed=0)
    action = {"direction": 2, "confidence": np.float32(0.8)}
    env.step(action)
    captured = capsys.readouterr()
    assert len(captured.out) > 0
    env.close()
