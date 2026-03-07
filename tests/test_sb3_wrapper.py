"""Tests for the Stable-Baselines3 wrapper."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import orin  # noqa: F401
from orin.data.sources import write_sample_data
from orin.wrappers.sb3 import make_sb3_env


def setup_module():
    write_sample_data()


def test_sb3_wrapper_spaces():
    env = make_sb3_env("orin/FinText-Earnings-v0", obs_size=128, confidence_bins=5)
    assert env.observation_space.shape == (128,)
    assert env.observation_space.dtype == np.float32
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 15  # 3 * 5
    env.close()


def test_sb3_wrapper_reset():
    env = make_sb3_env("orin/FinText-Earnings-v0", obs_size=128)
    obs, info = env.reset(seed=42)
    assert obs.shape == (128,)
    assert obs.dtype == np.float32
    assert obs.min() >= 0.0
    assert obs.max() <= 1.0
    env.close()


def test_sb3_wrapper_step():
    env = make_sb3_env("orin/FinText-Earnings-v0", obs_size=128)
    obs, info = env.reset(seed=42)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (128,)
    assert isinstance(reward, float)
    assert terminated is True
    env.close()


def test_sb3_action_decode():
    env = make_sb3_env("orin/FinText-Earnings-v0", confidence_bins=5)
    wrapper = env  # already an SB3Wrapper

    # Action 0: direction=0 (down), conf bin 0 -> 0.1
    a = wrapper._decode_action(0)
    assert a["direction"] == 0
    assert 0.0 < float(a["confidence"]) < 0.2

    # Action 14: direction=2 (up), conf bin 4 -> 0.9
    a = wrapper._decode_action(14)
    assert a["direction"] == 2
    assert float(a["confidence"]) > 0.8

    env.close()


def test_sb3_multiple_episodes():
    env = make_sb3_env("orin/FinText-Earnings-v0", obs_size=64)
    for ep in range(10):
        obs, info = env.reset(seed=ep)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert terminated is True
    env.close()


def test_sb3_byte_encoding_normalized():
    env = make_sb3_env("orin/FinText-Earnings-v0", obs_size=256, obs_mode="byte")
    obs, info = env.reset(seed=0)
    # Byte encoding should produce values in [0, 1]
    assert obs.min() >= 0.0
    assert obs.max() <= 1.0
    # Some values should be non-zero (text was encoded)
    assert obs.sum() > 0.0
    env.close()
