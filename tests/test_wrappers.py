"""Tests for PufferLib wrapper."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import orin  # noqa: F401 -- registers environments
from orin.data.sources import write_sample_data
from orin.wrappers.pufferlib import PufferLibWrapper


def setup_module():
    write_sample_data()


def test_pufferlib_wrapper_spaces():
    env = gym.make("orin/FinText-Earnings-v0")
    wrapped = PufferLibWrapper(env, max_tokens=128)
    assert wrapped.observation_space.shape == (128,)
    assert isinstance(wrapped.action_space, gym.spaces.Discrete)
    wrapped.close()


def test_pufferlib_wrapper_reset():
    env = gym.make("orin/FinText-Earnings-v0")
    wrapped = PufferLibWrapper(env, max_tokens=128)
    obs, info = wrapped.reset(seed=42)
    assert obs.shape == (128,)
    assert obs.dtype == np.int32
    wrapped.close()


def test_pufferlib_wrapper_step():
    env = gym.make("orin/FinText-Earnings-v0")
    wrapped = PufferLibWrapper(env, max_tokens=128)
    obs, info = wrapped.reset(seed=42)
    action = wrapped.action_space.sample()
    obs, reward, terminated, truncated, info = wrapped.step(action)
    assert obs.shape == (128,)
    assert isinstance(reward, float)
    assert terminated is True
    wrapped.close()


def test_pufferlib_action_decode():
    env = gym.make("orin/FinText-Earnings-v0")
    wrapped = PufferLibWrapper(env, max_tokens=128)
    # Action 0 -> direction=0 (down), confidence bin 0
    action_dict = wrapped._decode_action(0)
    assert action_dict["direction"] == 0
    assert 0.0 < action_dict["confidence"] < 0.2
    # Action 25 -> direction=2 (up), confidence bin 5
    action_dict = wrapped._decode_action(25)
    assert action_dict["direction"] == 2
    wrapped.close()
