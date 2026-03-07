"""PufferLib training example.

Demonstrates how to use orin with PufferLib for vectorized RL training.
Requires: pip install orin[all]
"""

from __future__ import annotations

import numpy as np

import orin  # noqa: F401 -- registers environments
from orin.wrappers.pufferlib import make_pufferlib_env


def main() -> None:
    env = make_pufferlib_env("orin/FinText-Earnings-v0", max_tokens=256)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    total_reward = 0.0
    n_episodes = 20

    for episode in range(n_episodes):
        obs, info = env.reset(seed=episode)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Decode what the flat action means
        direction = action // env.confidence_bins
        confidence = (action % env.confidence_bins + 0.5) / env.confidence_bins
        direction_labels = ["down", "flat", "up"]

        print(
            f"Episode {episode + 1}: "
            f"action={action} -> {direction_labels[direction]} "
            f"(conf={confidence:.2f}), "
            f"reward={reward:.3f}"
        )

    print(f"\nAverage reward: {total_reward / n_episodes:.3f}")

    # PufferLib vectorized usage (requires pufferlib installed):
    try:
        import pufferlib
        import pufferlib.vector

        def env_creator():
            return make_pufferlib_env("orin/FinText-Earnings-v0", max_tokens=256)

        vec_env = pufferlib.vector.make(env_creator, num_envs=4)
        obs, infos = vec_env.reset()
        print(f"\nVectorized env obs shape: {np.array(obs).shape}")
        vec_env.close()
    except ImportError:
        print("\nInstall pufferlib for vectorized training: pip install pufferlib")

    env.close()


if __name__ == "__main__":
    main()
