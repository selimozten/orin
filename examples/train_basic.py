"""Basic training example using a random agent.

Demonstrates the core gym loop with orin environments.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import orin  # noqa: F401 -- registers environments


def main() -> None:
    env = gym.make("orin/FinText-Earnings-v0")

    total_reward = 0.0
    n_episodes = 20

    for episode in range(n_episodes):
        obs, info = env.reset(seed=episode)

        # Random agent: pick a direction and confidence
        action = {
            "direction": np.random.randint(0, 3),
            "confidence": np.float32(np.random.random()),
        }

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(
            f"Episode {episode + 1}: "
            f"ticker={info['ticker']}, "
            f"direction={action['direction']}, "
            f"confidence={action['confidence']:.2f}, "
            f"reward={reward:.3f}, "
            f"actual_return={info['actual_return']:.3f}"
        )

    print(f"\nAverage reward over {n_episodes} episodes: {total_reward / n_episodes:.3f}")
    env.close()


if __name__ == "__main__":
    main()
