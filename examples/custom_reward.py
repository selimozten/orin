"""Custom reward function example.

Shows how to create and use a custom reward function with orin environments.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

import orin  # noqa: F401 -- registers environments
from orin.rewards.composite import CompositeReward
from orin.rewards.directional import DirectionalReward


@dataclass
class AsymmetricReward:
    """Custom reward that penalizes losses more than it rewards gains.

    Mimics real-world trading where drawdowns are more costly than gains.
    """

    loss_multiplier: float = 2.0
    flat_threshold: float = 0.005

    def compute(
        self,
        predicted_direction: int,
        actual_return: float,
        confidence: float = 1.0,
    ) -> float:
        base = DirectionalReward(flat_threshold=self.flat_threshold)
        reward = base.compute(predicted_direction, actual_return, confidence)
        if reward < 0:
            reward *= self.loss_multiplier
        return reward


def main() -> None:
    # Use a custom composite reward
    custom_reward = CompositeReward(
        directional_weight=0.5,
        calibration_weight=0.5,
    )

    env = gym.make("orin/FinText-Earnings-v0", reward_fn=custom_reward)

    total_reward = 0.0
    n_episodes = 10

    for episode in range(n_episodes):
        obs, info = env.reset(seed=episode)
        action = {
            "direction": np.random.randint(0, 3),
            "confidence": np.float32(0.9),  # high confidence to test calibration
        }
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Episode {episode + 1}: reward={reward:.3f}")

    print(f"\nAverage reward (custom): {total_reward / n_episodes:.3f}")
    env.close()

    # Use the fully custom AsymmetricReward directly
    print("\n--- Asymmetric Reward ---")
    asym = AsymmetricReward(loss_multiplier=3.0)
    r1 = asym.compute(predicted_direction=2, actual_return=0.05)
    r2 = asym.compute(predicted_direction=2, actual_return=-0.05)
    print(f"Correct prediction reward: {r1:.3f}")
    print(f"Wrong prediction reward:   {r2:.3f}")


if __name__ == "__main__":
    main()
