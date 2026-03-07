"""Train a PPO agent on orin with Stable-Baselines3.

Usage:
    pip install stable-baselines3
    python examples/train_sb3.py

Trains a PPO agent on FinText-Earnings-v0 and evaluates against
a random baseline.
"""

from __future__ import annotations

import argparse

import numpy as np

import orin  # noqa: F401
from orin.wrappers.sb3 import make_sb3_env


def evaluate(env, policy, n_episodes: int = 50, deterministic: bool = True) -> dict:
    """Run evaluation episodes and return metrics."""
    rewards = []
    correct = 0
    total = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep + 10000)
        action, _ = policy(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(reward)
        actual = info.get("actual_return", 0.0)
        pred_dir = info.get("predicted_direction", 1)
        if pred_dir == 2 and actual > 0.005:
            correct += 1
        elif pred_dir == 0 and actual < -0.005:
            correct += 1
        elif pred_dir == 1 and abs(actual) <= 0.005:
            correct += 1
        total += 1

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "accuracy": correct / total if total > 0 else 0.0,
        "n_episodes": n_episodes,
    }


def random_policy(obs, deterministic=False):
    """Random baseline policy."""
    return np.random.randint(0, 15), None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on orin")
    parser.add_argument("--env", default="orin/FinText-Earnings-v0")
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--obs-size", type=int, default=256)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--save-path", default=None)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("Install stable-baselines3: pip install stable-baselines3")
        return

    # Create training environment
    env = make_sb3_env(args.env, obs_size=args.obs_size)

    print(f"Training PPO on {args.env} for {args.timesteps} timesteps...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=64,
        batch_size=32,
        n_epochs=5,
        learning_rate=3e-4,
        ent_coef=0.05,
        gamma=0.0,  # single-step episodes, no discounting needed
    )

    model.learn(total_timesteps=args.timesteps)

    if args.save_path:
        model.save(args.save_path)
        print(f"Model saved to {args.save_path}")

    # Evaluate trained agent
    eval_env = make_sb3_env(args.env, obs_size=args.obs_size)
    print("\n--- Trained Agent ---")
    trained_metrics = evaluate(eval_env, model.predict, n_episodes=args.eval_episodes)
    m = trained_metrics
    print(f"Mean reward: {m['mean_reward']:.3f} +/- {m['std_reward']:.3f}")
    print(f"Accuracy:    {m['accuracy']:.1%}")

    # Random baseline
    print("\n--- Random Baseline ---")
    random_metrics = evaluate(eval_env, random_policy, n_episodes=args.eval_episodes)
    m = random_metrics
    print(f"Mean reward: {m['mean_reward']:.3f} +/- {m['std_reward']:.3f}")
    print(f"Accuracy:    {random_metrics['accuracy']:.1%}")

    delta = trained_metrics["mean_reward"] - random_metrics["mean_reward"]
    print(f"\nDelta vs random: {delta:+.3f}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
