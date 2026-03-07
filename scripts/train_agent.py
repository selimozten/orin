"""Train a PPO agent on orin and benchmark against baselines.

Usage:
    python scripts/train_agent.py
    python scripts/train_agent.py --timesteps 50000 --env orin/FinText-News-v0

Saves model to models/ and results to results/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import orin  # noqa: F401
from orin.wrappers.sb3 import make_sb3_env


class RewardLogger(BaseCallback):
    """Log episode rewards during training."""

    def __init__(self):
        super().__init__()
        self.episode_rewards: list[float] = []
        self._current_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])[0]
        done = self.locals.get("dones", [False])[0]
        self._current_reward += reward
        if done:
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0
        return True


def evaluate_policy(env, predict_fn, n_episodes: int = 100) -> dict:
    """Evaluate a policy over n episodes."""
    rewards = []
    correct = 0
    total = 0
    pred_counts = {0: 0, 1: 0, 2: 0}

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep + 50000)
        action, _ = predict_fn(obs, deterministic=True)
        action = int(np.asarray(action).flat[0])
        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(reward)
        actual = info.get("actual_return", 0.0)
        pred_dir = info.get("predicted_direction", 1)
        pred_counts[pred_dir] = pred_counts.get(pred_dir, 0) + 1

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
        "accuracy": correct / total if total else 0.0,
        "predictions": {
            "down": pred_counts.get(0, 0),
            "flat": pred_counts.get(1, 0),
            "up": pred_counts.get(2, 0),
        },
    }


def random_predict(obs, deterministic=False):
    return np.random.randint(0, 15), None


def always_up_predict(obs, deterministic=False):
    return 12, None  # direction=2, mid confidence


def always_down_predict(obs, deterministic=False):
    return 2, None  # direction=0, mid confidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="orin/FinText-Earnings-v0")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--obs-size", type=int, default=256)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env_short = args.env.split("/")[-1].lower()
    model_dir = Path("models")
    results_dir = Path("results")
    model_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # Train
    print(f"Training PPO on {args.env}")
    print(f"  timesteps: {args.timesteps}")
    print(f"  obs_size:  {args.obs_size}")
    print()

    env = make_sb3_env(args.env, obs_size=args.obs_size)
    callback = RewardLogger()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=args.seed,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        ent_coef=0.05,
        gamma=0.0,
        gae_lambda=1.0,
        clip_range=0.2,
        policy_kwargs={"net_arch": [128, 64]},
    )

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callback)
    train_time = time.time() - t0

    model_path = model_dir / f"ppo_{env_short}"
    model.save(str(model_path))
    print(f"Model saved to {model_path}.zip")
    print(f"Training time: {train_time:.1f}s")

    # Training curve
    ep_rewards = callback.episode_rewards
    if len(ep_rewards) >= 20:
        first_20 = np.mean(ep_rewards[:20])
        last_20 = np.mean(ep_rewards[-20:])
        print(f"Training curve: first 20 eps avg={first_20:.3f}, last 20 eps avg={last_20:.3f}")

    # Evaluate
    print(f"\nEvaluating over {args.eval_episodes} episodes...")
    eval_env = make_sb3_env(args.env, obs_size=args.obs_size)

    results = {}

    baselines = {
        "random": random_predict,
        "always_up": always_up_predict,
        "always_down": always_down_predict,
    }

    for name, policy in baselines.items():
        metrics = evaluate_policy(eval_env, policy, args.eval_episodes)
        results[name] = metrics
        print(
            f"  {name:12s}: reward={metrics['mean_reward']:+.3f} "
            f"acc={metrics['accuracy']:.1%} "
            f"preds={metrics['predictions']}"
        )

    # Trained agent
    trained = evaluate_policy(eval_env, model.predict, args.eval_episodes)
    results["ppo_trained"] = trained
    print(
        f"  {'ppo_trained':12s}: reward={trained['mean_reward']:+.3f} "
        f"acc={trained['accuracy']:.1%} "
        f"preds={trained['predictions']}"
    )

    # Summary
    delta = trained["mean_reward"] - results["random"]["mean_reward"]
    print(f"\nPPO vs Random: {delta:+.3f} reward delta")
    if delta > 0:
        print("Agent learned to outperform random baseline.")
    else:
        print("Agent did not beat random (may need more data or timesteps).")

    # Save results
    results["config"] = {
        "env": args.env,
        "timesteps": args.timesteps,
        "obs_size": args.obs_size,
        "eval_episodes": args.eval_episodes,
        "train_time_seconds": round(train_time, 1),
        "training_episodes": len(ep_rewards),
    }
    results_path = results_dir / f"eval_{env_short}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
