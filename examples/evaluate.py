"""Evaluate agents on orin environments and report metrics.

Usage:
    python examples/evaluate.py
    python examples/evaluate.py --env orin/FinText-News-v0 --episodes 100
    python examples/evaluate.py --model path/to/model.zip
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import orin  # noqa: F401
from orin.wrappers.sb3 import make_sb3_env


def random_agent(obs, deterministic=False):
    """Uniformly random policy."""
    return np.random.randint(0, 15), None


def majority_up_agent(obs, deterministic=False):
    """Always predicts up with medium confidence."""
    return 12, None  # direction=2 (up), confidence bin=2 -> 0.5


def majority_down_agent(obs, deterministic=False):
    """Always predicts down with medium confidence."""
    return 2, None  # direction=0 (down), confidence bin=2 -> 0.5


def run_evaluation(
    env_id: str,
    policy,
    n_episodes: int = 100,
    obs_size: int = 256,
    label: str = "agent",
) -> dict:
    """Evaluate a policy and return detailed metrics."""
    env = make_sb3_env(env_id, obs_size=obs_size)

    rewards = []
    directions_correct = 0
    directions_total = 0
    confidence_when_correct = []
    confidence_when_wrong = []
    returns_by_prediction = {"up": [], "down": [], "flat": []}
    direction_labels = {0: "down", 1: "flat", 2: "up"}

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        action, _ = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(reward)
        actual = info.get("actual_return", 0.0)
        pred_dir = info.get("predicted_direction", 1)
        conf = info.get("confidence", 0.5)
        pred_label = direction_labels.get(pred_dir, "flat")

        correct = (
            (pred_dir == 2 and actual > 0.005)
            or (pred_dir == 0 and actual < -0.005)
            or (pred_dir == 1 and abs(actual) <= 0.005)
        )

        if correct:
            directions_correct += 1
            confidence_when_correct.append(conf)
        else:
            confidence_when_wrong.append(conf)
        directions_total += 1
        returns_by_prediction[pred_label].append(actual)

    env.close()

    metrics = {
        "label": label,
        "env": env_id,
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "accuracy": directions_correct / directions_total,
        "avg_confidence_correct": (
            float(np.mean(confidence_when_correct)) if confidence_when_correct else 0.0
        ),
        "avg_confidence_wrong": (
            float(np.mean(confidence_when_wrong)) if confidence_when_wrong else 0.0
        ),
        "prediction_distribution": {k: len(v) for k, v in returns_by_prediction.items()},
    }
    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty-print evaluation metrics."""
    print(f"\n{'=' * 50}")
    print(f"  {metrics['label']} on {metrics['env']}")
    print(f"{'=' * 50}")
    print(f"  Episodes:          {metrics['n_episodes']}")
    print(f"  Mean reward:       {metrics['mean_reward']:+.3f} (std {metrics['std_reward']:.3f})")
    print(f"  Reward range:      [{metrics['min_reward']:+.3f}, {metrics['max_reward']:+.3f}]")
    print(f"  Accuracy:          {metrics['accuracy']:.1%}")
    print(f"  Conf (correct):    {metrics['avg_confidence_correct']:.2f}")
    print(f"  Conf (wrong):      {metrics['avg_confidence_wrong']:.2f}")
    dist = metrics["prediction_distribution"]
    up, down, flat = dist.get("up", 0), dist.get("down", 0), dist.get("flat", 0)
    print(f"  Predictions:       up={up} down={down} flat={flat}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate agents on orin")
    parser.add_argument("--env", default="orin/FinText-Earnings-v0")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--obs-size", type=int, default=256)
    parser.add_argument("--model", default=None, help="Path to saved SB3 model (.zip)")
    parser.add_argument("--json", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    all_results = []

    # Baseline agents
    baselines = [
        ("Random", random_agent),
        ("Always Up", majority_up_agent),
        ("Always Down", majority_down_agent),
    ]

    for label, policy in baselines:
        metrics = run_evaluation(args.env, policy, args.episodes, args.obs_size, label)
        print_metrics(metrics)
        all_results.append(metrics)

    # Trained agent (if model provided)
    if args.model:
        try:
            from stable_baselines3 import PPO

            model = PPO.load(args.model)
            metrics = run_evaluation(
                args.env,
                model.predict,
                args.episodes,
                args.obs_size,
                f"PPO ({args.model})",
            )
            print_metrics(metrics)
            all_results.append(metrics)
        except ImportError:
            print("Install stable-baselines3 to evaluate trained models")
        except FileNotFoundError:
            print(f"Model not found: {args.model}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.json}")


if __name__ == "__main__":
    main()
