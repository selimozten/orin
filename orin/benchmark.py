"""Benchmarking suite for orin environments."""
from __future__ import annotations

import time

import gymnasium as gym
import numpy as np

import orin  # noqa: F401
from orin.data.generator import generate_earnings, generate_filings, generate_macro, generate_news


def run_benchmark(
    n_episodes: int = 500,
    seed: int = 42,
    difficulties: list[str] | None = None,
) -> dict:
    """Run benchmarks across environment types and difficulties.

    Returns dict with results per env_type per difficulty.
    """
    if difficulties is None:
        difficulties = ["easy", "medium"]

    generators = {
        "earnings": (generate_earnings, "orin/FinText-Earnings-v0"),
        "news": (generate_news, "orin/FinText-News-v0"),
        "filing": (generate_filings, "orin/FinText-Filing-v0"),
        "macro": (generate_macro, "orin/FinText-Macro-v0"),
    }

    results = {}
    rng = np.random.RandomState(seed)

    for env_type, (gen_fn, env_id) in generators.items():
        results[env_type] = {}
        for diff in difficulties:
            data = gen_fn(n=max(n_episodes, 200), seed=seed, difficulty=diff)
            env = gym.make(env_id, data=data)

            rewards = []
            correct = 0
            t0 = time.time()

            for ep in range(n_episodes):
                obs, info = env.reset(seed=ep)
                direction = rng.randint(0, 3)
                confidence = rng.uniform(0.0, 1.0)
                action = {"direction": direction, "confidence": np.float32(confidence)}
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)

                actual = info.get("actual_return", 0.0)
                if (
                    (direction == 2 and actual > 0.005)
                    or (direction == 0 and actual < -0.005)
                    or (direction == 1 and abs(actual) <= 0.005)
                ):
                    correct += 1

            elapsed = time.time() - t0
            env.close()

            results[env_type][diff] = {
                "n_episodes": n_episodes,
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "random_accuracy": correct / n_episodes,
                "episodes_per_sec": n_episodes / elapsed,
                "total_time_sec": round(elapsed, 3),
            }

    return results


def format_report(results: dict) -> str:
    """Format benchmark results as Markdown table."""
    lines = ["# Orin Benchmark Report", ""]
    lines.append("| Env Type | Difficulty | Mean Reward | Accuracy | Eps/sec |")
    lines.append("|----------|-----------|-------------|----------|---------|")

    for env_type, diffs in results.items():
        for diff, metrics in diffs.items():
            lines.append(
                f"| {env_type:10s} | {diff:9s} | "
                f"{metrics['mean_reward']:+.3f}       | "
                f"{metrics['random_accuracy']:.1%}     | "
                f"{metrics['episodes_per_sec']:.0f}     |"
            )

    return "\n".join(lines)
