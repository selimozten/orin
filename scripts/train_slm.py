"""Train a PPO agent using SLM embeddings on scaled synthetic data.

Usage:
    pip install sentence-transformers stable-baselines3
    python scripts/train_slm.py
    python scripts/train_slm.py --n-records 500 --timesteps 50000

This is the "real" training pipeline:
1. Generate a large synthetic dataset
2. Encode observations with a pretrained sentence-transformer
3. Train PPO on dense embeddings
4. Evaluate against baselines
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

import orin  # noqa: F401
from orin.callbacks import OrinCallback
from orin.data.generator import generate_earnings, generate_news
from orin.eval.metrics import calibration_curve, confusion_matrix, direction_metrics
from orin.wrappers.slm import make_slm_env


def _actual_to_label(actual: float) -> int:
    """Convert actual return to direction label: 0=down, 1=flat, 2=up."""
    if actual > 0.005:
        return 2
    elif actual < -0.005:
        return 0
    return 1


def evaluate_policy(env, predict_fn, n_episodes: int = 200) -> dict:
    rewards = []
    correct = 0
    total = 0
    pred_counts = {0: 0, 1: 0, 2: 0}
    confidences_correct = []
    confidences_wrong = []
    y_true: list[int] = []
    y_pred: list[int] = []
    confidences: list[float] = []
    corrects: list[bool] = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep + 50000)
        action, _ = predict_fn(obs, deterministic=True)
        action = int(np.asarray(action).flat[0])
        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(reward)
        actual = info.get("actual_return", 0.0)
        pred_dir = info.get("predicted_direction", 1)
        conf = info.get("confidence", 0.5)
        pred_counts[pred_dir] = pred_counts.get(pred_dir, 0) + 1

        true_dir = _actual_to_label(actual)
        y_true.append(true_dir)
        y_pred.append(pred_dir)
        confidences.append(conf)

        is_correct = pred_dir == true_dir
        corrects.append(is_correct)
        if is_correct:
            correct += 1
            confidences_correct.append(conf)
        else:
            confidences_wrong.append(conf)
        total += 1

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "accuracy": correct / total if total else 0.0,
        "avg_conf_correct": float(np.mean(confidences_correct)) if confidences_correct else 0.0,
        "avg_conf_wrong": float(np.mean(confidences_wrong)) if confidences_wrong else 0.0,
        "predictions": {
            "down": pred_counts.get(0, 0),
            "flat": pred_counts.get(1, 0),
            "up": pred_counts.get(2, 0),
        },
        "_y_true": y_true,
        "_y_pred": y_pred,
        "_confidences": confidences,
        "_corrects": corrects,
    }


def random_predict(obs, deterministic=False):
    return np.random.randint(0, 15), None


def always_up_predict(obs, deterministic=False):
    return 12, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-records", type=int, default=300, help="Number of training records to generate"
    )
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-type", default="earnings", choices=["earnings", "news"])
    args = parser.parse_args()

    results_dir = Path("results")
    model_dir = Path("models")
    results_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    # 1. Generate data
    print(f"Generating {args.n_records} {args.env_type} records...")
    if args.env_type == "earnings":
        train_data = generate_earnings(args.n_records, seed=args.seed)
        eval_data = generate_earnings(args.n_records, seed=args.seed + 100)
        env_id = "orin/FinText-Earnings-v0"
    else:
        train_data = generate_news(args.n_records, seed=args.seed)
        eval_data = generate_news(args.n_records, seed=args.seed + 100)
        env_id = "orin/FinText-News-v0"

    up_count = sum(1 for r in train_data if r["outcome"]["direction"] == "up")
    down_count = sum(1 for r in train_data if r["outcome"]["direction"] == "down")
    flat_count = sum(1 for r in train_data if r["outcome"]["direction"] == "flat")
    print(f"  Train split: up={up_count} down={down_count} flat={flat_count}")
    print(f"  Eval split:  {len(eval_data)} records (different seed)")

    # 2. Create environments
    print(f"\nLoading SLM: {args.model_name}...")
    train_env = make_slm_env(env_id, model_name=args.model_name, data=train_data)
    eval_env = make_slm_env(env_id, model_name=args.model_name, data=eval_data)
    print(f"  Embedding dim: {train_env.embed_dim}")
    print(f"  Action space:  {train_env.action_space}")

    # 3. Train
    print(f"\nTraining PPO for {args.timesteps} timesteps...")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        seed=args.seed,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        ent_coef=0.03,
        gamma=0.0,
        gae_lambda=1.0,
        clip_range=0.2,
        policy_kwargs={"net_arch": [256, 128]},
    )

    callback = OrinCallback(log_interval=1000, verbose=1)
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callback)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f}s")

    model_path = model_dir / f"slm_ppo_{args.env_type}"
    model.save(str(model_path))
    print(f"Model saved to {model_path}.zip")

    # 4. Evaluate on held-out data
    print(f"\nEvaluating on {args.eval_episodes} held-out episodes...")
    all_results = {}

    for name, policy in [("random", random_predict), ("always_up", always_up_predict)]:
        metrics = evaluate_policy(eval_env, policy, args.eval_episodes)
        all_results[name] = metrics
        print(
            f"  {name:12s}: reward={metrics['mean_reward']:+.3f} "
            f"acc={metrics['accuracy']:.1%} "
            f"preds={metrics['predictions']}"
        )

    trained = evaluate_policy(eval_env, model.predict, args.eval_episodes)
    all_results["slm_ppo"] = trained
    print(
        f"  {'slm_ppo':12s}: reward={trained['mean_reward']:+.3f} "
        f"acc={trained['accuracy']:.1%} "
        f"preds={trained['predictions']}"
    )
    print(
        f"  Confidence: correct={trained['avg_conf_correct']:.2f} "
        f"wrong={trained['avg_conf_wrong']:.2f}"
    )

    # Confusion matrix
    y_true = trained["_y_true"]
    y_pred = trained["_y_pred"]
    cm = confusion_matrix(y_true, y_pred)
    labels = ["down", "flat", "  up"]
    print("\n  Confusion Matrix (rows=true, cols=predicted):")
    print(f"           {'down':>6s} {'flat':>6s} {'up':>6s}")
    for i, lbl in enumerate(labels):
        print(f"    {lbl:>4s}  {cm[i, 0]:6d} {cm[i, 1]:6d} {cm[i, 2]:6d}")

    # Per-direction metrics
    dm = direction_metrics(y_true, y_pred)
    print("\n  Per-direction metrics:")
    print(f"    {'':8s} {'Prec':>6s} {'Recall':>6s} {'F1':>6s}")
    for name in ("down", "flat", "up"):
        d = dm[name]
        print(f"    {name:8s} {d['precision']:6.3f} {d['recall']:6.3f} {d['f1']:6.3f}")

    # Calibration
    cal = calibration_curve(trained["_confidences"], trained["_corrects"])
    print(f"\n  Calibration (ECE={cal['ece']:.4f}):")
    print(f"    {'Bin':>6s} {'Acc':>6s} {'Count':>6s}")
    for b, a, c in zip(cal["bins"], cal["accuracy"], cal["counts"]):
        if c > 0:
            print(f"    {b:6.2f} {a:6.3f} {c:6d}")

    # 5. Summary
    delta = trained["mean_reward"] - all_results["random"]["mean_reward"]
    print(f"\nSLM PPO vs Random: {delta:+.3f} reward delta")
    acc_delta = trained["accuracy"] - all_results["random"]["accuracy"]
    print(f"SLM PPO vs Random: {acc_delta:+.1%} accuracy delta")

    if trained["accuracy"] > all_results["always_up"]["accuracy"]:
        print("Agent outperforms always-up baseline -- learned directional signal.")
    elif trained["accuracy"] > all_results["random"]["accuracy"]:
        print("Agent beats random but not always-up -- partial learning.")
    else:
        print("Agent did not beat baselines -- needs more data or timesteps.")

    # Save results (strip internal keys)
    saveable = {}
    for key, val in all_results.items():
        if isinstance(val, dict):
            saveable[key] = {k: v for k, v in val.items() if not k.startswith("_")}
        else:
            saveable[key] = val
    saveable["config"] = {
        "env_type": args.env_type,
        "n_records": args.n_records,
        "timesteps": args.timesteps,
        "eval_episodes": args.eval_episodes,
        "model_name": args.model_name,
        "embed_dim": train_env.embed_dim,
        "train_time_seconds": round(train_time, 1),
    }
    results_path = results_dir / f"slm_{args.env_type}.json"
    with open(results_path, "w") as f:
        json.dump(saveable, f, indent=2)
    print(f"Results saved to {results_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
