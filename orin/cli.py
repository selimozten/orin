"""Command-line interface for orin."""
from __future__ import annotations

import argparse
import json
import sys


def cmd_train(args):
    """Train an agent."""
    from orin.config import TrainConfig, load_config

    if args.config:
        config = load_config(args.config)
    else:
        config = TrainConfig(
            env_type=args.env_type,
            n_records=args.n_records,
            timesteps=args.timesteps,
            difficulty=args.difficulty,
            seed=args.seed,
        )

    # Map env_type to env_id
    env_map = {
        "earnings": "orin/FinText-Earnings-v0",
        "news": "orin/FinText-News-v0",
        "filing": "orin/FinText-Filing-v0",
        "macro": "orin/FinText-Macro-v0",
    }
    if not config.env_id or config.env_id == "orin/FinText-Earnings-v0":
        config.env_id = env_map.get(config.env_type, config.env_id)

    print(f"Training {config.env_type} agent...")
    print(f"  Records: {config.n_records}, Timesteps: {config.timesteps}")
    print(f"  Difficulty: {config.difficulty}")

    try:
        from stable_baselines3 import PPO  # noqa: F401
    except ImportError:
        print("Error: stable-baselines3 required. Run: pip install orin[rl]")
        sys.exit(1)

    from pathlib import Path

    import numpy as np
    from stable_baselines3 import PPO

    import orin  # noqa: F401
    from orin.data.generator import generate_earnings, generate_news
    from orin.wrappers.slm import make_slm_env

    gen_fn = generate_earnings if config.env_type == "earnings" else generate_news
    train_data = gen_fn(config.n_records, seed=config.seed, difficulty=config.difficulty)
    eval_data = gen_fn(config.n_records, seed=config.seed + 100, difficulty=config.difficulty)

    train_env = make_slm_env(config.env_id, model_name=config.model_name, data=train_data)
    eval_env = make_slm_env(config.env_id, model_name=config.model_name, data=eval_data)

    model = PPO(
        config.policy,
        train_env,
        verbose=0,
        seed=config.seed,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        learning_rate=config.learning_rate,
        ent_coef=config.ent_coef,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        policy_kwargs={"net_arch": config.net_arch},
    )

    model.learn(total_timesteps=config.timesteps)

    model_dir = Path(config.model_dir)
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"slm_ppo_{config.env_type}"
    model.save(str(model_path))
    print(f"Model saved to {model_path}.zip")

    # Quick eval
    rewards = []
    correct = 0
    for ep in range(min(config.eval_episodes, 100)):
        obs, info = eval_env.reset(seed=ep + 50000)
        action, _ = model.predict(obs, deterministic=True)
        action = int(np.asarray(action).flat[0])
        obs, reward, terminated, truncated, info = eval_env.step(action)
        rewards.append(reward)
        actual = info.get("actual_return", 0.0)
        pred_dir = info.get("predicted_direction", 1)
        is_correct = (
            (pred_dir == 2 and actual > 0.005)
            or (pred_dir == 0 and actual < -0.005)
            or (pred_dir == 1 and abs(actual) <= 0.005)
        )
        if is_correct:
            correct += 1

    total = min(config.eval_episodes, 100)
    print(f"Eval: reward={np.mean(rewards):+.3f} accuracy={correct / total:.1%}")
    train_env.close()
    eval_env.close()


def cmd_eval(args):
    """Evaluate an agent or baseline."""
    import numpy as np

    import orin  # noqa: F401
    from orin.wrappers.sb3 import make_sb3_env

    env = make_sb3_env(args.env, obs_size=256)
    rewards = []
    correct = 0

    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        action = np.random.randint(0, 15)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        actual = info.get("actual_return", 0.0)
        pred_dir = info.get("predicted_direction", 1)
        if (
            (pred_dir == 2 and actual > 0.005)
            or (pred_dir == 0 and actual < -0.005)
            or (pred_dir == 1 and abs(actual) <= 0.005)
        ):
            correct += 1

    env.close()
    print(f"Random baseline on {args.env}:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Mean reward: {np.mean(rewards):+.3f}")
    print(f"  Accuracy: {correct / args.episodes:.1%}")

    if args.model:
        try:
            from stable_baselines3 import PPO

            model = PPO.load(args.model)
            # Re-run with model
            env = make_sb3_env(args.env, obs_size=256)
            rewards = []
            correct = 0
            for ep in range(args.episodes):
                obs, info = env.reset(seed=ep)
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)
                actual = info.get("actual_return", 0.0)
                pred_dir = info.get("predicted_direction", 1)
                if (
                    (pred_dir == 2 and actual > 0.005)
                    or (pred_dir == 0 and actual < -0.005)
                    or (pred_dir == 1 and abs(actual) <= 0.005)
                ):
                    correct += 1
            env.close()
            print(f"\nTrained model ({args.model}):")
            print(f"  Mean reward: {np.mean(rewards):+.3f}")
            print(f"  Accuracy: {correct / args.episodes:.1%}")
        except ImportError:
            print("stable-baselines3 required for model evaluation")


def cmd_generate(args):
    """Generate synthetic data."""
    from orin.data.generator import (
        generate_earnings,
        generate_filings,
        generate_macro,
        generate_news,
    )

    generators = {
        "earnings": generate_earnings,
        "news": generate_news,
        "filing": generate_filings,
        "macro": generate_macro,
    }

    gen_fn = generators.get(args.type)
    if gen_fn is None:
        print(f"Unknown type: {args.type}. Choose from: {list(generators.keys())}")
        sys.exit(1)

    records = gen_fn(n=args.n, seed=args.seed, difficulty=args.difficulty)

    if args.output:
        from pathlib import Path

        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"Generated {len(records)} {args.type} records -> {path}")
    else:
        up = sum(1 for r in records if r["outcome"]["direction"] == "up")
        down = sum(1 for r in records if r["outcome"]["direction"] == "down")
        flat = sum(1 for r in records if r["outcome"]["direction"] == "flat")
        print(f"Generated {len(records)} {args.type} records (difficulty={args.difficulty})")
        print(f"  up={up} down={down} flat={flat}")


def cmd_info(args):
    """Show orin package information."""
    import gymnasium as gym

    import orin

    print(f"orin v{orin.__version__}")
    print(f"gymnasium v{gym.__version__}")
    print()

    print("Registered environments:")
    env_ids = [eid for eid in gym.registry if eid.startswith("orin/")]
    for eid in sorted(env_ids):
        print(f"  {eid}")

    print()
    print("Data generators: earnings, news, filing, macro")
    print("Wrappers: SB3Wrapper, PufferLibWrapper, SLMWrapper")
    print("Rewards: DirectionalReward, CalibrationReward, CompositeReward, SharpeReward")


def main():
    parser = argparse.ArgumentParser(
        prog="orin", description="orin RL gym for financial text"
    )
    sub = parser.add_subparsers(dest="command")

    # train
    p_train = sub.add_parser("train", help="Train an agent")
    p_train.add_argument("--config", default=None, help="Path to config file")
    p_train.add_argument(
        "--env-type",
        default="earnings",
        choices=["earnings", "news", "filing", "macro"],
    )
    p_train.add_argument("--n-records", type=int, default=300)
    p_train.add_argument("--timesteps", type=int, default=30000)
    p_train.add_argument(
        "--difficulty", default="easy", choices=["easy", "medium", "hard"]
    )
    p_train.add_argument("--seed", type=int, default=42)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate an agent")
    p_eval.add_argument("--env", default="orin/FinText-Earnings-v0")
    p_eval.add_argument("--episodes", type=int, default=100)
    p_eval.add_argument("--model", default=None, help="Path to model .zip")

    # generate
    p_gen = sub.add_parser("generate", help="Generate synthetic data")
    p_gen.add_argument(
        "--type",
        default="earnings",
        choices=["earnings", "news", "filing", "macro"],
    )
    p_gen.add_argument("--n", type=int, default=200)
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument(
        "--difficulty", default="easy", choices=["easy", "medium", "hard"]
    )
    p_gen.add_argument("--output", default=None, help="Output JSONL path")

    # info
    sub.add_parser("info", help="Show package info")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
