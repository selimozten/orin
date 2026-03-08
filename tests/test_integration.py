"""End-to-end integration tests."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

import orin  # noqa: F401
from orin.data.generator import generate_earnings


def test_generate_wrap_step():
    """Generate data -> create env -> step through episodes."""
    data = generate_earnings(50, seed=99)
    env = gym.make("orin/FinText-Earnings-v0", data=data)

    total_reward = 0.0
    for ep in range(10):
        obs, info = env.reset(seed=ep)
        action = {"direction": 2, "confidence": np.float32(0.7)}
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        assert terminated

    env.close()
    assert total_reward != 0.0  # non-trivial


def test_sb3_wrapper_integration():
    """SB3 wrapper should produce valid obs and rewards."""
    from orin.wrappers.sb3 import make_sb3_env

    data = generate_earnings(20, seed=42)
    env = make_sb3_env("orin/FinText-Earnings-v0", data=data)

    obs, info = env.reset(seed=0)
    assert obs.shape == (512,)

    for i in range(5):
        obs, reward, terminated, truncated, info = env.step(7)
        assert isinstance(reward, float)

    env.close()


def test_metrics_module():
    """Verify metrics module works standalone."""
    from orin.eval.metrics import calibration_curve, confusion_matrix, direction_metrics

    y_true = [0, 1, 2, 0, 1, 2, 2, 0]
    y_pred = [0, 1, 2, 1, 0, 2, 0, 0]

    cm = confusion_matrix(y_true, y_pred)
    assert cm.shape == (3, 3)
    assert cm.sum() == len(y_true)

    dm = direction_metrics(y_true, y_pred)
    assert "down" in dm and "flat" in dm and "up" in dm

    confs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    corrects = [True, True, False, True, False, True, False, False]
    cc = calibration_curve(confs, corrects, n_bins=4)
    assert "ece" in cc


def test_config_roundtrip(tmp_path):
    """Config save/load roundtrip."""
    from orin.config import TrainConfig, load_config, save_config

    config = TrainConfig(env_type="news", timesteps=5000)
    path = tmp_path / "test_config.json"
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.env_type == "news"
    assert loaded.timesteps == 5000


def test_curriculum_scheduler():
    """Curriculum scheduler progresses through difficulties."""
    from orin.curriculum import CurriculumScheduler

    sched = CurriculumScheduler(window_size=10, easy_to_medium_accuracy=0.7)
    assert sched.current_difficulty == "easy"

    # Feed enough correct answers to trigger transition
    for _ in range(10):
        sched.update(True)

    assert sched.current_difficulty == "medium"


def test_benchmark_runs():
    """Benchmark suite runs without error."""
    from orin.benchmark import format_report, run_benchmark

    results = run_benchmark(n_episodes=10, difficulties=["easy"])
    assert "earnings" in results
    report = format_report(results)
    assert "Benchmark" in report
