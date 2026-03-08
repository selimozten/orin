"""Configuration management for orin training runs."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    """Training configuration."""

    # Environment
    env_type: str = "earnings"
    env_id: str = "orin/FinText-Earnings-v0"
    episode_length: int = 1

    # Data
    n_records: int = 300
    difficulty: str = "easy"
    noise_rate: float | None = None
    augment: bool = False

    # Model
    model_name: str = "all-MiniLM-L6-v2"
    policy: str = "MlpPolicy"
    net_arch: list[int] = field(default_factory=lambda: [256, 128])

    # Training
    timesteps: int = 30000
    n_steps: int = 128
    batch_size: int = 64
    n_epochs: int = 10
    learning_rate: float = 3e-4
    ent_coef: float = 0.03
    gamma: float = 0.0
    gae_lambda: float = 1.0
    clip_range: float = 0.2
    seed: int = 42

    # Evaluation
    eval_episodes: int = 300

    # Reward
    directional_weight: float = 0.7
    calibration_weight: float = 0.3
    calibration_mode: str = "brier"

    # Output
    model_dir: str = "models"
    results_dir: str = "results"


def load_config(path: str | Path) -> TrainConfig:
    """Load config from JSON or YAML file."""
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml

            data = yaml.safe_load(text)
        except ImportError:
            raise ImportError("PyYAML required for YAML configs: pip install pyyaml")
    else:
        data = json.loads(text)
    return TrainConfig(**{k: v for k, v in data.items() if k in TrainConfig.__dataclass_fields__})


def save_config(config: TrainConfig, path: str | Path) -> None:
    """Save config to JSON or YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(config)
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml

            path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
        except ImportError:
            raise ImportError("PyYAML required for YAML configs: pip install pyyaml")
    else:
        path.write_text(json.dumps(data, indent=2) + "\n")
