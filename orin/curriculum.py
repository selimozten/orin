"""Curriculum learning for progressive difficulty."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CurriculumScheduler:
    """Schedule difficulty progression: easy -> medium -> hard.

    Transitions based on accuracy thresholds over a rolling window.
    """

    easy_to_medium_accuracy: float = 0.65
    medium_to_hard_accuracy: float = 0.70
    window_size: int = 200
    current_difficulty: str = "easy"
    _history: list[float] = field(default_factory=list, init=False, repr=False)

    def update(self, correct: bool) -> str:
        """Record a prediction result and return current difficulty."""
        self._history.append(float(correct))
        if len(self._history) > self.window_size:
            self._history = self._history[-self.window_size:]

        if len(self._history) >= self.window_size:
            acc = np.mean(self._history)
            if self.current_difficulty == "easy" and acc >= self.easy_to_medium_accuracy:
                self.current_difficulty = "medium"
                self._history.clear()
            elif self.current_difficulty == "medium" and acc >= self.medium_to_hard_accuracy:
                self.current_difficulty = "hard"
                self._history.clear()

        return self.current_difficulty

    def reset(self) -> None:
        self._history.clear()
        self.current_difficulty = "easy"


class CurriculumCallback:
    """SB3 callback that updates environment data at difficulty transitions.

    Usage with stable-baselines3:
        scheduler = CurriculumScheduler()
        callback = CurriculumCallback(scheduler, generate_fn, env)
        model.learn(total_timesteps=100000, callback=callback)
    """

    def __init__(self, scheduler, generate_fn, env, n_records=300, seed=42):
        self.scheduler = scheduler
        self.generate_fn = generate_fn
        self.env = env
        self.n_records = n_records
        self.seed = seed
        self._last_difficulty = scheduler.current_difficulty

        try:
            from stable_baselines3.common.callbacks import BaseCallback  # noqa: F401

            self._base_class = BaseCallback
        except ImportError:
            self._base_class = object

    def _on_step(self):
        # This would be called by SB3 training loop
        infos = getattr(self, "locals", {}).get("infos", [])
        for info in infos:
            if "actual_return" in info:
                actual = info["actual_return"]
                pred_dir = info.get("predicted_direction", 1)
                correct = (
                    (pred_dir == 2 and actual > 0.005)
                    or (pred_dir == 0 and actual < -0.005)
                    or (pred_dir == 1 and abs(actual) <= 0.005)
                )
                new_diff = self.scheduler.update(correct)

                if new_diff != self._last_difficulty:
                    print(f"Curriculum: {self._last_difficulty} -> {new_diff}")
                    new_data = self.generate_fn(
                        self.n_records, seed=self.seed, difficulty=new_diff
                    )
                    # Update the underlying environment's data
                    unwrapped = self.env
                    while hasattr(unwrapped, "env"):
                        unwrapped = unwrapped.env
                    unwrapped.data = new_data
                    unwrapped._indices = []
                    unwrapped._current_idx = 0
                    self._last_difficulty = new_diff

        return True
