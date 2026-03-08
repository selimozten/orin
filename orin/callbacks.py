"""Training callbacks for monitoring orin RL training."""

from __future__ import annotations

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    BaseCallback = object  # stub for when SB3 not installed

try:
    import wandb as _wandb
except ImportError:
    _wandb = None


class OrinCallback(BaseCallback):
    """Callback logging reward, accuracy, confidence distribution per epoch."""

    def __init__(
        self,
        log_interval: int = 1000,
        verbose: int = 0,
        use_wandb: bool = False,
    ):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.use_wandb = use_wandb and _wandb is not None
        self._rewards: list[float] = []
        self._accuracies: list[float] = []
        self._confidences: list[float] = []
        self._predictions: list[int] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "actual_return" in info:
                actual = info["actual_return"]
                pred_dir = info.get("predicted_direction", 1)
                conf = info.get("confidence", 0.5)
                correct = (
                    (pred_dir == 2 and actual > 0.005)
                    or (pred_dir == 0 and actual < -0.005)
                    or (pred_dir == 1 and abs(actual) <= 0.005)
                )
                raw_rewards = self.locals.get("rewards", [])
                reward = float(raw_rewards[-1]) if len(raw_rewards) else 0.0
                self._rewards.append(reward)
                self._accuracies.append(float(correct))
                self._confidences.append(conf)
                self._predictions.append(pred_dir)

        if self.num_timesteps % self.log_interval == 0 and self._rewards:
            n = min(len(self._rewards), self.log_interval)
            recent_rewards = self._rewards[-n:]
            recent_acc = self._accuracies[-n:]
            recent_conf = self._confidences[-n:]
            recent_preds = self._predictions[-n:]

            mean_reward = float(np.mean(recent_rewards))
            accuracy = float(np.mean(recent_acc))
            mean_conf = float(np.mean(recent_conf))
            std_conf = float(np.std(recent_conf))

            self.logger.record("orin/mean_reward", mean_reward)
            self.logger.record("orin/accuracy", accuracy)
            self.logger.record("orin/mean_confidence", mean_conf)
            self.logger.record("orin/std_confidence", std_conf)

            pred_counts = {0: 0, 1: 0, 2: 0}
            for p in recent_preds:
                pred_counts[p] = pred_counts.get(p, 0) + 1
            total = len(recent_preds)
            pct_up = pred_counts[2] / total
            pct_down = pred_counts[0] / total
            pct_flat = pred_counts[1] / total
            self.logger.record("orin/pct_up", pct_up)
            self.logger.record("orin/pct_down", pct_down)
            self.logger.record("orin/pct_flat", pct_flat)

            if self.use_wandb:
                _wandb.log(
                    {
                        "orin/mean_reward": mean_reward,
                        "orin/accuracy": accuracy,
                        "orin/mean_confidence": mean_conf,
                        "orin/std_confidence": std_conf,
                        "orin/pct_up": pct_up,
                        "orin/pct_down": pct_down,
                        "orin/pct_flat": pct_flat,
                    },
                    step=self.num_timesteps,
                )

            if self.verbose:
                print(
                    f"[{self.num_timesteps}] "
                    f"reward={mean_reward:.3f} "
                    f"acc={accuracy:.1%} "
                    f"conf={mean_conf:.2f}"
                )

        return True

    @property
    def metrics(self) -> dict:
        """Return all collected metrics."""
        return {
            "rewards": list(self._rewards),
            "accuracies": list(self._accuracies),
            "confidences": list(self._confidences),
            "predictions": list(self._predictions),
        }
