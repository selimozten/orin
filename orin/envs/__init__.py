"""Gymnasium environments for financial text reasoning."""

from __future__ import annotations

import gymnasium as gym

_registered = False


def register_envs() -> None:
    """Register all orin environments with gymnasium (idempotent)."""
    global _registered
    if _registered:
        return
    _registered = True

    _envs = {
        "orin/FinText-Earnings-v0": "orin.envs.earnings:FinTextEarnings",
        "orin/FinText-News-v0": "orin.envs.news:FinTextNews",
        "orin/FinText-Filing-v0": "orin.envs.filing:FinTextFiling",
        "orin/FinText-Macro-v0": "orin.envs.macro:FinTextMacro",
    }
    for env_id, entry_point in _envs.items():
        gym.register(id=env_id, entry_point=entry_point, disable_env_checker=True)

    # v1 variants with multi-step episodes
    _envs_v1 = {
        "orin/FinText-Earnings-v1": (
            "orin.envs.earnings:FinTextEarnings",
            {"episode_length": 5},
        ),
        "orin/FinText-News-v1": (
            "orin.envs.news:FinTextNews",
            {"episode_length": 5},
        ),
        "orin/FinText-Filing-v1": (
            "orin.envs.filing:FinTextFiling",
            {"episode_length": 5},
        ),
        "orin/FinText-Macro-v1": (
            "orin.envs.macro:FinTextMacro",
            {"episode_length": 5},
        ),
    }
    for env_id, (entry_point, kwargs) in _envs_v1.items():
        gym.register(
            id=env_id,
            entry_point=entry_point,
            kwargs=kwargs,
            disable_env_checker=True,
        )
