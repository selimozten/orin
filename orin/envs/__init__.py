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
        gym.register(id=env_id, entry_point=entry_point)
