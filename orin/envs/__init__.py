"""Gymnasium environments for financial text reasoning."""

from __future__ import annotations

import gymnasium as gym


def register_envs() -> None:
    """Register all orin environments with gymnasium."""
    gym.register(
        id="orin/FinText-Earnings-v0",
        entry_point="orin.envs.earnings:FinTextEarnings",
    )
    gym.register(
        id="orin/FinText-News-v0",
        entry_point="orin.envs.news:FinTextNews",
    )
    gym.register(
        id="orin/FinText-Filing-v0",
        entry_point="orin.envs.filing:FinTextFiling",
    )
    gym.register(
        id="orin/FinText-Macro-v0",
        entry_point="orin.envs.macro:FinTextMacro",
    )
