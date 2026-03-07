"""orin -- RL gym for training LLMs on financial text reasoning."""

from orin.envs import register_envs

register_envs()

__all__ = ["__version__"]
__version__ = "0.1.0"
