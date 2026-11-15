"""Functional MinAtar environments and a small Flax actor–critic."""

from minatar_jax.agent import Agent as Agent
from minatar_jax.environments.base import Environment as Environment
from minatar_jax.environments.base import State as State
from minatar_jax.factory import NAMES as NAMES
from minatar_jax.factory import Name as Name
from minatar_jax.factory import make as make

__all__ = ["Agent", "Environment", "State", "NAMES", "Name", "make"]
