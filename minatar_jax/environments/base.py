"""Shared functional API and batching for the five games."""

import abc
import typing

import jax
import jax.numpy as jnp


class State(typing.Protocol):
    """Episode data carries its own random key."""

    @property
    def key(self) -> jax.Array: ...


class Environment[StateT: State](abc.ABC):
    """Immutable settings; reset and step return all episode data explicitly."""

    observation_shape: typing.ClassVar[tuple[int, int, int]]
    num_actions: typing.ClassVar[int]

    def reset(self, key: jax.Array, *, n_envs: int = 1) -> tuple[StateT, jax.Array]:
        """Reset B games: batched state and observations (B, 10, 10, C)."""
        if n_envs < 1:
            raise ValueError("n_envs must be positive")
        _, reset_key = jax.random.split(key)
        keys = jax.random.split(reset_key, n_envs)
        return jax.vmap(self._reset_and_observe)(keys)

    def step(
        self,
        state: StateT,
        actions: jax.Array,
    ) -> tuple[StateT, jax.Array, jax.Array, jax.Array]:
        """Step actions (B,); auto-reset done games. Rewards/dones are (B, 1)."""
        return jax.vmap(self._step_and_reset)(state, actions)

    def observe(self, state: StateT) -> jax.Array:
        """Encode batched state as observations (B, 10, 10, C)."""
        return jax.vmap(self._observe)(state)

    def _reset_and_observe(self, key: jax.Array) -> tuple[StateT, jax.Array]:
        state = self._reset(key)
        return state, self._observe(state)

    def _step_and_reset(
        self,
        state: StateT,
        action: jax.Array,
    ) -> tuple[StateT, jax.Array, jax.Array, jax.Array]:
        state, reward, done = self._step(state, action)
        initial = self._reset(state.key)
        state = jax.tree_util.tree_map(
            lambda start, end: jnp.where(done, start, end),
            initial,
            state,
        )
        return state, self._observe(state), reward, done

    @abc.abstractmethod
    def _reset(self, key: jax.Array) -> StateT:
        """Build one game's initial state."""
        ...

    @abc.abstractmethod
    def _step(self, state: StateT, action: jax.Array) -> tuple[StateT, jax.Array, jax.Array]:
        """Advance one game before automatic reset."""
        ...

    @abc.abstractmethod
    def _observe(self, state: StateT) -> jax.Array:
        """Encode one game's state as (10, 10, C)."""
        ...
