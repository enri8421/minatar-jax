"""Asterix: collect gold and avoid enemies on eight lanes.

Actions: right, up, left, down, stay. Channels: player, enemies, trails, gold.
Difficulty rises every ``ramp_interval`` frames.
An enemy collision ends the episode; each collected gold piece rewards one point.
"""

import dataclasses
import typing

import jax
import jax.numpy as jnp
from flax import struct

from minatar_jax.environments import base


class AsterixState(struct.PyTreeNode):
    """Unbatched game data; scalar fields have shape (1,)."""

    player_x: jax.Array
    player_y: jax.Array
    entities: jax.Array  # (8, 3): x, direction, kind (-1 empty, 0 enemy, 1 gold).
    spawn_speed: jax.Array
    spawn_timer: jax.Array
    move_timer: jax.Array
    move_speed: jax.Array
    ramp_timer: jax.Array
    ramp_index: jax.Array
    key: jax.Array  # PRNG key for the next random transition.


@dataclasses.dataclass(frozen=True, kw_only=True)
class Asterix(base.Environment[AsterixState]):
    """Immutable settings for Asterix."""

    ramp_interval: int = 100

    observation_shape: typing.ClassVar[tuple[int, int, int]] = (10, 10, 4)
    num_actions: typing.ClassVar[int] = 5
    _INIT_SPAWN_SPEED: typing.ClassVar[int] = 10
    _INIT_MOVE_INTERVAL: typing.ClassVar[int] = 5

    def _reset(self, key: jax.Array) -> AsterixState:
        """Create one initial game state from an explicit key."""
        entities = jnp.zeros((8, 3), dtype=int).at[:, 2].set(-1)
        return AsterixState(
            player_x=jnp.array([5]),
            player_y=jnp.array([5]),
            entities=entities,
            spawn_speed=jnp.array([self._INIT_SPAWN_SPEED]),
            spawn_timer=jnp.array([self._INIT_SPAWN_SPEED]),
            move_timer=jnp.array([self._INIT_MOVE_INTERVAL]),
            move_speed=jnp.array([self._INIT_MOVE_INTERVAL]),
            ramp_timer=jnp.array([self.ramp_interval]),
            ramp_index=jnp.array([0]),
            key=key,
        )

    def _step(
        self,
        state: AsterixState,
        action: jax.Array,
    ) -> tuple[AsterixState, jax.Array, jax.Array]:
        """Advance one game; automatic reset is handled by the base class."""

        # Spawn entities, then resolve contact on both sides of their movement.
        key, entity_key = jax.random.split(state.key)
        entities = jnp.where(
            state.spawn_timer == 0,
            self._spawn_entity(entity_key, state.entities),
            state.entities,
        )
        spawn_timer = state.spawn_timer + state.spawn_speed * (state.spawn_timer == 0)
        player_x = (
            state.player_x
            + (action == 0) * (state.player_x < 9)
            - (action == 2) * (state.player_x > 0)
        )
        player_y = (
            state.player_y
            + (action == 3) * (state.player_y < 8)
            - (action == 1) * (state.player_y > 1)
        )
        reward = jnp.array([0])
        done = jnp.array([False])
        entities, reward, done = self._check_collision(entities, player_x, player_y, reward, done)
        can_move = state.move_timer == 0
        move_timer = state.move_timer + state.move_speed * (state.move_timer == 0)
        entities = entities.at[:, 0].add(
            (entities[:, 1] * 2 - 1) * (entities[:, 2] >= 0) * can_move,
        )
        overflow = (entities[:, 0] < 0) + (entities[:, 0] > 9)
        entities = entities.at[:, 2].set(entities[:, 2] * (1 - overflow) - 1 * overflow)
        entities, reward, done = self._check_collision(entities, player_x, player_y, reward, done)
        spawn_timer -= 1
        move_timer -= 1

        # Increase difficulty when the ramp timer expires.
        ramp_timer = state.ramp_timer - (state.ramp_timer >= 0) * jnp.logical_or(
            state.spawn_speed > 1,
            state.move_speed > 1,
        )
        move_speed = state.move_speed - (state.ramp_timer < 0) * (state.move_speed > 1) * (
            state.ramp_index % 2 == 1
        )
        spawn_speed = state.spawn_speed - (state.ramp_timer < 0) * (state.spawn_speed > 1)
        ramp_index = state.ramp_index + (state.ramp_timer < 0)
        ramp_timer = ramp_timer * (state.ramp_timer >= 0) + self.ramp_interval * (
            state.ramp_timer < 0
        )
        state = AsterixState(
            player_x=player_x,
            player_y=player_y,
            entities=entities,
            spawn_speed=spawn_speed,
            spawn_timer=spawn_timer,
            move_timer=move_timer,
            move_speed=move_speed,
            ramp_timer=ramp_timer,
            ramp_index=ramp_index,
            key=key,
        )
        return state, reward, done

    def _observe(self, state: AsterixState) -> jax.Array:
        """Encode state as a (10, 10, 4) observation."""
        obs = jnp.zeros((10, 10, 4))
        obs = obs.at[state.player_y, state.player_x, 0].set(1)
        obs = obs.at[jnp.arange(1, 9), state.entities[:, 0], 1].set(state.entities[:, 2] == 0)
        obs = obs.at[jnp.arange(1, 9), state.entities[:, 0], 3].set(state.entities[:, 2] == 1)
        back_x = state.entities[:, 0] - 2 * state.entities[:, 1] + 1
        back_not_overflow = (back_x >= 0) * (back_x <= 9)
        obs = obs.at[jnp.arange(1, 9), back_x, 2].set(
            back_not_overflow * (state.entities[:, 2] >= 0),
            mode="drop",
        )
        return obs

    def _spawn_entity(self, key: jax.Array, entities: jax.Array) -> jax.Array:
        """Spawn in an empty lane when space is available."""
        key_y, key_dir, key_is_gold = jax.random.split(key, 3)
        direction = jax.random.bernoulli(key_dir)
        is_gold = jax.random.bernoulli(key_is_gold, 1 / 3)
        x = 9 * (1 - direction)
        y = jnp.argmax((entities[:, 2] < 0) * (jax.random.permutation(key_y, 8) + 1))
        new_entity = jnp.array([x, direction, is_gold])
        free_slot = jnp.sum(entities[:, 2] < 0) > 0
        entities = entities.at[y].set(jnp.where(free_slot, new_entity, entities[y]))
        return entities

    def _check_collision(
        self,
        entities: jax.Array,
        player_x: jax.Array,
        player_y: jax.Array,
        reward: jax.Array,
        done: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Collect gold or flag an enemy collision."""
        possible_collision = entities[player_y - 1, 0] == player_x
        collision_object = entities[player_y - 1, 2]
        collision = jnp.logical_and(collision_object >= 0, possible_collision > 0)
        is_gold = jnp.logical_and(collision, collision_object == 1)
        entities = entities.at[player_y - 1, 2].set(
            entities[player_y - 1, 2] * (1 - is_gold) - 1 * is_gold,
        )
        reward += is_gold
        done = jnp.logical_or(done, jnp.logical_and(collision, jnp.logical_not(is_gold)))
        return entities, reward, done
