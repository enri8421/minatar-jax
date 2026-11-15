"""Freeway: cross eight lanes of traffic.

Actions: up, stay, down. Channels: player, cars, and five speed-specific trails.
Reaching the top rewards one point and changes
car speeds. A collision returns the player to the bottom; the time limit ends play.
"""

import dataclasses
import typing

import jax
import jax.numpy as jnp
from flax import struct

from minatar_jax.environments import base


class FreewayState(struct.PyTreeNode):
    """Unbatched game data; scalar fields have shape (1,)."""

    pos: jax.Array
    cars: jax.Array  # (8, 3): x, movement timer, signed speed.
    move_timer: jax.Array
    current_frame: jax.Array
    key: jax.Array  # PRNG key for the next random transition.


@dataclasses.dataclass(frozen=True, kw_only=True)
class Freeway(base.Environment[FreewayState]):
    """Immutable settings for Freeway."""

    player_speed: int = 3
    time_limit: int = 2500

    observation_shape: typing.ClassVar[tuple[int, int, int]] = (10, 10, 7)
    num_actions: typing.ClassVar[int] = 3

    def _reset(self, key: jax.Array) -> FreewayState:
        """Create one initial game state from an explicit key."""
        key, cars_key = jax.random.split(key)
        cars = jnp.zeros((8, 3), dtype=int)
        cars = self._randomize_speeds(cars_key, cars)
        return FreewayState(
            pos=jnp.array([9]),
            cars=cars,
            move_timer=jnp.array([self.player_speed]),
            current_frame=jnp.array([0]),
            key=key,
        )

    def _step(
        self,
        state: FreewayState,
        action: jax.Array,
    ) -> tuple[FreewayState, jax.Array, jax.Array]:
        """Advance one game; automatic reset is handled by the base class."""

        # Move the player; reaching the top starts another crossing.
        pos = (
            state.pos
            - (action == 0) * (state.move_timer == 0)
            + (action == 2) * (state.move_timer == 0) * (state.pos < 9)
        )
        has_moved = jnp.logical_and(state.move_timer == 0, jnp.logical_or(action == 0, action == 2))
        move_timer = state.move_timer * (1 - has_moved) + self.player_speed * has_moved
        reward = jnp.logical_not(pos)
        pos = 9 * reward + pos * (1 - reward)
        key, cars_key = jax.random.split(state.key)
        cars = jnp.where(reward, self._randomize_speeds(cars_key, state.cars), state.cars)

        # Detect contact before and after the cars move.
        car_agent_map = (pos - 1) * (pos > 0) * (pos < 9) + 8 * (pos == 0) + 8 * (pos == 9)
        hit_by_car = cars.at[car_agent_map, 0].get(mode="fill", fill_value=-1) == 4
        cars_can_move = cars[:, 1] == 0
        cars = cars.at[:, 1].set(jnp.where(cars_can_move, jnp.abs(cars[:, 2]), cars[:, 1] - 1))
        cars = cars.at[:, 0].set(
            jnp.where(cars_can_move, (cars[:, 0] + jnp.sign(cars[:, 2])) % 10, cars[:, 0]),
        )
        hit_by_car_after = cars.at[car_agent_map, 0].get(mode="fill", fill_value=-1) == 4
        go_back = jnp.logical_or(hit_by_car, hit_by_car_after)
        pos = 9 * go_back + pos * (1 - go_back)
        current_frame = state.current_frame + 1
        move_timer -= move_timer > 0
        done = current_frame == self.time_limit
        state = FreewayState(
            pos=pos,
            cars=cars,
            move_timer=move_timer,
            current_frame=current_frame,
            key=key,
        )
        return state, reward.astype(jnp.int32), done

    def _observe(self, state: FreewayState) -> jax.Array:
        """Encode state as a (10, 10, 7) observation."""
        obs = jnp.zeros((10, 10, 7))
        obs = obs.at[state.pos, 4, 0].set(1)
        obs = obs.at[jnp.arange(1, 9), state.cars[:, 0], 1].set(1)
        cars_back = (state.cars[:, 0] - jnp.sign(state.cars[:, 2])) % 10
        channel_back = jnp.abs(state.cars[:, 2]) + 1
        obs = obs.at[jnp.arange(1, 9), cars_back, channel_back].set(1)
        return obs

    def _randomize_speeds(self, key: jax.Array, cars: jax.Array) -> jax.Array:
        """Draw car speeds and directions after a crossing."""
        speed_key, direction_key = jax.random.split(key)
        speeds = jax.random.randint(speed_key, (8,), 1, 6)
        directions = jax.random.bernoulli(direction_key, 0.5, (8,)) * 2 - 1
        cars = cars.at[:, 1].set(speeds)
        cars = cars.at[:, 2].set(speeds * directions)
        return cars
