"""Space Invaders: shoot descending aliens while avoiding their bullets.

Actions: left, stay, right, shoot. Channels: player, aliens, left-moving aliens,
right-moving aliens, friendly bullets, enemy bullets. Aliens start in a random
direction. Alien hits reward one point; a collision ends play.
"""

import dataclasses
import typing

import jax
import jax.numpy as jnp
from flax import struct

from minatar_jax.environments import base


class SpaceInvadersState(struct.PyTreeNode):
    """Unbatched game data; scalar fields have shape (1,)."""

    pos: jax.Array
    f_bullet: jax.Array  # (10, 10): friendly bullets.
    e_bullet: jax.Array  # (10, 10): enemy bullets.
    alien_pos: jax.Array  # (10, 10): alien locations.
    alien_dir: jax.Array
    shot_timer: jax.Array
    alien_shot_timer: jax.Array
    alien_move_timer: jax.Array
    alien_move_interval: jax.Array
    key: jax.Array  # PRNG key for the next random transition.


@dataclasses.dataclass(frozen=True, kw_only=True)
class SpaceInvaders(base.Environment[SpaceInvadersState]):
    """Immutable settings for SpaceInvaders."""

    shot_cool_down: int = 5

    observation_shape: typing.ClassVar[tuple[int, int, int]] = (10, 10, 6)
    num_actions: typing.ClassVar[int] = 4
    _ENEMY_MOVE_INTERVAL: typing.ClassVar[int] = 12
    _ENEMY_SHOT_INTERVAL: typing.ClassVar[int] = 10
    _COLUMN_ORDER: typing.ClassVar[jax.Array] = jnp.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 0, 2, 3, 4, 5, 6, 7, 8, 9],
            [2, 1, 3, 0, 4, 5, 6, 7, 8, 9],
            [3, 2, 4, 1, 5, 0, 6, 7, 8, 9],
            [4, 3, 5, 2, 6, 1, 7, 0, 8, 9],
            [5, 4, 6, 3, 7, 2, 8, 1, 9, 0],
            [6, 5, 7, 4, 8, 3, 9, 2, 1, 0],
            [7, 6, 8, 5, 9, 4, 3, 2, 1, 0],
            [8, 7, 9, 6, 5, 4, 3, 2, 1, 0],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        ],
    )

    def _reset(self, key: jax.Array) -> SpaceInvadersState:
        """Create one initial game state from an explicit key."""
        key, sample_key = jax.random.split(key)
        alien_pos = jnp.zeros((10, 10), dtype=int).at[0:4, 2:8].set(1)
        direction = jnp.array([2 * jax.random.bernoulli(sample_key) - 1])
        return SpaceInvadersState(
            pos=jnp.array([5]),
            f_bullet=jnp.zeros((10, 10), dtype=int),
            e_bullet=jnp.zeros((10, 10), dtype=int),
            alien_pos=alien_pos,
            alien_dir=direction,
            shot_timer=jnp.array([0]),
            alien_shot_timer=jnp.array([self._ENEMY_SHOT_INTERVAL]),
            alien_move_timer=jnp.array([self._ENEMY_MOVE_INTERVAL]),
            alien_move_interval=jnp.array([self._ENEMY_MOVE_INTERVAL]),
            key=key,
        )

    def _step(
        self,
        state: SpaceInvadersState,
        action: jax.Array,
    ) -> tuple[SpaceInvadersState, jax.Array, jax.Array]:
        """Advance one game; automatic reset is handled by the base class."""

        # Move the player and bullets.
        done = jnp.array([False])
        pos = state.pos - (action == 0) * (state.pos > 0) + (action == 2) * (state.pos < 9)
        has_shot = jnp.logical_and(action == 3, state.shot_timer == 0)
        f_bullet = state.f_bullet.at[9, pos].set(has_shot)
        shot_timer = state.shot_timer * (1 - has_shot) + self.shot_cool_down * has_shot
        f_bullet = jnp.roll(f_bullet, -1, axis=0)
        f_bullet = f_bullet.at[9, :].set(0)
        e_bullet = jnp.roll(state.e_bullet, 1, axis=0)
        e_bullet = e_bullet.at[0, :].set(0)
        done = jnp.logical_or(e_bullet.at[9, pos].get(), done)
        done = jnp.logical_or(state.alien_pos.at[9, pos].get(), done)

        # Aliens reverse direction and descend when they reach an edge.
        alien_can_move = state.alien_move_timer == 0
        alien_move_timer = (
            state.alien_move_timer * (1 - alien_can_move)
            + jnp.minimum(jnp.count_nonzero(state.alien_pos), state.alien_move_interval)
            * alien_can_move
        )
        move_vertically = jnp.logical_or(
            jnp.logical_and(jnp.sum(state.alien_pos[:, 0]) > 0, state.alien_dir < 0),
            jnp.logical_and(jnp.sum(state.alien_pos[:, 9]) > 0, state.alien_dir > 0),
        )
        alien_dir = (
            state.alien_dir * (1 - move_vertically * alien_can_move)
            - state.alien_dir * move_vertically * alien_can_move
        )
        reached_bottom = jnp.any(state.alien_pos[9]) & move_vertically & alien_can_move
        done = jnp.logical_or(reached_bottom, done)
        alien_pos = jnp.where(
            alien_can_move,
            jnp.where(
                move_vertically,
                jnp.roll(state.alien_pos, 1, axis=0),
                jnp.where(
                    alien_dir < 0,
                    jnp.roll(state.alien_pos, -1, axis=1),
                    jnp.roll(state.alien_pos, 1, axis=1),
                ),
            ),
            state.alien_pos,
        )
        done = jnp.logical_or(alien_pos.at[9, pos].get(), done)
        alien_can_shot = state.alien_shot_timer == 0
        e_bullet = jnp.where(
            alien_can_shot,
            self._nearest_alien_shot(alien_pos, pos, e_bullet),
            e_bullet,
        )
        alien_shot_timer = (
            state.alien_shot_timer * (1 - alien_can_shot)
            + self._ENEMY_SHOT_INTERVAL * alien_can_shot
        )

        # Resolve hits and repopulate a cleared wave.
        kill_locations = jnp.logical_and(alien_pos, f_bullet)
        reward = jnp.expand_dims(jnp.sum(kill_locations), axis=0)
        alien_pos = jnp.logical_and(alien_pos, jnp.logical_not(kill_locations))
        f_bullet = jnp.logical_and(f_bullet, jnp.logical_not(kill_locations))
        shot_timer -= shot_timer > 0
        alien_move_timer -= 1
        alien_shot_timer -= 1
        all_alien_kill = jnp.count_nonzero(alien_pos) == 0
        alien_move_interval = (
            state.alien_move_interval * (1 - all_alien_kill)
            + jnp.maximum(6, state.alien_move_interval - 1) * all_alien_kill
        )
        alien_pos = jnp.where(
            all_alien_kill,
            jnp.zeros((10, 10), dtype=int).at[0:4, 2:8].set(1),
            alien_pos,
        )
        state = SpaceInvadersState(
            pos=pos,
            f_bullet=f_bullet,
            e_bullet=e_bullet,
            alien_pos=alien_pos,
            alien_dir=alien_dir,
            shot_timer=shot_timer,
            alien_shot_timer=alien_shot_timer,
            alien_move_timer=alien_move_timer,
            alien_move_interval=alien_move_interval,
            key=state.key,
        )
        return state, reward, done

    def _observe(self, state: SpaceInvadersState) -> jax.Array:
        """Encode state as a (10, 10, 6) observation."""
        obs = jnp.zeros((10, 10, 6))
        obs = obs.at[9, state.pos, 0].set(1)
        obs = obs.at[:, :, 1].set(state.alien_pos)
        obs = obs.at[:, :, 2].set(
            jnp.where(state.alien_dir > 0, jnp.zeros((10, 10)), state.alien_pos),
        )
        obs = obs.at[:, :, 3].set(
            jnp.where(state.alien_dir > 0, state.alien_pos, jnp.zeros((10, 10))),
        )
        obs = obs.at[:, :, 4].set(state.f_bullet)
        obs = obs.at[:, :, 5].set(state.e_bullet)
        return obs

    def _nearest_alien_shot(
        self,
        alien_map: jax.Array,
        pos: jax.Array,
        e_bullet: jax.Array,
    ) -> jax.Array:
        """Fire from the lowest alien in the nearest occupied column."""
        x = self._COLUMN_ORDER[
            pos,
            jnp.where(jnp.sum(alien_map, axis=0)[self._COLUMN_ORDER[pos]] > 0, size=1)[1],
        ].squeeze()
        y = jnp.argmax(alien_map[:, x] * jnp.arange(10))
        return e_bullet.at[y, x].set(1)
