"""Seaquest: rescue divers, shoot enemies, and surface to refill oxygen.

Actions: right, up, left, down, stay, shoot. Channels: player, player trail,
friendly bullets, object trails, enemy bullets, fish, submarines, oxygen,
passenger count, divers. Surfacing unloads one diver, or all six for a bonus.
Enemy hits, exhausted oxygen, or surfacing without divers end the episode.
"""

import dataclasses
import typing

import jax
import jax.numpy as jnp
from flax import struct

from minatar_jax.environments import base


class SeaquestState(struct.PyTreeNode):
    """Unbatched game data; scalar fields have shape (1,)."""

    oxygen: jax.Array
    diver_count: jax.Array
    sub_x: jax.Array
    sub_y: jax.Array
    sub_dir: jax.Array
    bullets: jax.Array  # (2, 9, 10): friendly bullets by direction.
    e_bullets: jax.Array  # (2, 8, 10): enemy bullets by direction.
    fish: jax.Array  # (3, 2, 8, 10): stack, direction, y, x; movement timers.
    e_sub: jax.Array  # (3, 2, 2, 8, 10): stack, direction, movement/shooting timer, y, x.
    divers: jax.Array  # (2, 8, 10): movement timers by direction.
    e_spawn_speed: jax.Array
    e_spawn_timer: jax.Array
    d_spawn_timer: jax.Array
    move_speed: jax.Array
    ramp_index: jax.Array
    shot_timer: jax.Array
    surface: jax.Array
    where_to_spawn: jax.Array
    key: jax.Array  # PRNG key for the next random transition.


@dataclasses.dataclass(frozen=True, kw_only=True)
class Seaquest(base.Environment[SeaquestState]):
    """Immutable settings for Seaquest."""

    max_oxygen: int = 200
    shot_cool_down: int = 5

    observation_shape: typing.ClassVar[tuple[int, int, int]] = (10, 10, 10)
    num_actions: typing.ClassVar[int] = 6
    _INIT_SPAWN_SPEED: typing.ClassVar[int] = 20
    _DIVER_SPAWN_SPEED: typing.ClassVar[int] = 30
    _INIT_MOVE_INTERVAL: typing.ClassVar[int] = 5
    _ENEMY_SHOT_INTERVAL: typing.ClassVar[int] = 10
    _DIVER_MOVE_INTERVAL: typing.ClassVar[int] = 5
    _MAX_DIVERS: typing.ClassVar[int] = 6

    def _reset(self, key: jax.Array) -> SeaquestState:
        """Create one initial game state from an explicit key."""
        return SeaquestState(
            oxygen=jnp.array([self.max_oxygen]),
            diver_count=jnp.array([0]),
            sub_x=jnp.array([5]),
            sub_y=jnp.array([0]),
            sub_dir=jnp.array([0]),
            bullets=jnp.zeros((2, 9, 10), dtype=int),
            e_bullets=jnp.zeros((2, 8, 10), dtype=int),
            fish=jnp.zeros((3, 2, 8, 10), dtype=int),
            e_sub=jnp.zeros((3, 2, 2, 8, 10), dtype=int),
            divers=jnp.zeros((2, 8, 10), dtype=int),
            e_spawn_speed=jnp.array([self._INIT_SPAWN_SPEED]),
            e_spawn_timer=jnp.array([self._INIT_SPAWN_SPEED]),
            d_spawn_timer=jnp.array([self._DIVER_SPAWN_SPEED]),
            move_speed=jnp.array([self._INIT_MOVE_INTERVAL]),
            ramp_index=jnp.array([0]),
            shot_timer=jnp.array([0]),
            surface=jnp.array([True]),
            where_to_spawn=jnp.array([0]),
            key=key,
        )

    def _step(
        self,
        state: SeaquestState,
        action: jax.Array,
    ) -> tuple[SeaquestState, jax.Array, jax.Array]:
        """Advance one game; automatic reset is handled by the base class."""

        # Spawn enemies and divers.
        key, key_spawn_e, key_spawn_d = jax.random.split(state.key, 3)
        done = jnp.array([False])
        reward = jnp.array([0])
        fish, e_sub, e_spawn_timer, where_to_spawn = self._spawn_enemy(
            key_spawn_e,
            state.fish,
            state.e_sub,
            state.e_spawn_timer,
            state.where_to_spawn,
            state.e_spawn_speed,
            state.move_speed,
        )
        divers, d_spawn_timer = self._spawn_diver(key_spawn_d, state.divers, state.d_spawn_timer)

        # Move the player and fire if the weapon is ready.
        sub_x = state.sub_x + (action == 0) * (state.sub_x < 9) - (action == 2) * (state.sub_x > 0)
        sub_y = state.sub_y + (action == 3) * (state.sub_y < 8) - (action == 1) * (state.sub_y > 0)
        sub_dir = state.sub_dir * jnp.logical_or(
            jnp.logical_or(action == 1, action == 3),
            jnp.logical_or(action == 4, action == 5),
        ) + (action == 0)
        can_shot = jnp.logical_and(state.shot_timer == 0, action == 5)
        shot_timer = state.shot_timer + can_shot * self.shot_cool_down
        bullets = jnp.where(
            can_shot,
            state.bullets.at[1 - sub_dir, sub_y, sub_x].add(1),
            state.bullets,
        )

        # Resolve contact before and after moving the other objects.
        bullets = self._move_bullets(bullets)
        done = jnp.logical_or(
            done,
            self._enemy_hits_player(fish, e_sub, state.e_bullets, sub_x, sub_y),
        )
        divers, diver_count = self._pick_up_divers(divers, state.diver_count, sub_x, sub_y)
        divers = self._move_divers(divers)
        divers, diver_count = self._pick_up_divers(divers, diver_count, sub_x, sub_y)
        bullets, fish, e_sub, new_reward = self._remove_hit_targets(bullets, fish, e_sub)
        reward = reward + new_reward
        fish = self._move_fish(fish, state.move_speed)
        e_sub, e_bullets = self._move_and_shoot_submarines(e_sub, state.e_bullets, state.move_speed)
        bullets, fish, e_sub, new_reward = self._remove_hit_targets(bullets, fish, e_sub)
        reward = reward + new_reward
        e_bullets = self._move_bullets(e_bullets)
        done = jnp.logical_or(done, self._enemy_hits_player(fish, e_sub, e_bullets, sub_x, sub_y))
        e_spawn_timer -= e_spawn_timer > 0
        d_spawn_timer -= d_spawn_timer > 0
        shot_timer -= shot_timer > 0
        done = jnp.logical_or(done, state.oxygen <= 0)

        # Consume oxygen underwater and unload divers on surfacing.
        oxygen = state.oxygen - (sub_y > 0)
        surface = jnp.logical_and(state.surface, sub_y == 0)
        done = jnp.logical_or(
            done,
            jnp.logical_and(
                jnp.logical_and(sub_y == 0, jnp.logical_not(surface)),
                diver_count == 0,
            ),
        )
        surface_with_people = jnp.logical_and(
            jnp.logical_and(sub_y == 0, jnp.logical_not(surface)),
            jnp.logical_not(diver_count == 0),
        )
        surface = jnp.logical_or(surface_with_people, surface)
        reward = reward + oxygen * 10 // self.max_oxygen * jnp.logical_and(
            surface_with_people,
            diver_count == self._MAX_DIVERS,
        )
        diver_count = diver_count * jnp.logical_not(surface_with_people) + (
            diver_count - 1
        ) * jnp.logical_and(surface_with_people, diver_count < self._MAX_DIVERS)
        oxygen = oxygen * (1 - surface_with_people) + self.max_oxygen * surface_with_people

        # A successful return to the surface increases difficulty.
        move_speed = state.move_speed - surface_with_people * (state.move_speed > 2) * (
            state.ramp_index % 2 == 1
        )
        e_spawn_speed = state.e_spawn_speed - surface_with_people * (state.e_spawn_speed > 1)
        ramp_index = state.ramp_index + surface_with_people * jnp.logical_or(
            state.e_spawn_speed > 1,
            state.move_speed > 2,
        )
        state = SeaquestState(
            oxygen=oxygen,
            diver_count=diver_count,
            sub_x=sub_x,
            sub_y=sub_y,
            sub_dir=sub_dir,
            bullets=bullets,
            e_bullets=e_bullets,
            fish=fish,
            e_sub=e_sub,
            divers=divers,
            e_spawn_speed=e_spawn_speed,
            e_spawn_timer=e_spawn_timer,
            d_spawn_timer=d_spawn_timer,
            move_speed=move_speed,
            ramp_index=ramp_index,
            shot_timer=shot_timer,
            surface=surface,
            where_to_spawn=where_to_spawn,
            key=key,
        )
        return state, reward, done

    def _observe(self, state: SeaquestState) -> jax.Array:
        """Encode state as a (10, 10, 10) observation."""
        obs = jnp.zeros((10, 10, 10))
        obs = obs.at[state.sub_y, state.sub_x, 0].set(1)
        back_x = state.sub_x - 2 * state.sub_dir + 1
        obs = obs.at[state.sub_y, back_x, 1].set(1)
        obs = obs.at[9, :, 7].set(
            jnp.where(jnp.arange(10) < state.oxygen * 10 // self.max_oxygen, 1, 0),
        )
        obs = obs.at[9, 3:9, 8].set(
            jnp.where(jnp.arange(self._MAX_DIVERS) >= self._MAX_DIVERS - state.diver_count, 1, 0),
        )
        obs = obs.at[:9, :, 2].set(jnp.sum(state.bullets, axis=0) > 0)
        obs = obs.at[1:9, :, 4].set(jnp.sum(state.e_bullets, axis=0) > 0)
        obs = obs.at[1:9, :, 5].set(jnp.sum(state.fish, axis=(0, 1)) > 0)
        obs = obs.at[1:9, :, 6].set(jnp.sum(state.e_sub[:, :, 0], axis=(0, 1)) > 0)
        obs = obs.at[1:9, :, 9].set(jnp.sum(state.divers, axis=0) > 0)
        all_object_l = jnp.logical_or(
            jnp.logical_or(
                jnp.sum(state.fish[:, 1], axis=0),
                jnp.sum(state.e_sub[:, 1, 0], axis=0),
            ),
            state.divers[1],
        )
        trails_l = jnp.roll(all_object_l.at[:, 9].set(False), 1, 1)
        all_object_r = jnp.logical_or(
            jnp.logical_or(
                jnp.sum(state.fish[:, 0], axis=0),
                jnp.sum(state.e_sub[:, 0, 0], axis=0),
            ),
            state.divers[0],
        )
        trails_r = jnp.roll(all_object_r.at[:, 0].set(False), -1, 1)
        obs = obs.at[1:9, :, 3].set(jnp.logical_or(trails_l, trails_r))
        return obs

    def _spawn_enemy(
        self,
        key: jax.Array,
        fish: jax.Array,
        e_sub: jax.Array,
        e_spawn_timer: jax.Array,
        where_to_spawn: jax.Array,
        e_spawn_speed: jax.Array,
        move_speed: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Spawn an enemy when its timer expires."""
        ready_to_spawn = e_spawn_timer == 0
        e_spawn_timer += e_spawn_speed * ready_to_spawn
        where_to_spawn = (where_to_spawn + ready_to_spawn) % 3
        key_dir, key_sub, key_pos = jax.random.split(key, 3)
        e_goes_l = jax.random.bernoulli(key_dir)
        x = 9 * e_goes_l
        is_sub = jax.random.bernoulli(key_sub, p=1 / 3)
        y = jax.random.randint(key_pos, shape=(1,), minval=0, maxval=8)
        not_occupied = 1 - jnp.logical_or(
            jnp.sum(fish[:, 1 - e_goes_l, y]),
            jnp.sum(e_sub[:, 1 - e_goes_l, 0, y]),
        )
        spawn_sub = jnp.logical_and(is_sub, jnp.logical_and(ready_to_spawn, not_occupied))
        spawn_fish = jnp.logical_and(1 - is_sub, jnp.logical_and(ready_to_spawn, not_occupied))
        e_sub = e_sub.at[where_to_spawn, 1 * e_goes_l, 0, y, x].add(
            jnp.where(spawn_sub, move_speed + 1, 0),
        )
        e_sub = e_sub.at[where_to_spawn, 1 * e_goes_l, 1, y, x].add(
            jnp.where(spawn_sub, self._ENEMY_SHOT_INTERVAL + 1, 0),
        )
        fish = fish.at[where_to_spawn, 1 * e_goes_l, y, x].add(
            jnp.where(spawn_fish, move_speed + 1, 0),
        )
        return fish, e_sub, e_spawn_timer, where_to_spawn

    def _spawn_diver(
        self,
        key: jax.Array,
        divers: jax.Array,
        d_spawn_timer: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Spawn a diver when its timer expires."""
        ready_to_spawn = d_spawn_timer == 0
        d_spawn_timer += self._DIVER_SPAWN_SPEED * ready_to_spawn
        key_dir, key_pos = jax.random.split(key, 2)
        d_goes_l = jax.random.bernoulli(key_dir)
        x = 9 * d_goes_l
        y = jax.random.randint(key_pos, shape=(1,), minval=0, maxval=8)
        divers = divers.at[1 * d_goes_l, y, x].add(
            jnp.where(ready_to_spawn, self._DIVER_MOVE_INTERVAL + 1, 0),
        )
        return divers, d_spawn_timer

    def _move_divers(self, entities: jax.Array) -> jax.Array:
        """Move divers and restart their movement timers."""
        moved_map = jnp.zeros_like(entities)
        moved_map = moved_map.at[0].set(jnp.roll(entities[0].at[:, 9].set(0), 1, 1))
        moved_map = moved_map.at[1].set(jnp.roll(entities[1].at[:, 0].set(0), -1, 1))
        entities = (entities > 1) * (entities - 1) + (moved_map == 1) * (
            self._DIVER_MOVE_INTERVAL + 1
        )
        return entities

    def _move_and_shoot_submarines(
        self,
        entities: jax.Array,
        bullets: jax.Array,
        speed: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Move submarines and fire when their timers expire."""
        moved_map = jnp.zeros_like(entities)
        moved_map = moved_map.at[:, 0].set(jnp.roll(entities[:, 0].at[:, :, :, 9].set(0), 1, 3))
        moved_map = moved_map.at[:, 1].set(jnp.roll(entities[:, 1].at[:, :, :, 0].set(0), -1, 3))
        not_moved_location = entities[:, :, 0] > 1
        entities = entities.at[:, :, 0].set(
            not_moved_location * (entities[:, :, 0] - 1) + (moved_map[:, :, 0] == 1) * (speed + 1),
        )
        entities = entities.at[:, :, 1].set(
            not_moved_location * entities[:, :, 1] + (moved_map[:, :, 0] == 1) * moved_map[:, :, 1],
        )
        bullets = jnp.logical_or(bullets, jnp.sum(entities[:, :, 1] == 1, 0))
        entities = entities.at[:, :, 1].set(
            jnp.where(
                entities[:, :, 1] == 1,
                self._ENEMY_SHOT_INTERVAL + 1,
                jnp.where(entities[:, :, 1] > 1, entities[:, :, 1] - 1, 0),
            ),
        )
        return entities, bullets

    def _move_fish(self, entities: jax.Array, speed: jax.Array) -> jax.Array:
        """Move fish and restart their movement timers."""
        moved_map = jnp.zeros_like(entities)
        moved_map = moved_map.at[:, 0].set(jnp.roll(entities[:, 0].at[:, :, 9].set(0), 1, 2))
        moved_map = moved_map.at[:, 1].set(jnp.roll(entities[:, 1].at[:, :, 0].set(0), -1, 2))
        entities = (entities > 1) * (entities - 1) + (moved_map == 1) * (speed + 1)
        return entities

    def _move_bullets(self, entities: jax.Array) -> jax.Array:
        """Move bullets horizontally without wrapping at the edges."""
        entities = entities.at[0].set(jnp.roll(entities[0].at[:, 9].set(False), 1, 1))
        entities = entities.at[1].set(jnp.roll(entities[1].at[:, 0].set(False), -1, 1))
        return entities

    def _clear_oldest(self, x: jax.Array) -> jax.Array:
        """Remove the oldest occupied slot in a stack."""
        z = jnp.argmin(jnp.where(x > 0, x, jnp.max(x) + 1))
        return x.at[z].set(0)

    def _remove_hit_targets(
        self,
        bullets: jax.Array,
        fish: jax.Array,
        e_sub: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Remove hit enemies and consume the corresponding bullets."""
        comp_bullets = jnp.logical_or(bullets[0], bullets[1])[1:]
        all_fish = jnp.sum(fish, (0, 1))
        all_e_sub = jnp.sum(e_sub, (0, 1, 2))
        all_targets = jnp.logical_or(all_fish, all_e_sub)
        hit_map = all_targets * comp_bullets
        reward = jnp.sum(hit_map)
        fish_remove_old = jnp.apply_along_axis(self._clear_oldest, 0, fish)
        fish = jnp.where(comp_bullets, fish_remove_old, fish)
        comp_bullets = jnp.logical_and(comp_bullets, jnp.logical_not(all_fish))
        sub_remove_old = jnp.apply_along_axis(self._clear_oldest, 0, e_sub[:, :, 0])
        e_sub = e_sub.at[:, :, 0].set(jnp.where(comp_bullets, sub_remove_old, e_sub[:, :, 0]))
        e_sub = e_sub.at[:, :, 1].set(jnp.where(e_sub[:, :, 0], e_sub[:, :, 1], 0))
        comp_bullets = jnp.logical_and(comp_bullets, jnp.logical_not(all_e_sub))
        bullets = bullets.at[0, 1:].multiply(comp_bullets)
        bullets = bullets.at[1, 1:].multiply(comp_bullets)
        return bullets, fish, e_sub, reward

    def _pick_up_diver(
        self,
        divers: jax.Array,
        diver_count: jax.Array,
        x: jax.Array,
        y: jax.Array,
        direction: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Pick up one diver if passenger capacity remains."""
        free_space = diver_count < self._MAX_DIVERS
        diver_count = diver_count + jnp.logical_and(
            jnp.logical_and(y > 0, divers[direction, y - 1, x]),
            free_space,
        )
        divers = divers.at[direction, y - 1, x].set(
            jnp.where(jnp.logical_and(free_space, y > 0), 0, divers[direction, y - 1, x]),
        )
        return divers, diver_count

    def _pick_up_divers(
        self,
        divers: jax.Array,
        diver_count: jax.Array,
        x: jax.Array,
        y: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Pick up divers in the original direction priority."""
        first_diver = 1 * (x > 4)
        divers, diver_count = self._pick_up_diver(divers, diver_count, x, y, first_diver)
        divers, diver_count = self._pick_up_diver(divers, diver_count, x, y, 1 - first_diver)
        return divers, diver_count

    def _enemy_hits_player(
        self,
        fish: jax.Array,
        e_sub: jax.Array,
        e_bullets: jax.Array,
        x: jax.Array,
        y: jax.Array,
    ) -> jax.Array:
        """Detect enemy or bullet contact below the surface."""
        return jnp.logical_and(
            y > 0,
            jnp.logical_or(
                jnp.logical_or(jnp.sum(fish[:, :, y - 1, x]), jnp.sum(e_sub[:, :, 0, y - 1, x])),
                jnp.sum(e_bullets[:, y - 1, x]),
            ),
        )
