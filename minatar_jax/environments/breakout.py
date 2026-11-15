"""Breakout: bounce a ball into three rows of bricks.

Actions: left, stay, right. Channels: paddle, ball, trail, bricks.
The ball starts at a random horizontal position and reflects at brick corners.
Each broken brick rewards one point. Missing the ball, clearing the board, or a frame limit ends play.
"""

import dataclasses
import typing

import jax
import jax.numpy as jnp
from flax import struct

from minatar_jax.environments import base


class BreakoutState(struct.PyTreeNode):
    """Unbatched game data; scalar fields have shape (1,)."""

    ball_x: jax.Array
    ball_y: jax.Array
    trail_x: jax.Array
    trail_y: jax.Array
    pad: jax.Array
    direction: jax.Array
    bricks: jax.Array  # (3, 10): remaining bricks.
    key: jax.Array  # PRNG key for the next random transition.
    current_frame: jax.Array


@dataclasses.dataclass(frozen=True, kw_only=True)
class Breakout(base.Environment[BreakoutState]):
    """Immutable settings for Breakout."""

    block_probability: float = 1.0
    max_frames: int | None = None

    observation_shape: typing.ClassVar[tuple[int, int, int]] = (10, 10, 4)
    num_actions: typing.ClassVar[int] = 3

    def _reset(self, key: jax.Array) -> BreakoutState:
        """Create one initial game state from an explicit key."""
        key, bricks_key, direction_key, position_key = jax.random.split(key, 4)
        if self.block_probability == 1:
            bricks = jnp.ones((3, 10), dtype=int)
        else:
            bricks = 1 * jax.random.bernoulli(bricks_key, self.block_probability, (3, 10))
        initial_direction = jnp.array([jax.random.bernoulli(direction_key) + 2])
        initial_pos = jax.random.randint(position_key, (1,), 0, 10)
        ball_y_initial = jnp.array([4])
        return BreakoutState(
            ball_x=initial_pos,
            ball_y=ball_y_initial,
            trail_x=initial_pos,
            trail_y=jnp.array([3]),
            pad=jnp.array([4]),
            direction=initial_direction,
            bricks=bricks,
            key=key,
            current_frame=jnp.array([0]),
        )

    def _step(
        self,
        state: BreakoutState,
        action: jax.Array,
    ) -> tuple[BreakoutState, jax.Array, jax.Array]:
        """Advance one game; automatic reset is handled by the base class."""
        pad = state.pad + (action == 2) * (state.pad < 9) - (action == 0) * (state.pad > 0)
        trail_x = state.ball_x
        trail_y = state.ball_y
        ball_x = (
            state.ball_x
            + (state.direction == 1)
            + (state.direction == 2)
            - (state.direction == 0)
            - (state.direction == 3)
        )
        ball_y = (
            state.ball_y
            + (state.direction == 2)
            + (state.direction == 3)
            - (state.direction == 0)
            - (state.direction == 1)
        )

        # Reflect at the walls and bricks.
        overflow_x = jnp.logical_or(ball_x < 0, ball_x > 9)
        direction = state.direction + overflow_x * (
            (state.direction % 2 == 0) + (-1) * (state.direction % 2 == 1)
        )
        direction = direction + (3 - 2 * direction) * (ball_y < 0)
        ball_x = jnp.maximum(jnp.minimum(ball_x, 9), 0)
        ball_y = jnp.maximum(jnp.minimum(ball_y, 9), 0)
        hit_brick = state.bricks.at[(ball_y - 1) % 10, ball_x].get(mode="fill", fill_value=0)
        bricks = state.bricks.at[(ball_y - 1) % 10, ball_x].add(-hit_brick.squeeze())
        direction = direction + (3 - 2 * direction) * hit_brick
        ball_y = trail_y * hit_brick + ball_y * (1 - hit_brick)
        still_hit_brick = state.bricks.at[(ball_y - 1) % 10, ball_x].get(mode="fill", fill_value=0)
        direction = direction + still_hit_brick * (
            (direction % 2 == 0) + (-1) * (direction % 2 == 1)
        )
        ball_x = trail_x * still_hit_brick + ball_x * (1 - still_hit_brick)

        # Either contact point can bounce the ball off the paddle.
        last_line = ball_y == 9
        ball_y = trail_y * last_line + ball_y * (1 - last_line)
        trail_hit = last_line * (trail_x == pad)
        ball_hit = last_line * (ball_x == pad) * (1 - trail_hit)
        direction = direction + (3 - 2 * direction) * trail_hit
        direction = (direction + 2 * ball_hit) % 4

        # A miss, cleared board, or configured frame limit ends the episode.
        fail = jnp.logical_and(last_line, jnp.logical_not(jnp.logical_or(ball_hit, trail_hit)))
        limit_reached = (
            state.current_frame + 1 >= self.max_frames if self.max_frames is not None else False
        )
        done = jnp.logical_or(jnp.logical_or(fail, jnp.sum(bricks) == 0), limit_reached)
        state = BreakoutState(
            ball_x=ball_x,
            ball_y=ball_y,
            trail_x=trail_x,
            trail_y=trail_y,
            pad=pad,
            direction=direction,
            bricks=bricks,
            key=state.key,
            current_frame=state.current_frame + 1,
        )
        return state, hit_brick, done

    def _observe(self, state: BreakoutState) -> jax.Array:
        """Encode state as a (10, 10, 4) observation."""
        obs = jnp.zeros((10, 10, 4))
        obs = obs.at[9, state.pad, 0].set(1)
        obs = obs.at[state.ball_y, state.ball_x, 1].set(1)
        obs = obs.at[state.trail_y, state.trail_x, 2].set(1)
        obs = obs.at[1:4, :, 3].set(state.bricks)
        return obs
