import dataclasses
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from minatar_jax import factory
from minatar_jax.environments import asterix, breakout, freeway, seaquest, space_invaders


class EnvironmentTests(unittest.TestCase):
    def test_reset_is_pure_and_settings_are_frozen(self):
        for name in factory.NAMES:
            with self.subTest(env=name):
                env = factory.make(name)
                settings = vars(env).copy()
                key = jax.random.PRNGKey(7)
                reset = jax.jit(lambda key: env.reset(key, n_envs=3))
                state, observations = reset(key)
                expected = jax.tree_util.tree_map(np.asarray, state)
                actions = jnp.zeros(3, dtype=jnp.int32)
                jax.jit(env.step)(state, actions)
                repeated, repeated_obs = reset(key)
                for before, after in zip(
                    jax.tree_util.tree_leaves(expected),
                    jax.tree_util.tree_leaves(repeated),
                ):
                    np.testing.assert_array_equal(before, after)
                np.testing.assert_array_equal(observations, repeated_obs)
                np.testing.assert_array_equal(observations, jax.jit(env.observe)(state))
                self.assertEqual(vars(env), settings)
                self.assertNotIn("key", vars(env))
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    setattr(env, "num_actions", 0)

    def test_factory_and_reset_reject_invalid_settings(self):
        with self.assertRaisesRegex(ValueError, "Unknown environment"):
            factory.make("missing")
        with self.assertRaisesRegex(ValueError, "n_envs must be positive"):
            factory.make("breakout").reset(jax.random.PRNGKey(0), n_envs=0)

    def test_asterix_collects_gold_and_resets_enemy_collision(self):
        env = asterix.Asterix()
        state, _ = env.reset(jax.random.PRNGKey(0), n_envs=2)
        entities = state.entities.at[:, 4, 0].set(5).at[:, 4, 2].set(jnp.array([1, 0]))
        state = state.replace(entities=entities)
        updated, _, rewards, dones = jax.jit(env.step)(state, jnp.array([4, 4]))
        np.testing.assert_array_equal(rewards, [[1], [0]])
        np.testing.assert_array_equal(dones, [[False], [True]])
        self.assertEqual(int(updated.entities[0, 4, 2]), -1)
        np.testing.assert_array_equal(updated.entities[1, :, 2], -np.ones(8))

    def test_breakout_hit_rewards_and_removes_one_brick(self):
        env = breakout.Breakout()
        state, _ = env.reset(jax.random.PRNGKey(0))
        state = state.replace(
            ball_x=jnp.array([[5]]),
            ball_y=jnp.array([[4]]),
            direction=jnp.array([[0]]),
        )
        updated, _, rewards, dones = jax.jit(env.step)(state, jnp.array([1]))
        self.assertEqual(int(rewards[0, 0]), 1)
        self.assertFalse(bool(dones[0, 0]))
        self.assertEqual(int(updated.bricks[0, 2, 4]), 0)
        self.assertEqual(int(updated.bricks.sum()), 29)

    def test_freeway_time_limit_resets_only_finished_games(self):
        env = freeway.Freeway(time_limit=2)
        state, _ = env.reset(jax.random.PRNGKey(0), n_envs=2)
        state = state.replace(current_frame=jnp.array([[0], [1]]))
        updated, _, _, dones = jax.jit(env.step)(state, jnp.array([1, 1]))
        np.testing.assert_array_equal(dones, [[False], [True]])
        np.testing.assert_array_equal(updated.current_frame, [[1], [0]])
        np.testing.assert_array_equal(updated.pos, [[9], [9]])
        self.assertFalse(np.array_equal(updated.key[1], state.key[1]))

    def test_seaquest_advances_then_resets_on_oxygen_exhaustion(self):
        env = seaquest.Seaquest()
        initial, _ = env.reset(jax.random.PRNGKey(0))
        step = jax.jit(env.step)
        state, observations, _, dones = step(initial, jnp.array([3]))
        self.assertFalse(bool(dones[0, 0]))
        self.assertEqual(int(state.sub_y[0, 0]), 1)
        self.assertEqual(int(state.oxygen[0, 0]), 199)
        self.assertEqual(float(observations[0, 1, 5, 0]), 1.0)
        self.assertFalse(np.array_equal(initial.key, state.key))
        self.assertEqual(int(initial.oxygen[0, 0]), 200)
        state = state.replace(oxygen=jnp.array([[1]]))
        state, _, _, dones = step(state, jnp.array([4]))
        self.assertFalse(bool(dones[0, 0]))
        self.assertEqual(int(state.oxygen[0, 0]), 0)
        state, _, _, dones = step(state, jnp.array([4]))
        self.assertTrue(bool(dones[0, 0]))
        self.assertEqual(int(state.oxygen[0, 0]), 200)
        self.assertEqual(int(state.sub_y[0, 0]), 0)

    def test_space_invaders_waits_for_movement_before_bottom_termination(self):
        env = space_invaders.SpaceInvaders()
        state, _ = env.reset(jax.random.PRNGKey(0), n_envs=2)
        aliens = jnp.zeros_like(state.alien_pos).at[:, 9, 0].set(1)
        state = state.replace(
            alien_pos=aliens,
            alien_dir=jnp.array([[-1], [-1]]),
            alien_move_timer=jnp.array([[10], [0]]),
        )
        updated, _, _, dones = jax.jit(env.step)(state, jnp.array([1, 1]))
        np.testing.assert_array_equal(dones, [[False], [True]])
        self.assertEqual(int(updated.alien_move_timer[0, 0]), 9)
        self.assertEqual(int(updated.alien_pos[0, 9, 0]), 1)

    def test_breakout_frame_limit_counts_completed_steps(self):
        for limit in (1, 3):
            with self.subTest(limit=limit):
                env = breakout.Breakout(max_frames=limit)
                state, _ = env.reset(jax.random.PRNGKey(0))
                step = jax.jit(env.step)
                for index in range(limit):
                    state, _, _, dones = step(state, jnp.array([1]))
                    self.assertEqual(bool(dones[0, 0]), index == limit - 1)
                self.assertEqual(int(state.current_frame[0, 0]), 0)

    def test_breakout_starts_cover_board_without_direction_brick_coupling(self):
        env = breakout.Breakout(block_probability=0.5)
        state, _ = jax.jit(lambda key: env.reset(key, n_envs=512))(jax.random.PRNGKey(0))
        np.testing.assert_array_equal(np.unique(state.ball_x), np.arange(10))
        directions = np.asarray(state.direction).ravel()
        bricks = np.asarray(state.bricks[:, 0, 0])
        for direction in (2, 3):
            with self.subTest(direction=direction):
                np.testing.assert_array_equal(np.unique(bricks[directions == direction]), [0, 1])

    def test_seaquest_enemy_spawn_mix_matches_minatar(self):
        env = seaquest.Seaquest()
        state, _ = env.reset(jax.random.PRNGKey(0))
        fish, submarines = state.fish[0], state.e_sub[0]

        def spawns_submarine(key: jax.Array) -> jax.Array:
            _, subs, _, _ = env._spawn_enemy(
                key,
                fish,
                submarines,
                jnp.array([0]),
                jnp.array([0]),
                jnp.array([20]),
                jnp.array([5]),
            )
            return jnp.any(subs)

        keys = jax.random.split(jax.random.PRNGKey(11), 4096)
        fraction = float(jax.jit(jax.vmap(spawns_submarine))(keys).mean())
        self.assertAlmostEqual(fraction, 1 / 3, delta=0.05)

    def test_seaquest_removes_occupied_slot_even_with_large_timer(self):
        env = seaquest.Seaquest()
        cleared = env._clear_oldest(jnp.array([0, 18, 0]))
        np.testing.assert_array_equal(cleared, [0, 0, 0])
        cleared = env._clear_oldest(jnp.array([8, 18, 0]))
        np.testing.assert_array_equal(cleared, [0, 18, 0])

    def test_space_invaders_respects_weapon_cooldown(self):
        env = space_invaders.SpaceInvaders(shot_cool_down=5)
        state, _ = env.reset(jax.random.PRNGKey(0))
        step = jax.jit(env.step)
        state, _, _, _ = step(state, jnp.array([3]))
        self.assertEqual(int(state.shot_timer[0, 0]), 4)
        self.assertEqual(int(state.f_bullet[0, 8, 5]), 1)
        state, _, _, _ = step(state, jnp.array([3]))
        self.assertEqual(int(state.shot_timer[0, 0]), 3)
        self.assertEqual(int(state.f_bullet[0, 7, 5]), 1)
        self.assertEqual(int(state.f_bullet[0].sum()), 1)


if __name__ == "__main__":
    unittest.main()
