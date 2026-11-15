import pathlib
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import PIL.GifImagePlugin
import PIL.Image
from flax import nnx

import demo
import minatar_jax as minatar


class SmokeTests(unittest.TestCase):
    def test_compiled_environment_steps(self):
        for name in minatar.NAMES:
            with self.subTest(env=name):
                env = minatar.make(name)
                state, observations = env.reset(jax.random.PRNGKey(0), n_envs=2)
                step = jax.jit(env.step)
                for index in range(4):
                    state, observations, rewards, dones = step(
                        state,
                        jnp.full((2,), index % env.num_actions, dtype=jnp.int32),
                    )
                    self.assertEqual(observations.shape, (2, *env.observation_shape))
                    self.assertEqual(rewards.shape, (2, 1))
                    self.assertEqual(dones.dtype, jnp.bool_)
                    self.assertEqual(rewards.dtype, jnp.int32)
                    self.assertTrue(np.isfinite(np.asarray(observations)).all())

    def initialize(self, name="breakout", lr_value=0.001):
        env = minatar.make(name)
        key, environment_key, network_key = jax.random.split(jax.random.PRNGKey(0), 3)
        policy = minatar.Agent(
            network_key,
            env.observation_shape,
            env.num_actions,
            lr_policy=0.003,
            lr_value=lr_value,
            gamma=0.99,
            beta=0.04,
        )
        state, observations = env.reset(environment_key, n_envs=2)
        return env, policy, (state, observations, key)

    def test_compiled_training_is_repeatable_and_updates_parameters(self):
        for name in minatar.NAMES:
            with self.subTest(env=name):
                env, policy, initial = self.initialize(name)
                first_policy, repeated_policy = nnx.clone(policy), nnx.clone(policy)
                first = demo.learn(first_policy, *initial, env=env, updates=2)
                repeated = demo.learn(repeated_policy, *initial, env=env, updates=2)
                for left, right in zip(
                    jax.tree.leaves(nnx.state(first_policy)),
                    jax.tree.leaves(nnx.state(repeated_policy)),
                ):
                    np.testing.assert_array_equal(left, right)
                    self.assertTrue(np.isfinite(np.asarray(left)).all())
                self.assertEqual(int(policy.policy_optimizer.step[...]), 0)
                self.assertEqual(int(first_policy.policy_optimizer.step[...]), 2)
                for left, right in zip(jax.tree.leaves(first), jax.tree.leaves(repeated)):
                    np.testing.assert_array_equal(left, right)
                self.assertTrue(
                    any(
                        not np.array_equal(before, after)
                        for before, after in zip(
                            jax.tree.leaves(nnx.state(policy.policy)),
                            jax.tree.leaves(nnx.state(first_policy.policy)),
                        )
                    ),
                )

    def test_terminal_targets_and_detached_bootstrap(self):
        _, policy, (_, observations, _) = self.initialize()
        rewards = jnp.array([[1], [2]])
        dones = jnp.array([[True], [False]])
        values = policy.value(observations)[:, 0]
        actual = policy.targets(observations, rewards, dones)
        np.testing.assert_allclose(actual, [1, 2 + 0.99 * float(values[1])], rtol=1e-6)
        gradients = nnx.grad(lambda p: p.targets(observations, rewards, dones).sum())(policy)
        self.assertTrue(all(np.all(np.asarray(value) == 0) for value in jax.tree.leaves(gradients)))

    def test_critic_learning_rate_is_independent(self):
        env, left_policy, left = self.initialize(lr_value=0.001)
        _, right_policy, right = self.initialize(lr_value=0.01)
        demo.learn(left_policy, *left, env=env, updates=1)
        demo.learn(right_policy, *right, env=env, updates=1)
        for a, b in zip(
            jax.tree.leaves(nnx.state(left_policy.policy)),
            jax.tree.leaves(nnx.state(right_policy.policy)),
        ):
            np.testing.assert_array_equal(a, b)
        self.assertTrue(
            any(
                not np.array_equal(a, b)
                for a, b in zip(
                    jax.tree.leaves(nnx.state(left_policy.value)),
                    jax.tree.leaves(nnx.state(right_policy.value)),
                )
            ),
        )

    def test_demo_exports_only_a_repeatable_animated_gif(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            cfg = demo.Config(n_envs=2, updates=2, max_steps=30, output=output)
            path = demo.demo(cfg)
            expected = path.read_bytes()
            self.assertEqual({p.name for p in output.iterdir()}, {"breakout.gif"})
            with PIL.Image.open(path) as image:
                assert isinstance(image, PIL.GifImagePlugin.GifImageFile)
                self.assertGreater(image.n_frames, 1)
                self.assertEqual(image.size, (220, 252))
            demo.demo(cfg)
            self.assertEqual(path.read_bytes(), expected)


if __name__ == "__main__":
    unittest.main()
