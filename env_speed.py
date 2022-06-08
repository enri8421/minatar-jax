import jax.numpy as jnp
import jax
from alive_progress import alive_bar
from absl import flags
import sys
from  environments.build_environment import get

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', "breakout", 'environment')
FLAGS(sys.argv)


N = 100
key = jax.random.PRNGKey(0)

state, obs, step, n_action = get(FLAGS.env_name, key = key, n_envs = N)


@jax.jit
def make_random_iteration(key, state, obs):
    key, step_key = jax.random.split(key)
    action = jax.random.randint(step_key, (N,), 0, n_action)
    state, obs, _, _ = step(state ,action)
    return key, state, obs




print("running a random agent in 100 parallel environments for 100k iteration:")
with alive_bar(100_000) as bar:
    for i in range(100_000):
        key, state, obs = make_random_iteration(key, state, obs)
        bar()


