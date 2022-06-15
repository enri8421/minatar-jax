from importlib import import_module
import jax.numpy as jnp
from jax import jit, vmap
import jax
from flax import struct

#this are not final, do not know yet how to put upperbound on reward
max_reward_4_env = {'breakout' : 30, 'freeway' : 70, 'space_invaders' : 2_000, 'asterix' : 500, 'seaquest' : 500, }

@struct.dataclass
class ExpandedState:
    internal_state: any
    tot_rewards: int


def build_sparse_reward_env(env_name, state, obs, step):

    n_channel = obs.shape[-1]
    n_envs = obs.shape[0]
    max_reward = max_reward_4_env[env_name]

    def new_step(state, action):
        internal_state = state.internal_state
        internal_state, obs, reward, done = step(internal_state, action)

        tot_rewards = state.tot_rewards + reward

        out_reward = done*tot_rewards

        state = ExpandedState(internal_state, tot_rewards)

        new_obs = jnp.zeros((10, 10, n_channel + 1))

        new_obs = new_obs.at[:, :, :-1].set(obs)

        display_reward = jnp.minimum(1, tot_rewards[0]/max_reward)

        new_obs = new_obs.at[:, :, -1].set(obs[:, :, 0]*display_reward)

        return state, new_obs, out_reward, done

    
    initial_obs = jnp.zeros((n_envs, 10, 10, n_channel + 1))
    initial_obs = initial_obs.at[:, :, :, :-1].set(obs)
    state = ExpandedState(state, jnp.zeros((n_envs, 1), dtype = int))
    jitnewstep = jit(vmap(new_step, in_axes=0, out_axes = 0))

    return state, initial_obs, jitnewstep
    


def get(env_name, sparse_rewards = False, **args):

    env_factory = import_module("environments."+env_name)

    state, obs, step, n_actions = env_factory.make_env(**args)

    if sparse_rewards:
        step =  env_factory.make_env(only_step = True, **args)
        state, obs, step = build_sparse_reward_env(env_name, state, obs, step)

    return state, obs, step, n_actions





    


