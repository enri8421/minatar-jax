import jax
import jax.numpy as jnp
import haiku as hk
import optax
from flax import struct
from functools import partial
from collections import namedtuple
from alive_progress import alive_bar

from utils.agent import A2C_Agent
from  environments.build_environment import get






@struct.dataclass
class Transition:
    obs: jnp.array
    action: jnp.array
    next_obs: jnp.array
    reward: jnp.array
    done: jnp.array

@struct.dataclass
class State:
    internal_state: struct.dataclass
    obs: jnp.array

@struct.dataclass
class Performance:
    rewards: jnp.array
    dones: jnp.array

class Trainer:

    def __init__(self, key, config):
        self.N = 1_000
        key, train_env_key, test_env_key, agent_key = jax.random.split(key, 4)
        self.key = key

        self.train_env = get(config.env_name, key = train_env_key, n_envs = config.n_envs, version = config.version)
        self.test_env = get(config.env_name, key = test_env_key, n_envs = self.N, version = config.version)

        self.agent = A2C_Agent(agent_key, 
                               self.train_env.obs_shape, 
                               self.train_env.n_actions, 
                               config.policy_architecture_config, 
                               config.value_architecture_config, 
                               config.lr_policy, 
                               config.lr_policy, 
                               config.gamma, 
                               config.beta)
        
        self.params = self.agent.get_init_params()
        self.state = State(*self.train_env.reset())
        self.max_iter = config.max_iter
        self.T = config.T

    
    def make_train_env_step(self, state, action):
        next_internal_state, next_obs, reward, done = self.train_env.step(state.internal_state, action)
        transition = Transition(state.obs, action, next_obs, reward, done)
        state = State(next_internal_state, next_obs)
        return state, transition

    def make_test_env_step(self, state, action):
        next_internal_state, next_obs, reward, done = self.test_env.step(state.internal_state, action)
        transition = Transition(state.obs, action, next_obs, reward, done)
        state = State(next_internal_state, next_obs)
        return state, transition

    def update_step(self, key, params, state):
        #sample new action using the current parameters
        key, step_key = jax.random.split(key)
        action = self.agent.sample_action(step_key, params, state.obs)

        #make a step in the environment and get transitions
        state, transition = self.make_train_env_step(state, action)

        #update parameters using single TD1 information
        params = self.agent.learning_step(params, transition)

        return key, params, state


    @partial(jax.jit, static_argnums=0)
    def jit_train(self, key, params, state):
        return jax.lax.fori_loop(0, self.max_iter, lambda _,x: self.update_step(*x), (key, params, state))

    def train(self):
        self.key, self.params, self.state = self.jit_train(self.key, self.params, self.state)

    def compute_performance(self, perfromance, transition):
        new_rewards = perfromance.rewards + transition.reward.squeeze()*(1 - perfromance.dones)
        new_dones = jnp.logical_or(transition.done.squeeze(), perfromance.dones)
        return Performance(new_rewards, new_dones)


    def test_step(self, params, key, state, perfromance):
        key, step_key = jax.random.split(key)
        action = self.agent.sample_action(step_key, params, state.obs, T = self.T)
        state, transition = self.make_test_env_step(state, action)
        perfromance = self.compute_performance(perfromance, transition)
        return key, state, perfromance


    def cond(self, val):
        return jnp.sum(val[-1].dones) < self.N

    partial(jax.jit, static_argnums=0)
    def test_jit(self, key, params, state):
        perfromance = Performance(jnp.zeros(self.N), jnp.full(self.N, False))
        key, _, perfromance = jax.lax.while_loop(self.cond, lambda val: self.test_step(params, *val), (key, state, perfromance))
        return key, jnp.mean(perfromance.rewards)


        
    def test(self):
        initial_state = State(*self.test_env.reset())
        self.key, score = self.test_jit(self.key, self.params, initial_state)
        return score

    


    