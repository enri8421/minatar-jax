import jax.numpy as jnp
from jax import jit, grad
import jax
import haiku as hk
import optax 
from flax import struct
from functools import partial
from collections import namedtuple
from alive_progress import alive_bar

from  environments.build_environment import get


@struct.dataclass
class Environment:
    internal_state: struct.dataclass
    obs: jnp.array

@struct.dataclass
class Params:
    value_params: dict
    policy_params: dict
    
    value_opt_state: tuple
    policy_opt_state: tuple

Nets_and_Opts = namedtuple('Nets_and_Opts', 'value_net policy_net optimizer_value optimizer_policy')

@struct.dataclass
class Performance:
    rewards: jnp.array
    finished_game_reward: int
    finished_game: int


def build_value_net(x, config):
    f = hk.Sequential((hk.Conv2D(16,3),
                    jax.nn.relu,
                    hk.Flatten(),
                    hk.nets.MLP([128, 1], activation = jax.nn.relu)))
    return f(x)

def build_policy_net(x, n_action):
    f = hk.Sequential((hk.Conv2D(16,3),
                    jax.nn.relu,
                    hk.Flatten(),
                    hk.nets.MLP([128, n_action], activation = jax.nn.relu)))
    return f(x)
    



def inizialize_nets_and_optimizers(key, obs, config, n_action):

    key_val, key_policy = jax.random.split(key)

    value_net = hk.without_apply_rng(hk.transform(lambda x: build_value_net(x, config)))
    policy_net = hk.without_apply_rng(hk.transform(lambda x: build_policy_net(x, n_action)))

    value_params = value_net.init(key_val, obs)
    policy_params = policy_net.init(key_policy, obs)

    optimizer_value = optax.adam(config.lr_actor)
    value_opt_state = optimizer_value.init(value_params)

    optimizer_policy = optax.chain(optax.clip_by_global_norm(config.max_2_norm_grad),  optax.adam(config.lr_policy))
    policy_opt_state = optimizer_policy.init(policy_params)

    nets_and_opts = Nets_and_Opts(value_net, policy_net, optimizer_value, optimizer_policy)
    params = Params(value_params,policy_params, value_opt_state, policy_opt_state)

    return params, nets_and_opts



@partial(grad, has_aux = True)
def value_loss_grad(value_params, obs,next_obs, reward, done, config, value_net):
    #compute TD(1) value loss from (obs,next_obs,reward,done)

    next_value = jnp.where(done,
                           jnp.zeros((config.n_envs,1)),                                  #if done next value is zero
                           jax.lax.stop_gradient(value_net.apply(value_params, next_obs)))    #otherwise we compute from next obs without taking derivates

    value = value_net.apply(value_params, obs)

    advantage = reward + config.gamma*next_value - value

    return jnp.mean(jnp.square(advantage)), advantage

@grad
def policy_loss_grad(policy_params, obs, action, advantage, config, policy_net):
    #compute actor-crtic policy loss and add an entropy regularizer

    pi = policy_net.apply(policy_params, obs) #compute the de-normalized probability from the actor network

    entropy = -jnp.sum(jax.nn.log_softmax(pi) * jax.nn.softmax(pi), axis=-1)     #compute the entropy of the current actor policy

    pi_log_loss_action = jnp.take_along_axis(jax.nn.log_softmax(pi),jnp.expand_dims(action, axis=1),1)   #take the probability value of the action we have taken
    
    return -jnp.mean(advantage*pi_log_loss_action) - config.beta_entropy*jnp.mean(entropy)


def update_params(params, value_grad, policy_grad, nets_and_opts, config):

    update_value, value_opt_state = nets_and_opts.optimizer_value.update(value_grad, params.value_opt_state, params.value_params)
    value_params = optax.apply_updates(params.value_params, update_value)
    
    update_policy, policy_opt_state = nets_and_opts.optimizer_policy.update(policy_grad, params.policy_opt_state, params.policy_params)
    policy_params = optax.apply_updates(params.policy_params, update_policy)

    params = Params(value_params, policy_params, value_opt_state, policy_opt_state)
    
    return params

def track_performance(performance_tracker, reward, done):
    #keeps track of the rewards across the game, if the game is done it summarize the results in finish_game_rewards and reset the tracker

    rewards = performance_tracker.rewards +  reward.squeeze()     

    finish_game_rewards = performance_tracker.finished_game_reward + jnp.sum(rewards*done.squeeze())

    finished_game =  performance_tracker.finished_game +  jnp.sum(done)

    rewards *= (1-done.squeeze())

    return Performance(rewards, finish_game_rewards, finished_game)




def build_update(key, config):

    key, env_key = jax.random.split(key)
    internal_state, obs, step, n_action = get(config.env_name, key = env_key, n_envs = config.n_envs, version = config.version)

    state = Environment(internal_state, obs)

    params, nets_and_opts = inizialize_nets_and_optimizers(key, obs, config, n_action)

    @jit
    def update(key, params, state, performance_tracker):

        key, step_key = jax.random.split(key)
        #compute next action
        pi = nets_and_opts.policy_net.apply(params.policy_params,state.obs)
        action = jax.random.categorical(step_key, pi)

        #get next obs
        next_internal_state, next_obs, reward, done = step(state.internal_state, action)

        #compute grads
        value_grad, advantage = value_loss_grad(params.value_params, state.obs, next_obs, reward, done, config, nets_and_opts.value_net)
        policy_grad = policy_loss_grad(params.policy_params, state.obs, action, advantage, config, nets_and_opts.policy_net)

        #update
        params = update_params(params, value_grad, policy_grad, nets_and_opts, config)

        #prepare next state
        state = Environment(next_internal_state ,next_obs)

        #compute actual performance to supervise training
        performance_tracker = track_performance(performance_tracker, reward, done)

        return key, params,state, performance_tracker
    
    return update, state, params, nets_and_opts




def train(config):

    key = jax.random.PRNGKey(config.seed)

    key, build_update_key = jax.random.split(key)

    update, state, params, nets_and_opts = build_update(build_update_key, config)

    performance_tracker = Performance(jnp.zeros(config.n_envs), 0, 0)


    performance = []

    with alive_bar(config.max_iter) as bar:
        for i in range(config.max_iter):
                key,params ,state, performance_tracker = \
                                        update(key, params, state, performance_tracker)

                if((i+1) % 1000 == 0):

                    performance.append(performance_tracker.finished_game_reward/performance_tracker.finished_game)

                    if(config.verbose):
                        print("avg per: ",performance_tracker.finished_game_reward/performance_tracker.finished_game)

                    performance_tracker = Performance(performance_tracker.rewards, 0, 0)
            
                bar()


    return lambda x: nets_and_opts.policy_net.apply(params.policy_params, x), performance
