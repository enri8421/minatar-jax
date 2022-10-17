import sys
from absl import flags
import os
import jax.numpy as jnp
from jax import jit
import jax

from utils.configuration import default
from utils.train import Trainer
import time


"""
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
#os.environ['TF_CUDNN_DETERMINISTIC']='1'


# Reads the flags from the command line arguments
def get_flags(config):
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('seed', config.seed, 'seed')

    flags.DEFINE_float('lr_policy', config.lr_policy, 'Learning Rate Policy')
    flags.DEFINE_float('lr_actor', config.lr_actor, 'Learning Rate Actor')

    flags.DEFINE_integer('n_envs', config.n_envs, 'concurrent number of enviroments')
    flags.DEFINE_integer('version', config.version, 'version of the env')

    flags.DEFINE_float('gamma', config.gamma, 'discount factor')
    flags.DEFINE_integer('max_iter', config.max_iter, 'number of iteration')

    flags.DEFINE_float('beta_entropy', config.beta_entropy, 'regularize coefficient for entropy loss')

    flags.DEFINE_float('max_2_norm_grad', config.max_2_norm_grad, 'maximum norm fro policy loss grad' )

    flags.DEFINE_bool('verbose', config.verbose, 'plot performance during training' )
    flags.DEFINE_bool('sparse_rewards', config.sparse_rewards, 'incorporate the rewards in the obs' )
    flags.DEFINE_bool('markov', config.markov, 'use complete observation' )

    flags.DEFINE_string('env_name', config.env_name, 'environment')

    FLAGS(sys.argv)

    return FLAGS

# Write the flags from FLAGS into variables in config
def flags2config(FLAGS, config):

    config.seed = FLAGS.seed
    config.lr_policy = FLAGS.lr_policy
    config.lr_actor = FLAGS.lr_actor
    config.n_envs = FLAGS.n_envs
    config.version = FLAGS.version
    config.sparse_rewards = FLAGS.sparse_rewards
    config.gamma = FLAGS.gamma
    config.markov = FLAGS.markov
    config.max_iter = FLAGS.max_iter
    config.beta_entropy = FLAGS.beta_entropy
    config.max_2_norm_grad = FLAGS.max_2_norm_grad
    config.verbose = FLAGS.verbose
    config.env_name = FLAGS.env_name

    return config

config = default()
FLAGS = get_flags(config)
config = flags2config(FLAGS, config)



policy, performance = train(config)


print("testing ...")

N = 1_000
seed = config.seed
key = jax.random.PRNGKey(seed)

state, obs, step, n_action = get(config.env_name, sparse_rewards = config.sparse_rewards, key = key, n_envs = N, version = config.version, markov = config.markov)
rewards = jnp.zeros(N)
dones = jnp.zeros(N)

@jit
def iteration(key,state,obs,rewards,dones):
  pi = policy(obs)
  key,sample_key = jax.random.split(key)
  action = jax.random.categorical(sample_key, pi)
  state, obs, reward, done = step(state ,action)
  rewards += reward.squeeze()*(1-dones)
  dones = jnp.logical_or(done.squeeze(), dones)
  return key, state, obs, rewards, dones


while(not (dones > 0).all()):
  key,state,obs,rewards,dones = iteration(key,state,obs,rewards,dones)

print(f"avg performance after {config.max_iter*config.n_envs} frames in {config.env_name} is {jnp.mean(rewards)}")
"""

key = jax.random.PRNGKey(0)
env_name = "seaquest"

config = default(env_name)

trainer = Trainer(key, config)

print("start training...")
s = time.time()

trainer.train()

print("train finished in ", time.time() - s, "s")

print("start testing...")


score = trainer.test()



print(f"avg performance after {config.max_iter*config.n_envs} frames in {config.env_name} is {score}")