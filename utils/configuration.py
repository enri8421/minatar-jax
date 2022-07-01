from absl import flags

class default(object):

    env_name = "breakout"
    version = 0
    sparse_rewards = False
    markov = False

    seed = 423

    lr_actor = 0.001
    lr_policy = 0.003
    n_envs = 64
    gamma = 0.99
    beta_entropy = 0.04
    max_2_norm_grad = 0.5
    
    
    
    
    max_iter = 50_000
    verbose = False