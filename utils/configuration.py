import yaml
from collections import namedtuple

class default(object):

    def __init__(self, env_name):
        with open("utils/a2c_hp.yaml") as file:
            documents = yaml.full_load(file)
            hp = documents[env_name]
        self.env_name = env_name
        self.lr_policy = hp["lr_policy"]
        self.lr_value = hp["lr_value"]
        self.beta = hp["beta"]



    n_envs = 64
    version = 1

    #architecture_config = (convolutional part: (n_output, kernel_size) ..., feedforward part: (n_neuron))
    #remark: last layer is added based on n_action
    policy_architecture_config = ([(16,3)], [128])
    value_architecture_config = ([(32,5),(16,3)], [256, 128])


    gamma = 0.99

    max_iter = 200_000

    T = 1

