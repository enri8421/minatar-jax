import jax
import jax.numpy as jnp
import haiku as hk
import optax
from flax import struct
from functools import partial



class ConvNN(hk.Module):
    def __init__(self, architecture_config, n_outputs, name = None):
        super().__init__(name)

        convolution_config, feedforward_config = architecture_config

        model = []
        for out_size, kernel_size in convolution_config:
            model.append(hk.Conv2D(out_size, kernel_size))
            model.append(jax.nn.relu)

        model.append(hk.Flatten())
        model.append(hk.nets.MLP(feedforward_config + [n_outputs], activation = jax.nn.relu))

        self.model = hk.Sequential(model)

    def __call__(self, x):
        return self.model(x)

@struct.dataclass
class Params:
    value_params: dict
    policy_params: dict

    value_opt_state: tuple
    policy_opt_state: tuple


class A2C_Agent():
    def __init__(self, key, input_shape, n_action, policy_sizes, value_sizes, lr_policy, lr_value, gamma, beta):
        key_val, key_policy = jax.random.split(key)
        sample_obs = jnp.zeros((1,) + input_shape)
        self.gamma = gamma
        self.beta = beta
        self.value_net = hk.without_apply_rng(hk.transform(lambda x: ConvNN(value_sizes, 1)(x)))
        self.policy_net = hk.without_apply_rng(hk.transform(lambda x: ConvNN(policy_sizes, n_action)(x)))


        value_params = self.value_net.init(key_val, sample_obs)
        policy_params = self.policy_net.init(key_policy, sample_obs)

        self.optimizer_value = optax.adam(lr_value)
        value_opt_state = self.optimizer_value.init(value_params)
        self.optimizer_policy = optax.adam(lr_policy)
        policy_opt_state = self.optimizer_policy.init(policy_params)

        self.init_params = Params(value_params, policy_params, value_opt_state, policy_opt_state)

    def get_init_params(self):
        return self.init_params

    @partial(jax.grad, argnums = 1, has_aux = True)
    def value_loss_grad(self, value_params, transition):
        next_value_no_done = jax.lax.stop_gradient(self.value_net.apply(value_params, transition.next_obs))
        next_value = jnp.where(transition.done,
                               jnp.zeros_like(transition.reward),    #if done next value is zero
                               next_value_no_done)    #otherwise we compute from next obs without taking derivates

        value = self.value_net.apply(value_params, transition.obs)

        advantage = transition.reward + self.gamma*next_value - value

        return jnp.mean(jnp.square(advantage)), advantage

    @partial(jax.grad, argnums = 1, has_aux = False)
    def policy_loss_grad(self, policy_params, transition, advantage):
        pi = self.policy_net.apply(policy_params, transition.obs) #compute the de-normalized probability from the actor network
        entropy = -jnp.sum(jax.nn.log_softmax(pi) * jax.nn.softmax(pi), axis=-1)     #compute the entropy of the current actor policy
        pi_log_loss_action = jnp.take_along_axis(jax.nn.log_softmax(pi),jnp.expand_dims(transition.action, axis=1),1)   #take the probability value of the action we have taken
        return -jnp.mean(advantage*pi_log_loss_action) - self.beta*jnp.mean(entropy)

    def update_params(self, params, value_grad, policy_grad):
        update_value, value_opt_state = self.optimizer_value.update(value_grad, params.value_opt_state, params.value_params)
        value_params = optax.apply_updates(params.value_params, update_value)
        update_policy, policy_opt_state = self.optimizer_policy.update(policy_grad, params.policy_opt_state, params.policy_params)
        policy_params = optax.apply_updates(params.policy_params, update_policy)
        params = Params(value_params, policy_params, value_opt_state, policy_opt_state)
        return params

    def learning_step(self, params, transition):
        value_grad, advantage = self.value_loss_grad(params.value_params, transition)
        policy_grad = self.policy_loss_grad(params.policy_params, transition, advantage)
        params = self.update_params(params, value_grad, policy_grad)
        return params
    
    def sample_action(self, key, params, obs, T = 1):
        pi = T*self.policy_net.apply(params.policy_params, obs)
        action = jax.random.categorical(key, pi)
        return action




