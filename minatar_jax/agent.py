"""Small convolutional TD(0) actor–critic using Flax NNX and Optax."""

import jax
import jax.numpy as jnp
import optax
from flax import nnx


class Network(nnx.Module):
    """SAME convolutions, ReLU hidden layers, and a linear output."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        convolutions: tuple[tuple[int, int], ...],
        hidden: tuple[int, ...],
        outputs: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        channels = shape[-1]
        self.convolutions = nnx.List([])
        for features, kernel in convolutions:
            convolution = nnx.Conv(
                channels,
                features,
                (kernel, kernel),
                padding="SAME",
                rngs=rngs,
            )
            self.convolutions.append(convolution)
            channels = features

        features = shape[0] * shape[1] * channels

        self.hidden = nnx.List([])
        for width in hidden:
            self.hidden.append(nnx.Linear(features, width, rngs=rngs))
            features = width

        self.output = nnx.Linear(features, outputs, rngs=rngs)

    def __call__(self, observations: jax.Array) -> jax.Array:
        x = observations
        for convolution in self.convolutions:
            x = jax.nn.relu(convolution(x))

        x = x.reshape(x.shape[0], -1)

        for layer in self.hidden:
            x = jax.nn.relu(layer(x))

        return self.output(x)


class Agent(nnx.Module):
    """Separate actor/critic, one-step targets, and entropy regularization."""

    def __init__(
        self,
        key: jax.Array,
        shape: tuple[int, int, int],
        actions: int,
        *,
        lr_policy: float,
        lr_value: float,
        gamma: float,
        beta: float,
    ) -> None:
        value_key, policy_key = jax.random.split(key)
        self.policy = Network(
            shape,
            ((16, 3),),
            (128,),
            actions,
            rngs=nnx.Rngs(policy_key),
        )
        self.value = Network(
            shape,
            ((32, 5), (16, 3)),
            (256, 128),
            1,
            rngs=nnx.Rngs(value_key),
        )

        self.policy_optimizer = nnx.Optimizer(self.policy, optax.adam(lr_policy), wrt=nnx.Param)
        self.value_optimizer = nnx.Optimizer(self.value, optax.adam(lr_value), wrt=nnx.Param)
        self.gamma, self.beta = gamma, beta

    def actions(
        self,
        observations: jax.Array,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Sample during training; choose argmax when no key is supplied."""
        logits = self.policy(observations)
        return logits.argmax(axis=-1) if key is None else jax.random.categorical(key, logits)

    def targets(
        self,
        observations: jax.Array,
        rewards: jax.Array,
        dones: jax.Array,
    ) -> jax.Array:
        """Terminal transitions do not bootstrap from auto-reset observations."""
        values = self.value(observations)[:, 0]
        targets = rewards[:, 0] + self.gamma * jnp.where(dones[:, 0], 0.0, values)
        return jax.lax.stop_gradient(targets)

    def update(
        self,
        observations: jax.Array,
        actions: jax.Array,
        next_observations: jax.Array,
        rewards: jax.Array,
        dones: jax.Array,
    ) -> jax.Array:
        """Apply one critic update and one entropy-regularized policy update."""
        targets = self.targets(next_observations, rewards, dones)

        def value_loss(model: Network) -> tuple[jax.Array, jax.Array]:
            values = model(observations)[:, 0]
            advantage = targets - values
            return jnp.mean(advantage**2), advantage

        losses = nnx.value_and_grad(value_loss, has_aux=True)(self.value)
        (critic_loss, advantage), value_gradients = losses
        advantage = jax.lax.stop_gradient(advantage)

        def policy_loss(model: Network) -> jax.Array:
            logits = model(observations)
            log_probs = jax.nn.log_softmax(logits)
            selected = jnp.take_along_axis(log_probs, actions[:, None], axis=-1)[:, 0]
            entropy = -(jax.nn.softmax(logits) * log_probs).sum(axis=-1)
            return -(selected * advantage).mean() - self.beta * entropy.mean()

        actor_loss, policy_gradients = nnx.value_and_grad(policy_loss)(self.policy)

        self.policy_optimizer.update(self.policy, policy_gradients)
        self.value_optimizer.update(self.value, value_gradients)
        return jnp.stack((actor_loss, critic_loss))
