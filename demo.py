"""Train a compiled MinAtar actor–critic and save one episode as a GIF."""

import pathlib
import time
import typing

import jax
import jax.numpy as jnp
import numpy as np
import PIL.Image
import PIL.ImageDraw
import pydantic
import pydantic_settings
from flax import nnx

import minatar_jax as minatar


class Config(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(cli_kebab_case=True)

    env: minatar.Name = "breakout"
    seed: int = pydantic.Field(default=0, ge=0, le=2**32 - 1)
    n_envs: pydantic.PositiveInt = 64
    updates: pydantic.PositiveInt = 5000
    lr_policy: pydantic.PositiveFloat = 0.003
    lr_value: pydantic.PositiveFloat = 0.001
    beta: pydantic.NonNegativeFloat = 0.04
    max_steps: pydantic.PositiveInt = 5000
    output: pathlib.Path = pathlib.Path("results")


@nnx.jit(static_argnames=("env", "updates"))
def learn(
    policy: minatar.Agent,
    state: minatar.State,
    observations: jax.Array,
    key: jax.Array,
    *,
    env: minatar.Environment[typing.Any],
    updates: int,
) -> tuple[minatar.State, jax.Array, jax.Array]:
    """Compile sampling, game dynamics, gradients, and Adam into one loop."""

    def step(_: int, carry: tuple) -> tuple:
        policy, state, observations, key = carry
        key, action_key = jax.random.split(key)
        actions = policy.actions(observations, action_key)
        state, next_observations, rewards, dones = env.step(state, actions)
        policy.update(observations, actions, next_observations, rewards, dones)
        return policy, state, next_observations, key

    _, state, observations, key = nnx.fori_loop(
        0,
        updates,
        step,
        (policy, state, observations, key),
    )
    return state, observations, key


def frame(observation: np.ndarray, name: str, score: float) -> PIL.Image.Image:
    """Render object channels on a small grid, with the player on top."""
    colors = np.array(
        [
            [85, 220, 170],
            [100, 160, 220],
            [255, 110, 110],
            [244, 196, 90],
            [230, 140, 70],
            [170, 120, 230],
            [230, 100, 180],
            [80, 180, 240],
            [240, 220, 90],
            [120, 220, 220],
        ],
        dtype=np.uint8,
    )
    pixels = np.full((10, 10, 3), [22, 27, 34], dtype=np.uint8)
    for channel in reversed(range(observation.shape[-1])):
        pixels[observation[:, :, channel] > 0] = colors[channel]
    board = PIL.Image.fromarray(pixels).resize((220, 220), PIL.Image.Resampling.NEAREST)
    image = PIL.Image.new("RGB", (220, 252), (22, 27, 34))
    image.paste(board, (0, 32))
    PIL.ImageDraw.Draw(image).text((7, 7), f"{name}  |  score {score:g}", fill=(235, 240, 245))
    return image.convert("P", palette=PIL.Image.Palette.ADAPTIVE)


def record(
    env: minatar.Environment[typing.Any],
    policy: minatar.Agent,
    name: str,
    output: pathlib.Path,
    max_steps: int,
) -> tuple[float, int]:
    """Record seed 1000, sampling the display every four game steps."""
    state, observations = env.reset(jax.random.PRNGKey(1000))
    step = nnx.jit(lambda policy, state, obs: env.step(state, policy.actions(obs)))
    images, score = [], 0.0
    length = 0
    for index in range(max_steps):
        length += 1
        if index % 4 == 0:
            images.append(frame(np.asarray(observations[0]), name, score))
        state, observations, rewards, dones = step(policy, state, observations)
        score += float(rewards[0, 0])
        if bool(dones[0, 0]):
            break
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=80,
        loop=0,
        optimize=False,
    )
    return score, length


def demo(cfg: Config) -> pathlib.Path:
    """Train from a fixed seed, then record greedy actions with seed 1000."""
    env = minatar.make(cfg.env)
    key, environment_key, network_key = jax.random.split(jax.random.PRNGKey(cfg.seed), 3)
    policy = minatar.Agent(
        network_key,
        env.observation_shape,
        env.num_actions,
        lr_policy=cfg.lr_policy,
        lr_value=cfg.lr_value,
        gamma=0.99,
        beta=cfg.beta,
    )
    state, observations = env.reset(environment_key, n_envs=cfg.n_envs)
    start = time.perf_counter()
    jax.block_until_ready(learn(policy, state, observations, key, env=env, updates=cfg.updates))
    seconds = time.perf_counter() - start
    cfg.output.mkdir(parents=True, exist_ok=True)
    output = cfg.output / f"{cfg.env}.gif"
    score, length = record(env, policy, cfg.env, output, cfg.max_steps)
    print(
        f"{cfg.env}: trained in {seconds:.1f}s (including compilation); episode {score:g} points in {length} steps → {output}",
    )
    return output


if __name__ == "__main__":
    demo(pydantic_settings.CliApp.run(Config))
