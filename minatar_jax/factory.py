"""Explicit constructors for the supported games."""

import typing

from minatar_jax.environments import asterix, base, breakout, freeway, seaquest, space_invaders

type Name = typing.Literal["asterix", "breakout", "freeway", "seaquest", "space_invaders"]

_ENVIRONMENTS: dict[str, type[base.Environment[typing.Any]]] = {
    "asterix": asterix.Asterix,
    "breakout": breakout.Breakout,
    "freeway": freeway.Freeway,
    "seaquest": seaquest.Seaquest,
    "space_invaders": space_invaders.SpaceInvaders,
}
NAMES = tuple(_ENVIRONMENTS)


def make(name: str) -> base.Environment[typing.Any]:
    """Create immutable game settings; pass random keys to reset explicitly."""
    constructor = _ENVIRONMENTS.get(name)
    if constructor is None:
        raise ValueError(f"Unknown environment {name!r}; choose from {', '.join(NAMES)}")
    return constructor()
