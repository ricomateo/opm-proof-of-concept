"""Per-model deck configurations consumed by the unified pipeline.

Add a new reservoir by creating a sibling module that exposes a
`CONFIG: DeckConfig` symbol, then registering it in `_REGISTRY` below.
"""

from __future__ import annotations

from . import norne, spe9, volve
from .base import DeckConfig

_REGISTRY: dict[str, DeckConfig] = {
    "spe9": spe9.CONFIG,
    "norne": norne.CONFIG,
    "volve": volve.CONFIG,
}


def get_config(name: str) -> DeckConfig:
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_models() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["DeckConfig", "get_config", "available_models"]
