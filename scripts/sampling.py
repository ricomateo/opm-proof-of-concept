"""Generic Latin Hypercube sampling.

Takes a `lever_ranges` dict (from a `DeckConfig`) and an optional
validator callable; returns N samples where each is a dict mapping lever
name to a value drawn from its range. Validator-rejected samples are
regenerated with a fresh stratum draw, up to 50 attempts each.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def sample_lhs(
    n: int,
    lever_ranges: dict[str, tuple[float, float]],
    validator: Optional[Callable[[dict], bool]] = None,
    seed: int = 42,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    lever_names = list(lever_ranges)
    dim = len(lever_names)

    raw = _lhs_unit(n, dim=dim, rng=rng)

    out: list[dict] = []
    for i in range(n):
        sample = _materialize(raw[i], lever_names, lever_ranges)
        attempts = 0
        while validator is not None and not validator(sample) and attempts < 50:
            attempts += 1
            sample = _materialize(rng.random(dim), lever_names, lever_ranges)
        out.append(sample)
    return out


def _lhs_unit(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    cuts = np.linspace(0.0, 1.0, n + 1)
    u = rng.random((n, dim))
    a = cuts[:n]
    b = cuts[1 : n + 1]
    points = u * (b - a)[:, None] + a[:, None]
    for j in range(dim):
        rng.shuffle(points[:, j])
    return points


def _materialize(unit_row, lever_names, lever_ranges) -> dict:
    out: dict = {}
    for j, name in enumerate(lever_names):
        lo, hi = lever_ranges[name]
        out[name] = float(lo + unit_row[j] * (hi - lo))
    return out
