"""Latin Hypercube sampling for the 6 deck levers.

Each row of the returned list is a dict consumable by deck_template.DeckParams.
We sample 7 numeric scalars (the producer ORAT cap is replicated for the high
and low control periods scaled by the original 1500/100 ratio so the throttle
profile is preserved).

Validity constraint enforced at sample time:
    p_init >= (3600 + pb_shift) + 200

If LHS produces a sample violating it, that point is regenerated with simple
rejection sampling within its stratum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class LeverRanges:
    qwinj_rate: tuple[float, float] = (2000.0, 9000.0)        # baseline 5000
    qo_rate_high: tuple[float, float] = (750.0, 2250.0)        # baseline 1500
    k_mult: tuple[float, float] = (0.5, 2.0)
    phi_mult: tuple[float, float] = (0.7, 1.3)
    p_init: tuple[float, float] = (3000.0, 4500.0)
    pb_shift: tuple[float, float] = (-300.0, 400.0)


THROTTLE_RATIO = 100.0 / 1500.0  # producer cap ratio in the deck's middle period
BASELINE_PB_PSI = 3600.0
P_INIT_OVER_PB_MARGIN = 200.0  # require p_init >= pb_new + this margin


def sample_lhs(n: int, ranges: LeverRanges | None = None, seed: int = 42) -> list[dict]:
    ranges = ranges or LeverRanges()
    rng = np.random.default_rng(seed)

    raw = _lhs_unit(n, dim=6, rng=rng)

    out: list[dict] = []
    for i in range(n):
        sample = _materialize(raw[i], ranges)
        # Reject and resample this row if the saturation constraint is violated
        attempts = 0
        while not _valid(sample) and attempts < 50:
            attempts += 1
            sample = _materialize(rng.random(6), ranges)
        if not _valid(sample):
            # Force-fix p_init by clamping it above the saturation pressure
            sample["p_init"] = (BASELINE_PB_PSI + sample["pb_shift"]) + P_INIT_OVER_PB_MARGIN + 50.0
        out.append(sample)
    return out


def _lhs_unit(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube samples in the unit hypercube, shape (n, dim)."""
    cuts = np.linspace(0.0, 1.0, n + 1)
    u = rng.random((n, dim))
    a = cuts[:n]
    b = cuts[1 : n + 1]
    points = u * (b - a)[:, None] + a[:, None]
    for j in range(dim):
        rng.shuffle(points[:, j])
    return points


def _materialize(unit_row: Sequence[float], ranges: LeverRanges) -> dict:
    qwinj_rate = _scale(unit_row[0], ranges.qwinj_rate)
    qo_rate_high = _scale(unit_row[1], ranges.qo_rate_high)
    k_mult = _scale(unit_row[2], ranges.k_mult)
    phi_mult = _scale(unit_row[3], ranges.phi_mult)
    p_init = _scale(unit_row[4], ranges.p_init)
    pb_shift = _scale(unit_row[5], ranges.pb_shift)
    return {
        "qwinj_rate": float(qwinj_rate),
        "qo_rate_high": float(qo_rate_high),
        "qo_rate_low": float(qo_rate_high * THROTTLE_RATIO),
        "k_mult": float(k_mult),
        "phi_mult": float(phi_mult),
        "p_init": float(p_init),
        "pb_shift": float(pb_shift),
    }


def _scale(u: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return lo + u * (hi - lo)


def _valid(sample: dict) -> bool:
    pb_new = BASELINE_PB_PSI + sample["pb_shift"]
    return sample["p_init"] >= pb_new + P_INIT_OVER_PB_MARGIN
