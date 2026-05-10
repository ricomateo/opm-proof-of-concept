"""Volve deck templating.

The patched Volve deck (`volve/VOLVE_2016.DATA`) is read once. For each
simulation we apply four parameter levers:

  1. k_mult                multiplier on PERMX/Y/Z
  2. phi_mult              multiplier on PORO
  3. p_init_shift_bar      shift on column 2 of every EQUIL row (bar)
  4. qwinj_group_mult      multiplier on the FIELD WAT RATE in GCONINJE

Unlike Norne, the Volve EQUIL block is inline in the main deck (not in an
include), so only one file is rendered per simulation.

The historical schedule (WCONHIST/WCONINJE per-well rates) is left
untouched: rewriting 8.7 years of history controls is out of scope for
this evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


VOLVE_DIR = Path(__file__).resolve().parents[1] / "volve"
BASELINE_DECK_PATH = VOLVE_DIR / "VOLVE_2016.DATA"


@dataclass
class VolveDeckParams:
    k_mult: float
    phi_mult: float
    p_init_shift_bar: float
    qwinj_group_mult: float


def load_baseline() -> str:
    return BASELINE_DECK_PATH.read_bytes().decode("latin-1")


def render_deck(deck_text: str, params: VolveDeckParams) -> str:
    text = _insert_multiply(deck_text, params.k_mult, params.phi_mult)
    text = _shift_equil(text, params.p_init_shift_bar)
    text = _scale_gconinje(text, params.qwinj_group_mult)
    return text


def _insert_multiply(text: str, k_mult: float, phi_mult: float) -> str:
    """Insert a global MULTIPLY block just before the EDIT section header.

    PORO and PERMX/Y/Z have already been declared by the existing GRID
    section; our block scales the post-existing values uniformly.
    """
    block = (
        "\n"
        "-- Per-simulation MULTIPLY block injected by deck_template_volve.py\n"
        "MULTIPLY\n"
        f"   'PORO'  {phi_mult:.5f} /\n"
        f"   'PERMX' {k_mult:.5f} /\n"
        f"   'PERMY' {k_mult:.5f} /\n"
        f"   'PERMZ' {k_mult:.5f} /\n"
        "/\n"
    )
    marker = "\nEDIT\n"
    if marker not in text:
        raise RuntimeError("EDIT marker not found in deck; cannot place MULTIPLY block")
    return text.replace(marker, block + marker, 1)


_EQUIL_ROW = re.compile(
    r"^(\s*[-\d.]+\s+)([-\d.]+)(\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+\d+\s+\d+\s+\d+\s*/.*)$"
)


def _shift_equil(text: str, p_shift_bar: float) -> str:
    """Shift column 2 of each EQUIL data row by p_shift_bar (bar)."""
    if p_shift_bar == 0:
        return text

    out: list[str] = []
    in_equil = False
    rows_shifted = 0

    for line in text.split("\n"):
        stripped = line.strip()

        if not in_equil:
            if stripped == "EQUIL":
                in_equil = True
            out.append(line)
            continue

        if not stripped or stripped.startswith("--"):
            out.append(line)
            continue

        match = _EQUIL_ROW.match(line)
        if match is None:
            in_equil = False
            out.append(line)
            continue

        prefix, pressure, tail = match.groups()
        new_pressure = float(pressure) + p_shift_bar
        out.append(f"{prefix}{new_pressure:.4f}{tail}")
        rows_shifted += 1

    if rows_shifted != 12:
        raise RuntimeError(f"Expected 12 EQUIL rows, shifted {rows_shifted}")
    return "\n".join(out)


_GCONINJE_LINE = re.compile(
    r"('FIELD'\s+'WAT'\s+'RATE'\s+)([\d.]+)(\s+\S+\s+\S+\s+[\d.]+\s+/)"
)


def _scale_gconinje(text: str, mult: float) -> str:
    """Scale the FIELD WAT RATE token in the GCONINJE record."""
    if mult == 1.0:
        return text

    def _sub(match: re.Match) -> str:
        prefix, rate, tail = match.group(1), float(match.group(2)), match.group(3)
        return f"{prefix}{rate * mult:.2f}{tail}"

    new, n = _GCONINJE_LINE.subn(_sub, text, count=1)
    if n != 1:
        raise RuntimeError("GCONINJE FIELD WAT RATE record not found")
    return new
