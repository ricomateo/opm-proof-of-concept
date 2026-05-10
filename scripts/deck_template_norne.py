"""Norne deck templating.

The Norne baseline deck (`NORNE_ATW2013.DATA`) plus the equilibration include
(`INCLUDE/PETRO/E3.prop`) are read once. For each simulation we apply three
parameter levers:

  1. k_mult              multiplier on PERMX/Y/Z
  2. phi_mult            multiplier on PORO
  3. p_init_shift_bar    shift on column 2 of every EQUIL row (bar)

We keep the historical schedule (WCONHIST/WCONINJH) untouched: tweaking
producer/injector rates over a 9-year history would require rewriting the
entire schedule include and is out of scope for this evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


NORNE_DIR = Path(__file__).resolve().parents[1] / "models" / "norne"
BASELINE_DECK_PATH = NORNE_DIR / "NORNE_ATW2013.DATA"
EQUIL_INCLUDE_REL = Path("INCLUDE/PETRO/E3.prop")
EQUIL_INCLUDE_PATH = NORNE_DIR / EQUIL_INCLUDE_REL


@dataclass
class NorneDeckParams:
    k_mult: float
    phi_mult: float
    p_init_shift_bar: float


def load_baseline() -> tuple[str, str]:
    return BASELINE_DECK_PATH.read_text(), EQUIL_INCLUDE_PATH.read_text()


def render_deck(deck_text: str, equil_text: str, params: NorneDeckParams) -> tuple[str, str]:
    """Returns (modified_main_deck, modified_equil_include)."""
    new_deck = _insert_multiply(deck_text, params.k_mult, params.phi_mult)
    new_equil = _shift_equil(equil_text, params.p_init_shift_bar)
    return new_deck, new_equil


def _insert_multiply(text: str, k_mult: float, phi_mult: float) -> str:
    """Insert a MULTIPLY block just before the EDIT section header.

    PORO and PERMX/Y/Z have already been declared and the existing per-layer
    PERMZ MULTIPLY has already been applied by the time we reach EDIT, so our
    block scales the post-existing values uniformly.
    """
    block = (
        "\n"
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
    r"^(\s*)([-\d.]+)(\s+)([-\d.]+)(\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+\d+\s+\d+\s+\d+\s*/.*)$"
)


def _shift_equil(text: str, p_shift_bar: float) -> str:
    """Shift column 2 of each EQUIL row by p_shift_bar (bar)."""
    if p_shift_bar == 0:
        return text

    out_lines: list[str] = []
    in_equil = False
    rows_shifted = 0

    for line in text.split("\n"):
        stripped = line.strip()

        if not in_equil:
            if stripped == "EQUIL":
                in_equil = True
            out_lines.append(line)
            continue

        if not stripped or stripped.startswith("--"):
            out_lines.append(line)
            continue

        match = _EQUIL_ROW.match(line)
        if match is None:
            in_equil = False
            out_lines.append(line)
            continue

        leading_ws, datum, sep, pressure, tail = match.groups()
        new_pressure = float(pressure) + p_shift_bar
        out_lines.append(f"{leading_ws}{datum}{sep}{new_pressure:.4f}{tail}")
        rows_shifted += 1

    if rows_shifted == 0:
        raise RuntimeError("No EQUIL data rows matched; pattern may be wrong")
    return "\n".join(out_lines)
