"""SPE9 deck templating.

Loads the baseline SPE9.DATA once, then renders a per-simulation deck by applying
six parameter levers:

  1. qwinj_rate      water injector rate cap (STB/day)
  2. qo_rate_high    producer ORAT cap during full production (STB/day)
  3. qo_rate_low     producer ORAT cap during the throttled period (STB/day)
  4. k_mult          multiplier on PERMX/Y/Z
  5. phi_mult        multiplier on PORO
  6. p_init          datum pressure in EQUIL (psia)
  7. pb_shift        shift applied to PVTO saturated Pb column (psi)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


SPE9_DIR = Path(__file__).resolve().parents[1] / "models" / "spe9"
BASELINE_DECK = SPE9_DIR / "SPE9.DATA"
INCLUDE_FILES = ("PERMVALUES.DATA", "TOPSVALUES.DATA")


@dataclass
class DeckParams:
    qwinj_rate: float
    qo_rate_high: float
    qo_rate_low: float
    k_mult: float
    phi_mult: float
    p_init: float
    pb_shift: float


def load_baseline() -> str:
    return BASELINE_DECK.read_text()


def render_deck(baseline_text: str, params: DeckParams) -> str:
    text = baseline_text
    text = _replace_wconinje(text, params.qwinj_rate)
    text = _replace_wconprod(text, params.qo_rate_high, params.qo_rate_low)
    text = _replace_equil(text, params.p_init)
    text = _insert_multiply(text, params.k_mult, params.phi_mult)
    text = _shift_pvto(text, params.pb_shift)
    return text


def _replace_wconinje(text: str, qwinj_rate: float) -> str:
    pattern = re.compile(
        r"('INJE1'\s+'WATER'\s+'OPEN'\s+'RATE'\s+)\d+(\.\d+)?",
    )
    new, n = pattern.subn(rf"\g<1>{qwinj_rate:.2f}", text, count=1)
    if n != 1:
        raise RuntimeError("WCONINJE rate line not found in deck")
    return new


def _replace_wconprod(text: str, qo_high: float, qo_low: float) -> str:
    """The deck has three WCONPROD blocks: high-low-high (1500, 100, 1500).

    We rewrite each based on the literal numeric token currently there, so the
    first/third occurrences get qo_high and the middle one gets qo_low.
    """
    pattern = re.compile(
        r"('PRODU\*'\s+'OPEN'\s+'ORAT'\s+)(\d+)(\.\d+)?",
    )

    def _sub(match: re.Match) -> str:
        prefix = match.group(1)
        current = int(match.group(2))
        if current == 1500:
            return f"{prefix}{qo_high:.2f}"
        if current == 100:
            return f"{prefix}{qo_low:.2f}"
        return match.group(0)

    new, n = pattern.subn(_sub, text)
    if n != 3:
        raise RuntimeError(f"Expected 3 WCONPROD ORAT matches, found {n}")
    return new


def _replace_equil(text: str, p_init: float) -> str:
    pattern = re.compile(
        r"(9035\s+)\d+(\.\d+)?(\s+9950\s+0\s+8800\s+0\s+1\s+0\s+0)",
    )
    new, n = pattern.subn(rf"\g<1>{p_init:.2f}\g<3>", text, count=1)
    if n != 1:
        raise RuntimeError("EQUIL line not found in deck")
    return new


def _insert_multiply(text: str, k_mult: float, phi_mult: float) -> str:
    """Insert MULTIPLY block at the end of the GRID section.

    The deck contains a literal `ECHO` line that closes a NOECHO/ECHO bracketing
    in the GRID section. We insert just before it so PORO and PERM declarations
    are already in scope.
    """
    multiply_block = (
        "\n"
        "MULTIPLY\n"
        f"   'PORO'  {phi_mult:.5f} /\n"
        f"   'PERMX' {k_mult:.5f} /\n"
        f"   'PERMY' {k_mult:.5f} /\n"
        f"   'PERMZ' {k_mult:.5f} /\n"
        "/\n"
    )
    marker = "\nECHO\n"
    if marker not in text:
        raise RuntimeError("ECHO marker not found; cannot place MULTIPLY block")
    return text.replace(marker, multiply_block + marker, 1)


def _shift_pvto(text: str, pb_shift: float) -> str:
    """Shift column 2 of saturated PVTO rows by pb_shift psi.

    Atmospheric anchor (Pb = 14.7) and the under-saturated extension row are
    not shifted. Block detection is line-based: we enter on a standalone PVTO
    line and exit on the first standalone `/` line afterward.
    """
    if pb_shift == 0:
        return text

    out_lines = []
    in_pvto = False

    for line in text.split("\n"):
        if not in_pvto:
            if line.strip() == "PVTO":
                in_pvto = True
            out_lines.append(line)
            continue

        stripped = line.strip()

        if stripped == "/":
            out_lines.append(line)
            in_pvto = False
            continue

        if not stripped or stripped.startswith("--"):
            out_lines.append(line)
            continue

        had_terminator = stripped.endswith("/")
        body = stripped.rstrip("/").strip()
        tokens = body.split()
        try:
            nums = [float(t) for t in tokens]
        except ValueError:
            out_lines.append(line)
            continue

        if len(nums) == 4 and nums[1] > 14.7:
            new_pb = nums[1] + pb_shift
            new_tokens = [
                f"{nums[0]:.3f}",
                f"{new_pb:.2f}",
                f"{nums[2]:.4f}",
                f"{nums[3]:.4f}",
            ]
            terminator = " /" if had_terminator else ""
            out_lines.append("\t" + "\t".join(new_tokens) + terminator)
        else:
            out_lines.append(line)

    return "\n".join(out_lines)
