"""SPE9 model configuration: 6 levers, FIELD units, single-deck render."""

from __future__ import annotations

import re
from pathlib import Path

from .base import DeckConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DECK_DIR = PROJECT_ROOT / "models" / "spe9"
MAIN_DECK = "SPE9.DATA"

THROTTLE_RATIO = 100.0 / 1500.0  # producer cap ratio in deck's middle period
BASELINE_PB_PSI = 3600.0
P_INIT_OVER_PB_MARGIN = 200.0


LEVER_RANGES = {
    "qwinj_rate": (2000.0, 9000.0),
    "qo_rate_high": (750.0, 2250.0),
    "k_mult": (0.5, 2.0),
    "phi_mult": (0.7, 1.3),
    "p_init": (3000.0, 4500.0),
    "pb_shift": (-300.0, 400.0),
}


def _expand_levers(params: dict) -> dict:
    """Add the dependent qo_rate_low (THROTTLE_RATIO * qo_rate_high)."""
    expanded = dict(params)
    if "qo_rate_low" not in expanded:
        expanded["qo_rate_low"] = expanded["qo_rate_high"] * THROTTLE_RATIO
    return expanded


def sample_validator(sample: dict) -> bool:
    pb_new = BASELINE_PB_PSI + sample["pb_shift"]
    return sample["p_init"] >= pb_new + P_INIT_OVER_PB_MARGIN


# Layer thicknesses and porosities (15 layers, used to derive volume-weighted statics)
import numpy as _np

_LAYER_THICKNESS_FT = _np.array(
    [20, 15, 26, 15, 16, 14, 8, 8, 18, 12, 19, 18, 20, 50, 100], dtype=float
)
_LAYER_POROSITY = _np.array(
    [0.087, 0.097, 0.111, 0.16, 0.13, 0.17, 0.17, 0.08,
     0.14, 0.13, 0.12, 0.105, 0.12, 0.116, 0.157], dtype=float
)
_NX, _NY = 24, 25
_DX_FT = _DY_FT = 300.0
_FT_TO_M = 0.3048

STATIC_FEATURES = {
    "baseline_porosity": float(
        _np.sum(_LAYER_THICKNESS_FT * _LAYER_POROSITY) / _LAYER_THICKNESS_FT.sum()
    ),
    "baseline_perm_md": 108.07,
    "espesor_neto_m": float(_LAYER_THICKNESS_FT.sum() * _FT_TO_M),
    "area_m2": float((_NX * _DX_FT * _FT_TO_M) * (_NY * _DY_FT * _FT_TO_M)),
}


def render_deck(params: dict) -> dict[str, str]:
    text = (DECK_DIR / MAIN_DECK).read_text()
    p = _expand_levers(params)
    text = _replace_wconinje(text, p["qwinj_rate"])
    text = _replace_wconprod(text, p["qo_rate_high"], p["qo_rate_low"])
    text = _replace_equil(text, p["p_init"])
    text = _insert_multiply(text, p["k_mult"], p["phi_mult"])
    text = _shift_pvto(text, p["pb_shift"])
    return {MAIN_DECK: text}


# --- regex helpers (lifted verbatim from the legacy deck_template.py) ---


def _replace_wconinje(text: str, qwinj_rate: float) -> str:
    pattern = re.compile(r"('INJE1'\s+'WATER'\s+'OPEN'\s+'RATE'\s+)\d+(\.\d+)?")
    new, n = pattern.subn(rf"\g<1>{qwinj_rate:.2f}", text, count=1)
    if n != 1:
        raise RuntimeError("WCONINJE rate line not found in deck")
    return new


def _replace_wconprod(text: str, qo_high: float, qo_low: float) -> str:
    pattern = re.compile(r"('PRODU\*'\s+'OPEN'\s+'ORAT'\s+)(\d+)(\.\d+)?")

    def _sub(match: re.Match) -> str:
        prefix, current = match.group(1), int(match.group(2))
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
        r"(9035\s+)\d+(\.\d+)?(\s+9950\s+0\s+8800\s+0\s+1\s+0\s+0)"
    )
    new, n = pattern.subn(rf"\g<1>{p_init:.2f}\g<3>", text, count=1)
    if n != 1:
        raise RuntimeError("EQUIL line not found in deck")
    return new


def _insert_multiply(text: str, k_mult: float, phi_mult: float) -> str:
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


CONFIG = DeckConfig(
    name="spe9",
    deck_dir=DECK_DIR,
    main_deck_filename=MAIN_DECK,
    lever_ranges=LEVER_RANGES,
    sample_validator=sample_validator,
    render_deck=render_deck,
    static_features=STATIC_FEATURES,
    unit_system="FIELD",
    baseline_pb=BASELINE_PB_PSI,
    flow_timeout_s=600,
)
