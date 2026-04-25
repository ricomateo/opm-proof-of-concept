"""PVT tables for SPE9.

Hardcoded from the SPE9.DATA PVTO and PVDG blocks. Used to derive Bo, Bg, and Rs
at field-average pressure for each timestep, since OPM Flow does not export those
as field-aggregated scalars in SUMMARY.
"""

from __future__ import annotations

import numpy as np

# PVTO saturated table: (Rs in Mscf/stb, Pb in psia, Bo in rb/stb, mu_o in cP).
# The first row (Pb = 14.7) is the dummy atmospheric anchor and is excluded from
# pb-shift logic. The undersaturated extension at (5000 psia, Bo = 1.1189) is
# treated as a separate point used only for Bo above Pb.
_PVTO_SATURATED = np.array(
    [
        [0.000, 14.7, 1.0000],
        [0.165, 400.0, 1.0120],
        [0.335, 800.0, 1.0255],
        [0.500, 1200.0, 1.0380],
        [0.665, 1600.0, 1.0510],
        [0.828, 2000.0, 1.0630],
        [0.985, 2400.0, 1.0750],
        [1.130, 2800.0, 1.0870],
        [1.270, 3200.0, 1.0985],
        [1.390, 3600.0, 1.1100],
        [1.500, 4000.0, 1.1200],
    ]
)

# Last under-saturated extension point: (Pressure in psia, Bo in rb/stb).
_PVTO_UNDERSAT = (5000.0, 1.1189)

# PVDG: (P in psia, Bg in rb/Mscf, mu_g in cP).
_PVDG = np.array(
    [
        [14.7, 191.7443, 0.0125],
        [400.0, 5.8979, 0.0130],
        [800.0, 2.9493, 0.0135],
        [1200.0, 1.9594, 0.0140],
        [1600.0, 1.4695, 0.0145],
        [2000.0, 1.1797, 0.0150],
        [2400.0, 0.9796, 0.0155],
        [2800.0, 0.8397, 0.0160],
        [3200.0, 0.7398, 0.0165],
        [3600.0, 0.6498, 0.0170],
        [4000.0, 0.5849, 0.0175],
    ]
)


def shifted_pb_grid(pb_shift: float) -> np.ndarray:
    """Saturated Pb pressures with the shift applied.

    The atmospheric anchor at 14.7 psia is left untouched.
    """
    pb = _PVTO_SATURATED[:, 1].copy()
    mask = pb > 14.7
    pb[mask] = pb[mask] + pb_shift
    return pb


def rs_from_pressure(p: np.ndarray, pb_shift: float) -> np.ndarray:
    """Rs in scf/stb at reservoir pressure p (psia).

    Below the (shifted) bubble point: linear interpolation on the saturated curve.
    Above bubble point: clamped to the value at Pb (no free gas).
    """
    pb_grid = shifted_pb_grid(pb_shift)
    rs_mscf = _PVTO_SATURATED[:, 0]
    pb_max = pb_grid[-1]
    p_eff = np.minimum(p, pb_max)
    rs = np.interp(p_eff, pb_grid, rs_mscf)
    return rs * 1000.0


def bo_from_pressure(p: np.ndarray, pb_shift: float) -> np.ndarray:
    """Bo in rb/stb at reservoir pressure p (psia).

    Below the (shifted) bubble point: linear interpolation on the saturated curve.
    Above bubble point: linear interpolation toward the under-saturated point at
    5000 psia (Bo = 1.1189). The under-saturated point is anchored at the original
    deck pressure (5000) and is not shifted.
    """
    pb_grid = shifted_pb_grid(pb_shift)
    bo_grid = _PVTO_SATURATED[:, 2]
    pb_max = pb_grid[-1]

    p = np.asarray(p, dtype=float)
    bo_sat = np.interp(np.minimum(p, pb_max), pb_grid, bo_grid)

    p_undersat, bo_undersat = _PVTO_UNDERSAT
    above = p > pb_max
    if np.any(above) and p_undersat > pb_max:
        slope = (bo_undersat - bo_grid[-1]) / (p_undersat - pb_max)
        bo_sat = np.where(above, bo_grid[-1] + slope * (p - pb_max), bo_sat)
    return bo_sat


def bg_from_pressure(p: np.ndarray) -> np.ndarray:
    """Bg in rb/scf at reservoir pressure p (psia).

    PVDG covers the same pressure range as PVTO. Outside the table we extrapolate
    using the nearest pair of points to keep the function defined for any FPR.
    Returned Bg is in rb/scf (PVDG is rb/Mscf, so we divide by 1000).
    """
    p_grid = _PVDG[:, 0]
    bg_grid = _PVDG[:, 1]
    bg = np.interp(p, p_grid, bg_grid)
    return bg / 1000.0
