"""Build the 16-column dataset from a SPE9 simulation summary.

Static columns are derived from the deck constants and the simulation parameters
(porosity and permeability multipliers, bubble-point shift). Dynamic columns are
read from the .UNSMRY file via resdata. PVT-derived columns (Bo, Bg, Rs) are
computed from the field pressure trajectory using the deck's PVTO and PVDG
tables (OPM does not export field-aggregated PVT scalars).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from resdata.summary import Summary

from pvt_tables import bg_from_pressure, bo_from_pressure, rs_from_pressure


# SPE9 grid constants
LAYER_THICKNESS_FT = np.array(
    [20, 15, 26, 15, 16, 14, 8, 8, 18, 12, 19, 18, 20, 50, 100], dtype=float
)
LAYER_POROSITY = np.array(
    [0.087, 0.097, 0.111, 0.16, 0.13, 0.17, 0.17, 0.08,
     0.14, 0.13, 0.12, 0.105, 0.12, 0.116, 0.157], dtype=float
)
NX, NY = 24, 25
DX_FT = DY_FT = 300.0
FT_TO_M = 0.3048

BASELINE_MEAN_PERM_MD = 108.07
BASELINE_PB_PSI = 3600.0

ESPESOR_NETO_M = float(LAYER_THICKNESS_FT.sum() * FT_TO_M)
AREA_M2 = float((NX * DX_FT * FT_TO_M) * (NY * DY_FT * FT_TO_M))
BASELINE_PORO_VOL_WEIGHTED = float(
    np.sum(LAYER_THICKNESS_FT * LAYER_POROSITY) / LAYER_THICKNESS_FT.sum()
)


SCHEMA_COLUMNS = [
    "tiempo_dias",
    "Porosidad",
    "Permeabilidad_mD",
    "Espesor_Neto_m",
    "Area",
    "Presion_Burbuja_psi",
    "Bo_rb_stb",
    "Bg_rb_scf",
    "Rs_scf_stb",
    "Caudal_Prod_Petroleo_bbl",
    "Caudal_Prod_Gas_Mpc",
    "Caudal_Iny_Agua_bbl",
    "Prod_Acumulada_Petroleo",
    "Prod_Acumulada_Gas",
    "Prod_Acumulada_Agua",
    "Iny_Acumulada_Agua",
    "Presion_Reservorio_psi",
]


def extract_features(summary_basename: Path | str, sim_id: int, params: dict) -> pd.DataFrame:
    sm = Summary(str(summary_basename))

    tiempo_dias = sm.numpy_vector("TIME")
    fpr = sm.numpy_vector("FPR")
    fopr = sm.numpy_vector("FOPR")
    fgpr = sm.numpy_vector("FGPR")
    fwir = sm.numpy_vector("FWIR")
    fopt = sm.numpy_vector("FOPT")
    fgpt = sm.numpy_vector("FGPT")
    fwpt = sm.numpy_vector("FWPT")
    fwit = sm.numpy_vector("FWIT")

    n = len(fpr)
    pb_shift = params["pb_shift"]

    porosidad = BASELINE_PORO_VOL_WEIGHTED * params["phi_mult"]
    permeabilidad = BASELINE_MEAN_PERM_MD * params["k_mult"]
    pb = BASELINE_PB_PSI + pb_shift

    rs = rs_from_pressure(fpr, pb_shift)
    bo = bo_from_pressure(fpr, pb_shift)
    bg = bg_from_pressure(fpr)

    df = pd.DataFrame(
        {
            "sim_id": np.full(n, sim_id, dtype=int),
            "tiempo_dias": tiempo_dias,
            "Porosidad": porosidad,
            "Permeabilidad_mD": permeabilidad,
            "Espesor_Neto_m": ESPESOR_NETO_M,
            "Area": AREA_M2,
            "Presion_Burbuja_psi": pb,
            "Bo_rb_stb": bo,
            "Bg_rb_scf": bg,
            "Rs_scf_stb": rs,
            "Caudal_Prod_Petroleo_bbl": fopr,
            "Caudal_Prod_Gas_Mpc": fgpr,
            "Caudal_Iny_Agua_bbl": fwir,
            "Prod_Acumulada_Petroleo": fopt,
            "Prod_Acumulada_Gas": fgpt * 1000.0,
            "Prod_Acumulada_Agua": fwpt,
            "Iny_Acumulada_Agua": fwit,
            "Presion_Reservorio_psi": fpr,
        }
    )
    return df
