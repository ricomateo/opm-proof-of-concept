"""Build the 16-column dataset row from a Norne simulation.

Two complications relative to the SPE9 extractor:

  1. Norne runs in METRIC units (bar, SM3) but the schema and the trained
     XGBoost model expect FIELD units (psi, STB, SCF). Every dynamic column
     gets a unit conversion applied at extraction time.

  2. The Norne grid is irregular and heterogeneous, so the static features
     (porosity, permeability, geometry, Pb) are derived from a one-shot
     baseline calibration instead of the deck text. The baseline values
     below were computed once from `runs/norne_baseline/NORNE_ATW2013.INIT`
     using the helper script and are scaled by the per-sim levers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from resdata.summary import Summary

from pvt_tables import bg_from_pressure, bo_from_pressure, rs_from_pressure


# Calibrated once from runs/norne_baseline INIT (volume-weighted, NTG-aware).
# These are the values BEFORE per-sim phi_mult / k_mult are applied.
NORNE_BASELINE_PORO = 0.2470
NORNE_BASELINE_PERM_MD = 487.49
NORNE_ESPESOR_NETO_M = 172.41
NORNE_AREA_M2 = 1.6962e7

# Norne PVT-WET-GAS at reservoir conditions: bubble point sits around 250 bar
# (Statoil deck). Pb is held constant per simulation since pb_shift is not a
# Norne lever. In psi for schema consistency.
NORNE_PB_BASELINE_BAR = 250.0

# METRIC -> FIELD conversions
BAR_TO_PSI = 14.5037738
SM3_TO_STB = 6.28981077
SM3_TO_SCF = 35.3146667
SM3_TO_MSCF = SM3_TO_SCF / 1000.0


def extract_features_norne(summary_basename: Path | str, sim_id: int, params: dict) -> pd.DataFrame:
    sm = Summary(str(summary_basename))

    tiempo_dias = sm.numpy_vector("TIME")
    fpr_bar = sm.numpy_vector("FPR")
    fopr_sm3d = sm.numpy_vector("FOPR")
    fgpr_sm3d = sm.numpy_vector("FGPR")
    fwir_sm3d = sm.numpy_vector("FWIR")
    fopt_sm3 = sm.numpy_vector("FOPT")
    fgpt_sm3 = sm.numpy_vector("FGPT")
    fwpt_sm3 = sm.numpy_vector("FWPT")
    fwit_sm3 = sm.numpy_vector("FWIT")

    fpr_psi = fpr_bar * BAR_TO_PSI
    pb_psi = (NORNE_PB_BASELINE_BAR + params.get("p_init_shift_bar", 0.0)) * BAR_TO_PSI

    # PVT columns (not used by the model since they leak the target; filled
    # with the SPE9 PVT interpolated at the FPR-in-psi for schema consistency).
    rs = rs_from_pressure(fpr_psi, pb_shift=0.0)
    bo = bo_from_pressure(fpr_psi, pb_shift=0.0)
    bg = bg_from_pressure(fpr_psi)

    porosidad = NORNE_BASELINE_PORO * params["phi_mult"]
    permeabilidad = NORNE_BASELINE_PERM_MD * params["k_mult"]

    n = len(fpr_bar)
    df = pd.DataFrame(
        {
            "sim_id": np.full(n, sim_id, dtype=int),
            "tiempo_dias": tiempo_dias,
            "Porosidad": porosidad,
            "Permeabilidad_mD": permeabilidad,
            "Espesor_Neto_m": NORNE_ESPESOR_NETO_M,
            "Area": NORNE_AREA_M2,
            "Presion_Burbuja_psi": pb_psi,
            "Bo_rb_stb": bo,
            "Bg_rb_scf": bg,
            "Rs_scf_stb": rs,
            "Caudal_Prod_Petroleo_bbl": fopr_sm3d * SM3_TO_STB,
            "Caudal_Prod_Gas_Mpc": fgpr_sm3d * SM3_TO_MSCF,
            "Caudal_Iny_Agua_bbl": fwir_sm3d * SM3_TO_STB,
            "Prod_Acumulada_Petroleo": fopt_sm3 * SM3_TO_STB,
            "Prod_Acumulada_Gas": fgpt_sm3 * SM3_TO_SCF,
            "Prod_Acumulada_Agua": fwpt_sm3 * SM3_TO_STB,
            "Iny_Acumulada_Agua": fwit_sm3 * SM3_TO_STB,
            "Presion_Reservorio_psi": fpr_psi,
        }
    )
    return df
