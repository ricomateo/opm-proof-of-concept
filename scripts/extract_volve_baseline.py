"""One-shot extractor for the existing Volve baseline UNSMRY.

Reads `VOLVE_2016.UNSMRY` (Eclipse-generated, already shipped with the model)
and produces `dataset_volve.csv` with the same 18-column schema as
`dataset.csv` and `dataset_norne.csv`. Single simulation, no parameter
levers, no OPM Flow run.

Static features (Porosidad, Permeabilidad, Espesor, Area) were calibrated
once from `VOLVE_2016.INIT` (volume-weighted, NTG-aware). Pb is taken from
the EQUIL/PVT block of the deck; its exact value does not affect the
predictor since Pb is excluded from the model features (leakage filter).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from resdata.summary import Summary

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pvt_tables import bg_from_pressure, bo_from_pressure, rs_from_pressure  # noqa: E402

VOLVE_DIR = PROJECT_ROOT / "Volve_sim_model_PPA-Eclipse Res Model"
SUMMARY_BASE = VOLVE_DIR / "VOLVE_2016"

# Volve baseline static (calibrated from VOLVE_2016.INIT).
VOLVE_BASELINE_PORO = 0.2212
VOLVE_BASELINE_PERM_MD = 1645.55
VOLVE_ESPESOR_NETO_M = 77.75
VOLVE_AREA_M2 = 6.7962e6

# Approximate bubble point (METRIC). Volve PVT, nominal mid-region.
VOLVE_PB_BASELINE_BAR = 280.0

# METRIC -> FIELD
BAR_TO_PSI = 14.5037738
SM3_TO_STB = 6.28981077
SM3_TO_SCF = 35.3146667
SM3_TO_MSCF = SM3_TO_SCF / 1000.0


def main() -> int:
    sm = Summary(str(SUMMARY_BASE))
    n = len(sm.numpy_vector("FPR"))
    print(f"loaded UNSMRY: {n} timesteps")

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
    pb_psi = VOLVE_PB_BASELINE_BAR * BAR_TO_PSI

    # PVT columns: filled with SPE9 PVT interpolation for schema consistency.
    rs = rs_from_pressure(fpr_psi, pb_shift=0.0)
    bo = bo_from_pressure(fpr_psi, pb_shift=0.0)
    bg = bg_from_pressure(fpr_psi)

    df = pd.DataFrame(
        {
            "sim_id": np.full(n, 1, dtype=int),
            "tiempo_dias": tiempo_dias,
            "Porosidad": VOLVE_BASELINE_PORO,
            "Permeabilidad_mD": VOLVE_BASELINE_PERM_MD,
            "Espesor_Neto_m": VOLVE_ESPESOR_NETO_M,
            "Area": VOLVE_AREA_M2,
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

    out_path = PROJECT_ROOT / "dataset_volve.csv"
    df.to_csv(out_path, index=False)

    print(f"wrote: {out_path} ({len(df)} rows, {len(df.columns)} cols)")
    print()
    print("Quick stats:")
    print(f"  tiempo_dias: {df['tiempo_dias'].min():.1f} -> {df['tiempo_dias'].max():.1f}")
    print(f"  FPR (psi): {df['Presion_Reservorio_psi'].min():.1f} -> {df['Presion_Reservorio_psi'].max():.1f}")
    print(f"  FOPT final (STB): {df['Prod_Acumulada_Petroleo'].iloc[-1]:.2e}")
    print(f"  FWIT final (STB): {df['Iny_Acumulada_Agua'].iloc[-1]:.2e}")
    print(f"  Porosidad: {df['Porosidad'].iloc[0]}")
    print(f"  Permeabilidad: {df['Permeabilidad_mD'].iloc[0]} mD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
