"""Train an XGBoost predictor on dimensionless features.

Hypothesis: replacing absolute cumulative volumes (STB, SCF) with ratios
normalized by the reservoir pore volume should let a model trained on SPE9
generalize to Norne, because the dimensionless quantities live in similar
ranges across reservoirs of different scale.

Feature engineering:
    Pi (psi)         := first FPR of each simulation (initial reservoir pressure)
    PV_bbl           := Area * Espesor_Neto * Porosidad * 6.28981  (pore volume in bbl)

Static features (3):
    Porosidad
    log10(Permeabilidad_mD)
    Pb_over_Pi = Presion_Burbuja_psi / Pi

Dynamic dimensionless features (5):
    Np_over_PV    = Prod_Acumulada_Petroleo / PV_bbl
    Winj_over_PV  = Iny_Acumulada_Agua / PV_bbl
    Wp_over_PV    = Prod_Acumulada_Agua / PV_bbl
    qo_over_PV    = Caudal_Prod_Petroleo_bbl / PV_bbl    (units: 1/day)
    qwinj_over_PV = Caudal_Iny_Agua_bbl / PV_bbl

Target:
    Pr_over_Pi = Presion_Reservorio_psi / Pi

After prediction, the model output is multiplied by each row's Pi to recover
FPR in psi for comparison against the previous absolute-features baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

M3_TO_BBL = 6.28981

FEATURES = [
    "Porosidad",
    "log10_Permeabilidad_mD",
    "Pb_over_Pi",
    "Np_over_PV",
    "Winj_over_PV",
    "Wp_over_PV",
    "qo_over_PV",
    "qwinj_over_PV",
]
TARGET = "Pr_over_Pi"


def add_dimensionless_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pi_per_sim = df.groupby("sim_id")["Presion_Reservorio_psi"].transform("first")
    pv_bbl = df["Area"] * df["Espesor_Neto_m"] * df["Porosidad"] * M3_TO_BBL

    df["Pi_psi"] = pi_per_sim
    df["PV_bbl"] = pv_bbl
    df["log10_Permeabilidad_mD"] = np.log10(df["Permeabilidad_mD"].clip(lower=1e-3))
    df["Pb_over_Pi"] = df["Presion_Burbuja_psi"] / pi_per_sim
    df["Np_over_PV"] = df["Prod_Acumulada_Petroleo"] / pv_bbl
    df["Winj_over_PV"] = df["Iny_Acumulada_Agua"] / pv_bbl
    df["Wp_over_PV"] = df["Prod_Acumulada_Agua"] / pv_bbl
    df["qo_over_PV"] = df["Caudal_Prod_Petroleo_bbl"] / pv_bbl
    df["qwinj_over_PV"] = df["Caudal_Iny_Agua_bbl"] / pv_bbl
    df["Pr_over_Pi"] = df["Presion_Reservorio_psi"] / pi_per_sim
    return df


def split_by_sim(df: pd.DataFrame, n_test: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sim_ids = df["sim_id"].unique().copy()
    rng.shuffle(sim_ids)
    test_sims = sim_ids[:n_test]
    train_sims = sim_ids[n_test:]
    return train_sims, test_sims


def report(label: str, y_true_psi: np.ndarray, y_pred_psi: np.ndarray) -> None:
    mae = mean_absolute_error(y_true_psi, y_pred_psi)
    rmse = float(np.sqrt(mean_squared_error(y_true_psi, y_pred_psi)))
    r2 = r2_score(y_true_psi, y_pred_psi)
    print(f"  {label:<30s}  MAE = {mae:8.2f} psi  RMSE = {rmse:8.2f} psi  R^2 = {r2:7.4f}")


def main() -> None:
    spe9 = add_dimensionless_columns(pd.read_csv(PROJECT_ROOT / "datasets" / "dataset_spe9.csv"))
    norne = add_dimensionless_columns(pd.read_csv(PROJECT_ROOT / "datasets" / "dataset_norne.csv"))

    print(f"SPE9: {spe9.shape}, Norne: {norne.shape}")
    print()
    print("Range comparison of dimensionless features:")
    print(f"  {'feature':<25s}  {'SPE9 min':>12s}  {'SPE9 max':>12s}  {'Norne min':>12s}  {'Norne max':>12s}")
    for col in FEATURES + [TARGET]:
        a, b = spe9[col].min(), spe9[col].max()
        c, d = norne[col].min(), norne[col].max()
        flag = " *" if (c < a or d > b) else ""
        print(f"  {col:<25s}  {a:12.4f}  {b:12.4f}  {c:12.4f}  {d:12.4f}{flag}")
    print("  (* = Norne value falls outside the SPE9 training range)")

    # Sim-level split on SPE9
    train_sims, test_sims = split_by_sim(spe9, n_test=20, seed=42)
    train_mask = spe9["sim_id"].isin(train_sims).to_numpy()
    test_mask = spe9["sim_id"].isin(test_sims).to_numpy()

    val_sims = train_sims[:16]
    train_inner_mask = train_mask & ~spe9["sim_id"].isin(val_sims).to_numpy()
    val_mask = train_mask & spe9["sim_id"].isin(val_sims).to_numpy()

    X_train = spe9.loc[train_inner_mask, FEATURES].to_numpy()
    y_train = spe9.loc[train_inner_mask, TARGET].to_numpy()
    X_val = spe9.loc[val_mask, FEATURES].to_numpy()
    y_val = spe9.loc[val_mask, TARGET].to_numpy()
    X_test = spe9.loc[test_mask, FEATURES].to_numpy()
    y_test = spe9.loc[test_mask, TARGET].to_numpy()
    pi_test = spe9.loc[test_mask, "Pi_psi"].to_numpy()

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        tree_method="hist",
        early_stopping_rounds=30,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"\ntrained: best_iteration = {model.best_iteration}, n_features = {len(FEATURES)}")

    print("\nResults (all metrics on absolute FPR in psi after denormalization):")
    pred_test = model.predict(X_test) * pi_test
    true_test = y_test * pi_test
    report("SPE9 held-out (20 sims)", true_test, pred_test)

    pi_norne = norne["Pi_psi"].to_numpy()
    X_norne = norne[FEATURES].to_numpy()
    y_norne = norne[TARGET].to_numpy()
    pred_norne = model.predict(X_norne) * pi_norne
    true_norne = y_norne * pi_norne
    report("Norne (full dataset, OOD)", true_norne, pred_norne)

    spe9_median_psi = spe9["Presion_Reservorio_psi"].median()
    baseline_pred = np.full_like(true_norne, spe9_median_psi)
    report("Baseline (SPE9 median)", true_norne, baseline_pred)

    # Per-sim Norne summary
    print("\nPer-sim Norne MAE distribution (psi):")
    per_sim = []
    for sid, grp in norne.groupby("sim_id"):
        p = model.predict(grp[FEATURES].to_numpy()) * grp["Pi_psi"].to_numpy()
        per_sim.append(mean_absolute_error(grp["Presion_Reservorio_psi"], p))
    per_sim = np.array(per_sim)
    print(f"  min = {per_sim.min():.1f}, p25 = {np.quantile(per_sim, 0.25):.1f}, "
          f"median = {np.quantile(per_sim, 0.5):.1f}, p75 = {np.quantile(per_sim, 0.75):.1f}, "
          f"max = {per_sim.max():.1f}")

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nFeature importance (gain):")
    print(imp.round(4).to_string())

    out_path = PROJECT_ROOT / "notebooks" / "xgboost_fpr_dimensionless.json"
    model.save_model(out_path)
    print(f"\nsaved model: {out_path}")


if __name__ == "__main__":
    sys.exit(main())
