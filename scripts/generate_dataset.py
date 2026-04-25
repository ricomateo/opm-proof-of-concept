"""Orchestrator: generate the SPE9 + OPM Flow dataset.

Usage:
    python scripts/generate_dataset.py --n 100 --workers 10 --out dataset.csv

Steps:
    1. Render and run two smoke simulations (baseline at 1.0x, extremes).
    2. If both converge, run the full LHS batch in parallel.
    3. Concatenate results, validate, and write dataset.csv plus runs_log.csv.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# Make this script runnable both as `python scripts/generate_dataset.py`
# and as a module from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extractor import SCHEMA_COLUMNS  # noqa: E402
from runner import run_simulation  # noqa: E402
from sampling import sample_lhs  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SPE9 dataset")
    p.add_argument("--n", type=int, default=100, help="Number of simulations")
    p.add_argument("--workers", type=int, default=10, help="Parallel docker workers")
    p.add_argument("--out", type=str, default="dataset.csv", help="Output CSV path")
    p.add_argument("--log", type=str, default="runs_log.csv", help="Per-sim log CSV path")
    p.add_argument("--seed", type=int, default=42, help="LHS seed")
    p.add_argument("--skip-smoke", action="store_true", help="Skip the two smoke runs")
    return p.parse_args()


def baseline_params() -> dict:
    return {
        "qwinj_rate": 5000.0,
        "qo_rate_high": 1500.0,
        "qo_rate_low": 100.0,
        "k_mult": 1.0,
        "phi_mult": 1.0,
        "p_init": 3600.0,
        "pb_shift": 0.0,
    }


def extremes_params() -> dict:
    return {
        "qwinj_rate": 9000.0,
        "qo_rate_high": 750.0,
        "qo_rate_low": 50.0,
        "k_mult": 2.0,
        "phi_mult": 1.3,
        "p_init": 4500.0,
        "pb_shift": 400.0,
    }


def run_smoke_tests() -> bool:
    print("[smoke] running baseline simulation...")
    base_res = run_simulation(sim_id=9001, params=baseline_params(), runs_dir=RUNS_DIR)
    _print_result(base_res, "baseline")
    if not base_res["ok"]:
        return False

    print("[smoke] running extremes simulation...")
    ext_res = run_simulation(sim_id=9002, params=extremes_params(), runs_dir=RUNS_DIR)
    _print_result(ext_res, "extremes")
    if not ext_res["ok"]:
        return False

    base_df = base_res["df"]
    ext_df = ext_res["df"]
    base_range = base_df["Presion_Reservorio_psi"].max() - base_df["Presion_Reservorio_psi"].min()
    ext_range = ext_df["Presion_Reservorio_psi"].max() - ext_df["Presion_Reservorio_psi"].min()
    print(f"[smoke] FPR range baseline = {base_range:.0f} psi, extremes = {ext_range:.0f} psi")
    fpr_diff = abs(base_df["Presion_Reservorio_psi"].iloc[-1] - ext_df["Presion_Reservorio_psi"].iloc[-1])
    print(f"[smoke] |FPR_final difference| = {fpr_diff:.0f} psi (expect > 100 to confirm levers move output)")
    return fpr_diff > 100.0


def _print_result(res: dict, label: str) -> None:
    if res["ok"]:
        print(f"[smoke] {label}: ok in {res['runtime_s']:.1f}s, {len(res['df'])} timesteps")
    else:
        print(f"[smoke] {label}: FAILED - {res['error']}")


def run_full_batch(n: int, workers: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = sample_lhs(n, seed=seed)
    print(f"[batch] launching {n} simulations on {workers} workers")

    started = time.perf_counter()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_simulation, sim_id=i + 1, params=samples[i], runs_dir=RUNS_DIR): i + 1
            for i in range(n)
        }
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            tag = "ok" if res["ok"] else f"FAIL ({res['error']})"
            print(f"  sim_{res['sim_id']:04d}: {tag} in {res['runtime_s']:.1f}s")

    elapsed = time.perf_counter() - started
    n_ok = sum(1 for r in results if r["ok"])
    print(f"[batch] {n_ok}/{n} converged in {elapsed:.1f}s")

    results.sort(key=lambda r: r["sim_id"])
    dfs = [r["df"] for r in results if r["ok"]]
    dataset = pd.concat(dfs, ignore_index=True)

    log_rows = []
    for r in results:
        params = r["params"]
        if r["ok"]:
            df = r["df"]
            fpr_min = float(df["Presion_Reservorio_psi"].min())
            fpr_max = float(df["Presion_Reservorio_psi"].max())
        else:
            fpr_min = fpr_max = float("nan")
        log_rows.append(
            {
                "sim_id": r["sim_id"],
                "ok": r["ok"],
                "error": r["error"] or "",
                "runtime_s": r["runtime_s"],
                "qwinj_rate": params["qwinj_rate"],
                "qo_rate_high": params["qo_rate_high"],
                "qo_rate_low": params["qo_rate_low"],
                "k_mult": params["k_mult"],
                "phi_mult": params["phi_mult"],
                "p_init": params["p_init"],
                "pb_shift": params["pb_shift"],
                "fpr_min_psi": fpr_min,
                "fpr_max_psi": fpr_max,
                "fpr_range_psi": fpr_max - fpr_min if r["ok"] else float("nan"),
            }
        )
    return dataset, pd.DataFrame(log_rows)


def validate(df: pd.DataFrame, n_requested: int) -> list[str]:
    errors: list[str] = []
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
    if df[SCHEMA_COLUMNS].isnull().any().any():
        errors.append("NaN values present in schema columns")

    n_sims = df["sim_id"].nunique()
    if n_sims < int(0.9 * n_requested):
        errors.append(f"only {n_sims}/{n_requested} simulations converged (< 90%)")

    cum_cols = [
        "Prod_Acumulada_Petroleo",
        "Prod_Acumulada_Gas",
        "Prod_Acumulada_Agua",
        "Iny_Acumulada_Agua",
    ]
    for col in cum_cols:
        diffs = df.groupby("sim_id")[col].diff().dropna()
        if (diffs < -1e-3).any():
            errors.append(f"{col} decreases within at least one simulation")

    bo_min = df["Bo_rb_stb"].min()
    bo_max = df["Bo_rb_stb"].max()
    if bo_min < 1.00 or bo_max > 1.20:
        errors.append(f"Bo out of expected range [1.00, 1.20]: actual [{bo_min:.3f}, {bo_max:.3f}]")

    rs_min = df["Rs_scf_stb"].min()
    rs_max = df["Rs_scf_stb"].max()
    if rs_min < 0 or rs_max > 1600:
        errors.append(f"Rs out of expected range [0, 1600]: actual [{rs_min:.1f}, {rs_max:.1f}]")

    fpr_global_range = df["Presion_Reservorio_psi"].max() - df["Presion_Reservorio_psi"].min()
    if fpr_global_range < 2000:
        errors.append(
            f"FPR variance across batch is small ({fpr_global_range:.0f} psi); "
            "consider widening the lever ranges"
        )
    return errors


def main() -> int:
    args = parse_args()
    RUNS_DIR.mkdir(exist_ok=True)

    if not args.skip_smoke:
        if not run_smoke_tests():
            print("[smoke] aborting; smoke test did not pass")
            return 1

    dataset, log_df = run_full_batch(args.n, args.workers, args.seed)

    out_path = (PROJECT_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    log_path = (PROJECT_ROOT / args.log) if not Path(args.log).is_absolute() else Path(args.log)

    dataset.to_csv(out_path, index=False)
    log_df.to_csv(log_path, index=False)
    print(f"[write] dataset: {out_path} ({len(dataset)} rows, {len(dataset.columns)} cols)")
    print(f"[write] runs log: {log_path}")

    print("[validate] running checks...")
    errors = validate(dataset, args.n)
    if errors:
        print("[validate] issues found:")
        for e in errors:
            print(f"  - {e}")
        return 2
    print("[validate] all checks passed")

    print("\n[summary] dataset shape:", dataset.shape)
    print("[summary] FPR range across batch:",
          f"{dataset['Presion_Reservorio_psi'].min():.0f} -> {dataset['Presion_Reservorio_psi'].max():.0f} psi")
    print("[summary] simulations included:", dataset["sim_id"].nunique())
    return 0


if __name__ == "__main__":
    sys.exit(main())
