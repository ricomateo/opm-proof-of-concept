"""Orchestrator: generate the Volve evaluation dataset.

Usage:
    python scripts/generate_volve_dataset.py --n 15 --workers 4 --out dataset_volve_batch.csv

LHS over 4 levers, parallel docker workers, validation, single CSV output.
The default `--out` is `dataset_volve_batch.csv` so the existing Eclipse-
generated `dataset_volve.csv` stays as the reference baseline.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extractor import SCHEMA_COLUMNS  # noqa: E402
from runner_volve import run_simulation_volve  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"


@dataclass(frozen=True)
class VolveLeverRanges:
    k_mult: tuple[float, float] = (0.7, 1.5)
    phi_mult: tuple[float, float] = (0.85, 1.15)
    p_init_shift_bar: tuple[float, float] = (-13.79, 13.79)  # ~ +/- 200 psi
    qwinj_group_mult: tuple[float, float] = (0.6, 1.4)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Volve dataset")
    p.add_argument("--n", type=int, default=15, help="Number of simulations")
    p.add_argument("--workers", type=int, default=4, help="Parallel docker workers")
    p.add_argument("--out", type=str, default="dataset_volve_batch.csv")
    p.add_argument("--log", type=str, default="runs_log_volve.csv")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def sample_lhs_volve(n: int, ranges: VolveLeverRanges, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    cuts = np.linspace(0.0, 1.0, n + 1)
    u = rng.random((n, 4))
    a = cuts[:n]
    b = cuts[1 : n + 1]
    points = u * (b - a)[:, None] + a[:, None]
    for j in range(4):
        rng.shuffle(points[:, j])

    out: list[dict] = []
    for row in points:
        out.append(
            {
                "k_mult": _scale(row[0], ranges.k_mult),
                "phi_mult": _scale(row[1], ranges.phi_mult),
                "p_init_shift_bar": _scale(row[2], ranges.p_init_shift_bar),
                "qwinj_group_mult": _scale(row[3], ranges.qwinj_group_mult),
            }
        )
    return out


def _scale(u: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return float(lo + u * (hi - lo))


def run_full_batch(n: int, workers: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = sample_lhs_volve(n, VolveLeverRanges(), seed=seed)
    print(f"[batch] launching {n} Volve simulations on {workers} workers")

    started = time.perf_counter()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_simulation_volve, sim_id=i + 1, params=samples[i], runs_dir=RUNS_DIR): i + 1
            for i in range(n)
        }
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            tag = "ok" if res["ok"] else f"FAIL ({res['error']})"
            print(f"  volve_sim_{res['sim_id']:04d}: {tag} in {res['runtime_s']:.1f}s")

    elapsed = time.perf_counter() - started
    n_ok = sum(1 for r in results if r["ok"])
    print(f"[batch] {n_ok}/{n} converged in {elapsed:.1f}s")

    results.sort(key=lambda r: r["sim_id"])
    dfs = [r["df"] for r in results if r["ok"]]
    dataset = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

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
                "k_mult": params["k_mult"],
                "phi_mult": params["phi_mult"],
                "p_init_shift_bar": params["p_init_shift_bar"],
                "qwinj_group_mult": params["qwinj_group_mult"],
                "fpr_min_psi": fpr_min,
                "fpr_max_psi": fpr_max,
                "fpr_range_psi": fpr_max - fpr_min if r["ok"] else float("nan"),
            }
        )
    return dataset, pd.DataFrame(log_rows)


def validate(df: pd.DataFrame, n_requested: int) -> list[str]:
    errors: list[str] = []
    if df.empty:
        return ["dataset is empty (no sims converged)"]
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
    if df[SCHEMA_COLUMNS].isnull().any().any():
        errors.append("NaN in schema columns")

    n_sims = df["sim_id"].nunique()
    if n_sims < int(0.8 * n_requested):
        errors.append(f"only {n_sims}/{n_requested} simulations converged (< 80%)")

    cum_cols = [
        "Prod_Acumulada_Petroleo",
        "Prod_Acumulada_Gas",
        "Prod_Acumulada_Agua",
        "Iny_Acumulada_Agua",
    ]
    for col in cum_cols:
        diffs = df.groupby("sim_id")[col].diff().dropna()
        if (diffs < -1.0).any():
            errors.append(f"{col} decreases within at least one simulation")

    fpr_global_range = df["Presion_Reservorio_psi"].max() - df["Presion_Reservorio_psi"].min()
    if fpr_global_range < 100:
        errors.append(
            f"FPR variance across batch is small ({fpr_global_range:.0f} psi); "
            "consider widening the lever ranges"
        )
    return errors


def main() -> int:
    args = parse_args()
    RUNS_DIR.mkdir(exist_ok=True)

    dataset, log_df = run_full_batch(args.n, args.workers, args.seed)

    out_path = (PROJECT_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    log_path = (PROJECT_ROOT / args.log) if not Path(args.log).is_absolute() else Path(args.log)

    if not dataset.empty:
        dataset.to_csv(out_path, index=False)
        print(f"[write] dataset: {out_path} ({len(dataset)} rows, {len(dataset.columns)} cols)")
    log_df.to_csv(log_path, index=False)
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
