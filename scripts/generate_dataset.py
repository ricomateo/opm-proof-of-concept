"""Unified dataset generator for any registered reservoir model.

Usage:
    python scripts/generate_dataset.py --model spe9   --n 100 --workers 10
    python scripts/generate_dataset.py --model norne  --n 30  --workers 4
    python scripts/generate_dataset.py --model volve  --n 15  --workers 2

LHS sampling, parallel docker workers, validation, single CSV output.
The model selector resolves to a `DeckConfig` from `scripts/decks/`. The
runner, extractor, and validation are model-agnostic.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decks import available_models, get_config  # noqa: E402
from extractor import SCHEMA_COLUMNS  # noqa: E402
from runner import run_simulation  # noqa: E402
from sampling import sample_lhs  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a reservoir dataset")
    p.add_argument("--model", required=True, choices=available_models(),
                   help="Reservoir model to simulate")
    p.add_argument("--n", type=int, default=30, help="Number of simulations")
    p.add_argument("--workers", type=int, default=4, help="Parallel docker workers")
    p.add_argument("--out", type=str, default=None,
                   help="Output CSV (default: datasets/dataset_<model>.csv)")
    p.add_argument("--log", type=str, default=None,
                   help="Per-sim log CSV (default: datasets/runs_log_<model>.csv)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_full_batch(config, n: int, workers: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = sample_lhs(n, config.lever_ranges,
                         validator=config.sample_validator,
                         seed=seed)
    print(f"[batch] {config.name}: launching {n} simulations on {workers} workers")

    started = time.perf_counter()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_simulation, config, sim_id=i + 1,
                        params=samples[i], runs_dir=RUNS_DIR): i + 1
            for i in range(n)
        }
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            tag = "ok" if res["ok"] else f"FAIL ({res['error']})"
            print(f"  {config.name}_sim_{res['sim_id']:04d}: {tag} in {res['runtime_s']:.1f}s")

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
        row = {
            "sim_id": r["sim_id"],
            "ok": r["ok"],
            "error": r["error"] or "",
            "runtime_s": r["runtime_s"],
            **{k: params.get(k, float("nan")) for k in config.lever_ranges},
            "fpr_min_psi": fpr_min,
            "fpr_max_psi": fpr_max,
            "fpr_range_psi": fpr_max - fpr_min if r["ok"] else float("nan"),
        }
        log_rows.append(row)
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
            f"FPR variance across batch is small ({fpr_global_range:.0f} psi)"
        )
    return errors


def main() -> int:
    args = parse_args()
    RUNS_DIR.mkdir(exist_ok=True)
    config = get_config(args.model)

    dataset, log_df = run_full_batch(config, args.n, args.workers, args.seed)

    out = args.out or f"datasets/dataset_{config.name}.csv"
    log = args.log or f"datasets/runs_log_{config.name}.csv"
    out_path = (PROJECT_ROOT / out) if not Path(out).is_absolute() else Path(out)
    log_path = (PROJECT_ROOT / log) if not Path(log).is_absolute() else Path(log)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    if not dataset.empty:
        print("\n[summary] dataset shape:", dataset.shape)
        print("[summary] FPR range across batch:",
              f"{dataset['Presion_Reservorio_psi'].min():.0f} -> {dataset['Presion_Reservorio_psi'].max():.0f} psi")
        print("[summary] simulations included:", dataset["sim_id"].nunique())
    return 0


if __name__ == "__main__":
    sys.exit(main())
