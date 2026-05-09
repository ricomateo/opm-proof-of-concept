"""Per-simulation worker for Norne: render deck, run OPM Flow, parse summary.

Differences vs the SPE9 runner:
  - Recursively copies the entire `norne/` tree into the per-sim dir because
    the deck includes use relative paths (`./INCLUDE/...`).
  - Overwrites two files in the per-sim copy: the main deck and the EQUIL
    include (the only include that varies per simulation).
  - Calls the Norne extractor and Norne templater.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from deck_template_norne import (
    EQUIL_INCLUDE_REL,
    NORNE_DIR,
    NorneDeckParams,
    load_baseline,
    render_deck,
)
from extractor_norne import extract_features_norne

DOCKER_IMAGE = "openporousmedia/opmreleases:latest"
FLOW_TIMEOUT_SECONDS = 1800
DECK_NAME = "NORNE_ATW2013.DATA"


def run_simulation_norne(
    sim_id: int,
    params: dict,
    runs_dir: Path,
    keep_outputs: bool = False,
) -> dict:
    sim_dir = runs_dir / f"norne_sim_{sim_id:04d}"
    if sim_dir.exists():
        shutil.rmtree(sim_dir)
    started = time.perf_counter()
    try:
        # Recursive copy of the Norne tree (~30 MB)
        shutil.copytree(NORNE_DIR, sim_dir)

        deck_params = NorneDeckParams(
            k_mult=params["k_mult"],
            phi_mult=params["phi_mult"],
            p_init_shift_bar=params["p_init_shift_bar"],
        )
        baseline_deck, baseline_equil = load_baseline()
        new_deck, new_equil = render_deck(baseline_deck, baseline_equil, deck_params)
        (sim_dir / DECK_NAME).write_text(new_deck)
        (sim_dir / EQUIL_INCLUDE_REL).write_text(new_equil)

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{sim_dir.resolve()}:/shared_host",
            DOCKER_IMAGE,
            "flow",
            "--output-dir=/shared_host",
            f"/shared_host/{DECK_NAME}",
        ]
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=FLOW_TIMEOUT_SECONDS,
        )
        if proc.returncode != 0:
            _persist_failure_log(sim_dir, proc.stdout, proc.stderr)
            return {
                "sim_id": sim_id,
                "ok": False,
                "error": f"docker exit {proc.returncode}",
                "runtime_s": time.perf_counter() - started,
                "df": None,
                "params": params,
            }

        summary_base = sim_dir / "NORNE_ATW2013"
        df = extract_features_norne(summary_base, sim_id, params)
        runtime = time.perf_counter() - started
        return {
            "sim_id": sim_id,
            "ok": True,
            "error": None,
            "runtime_s": runtime,
            "df": df,
            "params": params,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "sim_id": sim_id,
            "ok": False,
            "error": f"timeout after {exc.timeout:.0f}s",
            "runtime_s": time.perf_counter() - started,
            "df": None,
            "params": params,
        }
    except Exception as exc:
        return {
            "sim_id": sim_id,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_s": time.perf_counter() - started,
            "df": None,
            "params": params,
        }
    finally:
        if not keep_outputs and sim_dir.exists():
            shutil.rmtree(sim_dir, ignore_errors=True)


def _persist_failure_log(sim_dir: Path, stdout: str, stderr: str) -> None:
    log_path = sim_dir.parent / f"{sim_dir.name}.failed.log"
    tail_lines = 300
    body = (
        "=== STDOUT (tail) ===\n"
        + "\n".join(stdout.splitlines()[-tail_lines:])
        + "\n\n=== STDERR (tail) ===\n"
        + "\n".join(stderr.splitlines()[-tail_lines:])
        + "\n"
    )
    log_path.write_text(body)
