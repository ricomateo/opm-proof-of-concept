"""Per-simulation worker for Volve: render deck, run OPM Flow, parse summary.

Differences vs the Norne runner:
  - Single file overwritten in the per-sim copy (the main deck). Volve's
    EQUIL is inline in the deck, so no second-file rewrite is needed.
  - Longer timeout: baseline run takes ~21 min, generous parameterizations
    can push past 30 min.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from deck_template_volve import (
    VOLVE_DIR,
    VolveDeckParams,
    load_baseline,
    render_deck,
)
from extractor_volve import extract_features_volve

DOCKER_IMAGE = "openporousmedia/opmreleases:latest"
FLOW_TIMEOUT_SECONDS = 5400
DECK_NAME = "VOLVE_2016.DATA"


def run_simulation_volve(
    sim_id: int,
    params: dict,
    runs_dir: Path,
    keep_outputs: bool = False,
) -> dict:
    sim_dir = runs_dir / f"volve_sim_{sim_id:04d}"
    if sim_dir.exists():
        shutil.rmtree(sim_dir)
    started = time.perf_counter()
    try:
        # Recursive copy of the Volve tree (~167 MB committed; the per-sim
        # copy uses APFS clones via cp -c when available).
        shutil.copytree(VOLVE_DIR, sim_dir)

        deck_params = VolveDeckParams(
            k_mult=params["k_mult"],
            phi_mult=params["phi_mult"],
            p_init_shift_bar=params["p_init_shift_bar"],
            qwinj_group_mult=params["qwinj_group_mult"],
        )
        rendered = render_deck(load_baseline(), deck_params)
        (sim_dir / DECK_NAME).write_bytes(rendered.encode("latin-1"))

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

        summary_base = sim_dir / "VOLVE_2016"
        df = extract_features_volve(summary_base, sim_id, params)
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
