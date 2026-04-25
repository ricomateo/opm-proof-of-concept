"""Per-simulation worker: render deck, run OPM Flow in Docker, parse summary."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd

from deck_template import (
    BASELINE_DECK,
    INCLUDE_FILES,
    SPE9_DIR,
    DeckParams,
    load_baseline,
    render_deck,
)
from extractor import extract_features

DOCKER_IMAGE = "openporousmedia/opmreleases:latest"
FLOW_TIMEOUT_SECONDS = 600
DECK_NAME = "SPE9.DATA"


def run_simulation(
    sim_id: int,
    params: dict,
    runs_dir: Path,
    keep_outputs: bool = False,
) -> dict:
    sim_dir = runs_dir / f"sim_{sim_id:04d}"
    sim_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        deck_params = DeckParams(
            qwinj_rate=params["qwinj_rate"],
            qo_rate_high=params["qo_rate_high"],
            qo_rate_low=params["qo_rate_low"],
            k_mult=params["k_mult"],
            phi_mult=params["phi_mult"],
            p_init=params["p_init"],
            pb_shift=params["pb_shift"],
        )
        rendered = render_deck(load_baseline(), deck_params)
        (sim_dir / DECK_NAME).write_text(rendered)
        for include in INCLUDE_FILES:
            shutil.copyfile(SPE9_DIR / include, sim_dir / include)

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

        summary_base = sim_dir / "SPE9"
        df = extract_features(summary_base, sim_id, params)
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
