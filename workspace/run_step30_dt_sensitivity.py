"""Run or evaluate the required three-step Step30 timestep matrix.

By default this script evaluates already-produced immutable runs.  Pass
``--execute`` to invoke the particle tracker for each arm; the command is
constructed from explicit paths and all results remain run-local.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from step30_particle_sensitivity import run as evaluate_sensitivity

DT_VALUES = (1.0e-5, 5.0e-6, 2.5e-6)


def _parse_run(value: str) -> tuple[str, Path, Path]:
    label, raw = value.split("=", 1)
    npz, summary = raw.split(",", 1)
    return label, Path(npz), Path(summary)


def run_existing(runs: list[tuple[str, Path, Path]], output: Path) -> dict:
    return evaluate_sensitivity(runs, output)


def run_execute(args: argparse.Namespace) -> dict:
    if not args.native_si_directory or not args.topology:
        raise ValueError("--execute requires --native-si-directory and --topology")
    out_root = args.output.parent / "dt_runs"
    out_root.mkdir(parents=True, exist_ok=True)
    produced = []
    tracker = Path(args.tracker)
    for dt in DT_VALUES:
        tag = f"dt_{dt:.1e}".replace(".", "p").replace("+", "")
        run_dir = out_root / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(tracker), "--cfd", "native_si", "--collision-backend", "cfd_topology",
            "--collision-policy", "strict_v2", "--native-si-directory", str(args.native_si_directory),
            "--topology", str(args.topology), "--parcels", "500", "--max-impacts", str(args.max_impacts),
            "--dt", str(dt), "--strict-max-substep", str(dt), "--immutable-output", "--tag", tag,
        ]
        subprocess.run(command, check=True)
        candidates = sorted(run_dir.parent.parent.glob(f"**/{tag}/summary.json"))
        if not candidates:
            raise FileNotFoundError(f"tracker did not create immutable summary for {tag}")
        summary = candidates[-1]
        produced.append((tag, summary.with_name("particle_hits.npz"), summary))
    return run_existing(produced, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=_parse_run)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--tracker", type=Path, default=Path(__file__).with_name("step30_lagrangian_particle_tracking.py"))
    parser.add_argument("--native-si-directory", type=Path)
    parser.add_argument("--topology", type=Path)
    parser.add_argument("--max-impacts", type=int, default=3)
    args = parser.parse_args()
    if args.execute:
        result = run_execute(args)
    else:
        if not args.run or len(args.run) != 3:
            parser.error("evaluation mode requires exactly three --run LABEL=NPZ,SUMMARY_JSON arguments")
        result = run_existing(args.run, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 2)
