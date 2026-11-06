"""Aggregate the non-overlapping v5 confirmation runs on a remote host.

This is an orchestration script only. It reads the DCU run-status manifest and
delegates all evaluator work to the existing v5 evaluator on that server.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_STATUS = Path("output/step30_v5_confirmation_parallel_v1/run_status.json")
DEFAULT_DISCOVERY = Path(
    "output/step30_particle_convergence_5k_10k_20k_topology_aggregate_v5_discovery_r2.json"
)
DEFAULT_TOPOLOGY = Path(
    "output/step30_acceptance_0.8684190921/bucket_topology_v3_0.8684190921.npz"
)
DEFAULT_OUTPUT = Path(
    "output/step30_particle_convergence_20k_confirmation_20260830_v2.json"
)
DEFAULT_FIELDS = Path(
    "output/step30_particle_surface_aggregate_v5_confirmation_20260830_v2"
)


def load_status(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "completed":
        raise RuntimeError("confirmation run status is not completed")
    return value


def select_runs(status: dict, *, minimum: int = 5) -> list[Path]:
    rows = status.get("seeds")
    if not isinstance(rows, dict):
        raise RuntimeError("run status has no seed records")
    selected: list[Path] = []
    for seed_text, row in sorted(rows.items(), key=lambda item: int(item[0])):
        if not isinstance(row, dict) or row.get("status") != "completed":
            continue
        directory = row.get("run_directory")
        if not isinstance(directory, str) or not directory:
            raise RuntimeError(f"completed seed {seed_text} has no run_directory")
        selected.append(Path(directory))
    if len(selected) < minimum:
        raise RuntimeError(f"only {len(selected)} completed confirmation runs; need {minimum}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--discovery-report", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fields-directory", type=Path, default=DEFAULT_FIELDS)
    parser.add_argument("--ensemble-block-size", type=int, default=1)
    args = parser.parse_args()

    status = load_status(args.status)
    runs = select_runs(status)
    command = [
        sys.executable,
        "step30_particle_convergence_v5.py",
        "--analysis-role",
        "confirmation",
        "--discovery-report",
        str(args.discovery_report),
        "--topology",
        str(args.topology),
        "--output",
        str(args.output),
        "--fields-directory",
        str(args.fields_directory),
    ]
    for run in runs:
        command.extend(["--run", str(run)])
    command.extend(["--ensemble-block-size", str(args.ensemble_block_size)])
    print(json.dumps({"selected_confirmation_runs": [str(path) for path in runs], "command": command}, indent=2))
    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
