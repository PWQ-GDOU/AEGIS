"""Evaluate collision-parameter sensitivity from immutable particle arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from step30_particle_parameter_sensitivity import run as _run


def run(arms, output: Path) -> dict:
    result = _run(arms, output)
    result["version"] = "step30-collision-parameter-sensitivity-v1"
    result["variable"] = "collision_parameters"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _parse(value: str):
    label, paths = value.split("=", 1)
    npz, summary = paths.split(",", 1)
    return label, Path(npz), Path(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", type=_parse, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.arm, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 2)
