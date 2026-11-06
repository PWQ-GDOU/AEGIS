"""Validate mean-field presence and finite values for a staged OpenFOAM case."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def read_internal(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"internalField\s+nonuniform\s+List<(vector|scalar)>\s+(\d+)\s*\((.*?)\)\s*;", text, re.S)
    if not match:
        raise ValueError(f"cannot parse internal field: {path}")
    kind, count, body = match.groups()
    values = np.fromstring(body.replace("(", " ").replace(")", " "), sep=" ")
    if kind == "vector":
        values = values.reshape(-1, 3)
    if len(values) != int(count):
        raise ValueError(f"field count mismatch: {path}")
    return values


def read_bucket_boundary_vector(path: Path) -> np.ndarray:
    """Read a calculated bucket boundary vector field such as wall shear."""
    text = path.read_text(encoding="utf-8", errors="replace")
    patch = re.search(r"\bbucket\s*\{(?P<body>.*?)\n\s*\}", text, re.S)
    if not patch:
        raise ValueError(f"bucket boundary is missing: {path}")
    match = re.search(r"value\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;", patch.group("body"), re.S)
    if not match:
        raise ValueError(f"bucket boundary vector is missing: {path}")
    values = np.fromstring(match.group(2).replace("(", " ").replace(")", " "), sep=" ").reshape(-1, 3)
    if len(values) != int(match.group(1)):
        raise ValueError(f"bucket boundary count mismatch: {path}")
    return values


def validate(case: Path, field_time: str, *, require: tuple[str, ...] = ("UMean", "pMean", "kMean", "omegaMean", "wallShearStressMean")) -> dict:
    directory = case / field_time
    fields = {}
    errors = []
    for name in require:
        path = directory / name
        if not path.exists():
            errors.append(f"missing field: {name}")
            continue
        try:
            values = read_bucket_boundary_vector(path) if name == "wallShearStressMean" else read_internal(path)
        except Exception as exc:
            errors.append(f"unreadable field {name}: {exc}")
            continue
        fields[name] = {"count": int(len(values)), "finite": bool(np.isfinite(values).all()), "min": float(np.min(values)), "max": float(np.max(values))}
        if not np.isfinite(values).all():
            errors.append(f"nonfinite field: {name}")
    return {"version": "step30-mean-field-validation-v1", "case": str(case), "field_time": field_time, "status": "passed" if not errors else "failed", "required_fields": list(require), "fields": fields, "errors": errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--field-time", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.case, args.field_time)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 2)
