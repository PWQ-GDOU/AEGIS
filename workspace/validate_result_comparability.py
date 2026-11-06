"""Fail-closed comparability checks for Step30 experiment arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def comparable(left: dict, right: dict, *, allowed_changes: set[str] | None = None) -> dict:
    allowed_changes = allowed_changes or set()
    keys = set(left) | set(right)
    differences = {}
    for key in sorted(keys):
        if key in allowed_changes:
            continue
        if left.get(key) != right.get(key):
            differences[key] = {"left": left.get(key), "right": right.get(key)}
    checks = {
        "artifact_set_id": left.get("artifact_set_id") == right.get("artifact_set_id"),
        "mesh_signature": left.get("mesh_signature") == right.get("mesh_signature"),
        "collision_policy": left.get("collision_policy") == right.get("collision_policy"),
        "seed": left.get("seed") == right.get("seed"),
        "n_parcels": left.get("n_parcels") == right.get("n_parcels"),
        "tracking_velocity": left.get("tracking_velocity") == right.get("tracking_velocity"),
        "dt": ("dt" in allowed_changes) or left.get("dt") == right.get("dt"),
    }
    checks["no_unapproved_differences"] = not differences
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "differences": differences}


def _descriptor(path: Path) -> dict:
    data = _load(path)
    config = data.get("config", data.get("parameters", data))
    summary = data.get("summary", {})
    return {
        "artifact_set_id": data.get("artifact_set_id", data.get("artifact_identity", {}).get("artifact_set_id")),
        "mesh_signature": data.get("mesh_signature", data.get("geometry", {}).get("mesh_signature")),
        "collision_policy": config.get("collision_policy", data.get("collision_policy")),
        "seed": config.get("seed", data.get("seed")),
        "n_parcels": config.get("n_parcels", data.get("n_parcels")),
        "tracking_velocity": data.get("tracking_velocity", summary.get("cfd", {}).get("solver_semantics", {}).get("tracking_velocity_object")),
        "dt": config.get("dt_s", data.get("dt_s")),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--allow", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = comparable(_descriptor(args.left), _descriptor(args.right), allowed_changes=set(args.allow))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    raise SystemExit(0 if result["status"] == "passed" else 2)
