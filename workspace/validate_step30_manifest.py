"""Validate Step30 native-SI manifests and immutable run descriptors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from step30_artifact_identity import canonical_sha256, file_sha256


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def validate_artifact_manifest(path: Path) -> dict[str, Any]:
    data = _load(path)
    errors: list[str] = []
    if data.get("version") != "step30-artifact-set-descriptor-v2":
        errors.append("artifact set descriptor version mismatch")
    cfd = data.get("cfd", {})
    topology = data.get("topology", {})
    units = data.get("units", {})
    semantics = cfd.get("solver_semantics", {})
    if cfd.get("variant") != "native_si":
        errors.append("artifact set is not native_si")
    if semantics.get("field_statistic") == "time_mean":
        if semantics.get("application") != "pimpleFoam" or semantics.get("mode") != "urans":
            errors.append("time_mean must use pimpleFoam/urans")
        if semantics.get("tracking_velocity_object") != "UMean":
            errors.append("time_mean must track UMean")
        if not isinstance(semantics.get("averaging_window"), dict):
            errors.append("time_mean averaging window missing")
    elif semantics.get("field_statistic") == "snapshot":
        if semantics.get("application") != "simpleFoam" or semantics.get("tracking_velocity_object") != "U":
            errors.append("snapshot must use simpleFoam/U")
    else:
        errors.append("unknown field statistic")
    if units != {"source_coordinate_units": "m", "internal_coordinate_units": "mm", "coordinate_scale_to_mm": 1000.0}:
        errors.append("SI coordinate contract mismatch")
    if topology.get("artifact_version") != "step30-native-openfoam-topology-v3":
        errors.append("topology version mismatch")
    if topology.get("field_time", topology.get("time")) != cfd.get("field_time"):
        errors.append("field/topology time mismatch")
    expected_id = dict(data)
    expected_id.pop("artifact_set_id", None)
    if data.get("artifact_set_id") != canonical_sha256(expected_id):
        errors.append("artifact_set_id does not match descriptor content")
    return {"path": str(path), "status": "passed" if not errors else "failed", "errors": errors, "artifact_set_id": data.get("artifact_set_id"), "descriptor": data}


def validate_particle_run(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "particle_run_manifest.json"
    result: dict[str, Any] = {"run_dir": str(run_dir), "status": "passed", "errors": []}
    if not manifest_path.exists():
        result.update(status="failed", errors=["particle_run_manifest.json missing"])
        return result
    data = _load(manifest_path)
    errors = result["errors"]
    required = {"version", "artifact_set_id", "collision_backend", "implementation_hashes", "input_hashes", "config", "particle_run_id", "outputs"}
    errors.extend([f"missing {key}" for key in sorted(required - set(data))])
    if data.get("version") != "step30-particle-run-descriptor-v2":
        errors.append("particle descriptor version mismatch")
    unhashed = dict(data)
    unhashed.pop("particle_run_id", None)
    unhashed.pop("outputs", None)
    if data.get("particle_run_id") != canonical_sha256(unhashed):
        errors.append("particle_run_id does not match descriptor content")
    config = data.get("config", {})
    if config.get("collision_policy") != "strict_v2":
        errors.append("formal validation requires strict_v2")
    if data.get("collision_backend") != "cfd_topology" or config.get("cfd_variant") != "native_si":
        errors.append("formal validation requires cfd_topology/native_si")
    for name, entry in (data.get("outputs") or {}).items():
        if not isinstance(entry, dict) or "file" not in entry or "sha256" not in entry:
            errors.append(f"invalid output entry: {name}")
            continue
        target = run_dir / entry["file"]
        if not target.exists():
            errors.append(f"missing output: {target.name}")
        elif file_sha256(target) != entry["sha256"]:
            errors.append(f"output hash mismatch: {target.name}")
    result["status"] = "passed" if not errors else "failed"
    result["particle_run_id"] = data.get("particle_run_id")
    result["artifact_set_id"] = data.get("artifact_set_id")
    result["config"] = config
    return result


def validate(path: Path) -> dict[str, Any]:
    if path.name == "artifact_set_manifest.json":
        return validate_artifact_manifest(path)
    return validate_particle_run(path if path.is_dir() else path.parent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    report = validate(args.path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 2)
