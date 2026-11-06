"""Immutable identity helpers for Step30 CFD and particle artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


ARTIFACT_DESCRIPTOR_VERSION = "step30-artifact-set-descriptor-v2"
PARTICLE_DESCRIPTOR_VERSION = "step30-particle-run-descriptor-v2"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(value: Any) -> str:
    """Hash structured data with stable key ordering and JSON encoding."""
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _solver_semantics(cfd: dict) -> dict:
    manifest = cfd.get("artifact_manifest") or {}
    if manifest.get("version") != "step30-native-si-manifest-v2":
        raise ValueError("immutable artifact identity requires a native SI manifest v2")
    semantics = manifest.get("solver_semantics", {})
    required = {
        "application",
        "mode",
        "field_statistic",
        "tracking_velocity_object",
        "averaging_window",
    }
    missing = sorted(required - set(semantics))
    if missing:
        raise ValueError(f"solver semantics missing fields: {missing}")
    statistic = semantics["field_statistic"]
    velocity_object = semantics["tracking_velocity_object"]
    averaging_window = semantics["averaging_window"]
    if statistic == "snapshot":
        if (
            semantics["application"] != "simpleFoam"
            or semantics["mode"] != "steady_rans"
            or velocity_object != "U"
            or averaging_window is not None
        ):
            raise ValueError("snapshot artifacts require simpleFoam/steady_rans/U and no averaging window")
    elif statistic == "time_mean":
        if (
            semantics["application"] != "pimpleFoam"
            or semantics["mode"] != "urans"
            or velocity_object != "UMean"
            or not isinstance(averaging_window, dict)
        ):
            raise ValueError("time-mean artifacts require pimpleFoam/urans/UMean and an averaging window")
        for key in ("start_time", "end_time", "sample_count"):
            if key not in averaging_window:
                raise ValueError(f"averaging window missing {key}")
        if int(averaging_window["sample_count"]) <= 0:
            raise ValueError("averaging sample count must be positive")
    else:
        raise ValueError("field_statistic must be 'snapshot' or 'time_mean'")
    return semantics


def build_artifact_set_descriptor(
    cfd: dict,
    topology: dict,
    topology_path: Path,
) -> dict:
    """Describe one immutable CFD field/topology pairing."""
    manifest = cfd.get("artifact_manifest") or {}
    field_hashes = {
        name: entry.get("sha256")
        for name, entry in sorted(manifest.get("fields", {}).items())
        if isinstance(entry, dict) and entry.get("sha256")
    }
    descriptor = {
        "version": ARTIFACT_DESCRIPTOR_VERSION,
        "cfd": {
            "variant": cfd["variant"],
            "field_time": str(cfd["field_time"]),
            "manifest_version": manifest.get("version"),
            "manifest_sha256": cfd.get("artifact_manifest_sha256"),
            "polyMesh_signature": manifest.get("polyMesh", {}).get("signature"),
            "field_sha256": field_hashes,
            "solver_semantics": _solver_semantics(cfd),
        },
        "topology": {
            "file_sha256": file_sha256(topology_path),
            "artifact_version": topology.get("artifact_version"),
            "mesh_signature_algorithm": topology.get("mesh_signature_algorithm"),
            "mesh_signature": topology["mesh_signature"],
            "source_vtp_sha256": topology.get("source_vtp_sha256"),
            "patch_name": topology["patch_name"],
            "time": str(topology["time_name"]),
            "n_parent_faces": int(topology["n_parent_faces"]),
        },
        "units": {
            "source_coordinate_units": cfd["source_coordinate_units"],
            "internal_coordinate_units": cfd["internal_coordinate_units"],
            "coordinate_scale_to_mm": float(cfd["coordinate_scale_to_mm"]),
        },
    }
    if descriptor["topology"]["time"] != descriptor["cfd"]["field_time"]:
        raise ValueError("artifact topology and CFD field times differ")
    descriptor["artifact_set_id"] = canonical_sha256(descriptor)
    return descriptor


def build_particle_run_descriptor(
    artifact_set_id: str,
    config: Any,
    *,
    collision_backend: str,
    implementation_hashes: dict[str, str] | None = None,
    input_hashes: dict[str, str] | None = None,
    implementation_sha256: str | None = None,
) -> dict:
    """Describe every setting and external input that can affect one particle result."""
    if implementation_hashes is None:
        if implementation_sha256 is None:
            raise ValueError("particle identity requires implementation hashes")
        implementation_hashes = {"particle_tracking": implementation_sha256}
    if not implementation_hashes or not all(implementation_hashes.values()):
        raise ValueError("particle identity has an empty implementation hash")
    if input_hashes is None:
        input_hashes = {}
    if not all(input_hashes.values()):
        raise ValueError("particle identity has an empty input hash")
    values = asdict(config) if is_dataclass(config) else dict(config)
    values.pop("output_tag", None)
    values.pop("immutable_output", None)
    values.pop("topology_path", None)
    values.pop("native_si_directory", None)
    values.pop("analytic_reference_transform_path", None)
    descriptor = {
        "version": PARTICLE_DESCRIPTOR_VERSION,
        "artifact_set_id": artifact_set_id,
        "collision_backend": collision_backend,
        "implementation_hashes": dict(sorted(implementation_hashes.items())),
        "input_hashes": dict(sorted(input_hashes.items())),
        "config": values,
    }
    descriptor["particle_run_id"] = canonical_sha256(descriptor)
    return descriptor


def validate_i1_i3_descriptors(single: dict, multiple: dict) -> None:
    """Require immutable i1/i3 descriptors to differ only in max_impacts."""
    if single.get("version") != PARTICLE_DESCRIPTOR_VERSION or multiple.get("version") != PARTICLE_DESCRIPTOR_VERSION:
        raise ValueError("unsupported particle run descriptor version")
    if single.get("artifact_set_id") != multiple.get("artifact_set_id"):
        raise ValueError("i1/i3 artifact_set_id values differ")
    single_config = dict(single.get("config", {}))
    multiple_config = dict(multiple.get("config", {}))
    if single_config.pop("max_impacts", None) != 1 or multiple_config.pop("max_impacts", None) != 3:
        raise ValueError("i1/i3 descriptors require max_impacts 1 and 3")
    single_copy = dict(single)
    multiple_copy = dict(multiple)
    single_copy.pop("particle_run_id", None)
    multiple_copy.pop("particle_run_id", None)
    single_copy["config"] = single_config
    multiple_copy["config"] = multiple_config
    if single_copy != multiple_copy:
        raise ValueError("i1/i3 particle descriptors differ beyond max_impacts")
