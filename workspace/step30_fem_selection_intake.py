"""Fail-closed intake for FEM physical selections and material density.

The recovered STEP is usable geometry, but raw surface tags are not physical
meaning. This module accepts only a hash-bound selection package with explicit
CAD/independent confirmation, AI5 gauge directions, and an admissible material
source. It does not generate or promote a mesh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from step30_validation_common import canonical_sha256, safe_json_dump, safe_json_load, sha256_file


VERSION = "step30-fem-selection-material-intake-v1"
EXPECTED_STEP_SHA256 = "521fbaf7688ad1f0e063dedb08c360135026ce60e5e371b4a196294b18cfe21d"
_REQUIRED_SELECTIONS = ("ROOT_INTERFACE", "WETTED_SURFACE", "AI5_04", "MATERIAL_DOMAIN")
_ALLOWED_CONFIRMATIONS = frozenset({"cad_named_selection", "independent_cad_review"})
_FORBIDDEN_CONFIRMATIONS = frozenset({"heuristic_surface_tag", "bbox_heuristic", "visual_guess"})


class FEMSelectionIntakeError(ValueError):
    """Raised when a physical-selection package is incomplete or unverifiable."""


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdefABCDEF" for char in value
    ):
        raise FEMSelectionIntakeError(f"{name} must be a SHA-256 digest")
    return value.lower()


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise FEMSelectionIntakeError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FEMSelectionIntakeError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise FEMSelectionIntakeError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _file_record(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FEMSelectionIntakeError(f"{name} must be an object")
    path_value = value.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        raise FEMSelectionIntakeError(f"{name}.path is missing")
    target = Path(path_value).resolve()
    if not target.is_file():
        raise FEMSelectionIntakeError(f"{name} file does not exist: {target}")
    digest = _hash(value.get("sha256"), f"{name}.sha256")
    byte_count = value.get("bytes")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise FEMSelectionIntakeError(f"{name}.bytes is missing or invalid")
    if target.stat().st_size != byte_count or sha256_file(target) != digest:
        raise FEMSelectionIntakeError(f"{name} hash/byte metadata mismatch")
    return {"path": str(target), "sha256": digest, "bytes": byte_count}


def _tags(value: Any, name: str, dimension: int) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or not value:
        raise FEMSelectionIntakeError(f"{name}.entity_tags must be a non-empty list")
    result: list[int] = []
    for index, raw in enumerate(value):
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise FEMSelectionIntakeError(f"{name}.entity_tags[{index}] must be a positive integer")
        result.append(raw)
    if len(set(result)) != len(result):
        raise FEMSelectionIntakeError(f"{name}.entity_tags must be unique")
    if dimension not in {2, 3}:
        raise FEMSelectionIntakeError(f"{name}.entity_dimension must be 2 or 3")
    return sorted(result)


def _unit_vector(value: Any, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or len(value) != 3:
        raise FEMSelectionIntakeError(f"{name} must be a 3-vector")
    vector = np.asarray([_number(item, f"{name}[{index}]") for index, item in enumerate(value)], dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise FEMSelectionIntakeError(f"{name} must be nonzero")
    normalized = vector / norm
    return [float(item) for item in normalized]


def _selection(value: Any, name: str, *, expected_dimension: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FEMSelectionIntakeError(f"selection {name} is missing")
    item = dict(value)
    if item.get("status") != "passed":
        raise FEMSelectionIntakeError(f"selection {name} must have status=passed")
    confirmation = str(item.get("confirmation_method", "")).strip()
    if confirmation in _FORBIDDEN_CONFIRMATIONS or confirmation not in _ALLOWED_CONFIRMATIONS:
        raise FEMSelectionIntakeError(f"selection {name} lacks admissible CAD confirmation")
    if item.get("independent_confirmation") is not True:
        raise FEMSelectionIntakeError(f"selection {name} needs independent_confirmation=true")
    dimension = item.get("entity_dimension")
    if dimension != expected_dimension:
        raise FEMSelectionIntakeError(f"selection {name} must use entity_dimension={expected_dimension}")
    tags = _tags(item.get("entity_tags"), name, dimension)
    evidence = item.get("evidence_files")
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes, bytearray)) or not evidence:
        raise FEMSelectionIntakeError(f"selection {name}.evidence_files is missing")
    checked = [_file_record(value, f"{name}.evidence_files[{index}]") for index, value in enumerate(evidence)]
    return {**item, "entity_tags": tags, "evidence_files": checked}


def validate_selection_package(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Validate a completed selection/material package and return normalized data."""
    if isinstance(value, Mapping):
        document = dict(value)
    else:
        path = Path(value).resolve()
        loaded = safe_json_load(path)
        if not isinstance(loaded, Mapping):
            raise FEMSelectionIntakeError("selection package must be an object")
        document = dict(loaded)
    stored = document.get("package_sha256")
    unsigned = dict(document)
    unsigned.pop("package_sha256", None)
    if _hash(stored, "package_sha256") != canonical_sha256(unsigned):
        raise FEMSelectionIntakeError("selection package canonical hash mismatch")
    if document.get("version") != VERSION:
        raise FEMSelectionIntakeError("unsupported FEM selection package version")
    if document.get("status") != "passed":
        raise FEMSelectionIntakeError("selection package is not passed")

    geometry = document.get("geometry")
    if not isinstance(geometry, Mapping):
        raise FEMSelectionIntakeError("geometry binding is missing")
    geometry = dict(geometry)
    geometry_file = _file_record(geometry.get("artifact"), "geometry.artifact")
    if geometry_file["sha256"] != EXPECTED_STEP_SHA256:
        raise FEMSelectionIntakeError("geometry does not bind the recovered authoritative STEP")
    if geometry.get("coordinate_units") not in {"m", "mm"}:
        raise FEMSelectionIntakeError("geometry.coordinate_units must be exactly m or mm")

    selections = document.get("selections")
    if not isinstance(selections, Mapping) or set(selections) != set(_REQUIRED_SELECTIONS):
        raise FEMSelectionIntakeError("all four required physical selections must be present")
    root = _selection(selections["ROOT_INTERFACE"], "ROOT_INTERFACE", expected_dimension=2)
    wetted = _selection(selections["WETTED_SURFACE"], "WETTED_SURFACE", expected_dimension=2)
    ai5 = _selection(selections["AI5_04"], "AI5_04", expected_dimension=2)
    material_domain = _selection(selections["MATERIAL_DOMAIN"], "MATERIAL_DOMAIN", expected_dimension=3)
    root_tags = set(root["entity_tags"])
    wetted_tags = set(wetted["entity_tags"])
    ai5_tags = set(ai5["entity_tags"])
    if root_tags & wetted_tags:
        raise FEMSelectionIntakeError("ROOT_INTERFACE and WETTED_SURFACE selections overlap")
    if root_tags & ai5_tags:
        raise FEMSelectionIntakeError("AI5_04 selection overlaps fixed ROOT_INTERFACE")
    if wetted.get("excludes_root_interface") is not True:
        raise FEMSelectionIntakeError("WETTED_SURFACE must explicitly exclude ROOT_INTERFACE")
    if ai5.get("outside_fixed_root") is not True:
        raise FEMSelectionIntakeError("AI5_04 must explicitly be outside fixed root nodes")

    coordinates = ai5.get("patch_center")
    if not isinstance(coordinates, Mapping):
        raise FEMSelectionIntakeError("AI5_04.patch_center is missing")
    if coordinates.get("units") != geometry["coordinate_units"]:
        raise FEMSelectionIntakeError("AI5_04 patch units differ from geometry units")
    patch_center = [_number(item, f"AI5_04.patch_center.coordinates[{index}]") for index, item in enumerate(coordinates.get("coordinates", []))]
    if len(patch_center) != 3:
        raise FEMSelectionIntakeError("AI5_04.patch_center.coordinates must contain three values")
    directions = ai5.get("gauge_directions")
    if not isinstance(directions, Sequence) or isinstance(directions, (str, bytes, bytearray)) or len(directions) not in {1, 2}:
        raise FEMSelectionIntakeError("AI5_04.gauge_directions must contain one or two directions")
    normalized_directions = [_unit_vector(item, f"AI5_04.gauge_directions[{index}]") for index, item in enumerate(directions)]
    if ai5.get("direction_frame") not in {"step_global", "cad_local"}:
        raise FEMSelectionIntakeError("AI5_04.direction_frame must be step_global or cad_local")

    material = document.get("material")
    if not isinstance(material, Mapping):
        raise FEMSelectionIntakeError("material source is missing")
    material = dict(material)
    for field in ("grade", "heat_or_batch", "source_kind"):
        if not isinstance(material.get(field), str) or not material[field].strip():
            raise FEMSelectionIntakeError(f"material.{field} is missing")
    if material["source_kind"] not in {"material_certificate", "official_design_record"}:
        raise FEMSelectionIntakeError("material source_kind is not admissible for formal S1")
    density = _number(material.get("density_kg_m3"), "material.density_kg_m3", positive=True)
    material_source = _file_record(material.get("source_artifact"), "material.source_artifact")
    normalized_material = {**material, "density_kg_m3": density, "source_artifact": material_source}
    normalized = {
        **document,
        "geometry": {**geometry, "artifact": geometry_file},
        "selections": {
            "ROOT_INTERFACE": root,
            "WETTED_SURFACE": wetted,
            "AI5_04": {**ai5, "patch_center": {**coordinates, "coordinates": patch_center}, "gauge_directions": normalized_directions},
            "MATERIAL_DOMAIN": material_domain,
        },
        "material": normalized_material,
    }
    return normalized


def write_template(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    template = {
        "version": VERSION,
        "status": "blocked",
        "geometry": {
            "coordinate_units": "m",
            "artifact": {"path": "ABSOLUTE_PATH_TO_SINGLE_BUCKET_TARGET_AUTHORITATIVE.step", "sha256": EXPECTED_STEP_SHA256, "bytes": 0},
        },
        "selections": {
            "ROOT_INTERFACE": {"status": "blocked", "confirmation_method": "cad_named_selection", "independent_confirmation": False, "entity_dimension": 2, "entity_tags": [], "evidence_files": []},
            "WETTED_SURFACE": {"status": "blocked", "confirmation_method": "cad_named_selection", "independent_confirmation": False, "entity_dimension": 2, "entity_tags": [], "excludes_root_interface": False, "evidence_files": []},
            "AI5_04": {"status": "blocked", "confirmation_method": "cad_named_selection", "independent_confirmation": False, "entity_dimension": 2, "entity_tags": [], "patch_center": {"coordinates": [], "units": "m"}, "direction_frame": "step_global", "gauge_directions": [], "outside_fixed_root": False, "evidence_files": []},
            "MATERIAL_DOMAIN": {"status": "blocked", "confirmation_method": "cad_named_selection", "independent_confirmation": False, "entity_dimension": 3, "entity_tags": [], "evidence_files": []},
        },
        "material": {"grade": "", "heat_or_batch": "", "source_kind": "material_certificate", "density_kg_m3": None, "source_artifact": {"path": "ABSOLUTE_PATH_TO_MATERIAL_CERTIFICATE.pdf", "sha256": "", "bytes": 0}},
        "package_sha256": "",
    }
    unsigned = dict(template)
    unsigned.pop("package_sha256")
    template["package_sha256"] = canonical_sha256(unsigned)
    safe_json_dump(template, target)
    return {"path": str(target), "sha256": sha256_file(target)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate FEM CAD selections and material density evidence")
    parser.add_argument("--package", type=Path)
    parser.add_argument("--template", type=Path)
    args = parser.parse_args(argv)
    if args.template:
        print(json.dumps(write_template(args.template), indent=2))
        return 0
    if not args.package:
        parser.error("provide --package or --template")
    try:
        result = validate_selection_package(args.package)
    except (FEMSelectionIntakeError, OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "blocked", "fem_gate": "blocked", "reason": str(exc)}, indent=2))
        return 3
    print(json.dumps({"status": "passed", "fem_gate": "ready_for_s0", "package_sha256": result["package_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
