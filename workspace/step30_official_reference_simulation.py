"""Fail-closed comparison against a contest-provided reference simulation.

Contest reference data is synthetic evidence in the taxonomy sense: it is a
simulation supplied by the organizer, not a field wear measurement. This
module binds the comparison to the immutable native-SI artifact set, the
validated topology-v3 NPZ, and a hash-bound observation artifact before it
computes any accuracy metric.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from step30_cfd_topology import load_topology_artifact
from step30_validation_common import (
    EvidenceContractError,
    ValidationError,
    canonical_sha256,
    evidence_metadata,
    safe_json_dump,
    safe_json_load,
    sha256_file,
)

VERSION = "step30-official-reference-simulation-v2"
REFERENCE_MANIFEST_VERSION = "step30-official-reference-manifest-v2"
MANIFEST_KIND = "official_reference_package"
EVALUATION_VERSION = "step30-reference-evaluation-contract-v1"
ARTIFACT_DESCRIPTOR_VERSION = "step30-artifact-set-descriptor-v2"
NATIVE_MANIFEST_VERSION = "step30-native-si-manifest-v2"
TOPOLOGY_VERSION = "step30-native-openfoam-topology-v3"
TOPOLOGY_SIGNATURE_ALGORITHM = "sha256-vtk-triangulation-source-units-points-triangles-parent-ids-v3"
EXPECTED_UNITS = {
    "source_coordinate_units": "m",
    "internal_coordinate_units": "mm",
    "coordinate_scale_to_mm": 1000.0,
}
SUPPORTED_UNITS = {"mm", "mm/h", "mm2", "mm3", "kg", "kg/kg", "mm3/kg", "dimensionless"}
METRIC_UNITS = {
    "wear_depth": {"mm"},
    "erosion_rate": {"mm/h"},
    "erosion_area": {"mm2"},
    "erosion_volume": {"mm3"},
    "erosion_mass": {"kg"},
    "erosion_ratio": {"kg/kg", "mm3/kg"},
    "dimensionless": {"dimensionless"},
}
REQUIRED_ERROR_THRESHOLDS = (
    "total_quantity_relative_error",
    "maximum_value_relative_error",
    "hotspot_area_relative_error",
    "hotspot_iou_error",
)


class OfficialReferenceError(ValueError):
    """Raised when contest reference evidence is incomplete or inconsistent."""


def _load(value: Any) -> tuple[Any, Path | None]:
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            raise OfficialReferenceError(f"input does not exist: {path}")
        try:
            return safe_json_load(path), path.resolve()
        except (OSError, UnicodeError, ValidationError) as exc:
            raise OfficialReferenceError(f"invalid JSON input: {path}") from exc
    return value, None


def _hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        c in "0123456789abcdefABCDEF" for c in value
    )


def _canonical_checked(document: Mapping[str, Any], field: str) -> None:
    stored = document.get(field)
    unsigned = dict(document)
    unsigned.pop(field, None)
    if not _hash(stored) or canonical_sha256(unsigned) != str(stored).lower():
        raise OfficialReferenceError(f"{field} is missing or invalid")


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise OfficialReferenceError(f"{label} is not numeric")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise OfficialReferenceError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise OfficialReferenceError(f"{label} is not finite")
    return number


def _records(value: Any, label: str) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        raw = value.get("records", value.get(label, value.get("data")))
        if raw is None and any(key in value for key in ("record_id", "parent_face_id", "value")):
            raw = [value]
    else:
        raw = value
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)) or not raw:
        raise OfficialReferenceError(f"{label} records are missing or empty")
    rows = [dict(item) for item in raw if isinstance(item, Mapping)]
    if len(rows) != len(raw):
        raise OfficialReferenceError(f"{label} records must contain objects")
    return rows


def _number(row: Mapping[str, Any], names: Sequence[str], label: str) -> float:
    value = next((row[name] for name in names if name in row), None)
    return _finite_number(value, label)


def _resolve_root(document: Mapping[str, Any], origin: Path | None) -> Path:
    raw = document.get("source_root")
    if raw is None:
        return (origin.parent if origin else Path.cwd()).resolve()
    if not isinstance(raw, str) or not raw.strip():
        raise OfficialReferenceError("source_root must be a non-empty path")
    root = Path(raw)
    if not root.is_absolute() and origin is not None:
        root = origin.parent / root
    root = root.resolve()
    if not root.is_dir():
        raise OfficialReferenceError(f"source_root does not exist: {root}")
    return root


def _safe_path(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise OfficialReferenceError(f"{label} path is missing")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise OfficialReferenceError(f"{label} path must stay inside source_root")
    target = (root / path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise OfficialReferenceError(f"{label} path escapes source_root") from exc
    if not target.is_file():
        raise OfficialReferenceError(f"{label} does not exist: {raw}")
    return target


def _artifact_entries(document: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    raw = document.get("artifacts")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)) or not raw:
        raise OfficialReferenceError("reference artifacts are missing")
    checked: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_roles: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise OfficialReferenceError("reference artifact entries must be objects")
        role = item.get("role")
        if not isinstance(role, str) or not role.strip():
            raise OfficialReferenceError("reference artifact needs an explicit role")
        declared_path = item.get("path")
        path = _safe_path(root, declared_path, f"artifact/{role}")
        digest = item.get("sha256", item.get("file_sha256"))
        if not _hash(digest):
            raise OfficialReferenceError(f"artifact/{role} needs a valid sha256")
        digest = str(digest).lower()
        if sha256_file(path) != digest:
            raise OfficialReferenceError(f"reference artifact hash mismatch: {declared_path}")
        path_key = path.as_posix()
        if path_key in seen_paths or role in seen_roles:
            raise OfficialReferenceError("reference artifact paths and roles must be unique")
        seen_paths.add(path_key)
        seen_roles.add(role)
        declared_bytes = item.get("bytes")
        if declared_bytes is not None and declared_bytes != path.stat().st_size:
            raise OfficialReferenceError(f"artifact byte count mismatch: {role}")
        checked.append(
            {
                "role": role,
                "declared_path": str(declared_path),
                "path": str(path),
                "sha256": digest,
                "bytes": int(path.stat().st_size),
            }
        )
    return checked


def _binding(
    document: Mapping[str, Any],
    name: str,
    artifacts: Sequence[Mapping[str, Any]],
    roles: Sequence[str],
    root: Path,
) -> tuple[dict[str, Any], Path]:
    raw = document.get(name)
    if not isinstance(raw, Mapping):
        candidates = [item for item in artifacts if item.get("role") in roles]
        if len(candidates) != 1:
            raise OfficialReferenceError(f"{name} binding is missing")
        raw = candidates[0]
    path_value = raw.get("path", raw.get("manifest_path", raw.get("file")))
    digest = raw.get("sha256", raw.get("file_sha256"))
    if not isinstance(path_value, str) or not _hash(digest):
        raise OfficialReferenceError(f"{name} binding needs path and sha256")
    digest = str(digest).lower()
    matches = [
        item
        for item in artifacts
        if item.get("declared_path") == path_value and item.get("sha256") == digest
    ]
    if len(matches) != 1:
        raise OfficialReferenceError(f"{name} binding is not present in artifacts")
    path = _safe_path(root, path_value, name)
    return {**dict(raw), "path": str(path), "declared_path": path_value, "sha256": digest}, path


def _solver_semantics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OfficialReferenceError("solver_semantics must be an object")
    semantics = dict(value)
    required = {
        "application",
        "mode",
        "field_statistic",
        "tracking_velocity_object",
        "averaging_window",
    }
    if set(semantics) != required:
        raise OfficialReferenceError("solver_semantics has incomplete or unexpected fields")
    if semantics["field_statistic"] != "time_mean":
        raise OfficialReferenceError("official reference requires time_mean solver semantics")
    if semantics["application"] != "pimpleFoam" or semantics["mode"] != "urans":
        raise OfficialReferenceError("official reference requires pimpleFoam/urans")
    if semantics["tracking_velocity_object"] != "UMean":
        raise OfficialReferenceError("official reference requires UMean tracking semantics")
    window = semantics["averaging_window"]
    if not isinstance(window, Mapping) or set(window) != {"start_time", "end_time", "sample_count"}:
        raise OfficialReferenceError("time_mean averaging_window is incomplete")
    if (
        isinstance(window["sample_count"], bool)
        or not isinstance(window["sample_count"], int)
        or window["sample_count"] <= 0
    ):
        raise OfficialReferenceError("time_mean sample_count must be a positive integer")
    start = _finite_number(window["start_time"], "averaging_window.start_time")
    end = _finite_number(window["end_time"], "averaging_window.end_time")
    if end <= start:
        raise OfficialReferenceError("averaging_window end_time must exceed start_time")
    return semantics


def _reference_solver_semantics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OfficialReferenceError("reference_solver_semantics must be an object")
    semantics = dict(value)
    required = {"application", "mode", "field_statistic", "value_object"}
    if not required <= set(semantics):
        raise OfficialReferenceError("reference_solver_semantics is incomplete")
    for field in required:
        if not isinstance(semantics.get(field), str) or not semantics[field].strip():
            raise OfficialReferenceError(f"reference_solver_semantics missing {field}")
    if semantics["field_statistic"] not in {"snapshot", "time_mean", "time_integrated", "cumulative"}:
        raise OfficialReferenceError("reference field_statistic is unsupported")
    if semantics["field_statistic"] == "time_mean":
        window = semantics.get("averaging_window")
        if not isinstance(window, Mapping):
            raise OfficialReferenceError("reference time_mean semantics needs averaging_window")
        for field in ("start_time", "end_time", "sample_count"):
            if field not in window:
                raise OfficialReferenceError(f"reference averaging_window missing {field}")
    return semantics


def _validate_native_descriptor(
    path: Path,
    binding: Mapping[str, Any],
    artifact_set_id: str,
) -> dict[str, Any]:
    try:
        native = safe_json_load(path)
    except (OSError, UnicodeError, ValidationError) as exc:
        raise OfficialReferenceError("native artifact-set manifest is invalid") from exc
    if not isinstance(native, Mapping):
        raise OfficialReferenceError("native artifact-set manifest must be an object")
    native = dict(native)
    if native.get("version") != ARTIFACT_DESCRIPTOR_VERSION:
        raise OfficialReferenceError("native artifact-set manifest version is invalid")
    stored_id = native.get("artifact_set_id")
    unsigned = dict(native)
    unsigned.pop("artifact_set_id", None)
    if not _hash(stored_id) or canonical_sha256(unsigned) != str(stored_id).lower():
        raise OfficialReferenceError("native artifact_set_id/content mismatch")
    if str(stored_id).lower() != artifact_set_id:
        raise OfficialReferenceError("reference and native artifact_set_id values differ")
    if binding.get("artifact_set_id", artifact_set_id) != str(stored_id).lower():
        raise OfficialReferenceError("native binding artifact_set_id differs from descriptor")
    cfd = native.get("cfd")
    topology = native.get("topology")
    if not isinstance(cfd, Mapping) or not isinstance(topology, Mapping):
        raise OfficialReferenceError("native artifact-set manifest is incomplete")
    if cfd.get("variant") != "native_si" or cfd.get("manifest_version") != NATIVE_MANIFEST_VERSION:
        raise OfficialReferenceError("native artifact set is not native_si manifest v2")
    if not _hash(cfd.get("manifest_sha256")) or not _hash(cfd.get("polyMesh_signature")):
        raise OfficialReferenceError("native CFD manifest/polyMesh hash is invalid")
    field_hashes = cfd.get("field_sha256")
    if not isinstance(field_hashes, Mapping) or not field_hashes or not all(_hash(item) for item in field_hashes.values()):
        raise OfficialReferenceError("native CFD field hashes are missing or invalid")
    if native.get("units") != EXPECTED_UNITS:
        raise OfficialReferenceError("native artifact set has an invalid SI unit contract")
    semantics = _solver_semantics(cfd.get("solver_semantics"))
    if not _hash(topology.get("file_sha256")):
        raise OfficialReferenceError("native topology file hash is missing")
    if topology.get("artifact_version") != TOPOLOGY_VERSION:
        raise OfficialReferenceError("native artifact set does not use topology v3")
    if topology.get("mesh_signature_algorithm") != TOPOLOGY_SIGNATURE_ALGORITHM:
        raise OfficialReferenceError("native topology signature algorithm is invalid")
    if not _hash(topology.get("mesh_signature")) or not _hash(topology.get("source_vtp_sha256")):
        raise OfficialReferenceError("native topology hashes are invalid")
    if not isinstance(topology.get("patch_name"), str) or not topology["patch_name"]:
        raise OfficialReferenceError("native topology patch_name is missing")
    if str(topology.get("time")) != str(cfd.get("field_time")):
        raise OfficialReferenceError("native topology and CFD field times differ")
    return {
        "document": native,
        "solver_semantics": semantics,
        "topology": dict(topology),
        "artifact_set_id": str(stored_id).lower(),
    }


def _validate_thresholds(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OfficialReferenceError("thresholds must be a pre-registered object")
    thresholds = dict(value)
    missing = [name for name in REQUIRED_ERROR_THRESHOLDS if name not in thresholds]
    if missing:
        raise OfficialReferenceError(f"thresholds missing: {missing}")
    for name in REQUIRED_ERROR_THRESHOLDS:
        number = _finite_number(thresholds[name], f"thresholds.{name}")
        if number <= 0.0 or number > 0.05:
            raise OfficialReferenceError(f"thresholds.{name} must be in (0, 0.05]")
    percentile = _finite_number(thresholds.get("hotspot_percentile"), "thresholds.hotspot_percentile")
    if not 0.0 < percentile < 100.0:
        raise OfficialReferenceError("thresholds.hotspot_percentile must be in (0, 100)")
    top_k = thresholds.get("top_k", [])
    if not isinstance(top_k, Sequence) or isinstance(top_k, (str, bytes, bytearray)):
        raise OfficialReferenceError("thresholds.top_k must be a list")
    normalized_top_k = []
    seen_k: set[int] = set()
    for item in top_k:
        if not isinstance(item, Mapping):
            raise OfficialReferenceError("thresholds.top_k entries must be objects")
        k = item.get("k")
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0 or k in seen_k:
            raise OfficialReferenceError("thresholds.top_k has an invalid or duplicate k")
        overlap = _finite_number(item.get("minimum_overlap"), f"thresholds.top_k[{k}].minimum_overlap")
        if not 0.0 <= overlap <= 1.0:
            raise OfficialReferenceError("top-k minimum_overlap must be in [0, 1]")
        seen_k.add(k)
        normalized_top_k.append({"k": k, "minimum_overlap": overlap})
    normalized = {name: float(thresholds[name]) for name in REQUIRED_ERROR_THRESHOLDS}
    normalized["hotspot_percentile"] = percentile
    normalized["top_k"] = sorted(normalized_top_k, key=lambda item: item["k"])
    return normalized


def _validate_evaluation_contract(
    value: Any,
    record_contract: Mapping[str, Any],
    evaluation_definition_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OfficialReferenceError("evaluation_contract is missing")
    contract = dict(value)
    if contract.get("version") != EVALUATION_VERSION:
        raise OfficialReferenceError(f"evaluation_contract must use {EVALUATION_VERSION}")
    if contract.get("definition_sha256") != evaluation_definition_sha256:
        raise OfficialReferenceError("evaluation_contract definition_sha256 mismatch")
    metric = contract.get("value_metric")
    units = contract.get("canonical_units")
    if metric != record_contract.get("value_metric") or units != record_contract.get("value_units"):
        raise OfficialReferenceError("evaluation and record contracts use different metric/units")
    allowed_aggregation = {
        "wear_depth": "area_weighted_integral",
        "erosion_rate": "area_weighted_integral",
        "erosion_area": "explicit_sum",
        "erosion_volume": "explicit_sum",
        "erosion_mass": "explicit_sum",
        "erosion_ratio": "explicit_sum",
        "dimensionless": "explicit_sum",
    }
    aggregation = contract.get("aggregation")
    if aggregation != allowed_aggregation.get(str(metric)):
        raise OfficialReferenceError("evaluation aggregation is invalid for value_metric")
    if contract.get("maximum_metric") != "maximum_value":
        raise OfficialReferenceError("evaluation maximum_metric must be maximum_value")
    if contract.get("relative_error_denominator") != "absolute_reference":
        raise OfficialReferenceError("evaluation relative_error_denominator must be absolute_reference")
    if contract.get("zero_reference_policy") != "block":
        raise OfficialReferenceError("evaluation zero_reference_policy must be block")
    hotspot = contract.get("hotspot")
    if not isinstance(hotspot, Mapping):
        raise OfficialReferenceError("evaluation hotspot definition is missing")
    if hotspot.get("mode") != "weighted_percentile":
        raise OfficialReferenceError("evaluation hotspot mode must be weighted_percentile")
    if hotspot.get("area_source") != record_contract.get("area_source"):
        raise OfficialReferenceError("evaluation and record contracts use different area_source")
    percentile = _finite_number(hotspot.get("percentile"), "evaluation hotspot percentile")
    if not 0.0 < percentile < 100.0:
        raise OfficialReferenceError("evaluation hotspot percentile must be in (0, 100)")
    if contract.get("top_k_tie_break") != "parent_face_id_ascending":
        raise OfficialReferenceError("evaluation top_k_tie_break is invalid")
    return {
        "version": EVALUATION_VERSION,
        "definition_sha256": evaluation_definition_sha256,
        "metric_name": str(contract.get("metric_name", metric)),
        "value_metric": metric,
        "canonical_units": units,
        "aggregation": aggregation,
        "maximum_metric": "maximum_value",
        "relative_error_denominator": "absolute_reference",
        "zero_reference_policy": "block",
        "hotspot": {
            "mode": "weighted_percentile",
            "percentile": percentile,
            "area_source": hotspot["area_source"],
        },
        "top_k_tie_break": "parent_face_id_ascending",
    }


def _validate_record_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OfficialReferenceError("record_contract is missing")
    contract = dict(value)
    if contract.get("parent_face_id_field", contract.get("id_field")) != "parent_face_id":
        raise OfficialReferenceError("record_contract must use parent_face_id")
    value_field = contract.get("value_field")
    value_units = contract.get("value_units", contract.get("units"))
    value_metric = contract.get("value_metric")
    coverage = contract.get("coverage")
    if not isinstance(value_field, str) or not value_field.strip():
        raise OfficialReferenceError("record_contract.value_field is missing")
    if not isinstance(value_units, str) or value_units not in SUPPORTED_UNITS:
        raise OfficialReferenceError("record_contract.value_units is invalid")
    if not isinstance(value_metric, str) or value_metric not in METRIC_UNITS:
        raise OfficialReferenceError("record_contract.value_metric is invalid")
    if value_units not in METRIC_UNITS[value_metric]:
        raise OfficialReferenceError("record_contract value_metric and value_units differ")
    if coverage not in {"full_parent_faces", "subset"}:
        raise OfficialReferenceError("record_contract.coverage must be full_parent_faces or subset")
    prediction_field = contract.get("prediction_field", value_field)
    observation_field = contract.get("observation_field", value_field)
    if not isinstance(prediction_field, str) or not isinstance(observation_field, str):
        raise OfficialReferenceError("prediction/observation fields must be strings")
    area_source = contract.get("area_source", "topology_parent_face_area_m2")
    if area_source not in {"topology_parent_face_area_m2", "record"}:
        raise OfficialReferenceError("record_contract.area_source is invalid")
    area_field = contract.get("area_field")
    area_units = contract.get("area_units")
    if area_source == "record":
        if not isinstance(area_field, str) or not area_field.strip():
            raise OfficialReferenceError("record area_source requires area_field")
        if area_units not in {"m2", "mm2"}:
            raise OfficialReferenceError("record area_source requires area_units m2 or mm2")
    elif area_field is not None and area_units not in {"m2", "mm2"}:
        raise OfficialReferenceError("declared record area units are invalid")
    return {
        "parent_face_id_field": "parent_face_id",
        "value_field": value_field,
        "value_metric": value_metric,
        "prediction_field": prediction_field,
        "observation_field": observation_field,
        "value_units": value_units,
        "coverage": coverage,
        "area_source": area_source,
        "area_field": area_field,
        "area_units": area_units,
    }


def _validate_split(
    value: Any,
    reference_artifact_sha256: str,
    split_artifact_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OfficialReferenceError("independent_split is missing")
    split = dict(value)
    provider = str(split.get("provider", split.get("source", ""))).strip().lower()
    if "organizer" not in provider:
        raise OfficialReferenceError("independent_split must be organizer-provided")
    if split.get("is_independent") is not True:
        raise OfficialReferenceError("independent_split must declare is_independent=true")
    if not isinstance(split.get("split_id"), str) or not split["split_id"].strip():
        raise OfficialReferenceError("independent_split.split_id is missing")
    record_ids = split.get("record_ids")
    if not isinstance(record_ids, Sequence) or isinstance(record_ids, (str, bytes, bytearray)) or not record_ids:
        raise OfficialReferenceError("independent_split.record_ids is empty")
    normalized_ids = []
    seen: set[int] = set()
    for value_id in record_ids:
        if isinstance(value_id, bool) or not isinstance(value_id, int) or value_id < 0 or value_id in seen:
            raise OfficialReferenceError("independent_split.record_ids must be unique non-negative integers")
        seen.add(value_id)
        normalized_ids.append(value_id)
    reference_hash = split.get("reference_source_sha256")
    split_hash = split.get("split_artifact_sha256")
    if not _hash(reference_hash) or str(reference_hash).lower() != reference_artifact_sha256:
        raise OfficialReferenceError("independent_split does not bind normalized reference observations")
    if not _hash(split_hash) or str(split_hash).lower() != split_artifact_sha256:
        raise OfficialReferenceError("independent_split does not bind organizer split artifact")
    return {
        "provider": provider,
        "is_independent": True,
        "split_id": split["split_id"],
        "record_ids": sorted(normalized_ids),
        "reference_source_sha256": str(reference_hash).lower(),
        "split_artifact_sha256": str(split_hash).lower(),
    }


def load_official_reference_manifest(value: Any) -> dict[str, Any]:
    """Load and fully validate a v2 organizer reference manifest."""
    document, origin = _load(value)
    return _reference_manifest(document, origin)


def _reference_manifest(value: Any, origin: Path | None) -> dict[str, Any]:
    document, _ = _load(value)
    if not isinstance(document, Mapping):
        raise OfficialReferenceError("reference manifest must be an object")
    document = dict(document)
    if document.get("version") != REFERENCE_MANIFEST_VERSION:
        raise OfficialReferenceError(f"reference manifest must use {REFERENCE_MANIFEST_VERSION}")
    if document.get("manifest_kind") != MANIFEST_KIND:
        raise OfficialReferenceError(f"reference manifest must use manifest_kind={MANIFEST_KIND}")
    _canonical_checked(document, "manifest_sha256")
    if document.get("status") != "synthetic":
        raise OfficialReferenceError("contest reference simulation must use status=synthetic")
    try:
        evidence = evidence_metadata(document, status="synthetic")
    except EvidenceContractError as exc:
        raise OfficialReferenceError(str(exc)) from exc
    if (
        evidence["source_type"] != "contest_provided_simulation"
        or evidence["reference_role"] != "official_reference_simulation"
        or evidence["official_reference"] is not True
    ):
        raise OfficialReferenceError("reference manifest evidence role is not official contest simulation")
    provider = document.get("contest")
    if not isinstance(provider, Mapping):
        raise OfficialReferenceError("contest metadata is missing")
    for field in ("competition", "organizer", "dataset_id", "dataset_version"):
        if not isinstance(provider.get(field), str) or not provider[field].strip():
            raise OfficialReferenceError(f"contest metadata missing {field}")
    artifact_set_id = document.get("artifact_set_id")
    if not _hash(artifact_set_id):
        raise OfficialReferenceError("artifact_set_id is missing or invalid")
    root = _resolve_root(document, origin)
    artifacts = _artifact_entries(document, root)
    role_map = {str(item["role"]): item for item in artifacts}
    required_roles = {
        "official_description",
        "official_evaluation_definition",
        "official_reference_raw",
        "official_independent_split",
        "official_reference_observations",
        "native_artifact_set_manifest",
        "topology_v3",
    }
    missing_roles = sorted(required_roles - set(role_map))
    if missing_roles:
        raise OfficialReferenceError(f"reference manifest missing artifact roles: {missing_roles}")
    native_binding, native_path = _binding(
        document, "native_artifact", artifacts,
        ("native_artifact_set_manifest", "artifact_set_manifest"), root,
    )
    native = _validate_native_descriptor(native_path, native_binding, str(artifact_set_id).lower())
    solver = _solver_semantics(document.get("solver_semantics"))
    if solver != native["solver_semantics"]:
        raise OfficialReferenceError("prediction and native solver semantics differ")
    reference_solver = _reference_solver_semantics(document.get("reference_solver_semantics"))
    units = document.get("units")
    if units != EXPECTED_UNITS:
        raise OfficialReferenceError("reference unit contract differs from native SI units")

    topology_meta = document.get("topology")
    if not isinstance(topology_meta, Mapping):
        raise OfficialReferenceError("topology metadata is missing")
    topology_binding, topology_path = _binding(
        document, "topology", artifacts, ("topology_v3", "native_topology_v3"), root,
    )
    topology_digest = str(topology_meta.get("sha256", topology_meta.get("file_sha256", ""))).lower()
    if topology_digest != topology_binding["sha256"]:
        raise OfficialReferenceError("topology binding hash differs from topology metadata")
    if topology_digest != native["topology"].get("file_sha256"):
        raise OfficialReferenceError("topology binding hash differs from native artifact set")
    try:
        topology = load_topology_artifact(topology_path)
    except (OSError, ValueError, TypeError) as exc:
        raise OfficialReferenceError(f"topology v3 validation failed: {exc}") from exc
    if topology.get("artifact_version") != TOPOLOGY_VERSION:
        raise OfficialReferenceError("topology artifact is not v3")
    expected_topology = {
        "artifact_version": TOPOLOGY_VERSION,
        "mesh_signature_algorithm": TOPOLOGY_SIGNATURE_ALGORITHM,
        "mesh_signature": native["topology"].get("mesh_signature"),
        "source_vtp_sha256": native["topology"].get("source_vtp_sha256"),
        "patch_name": native["topology"].get("patch_name"),
        "time": native["topology"].get("time"),
        "n_parent_faces": topology.get("n_parent_faces"),
        "n_collision_triangles": len(topology["collision_triangles_mm"]),
    }
    for key, expected in expected_topology.items():
        actual = topology_meta.get(key)
        if actual != expected:
            raise OfficialReferenceError(f"topology metadata {key} differs from validated identity")
    if topology.get("mesh_signature") != native["topology"].get("mesh_signature"):
        raise OfficialReferenceError("validated topology mesh signature differs from native identity")
    if topology.get("source_vtp_sha256") != native["topology"].get("source_vtp_sha256"):
        raise OfficialReferenceError("validated topology source VTP hash differs from native identity")
    if topology.get("patch_name") != native["topology"].get("patch_name"):
        raise OfficialReferenceError("validated topology patch differs from native identity")
    if str(topology.get("time_name")) != str(native["topology"].get("time")):
        raise OfficialReferenceError("validated topology time differs from native identity")
    if (topology.get("source_coordinate_units"), topology.get("internal_coordinate_units"), topology.get("coordinate_scale_to_mm")) != ("m", "mm", 1000.0):
        raise OfficialReferenceError("validated topology has an invalid SI coordinate contract")
    if not math.isclose(
        float(topology_meta.get("surface_area_m2")),
        float(topology["surface_area_m2"]),
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise OfficialReferenceError("declared topology surface area differs from validated NPZ")
    if topology.get("n_parent_faces") != native["topology"].get("n_parent_faces"):
        raise OfficialReferenceError("topology parent-face count differs from native identity")

    reference_binding, reference_path = _binding(
        document, "reference_records", artifacts,
        ("official_reference_observations", "reference_observations"), root,
    )
    try:
        reference_document = safe_json_load(reference_path)
    except (OSError, UnicodeError, ValidationError) as exc:
        raise OfficialReferenceError("reference observation artifact is invalid JSON") from exc
    contract = _validate_record_contract(document.get("record_contract"))
    evaluation = _validate_evaluation_contract(
        document.get("evaluation_contract"),
        contract,
        role_map["official_evaluation_definition"]["sha256"],
    )
    records_hash = _validate_normalized_reference_document(reference_document, contract)
    provenance = document.get("record_provenance")
    if not isinstance(provenance, Mapping):
        raise OfficialReferenceError("record_provenance is missing")
    if provenance.get("raw_reference_sha256") != role_map["official_reference_raw"]["sha256"]:
        raise OfficialReferenceError("record_provenance raw reference hash mismatch")
    if provenance.get("normalized_reference_sha256") != reference_binding["sha256"]:
        raise OfficialReferenceError("record_provenance normalized reference hash mismatch")
    if provenance.get("normalized_records_sha256") != records_hash:
        raise OfficialReferenceError("record_provenance records hash mismatch")
    if provenance.get("inferred_fields") is not False or not isinstance(provenance.get("adapter"), Mapping):
        raise OfficialReferenceError("record_provenance requires an explicit non-inferred adapter")
    split = _validate_split(
        document.get("independent_split"),
        reference_binding["sha256"],
        role_map["official_independent_split"]["sha256"],
    )
    thresholds = _validate_thresholds(document.get("thresholds"))
    if thresholds["hotspot_percentile"] != evaluation["hotspot"]["percentile"]:
        raise OfficialReferenceError("threshold and evaluation hotspot percentiles differ")
    return {
        "document": document,
        "manifest_sha256": str(document["manifest_sha256"]).lower(),
        "manifest_path": str(origin) if origin else None,
        "source_root": root,
        "contest": dict(provider),
        "artifact_set_id": str(artifact_set_id).lower(),
        "artifacts": artifacts,
        "native_binding": {**native_binding, "path": str(native_path)},
        "native": native,
        "solver_semantics": solver,
        "reference_solver_semantics": reference_solver,
        "units": dict(units),
        "topology_binding": {**topology_binding, "path": str(topology_path)},
        "topology": dict(topology_meta),
        "topology_data": topology,
        "reference_binding": {**reference_binding, "path": str(reference_path)},
        "reference_document": reference_document,
        "record_contract": contract,
        "evaluation_contract": evaluation,
        "record_provenance": dict(provenance),
        "source_artifact_hashes": {
            role: str(role_map[role]["sha256"])
            for role in sorted(required_roles)
        },
        "independent_split": split,
        "thresholds": thresholds,
        "evidence": evidence,
    }


def _explicit_unit(row: Mapping[str, Any], field: str) -> str | None:
    units = row.get("units")
    if isinstance(units, Mapping) and field in units:
        return str(units[field])
    for key in (f"{field}_units", f"{field}_unit", "value_units", "unit"):
        if key in row:
            return str(row[key])
    return None


def _validated_records(
    value: Any,
    label: str,
    contract: Mapping[str, Any],
    n_parent_faces: int,
) -> dict[int, dict[str, Any]]:
    rows = _records(value, label)
    result: dict[int, dict[str, Any]] = {}
    field = contract["prediction_field"] if label == "predictions" else contract["observation_field"]
    for row in rows:
        identifier = row.get("parent_face_id")
        if isinstance(identifier, bool) or not isinstance(identifier, int):
            raise OfficialReferenceError(f"{label} parent_face_id must be an integer")
        if identifier < 0 or identifier >= n_parent_faces:
            raise OfficialReferenceError(f"{label} parent_face_id is out of range: {identifier}")
        if identifier in result:
            raise OfficialReferenceError(f"duplicate {label} parent_face_id: {identifier}")
        if field not in row:
            raise OfficialReferenceError(f"{label} record {identifier} has no {field}")
        number = _finite_number(row[field], f"{label}/{identifier}/{field}")
        if number < 0.0:
            raise OfficialReferenceError(f"{label}/{identifier}/{field} must be non-negative")
        explicit_unit = _explicit_unit(row, field)
        if explicit_unit is not None and explicit_unit != contract["value_units"]:
            raise OfficialReferenceError(f"{label}/{identifier} unit differs from record contract")
        normalized = dict(row)
        normalized["parent_face_id"] = identifier
        normalized[field] = number
        result[identifier] = normalized
    if contract["coverage"] == "full_parent_faces" and set(result) != set(range(n_parent_faces)):
        raise OfficialReferenceError("full_parent_faces contract is not completely covered")
    return result


def _validate_normalized_reference_document(
    document: Any,
    contract: Mapping[str, Any],
) -> str:
    if not isinstance(document, Mapping):
        raise OfficialReferenceError("normalized reference artifact must be an object")
    if document.get("version") != "step30-official-reference-records-v1":
        raise OfficialReferenceError("normalized reference artifact version is invalid")
    if (
        document.get("status") != "synthetic"
        or document.get("source_type") != "contest_provided_simulation"
        or document.get("reference_role") != "official_reference_simulation"
        or document.get("pipeline_validation") is not False
        or document.get("official_reference") is not True
    ):
        raise OfficialReferenceError("normalized reference artifact evidence role is invalid")
    if document.get("value_metric") != contract.get("value_metric"):
        raise OfficialReferenceError("normalized reference metric differs from record contract")
    unsigned = dict(document)
    stored_document_hash = unsigned.pop("document_sha256", None)
    if not _hash(stored_document_hash) or canonical_sha256(unsigned) != str(stored_document_hash).lower():
        raise OfficialReferenceError("normalized reference document_sha256 is invalid")
    records_hash = _canonical_record_payload(document, "observations")
    if document.get("records_sha256") != records_hash:
        raise OfficialReferenceError("normalized reference records_sha256 is invalid")
    return records_hash


def _canonical_record_payload(value: Any, label: str) -> str:
    rows = _records(value, label)
    normalized = sorted((dict(row) for row in rows), key=lambda row: row.get("parent_face_id", -1))
    return canonical_sha256({"records": normalized})


def _check_observation_binding(observation: Any, reference_document: Any) -> None:
    if _canonical_record_payload(observation, "observations") != _canonical_record_payload(reference_document, "observations"):
        raise OfficialReferenceError("observations differ from the hash-bound reference artifact")


def _record_weights(
    observed: Mapping[int, Mapping[str, Any]],
    contract: Mapping[str, Any],
    topology: Mapping[str, Any],
) -> dict[int, float]:
    if contract["area_source"] == "topology_parent_face_area_m2":
        areas = np.asarray(topology["face_area_m2"], dtype=np.float64)
        return {identifier: float(areas[identifier]) for identifier in observed}
    field = contract["area_field"]
    units = contract["area_units"]
    result: dict[int, float] = {}
    for identifier, row in observed.items():
        if field not in row:
            raise OfficialReferenceError(f"observation record {identifier} has no {field}")
        value = _finite_number(row[field], f"area/{identifier}")
        if value < 0.0:
            raise OfficialReferenceError(f"area/{identifier} must be non-negative")
        explicit = _explicit_unit(row, field)
        if explicit is not None and explicit != units:
            raise OfficialReferenceError(f"area/{identifier} unit differs from record contract")
        result[identifier] = value / 1.0e6 if units == "mm2" else value
    if not any(value > 0.0 for value in result.values()):
        raise OfficialReferenceError("record areas must have a positive total")
    return result


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(np.interp(percentile / 100.0 * float(cumulative[-1]), cumulative, ordered_values))


def _comparison(prediction: Any, observation: Any, manifest: Mapping[str, Any]) -> dict[str, Any]:
    contract = manifest["record_contract"]
    topology = manifest["topology_data"]
    n_parent_faces = int(topology["n_parent_faces"])
    predicted_rows = _validated_records(prediction, "predictions", contract, n_parent_faces)
    observed_rows = _validated_records(observation, "observations", contract, n_parent_faces)
    if set(predicted_rows) != set(observed_rows):
        raise OfficialReferenceError("prediction and reference parent_face_id sets differ")
    split_ids = set(manifest["independent_split"]["record_ids"])
    if set(observed_rows) != split_ids:
        raise OfficialReferenceError("comparison records do not exactly match organizer independent split")
    identifiers = sorted(predicted_rows)
    predicted = np.asarray(
        [predicted_rows[key][contract["prediction_field"]] for key in identifiers], dtype=np.float64
    )
    observed = np.asarray(
        [observed_rows[key][contract["observation_field"]] for key in identifiers], dtype=np.float64
    )
    weights_map = _record_weights(observed_rows, contract, topology)
    weights = np.asarray([weights_map[key] for key in identifiers], dtype=np.float64)
    if np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise OfficialReferenceError("comparison areas must be non-negative with a positive total")
    evaluation = manifest["evaluation_contract"]
    residual = predicted - observed
    absolute = np.abs(residual)
    if evaluation["aggregation"] == "area_weighted_integral":
        total_pred = float(np.sum(predicted * weights))
        total_obs = float(np.sum(observed * weights))
        total_units = f"{contract['value_units']}*m2"
    elif evaluation["aggregation"] == "explicit_sum":
        total_pred = float(np.sum(predicted))
        total_obs = float(np.sum(observed))
        total_units = contract["value_units"]
    else:
        raise OfficialReferenceError("unsupported evaluation aggregation")
    max_pred = float(np.max(predicted))
    max_obs = float(np.max(observed))
    percentile = manifest["thresholds"]["hotspot_percentile"]
    pred_threshold = _weighted_percentile(predicted, weights, percentile)
    obs_threshold = _weighted_percentile(observed, weights, percentile)
    pred_hot = predicted >= pred_threshold
    obs_hot = observed >= obs_threshold
    intersection = float(np.sum(weights[pred_hot & obs_hot]))
    union = float(np.sum(weights[pred_hot | obs_hot]))
    iou = intersection / union if union else 1.0
    area_pred = float(np.sum(weights[pred_hot]))
    area_obs = float(np.sum(weights[obs_hot]))
    if evaluation["zero_reference_policy"] == "block" and (total_obs == 0.0 or max_obs == 0.0 or area_obs == 0.0):
        raise OfficialReferenceError("reference total/maximum/hotspot area is zero under block policy")
    total_relative = abs(total_pred - total_obs) / abs(total_obs)
    max_relative = abs(max_pred - max_obs) / abs(max_obs)
    area_relative = abs(area_pred - area_obs) / abs(area_obs)
    iou_error = 1.0 - iou
    thresholds = manifest["thresholds"]
    threshold_checks = {
        "total_quantity_relative_error": total_relative < thresholds["total_quantity_relative_error"],
        "maximum_value_relative_error": max_relative < thresholds["maximum_value_relative_error"],
        "hotspot_area_relative_error": area_relative < thresholds["hotspot_area_relative_error"],
        "hotspot_iou_error": iou_error < thresholds["hotspot_iou_error"],
    }
    top_k_results = []
    for item in thresholds["top_k"]:
        k = item["k"]
        if k > len(identifiers):
            raise OfficialReferenceError(f"top-k k exceeds the number of records: {k}")
        pred_order = sorted(range(len(identifiers)), key=lambda index: (-predicted[index], identifiers[index]))[:k]
        obs_order = sorted(range(len(identifiers)), key=lambda index: (-observed[index], identifiers[index]))[:k]
        overlap = len(set(pred_order) & set(obs_order)) / float(k)
        top_k_results.append(
            {
                "k": k,
                "predicted_parent_face_ids": [identifiers[index] for index in pred_order],
                "observed_parent_face_ids": [identifiers[index] for index in obs_order],
                "overlap": overlap,
                "minimum_overlap": item["minimum_overlap"],
                "passed": overlap >= item["minimum_overlap"],
            }
        )
    result = {
        "n": len(identifiers),
        "record_ids": identifiers,
        "metric_name": evaluation["metric_name"],
        "value_metric": evaluation["value_metric"],
        "value_units": contract["value_units"],
        "total_units": total_units,
        "area_units": "m2",
        "aggregation": evaluation["aggregation"],
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "p95": float(np.percentile(absolute, 95.0)),
        "max_absolute_error": float(np.max(absolute)),
        "total_error": {"absolute": abs(total_pred - total_obs), "relative": total_relative},
        "maximum_value_error": {"absolute": abs(max_pred - max_obs), "relative": max_relative},
        "maximum_depth_error": {"absolute": abs(max_pred - max_obs), "relative": max_relative},
        "hotspot": {
            "percentile": percentile,
            "predicted_threshold": pred_threshold,
            "observed_threshold": obs_threshold,
            "predicted_area_m2": area_pred,
            "observed_area_m2": area_obs,
            "iou": iou,
            "area_relative_error": area_relative,
            "iou_error": iou_error,
        },
        "thresholds": thresholds,
        "threshold_checks": threshold_checks,
        "top_k": top_k_results,
    }
    result["under_thresholds"] = bool(
        all(threshold_checks.values()) and all(item["passed"] for item in top_k_results)
    )
    result["under_5pct"] = bool(
        total_relative < 0.05
        and max_relative < 0.05
        and area_relative < 0.05
        and iou_error < 0.05
        and all(item["passed"] for item in top_k_results)
    )
    return result


def _artifact_descriptor(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path)}


def validate_official_reference(
    prediction: Any,
    observation: Any,
    reference_manifest: Any,
    *,
    trusted_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Return an authenticated contest-reference comparison.

    A valid manifest can still produce ``status=failed`` when its declared
    numerical thresholds are missed. Missing provenance, identity, units, or
    record evidence always produces ``status=blocked``.
    """
    try:
        prediction_document, prediction_origin = _load(prediction)
        observation_document, _ = _load(observation)
        manifest_document, manifest_origin = _load(reference_manifest)
        manifest = _reference_manifest(manifest_document, manifest_origin)
        _check_observation_binding(observation_document, manifest["reference_document"])
        from step30_official_reference_prediction import validate_prediction_document

        prediction_records_sha256 = validate_prediction_document(prediction_document, manifest)
        observation_records_sha256 = _canonical_record_payload(observation_document, "observations")
        comparison = _comparison(prediction_document, observation_document, manifest)
        manifest_digest = manifest["manifest_sha256"]
        if trusted_manifest_sha256 is not None and (
            not _hash(trusted_manifest_sha256)
            or str(trusted_manifest_sha256).lower() != manifest_digest
        ):
            raise OfficialReferenceError("trusted reference manifest SHA-256 does not match manifest")
        trusted = trusted_manifest_sha256 is not None
        report = {
            "version": VERSION,
            "status": "passed" if comparison["under_thresholds"] else "failed",
            "numerical_status": "passed" if comparison["under_thresholds"] else "failed",
            "official_reference": True,
            "official_reference_gate": bool(comparison["under_thresholds"] and trusted),
            "trusted_manifest_sha256": str(trusted_manifest_sha256).lower() if trusted_manifest_sha256 is not None else None,
            "source_type": "contest_provided_simulation",
            "reference_role": "official_reference_simulation",
            "pipeline_validation": False,
            "reference_dataset_id": manifest["contest"]["dataset_id"],
            "reference_manifest_sha256": manifest["manifest_sha256"],
            "reference_manifest_path": manifest["manifest_path"],
            "artifact_set_id": manifest["artifact_set_id"],
            "native_artifact_set_manifest": _artifact_descriptor(Path(manifest["native_binding"]["path"])),
            "reference_observation_artifact": {
                **_artifact_descriptor(Path(manifest["reference_binding"]["path"])),
                "role": "official_reference_observations",
            },
            "prediction_artifact": _artifact_descriptor(prediction_origin) if prediction_origin else None,
            "solver_semantics": manifest["solver_semantics"],
            "reference_solver_semantics": manifest["reference_solver_semantics"],
            "units": manifest["units"],
            "topology": {
                "sha256": manifest["topology_binding"]["sha256"],
                **{
                    key: manifest["topology"].get(key)
                    for key in (
                        "artifact_version",
                        "mesh_signature_algorithm",
                        "mesh_signature",
                        "source_vtp_sha256",
                        "patch_name",
                        "time",
                        "n_parent_faces",
                        "n_collision_triangles",
                        "surface_area_m2",
                    )
                },
            },
            "independent_split": manifest["independent_split"],
            "record_contract": manifest["record_contract"],
            "evaluation_contract": manifest["evaluation_contract"],
            "record_provenance": manifest["record_provenance"],
            "source_artifact_hashes": manifest["source_artifact_hashes"],
            "prediction_records_sha256": prediction_records_sha256,
            "observation_records_sha256": observation_records_sha256,
            "comparison": comparison,
            "provenance_note": "contest-provided reference simulation; this is not field wear measurement",
        }
        report["report_sha256"] = canonical_sha256(report)
        return report
    except (OfficialReferenceError, OSError, TypeError, ValueError, KeyError, OverflowError) as exc:
        report = {
            "version": VERSION,
            "status": "blocked",
            "numerical_status": "blocked",
            "official_reference": False,
            "official_reference_gate": False,
            "source_type": "contest_provided_simulation",
            "reference_role": "official_reference_simulation",
            "pipeline_validation": False,
            "reason": str(exc) or type(exc).__name__,
            "provenance_note": "blocked contest reference comparison; no accuracy claim",
        }
        report["report_sha256"] = canonical_sha256(report)
        return report


def _selftest() -> None:
    """Exercise the complete contract with a two-face local fixture."""
    import tempfile

    import vtk

    from step30_artifact_identity import ARTIFACT_DESCRIPTOR_VERSION as ID_VERSION
    from step30_cfd_topology import build_topology_from_vtp

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "bucket.vtp"
        points = vtk.vtkPoints()
        for point in (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
        ):
            points.InsertNextPoint(point)
        polys = vtk.vtkCellArray()
        quad = vtk.vtkPolygon()
        quad.GetPointIds().SetNumberOfIds(4)
        for index, point_id in enumerate((0, 1, 2, 3)):
            quad.GetPointIds().SetId(index, point_id)
        polys.InsertNextCell(quad)
        triangle = vtk.vtkTriangle()
        for index, point_id in enumerate((4, 5, 6)):
            triangle.GetPointIds().SetId(index, point_id)
        polys.InsertNextCell(triangle)
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(polys)
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(source))
        writer.SetInputData(polydata)
        assert writer.Write() == 1
        topology_path = root / "topology.npz"
        topology_summary = root / "topology.json"
        build_topology_from_vtp(
            source,
            patch_name="bucket",
            time_name="0.8684190921",
            units="m",
            output_npz=topology_path,
            output_json=topology_summary,
        )
        topology = load_topology_artifact(topology_path)
        semantics = {
            "application": "pimpleFoam",
            "mode": "urans",
            "field_statistic": "time_mean",
            "tracking_velocity_object": "UMean",
            "averaging_window": {"start_time": "0.7", "end_time": "0.8684190921", "sample_count": 12},
        }
        native = {
            "version": ID_VERSION,
            "cfd": {
                "variant": "native_si",
                "field_time": "0.8684190921",
                "manifest_version": NATIVE_MANIFEST_VERSION,
                "manifest_sha256": "a" * 64,
                "polyMesh_signature": "b" * 64,
                "field_sha256": {"UMean": "c" * 64},
                "solver_semantics": semantics,
            },
            "topology": {
                "file_sha256": sha256_file(topology_path),
                "artifact_version": TOPOLOGY_VERSION,
                "mesh_signature_algorithm": TOPOLOGY_SIGNATURE_ALGORITHM,
                "mesh_signature": topology["mesh_signature"],
                "source_vtp_sha256": topology["source_vtp_sha256"],
                "patch_name": "bucket",
                "time": "0.8684190921",
                "n_parent_faces": topology["n_parent_faces"],
            },
            "units": EXPECTED_UNITS,
        }
        native["artifact_set_id"] = canonical_sha256(native)
        native_path = root / "artifact_set_manifest.json"
        native_path.write_text(json.dumps(native, ensure_ascii=False, indent=2), encoding="utf-8")
        from step30_official_reference_intake import build_intake, preflight_intake

        (root / "official-description.txt").write_text("selftest organizer package", encoding="utf-8")
        (root / "official-evaluation-definition.txt").write_text("wear-depth metrics", encoding="utf-8")
        (root / "raw-reference.csv").write_text(
            "parent_face_id,wear_um,unit\n0,1000,um\n1,2000,um\n", encoding="utf-8"
        )
        (root / "independent-split.csv").write_text(
            "parent_face_id\n0\n1\n", encoding="utf-8"
        )
        spec = {
            "version": "step30-official-reference-intake-spec-v1",
            "source_root": str(root),
            "contest": {
                "competition": "Step30 selftest",
                "organizer": "organizer",
                "dataset_id": "reference-selftest-v2",
                "dataset_version": "v2",
            },
            "official_description": "official-description.txt",
            "official_evaluation_definition": "official-evaluation-definition.txt",
            "native_artifact_set_manifest": native_path.name,
            "topology_v3": topology_path.name,
            "reference_source": {
                "path": "raw-reference.csv",
                "adapter": {
                    "format": "csv",
                    "parent_face_id_field": "parent_face_id",
                    "value_field": "wear_um",
                    "canonical_value_field": "wear_depth_mm",
                    "unit_field": "unit",
                    "source_units": "um",
                    "canonical_units": "mm",
                    "value_metric": "wear_depth",
                },
            },
            "independent_split": {
                "provider": "organizer_provided",
                "is_independent": True,
                "split_id": "selftest-independent-v2",
                "path": "independent-split.csv",
                "format": "csv",
                "parent_face_id_field": "parent_face_id",
            },
            "record_contract": {
                "parent_face_id_field": "parent_face_id",
                "value_field": "wear_depth_mm",
                "value_metric": "wear_depth",
                "value_units": "mm",
                "coverage": "full_parent_faces",
                "area_source": "topology_parent_face_area_m2",
            },
            "evaluation_contract": {
                "version": EVALUATION_VERSION,
                "definition_sha256": sha256_file(root / "official-evaluation-definition.txt"),
                "metric_name": "wear_depth_mm",
                "value_metric": "wear_depth",
                "canonical_units": "mm",
                "aggregation": "area_weighted_integral",
                "relative_error_denominator": "absolute_reference",
                "zero_reference_policy": "block",
                "hotspot": {
                    "mode": "weighted_percentile",
                    "percentile": 90.0,
                    "area_source": "topology_parent_face_area_m2",
                },
                "top_k_tie_break": "parent_face_id_ascending",
            },
            "reference_solver_semantics": {
                "application": "organizerSolver",
                "mode": "organizer_reference",
                "field_statistic": "cumulative",
                "value_object": "wear_depth_mm",
            },
            "thresholds": {
                "total_quantity_relative_error": 0.05,
                "maximum_value_relative_error": 0.05,
                "hotspot_area_relative_error": 0.05,
                "hotspot_iou_error": 0.05,
                "top_k": [{"k": 1, "minimum_overlap": 1.0}],
            },
        }
        spec_path = root / "intake-spec.json"
        spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        assert preflight_intake(spec_path)["status"] == "passed"
        intake = build_intake(spec_path, root / "intake")
        manifest_path = Path(intake["manifest"]["path"])
        observation_path = Path(intake["records"]["path"])
        manifest = safe_json_load(manifest_path)
        observations = safe_json_load(observation_path)
        prediction_path = root / "predictions.json"
        prediction_path.write_text(
            json.dumps({"records": observations["records"]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result = validate_official_reference(
            prediction_path,
            observation_path,
            manifest_path,
            trusted_manifest_sha256=manifest["manifest_sha256"],
        )
        assert result["status"] == "passed"
        assert result["official_reference_gate"] is True
        tampered = {"records": [dict(row) for row in observations["records"]]}
        tampered["records"][0]["wear_depth_mm"] = 9.0
        tampered_path = root / "tampered.json"
        tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
        assert validate_official_reference(prediction_path, tampered_path, manifest_path)["status"] == "blocked"


def selftest() -> bool:
    _selftest()
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate against authenticated contest reference simulation")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument(
        "--trusted-reference-manifest-sha256",
        help="out-of-band SHA-256 pin for the organizer reference manifest",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        selftest()
        print("step30_official_reference_simulation selftest: PASS")
        return 0
    if not args.predictions or not args.observations or not args.reference_manifest:
        parser.error("--predictions, --observations, and --reference-manifest are required")
    report = validate_official_reference(
        args.predictions,
        args.observations,
        args.reference_manifest,
        trusted_manifest_sha256=args.trusted_reference_manifest_sha256,
    )
    if args.output:
        safe_json_dump(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
