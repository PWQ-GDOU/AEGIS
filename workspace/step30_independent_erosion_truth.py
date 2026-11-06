"""Fail-closed construction of independent measured spatial erosion truth."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from step30_independent_erosion_protocol import (
    IndependentErosionProtocolError,
    PredictionCommitmentError,
    validate_prediction_commitment,
    validate_protocol,
)
from step30_measurement_io import read_point_cloud, read_sparse_measurement_table
from step30_measurement_parent_face_mapping import (
    TOPOLOGY_V3_VERSION,
    load_topology_v3,
    map_measurements_to_parent_faces,
)
from step30_sparse_thickness import interpolate_sparse_thickness
from step30_surface_difference import surface_difference
from step30_surface_registration import apply_transform, register_surfaces
from step30_validation_common import (
    ValidationError,
    canonical_sha256,
    convert_units,
    jsonable,
    safe_json_dump,
    safe_json_load,
    sha256_file,
    validate_units,
)


TRUTH_VERSION = "step30-independent-erosion-truth-v1"
REVEAL_RECEIPT_VERSION = "step30-independent-erosion-reveal-receipt-v1"
RECORD_SOURCE_TYPES = frozenset({"baseline_worn_surface", "sparse_thickness"})
_HEX = frozenset("0123456789abcdef")
_NON_MEASURED_TOKENS = frozenset(
    {
        "synthetic",
        "simulation",
        "simulated",
        "dec",
        "cfd",
        "numerical",
        "fixed_seed",
        "generator",
    }
)
_ROW_KEYS = frozenset(
    {
        "record_id",
        "parent_face_id",
        "wear_depth_mm",
        "area_mm2",
        "uncertainty_mm",
        "valid_count",
    }
)
_RECEIPT_KEYS = frozenset(
    {
        "version",
        "evaluation_status",
        "reason",
        "protocol_sha256",
        "prediction_commitment_sha256",
        "truth_sha256",
        "official_reference_gate",
        "receipt_sha256",
    }
)
_TRUTH_KEYS = frozenset(
    {
        "version",
        "status",
        "evaluation_status",
        "protocol_sha256",
        "artifact_set_id",
        "topology_path",
        "topology_v3_sha256",
        "topology_artifact_version",
        "topology_source_vtp_sha256",
        "mesh_signature",
        "coordinate_units",
        "wear_units",
        "area_units",
        "uncertainty_units",
        "record_sources",
        "registration_bindings",
        "rows",
        "n_records",
        "n_parent_faces",
        "official_reference_gate",
        "truth_sha256",
    }
)


class IndependentErosionTruthError(ValidationError):
    """Raised when measured erosion truth cannot be established exactly."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value.lower())
    )


def _sha256(value: Any, name: str) -> str:
    if not _is_sha256(value):
        raise IndependentErosionTruthError(
            f"{name} must be a 64-character SHA-256 hex digest"
        )
    return str(value).lower()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IndependentErosionTruthError(f"{name} must be a non-empty string")
    return value.strip()


def _number(
    value: Any,
    name: str,
    *,
    positive: bool = False,
    integer: bool = False,
) -> float | int:
    if isinstance(value, (bool, np.bool_)):
        raise IndependentErosionTruthError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise IndependentErosionTruthError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or number < 0.0 or (positive and number <= 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise IndependentErosionTruthError(f"{name} must be finite and {qualifier}")
    if integer:
        if not number.is_integer():
            raise IndependentErosionTruthError(f"{name} must be an integer")
        return int(number)
    return number


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IndependentErosionTruthError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise IndependentErosionTruthError(f"{name} must be an array")
    return list(value)


def _require_measured(value: Mapping[str, Any], name: str) -> str:
    if value.get("status") != "measured":
        raise IndependentErosionTruthError(f"{name} status must be measured")
    source_type = _text(value.get("source_type"), f"{name} source_type")
    declared = [
        source_type,
        *(value[field] for field in ("origin", "data_origin", "truth_source") if value.get(field) is not None),
    ]
    for raw in declared:
        normalized = str(raw).strip().lower().replace("-", "_")
        if any(token in normalized for token in _NON_MEASURED_TOKENS):
            raise IndependentErosionTruthError(
                f"{name} synthetic/DEC/CFD source cannot be classified as measured"
            )
    return source_type


def _source_descriptor(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    if not target.is_file():
        raise IndependentErosionTruthError(
            f"measured source file does not exist: {target}"
        )
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "bytes": int(target.stat().st_size),
    }


def _validated_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return validate_protocol(protocol)
    except (IndependentErosionProtocolError, OSError) as exc:
        raise IndependentErosionTruthError(f"protocol validation failed: {exc}") from exc


def _bind_source(
    path: str | Path,
    *,
    protocol_asset: Mapping[str, Any] | None,
    label: str,
) -> dict[str, Any]:
    actual = _source_descriptor(path)
    if protocol_asset is not None:
        if Path(actual["path"]) != Path(str(protocol_asset["path"])).resolve():
            raise IndependentErosionTruthError(
                f"{label} path does not match the frozen protocol source"
            )
        if actual["sha256"] != str(protocol_asset["sha256"]).lower():
            raise IndependentErosionTruthError(f"{label} sha256 mismatch")
        if actual["bytes"] != protocol_asset["bytes"]:
            raise IndependentErosionTruthError(f"{label} byte count mismatch")
    return actual


def _to_mm(values: Any, units: Any, name: str) -> np.ndarray:
    try:
        source_units = validate_units(units, expected_dimension="length")
        converted = convert_units(values, source_units, "mm")
    except ValidationError as exc:
        raise IndependentErosionTruthError(f"invalid {name} units: {exc}") from exc
    array = np.asarray(converted, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise IndependentErosionTruthError(f"{name} must be nonempty and finite")
    return array


def _clean_registration_options(options: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = dict(options)
    for name in ("measured_units", "reference_units", "allow_unit_conversion"):
        cleaned.pop(name, None)
    return cleaned


def _registration_hashes(registration: Mapping[str, Any]) -> tuple[str, str]:
    if registration.get("status") != "measured":
        raise IndependentErosionTruthError("registration status must be measured")
    transform_hash = _sha256(
        registration.get("transform_sha256", registration.get("transform_hash")),
        "registration transform hash",
    )
    transform = _mapping(registration.get("transform"), "registration transform")
    if canonical_sha256(transform) != transform_hash:
        raise IndependentErosionTruthError("registration transform hash mismatch")
    if registration.get("passed") is not True:
        raise IndependentErosionTruthError("registration quality did not pass")
    quality = _mapping(registration.get("quality"), "registration quality")
    quality_hash = canonical_sha256(quality)
    stored_quality_hash = registration.get("quality_sha256")
    if stored_quality_hash is not None and _sha256(
        stored_quality_hash, "registration quality hash"
    ) != quality_hash:
        raise IndependentErosionTruthError("registration quality hash mismatch")
    return transform_hash, quality_hash


def _registration_binding(registration: Mapping[str, Any]) -> dict[str, Any]:
    transform_hash, quality_hash = _registration_hashes(registration)
    return {
        "status": "measured",
        "passed": True,
        "transform_sha256": transform_hash,
        "quality_sha256": quality_hash,
        "registration_result_sha256": _sha256(
            registration.get("result_sha256"), "registration result hash"
        ),
        "quality": jsonable(registration["quality"]),
        "uncertainty": jsonable(registration["uncertainty"]),
        "transform": jsonable(registration["transform"]),
    }


def _register_to_topology(
    points_mm: np.ndarray,
    topology: Mapping[str, Any],
    options: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        registration = register_surfaces(
            points_mm,
            topology["points_mm"],
            measured_units="mm",
            reference_units="mm",
            allow_unit_conversion=False,
            **_clean_registration_options(options),
        )
    except (ValidationError, ValueError) as exc:
        raise IndependentErosionTruthError(f"registration failed: {exc}") from exc
    return (
        np.asarray(registration["registered_measured_points"], dtype=np.float64),
        _registration_binding(registration),
    )


def _registration_uncertainty(binding: Mapping[str, Any]) -> float:
    uncertainty = _mapping(binding.get("uncertainty"), "registration uncertainty")
    if uncertainty.get("available") is not True:
        raise IndependentErosionTruthError("registration uncertainty is unavailable")
    return float(
        _number(
            uncertainty.get(
                "residual_sigma_mm", uncertainty.get("residual_rms_mm")
            ),
            "registration uncertainty",
        )
    )


def _map_samples(
    topology_path: str | Path,
    points_mm: np.ndarray,
    values_mm: np.ndarray,
    *,
    topology: Mapping[str, Any],
    artifact_set_id: str,
    transform_sha256: str,
    maximum_projection_distance_mm: float,
) -> dict[str, Any]:
    try:
        result = map_measurements_to_parent_faces(
            topology_path,
            points_mm,
            sample_values=values_mm,
            expected_mesh_signature=topology["mesh_signature"],
            expected_source_hash=topology["source_vtp_sha256"],
            artifact_set_id=artifact_set_id,
            transform_sha256=transform_sha256,
            coordinate_frame="topology_local_mm",
            require_artifact_identity=True,
            maximum_projection_distance_mm=maximum_projection_distance_mm,
        )
    except (ValidationError, ValueError) as exc:
        raise IndependentErosionTruthError(f"parent-face mapping failed: {exc}") from exc
    if (
        result.get("artifact_identity_required") is not True
        or result.get("no_identity_fallback") is not True
        or result.get("artifact_set_id") != artifact_set_id
        or result.get("transform_sha256") != transform_sha256
        or result.get("mesh_signature") != topology["mesh_signature"]
    ):
        raise IndependentErosionTruthError(
            "mapper did not retain required artifact identity"
        )
    return result


def _rows_from_mapping(
    record_id: str,
    mapping: Mapping[str, Any],
    values_mm: np.ndarray,
    sample_uncertainty_mm: np.ndarray,
    topology: Mapping[str, Any],
) -> list[dict[str, Any]]:
    parent_ids = np.asarray(mapping["point_parent_face_id"], dtype=np.int64)
    missing = np.asarray(mapping["point_missing_mask"], dtype=bool)
    values = np.asarray(values_mm, dtype=np.float64)
    uncertainty = np.asarray(sample_uncertainty_mm, dtype=np.float64)
    if not (len(parent_ids) == len(missing) == len(values) == len(uncertainty)):
        raise IndependentErosionTruthError("mapped sample arrays have inconsistent lengths")
    valid = ~missing
    valid &= np.isfinite(values) & np.isfinite(uncertainty)
    if np.any(values[valid] < 0.0):
        raise IndependentErosionTruthError("negative wear is not permitted")
    if not np.any(valid):
        raise IndependentErosionTruthError("empty valid wear sample set")
    area_mm2 = np.asarray(topology["face_area_m2"], dtype=np.float64) * 1.0e6
    rows: list[dict[str, Any]] = []
    for face_id in sorted(set(parent_ids[valid].tolist())):
        face_rows = valid & (parent_ids == face_id)
        count = int(np.sum(face_rows))
        if count == 0:
            continue
        face_values = values[face_rows]
        face_sigma = uncertainty[face_rows]
        sampling_sigma = (
            float(np.std(face_values, ddof=1) / math.sqrt(count))
            if count > 1
            else 0.0
        )
        combined_sigma = math.sqrt(
            float(np.mean(np.square(face_sigma))) + sampling_sigma**2
        )
        rows.append(
            {
                "record_id": record_id,
                "parent_face_id": int(face_id),
                "wear_depth_mm": float(np.mean(face_values)),
                "area_mm2": float(
                    _number(area_mm2[face_id], "parent-face area", positive=True)
                ),
                "uncertainty_mm": float(
                    _number(combined_sigma, "parent-face uncertainty")
                ),
                "valid_count": count,
            }
        )
    if not rows:
        raise IndependentErosionTruthError("empty valid parent-face set")
    return rows


def _surface_record(
    spec: Mapping[str, Any],
    *,
    protocol_asset: Mapping[str, Any],
    topology_path: Path,
    topology: Mapping[str, Any],
    artifact_set_id: str,
    registration_options: Mapping[str, Any],
    maximum_projection_distance_mm: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    baseline_path = _text(spec.get("baseline_path"), "baseline_path")
    worn_path = _text(spec.get("worn_path"), "worn_path")
    baseline_source = _bind_source(
        baseline_path, protocol_asset=protocol_asset, label="baseline source"
    )
    frozen_worn = protocol_asset.get("truth_source")
    worn_source = _bind_source(
        worn_path,
        protocol_asset=frozen_worn,
        label="worn source",
    )
    baseline_units = spec.get("baseline_units")
    worn_units = spec.get("worn_units")
    try:
        baseline_unit = validate_units(baseline_units, expected_dimension="length")
        worn_unit = validate_units(worn_units, expected_dimension="length")
        baseline = read_point_cloud(
            baseline_path, status="measured", units=baseline_unit
        )
        worn = read_point_cloud(worn_path, status="measured", units=worn_unit)
    except (ValidationError, OSError) as exc:
        raise IndependentErosionTruthError(
            f"cannot read measured surface source: {exc}"
        ) from exc
    baseline_mm = _to_mm(
        baseline["points"], baseline_unit, "baseline coordinates"
    )
    worn_mm = _to_mm(worn["points"], worn_unit, "worn coordinates")
    registered_baseline, registration = _register_to_topology(
        baseline_mm, topology, registration_options
    )
    registered_worn = apply_transform(worn_mm, registration["transform"])
    baseline_mapping = _map_samples(
        topology_path,
        registered_baseline,
        np.zeros(len(registered_baseline), dtype=np.float64),
        topology=topology,
        artifact_set_id=artifact_set_id,
        transform_sha256=registration["transform_sha256"],
        maximum_projection_distance_mm=maximum_projection_distance_mm,
    )
    baseline_triangle_ids = np.asarray(
        baseline_mapping["point_triangle_id"], dtype=np.int64
    )
    baseline_normals = np.asarray(topology["triangle_normals"], dtype=np.float64)[
        baseline_triangle_ids
    ]
    try:
        difference = surface_difference(
            registered_worn,
            registered_baseline,
            reference_normals=baseline_normals,
            maximum_projection_distance_mm=maximum_projection_distance_mm,
            maximum_tangent_residual_mm=maximum_projection_distance_mm,
            estimate_worn_normals=False,
        )
    except (ValidationError, ValueError) as exc:
        raise IndependentErosionTruthError(f"surface difference failed: {exc}") from exc
    values = np.asarray(difference["wear_depth_mm"], dtype=np.float64)
    valid_difference = ~np.asarray(difference["missing_mask"], dtype=bool)
    reference_indices = np.asarray(
        difference["reference_point_index"], dtype=np.int64
    )
    valid_difference &= ~np.asarray(
        baseline_mapping["point_missing_mask"], dtype=bool
    )[reference_indices]
    if not np.any(valid_difference):
        raise IndependentErosionTruthError("empty valid surface-difference set")
    if not np.isfinite(values[valid_difference]).all():
        raise IndependentErosionTruthError(
            "surface difference contains nonfinite wear"
        )
    if np.any(values[valid_difference] < 0.0):
        raise IndependentErosionTruthError("negative wear is not permitted")
    supported_points = np.asarray(
        difference["projected_points_mm"], dtype=np.float64
    )[valid_difference]
    supported_values = values[valid_difference]
    mapping = _map_samples(
        topology_path,
        supported_points,
        supported_values,
        topology=topology,
        artifact_set_id=artifact_set_id,
        transform_sha256=registration["transform_sha256"],
        maximum_projection_distance_mm=maximum_projection_distance_mm,
    )
    registration_sigma = _registration_uncertainty(registration)
    sample_sigma = np.sqrt(
        registration_sigma**2
        + np.square(
            np.asarray(difference["tangent_residual_mm"], dtype=np.float64)[
                valid_difference
            ]
        )
    )
    rows = _rows_from_mapping(
        str(spec["record_id"]),
        mapping,
        supported_values,
        sample_sigma,
        topology,
    )
    sources = {
        "record_id": str(spec["record_id"]),
        "record_source": "baseline_worn_surface",
        "status": "measured",
        "source_type": _text(spec.get("source_type"), "source_type"),
        "baseline": baseline_source,
        "worn": worn_source,
        "source_sha256": baseline_source["sha256"],
        "source_bytes": baseline_source["bytes"],
    }
    bindings = {
        "record_id": str(spec["record_id"]),
        "topology_registration": registration,
        "surface_difference_sha256": _sha256(
            difference["result_sha256"], "surface difference hash"
        ),
        "baseline_mapping_sha256": _sha256(
            baseline_mapping["result_sha256"], "baseline mapping hash"
        ),
        "mapping_sha256": _sha256(mapping["result_sha256"], "mapping hash"),
    }
    return rows, sources, bindings


def _sparse_record(
    spec: Mapping[str, Any],
    *,
    protocol_asset: Mapping[str, Any],
    topology_path: Path,
    topology: Mapping[str, Any],
    artifact_set_id: str,
    registration_options: Mapping[str, Any],
    maximum_projection_distance_mm: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    baseline_path = _text(spec.get("baseline_path"), "baseline_path")
    worn_path = _text(spec.get("worn_path"), "worn_path")
    baseline_source = _bind_source(
        baseline_path,
        protocol_asset=protocol_asset,
        label="baseline thickness source",
    )
    frozen_worn = protocol_asset.get("truth_source")
    worn_source = _bind_source(
        worn_path,
        protocol_asset=frozen_worn,
        label="worn thickness source",
    )
    coordinate_units = spec.get("coordinate_units")
    thickness_units = spec.get("thickness_units")
    try:
        coordinate_unit = validate_units(
            coordinate_units, expected_dimension="length"
        )
        thickness_unit = validate_units(
            thickness_units, expected_dimension="length"
        )
        baseline = read_sparse_measurement_table(
            baseline_path,
            kind="thickness",
            status="measured",
            coordinate_units=coordinate_unit,
            value_units=thickness_unit,
        )
        worn = read_sparse_measurement_table(
            worn_path,
            kind="thickness",
            status="measured",
            coordinate_units=coordinate_unit,
            value_units=thickness_unit,
        )
    except (ValidationError, OSError) as exc:
        raise IndependentErosionTruthError(
            f"cannot read sparse thickness source: {exc}"
        ) from exc
    baseline_points_mm = _to_mm(
        baseline["coordinates"], coordinate_unit, "baseline thickness coordinates"
    )
    worn_points_mm = _to_mm(
        worn["coordinates"], coordinate_unit, "worn thickness coordinates"
    )
    baseline_values_mm = _to_mm(
        baseline["values"], thickness_unit, "baseline thickness"
    )
    worn_values_mm = _to_mm(
        worn["values"], thickness_unit, "worn thickness"
    )
    if len(baseline_points_mm) != len(worn_points_mm) or not np.allclose(
        baseline_points_mm, worn_points_mm, rtol=0.0, atol=1.0e-9
    ):
        raise IndependentErosionTruthError(
            "baseline and worn sparse thickness coordinates must match exactly"
        )
    if np.any(baseline_values_mm < 0.0) or np.any(worn_values_mm < 0.0):
        raise IndependentErosionTruthError("thickness values must be non-negative")
    if np.any((baseline_values_mm - worn_values_mm) < 0.0):
        raise IndependentErosionTruthError("negative wear is not permitted")
    registered_points, registration = _register_to_topology(
        baseline_points_mm, topology, registration_options
    )
    try:
        sparse = interpolate_sparse_thickness(
            registered_points,
            baseline_values_mm,
            worn_values_mm,
            query_points=registered_points,
            method=str(spec.get("interpolation_method", "nearest")),
            support_radius_mm=spec.get("support_radius_mm"),
            coordinate_units="mm",
            thickness_units="mm",
        )
    except (ValidationError, ValueError) as exc:
        raise IndependentErosionTruthError(
            f"sparse thickness interpolation failed: {exc}"
        ) from exc
    values = np.asarray(sparse["thickness_change_mm"], dtype=np.float64)
    valid_sparse = ~np.asarray(sparse["missing_mask"], dtype=bool)
    if not np.any(valid_sparse):
        raise IndependentErosionTruthError("empty valid sparse thickness set")
    if not np.isfinite(values[valid_sparse]).all():
        raise IndependentErosionTruthError(
            "sparse thickness contains nonfinite wear"
        )
    if np.any(values[valid_sparse] < 0.0):
        raise IndependentErosionTruthError("negative wear is not permitted")
    supported = valid_sparse & np.isfinite(values)
    mapping = _map_samples(
        topology_path,
        registered_points[supported],
        values[supported],
        topology=topology,
        artifact_set_id=artifact_set_id,
        transform_sha256=registration["transform_sha256"],
        maximum_projection_distance_mm=maximum_projection_distance_mm,
    )
    registration_sigma = _registration_uncertainty(registration)
    sparse_sigma = np.asarray(sparse["sigma_mm"], dtype=np.float64)[supported]
    values = values[supported]
    sample_sigma = np.sqrt(registration_sigma**2 + np.square(sparse_sigma))
    rows = _rows_from_mapping(
        str(spec["record_id"]), mapping, values, sample_sigma, topology
    )
    sources = {
        "record_id": str(spec["record_id"]),
        "record_source": "sparse_thickness",
        "status": "measured",
        "source_type": _text(spec.get("source_type"), "source_type"),
        "baseline": baseline_source,
        "worn": worn_source,
        "source_sha256": baseline_source["sha256"],
        "source_bytes": baseline_source["bytes"],
    }
    bindings = {
        "record_id": str(spec["record_id"]),
        "topology_registration": registration,
        "sparse_thickness_sha256": _sha256(
            sparse["result_sha256"], "sparse thickness hash"
        ),
        "mapping_sha256": _sha256(mapping["result_sha256"], "mapping hash"),
    }
    return rows, sources, bindings


def recompute_truth_hash(truth: Mapping[str, Any]) -> str:
    unsigned = dict(jsonable(truth))
    unsigned.pop("truth_sha256", None)
    return canonical_sha256(unsigned)


def build_independent_erosion_truth(
    protocol: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    topology_path: str | Path,
    topology_v3_sha256: str,
    mesh_signature: str,
    artifact_set_id: str,
    registration_options: Mapping[str, Any] | None = None,
    maximum_projection_distance_mm: float = 2.0,
) -> dict[str, Any]:
    """Construct measured parent-face truth for the frozen independent cohort."""
    frozen = _validated_protocol(protocol)
    assets = {str(asset["record_id"]): asset for asset in frozen["assets"]}
    artifact_id = _sha256(artifact_set_id, "artifact_set_id")
    expected_topology_hash = _sha256(
        topology_v3_sha256, "topology_v3_sha256"
    )
    expected_mesh_signature = _sha256(mesh_signature, "mesh_signature")
    topology_file = Path(topology_path).resolve()
    if not topology_file.is_file():
        raise IndependentErosionTruthError(
            f"topology v3 artifact does not exist: {topology_file}"
        )
    if sha256_file(topology_file) != expected_topology_hash:
        raise IndependentErosionTruthError("topology v3 sha256 mismatch")
    try:
        topology = load_topology_v3(
            topology_file, expected_mesh_signature=expected_mesh_signature
        )
    except (ValidationError, ValueError) as exc:
        raise IndependentErosionTruthError(
            f"topology v3 validation failed: {exc}"
        ) from exc
    if topology.get("artifact_version") != TOPOLOGY_V3_VERSION:
        raise IndependentErosionTruthError("topology artifact is not v3")
    if not math.isfinite(maximum_projection_distance_mm) or (
        maximum_projection_distance_mm <= 0.0
    ):
        raise IndependentErosionTruthError(
            "maximum_projection_distance_mm must be finite and positive"
        )

    record_specs = _sequence(records, "records")
    if not record_specs:
        raise IndependentErosionTruthError("records must not be empty")
    normalized: list[Mapping[str, Any]] = []
    seen_records: set[str] = set()
    for raw in record_specs:
        spec = _mapping(raw, "record")
        record_id = _text(spec.get("record_id"), "record_id")
        if record_id in seen_records:
            raise IndependentErosionTruthError("record_id values must be unique")
        seen_records.add(record_id)
        if record_id not in assets:
            raise IndependentErosionTruthError(
                f"record_id is absent from frozen protocol: {record_id}"
            )
        _require_measured(spec, f"record {record_id}")
        if "truth_source" not in assets[str(record_id)]:
            raise IndependentErosionTruthError(
                f"record {record_id} has no frozen truth_source in protocol"
            )
        if spec.get("record_source") not in RECORD_SOURCE_TYPES:
            raise IndependentErosionTruthError(
                f"record_source must be one of {sorted(RECORD_SOURCE_TYPES)}"
            )
        normalized.append(spec)
    expected_records = set(frozen["splits"]["independent_test"])
    if seen_records != expected_records:
        raise IndependentErosionTruthError(
            "truth records must exactly match the frozen independent_test cohort"
        )

    options = dict(registration_options or {})
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    registrations: list[dict[str, Any]] = []
    for spec in sorted(normalized, key=lambda item: str(item["record_id"])):
        common = {
            "protocol_asset": assets[str(spec["record_id"])],
            "topology_path": topology_file,
            "topology": topology,
            "artifact_set_id": artifact_id,
            "registration_options": options,
            "maximum_projection_distance_mm": float(
                maximum_projection_distance_mm
            ),
        }
        if spec["record_source"] == "baseline_worn_surface":
            record_rows, source_binding, registration_binding = _surface_record(
                spec, **common
            )
        else:
            record_rows, source_binding, registration_binding = _sparse_record(
                spec, **common
            )
        rows.extend(record_rows)
        sources.append(source_binding)
        registrations.append(registration_binding)
    rows.sort(key=lambda row: (str(row["record_id"]), row["parent_face_id"]))
    parent_ids = [(row["record_id"], row["parent_face_id"]) for row in rows]
    if len(parent_ids) != len(set(parent_ids)):
        raise IndependentErosionTruthError(
            "(record_id, parent_face_id) values must be unique across truth records"
        )
    result = {
        "version": TRUTH_VERSION,
        "status": "measured",
        "evaluation_status": "truth_built",
        "protocol_sha256": frozen["protocol_sha256"],
        "artifact_set_id": artifact_id,
        "topology_path": str(topology_file),
        "topology_v3_sha256": expected_topology_hash,
        "topology_artifact_version": TOPOLOGY_V3_VERSION,
        "topology_source_vtp_sha256": topology["source_vtp_sha256"],
        "mesh_signature": topology["mesh_signature"],
        "coordinate_units": "mm",
        "wear_units": "mm",
        "area_units": "mm2",
        "uncertainty_units": "mm",
        "record_sources": sources,
        "registration_bindings": registrations,
        "rows": rows,
        "n_records": len(sources),
        # This is the frozen topology cardinality; rows remain sparse explicit coverage.
        "n_parent_faces": int(topology["n_parent_faces"]),
        "official_reference_gate": False,
    }
    result["truth_sha256"] = recompute_truth_hash(result)
    return validate_independent_erosion_truth(result, protocol=protocol)


def validate_independent_erosion_truth(
    truth: Any,
    *,
    protocol: Mapping[str, Any] | None = None,
    verify_sources: bool = True,
    topology_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate content hashes, source bytes, identities, units, and rows."""
    if not isinstance(truth, Mapping) or set(truth) != _TRUTH_KEYS:
        raise IndependentErosionTruthError(
            "independent erosion truth fields are invalid"
        )
    if (
        truth.get("version") != TRUTH_VERSION
        or truth.get("status") != "measured"
        or truth.get("evaluation_status") != "truth_built"
        or truth.get("official_reference_gate") is not False
    ):
        raise IndependentErosionTruthError(
            "independent erosion truth contract is invalid"
        )
    if (
        truth.get("coordinate_units"),
        truth.get("wear_units"),
        truth.get("area_units"),
        truth.get("uncertainty_units"),
    ) != ("mm", "mm", "mm2", "mm"):
        raise IndependentErosionTruthError("truth units must be canonical mm/mm2")
    for field in (
        "protocol_sha256",
        "artifact_set_id",
        "topology_v3_sha256",
        "topology_source_vtp_sha256",
        "mesh_signature",
        "truth_sha256",
    ):
        _sha256(truth.get(field), field)
    if truth.get("topology_artifact_version") != TOPOLOGY_V3_VERSION:
        raise IndependentErosionTruthError(
            "truth topology artifact version mismatch"
        )
    if truth.get("truth_sha256") != recompute_truth_hash(truth):
        raise IndependentErosionTruthError("truth_sha256/content mismatch")
    protocol_assets: dict[str, Mapping[str, Any]] | None = None
    if protocol is not None:
        frozen = _validated_protocol(protocol)
        if truth.get("protocol_sha256") != frozen["protocol_sha256"]:
            raise IndependentErosionTruthError("truth protocol_sha256 mismatch")
        protocol_assets = {
            str(asset["record_id"]): asset for asset in frozen["assets"]
        }

    rows_raw = _sequence(truth.get("rows"), "truth rows")
    if not rows_raw:
        raise IndependentErosionTruthError("truth rows must not be empty")
    rows: list[dict[str, Any]] = []
    seen_faces: set[tuple[str, int]] = set()
    for raw in rows_raw:
        if not isinstance(raw, Mapping) or set(raw) != _ROW_KEYS:
            raise IndependentErosionTruthError("truth row fields are invalid")
        row = dict(raw)
        _text(row.get("record_id"), "truth row record_id")
        face_id = int(
            _number(row.get("parent_face_id"), "parent_face_id", integer=True)
        )
        identity = (str(row["record_id"]), face_id)
        if identity in seen_faces:
            raise IndependentErosionTruthError(
                "(record_id, parent_face_id) values must be unique"
            )
        seen_faces.add(identity)
        _number(row.get("wear_depth_mm"), "wear_depth_mm")
        _number(row.get("area_mm2"), "area_mm2", positive=True)
        _number(row.get("uncertainty_mm"), "uncertainty_mm")
        _number(
            row.get("valid_count"), "valid_count", positive=True, integer=True
        )
        rows.append(row)
    if rows != sorted(
        rows, key=lambda item: (str(item["record_id"]), item["parent_face_id"])
    ):
        raise IndependentErosionTruthError(
            "truth rows must be sorted by record_id and parent_face_id"
        )

    sources = _sequence(truth.get("record_sources"), "record_sources")
    registrations = _sequence(
        truth.get("registration_bindings"), "registration_bindings"
    )
    if not sources or len(sources) != len(registrations):
        raise IndependentErosionTruthError(
            "truth source/registration bindings are incomplete"
        )
    source_ids: list[str] = []
    for raw in sources:
        source = _mapping(raw, "record source binding")
        _require_measured(source, "record source binding")
        record_id = _text(source.get("record_id"), "record source record_id")
        source_ids.append(record_id)
        if source.get("record_source") not in RECORD_SOURCE_TYPES:
            raise IndependentErosionTruthError("record source type is invalid")
        frozen_hash = _sha256(source.get("source_sha256"), "source_sha256")
        frozen_bytes = int(
            _number(source.get("source_bytes"), "source_bytes", integer=True)
        )
        baseline = _mapping(source.get("baseline"), "baseline source")
        if (
            baseline.get("sha256") != frozen_hash
            or baseline.get("bytes") != frozen_bytes
        ):
            raise IndependentErosionTruthError(
                "protocol source binding is inconsistent"
            )
        if protocol_assets is not None:
            asset = protocol_assets.get(record_id)
            if asset is None or (
                Path(str(baseline.get("path"))).resolve()
                != Path(str(asset.get("path"))).resolve()
                or frozen_hash != str(asset.get("sha256", "")).lower()
                or frozen_bytes != asset.get("bytes")
            ):
                raise IndependentErosionTruthError(
                    "record source differs from frozen protocol asset"
                )
            frozen_truth_source = asset.get("truth_source")
            worn_binding = _mapping(source.get("worn"), "worn source")
            if not isinstance(frozen_truth_source, Mapping) or (
                Path(str(worn_binding.get("path"))).resolve()
                != Path(str(frozen_truth_source.get("path"))).resolve()
                or str(worn_binding.get("sha256", "")).lower()
                != str(frozen_truth_source.get("sha256", "")).lower()
                or worn_binding.get("bytes") != frozen_truth_source.get("bytes")
            ):
                raise IndependentErosionTruthError(
                    "worn source differs from frozen protocol truth_source"
                )
        for name in ("baseline", "worn"): 
            descriptor = _mapping(source.get(name), f"{name} source")
            digest = _sha256(
                descriptor.get("sha256"), f"{name} source sha256"
            )
            byte_count = int(
                _number(
                    descriptor.get("bytes"), f"{name} source bytes", integer=True
                )
            )
            path = Path(_text(descriptor.get("path"), f"{name} source path"))
            if verify_sources:
                if not path.is_file():
                    raise IndependentErosionTruthError(
                        f"measured source file does not exist: {path}"
                    )
                if path.stat().st_size != byte_count or sha256_file(path) != digest:
                    raise IndependentErosionTruthError(
                        f"measured source hash/bytes mismatch: {path}"
                    )
    if len(source_ids) != len(set(source_ids)):
        raise IndependentErosionTruthError("record source IDs must be unique")
    if set(source_ids) != {row["record_id"] for row in rows}:
        raise IndependentErosionTruthError("truth rows/source record IDs differ")
    if truth.get("n_records") != len(sources):
        raise IndependentErosionTruthError("truth record count is inconsistent")
    n_parent_faces = truth.get("n_parent_faces")
    if isinstance(n_parent_faces, bool) or not isinstance(n_parent_faces, int) or n_parent_faces <= 0:
        raise IndependentErosionTruthError("truth n_parent_faces must be a positive integer")
    if any(identity[1] >= n_parent_faces for identity in seen_faces):
        raise IndependentErosionTruthError("truth parent-face identity exceeds n_parent_faces")

    registration_ids: list[str] = []
    for raw in registrations:
        binding = _mapping(raw, "registration binding")
        registration_ids.append(
            _text(binding.get("record_id"), "registration record_id")
        )
        found_registration = False
        for value in binding.values():
            if isinstance(value, Mapping) and "transform_sha256" in value:
                found_registration = True
                _registration_hashes(value)
                _sha256(
                    value.get("registration_result_sha256"),
                    "registration result hash",
                )
        if not found_registration:
            raise IndependentErosionTruthError(
                "registration transform/quality binding is missing"
            )
        for field, value in binding.items():
            if field.endswith("_sha256") and not isinstance(value, Mapping):
                _sha256(value, field)
    if set(registration_ids) != set(source_ids):
        raise IndependentErosionTruthError(
            "registration/source record IDs differ"
        )

    frozen_topology_path = Path(
        _text(truth.get("topology_path"), "topology_path")
    ).resolve()
    if topology_path is not None and Path(topology_path).resolve() != frozen_topology_path:
        raise IndependentErosionTruthError("topology path differs from truth binding")
    if verify_sources:
        if not frozen_topology_path.is_file():
            raise IndependentErosionTruthError(
                "topology v3 artifact does not exist"
            )
        if sha256_file(frozen_topology_path) != truth["topology_v3_sha256"]:
            raise IndependentErosionTruthError("topology v3 sha256 mismatch")
        try:
            topology = load_topology_v3(
                frozen_topology_path,
                expected_mesh_signature=truth["mesh_signature"],
                expected_source_hash=truth["topology_source_vtp_sha256"],
            )
        except (ValidationError, ValueError) as exc:
            raise IndependentErosionTruthError(
                f"topology identity mismatch: {exc}"
            ) from exc
        if int(topology["n_parent_faces"]) != n_parent_faces:
            raise IndependentErosionTruthError(
                "truth n_parent_faces differs from topology"
            )
    return dict(truth)


def save_independent_erosion_truth(
    truth: Mapping[str, Any],
    path: str | Path,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return safe_json_dump(
        validate_independent_erosion_truth(truth, protocol=protocol), path
    )


def load_independent_erosion_truth(
    path: str | Path,
    *,
    protocol: Mapping[str, Any] | None = None,
    topology_path: str | Path | None = None,
) -> dict[str, Any]:
    try:
        document = safe_json_load(path)
    except (OSError, ValidationError) as exc:
        raise IndependentErosionTruthError(f"cannot load truth: {path}") from exc
    return validate_independent_erosion_truth(
        document, protocol=protocol, topology_path=topology_path
    )


def recompute_reveal_receipt_hash(receipt: Mapping[str, Any]) -> str:
    unsigned = dict(jsonable(receipt))
    unsigned.pop("receipt_sha256", None)
    return canonical_sha256(unsigned)


def build_reveal_receipt(
    protocol: Mapping[str, Any],
    commitment: Mapping[str, Any] | None = None,
    truth: Mapping[str, Any] | None = None,
    *,
    commitment_path: str | Path | None = None,
    truth_path: str | Path | None = None,
    topology_path: str | Path | None = None,
) -> dict[str, Any]:
    """Bind a pre-reveal commitment to truth or return a blocked receipt."""
    reason: str | None = None
    frozen: dict[str, Any] | None = None
    valid_commitment: dict[str, Any] | None = None
    valid_truth: dict[str, Any] | None = None
    try:
        frozen = validate_protocol(protocol)
    except (ValidationError, OSError) as exc:
        reason = f"protocol validation failed: {exc}"
    if reason is None:
        try:
            if commitment is not None and commitment_path is not None:
                raise IndependentErosionTruthError(
                    "provide commitment or commitment_path, not both"
                )
            if commitment_path is not None:
                path = Path(commitment_path)
                if not path.is_file():
                    raise IndependentErosionTruthError(
                        "prediction commitment file is missing"
                    )
                commitment = safe_json_load(path)
            if commitment is None:
                raise IndependentErosionTruthError(
                    "prediction commitment is missing"
                )
            valid_commitment = validate_prediction_commitment(commitment, frozen)
        except (PredictionCommitmentError, ValidationError, OSError) as exc:
            reason = f"prediction commitment invalid: {exc}"
    if reason is None:
        try:
            if truth is not None and truth_path is not None:
                raise IndependentErosionTruthError(
                    "provide truth or truth_path, not both"
                )
            if truth_path is not None:
                path = Path(truth_path)
                if not path.is_file():
                    raise IndependentErosionTruthError("truth file is missing")
                truth = safe_json_load(path)
            if truth is None:
                raise IndependentErosionTruthError("truth file is missing")
            valid_truth = validate_independent_erosion_truth(
                truth, protocol=frozen, topology_path=topology_path
            )
            prediction_rows = {
                (row["record_id"], row["parent_face_id"]): row
                for row in valid_commitment["predictions"]
            }
            truth_rows = {
                (row["record_id"], row["parent_face_id"]): row
                for row in valid_truth["rows"]
            }
            if set(prediction_rows) != set(truth_rows):
                raise IndependentErosionTruthError(
                    "prediction commitment and truth parent-face identities differ"
                )
            for identity, prediction in prediction_rows.items():
                truth_area = float(truth_rows[identity]["area_mm2"])
                prediction_area = float(prediction["parent_face_area_mm2"])
                if not math.isclose(
                    truth_area, prediction_area, rel_tol=1.0e-12, abs_tol=1.0e-12
                ):
                    raise IndependentErosionTruthError(
                        "prediction commitment and truth parent-face areas differ"
                    )
        except (ValidationError, OSError) as exc:
            reason = f"truth invalid: {exc}"
    receipt = {
        "version": REVEAL_RECEIPT_VERSION,
        "evaluation_status": "blocked" if reason is not None else "ready",
        "reason": reason,
        "protocol_sha256": None if frozen is None else frozen["protocol_sha256"],
        "prediction_commitment_sha256": (
            None
            if valid_commitment is None
            else valid_commitment["commitment_sha256"]
        ),
        "truth_sha256": (
            None if valid_truth is None else valid_truth["truth_sha256"]
        ),
        "official_reference_gate": False,
    }
    receipt["receipt_sha256"] = recompute_reveal_receipt_hash(receipt)
    return validate_reveal_receipt(receipt)


def validate_reveal_receipt(receipt: Any) -> dict[str, Any]:
    """Validate a ready or blocked reveal receipt without opening truth."""
    if not isinstance(receipt, Mapping) or set(receipt) != _RECEIPT_KEYS:
        raise IndependentErosionTruthError("reveal receipt fields are invalid")
    if (
        receipt.get("version") != REVEAL_RECEIPT_VERSION
        or receipt.get("official_reference_gate") is not False
        or receipt.get("evaluation_status") not in {"ready", "blocked"}
    ):
        raise IndependentErosionTruthError("reveal receipt contract is invalid")
    _sha256(receipt.get("receipt_sha256"), "receipt_sha256")
    protocol_hash = receipt.get("protocol_sha256")
    if protocol_hash is not None:
        _sha256(protocol_hash, "receipt protocol_sha256")
    if receipt["evaluation_status"] == "ready":
        _sha256(protocol_hash, "receipt protocol_sha256")
        if receipt.get("reason") is not None:
            raise IndependentErosionTruthError("ready receipt must not have a reason")
        _sha256(
            receipt.get("prediction_commitment_sha256"),
            "prediction_commitment_sha256",
        )
        _sha256(receipt.get("truth_sha256"), "truth_sha256")
    else:
        _text(receipt.get("reason"), "blocked receipt reason")
        for field in ("prediction_commitment_sha256", "truth_sha256"):
            if receipt.get(field) is not None:
                _sha256(receipt[field], field)
    if receipt["receipt_sha256"] != recompute_reveal_receipt_hash(receipt):
        raise IndependentErosionTruthError("receipt_sha256/content mismatch")
    return dict(receipt)


def save_reveal_receipt(
    receipt: Mapping[str, Any], path: str | Path
) -> dict[str, Any]:
    return safe_json_dump(validate_reveal_receipt(receipt), path)


def load_reveal_receipt(path: str | Path) -> dict[str, Any]:
    try:
        document = safe_json_load(path)
    except (OSError, ValidationError) as exc:
        raise IndependentErosionTruthError(
            f"cannot load reveal receipt: {path}"
        ) from exc
    return validate_reveal_receipt(document)


build_truth = build_independent_erosion_truth
validate_truth = validate_independent_erosion_truth
save_truth = save_independent_erosion_truth
load_truth = load_independent_erosion_truth
build_receipt = build_reveal_receipt
validate_receipt = validate_reveal_receipt
save_receipt = save_reveal_receipt
load_receipt = load_reveal_receipt
