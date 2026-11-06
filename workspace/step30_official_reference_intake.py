"""Read-only intake for a contest-provided reference simulation package.

The intake never upgrades local fixtures or DEC engineering context into
contest evidence. It normalizes explicitly declared organizer records, binds
all source files by SHA-256, and creates immutable inputs for the strict
reference comparator.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from step30_cfd_topology import load_topology_artifact
from step30_validation_common import (
    ValidationError,
    canonical_sha256,
    safe_json_dump,
    safe_json_load,
    sha256_file,
)

VERSION = "step30-official-reference-intake-v1"
SPEC_VERSION = "step30-official-reference-intake-spec-v1"
RECORDS_VERSION = "step30-official-reference-records-v1"
MANIFEST_VERSION = "step30-official-reference-manifest-v2"
MANIFEST_KIND = "official_reference_package"
EVALUATION_VERSION = "step30-reference-evaluation-contract-v1"
TOPOLOGY_VERSION = "step30-native-openfoam-topology-v3"
TOPOLOGY_SIGNATURE_ALGORITHM = "sha256-vtk-triangulation-source-units-points-triangles-parent-ids-v3"

_UNIT_INFO = {
    "m": ("length", 1.0),
    "cm": ("length", 1.0e-2),
    "mm": ("length", 1.0e-3),
    "um": ("length", 1.0e-6),
    "m2": ("area", 1.0),
    "cm2": ("area", 1.0e-4),
    "mm2": ("area", 1.0e-6),
    "m3": ("volume", 1.0),
    "cm3": ("volume", 1.0e-6),
    "mm3": ("volume", 1.0e-9),
    "kg": ("mass", 1.0),
    "g": ("mass", 1.0e-3),
    "mg": ("mass", 1.0e-6),
    "m/h": ("length_rate", 1.0),
    "mm/h": ("length_rate", 1.0e-3),
    "um/h": ("length_rate", 1.0e-6),
    "kg/kg": ("mass_ratio", 1.0),
    "mm3/kg": ("volume_ratio", 1.0),
    "dimensionless": ("dimensionless", 1.0),
}
_METRIC_UNITS = {
    "wear_depth": {"m", "cm", "mm", "um"},
    "erosion_rate": {"m/h", "mm/h", "um/h"},
    "erosion_area": {"m2", "cm2", "mm2"},
    "erosion_volume": {"m3", "cm3", "mm3"},
    "erosion_mass": {"kg", "g", "mg"},
    "erosion_ratio": {"kg/kg", "mm3/kg"},
    "dimensionless": {"dimensionless"},
}
_UNIT_ALIASES = {
    "m^2": "m2",
    "cm^2": "cm2",
    "mm^2": "mm2",
    "m^3": "m3",
    "cm^3": "cm3",
    "mm^3": "mm3",
    "micrometer": "um",
    "micrometre": "um",
    "micrometers": "um",
    "micrometres": "um",
    "μm": "um",
    "µm": "um",
    "1": "dimensionless",
}


class OfficialReferenceIntakeError(ValueError):
    """Raised when an organizer package cannot be normalized safely."""


def _hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdefABCDEF" for character in value
    )


def _unit(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OfficialReferenceIntakeError("units must be explicit non-empty strings")
    normalized = value.strip().lower().replace("²", "2").replace("³", "3")
    normalized = _UNIT_ALIASES.get(normalized, normalized)
    if normalized not in _UNIT_INFO:
        raise OfficialReferenceIntakeError(f"unsupported official reference unit: {value!r}")
    return normalized


def _convert(value: Any, source_units: str, target_units: str) -> float:
    if isinstance(value, bool):
        raise OfficialReferenceIntakeError("reference value must be numeric")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise OfficialReferenceIntakeError("reference value must be numeric") from exc
    if not math.isfinite(number) or number < 0.0:
        raise OfficialReferenceIntakeError("reference value must be finite and non-negative")
    source = _unit(source_units)
    target = _unit(target_units)
    source_dimension, source_factor = _UNIT_INFO[source]
    target_dimension, target_factor = _UNIT_INFO[target]
    if source_dimension != target_dimension:
        raise OfficialReferenceIntakeError(
            f"incompatible official reference units: {source_units!r} and {target_units!r}"
        )
    return number * source_factor / target_factor


def _safe_relative(root: Path, raw: Any, label: str) -> tuple[str, Path]:
    if not isinstance(raw, str) or not raw.strip():
        raise OfficialReferenceIntakeError(f"{label} path is missing")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise OfficialReferenceIntakeError(f"{label} path must be relative and stay inside source_root")
    target = (root / relative).resolve()
    try:
        normalized = target.relative_to(root).as_posix()
    except ValueError as exc:
        raise OfficialReferenceIntakeError(f"{label} path escapes source_root") from exc
    if not target.is_file():
        raise OfficialReferenceIntakeError(f"{label} does not exist: {raw}")
    return normalized, target


def _artifact(root: Path, role: str, raw: Any) -> dict[str, Any]:
    relative, path = _safe_relative(root, raw, role)
    return {
        "role": role,
        "path": relative,
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def _load_rows(path: Path, format_name: str, records_key: str) -> tuple[list[dict[str, Any]], list[str]]:
    if format_name == "json":
        try:
            document = safe_json_load(path)
        except (OSError, UnicodeError, ValidationError) as exc:
            raise OfficialReferenceIntakeError(f"invalid JSON record source: {path}") from exc
        raw = document.get(records_key) if isinstance(document, Mapping) else document
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)) or not raw:
            raise OfficialReferenceIntakeError(f"JSON source has no non-empty {records_key!r} list")
        if not all(isinstance(row, Mapping) for row in raw):
            raise OfficialReferenceIntakeError("JSON reference records must be objects")
        rows = [dict(row) for row in raw]
        headers = sorted({str(key) for row in rows for key in row})
        return rows, headers
    if format_name == "csv":
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                headers = list(reader.fieldnames or [])
                if len(headers) != len(set(headers)):
                    raise OfficialReferenceIntakeError("CSV reference source has duplicate headers")
                rows = [dict(row) for row in reader]
        except (OSError, UnicodeError, csv.Error) as exc:
            raise OfficialReferenceIntakeError(f"invalid CSV record source: {path}") from exc
        if not headers or not rows:
            raise OfficialReferenceIntakeError("CSV reference source is empty")
        return rows, headers
    raise OfficialReferenceIntakeError("record format must be json or csv")


def _parent_face_id(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise OfficialReferenceIntakeError(f"{label} must be an integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        result = int(value.strip())
    else:
        raise OfficialReferenceIntakeError(f"{label} must be an integer")
    if result < 0:
        raise OfficialReferenceIntakeError(f"{label} must be non-negative")
    return result


def normalize_records(source_path: str | Path, adapter: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize explicitly declared JSON/CSV columns without inferring semantics."""
    path = Path(source_path).resolve()
    if not path.is_file():
        raise OfficialReferenceIntakeError(f"record source does not exist: {path}")
    format_name = str(adapter.get("format", path.suffix.lstrip("."))).lower()
    records_key = str(adapter.get("records_key", "records"))
    id_field = adapter.get("parent_face_id_field")
    value_field = adapter.get("value_field")
    output_field = adapter.get("canonical_value_field", value_field)
    unit_field = adapter.get("unit_field")
    source_units = _unit(adapter.get("source_units"))
    canonical_units = _unit(adapter.get("canonical_units"))
    value_metric = adapter.get("value_metric")
    if id_field != "parent_face_id" and not isinstance(id_field, str):
        raise OfficialReferenceIntakeError("adapter.parent_face_id_field is missing")
    if not isinstance(value_field, str) or not value_field:
        raise OfficialReferenceIntakeError("adapter.value_field is missing")
    if not isinstance(output_field, str) or not output_field:
        raise OfficialReferenceIntakeError("adapter.canonical_value_field is invalid")
    if not isinstance(value_metric, str) or value_metric not in _METRIC_UNITS:
        raise OfficialReferenceIntakeError("adapter.value_metric is invalid")
    if source_units not in _METRIC_UNITS[value_metric] or canonical_units not in _METRIC_UNITS[value_metric]:
        raise OfficialReferenceIntakeError("adapter metric and units are dimensionally inconsistent")
    rows, headers = _load_rows(path, format_name, records_key)
    required = {str(id_field), value_field}
    if unit_field is not None:
        if not isinstance(unit_field, str) or not unit_field:
            raise OfficialReferenceIntakeError("adapter.unit_field is invalid")
        required.add(unit_field)
    missing = sorted(required - set(headers))
    if missing:
        raise OfficialReferenceIntakeError(f"record source missing declared columns: {missing}")
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, row in enumerate(rows):
        identifier = _parent_face_id(row.get(str(id_field)), f"row {index} parent_face_id")
        if identifier in seen:
            raise OfficialReferenceIntakeError(f"duplicate parent_face_id: {identifier}")
        if unit_field is not None and _unit(row.get(unit_field)) != source_units:
            raise OfficialReferenceIntakeError(f"row {index} unit differs from declared source_units")
        value = _convert(row.get(value_field), source_units, canonical_units)
        normalized.append(
            {
                "parent_face_id": identifier,
                output_field: value,
                "value_units": canonical_units,
            }
        )
        seen.add(identifier)
    normalized.sort(key=lambda row: row["parent_face_id"])
    records_hash = canonical_sha256({"records": normalized})
    result = {
        "version": RECORDS_VERSION,
        "status": "synthetic",
        "source_type": "contest_provided_simulation",
        "reference_role": "official_reference_simulation",
        "pipeline_validation": False,
        "official_reference": True,
        "value_metric": value_metric,
        "source": {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
            "format": format_name,
        },
        "normalization": {
            "parent_face_id_field": id_field,
            "source_value_field": value_field,
            "canonical_value_field": output_field,
            "unit_field": unit_field,
            "source_units": source_units,
            "canonical_units": canonical_units,
            "conversion_factor": _UNIT_INFO[source_units][1] / _UNIT_INFO[canonical_units][1],
            "inferred_fields": False,
        },
        "records": normalized,
        "records_sha256": records_hash,
    }
    result["document_sha256"] = canonical_sha256(result)
    return result


def _load_split(path: Path, spec: Mapping[str, Any]) -> list[int]:
    rows, headers = _load_rows(
        path,
        str(spec.get("format", path.suffix.lstrip("."))).lower(),
        str(spec.get("records_key", "records")),
    )
    field = spec.get("parent_face_id_field")
    if not isinstance(field, str) or field not in headers:
        raise OfficialReferenceIntakeError("split source lacks declared parent_face_id_field")
    values = [_parent_face_id(row.get(field), f"split row {index}") for index, row in enumerate(rows)]
    if len(values) != len(set(values)) or not values:
        raise OfficialReferenceIntakeError("split parent_face_ids must be non-empty and unique")
    return sorted(values)


def _load_spec(value: Mapping[str, Any] | str | Path) -> tuple[dict[str, Any], Path | None]:
    if isinstance(value, (str, Path)):
        path = Path(value).resolve()
        try:
            document = safe_json_load(path)
        except (OSError, UnicodeError, ValidationError) as exc:
            raise OfficialReferenceIntakeError(f"invalid intake spec: {path}") from exc
        origin = path
    else:
        document, origin = value, None
    if not isinstance(document, Mapping):
        raise OfficialReferenceIntakeError("intake spec must be an object")
    document = dict(document)
    if document.get("version") != SPEC_VERSION:
        raise OfficialReferenceIntakeError(f"intake spec must use {SPEC_VERSION}")
    return document, origin


def _required_mapping(document: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = document.get(field)
    if not isinstance(value, Mapping):
        raise OfficialReferenceIntakeError(f"intake spec missing {field}")
    return dict(value)


def _evaluation_contract(
    value: Any,
    record_contract: Mapping[str, Any],
    evaluation_definition_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OfficialReferenceIntakeError("intake spec missing evaluation_contract")
    contract = dict(value)
    if contract.get("version") != EVALUATION_VERSION:
        raise OfficialReferenceIntakeError(f"evaluation_contract must use {EVALUATION_VERSION}")
    declared_definition_hash = contract.get("definition_sha256")
    if declared_definition_hash != evaluation_definition_sha256:
        raise OfficialReferenceIntakeError("evaluation_contract definition_sha256 differs from official definition artifact")
    metric = contract.get("value_metric")
    units = _unit(contract.get("canonical_units"))
    if metric != record_contract.get("value_metric") or units != _unit(record_contract.get("value_units")):
        raise OfficialReferenceIntakeError("evaluation and record contracts use different metric/units")
    aggregation = contract.get("aggregation")
    allowed = {
        "wear_depth": {"area_weighted_integral"},
        "erosion_rate": {"area_weighted_integral"},
        "erosion_area": {"explicit_sum"},
        "erosion_volume": {"explicit_sum"},
        "erosion_mass": {"explicit_sum"},
        "erosion_ratio": {"explicit_sum"},
        "dimensionless": {"explicit_sum"},
    }
    if aggregation not in allowed.get(str(metric), set()):
        raise OfficialReferenceIntakeError("evaluation aggregation is invalid for value_metric")
    if contract.get("relative_error_denominator") != "absolute_reference":
        raise OfficialReferenceIntakeError("relative_error_denominator must be absolute_reference")
    if contract.get("zero_reference_policy") != "block":
        raise OfficialReferenceIntakeError("zero_reference_policy must be block")
    hotspot = contract.get("hotspot")
    if not isinstance(hotspot, Mapping):
        raise OfficialReferenceIntakeError("evaluation_contract.hotspot is missing")
    if hotspot.get("mode") != "weighted_percentile" or hotspot.get("area_source") != record_contract.get("area_source"):
        raise OfficialReferenceIntakeError("hotspot definition differs from record area contract")
    try:
        percentile = float(hotspot.get("percentile"))
    except (TypeError, ValueError) as exc:
        raise OfficialReferenceIntakeError("hotspot percentile is invalid") from exc
    if not math.isfinite(percentile) or not 0.0 < percentile < 100.0:
        raise OfficialReferenceIntakeError("hotspot percentile must be in (0, 100)")
    if contract.get("top_k_tie_break") != "parent_face_id_ascending":
        raise OfficialReferenceIntakeError("top_k_tie_break must be parent_face_id_ascending")
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


def preflight_intake(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Validate all package inputs without creating normalized outputs."""
    try:
        spec, origin = _load_spec(value)
        raw_root = spec.get("source_root")
        if not isinstance(raw_root, str) or not raw_root.strip():
            raise OfficialReferenceIntakeError("intake spec missing source_root")
        root = Path(raw_root)
        if not root.is_absolute() and origin is not None:
            root = origin.parent / root
        root = root.resolve()
        if not root.is_dir():
            raise OfficialReferenceIntakeError(f"source_root does not exist: {root}")
        contest = _required_mapping(spec, "contest")
        for field in ("competition", "organizer", "dataset_id", "dataset_version"):
            if not isinstance(contest.get(field), str) or not contest[field].strip():
                raise OfficialReferenceIntakeError(f"contest metadata missing {field}")
        reference = _required_mapping(spec, "reference_source")
        reference_relative, reference_path = _safe_relative(root, reference.get("path"), "reference_source")
        normalized = normalize_records(reference_path, _required_mapping(reference, "adapter"))
        split = _required_mapping(spec, "independent_split")
        if split.get("provider") != "organizer_provided" or split.get("is_independent") is not True:
            raise OfficialReferenceIntakeError("independent_split must be organizer-provided and independent")
        split_relative, split_path = _safe_relative(root, split.get("path"), "independent_split")
        split_ids = _load_split(split_path, split)
        record_ids = [row["parent_face_id"] for row in normalized["records"]]
        if split_ids != record_ids:
            raise OfficialReferenceIntakeError("organizer split does not exactly match normalized reference IDs")
        required_paths = {
            "official_description": spec.get("official_description"),
            "official_evaluation_definition": spec.get("official_evaluation_definition"),
            "native_artifact_set_manifest": spec.get("native_artifact_set_manifest"),
            "topology_v3": spec.get("topology_v3"),
        }
        checked_paths = {
            role: _safe_relative(root, raw, role)
            for role, raw in required_paths.items()
        }
        native = safe_json_load(checked_paths["native_artifact_set_manifest"][1])
        if not isinstance(native, Mapping) or not _hash(native.get("artifact_set_id")):
            raise OfficialReferenceIntakeError("native artifact-set manifest is invalid")
        native_unsigned = dict(native)
        native_id = str(native_unsigned.pop("artifact_set_id")).lower()
        if canonical_sha256(native_unsigned) != native_id:
            raise OfficialReferenceIntakeError("native artifact_set_id/content mismatch")
        topology = load_topology_artifact(checked_paths["topology_v3"][1])
        if topology.get("artifact_version") != TOPOLOGY_VERSION:
            raise OfficialReferenceIntakeError("topology artifact is not v3")
        if topology.get("mesh_signature_algorithm") != TOPOLOGY_SIGNATURE_ALGORITHM:
            raise OfficialReferenceIntakeError("topology signature algorithm is invalid")
        if native.get("topology", {}).get("file_sha256") != sha256_file(checked_paths["topology_v3"][1]):
            raise OfficialReferenceIntakeError("native descriptor and topology file hash differ")
        if native.get("topology", {}).get("mesh_signature") != topology.get("mesh_signature"):
            raise OfficialReferenceIntakeError("native descriptor and topology mesh signature differ")
        record_contract = _required_mapping(spec, "record_contract")
        adapter = reference["adapter"]
        if record_contract.get("parent_face_id_field") != "parent_face_id":
            raise OfficialReferenceIntakeError("record_contract must use parent_face_id")
        if record_contract.get("value_field") != adapter.get("canonical_value_field", adapter.get("value_field")):
            raise OfficialReferenceIntakeError("record contract and adapter value fields differ")
        if record_contract.get("value_metric") != adapter.get("value_metric"):
            raise OfficialReferenceIntakeError("record contract and adapter metrics differ")
        if _unit(record_contract.get("value_units")) != _unit(adapter.get("canonical_units")):
            raise OfficialReferenceIntakeError("record contract and adapter units differ")
        if record_contract.get("coverage") not in {"full_parent_faces", "subset"}:
            raise OfficialReferenceIntakeError("record_contract coverage is invalid")
        if record_contract.get("coverage") == "full_parent_faces" and record_ids != list(range(topology["n_parent_faces"])):
            raise OfficialReferenceIntakeError("full_parent_faces reference does not cover topology")
        if any(identifier >= topology["n_parent_faces"] for identifier in record_ids):
            raise OfficialReferenceIntakeError("reference parent_face_id is outside topology")
        evaluation = _evaluation_contract(
            spec.get("evaluation_contract"),
            record_contract,
            sha256_file(checked_paths["official_evaluation_definition"][1]),
        )
        thresholds = _required_mapping(spec, "thresholds")
        required_thresholds = {
            "total_quantity_relative_error",
            "maximum_value_relative_error",
            "hotspot_area_relative_error",
            "hotspot_iou_error",
        }
        if not required_thresholds <= set(thresholds):
            raise OfficialReferenceIntakeError("thresholds are incomplete")
        for name in required_thresholds:
            try:
                threshold = float(thresholds[name])
            except (TypeError, ValueError) as exc:
                raise OfficialReferenceIntakeError(f"threshold {name} is invalid") from exc
            if not math.isfinite(threshold) or not 0.0 < threshold <= 0.05:
                raise OfficialReferenceIntakeError(f"threshold {name} must be in (0, 0.05]")
        result = {
            "version": VERSION,
            "status": "passed",
            "blocked": False,
            "source_root": str(root),
            "contest": contest,
            "reference_source": {
                "path": reference_relative,
                "sha256": normalized["source"]["sha256"],
                "n_records": len(record_ids),
                "records_sha256": normalized["records_sha256"],
            },
            "independent_split": {
                "path": split_relative,
                "sha256": sha256_file(split_path),
                "n_records": len(split_ids),
            },
            "artifact_set_id": native_id,
            "topology": {
                "mesh_signature": topology["mesh_signature"],
                "n_parent_faces": topology["n_parent_faces"],
                "n_collision_triangles": len(topology["collision_triangles_mm"]),
            },
            "evaluation_contract": evaluation,
            "required_outputs": ["official-reference-records.json", "official-reference-manifest.json"],
        }
        result["preflight_sha256"] = canonical_sha256(result)
        return result
    except (OfficialReferenceIntakeError, OSError, TypeError, ValueError, KeyError, ValidationError) as exc:
        result = {
            "version": VERSION,
            "status": "blocked",
            "blocked": True,
            "reason": str(exc) or type(exc).__name__,
            "official_reference_gate": False,
            "provenance_note": "official reference package incomplete; no accuracy claim",
        }
        result["preflight_sha256"] = canonical_sha256(result)
        return result


def build_intake(
    value: Mapping[str, Any] | str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create immutable normalized records and a strict reference manifest."""
    preflight = preflight_intake(value)
    if preflight["status"] != "passed":
        return preflight
    spec, origin = _load_spec(value)
    root = Path(preflight["source_root"])
    output = Path(output_dir).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise OfficialReferenceIntakeError("output_dir must be inside source_root") from exc
    records_path = output / "official-reference-records.json"
    manifest_path = output / "official-reference-manifest.json"
    if records_path.exists() or manifest_path.exists():
        raise FileExistsError("immutable intake output already exists")
    reference = _required_mapping(spec, "reference_source")
    _, reference_path = _safe_relative(root, reference["path"], "reference_source")
    normalized = normalize_records(reference_path, reference["adapter"])
    safe_json_dump(normalized, records_path)
    split = _required_mapping(spec, "independent_split")
    split_relative, split_path = _safe_relative(root, split["path"], "independent_split")
    split_ids = _load_split(split_path, split)
    native_relative, native_path = _safe_relative(root, spec["native_artifact_set_manifest"], "native_artifact_set_manifest")
    topology_relative, topology_path = _safe_relative(root, spec["topology_v3"], "topology_v3")
    native = safe_json_load(native_path)
    topology = load_topology_artifact(topology_path)
    records_relative = records_path.relative_to(root).as_posix()
    description = _artifact(root, "official_description", spec["official_description"])
    evaluation_definition = _artifact(root, "official_evaluation_definition", spec["official_evaluation_definition"])
    raw_reference = _artifact(root, "official_reference_raw", reference["path"])
    split_artifact = _artifact(root, "official_independent_split", split["path"])
    normalized_artifact = {
        "role": "official_reference_observations",
        "path": records_relative,
        "sha256": sha256_file(records_path),
        "bytes": int(records_path.stat().st_size),
    }
    native_artifact = _artifact(root, "native_artifact_set_manifest", native_relative)
    topology_artifact = _artifact(root, "topology_v3", topology_relative)
    adapter = reference["adapter"]
    record_contract = dict(spec["record_contract"])
    record_contract.update(
        {
            "prediction_field": record_contract["value_field"],
            "observation_field": record_contract["value_field"],
            "value_units": _unit(record_contract["value_units"]),
        }
    )
    evaluation = _evaluation_contract(
        spec["evaluation_contract"],
        record_contract,
        evaluation_definition["sha256"],
    )
    thresholds = dict(spec["thresholds"])
    thresholds["hotspot_percentile"] = evaluation["hotspot"]["percentile"]
    manifest = {
        "version": MANIFEST_VERSION,
        "manifest_kind": MANIFEST_KIND,
        "status": "synthetic",
        "source_type": "contest_provided_simulation",
        "reference_role": "official_reference_simulation",
        "pipeline_validation": False,
        "official_reference": True,
        "contest": dict(spec["contest"]),
        "source_root": str(root),
        "artifacts": [
            description,
            evaluation_definition,
            raw_reference,
            split_artifact,
            normalized_artifact,
            native_artifact,
            topology_artifact,
        ],
        "artifact_set_id": native["artifact_set_id"],
        "native_artifact": {
            "path": native_relative,
            "sha256": native_artifact["sha256"],
            "artifact_set_id": native["artifact_set_id"],
        },
        "topology": {
            "path": topology_relative,
            "sha256": topology_artifact["sha256"],
            "artifact_version": topology["artifact_version"],
            "mesh_signature_algorithm": topology["mesh_signature_algorithm"],
            "mesh_signature": topology["mesh_signature"],
            "source_vtp_sha256": topology["source_vtp_sha256"],
            "patch_name": topology["patch_name"],
            "time": topology["time_name"],
            "n_parent_faces": topology["n_parent_faces"],
            "n_collision_triangles": len(topology["collision_triangles_mm"]),
            "surface_area_m2": topology["surface_area_m2"],
        },
        "reference_records": {
            "path": records_relative,
            "sha256": normalized_artifact["sha256"],
        },
        "solver_semantics": dict(native["cfd"]["solver_semantics"]),
        "reference_solver_semantics": dict(spec["reference_solver_semantics"]),
        "units": dict(native["units"]),
        "independent_split": {
            "provider": "organizer_provided",
            "is_independent": True,
            "split_id": split["split_id"],
            "record_ids": split_ids,
            "split_artifact_sha256": split_artifact["sha256"],
            "reference_source_sha256": normalized_artifact["sha256"],
        },
        "record_contract": record_contract,
        "evaluation_contract": evaluation,
        "thresholds": thresholds,
        "record_provenance": {
            "raw_reference_sha256": raw_reference["sha256"],
            "normalized_reference_sha256": normalized_artifact["sha256"],
            "normalized_records_sha256": normalized["records_sha256"],
            "adapter": dict(adapter),
            "inferred_fields": False,
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    safe_json_dump(manifest, manifest_path)
    return {
        "version": VERSION,
        "status": "passed",
        "blocked": False,
        "records": {
            "path": str(records_path),
            "sha256": sha256_file(records_path),
            "records_sha256": normalized["records_sha256"],
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "manifest_canonical_sha256": manifest["manifest_sha256"],
        "trusted_manifest_sha256": None,
        "official_reference_gate": False,
        "provenance_note": "intake complete; the manifest digest is not trusted until pinned through an independent organizer channel",
    }


def _selftest() -> None:
    assert _convert(1000.0, "um", "mm") == 1.0
    assert math.isclose(_convert(1.0, "m3", "mm3"), 1.0e9)
    assert preflight_intake({"version": SPEC_VERSION})["status"] == "blocked"


def selftest() -> bool:
    _selftest()
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Intake an organizer reference simulation package")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output", type=Path, help="optional immutable preflight report")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        selftest()
        print("step30_official_reference_intake selftest: PASS")
        return 0
    if args.spec is None:
        parser.error("--spec is required")
    if args.preflight:
        result = preflight_intake(args.spec)
        if args.output:
            safe_json_dump(result, args.output)
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required unless --preflight is used")
        result = build_intake(args.spec, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
