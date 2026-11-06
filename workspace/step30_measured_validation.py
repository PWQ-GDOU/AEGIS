"""Formal, fail-closed measured validation for Step 30.

The module accepts prediction/observation records plus auditable manifests.  It
never promotes a command-line flag to evidence: measured status, source files,
independent split membership, quality gates, and pre-registered metrics all
come from the supplied documents.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from step30_validation_common import ValidationError, canonical_sha256, jsonable, safe_json_dump, safe_json_load, sha256_file

VERSION = "step30-measured-validation-v2"
SUPPORTED_METRICS = ("mass_loss_mg", "wear_volume_mm3", "wear_depth_mm", "dhdt_mm_h")
SPATIAL_METRICS = frozenset({"wear_depth_mm", "dhdt_mm_h"})
ACCEPTANCE_ERROR_LIMIT = 0.05
QUALITY_GATES = ("registration", "geometry", "measurement")
TARGET_UNITS = {
    "mass_loss_mg": "mg",
    "wear_volume_mm3": "mm3",
    "wear_depth_mm": "mm",
    "dhdt_mm_h": "mm/h",
}
DIMENSIONS = {"mass_loss_mg": "mass", "wear_volume_mm3": "volume", "wear_depth_mm": "length", "dhdt_mm_h": "rate"}
UNIT_INFO = {
    "ug": ("mass", 1e-3), "mcg": ("mass", 1e-3), "mg": ("mass", 1.0), "g": ("mass", 1e3), "kg": ("mass", 1e6),
    "mm3": ("volume", 1.0), "cm3": ("volume", 1e3), "m3": ("volume", 1e9),
    "um": ("length", 1e-3), "micron": ("length", 1e-3), "micrometer": ("length", 1e-3), "mm": ("length", 1.0), "cm": ("length", 10.0), "m": ("length", 1e3),
    "mm/h": ("rate", 1.0), "mm/hr": ("rate", 1.0), "mm/s": ("rate", 3600.0), "m/s": ("rate", 3.6e6),
}
INDEPENDENT_NAMES = {"independent", "independent_test", "test", "holdout", "independent-test"}
DEVELOPMENT_NAMES = {"train", "training", "calibration", "validation", "dev", "development"}
MEASURED_ORIGINS = {
    "laboratory_measurement", "field_measurement", "physical_measurement",
    "inspection_measurement", "metrology_measurement",
}
CORE_KEYS = ("mae", "rmse", "p50", "p95", "max", "bias", "total_relative_error", "max_relative_error")
HASH_BINDING_ALIASES = {
    "source": ("source_manifest_sha256", "source_sha256"),
    "protocol": ("protocol_sha256", "split_manifest_sha256"),
    "transform": ("transform_sha256", "transform_hash", "registration_sha256"),
    "topology": ("topology_sha256", "topology_hash", "mesh_signature"),
}


class MeasuredValidationError(ValidationError):
    """Raised internally when measured evidence is incomplete or inconsistent."""


def _hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value)


def _path(value: Any) -> Path | None:
    if isinstance(value, Path):
        return value
    return Path(value) if isinstance(value, str) and value else None


def _load(value: Any) -> tuple[Any, Path | None]:
    path = _path(value)
    if path is None:
        return value, None
    if not path.is_file():
        raise MeasuredValidationError(f"input file does not exist: {path}")
    if path.suffix.lower() == ".json":
        try:
            return safe_json_load(path), path.resolve()
        except Exception as exc:
            raise MeasuredValidationError(f"invalid JSON input: {path}") from exc
    if path.suffix.lower() == ".npz":
        try:
            from step30_measurement_io import load_measurement_record
            return load_measurement_record(path), path.resolve()
        except Exception as exc:
            raise MeasuredValidationError(f"invalid measurement record: {path}") from exc
    raise MeasuredValidationError(f"unsupported input extension: {path.suffix}")


def _records(document: Any, label: str) -> tuple[str | None, list[dict[str, Any]]]:
    if isinstance(document, Mapping):
        top_status = document.get("status") if isinstance(document.get("status"), str) else None
        raw = document.get("records", document.get(label, document.get("data")))
        if raw is None and any(k in document for k in ("record_id", "id", "metric")):
            raw = [document]
    elif isinstance(document, Sequence) and not isinstance(document, (str, bytes, bytearray)):
        top_status, raw = None, document
    else:
        raise MeasuredValidationError(f"{label} records must be an object or list")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)) or not raw:
        raise MeasuredValidationError(f"{label} records are empty or missing")
    result = [dict(item) for item in raw if isinstance(item, Mapping)]
    if len(result) != len(raw):
        raise MeasuredValidationError(f"{label} records must contain objects")
    return top_status, result


def _record_id(record: Mapping[str, Any]) -> str | None:
    for key in ("record_id", "id", "sample_id", "run_id", "trial_id"):
        if record.get(key) is not None and str(record[key]).strip():
            return str(record[key])
    return None


def _hashes(value: Any) -> set[str]:
    if _hash(value):
        return {str(value).lower()}
    if isinstance(value, Mapping):
        result: set[str] = set()
        for item in value.values():
            result.update(_hashes(item))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result: set[str] = set()
        for item in value:
            result.update(_hashes(item))
        return result
    return set()


def _record_hashes(record: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for key in ("source_sha256", "source_hash", "source_hashes"):
        result.update(_hashes(record.get(key)))
    for container_name in ("source", "provenance"):
        container = record.get(container_name)
        if _hash(container):
            result.add(str(container).lower())
        elif isinstance(container, Mapping):
            for key in ("sha256", "source_sha256", "source_hash", "source_hashes"):
                result.update(_hashes(container.get(key)))
    return result


def _canonical_checked(document: Mapping[str, Any], label: str) -> None:
    stored = document.get("manifest_sha256")
    without = dict(document)
    without.pop("manifest_sha256", None)
    if not _hash(stored) or canonical_sha256(without) != str(stored).lower():
        raise MeasuredValidationError(f"{label} canonical hash is missing or invalid")


def _source_manifest(document: Any, origin: Path | None, source_root: str | Path | None) -> tuple[dict[str, Any], dict[str, str], set[str]]:
    if not isinstance(document, Mapping) or document.get("status") != "measured":
        raise MeasuredValidationError("source manifest status must be measured")
    _canonical_checked(document, "source manifest")
    if not isinstance(document.get("version"), str) or not document["version"]:
        raise MeasuredValidationError("source manifest version metadata is missing")
    provenance = document.get("provenance")
    if not isinstance(provenance, Mapping):
        raise MeasuredValidationError("measured source manifest provenance is missing")
    origin_name = str(provenance.get("origin", "")).strip().lower()
    if origin_name not in MEASURED_ORIGINS:
        raise MeasuredValidationError("measured source manifest origin is not a physical measurement")
    for field in ("acquisition_method", "acquired_at", "organization"):
        if not str(provenance.get(field, "")).strip():
            raise MeasuredValidationError(f"measured source manifest provenance missing {field}")
    raw = document.get("entries", document.get("sources"))
    if isinstance(raw, Mapping):
        entries = {str(k): v for k, v in raw.items()}
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        entries = {str(i): v for i, v in enumerate(raw)}
    else:
        raise MeasuredValidationError("source manifest entries are missing")
    root = Path(source_root).resolve() if source_root is not None else (origin.parent if origin else Path.cwd()).resolve()
    by_record: dict[str, str] = {}
    all_hashes: set[str] = set()
    for key, value in entries.items():
        if not isinstance(value, Mapping):
            raise MeasuredValidationError(f"source entry {key} is not an object")
        entry = dict(value)
        if entry.get("status", "measured") != "measured":
            raise MeasuredValidationError(f"source entry {key} is not measured")
        path_value = entry.get("path")
        digest = entry.get("sha256")
        byte_count = entry.get("bytes")
        metadata = entry.get("metadata")
        if metadata is None:
            # ``step30_measurement_io.write_measurement_manifest`` keeps
            # kind/units/sample metadata at entry level.  Normalize that
            # established shape without weakening the completeness contract.
            metadata = {
                field: entry[field]
                for field in ("record_id", "specimen_id", "campaign_id", "group_id", "kind", "units", "n_samples", "schema")
                if field in entry
            }
        if not isinstance(path_value, str) or not path_value or Path(path_value).is_absolute():
            if entry.get("external") is not True:
                raise MeasuredValidationError(f"source entry {key} has unsafe path")
        if not _hash(digest):
            raise MeasuredValidationError(f"source entry {key} has no complete sha256")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise MeasuredValidationError(f"source entry {key} has no complete byte metadata")
        if not isinstance(metadata, Mapping) or not metadata:
            raise MeasuredValidationError(f"source entry {key} has incomplete metadata")
        metadata = dict(metadata)
        for field in ("record_id", "specimen_id", "campaign_id", "group_id", "kind", "units", "schema"):
            if field in entry and field not in metadata:
                metadata[field] = entry[field]
        if not any(field in metadata for field in ("record_id", "specimen_id", "campaign_id", "kind", "units", "schema")):
            raise MeasuredValidationError(f"source entry {key} has incomplete metadata")
        if entry.get("external") is True:
            absolute = entry.get("absolute_path")
            target = Path(absolute).resolve() if isinstance(absolute, str) and Path(absolute).is_absolute() else None
        else:
            target = (root / path_value).resolve() if isinstance(path_value, str) else None
            if target is not None:
                try:
                    target.relative_to(root)
                except ValueError as exc:
                    raise MeasuredValidationError(f"source path escapes root: {path_value}") from exc
        if target is None or not target.is_file():
            raise MeasuredValidationError(f"source file does not exist: {path_value}")
        if int(target.stat().st_size) != byte_count or sha256_file(target) != str(digest).lower():
            raise MeasuredValidationError(f"source hash/metadata mismatch: {path_value}")
        all_hashes.add(str(digest).lower())
        identifier = metadata.get("record_id", metadata.get("id"))
        if identifier is not None:
            if str(identifier) in by_record:
                raise MeasuredValidationError(f"duplicate source record_id: {identifier}")
            by_record[str(identifier)] = str(digest).lower()
    if not all_hashes:
        raise MeasuredValidationError("source manifest has no hashed files")
    return dict(document), by_record, all_hashes


def _split_rows(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = document.get("records", document.get("entries"))
    if isinstance(raw, Mapping):
        return [dict(v, record_id=v.get("record_id", k)) for k, v in raw.items() if isinstance(v, Mapping)]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return [dict(v) for v in raw if isinstance(v, Mapping)]
    result: list[dict[str, Any]] = []
    split_mapping = document.get("splits")
    if isinstance(split_mapping, Mapping):
        for split_name, values in split_mapping.items():
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
                continue
            for value in values:
                item = dict(value) if isinstance(value, Mapping) else {"record_id": value}
                item.setdefault("split", split_name)
                result.append(item)
    for split_name in ("independent_test", "independent", "test", "train", "training", "validation", "dev"):
        values = document.get(split_name)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
            for value in values:
                item = dict(value) if isinstance(value, Mapping) else {"record_id": value}
                item.setdefault("split", split_name)
                result.append(item)
    return result


def _split_manifest(document: Any, ids: set[str], record_hashes: Mapping[str, set[str]], source_by_record: Mapping[str, str], source_hashes: set[str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(document, Mapping):
        raise MeasuredValidationError("split manifest must be an object")
    if document.get("status") != "measured":
        raise MeasuredValidationError("split manifest status must be measured")
    _canonical_checked(document, "split manifest")
    rows = _split_rows(document)
    if not rows:
        raise MeasuredValidationError("split manifest is empty")
    seen: set[str] = set()
    independent: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    for row in rows:
        identifier = row.get("record_id", row.get("id"))
        if identifier is None or not str(identifier).strip():
            raise MeasuredValidationError("split entries need record_id")
        identifier = str(identifier)
        if identifier in seen:
            raise MeasuredValidationError(f"duplicate split record_id: {identifier}")
        seen.add(identifier)
        if identifier not in ids:
            raise MeasuredValidationError(f"split references unknown record: {identifier}")
        split_name = str(row.get("split", row.get("partition", row.get("set", "")))).strip().lower()
        if row.get("is_independent") is True:
            split_name = "independent_test"
        if not split_name:
            raise MeasuredValidationError(f"split has no partition: {identifier}")
        if split_name not in INDEPENDENT_NAMES | DEVELOPMENT_NAMES:
            raise MeasuredValidationError(f"unsupported split partition: {split_name}")
        group = row.get("group_id", row.get("group", row.get("split_group")))
        if group is None or not str(group).strip():
            raise MeasuredValidationError(f"split has no group_id: {identifier}")
        declared_hashes = _hashes(row.get("source_sha256", row.get("source_hash", row.get("source_hashes"))))
        hashes = set(record_hashes.get(identifier, set()))
        if not hashes and identifier in source_by_record:
            hashes.add(source_by_record[identifier])
        if not hashes:
            raise MeasuredValidationError(f"split has no source hash: {identifier}")
        if declared_hashes and not declared_hashes <= hashes:
            raise MeasuredValidationError(f"split source hash disagrees with record: {identifier}")
        if source_hashes is not None and not hashes <= source_hashes:
            raise MeasuredValidationError(f"split source hash is absent from source manifest: {identifier}")
        normalized = {"record_id": identifier, "split": split_name, "group_id": str(group), "hashes": hashes}
        (independent if split_name in INDEPENDENT_NAMES else other).append(normalized)
    if seen != ids:
        raise MeasuredValidationError("all paired records must be listed in split manifest")
    if not independent:
        raise MeasuredValidationError("independent test split is empty")
    if not other:
        raise MeasuredValidationError("split manifest has no non-independent partition")
    independent_groups = {r["group_id"] for r in independent}
    other_groups = {r["group_id"] for r in other}
    independent_hashes = set().union(*(r["hashes"] for r in independent))
    other_hashes = set().union(*(r["hashes"] for r in other)) if other else set()
    if independent_groups & other_groups:
        raise MeasuredValidationError("independent and non-independent groups overlap")
    if independent_hashes & other_hashes:
        raise MeasuredValidationError("independent and non-independent source hashes overlap")
    if len(independent_hashes) != sum(len(r["hashes"]) for r in independent):
        raise MeasuredValidationError("independent records reuse a source hash")
    core = document.get("pre_registered_core_metrics", document.get("core_metrics"))
    metadata = document.get("metadata")
    if core is None and isinstance(metadata, Mapping):
        core = metadata.get("pre_registered_core_metrics", metadata.get("core_metrics"))
    if not isinstance(core, Sequence) or isinstance(core, (str, bytes, bytearray)) or not core:
        raise MeasuredValidationError("split manifest needs explicit pre-registered core metrics and thresholds")
    core_metrics: list[str] = []
    core_thresholds: dict[str, float] = {}
    for item in core:
        if not isinstance(item, Mapping):
            raise MeasuredValidationError("pre-registered core metric entries need explicit thresholds")
        name = str(item.get("metric", item.get("name", "")))
        if name not in SUPPORTED_METRICS or name in core_thresholds:
            raise MeasuredValidationError(f"unsupported or duplicate pre-registered core metric: {name}")
        raw_limit = item.get("total_relative_error_max", item.get("maximum_relative_error", item.get("threshold")))
        try:
            limit = float(raw_limit)
        except (TypeError, ValueError) as exc:
            raise MeasuredValidationError(f"pre-registered threshold is invalid: {name}") from exc
        if not math.isfinite(limit) or limit <= 0.0 or limit > ACCEPTANCE_ERROR_LIMIT:
            raise MeasuredValidationError(f"pre-registered threshold must be in (0, {ACCEPTANCE_ERROR_LIMIT}]: {name}")
        core_metrics.append(name)
        core_thresholds[name] = limit
    hotspot = document.get("pre_registered_hotspot", document.get("hotspot_definition"))
    if hotspot is None and isinstance(metadata, Mapping):
        hotspot = metadata.get("pre_registered_hotspot", metadata.get("hotspot_definition"))
    if not isinstance(hotspot, Mapping):
        raise MeasuredValidationError("split manifest needs a pre-registered hotspot definition")
    try:
        hotspot_metric = str(hotspot.get("metric"))
        percentile = float(hotspot.get("percentile"))
        area_limit = float(hotspot.get("area_relative_error_max"))
        iou_limit = float(hotspot.get("iou_error_max"))
    except (TypeError, ValueError) as exc:
        raise MeasuredValidationError("pre-registered hotspot thresholds are invalid") from exc
    if hotspot_metric != "wear_depth_mm" or not 0.0 < percentile < 100.0:
        raise MeasuredValidationError("pre-registered hotspot definition must target wear_depth_mm at a valid percentile")
    if any(not math.isfinite(limit) or limit <= 0.0 or limit > ACCEPTANCE_ERROR_LIMIT for limit in (area_limit, iou_limit)):
        raise MeasuredValidationError(f"hotspot thresholds must be in (0, {ACCEPTANCE_ERROR_LIMIT}]")
    hotspot_definition = {
        "metric": hotspot_metric,
        "percentile": percentile,
        "area_relative_error_max": area_limit,
        "iou_error_max": iou_limit,
    }
    return dict(document), {
        "independent": independent,
        "other": other,
        "core_metrics": core_metrics,
        "core_thresholds": core_thresholds,
        "hotspot": hotspot_definition,
    }


def _canonical_report_checked(document: Mapping[str, Any], label: str) -> str:
    stored = document.get("report_sha256")
    unsigned = dict(document)
    unsigned.pop("report_sha256", None)
    if not _hash(stored) or canonical_sha256(unsigned) != str(stored).lower():
        raise MeasuredValidationError(f"{label} report canonical hash is missing or invalid")
    return str(stored).lower()


def _binding_value(document: Mapping[str, Any], kind: str) -> str:
    containers = [document]
    for key in ("bindings", "identity", "provenance"):
        value = document.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
    values: set[str] = set()
    for container in containers:
        for alias in HASH_BINDING_ALIASES[kind]:
            value = container.get(alias)
            if value is not None:
                if not _hash(value):
                    raise MeasuredValidationError(f"{kind} binding is not a SHA-256 digest")
                values.add(str(value).lower())
    if len(values) != 1:
        raise MeasuredValidationError(f"{kind} binding is missing or inconsistent")
    return next(iter(values))


def _record_identity(records: Sequence[Mapping[str, Any]], kind: str) -> str:
    values = {_binding_value(record, kind) for record in records}
    if len(values) != 1:
        raise MeasuredValidationError(f"record {kind} bindings differ")
    return next(iter(values))


def _numeric_quality_value(value: Any, name: str) -> tuple[float, str | None]:
    raw = value
    units = None
    if isinstance(value, Mapping):
        raw = value.get("value", value.get("actual"))
        units_value = value.get("units", value.get("unit"))
        if units_value is not None:
            if not isinstance(units_value, str) or not units_value.strip():
                raise MeasuredValidationError(f"quality units are invalid: {name}")
            units = units_value.strip()
    if isinstance(raw, (bool, np.bool_)):
        raise MeasuredValidationError(f"quality actual is not numeric: {name}")
    try:
        number = float(raw)
    except (TypeError, ValueError) as exc:
        raise MeasuredValidationError(f"quality actual is not numeric: {name}") from exc
    if not math.isfinite(number):
        raise MeasuredValidationError(f"quality actual is not finite: {name}")
    return number, units


def _quality_comparison(spec: Any, actual: float, actual_units: str | None, name: str) -> tuple[bool, dict[str, Any]]:
    if not isinstance(spec, Mapping):
        raise MeasuredValidationError(f"quality threshold needs an explicit operator: {name}")
    units_value = spec.get("units", spec.get("unit"))
    threshold_units = None if units_value is None else str(units_value).strip()
    if bool(threshold_units) != bool(actual_units) or (threshold_units and threshold_units != actual_units):
        raise MeasuredValidationError(f"quality threshold/actual units differ: {name}")
    if "maximum" in spec:
        operator, raw_limit = "<=", spec["maximum"]
    elif "minimum" in spec:
        operator, raw_limit = ">=", spec["minimum"]
    else:
        operator = str(spec.get("operator", ""))
        raw_limit = spec.get("value")
        if operator not in {"<=", ">="}:
            raise MeasuredValidationError(f"quality threshold operator is invalid: {name}")
    if isinstance(raw_limit, (bool, np.bool_)):
        raise MeasuredValidationError(f"quality threshold is not numeric: {name}")
    try:
        limit = float(raw_limit)
    except (TypeError, ValueError) as exc:
        raise MeasuredValidationError(f"quality threshold is not numeric: {name}") from exc
    if not math.isfinite(limit):
        raise MeasuredValidationError(f"quality threshold is not finite: {name}")
    passed = actual <= limit if operator == "<=" else actual >= limit
    return passed, {"actual": actual, "operator": operator, "threshold": limit, "units": actual_units, "passed": passed}


def _quality(document: Any, expected_bindings: Mapping[str, str]) -> dict[str, Any]:
    if not isinstance(document, Mapping) or document.get("status") != "measured":
        raise MeasuredValidationError("quality report bundle is missing or not measured")
    stored = document.get("quality_sha256")
    unsigned = dict(document)
    unsigned.pop("quality_sha256", None)
    if not _hash(stored) or canonical_sha256(unsigned) != str(stored).lower():
        raise MeasuredValidationError("quality report bundle canonical hash is missing or invalid")
    gate_results: dict[str, Any] = {}
    report_hashes: dict[str, str] = {}
    for gate_name in QUALITY_GATES:
        gate = document.get(gate_name)
        if not isinstance(gate, Mapping) or gate.get("status") != "measured":
            raise MeasuredValidationError(f"{gate_name} quality report is missing or not measured")
        report_hashes[gate_name] = _canonical_report_checked(gate, gate_name)
        for kind, expected in expected_bindings.items():
            if _binding_value(gate, kind) != expected:
                raise MeasuredValidationError(f"{gate_name} quality {kind} binding mismatch")
        thresholds, actuals = gate.get("thresholds"), gate.get("actuals")
        if not isinstance(thresholds, Mapping) or not thresholds or not isinstance(actuals, Mapping):
            raise MeasuredValidationError(f"{gate_name} quality thresholds/actuals are missing")
        if set(thresholds) != set(actuals):
            raise MeasuredValidationError(f"{gate_name} quality threshold/actual keys differ")
        checks: dict[str, Any] = {}
        for name in sorted(thresholds):
            actual, units = _numeric_quality_value(actuals[name], f"{gate_name}.{name}")
            passed, detail = _quality_comparison(thresholds[name], actual, units, f"{gate_name}.{name}")
            checks[str(name)] = detail
            if not passed:
                raise MeasuredValidationError(f"{gate_name} quality threshold failed: {name}")
        gate_results[gate_name] = {"passed": True, "checks": checks, "report_sha256": report_hashes[gate_name]}
    return {
        "passed": True,
        "gates": gate_results,
        "report_hashes": report_hashes,
        "bindings": dict(expected_bindings),
        "quality_sha256": str(stored).lower(),
        "passed_bool_trusted": False,
    }


def _pair(document: Any, label: str) -> tuple[str | None, dict[str, dict[str, Any]]]:
    top, rows = _records(document, label)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = _record_id(row)
        if identifier is None:
            raise MeasuredValidationError(f"{label} record has no record_id")
        if identifier in result:
            raise MeasuredValidationError(f"duplicate {label} record_id: {identifier}")
        status = row.get("status", top)
        if not isinstance(status, str):
            raise MeasuredValidationError(f"{label} record has no status")
        hashes = _record_hashes(row)
        result[identifier] = dict(row, record_id=identifier, _status=status, _hashes=hashes)
    return top, result


def _metric_spec(record: Mapping[str, Any], metric: str, role: str) -> Any:
    metrics = record.get("metrics")
    if isinstance(metrics, Mapping) and metric in metrics:
        return metrics[metric]
    if metric in record:
        return record[metric]
    named = record.get("metric", record.get("field", record.get("target")))
    if named == metric:
        for key in (f"{role}_values", f"{role}_value", "values", "value", "data", "prediction", "predicted", "observation", "observed"):
            if key in record:
                return record[key]
    nested = record.get(role)
    if isinstance(nested, Mapping):
        return nested.get(metric, nested.get("values", nested.get("value")) if named == metric else None)
    return None


def _unit(record: Mapping[str, Any], metric: str, spec: Any) -> str:
    if isinstance(spec, Mapping):
        for key in ("unit", "units", "value_units"):
            if key in spec:
                return _normal_unit(spec[key])
    units = record.get("units")
    if isinstance(units, Mapping):
        for key in (metric, "value", "measurement"):
            if key in units:
                return _normal_unit(units[key])
    for key in (f"{metric}_unit", f"{metric}_units", "unit", "units", "value_units"):
        if key in record and isinstance(record[key], str):
            return _normal_unit(record[key])
    raise MeasuredValidationError(f"{metric} has no explicit units")


def _normal_unit(value: Any) -> str:
    if not isinstance(value, str):
        raise MeasuredValidationError("metric units must be strings")
    unit = value.strip().lower().replace("²", "2").replace("³", "3").replace(" ", "").replace("_", "")
    unit = {
        "microgram": "ug", "micrograms": "ug", "milligram": "mg", "milligrams": "mg", "gram": "g", "grams": "g",
        "millimeter": "mm", "millimetre": "mm", "millimeters": "mm", "millimetres": "mm", "micrometre": "um", "micrometers": "um", "micrometres": "um",
        "mm^3": "mm3", "cm^3": "cm3", "m^3": "m3", "mm/hour": "mm/h", "millimeter/hour": "mm/h", "millimetre/hour": "mm/h",
    }.get(unit, unit)
    if unit not in UNIT_INFO:
        raise MeasuredValidationError(f"unsupported metric unit: {value!r}")
    return unit


def _values(spec: Any, name: str) -> np.ndarray:
    raw = spec
    if isinstance(spec, Mapping):
        for key in ("values", "value", "data", "observations", "prediction", "predicted", "observed"):
            if key in spec:
                raw = spec[key]
                break
    try:
        array = np.asarray(raw, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise MeasuredValidationError(f"{name} is not numeric") from exc
    if array.size == 0 or not np.isfinite(array).all():
        raise MeasuredValidationError(f"{name} is empty or non-finite")
    return array


def _convert(spec: Any, source_unit: str, metric: str) -> np.ndarray:
    array = _values(spec, metric)
    dimension, factor = UNIT_INFO[source_unit]
    if dimension != DIMENSIONS[metric]:
        raise MeasuredValidationError(f"{metric} has incompatible unit {source_unit}")
    return array * factor / UNIT_INFO[TARGET_UNITS[metric]][1]


def _field(record: Mapping[str, Any], spec: Any, names: Sequence[str]) -> Any:
    for container in (spec, record):
        if not isinstance(container, Mapping):
            continue
        for name in names:
            if name in container:
                return container[name]
    return None


def _parent_face_ids(record: Mapping[str, Any], spec: Any, n: int, role: str) -> tuple[np.ndarray, int, str]:
    raw = _field(record, spec, ("parent_face_id", "parent_face_ids"))
    values = raw.get("values", raw.get("value")) if isinstance(raw, Mapping) else raw
    array = np.asarray(values) if values is not None else np.asarray([])
    if array.ndim != 1 or len(array) != n or not np.issubdtype(array.dtype, np.integer):
        raise MeasuredValidationError(f"{role} parent_face_id must be an integer array matching metric values")
    identifiers = array.astype(np.int64)
    if len(np.unique(identifiers)) != n:
        raise MeasuredValidationError(f"duplicate {role} parent_face_id")
    raw_count = _field(record, spec, ("n_parent_faces",))
    if isinstance(raw_count, bool) or not isinstance(raw_count, (int, np.integer)) or int(raw_count) <= 0:
        raise MeasuredValidationError(f"{role} n_parent_faces is missing or invalid")
    count = int(raw_count)
    if np.any(identifiers < 0) or np.any(identifiers >= count):
        raise MeasuredValidationError(f"{role} parent_face_id is out of range")
    if len(identifiers) != count or set(identifiers.tolist()) != set(range(count)):
        raise MeasuredValidationError(f"{role} parent-face cohort is incomplete")
    cohort = _field(record, spec, ("cohort_id", "parent_face_cohort_id", "cohort"))
    if not isinstance(cohort, str) or not cohort.strip():
        cohort = _binding_value(record, "topology")
    return identifiers, count, cohort.strip()


def _area_mm2(record: Mapping[str, Any], spec: Any, n: int, role: str) -> np.ndarray:
    area = _field(record, spec, ("area_mm2",))
    if area is None:
        raise MeasuredValidationError(f"{role} area_mm2 is missing")
    unit_declarations: list[Any] = []
    for container in (spec, record):
        if not isinstance(container, Mapping):
            continue
        for key in ("area_unit", "area_units"):
            if key in container:
                unit_declarations.append(container[key])
        units = container.get("units")
        if isinstance(units, Mapping):
            for key in ("area_mm2", "area"):
                if key in units:
                    unit_declarations.append(units[key])
    if isinstance(area, Mapping):
        unit_declarations.append(area.get("units", area.get("unit")))
        raw = area.get("values", area.get("value"))
    else:
        raw = area
    for units in unit_declarations:
        normalized = str(units).strip().lower().replace("²", "2").replace("^", "") if isinstance(units, str) else ""
        if normalized != "mm2":
            raise MeasuredValidationError(f"{role} area_mm2 units must be mm2")
    try:
        result = np.asarray(raw, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise MeasuredValidationError(f"{role} area_mm2 is not numeric") from exc
    if result.size != n or not np.isfinite(result).all() or np.any(result <= 0.0):
        raise MeasuredValidationError(f"{role} area_mm2 must match values and be strictly positive")
    return result


def _spatial_alignment(
    prediction: Mapping[str, Any], observation: Mapping[str, Any], pred_spec: Any, obs_spec: Any,
    predicted: np.ndarray, observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], str]:
    pred_ids, pred_count, pred_cohort = _parent_face_ids(prediction, pred_spec, len(predicted), "prediction")
    obs_ids, obs_count, obs_cohort = _parent_face_ids(observation, obs_spec, len(observed), "observation")
    if set(pred_ids.tolist()) != set(obs_ids.tolist()):
        raise MeasuredValidationError("prediction and observation parent_face_id sets differ")
    if pred_count != obs_count or pred_cohort != obs_cohort:
        raise MeasuredValidationError("prediction and observation parent-face cohorts differ")
    pred_area = _area_mm2(prediction, pred_spec, len(predicted), "prediction")
    obs_area = _area_mm2(observation, obs_spec, len(observed), "observation")
    pred_order, obs_order = np.argsort(pred_ids), np.argsort(obs_ids)
    aligned_ids = pred_ids[pred_order]
    if not np.array_equal(aligned_ids, obs_ids[obs_order]):
        raise MeasuredValidationError("prediction and observation parent_face_id alignment failed")
    if not np.array_equal(pred_area[pred_order], obs_area[obs_order]):
        raise MeasuredValidationError("prediction and observation area_mm2 differ by parent_face_id")
    return predicted[pred_order], observed[obs_order], pred_area[pred_order], aligned_ids.tolist(), pred_cohort


def _percentile(values: np.ndarray, weights: np.ndarray | None, p: float) -> float:
    if weights is None:
        return float(np.percentile(values, p))
    order = np.argsort(values)
    x, w = values[order], weights[order]
    return float(np.interp(p / 100.0 * float(w.sum()), np.cumsum(w), x))


def _relative(delta: float, reference: float, label: str) -> float:
    if reference == 0.0:
        raise MeasuredValidationError(f"zero reference is undefined for relative error: {label}")
    return float(abs(delta) / abs(reference))


def _calculate(metric: str, predicted: np.ndarray, observed: np.ndarray, weights: np.ndarray | None, hotspot_percentile: float) -> dict[str, Any]:
    if predicted.shape != observed.shape or predicted.size == 0:
        raise MeasuredValidationError(f"{metric} prediction/observation shapes differ")
    residual = predicted - observed
    absolute = np.abs(residual)
    effective = weights if metric in SPATIAL_METRICS else None
    if metric in SPATIAL_METRICS and weights is None:
        raise MeasuredValidationError(f"{metric} formal spatial validation requires explicit area_mm2")
    average = (lambda v: float(np.average(v, weights=effective))) if effective is not None else (lambda v: float(np.mean(v)))
    if metric in SPATIAL_METRICS:
        predicted_total, observed_total = average(predicted), average(observed)
        aggregation = "area_weighted_mean"
    else:
        predicted_total, observed_total = float(predicted.sum()), float(observed.sum())
        aggregation = "total"
    p_pred = _percentile(predicted, weights, hotspot_percentile)
    p_obs = _percentile(observed, weights, hotspot_percentile)
    predicted_hot, observed_hot = predicted >= p_pred, observed >= p_obs
    if weights is None:
        area_pred, area_obs = float(predicted_hot.sum()), float(observed_hot.sum())
        intersection, union = float(np.sum(predicted_hot & observed_hot)), float(np.sum(predicted_hot | observed_hot))
    else:
        area_pred, area_obs = float(weights[predicted_hot].sum()), float(weights[observed_hot].sum())
        intersection, union = float(weights[predicted_hot & observed_hot].sum()), float(weights[predicted_hot | observed_hot].sum())
    total_signed = predicted_total - observed_total
    max_signed = float(predicted.max() - observed.max())
    distribution = {
        "mae": average(absolute), "rmse": float(math.sqrt(average(residual ** 2))), "p50": _percentile(absolute, effective, 50),
        "p95": _percentile(absolute, effective, 95), "max": float(absolute.max()),
    }
    total_error = {"signed": float(total_signed), "absolute": float(abs(total_signed)), "relative": _relative(total_signed, observed_total, f"{metric} total")}
    observed_max = float(observed.max())
    max_error = {"signed": max_signed, "absolute": float(abs(max_signed)), "relative": _relative(max_signed, observed_max, f"{metric} maximum")}
    if np.any(observed == 0.0):
        raise MeasuredValidationError(f"zero reference is undefined for pointwise relative error: {metric}")
    max_point_relative = float(np.max(np.abs(residual) / np.abs(observed)))
    return {
        "metric": metric, "units": TARGET_UNITS[metric], "n": int(predicted.size), "aggregation": aggregation,
        "residual_distribution": distribution, **distribution, "bias": average(residual), "signed_bias": average(residual),
        "predicted_total": predicted_total, "observed_total": observed_total, "total_error": total_error,
        "total_quantity_error": dict(total_error), "total_relative_error": total_error["relative"],
        "maximum_error": max_error, "maximum_depth_error": max_error, "max_relative_error": max_error["relative"],
        "maximum_relative_error": max_error["relative"], "pointwise_max_relative_error": max_point_relative,
        "hotspot": {"definition": "independent weighted percentile per prediction/observation cohort", "percentile": hotspot_percentile, "area_units": "mm2" if weights is not None else None, "predicted_threshold": p_pred, "observed_threshold": p_obs, "predicted_area": area_pred, "observed_area": area_obs, "intersection_area": intersection, "union_area": union, "area": {"predicted": area_pred, "observed": area_obs}, "iou": float(intersection / union) if union else 1.0},
        "hotspot_area_relative_error": _relative(area_pred - area_obs, area_obs, f"{metric} hotspot area"),
        "hotspot_iou_error": float(1.0 - (intersection / union)) if union else 0.0,
    }


def _payload(pair: Mapping[str, Any], metric: str) -> dict[str, Any] | None:
    prediction, observation = pair["prediction"], pair["observation"]
    pred_spec, obs_spec = _metric_spec(prediction, metric, "prediction"), _metric_spec(observation, metric, "observation")
    if pred_spec is None or obs_spec is None:
        return None
    predicted = _convert(pred_spec, _unit(prediction, metric, pred_spec), metric)
    observed = _convert(obs_spec, _unit(observation, metric, obs_spec), metric)
    weights = None
    parent_face_ids: list[int] | None = None
    cohort: str | None = None
    if metric in SPATIAL_METRICS:
        predicted, observed, weights, parent_face_ids, cohort = _spatial_alignment(
            prediction, observation, pred_spec, obs_spec, predicted, observed,
        )
    elif predicted.shape != observed.shape:
        raise MeasuredValidationError(f"{metric} prediction/observation shapes differ")
    return {
        "record_id": pair["record_id"],
        "specimen_id": observation.get("specimen_id", prediction.get("specimen_id", "unknown")),
        "campaign_id": observation.get("campaign_id", prediction.get("campaign_id", "unknown")),
        "predicted": predicted,
        "observed": observed,
        "weights": weights,
        "parent_face_ids": parent_face_ids,
        "cohort_id": cohort,
    }


def _aggregate(items: Sequence[Mapping[str, Any]], metric: str, hotspot_percentile: float) -> dict[str, Any]:
    predicted = np.concatenate([item["predicted"] for item in items])
    observed = np.concatenate([item["observed"] for item in items])
    weights = np.concatenate([item["weights"] for item in items]) if items and all(item["weights"] is not None for item in items) else None
    result = _calculate(metric, predicted, observed, weights, hotspot_percentile)
    result["record_ids"], result["n_records"] = [str(item["record_id"]) for item in items], len(items)
    return result


def _bootstrap(items: Sequence[Mapping[str, Any]], metric: str, count: int, confidence: float, seed: int, hotspot_percentile: float) -> dict[str, Any]:
    result: dict[str, Any] = {"method": "record_percentile", "n_bootstrap": int(max(0, count)), "confidence": float(confidence), "metrics": {}}
    if count <= 0 or not items:
        return result
    rng = np.random.default_rng(seed)
    values = {key: [] for key in CORE_KEYS}
    for _ in range(count):
        selected = [items[int(i)] for i in rng.integers(0, len(items), size=len(items))]
        report = _aggregate(selected, metric, hotspot_percentile)
        for key in CORE_KEYS:
            if key in report:
                values[key].append(float(report[key]))
    alpha = (1.0 - confidence) / 2.0
    result["metrics"] = {key: [float(np.quantile(v, alpha)), float(np.quantile(v, 1 - alpha))] for key, v in values.items() if v}
    return result


def _grouped(items: Sequence[Mapping[str, Any]], metric: str, key: str, count: int, confidence: float, seed: int, hotspot_percentile: float) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for item in items:
        groups.setdefault(str(item.get(key, "unknown")), []).append(item)
    return {
        group: dict(
            _aggregate(values, metric, hotspot_percentile),
            bootstrap_ci=_bootstrap(values, metric, count, confidence, seed + i, hotspot_percentile),
        )
        for i, (group, values) in enumerate(sorted(groups.items()))
    }


def _blocked(reason: str, evidence_status: str | None = None) -> dict[str, Any]:
    result = {
        "version": VERSION,
        "status": "blocked",
        "evaluation_status": "blocked",
        "evidence_status": evidence_status,
        "blocked": True,
        "official_accuracy_gate": False,
        "official_reference_gate": False,
        "reasons": [reason],
        "provenance_note": "blocked evidence; not eligible for measured accuracy claims",
    }
    result["validation_sha256"] = canonical_sha256(result)
    return result


def validate_measured(prediction_records: Any, observation_records: Any, source_manifest: Any, split_manifest: Any, quality_json: Any = None, *, source_root: str | Path | None = None, bootstrap_samples: int = 1000, confidence: float = 0.95, seed: int = 30, **kwargs: Any) -> dict[str, Any]:
    """Validate records and return a report; malformed/missing data is blocked."""
    try:
        prediction_document, _ = _load(prediction_records)
        observation_document, _ = _load(observation_records)
        source_document, source_origin = _load(source_manifest)
        split_document, _ = _load(split_manifest)
        quality_document, _ = _load(quality_json) if quality_json is not None else (None, None)
        quality_sources = {
            "registration": kwargs.get("registration_quality_json", kwargs.get("registration_json")),
            "geometry": kwargs.get("geometry_quality_json", kwargs.get("geometry_json")),
            "measurement": kwargs.get("measurement_quality_json", kwargs.get("measurement_json")),
        }
        if any(value is not None for value in quality_sources.values()):
            merged_quality = dict(quality_document) if isinstance(quality_document, Mapping) else {"status": "measured"}
            for gate_name, gate_input in quality_sources.items():
                if gate_input is None:
                    continue
                gate_document, _ = _load(gate_input)
                if not isinstance(gate_document, Mapping):
                    raise MeasuredValidationError(f"{gate_name} quality JSON is not an object")
                # Accept either {passed: bool}, {status: ...}, or a report
                # already nested under its gate name.
                merged_quality[gate_name] = gate_document.get(gate_name, gate_document)
            quality_document = merged_quality
        pred_top, predictions = _pair(prediction_document, "predictions")
        obs_top, observations = _pair(observation_document, "observations")
        statuses = {pred_top, obs_top} | {r["_status"] for r in predictions.values()} | {r["_status"] for r in observations.values()}
        statuses.discard(None)
        if "synthetic" in statuses:
            return _blocked("synthetic_status", "synthetic")
        if "blocked" in statuses:
            return _blocked("blocked_input", "blocked")
        if not statuses or statuses != {"measured"}:
            return _blocked("legacy_status_mismatch", "legacy_status_mismatch")
        if set(predictions) != set(observations):
            raise MeasuredValidationError("prediction and observation record IDs differ")
        source_document, source_by_record, source_hashes = _source_manifest(source_document, source_origin, source_root)
        pairs: list[dict[str, Any]] = []
        pair_hashes: dict[str, set[str]] = {}
        for identifier in sorted(predictions):
            prediction, observation = predictions[identifier], observations[identifier]
            if prediction["_status"] == "blocked" or observation["_status"] == "blocked":
                return _blocked("blocked_input", "blocked")
            if prediction["_status"] != observation["_status"]:
                return _blocked("legacy_status_mismatch", "legacy_status_mismatch")
            prediction_hashes = set(prediction["_hashes"])
            observation_hashes = set(observation["_hashes"])
            if not prediction_hashes and identifier in source_by_record:
                prediction_hashes = {source_by_record[identifier]}
            if not observation_hashes and identifier in source_by_record:
                observation_hashes = {source_by_record[identifier]}
            if prediction_hashes != observation_hashes:
                raise MeasuredValidationError(f"prediction and observation source hashes differ: {identifier}")
            hashes = prediction_hashes
            if not hashes or not hashes <= source_hashes:
                raise MeasuredValidationError(f"record source hash is absent or not in source manifest: {identifier}")
            pair_hashes[identifier] = hashes
            pairs.append({"record_id": identifier, "prediction": prediction, "observation": observation})
        _, split = _split_manifest(split_document, set(predictions), pair_hashes, source_by_record, source_hashes)
        record_documents = [*predictions.values(), *observations.values()]
        expected_bindings = {
            "source": str(source_document["manifest_sha256"]).lower(),
            "protocol": str(split_document["manifest_sha256"]).lower(),
            "transform": _record_identity(record_documents, "transform"),
            "topology": _record_identity(record_documents, "topology"),
        }
        quality = _quality(quality_document, expected_bindings)
        independent_ids = {row["record_id"] for row in split["independent"]}
        independent_pairs = [pair for pair in pairs if pair["record_id"] in independent_ids]
        if not independent_pairs:
            raise MeasuredValidationError("independent test records are empty")
        metric_items: dict[str, list[dict[str, Any]]] = {}
        for pair in independent_pairs:
            names = [metric for metric in SUPPORTED_METRICS if _metric_spec(pair["prediction"], metric, "prediction") is not None or _metric_spec(pair["observation"], metric, "observation") is not None]
            if not names:
                raise MeasuredValidationError(f"record has no supported metric: {pair['record_id']}")
            for metric in names:
                item = _payload(pair, metric)
                if item is None:
                    raise MeasuredValidationError(f"metric is missing on one side: {metric}/{pair['record_id']}")
                metric_items.setdefault(metric, []).append(item)
        hotspot = split["hotspot"]
        metrics: dict[str, Any] = {}
        for index, (metric, items) in enumerate(sorted(metric_items.items())):
            percentile = hotspot["percentile"]
            report = _aggregate(items, metric, percentile)
            report["bootstrap_ci"] = _bootstrap(items, metric, bootstrap_samples, confidence, seed + index, percentile)
            report["by_specimen"] = _grouped(items, metric, "specimen_id", bootstrap_samples, confidence, seed + 100 + index, percentile)
            report["by_campaign"] = _grouped(items, metric, "campaign_id", bootstrap_samples, confidence, seed + 200 + index, percentile)
            metrics[metric] = report
        missing_core = sorted(set(split["core_metrics"]) - set(metrics))
        if missing_core:
            raise MeasuredValidationError(f"pre-registered core metrics are absent: {missing_core}")
        core_results = {metric: float(metrics[metric]["total_relative_error"]) for metric in split["core_metrics"]}
        core_passed = {metric: error <= split["core_thresholds"][metric] for metric, error in core_results.items()}
        spatial_metric = metrics.get(hotspot["metric"])
        if not isinstance(spatial_metric, Mapping):
            raise MeasuredValidationError(f"pre-registered hotspot metric is absent: {hotspot['metric']}")
        spatial_hotspot_gate = bool(
            float(spatial_metric.get("hotspot_area_relative_error", float("inf"))) <= hotspot["area_relative_error_max"]
            and float(spatial_metric.get("hotspot_iou_error", float("inf"))) <= hotspot["iou_error_max"]
        )
        numeric_passed = bool(core_passed) and all(core_passed.values()) and spatial_hotspot_gate
        evaluation_status = "passed" if numeric_passed else "failed"
        failed_reasons = [] if numeric_passed else [
            *[f"core_metric_threshold:{metric}" for metric, passed in core_passed.items() if not passed],
            *([] if spatial_hotspot_gate else ["pre_registered_hotspot_threshold"]),
        ]
        result = {
            "version": VERSION,
            "status": evaluation_status,
            "evaluation_status": evaluation_status,
            "evidence_status": "measured",
            "blocked": False,
            "reasons": failed_reasons,
            "n_records": len(independent_pairs),
            "independent_validation": {"is_independent": True, "n": len(independent_pairs), "record_ids": sorted(independent_ids), "source_hashes": sorted(set().union(*(r["hashes"] for r in split["independent"]))), "groups": sorted({r["group_id"] for r in split["independent"]}), "group_source_record_disjoint": True},
            "quality": quality,
            "metrics": metrics,
            "pre_registered_core_metrics": split["core_metrics"],
            "pre_registered_core_thresholds": split["core_thresholds"],
            "pre_registered_hotspot": hotspot,
            "core_metric_total_relative_error": core_results,
            "core_metric_under_threshold": core_passed,
            "core_metric_under_5pct": core_passed,
            "declared_error_below_5pct": numeric_passed,
            "spatial_hotspot_gate": spatial_hotspot_gate,
            "official_accuracy_gate": numeric_passed,
            "official_reference_gate": False,
            "provenance_note": "independently replayed measured evidence; never an organizer reference",
            "source_manifest_sha256": source_document["manifest_sha256"],
            "split_manifest_sha256": split_document["manifest_sha256"],
            "identity_bindings": expected_bindings,
        }
        result["validation_sha256"] = canonical_sha256(result)
        return result
    except (MeasuredValidationError, ValidationError, OSError, TypeError, ValueError, KeyError) as exc:
        return _blocked(str(exc) or type(exc).__name__)


# Stable aliases for callers that used the earlier experimental naming.
validate = validate_measured
run = validate_measured
run_validation = validate_measured
validate_records = validate_measured
build_measured_validation = validate_measured
calculate_measured_validation = validate_measured


def save_validation_report(report: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    return safe_json_dump(report, path)


def _selftest() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        entries = {}
        for index, identifier in enumerate(("r-test", "r-train")):
            path = root / f"source-{index}.json"
            path.write_text(json.dumps({"record_id": identifier, "index": index}), encoding="utf-8")
            entries[identifier] = {"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size, "status": "measured", "metadata": {"record_id": identifier, "kind": "depth", "schema": "v1"}}
        source = {"version": "selftest-source-v1", "status": "measured", "provenance": {"origin": "laboratory_measurement", "acquisition_method": "profilometer", "acquired_at": "2026-07-30T00:00:00Z", "organization": "selftest-laboratory"}, "entries": entries}
        source["manifest_sha256"] = canonical_sha256(source)
        hashes = [entries[x]["sha256"] for x in ("r-test", "r-train")]
        split = {"version": "selftest-split-v1", "status": "measured", "pre_registered_core_metrics": [{"metric": "wear_depth_mm", "total_relative_error_max": 0.05}], "pre_registered_hotspot": {"metric": "wear_depth_mm", "percentile": 75.0, "area_relative_error_max": 0.05, "iou_error_max": 0.05}, "records": [{"record_id": "r-test", "split": "independent_test", "group_id": "test-group", "source_sha256": hashes[0]}, {"record_id": "r-train", "split": "train", "group_id": "train-group", "source_sha256": hashes[1]}]}
        split["manifest_sha256"] = canonical_sha256(split)
        transform_hash, topology_hash = "1" * 64, "2" * 64

        def record(identifier: str, source_hash: str, values: list[float]) -> dict[str, Any]:
            metric = {"values": values, "units": "mm", "parent_face_id": [0, 1], "n_parent_faces": 2, "cohort_id": "selftest-faces", "area_mm2": {"values": [2.0, 1.0], "units": "mm2"}}
            return {"record_id": identifier, "status": "measured", "source_sha256": source_hash, "transform_sha256": transform_hash, "topology_sha256": topology_hash, "wear_depth_mm": metric}

        prediction = {"status": "measured", "records": [record("r-test", hashes[0], [1.0, 2.0]), record("r-train", hashes[1], [1.0, 2.0])]}
        observation = {"status": "measured", "records": [record("r-test", hashes[0], [1.0, 2.01]), record("r-train", hashes[1], [1.0, 2.0])]}
        bindings = {"source_manifest_sha256": source["manifest_sha256"], "protocol_sha256": split["manifest_sha256"], "transform_sha256": transform_hash, "topology_sha256": topology_hash}
        quality: dict[str, Any] = {"version": "selftest-quality-v1", "status": "measured"}
        for gate in QUALITY_GATES:
            report = {"version": f"selftest-{gate}-v1", "status": "measured", "bindings": bindings, "thresholds": {"coverage": {"minimum": 0.95}}, "actuals": {"coverage": 0.99}, "passed": False}
            report["report_sha256"] = canonical_sha256(report)
            quality[gate] = report
        quality["quality_sha256"] = canonical_sha256(quality)
        report = validate_measured(prediction, observation, source, split, quality, source_root=root, bootstrap_samples=20)
        assert report["status"] == "passed" and report["official_accuracy_gate"] is True
        assert report["metrics"]["wear_depth_mm"]["aggregation"] == "area_weighted_mean"
        assert validate_measured(prediction, observation, source, split, {"quality_passed": True}, source_root=root)["status"] == "blocked"
        synthetic = dict(prediction, status="synthetic")
        assert validate_measured(synthetic, observation, source, split, quality, source_root=root)["evidence_status"] == "synthetic"


def selftest() -> bool:
    _selftest()
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Formal Step30 measured validation")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--quality", "--quality-json", dest="quality", type=Path)
    parser.add_argument("--registration-quality-json", type=Path)
    parser.add_argument("--geometry-quality-json", type=Path)
    parser.add_argument("--measurement-quality-json", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--independent", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--quality-passed", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.selftest:
        selftest()
        print("step30_measured_validation selftest: PASS")
        return 0
    required = (args.predictions, args.observations, args.source_manifest, args.split_manifest)
    has_quality = args.quality is not None or any(
        value is not None
        for value in (args.registration_quality_json, args.geometry_quality_json, args.measurement_quality_json)
    )
    if any(value is None for value in required) or not has_quality:
        report = _blocked("missing required validation input")
    else:
        report = validate_measured(
            args.predictions,
            args.observations,
            args.source_manifest,
            args.split_manifest,
            args.quality,
            source_root=args.source_root,
            bootstrap_samples=args.bootstrap_samples,
            registration_quality_json=args.registration_quality_json,
            geometry_quality_json=args.geometry_quality_json,
            measurement_quality_json=args.measurement_quality_json,
        )
    if args.output is not None:
        save_validation_report(report, args.output)
    print(json.dumps(jsonable(report), ensure_ascii=False, indent=2))
    return 0 if report.get("official_accuracy_gate") else 3


if __name__ == "__main__":
    raise SystemExit(main())
