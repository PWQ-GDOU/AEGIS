"""Fail-closed material and bucket measurement schemas.

A ``measured`` record is eligible for downstream use only when its source and
measurement provenance are complete.  ``synthetic`` and ``blocked`` are valid
explicit states, but they carry explicit explanations and are never silently
promoted to measured data.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - JSON/standard-library use remains available
    np = None  # type: ignore[assignment]

from step30_units import UnitError, normalize_unit, unit_dimension


VERSION = "step30-measurement-schema-v1"
VALID_STATUSES = frozenset({"measured", "synthetic", "blocked"})
REQUIRED_MEASURED_PROVENANCE = (
    "source_sha256",
    "instrument",
    "calibration",
    "timestamp",
    "frame",
    "uncertainty",
    "campaign",
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class MeasurementSchemaError(ValueError):
    """Raised when a measurement record violates the Step30 schema."""


# Canonical field aliases and their physical dimensions.  Units are still
# required in the input ``units`` mapping or directly on a measurement entry.
_MATERIAL_FIELDS: dict[str, str] = {
    "speed": "speed", "speed_m_s": "speed", "velocity": "speed",
    "impact_speed": "speed", "impact_angle": "angle", "impact_angle_deg": "angle",
    "angle": "angle", "angle_deg": "angle", "particle_size": "length",
    "particle_size_um": "length", "d50": "length", "d50_um": "length",
    "psd_d10": "length", "psd_d10_um": "length", "psd_d50": "length",
    "psd_d50_um": "length", "psd_d90": "length", "psd_d90_um": "length",
    "sediment_mass": "mass", "sediment_mass_kg": "mass", "mass": "mass",
    "before_mass": "mass", "before_mass_g": "mass", "after_mass": "mass",
    "after_mass_g": "mass", "mass_loss": "mass", "mass_loss_g": "mass",
    "exposure": "time", "exposure_s": "time", "exposure_time": "time",
    "area": "area", "exposed_area": "area", "exposed_area_mm2": "area",
    "wear_depth": "length", "wear_depth_mm": "length", "depth": "length",
    "depth_mm": "length", "wear_volume": "volume", "wear_volume_mm3": "volume",
    "volume": "volume", "volume_mm3": "volume", "wear_rate": "linear_rate",
    "wear_rate_mm_h": "linear_rate", "erosion_volume_per_kg": "specific_volume",
    "wear_volume_per_kg": "specific_volume", "erosion_mass_per_kg": "mass_ratio",
    "erosion_volume_mm3_per_kg": "specific_volume", "erosion_mass_kg_per_kg": "mass_ratio",
    "mass_ratio": "mass_ratio",
}

_BUCKET_FIELDS: dict[str, str] = {
    "coordinates": "length", "coordinate": "length", "points": "length",
    "coordinates_mm": "length", "points_mm": "length", "original_xyz_mm": "length",
    "original_coordinates": "length", "original_points": "length",
    "worn_xyz_mm": "length", "worn_coordinates": "length", "worn_points": "length",
    "wear_depth": "length", "wear_depth_mm": "length", "depth": "length",
    "erosion_depth": "length", "erosion": "length", "wear_rate": "linear_rate",
    "wear_rate_mm_h": "linear_rate", "surface_area": "area", "area": "area",
    "wear_volume": "volume", "wear_volume_mm3": "volume", "volume": "volume",
    "wear_depth_mm": "length", "erosion_volume_mm3_per_kg": "specific_volume",
    "erosion_mass_kg_per_kg": "mass_ratio",
    "impact_speed": "speed", "speed": "speed", "impact_angle": "angle",
    "impact_angle_deg": "angle",
}

# Units accepted for each dimension.  This keeps a field such as an angle
# from being accidentally supplied in a length unit while allowing explicit
# conversion from m/cm/mm/um and the other requested units.
_DIMENSION_UNITS: dict[str, frozenset[str]] = {
    "length": frozenset({"m", "cm", "mm", "um"}),
    "time": frozenset({"s", "h"}),
    "mass": frozenset({"kg", "g", "mg"}),
    "area": frozenset({"m2", "mm2"}),
    "volume": frozenset({"mm3"}),
    "specific_volume": frozenset({"mm3/kg"}),
    "mass_ratio": frozenset({"kg/kg"}),
    "linear_rate": frozenset({"mm/h"}),
    "speed": frozenset({"m/s"}),
    "angle": frozenset({"deg"}),
}


def _error(message: str) -> MeasurementSchemaError:
    return MeasurementSchemaError(message)


def _nonblank(value: Any, field: str) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise _error(f"{field} must be present and non-empty")
    if isinstance(value, Mapping) and not value:
        raise _error(f"{field} must not be empty")
    if isinstance(value, (list, tuple)) and not value:
        raise _error(f"{field} must not be empty")
    return value


def _finite(value: Any, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise _error(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise _error(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise _error(f"{field} must be finite")
    if nonnegative and result < 0.0:
        raise _error(f"{field} must be non-negative")
    return result


def _finite_values(value: Any, field: str) -> None:
    if isinstance(value, bool):
        raise _error(f"{field} must contain numeric values")
    if np is not None and isinstance(value, np.ndarray):
        if np.iscomplexobj(value) or not np.issubdtype(value.dtype, np.number):
            raise _error(f"{field} must contain numeric values")
        if not np.isfinite(value).all():
            raise _error(f"{field} must contain only finite values")
        return
    if isinstance(value, Mapping):
        for name, item in value.items():
            _finite_values(item, f"{field}.{name}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_values(item, f"{field}[{index}]")
        return
    _finite(value, field)


def _uncertainty(value: Any) -> Any:
    _nonblank(value, "uncertainty")
    if isinstance(value, Mapping):
        for name, item in value.items():
            if str(name).lower() in {"unit", "units", "method", "type", "confidence"}:
                continue
            if isinstance(item, Mapping):
                _uncertainty(item)
            elif isinstance(item, (list, tuple)):
                for index, number in enumerate(item):
                    _finite(number, f"uncertainty.{name}[{index}]", nonnegative=True)
            else:
                _finite(item, f"uncertainty.{name}", nonnegative=True)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite(item, f"uncertainty[{index}]", nonnegative=True)
    else:
        _finite(value, "uncertainty", nonnegative=True)
    return value


def _timestamp(value: Any) -> str:
    _nonblank(value, "timestamp")
    if not isinstance(value, str):
        raise _error("timestamp must be an ISO-8601 string")
    text = value.strip()
    try:
        parsed = _datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _error("timestamp must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _error("timestamp must include an explicit timezone")
    return text


def _provenance(record: Mapping[str, Any], status: str) -> dict[str, Any]:
    nested = record.get("provenance")
    nested = nested if isinstance(nested, Mapping) else {}
    result: dict[str, Any] = {}
    missing: list[str] = []
    for field in REQUIRED_MEASURED_PROVENANCE:
        value = record.get(field, nested.get(field))
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field)
        else:
            result[field] = value
    if status == "measured" and missing:
        raise _error(f"measured provenance missing: {', '.join(missing)}")
    if status == "measured":
        if record.get("synthetic") is True:
            raise _error("measured record cannot set synthetic=true")
        if record.get("blocked_reason") not in (None, ""):
            raise _error("measured record cannot carry blocked_reason")
        source_hash = result["source_sha256"]
        if not isinstance(source_hash, str) or not _SHA256_RE.fullmatch(source_hash.strip()):
            raise _error("source_sha256 must be a 64-character hexadecimal SHA-256")
        result["source_sha256"] = source_hash.strip().lower()
        for field in ("instrument", "calibration", "frame", "campaign"):
            _nonblank(result[field], field)
        result["timestamp"] = _timestamp(result["timestamp"])
        result["uncertainty"] = _uncertainty(result["uncertainty"])
    elif status == "synthetic":
        if record.get("blocked_reason") not in (None, ""):
            raise _error("synthetic record cannot carry blocked_reason")
        explanation = record.get("synthetic_reason", record.get("truth_source"))
        if explanation is None and isinstance(nested, Mapping):
            explanation = nested.get("synthetic_reason", nested.get("truth_source"))
        if explanation is None or (isinstance(explanation, str) and not explanation.strip()):
            raise _error("synthetic record requires an explicit synthetic_reason or truth_source")
        if "synthetic" in record and record["synthetic"] is not True:
            raise _error("synthetic record must set synthetic=true when the marker is present")
        result["synthetic_reason"] = explanation
    else:
        if record.get("synthetic") is True:
            raise _error("blocked record cannot set synthetic=true")
        reason = record.get("blocked_reason")
        if reason is None and isinstance(nested, Mapping):
            reason = nested.get("blocked_reason")
        _nonblank(reason, "blocked_reason")
        result["blocked_reason"] = reason
    return result


def _status(record: Mapping[str, Any]) -> str:
    value = record.get("status", record.get("measurement_status"))
    if not isinstance(value, str) or value not in VALID_STATUSES:
        raise _error(f"status must be one of {sorted(VALID_STATUSES)}")
    return value


def _kind(record: Mapping[str, Any], requested: str | None) -> str:
    if requested is not None and requested not in {"material", "bucket"}:
        raise _error("kind must be material or bucket")
    supplied = record.get("kind", record.get("schema", record.get("record_type")))
    if isinstance(supplied, str):
        lowered = supplied.lower().replace("_measurement", "").replace("-measurement", "")
        if lowered in {"material", "bucket"}:
            supplied = lowered
        else:
            supplied = None
    if requested is not None and supplied is not None and supplied != requested:
        raise _error(f"record kind {supplied!r} differs from requested {requested!r}")
    if requested is not None:
        return requested
    if supplied is not None:
        return supplied
    if any(name in record for name in ("coordinates", "points", "wear_depth", "wear_depth_mm", "topology")):
        return "bucket"
    return "material"


def _measurements(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("measurements", record.get("values"))
    if value is None:
        return record
    if not isinstance(value, Mapping):
        raise _error("measurements/values must be an object")
    return value


def _field_name(name: Any) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def _field_dimension(name: str, kind: str) -> str | None:
    fields = _MATERIAL_FIELDS if kind == "material" else _BUCKET_FIELDS
    normalized = _field_name(name)
    if normalized in fields:
        return fields[normalized]
    # IDs, flags, counts, and metadata are not dimensional measurements.
    if normalized.endswith(("_id", "_count", "_index")) or normalized in {
        "status", "material", "specimen", "specimen_id", "campaign", "run_id",
        "parent_face_id", "point_id", "frame", "instrument",
    }:
        return None
    return None


def _units(record: Mapping[str, Any], values: Mapping[str, Any], kind: str) -> dict[str, str]:
    supplied = record.get("units")
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, Mapping):
        raise _error("units must be an object with an explicit unit per measurement field")
    normalized: dict[str, str] = {}
    for field, raw_unit in supplied.items():
        try:
            unit = normalize_unit(raw_unit)
        except UnitError as exc:
            raise _error(str(exc)) from exc
        normalized[_field_name(field)] = unit

    for field, entry in values.items():
        name = _field_name(field)
        dimension = _field_dimension(name, kind)
        if dimension is None:
            continue
        entry_unit = entry.get("unit") if isinstance(entry, Mapping) and "value" in entry else None
        raw_unit = entry_unit if entry_unit is not None else normalized.get(name)
        if raw_unit is None:
            raise _error(f"measurement field {field!r} has no explicit unit")
        try:
            unit = normalize_unit(raw_unit)
        except UnitError as exc:
            raise _error(f"{field}: {exc}") from exc
        if unit not in _DIMENSION_UNITS[dimension]:
            raise _error(f"{field} unit {unit!r} is incompatible with dimension {dimension!r}")
        normalized[name] = unit
    return normalized


def _entry_value(entry: Any, field: str) -> Any:
    if isinstance(entry, Mapping) and "value" in entry:
        return entry["value"]
    return entry


def _validate_measurement_values(values: Mapping[str, Any], kind: str) -> dict[str, Any]:
    if not values:
        raise _error(f"{kind} record has no measurement values")
    result: dict[str, Any] = {}
    for field, entry in values.items():
        name = _field_name(field)
        value = _entry_value(entry, str(field))
        dimension = _field_dimension(name, kind)
        if dimension is not None:
            _finite_values(value, str(field))
        result[name] = value
    return result


def _material(record: Mapping[str, Any], values: Mapping[str, Any]) -> None:
    material = record.get("material", record.get("material_name", record.get("alloy")))
    _nonblank(material, "material")
    specimen = record.get("specimen_id", record.get("sample_id", record.get("coupon_id")))
    _nonblank(specimen, "specimen_id")
    if not any(_field_dimension(_field_name(field), "material") is not None for field in values):
        raise _error("material record has no recognized dimensional measurement")


def _shape(value: Any, field: str) -> tuple[int, ...] | None:
    if np is not None and isinstance(value, np.ndarray):
        if value.size == 0:
            raise _error(f"{field} must not be empty")
        return tuple(int(item) for item in value.shape)
    if not isinstance(value, (list, tuple)):
        return None
    if not value:
        raise _error(f"{field} must not be empty")
    if isinstance(value[0], (list, tuple)):
        width = len(value[0])
        if width == 0 or any(not isinstance(row, (list, tuple)) or len(row) != width for row in value):
            raise _error(f"{field} must be a rectangular array")
        return (len(value), width)
    return (len(value),)


def _bucket(record: Mapping[str, Any], values: Mapping[str, Any]) -> None:
    coordinate_field = next((field for field in values if _field_name(field) in {"coordinates", "coordinate", "points", "original_coordinates", "original_points"}), None)
    depth_field = next((field for field in values if _field_name(field) in {"wear_depth", "wear_depth_mm", "depth", "erosion_depth", "erosion"}), None)
    if coordinate_field is None:
        raise _error("bucket record requires coordinates/points")
    coordinates = _entry_value(values[coordinate_field], str(coordinate_field))
    coordinate_shape = _shape(coordinates, str(coordinate_field))
    if coordinate_shape is None or len(coordinate_shape) != 2 or coordinate_shape[1] != 3:
        raise _error("bucket coordinates must be an N x 3 array")
    _finite_values(coordinates, str(coordinate_field))
    if depth_field is None:
        raise _error("bucket record requires wear_depth/erosion_depth")
    depths = _entry_value(values[depth_field], str(depth_field))
    depth_shape = _shape(depths, str(depth_field))
    _finite_values(depths, str(depth_field))
    if depth_shape is not None and depth_shape != (coordinate_shape[0],):
        raise _error("bucket wear depth length must match coordinate count")
    for name in ("point_id", "parent_face_id"):
        if name in values:
            ids = _entry_value(values[name], name)
            shape = _shape(ids, name)
            if shape is None or shape != (coordinate_shape[0],):
                raise _error(f"bucket {name} length must match coordinate count")
            items = ids.tolist() if np is not None and isinstance(ids, np.ndarray) else ids
            for item in items:
                if isinstance(item, bool) or int(item) != item or int(item) < 0:
                    raise _error(f"bucket {name} must contain non-negative integer IDs")


def validate_measurement(record: Mapping[str, Any], kind: str | None = None) -> dict[str, Any]:
    """Validate one material or bucket record and return a normalized copy."""
    if not isinstance(record, Mapping):
        raise _error("measurement record must be an object")
    status = _status(record)
    resolved_kind = _kind(record, kind)
    provenance = _provenance(record, status)
    values = _measurements(record)
    normalized_values = _validate_measurement_values(values, resolved_kind)
    normalized_units = _units(record, values, resolved_kind)
    if status != "blocked":
        if resolved_kind == "material":
            _material(record, values)
        else:
            _bucket(record, values)
    result = dict(record)
    result["version"] = VERSION
    result["kind"] = resolved_kind
    result["status"] = status
    result["provenance"] = provenance
    result["measurements"] = normalized_values
    result["units"] = normalized_units
    if status == "synthetic":
        result["synthetic"] = True
        result["provenance_note"] = "synthetic result; not official accuracy"
    elif status == "blocked":
        result["provenance_note"] = "blocked result; not eligible for accuracy claims"
    else:
        result["provenance_note"] = "measured result; provenance validated"
    return result


def validate_material_measurement(record: Mapping[str, Any]) -> dict[str, Any]:
    return validate_measurement(record, "material")


def validate_bucket_measurement(record: Mapping[str, Any]) -> dict[str, Any]:
    return validate_measurement(record, "bucket")


def validate_records(records: Sequence[Mapping[str, Any]], kind: str | None = None) -> list[dict[str, Any]]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise _error("records must be a sequence of objects")
    if not records:
        raise _error("records must not be empty")
    return [validate_measurement(record, kind) for record in records]


def _selftest_provenance(status: str = "measured") -> dict[str, Any]:
    if status == "measured":
        return {
            "source_sha256": hashlib.sha256(b"measured").hexdigest(),
            "instrument": "profilometer-01",
            "calibration": {"certificate": "CAL-1", "valid": True},
            "timestamp": "2026-07-30T12:00:00Z",
            "frame": "bucket-local",
            "uncertainty": {"wear_depth": 0.01},
            "campaign": "C-2026-01",
        }
    if status == "synthetic":
        return {"truth_source": "fixed_seed_generator", "synthetic_reason": "deterministic fixture"}
    return {"blocked_reason": "source measurement was not supplied"}


def _selftest() -> None:
    measured = {
        "kind": "material", "status": "measured", "material": "ZG0Cr13Ni4Mo", "specimen_id": "S-1",
        "measurements": {"speed": 50.0, "impact_angle": 45.0, "wear_depth": 0.2, "wear_volume": 3.0},
        "units": {"speed": "m/s", "impact_angle": "deg", "wear_depth": "mm", "wear_volume": "mm3"},
        **_selftest_provenance(),
    }
    normalized = validate_material_measurement(measured)
    assert normalized["status"] == "measured" and normalized["units"]["wear_depth"] == "mm"
    bucket = {
        "kind": "bucket", "status": "measured",
        "measurements": {"coordinates": [[0, 0, 0], [1, 0, 0]], "wear_depth": [0.1, 0.2]},
        "units": {"coordinates": "mm", "wear_depth": "mm"},
        **_selftest_provenance(),
    }
    assert validate_bucket_measurement(bucket)["kind"] == "bucket"
    synthetic = {"kind": "bucket", "status": "synthetic", "measurements": {"coordinates": [[0, 0, 0]], "wear_depth": [0.1]}, "units": {"coordinates": "mm", "wear_depth": "mm"}, **_selftest_provenance("synthetic")}
    assert validate_bucket_measurement(synthetic)["synthetic"] is True
    blocked = {"kind": "material", "status": "blocked", "blocked_reason": "no source"}
    assert validate_material_measurement(blocked)["status"] == "blocked"
    for missing in ("source_sha256", "instrument", "calibration", "timestamp", "frame", "uncertainty", "campaign"):
        broken = dict(measured)
        broken.pop(missing)
        try:
            validate_material_measurement(broken)
        except MeasurementSchemaError:
            pass
        else:
            raise AssertionError(f"missing provenance was accepted: {missing}")
    try:
        validate_material_measurement({**measured, "units": {"speed": "mm"}})
    except MeasurementSchemaError:
        pass
    else:
        raise AssertionError("incompatible field unit was accepted")


def selftest() -> bool:
    _selftest()
    return True


def _load_json(path: Path) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MeasurementSchemaError(f"invalid JSON input: {path}") from exc
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Step30 material/bucket measurement provenance")
    parser.add_argument("--input", type=Path, help="JSON record or {records: [...]} input")
    parser.add_argument("--kind", choices=("material", "bucket"), help="schema kind; infer when omitted")
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        selftest()
        print("step30_measurement_schema selftest: PASS")
        return 0
    if args.input is None:
        build_parser().error("--input is required unless --selftest is used")
    try:
        payload = _load_json(args.input)
        if isinstance(payload, Mapping) and isinstance(payload.get("records"), list):
            result: Any = validate_records(payload["records"], args.kind)
        else:
            result = validate_measurement(payload, args.kind)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (MeasurementSchemaError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "schema": VERSION, "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
