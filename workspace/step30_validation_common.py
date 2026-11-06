"""Common fail-closed validation, provenance, hashing, and safe-output helpers.

This module deliberately uses only the Python standard library and NumPy.  The
other Step30 validation modules use these helpers so that measured, synthetic,
and blocked records cannot be confused accidentally.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


COMMON_VERSION = "step30-validation-common-v1"
VALID_STATUSES = frozenset({"measured", "synthetic", "blocked"})
SOURCE_TYPES = frozenset({
    "contest_provided_simulation",
    "external_measured",
    "local_synthetic_fixture",
    "none",
})
REFERENCE_ROLES = frozenset({
    "official_reference_simulation",
    "external_measured_calibration",
    "synthetic_pipeline_fixture",
    "none",
})


class EvidenceContractError(ValueError):
    """Raised when evidence provenance fields are incomplete or inconsistent."""


def evidence_metadata(
    value: Mapping[str, Any] | None = None,
    *,
    status: str | None = None,
    source_type: str | None = None,
    reference_role: str | None = None,
    pipeline_validation: bool | None = None,
    official_reference: bool | None = None,
) -> dict[str, Any]:
    """Validate the non-numerical evidence taxonomy without changing status semantics."""
    data = dict(value or {})
    if status is not None:
        data["status"] = status
    if source_type is not None:
        data["source_type"] = source_type
    if reference_role is not None:
        data["reference_role"] = reference_role
    if pipeline_validation is not None:
        data["pipeline_validation"] = pipeline_validation
    if official_reference is not None:
        data["official_reference"] = official_reference
    if "status" in data:
        require_status(data["status"])
    source = data.get("source_type", "none")
    role = data.get("reference_role", "none")
    if source not in SOURCE_TYPES:
        raise EvidenceContractError(f"unsupported source_type: {source!r}")
    if role not in REFERENCE_ROLES:
        raise EvidenceContractError(f"unsupported reference_role: {role!r}")
    pipeline = data.get("pipeline_validation", False)
    official = data.get("official_reference", False)
    if not isinstance(pipeline, bool) or not isinstance(official, bool):
        raise EvidenceContractError("pipeline_validation and official_reference must be booleans")
    if source == "local_synthetic_fixture":
        if data.get("status") not in (None, "synthetic") or role != "synthetic_pipeline_fixture" or not pipeline or official:
            raise EvidenceContractError("local synthetic evidence has an invalid provenance combination")
    elif source == "external_measured":
        if data.get("status") not in (None, "measured") or role != "external_measured_calibration" or official:
            raise EvidenceContractError("external measured evidence has an invalid provenance combination")
    elif source == "contest_provided_simulation":
        if role != "official_reference_simulation" or pipeline or not official:
            raise EvidenceContractError("contest reference evidence has an invalid provenance combination")
    elif role != "none" or pipeline or official:
        raise EvidenceContractError("source_type=none cannot claim a reference role")
    data.update({
        "source_type": source,
        "reference_role": role,
        "pipeline_validation": pipeline,
        "official_reference": official,
    })
    return data


class ValidationError(ValueError):
    """Raised when a validation contract is not satisfied."""


def require_status(status: str, *, allowed: Iterable[str] = VALID_STATUSES) -> str:
    """Validate and return one of the explicit Step30 provenance states."""
    if not isinstance(status, str) or status not in set(allowed):
        raise ValidationError(
            f"status must be one of {sorted(set(allowed))}, got {status!r}"
        )
    return status


def status_is_measured(status: str) -> bool:
    return require_status(status) == "measured"


def status_is_synthetic(status: str) -> bool:
    return require_status(status) == "synthetic"


def status_is_blocked(status: str) -> bool:
    return require_status(status) == "blocked"


def finite_array(
    value: Any,
    name: str = "array",
    *,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
    nonempty: bool = True,
    dtype: Any = np.float64,
) -> np.ndarray:
    """Return a numeric finite array or fail closed."""
    try:
        raw_array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} is not a numeric array") from exc
    if np.iscomplexobj(raw_array):
        raise ValidationError(f"{name} must contain real values")
    try:
        array = np.asarray(raw_array, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} is not a numeric array") from exc
    if ndim is not None and array.ndim != ndim:
        raise ValidationError(f"{name} must have ndim={ndim}, got {array.ndim}")
    if shape is not None and tuple(array.shape) != tuple(shape):
        raise ValidationError(f"{name} must have shape {shape}, got {array.shape}")
    if nonempty and array.size == 0:
        raise ValidationError(f"{name} must not be empty")
    if np.iscomplexobj(array):
        raise ValidationError(f"{name} must contain real values")
    if not np.isfinite(array).all():
        raise ValidationError(f"{name} contains NaN or infinite values")
    return array


def finite_scalar(value: Any, name: str = "value") -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} is not numeric") from exc
    if not math.isfinite(result):
        raise ValidationError(f"{name} is not finite")
    return result


# Factors convert a quantity to the SI base unit of its dimension.
_UNIT_FACTORS: dict[str, tuple[str, float]] = {
    "m": ("length", 1.0),
    "cm": ("length", 1.0e-2),
    "mm": ("length", 1.0e-3),
    "um": ("length", 1.0e-6),
    "micrometer": ("length", 1.0e-6),
    "micrometre": ("length", 1.0e-6),
    "s": ("time", 1.0),
    "ms": ("time", 1.0e-3),
    "us": ("time", 1.0e-6),
    "kg/m3": ("density", 1.0),
    "kg m-3": ("density", 1.0),
    "m2": ("area", 1.0),
    "m^2": ("area", 1.0),
    "cm2": ("area", 1.0e-4),
    "cm^2": ("area", 1.0e-4),
    "mm2": ("area", 1.0e-6),
    "mm^2": ("area", 1.0e-6),
    "pa": ("pressure", 1.0),
    "m2/s2": ("specific_energy", 1.0),
    "m^2/s^2": ("specific_energy", 1.0),
    "m/s": ("velocity", 1.0),
    "mm/s": ("velocity", 1.0e-3),
    "1/s": ("rate", 1.0),
    "s-1": ("rate", 1.0),
    "n": ("force", 1.0),
    "kg": ("mass", 1.0),
    "dimensionless": ("dimensionless", 1.0),
    "1": ("dimensionless", 1.0),
}


def normalize_unit(unit: str) -> str:
    if not isinstance(unit, str):
        raise ValidationError(f"unit must be a string, got {type(unit).__name__}")
    normalized = unit.strip().lower().replace("²", "^2")
    aliases = {
        "meter": "m",
        "metre": "m",
        "meters": "m",
        "metres": "m",
        "millimeter": "mm",
        "millimetre": "mm",
        "millimeters": "mm",
        "millimetres": "mm",
        "microns": "um",
        "μm": "um",
        "µm": "um",
        "pascal": "pa",
        "pascals": "pa",
        "m^2 s^-2": "m2/s2",
        "m2 s-2": "m2/s2",
        "m/s": "m/s",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _UNIT_FACTORS:
        raise ValidationError(f"unsupported unit {unit!r}")
    return normalized


def validate_units(
    units: str,
    *,
    expected_dimension: str | None = None,
    expected: str | None = None,
) -> str:
    """Validate a unit and optionally its physical dimension or exact value."""
    normalized = normalize_unit(units)
    dimension, _ = _UNIT_FACTORS[normalized]
    if expected_dimension is not None and dimension != expected_dimension:
        raise ValidationError(
            f"unit {units!r} has dimension {dimension!r}, expected {expected_dimension!r}"
        )
    if expected is not None and normalized != normalize_unit(expected):
        raise ValidationError(f"unit {units!r} does not match expected unit {expected!r}")
    return normalized


def unit_dimension(units: str) -> str:
    return _UNIT_FACTORS[normalize_unit(units)][0]


def unit_factor_to_si(units: str) -> float:
    return _UNIT_FACTORS[normalize_unit(units)][1]


def convert_units(value: Any, from_units: str, to_units: str) -> np.ndarray:
    """Convert a finite numeric array between compatible units."""
    source = normalize_unit(from_units)
    target = normalize_unit(to_units)
    if unit_dimension(source) != unit_dimension(target):
        raise ValidationError(f"incompatible units: {from_units!r} and {to_units!r}")
    array = finite_array(value, "value", nonempty=False)
    return array * (unit_factor_to_si(source) / unit_factor_to_si(target))


def _jsonable(value: Any) -> Any:
    """Convert supported structured values to strict JSON-compatible values."""
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            raise ValidationError("object arrays are not safe JSON values")
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError("non-finite float cannot be serialized")
        return value
    raise ValidationError(f"unsupported JSON value type {type(value).__name__}")


def jsonable(value: Any) -> Any:
    """Public strict conversion helper used by record builders."""
    return _jsonable(value)


def canonical_json(value: Any) -> str:
    """Return deterministic, finite, compact JSON for hashing and manifests."""
    return json.dumps(
        _jsonable(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Existing Step30 modules use this spelling for file hashes.
file_sha256 = sha256_file


def write_json_immutable(value: Any, path: str | Path, *, indent: int = 2) -> dict[str, Any]:
    """Compatibility alias for the immutable JSON writer."""
    return safe_json_dump(value, path, indent=indent)


def _exclusive_write(path: str | Path, data: bytes) -> Path:
    """Create a file exactly once; never overwrite an existing artifact."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o644,
        )
    except FileExistsError as exc:
        raise FileExistsError(f"immutable output already exists: {target}") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
    except Exception:
        try:
            target.unlink()
        except OSError:
            pass
        raise
    return target


def safe_json_dump(value: Any, path: str | Path, *, indent: int = 2) -> dict[str, Any]:
    """Write strict JSON once and return the written path/hash record."""
    payload = json.dumps(
        _jsonable(value), ensure_ascii=False, indent=indent, allow_nan=False
    ).encode("utf-8")
    target = _exclusive_write(path, payload)
    return {"path": str(target), "sha256": sha256_bytes(payload), "bytes": len(payload)}


def safe_json_load(path: str | Path, *, max_bytes: int = 128 * 1024 * 1024) -> Any:
    target = Path(path)
    size = target.stat().st_size
    if size > max_bytes:
        raise ValidationError(f"JSON input exceeds safety limit: {target}")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid JSON input: {target}") from exc
    # Round-trip through the strict converter to reject non-finite values and
    # unsupported structures even when the parser accepted the input.
    _jsonable(value)
    return value


def safe_npz_dump(
    arrays: Mapping[str, Any],
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a compressed NPZ once, with pickle disabled on the read path."""
    if not arrays:
        raise ValidationError("NPZ output requires at least one array")
    payload: dict[str, np.ndarray] = {}
    for name, value in sorted(arrays.items()):
        if not isinstance(name, str) or not name or name.startswith("__"):
            raise ValidationError(f"invalid NPZ array name: {name!r}")
        array = np.asarray(value)
        if array.dtype == object or np.iscomplexobj(array):
            raise ValidationError(f"object or complex dtype is not safe for NPZ: {name}")
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValidationError(f"NPZ array contains non-finite values: {name}")
        payload[name] = array
    if metadata is not None:
        payload["__metadata_json__"] = np.asarray(canonical_json(metadata))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **payload)
    data = buffer.getvalue()
    target = _exclusive_write(path, data)
    return {"path": str(target), "sha256": sha256_bytes(data), "bytes": len(data)}


def safe_npz_load(
    path: str | Path,
    *,
    allowed_keys: Iterable[str] | None = None,
    require_finite: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load an NPZ without pickle and validate its numeric arrays."""
    allowed = None if allowed_keys is None else set(allowed_keys)
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    try:
        with np.load(str(path), allow_pickle=False) as loaded:
            for key in loaded.files:
                if allowed is not None and key not in allowed and key != "__metadata_json__":
                    raise ValidationError(f"unexpected NPZ key: {key}")
                value = np.asarray(loaded[key])
                if key == "__metadata_json__":
                    if value.ndim != 0:
                        raise ValidationError("invalid NPZ metadata")
                    metadata = safe_json_load_from_text(str(value.item()))
                    continue
                if value.dtype == object or np.iscomplexobj(value):
                    raise ValidationError(f"object or complex dtype is not safe: {key}")
                if require_finite and np.issubdtype(value.dtype, np.number):
                    if not np.isfinite(value).all():
                        raise ValidationError(f"NPZ array contains non-finite values: {key}")
                arrays[key] = value
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError(f"invalid NPZ input: {path}") from exc
    return arrays, metadata


def safe_json_load_from_text(text: str) -> Any:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError("invalid embedded NPZ metadata JSON") from exc
    _jsonable(value)
    return value


def manifest_entry(
    path: str | Path,
    *,
    root: str | Path | None = None,
    kind: str | None = None,
    status: str = "measured",
    units: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a source-file manifest entry with a hash and safe relative path."""
    require_status(status)
    target = Path(path).resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    if root is None:
        stored_path = target.name
    else:
        root_path = Path(root).resolve()
        try:
            stored_path = target.relative_to(root_path).as_posix()
        except ValueError as exc:
            raise ValidationError(f"manifest file is outside root: {target}") from exc
    result: dict[str, Any] = {
        "path": stored_path,
        "sha256": sha256_file(target),
        "bytes": int(target.stat().st_size),
        "status": status,
    }
    if kind is not None:
        result["kind"] = str(kind)
    if units is not None:
        result["units"] = {
            str(name): validate_units(unit) for name, unit in units.items()
        }
    return result


def build_manifest(
    *,
    version: str,
    entries: Mapping[str, Mapping[str, Any]],
    status: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    require_status(status)
    if not version:
        raise ValidationError("manifest version must not be empty")
    normalized_entries = {str(key): jsonable(value) for key, value in sorted(entries.items())}
    result: dict[str, Any] = {
        "version": version,
        "status": status,
        "entries": normalized_entries,
    }
    if metadata is not None:
        result["metadata"] = jsonable(metadata)
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    root: str | Path | None = None,
    expected_version: str | None = None,
    require_files: bool = True,
) -> dict[str, Any]:
    """Validate manifest structure, hash references, paths, and status."""
    if not isinstance(manifest, Mapping):
        raise ValidationError("manifest must be an object")
    require_status(manifest.get("status"))
    if expected_version is not None and manifest.get("version") != expected_version:
        raise ValidationError("manifest version mismatch")
    entries = manifest.get("entries")
    if not isinstance(entries, Mapping) or not entries:
        raise ValidationError("manifest has no entries")
    stored_hash = manifest.get("manifest_sha256")
    if not isinstance(stored_hash, str):
        raise ValidationError("manifest has no manifest_sha256")
    without_hash = dict(manifest)
    without_hash.pop("manifest_sha256", None)
    if canonical_sha256(without_hash) != stored_hash:
        raise ValidationError("manifest canonical hash mismatch")
    root_path = None if root is None else Path(root).resolve()
    for key, entry in entries.items():
        if not isinstance(entry, Mapping):
            raise ValidationError(f"manifest entry {key!r} is not an object")
        path_value = entry.get("path")
        file_hash = entry.get("sha256")
        if not isinstance(path_value, str) or not path_value or Path(path_value).is_absolute():
            raise ValidationError(f"manifest entry {key!r} has an unsafe path")
        if (
            not isinstance(file_hash, str)
            or len(file_hash) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in file_hash)
        ):
            raise ValidationError(f"manifest entry {key!r} has an invalid hash")
        require_status(entry.get("status", manifest["status"]))
        if root_path is not None and require_files:
            if entry.get("external") is True:
                absolute = entry.get("absolute_path")
                if not isinstance(absolute, str) or not Path(absolute).is_absolute():
                    raise ValidationError(f"external manifest entry has no absolute_path: {key}")
                target = Path(absolute).resolve()
            else:
                target = (root_path / Path(path_value)).resolve()
                try:
                    target.relative_to(root_path)
                except ValueError as exc:
                    raise ValidationError(f"manifest path escapes root: {path_value}") from exc
            if not target.is_file() or sha256_file(target) != file_hash:
                raise ValidationError(f"manifest hash mismatch: {path_value}")
    return dict(manifest)


def quality_gate(
    gates: Mapping[str, Any] | None,
    *,
    required: Iterable[str] | None = None,
) -> bool:
    """Return true only for an explicit all-boolean-true quality contract."""
    if not isinstance(gates, Mapping) or not gates:
        return False
    if required is not None and set(gates) != set(required):
        return False
    return all(isinstance(value, bool) and value for value in gates.values())


def independent_validation_ok(value: Any) -> bool:
    """Require explicit independence, a non-empty validation set, and provenance."""
    if not isinstance(value, Mapping):
        return False
    if value.get("is_independent") is not True:
        return False
    count = value.get("n", value.get("count", 0))
    try:
        if isinstance(count, bool) or int(count) <= 0:
            return False
    except (TypeError, ValueError):
        return False
    hashes = value.get("source_hashes")
    if isinstance(hashes, Mapping):
        return bool(hashes) and all(
            isinstance(item, str)
            and len(item) == 64
            and all(character in "0123456789abcdefABCDEF" for character in item)
            for item in hashes.values()
        )
    source_hash = value.get("source_sha256")
    return bool(
        isinstance(source_hash, str)
        and len(source_hash) == 64
        and all(character in "0123456789abcdefABCDEF" for character in source_hash)
    )



def official_accuracy_gate(
    *,
    status: str,
    independent_validation_set: Any,
    quality_passed: bool | None = None,
    quality_gates: Mapping[str, Any] | None = None,
) -> bool:
    """The sole official gate: measured + independent data + explicit quality pass."""
    require_status(status)
    quality_ok = bool(quality_passed) if quality_passed is not None else quality_gate(quality_gates)
    return bool(
        status == "measured"
        and independent_validation_ok(independent_validation_set)
        and quality_ok
    )


def provenance_note(status: str, evidence: Mapping[str, Any] | None = None) -> str:
    """Describe evidence role without conflating execution and provenance status."""
    require_status(status)
    metadata = evidence_metadata(evidence, status=status)
    role = metadata["reference_role"]
    if role == "official_reference_simulation":
        return "contest-provided reference simulation; eligible only for authenticated contest comparison"
    if role == "external_measured_calibration":
        return "external measured calibration evidence; not a bucket spatial ground truth"
    if role == "synthetic_pipeline_fixture":
        return "local synthetic pipeline fixture; not official accuracy"
    if status == "blocked":
        return "blocked result; not eligible for accuracy claims"
    if status == "synthetic":
        return "synthetic result; not official accuracy"
    return "measured result; official accuracy requires the independent-validation gate"


def _selftest() -> None:
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})
    assert validate_units("mm", expected_dimension="length") == "mm"
    assert np.allclose(convert_units([1000.0], "mm", "m"), [1.0])
    assert quality_gate({"finite": True})
    assert not quality_gate({"finite": False})
    assert official_accuracy_gate(
        status="measured",
        independent_validation_set={"is_independent": True, "n": 2, "source_sha256": "a" * 64},
        quality_gates={"finite": True},
    )
    assert not official_accuracy_gate(
        status="synthetic",
        independent_validation_set=True,
        quality_passed=True,
    )
    synthetic = evidence_metadata({"status": "synthetic", "source_type": "local_synthetic_fixture", "reference_role": "synthetic_pipeline_fixture", "pipeline_validation": True})
    assert synthetic["official_reference"] is False
    contest = evidence_metadata({"status": "synthetic", "source_type": "contest_provided_simulation", "reference_role": "official_reference_simulation", "official_reference": True})
    assert contest["pipeline_validation"] is False
    try:
        evidence_metadata({"status": "synthetic", "source_type": "local_synthetic_fixture", "reference_role": "official_reference_simulation", "pipeline_validation": True})
    except EvidenceContractError:
        pass
    else:
        raise AssertionError("invalid evidence role combination was accepted")
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "record.json"
        safe_json_dump({"finite": 1.0}, target)
        try:
            safe_json_dump({"finite": 2.0}, target)
        except FileExistsError:
            pass
        else:
            raise AssertionError("safe_json_dump overwrote an immutable file")
        npz = Path(directory) / "record.npz"
        safe_npz_dump({"x": np.array([1.0, 2.0])}, npz, metadata={"status": "measured"})
        arrays, metadata = safe_npz_load(npz)
        assert np.allclose(arrays["x"], [1.0, 2.0])
        assert metadata["status"] == "measured"
        entry = manifest_entry(target, root=directory)
        manifest = build_manifest(version="test-v1", entries={"record": entry}, status="measured")
        validate_manifest(manifest, root=directory)


def selftest() -> bool:
    _selftest()
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step30 common validation selftest")
    parser.add_argument("--selftest", action="store_true", help="run the minimal selftest")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        print("step30_validation_common selftest: PASS")
    else:
        parser.print_help()
