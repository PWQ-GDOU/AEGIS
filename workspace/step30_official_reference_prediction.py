"""Export immutable parent-face predictions for official reference comparison."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from step30_official_reference_simulation import (
    OfficialReferenceError,
    load_official_reference_manifest,
)
from step30_validation_common import (
    ValidationError,
    canonical_sha256,
    safe_json_dump,
    safe_json_load,
    sha256_file,
)

VERSION = "step30-official-reference-prediction-v1"
RUN_VERSION = "step30-particle-run-descriptor-v2"
_ARRAYS = {
    "erosion_rate": ("dhdt_face_mm_h", "mm/h"),
    "erosion_mass": ("face_target_mass_kg", "kg"),
    "erosion_ratio": ("ER_mass_ratio_face", "kg/kg"),
}


class OfficialReferencePredictionError(ValueError):
    """Raised when frozen production outputs cannot support the reference metric."""


def _hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdefABCDEF" for character in value
    )


def _scalar(data: Mapping[str, np.ndarray], field: str) -> Any:
    if field not in data:
        raise OfficialReferencePredictionError(f"particle artifact missing {field}")
    value = np.asarray(data[field])
    if value.ndim != 0:
        raise OfficialReferencePredictionError(f"particle artifact {field} must be scalar")
    return value.item()


def _run_manifest(path: Path) -> tuple[dict[str, Any], Path, str]:
    try:
        document = safe_json_load(path)
    except (OSError, UnicodeError, ValidationError) as exc:
        raise OfficialReferencePredictionError(f"invalid particle run manifest: {path}") from exc
    if not isinstance(document, Mapping):
        raise OfficialReferencePredictionError("particle run manifest must be an object")
    document = dict(document)
    if document.get("version") != RUN_VERSION:
        raise OfficialReferencePredictionError("unsupported particle run manifest version")
    outputs = document.get("outputs")
    if not isinstance(outputs, Mapping) or not isinstance(outputs.get("particle_hits"), Mapping):
        raise OfficialReferencePredictionError("particle run manifest lacks particle_hits output")
    descriptor = dict(document)
    descriptor.pop("outputs", None)
    particle_id = descriptor.get("particle_run_id")
    unsigned = dict(descriptor)
    unsigned.pop("particle_run_id", None)
    if not _hash(particle_id) or canonical_sha256(unsigned) != str(particle_id).lower():
        raise OfficialReferencePredictionError("particle_run_id/content mismatch")
    entry = outputs["particle_hits"]
    file_name = entry.get("file")
    expected_hash = entry.get("sha256")
    if not isinstance(file_name, str) or Path(file_name).name != file_name or not _hash(expected_hash):
        raise OfficialReferencePredictionError("particle_hits output entry is invalid")
    hits_path = path.parent / file_name
    if not hits_path.is_file() or sha256_file(hits_path) != str(expected_hash).lower():
        raise OfficialReferencePredictionError("particle_hits output hash mismatch")
    return document, hits_path, str(expected_hash).lower()


def _prediction_values(
    metric: str,
    arrays: Mapping[str, np.ndarray],
    exposure_hours: float | None,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    if metric == "wear_depth":
        if exposure_hours is None or not math.isfinite(exposure_hours) or exposure_hours <= 0.0:
            raise OfficialReferencePredictionError("wear_depth export requires positive --exposure-hours")
        if "dhdt_face_mm_h" not in arrays:
            raise OfficialReferencePredictionError("particle artifact lacks dhdt_face_mm_h")
        values = np.asarray(arrays["dhdt_face_mm_h"], dtype=np.float64) * exposure_hours
        return values, "mm", {
            "source_array": "dhdt_face_mm_h",
            "operation": "linear_rate_times_exposure",
            "exposure_hours": float(exposure_hours),
        }
    if metric not in _ARRAYS:
        raise OfficialReferencePredictionError(
            f"frozen production artifact cannot export value_metric={metric!r}"
        )
    field, units = _ARRAYS[metric]
    if field not in arrays:
        raise OfficialReferencePredictionError(f"particle artifact lacks {field}")
    return np.asarray(arrays[field], dtype=np.float64), units, {
        "source_array": field,
        "operation": "identity",
        "exposure_hours": None,
    }


def build_prediction(
    run_manifest: str | Path,
    reference_manifest: str | Path,
    *,
    exposure_hours: float | None = None,
) -> dict[str, Any]:
    """Build a hash-bound prediction document from a frozen particle run."""
    run_path = Path(run_manifest).resolve()
    reference_path = Path(reference_manifest).resolve()
    run, hits_path, hits_hash = _run_manifest(run_path)
    try:
        reference = load_official_reference_manifest(reference_path)
    except OfficialReferenceError as exc:
        raise OfficialReferencePredictionError(str(exc)) from exc
    artifact_set_id = reference["artifact_set_id"]
    if run.get("artifact_set_id") != artifact_set_id:
        raise OfficialReferencePredictionError("particle run and reference use different artifact_set_id values")
    config = run.get("config")
    if not isinstance(config, Mapping):
        raise OfficialReferencePredictionError("particle run config is missing")
    if config.get("collision_policy") != "strict_v2" or run.get("collision_backend") != "cfd_topology":
        raise OfficialReferencePredictionError("official prediction requires strict_v2 cfd_topology run")
    try:
        with np.load(hits_path, allow_pickle=False) as loaded:
            arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    except (OSError, ValueError) as exc:
        raise OfficialReferencePredictionError("invalid particle_hits NPZ") from exc
    if str(_scalar(arrays, "artifact_set_id")) != artifact_set_id:
        raise OfficialReferencePredictionError("particle artifact artifact_set_id mismatch")
    if str(_scalar(arrays, "particle_run_id")) != run.get("particle_run_id"):
        raise OfficialReferencePredictionError("particle artifact particle_run_id mismatch")
    if str(_scalar(arrays, "mesh_signature")) != reference["topology"]["mesh_signature"]:
        raise OfficialReferencePredictionError("particle artifact mesh signature mismatch")
    if (
        str(_scalar(arrays, "source_coordinate_units")),
        str(_scalar(arrays, "internal_coordinate_units")),
        float(_scalar(arrays, "coordinate_scale_to_mm")),
    ) != ("m", "mm", 1000.0):
        raise OfficialReferencePredictionError("particle artifact coordinate unit contract mismatch")
    contract = reference["record_contract"]
    metric = contract["value_metric"]
    values, value_units, derivation = _prediction_values(metric, arrays, exposure_hours)
    if value_units != contract["value_units"]:
        raise OfficialReferencePredictionError("prediction units differ from official reference contract")
    n_faces = int(reference["topology_data"]["n_parent_faces"])
    if values.shape != (n_faces,) or not np.isfinite(values).all() or np.any(values < 0.0):
        raise OfficialReferencePredictionError("prediction array is malformed or non-physical")
    split_ids = reference["independent_split"]["record_ids"]
    field = contract["prediction_field"]
    records = [
        {
            "parent_face_id": identifier,
            field: float(values[identifier]),
            "value_units": value_units,
        }
        for identifier in split_ids
    ]
    records_hash = canonical_sha256({"records": records})
    result = {
        "version": VERSION,
        "status": "synthetic",
        "source_kind": "frozen_native_prediction",
        "artifact_set_id": artifact_set_id,
        "particle_run_id": run["particle_run_id"],
        "mesh_signature": reference["topology"]["mesh_signature"],
        "value_metric": metric,
        "value_units": value_units,
        "record_contract_sha256": canonical_sha256(contract),
        "evaluation_contract_sha256": canonical_sha256(reference["evaluation_contract"]),
        "reference_manifest_sha256": reference["manifest_sha256"],
        "source_artifacts": {
            "particle_run_manifest": {
                "path": str(run_path),
                "sha256": sha256_file(run_path),
            },
            "particle_hits": {
                "path": str(hits_path),
                "sha256": hits_hash,
            },
        },
        "derivation": derivation,
        "records": records,
        "records_sha256": records_hash,
    }
    result["document_sha256"] = canonical_sha256(result)
    return result


def validate_prediction_document(
    document: Any,
    reference: Mapping[str, Any],
) -> str:
    """Revalidate an exported prediction and its frozen source artifacts."""
    if not isinstance(document, Mapping) or document.get("version") != VERSION:
        raise OfficialReferencePredictionError("official prediction document version is invalid")
    if document.get("source_kind") != "frozen_native_prediction":
        raise OfficialReferencePredictionError("official prediction source_kind is invalid")
    unsigned = dict(document)
    stored_document_hash = unsigned.pop("document_sha256", None)
    if not _hash(stored_document_hash) or canonical_sha256(unsigned) != str(stored_document_hash).lower():
        raise OfficialReferencePredictionError("official prediction document_sha256 is invalid")
    for field, expected in (
        ("artifact_set_id", reference["artifact_set_id"]),
        ("mesh_signature", reference["topology"]["mesh_signature"]),
        ("value_metric", reference["record_contract"]["value_metric"]),
        ("value_units", reference["record_contract"]["value_units"]),
        ("record_contract_sha256", canonical_sha256(reference["record_contract"])),
        ("evaluation_contract_sha256", canonical_sha256(reference["evaluation_contract"])),
        ("reference_manifest_sha256", reference["manifest_sha256"]),
    ):
        if document.get(field) != expected:
            raise OfficialReferencePredictionError(f"official prediction {field} mismatch")
    source_artifacts = document.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping) or set(source_artifacts) != {
        "particle_run_manifest", "particle_hits"
    }:
        raise OfficialReferencePredictionError("official prediction source artifacts are incomplete")
    for label, entry in source_artifacts.items():
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str) or not _hash(entry.get("sha256")):
            raise OfficialReferencePredictionError(f"official prediction {label} source is invalid")
        path = Path(entry["path"])
        if not path.is_file() or sha256_file(path) != str(entry["sha256"]).lower():
            raise OfficialReferencePredictionError(f"official prediction {label} hash mismatch")
    run, hits_path, hits_hash = _run_manifest(Path(source_artifacts["particle_run_manifest"]["path"]))
    if run.get("particle_run_id") != document.get("particle_run_id"):
        raise OfficialReferencePredictionError("official prediction particle_run_id mismatch")
    if str(hits_path) != str(Path(source_artifacts["particle_hits"]["path"])) or hits_hash != source_artifacts["particle_hits"]["sha256"]:
        raise OfficialReferencePredictionError("official prediction particle_hits binding mismatch")
    records = document.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)) or not records:
        raise OfficialReferencePredictionError("official prediction records are missing")
    records_hash = canonical_sha256({"records": list(records)})
    if document.get("records_sha256") != records_hash:
        raise OfficialReferencePredictionError("official prediction records_sha256 is invalid")
    manifest_path = reference.get("manifest_path")
    if not isinstance(manifest_path, str) or not Path(manifest_path).is_file():
        raise OfficialReferencePredictionError("official prediction cannot revalidate an in-memory reference manifest")
    derivation = document.get("derivation")
    if not isinstance(derivation, Mapping):
        raise OfficialReferencePredictionError("official prediction derivation is missing")
    recomputed = build_prediction(
        source_artifacts["particle_run_manifest"]["path"],
        manifest_path,
        exposure_hours=derivation.get("exposure_hours"),
    )
    if recomputed != dict(document):
        raise OfficialReferencePredictionError("official prediction differs from frozen source recomputation")
    return records_hash


def _selftest() -> None:
    try:
        build_prediction("missing.json", "missing-reference.json")
    except OfficialReferencePredictionError:
        pass
    else:
        raise AssertionError("missing frozen prediction inputs were accepted")


def selftest() -> bool:
    _selftest()
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export frozen parent-face predictions")
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--exposure-hours", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        selftest()
        print("step30_official_reference_prediction selftest: PASS")
        return 0
    if args.run_manifest is None or args.reference_manifest is None:
        parser.error("--run-manifest and --reference-manifest are required")
    try:
        result = build_prediction(
            args.run_manifest,
            args.reference_manifest,
            exposure_hours=args.exposure_hours,
        )
    except (OfficialReferencePredictionError, OSError, ValueError, KeyError) as exc:
        result = {
            "version": VERSION,
            "status": "blocked",
            "reason": str(exc) or type(exc).__name__,
            "official_reference_gate": False,
        }
        result["document_sha256"] = canonical_sha256(result)
    safe_json_dump(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "synthetic" else 3


if __name__ == "__main__":
    raise SystemExit(main())
