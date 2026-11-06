"""Acceptance for independently replayed Step 30 measured evidence.

A validation report can be inspected diagnostically, but it cannot grant an
acceptance gate. Official measured acceptance is available only through the
full-evidence replay API, which calls :func:`validate_measured` again.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from step30_independent_erosion_protocol import (
    validate_prediction_commitment,
    validate_protocol,
)
from step30_independent_erosion_truth import (
    build_independent_erosion_truth,
    validate_independent_erosion_truth,
    validate_reveal_receipt,
)
from step30_measurement_parent_face_mapping import load_topology_v3
from step30_measured_validation import validate_measured
from step30_validation_common import (
    ValidationError,
    canonical_sha256,
    jsonable,
    safe_json_dump,
    safe_json_load,
    sha256_file,
)

VERSION = "step30-measurement-acceptance-v2"
INDEPENDENT_EROSION_ACCEPTANCE_VERSION = "step30-independent-erosion-acceptance-v1"
TRUTH_BUILD_CONTRACT_VERSION = "step30-independent-erosion-truth-build-v1"
OUTCOMES = frozenset({"passed", "failed", "blocked"})


class MeasurementAcceptanceError(ValidationError):
    """Raised internally when acceptance evidence is malformed."""


def _hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value)


def _load(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            raise MeasurementAcceptanceError(f"validation report does not exist: {path}")
        document = safe_json_load(path)
        if isinstance(document, Mapping):
            return document
    raise MeasurementAcceptanceError("validation report must be a JSON object")


def _load_artifact(value: Any, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            raise MeasurementAcceptanceError(f"{name} does not exist: {path}")
        try:
            document = safe_json_load(path)
        except (OSError, ValidationError) as exc:
            raise MeasurementAcceptanceError(f"cannot load {name}: {path}") from exc
        if isinstance(document, Mapping):
            return document
    raise MeasurementAcceptanceError(f"{name} must be a JSON object or mapping")


def _sha256(value: Any, name: str) -> str:
    if not _hash(value):
        raise MeasurementAcceptanceError(f"{name} must be a SHA-256 digest")
    return str(value).lower()


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise MeasurementAcceptanceError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementAcceptanceError(f"{name} must be finite and positive") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise MeasurementAcceptanceError(f"{name} must be finite and positive")
    return result


def build_truth_build_contract(
    *,
    topology_path: str | Path,
    topology_v3_sha256: Any,
    mesh_signature: Any,
    artifact_set_id: Any,
    registration_options: Mapping[str, Any],
    maximum_projection_distance_mm: Any,
) -> dict[str, Any]:
    """Canonical pre-reveal binding for every truth-building parameter."""
    if not isinstance(registration_options, Mapping):
        raise MeasurementAcceptanceError("registration_options must be an object")
    contract = {
        "version": TRUTH_BUILD_CONTRACT_VERSION,
        "topology_path": str(Path(topology_path).resolve()),
        "topology_v3_sha256": _sha256(topology_v3_sha256, "topology_v3_sha256"),
        "mesh_signature": _sha256(mesh_signature, "mesh_signature"),
        "artifact_set_id": _sha256(artifact_set_id, "artifact_set_id"),
        "registration_options": jsonable(dict(registration_options)),
        "maximum_projection_distance_mm": _finite_positive(
            maximum_projection_distance_mm, "maximum_projection_distance_mm"
        ),
    }
    contract["truth_build_contract_sha256"] = recompute_truth_build_contract_sha256(
        contract
    )
    return contract


def recompute_truth_build_contract_sha256(contract: Mapping[str, Any]) -> str:
    unsigned = dict(jsonable(contract))
    unsigned.pop("truth_build_contract_sha256", None)
    return canonical_sha256(unsigned)


def _truth_build_contract_hash_candidates(contract: Mapping[str, Any]) -> set[str]:
    normalized = {
        key: contract[key]
        for key in (
            "topology_path",
            "topology_v3_sha256",
            "mesh_signature",
            "artifact_set_id",
            "registration_options",
            "maximum_projection_distance_mm",
        )
    }
    candidates = {
        recompute_truth_build_contract_sha256(contract),
        canonical_sha256(normalized),
    }
    for topology_key in (
        "topology_v3_sha256",
        "topology_hash",
        "topology_sha256",
    ):
        for projection_key in (
            "maximum_projection_distance_mm",
            "max_projection_distance_mm",
            "max_projection",
        ):
            candidates.add(
                canonical_sha256(
                    {
                        "topology_path": contract["topology_path"],
                        topology_key: contract["topology_v3_sha256"],
                        "mesh_signature": contract["mesh_signature"],
                        "artifact_set_id": contract["artifact_set_id"],
                        "registration_options": contract["registration_options"],
                        projection_key: contract["maximum_projection_distance_mm"],
                    }
                )
            )
    return candidates


def _resolve_truth_build_config(
    *,
    build_config: Mapping[str, Any] | None,
    topology_path: str | Path | None,
    topology_v3_sha256: Any,
    mesh_signature: Any,
    artifact_set_id: Any,
    registration_options: Mapping[str, Any] | None,
    maximum_projection_distance_mm: Any,
) -> dict[str, Any]:
    if build_config is not None and not isinstance(build_config, Mapping):
        raise MeasurementAcceptanceError("truth build config must be an object")
    values = dict(build_config or {})
    aliases = {
        "topology_hash": "topology_v3_sha256",
        "topology_sha256": "topology_v3_sha256",
        "max_projection": "maximum_projection_distance_mm",
        "max_projection_distance_mm": "maximum_projection_distance_mm",
    }
    canonical_fields = {
        "topology_path",
        "topology_v3_sha256",
        "mesh_signature",
        "artifact_set_id",
        "registration_options",
        "maximum_projection_distance_mm",
    }
    allowed_fields = canonical_fields | set(aliases) | {
        "version",
        "truth_build_contract_sha256",
    }
    unknown = sorted(set(values) - allowed_fields)
    if unknown:
        raise MeasurementAcceptanceError(
            f"truth build config fields are invalid: {unknown}"
        )
    if values.get("version", TRUTH_BUILD_CONTRACT_VERSION) != TRUTH_BUILD_CONTRACT_VERSION:
        raise MeasurementAcceptanceError("truth build config version mismatch")
    declared_hash = values.get("truth_build_contract_sha256")
    for alias, field in aliases.items():
        if alias in values:
            if field in values and jsonable(values[field]) != jsonable(values[alias]):
                raise MeasurementAcceptanceError(
                    f"truth build config aliases conflict on {field}"
                )
            values[field] = values[alias]
    explicit = {
        "topology_path": topology_path,
        "topology_v3_sha256": topology_v3_sha256,
        "mesh_signature": mesh_signature,
        "artifact_set_id": artifact_set_id,
        "registration_options": registration_options,
        "maximum_projection_distance_mm": maximum_projection_distance_mm,
    }
    for name, value in explicit.items():
        if value is not None:
            if name in values and jsonable(values[name]) != jsonable(value):
                raise MeasurementAcceptanceError(f"truth build config conflicts on {name}")
            values[name] = value
    required = (
        "topology_path",
        "topology_v3_sha256",
        "mesh_signature",
        "artifact_set_id",
        "registration_options",
        "maximum_projection_distance_mm",
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise MeasurementAcceptanceError(
            f"truth build config is incomplete: missing {', '.join(missing)}"
        )
    contract = build_truth_build_contract(
        **{name: values[name] for name in required}
    )
    if declared_hash is not None:
        if not _hash(declared_hash) or str(declared_hash).lower() not in _truth_build_contract_hash_candidates(contract):
            raise MeasurementAcceptanceError(
                "truth build config hash does not match its frozen parameters"
            )
    return contract


def _require_truth_sources(protocol: Mapping[str, Any]) -> None:
    assets = protocol.get("assets")
    if not isinstance(assets, Sequence) or isinstance(assets, (str, bytes, bytearray)):
        raise MeasurementAcceptanceError("protocol assets are missing")
    if any(
        not isinstance(asset, Mapping)
        or not isinstance(asset.get("truth_source"), Mapping)
        for asset in assets
    ):
        raise MeasurementAcceptanceError(
            "formal protocol requires baseline source and truth_source for every asset"
        )


def _validate_commitment_topology(
    commitment: Mapping[str, Any], truth: Mapping[str, Any], topology: Mapping[str, Any]
) -> None:
    binding = commitment.get("topology_binding")
    if not isinstance(binding, Mapping):
        raise MeasurementAcceptanceError("commitment topology_binding is required")
    required = {
        "topology_v3_sha256",
        "mesh_signature",
        "artifact_set_id",
        "n_parent_faces",
    }
    if set(binding) != required:
        raise MeasurementAcceptanceError("commitment topology_binding fields are invalid")
    expected = {
        "topology_v3_sha256": str(truth["topology_v3_sha256"]).lower(),
        "mesh_signature": str(truth["mesh_signature"]).lower(),
        "artifact_set_id": str(truth["artifact_set_id"]).lower(),
        "n_parent_faces": int(truth["n_parent_faces"]),
    }
    if dict(binding) != expected:
        raise MeasurementAcceptanceError(
            "commitment topology_binding differs from truth topology identity"
        )
    if int(binding["n_parent_faces"]) != int(topology["n_parent_faces"]):
        raise MeasurementAcceptanceError("topology parent-face count differs from binding")
    if int(truth["n_parent_faces"]) != int(topology["n_parent_faces"]):
        raise MeasurementAcceptanceError("truth n_parent_faces differs from topology")


def _aligned_rows(
    commitment: Mapping[str, Any], truth: Mapping[str, Any], topology: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, int]]]:
    prediction_rows = {
        (str(row["record_id"]), int(row["parent_face_id"])): row
        for row in commitment["predictions"]
    }
    truth_rows = {
        (str(row["record_id"]), int(row["parent_face_id"])): row
        for row in truth["rows"]
    }
    if set(prediction_rows) != set(truth_rows):
        raise MeasurementAcceptanceError(
            "prediction commitment and truth parent-face identities differ"
        )
    n_faces = int(topology["n_parent_faces"])
    topology_areas = np.asarray(topology["face_area_m2"], dtype=np.float64) * 1.0e6
    identities = sorted(truth_rows)
    predicted: list[float] = []
    observed: list[float] = []
    areas: list[float] = []
    for identity in identities:
        _, face_id = identity
        if face_id < 0 or face_id >= n_faces:
            raise MeasurementAcceptanceError("parent-face identity is outside topology")
        prediction = prediction_rows[identity]
        observation = truth_rows[identity]
        expected_area = float(topology_areas[face_id])
        prediction_area = float(prediction["parent_face_area_mm2"])
        truth_area = float(observation["area_mm2"])
        if not math.isclose(
            prediction_area, expected_area, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise MeasurementAcceptanceError("prediction parent-face area differs from topology")
        if not math.isclose(
            truth_area, expected_area, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise MeasurementAcceptanceError("truth parent-face area differs from topology")
        predicted.append(float(prediction["prediction_value"]))
        observed.append(float(observation["wear_depth_mm"]))
        areas.append(expected_area)
    if not identities:
        raise MeasurementAcceptanceError("explicit measurement coverage is empty")
    return (
        np.asarray(predicted, dtype=np.float64),
        np.asarray(observed, dtype=np.float64),
        np.asarray(areas, dtype=np.float64),
        identities,
    )


def _relative_error(delta: float, observed: float, name: str) -> float:
    if observed == 0.0:
        raise MeasurementAcceptanceError(
            f"zero observed baseline is undefined for relative error: {name}"
        )
    return float(abs(delta) / abs(observed))


def _spatial_metrics(
    predicted: np.ndarray, observed: np.ndarray, areas: np.ndarray
) -> dict[str, Any]:
    if not (
        np.isfinite(predicted).all()
        and np.isfinite(observed).all()
        and np.isfinite(areas).all()
    ):
        raise MeasurementAcceptanceError("formal metric arrays must be finite")
    if np.any(predicted < 0.0) or np.any(observed < 0.0) or np.any(areas <= 0.0):
        raise MeasurementAcceptanceError("formal metric arrays have invalid values")
    if np.any(observed == 0.0):
        raise MeasurementAcceptanceError(
            "zero observed baseline is undefined for relative error"
        )
    area_total = float(np.sum(areas))
    predicted_wear = float(np.sum(predicted * areas) / area_total)
    observed_wear = float(np.sum(observed * areas) / area_total)
    predicted_volume = float(np.sum(predicted * areas))
    observed_volume = float(np.sum(observed * areas))
    predicted_max = float(np.max(predicted))
    observed_max = float(np.max(observed))

    def weighted_percentile(values: np.ndarray, percentile: float) -> float:
        order = np.argsort(values, kind="mergesort")
        cumulative = np.cumsum(areas[order])
        return float(
            np.interp(percentile / 100.0 * area_total, cumulative, values[order])
        )

    predicted_threshold = weighted_percentile(predicted, 90.0)
    observed_threshold = weighted_percentile(observed, 90.0)
    predicted_hot = predicted >= predicted_threshold
    observed_hot = observed >= observed_threshold
    predicted_hot_area = float(np.sum(areas[predicted_hot]))
    observed_hot_area = float(np.sum(areas[observed_hot]))
    intersection = float(np.sum(areas[predicted_hot & observed_hot]))
    union = float(np.sum(areas[predicted_hot | observed_hot]))
    iou = float(intersection / union) if union else 1.0
    return {
        "wear_depth_mm": {
            "metric": "wear_depth_mm",
            "units": "mm",
            "aggregation": "area_weighted_mean",
            "predicted_total": predicted_wear,
            "observed_total": observed_wear,
            "total_relative_error": _relative_error(
                predicted_wear - observed_wear, observed_wear, "wear_depth_mm total"
            ),
        },
        "total_volume_mm3": {
            "metric": "total_volume_mm3",
            "units": "mm3",
            "aggregation": "sum(depth_mm * parent_face_area_mm2)",
            "predicted_total": predicted_volume,
            "observed_total": observed_volume,
            "total_relative_error": _relative_error(
                predicted_volume - observed_volume,
                observed_volume,
                "total_volume_mm3 total",
            ),
        },
        "max_depth_mm": {
            "metric": "max_depth_mm",
            "units": "mm",
            "predicted": predicted_max,
            "observed": observed_max,
            "relative_error": _relative_error(
                predicted_max - observed_max, observed_max, "max_depth_mm"
            ),
        },
        "hotspot_area_mm2": {
            "metric": "hotspot_area_mm2",
            "units": "mm2",
            "percentile": 90.0,
            "predicted": predicted_hot_area,
            "observed": observed_hot_area,
            "relative_error": _relative_error(
                predicted_hot_area - observed_hot_area,
                observed_hot_area,
                "90th percentile hotspot area",
            ),
            "predicted_threshold": predicted_threshold,
            "observed_threshold": observed_threshold,
        },
        "hotspot_iou_error": {
            "metric": "hotspot_iou_error",
            "units": "dimensionless",
            "percentile": 90.0,
            "iou": iou,
            "error": float(1.0 - iou),
        },
    }


def _seal_independent(result: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(result)
    unsigned.pop("acceptance_sha256", None)
    result["acceptance_sha256"] = canonical_sha256(unsigned)
    return result


def _independent_blocked(
    reason: str, *, checks: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    return _seal_independent(
        {
            "version": INDEPENDENT_EROSION_ACCEPTANCE_VERSION,
            "status": "blocked",
            "evaluation_status": "blocked",
            "validation_status": "blocked",
            "blocked": True,
            "accepted": False,
            "official_accuracy_gate": False,
            "official_reference_gate": False,
            "diagnostic_only": False,
            "checks": dict(checks or {}),
            "metrics": {},
            "reasons": [reason],
            "official_accuracy_note": "Independent erosion acceptance requires replay from all pinned raw evidence.",
        }
    )


def _load_truth_records(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, (str, Path)):
        value = safe_json_load(Path(value))
    if isinstance(value, Mapping):
        value = value.get("records")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise MeasurementAcceptanceError("truth_records must be an array")
    return list(value)


def _report_hash(report: Mapping[str, Any]) -> str:
    stored = report.get("validation_sha256")
    payload = dict(report)
    payload.pop("validation_sha256", None)
    if not _hash(stored) or canonical_sha256(payload) != str(stored).lower():
        raise MeasurementAcceptanceError("validation report canonical hash is missing or invalid")
    return str(stored).lower()


def _seal(result: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(result)
    unsigned.pop("acceptance_sha256", None)
    result["acceptance_sha256"] = canonical_sha256(unsigned)
    return result


def _blocked(reason: str, *, evidence_status: str | None = None, diagnostic_only: bool = False) -> dict[str, Any]:
    return _seal({
        "version": VERSION,
        "status": "blocked",
        "evaluation_status": "blocked",
        "validation_status": "blocked",
        "evidence_status": evidence_status,
        "blocked": True,
        "accepted": False,
        "official_accuracy_gate": False,
        "official_reference_gate": False,
        "diagnostic_only": diagnostic_only,
        "checks": {},
        "reasons": [reason],
        "official_accuracy_note": "Official measured acceptance requires replay from all raw evidence inputs.",
    })


def evaluate_acceptance(validation_report: Any) -> dict[str, Any]:
    """Inspect a report only; report-only evaluation can never grant a gate."""
    try:
        report = _load(validation_report)
        digest = _report_hash(report)
        status = str(report.get("evaluation_status", report.get("status", "blocked")))
        evidence_status = report.get("evidence_status")
        if evidence_status is None and report.get("status") == "measured":
            evidence_status = "measured"
        return _seal({
            "version": VERSION,
            "status": "blocked",
            "evaluation_status": "blocked",
            "validation_status": status,
            "evidence_status": evidence_status,
            "diagnostic_validation_status": status,
            "blocked": True,
            "accepted": False,
            "official_accuracy_gate": False,
            "official_reference_gate": False,
            "diagnostic_only": True,
            "checks": {
                "validation_report_hash_valid": True,
                "full_evidence_replay_performed": False,
            },
            "validation_sha256": digest,
            "reasons": ["report_only_diagnostic_cannot_grant_official_gate"],
            "official_accuracy_note": "The report hash is diagnostic integrity only, not independent evidence replay.",
        })
    except (MeasurementAcceptanceError, ValidationError, OSError, TypeError, ValueError, KeyError) as exc:
        return _blocked(str(exc) or type(exc).__name__, diagnostic_only=True)


def replay_independent_erosion_acceptance(
    protocol: Any,
    prediction_commitment: Any,
    truth: Any,
    reveal_receipt: Any,
    *,
    trusted_commitment_sha256: Any = None,
    trusted_protocol_sha256: Any = None,
    truth_records: Any = None,
    build_config: Mapping[str, Any] | None = None,
    truth_build_config: Mapping[str, Any] | None = None,
    frozen_build_config: Mapping[str, Any] | None = None,
    topology_path: str | Path | None = None,
    topology_v3_sha256: Any = None,
    topology_hash: Any = None,
    mesh_signature: Any = None,
    artifact_set_id: Any = None,
    registration_options: Mapping[str, Any] | None = None,
    maximum_projection_distance_mm: Any = None,
    max_projection_distance_mm: Any = None,
) -> dict[str, Any]:
    """Replay pinned independent erosion evidence and classify it.

    The trusted commitment digest is an external pre-reveal pin. Raw truth is
    rebuilt with the frozen processing contract before the revealed truth hash
    can be used for numerical scoring.
    """
    checks: dict[str, Any] = {
        "trusted_commitment_pin_present": trusted_commitment_sha256 is not None,
        "trusted_protocol_pin_present": trusted_protocol_sha256 is not None,
        "protocol_hash_valid": False,
        "commitment_hash_valid": False,
        "truth_hash_valid": False,
        "receipt_hash_valid": False,
        "truth_rebuilt_from_raw_sources": False,
    }
    try:
        if not _hash(trusted_commitment_sha256):
            raise MeasurementAcceptanceError(
                "trusted_commitment_sha256 is required and must be a SHA-256 digest"
            )
        if trusted_protocol_sha256 is not None and not _hash(trusted_protocol_sha256):
            raise MeasurementAcceptanceError(
                "trusted_protocol_sha256 must be a SHA-256 digest"
            )

        frozen_protocol = validate_protocol(_load_artifact(protocol, "protocol"))
        checks["protocol_hash_valid"] = True
        protocol_digest = str(frozen_protocol["protocol_sha256"]).lower()
        if trusted_protocol_sha256 is not None and str(trusted_protocol_sha256).lower() != protocol_digest:
            raise MeasurementAcceptanceError(
                "trusted_protocol_sha256 does not match protocol"
            )
        _require_truth_sources(frozen_protocol)

        frozen_commitment = validate_prediction_commitment(
            _load_artifact(prediction_commitment, "prediction commitment"),
            frozen_protocol,
        )
        checks["commitment_hash_valid"] = True
        commitment_digest = str(frozen_commitment["commitment_sha256"]).lower()
        if str(trusted_commitment_sha256).lower() != commitment_digest:
            raise MeasurementAcceptanceError(
                "trusted_commitment_sha256 does not match commitment digest"
            )

        frozen_truth = validate_independent_erosion_truth(
            _load_artifact(truth, "truth"), protocol=frozen_protocol, verify_sources=False
        )
        checks["truth_hash_valid"] = True
        frozen_receipt = validate_reveal_receipt(
            _load_artifact(reveal_receipt, "reveal receipt")
        )
        checks["receipt_hash_valid"] = True
        if frozen_receipt.get("evaluation_status") != "ready":
            raise MeasurementAcceptanceError("reveal receipt is not ready")
        if (
            frozen_receipt.get("protocol_sha256") != protocol_digest
            or frozen_receipt.get("prediction_commitment_sha256") != commitment_digest
            or frozen_receipt.get("truth_sha256") != frozen_truth["truth_sha256"]
        ):
            raise MeasurementAcceptanceError(
                "reveal receipt does not point to the same protocol, commitment, and truth"
            )

        provided_configs = [
            value
            for value in (build_config, truth_build_config, frozen_build_config)
            if value is not None
        ]
        if len(provided_configs) > 1:
            raise MeasurementAcceptanceError(
                "provide only one frozen truth build config object"
            )
        if (
            topology_v3_sha256 is not None
            and topology_hash is not None
            and str(topology_v3_sha256).lower() != str(topology_hash).lower()
        ):
            raise MeasurementAcceptanceError(
                "topology_v3_sha256 conflicts with topology_hash"
            )
        if (
            maximum_projection_distance_mm is not None
            and max_projection_distance_mm is not None
            and float(maximum_projection_distance_mm)
            != float(max_projection_distance_mm)
        ):
            raise MeasurementAcceptanceError(
                "maximum projection distance parameters conflict"
            )
        config = _resolve_truth_build_config(
            build_config=provided_configs[0] if provided_configs else None,
            topology_path=topology_path,
            topology_v3_sha256=(
                topology_v3_sha256
                if topology_v3_sha256 is not None
                else topology_hash
            ),
            mesh_signature=mesh_signature,
            artifact_set_id=artifact_set_id,
            registration_options=registration_options,
            maximum_projection_distance_mm=(
                maximum_projection_distance_mm
                if maximum_projection_distance_mm is not None
                else max_projection_distance_mm
            ),
        )
        descriptor = frozen_commitment.get("model_descriptor")
        if not isinstance(descriptor, Mapping):
            raise MeasurementAcceptanceError("commitment model_descriptor is missing")
        pinned_contract_hash = descriptor.get("truth_build_contract_sha256")
        if not _hash(pinned_contract_hash):
            raise MeasurementAcceptanceError(
                "commitment does not contain truth_build_contract_sha256 pin"
            )
        if str(pinned_contract_hash).lower() not in _truth_build_contract_hash_candidates(config):
            raise MeasurementAcceptanceError(
                "truth build config differs from pre-reveal commitment contract"
            )

        topology_file = Path(config["topology_path"]).resolve()
        topology = load_topology_v3(
            topology_file,
            expected_mesh_signature=config["mesh_signature"],
        )
        if sha256_file(topology_file) != config["topology_v3_sha256"]:
            raise MeasurementAcceptanceError("truth build topology file hash mismatch")
        if config["artifact_set_id"] != frozen_truth["artifact_set_id"]:
            raise MeasurementAcceptanceError("truth build artifact_set_id differs from truth")
        if config["topology_v3_sha256"] != frozen_truth["topology_v3_sha256"]:
            raise MeasurementAcceptanceError("truth build topology hash differs from truth")
        if config["mesh_signature"] != frozen_truth["mesh_signature"]:
            raise MeasurementAcceptanceError("truth build mesh signature differs from truth")
        if truth_records is None:
            raise MeasurementAcceptanceError("truth_records are required for raw truth replay")
        rebuilt_truth = build_independent_erosion_truth(
            frozen_protocol,
            _load_truth_records(truth_records),
            topology_path=topology_file,
            topology_v3_sha256=config["topology_v3_sha256"],
            mesh_signature=config["mesh_signature"],
            artifact_set_id=config["artifact_set_id"],
            registration_options=config["registration_options"],
            maximum_projection_distance_mm=config["maximum_projection_distance_mm"],
        )
        checks["truth_rebuilt_from_raw_sources"] = True
        if rebuilt_truth["truth_sha256"] != frozen_truth["truth_sha256"]:
            raise MeasurementAcceptanceError(
                "revealed truth_sha256 differs from reconstruction from raw sources"
            )

        _validate_commitment_topology(frozen_commitment, frozen_truth, topology)
        predicted, observed, areas, identities = _aligned_rows(
            frozen_commitment, frozen_truth, topology
        )
        metrics = _spatial_metrics(predicted, observed, areas)
        thresholds = frozen_protocol["metric_contract"]["thresholds"]
        relative_errors = {
            "wear_depth_mm": metrics["wear_depth_mm"]["total_relative_error"],
            "total_volume_mm3": metrics["total_volume_mm3"]["total_relative_error"],
            "max_depth_mm": metrics["max_depth_mm"]["relative_error"],
            "hotspot_area_mm2": metrics["hotspot_area_mm2"]["relative_error"],
            "hotspot_iou_error": metrics["hotspot_iou_error"]["error"],
        }
        numeric_passed = all(
            float(relative_errors[name]) <= float(thresholds[name])
            for name in relative_errors
        )
        status = "passed" if numeric_passed else "failed"
        return _seal_independent(
            {
                "version": INDEPENDENT_EROSION_ACCEPTANCE_VERSION,
                "status": status,
                "evaluation_status": status,
                "validation_status": status,
                "evidence_status": "measured",
                "blocked": False,
                "accepted": numeric_passed,
                "independent_accuracy_gate": numeric_passed,
                # Independent physical truth is diagnostic until the
                # organizer package and trusted organizer pin are present.
                "official_accuracy_gate": False,
                "official_reference_gate": False,
                "production_acceptance_status": "blocked",
                "production_acceptance_reason": (
                    "organizer reference package and trusted organizer SHA-256 pin are absent"
                ),
                "diagnostic_only": False,
                "checks": {
                    **checks,
                    "receipt_bindings_valid": True,
                    "topology_binding_valid": True,
                    "parent_face_identity_and_area_valid": True,
                    "metric_contract_valid": True,
                    "numeric_thresholds_passed": numeric_passed,
                },
                "artifact_hashes": {
                    "protocol_sha256": protocol_digest,
                    "commitment_sha256": commitment_digest,
                    "truth_sha256": frozen_truth["truth_sha256"],
                    "receipt_sha256": frozen_receipt["receipt_sha256"],
                },
                "n_explicit_measurements": len(identities),
                "metrics": metrics,
                "relative_errors": relative_errors,
                "reasons": [] if numeric_passed else [
                    "one or more preregistered metric thresholds exceeded"
                ],
                "official_accuracy_note": "Independent measured erosion acceptance is not an official reference.",
            }
        )
    except (MeasurementAcceptanceError, ValidationError, OSError, TypeError, ValueError, KeyError) as exc:
        return _independent_blocked(str(exc) or type(exc).__name__, checks=checks)


def replay_acceptance(
    prediction_records: Any,
    observation_records: Any,
    source_manifest: Any,
    split_manifest: Any,
    quality_json: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Replay all evidence with ``validate_measured`` and classify its outcome."""
    validation = validate_measured(
        prediction_records,
        observation_records,
        source_manifest,
        split_manifest,
        quality_json,
        **kwargs,
    )
    try:
        digest = _report_hash(validation)
        status = str(validation.get("evaluation_status", validation.get("status", "blocked")))
        if status not in OUTCOMES:
            raise MeasurementAcceptanceError(f"validation outcome is invalid: {status}")
        evidence_status = validation.get("evidence_status")
        contract_valid = bool(
            status in {"passed", "failed"}
            and evidence_status == "measured"
            and validation.get("blocked") is False
        )
        if status == "blocked" or not contract_valid:
            status = "blocked"
        passed = status == "passed"
        reasons = list(validation.get("reasons", []))
        if status == "blocked" and not reasons:
            reasons = ["validation evidence contract is blocked"]
        result = {
            "version": VERSION,
            "status": status,
            "evaluation_status": status,
            "validation_status": status,
            "evidence_status": evidence_status,
            "blocked": status == "blocked",
            "accepted": passed,
            "official_accuracy_gate": passed,
            "official_reference_gate": False,
            "diagnostic_only": False,
            "checks": {
                "full_evidence_replay_performed": True,
                "validation_report_hash_valid": True,
                "measured_evidence_contract_valid": contract_valid,
                "pre_registered_numeric_thresholds_passed": passed,
            },
            "validation_sha256": digest,
            "reasons": reasons,
            "official_accuracy_note": "Measured acceptance is replayed evidence and is never an organizer reference.",
            "validation_report": validation,
        }
        return _seal(result)
    except (MeasurementAcceptanceError, ValidationError, OSError, TypeError, ValueError, KeyError) as exc:
        result = _blocked(str(exc) or type(exc).__name__, evidence_status=validation.get("evidence_status"))
        result["validation_report"] = validation
        return _seal(result)


def accept_measurements(
    prediction_records: Any,
    observation_records: Any,
    source_manifest: Any,
    split_manifest: Any,
    quality_json: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compatibility name for full evidence replay acceptance."""
    return replay_acceptance(
        prediction_records,
        observation_records,
        source_manifest,
        split_manifest,
        quality_json,
        **kwargs,
    )


# Report-only legacy calls remain callable but are intentionally diagnostic.
accept = evaluate_acceptance
measurement_acceptance = accept_measurements
run_acceptance = accept_measurements
replay_evidence = replay_acceptance


def save_acceptance_report(report: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    return safe_json_dump(report, path)


def selftest() -> bool:
    report = {
        "version": "selftest",
        "status": "passed",
        "evaluation_status": "passed",
        "evidence_status": "measured",
        "blocked": False,
        "official_accuracy_gate": True,
        "official_reference_gate": False,
    }
    report["validation_sha256"] = canonical_sha256(report)
    diagnostic = evaluate_acceptance(report)
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["official_accuracy_gate"] is False
    assert diagnostic["official_reference_gate"] is False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step30 independently replayed measurement acceptance")
    parser.add_argument("--validation-report", type=Path)
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
        print("step30_measurement_acceptance selftest: PASS")
        return 0
    if args.validation_report is not None:
        report = evaluate_acceptance(args.validation_report)
    else:
        required = (args.predictions, args.observations, args.source_manifest, args.split_manifest, args.quality)
        report = _blocked("missing required acceptance input") if any(value is None for value in required) else replay_acceptance(
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
        save_acceptance_report(report, args.output)
    print(json.dumps(jsonable(report), ensure_ascii=False, indent=2))
    return 0 if report.get("official_accuracy_gate") else 3


if __name__ == "__main__":
    raise SystemExit(main())
