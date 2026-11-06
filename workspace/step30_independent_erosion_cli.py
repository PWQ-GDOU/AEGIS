"""Independent measured erosion v2 delivery CLI.

This module is a strict delivery layer over the current v1 protocol, truth, and
acceptance APIs. It freezes scan-pair provenance and evaluation coverage while
leaving the validated numerical replay implementation unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from step30_external_commitment_pin import (
    ExternalCommitmentPinError,
    parse_utc,
    utc_now,
    verify_pin_envelope,
)
from step30_independent_erosion_protocol import (
    build_prediction_commitment,
    build_protocol,
    validate_prediction_commitment,
    validate_protocol,
)
from step30_independent_erosion_truth import (
    build_independent_erosion_truth,
    build_reveal_receipt,
    validate_independent_erosion_truth,
    validate_reveal_receipt,
)
from step30_measurement_acceptance import (
    build_truth_build_contract,
    replay_independent_erosion_acceptance,
)
from step30_measurement_parent_face_mapping import load_topology_v3
from step30_validation_common import (
    ValidationError,
    canonical_sha256,
    jsonable,
    safe_json_dump,
    safe_json_load,
    sha256_file,
)


PROTOCOL_V2_VERSION = "step30-independent-erosion-protocol-v2"
SCAN_MANIFEST_VERSION = "step30-scan-pair-manifest-v1"
EVALUATION_MASK_VERSION = "step30-independent-erosion-evaluation-mask-v1"
DELIVERY_ACCEPTANCE_VERSION = "step30-independent-erosion-delivery-acceptance-v2"
MINIMUM_INDEPENDENT_GROUPS = 3
RECOMMENDED_FORMAL_GROUPS = 5
_HEX = frozenset("0123456789abcdef")
_PROTOCOL_FIELDS = frozenset(
    {
        "version",
        "schema_version",
        "status",
        "immutable",
        "formal",
        "study_phase",
        "minimum_independent_groups",
        "recommended_formal_groups",
        "formal_eligibility_reason",
        "created_at_utc",
        "scan_manifest",
        "scan_manifest_sha256",
        "evaluation_mask",
        "evaluation_mask_sha256",
        "evaluation_mask_file_sha256",
        "truth_build_contract",
        "core_protocol",
        "protocol_sha256",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "version",
        "schema_version",
        "status",
        "formal_requested",
        "created_at_utc",
        "minimum_independent_groups",
        "recommended_formal_groups",
        "blocked_reason",
        "scan_pairs",
    }
)
_PAIR_FIELDS = frozenset(
    {
        "record_id",
        "asset_id",
        "component_id",
        "campaign_id",
        "service_episode",
        "baseline_scan",
        "worn_scan",
    }
)
_EPISODE_FIELDS = frozenset({"service_episode_id", "started_at", "ended_at"})
_SCAN_FIELDS = frozenset(
    {
        "scan_id",
        "acquired_at",
        "instrument_serial",
        "calibration_certificate",
        "coordinate_frame",
        "fiducial_id",
        "fixture_id",
        "coordinate_units",
        "source",
    }
)
_SOURCE_FIELDS = frozenset({"path", "sha256", "bytes"})
_MASK_FIELDS = frozenset(
    {
        "version",
        "status",
        "immutable",
        "topology_v3_sha256",
        "episodes",
        "mask_sha256",
    }
)
_MASK_EPISODE_FIELDS = frozenset(
    {
        "record_id",
        "service_episode_id",
        "parent_face_ids",
        "minimum_area_coverage_fraction",
        "minimum_valid_count",
    }
)


class IndependentErosionDeliveryError(ValidationError):
    """Raised when v2 delivery evidence is incomplete or inconsistent."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IndependentErosionDeliveryError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise IndependentErosionDeliveryError(f"{name} must be an array")
    return list(value)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IndependentErosionDeliveryError(f"{name} must be a non-empty string")
    return value.strip()


def _sha256(value: Any, name: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value.lower())
    ):
        raise IndependentErosionDeliveryError(f"{name} must be a SHA-256 digest")
    return value.lower()


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise IndependentErosionDeliveryError(f"{name} must be an integer")
    if value < 0 or (positive and value <= 0):
        qualifier = "positive" if positive else "non-negative"
        raise IndependentErosionDeliveryError(f"{name} must be {qualifier}")
    return value


def _utc(value: Any, name: str) -> datetime:
    try:
        return parse_utc(value, name)
    except ExternalCommitmentPinError as exc:
        raise IndependentErosionDeliveryError(str(exc)) from exc


def _load_json(value: Mapping[str, Any] | str | Path, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    if not path.is_file():
        raise IndependentErosionDeliveryError(f"{name} does not exist: {path}")
    try:
        document = safe_json_load(path)
    except (OSError, ValidationError) as exc:
        raise IndependentErosionDeliveryError(f"cannot load {name}: {path}") from exc
    if not isinstance(document, Mapping):
        raise IndependentErosionDeliveryError(f"{name} must be a JSON object")
    return dict(document)


def _source_descriptor(
    value: Any,
    name: str,
    *,
    source_base: str | Path | None,
    verify_source: bool,
) -> dict[str, Any]:
    source = _mapping(value, name)
    if set(source) != _SOURCE_FIELDS:
        raise IndependentErosionDeliveryError(f"{name} fields are invalid")
    raw_path = _text(source.get("path"), f"{name} path")
    path = Path(raw_path)
    if not path.is_absolute() and source_base is not None:
        path = Path(source_base) / path
    path = path.resolve()
    digest = _sha256(source.get("sha256"), f"{name} sha256")
    byte_count = _integer(source.get("bytes"), f"{name} bytes")
    if verify_source:
        if not path.is_file():
            raise IndependentErosionDeliveryError(
                f"real measured source does not exist: {path}"
            )
        if path.stat().st_size != byte_count:
            raise IndependentErosionDeliveryError(
                f"real measured source byte count mismatch: {path}"
            )
        if sha256_file(path) != digest:
            raise IndependentErosionDeliveryError(
                f"real measured source hash mismatch: {path}"
            )
    return {"path": str(path), "sha256": digest, "bytes": byte_count}


def _normalize_scan(
    value: Any,
    name: str,
    *,
    source_base: str | Path | None,
    verify_sources: bool,
) -> dict[str, Any]:
    scan = _mapping(value, name)
    if set(scan) != _SCAN_FIELDS:
        raise IndependentErosionDeliveryError(f"{name} fields are invalid")
    units = _text(scan.get("coordinate_units"), f"{name} coordinate_units")
    if units not in {"m", "cm", "mm", "um"}:
        raise IndependentErosionDeliveryError(
            f"{name} coordinate_units must be m, cm, mm, or um"
        )
    return {
        "scan_id": _text(scan.get("scan_id"), f"{name} scan_id"),
        "acquired_at": _text(scan.get("acquired_at"), f"{name} acquired_at"),
        "instrument_serial": _text(
            scan.get("instrument_serial"), f"{name} instrument_serial"
        ),
        "calibration_certificate": _source_descriptor(
            scan.get("calibration_certificate"),
            f"{name} calibration certificate",
            source_base=source_base,
            verify_source=verify_sources,
        ),
        "coordinate_frame": _text(
            scan.get("coordinate_frame"), f"{name} coordinate_frame"
        ),
        "fiducial_id": _text(scan.get("fiducial_id"), f"{name} fiducial_id"),
        "fixture_id": _text(scan.get("fixture_id"), f"{name} fixture_id"),
        "coordinate_units": units,
        "source": _source_descriptor(
            scan.get("source"),
            f"{name} source",
            source_base=source_base,
            verify_source=verify_sources,
        ),
    }


def validate_scan_manifest(
    manifest: Any,
    *,
    source_base: str | Path | None = None,
    verify_sources: bool = True,
    allow_blocked_template: bool = False,
) -> dict[str, Any]:
    """Validate scan-pair identity, provenance, sources, and UTC chronology."""
    document = _mapping(manifest, "scan manifest")
    if set(document) != _MANIFEST_FIELDS:
        raise IndependentErosionDeliveryError("scan manifest fields are invalid")
    if (
        document.get("version") != SCAN_MANIFEST_VERSION
        or document.get("schema_version") != SCAN_MANIFEST_VERSION
    ):
        raise IndependentErosionDeliveryError("scan manifest version mismatch")
    status = document.get("status")
    if status not in {"measured", "blocked"}:
        raise IndependentErosionDeliveryError(
            "scan manifest status must be measured or blocked"
        )
    if status == "blocked" and not allow_blocked_template:
        raise IndependentErosionDeliveryError(
            "scan manifest is blocked and has no real measured delivery"
        )
    formal_requested = document.get("formal_requested")
    if not isinstance(formal_requested, bool):
        raise IndependentErosionDeliveryError("formal_requested must be a boolean")
    created_at = _text(document.get("created_at_utc"), "created_at_utc")
    created_time = _utc(created_at, "created_at_utc")
    if document.get("minimum_independent_groups") != MINIMUM_INDEPENDENT_GROUPS:
        raise IndependentErosionDeliveryError(
            "minimum_independent_groups must remain 3"
        )
    if document.get("recommended_formal_groups") != RECOMMENDED_FORMAL_GROUPS:
        raise IndependentErosionDeliveryError(
            "recommended_formal_groups must remain 5"
        )
    blocked_reason = document.get("blocked_reason")
    if status == "blocked":
        _text(blocked_reason, "blocked_reason")
    elif blocked_reason is not None:
        raise IndependentErosionDeliveryError(
            "measured scan manifest blocked_reason must be null"
        )

    pairs = _sequence(document.get("scan_pairs"), "scan_pairs")
    if status == "measured" and not pairs:
        raise IndependentErosionDeliveryError("scan_pairs must not be empty")
    normalized_pairs: list[dict[str, Any]] = []
    identities: set[str] = set()
    scan_ids: set[str] = set()
    for index, raw in enumerate(pairs):
        pair = _mapping(raw, f"scan_pairs[{index}]")
        if set(pair) != _PAIR_FIELDS:
            raise IndependentErosionDeliveryError("scan pair fields are invalid")
        record_id = _text(pair.get("record_id"), "record_id")
        if record_id in identities:
            raise IndependentErosionDeliveryError("record_id values must be unique")
        identities.add(record_id)
        episode = _mapping(pair.get("service_episode"), "service_episode")
        if set(episode) != _EPISODE_FIELDS:
            raise IndependentErosionDeliveryError(
                "service_episode fields are invalid"
            )
        episode_id = _text(
            episode.get("service_episode_id"), "service_episode_id"
        )
        started_text = _text(episode.get("started_at"), "service episode started_at")
        ended_text = _text(episode.get("ended_at"), "service episode ended_at")
        started = _utc(started_text, "service episode started_at")
        ended = _utc(ended_text, "service episode ended_at")
        baseline = _normalize_scan(
            pair.get("baseline_scan"),
            "baseline_scan",
            source_base=source_base,
            verify_sources=verify_sources,
        )
        worn = _normalize_scan(
            pair.get("worn_scan"),
            "worn_scan",
            source_base=source_base,
            verify_sources=verify_sources,
        )
        baseline_time = _utc(baseline["acquired_at"], "baseline acquired_at")
        worn_time = _utc(worn["acquired_at"], "worn acquired_at")
        if not baseline_time < started <= ended < worn_time:
            raise IndependentErosionDeliveryError(
                "time order must be baseline < service start <= service end < worn"
            )
        if created_time < worn_time:
            raise IndependentErosionDeliveryError(
                "scan manifest created_at_utc predates a worn scan"
            )
        for scan_id in (baseline["scan_id"], worn["scan_id"]):
            if scan_id in scan_ids:
                raise IndependentErosionDeliveryError(
                    "baseline/worn scan_id values must be globally unique"
                )
            scan_ids.add(scan_id)
        for field in ("coordinate_frame", "fiducial_id", "fixture_id"):
            if baseline[field] != worn[field]:
                raise IndependentErosionDeliveryError(
                    f"baseline/worn {field} must match"
                )
        normalized_pairs.append(
            {
                "record_id": record_id,
                "asset_id": _text(pair.get("asset_id"), "asset_id"),
                "component_id": _text(
                    pair.get("component_id"), "component_id"
                ),
                "campaign_id": _text(pair.get("campaign_id"), "campaign_id"),
                "service_episode": {
                    "service_episode_id": episode_id,
                    "started_at": started_text,
                    "ended_at": ended_text,
                },
                "baseline_scan": baseline,
                "worn_scan": worn,
            }
        )
    normalized_pairs.sort(key=lambda item: item["record_id"])
    return {
        "version": SCAN_MANIFEST_VERSION,
        "schema_version": SCAN_MANIFEST_VERSION,
        "status": status,
        "formal_requested": formal_requested,
        "created_at_utc": created_at,
        "minimum_independent_groups": MINIMUM_INDEPENDENT_GROUPS,
        "recommended_formal_groups": RECOMMENDED_FORMAL_GROUPS,
        "blocked_reason": blocked_reason,
        "scan_pairs": normalized_pairs,
    }


def recompute_evaluation_mask_sha256(mask: Mapping[str, Any]) -> str:
    unsigned = dict(jsonable(mask))
    unsigned.pop("mask_sha256", None)
    return canonical_sha256(unsigned)


def validate_evaluation_mask(
    mask: Any,
    *,
    expected_topology_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate preregistered per-episode face masks and minimum coverage."""
    document = _mapping(mask, "evaluation mask")
    if set(document) != _MASK_FIELDS:
        raise IndependentErosionDeliveryError("evaluation mask fields are invalid")
    if (
        document.get("version") != EVALUATION_MASK_VERSION
        or document.get("status") != "measured"
        or document.get("immutable") is not True
    ):
        raise IndependentErosionDeliveryError(
            "evaluation mask must be measured and immutable"
        )
    topology_hash = _sha256(
        document.get("topology_v3_sha256"), "mask topology_v3_sha256"
    )
    if expected_topology_sha256 is not None and topology_hash != _sha256(
        expected_topology_sha256, "expected topology_v3_sha256"
    ):
        raise IndependentErosionDeliveryError(
            "evaluation mask topology_v3_sha256 mismatch"
        )
    episodes_raw = _sequence(document.get("episodes"), "mask episodes")
    if not episodes_raw:
        raise IndependentErosionDeliveryError("mask episodes must not be empty")
    episodes: list[dict[str, Any]] = []
    record_ids: set[str] = set()
    episode_ids: set[str] = set()
    for raw in episodes_raw:
        episode = _mapping(raw, "mask episode")
        if set(episode) != _MASK_EPISODE_FIELDS:
            raise IndependentErosionDeliveryError(
                "evaluation mask episode fields are invalid"
            )
        record_id = _text(episode.get("record_id"), "mask record_id")
        episode_id = _text(
            episode.get("service_episode_id"), "mask service_episode_id"
        )
        if record_id in record_ids or episode_id in episode_ids:
            raise IndependentErosionDeliveryError(
                "mask record_id and service_episode_id values must be unique"
            )
        record_ids.add(record_id)
        episode_ids.add(episode_id)
        face_ids = [
            _integer(value, "mask parent_face_id")
            for value in _sequence(
                episode.get("parent_face_ids"), "mask parent_face_ids"
            )
        ]
        if not face_ids or len(face_ids) != len(set(face_ids)):
            raise IndependentErosionDeliveryError(
                "mask parent_face_ids must be nonempty and unique"
            )
        if face_ids != sorted(face_ids):
            raise IndependentErosionDeliveryError(
                "mask parent_face_ids must be sorted"
            )
        coverage = episode.get("minimum_area_coverage_fraction")
        if isinstance(coverage, bool):
            raise IndependentErosionDeliveryError(
                "minimum_area_coverage_fraction must be numeric"
            )
        try:
            coverage_value = float(coverage)
        except (TypeError, ValueError) as exc:
            raise IndependentErosionDeliveryError(
                "minimum_area_coverage_fraction must be numeric"
            ) from exc
        if not math.isfinite(coverage_value) or not 0.0 < coverage_value <= 1.0:
            raise IndependentErosionDeliveryError(
                "minimum_area_coverage_fraction must be in (0, 1]"
            )
        episodes.append(
            {
                "record_id": record_id,
                "service_episode_id": episode_id,
                "parent_face_ids": face_ids,
                "minimum_area_coverage_fraction": coverage_value,
                "minimum_valid_count": _integer(
                    episode.get("minimum_valid_count"),
                    "minimum_valid_count",
                    positive=True,
                ),
            }
        )
    episodes.sort(key=lambda item: item["record_id"])
    normalized = {
        "version": EVALUATION_MASK_VERSION,
        "status": "measured",
        "immutable": True,
        "topology_v3_sha256": topology_hash,
        "episodes": episodes,
    }
    expected_mask_hash = recompute_evaluation_mask_sha256(normalized)
    if document.get("mask_sha256") != expected_mask_hash:
        raise IndependentErosionDeliveryError("mask_sha256/content mismatch")
    normalized["mask_sha256"] = expected_mask_hash
    return normalized


def recompute_protocol_v2_sha256(protocol: Mapping[str, Any]) -> str:
    unsigned = dict(jsonable(protocol))
    unsigned.pop("protocol_sha256", None)
    return canonical_sha256(unsigned)


def _verify_topology_source(contract: Mapping[str, Any]) -> None:
    path = Path(_text(contract.get("topology_path"), "topology_path")).resolve()
    if not path.is_file():
        raise IndependentErosionDeliveryError(
            f"topology v3 artifact does not exist: {path}"
        )
    if sha256_file(path) != _sha256(
        contract.get("topology_v3_sha256"), "topology_v3_sha256"
    ):
        raise IndependentErosionDeliveryError("topology v3 SHA-256 mismatch")


def _manifest_assets(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "record_id": pair["record_id"],
            "asset_id": pair["asset_id"],
            "component_id": pair["component_id"],
            "service_episode_id": pair["service_episode"]["service_episode_id"],
            "campaign_id": pair["campaign_id"],
            "scan_id": (
                f"{pair['baseline_scan']['scan_id']}::{pair['worn_scan']['scan_id']}"
            ),
            "status": "measured",
            "source": pair["baseline_scan"]["source"],
            "truth_source": pair["worn_scan"]["source"],
        }
        for pair in manifest["scan_pairs"]
    ]


def build_protocol_v2(
    scan_manifest: Mapping[str, Any] | str | Path,
    evaluation_mask: Mapping[str, Any] | str | Path,
    truth_build_config: Mapping[str, Any] | str | Path,
    *,
    seed: int = 30,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Freeze scan pairs, mask, current core protocol, and truth build inputs."""
    manifest_path = None if isinstance(scan_manifest, Mapping) else Path(scan_manifest)
    manifest = validate_scan_manifest(
        _load_json(scan_manifest, "scan manifest"),
        source_base=None if manifest_path is None else manifest_path.resolve().parent,
        verify_sources=True,
    )
    mask_path = None if isinstance(evaluation_mask, Mapping) else Path(evaluation_mask)
    if mask_path is None:
        raise IndependentErosionDeliveryError(
            "evaluation_mask must be a file so its bytes can be frozen"
        )
    raw_mask = _load_json(mask_path, "evaluation mask")
    raw_build = _load_json(truth_build_config, "truth build config")
    required_build_fields = {
        "topology_path",
        "topology_v3_sha256",
        "mesh_signature",
        "artifact_set_id",
        "registration_options",
        "maximum_projection_distance_mm",
    }
    values = {
        name: raw_build[name]
        for name in required_build_fields
        if name in raw_build
    }
    missing = sorted(required_build_fields - set(values))
    if missing:
        raise IndependentErosionDeliveryError(
            f"truth build config is incomplete: {missing}"
        )
    build_contract = build_truth_build_contract(**values)
    _verify_topology_source(build_contract)
    mask = validate_evaluation_mask(
        raw_mask,
        expected_topology_sha256=build_contract["topology_v3_sha256"],
    )
    manifest_records = {pair["record_id"] for pair in manifest["scan_pairs"]}
    mask_records = {episode["record_id"] for episode in mask["episodes"]}
    if manifest_records != mask_records:
        raise IndependentErosionDeliveryError(
            "evaluation mask must preregister every scan-pair record exactly"
        )
    episode_by_record = {
        pair["record_id"]: pair["service_episode"]["service_episode_id"]
        for pair in manifest["scan_pairs"]
    }
    if any(
        episode_by_record[episode["record_id"]] != episode["service_episode_id"]
        for episode in mask["episodes"]
    ):
        raise IndependentErosionDeliveryError(
            "evaluation mask service episode identity mismatch"
        )
    core = build_protocol(_manifest_assets(manifest), seed=seed)
    group_count = int(core["n_independent_groups"])
    formal = bool(
        manifest["formal_requested"]
        and group_count >= RECOMMENDED_FORMAL_GROUPS
    )
    if formal:
        phase = "formal_candidate"
        reason = None
    else:
        phase = "pilot"
        reason = (
            "pilot has fewer than five recommended independent groups"
            if group_count < RECOMMENDED_FORMAL_GROUPS
            else "formal_requested is false"
        )
    created = created_at_utc or utc_now()
    if _utc(created, "protocol created_at_utc") < _utc(
        manifest["created_at_utc"], "scan manifest created_at_utc"
    ):
        raise IndependentErosionDeliveryError(
            "protocol created_at_utc predates scan manifest"
        )
    result = {
        "version": PROTOCOL_V2_VERSION,
        "schema_version": PROTOCOL_V2_VERSION,
        "status": "measured",
        "immutable": True,
        "formal": formal,
        "study_phase": phase,
        "minimum_independent_groups": MINIMUM_INDEPENDENT_GROUPS,
        "recommended_formal_groups": RECOMMENDED_FORMAL_GROUPS,
        "formal_eligibility_reason": reason,
        "created_at_utc": created,
        "scan_manifest": manifest,
        "scan_manifest_sha256": canonical_sha256(manifest),
        "evaluation_mask": mask,
        "evaluation_mask_sha256": mask["mask_sha256"],
        "evaluation_mask_file_sha256": sha256_file(mask_path),
        "truth_build_contract": build_contract,
        "core_protocol": core,
    }
    result["protocol_sha256"] = recompute_protocol_v2_sha256(result)
    return validate_protocol_v2(result, verify_sources=True)


def validate_protocol_v2(
    protocol: Any,
    *,
    verify_sources: bool = True,
) -> dict[str, Any]:
    """Validate every v2 binding and recompute the identity-only core split."""
    document = _mapping(protocol, "v2 protocol")
    if set(document) != _PROTOCOL_FIELDS:
        raise IndependentErosionDeliveryError("v2 protocol fields are invalid")
    if (
        document.get("version") != PROTOCOL_V2_VERSION
        or document.get("schema_version") != PROTOCOL_V2_VERSION
        or document.get("status") != "measured"
        or document.get("immutable") is not True
    ):
        raise IndependentErosionDeliveryError("v2 protocol contract is invalid")
    protocol_created = _utc(
        document.get("created_at_utc"), "protocol created_at_utc"
    )
    if document.get("minimum_independent_groups") != MINIMUM_INDEPENDENT_GROUPS:
        raise IndependentErosionDeliveryError("v2 minimum group count changed")
    if document.get("recommended_formal_groups") != RECOMMENDED_FORMAL_GROUPS:
        raise IndependentErosionDeliveryError("v2 recommended group count changed")
    manifest = validate_scan_manifest(
        document.get("scan_manifest"), verify_sources=verify_sources
    )
    if protocol_created < _utc(
        manifest["created_at_utc"], "scan manifest created_at_utc"
    ):
        raise IndependentErosionDeliveryError(
            "protocol created_at_utc predates scan manifest"
        )
    if document.get("scan_manifest_sha256") != canonical_sha256(manifest):
        raise IndependentErosionDeliveryError("scan_manifest_sha256 mismatch")
    build_contract = _mapping(
        document.get("truth_build_contract"), "truth_build_contract"
    )
    rebuilt_contract = build_truth_build_contract(
        topology_path=build_contract.get("topology_path"),
        topology_v3_sha256=build_contract.get("topology_v3_sha256"),
        mesh_signature=build_contract.get("mesh_signature"),
        artifact_set_id=build_contract.get("artifact_set_id"),
        registration_options=build_contract.get("registration_options"),
        maximum_projection_distance_mm=build_contract.get(
            "maximum_projection_distance_mm"
        ),
    )
    if dict(build_contract) != rebuilt_contract:
        raise IndependentErosionDeliveryError(
            "truth_build_contract hash or fields changed"
        )
    if verify_sources:
        _verify_topology_source(rebuilt_contract)
    mask = validate_evaluation_mask(
        document.get("evaluation_mask"),
        expected_topology_sha256=rebuilt_contract["topology_v3_sha256"],
    )
    if document.get("evaluation_mask_sha256") != mask["mask_sha256"]:
        raise IndependentErosionDeliveryError("evaluation_mask_sha256 mismatch")
    _sha256(
        document.get("evaluation_mask_file_sha256"),
        "evaluation_mask_file_sha256",
    )
    core = validate_protocol(document.get("core_protocol")) if verify_sources else None
    if core is None:
        from step30_independent_erosion_protocol import _validate_protocol

        core = _validate_protocol(document.get("core_protocol"), verify_sources=False)
    expected_assets = _manifest_assets(manifest)
    frozen_assets = [
        {
            key: asset[key]
            for key in (
                "record_id",
                "asset_id",
                "component_id",
                "service_episode_id",
                "campaign_id",
                "scan_id",
                "status",
            )
        }
        for asset in core["assets"]
    ]
    expected_identities = [
        {
            key: asset[key]
            for key in (
                "record_id",
                "asset_id",
                "component_id",
                "service_episode_id",
                "campaign_id",
                "scan_id",
                "status",
            )
        }
        for asset in expected_assets
    ]
    if frozen_assets != sorted(expected_identities, key=lambda item: item["record_id"]):
        raise IndependentErosionDeliveryError(
            "core protocol identities differ from scan manifest"
        )
    expected_by_record = {asset["record_id"]: asset for asset in expected_assets}
    for asset in core["assets"]:
        expected = expected_by_record[asset["record_id"]]
        if (
            Path(asset["path"]).resolve()
            != Path(expected["source"]["path"]).resolve()
            or asset["sha256"] != expected["source"]["sha256"]
            or asset["bytes"] != expected["source"]["bytes"]
            or asset.get("truth_source") != expected["truth_source"]
        ):
            raise IndependentErosionDeliveryError(
                "core protocol baseline/worn sources differ from scan manifest"
            )
    group_count = int(core["n_independent_groups"])
    expected_formal = bool(
        manifest["formal_requested"]
        and group_count >= RECOMMENDED_FORMAL_GROUPS
    )
    expected_phase = "formal_candidate" if expected_formal else "pilot"
    expected_reason = (
        None
        if expected_formal
        else (
            "pilot has fewer than five recommended independent groups"
            if group_count < RECOMMENDED_FORMAL_GROUPS
            else "formal_requested is false"
        )
    )
    if (
        document.get("formal") is not expected_formal
        or document.get("study_phase") != expected_phase
        or document.get("formal_eligibility_reason") != expected_reason
    ):
        raise IndependentErosionDeliveryError(
            "formal/pilot classification is inconsistent"
        )
    mask_identity = {
        (episode["record_id"], episode["service_episode_id"])
        for episode in mask["episodes"]
    }
    manifest_identity = {
        (pair["record_id"], pair["service_episode"]["service_episode_id"])
        for pair in manifest["scan_pairs"]
    }
    if mask_identity != manifest_identity:
        raise IndependentErosionDeliveryError(
            "mask and scan manifest episode identities differ"
        )
    if document.get("protocol_sha256") != recompute_protocol_v2_sha256(document):
        raise IndependentErosionDeliveryError("protocol_sha256/content mismatch")
    return dict(document)


def _mask_for_record(protocol: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        episode["record_id"]: episode
        for episode in protocol["evaluation_mask"]["episodes"]
    }


def build_commitment_v2(
    protocol: Mapping[str, Any],
    model_descriptor: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    *,
    committed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Commit the complete preregistered independent-test evaluation mask."""
    frozen = validate_protocol_v2(protocol, verify_sources=False)
    descriptor = dict(_mapping(model_descriptor, "model_descriptor"))
    timestamp = committed_at_utc or utc_now()
    committed = _utc(timestamp, "commitment_created_at_utc")
    if committed < _utc(frozen["created_at_utc"], "protocol created_at_utc"):
        raise IndependentErosionDeliveryError("commitment predates protocol")
    descriptor.update(
        {
            "delivery_protocol_sha256": frozen["protocol_sha256"],
            "truth_build_contract_sha256": frozen["truth_build_contract"][
                "truth_build_contract_sha256"
            ],
            "evaluation_mask_sha256": frozen["evaluation_mask_sha256"],
            "commitment_created_at_utc": timestamp,
        }
    )
    topology = load_topology_v3(
        frozen["truth_build_contract"]["topology_path"],
        expected_mesh_signature=frozen["truth_build_contract"]["mesh_signature"],
    )
    commitment = build_prediction_commitment(
        frozen["core_protocol"],
        descriptor,
        predictions,
        topology_binding={
            "topology_v3_sha256": frozen["truth_build_contract"][
                "topology_v3_sha256"
            ],
            "mesh_signature": frozen["truth_build_contract"]["mesh_signature"],
            "artifact_set_id": frozen["truth_build_contract"]["artifact_set_id"],
            "n_parent_faces": int(topology["n_parent_faces"]),
        },
    )
    return validate_commitment_v2(commitment, frozen)


def validate_commitment_v2(
    commitment: Any, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    frozen = validate_protocol_v2(protocol, verify_sources=False)
    valid = validate_prediction_commitment(commitment, frozen["core_protocol"])
    descriptor = _mapping(valid.get("model_descriptor"), "model_descriptor")
    expected = {
        "delivery_protocol_sha256": frozen["protocol_sha256"],
        "truth_build_contract_sha256": frozen["truth_build_contract"][
            "truth_build_contract_sha256"
        ],
        "evaluation_mask_sha256": frozen["evaluation_mask_sha256"],
    }
    for field, value in expected.items():
        if descriptor.get(field) != value:
            raise IndependentErosionDeliveryError(
                f"commitment {field} binding mismatch"
            )
    timestamp = descriptor.get("commitment_created_at_utc")
    committed = _utc(timestamp, "commitment_created_at_utc")
    if committed < _utc(frozen["created_at_utc"], "protocol created_at_utc"):
        raise IndependentErosionDeliveryError("commitment predates protocol")
    mask = _mask_for_record(frozen)
    expected_records = set(frozen["core_protocol"]["splits"]["independent_test"])
    expected_faces = {
        record_id: set(mask[record_id]["parent_face_ids"])
        for record_id in expected_records
    }
    actual_faces: dict[str, set[int]] = {record_id: set() for record_id in expected_records}
    for row in valid["predictions"]:
        if row["record_id"] in actual_faces:
            actual_faces[row["record_id"]].add(int(row["parent_face_id"]))
    if actual_faces != expected_faces:
        raise IndependentErosionDeliveryError(
            "commitment must exactly cover the preregistered independent-test mask"
        )
    return valid


def external_pin_bindings(protocol: Mapping[str, Any]) -> dict[str, str]:
    frozen = validate_protocol_v2(protocol, verify_sources=False)
    return {
        "protocol_sha256": frozen["protocol_sha256"],
        "truth_build_contract_sha256": frozen["truth_build_contract"][
            "truth_build_contract_sha256"
        ],
        "topology_v3_sha256": frozen["truth_build_contract"][
            "topology_v3_sha256"
        ],
        "evaluation_mask_sha256": frozen["evaluation_mask_sha256"],
    }


def validate_truth_coverage(
    protocol: Mapping[str, Any], truth: Mapping[str, Any]
) -> dict[str, Any]:
    """Enforce preregistered area and valid sample minima per test episode."""
    frozen = validate_protocol_v2(protocol, verify_sources=False)
    valid_truth = validate_independent_erosion_truth(
        truth, protocol=frozen["core_protocol"], verify_sources=False
    )
    topology = load_topology_v3(
        frozen["truth_build_contract"]["topology_path"],
        expected_mesh_signature=frozen["truth_build_contract"]["mesh_signature"],
    )
    areas = [float(value) * 1.0e6 for value in topology["face_area_m2"]]
    rows_by_record: dict[str, dict[int, Mapping[str, Any]]] = {}
    for row in valid_truth["rows"]:
        rows_by_record.setdefault(str(row["record_id"]), {})[
            int(row["parent_face_id"])
        ] = row
    required_records = set(frozen["core_protocol"]["splits"]["independent_test"])
    mask = _mask_for_record(frozen)
    results = []
    for record_id in sorted(required_records):
        episode = mask[record_id]
        face_ids = episode["parent_face_ids"]
        if any(face_id >= len(areas) for face_id in face_ids):
            raise IndependentErosionDeliveryError(
                "evaluation mask parent_face_id exceeds topology"
            )
        denominator = sum(areas[face_id] for face_id in face_ids)
        measured = rows_by_record.get(record_id, {})
        numerator = sum(
            areas[face_id] for face_id in face_ids if face_id in measured
        )
        coverage = numerator / denominator
        valid_count = sum(
            int(measured[face_id]["valid_count"])
            for face_id in face_ids
            if face_id in measured
        )
        coverage_passed = coverage >= float(
            episode["minimum_area_coverage_fraction"]
        )
        count_passed = valid_count >= int(episode["minimum_valid_count"])
        results.append(
            {
                "record_id": record_id,
                "service_episode_id": episode["service_episode_id"],
                "area_coverage_fraction": coverage,
                "minimum_area_coverage_fraction": episode[
                    "minimum_area_coverage_fraction"
                ],
                "valid_count": valid_count,
                "minimum_valid_count": episode["minimum_valid_count"],
                "passed": coverage_passed and count_passed,
            }
        )
    if not all(result["passed"] for result in results):
        raise IndependentErosionDeliveryError(
            "preregistered per-episode area coverage or valid_count is insufficient"
        )
    return {"passed": True, "episodes": results}


def build_truth_v2(
    protocol: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Build current truth and reject it immediately if coverage is insufficient."""
    frozen = validate_protocol_v2(protocol, verify_sources=True)
    config = frozen["truth_build_contract"]
    truth = build_independent_erosion_truth(
        frozen["core_protocol"],
        records,
        topology_path=config["topology_path"],
        topology_v3_sha256=config["topology_v3_sha256"],
        mesh_signature=config["mesh_signature"],
        artifact_set_id=config["artifact_set_id"],
        registration_options=config["registration_options"],
        maximum_projection_distance_mm=config[
            "maximum_projection_distance_mm"
        ],
    )
    validate_truth_coverage(frozen, truth)
    return truth


def _blocked(reason: str, checks: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "version": DELIVERY_ACCEPTANCE_VERSION,
        "status": "blocked",
        "formal": False,
        "accepted": False,
        "blocked": True,
        "official_reference_gate": False,
        "official_accuracy_gate": False,
        "checks": dict(checks),
        "reasons": [reason],
    }
    result["report_sha256"] = canonical_sha256(result)
    return result


def accept_independent_erosion_delivery(
    protocol: Mapping[str, Any] | str | Path,
    commitment: Mapping[str, Any] | str | Path,
    truth: Mapping[str, Any] | str | Path,
    reveal_receipt: Mapping[str, Any] | str | Path,
    pin_envelope: Mapping[str, Any] | str | Path | None,
    *,
    truth_records: Any,
    trusted_public_key: str | Path | bytes | None = None,
    test_hmac_key: str | bytes | None = None,
) -> dict[str, Any]:
    """Verify a third-party receipt, then and only then invoke core replay."""
    checks: dict[str, Any] = {
        "protocol_v2_verified": False,
        "third_party_receipt_verified": False,
        "coverage_verified": False,
        "replay_invoked": False,
    }
    try:
        frozen = validate_protocol_v2(
            _load_json(protocol, "v2 protocol"), verify_sources=True
        )
        checks["protocol_v2_verified"] = True
        if frozen["formal"] is not True:
            raise IndependentErosionDeliveryError(
                "pilot protocol cannot produce formal acceptance"
            )
        commitment_path = None if isinstance(commitment, Mapping) else Path(commitment)
        if commitment_path is None:
            raise IndependentErosionDeliveryError(
                "formal accept requires the serialized commitment file"
            )
        frozen_commitment = validate_commitment_v2(
            _load_json(commitment, "commitment"), frozen
        )
        if pin_envelope is None:
            raise IndependentErosionDeliveryError(
                "verifiable third-party receipt is required"
            )
        envelope = _load_json(pin_envelope, "prediction pin envelope")
        pin_result = verify_pin_envelope(
            envelope,
            commitment=frozen_commitment,
            commitment_path=commitment_path,
            expected_bindings=external_pin_bindings(frozen),
            trusted_public_key=trusted_public_key,
            test_hmac_key=test_hmac_key,
            not_before=frozen_commitment["model_descriptor"][
                "commitment_created_at_utc"
            ],
            require_formal=True,
        )
        checks["third_party_receipt_verified"] = True
        frozen_truth = validate_independent_erosion_truth(
            _load_json(truth, "truth"),
            protocol=frozen["core_protocol"],
            verify_sources=False,
        )
        coverage = validate_truth_coverage(frozen, frozen_truth)
        checks["coverage_verified"] = True
        frozen_receipt = validate_reveal_receipt(
            _load_json(reveal_receipt, "reveal receipt")
        )
        checks["replay_invoked"] = True
        replay = replay_independent_erosion_acceptance(
            frozen["core_protocol"],
            frozen_commitment,
            frozen_truth,
            frozen_receipt,
            trusted_commitment_sha256=pin_result[
                "canonical_commitment_digest"
            ],
            trusted_protocol_sha256=frozen["core_protocol"]["protocol_sha256"],
            truth_records=truth_records,
            build_config=frozen["truth_build_contract"],
        )
        if replay.get("status") == "blocked":
            return _blocked(
                f"core acceptance replay blocked: {replay.get('reasons', ['unknown'])[0]}",
                checks,
            )
        result = {
            "version": DELIVERY_ACCEPTANCE_VERSION,
            "status": replay["status"],
            "formal": True,
            "accepted": bool(replay.get("accepted")),
            "blocked": False,
            "official_reference_gate": False,
            "official_accuracy_gate": False,
            "checks": checks,
            "third_party_receipt": pin_result,
            "coverage": coverage,
            "core_replay": replay,
            "reasons": list(replay.get("reasons", [])),
        }
        result["report_sha256"] = canonical_sha256(result)
        return result
    except (
        OSError,
        ValidationError,
        ValueError,
        TypeError,
        KeyError,
        ExternalCommitmentPinError,
    ) as exc:
        return _blocked(str(exc) or type(exc).__name__, checks)


def delivery_status(
    protocol: Mapping[str, Any] | str | Path,
    *,
    commitment: Mapping[str, Any] | str | Path | None = None,
    pin_envelope: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Report artifact readiness without granting or replaying acceptance."""
    try:
        frozen = validate_protocol_v2(
            _load_json(protocol, "v2 protocol"), verify_sources=True
        )
        commitment_valid = False
        receipt_present = pin_envelope is not None
        if commitment is not None:
            validate_commitment_v2(_load_json(commitment, "commitment"), frozen)
            commitment_valid = True
        return {
            "version": "step30-independent-erosion-delivery-status-v2",
            "status": "ready" if commitment_valid and receipt_present else "incomplete",
            "study_phase": frozen["study_phase"],
            "formal_candidate": frozen["formal"],
            "group_count": frozen["core_protocol"]["n_independent_groups"],
            "protocol_valid": True,
            "commitment_valid": commitment_valid,
            "pin_envelope_present": receipt_present,
            "formal_acceptance_granted": False,
        }
    except (OSError, ValidationError, ValueError, TypeError, KeyError) as exc:
        return {
            "version": "step30-independent-erosion-delivery-status-v2",
            "status": "blocked",
            "formal_acceptance_granted": False,
            "reason": str(exc) or type(exc).__name__,
        }


def validate_template(document: Mapping[str, Any], *, kind: str = "auto") -> dict[str, Any]:
    """Validate shipped blocked templates without treating them as evidence."""
    version = document.get("version")
    selected = kind
    if selected == "auto":
        if version == SCAN_MANIFEST_VERSION:
            selected = "scan-manifest"
        elif version == "step30-prediction-pin-envelope-v1":
            selected = "pin-envelope"
        else:
            raise IndependentErosionDeliveryError("cannot infer template kind")
    if selected == "scan-manifest":
        validated = validate_scan_manifest(
            document,
            verify_sources=False,
            allow_blocked_template=True,
        )
    elif selected == "pin-envelope":
        required = {
            "version",
            "formal",
            "test_only",
            "canonical_commitment_digest",
            "commitment_file_sha256",
            "bindings",
            "issuer",
            "received_at",
            "receipt_id",
            "signature_algorithm",
            "signature",
            "certificate_pem",
            "public_key_url",
            "envelope_sha256",
        }
        if set(document) != required or version != "step30-prediction-pin-envelope-v1":
            raise IndependentErosionDeliveryError("pin envelope template fields are invalid")
        if document.get("formal") is not False or document.get("test_only") is not False:
            raise IndependentErosionDeliveryError(
                "unissued pin template must be non-formal and non-test"
            )
        validated = dict(document)
    else:
        raise IndependentErosionDeliveryError("template kind is invalid")
    return {"status": "valid_template", "kind": selected, "document": validated}


def _write_or_print(document: Mapping[str, Any], output: str | Path | None) -> None:
    if output is None:
        print(json.dumps(document, ensure_ascii=True, indent=2, allow_nan=False))
    else:
        safe_json_dump(document, output)
        print(json.dumps({"status": "written", "path": str(Path(output).resolve())}))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deliver independent measured erosion truth protocol v2"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-template")
    validate.add_argument("input")
    validate.add_argument(
        "--kind", choices=("auto", "scan-manifest", "pin-envelope"), default="auto"
    )

    protocol = subparsers.add_parser("build-protocol")
    protocol.add_argument("--scan-manifest", required=True)
    protocol.add_argument("--evaluation-mask", required=True)
    protocol.add_argument("--truth-build-config", required=True)
    protocol.add_argument("--seed", type=int, default=30)
    protocol.add_argument("--created-at-utc")
    protocol.add_argument("--output", required=True)

    commit = subparsers.add_parser("commit")
    commit.add_argument("--protocol", required=True)
    commit.add_argument("--model-descriptor", required=True)
    commit.add_argument("--predictions", required=True)
    commit.add_argument("--committed-at-utc")
    commit.add_argument("--output", required=True)

    truth = subparsers.add_parser("build-truth")
    truth.add_argument("--protocol", required=True)
    truth.add_argument("--records", required=True)
    truth.add_argument("--output", required=True)

    reveal = subparsers.add_parser("reveal")
    reveal.add_argument("--protocol", required=True)
    reveal.add_argument("--commitment", required=True)
    reveal.add_argument("--truth", required=True)
    reveal.add_argument("--output", required=True)

    accept = subparsers.add_parser("accept")
    accept.add_argument("--protocol", required=True)
    accept.add_argument("--commitment", required=True)
    accept.add_argument("--truth", required=True)
    accept.add_argument("--reveal-receipt", required=True)
    accept.add_argument("--pin-envelope")
    accept.add_argument("--truth-records", required=True)
    accept.add_argument("--trusted-public-key")
    accept.add_argument("--test-hmac-key")
    accept.add_argument("--output")

    status = subparsers.add_parser("status")
    status.add_argument("--protocol", required=True)
    status.add_argument("--commitment")
    status.add_argument("--pin-envelope")
    status.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-template":
            result = validate_template(_load_json(args.input, "template"), kind=args.kind)
            _write_or_print(result, None)
        elif args.command == "build-protocol":
            result = build_protocol_v2(
                args.scan_manifest,
                args.evaluation_mask,
                args.truth_build_config,
                seed=args.seed,
                created_at_utc=args.created_at_utc,
            )
            _write_or_print(result, args.output)
        elif args.command == "commit":
            protocol = validate_protocol_v2(
                _load_json(args.protocol, "v2 protocol"), verify_sources=False
            )
            model = _load_json(args.model_descriptor, "model descriptor")
            predictions_doc = safe_json_load(args.predictions)
            predictions = (
                predictions_doc.get("predictions")
                if isinstance(predictions_doc, Mapping)
                else predictions_doc
            )
            result = build_commitment_v2(
                protocol,
                model,
                _sequence(predictions, "predictions"),
                committed_at_utc=args.committed_at_utc,
            )
            _write_or_print(result, args.output)
        elif args.command == "build-truth":
            protocol = validate_protocol_v2(
                _load_json(args.protocol, "v2 protocol"), verify_sources=True
            )
            records_doc = safe_json_load(args.records)
            records = (
                records_doc.get("records")
                if isinstance(records_doc, Mapping)
                else records_doc
            )
            result = build_truth_v2(
                protocol, _sequence(records, "truth records")
            )
            _write_or_print(result, args.output)
        elif args.command == "reveal":
            protocol = validate_protocol_v2(
                _load_json(args.protocol, "v2 protocol"), verify_sources=True
            )
            commitment = validate_commitment_v2(
                _load_json(args.commitment, "commitment"), protocol
            )
            truth = validate_independent_erosion_truth(
                _load_json(args.truth, "truth"),
                protocol=protocol["core_protocol"],
            )
            validate_truth_coverage(protocol, truth)
            result = build_reveal_receipt(
                protocol["core_protocol"], commitment, truth
            )
            _write_or_print(result, args.output)
        elif args.command == "accept":
            result = accept_independent_erosion_delivery(
                args.protocol,
                args.commitment,
                args.truth,
                args.reveal_receipt,
                args.pin_envelope,
                truth_records=args.truth_records,
                trusted_public_key=args.trusted_public_key,
                test_hmac_key=args.test_hmac_key,
            )
            _write_or_print(result, args.output)
            if result.get("status") == "blocked":
                return 2
        elif args.command == "status":
            result = delivery_status(
                args.protocol,
                commitment=args.commitment,
                pin_envelope=args.pin_envelope,
            )
            _write_or_print(result, args.output)
        else:
            raise IndependentErosionDeliveryError("unknown command")
        return 0
    except (OSError, ValidationError, ValueError, TypeError, KeyError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "formal": False,
                    "reason": str(exc) or type(exc).__name__,
                },
                ensure_ascii=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
