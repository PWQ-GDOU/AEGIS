"""Frozen independent spatial erosion truth and prediction commitments.

Protocol construction verifies measured truth files. Prediction commitment
construction uses only frozen protocol metadata and never opens truth files.
Splits depend exclusively on preregistered identities and an integer seed;
source hashes bind bytes but are not partitioning keys.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from step30_validation_common import (
    ValidationError,
    canonical_sha256,
    jsonable,
    safe_json_dump,
    safe_json_load,
    sha256_file,
)


PROTOCOL_SCHEMA_VERSION = "step30-independent-erosion-protocol-v1"
SCHEMA_VERSION = PROTOCOL_SCHEMA_VERSION
VERSION = PROTOCOL_SCHEMA_VERSION
SPLIT_CONTRACT_VERSION = "step30-independent-erosion-split-v1"
METRIC_CONTRACT_VERSION = "step30-independent-erosion-metrics-v1"
PREDICTION_COMMITMENT_VERSION = (
    "step30-independent-erosion-prediction-commitment-v1"
)
COMMITMENT_VERSION = PREDICTION_COMMITMENT_VERSION
BLOCKED_REPORT_VERSION = "step30-independent-erosion-blocked-report-v1"
DEFAULT_SPLIT_SEED = 30

PARTITIONS = ("train", "calibration", "independent_test")
IDENTITY_FIELDS = (
    "record_id",
    "asset_id",
    "component_id",
    "service_episode_id",
    "campaign_id",
    "scan_id",
)
HARD_GROUP_FIELDS = ("asset_id", "service_episode_id")
FORBIDDEN_SPLIT_FIELD_TOKENS = (
    "wear",
    "depth",
    "label",
    "observation",
    "value",
    "metric",
)
SPLIT_RATIOS = {
    "train": 0.6,
    "calibration": 0.2,
    "independent_test": 0.2,
}

# Exact comparison against this constant prevents a caller from relaxing a
# threshold and merely recomputing the outer protocol hash after evaluation.
REGISTERED_SPATIAL_METRICS = {
    "wear_depth_mm": {
        "units": "mm",
        "error_measure": "relative_error",
        "maximum_error_fraction": 0.05,
    },
    "total_volume_mm3": {
        "units": "mm3",
        "error_measure": "relative_error",
        "maximum_error_fraction": 0.05,
    },
    "max_depth_mm": {
        "units": "mm",
        "error_measure": "relative_error",
        "maximum_error_fraction": 0.05,
    },
    "hotspot_area_mm2": {
        "units": "mm2",
        "error_measure": "relative_error",
        "maximum_error_fraction": 0.05,
    },
    "hotspot_iou_error": {
        "units": "dimensionless",
        "error_measure": "absolute_error",
        "maximum_error_fraction": 0.05,
    },
}

_HEX = frozenset("0123456789abcdef")
_ASSET_KEYS = frozenset(
    (*IDENTITY_FIELDS, "status", "path", "sha256", "bytes", "group_id", "partition")
)
_ASSET_KEYS_WITH_TRUTH_SOURCE = frozenset((*_ASSET_KEYS, "truth_source"))
_PROTOCOL_KEYS = frozenset(
    {
        "version",
        "schema_version",
        "status",
        "immutable",
        "official_reference_gate",
        "split_contract",
        "metric_contract",
        "n_records",
        "n_independent_groups",
        "assets",
        "groups",
        "splits",
        "protocol_sha256",
    }
)
_PREDICTION_KEYS = frozenset(
    {
        "record_id",
        "parent_face_id",
        "prediction_value",
        "value_units",
        "parent_face_area_mm2",
        "area_units",
    }
)
_COMMITMENT_KEYS = frozenset(
    {
        "version",
        "protocol_schema_version",
        "protocol_sha256",
        "model_descriptor",
        "model_descriptor_sha256",
        "topology_binding",
        "topology_binding_sha256",
        "independent_test_record_ids",
        "independent_test_cohort_sha256",
        "value_metric",
        "value_units",
        "area_units",
        "predictions",
        "predictions_sha256",
        "official_reference_gate",
        "commitment_sha256",
    }
)
_BLOCKED_KEYS = frozenset(
    {
        "version",
        "schema_version",
        "status",
        "stage",
        "reason",
        "details",
        "official_reference_gate",
        "report_sha256",
    }
)


class IndependentErosionProtocolError(ValidationError):
    """Raised when the independent truth protocol fails closed."""


class PredictionCommitmentError(ValidationError):
    """Raised when a prediction commitment is incomplete or mutable."""


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value.lower())
    )


def _required_text(value: Any, name: str, error_type: type[ValidationError]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error_type(f"{name} must be a non-empty string")
    return value.strip()


def _sequence(value: Any, name: str, error_type: type[ValidationError]) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise error_type(f"{name} must be an array")
    return list(value)


def _metric_contract() -> dict[str, Any]:
    return {
        "version": METRIC_CONTRACT_VERSION,
        "preregistered": True,
        "posthoc_relaxation_allowed": False,
        "threshold_ceiling_fraction": 0.05,
        "thresholds": {
            name: metric["maximum_error_fraction"]
            for name, metric in REGISTERED_SPATIAL_METRICS.items()
        },
        "hotspot_percentile": 90.0,
        "metrics": jsonable(REGISTERED_SPATIAL_METRICS),
    }


def _split_contract(seed: int) -> dict[str, Any]:
    return {
        "version": SPLIT_CONTRACT_VERSION,
        "seed": seed,
        "partitions": list(PARTITIONS),
        "ratios": dict(SPLIT_RATIOS),
        "identity_fields": list(IDENTITY_FIELDS),
        "hard_group_fields": list(HARD_GROUP_FIELDS),
        "forbidden_field_tokens": list(FORBIDDEN_SPLIT_FIELD_TOKENS),
        "source_hash_role": "integrity_binding_only_not_partitioning",
    }


def source_file_descriptor(path: str | Path) -> dict[str, Any]:
    """Describe one existing truth source without interpreting its content."""
    target = Path(path).resolve()
    if not target.is_file():
        raise IndependentErosionProtocolError(f"truth source file does not exist: {target}")
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "bytes": int(target.stat().st_size),
    }


def _declared_source(record: Mapping[str, Any]) -> Mapping[str, Any]:
    source = record.get("source")
    if source is not None:
        if not isinstance(source, Mapping):
            raise IndependentErosionProtocolError("asset source must be an object")
        return source
    return record


def _normalize_asset_input(
    record: Any,
    *,
    source_base: str | Path | None,
    verify_source: bool,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise IndependentErosionProtocolError("each asset must be an object")
    normalized = {
        field: _required_text(
            record.get(field), f"asset {field}", IndependentErosionProtocolError
        )
        for field in IDENTITY_FIELDS
    }
    if record.get("status") != "measured":
        raise IndependentErosionProtocolError("asset status must be measured")

    source = _declared_source(record)
    raw_path = _required_text(
        source.get("path"), "asset source path", IndependentErosionProtocolError
    )
    expected_hash = source.get("sha256")
    expected_bytes = source.get("bytes")
    if not _is_sha256(expected_hash):
        raise IndependentErosionProtocolError(
            "asset source sha256 must be a 64-character digest"
        )
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
    ):
        raise IndependentErosionProtocolError(
            "asset source bytes must be a non-negative integer"
        )

    path = Path(raw_path)
    if not path.is_absolute() and source_base is not None:
        path = Path(source_base) / path
    path = path.resolve()
    if verify_source:
        if not path.is_file():
            raise IndependentErosionProtocolError(
                f"truth source file does not exist: {path}"
            )
        if int(path.stat().st_size) != expected_bytes:
            raise IndependentErosionProtocolError(
                f"truth source byte count mismatch: {path}"
            )
        if sha256_file(path) != str(expected_hash).lower():
            raise IndependentErosionProtocolError(f"truth source hash mismatch: {path}")

    normalized.update(
        {
            "status": "measured",
            "path": str(path),
            "sha256": str(expected_hash).lower(),
            "bytes": expected_bytes,
        }
    )

    # A surface/thickness record has a second, immutable worn/after source.
    # It is optional for legacy protocol fixtures, but required by the truth
    # builder before a record can be used as formal independent evidence.
    if record.get("truth_source") is not None:
        truth_source = record.get("truth_source")
        if not isinstance(truth_source, Mapping):
            raise IndependentErosionProtocolError(
                "asset truth_source must be an object"
            )
        truth_path_raw = _required_text(
            truth_source.get("path"),
            "asset truth_source path",
            IndependentErosionProtocolError,
        )
        truth_hash = truth_source.get("sha256")
        truth_bytes = truth_source.get("bytes")
        if not _is_sha256(truth_hash):
            raise IndependentErosionProtocolError(
                "asset truth_source sha256 must be a 64-character digest"
            )
        if isinstance(truth_bytes, bool) or not isinstance(truth_bytes, int) or truth_bytes < 0:
            raise IndependentErosionProtocolError(
                "asset truth_source bytes must be a non-negative integer"
            )
        truth_path = Path(truth_path_raw)
        if not truth_path.is_absolute() and source_base is not None:
            truth_path = Path(source_base) / truth_path
        truth_path = truth_path.resolve()
        if verify_source:
            if not truth_path.is_file():
                raise IndependentErosionProtocolError(
                    f"truth source file does not exist: {truth_path}"
                )
            if truth_path.stat().st_size != truth_bytes:
                raise IndependentErosionProtocolError(
                    f"truth source byte count mismatch: {truth_path}"
                )
            if sha256_file(truth_path) != str(truth_hash).lower():
                raise IndependentErosionProtocolError(
                    f"truth source hash mismatch: {truth_path}"
                )
        normalized["truth_source"] = {
            "path": str(truth_path),
            "sha256": str(truth_hash).lower(),
            "bytes": truth_bytes,
        }
    return normalized


def _identity_projection(asset: Mapping[str, Any]) -> dict[str, str]:
    """Return the only record fields allowed to influence group assignment."""
    return {field: str(asset[field]) for field in IDENTITY_FIELDS}


def _derive_groups(assets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    union = _UnionFind(len(assets))
    owners: dict[tuple[str, str], int] = {}
    for index, asset in enumerate(assets):
        for field in HARD_GROUP_FIELDS:
            token = (field, str(asset[field]))
            previous = owners.get(token)
            if previous is None:
                owners[token] = index
            else:
                union.union(index, previous)

    components: dict[int, list[Mapping[str, Any]]] = {}
    for index, asset in enumerate(assets):
        components.setdefault(union.find(index), []).append(asset)

    groups: list[dict[str, Any]] = []
    for members in components.values():
        identity_records = sorted(
            (_identity_projection(asset) for asset in members),
            key=lambda item: item["record_id"],
        )
        group_hash = canonical_sha256({"identity_records": identity_records})
        groups.append(
            {
                "group_id": f"group-{group_hash}",
                "record_ids": sorted(str(asset["record_id"]) for asset in members),
                "asset_ids": sorted({str(asset["asset_id"]) for asset in members}),
                "service_episode_ids": sorted(
                    {str(asset["service_episode_id"]) for asset in members}
                ),
                "n_records": len(members),
            }
        )
    return sorted(groups, key=lambda item: item["group_id"])


def _seeded_group_key(seed: int, group_id: str) -> str:
    # group_id is derived exclusively from preregistered identities.
    return hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()


def _assign_groups(groups: Sequence[Mapping[str, Any]], seed: int) -> dict[str, str]:
    if len(groups) < 3:
        raise IndependentErosionProtocolError(
            "at least three independent asset/service-episode groups are required"
        )
    ordered = sorted(
        groups,
        key=lambda item: _seeded_group_key(seed, str(item["group_id"])),
    )
    assignment: dict[str, str] = {}
    counts = {partition: 0 for partition in PARTITIONS}
    total_records = sum(int(group["n_records"]) for group in groups)
    targets = {
        partition: total_records * SPLIT_RATIOS[partition] for partition in PARTITIONS
    }
    for group, partition in zip(ordered[:3], PARTITIONS):
        assignment[str(group["group_id"])] = partition
        counts[partition] += int(group["n_records"])
    for group in ordered[3:]:
        partition = min(
            PARTITIONS,
            key=lambda name: (
                counts[name] / max(targets[name], 1e-12),
                counts[name],
                PARTITIONS.index(name),
            ),
        )
        assignment[str(group["group_id"])] = partition
        counts[partition] += int(group["n_records"])
    if any(counts[partition] == 0 for partition in PARTITIONS):
        raise IndependentErosionProtocolError("split contains an empty partition")
    return assignment


def _partitioned_assets(
    assets: Sequence[Mapping[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    groups = _derive_groups(assets)
    assignment = _assign_groups(groups, seed)
    record_group = {
        record_id: str(group["group_id"])
        for group in groups
        for record_id in group["record_ids"]
    }
    partitioned = []
    for asset in assets:
        group_id = record_group[str(asset["record_id"])]
        partitioned.append(
            {
                **dict(asset),
                "group_id": group_id,
                "partition": assignment[group_id],
            }
        )
    partitioned.sort(key=lambda item: item["record_id"])
    frozen_groups = [
        {**dict(group), "partition": assignment[str(group["group_id"])]}
        for group in groups
    ]
    frozen_groups.sort(key=lambda item: item["group_id"])
    splits = {
        partition: sorted(
            str(asset["record_id"])
            for asset in partitioned
            if asset["partition"] == partition
        )
        for partition in PARTITIONS
    }
    return partitioned, frozen_groups, splits


def recompute_protocol_hash(protocol: Mapping[str, Any]) -> str:
    unsigned = dict(jsonable(protocol))
    unsigned.pop("protocol_sha256", None)
    return canonical_sha256(unsigned)


def build_protocol(
    assets: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    source_base: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze measured source identities, a deterministic split, and metrics."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise IndependentErosionProtocolError("split seed must be an integer")
    records = _sequence(assets, "assets", IndependentErosionProtocolError)
    if not records:
        raise IndependentErosionProtocolError("assets must not be empty")
    normalized = [
        _normalize_asset_input(
            record, source_base=source_base, verify_source=True
        )
        for record in records
    ]
    record_ids = [record["record_id"] for record in normalized]
    if len(set(record_ids)) != len(record_ids):
        raise IndependentErosionProtocolError("asset record_id values must be unique")
    frozen_assets, groups, splits = _partitioned_assets(normalized, seed)
    result = {
        "version": PROTOCOL_SCHEMA_VERSION,
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "status": "measured",
        "immutable": True,
        "official_reference_gate": False,
        "split_contract": _split_contract(seed),
        "metric_contract": _metric_contract(),
        "n_records": len(frozen_assets),
        "n_independent_groups": len(groups),
        "assets": frozen_assets,
        "groups": groups,
        "splits": splits,
    }
    result["protocol_sha256"] = recompute_protocol_hash(result)
    return validate_protocol(result)


def _validate_asset_from_protocol(
    asset: Any, *, verify_source: bool
) -> dict[str, Any]:
    if not isinstance(asset, Mapping) or set(asset) not in {
        _ASSET_KEYS,
        _ASSET_KEYS_WITH_TRUTH_SOURCE,
    }:
        raise IndependentErosionProtocolError("protocol asset fields are invalid")
    normalized = _normalize_asset_input(
        asset, source_base=None, verify_source=verify_source
    )
    group_id = _required_text(
        asset.get("group_id"), "asset group_id", IndependentErosionProtocolError
    )
    partition = asset.get("partition")
    if partition not in PARTITIONS:
        raise IndependentErosionProtocolError("asset partition is invalid")
    normalized.update({"group_id": group_id, "partition": partition})
    if normalized != dict(asset):
        raise IndependentErosionProtocolError("protocol asset is not canonical")
    return normalized


def _assert_no_identity_leak(assets: Sequence[Mapping[str, Any]]) -> None:
    for field in HARD_GROUP_FIELDS:
        partitions: dict[str, set[str]] = {}
        for asset in assets:
            partitions.setdefault(str(asset[field]), set()).add(str(asset["partition"]))
        leaked = sorted(
            identity for identity, values in partitions.items() if len(values) > 1
        )
        if leaked:
            raise IndependentErosionProtocolError(
                f"{field} leakage across partitions: {leaked}"
            )


def _validate_protocol(
    protocol: Any, *, verify_sources: bool
) -> dict[str, Any]:
    """Validate immutable content, optionally without opening truth bytes."""
    if not isinstance(protocol, Mapping) or set(protocol) != _PROTOCOL_KEYS:
        raise IndependentErosionProtocolError(
            "independent erosion protocol fields are invalid"
        )
    if (
        protocol.get("version") != PROTOCOL_SCHEMA_VERSION
        or protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION
    ):
        raise IndependentErosionProtocolError(
            "independent erosion protocol version mismatch"
        )
    if protocol.get("status") != "measured":
        raise IndependentErosionProtocolError("protocol status must be measured")
    if protocol.get("immutable") is not True:
        raise IndependentErosionProtocolError("protocol must be immutable")
    if protocol.get("official_reference_gate") is not False:
        raise IndependentErosionProtocolError(
            "official_reference_gate must remain false"
        )
    stored_hash = protocol.get("protocol_sha256")
    if (
        not _is_sha256(stored_hash)
        or recompute_protocol_hash(protocol) != str(stored_hash).lower()
    ):
        raise IndependentErosionProtocolError("protocol_sha256/content mismatch")

    split_contract = protocol.get("split_contract")
    if not isinstance(split_contract, Mapping):
        raise IndependentErosionProtocolError("split contract is missing")
    seed = split_contract.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise IndependentErosionProtocolError("split seed must be an integer")
    if dict(split_contract) != _split_contract(seed):
        raise IndependentErosionProtocolError(
            "split contract is not the preregistered identity-only contract"
        )
    if protocol.get("metric_contract") != _metric_contract():
        raise IndependentErosionProtocolError(
            "spatial metric contract or frozen five-percent thresholds changed"
        )

    assets_raw = _sequence(
        protocol.get("assets"), "protocol assets", IndependentErosionProtocolError
    )
    if not assets_raw:
        raise IndependentErosionProtocolError("protocol assets must not be empty")
    assets = [
        _validate_asset_from_protocol(asset, verify_source=verify_sources)
        for asset in assets_raw
    ]
    if assets != sorted(assets, key=lambda item: item["record_id"]):
        raise IndependentErosionProtocolError(
            "protocol assets are not canonically sorted"
        )
    record_ids = [asset["record_id"] for asset in assets]
    if len(set(record_ids)) != len(record_ids):
        raise IndependentErosionProtocolError("asset record_id values must be unique")
    _assert_no_identity_leak(assets)

    identity_and_sources = []
    for asset in assets:
        item = {
            key: asset[key]
            for key in (*IDENTITY_FIELDS, "status", "path", "sha256", "bytes")
        }
        if "truth_source" in asset:
            item["truth_source"] = asset["truth_source"]
        identity_and_sources.append(item)
    expected_assets, expected_groups, expected_splits = _partitioned_assets(
        identity_and_sources, seed
    )
    if assets != expected_assets:
        raise IndependentErosionProtocolError(
            "stored split differs from identity-and-seed recomputation"
        )
    if protocol.get("groups") != expected_groups:
        raise IndependentErosionProtocolError("stored independent groups are invalid")
    if protocol.get("splits") != expected_splits:
        raise IndependentErosionProtocolError("stored split cohorts are invalid")
    if any(not expected_splits[partition] for partition in PARTITIONS):
        raise IndependentErosionProtocolError(
            "train/calibration/independent_test must be non-empty"
        )
    if protocol.get("n_records") != len(assets):
        raise IndependentErosionProtocolError("protocol n_records mismatch")
    if (
        protocol.get("n_independent_groups") != len(expected_groups)
        or len(expected_groups) < 3
    ):
        raise IndependentErosionProtocolError(
            "at least three independent groups are required"
        )
    return dict(protocol)


def validate_protocol(protocol: Any) -> dict[str, Any]:
    """Validate protocol content and every measured truth source fail-closed."""
    return _validate_protocol(protocol, verify_sources=True)


def save_protocol(protocol: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    """Validate and write a protocol exactly once."""
    validated = validate_protocol(protocol)
    return safe_json_dump(validated, path)


def load_protocol(path: str | Path) -> dict[str, Any]:
    try:
        document = safe_json_load(path)
    except (OSError, ValidationError) as exc:
        if isinstance(exc, IndependentErosionProtocolError):
            raise
        raise IndependentErosionProtocolError(f"cannot load protocol: {path}") from exc
    return validate_protocol(document)


def _prediction_field(
    prediction: Mapping[str, Any], names: tuple[str, ...], label: str
) -> Any:
    present = [name for name in names if name in prediction]
    if len(present) != 1:
        raise PredictionCommitmentError(
            f"prediction requires exactly one {label} field"
        )
    return prediction[present[0]]


def _finite_nonnegative(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise PredictionCommitmentError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PredictionCommitmentError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or result < 0.0 or (positive and result <= 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise PredictionCommitmentError(
            f"{label} must be finite and {qualifier}"
        )
    return result


def _normalize_predictions(
    predictions: Any,
    cohort: Sequence[str],
    *,
    require_canonical_keys: bool,
) -> list[dict[str, Any]]:
    rows = _sequence(predictions, "predictions", PredictionCommitmentError)
    if not rows:
        raise PredictionCommitmentError("predictions must not be empty")
    normalized: list[dict[str, Any]] = []
    seen_faces: set[tuple[str, int]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise PredictionCommitmentError("each prediction must be an object")
        if require_canonical_keys and set(row) != _PREDICTION_KEYS:
            raise PredictionCommitmentError("prediction fields are not canonical")
        record_id = _required_text(
            row.get("record_id"), "prediction record_id", PredictionCommitmentError
        )
        parent_face_id = row.get("parent_face_id")
        if (
            isinstance(parent_face_id, bool)
            or not isinstance(parent_face_id, int)
            or parent_face_id < 0
        ):
            raise PredictionCommitmentError(
                "parent_face_id must be a non-negative integer"
            )
        face_identity = (record_id, parent_face_id)
        if face_identity in seen_faces:
            raise PredictionCommitmentError(
                "(record_id, parent_face_id) values must be unique"
            )
        seen_faces.add(face_identity)
        value = _prediction_field(
            row,
            (
                "prediction_value",
                "predicted_value",
                "predicted_wear_depth_mm",
                "wear_depth_mm",
            ),
            "prediction value",
        )
        area = _prediction_field(
            row,
            ("parent_face_area_mm2", "area_mm2", "area"),
            "parent-face area",
        )
        value_units = _prediction_field(
            row, ("value_units", "units", "unit"), "value units"
        )
        area_units = _prediction_field(
            row, ("area_units", "area_unit"), "area units"
        )
        if value_units != "mm":
            raise PredictionCommitmentError("prediction value_units must be mm")
        if area_units != "mm2":
            raise PredictionCommitmentError("prediction area_units must be mm2")
        normalized.append(
            {
                "record_id": record_id,
                "parent_face_id": parent_face_id,
                "prediction_value": _finite_nonnegative(
                    value, "prediction value"
                ),
                "value_units": "mm",
                "parent_face_area_mm2": _finite_nonnegative(
                    area, "parent-face area", positive=True
                ),
                "area_units": "mm2",
            }
        )
    cohort_set = set(cohort)
    prediction_records = {row["record_id"] for row in normalized}
    if prediction_records != cohort_set:
        missing = sorted(cohort_set - prediction_records)
        unexpected = sorted(prediction_records - cohort_set)
        raise PredictionCommitmentError(
            f"prediction cohort mismatch; missing={missing}, unexpected={unexpected}"
        )
    return sorted(
        normalized,
        key=lambda item: (str(item["record_id"]), item["parent_face_id"]),
    )


def recompute_commitment_hash(commitment: Mapping[str, Any]) -> str:
    unsigned = dict(jsonable(commitment))
    unsigned.pop("commitment_sha256", None)
    return canonical_sha256(unsigned)


def build_prediction_commitment(
    protocol: Mapping[str, Any],
    model_descriptor: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    *,
    topology_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Commit predictions without opening or hashing a measured truth source."""
    frozen = _validate_protocol(protocol, verify_sources=False)
    if not isinstance(model_descriptor, Mapping) or not model_descriptor:
        raise PredictionCommitmentError(
            "model_descriptor must be a non-empty object"
        )
    descriptor = jsonable(model_descriptor)
    if not isinstance(descriptor, dict) or not descriptor:
        raise PredictionCommitmentError(
            "model_descriptor must be a non-empty object"
        )
    cohort = list(frozen["splits"]["independent_test"])
    normalized = _normalize_predictions(
        predictions, cohort, require_canonical_keys=False
    )
    if topology_binding is None:
        topology = None
    else:
        if not isinstance(topology_binding, Mapping):
            raise PredictionCommitmentError("topology_binding must be an object")
        topology = {
            "topology_v3_sha256": _required_text(
                topology_binding.get("topology_v3_sha256"),
                "topology_v3_sha256",
                PredictionCommitmentError,
            ).lower(),
            "mesh_signature": _required_text(
                topology_binding.get("mesh_signature"),
                "mesh_signature",
                PredictionCommitmentError,
            ).lower(),
            "artifact_set_id": _required_text(
                topology_binding.get("artifact_set_id"),
                "artifact_set_id",
                PredictionCommitmentError,
            ).lower(),
            "n_parent_faces": topology_binding.get("n_parent_faces"),
        }
        for field in ("topology_v3_sha256", "mesh_signature", "artifact_set_id"):
            if not _is_sha256(topology[field]):
                raise PredictionCommitmentError(f"{field} must be a SHA-256 digest")
        if (
            isinstance(topology["n_parent_faces"], bool)
            or not isinstance(topology["n_parent_faces"], int)
            or topology["n_parent_faces"] <= 0
        ):
            raise PredictionCommitmentError("n_parent_faces must be a positive integer")
        if any(row["parent_face_id"] >= topology["n_parent_faces"] for row in normalized):
            raise PredictionCommitmentError("parent_face_id exceeds topology bounds")
    result = {
        "version": PREDICTION_COMMITMENT_VERSION,
        "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_sha256": frozen["protocol_sha256"],
        "model_descriptor": descriptor,
        "model_descriptor_sha256": canonical_sha256(descriptor),
        "topology_binding": topology,
        "topology_binding_sha256": (
            None if topology is None else canonical_sha256(topology)
        ),
        "independent_test_record_ids": cohort,
        "independent_test_cohort_sha256": canonical_sha256(
            {"record_ids": cohort}
        ),
        "value_metric": "wear_depth_mm",
        "value_units": "mm",
        "area_units": "mm2",
        "predictions": normalized,
        "predictions_sha256": canonical_sha256({"predictions": normalized}),
        "official_reference_gate": False,
    }
    result["commitment_sha256"] = recompute_commitment_hash(result)
    return validate_prediction_commitment(result, frozen)


def validate_prediction_commitment(
    commitment: Any, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a commitment against metadata without reading measured truth."""
    frozen = _validate_protocol(protocol, verify_sources=False)
    if not isinstance(commitment, Mapping) or set(commitment) != _COMMITMENT_KEYS:
        raise PredictionCommitmentError(
            "prediction commitment fields are invalid"
        )
    if commitment.get("version") != PREDICTION_COMMITMENT_VERSION:
        raise PredictionCommitmentError("prediction commitment version mismatch")
    if commitment.get("protocol_schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise PredictionCommitmentError("prediction protocol schema mismatch")
    if commitment.get("protocol_sha256") != frozen["protocol_sha256"]:
        raise PredictionCommitmentError("prediction protocol hash mismatch")
    if commitment.get("official_reference_gate") is not False:
        raise PredictionCommitmentError(
            "official_reference_gate must remain false"
        )
    topology = commitment.get("topology_binding")
    topology_hash = commitment.get("topology_binding_sha256")
    if topology is None:
        if topology_hash is not None:
            raise PredictionCommitmentError(
                "topology_binding_sha256 requires topology_binding"
            )
    else:
        if not isinstance(topology, Mapping):
            raise PredictionCommitmentError("topology_binding must be an object")
        required = {
            "topology_v3_sha256",
            "mesh_signature",
            "artifact_set_id",
            "n_parent_faces",
        }
        if set(topology) != required:
            raise PredictionCommitmentError("topology_binding fields are invalid")
        for field in ("topology_v3_sha256", "mesh_signature", "artifact_set_id"):
            if not _is_sha256(topology.get(field)):
                raise PredictionCommitmentError(
                    f"topology_binding {field} must be a SHA-256 digest"
                )
        if (
            isinstance(topology.get("n_parent_faces"), bool)
            or not isinstance(topology.get("n_parent_faces"), int)
            or topology["n_parent_faces"] <= 0
        ):
            raise PredictionCommitmentError(
                "topology_binding n_parent_faces must be positive"
            )
        if not _is_sha256(topology_hash) or topology_hash != canonical_sha256(topology):
            raise PredictionCommitmentError("topology_binding_sha256 mismatch")
        predictions = commitment.get("predictions", [])
        if any(
            int(row.get("parent_face_id", -1)) >= topology["n_parent_faces"]
            for row in predictions
            if isinstance(row, Mapping)
        ):
            raise PredictionCommitmentError("prediction parent_face_id exceeds topology")
    if (
        commitment.get("value_metric"),
        commitment.get("value_units"),
        commitment.get("area_units"),
    ) != ("wear_depth_mm", "mm", "mm2"):
        raise PredictionCommitmentError("prediction metric or units changed")
    stored_hash = commitment.get("commitment_sha256")
    if (
        not _is_sha256(stored_hash)
        or recompute_commitment_hash(commitment) != str(stored_hash).lower()
    ):
        raise PredictionCommitmentError("commitment_sha256/content mismatch")

    descriptor = commitment.get("model_descriptor")
    if not isinstance(descriptor, Mapping) or not descriptor:
        raise PredictionCommitmentError("model_descriptor is missing")
    if commitment.get("model_descriptor_sha256") != canonical_sha256(descriptor):
        raise PredictionCommitmentError("model_descriptor_sha256 mismatch")
    cohort = list(frozen["splits"]["independent_test"])
    if commitment.get("independent_test_record_ids") != cohort:
        raise PredictionCommitmentError(
            "complete independent_test cohort is not bound"
        )
    if commitment.get("independent_test_cohort_sha256") != canonical_sha256(
        {"record_ids": cohort}
    ):
        raise PredictionCommitmentError("independent_test cohort hash mismatch")
    predictions = _normalize_predictions(
        commitment.get("predictions"), cohort, require_canonical_keys=True
    )
    if predictions != commitment.get("predictions"):
        raise PredictionCommitmentError(
            "predictions must be sorted by ascending parent_face_id"
        )
    if commitment.get("predictions_sha256") != canonical_sha256(
        {"predictions": predictions}
    ):
        raise PredictionCommitmentError("predictions_sha256 mismatch")
    return dict(commitment)


def save_prediction_commitment(
    commitment: Mapping[str, Any],
    path: str | Path,
    *,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_prediction_commitment(commitment, protocol)
    return safe_json_dump(validated, path)


def load_prediction_commitment(
    path: str | Path, *, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        document = safe_json_load(path)
    except (OSError, ValidationError) as exc:
        if isinstance(exc, PredictionCommitmentError):
            raise
        raise PredictionCommitmentError(
            f"cannot load prediction commitment: {path}"
        ) from exc
    return validate_prediction_commitment(document, protocol)


def recompute_blocked_report_hash(report: Mapping[str, Any]) -> str:
    unsigned = dict(jsonable(report))
    unsigned.pop("report_sha256", None)
    return canonical_sha256(unsigned)


def build_blocked_report(
    reason: str,
    *,
    stage: str = "independent_erosion_protocol",
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a hash-bound report that can never claim an official gate."""
    result = {
        "version": BLOCKED_REPORT_VERSION,
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "status": "blocked",
        "stage": _required_text(
            stage, "blocked stage", IndependentErosionProtocolError
        ),
        "reason": _required_text(
            reason, "blocked reason", IndependentErosionProtocolError
        ),
        "details": jsonable(details or {}),
        "official_reference_gate": False,
    }
    result["report_sha256"] = recompute_blocked_report_hash(result)
    return result


def validate_blocked_report(report: Any) -> dict[str, Any]:
    if not isinstance(report, Mapping) or set(report) != _BLOCKED_KEYS:
        raise IndependentErosionProtocolError("blocked report fields are invalid")
    if (
        report.get("version") != BLOCKED_REPORT_VERSION
        or report.get("schema_version") != PROTOCOL_SCHEMA_VERSION
        or report.get("status") != "blocked"
        or report.get("official_reference_gate") is not False
    ):
        raise IndependentErosionProtocolError("blocked report contract is invalid")
    _required_text(
        report.get("stage"), "blocked stage", IndependentErosionProtocolError
    )
    _required_text(
        report.get("reason"), "blocked reason", IndependentErosionProtocolError
    )
    if not isinstance(report.get("details"), Mapping):
        raise IndependentErosionProtocolError("blocked report details must be an object")
    if report.get("report_sha256") != recompute_blocked_report_hash(report):
        raise IndependentErosionProtocolError("blocked report hash mismatch")
    return dict(report)


def save_blocked_report(report: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    return safe_json_dump(validate_blocked_report(report), path)


def load_blocked_report(path: str | Path) -> dict[str, Any]:
    try:
        document = safe_json_load(path)
    except (OSError, ValidationError) as exc:
        raise IndependentErosionProtocolError(
            f"cannot load blocked report: {path}"
        ) from exc
    return validate_blocked_report(document)


# Explicit aliases keep artifact types obvious at integration call sites.
build_independent_erosion_protocol = build_protocol
validate_independent_erosion_protocol = validate_protocol
save_independent_erosion_protocol = save_protocol
load_independent_erosion_protocol = load_protocol
build_commitment = build_prediction_commitment
validate_commitment = validate_prediction_commitment
save_commitment = save_prediction_commitment
load_commitment = load_prediction_commitment
