"""Leakage-resistant group-disjoint validation splits for Step30 data.

The module treats provenance as part of the data contract.  Records are grouped
by material ``campaign/specimen`` identities or bucket
``asset/component/service_episode/scan`` identities.  Records sharing any group
identity or source hash are kept in one connected group.  This is deliberately
stricter than assigning a single row-level group label: a campaign, asset, or
source file cannot leak into another partition through a different companion
identifier.

The public API is intentionally dependency-light so it can be used by the
existing Step30 scripts and by command-line checks:

* :func:`group_disjoint_split` creates train/calibration/test assignments.
* :func:`leave_one_group_out` creates one held-out-group fold at a time.
* :func:`write_validation_manifest` writes a canonical, immutable manifest.

Every record must carry at least one real SHA-256 source hash.  A hash is never
inferred from a row index, and missing provenance fails closed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from step30_validation_common import (
    ValidationError,
    canonical_json,
    canonical_sha256,
    jsonable,
    safe_json_dump,
    sha256_file,
)


VERSION = "step30-validation-split-v1"
MANIFEST_VERSION = "step30-validation-manifest-v1"
SCHEME_THREE_WAY = "group_disjoint_train_calibration_test"
SCHEME_LOGO = "leave_one_group_out"
PARTITIONS = ("train", "calibration", "test")

_MATERIAL_FIELDS = ("campaign_id", "specimen_id", "material_batch")
_BUCKET_FIELDS = ("asset", "component", "service_episode", "scan")
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "campaign_id": ("campaign_id", "campaign", "campaignid"),
    "specimen_id": ("specimen_id", "specimen", "coupon_id", "coupon", "specimenid"),
    "material_batch": ("material_batch", "materialbatch", "batch", "heat_id", "heat"),
    "asset": ("asset", "asset_id", "assetid"),
    "component": ("component", "component_id", "componentid", "part"),
    "service_episode": (
        "service_episode", "service_episode_id", "serviceepisode", "episode",
    ),
    "scan": ("scan", "scan_id", "scanid", "scan_number"),
    "group_id": ("group_id", "split_group", "batch_id", "group"),
}
_SOURCE_KEYS = (
    "source_hash", "source_sha256", "source_file_sha256", "source_hashes",
)
_ASSIGNMENT_KEYS = {
    "split", "partition", "fold", "fold_id", "record_hash", "group_hash",
    "group_id", "_split", "_partition",
}
_HEX = frozenset("0123456789abcdef")


class SplitError(ValidationError):
    """Raised when a split cannot satisfy the provenance contract."""


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _load_document(path: str | Path) -> Any:
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise SplitError(f"cannot read input: {target}") from exc
    if target.suffix.lower() == ".csv":
        try:
            rows = list(csv.DictReader(text.splitlines()))
        except (csv.Error, UnicodeError) as exc:
            raise SplitError(f"invalid CSV input: {target}") from exc
        if not rows or not csv.DictReader(text.splitlines()).fieldnames:
            raise SplitError("CSV input requires a named non-empty header and rows")
        return rows
    if target.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SplitError(f"invalid JSONL line {line_number}: {target}") from exc
            if not isinstance(value, Mapping):
                raise SplitError(f"JSONL line {line_number} is not an object")
            rows.append(dict(value))
        return rows
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SplitError(f"invalid JSON input: {target}") from exc


def _records_from_value(value: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract rows while retaining top-level provenance defaults."""
    if isinstance(value, (str, Path)):
        value = _load_document(value)
    defaults: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key in ("source", "source_manifest", "dataset", "provenance", "status"):
            if key in value:
                defaults[key] = value[key]
        rows_value = value.get("records", value.get("rows", value.get("data")))
        if rows_value is None and all(isinstance(item, Mapping) for item in value.values()):
            rows_value = list(value.values())
    else:
        rows_value = value
    if not isinstance(rows_value, Sequence) or isinstance(rows_value, (str, bytes)):
        raise SplitError("input must contain a records/rows list")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows_value, 1):
        if not isinstance(row, Mapping):
            raise SplitError(f"record {index} is not an object")
        rows.append(dict(row))
    if not rows:
        raise SplitError("input contains no records")
    return rows, defaults


def _normal_key(value: Any) -> str:
    if value is None or isinstance(value, bool):
        raise SplitError("group/source identity cannot be null or boolean")
    text = str(value).strip()
    if not text:
        raise SplitError("group/source identity cannot be empty")
    return text


def _find_field(row: Mapping[str, Any], logical_name: str) -> tuple[str, Any] | None:
    aliases = _FIELD_ALIASES.get(logical_name, (logical_name,))
    normalized = {
        str(key).strip().lower().replace("-", "_").replace(" ", "_"): key
        for key in row
    }
    for alias in aliases:
        key = normalized.get(alias.lower().replace("-", "_").replace(" ", "_"))
        if key is not None and row.get(key) not in (None, ""):
            return str(key), row[key]
    return None


def _hash_value(value: Any, label: str) -> str:
    if not isinstance(value, str):
        value = str(value)
    value = value.strip().lower()
    if len(value) != 64 or any(character not in _HEX for character in value):
        raise SplitError(f"{label} must be a 64-character SHA-256 hex digest")
    return value


def _hashes_from_value(value: Any, label: str) -> list[str]:
    if isinstance(value, Mapping):
        values = list(value.values())
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = list(value)
    else:
        values = [value]
    hashes = sorted({_hash_value(item, label) for item in values})
    if not hashes:
        raise SplitError(f"{label} is empty")
    return hashes


def _source_hashes(
    row: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    source_base: Path | None = None,
) -> list[str]:
    values: list[str] = []
    for key in _SOURCE_KEYS:
        if key in row and row[key] not in (None, "", [], {}):
            values.extend(_hashes_from_value(row[key], f"record {key}"))
    source_value = row.get("source")
    if isinstance(source_value, Mapping):
        for key in ("sha256", "source_sha256", "hash"):
            if source_value.get(key) not in (None, ""):
                values.extend(_hashes_from_value(source_value[key], f"record source.{key}"))
        if not values and source_value.get("path"):
            path = Path(str(source_value["path"]))
            if not path.is_absolute() and source_base is not None:
                path = source_base / path
            if path.is_file():
                values.append(sha256_file(path))
    if not values and row.get("provenance") and isinstance(row["provenance"], Mapping):
        provenance = row["provenance"]
        for key in ("source_sha256", "source_hash"):
            if provenance.get(key) not in (None, ""):
                values.extend(_hashes_from_value(provenance[key], f"record provenance.{key}"))
    if not values:
        default_source = defaults.get("source")
        if isinstance(default_source, Mapping):
            for key in ("sha256", "source_sha256", "hash"):
                if default_source.get(key) not in (None, ""):
                    values.extend(_hashes_from_value(default_source[key], f"dataset source.{key}"))
                    break
        elif default_source not in (None, ""):
            values.extend(_hashes_from_value(default_source, "dataset source"))
    if not values:
        raise SplitError("every record requires a source SHA-256 hash")
    return sorted(set(values))


def _record_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): jsonable(value)
        for key, value in row.items()
        if str(key) not in _ASSIGNMENT_KEYS and not str(key).startswith("_")
    }


def _resolve_group_fields(
    rows: Sequence[Mapping[str, Any]],
    domain: str | None,
    group_fields: Iterable[str] | str | None,
) -> tuple[str, tuple[str, ...]]:
    if group_fields is not None:
        if isinstance(group_fields, str):
            selected = tuple(item.strip() for item in group_fields.split(",") if item.strip())
        else:
            selected = tuple(str(item).strip() for item in group_fields if str(item).strip())
        if not selected:
            raise SplitError("group_fields cannot be empty")
        resolved_domain = str(domain).strip().lower() if domain is not None else "custom"
        return resolved_domain, selected
    if domain is None:
        keys = {str(key).strip().lower() for row in rows for key in row}
        domain = "material" if keys & set(_MATERIAL_FIELDS) else "bucket"
    domain = str(domain).strip().lower()
    if domain not in {"material", "bucket"}:
        raise SplitError("domain must be material or bucket")
    candidates = _MATERIAL_FIELDS if domain == "material" else _BUCKET_FIELDS
    present: list[str] = []
    for logical in candidates:
        if any(_find_field(row, logical) is not None for row in rows):
            present.append(logical)
    if not present:
        raise SplitError(f"no {domain} grouping fields were found")
    return domain, tuple(present)


def _prepare(
    value: Any,
    *,
    domain: str | None = None,
    group_fields: Iterable[str] | str | None = None,
    source_base: str | Path | None = None,
) -> dict[str, Any]:
    rows, defaults = _records_from_value(value)
    declared_status = defaults.get("status")
    row_statuses = {
        str(row["status"]).strip()
        for row in rows
        if row.get("status") not in (None, "")
    }
    if declared_status is None:
        if len(row_statuses) != 1:
            raise SplitError("records require one explicit measured, synthetic, or blocked status")
        declared_status = next(iter(row_statuses))
    elif row_statuses and row_statuses != {str(declared_status)}:
        raise SplitError("record statuses disagree with the dataset status")
    if declared_status not in {"measured", "synthetic", "blocked"}:
        raise SplitError("record status must be measured, synthetic, or blocked")
    defaults["status"] = declared_status
    base = Path(source_base).resolve() if source_base is not None else None
    resolved_domain, fields = _resolve_group_fields(rows, domain, group_fields)
    record_hashes: list[str] = []
    record_ids: list[str] = []
    source_hashes: list[list[str]] = []
    identities: list[list[tuple[str, str]]] = []
    seen_records: set[str] = set()
    seen_record_ids: set[str] = set()
    for index, row in enumerate(rows):
        record_hash = canonical_sha256(_record_identity(row))
        if record_hash in seen_records:
            raise SplitError(f"duplicate record hash at row {index + 1}: {record_hash}")
        seen_records.add(record_hash)
        record_hashes.append(record_hash)
        record_id_value = None
        for key in ("record_id", "id", "sample_id", "trial_id", "run_id"):
            if row.get(key) not in (None, ""):
                record_id_value = _normal_key(row[key])
                break
        if record_id_value is None:
            raise SplitError(f"record {index + 1} requires an explicit record_id")
        if record_id_value in seen_record_ids:
            raise SplitError(f"duplicate record_id: {record_id_value}")
        seen_record_ids.add(record_id_value)
        record_ids.append(record_id_value)
        hashes = _source_hashes(row, defaults, source_base=base)
        source_hashes.append(hashes)
        group_values: list[tuple[str, str]] = []
        for field in fields:
            found = _find_field(row, field)
            if found is None:
                raise SplitError(f"record {index + 1} is missing required group field {field}")
            group_values.append((field, _normal_key(found[1])))
        # The selected fields form one composite identity.  Using individual
        # fields as union keys would merge every bucket row sharing only
        # component=bucket or every material row sharing only a campaign.
        group_key = canonical_json(dict(group_values))
        row_identities: list[tuple[str, str]] = [("group_key", group_key)]
        row_identities.extend(group_values)
        explicit = _find_field(row, "group_id")
        if explicit is not None:
            row_identities.append(("group_id", _normal_key(explicit[1])))
        identities.append(row_identities)

    union = _UnionFind(len(rows))
    token_owner: dict[tuple[str, str], int] = {}
    for index, row_tokens in enumerate(identities):
        # Composite group identity, explicit group_id, and source hashes are
        # all hard linkage constraints.  Individual display identities above
        # are intentionally not used as union keys.
        tokens = [
            token for token in row_tokens if token[0] in {"group_key", "group_id"}
        ]
        # Material campaigns and batches are hard leakage boundaries. Bucket
        # component names are intentionally not global union keys because a
        # common component label can legitimately occur on many assets.
        if resolved_domain == "material":
            tokens.extend(
                token for token in row_tokens
                if token[0] in {"campaign_id", "material_batch"}
            )
        tokens += [("source_sha256", item) for item in source_hashes[index]]
        for token in tokens:
            previous = token_owner.get(token)
            if previous is not None:
                union.union(index, previous)
            else:
                token_owner[token] = index
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[union.find(index)].append(index)

    groups: list[dict[str, Any]] = []
    row_group: dict[int, str] = {}
    for indices in sorted(components.values(), key=lambda item: min(item)):
        group_sources = sorted({item for index in indices for item in source_hashes[index]})
        group_records = sorted(record_hashes[index] for index in indices)
        group_identities = sorted({
            f"{field}={value}" for index in indices for field, value in identities[index]
        })
        group_hash = canonical_sha256({
            "identities": group_identities,
            "record_hashes": group_records,
            "source_hashes": group_sources,
        })
        group_id = f"group-{group_hash}"
        group = {
            "group_id": group_id,
            "group_hash": group_hash,
            "record_hashes": group_records,
            "source_hashes": group_sources,
            "identities": group_identities,
            "n_records": len(indices),
        }
        groups.append(group)
        for index in indices:
            row_group[index] = group_id
    groups.sort(key=lambda item: item["group_hash"])
    return {
        "rows": rows,
        "defaults": defaults,
        "domain": resolved_domain,
        "group_fields": list(fields),
        "record_ids": record_ids,
        "record_hashes": record_hashes,
        "source_hashes": source_hashes,
        "row_group": row_group,
        "groups": groups,
    }


def _seeded_key(seed: int, group_hash: str) -> str:
    return hashlib.sha256(f"{int(seed)}:{group_hash}".encode("ascii")).hexdigest()


def _assign_groups(
    groups: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    ratios: Mapping[str, float],
    require_all: bool = True,
) -> dict[str, str]:
    if set(ratios) != set(PARTITIONS):
        raise SplitError("ratios must contain train, calibration, and test")
    values = {name: float(ratios[name]) for name in PARTITIONS}
    if any(value <= 0.0 for value in values.values()) or abs(sum(values.values()) - 1.0) > 1e-9:
        raise SplitError("split ratios must be positive and sum to one")
    ordered = sorted(groups, key=lambda item: _seeded_key(seed, str(item["group_hash"])))
    if require_all and len(ordered) < len(PARTITIONS):
        raise SplitError("at least three independent groups are required")
    assignment: dict[str, str] = {}
    counts = {name: 0 for name in PARTITIONS}
    target = {name: values[name] * sum(int(item["n_records"]) for item in groups) for name in PARTITIONS}
    reserved = list(PARTITIONS) if require_all else []
    for index, group in enumerate(ordered):
        group_id = str(group["group_id"])
        if index < len(reserved):
            partition = reserved[index]
        else:
            partition = min(
                PARTITIONS,
                key=lambda name: (
                    counts[name] / max(target[name], 1e-12),
                    counts[name],
                    PARTITIONS.index(name),
                ),
            )
        assignment[group_id] = partition
        counts[partition] += int(group["n_records"])
    if require_all and any(counts[name] == 0 for name in PARTITIONS):
        raise SplitError("split assignment produced an empty partition")
    return assignment


def _check_disjoint(
    descriptors: Sequence[Mapping[str, Any]],
    *,
    partition_key: str = "partition",
) -> dict[str, Any]:
    by_partition: dict[str, dict[str, set[str]]] = {
        name: {"source_hashes": set(), "record_hashes": set(), "record_ids": set(), "group_ids": set()}
        for name in PARTITIONS
    }
    for descriptor in descriptors:
        partition = str(descriptor[partition_key])
        if partition not in by_partition:
            continue
        target = by_partition[partition]
        target["source_hashes"].update(descriptor["source_hashes"])
        target["record_hashes"].add(str(descriptor["record_hash"]))
        target["record_ids"].add(str(descriptor["record_id"]))
        target["group_ids"].add(str(descriptor["group_id"]))
    overlaps: dict[str, dict[str, list[str]]] = {}
    for left_index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[left_index + 1:]:
            for identity in ("source_hashes", "record_hashes", "group_ids"):
                common = sorted(by_partition[left][identity] & by_partition[right][identity])
                if common:
                    overlaps[f"{left}:{right}:{identity}"] = common
    if overlaps:
        raise SplitError(f"split provenance overlap: {overlaps}")
    return {
        name: {key: sorted(values) for key, values in identities.items()}
        for name, identities in by_partition.items()
    }


def group_disjoint_split(
    records: Any,
    *,
    domain: str | None = None,
    group_fields: Iterable[str] | str | None = None,
    ratios: Mapping[str, float] | Sequence[float] = (0.6, 0.2, 0.2),
    seed: int = 30,
    source_base: str | Path | None = None,
) -> dict[str, Any]:
    """Assign records to group-disjoint train/calibration/test partitions."""
    prepared = _prepare(records, domain=domain, group_fields=group_fields, source_base=source_base)
    if isinstance(ratios, Mapping):
        ratio_map = dict(ratios)
    else:
        if len(ratios) != 3:
            raise SplitError("ratios must have three values")
        ratio_map = dict(zip(PARTITIONS, (float(item) for item in ratios)))
    group_assignment = _assign_groups(prepared["groups"], seed=seed, ratios=ratio_map)
    descriptors: list[dict[str, Any]] = []
    assignments: list[str] = []
    for index, row in enumerate(prepared["rows"]):
        group_id = prepared["row_group"][index]
        partition = group_assignment[group_id]
        assignments.append(partition)
        descriptors.append({
            "index": index,
            "record_id": prepared["record_ids"][index],
            "record_hash": prepared["record_hashes"][index],
            "source_hashes": prepared["source_hashes"][index],
            "group_id": group_id,
            "partition": partition,
        })
    identity_sets = _check_disjoint(descriptors)
    status = prepared["defaults"].get("status")
    if status not in {"measured", "synthetic", "blocked"}:
        raise SplitError("record status must be measured, synthetic, or blocked")
    return {
        "version": VERSION,
        "scheme": SCHEME_THREE_WAY,
        "status": status,
        "domain": prepared["domain"],
        "group_fields": prepared["group_fields"],
        "seed": int(seed),
        "ratios": {name: float(ratio_map[name]) for name in PARTITIONS},
        "n_records": len(prepared["rows"]),
        "n_groups": len(prepared["groups"]),
        "assignments": assignments,
        "records": descriptors,
        "groups": [
            {**dict(group), "partition": group_assignment[str(group["group_id"])]}
            for group in prepared["groups"]
        ],
        "identity_sets": identity_sets,
        "source_manifest": prepared["defaults"].get("source_manifest"),
    }


def _assign_logo_remaining(
    groups: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    calibration_ratio: float,
) -> dict[str, str]:
    if not 0.0 < calibration_ratio < 1.0:
        raise SplitError("calibration_ratio must be in (0, 1)")
    if len(groups) < 2:
        raise SplitError("leave-one-group-out requires at least two groups")
    remaining = sorted(groups, key=lambda item: _seeded_key(seed, str(item["group_hash"])))
    ratio_map = {
        "train": 1.0 - calibration_ratio,
        "calibration": calibration_ratio,
        "test": 1.0,
    }
    # ``test`` is not assigned here; the caller assigns the held-out group.
    counts = {"train": 0, "calibration": 0}
    total = sum(int(group["n_records"]) for group in remaining)
    target = {name: ratio_map[name] * total for name in counts}
    result: dict[str, str] = {}
    if len(remaining) >= 2:
        result[str(remaining[0]["group_id"])] = "train"
        counts["train"] += int(remaining[0]["n_records"])
        result[str(remaining[1]["group_id"])] = "calibration"
        counts["calibration"] += int(remaining[1]["n_records"])
        remaining = remaining[2:]
    else:
        result[str(remaining[0]["group_id"])] = "train"
        return result
    for group in remaining:
        partition = min(
            counts,
            key=lambda name: (
                counts[name] / max(target[name], 1e-12),
                counts[name],
                ("train", "calibration").index(name),
            ),
        )
        result[str(group["group_id"])] = partition
        counts[partition] += int(group["n_records"])
    return result


def leave_one_group_out(
    records: Any,
    *,
    domain: str | None = None,
    group_fields: Iterable[str] | str | None = None,
    seed: int = 30,
    calibration_ratio: float = 0.2,
    source_base: str | Path | None = None,
) -> dict[str, Any]:
    """Create deterministic folds with exactly one group held out as test.

    For each held-out group, the remaining groups are split into train and
    calibration.  At least three independent groups are required so both
    calibration and held-out test remain non-empty in every fold.
    """
    prepared = _prepare(records, domain=domain, group_fields=group_fields, source_base=source_base)
    groups = list(prepared["groups"])
    if len(groups) < 3:
        raise SplitError("leave-one-group-out with calibration requires at least three groups")
    folds: list[dict[str, Any]] = []
    for fold_index, held_out in enumerate(groups):
        remaining = [group for group in groups if group["group_id"] != held_out["group_id"]]
        assignment = _assign_logo_remaining(
            remaining, seed=seed + fold_index, calibration_ratio=calibration_ratio,
        )
        assignment[str(held_out["group_id"])] = "test"
        descriptors: list[dict[str, Any]] = []
        for index, row in enumerate(prepared["rows"]):
            group_id = prepared["row_group"][index]
            descriptors.append({
                "index": index,
                "record_id": prepared["record_ids"][index],
                "record_hash": prepared["record_hashes"][index],
                "source_hashes": prepared["source_hashes"][index],
                "group_id": group_id,
                "partition": assignment[group_id],
            })
        identity_sets = _check_disjoint(descriptors)
        folds.append({
            "fold": fold_index,
            "held_out_group": held_out["group_id"],
            "held_out_group_hash": held_out["group_hash"],
            "assignments": [item["partition"] for item in descriptors],
            "records": descriptors,
            "identity_sets": identity_sets,
        })
    status = prepared["defaults"].get("status")
    if status not in {"measured", "synthetic", "blocked"}:
        raise SplitError("record status must be measured, synthetic, or blocked")
    return {
        "version": VERSION,
        "scheme": SCHEME_LOGO,
        "status": status,
        "domain": prepared["domain"],
        "group_fields": prepared["group_fields"],
        "seed": int(seed),
        "calibration_ratio": float(calibration_ratio),
        "n_records": len(prepared["rows"]),
        "n_groups": len(groups),
        "groups": groups,
        "folds": folds,
        "source_manifest": prepared["defaults"].get("source_manifest"),
    }


def build_validation_manifest(
    records: Any,
    *,
    scheme: str = SCHEME_THREE_WAY,
    domain: str | None = None,
    group_fields: Iterable[str] | str | None = None,
    seed: int = 30,
    ratios: Mapping[str, float] | Sequence[float] = (0.6, 0.2, 0.2),
    calibration_ratio: float = 0.2,
    source_base: str | Path | None = None,
) -> dict[str, Any]:
    """Build an immutable-ready manifest without writing it."""
    if scheme in {SCHEME_THREE_WAY, "three_way", "group_disjoint"}:
        result = group_disjoint_split(
            records, domain=domain, group_fields=group_fields, ratios=ratios,
            seed=seed, source_base=source_base,
        )
    elif scheme in {SCHEME_LOGO, "logo", "leave_one_out"}:
        result = leave_one_group_out(
            records, domain=domain, group_fields=group_fields, seed=seed,
            calibration_ratio=calibration_ratio, source_base=source_base,
        )
    else:
        raise SplitError(f"unsupported split scheme: {scheme}")
    result = jsonable(result)
    result["manifest_version"] = MANIFEST_VERSION
    if result.get("status") not in {"measured", "synthetic", "blocked"}:
        raise SplitError("validation manifest status must be measured, synthetic, or blocked")
    result["immutable"] = True
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def write_validation_manifest(manifest: Mapping[str, Any], output_path: str | Path) -> dict[str, Any]:
    """Write a manifest exactly once and return its file metadata."""
    candidate = dict(manifest)
    candidate.pop("manifest_sha256", None)
    candidate["manifest_sha256"] = canonical_sha256(candidate)
    if candidate.get("manifest_version") != MANIFEST_VERSION or candidate.get("immutable") is not True:
        raise SplitError("manifest must be built by build_validation_manifest")
    return safe_json_dump(candidate, output_path)


def load_validation_manifest(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SplitError(f"cannot load validation manifest: {path}") from exc
    if not isinstance(value, Mapping) or value.get("manifest_version") != MANIFEST_VERSION:
        raise SplitError("unsupported validation manifest")
    unsigned = dict(value)
    stored = unsigned.pop("manifest_sha256", None)
    if not isinstance(stored, str) or canonical_sha256(unsigned) != stored:
        raise SplitError("validation manifest canonical hash mismatch")
    if value.get("immutable") is not True:
        raise SplitError("validation manifest is not immutable")
    return dict(value)


def _read_source_manifest(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SplitError(f"invalid source manifest: {target}") from exc
    if not isinstance(value, Mapping):
        raise SplitError("source manifest must be an object")
    stored = value.get("manifest_sha256")
    unsigned = dict(value)
    unsigned.pop("manifest_sha256", None)
    if not isinstance(stored, str) or canonical_sha256(unsigned) != stored:
        raise SplitError("source manifest canonical hash mismatch")
    return dict(value)


def _selftest() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rows = []
        for index in range(6):
            source = hashlib.sha256(f"source-{index}".encode()).hexdigest()
            rows.append({
                "record_id": f"record-{index}",
                "status": "measured",
                "campaign_id": f"C-{index // 2}",
                "specimen_id": f"S-{index}",
                "run_id": f"R-{index}",
                "source_sha256": source,
                "value": index,
            })
        split = group_disjoint_split(rows, domain="material", seed=7)
        assert split["n_groups"] == 3
        assert set(split["assignments"]) == set(PARTITIONS)
        assert all(
            not (set(split["identity_sets"][left]["source_hashes"]) & set(split["identity_sets"][right]["source_hashes"]))
            for left_index, left in enumerate(PARTITIONS)
            for right in PARTITIONS[left_index + 1:]
        )
        manifest = build_validation_manifest(rows, domain="material", seed=7)
        output = root / "validation-manifest.json"
        write_validation_manifest(manifest, output)
        loaded = load_validation_manifest(output)
        assert loaded["manifest_sha256"] == manifest["manifest_sha256"]
        try:
            write_validation_manifest(manifest, output)
        except FileExistsError:
            pass
        else:
            raise AssertionError("validation manifest was overwritten")
        logo = leave_one_group_out(rows, domain="material", seed=7)
        assert len(logo["folds"]) == logo["n_groups"] == 3
        assert all({item["partition"] for item in fold["records"]} == set(PARTITIONS) for fold in logo["folds"])
        bucket_rows = [
            {"record_id": f"bucket-record-{i}", "status": "measured", "asset": f"A-{i}", "component": "bucket", "service_episode": f"E-{i}", "scan": f"S-{i}", "source_sha256": hashlib.sha256(f"b-{i}".encode()).hexdigest()}
            for i in range(3)
        ]
        assert group_disjoint_split(bucket_rows, domain="bucket")["n_groups"] == 3


def selftest() -> bool:
    _selftest()
    return True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build immutable group-disjoint Step30 validation manifests")
    parser.add_argument("--input", help="CSV, JSON, or JSONL records")
    parser.add_argument("--output", type=Path, help="immutable JSON manifest")
    parser.add_argument("--source-manifest", help="optional source manifest path to retain in the output")
    parser.add_argument("--domain", choices=("material", "bucket"))
    parser.add_argument("--group-fields", help="comma-separated explicit grouping fields")
    parser.add_argument("--scheme", choices=("three_way", "logo"), default="three_way")
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.selftest:
        selftest()
        print("step30_validation_split selftest: PASS")
        return 0
    if not args.input:
        _parser().error("--input is required unless --selftest is used")
    if not args.output:
        _parser().error("--output is required unless --selftest is used")
    value: Any = args.input
    # Keep the source manifest reference in the immutable manifest without
    # treating a CLI flag as provenance evidence.
    if args.source_manifest:
        source_manifest = _read_source_manifest(args.source_manifest)
        value = {
            "records": _records_from_value(args.input)[0],
            "source_manifest": {
                "path": str(Path(args.source_manifest).resolve()),
                "sha256": sha256_file(args.source_manifest),
                "manifest_sha256": source_manifest["manifest_sha256"],
            },
        }
    scheme = SCHEME_THREE_WAY if args.scheme == "three_way" else SCHEME_LOGO
    manifest = build_validation_manifest(
        value,
        scheme=scheme,
        domain=args.domain,
        group_fields=args.group_fields,
        seed=args.seed,
        ratios=(args.train_ratio, args.calibration_ratio, args.test_ratio),
        calibration_ratio=args.calibration_ratio,
        source_base=Path(args.input).resolve().parent,
    )
    metadata = write_validation_manifest(manifest, args.output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"written": metadata}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
