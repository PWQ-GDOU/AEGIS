"""Frozen v2 protocol and prediction commitment for material trials.

Partitioning uses identities only. Measurement values, source hashes, and
certificate contents are integrity bindings and never partitioning inputs.
"""
from __future__ import annotations

import hashlib
import json
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


PROTOCOL_VERSION = "step30-material-trial-source-manifest-v2"
SOURCE_MANIFEST_VERSION = PROTOCOL_VERSION
PREDICTION_COMMITMENT_VERSION = "step30-material-trial-prediction-commitment-v2"
BLOCKED_STATUS_VERSION = "step30-material-trial-blocked-status-v2"
PARTITIONS = ("train", "calibration", "independent_test")
HARD_GROUP_FIELDS = (
    "asset_id",
    "heat_id",
    "material_batch",
    "campaign_id",
    "specimen_id",
)
IDENTITY_FIELDS = (
    "trial_id",
    "run_id",
    "asset_id",
    "heat_id",
    "material_batch",
    "campaign_id",
    "specimen_id",
)
PILOT_MIN_GROUPS = 3
RECOMMENDED_FORMAL_GROUPS = 5
DEFAULT_SPLIT_SEED = 30
ERROR_THRESHOLD = 0.05
PREREGISTERED_METRICS = {
    "wear_volume_mm3": {"kind": "volume", "units": "mm3"},
    "mass_loss_g": {"kind": "mass", "units": "g"},
    "wear_depth_mm": {"kind": "depth", "units": "mm"},
    "roughness_ra_um": {"kind": "morphology", "units": "um"},
    "pit_volume_mm3": {"kind": "morphology", "units": "mm3"},
    "pit_max_depth_um": {"kind": "morphology", "units": "um"},
}

_HEX = frozenset("0123456789abcdef")
_UNKNOWN = frozenset({"", "unknown", "unspecified", "n/a", "na", "none"})
_CONDITION_FIELDS = (
    "speed_m_s",
    "impact_angle_deg",
    "psd_d10_um",
    "psd_d50_um",
    "psd_d90_um",
    "sediment_mass_kg",
    "exposure_s",
    "exposed_area_mm2",
)


class MaterialTrialProtocolError(ValidationError):
    """Raised when material evidence cannot satisfy the frozen protocol."""


class MaterialPredictionCommitmentError(ValidationError):
    """Raised when a material prediction commitment is incomplete or changed."""


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


def _sha256(value: Any, name: str, error: type[ValidationError]) -> str:
    if not _is_sha256(value):
        raise error(f"{name} must be a SHA-256 digest")
    return str(value).lower()


def _text(value: Any, name: str, error: type[ValidationError]) -> str:
    if not isinstance(value, str) or value.strip().lower() in _UNKNOWN:
        raise error(f"{name} must be known and non-empty")
    return value.strip()


def _number(
    value: Any,
    name: str,
    error: type[ValidationError],
    *,
    positive: bool = False,
) -> float:
    if isinstance(value, bool):
        raise error(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise error(f"{name} must be numeric") from exc
    if not math.isfinite(number) or number < 0.0 or (positive and number <= 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise error(f"{name} must be finite and {qualifier}")
    return number


def _mapping(value: Any, name: str, error: type[ValidationError]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str, error: type[ValidationError]) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise error(f"{name} must be an array")
    return list(value)


def source_file_descriptor(
    path: str | Path,
    *,
    status: str = "measured",
    kind: str | None = None,
) -> dict[str, Any]:
    """Return a byte/hash binding for one existing raw source."""
    if status not in {"measured", "synthetic"}:
        raise MaterialTrialProtocolError("source status must be measured or synthetic")
    target = Path(path).resolve()
    if not target.is_file():
        raise MaterialTrialProtocolError(f"source file does not exist: {target}")
    result: dict[str, Any] = {
        "path": str(target),
        "sha256": sha256_file(target),
        "bytes": int(target.stat().st_size),
        "status": status,
    }
    if kind is not None:
        result["kind"] = _text(kind, "source kind", MaterialTrialProtocolError)
    return result


def _source(
    value: Any,
    name: str,
    *,
    status: str,
    verify_source: bool,
    default_kind: str,
) -> dict[str, Any]:
    source = _mapping(value, name, MaterialTrialProtocolError)
    path = Path(
        _text(source.get("path"), f"{name} path", MaterialTrialProtocolError)
    ).resolve()
    digest = _sha256(
        source.get("sha256"), f"{name} sha256", MaterialTrialProtocolError
    )
    byte_count = source.get("bytes")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise MaterialTrialProtocolError(
            f"{name} bytes must be a non-negative integer"
        )
    if source.get("status", status) != status:
        raise MaterialTrialProtocolError(f"{name} status differs from protocol status")
    if verify_source:
        if not path.is_file():
            raise MaterialTrialProtocolError(f"{name} file does not exist: {path}")
        if path.stat().st_size != byte_count or sha256_file(path) != digest:
            raise MaterialTrialProtocolError(
                f"{name} hash/byte count mismatch: {path}"
            )
    return {
        "path": str(path),
        "sha256": digest,
        "bytes": byte_count,
        "status": status,
        "kind": _text(
            source.get("kind", default_kind),
            f"{name} kind",
            MaterialTrialProtocolError,
        ),
    }


def _chemistry(value: Any, name: str) -> dict[str, Any]:
    chemistry = _mapping(value, name, MaterialTrialProtocolError)
    unit = str(chemistry.get("unit", "wt_pct")).strip().lower()
    if unit not in {"wt_pct", "mass_percent"}:
        raise MaterialTrialProtocolError(f"{name} unit must be wt_pct")
    values = chemistry.get("values", chemistry)
    values = _mapping(values, f"{name} values", MaterialTrialProtocolError)
    normalized = {
        _text(element, f"{name} element", MaterialTrialProtocolError): _number(
            amount, f"{name} {element}", MaterialTrialProtocolError
        )
        for element, amount in values.items()
        if element != "unit"
    }
    if not normalized or sum(normalized.values()) > 100.000001:
        raise MaterialTrialProtocolError(
            f"{name} must contain plausible wt_pct values"
        )
    return {"unit": "wt_pct", "values": dict(sorted(normalized.items()))}


def _certificate_claims(path: Path) -> Mapping[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MaterialTrialProtocolError(
            "formal material certificate must be machine-readable JSON"
        ) from exc
    if not isinstance(document, Mapping):
        raise MaterialTrialProtocolError(
            "material certificate must be a JSON object"
        )
    nested = document.get("material")
    return nested if isinstance(nested, Mapping) else document


def _hardness(value: Any) -> dict[str, Any]:
    item = _mapping(value, "hardness", MaterialTrialProtocolError)
    source = str(item.get("source", "")).strip().lower()
    if source != "measured" or item.get("is_default") is not False:
        raise MaterialTrialProtocolError(
            "hardness default constant/assumption is forbidden; measured actual value required"
        )
    if "value_gpa" in item or "uncertainty_gpa" in item:
        if "value_gpa" not in item or "uncertainty_gpa" not in item:
            raise MaterialTrialProtocolError(
                "canonical hardness requires value_gpa and uncertainty_gpa"
            )
        value_gpa = _number(
            item.get("value_gpa"),
            "hardness value_gpa",
            MaterialTrialProtocolError,
            positive=True,
        )
        uncertainty_gpa = _number(
            item.get("uncertainty_gpa"),
            "hardness uncertainty_gpa",
            MaterialTrialProtocolError,
            positive=True,
        )
    else:
        unit = str(item.get("unit", "")).strip().lower()
        measured = _number(
            item.get("value"),
            "hardness value",
            MaterialTrialProtocolError,
            positive=True,
        )
        uncertainty = _number(
            item.get("uncertainty"),
            "hardness uncertainty",
            MaterialTrialProtocolError,
            positive=True,
        )
        if unit in {"gpa", "gigapascal", "gigapascals"}:
            value_gpa, uncertainty_gpa = measured, uncertainty
        elif unit in {"hv", "vickers"}:
            value_gpa = measured * 0.00980665
            uncertainty_gpa = uncertainty * 0.00980665
        else:
            raise MaterialTrialProtocolError("hardness unit must be GPa or HV")
    return {
        "value_gpa": value_gpa,
        "uncertainty_gpa": uncertainty_gpa,
        "source": "measured",
        "is_default": False,
        "measurement_id": _text(
            item.get("measurement_id"),
            "hardness measurement_id",
            MaterialTrialProtocolError,
        ),
        "instrument_id": _text(
            item.get("instrument_id"),
            "hardness instrument_id",
            MaterialTrialProtocolError,
        ),
    }


def _density(value: Any) -> dict[str, Any]:
    item = _mapping(value, "density", MaterialTrialProtocolError)
    source = str(item.get("source", "")).strip().lower()
    if source not in {"measured", "certificate"}:
        raise MaterialTrialProtocolError(
            "density source must be measured or certificate"
        )
    return {
        "value_kg_m3": _number(
            item.get("value_kg_m3", item.get("value")),
            "density value",
            MaterialTrialProtocolError,
            positive=True,
        ),
        "uncertainty_kg_m3": _number(
            item.get("uncertainty_kg_m3", item.get("uncertainty")),
            "density uncertainty",
            MaterialTrialProtocolError,
            positive=True,
        ),
        "source": source,
    }


def _instrument(value: Any, *, status: str, verify_source: bool) -> dict[str, Any]:
    item = _mapping(value, "instrument", MaterialTrialProtocolError)
    return {
        "instrument_id": _text(item.get("instrument_id"), "instrument_id", MaterialTrialProtocolError),
        "kind": _text(item.get("kind"), "instrument kind", MaterialTrialProtocolError).lower(),
        "model": _text(item.get("model"), "instrument model", MaterialTrialProtocolError),
        "serial_number": _text(item.get("serial_number"), "instrument serial_number", MaterialTrialProtocolError),
        "calibrated_at": _text(item.get("calibrated_at"), "instrument calibrated_at", MaterialTrialProtocolError),
        "valid_until": _text(item.get("valid_until"), "instrument valid_until", MaterialTrialProtocolError),
        "calibration": _source(
            item.get("calibration"),
            "instrument calibration",
            status=status,
            verify_source=verify_source,
            default_kind="calibration_certificate",
        ),
    }


def _conditions(value: Any) -> dict[str, float]:
    item = _mapping(value, "conditions", MaterialTrialProtocolError)
    result = {
        name: _number(item.get(name), name, MaterialTrialProtocolError, positive=True)
        for name in _CONDITION_FIELDS
    }
    if not 0.0 < result["impact_angle_deg"] <= 90.0:
        raise MaterialTrialProtocolError("impact_angle_deg must be in (0, 90]")
    if not result["psd_d10_um"] <= result["psd_d50_um"] <= result["psd_d90_um"]:
        raise MaterialTrialProtocolError("PSD must satisfy d10 <= d50 <= d90")
    return result


def _material(record: Mapping[str, Any], *, status: str, verify_source: bool) -> dict[str, Any]:
    raw = record.get("material_identity", record.get("material"))
    material = _mapping(raw, "material identity", MaterialTrialProtocolError)
    grade = _text(material.get("grade", record.get("material_grade")), "material grade", MaterialTrialProtocolError)
    heat_id = _text(material.get("heat_id", record.get("heat_id")), "heat_id", MaterialTrialProtocolError)
    batch = _text(material.get("material_batch", record.get("material_batch")), "material_batch", MaterialTrialProtocolError)
    chemistry = _chemistry(material.get("chemistry"), "material chemistry")
    heat_treatment = _text(material.get("heat_treatment"), "heat treatment", MaterialTrialProtocolError)
    certificate = _source(
        material.get("certificate"),
        "material certificate",
        status=status,
        verify_source=verify_source,
        default_kind="material_certificate",
    )
    if verify_source:
        claims = _certificate_claims(Path(certificate["path"]))
        comparisons = {
            "material grade": (claims.get("material_grade", claims.get("grade")), grade),
            "heat_id": (claims.get("heat_id"), heat_id),
            "material_batch": (claims.get("material_batch"), batch),
            "heat treatment": (claims.get("heat_treatment"), heat_treatment),
        }
        for name, (actual, expected) in comparisons.items():
            if str(actual).strip() != expected:
                raise MaterialTrialProtocolError(f"material certificate {name} mismatch")
        if _chemistry(claims.get("chemistry"), "certificate chemistry") != chemistry:
            raise MaterialTrialProtocolError("material certificate chemistry mismatch")
    coating = _text(material.get("coating_status"), "coating status", MaterialTrialProtocolError)
    weld = _text(material.get("weld_status"), "weld status", MaterialTrialProtocolError)
    return {
        "grade": grade,
        "heat_id": heat_id,
        "material_batch": batch,
        "certificate": certificate,
        "chemistry": chemistry,
        "heat_treatment": heat_treatment,
        "coating_status": coating,
        "weld_status": weld,
        "hardness": _hardness(material.get("hardness")),
        "density": _density(material.get("density")),
    }


def _provenance(
    value: Any, *, status: str, campaign_id: str
) -> dict[str, Any]:
    item = _mapping(value, "trial provenance", MaterialTrialProtocolError)
    expected_origin = (
        "laboratory_measurement"
        if status == "measured"
        else "deterministic_simulation"
    )
    origin = _text(item.get("origin"), "provenance origin", MaterialTrialProtocolError)
    if origin != expected_origin:
        raise MaterialTrialProtocolError(
            f"{status} trial provenance origin must be {expected_origin}"
        )
    frozen_campaign = _text(
        item.get("campaign_id"),
        "provenance campaign_id",
        MaterialTrialProtocolError,
    )
    if frozen_campaign != campaign_id:
        raise MaterialTrialProtocolError(
            "trial provenance campaign differs from frozen campaign identity"
        )
    return {
        "origin": origin,
        "campaign_id": frozen_campaign,
        "organization": _text(
            item.get("organization"),
            "provenance organization",
            MaterialTrialProtocolError,
        ),
        "acquisition_method": _text(
            item.get("acquisition_method"),
            "provenance acquisition_method",
            MaterialTrialProtocolError,
        ),
        "acquired_at": _text(
            item.get("acquired_at"),
            "provenance acquired_at",
            MaterialTrialProtocolError,
        ),
    }


def _normalize_trial(record: Any, *, status: str, verify_source: bool) -> dict[str, Any]:
    item = _mapping(record, "trial", MaterialTrialProtocolError)
    if item.get("status", status) != status:
        raise MaterialTrialProtocolError("trial status differs from protocol status")
    if item.get("material_identity_known") is not True:
        raise MaterialTrialProtocolError("bucket/asset material identity is unknown")
    if item.get("mixed_heat") is not False:
        raise MaterialTrialProtocolError("mixed heat material is formally blocked")
    material = _material(item, status=status, verify_source=verify_source)
    instruments = [
        _instrument(value, status=status, verify_source=verify_source)
        for value in _sequence(item.get("instruments"), "instruments", MaterialTrialProtocolError)
    ]
    kinds = {instrument["kind"] for instrument in instruments}
    required_kinds = {"balance", "hardness_tester", "profilometer"}
    if not required_kinds.issubset(kinds):
        raise MaterialTrialProtocolError(
            f"instruments require calibrated {sorted(required_kinds)}"
        )
    hardness_instrument_id = material["hardness"]["instrument_id"]
    if not any(
        instrument["instrument_id"] == hardness_instrument_id
        and instrument["kind"] == "hardness_tester"
        for instrument in instruments
    ):
        raise MaterialTrialProtocolError(
            "hardness measurement is not bound to the calibrated hardness tester"
        )
    source = _source(
        item.get("source"),
        "trial raw source",
        status=status,
        verify_source=verify_source,
        default_kind="trial_data",
    )
    morphology_source = _source(
        item.get("morphology_source"),
        "morphology raw source",
        status=status,
        verify_source=verify_source,
        default_kind="height_field",
    )
    morphology_config = jsonable(
        _mapping(item.get("morphology_config"), "morphology_config", MaterialTrialProtocolError)
    )
    if not morphology_config:
        raise MaterialTrialProtocolError("morphology_config must freeze replay parameters")
    units = jsonable(_mapping(item.get("units", {}), "trial units", MaterialTrialProtocolError))
    campaign_id = _text(
        item.get("campaign_id"), "campaign_id", MaterialTrialProtocolError
    )
    normalized = {
        "trial_id": _text(item.get("trial_id"), "trial_id", MaterialTrialProtocolError),
        "run_id": _text(item.get("run_id"), "run_id", MaterialTrialProtocolError),
        "asset_id": _text(item.get("asset_id"), "asset_id", MaterialTrialProtocolError),
        "asset_type": _text(item.get("asset_type"), "asset_type", MaterialTrialProtocolError),
        "material_identity_known": True,
        "mixed_heat": False,
        "campaign_id": campaign_id,
        "specimen_id": _text(item.get("specimen_id"), "specimen_id", MaterialTrialProtocolError),
        "status": status,
        "provenance": _provenance(
            item.get("provenance"), status=status, campaign_id=campaign_id
        ),
        "source": source,
        "morphology_source": morphology_source,
        "morphology_config": morphology_config,
        "material": material,
        "instruments": sorted(instruments, key=lambda value: value["instrument_id"]),
        "conditions": _conditions(item.get("conditions")),
        "units": dict(sorted((str(key), str(value)) for key, value in units.items())),
    }
    return normalized


def _assert_no_mixed_identity(trials: Sequence[Mapping[str, Any]]) -> None:
    for field in ("asset_id", "specimen_id"):
        identities: dict[str, set[tuple[str, str, str]]] = {}
        for trial in trials:
            material = trial["material"]
            identities.setdefault(str(trial[field]), set()).add(
                (material["grade"], material["heat_id"], material["material_batch"])
            )
        mixed = sorted(key for key, values in identities.items() if len(values) > 1)
        if mixed:
            raise MaterialTrialProtocolError(f"mixed heat/material identity for {field}: {mixed}")
    for key_field, value_fields in (
        ("material_batch", ("heat_id", "grade")),
        ("heat_id", ("material_batch", "grade")),
    ):
        identities: dict[str, set[tuple[str, str]]] = {}
        for trial in trials:
            material = trial["material"]
            identities.setdefault(str(material[key_field]), set()).add(
                tuple(str(material[field]) for field in value_fields)
            )
        mixed = sorted(key for key, values in identities.items() if len(values) > 1)
        if mixed:
            raise MaterialTrialProtocolError(
                f"mixed heat/material batch identity for {key_field}: {mixed}"
            )


def _identity(trial: Mapping[str, Any]) -> dict[str, str]:
    material = trial["material"]
    return {
        "trial_id": str(trial["trial_id"]),
        "run_id": str(trial["run_id"]),
        "asset_id": str(trial["asset_id"]),
        "heat_id": str(material["heat_id"]),
        "material_batch": str(material["material_batch"]),
        "campaign_id": str(trial["campaign_id"]),
        "specimen_id": str(trial["specimen_id"]),
    }


def _groups(trials: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    union = _UnionFind(len(trials))
    owners: dict[tuple[str, str], int] = {}
    identities = [_identity(trial) for trial in trials]
    for index, identity in enumerate(identities):
        for field in HARD_GROUP_FIELDS:
            token = (field, identity[field])
            if token in owners:
                union.union(index, owners[token])
            else:
                owners[token] = index
    components: dict[int, list[dict[str, str]]] = {}
    for index, identity in enumerate(identities):
        components.setdefault(union.find(index), []).append(identity)
    result = []
    for members in components.values():
        frozen = sorted(members, key=lambda value: value["trial_id"])
        digest = canonical_sha256({"identities": frozen})
        result.append(
            {
                "group_id": f"material-group-{digest}",
                "trial_ids": sorted(value["trial_id"] for value in members),
                "n_trials": len(members),
                "identities_sha256": digest,
            }
        )
    return sorted(result, key=lambda value: value["group_id"])


def _assign_groups(groups: Sequence[Mapping[str, Any]], seed: int) -> dict[str, str]:
    if len(groups) < PILOT_MIN_GROUPS:
        raise MaterialTrialProtocolError(
            "at least three independent identity groups are required for a pilot"
        )
    ordered = sorted(
        groups,
        key=lambda value: hashlib.sha256(
            f"{seed}:{value['group_id']}".encode("utf-8")
        ).hexdigest(),
    )
    assignment = {
        str(group["group_id"]): partition
        for group, partition in zip(ordered[:3], PARTITIONS)
    }
    counts = {
        name: int(ordered[index]["n_trials"])
        for index, name in enumerate(PARTITIONS)
    }
    targets = {"train": 0.6, "calibration": 0.2, "independent_test": 0.2}
    for group in ordered[3:]:
        partition = min(
            PARTITIONS,
            key=lambda name: (
                counts[name] / targets[name],
                counts[name],
                PARTITIONS.index(name),
            ),
        )
        assignment[str(group["group_id"])] = partition
        counts[partition] += int(group["n_trials"])
    return assignment


def _metric_contract() -> dict[str, Any]:
    return {
        "preregistered": True,
        "posthoc_change_allowed": False,
        "comparison": "absolute_relative_error",
        "failure_rule": (
            "failed if any independent_test value is > 0.05; "
            "passed only if all are <= 0.05"
        ),
        "maximum_error_fraction": ERROR_THRESHOLD,
        "metrics": jsonable(PREREGISTERED_METRICS),
    }


def _split_contract(seed: int) -> dict[str, Any]:
    return {
        "scheme": "identity_only_union_group_holdout",
        "seed": seed,
        "partitions": list(PARTITIONS),
        "identity_fields": list(IDENTITY_FIELDS),
        "hard_group_fields": list(HARD_GROUP_FIELDS),
        "measurement_labels_used": False,
        "source_hashes_used": False,
        "pilot_min_groups": PILOT_MIN_GROUPS,
        "recommended_formal_groups": RECOMMENDED_FORMAL_GROUPS,
    }


def _dataset_binding(trials: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "trial_id": trial["trial_id"],
            "source": trial["source"],
            "morphology_source": trial["morphology_source"],
            "certificate": trial["material"]["certificate"],
            "instrument_calibrations": [
                instrument["calibration"] for instrument in trial["instruments"]
            ],
        }
        for trial in trials
    ]


def recompute_protocol_hash(protocol: Mapping[str, Any]) -> str:
    unsigned = dict(jsonable(protocol))
    unsigned.pop("protocol_sha256", None)
    return canonical_sha256(unsigned)


def build_material_trial_protocol(
    trials: Sequence[Mapping[str, Any]],
    *,
    status: str,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, Any]:
    """Freeze material identity, all raw sources, and an identity-only split."""
    if status not in {"measured", "synthetic"}:
        raise MaterialTrialProtocolError("protocol status must be measured or synthetic")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise MaterialTrialProtocolError("split seed must be an integer")
    raw_trials = _sequence(trials, "trials", MaterialTrialProtocolError)
    if not raw_trials:
        raise MaterialTrialProtocolError("trials must not be empty")
    normalized = [
        _normalize_trial(value, status=status, verify_source=True)
        for value in raw_trials
    ]
    trial_ids = [trial["trial_id"] for trial in normalized]
    if len(trial_ids) != len(set(trial_ids)):
        raise MaterialTrialProtocolError("trial_id values must be unique")
    _assert_no_mixed_identity(normalized)
    groups = _groups(normalized)
    assignment = _assign_groups(groups, seed)
    group_by_trial = {
        trial_id: group["group_id"]
        for group in groups
        for trial_id in group["trial_ids"]
    }
    frozen_trials = []
    for trial in normalized:
        group_id = group_by_trial[trial["trial_id"]]
        frozen_trials.append(
            {**trial, "group_id": group_id, "partition": assignment[group_id]}
        )
    frozen_trials.sort(key=lambda value: value["trial_id"])
    frozen_groups = [
        {**group, "partition": assignment[group["group_id"]]}
        for group in groups
    ]
    splits = {
        partition: sorted(
            trial["trial_id"]
            for trial in frozen_trials
            if trial["partition"] == partition
        )
        for partition in PARTITIONS
    }
    split_contract = _split_contract(seed)
    result = {
        "version": PROTOCOL_VERSION,
        "schema_version": PROTOCOL_VERSION,
        "status": status,
        "immutable": True,
        "evidence_role": (
            "formal_candidate"
            if status == "measured"
            else "synthetic_software_contract_only"
        ),
        "study_phase": (
            "formal_candidate"
            if len(groups) >= RECOMMENDED_FORMAL_GROUPS
            else "pilot"
        ),
        "split_contract": split_contract,
        "metric_contract": _metric_contract(),
        "n_trials": len(frozen_trials),
        "n_independent_groups": len(groups),
        "trials": frozen_trials,
        "groups": frozen_groups,
        "splits": splits,
        "dataset_sha256": canonical_sha256(
            {"trial_sources": _dataset_binding(frozen_trials)}
        ),
        "split_sha256": canonical_sha256(
            {
                "split_contract": split_contract,
                "groups": frozen_groups,
                "splits": splits,
            }
        ),
        "official_reference": False,
        "official_accuracy": False,
    }
    result["protocol_sha256"] = recompute_protocol_hash(result)
    return validate_material_trial_protocol(result)


def _expected_partition_data(
    trials: Sequence[Mapping[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    groups = _groups(trials)
    assignment = _assign_groups(groups, seed)
    group_by_trial = {
        trial_id: group["group_id"]
        for group in groups
        for trial_id in group["trial_ids"]
    }
    frozen_trials = sorted(
        [
            {
                **dict(trial),
                "group_id": group_by_trial[trial["trial_id"]],
                "partition": assignment[group_by_trial[trial["trial_id"]]],
            }
            for trial in trials
        ],
        key=lambda value: value["trial_id"],
    )
    frozen_groups = sorted(
        [
            {**group, "partition": assignment[group["group_id"]]}
            for group in groups
        ],
        key=lambda value: value["group_id"],
    )
    splits = {
        partition: sorted(
            trial["trial_id"]
            for trial in frozen_trials
            if trial["partition"] == partition
        )
        for partition in PARTITIONS
    }
    return frozen_trials, frozen_groups, splits


def _validate_protocol(protocol: Any, *, verify_sources: bool) -> dict[str, Any]:
    if not isinstance(protocol, Mapping):
        raise MaterialTrialProtocolError("protocol must be an object")
    if (
        protocol.get("version") != PROTOCOL_VERSION
        or protocol.get("schema_version") != PROTOCOL_VERSION
    ):
        raise MaterialTrialProtocolError("material trial protocol version mismatch")
    if (
        protocol.get("status") not in {"measured", "synthetic"}
        or protocol.get("immutable") is not True
    ):
        raise MaterialTrialProtocolError("protocol status/immutability contract is invalid")
    if (
        protocol.get("official_reference") is not False
        or protocol.get("official_accuracy") is not False
    ):
        raise MaterialTrialProtocolError(
            "protocol cannot claim official reference or accuracy"
        )
    stored_hash = protocol.get("protocol_sha256")
    if not _is_sha256(stored_hash) or stored_hash != recompute_protocol_hash(protocol):
        raise MaterialTrialProtocolError("protocol_sha256/content mismatch")
    split_contract = _mapping(
        protocol.get("split_contract"), "split_contract", MaterialTrialProtocolError
    )
    seed = split_contract.get("seed")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or dict(split_contract) != _split_contract(seed)
    ):
        raise MaterialTrialProtocolError("split is not the frozen identity-only contract")
    if protocol.get("metric_contract") != _metric_contract():
        raise MaterialTrialProtocolError(
            "preregistered metric/five-percent contract changed"
        )
    trials_raw = _sequence(
        protocol.get("trials"), "protocol trials", MaterialTrialProtocolError
    )
    canonical_trials = []
    for raw in trials_raw:
        item = _mapping(raw, "protocol trial", MaterialTrialProtocolError)
        base = _normalize_trial(
            item, status=protocol["status"], verify_source=verify_sources
        )
        group_id = _text(item.get("group_id"), "group_id", MaterialTrialProtocolError)
        partition = item.get("partition")
        if partition not in PARTITIONS:
            raise MaterialTrialProtocolError("trial partition is invalid")
        canonical_trials.append(
            {**base, "group_id": group_id, "partition": partition}
        )
    canonical_trials.sort(key=lambda value: value["trial_id"])
    if canonical_trials != trials_raw:
        raise MaterialTrialProtocolError("protocol trials are not canonical")
    if protocol.get("n_trials") != len(canonical_trials):
        raise MaterialTrialProtocolError("protocol trial count mismatch")
    _assert_no_mixed_identity(canonical_trials)
    expected_trials, expected_groups, expected_splits = _expected_partition_data(
        canonical_trials, seed
    )
    if canonical_trials != expected_trials:
        raise MaterialTrialProtocolError(
            "stored split differs from identity-only recomputation"
        )
    if protocol.get("groups") != expected_groups or protocol.get("splits") != expected_splits:
        raise MaterialTrialProtocolError("stored groups/cohorts are invalid")
    if any(not expected_splits[partition] for partition in PARTITIONS):
        raise MaterialTrialProtocolError(
            "train/calibration/independent_test must all be non-empty"
        )
    if protocol.get("n_independent_groups") != len(expected_groups):
        raise MaterialTrialProtocolError("independent group count mismatch")
    expected_phase = (
        "formal_candidate"
        if len(expected_groups) >= RECOMMENDED_FORMAL_GROUPS
        else "pilot"
    )
    if protocol.get("study_phase") != expected_phase:
        raise MaterialTrialProtocolError("study phase/group count mismatch")
    expected_role = (
        "formal_candidate"
        if protocol["status"] == "measured"
        else "synthetic_software_contract_only"
    )
    if protocol.get("evidence_role") != expected_role:
        raise MaterialTrialProtocolError("evidence role/status mismatch")
    if protocol.get("dataset_sha256") != canonical_sha256(
        {"trial_sources": _dataset_binding(canonical_trials)}
    ):
        raise MaterialTrialProtocolError("dataset_sha256 mismatch")
    if protocol.get("split_sha256") != canonical_sha256(
        {
            "split_contract": _split_contract(seed),
            "groups": expected_groups,
            "splits": expected_splits,
        }
    ):
        raise MaterialTrialProtocolError("split_sha256 mismatch")
    return dict(protocol)


def validate_material_trial_protocol_metadata(protocol: Any) -> dict[str, Any]:
    """Validate frozen content and split without opening any raw source."""
    return _validate_protocol(protocol, verify_sources=False)


def validate_material_trial_protocol_sources(
    protocol: Any, trial_ids: Sequence[str]
) -> dict[str, Any]:
    """Validate only named raw sources after metadata/split validation."""
    frozen = _validate_protocol(protocol, verify_sources=False)
    requested = set(
        _text(value, "source trial_id", MaterialTrialProtocolError)
        for value in _sequence(
            trial_ids, "source trial_ids", MaterialTrialProtocolError
        )
    )
    available = {str(trial["trial_id"]) for trial in frozen["trials"]}
    if not requested or not requested.issubset(available):
        raise MaterialTrialProtocolError(
            "source trial_ids must be a non-empty protocol subset"
        )
    for trial in frozen["trials"]:
        if trial["trial_id"] not in requested:
            continue
        canonical = _normalize_trial(
            trial, status=frozen["status"], verify_source=True
        )
        stored = dict(trial)
        stored.pop("group_id", None)
        stored.pop("partition", None)
        if canonical != stored:
            raise MaterialTrialProtocolError(
                f"trial source is not canonical: {trial['trial_id']}"
            )
    return frozen


def validate_material_trial_protocol(protocol: Any) -> dict[str, Any]:
    return _validate_protocol(protocol, verify_sources=True)


def save_material_trial_protocol(
    protocol: Mapping[str, Any], path: str | Path
) -> dict[str, Any]:
    return safe_json_dump(validate_material_trial_protocol(protocol), path)


def load_material_trial_protocol(path: str | Path) -> dict[str, Any]:
    try:
        return validate_material_trial_protocol(safe_json_load(path))
    except (OSError, ValidationError) as exc:
        raise MaterialTrialProtocolError(
            f"cannot load material trial protocol: {exc}"
        ) from exc


def recompute_external_pin_digest(commitment: Mapping[str, Any]) -> str:
    """Digest every commitment field except the two enclosing digests."""
    unsigned = dict(jsonable(commitment))
    unsigned.pop("commitment_sha256", None)
    unsigned.pop("external_pin_digest", None)
    return canonical_sha256(unsigned)


def recompute_commitment_hash(commitment: Mapping[str, Any]) -> str:
    unsigned = dict(jsonable(commitment))
    unsigned.pop("commitment_sha256", None)
    return canonical_sha256(unsigned)


def _commitment_model(value: Any) -> dict[str, Any]:
    model = dict(_mapping(value, "model", MaterialPredictionCommitmentError))
    required = {"model_type", "code", "config", "code_sha256", "config_sha256"}
    if set(model) != required:
        raise MaterialPredictionCommitmentError(
            "model must bind model_type, code, config and their digests"
        )
    if not isinstance(model["code"], Mapping) or not isinstance(model["config"], Mapping):
        raise MaterialPredictionCommitmentError("model code/config must be objects")
    if model["code_sha256"] != canonical_sha256(model["code"]):
        raise MaterialPredictionCommitmentError("model code digest mismatch")
    if model["config_sha256"] != canonical_sha256(model["config"]):
        raise MaterialPredictionCommitmentError("model config digest mismatch")
    return model


def _commitment_calibration(value: Any, protocol: Mapping[str, Any]) -> dict[str, Any]:
    calibration = dict(_mapping(value, "calibration", MaterialPredictionCommitmentError))
    fit_ids = sorted(
        list(protocol["splits"]["train"]) + list(protocol["splits"]["calibration"])
    )
    if calibration.get("fit_trial_ids") != fit_ids:
        raise MaterialPredictionCommitmentError(
            "calibration must use exactly train/calibration trials"
        )
    if calibration.get("fit_partitions") != ["train", "calibration"]:
        raise MaterialPredictionCommitmentError("calibration partition binding is invalid")
    test_ids = set(protocol["splits"]["independent_test"])
    if test_ids.intersection(fit_ids):
        raise MaterialPredictionCommitmentError(
            "independent_test labels leaked into calibration"
        )
    for name in ("model_fit_hash", "calibration_config_sha256"):
        _sha256(calibration.get(name), name, MaterialPredictionCommitmentError)
    return calibration


def _commitment_predictions(value: Any, cohort: Sequence[str]) -> list[dict[str, Any]]:
    rows = _sequence(value, "predictions", MaterialPredictionCommitmentError)
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        row = _mapping(raw, "prediction", MaterialPredictionCommitmentError)
        trial_id = _text(
            row.get("trial_id"), "prediction trial_id", MaterialPredictionCommitmentError
        )
        metrics = _mapping(
            row.get("metrics"), "prediction metrics", MaterialPredictionCommitmentError
        )
        if set(metrics) != set(PREREGISTERED_METRICS):
            raise MaterialPredictionCommitmentError(
                "prediction must contain every preregistered metric"
            )
        normalized.append(
            {
                "trial_id": trial_id,
                "metrics": {
                    name: _number(
                        metrics[name],
                        f"prediction {name}",
                        MaterialPredictionCommitmentError,
                    )
                    for name in PREREGISTERED_METRICS
                },
            }
        )
    normalized.sort(key=lambda value: value["trial_id"])
    if [row["trial_id"] for row in normalized] != list(cohort):
        raise MaterialPredictionCommitmentError(
            "prediction cohort must exactly match independent_test"
        )
    return normalized


def build_material_prediction_commitment(
    protocol: Mapping[str, Any],
    *,
    model_descriptor: Mapping[str, Any],
    calibration_binding: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    external_pin_digest: str | None = None,
) -> dict[str, Any]:
    """Bind model/code/config/data/split/cohort and blind predictions."""
    frozen = _validate_protocol(protocol, verify_sources=False)
    model = _commitment_model(model_descriptor)
    calibration = _commitment_calibration(calibration_binding, frozen)
    cohort = list(frozen["splits"]["independent_test"])
    normalized = _commitment_predictions(predictions, cohort)
    result = {
        "version": PREDICTION_COMMITMENT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": frozen["protocol_sha256"],
        "dataset_sha256": frozen["dataset_sha256"],
        "split_sha256": frozen["split_sha256"],
        "independent_test_trial_ids": cohort,
        "independent_test_cohort_sha256": canonical_sha256({"trial_ids": cohort}),
        "model": model,
        "calibration": calibration,
        "predictions": normalized,
        "predictions_sha256": canonical_sha256({"predictions": normalized}),
        "test_labels_used_for_fit": False,
        "official_reference": False,
        "official_accuracy": False,
    }
    derived_pin = recompute_external_pin_digest(result)
    if external_pin_digest is not None:
        supplied_pin = _sha256(
            external_pin_digest,
            "external pin digest",
            MaterialPredictionCommitmentError,
        )
        if supplied_pin != derived_pin:
            raise MaterialPredictionCommitmentError(
                "external pin digest must bind the complete commitment core"
            )
    result["external_pin_digest"] = derived_pin
    result["commitment_sha256"] = recompute_commitment_hash(result)
    return validate_material_prediction_commitment(result, frozen)


def validate_material_prediction_commitment(
    commitment: Any, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    frozen = _validate_protocol(protocol, verify_sources=False)
    if not isinstance(commitment, Mapping) or commitment.get("version") != PREDICTION_COMMITMENT_VERSION:
        raise MaterialPredictionCommitmentError("prediction commitment version mismatch")
    if commitment.get("commitment_sha256") != recompute_commitment_hash(commitment):
        raise MaterialPredictionCommitmentError("commitment_sha256/content mismatch")
    if commitment.get("protocol_version") != PROTOCOL_VERSION:
        raise MaterialPredictionCommitmentError("commitment protocol version mismatch")
    for field in ("protocol_sha256", "dataset_sha256", "split_sha256"):
        if commitment.get(field) != frozen[field]:
            raise MaterialPredictionCommitmentError(f"commitment {field} mismatch")
    if commitment.get("official_reference") is not False or commitment.get("official_accuracy") is not False:
        raise MaterialPredictionCommitmentError("commitment cannot claim official status")
    if commitment.get("test_labels_used_for_fit") is not False:
        raise MaterialPredictionCommitmentError("independent_test labels cannot enter fitting")
    cohort = list(frozen["splits"]["independent_test"])
    if commitment.get("independent_test_trial_ids") != cohort:
        raise MaterialPredictionCommitmentError("independent_test cohort mismatch")
    if commitment.get("independent_test_cohort_sha256") != canonical_sha256({"trial_ids": cohort}):
        raise MaterialPredictionCommitmentError("independent_test cohort digest mismatch")
    pin = _sha256(
        commitment.get("external_pin_digest"),
        "external pin digest",
        MaterialPredictionCommitmentError,
    )
    if pin != recompute_external_pin_digest(commitment):
        raise MaterialPredictionCommitmentError(
            "external pin digest does not bind the complete commitment core"
        )
    _commitment_model(commitment.get("model"))
    _commitment_calibration(commitment.get("calibration"), frozen)
    predictions = _commitment_predictions(commitment.get("predictions"), cohort)
    if commitment.get("predictions_sha256") != canonical_sha256({"predictions": predictions}):
        raise MaterialPredictionCommitmentError("predictions_sha256 mismatch")
    if predictions != commitment.get("predictions"):
        raise MaterialPredictionCommitmentError("predictions are not canonical")
    return dict(commitment)


def save_material_prediction_commitment(
    commitment: Mapping[str, Any], path: str | Path, *, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    return safe_json_dump(
        validate_material_prediction_commitment(commitment, protocol), path
    )


def load_material_prediction_commitment(
    path: str | Path, *, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        return validate_material_prediction_commitment(safe_json_load(path), protocol)
    except (OSError, ValidationError) as exc:
        raise MaterialPredictionCommitmentError(
            f"cannot load material prediction commitment: {exc}"
        ) from exc


def build_blocked_status(
    reason: str, *, details: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    result = {
        "version": BLOCKED_STATUS_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "status": "blocked",
        "reason": _text(reason, "blocked reason", MaterialTrialProtocolError),
        "details": jsonable(details or {}),
        "formal": False,
        "independent_material_gate": False,
        "official_reference": False,
        "official_accuracy": False,
        "official_reference_gate": False,
        "official_accuracy_gate": False,
    }
    result["report_sha256"] = canonical_sha256(result)
    return result


# Short aliases for integration callers.
build_protocol = build_material_trial_protocol
validate_protocol = validate_material_trial_protocol
save_protocol = save_material_trial_protocol
load_protocol = load_material_trial_protocol
build_prediction_commitment = build_material_prediction_commitment
validate_prediction_commitment = validate_material_prediction_commitment
save_prediction_commitment = save_material_prediction_commitment
load_prediction_commitment = load_material_prediction_commitment
