"""Formal material-trial replay from frozen protocol and raw files.

The replay intentionally does not accept a self-authored report as evidence.
It re-ingests every pinned source, calls the existing material calibration
implementation, recalculates morphology, and evaluates the frozen test cohort.
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from step30_material_model_calibration import (
    calibrate_material_model,
    finnie_volume_mm3_per_kg,
    mansouri_mass_ratio_kg_per_kg,
    oka_volume_mm3_per_kg,
    tabakoff_volume_mm3_per_kg,
)
from step30_material_morphology_metrics import morphology_from_file
from step30_material_trial_ingest import _load_rows, ingest_material_trials, normalize_record
from step30_material_trial_report import build_material_trial_report
from step30_material_trial_protocol import (
    ERROR_THRESHOLD,
    PREREGISTERED_METRICS,
    PROTOCOL_VERSION,
    MaterialPredictionCommitmentError,
    MaterialTrialProtocolError,
    build_blocked_status,
    validate_material_prediction_commitment,
    validate_material_trial_protocol,
    validate_material_trial_protocol_metadata,
    validate_material_trial_protocol_sources,
)
from step30_validation_common import (
    ValidationError,
    canonical_sha256,
    jsonable,
    safe_json_dump,
    safe_json_load,
    sha256_file,
)


REPLAY_VERSION = "step30-material-trial-replay-v2"
MODEL_CODE_VERSION = "step30-material-trial-model-code-v2"
DEFAULT_MODEL_NAME = "oka"


class MaterialTrialReplayError(ValidationError):
    """Raised internally; public replay converts it to a blocked report."""


def _read_artifact(value: Mapping[str, Any] | str | Path, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        document = safe_json_load(value)
    except (OSError, ValidationError) as exc:
        raise MaterialTrialReplayError(f"cannot read {name}: {value}") from exc
    if not isinstance(document, Mapping):
        raise MaterialTrialReplayError(f"{name} must be a JSON object")
    return dict(document)


def _descriptor(path: str | Path, expected: Mapping[str, Any], name: str) -> dict[str, Any]:
    target = Path(path).resolve()
    if not target.is_file():
        raise MaterialTrialReplayError(f"{name} is missing: {target}")
    actual = {
        "path": str(target),
        "sha256": sha256_file(target),
        "bytes": int(target.stat().st_size),
    }
    for field in ("path", "sha256", "bytes"):
        if field == "path":
            if Path(str(expected.get(field))).resolve() != target:
                raise MaterialTrialReplayError(f"{name} path differs from protocol")
        elif actual[field] != expected.get(field):
            raise MaterialTrialReplayError(f"{name} {field} mismatch")
    return actual


def _raw_identity(row: Mapping[str, Any], trial: Mapping[str, Any]) -> None:
    aliases = {
        "trial_id": ("trial_id", "material_trial_id"),
        "run_id": ("run_id", "run", "trial_run_id"),
        "campaign_id": ("campaign_id", "campaign"),
        "specimen_id": ("specimen_id", "specimen", "coupon_id"),
        "asset_id": ("asset_id",),
        "heat_id": ("heat_id",),
        "material_batch": ("material_batch", "batch_id"),
    }
    for field, names in aliases.items():
        present = next((row[name] for name in names if name in row and row[name] not in (None, "")), None)
        if present is None:
            if field in {"asset_id", "heat_id", "material_batch", "trial_id"}:
                continue
            raise MaterialTrialReplayError(f"raw trial source is missing {field}")
        expected = trial[field] if field not in {"heat_id", "material_batch"} else trial["material"][field]
        if str(present).strip() != str(expected):
            raise MaterialTrialReplayError(f"raw trial {field} differs from frozen identity")
    if "material" in row and str(row["material"]).strip() not in {
        str(trial["material"]["grade"]),
        str(trial["material"].get("material_grade", "")),
    }:
        raise MaterialTrialReplayError("raw material grade differs from certificate identity")


def _load_trial_rows(trial: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = trial["source"]
    _descriptor(source["path"], source, "trial raw source")
    try:
        rows = _load_rows(Path(source["path"]))
    except (OSError, ValidationError, ValueError, TypeError) as exc:
        raise MaterialTrialReplayError(
            f"cannot load raw trial source {trial['trial_id']}: {exc}"
        ) from exc
    if len(rows) != 1:
        raise MaterialTrialReplayError(
            f"trial source {trial['trial_id']} must contain exactly one row"
        )
    _raw_identity(rows[0], trial)
    return rows


def _manifest_for_combined(path: Path, status: str) -> dict[str, Any]:
    manifest = {
        "version": "step30-material-trial-source-manifest-v1",
        "status": status,
        "source": {
            "path": path.name,
            "sha256": sha256_file(path),
            "status": status,
        },
        "provenance": {
            "origin": "laboratory_measurement" if status == "measured" else "deterministic_simulation",
            "campaign_id": "material-v2-replay-campaign",
            "acquisition_method": "frozen material trial raw replay",
            "acquired_at": "2026-01-01T00:00:00Z",
            "organization": "material-trial-v2",
        },
        "units": {},
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def _combined_ingest(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    status = str(protocol["status"])
    rows: list[dict[str, Any]] = []
    trial_order: list[str] = []
    for index, trial in enumerate(protocol["trials"]):
        trial_order.append(str(trial["trial_id"]))
        split = (
            "test" if trial["partition"] == "independent_test" else "calibration"
        )
        # Normalize with this trial's frozen source units first. Protocol split
        # remains authoritative; any split/label in the source is ignored.
        raw = _normalize_fit_record(
            trial, status, index + 1, split=split
        )
        raw.pop("derived", None)
        raw["campaign_id"] = "material-v2-replay-campaign"
        raw["specimen_id"] = str(trial["trial_id"])
        rows.append(raw)
    with tempfile.TemporaryDirectory(prefix="step30-material-v2-replay-") as directory:
        root = Path(directory)
        combined = root / "combined-trials.json"
        combined.write_text(
            json.dumps({"rows": rows}, ensure_ascii=True), encoding="utf-8"
        )
        manifest_path = root / "combined-source-manifest.json"
        manifest_path.write_text(
            json.dumps(_manifest_for_combined(combined, status), ensure_ascii=True),
            encoding="utf-8",
        )
        ingested = ingest_material_trials(
            combined, manifest_path, requested_mode=status
        )
    if ingested.get("status") == "blocked":
        raise MaterialTrialReplayError(
            f"raw material ingest blocked: {ingested.get('blocked_reason', 'unknown reason')}"
        )
    if len(ingested.get("records", [])) != len(rows):
        raise MaterialTrialReplayError(
            "raw ingest row count differs from frozen protocol"
        )
    # Adapter paths are temporary implementation details. Replace them with
    # stable protocol bindings before any calibration/report hash is computed.
    ingested["source"] = {
        "kind": "protocol_trial_sources",
        "sha256": protocol["dataset_sha256"],
    }
    ingested["source_manifest"] = {
        "version": PROTOCOL_VERSION,
        "manifest_sha256": protocol["protocol_sha256"],
    }
    ingested["provenance"] = {
        "origin": (
            "laboratory_measurement"
            if status == "measured"
            else "deterministic_simulation"
        ),
        "campaign_id": "material-v2-replay-campaign",
        "acquisition_method": "frozen material trial raw replay",
        "organization": "material-trial-v2",
    }
    unsigned_ingest = dict(ingested)
    unsigned_ingest.pop("dataset_sha256", None)
    ingested["dataset_sha256"] = canonical_sha256(unsigned_ingest)
    by_key = {}
    for index, record in enumerate(ingested["records"]):
        by_key[(str(record["run_id"]), str(record["specimen_id"]))] = trial_order[index]
    return ingested, {"trial_order": trial_order, "record_to_trial": by_key}


def _formula(
    model_name: str,
    records: Sequence[Mapping[str, Any]],
    trials: Mapping[str, Mapping[str, Any]],
) -> np.ndarray:
    speed = np.asarray([float(row["speed_m_s"]) for row in records], dtype=float)
    angle = np.asarray([float(row["impact_angle_deg"]) for row in records], dtype=float)
    d50 = np.asarray([float(row["psd_d50_um"]) for row in records], dtype=float)
    if model_name == "oka":
        # The reused formula accepts one hardness at a time. Vectorizing here
        # preserves each trial's actual measurement instead of a default/mean.
        return np.asarray(
            [
                float(
                    oka_volume_mm3_per_kg(
                        speed[index],
                        angle[index],
                        d50[index],
                        hardness_gpa=float(
                            trials[str(row["specimen_id"])]["material"]["hardness"]["value_gpa"]
                        ),
                    )
                )
                for index, row in enumerate(records)
            ],
            dtype=float,
        )
    if model_name == "finnie":
        return finnie_volume_mm3_per_kg(speed, angle)
    if model_name == "tabakoff":
        return tabakoff_volume_mm3_per_kg(speed, angle)
    if model_name == "mansouri":
        return mansouri_mass_ratio_kg_per_kg(speed, angle)
    raise MaterialTrialReplayError(f"unsupported model_name {model_name!r}")


def _model_descriptor(model_name: str, config: Mapping[str, Any]) -> dict[str, Any]:
    workspace = Path(__file__).resolve().parent
    code = {
        "module": "step30_material_model_calibration",
        "model_name": model_name,
        "formula_version": "step30-material-physical-formulas-v1",
        "implementation": MODEL_CODE_VERSION,
        "hardness_binding": (
            "protocol.material.hardness.value_gpa"
            if model_name == "oka"
            else "not_used"
        ),
        "source_file_sha256": {
            name: sha256_file(workspace / name)
            for name in (
                "step30_material_trial_replay.py",
                "step30_material_trial_ingest.py",
                "step30_material_model_calibration.py",
                "step30_material_morphology_metrics.py",
                "step30_material_trial_report.py",
                "step30_validation_common.py",
            )
        },
    }
    config_json = dict(config)
    return {
        "model_type": model_name,
        "code": code,
        "config": config_json,
        "code_sha256": canonical_sha256(code),
        "config_sha256": canonical_sha256(config_json),
    }


def _target(record: Mapping[str, Any], model_name: str) -> float:
    key = "erosion_mass_kg_per_kg" if model_name == "mansouri" else "erosion_volume_mm3_per_kg"
    value = float(record["derived"][key])
    if not math.isfinite(value) or value <= 0.0:
        raise MaterialTrialReplayError(f"fit target {key} must be positive")
    return value


def _baseline(value: float, record: Mapping[str, Any], model_name: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise MaterialTrialReplayError("physical model baseline must be positive")
    return value


def _normalize_fit_record(
    trial: Mapping[str, Any],
    status: str,
    index: int,
    *,
    split: str = "calibration",
) -> dict[str, Any]:
    raw = dict(_load_trial_rows(trial)[0])
    raw.update(
        {
            "split": split,
            "group_id": str(trial["group_id"]),
            "campaign_id": str(trial["campaign_id"]),
            "specimen_id": str(trial["trial_id"]),
            "run_id": str(trial["run_id"]),
            "material": str(trial["material"]["grade"]),
            "status": status,
        }
    )
    manifest = {
        "status": status,
        "units": dict(trial.get("units", {})),
        "provenance": {"campaign_id": str(trial["campaign_id"])},
    }
    try:
        return normalize_record(raw, manifest, index)
    except (ValidationError, ValueError, TypeError) as exc:
        raise MaterialTrialReplayError(
            f"cannot normalize fit trial {trial['trial_id']}: {exc}"
        ) from exc


def _prediction_from_conditions(
    trial: Mapping[str, Any], model_name: str, fit_scale: float
) -> dict[str, float]:
    conditions = trial["conditions"]
    pseudo_record = {
        "speed_m_s": conditions["speed_m_s"],
        "impact_angle_deg": conditions["impact_angle_deg"],
        "psd_d50_um": conditions["psd_d50_um"],
        "specimen_id": trial["trial_id"],
    }
    physical = float(
        _formula(model_name, [pseudo_record], {trial["trial_id"]: trial})[0]
    )
    model_value = physical * fit_scale
    sediment = float(conditions["sediment_mass_kg"])
    density = float(trial["material"]["density"]["value_kg_m3"])
    if model_name == "mansouri":
        volume = model_value * sediment * 1.0e9 / density
        mass = model_value * sediment * 1000.0
    else:
        volume = model_value * sediment
        mass = volume * density * 1.0e-6
    return {
        "wear_volume_mm3": volume,
        "mass_loss_g": mass,
        "wear_depth_mm": volume / float(conditions["exposed_area_mm2"]),
    }


def prepare_material_trial_replay(
    protocol: Mapping[str, Any] | str | Path,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, Any]:
    """Fit without opening independent-test labels or morphology sources."""
    document = _read_artifact(protocol, "material trial protocol")
    frozen = validate_material_trial_protocol_metadata(document)
    fit_ids = sorted(
        frozen["splits"]["train"] + frozen["splits"]["calibration"]
    )
    validate_material_trial_protocol_sources(frozen, fit_ids)
    trials = {str(trial["trial_id"]): trial for trial in frozen["trials"]}
    fit_records = [
        _normalize_fit_record(trials[trial_id], frozen["status"], index + 1)
        for index, trial_id in enumerate(fit_ids)
    ]
    physical = _formula(model_name, fit_records, trials)
    ratios = []
    labels_for_fit = []
    for index, trial_id in enumerate(fit_ids):
        record = fit_records[index]
        ratios.append(
            _target(record, model_name)
            / _baseline(float(physical[index]), record, model_name)
        )
        labels_for_fit.append(
            {
                "trial_id": trial_id,
                "target": _target(record, model_name),
                "source_sha256": trials[trial_id]["source"]["sha256"],
            }
        )
    if (
        not ratios
        or not np.isfinite(ratios).all()
        or any(value <= 0.0 for value in ratios)
    ):
        raise MaterialTrialReplayError(
            "calibration ratios are not finite and positive"
        )
    fit_scale = float(
        np.exp(np.mean(np.log(np.asarray(ratios, dtype=float))))
    )
    config = {
        "model_name": model_name,
        "fit_transform": "geometric_mean(target / physical_baseline)",
        "hardness_source": "protocol.material.hardness.value_gpa",
        "fit_partitions": ["train", "calibration"],
    }
    fit_contract = {
        "model_name": model_name,
        "fit_scale": fit_scale,
        "fit_transform": config["fit_transform"],
        "fit_trial_ids": fit_ids,
        "fit_partitions": ["train", "calibration"],
        "fit_labels_sha256": canonical_sha256({"labels": labels_for_fit}),
        "source_dataset_sha256": frozen["dataset_sha256"],
        "protocol_sha256": frozen["protocol_sha256"],
        "physical_formula_version": "step30-material-physical-formulas-v1",
    }
    descriptor = _model_descriptor(model_name, config)
    calibration_binding = {
        "fit_trial_ids": fit_ids,
        "fit_partitions": ["train", "calibration"],
        "model_fit_hash": canonical_sha256(fit_contract),
        "calibration_config_sha256": descriptor["config_sha256"],
    }
    predicted = {
        trial_id: _prediction_from_conditions(
            trials[trial_id], model_name, fit_scale
        )
        for trial_id in frozen["splits"]["independent_test"]
    }
    return {
        "status": frozen["status"],
        "protocol": frozen,
        "model_descriptor": descriptor,
        "calibration_binding": calibration_binding,
        "fit_contract": fit_contract,
        "predicted_metrics": predicted,
        "independent_test_labels_opened": False,
        "independent_test_morphology_opened": False,
    }


def commitment_predictions_from_preparation(
    preparation: Mapping[str, Any],
    *,
    morphology_predictions: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build complete blind prediction rows before test labels are evaluated."""
    protocol = preparation["protocol"]
    rows = []
    for trial_id in protocol["splits"]["independent_test"]:
        morphology = morphology_predictions.get(trial_id)
        if not isinstance(morphology, Mapping):
            raise MaterialTrialReplayError(
                f"morphology prediction missing for independent trial {trial_id}"
            )
        metrics = dict(preparation["predicted_metrics"][trial_id])
        for name in ("roughness_ra_um", "pit_volume_mm3", "pit_max_depth_um"):
            try:
                value = float(morphology[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise MaterialTrialReplayError(
                    f"morphology prediction {name} missing/non-numeric for {trial_id}"
                ) from exc
            if not math.isfinite(value) or value < 0.0:
                raise MaterialTrialReplayError(
                    f"morphology prediction {name} must be finite and non-negative"
                )
            metrics[name] = value
        rows.append({"trial_id": trial_id, "metrics": metrics})
    return rows


def _relative_error(predicted: float, observed: float) -> float:
    if not math.isfinite(predicted) or not math.isfinite(observed):
        raise MaterialTrialReplayError("metric values must be finite")
    if observed == 0.0:
        if predicted == 0.0:
            return 0.0
        raise MaterialTrialReplayError(
            "zero observed metric cannot support a relative-error acceptance claim"
        )
    return abs(predicted - observed) / abs(observed)


def _validate_protocol_conditions(
    protocol: Mapping[str, Any], ingest: Mapping[str, Any]
) -> None:
    by_trial = {str(row["specimen_id"]): row for row in ingest["records"]}
    for trial in protocol["trials"]:
        trial_id = str(trial["trial_id"])
        if trial_id not in by_trial:
            raise MaterialTrialReplayError(f"raw ingest is missing trial {trial_id}")
        record = by_trial[trial_id]
        for name, expected in trial["conditions"].items():
            if not math.isclose(
                float(record[name]), float(expected), rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise MaterialTrialReplayError(
                    f"raw {name} differs from frozen protocol for {trial_id}"
                )
        density = float(trial["material"]["density"]["value_kg_m3"])
        if not math.isclose(
            float(record["material_density_kg_m3"]),
            density,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise MaterialTrialReplayError(
                f"raw density differs from frozen material identity for {trial_id}"
            )
        if (
            record.get("registration_passed") is not True
            or record.get("measurement_quality_passed") is not True
        ):
            raise MaterialTrialReplayError(
                f"measurement/registration quality is not passed for {trial_id}"
            )


def _observed_test_metrics(
    protocol: Mapping[str, Any], ingest: Mapping[str, Any]
) -> dict[str, dict[str, float]]:
    rows = {str(row["specimen_id"]): row for row in ingest["records"]}
    trials = {str(trial["trial_id"]): trial for trial in protocol["trials"]}
    observed: dict[str, dict[str, float]] = {}
    for trial_id in protocol["splits"]["independent_test"]:
        if trial_id not in rows or trial_id not in trials:
            raise MaterialTrialReplayError(
                f"independent raw trial is missing: {trial_id}"
            )
        record = rows[trial_id]
        trial = trials[trial_id]
        morphology = morphology_from_file(
            trial["morphology_source"]["path"],
            **trial["morphology_config"],
        )
        observed[trial_id] = {
            "wear_volume_mm3": float(record["wear_volume_mm3"]),
            "mass_loss_g": float(record["mass_loss_g"]),
            "wear_depth_mm": float(record["wear_depth_mm"]),
            "roughness_ra_um": float(morphology["roughness"]["Ra_um"]),
            "pit_volume_mm3": float(
                morphology["pit_summary"]["total_volume_mm3"]
            ),
            "pit_max_depth_um": float(
                morphology["pit_summary"]["max_depth_um"]
            ),
        }
    return observed


def _external_pin(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise MaterialTrialReplayError("trusted external pin digest is required")
    try:
        int(value, 16)
    except ValueError as exc:
        raise MaterialTrialReplayError("trusted external pin digest is invalid") from exc
    return value.lower()


def _blocked(reason: str, *, checks: Mapping[str, Any] | None = None) -> dict[str, Any]:
    report = build_blocked_status(reason, details={"checks": dict(checks or {})})
    report["version"] = REPLAY_VERSION
    report["blocked"] = True
    report["failed"] = False
    report["checks"] = dict(checks or {})
    report["report_sha256"] = canonical_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    return report


def replay_material_trials(
    protocol: Mapping[str, Any] | str | Path,
    commitment: Mapping[str, Any] | str | Path,
    *,
    trusted_external_pin_digest: str | None,
) -> dict[str, Any]:
    """Replay raw evidence and return passed, failed, or formally blocked."""
    checks: dict[str, Any] = {
        "protocol_and_raw_hashes_verified": False,
        "certificate_identity_verified": False,
        "identity_only_split_recomputed": False,
        "raw_reingest_performed": False,
        "train_calibration_only_fit": False,
        "existing_calibration_reused": False,
        "morphology_recomputed_from_raw": False,
        "external_pin_verified": False,
        "self_authored_report_trusted": False,
    }
    try:
        raw_commitment = _read_artifact(commitment, "prediction commitment")
        model_section = raw_commitment.get("model")
        if not isinstance(model_section, Mapping):
            raise MaterialTrialReplayError("prediction commitment model is missing")
        model_name = str(model_section.get("model_type", ""))
        preparation = prepare_material_trial_replay(protocol, model_name=model_name)
        frozen = preparation["protocol"]
        checks["identity_only_split_recomputed"] = True
        fit_ids = set(preparation["calibration_binding"]["fit_trial_ids"])
        if fit_ids.intersection(frozen["splits"]["independent_test"]):
            raise MaterialTrialReplayError(
                "independent_test labels leaked into model fit"
            )
        if (
            preparation["independent_test_labels_opened"] is not False
            or preparation["independent_test_morphology_opened"] is not False
        ):
            raise MaterialTrialReplayError(
                "pre-commit preparation opened independent_test evidence"
            )
        checks["train_calibration_only_fit"] = True
        frozen_commitment = validate_material_prediction_commitment(
            raw_commitment, frozen
        )
        pin = _external_pin(trusted_external_pin_digest)
        if frozen_commitment["external_pin_digest"] != pin:
            raise MaterialTrialReplayError("external pin digest mismatch")
        checks["external_pin_verified"] = True
        if frozen_commitment["model"] != preparation["model_descriptor"]:
            raise MaterialTrialReplayError(
                "committed model type/code/config differs from raw replay"
            )
        if frozen_commitment["calibration"] != preparation["calibration_binding"]:
            raise MaterialTrialReplayError(
                "committed calibration binding differs from raw replay"
            )
        prediction_rows = {
            str(row["trial_id"]): row["metrics"]
            for row in frozen_commitment["predictions"]
        }
        for trial_id in frozen["splits"]["independent_test"]:
            recomputed = preparation["predicted_metrics"][trial_id]
            committed = prediction_rows[trial_id]
            for name in ("wear_volume_mm3", "mass_loss_g", "wear_depth_mm"):
                if not math.isclose(
                    float(committed[name]),
                    float(recomputed[name]),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                ):
                    raise MaterialTrialReplayError(
                        f"committed {name} differs from raw model replay"
                    )
        # Reveal starts only after the commitment and externally stored digest
        # have both validated against the fit-only preparation.
        validate_material_trial_protocol(frozen)
        checks.update(
            {
                "protocol_and_raw_hashes_verified": True,
                "certificate_identity_verified": True,
            }
        )
        ingested, _ = _combined_ingest(frozen)
        checks["raw_reingest_performed"] = True
        _validate_protocol_conditions(frozen, ingested)
        existing_calibration = calibrate_material_model(
            ingested, model_name=model_name
        )
        if existing_calibration.get("status") == "blocked":
            raise MaterialTrialReplayError(
                "existing model calibration returned blocked"
            )
        checks["existing_calibration_reused"] = True
        observed_metrics = _observed_test_metrics(frozen, ingested)
        checks["morphology_recomputed_from_raw"] = True
        metric_rows = []
        metric_max = {name: 0.0 for name in PREREGISTERED_METRICS}
        for trial_id in frozen["splits"]["independent_test"]:
            observed = observed_metrics[trial_id]
            predicted = prediction_rows[trial_id]
            errors = {
                name: _relative_error(float(predicted[name]), float(observed[name]))
                for name in PREREGISTERED_METRICS
            }
            for name, value in errors.items():
                metric_max[name] = max(metric_max[name], value)
            metric_rows.append(
                {
                    "trial_id": trial_id,
                    "predicted": dict(predicted),
                    "observed_from_raw": dict(observed),
                    "relative_errors": errors,
                }
            )
        numerical_passed = all(
            math.isfinite(value)
            and (
                value < ERROR_THRESHOLD
                or math.isclose(
                    value,
                    ERROR_THRESHOLD,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            )
            for value in metric_max.values()
        )
        formal = bool(
            frozen["status"] == "measured"
            and frozen["study_phase"] == "formal_candidate"
        )
        status = "passed" if numerical_passed else "failed"
        independent_gate = bool(formal and numerical_passed)
        # Existing report is regenerated only as a diagnostic cross-check. Its
        # own gate is never used by this v2 decision.
        diagnostic = build_material_trial_report(
            ingested,
            existing_calibration,
            make_png=False,
        )
        report = {
            "version": REPLAY_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "status": status,
            "evidence_status": frozen["status"],
            "formal": formal,
            "formal_status": (
                status
                if formal
                else "blocked"
            ),
            "formal_blocked_reason": (
                None
                if formal
                else (
                    "synthetic evidence cannot be formal"
                    if frozen["status"] == "synthetic"
                    else "pilot has fewer than five recommended independent groups"
                )
            ),
            "synthetic_software_contract_only": frozen["status"] == "synthetic",
            "blocked": False,
            "failed": not numerical_passed,
            "accepted": independent_gate,
            "independent_material_gate": independent_gate,
            "official_reference": False,
            "official_accuracy": False,
            "official_reference_gate": False,
            "official_accuracy_gate": False,
            "threshold_relative_error": ERROR_THRESHOLD,
            "threshold_rule": "all preregistered values <= 0.05",
            "metric_max_relative_errors": metric_max,
            "metric_rows": metric_rows,
            "checks": checks,
            "artifact_hashes": {
                "protocol_sha256": frozen["protocol_sha256"],
                "dataset_sha256": frozen["dataset_sha256"],
                "split_sha256": frozen["split_sha256"],
                "commitment_sha256": frozen_commitment["commitment_sha256"],
                "external_pin_digest": pin,
                "raw_ingest_sha256": ingested["dataset_sha256"],
                "model_fit_hash": preparation["calibration_binding"]["model_fit_hash"],
            },
            "group_count": frozen["n_independent_groups"],
            "study_phase": frozen["study_phase"],
            "existing_report_diagnostic": {
                "regenerated_from_raw": True,
                "report_sha256": diagnostic["report_sha256"],
                "trusted_for_v2_gate": False,
            },
            "note": (
                "synthetic data validates software contracts only; formal remains false"
                if frozen["status"] == "synthetic"
                else "independent material evidence; never an official reference"
            ),
        }
        report["report_sha256"] = canonical_sha256(report)
        return report
    except (
        OSError,
        ValidationError,
        ValueError,
        TypeError,
        KeyError,
        MaterialTrialProtocolError,
        MaterialPredictionCommitmentError,
    ) as exc:
        return _blocked(str(exc), checks=checks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay frozen material trial v2 evidence from raw sources"
    )
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--commitment", required=True)
    parser.add_argument("--trusted-external-pin-digest", required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = replay_material_trials(
        args.protocol,
        args.commitment,
        trusted_external_pin_digest=args.trusted_external_pin_digest,
    )
    if args.output is not None:
        safe_json_dump(report, args.output)
    print(json.dumps(jsonable(report), ensure_ascii=False, indent=2))
    return 0 if report.get("independent_material_gate") else 3


replay = replay_material_trials


if __name__ == "__main__":
    raise SystemExit(main())
