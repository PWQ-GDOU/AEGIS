"""Synthetic-only contract tests for the formal material trial v2 replay."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from step30_material_model_calibration import oka_volume_mm3_per_kg
from step30_material_morphology_metrics import morphology_from_file
from step30_material_trial_protocol import (
    MaterialPredictionCommitmentError,
    MaterialTrialProtocolError,
    build_material_prediction_commitment,
    build_material_trial_protocol,
    recompute_commitment_hash,
    recompute_external_pin_digest,
    source_file_descriptor,
)
from step30_material_trial_replay import (
    commitment_predictions_from_preparation,
    prepare_material_trial_replay,
    replay_material_trials,
)


STATUS = "synthetic"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, allow_nan=False), encoding="utf-8"
    )


def _instrument(root: Path, index: int, kind: str) -> dict[str, Any]:
    instrument_id = f"{kind}-{index}"
    calibration = root / f"{instrument_id}-calibration.txt"
    calibration.write_text(
        f"synthetic software fixture calibration {instrument_id}\n", encoding="utf-8"
    )
    return {
        "instrument_id": instrument_id,
        "kind": kind,
        "model": "fixture-model",
        "serial_number": f"fixture-serial-{index}-{kind}",
        "calibrated_at": "2026-01-01T00:00:00Z",
        "valid_until": "2027-01-01T00:00:00Z",
        "calibration": source_file_descriptor(
            calibration, status=STATUS, kind="calibration_certificate"
        ),
    }


def _trial(root: Path, index: int, *, label: str = "unseen") -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    trial_id = f"trial-{index}"
    run_id = f"run-{index}"
    campaign_id = f"campaign-{index}"
    specimen_id = f"specimen-{index}"
    asset_id = f"asset-{index}"
    heat_id = f"heat-{index}"
    batch = f"batch-{index}"
    hardness_gpa = 3.6 + 0.1 * index
    chemistry = {"unit": "wt_pct", "values": {"Cr": 13.0, "C": 0.05}}
    certificate_path = root / f"certificate-{index}.json"
    _write_json(
        certificate_path,
        {
            "material_grade": "GX4CrNi13-4",
            "heat_id": heat_id,
            "material_batch": batch,
            "chemistry": chemistry,
            "heat_treatment": "quenched-and-tempered",
        },
    )
    instruments = [
        _instrument(root, index, "balance"),
        _instrument(root, index, "hardness_tester"),
        _instrument(root, index, "profilometer"),
    ]
    speed = 44.0 + index
    angle = 35.0 + index
    d10, d50, d90 = 40.0, 55.0 + index, 80.0
    sediment_mass_kg = 0.1
    area_mm2 = 100.0
    density = 7850.0
    specific_volume = float(
        oka_volume_mm3_per_kg(
            speed, angle, d50, hardness_gpa=hardness_gpa
        )
    )
    volume = specific_volume * sediment_mass_kg
    mass_loss = volume * density * 1.0e-6
    before_mass = 100.0
    row = {
        "trial_id": trial_id,
        "run_id": run_id,
        "campaign_id": campaign_id,
        "specimen_id": specimen_id,
        "asset_id": asset_id,
        "heat_id": heat_id,
        "material_batch": batch,
        "material": "GX4CrNi13-4",
        # These source labels are deliberately non-authoritative.
        "split": "test" if index % 2 else "calibration",
        "posthoc_measurement_label": label,
        "speed_m_s": speed,
        "impact_angle_deg": angle,
        "psd_d10_um": d10,
        "psd_d50_um": d50,
        "psd_d90_um": d90,
        "sediment_mass_kg": sediment_mass_kg,
        "exposure_s": 600.0,
        "before_mass_g": before_mass,
        "after_mass_g": before_mass - mass_loss,
        "mass_loss_g": mass_loss,
        "mass_uncertainty_g": 1.0e-5,
        "material_density_kg_m3": density,
        "exposed_area_mm2": area_mm2,
        "wear_depth_mm": volume / area_mm2,
        "depth_uncertainty_mm": 1.0e-5,
        "wear_volume_mm3": volume,
        "volume_uncertainty_mm3": 0.001,
        "roughness_ra_um": 1.0,
        "roughness_rq_um": 1.2,
        "roughness_rz_um": 5.0,
        "roughness_uncertainty_um": 0.1,
        "pit_count": 1,
        "pit_max_depth_um": 100.0,
        "pit_mean_depth_um": 100.0,
        "pit_mean_width_um": 100.0,
        "pit_volume_mm3": 0.001,
        "pit_uncertainty_mm3": 0.0001,
        "registration_passed": True,
        "measurement_quality_passed": True,
        "status": STATUS,
    }
    source_path = root / f"trial-source-{index}.json"
    _write_json(source_path, {"rows": [row]})
    morphology_path = root / f"morphology-{index}.json"
    field = [[0.0 for _ in range(5)] for _ in range(5)]
    field[2][2] = -0.1 - index * 0.002
    _write_json(morphology_path, {"height_field": field})
    return {
        "trial_id": trial_id,
        "run_id": run_id,
        "asset_id": asset_id,
        "asset_type": "coupon",
        "material_identity_known": True,
        "mixed_heat": False,
        "campaign_id": campaign_id,
        "specimen_id": specimen_id,
        "status": STATUS,
        "provenance": {
            "origin": "deterministic_simulation",
            "campaign_id": campaign_id,
            "organization": "synthetic-contract-test",
            "acquisition_method": "deterministic synthetic fixture",
            "acquired_at": "2026-01-01T00:00:00Z",
        },
        "source": source_file_descriptor(
            source_path, status=STATUS, kind="trial_data"
        ),
        "morphology_source": source_file_descriptor(
            morphology_path, status=STATUS, kind="height_field"
        ),
        "morphology_config": {
            "input_kind": "height_field",
            "height_unit": "mm",
            "spacing_x": 0.1,
            "spacing_y": 0.1,
            "spacing_unit": "mm",
            "threshold": 0.02,
            "threshold_unit": "mm",
            "height_uncertainty": 0.001,
            "uncertainty_unit": "mm",
        },
        "material": {
            "grade": "GX4CrNi13-4",
            "heat_id": heat_id,
            "material_batch": batch,
            "certificate": source_file_descriptor(
                certificate_path, status=STATUS, kind="material_certificate"
            ),
            "chemistry": chemistry,
            "heat_treatment": "quenched-and-tempered",
            "coating_status": "uncoated",
            "weld_status": "unwelded",
            "hardness": {
                "value": hardness_gpa,
                "uncertainty": 0.03,
                "unit": "GPa",
                "source": "measured",
                "is_default": False,
                "measurement_id": f"hardness-measurement-{index}",
                "instrument_id": f"hardness_tester-{index}",
            },
            "density": {
                "value_kg_m3": density,
                "uncertainty_kg_m3": 10.0,
                "source": "measured",
            },
        },
        "instruments": instruments,
        "conditions": {
            "speed_m_s": speed,
            "impact_angle_deg": angle,
            "psd_d10_um": d10,
            "psd_d50_um": d50,
            "psd_d90_um": d90,
            "sediment_mass_kg": sediment_mass_kg,
            "exposure_s": 600.0,
            "exposed_area_mm2": area_mm2,
        },
        "units": {},
    }


def _evidence(
    root: Path, *, morphology_scale: float = 1.0, count: int = 5
) -> dict[str, Any]:
    trials = [_trial(root, index) for index in range(count)]
    protocol = build_material_trial_protocol(trials, status=STATUS, seed=731)
    preparation = prepare_material_trial_replay(protocol)
    by_id = {trial["trial_id"]: trial for trial in protocol["trials"]}
    morphology_predictions = {}
    for trial_id in protocol["splits"]["independent_test"]:
        trial = by_id[trial_id]
        metrics = morphology_from_file(
            trial["morphology_source"]["path"], **trial["morphology_config"]
        )
        morphology_predictions[trial_id] = {
            "roughness_ra_um": metrics["roughness"]["Ra_um"]
            * morphology_scale,
            "pit_volume_mm3": metrics["pit_summary"]["total_volume_mm3"]
            * morphology_scale,
            "pit_max_depth_um": metrics["pit_summary"]["max_depth_um"]
            * morphology_scale,
        }
    assert preparation["independent_test_labels_opened"] is False
    assert preparation["independent_test_morphology_opened"] is False
    assert "observed_metrics" not in preparation
    assert "ingest" not in preparation
    predictions = commitment_predictions_from_preparation(
        preparation, morphology_predictions=morphology_predictions
    )
    commitment = build_material_prediction_commitment(
        protocol,
        model_descriptor=preparation["model_descriptor"],
        calibration_binding=preparation["calibration_binding"],
        predictions=predictions,
    )
    return {
        "trials": trials,
        "protocol": protocol,
        "preparation": preparation,
        "commitment": commitment,
    }


def test_measurement_labels_and_source_hashes_do_not_influence_split(
    tmp_path: Path,
) -> None:
    trials = [_trial(tmp_path, index, label=f"low-{index}") for index in range(5)]
    first = build_material_trial_protocol(trials, status=STATUS, seed=1701)
    relabelled = deepcopy(trials)
    for index, trial in enumerate(relabelled):
        source = Path(trial["source"]["path"])
        document = json.loads(source.read_text(encoding="utf-8"))
        document["rows"][0]["posthoc_measurement_label"] = f"high-{999-index}"
        document["rows"][0]["unregistered_wear_label"] = 999.0 - index
        _write_json(source, document)
        trial["source"] = source_file_descriptor(
            source, status=STATUS, kind="trial_data"
        )
    second = build_material_trial_protocol(relabelled, status=STATUS, seed=1701)

    assert first["splits"] == second["splits"]
    assert first["groups"] == second["groups"]
    assert first["split_contract"]["measurement_labels_used"] is False
    assert first["split_contract"]["source_hashes_used"] is False


def test_three_groups_are_pilot_and_five_are_formal_candidate(tmp_path: Path) -> None:
    pilot = build_material_trial_protocol(
        [_trial(tmp_path / "pilot", index) for index in range(3)],
        status=STATUS,
    )
    formal_sized = build_material_trial_protocol(
        [_trial(tmp_path / "formal", index) for index in range(5)],
        status=STATUS,
    )

    assert pilot["study_phase"] == "pilot"
    assert formal_sized["study_phase"] == "formal_candidate"
    assert formal_sized["evidence_role"] == "synthetic_software_contract_only"


def test_mixed_heat_or_default_hardness_is_formally_blocked(tmp_path: Path) -> None:
    trials = [_trial(tmp_path, index) for index in range(5)]
    trials[1]["asset_id"] = trials[0]["asset_id"]
    with pytest.raises(MaterialTrialProtocolError, match="mixed heat"):
        build_material_trial_protocol(trials, status=STATUS)

    trials = [_trial(tmp_path / "unknown", index) for index in range(5)]
    trials[0]["material_identity_known"] = False
    with pytest.raises(MaterialTrialProtocolError, match="identity is unknown"):
        build_material_trial_protocol(trials, status=STATUS)

    trials = [_trial(tmp_path / "batch", index) for index in range(5)]
    shared_batch = trials[0]["material"]["material_batch"]
    trials[1]["material"]["material_batch"] = shared_batch
    certificate = Path(trials[1]["material"]["certificate"]["path"])
    certificate_document = json.loads(certificate.read_text(encoding="utf-8"))
    certificate_document["material_batch"] = shared_batch
    _write_json(certificate, certificate_document)
    trials[1]["material"]["certificate"] = source_file_descriptor(
        certificate, status=STATUS, kind="material_certificate"
    )
    with pytest.raises(MaterialTrialProtocolError, match="mixed heat/material batch"):
        build_material_trial_protocol(trials, status=STATUS)

    trials = [_trial(tmp_path / "hardness", index) for index in range(5)]
    trials[0]["material"]["hardness"]["is_default"] = True
    with pytest.raises(MaterialTrialProtocolError, match="default constant"):
        build_material_trial_protocol(trials, status=STATUS)


def test_independent_test_labels_do_not_change_fit_scale(tmp_path: Path) -> None:
    trials = [_trial(tmp_path, index) for index in range(5)]
    first_protocol = build_material_trial_protocol(trials, status=STATUS, seed=731)
    first = prepare_material_trial_replay(first_protocol)
    test_id = first_protocol["splits"]["independent_test"][0]
    changed_trial = next(trial for trial in trials if trial["trial_id"] == test_id)
    source = Path(changed_trial["source"]["path"])
    document = json.loads(source.read_text(encoding="utf-8"))
    row = document["rows"][0]
    row["wear_volume_mm3"] *= 1.5
    row["mass_loss_g"] = row["wear_volume_mm3"] * row["material_density_kg_m3"] * 1.0e-6
    row["after_mass_g"] = row["before_mass_g"] - row["mass_loss_g"]
    row["wear_depth_mm"] = row["wear_volume_mm3"] / row["exposed_area_mm2"]
    _write_json(source, document)
    changed_trial["source"] = source_file_descriptor(
        source, status=STATUS, kind="trial_data"
    )
    second_protocol = build_material_trial_protocol(trials, status=STATUS, seed=731)
    second = prepare_material_trial_replay(second_protocol)

    assert first_protocol["splits"] == second_protocol["splits"]
    assert first["fit_contract"]["fit_scale"] == pytest.approx(
        second["fit_contract"]["fit_scale"], rel=0.0, abs=0.0
    )
    assert test_id not in second["fit_contract"]["fit_trial_ids"]


def test_certificate_hash_tamper_is_blocked_on_raw_replay(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    certificate = Path(evidence["protocol"]["trials"][0]["material"]["certificate"]["path"])
    certificate.write_bytes(certificate.read_bytes() + b"\n")

    report = replay_material_trials(
        evidence["protocol"],
        evidence["commitment"],
        trusted_external_pin_digest=evidence["commitment"]["external_pin_digest"],
    )

    assert report["status"] == "blocked"
    assert report["independent_material_gate"] is False
    assert report["official_reference"] is False
    assert report["official_accuracy"] is False


def test_independent_test_leakage_in_commitment_is_blocked(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    protocol = evidence["protocol"]
    leaked = deepcopy(evidence["commitment"])
    leaked["calibration"]["fit_trial_ids"].append(
        protocol["splits"]["independent_test"][0]
    )
    leaked["external_pin_digest"] = recompute_external_pin_digest(leaked)
    leaked["commitment_sha256"] = recompute_commitment_hash(leaked)

    with pytest.raises(MaterialPredictionCommitmentError, match="train/calibration"):
        from step30_material_trial_protocol import validate_material_prediction_commitment
        validate_material_prediction_commitment(leaked, protocol)

    report = replay_material_trials(
        protocol,
        leaked,
        trusted_external_pin_digest=leaked["external_pin_digest"],
    )
    assert report["status"] == "blocked"
    assert report["failed"] is False


def test_failed_and_synthetic_passed_semantics_are_distinct(tmp_path: Path) -> None:
    passed_evidence = _evidence(tmp_path / "passed", morphology_scale=1.04)
    passed = replay_material_trials(
        passed_evidence["protocol"],
        passed_evidence["commitment"],
        trusted_external_pin_digest=passed_evidence["commitment"]["external_pin_digest"],
    )
    assert passed["status"] == "passed"
    assert passed["blocked"] is False
    assert passed["formal"] is False
    assert passed["synthetic_software_contract_only"] is True
    assert passed["independent_material_gate"] is False
    assert passed["official_reference"] is False
    assert passed["official_accuracy"] is False

    boundary_evidence = _evidence(tmp_path / "boundary", morphology_scale=1.05)
    boundary = replay_material_trials(
        boundary_evidence["protocol"],
        boundary_evidence["commitment"],
        trusted_external_pin_digest=boundary_evidence["commitment"]["external_pin_digest"],
    )
    assert boundary["status"] == "passed"
    assert max(boundary["metric_max_relative_errors"].values()) == pytest.approx(
        0.05
    )

    failed_evidence = _evidence(tmp_path / "failed", morphology_scale=1.051)
    failed = replay_material_trials(
        failed_evidence["protocol"],
        failed_evidence["commitment"],
        trusted_external_pin_digest=failed_evidence["commitment"]["external_pin_digest"],
    )
    assert failed["status"] == "failed"
    assert failed["failed"] is True
    assert failed["blocked"] is False
    assert max(failed["metric_max_relative_errors"].values()) > 0.05
    assert failed["independent_material_gate"] is False


def test_missing_or_wrong_external_pin_is_blocked(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    for pin in (None, "0" * 64):
        report = replay_material_trials(
            evidence["protocol"],
            evidence["commitment"],
            trusted_external_pin_digest=pin,
        )
        assert report["status"] == "blocked"
        assert report["independent_material_gate"] is False


def test_wrong_pin_blocks_before_independent_morphology_reveal(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    test_id = evidence["protocol"]["splits"]["independent_test"][0]
    trial = next(
        trial
        for trial in evidence["protocol"]["trials"]
        if trial["trial_id"] == test_id
    )
    Path(trial["morphology_source"]["path"]).unlink()

    report = replay_material_trials(
        evidence["protocol"],
        evidence["commitment"],
        trusted_external_pin_digest="0" * 64,
    )

    assert report["status"] == "blocked"
    assert "external pin digest mismatch" in report["reason"]
    assert report["checks"]["external_pin_verified"] is False
    assert report["checks"]["morphology_recomputed_from_raw"] is False


def test_blocked_template_cannot_claim_formal_evidence() -> None:
    template = Path(__file__).parent / "templates" / "step30-material-trial-source-manifest-v2.json"
    document = json.loads(template.read_text(encoding="utf-8"))

    assert document["status"] == "blocked"
    assert document["formal"] is False
    assert document["independent_material_gate"] is False
    assert document["official_reference"] is False
    assert document["official_accuracy"] is False
