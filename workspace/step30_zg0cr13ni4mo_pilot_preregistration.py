"""Preregistered ZG0Cr13Ni4Mo water-sand jet pilot evidence chain.

This module plans measurement without fabricating certificates, calibration
records, PSD values, or wear results.  A single heat/batch is explicitly a
measured-pilot design and cannot open the existing multi-group v2 independent
material replay gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from step30_validation_common import canonical_sha256, safe_json_dump, sha256_file


VERSION = "step30-zg0cr13ni4mo-water-sand-pilot-preregistration-v1"
MATERIAL_GRADE = "ZG0Cr13Ni4Mo"
SPEEDS_M_S = (50.0, 90.0)
ANGLES_DEG = (20.0, 45.0, 90.0)
REPEATS = (1, 2)
EROSION_MODELS = ("oka", "finnie", "tabakoff", "mansouri")
REQUIRED_INSTRUMENTS = (
    "balance",
    "hardness_tester",
    "profilometer",
    "psd_analyzer",
)


class ZG0Cr13Ni4MoPilotError(ValueError):
    """Raised for an invalid or mutation-prone pilot preregistration."""


def _trial_id(speed: float, angle: float, repeat: int) -> str:
    return f"ZG13-{int(speed):02d}-{int(angle):02d}-R{repeat}"


def _blind_code(trial_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{trial_id}".encode("ascii")).hexdigest()
    return f"PILOT-{digest[:8].upper()}"


def planned_trials(*, randomization_seed: int) -> list[dict[str, Any]]:
    """Return the fixed 2 x 3 x 2 coupon plan in a deterministic run order."""
    records = [
        {
            "trial_id": _trial_id(speed, angle, repeat),
            "speed_m_s": speed,
            "impact_angle_deg": angle,
            "repeat": repeat,
        }
        for speed in SPEEDS_M_S
        for angle in ANGLES_DEG
        for repeat in REPEATS
    ]
    if len(records) != 12:
        raise ZG0Cr13Ni4MoPilotError("pilot design must contain exactly 12 coupons")
    ordered = sorted(
        records,
        key=lambda item: hashlib.sha256(
            f"{randomization_seed}:{item['trial_id']}".encode("ascii")
        ).hexdigest(),
    )
    return [
        {
            **record,
            "blind_specimen_code": _blind_code(record["trial_id"], randomization_seed),
            "run_order": index + 1,
            "metrology_blind_to": [
                "speed_m_s",
                "impact_angle_deg",
                "repeat",
                "run_order",
            ],
        }
        for index, record in enumerate(ordered)
    ]


def model_commitment() -> dict[str, Any]:
    """Freeze comparison workflow before any pilot measurement is available."""
    workspace = Path(__file__).resolve().parent
    calibration = workspace / "step30_material_model_calibration.py"
    return {
        "models": list(EROSION_MODELS),
        "comparison_rule": (
            "Run all frozen models through the same ingest/replay-compatible metrics; "
            "do not choose a winner from pilot data before the comparison is replayed."
        ),
        "prediction_commitment": (
            "After pilot raw files are immutable, predictions may be replayed only "
            "with this frozen model list and raw-data hashes."
        ),
        "implementation": {
            "path": str(calibration),
            "sha256": sha256_file(calibration),
        },
        "post_measurement_change_allowed": False,
    }


def build_preregistration(
    *,
    campaign_id: str = "ZG0Cr13Ni4Mo-water-sand-pilot-001",
    randomization_seed: int = 20260821,
    cumulative_sediment_mass_kg: float = 1.0,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a pilot plan. The PSD/certificate gates remain blocked until supplied."""
    if (
        not isinstance(campaign_id, str)
        or not campaign_id.strip()
        or isinstance(randomization_seed, bool)
        or not isinstance(randomization_seed, int)
        or cumulative_sediment_mass_kg <= 0.0
    ):
        raise ZG0Cr13Ni4MoPilotError(
            "campaign ID, randomization seed, and sediment mass setpoint are invalid"
        )
    trials = planned_trials(randomization_seed=randomization_seed)
    result = {
        "version": VERSION,
        "campaign_id": campaign_id.strip(),
        "status": "preregistered_pending_execution_evidence",
        "study_phase": "single_heat_measured_pilot",
        "material_identity": {
            "grade": MATERIAL_GRADE,
            "same_material_rule": (
                "Mill/heat certificate must state exactly ZG0Cr13Ni4Mo; "
                "04Cr13Ni5Mo is not automatically equivalent."
            ),
            "certificate_required_before_exposure": True,
            "chemistry_required_before_exposure": True,
            "heat_treatment_required_before_exposure": True,
            "hardness_hv_required_before_exposure": True,
            "density_required_before_exposure": True,
        },
        "test_rig": {
            "type": "controlled_water_sand_jet",
            "abrasive": "quartz_sand",
            "psd_contract": {
                "required_before_first_exposure": True,
                "measurements": ["D10_um", "D50_um", "D90_um"],
                "fixed_for_all_12_coupons_after_measurement": True,
                "status": "blocked_pending_measured_psd",
            },
            "exposure_contract": {
                "cumulative_sediment_mass_kg_per_coupon": float(
                    cumulative_sediment_mass_kg
                ),
                "setpoint_status": "frozen_preregistration_setpoint",
                "not_a_measured_result": True,
            },
        },
        "design": {
            "n_coupons": 12,
            "speed_m_s": list(SPEEDS_M_S),
            "impact_angle_deg": list(ANGLES_DEG),
            "repeats_per_condition": 2,
            "trials": trials,
            "randomization_seed": randomization_seed,
            "blinding": {
                "specimen_codes_are_blinded_for_metrology": True,
                "profilometry_scan_repeats_minimum": 3,
            },
        },
        "measurement_contract": {
            "primary": ["mass_loss_g", "wear_volume_mm3", "wear_depth_mm"],
            "morphology": ["roughness_ra_um", "pit_morphology"],
            "profilometry_scan_repeats_minimum": 3,
            "required_raw_data": [
                "balance_raw",
                "three_dimensional_profile_raw",
                "roughness_raw",
                "pit_morphology_images_or_micrographs",
            ],
            "required_evidence_before_measurement_release": [
                "mill_heat_certificate",
                "chemistry_record",
                "heat_treatment_record",
                "hardness_hv_measurement",
                "density_measurement_or_certificate",
                "instrument_calibration_files",
                "specimen_photos",
                "raw_file_sha256_for_every_measurement",
            ],
        },
        "required_instrument_kinds": list(REQUIRED_INSTRUMENTS),
        "model_comparison_commitment": model_commitment(),
        "v2_ingest_replay_bridge": {
            "source_manifest_version": "step30-material-trial-source-manifest-v2",
            "ingest_module": "step30_material_trial_ingest.py",
            "replay_module": "step30_material_trial_replay.py",
            "single_heat_behavior": (
                "A completed pilot may be reported as measured_pilot, but v2 independent "
                "replay remains blocked because one heat/batch is one hard identity group."
            ),
            "upgrade_requirement": (
                "At least 3 independent heat/batch groups are required for a pilot split; "
                "5 independent groups are recommended before formal candidate status."
            ),
        },
        "execution_gate": "blocked_pending_certificate_psd_calibration_and_raw_measurement_evidence",
        "independent_material_gate": False,
        "official_reference_gate": False,
        "official_accuracy_gate": False,
        "fem_gate": "blocked",
        "limitations": [
            "This artifact contains no measured wear result.",
            "One furnace heat/batch cannot demonstrate independent material generalization.",
            "The pilot does not confirm DEC geometry, ROOT/WETTED selections, or official erosion accuracy.",
        ],
    }
    result["preregistration_sha256"] = canonical_sha256(result)
    if output_path is not None:
        safe_json_dump(result, output_path)
    return result


def validate_measured_pilot_handoff(
    preregistration: Mapping[str, Any],
    *,
    observed_trial_ids: Sequence[str],
    material_grade: str,
    distinct_heat_ids: Sequence[str],
) -> dict[str, Any]:
    """Validate future pilot completeness and explicitly keep gates closed."""
    expected = {row["trial_id"] for row in preregistration["design"]["trials"]}
    observed = {str(item) for item in observed_trial_ids}
    if material_grade != MATERIAL_GRADE:
        raise ZG0Cr13Ni4MoPilotError(
            "measured pilot grade is not ZG0Cr13Ni4Mo; no automatic material equivalence"
        )
    if observed != expected:
        raise ZG0Cr13Ni4MoPilotError("measured pilot trials differ from the preregistered 12 coupons")
    unique_heats = sorted({str(item) for item in distinct_heat_ids if str(item).strip()})
    if not unique_heats:
        raise ZG0Cr13Ni4MoPilotError("at least one heat ID is required")
    return {
        "version": VERSION,
        "status": "measured_pilot",
        "n_observed_trials": len(observed),
        "n_distinct_heat_ids": len(unique_heats),
        "independent_material_gate": False,
        "official_reference_gate": False,
        "official_accuracy_gate": False,
        "v2_replay_status": (
            "blocked_single_heat"
            if len(unique_heats) < 3
            else "eligible_for_v2_protocol_construction"
        ),
        "formal_candidate_status": (
            "blocked_less_than_five_independent_heats"
            if len(unique_heats) < 5
            else "requires_v2_protocol_and_replay"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a ZG0Cr13Ni4Mo water-sand pilot preregistration")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign-id", default="ZG0Cr13Ni4Mo-water-sand-pilot-001")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--sediment-mass-kg", type=float, default=1.0)
    args = parser.parse_args(argv)
    report = build_preregistration(
        campaign_id=args.campaign_id,
        randomization_seed=args.seed,
        cumulative_sediment_mass_kg=args.sediment_mass_kg,
        output_path=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
