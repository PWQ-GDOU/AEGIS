from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from step30_zg0cr13ni4mo_pilot_preregistration import (
    MATERIAL_GRADE,
    ZG0Cr13Ni4MoPilotError,
    build_preregistration,
    validate_measured_pilot_handoff,
)


def test_preregistration_has_exact_12_coupon_factorial_design() -> None:
    report = build_preregistration(campaign_id="ZG13-CAMPAIGN-FIXTURE")
    trials = report["design"]["trials"]
    assert len(trials) == 12
    assert {(row["speed_m_s"], row["impact_angle_deg"]) for row in trials} == {
        (50.0, 20.0),
        (50.0, 45.0),
        (50.0, 90.0),
        (90.0, 20.0),
        (90.0, 45.0),
        (90.0, 90.0),
    }
    assert report["material_identity"]["grade"] == MATERIAL_GRADE
    assert report["campaign_id"] == "ZG13-CAMPAIGN-FIXTURE"
    assert report["execution_gate"].startswith("blocked")


def test_campaign_id_must_be_a_nonempty_string() -> None:
    with pytest.raises(ZG0Cr13Ni4MoPilotError, match="campaign ID"):
        build_preregistration(campaign_id="  ")


def test_single_heat_measured_pilot_never_opens_independent_gate() -> None:
    prereg = build_preregistration()
    handoff = validate_measured_pilot_handoff(
        prereg,
        observed_trial_ids=[row["trial_id"] for row in prereg["design"]["trials"]],
        material_grade=MATERIAL_GRADE,
        distinct_heat_ids=["HEAT-001"],
    )
    assert handoff["status"] == "measured_pilot"
    assert handoff["v2_replay_status"] == "blocked_single_heat"
    assert handoff["independent_material_gate"] is False


def test_material_grade_is_not_silently_equated() -> None:
    prereg = build_preregistration()
    with pytest.raises(ZG0Cr13Ni4MoPilotError, match="not ZG0Cr13Ni4Mo"):
        validate_measured_pilot_handoff(
            prereg,
            observed_trial_ids=[row["trial_id"] for row in prereg["design"]["trials"]],
            material_grade="04Cr13Ni5Mo",
            distinct_heat_ids=["HEAT-001"],
        )
