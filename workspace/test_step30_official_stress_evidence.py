from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from step30_official_stress_evidence import (
    STRICT_RELATIVE_THRESHOLD,
    build_130mw_leave_one_report,
    build_official_stress_evidence,
    fit_dynamic_by_nozzle_and_opening,
    official_table_records,
    raw_ai5_diagnostic,
)


def test_130mw_is_excluded_from_six_nozzle_training() -> None:
    fitted = fit_dynamic_by_nozzle_and_opening(official_table_records())
    six = fitted["models_by_nozzle_count"]["6"]
    assert "130MW" not in six["training_conditions"]
    assert fitted["held_out_not_in_training"] is True
    assert fitted["prediction"]["predicted_peak_to_peak_mpa"] == pytest.approx(
        26.42370992564617
    )
    assert fitted["prediction"]["relative_error"] < STRICT_RELATIVE_THRESHOLD


def test_single_review_keeps_130mw_strictly_blocked(tmp_path: pathlib.Path) -> None:
    pdf = tmp_path / "official.pdf"
    pdf.write_bytes(b"official report fixture")
    evidence = build_official_stress_evidence(
        pdf, transcription_reviews=[{"reviewer_id": "reviewer_a", "status": "completed_match"}]
    )
    report = build_130mw_leave_one_report(evidence)
    assert evidence["double_review_completed"] is False
    assert report["status"] == "blocked"
    assert report["strict_scoring_status"] == "candidate_pass_pending_transcription_review"


def test_two_matching_reviews_make_table_eligible_but_not_official_accuracy(
    tmp_path: pathlib.Path,
) -> None:
    pdf = tmp_path / "official.pdf"
    pdf.write_bytes(b"official report fixture")
    evidence = build_official_stress_evidence(
        pdf,
        transcription_reviews=[
            {"reviewer_id": "reviewer_a", "status": "completed_match"},
            {"reviewer_id": "reviewer_b", "status": "completed_match"},
        ],
    )
    report = build_130mw_leave_one_report(evidence)
    assert evidence["double_review_completed"] is True
    assert report["status"] == "passed"
    assert report["official_accuracy_gate"] is False


def test_raw_ai5_uses_peak_to_peak_as_diagnostic(tmp_path: pathlib.Path) -> None:
    values = np.asarray([1.0, 3.0, 2.0])
    (tmp_path / "30MW.txt").write_text(
        "time[s]\tAI5-04 [MPa]\n0.0\t1.0\n0.1\t3.0\n0.2\t2.0\n",
        encoding="utf-8",
    )
    diagnostic = raw_ai5_diagnostic(tmp_path, official_table_records())
    row = next(item for item in diagnostic["records"] if item["condition"] == "30MW")
    assert row["raw_peak_to_peak_mpa"] == pytest.approx(2.0)
    assert diagnostic["raw_data_used_for_strict_scoring"] is False
