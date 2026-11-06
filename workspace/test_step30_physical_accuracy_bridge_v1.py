from __future__ import annotations

import json

import numpy as np

from step30_physical_accuracy_bridge_v1 import build_bridge


def test_bridge_keeps_physical_accuracy_blocked(tmp_path) -> None:
    report = {
        "status": "provisional",
        "prior_contract": {"ai5_used_for_calibration": False},
    }
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps(report), encoding="utf-8")
    np.savez(
        prior.with_suffix(".npz"),
        face_centers_local_mm=np.zeros((8, 3), dtype=np.float32),
        oka_p50_mm_h=np.linspace(1.0, 8.0, 8),
        finnie_p50_mm_h=np.linspace(1.1, 8.1, 8),
        tabakoff_p50_mm_h=np.linspace(0.9, 7.9, 8),
        mansouri_p50_mm_h=np.linspace(1.2, 8.2, 8),
    )
    output = tmp_path / "bridge.json"
    result = build_bridge(prior, output_path=output)
    assert result["status"] == "provisional_physical_accuracy_bridge"
    assert result["physical_spatial_accuracy_status"] == "blocked_without_independent_measured_truth"
    assert result["official_accuracy_gate"] is False
    assert result["model_diversity_diagnostic"]["status"] == "low_spatial_diversity_shared_particle_driver"
    assert result["upstream_particle_input"]["interpretation"].startswith("fixed common")
    assert output.is_file()
    assert output.with_suffix(".npz").is_file()
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["report_sha256"] == result["report_sha256"]
    assert persisted["report_sha256"]
    with np.load(output.with_suffix(".npz"), allow_pickle=False) as data:
        assert set(data.files) == {
            "__metadata_json__",
            "consensus_normalized",
            "disagreement_cv",
            "rank_consensus",
            "oka_normalized",
            "finnie_normalized",
            "tabakoff_normalized",
            "mansouri_normalized",
        }


def test_bridge_rejects_prior_that_used_ai5(tmp_path) -> None:
    report = {
        "status": "provisional",
        "prior_contract": {"ai5_used_for_calibration": True},
    }
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps(report), encoding="utf-8")
    np.savez(
        prior.with_suffix(".npz"),
        face_centers_local_mm=np.zeros((2, 3), dtype=np.float32),
        oka_p50_mm_h=np.ones(2),
        finnie_p50_mm_h=np.ones(2),
        tabakoff_p50_mm_h=np.ones(2),
        mansouri_p50_mm_h=np.ones(2),
    )
    try:
        build_bridge(prior)
    except ValueError as exc:
        assert "AI5-free" in str(exc)
    else:
        raise AssertionError("AI5-calibrated prior must be rejected")
