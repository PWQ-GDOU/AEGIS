from __future__ import annotations

import json

import numpy as np

from step30_eight_step_optimization_v1 import build_eight_step_package


def test_progress_package_keeps_missing_matrix_cells_blocked(tmp_path) -> None:
    convergence = tmp_path / "convergence.json"
    convergence.write_text(
        json.dumps(
            {
                "version": "test-convergence",
                "scales": {
                    "5000": {"n_runs": 5, "estimators": {"topology_aggregate": {"passed": False}}},
                    "20000": {
                        "n_runs": 5,
                        "estimators": {
                            "topology_aggregate": {
                                "passed": True,
                                "area_weighted_spearman_median": 0.91,
                                "mass_hotspot_iou_median": 1.0,
                                "spatial_js_divergence_max": 0.04,
                            }
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    participant = tmp_path / "participant.json"
    participant.write_text(
        json.dumps(
            {
                "version": "test-participant",
                "status": "conditional",
                "conservative_score": 62.0,
                "nominal_score": 82.0,
                "optimistic_score": 82.0,
                "competition_self_benchmark_score_interval": {"lower": 50.0, "upper": 90.0},
                "competition_self_benchmark_status": "conditional_participant_built_reference",
                "official_accuracy_gate": False,
                "official_reference_gate": False,
                "fem_gate": "blocked",
                "face_region_mapping": {"version": "mapping", "claim_boundary": "participant only"},
                "material_amplitude_calibration": {"status": "synthetic_material_reference", "material": "ZG0Cr13Ni4Mo", "independent_measured_pilot": False},
                "baseline_worn_reference": {"status": "participant_built_synthetic", "measured_truth": False},
                "model_combination": {"status": "preregistered", "weights": {"oka": 0.25}, "weight_sum": 1.0, "posthoc_weight_change_allowed": False},
            }
        ),
        encoding="utf-8",
    )
    domain = tmp_path / "domain.npz"
    np.savez(domain, face_centers_local_mm=np.zeros((4, 3)), face_area_m2=np.ones(4))

    result = build_eight_step_package(convergence, participant, domain)

    matrix = result["step_status"]["particle_convergence_matrix"]
    assert matrix["selected_candidate"]["particle_count"] == 20000
    assert matrix["selected_candidate"]["seed_count"] == 5
    assert any(
        cell["particle_count"] == 50000 and cell["status"] == "blocked_missing_runs"
        for cell in matrix["cells"]
    )
    assert result["step_status"]["score_recalculation"]["conservative_score"] == 62.0
    assert result["step_status"]["dec_mapping_hypotheses"]["face_count"] == 4

