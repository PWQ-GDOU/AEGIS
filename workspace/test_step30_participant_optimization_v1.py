from __future__ import annotations

import json

import numpy as np

from step30_participant_optimization_v1 import (
    MODEL_NAMES,
    REGIONS,
    build_face_region_mapping,
    build_optimized_pipeline,
)


def _centers_and_areas() -> tuple[np.ndarray, np.ndarray]:
    # A deterministic, non-degenerate face-center fixture for contract tests.
    centers = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.2),
            (2.0, 0.0, 0.4),
            (3.0, 0.0, 0.6),
            (4.0, 0.0, 0.8),
            (5.0, 0.0, 1.0),
            (0.0, 1.0, 0.1),
            (1.0, 1.0, 0.3),
            (2.0, 1.0, 0.5),
            (3.0, 1.0, 0.7),
            (4.0, 1.0, 0.9),
            (5.0, 1.0, 1.1),
        ],
        dtype=np.float64,
    )
    return centers, np.full(len(centers), 1.0e-4, dtype=np.float64)


def _prior(root, name: str, offset: float) -> str:
    path = root / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "status": "provisional",
                "prior_contract": {"ai5_used_for_calibration": False},
                "input_bindings": {"total_particle_mass_kg": 0.1},
            }
        ),
        encoding="utf-8",
    )
    centers, areas = _centers_and_areas()
    base = np.linspace(0.0, 0.12, len(centers), dtype=np.float64) + offset
    base[:2] = 0.0
    np.savez(
        path.with_suffix(".npz"),
        face_centers_local_mm=centers,
        face_area_m2=areas,
        oka_p50_mm_h=base,
        finnie_p50_mm_h=base * 0.9 + 0.002,
        tabakoff_p50_mm_h=base * 1.1,
        mansouri_p50_mm_h=base * 0.8 + 0.001,
    )
    return str(path)


def _literature(root) -> str:
    path = root / "literature.json"
    path.write_text(
        json.dumps(
            {
                "reference_kind": "literature_reference",
                "official_reference_gate": False,
                "regions": [
                    {"region_id": region, "literature_support_score": index + 1}
                    for index, region in enumerate(REGIONS)
                ],
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def test_region_mapping_covers_faces_and_depends_only_on_geometry() -> None:
    centers, areas = _centers_and_areas()
    first_report, first_labels = build_face_region_mapping(centers, areas)
    second_report, second_labels = build_face_region_mapping(centers, areas * 7.0)

    assert len(first_labels) == len(centers)
    assert set(first_labels).issubset(set(REGIONS))
    assert np.array_equal(first_labels, second_labels)
    assert first_report["rules"] == second_report["rules"]
    assert first_report["axis_metadata"] == second_report["axis_metadata"]


def test_pipeline_records_convergence_calibration_combination_and_gates(tmp_path) -> None:
    centers, areas = _centers_and_areas()
    domain = tmp_path / "domain.npz"
    np.savez(domain, face_centers_local_mm=centers, face_area_m2=areas)
    merged = _prior(tmp_path, "merged", 0.0)
    seeds = [_prior(tmp_path, f"seed_{index}", 0.002 * index) for index in range(5)]
    literature = _literature(tmp_path)
    output = tmp_path / "participant.json"

    result = build_optimized_pipeline(merged, seeds, domain, literature, output_path=output)

    assert result["active_face_convergence"]["n_seeds"] == 5
    assert result["active_face_convergence"]["models"]["oka"]["active_cv_p95"] > 0.0
    assert result["material_amplitude_calibration"]["status"] == "synthetic_material_reference"
    assert result["baseline_worn_reference"]["independence"] == "does not read any model erosion field"
    assert result["baseline_worn_reference"]["literature_binding"]["sha256"]
    assert result["model_combination"]["weight_sum"] == 1.0
    assert result["official_accuracy_gate"] is False
    assert result["official_reference_gate"] is False
    assert result["fem_gate"] == "blocked"
    assert result["competition_self_benchmark_status"] == "conditional_participant_built_reference"
    assert result["conservative_score"] <= result["nominal_score"] <= result["optimistic_score"]
    assert result["competition_self_benchmark_score_interval"]["lower"] <= result["nominal_score"] <= result["competition_self_benchmark_score_interval"]["upper"]
    assert output.is_file() and output.with_suffix(".npz").is_file()

    with np.load(output.with_suffix(".npz"), allow_pickle=False) as data:
        required = {
            "face_region_id",
            "composite_prediction_depth_mm",
            "baseline_depth_mm",
            "worn_depth_low_mm",
            "worn_depth_nominal_mm",
            "worn_depth_high_mm",
        }
        required.update(f"{model}_calibrated_depth_mm" for model in MODEL_NAMES)
        assert required.issubset(set(data.files))
        assert data["face_region_id"].shape == (len(centers),)
