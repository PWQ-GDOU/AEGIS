from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from step30_particle_spatial_estimator_v4 import (
    KERNEL_CONTRACT,
    ParticleSpatialEstimatorError,
    build_parent_face_adjacency,
    estimate_surface_kernel,
    save_estimate,
)
from step30_validation_common import canonical_sha256


def _topology() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Faces 0, 1, 2 form a connected strip. Face 3 is disconnected.
    triangles = np.asarray(
        [
            [0, 1, 2],
            [2, 1, 3],
            [2, 3, 4],
            [10, 11, 12],
        ],
        dtype=np.int64,
    )
    parents = np.asarray([0, 1, 2, 3], dtype=np.int64)
    areas = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    return triangles, parents, areas


def test_kernel_conserves_total_target_mass() -> None:
    triangles, parents, areas = _topology()
    adjacency, _ = build_parent_face_adjacency(triangles, parents, 4)
    result = estimate_surface_kernel(
        np.asarray([10.0, 0.0, 5.0, 0.0]), areas, adjacency
    )
    assert result["total_kernel_target_mass_kg"] == pytest.approx(15.0)
    assert result["mass_conservation_error_kg"] == pytest.approx(0.0, abs=1e-15)
    assert np.sum(result["surface_kernel_face_target_mass_kg"]) == pytest.approx(15.0)


def test_uniform_area_scaling_keeps_mass_fractions_invariant() -> None:
    triangles, parents, areas = _topology()
    adjacency, _ = build_parent_face_adjacency(triangles, parents, 4)
    raw = np.asarray([2.0, 0.0, 0.0, 0.0])
    first = estimate_surface_kernel(raw, areas, adjacency)
    second = estimate_surface_kernel(raw, areas * 7.0, adjacency)
    assert np.allclose(
        first["surface_kernel_face_target_mass_kg"] / raw.sum(),
        second["surface_kernel_face_target_mass_kg"] / raw.sum(),
    )


def test_disconnected_surface_does_not_receive_kernel_mass() -> None:
    triangles, parents, areas = _topology()
    adjacency, _ = build_parent_face_adjacency(triangles, parents, 4)
    result = estimate_surface_kernel(np.asarray([1.0, 0.0, 0.0, 0.0]), areas, adjacency)
    assert result["surface_kernel_face_target_mass_kg"][3] == pytest.approx(0.0)
    assert result["local_support_mass_kg"][3] == pytest.approx(0.0)


def test_estimator_is_deterministic_and_reports_frozen_bandwidth() -> None:
    triangles, parents, areas = _topology()
    adjacency, first_hash = build_parent_face_adjacency(triangles, parents, 4)
    _, second_hash = build_parent_face_adjacency(triangles, parents, 4)
    first = estimate_surface_kernel(np.asarray([1.0, 2.0, 0.0, 0.0]), areas, adjacency)
    second = estimate_surface_kernel(np.asarray([1.0, 2.0, 0.0, 0.0]), areas, adjacency)
    assert first_hash == second_hash
    assert first["bandwidth_hops"] == KERNEL_CONTRACT["default_bandwidth_hops"]
    assert np.array_equal(
        first["surface_kernel_face_target_mass_kg"],
        second["surface_kernel_face_target_mass_kg"],
    )


def test_nonfrozen_bandwidth_is_rejected() -> None:
    triangles, parents, areas = _topology()
    adjacency, _ = build_parent_face_adjacency(triangles, parents, 4)
    with pytest.raises(ParticleSpatialEstimatorError):
        estimate_surface_kernel(
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            areas,
            adjacency,
            bandwidth_hops=99,
        )


def test_v3_module_remains_available_for_historical_compatibility() -> None:
    import step30_particle_seed_convergence as legacy

    assert legacy.VERSION == "step30-particle-seed-convergence-v3"


def test_saved_field_uses_stable_field_contract_and_final_report_hash(
    tmp_path: pathlib.Path,
) -> None:
    triangles, parents, areas = _topology()
    adjacency, _ = build_parent_face_adjacency(triangles, parents, 4)
    field = estimate_surface_kernel(np.asarray([1.0, 0.0, 0.5, 0.0]), areas, adjacency)
    metadata = {
        "version": "fixture",
        "input_bindings": {"mesh_signature": "mesh-fixture"},
        "field_contract_sha256": "field-contract-fixture",
    }
    report = save_estimate(
        {"metadata": metadata, "field": field, "topology": {"face_area_m2": areas}},
        tmp_path / "field.npz",
    )
    with np.load(tmp_path / "field.npz", allow_pickle=False) as saved:
        assert str(saved["field_contract_sha256"].item()) == "field-contract-fixture"
        assert "report_sha256" not in saved.files
    assert report["report_sha256"] == canonical_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
