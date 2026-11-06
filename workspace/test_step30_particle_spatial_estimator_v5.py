from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from step30_particle_spatial_estimator_v4 import build_parent_face_adjacency
from step30_particle_spatial_estimator_v5 import (
    ParticleSpatialEstimatorV5Error,
    _validated_aggregate_context,
    build_topology_aggregate_context,
    build_connected_aggregate_partition,
    estimate_topology_aggregate,
    save_estimate_v5,
)
from step30_validation_common import canonical_sha256


def _topology() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Faces 0-4 form a strip; face 5 is a disconnected component.
    triangles = np.asarray(
        [
            [0, 1, 2],
            [2, 1, 3],
            [2, 3, 4],
            [4, 3, 5],
            [4, 5, 6],
            [10, 11, 12],
        ],
        dtype=np.int64,
    )
    parents = np.arange(6, dtype=np.int64)
    areas = np.asarray([1.0, 2.0, 3.0, 1.0, 2.0, 4.0], dtype=np.float64)
    return triangles, parents, areas


def test_connected_partition_is_deterministic_and_keeps_components_separate() -> None:
    triangles, parents, areas = _topology()
    adjacency, _ = build_parent_face_adjacency(triangles, parents, 6)
    first = build_connected_aggregate_partition(
        adjacency, areas, max_parent_faces_per_aggregate=3
    )
    second = build_connected_aggregate_partition(
        adjacency, areas, max_parent_faces_per_aggregate=3
    )
    assert np.array_equal(first["aggregate_id"], second["aggregate_id"])
    assert (
        first["partition"]["partition_sha256"]
        == second["partition"]["partition_sha256"]
    )
    assert first["aggregate_id"][5] != first["aggregate_id"][0]
    assert int(first["aggregate_face_count"].sum()) == 6


def test_topology_context_rejects_a_different_scale() -> None:
    triangles, parents, areas = _topology()
    topology = {
        "faces": triangles,
        "collision_face_ids": parents,
        "face_area_m2": areas,
        "n_parent_faces": 6,
    }
    context = build_topology_aggregate_context(
        topology, max_parent_faces_per_aggregate=4096
    )
    accepted = _validated_aggregate_context(
        topology, context, max_parent_faces_per_aggregate=4096
    )
    assert accepted["partition"]["partition"]["partition_sha256"] == context[
        "partition"
    ]["partition"]["partition_sha256"]
    with pytest.raises(ParticleSpatialEstimatorV5Error):
        _validated_aggregate_context(
            topology, context, max_parent_faces_per_aggregate=8192
        )


def test_aggregate_conserves_mass_and_is_area_scale_invariant() -> None:
    triangles, parents, areas = _topology()
    adjacency, _ = build_parent_face_adjacency(triangles, parents, 6)
    partition = build_connected_aggregate_partition(
        adjacency, areas, max_parent_faces_per_aggregate=3
    )
    raw = np.asarray([2.0, 0.0, 3.0, 0.0, 1.0, 0.0])
    face_id = np.asarray([0, 0, 2, 4], dtype=np.int64)
    contribution = np.asarray([1.0, 1.0, 3.0, 1.0])
    first = estimate_topology_aggregate(
        raw,
        areas,
        partition["aggregate_id"],
        contribution_parent_face_id=face_id,
        contribution_target_mass_kg=contribution,
    )
    second = estimate_topology_aggregate(
        raw,
        areas * 5.0,
        partition["aggregate_id"],
        contribution_parent_face_id=face_id,
        contribution_target_mass_kg=contribution,
    )
    assert first["total_projected_target_mass_kg"] == pytest.approx(raw.sum())
    assert first["mass_conservation_error_kg"] == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(
        first["surface_aggregate_face_target_mass_kg"] / raw.sum(),
        second["surface_aggregate_face_target_mass_kg"] / raw.sum(),
    )
    assert float(np.max(first["aggregate_local_ess"])) >= 2.0


def test_aggregate_rejects_mismatched_impact_support() -> None:
    triangles, parents, areas = _topology()
    adjacency, _ = build_parent_face_adjacency(triangles, parents, 6)
    partition = build_connected_aggregate_partition(
        adjacency, areas, max_parent_faces_per_aggregate=3
    )
    with pytest.raises(ParticleSpatialEstimatorV5Error):
        estimate_topology_aggregate(
            np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            areas,
            partition["aggregate_id"],
            contribution_parent_face_id=np.asarray([0], dtype=np.int64),
            contribution_target_mass_kg=np.asarray([0.5]),
        )


def test_aggregate_accepts_float32_serialization_scale_error() -> None:
    triangles, parents, areas = _topology()
    adjacency, _ = build_parent_face_adjacency(triangles, parents, 6)
    partition = build_connected_aggregate_partition(
        adjacency, areas, max_parent_faces_per_aggregate=3
    )
    result = estimate_topology_aggregate(
        np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        areas,
        partition["aggregate_id"],
        contribution_parent_face_id=np.asarray([0], dtype=np.int64),
        contribution_target_mass_kg=np.asarray([1.0 + 1.0e-8]),
    )
    assert result["impact_support_mass_relative_error"] == pytest.approx(1.0e-8)
    assert result["impact_support_mass_relative_tolerance"] == pytest.approx(2.0e-8)


def test_saved_v5_field_has_partition_and_report_hash(
    tmp_path: pathlib.Path,
) -> None:
    triangles, parents, areas = _topology()
    adjacency, _ = build_parent_face_adjacency(triangles, parents, 6)
    partition = build_connected_aggregate_partition(
        adjacency, areas, max_parent_faces_per_aggregate=3
    )
    field = estimate_topology_aggregate(
        np.asarray([1.0, 0.0, 0.0, 2.0, 0.0, 0.0]),
        areas,
        partition["aggregate_id"],
    )
    metadata = {
        "version": "fixture-v5",
        "input_bindings": {"mesh_signature": "mesh-fixture"},
        "partition": {"partition_sha256": "partition-fixture"},
        "field_contract_sha256": "field-contract-fixture",
    }
    report = save_estimate_v5(
        {
            "metadata": metadata,
            "field": field,
            "topology": {"face_area_m2": areas},
            "aggregate_id": partition["aggregate_id"],
            "aggregate_face_count": partition["aggregate_face_count"],
        },
        tmp_path / "field.npz",
    )
    with np.load(tmp_path / "field.npz", allow_pickle=False) as saved:
        assert str(saved["partition_sha256"].item()) == "partition-fixture"
        assert str(saved["field_contract_sha256"].item()) == "field-contract-fixture"
        assert "report_sha256" not in saved.files
    assert report["report_sha256"] == canonical_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
