"""Hermetic tests for strict-v2 continuous particle collisions."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import step30_lagrangian_particle_tracking as tracking


def _mesh(triangles: np.ndarray, parent_ids: np.ndarray | None = None) -> dict:
    triangles = np.asarray(triangles, dtype=np.float64)
    centers = triangles.mean(axis=1)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = cross / np.linalg.norm(cross, axis=1, keepdims=True)
    radii = np.linalg.norm(triangles - centers[:, None, :], axis=2).max(axis=1)
    ids = np.arange(len(triangles), dtype=np.int64) if parent_ids is None else parent_ids
    indices = np.arange(len(triangles), dtype=np.int64)
    return {
        "collision_triangles_mm": triangles,
        "triangles_mm": triangles,
        "collision_face_centers_mm": centers,
        "face_centers_mm": centers,
        "collision_face_normals": normals,
        "face_normals": normals,
        "face_radii_mm": radii,
        "collision_face_ids": np.asarray(ids, dtype=np.int64),
        "parent_face_ids": np.asarray(ids, dtype=np.int64),
        "collision_broadphase_radius_mm": float(max(15.0, 3.0 * np.median(radii))),
        "regular_face_indices": indices,
        "large_face_indices": np.empty(0, dtype=np.int64),
        "regular_face_tree": cKDTree(centers),
        "large_face_tree": None,
        "collision_bounds_mm": [
            triangles.min(axis=(0, 1)).tolist(),
            triangles.max(axis=(0, 1)).tolist(),
        ],
    }


def test_strict_crossing_and_legacy_near_miss() -> None:
    tri = np.array([[[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]]])
    mesh = _mesh(tri)
    tree = mesh["regular_face_tree"]
    hit = tracking.find_segment_hit(
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
        mesh,
        tree,
        1.5,
        collision_policy="strict_v2",
    )
    assert hit is not None and hit[3] == "intersection" and np.isclose(hit[1], 0.5)

    diagnostics = Counter()
    strict_near_miss = tracking.find_segment_hit(
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 0.5]),
        mesh,
        tree,
        1.5,
        collision_policy="strict_v2",
        diagnostics=diagnostics,
    )
    assert strict_near_miss is None
    assert diagnostics["endpoint_near_miss"] == 1
    legacy = tracking.find_segment_hit(
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 0.5]),
        mesh,
        tree,
        1.5,
        collision_policy="legacy_proximity_v1",
    )
    assert legacy is not None and legacy[3] == "fallback"


def test_strict_near_miss_distance_diagnostic_is_optional_but_crossing_is_unchanged() -> None:
    tri = np.array([[[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]]])
    mesh = _mesh(tri)
    tree = mesh["regular_face_tree"]
    diagnostics = Counter()
    near_miss = tracking.find_segment_hit(
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 0.5]),
        mesh,
        tree,
        1.5,
        collision_policy="strict_v2",
        diagnostics=diagnostics,
        strict_near_miss_distance_diagnostics=False,
    )
    assert near_miss is None
    assert diagnostics["endpoint_near_miss"] == 0
    crossing = tracking.find_segment_hit(
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
        mesh,
        tree,
        1.5,
        collision_policy="strict_v2",
        strict_near_miss_distance_diagnostics=False,
    )
    assert crossing is not None and crossing[3] == "intersection"


def test_segment_bounds_rejection_preserves_no_hit_and_hit_behavior() -> None:
    tri = np.array([[[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]]])
    mesh = _mesh(tri)
    tree = mesh["regular_face_tree"]
    assert tracking.find_segment_hit(
        np.array([10.0, 0.0, 1.0]),
        np.array([10.0, 0.0, -1.0]),
        mesh,
        tree,
        1.5,
        collision_policy="strict_v2",
    ) is None
    assert tracking.find_segment_hit(
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
        mesh,
        tree,
        1.5,
        collision_policy="strict_v2",
    ) is not None


def test_strict_broadphase_matches_brute_force_segments() -> None:
    rng = np.random.default_rng(20260728)
    centers = rng.uniform(-20.0, 20.0, size=(40, 3))
    edge_a = rng.normal(size=(40, 3))
    edge_b = rng.normal(size=(40, 3))
    triangles = np.stack([centers - edge_a, centers + edge_a, centers + edge_b], axis=1)
    mesh = _mesh(triangles)
    tree = mesh["regular_face_tree"]
    segments = [
        (center + 3.0 * normal, center - 3.0 * normal)
        for center, normal in zip(centers, mesh["collision_face_normals"])
    ]
    segments.extend(
        zip(
            rng.uniform(-25.0, 25.0, size=(160, 3)),
            rng.uniform(-25.0, 25.0, size=(160, 3)),
        )
    )
    for start, end in segments:
        broadphase = tracking.find_segment_hit(
            start,
            end,
            mesh,
            tree,
            1.5,
            collision_policy="strict_v2",
        )
        hit, toi = tracking.segment_triangle_intersection(start, end, triangles)
        if not hit.any():
            assert broadphase is None
            continue
        candidates = np.flatnonzero(hit)
        order = np.lexsort((candidates, toi[candidates]))
        expected = int(candidates[order[0]])
        assert broadphase is not None
        assert broadphase[4] == expected
        assert np.isclose(broadphase[1], toi[expected], atol=1e-12)


def test_strict_deterministic_tie_breaking_and_face_suppression() -> None:
    tri = np.array([
        [[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]],
        [[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]],
    ])
    mesh = _mesh(tri, np.array([10, 11]))
    tree = mesh["regular_face_tree"]
    hit = tracking.find_segment_hit(
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
        mesh,
        tree,
        0.0,
        collision_policy="strict_v2",
    )
    assert hit is not None and hit[0] == 10 and hit[4] == 0
    suppressed = tracking.find_segment_hit(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        mesh,
        tree,
        0.0,
        collision_policy="strict_v2",
        suppressed_parent_face_id=10,
    )
    assert suppressed is not None and suppressed[0] == 11


def test_cfd_velocity_interpolation_is_exact_and_symmetric() -> None:
    centers = np.array([
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    velocity = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    tree = cKDTree(centers)
    nearest, spacing, exact = tracking._sample_cfd_velocity(
        tree, velocity, centers[1], "idw_k8_v1"
    )
    assert nearest == 0.0
    assert spacing > 0.0
    assert np.array_equal(exact, velocity[1])

    _, _, symmetric = tracking._sample_cfd_velocity(
        tree, velocity, np.zeros(3), "idw_k8_v1"
    )
    assert np.allclose(symmetric, [1.0, 0.0, 0.0])


def test_cfd_velocity_interpolation_rejects_unknown_policy() -> None:
    tree = cKDTree(np.array([[0.0, 0.0, 0.0]]))
    with np.testing.assert_raises(ValueError):
        tracking._sample_cfd_velocity(tree, np.zeros((1, 3)), np.zeros(3), "unknown")


def test_drag_state_matches_constant_fluid_analytic_solution() -> None:
    end, velocity = tracking._drag_state(
        np.zeros(3),
        np.array([10.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        0.1,
        0.05,
        None,
    )
    decay = np.exp(-0.5)
    assert np.allclose(velocity, [2.0 + 8.0 * decay, 0.0, 0.0])
    assert np.allclose(end, [(2.0 * 0.05 + 8.0 * 0.1 * (1.0 - decay)) * 1000.0, 0.0, 0.0])


def test_strict_particle_integrator_policies_are_reproducible() -> None:
    centers = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    fluid = np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    tree = cKDTree(centers)
    position = np.zeros(3)
    velocity = np.array([10.0, 0.0, 0.0])
    particle_tau = 0.1
    diameter = 5.0e-5
    dt = 0.05
    tau_eff = tracking._effective_drag_tau_s(
        particle_tau, diameter, velocity, fluid[0]
    )
    expected = tracking._drag_state(position, velocity, fluid[0], tau_eff, dt, None)
    frozen = tracking._strict_particle_state(
        tree,
        fluid,
        position,
        velocity,
        particle_tau,
        diameter,
        dt,
        None,
        "idw_k8_v1",
        "frozen_exponential_v1",
    )
    assert np.array_equal(frozen[0], expected[0])
    assert np.array_equal(frozen[1], expected[1])

    no_slip_velocity = np.array([2.0, 0.0, 0.0])
    midpoint = tracking._strict_particle_state(
        tree,
        fluid,
        position,
        no_slip_velocity,
        particle_tau,
        diameter,
        dt,
        None,
        "idw_k8_v1",
        "midpoint_exponential_v2",
    )
    assert np.allclose(midpoint[0], [100.0, 0.0, 0.0])
    assert np.array_equal(midpoint[1], no_slip_velocity)


def test_strict_tracker_processes_multiple_toi_in_remaining_substep(tmp_path: Path) -> None:
    triangles = np.array([
        [[1.0, -500.0, -500.0], [1.0, 500.0, -500.0], [1.0, 0.0, 500.0]],
        [[2.0, -500.0, -500.0], [2.0, 500.0, -500.0], [2.0, 0.0, 500.0]],
    ])
    mesh = _mesh(triangles, np.array([0, 1]))
    mesh.update({
        "n_parent_faces": 2,
        "face_area_m2": np.ones(2),
        "vertices_mm": triangles.reshape(-1, 3),
        "faces": np.array([[0, 1, 2], [3, 4, 5]]),
        "bounds_mm": [[1.0, -500.0, -500.0], [2.0, 500.0, 500.0]],
        "surface_area_m2": 1.0,
    })
    np.savez(tmp_path / "jet_rotation.npz", R=np.eye(3), nozzle_pos=np.zeros(3))
    cfd = {
        "cell_centers_mm": np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]]),
        "U_ms": np.array([[104.8, 0.0, 0.0], [104.8, 0.0, 0.0]]),
        "center_bounds_mm": [[-100.0, -500.0, -500.0], [1000.0, 500.0, 500.0]],
    }
    cfg = tracking.TrackingConfig(
        collision_policy="strict_v2",
        n_parcels=1,
        n_injected=1,
        max_impacts=2,
        dt_s=1e-4,
        max_time_s=1e-4,
        max_query_distance_mm=1e9,
        particle_cfl=1.0,
        e_normal=-1.0,
        e_tangent=1.0,
    )
    result = tracking._run_tracking_strict(cfg, tmp_path, mesh, cfd, np.random.default_rng(42))
    rows = result["impact_rows"]
    assert len(rows) == 2
    assert [row["face_id"] for row in rows] == [0, 1]
    assert 0.0 < rows[0]["time_s"] < rows[1]["time_s"] < cfg.dt_s
    assert result["fallback_count"] == 0


def test_strict_max_substep_makes_outer_dt_schedule_invariant(tmp_path: Path) -> None:
    triangles = np.array([
        [[1.0, -500.0, -500.0], [1.0, 500.0, -500.0], [1.0, 0.0, 500.0]],
    ])
    mesh = _mesh(triangles, np.array([0]))
    mesh.update({
        "n_parent_faces": 1,
        "face_area_m2": np.ones(1),
        "vertices_mm": triangles.reshape(-1, 3),
        "faces": np.array([[0, 1, 2]]),
        "bounds_mm": [[1.0, -500.0, -500.0], [1.0, 500.0, 500.0]],
        "surface_area_m2": 1.0,
    })
    np.savez(tmp_path / "jet_rotation.npz", R=np.eye(3), nozzle_pos=np.zeros(3))
    cfd = {
        "cell_centers_mm": np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]]),
        "U_ms": np.array([[104.8, 0.0, 0.0], [104.8, 0.0, 0.0]]),
        "center_bounds_mm": [[-100.0, -500.0, -500.0], [1000.0, 500.0, 500.0]],
    }
    rows = []
    for dt_s in (1e-5, 5e-6, 2.5e-6):
        cfg = tracking.TrackingConfig(
            collision_policy="strict_v2",
            n_parcels=1,
            n_injected=1,
            max_impacts=1,
            dt_s=dt_s,
            strict_max_substep_s=2.5e-6,
            max_time_s=2e-5,
            max_query_distance_mm=1e9,
            particle_cfl=1.0,
        )
        result = tracking._run_tracking_strict(
            cfg, tmp_path, mesh, cfd, np.random.default_rng(42)
        )
        assert len(result["impact_rows"]) == 1
        rows.append(result["impact_rows"][0])
    assert [row["face_id"] for row in rows] == [0, 0, 0]
    assert np.allclose([row["time_s"] for row in rows], rows[0]["time_s"], atol=1e-15)
    assert np.allclose([row["position_local_mm"] for row in rows], rows[0]["position_local_mm"], atol=1e-12)
    assert np.allclose([row["velocity_ms"] for row in rows], rows[0]["velocity_ms"], atol=1e-12)
