"""Regression tests for the CFD-driven Step-30 particle post-processor."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import step30_lagrangian_particle_tracking as s


def _run_production_integration() -> None:
    si_case = Path(__file__).parent / "of_case" / "pelton_bucket_jet_si"
    block_mesh = (si_case / "system" / "blockMeshDict").read_text()
    omega_initial = (si_case / "0.orig" / "omega").read_text()
    assert "scale 0.001;" in block_mesh
    assert "radius 0.12;" in (si_case / "system" / "topoSetDict").read_text()
    assert "uniform 697.44025;" in omega_initial
    assert "4118640" not in omega_initial

    tri = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    hit, t = s.segment_triangle_intersection(
        np.array([0.2, 0.2, 1.0]), np.array([0.2, 0.2, -1.0]), tri
    )
    assert bool(hit[0]) and abs(float(t[0]) - 0.5) < 1e-12

    mesh = {
        "triangles_mm": tri,
        "collision_triangles_mm": tri,
        "face_centers_mm": tri.mean(axis=1),
        "face_radii_mm": np.array([1.0]),
        "face_normals": np.array([[0.0, 0.0, 1.0]]),
        "collision_face_normals": np.array([[0.0, 0.0, 1.0]]),
        "collision_face_ids": np.array([0]),
        "parent_face_ids": np.array([0]),
    }
    tree = cKDTree(mesh["face_centers_mm"])
    fallback = s.find_segment_hit(
        np.array([0.2, 0.2, 1.0]), np.array([0.2, 0.2, 0.2]), mesh, tree, 0.3
    )
    assert fallback is not None and fallback[3] == "fallback"
    assert fallback[0] == 0 and fallback[4] == 0
    assert np.allclose(fallback[2], [0.2, 0.2, 0.0])

    reflected = s.reflect_velocity(
        np.array([3.0, 4.0, 0.0]), np.array([0.0, 1.0, 0.0]), 0.8, 0.9
    )
    assert np.allclose(reflected, [2.7, -3.2, 0.0])

    cfg = s.TrackingConfig(n_parcels=32, n_injected=32)
    rng = np.random.default_rng(42)
    diam = s.sample_rosin_rammler(rng, 32, cfg)
    particle_mass, parcel_mass, total_mass = s.compute_parcel_masses(diam, cfg)
    assert np.all((diam >= cfg.d_min_m) & (diam <= cfg.d_max_m))
    assert np.all(particle_mass > 0) and np.all(parcel_mass > 0)
    assert abs(parcel_mass.sum() - total_mass) / total_mass < 1e-12
    assert parcel_mass[-1] != parcel_mass[0]

    agg = s.aggregate_face_impacts(
        2,
        np.array([1.0, 2.0]),
        np.zeros((2, 3)),
        np.zeros((2, 3)),
        np.array([0, 1, 1]),
        np.array([0.1, 0.2, 0.4]),
        np.array([2.0, 3.0, 5.0]),
        10.0,
    )
    assert np.allclose(agg["face_mass_kg"], [2.0, 8.0])
    assert np.allclose(agg["face_target_mass_kg"], [0.2, 2.6])
    assert np.allclose(agg["ER_mass_ratio_face"], [0.1, 0.325])

    for variant in ("baseline", "mrf"):
        field = s.load_cfd_velocity_field(s.OUTPUT_DIR, variant)
        assert field["n_cells"] == 722342
        assert np.isfinite(field["U_ms"]).all()
    assert s.load_cfd_velocity_field(s.OUTPUT_DIR, "baseline")["speed_stats_ms"]["max"] < 500.0
    assert s.load_cfd_velocity_field(s.OUTPUT_DIR, "mrf")["speed_stats_ms"]["max"] > 500.0

    from step30_cfd_topology import load_topology_artifact, validate_topology_arrays
    native = load_topology_artifact()
    assert native["backend"] == "cfd_topology"
    assert native["source_coordinate_units"] == "mm"
    assert native["internal_coordinate_units"] == "mm"
    assert native["coordinate_scale_to_mm"] == 1.0

    si_field = s.load_cfd_velocity_field(s.OUTPUT_DIR, "native_si", "3000")
    assert si_field["source_coordinate_units"] == "m"
    assert si_field["internal_coordinate_units"] == "mm"
    assert si_field["coordinate_scale_to_mm"] == 1000.0
    assert si_field["n_cells"] == 424146
    assert np.isclose(si_field["center_bounds_mm"][0][0], 10.0, atol=0.1)
    assert si_field["speed_stats_ms"]["max"] < 500.0
    si = load_topology_artifact(s.OUTPUT_DIR / "bucket_topology_si.npz")
    assert si["source_coordinate_units"] == "m"
    assert si["internal_coordinate_units"] == "mm"
    assert si["coordinate_scale_to_mm"] == 1000.0
    assert si["n_parent_faces"] == 74802
    assert np.isclose(si["surface_area_m2"], 1.318275465, rtol=1e-6)

    assert native["backend"] == "cfd_topology"
    assert native["face_id_namespace"] == "openfoam_bucket_patch"
    assert native["n_parent_faces"] == 74803
    assert len(native["collision_triangles_mm"]) == 179568
    assert np.all(native["face_area_m2"] > 0.0)
    assert np.allclose(np.linalg.norm(native["face_normals"], axis=1), 1.0, atol=1e-4)
    assert native["collision_face_ids"].min() == 0
    assert native["collision_face_ids"].max() == 74802

    rot = dict(np.load(str(s.OUTPUT_DIR / "jet_rotation.npz")))
    gpr = dict(np.load(str(s.OUTPUT_DIR / "vertices_full_gpr.npz")))
    mapped = s._sample_model_on_faces(native, gpr, rot)
    assert np.median(mapped["model_dist_mm"]) < 10.0
    assert np.percentile(mapped["model_dist_mm"], 95) < 15.0

    try:
        validate_topology_arrays(
            np.zeros((3, 3)), np.array([[0, 1, 2]]), np.array([1]),
            np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), np.array([1.0]),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range parent face ID was not rejected")

    print("step30 CFD regression tests passed")


def test_production_artifacts() -> None:
    if os.environ.get("STEP30_PRODUCTION_INTEGRATION") != "1":
        import pytest
        pytest.skip("set STEP30_PRODUCTION_INTEGRATION=1 to validate ignored production artifacts")
    _run_production_integration()


if __name__ == "__main__":
    _run_production_integration()
