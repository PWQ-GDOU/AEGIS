"""Regression tests for the CFD-driven Step-30 particle post-processor."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

import step30_lagrangian_particle_tracking as s


def main() -> None:
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


if __name__ == "__main__":
    main()
