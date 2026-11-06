"""Hermetic strict_v2 CCD validation cases.

The report is numerical-algorithm evidence only; it is not a physical erosion
validation.  Proximity hits are tested as diagnostics and never as strict
physical impacts.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from step30_lagrangian_particle_tracking import find_segment_hit, segment_triangle_intersection

VERSION = "step30-strict-collision-validation-v1"


def _mesh() -> dict:
    triangles = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
        ], dtype=float,
    )
    centers = triangles.mean(axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    radii = np.linalg.norm(triangles - centers[:, None, :], axis=2).max(axis=1)
    return {
        "collision_triangles_mm": triangles,
        "collision_face_ids": np.array([0, 1], dtype=np.int64),
        "collision_face_centers_mm": centers,
        "face_radii_mm": radii,
        "regular_face_indices": np.array([0, 1], dtype=np.int64),
        "large_face_indices": np.array([], dtype=np.int64),
        "regular_face_tree": cKDTree(centers),
        "large_face_tree": None,
        "collision_broadphase_radius_mm": 2.0,
    }


def run(output: Path | None = None) -> dict:
    tri = _mesh()["collision_triangles_mm"][:1]
    cases = []

    def case(name: str, start, end, expected: bool) -> None:
        hit, t = segment_triangle_intersection(np.asarray(start, float), np.asarray(end, float), tri)
        actual = bool(hit[0])
        cases.append({"name": name, "expected": expected, "actual": actual, "passed": actual == expected, "toi": float(t[0]) if actual else None})

    case("face_crossing", [0.2, 0.2, -1.0], [0.2, 0.2, 1.0], True)
    case("edge_crossing", [-0.2, 0.0, -1.0], [0.2, 0.0, 1.0], True)
    case("vertex_crossing", [-0.2, -0.2, -1.0], [0.2, 0.2, 1.0], True)
    case("parallel", [0.2, 0.2, 1.0], [0.8, 0.2, 1.0], False)
    case("coplanar", [0.2, 0.2, 0.0], [0.8, 0.2, 0.0], False)
    case("near_miss", [0.2, 0.2, 1.0], [0.2, 0.2, 1.0001], False)

    mesh = _mesh()
    diagnostics = Counter()
    strict_hit = find_segment_hit(
        np.array([0.2, 0.2, -1.0]), np.array([0.2, 0.2, 1.0]), mesh,
        mesh["regular_face_tree"], 0.25, "strict_v2", diagnostics=diagnostics,
    )
    strict_near = find_segment_hit(
        np.array([0.2, 0.2, 1.0]), np.array([0.2, 0.2, 1.1]), mesh,
        mesh["regular_face_tree"], 0.25, "strict_v2", diagnostics=diagnostics,
    )
    legacy_near = find_segment_hit(
        np.array([0.2, 0.2, 1.0]), np.array([0.2, 0.2, 0.1]), mesh,
        mesh["regular_face_tree"], 0.25, "legacy_proximity_v1", diagnostics=diagnostics,
    )
    policy_checks = {
        "strict_intersection_is_physical": strict_hit is not None and strict_hit[3] == "intersection",
        "strict_near_miss_is_not_physical": strict_near is None,
        "legacy_fallback_is_explicit": legacy_near is not None and legacy_near[3] == "fallback",
        "near_miss_diagnostic_only": diagnostics["endpoint_near_miss"] >= 1,
    }
    # Tie-breaking: two triangles at identical TOI must choose the lower index.
    tied_triangles = np.asarray([
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [1, 0, 0]],
    ], dtype=float)
    tie_mesh = dict(mesh)
    tie_mesh.update({
        "collision_triangles_mm": tied_triangles,
        "collision_face_ids": np.array([10, 11], dtype=np.int64),
        "collision_face_centers_mm": tied_triangles.mean(axis=1),
        "face_radii_mm": np.linalg.norm(tied_triangles - tied_triangles.mean(axis=1)[:, None, :], axis=2).max(axis=1),
        "regular_face_tree": cKDTree(tied_triangles.mean(axis=1)),
    })
    tie = find_segment_hit(np.array([0.2, 0.2, -1.0]), np.array([0.2, 0.2, 1.0]), tie_mesh, tie_mesh["regular_face_tree"], 0.1, "strict_v2")
    policy_checks["deterministic_tie_break"] = tie is not None and tie[4] == 0
    report = {
        "version": VERSION,
        "collision_policy": "strict_v2",
        "status": "passed" if all(item["passed"] for item in cases) and all(policy_checks.values()) else "failed",
        "cases": cases,
        "policy_checks": policy_checks,
        "diagnostics": dict(diagnostics),
        "physical_fallback_hits": 0,
        "interpretation": "strict_v2 excludes proximity fallback from physical impacts; legacy fallback is tested only under explicit legacy_proximity_v1.",
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 2)
