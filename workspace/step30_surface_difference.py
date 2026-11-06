"""Geometry-based wear depth between a registered worn scan and a reference.

Every worn sample is projected by spatial search onto reference points or
reference triangles. Array row position is never treated as correspondence.
For oriented reference normals, positive ``wear_depth_mm`` means material was
lost in the direction opposite the outward reference normal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from step30_validation_common import ValidationError, canonical_sha256, finite_array, jsonable


SURFACE_DIFFERENCE_VERSION = "step30-surface-difference-v1"


def _points(value: Any, name: str, *, minimum: int = 1) -> np.ndarray:
    array = finite_array(value, name, ndim=2)
    if array.shape[1] != 3 or len(array) < minimum:
        raise ValidationError(f"{name} must have shape (n >= {minimum}, 3)")
    return np.asarray(array, dtype=np.float64)


def _hashable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {"dtype": str(contiguous.dtype), "shape": list(contiguous.shape), "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {str(key): _hashable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_hashable(item) for item in value]
    return value


def _unit_normals(value: Any, name: str, count: int) -> np.ndarray:
    array = finite_array(value, name, ndim=2, shape=(count, 3))
    lengths = np.linalg.norm(array, axis=1)
    if np.any(lengths <= 1.0e-12):
        raise ValidationError(f"{name} contains zero-length normals")
    return array / lengths[:, None]


def _triangle_indices(value: Any, n_points: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) == 0:
        raise ValidationError("reference triangles must have nonempty shape (m, 3)")
    if not np.issubdtype(raw.dtype, np.integer):
        if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
            raise ValidationError("reference triangles must contain integer indices")
    triangles = np.asarray(raw, dtype=np.int64)
    if np.any(triangles < 0) or np.any(triangles >= n_points):
        raise ValidationError("reference triangle vertex index is out of range")
    return triangles


def _closest_point_triangle(point: np.ndarray, triangle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the Euclidean closest point and barycentric coordinates."""
    a, b, c = triangle
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a, np.array([1.0, 0.0, 0.0])
    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b, np.array([0.0, 1.0, 0.0])
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab, np.array([1.0 - v, v, 0.0])
    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c, np.array([0.0, 0.0, 1.0])
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac, np.array([1.0 - w, 0.0, w])
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), np.array([0.0, 1.0 - w, w])
    denominator = va + vb + vc
    if abs(denominator) <= 1.0e-18:
        raise ValidationError("reference contains a degenerate triangle")
    v = vb / denominator
    w = vc / denominator
    return a + ab * v + ac * w, np.array([1.0 - v - w, v, w])


def _mesh_geometry(
    reference: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xyz = reference[triangles]
    cross = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    if np.any(twice_area <= 1.0e-12):
        raise ValidationError("reference contains degenerate triangles")
    normals = cross / twice_area[:, None]
    centers = xyz.mean(axis=1)
    radii = np.linalg.norm(xyz - centers[:, None, :], axis=2).max(axis=1)
    return xyz, normals, centers, radii


def _project_to_triangles(
    query: np.ndarray,
    reference: np.ndarray,
    triangles: np.ndarray,
    vertex_normals: np.ndarray | None,
) -> dict[str, np.ndarray]:
    xyz, triangle_normals, centers, radii = _mesh_geometry(reference, triangles)
    center_tree = cKDTree(centers)
    maximum_radius = float(np.max(radii))
    projected = np.empty_like(query)
    normals = np.empty_like(query)
    triangle_id = np.empty(len(query), dtype=np.int64)
    barycentric = np.empty((len(query), 3), dtype=np.float64)
    distances = np.empty(len(query), dtype=np.float64)
    for row, point in enumerate(query):
        _, initial_id = center_tree.query(point, k=1)
        initial_id = int(initial_id)
        initial_point, initial_barycentric = _closest_point_triangle(point, xyz[initial_id])
        best_distance = float(np.linalg.norm(point - initial_point))
        candidates = center_tree.query_ball_point(point, r=best_distance + maximum_radius)
        if initial_id not in candidates:
            candidates.append(initial_id)
        best_id = initial_id
        best_point = initial_point
        best_barycentric = initial_barycentric
        for candidate in candidates:
            candidate_point, candidate_barycentric = _closest_point_triangle(point, xyz[candidate])
            candidate_distance = float(np.linalg.norm(point - candidate_point))
            if candidate_distance < best_distance:
                best_distance = candidate_distance
                best_id = int(candidate)
                best_point = candidate_point
                best_barycentric = candidate_barycentric
        projected[row] = best_point
        triangle_id[row] = best_id
        barycentric[row] = best_barycentric
        distances[row] = best_distance
        if vertex_normals is None:
            normals[row] = triangle_normals[best_id]
        else:
            interpolated = best_barycentric @ vertex_normals[triangles[best_id]]
            length = float(np.linalg.norm(interpolated))
            if length <= 1.0e-12:
                raise ValidationError("interpolated reference normal is zero")
            normals[row] = interpolated / length
    return {
        "projected_points_mm": projected,
        "reference_normals": normals,
        "triangle_index": triangle_id,
        "barycentric_coordinates": barycentric,
        "nearest_distance_mm": distances,
    }


def _project_to_points(
    query: np.ndarray,
    reference: np.ndarray,
    reference_normals: np.ndarray,
) -> dict[str, np.ndarray]:
    distances, indices = cKDTree(reference).query(query, k=1)
    indices = indices.astype(np.int64)
    return {
        "projected_points_mm": reference[indices],
        "reference_normals": reference_normals[indices],
        "reference_point_index": indices,
        "nearest_distance_mm": distances.astype(np.float64),
    }


def _estimate_query_normals(query: np.ndarray, mapped_normals: np.ndarray, neighbours: int = 12) -> np.ndarray:
    if len(query) < 3:
        return mapped_normals.copy()
    k = min(max(3, neighbours), len(query))
    _, indices = cKDTree(query).query(query, k=k)
    estimated = np.empty_like(query)
    for row, local_indices in enumerate(np.atleast_2d(indices)):
        local = query[np.asarray(local_indices, dtype=np.int64)]
        centered = local - local.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        normal = vt[-1]
        if np.dot(normal, mapped_normals[row]) < 0.0:
            normal = -normal
        estimated[row] = normal / max(np.linalg.norm(normal), 1.0e-12)
    return estimated


def surface_difference(
    registered_worn_points: Any,
    reference_points: Any,
    *,
    reference_triangles: Any | None = None,
    reference_normals: Any | None = None,
    worn_normals: Any | None = None,
    maximum_projection_distance_mm: float = 2.0,
    maximum_tangent_residual_mm: float | None = None,
    maximum_normal_angle_degrees: float = 45.0,
    estimate_worn_normals: bool = True,
) -> dict[str, Any]:
    """Map registered worn samples to reference geometry and decompose wear."""
    worn = _points(registered_worn_points, "registered worn points", minimum=1)
    reference = _points(reference_points, "reference points", minimum=3)
    if not np.isfinite(maximum_projection_distance_mm) or maximum_projection_distance_mm <= 0.0:
        raise ValidationError("maximum_projection_distance_mm must be finite and positive")
    tangent_limit = maximum_projection_distance_mm if maximum_tangent_residual_mm is None else float(maximum_tangent_residual_mm)
    if not np.isfinite(tangent_limit) or tangent_limit <= 0.0:
        raise ValidationError("maximum_tangent_residual_mm must be finite and positive")
    if not 0.0 <= maximum_normal_angle_degrees <= 180.0:
        raise ValidationError("maximum_normal_angle_degrees must be in [0, 180]")

    reference_normal_array = None
    if reference_normals is not None:
        reference_normal_array = _unit_normals(reference_normals, "reference normals", len(reference))
    if reference_triangles is not None:
        triangles = _triangle_indices(reference_triangles, len(reference))
        projection = _project_to_triangles(worn, reference, triangles, reference_normal_array)
        mapping = "nearest_reference_triangle"
    else:
        if reference_normal_array is None:
            raise ValidationError("point-reference mapping requires oriented reference_normals")
        projection = _project_to_points(worn, reference, reference_normal_array)
        mapping = "nearest_reference_point"

    mapped_normals = projection["reference_normals"]
    if worn_normals is not None:
        worn_normal_array = _unit_normals(worn_normals, "worn normals", len(worn))
        normal_source = "provided"
    elif estimate_worn_normals:
        worn_normal_array = _estimate_query_normals(worn, mapped_normals)
        normal_source = "local_pca_aligned_to_reference"
    else:
        worn_normal_array = mapped_normals.copy()
        normal_source = "not_measured_assumed_reference"
    cosine = np.einsum("ij,ij->i", worn_normal_array, mapped_normals)
    normal_angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    displacement = projection["projected_points_mm"] - worn
    wear_depth = np.einsum("ij,ij->i", displacement, mapped_normals)
    tangent_vector = displacement - wear_depth[:, None] * mapped_normals
    tangent_residual = np.linalg.norm(tangent_vector, axis=1)
    nearest_distance = projection["nearest_distance_mm"]
    distance_bad = nearest_distance > maximum_projection_distance_mm
    tangent_bad = tangent_residual > tangent_limit
    normal_bad = normal_angle > maximum_normal_angle_degrees
    missing_mask = distance_bad | tangent_bad | normal_bad

    distance_score = np.exp(-np.square(nearest_distance / maximum_projection_distance_mm))
    tangent_score = np.exp(-np.square(tangent_residual / tangent_limit))
    angle_scale = max(maximum_normal_angle_degrees, 1.0e-9)
    normal_score = np.exp(-np.square(normal_angle / angle_scale))
    confidence = np.clip(distance_score * tangent_score * normal_score, 0.0, 1.0)
    confidence[missing_mask] = 0.0

    result: dict[str, Any] = {
        "version": SURFACE_DIFFERENCE_VERSION,
        "mapping": mapping,
        "units": "mm",
        "n_worn": int(len(worn)),
        "n_reference_points": int(len(reference)),
        "wear_depth_mm": wear_depth,
        "tangent_residual_mm": tangent_residual,
        "nearest_distance_mm": nearest_distance,
        "normal_angle_degrees": normal_angle,
        "confidence": confidence,
        "missing_mask": missing_mask,
        "projected_points_mm": projection["projected_points_mm"],
        "mapped_reference_normals": mapped_normals,
        "normal_source": normal_source,
        "quality": {
            "coverage": float(np.mean(~missing_mask)),
            "missing_fraction": float(np.mean(missing_mask)),
            "median_nearest_distance_mm": float(np.median(nearest_distance)),
            "p95_nearest_distance_mm": float(np.percentile(nearest_distance, 95.0)),
            "median_tangent_residual_mm": float(np.median(tangent_residual)),
            "p95_tangent_residual_mm": float(np.percentile(tangent_residual, 95.0)),
            "median_normal_angle_degrees": float(np.median(normal_angle)),
            "p95_normal_angle_degrees": float(np.percentile(normal_angle, 95.0)),
            "median_confidence": float(np.median(confidence)),
        },
        "thresholds": {
            "maximum_projection_distance_mm": float(maximum_projection_distance_mm),
            "maximum_tangent_residual_mm": float(tangent_limit),
            "maximum_normal_angle_degrees": float(maximum_normal_angle_degrees),
        },
    }
    for key in ("reference_point_index", "triangle_index", "barycentric_coordinates"):
        if key in projection:
            result[key] = projection[key]
    result["result_sha256"] = canonical_sha256(_hashable({key: value for key, value in result.items() if key != "result_sha256"}))
    return result


def _save_npz(result: dict[str, Any], path: Path) -> None:
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    metadata = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
    arrays["metadata_json"] = np.asarray(json.dumps(jsonable(metadata), ensure_ascii=True, sort_keys=True))
    np.savez_compressed(path, **arrays)


def selftest() -> bool:
    rng = np.random.default_rng(30031)
    side = 18
    x, y = np.meshgrid(np.linspace(-2.0, 2.0, side), np.linspace(-1.5, 1.5, side))
    reference = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    triangles = []
    for row in range(side - 1):
        for column in range(side - 1):
            a = row * side + column
            triangles.extend(((a, a + 1, a + side + 1), (a, a + side + 1, a + side)))
    triangles = np.asarray(triangles, dtype=np.int64)
    keep = rng.choice(len(reference), size=190, replace=False)
    depth = 0.18 + 0.04 * np.cos(reference[keep, 0])
    worn = reference[keep].copy()
    worn[:, :2] += rng.normal(0.0, 0.003, (len(worn), 2))
    worn[:, 2] -= depth + rng.normal(0.0, 0.002, len(worn))
    permutation = rng.permutation(len(worn))
    worn = worn[permutation]
    worn_normals = np.tile([0.0, 0.0, 1.0], (len(worn), 1))
    worn = np.vstack((worn, [9.0, 9.0, 9.0]))
    worn_normals = np.vstack((worn_normals, [0.0, 0.0, 1.0]))
    result = surface_difference(
        worn, reference, reference_triangles=triangles, worn_normals=worn_normals,
        maximum_projection_distance_mm=0.5,
        maximum_tangent_residual_mm=0.1,
    )
    valid = ~result["missing_mask"]
    assert result["mapping"] == "nearest_reference_triangle"
    assert abs(float(np.median(result["wear_depth_mm"][valid])) - float(np.median(depth))) < 0.02
    assert float(np.percentile(result["tangent_residual_mm"][valid], 95.0)) < 0.02
    assert result["missing_mask"][-1]
    assert np.allclose(result["barycentric_coordinates"].sum(axis=1), 1.0)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step30 registered surface difference")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--worn", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--triangles", type=Path)
    parser.add_argument("--reference-normals", type=Path)
    parser.add_argument("--worn-normals", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--maximum-distance-mm", type=float, default=2.0)
    args = parser.parse_args()
    if args.selftest:
        selftest()
        print("step30_surface_difference selftest: PASS")
    elif args.worn is not None and args.reference is not None:
        result = surface_difference(
            np.load(args.worn, allow_pickle=False),
            np.load(args.reference, allow_pickle=False),
            reference_triangles=None if args.triangles is None else np.load(args.triangles, allow_pickle=False),
            reference_normals=None if args.reference_normals is None else np.load(args.reference_normals, allow_pickle=False),
            worn_normals=None if args.worn_normals is None else np.load(args.worn_normals, allow_pickle=False),
            maximum_projection_distance_mm=args.maximum_distance_mm,
        )
        if args.output is None:
            print(json.dumps(jsonable(result["quality"]), indent=2, ensure_ascii=True))
        else:
            _save_npz(result, args.output)
    else:
        parser.print_help()
