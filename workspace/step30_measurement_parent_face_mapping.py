"""Strict mapping of measured samples to topology-v3 parent faces.

The mapper consumes the validated native topology-v3 NPZ artifact. It projects
samples onto collision triangles and aggregates measurements by their explicit
``triangle_parent_face_id``. Parent face IDs are never inferred from array
position and no identity fallback is provided when topology metadata is
missing or invalid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from step30_surface_difference import _closest_point_triangle, _mesh_geometry
from step30_validation_common import ValidationError, canonical_sha256, finite_array, jsonable


PARENT_FACE_MAPPING_VERSION = "step30-measurement-parent-face-mapping-v1"
TOPOLOGY_V3_VERSION = "step30-native-openfoam-topology-v3"
TOPOLOGY_V3_SIGNATURE_ALGORITHM = "sha256-vtk-triangulation-source-units-points-triangles-parent-ids-v3"


def _text_scalar(data: Any, key: str) -> str:
    if key not in data.files:
        raise ValidationError(f"topology v3 artifact missing field: {key}")
    value = np.asarray(data[key])
    if value.ndim != 0:
        raise ValidationError(f"topology field {key} must be a scalar")
    return str(value.item())


def _hex_hash(value: str, name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value):
        raise ValidationError(f"{name} must be a 64-character SHA-256 hex digest")
    return value.lower()


def _mesh_signature(points: np.ndarray, triangles: np.ndarray, parent_ids: np.ndarray, source_units: str) -> str:
    digest = hashlib.sha256()
    digest.update(source_units.encode("ascii"))
    for value in (points, triangles, parent_ids):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _hashable(value: Any) -> Any:
    """Represent arrays by finite-independent byte digests for result hashing."""
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "dtype": str(contiguous.dtype),
            "shape": list(contiguous.shape),
            "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(key): _hashable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_hashable(item) for item in value]
    return value


def _validate_normals(value: Any, name: str, count: int) -> np.ndarray:
    array = finite_array(value, name, ndim=2, shape=(count, 3))
    norm = np.linalg.norm(array, axis=1)
    if np.any(norm <= 1.0e-12):
        raise ValidationError(f"{name} contains zero-length normals")
    return np.asarray(array / norm[:, None], dtype=np.float64)


def load_topology_v3(
    topology_path: str | Path,
    *,
    expected_mesh_signature: str | None = None,
    expected_source_hash: str | None = None,
) -> dict[str, Any]:
    """Load and strictly validate a topology-v3 NPZ artifact."""
    path = Path(topology_path)
    if not path.is_file():
        raise ValidationError(f"topology artifact does not exist: {path}")
    try:
        data = np.load(str(path), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValidationError(f"cannot load topology artifact: {path}") from exc
    required = {
        "points_local_mm", "triangles_local_mm", "triangle_parent_face_id",
        "face_centers_local_mm", "face_normals_local", "face_area_m2",
        "triangle_centers_local_mm", "triangle_normals_local", "triangle_radii_mm",
        "face_id", "parent_face_id", "mesh_signature", "mesh_signature_algorithm",
        "source_vtp_sha256", "artifact_version", "source_coordinate_units",
        "internal_coordinate_units", "coordinate_scale_to_mm",
    }
    missing = sorted(required - set(data.files))
    if missing:
        raise ValidationError(f"topology v3 artifact missing fields: {missing}")
    artifact_version = _text_scalar(data, "artifact_version")
    if artifact_version != TOPOLOGY_V3_VERSION:
        raise ValidationError(f"unsupported topology artifact version: {artifact_version}")
    algorithm = _text_scalar(data, "mesh_signature_algorithm")
    if algorithm != TOPOLOGY_V3_SIGNATURE_ALGORITHM:
        raise ValidationError("topology v3 mesh signature algorithm is invalid")
    source_units = _text_scalar(data, "source_coordinate_units")
    internal_units = _text_scalar(data, "internal_coordinate_units")
    scale = float(np.asarray(data["coordinate_scale_to_mm"]).item())
    if source_units not in {"mm", "m"} or internal_units != "mm" or scale != ({"mm": 1.0, "m": 1000.0}[source_units]):
        raise ValidationError("topology v3 coordinate unit contract is invalid")
    points = finite_array(data["points_local_mm"], "topology points", ndim=2)
    if points.shape[1] != 3 or len(points) == 0:
        raise ValidationError("topology points must have shape (n, 3)")
    triangles_raw = np.asarray(data["triangles_local_mm"])
    parent_raw = np.asarray(data["triangle_parent_face_id"])
    if triangles_raw.ndim != 2 or triangles_raw.shape[1] != 3 or len(triangles_raw) == 0:
        raise ValidationError("topology triangles must have shape (m, 3)")
    if not np.issubdtype(triangles_raw.dtype, np.integer) or not np.issubdtype(parent_raw.dtype, np.integer):
        raise ValidationError("topology indices must be integer arrays")
    triangles = triangles_raw.astype(np.int64)
    parent_ids = parent_raw.astype(np.int64)
    if len(parent_ids) != len(triangles):
        raise ValidationError("triangle parent IDs do not match triangle count")
    if np.any(triangles < 0) or np.any(triangles >= len(points)):
        raise ValidationError("topology triangle vertex index out of range")
    face_centers = finite_array(data["face_centers_local_mm"], "face centers", ndim=2)
    face_normals = finite_array(data["face_normals_local"], "face normals", ndim=2)
    face_areas = finite_array(data["face_area_m2"], "face areas", ndim=1)
    n_faces = len(face_areas)
    if face_centers.shape != (n_faces, 3) or face_normals.shape != (n_faces, 3):
        raise ValidationError("topology face arrays have inconsistent shapes")
    if n_faces == 0 or np.any(parent_ids < 0) or np.any(parent_ids >= n_faces):
        raise ValidationError("topology parent face ID is out of range")
    if not np.isfinite(face_areas).all() or np.any(face_areas <= 0.0):
        raise ValidationError("topology face areas must be finite and positive")
    normal_lengths = np.linalg.norm(face_normals, axis=1)
    if np.any(np.abs(normal_lengths - 1.0) > 1.0e-4):
        raise ValidationError("topology face normals are not unit vectors")
    face_ids = np.asarray(data["face_id"])
    stored_parent_ids = np.asarray(data["parent_face_id"])
    expected_faces = np.arange(n_faces, dtype=np.int64)
    if not np.array_equal(face_ids.astype(np.int64), expected_faces) or not np.array_equal(stored_parent_ids.astype(np.int64), expected_faces):
        raise ValidationError("topology v3 parent face IDs are not canonical and unique")
    if np.any(np.bincount(parent_ids, minlength=n_faces) == 0):
        raise ValidationError("topology v3 contains a parent face with no collision triangle")
    xyz, triangle_normals, triangle_centers, triangle_radii = _mesh_geometry(points, triangles)
    triangle_area_vectors = 0.5 * np.cross(
        xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]
    )
    tri_area = np.linalg.norm(triangle_area_vectors, axis=1)
    if np.any(~np.isfinite(tri_area)) or np.any(tri_area <= 0.0):
        raise ValidationError("topology collision triangles have invalid areas")
    stored_triangle_normals = finite_array(data["triangle_normals_local"], "triangle normals", ndim=2)
    stored_triangle_centers = finite_array(data["triangle_centers_local_mm"], "triangle centers", ndim=2)
    stored_triangle_radii = finite_array(data["triangle_radii_mm"], "triangle radii", ndim=1)
    if stored_triangle_normals.shape != triangle_normals.shape or stored_triangle_centers.shape != triangle_centers.shape or stored_triangle_radii.shape != triangle_radii.shape:
        raise ValidationError("topology triangle metadata has inconsistent shapes")
    if not np.allclose(stored_triangle_normals, triangle_normals, rtol=1.0e-5, atol=1.0e-6):
        raise ValidationError("topology triangle normals differ from vertices")
    if not np.allclose(stored_triangle_centers, triangle_centers, rtol=1.0e-6, atol=2.0e-5):
        raise ValidationError("topology triangle centers differ from vertices")
    if not np.allclose(stored_triangle_radii, triangle_radii, rtol=1.0e-5, atol=2.0e-5):
        raise ValidationError("topology triangle radii differ from vertices")
    triangle_keys = np.sort(triangles, axis=1)
    if len(np.unique(triangle_keys, axis=0)) != len(triangles):
        raise ValidationError("topology v3 contains duplicate collision triangles")
    orientation = np.einsum("ij,ij->i", triangle_normals, face_normals[parent_ids])
    if np.any(orientation <= 0.0):
        raise ValidationError("topology collision triangle normal disagrees with parent face")
    parent_area_vectors = np.zeros((n_faces, 3), dtype=np.float64)
    np.add.at(parent_area_vectors, parent_ids, triangle_area_vectors)
    parent_area_mm2 = np.linalg.norm(parent_area_vectors, axis=1)
    computed_face_areas = parent_area_mm2 * 1.0e-6
    computed_face_normals = parent_area_vectors / np.maximum(parent_area_mm2[:, None], 1.0e-30)
    scalar_area = np.bincount(parent_ids, weights=tri_area, minlength=n_faces)
    computed_face_centers = np.zeros((n_faces, 3), dtype=np.float64)
    np.add.at(computed_face_centers, parent_ids, triangle_centers * tri_area[:, None])
    computed_face_centers /= np.maximum(scalar_area[:, None], 1.0e-30)
    if not np.allclose(face_areas, computed_face_areas, rtol=2.0e-5, atol=1.0e-12):
        raise ValidationError("topology face areas differ from collision geometry")
    if not np.allclose(face_normals, computed_face_normals, rtol=1.0e-5, atol=2.0e-6):
        raise ValidationError("topology face normals differ from collision geometry")
    if not np.allclose(face_centers, computed_face_centers, rtol=1.0e-6, atol=2.0e-5):
        raise ValidationError("topology face centers differ from collision geometry")
    stored_signature = _hex_hash(_text_scalar(data, "mesh_signature"), "mesh_signature")
    computed_signature = _mesh_signature(np.asarray(data["points_local_mm"]), triangles_raw, parent_raw, source_units)
    if stored_signature != computed_signature:
        raise ValidationError("topology v3 mesh signature differs from stored arrays")
    source_hash = _hex_hash(_text_scalar(data, "source_vtp_sha256"), "source_vtp_sha256")
    if expected_mesh_signature is not None and stored_signature != _hex_hash(expected_mesh_signature, "expected_mesh_signature"):
        raise ValidationError("mesh signature mismatch")
    if expected_source_hash is not None and source_hash != _hex_hash(expected_source_hash, "expected_source_hash"):
        raise ValidationError("source hash mismatch")
    return {
        "path": str(path),
        "artifact_version": artifact_version,
        "mesh_signature_algorithm": algorithm,
        "mesh_signature": stored_signature,
        "source_vtp_sha256": source_hash,
        "source_coordinate_units": source_units,
        "internal_coordinate_units": internal_units,
        "coordinate_scale_to_mm": scale,
        "points_mm": np.asarray(points, dtype=np.float64),
        "triangles": triangles,
        "triangle_parent_face_id": parent_ids,
        "face_centers_mm": np.asarray(face_centers, dtype=np.float64),
        "face_normals": np.asarray(face_normals, dtype=np.float64),
        "face_area_m2": np.asarray(face_areas, dtype=np.float64),
        "triangle_normals": triangle_normals,
        "triangle_centers_mm": triangle_centers,
        "triangle_radii_mm": triangle_radii,
        "n_parent_faces": int(n_faces),
    }


def _map_samples(
    points: np.ndarray,
    topology: dict[str, Any],
    sample_normals: np.ndarray | None,
    maximum_projection_distance_mm: float,
    maximum_normal_angle_degrees: float,
) -> dict[str, np.ndarray]:
    reference = topology["points_mm"]
    triangles = topology["triangles"]
    parent_ids = topology["triangle_parent_face_id"]
    xyz = reference[triangles]
    centers = topology["triangle_centers_mm"]
    radii = topology["triangle_radii_mm"]
    tree = cKDTree(centers)
    projected = np.empty_like(points)
    triangle_id = np.empty(len(points), dtype=np.int64)
    barycentric = np.empty((len(points), 3), dtype=np.float64)
    distances = np.empty(len(points), dtype=np.float64)
    mapped_normals = np.empty_like(points)
    for row, point in enumerate(points):
        _, first = tree.query(point, k=1)
        first = int(first)
        first_point, first_bary = _closest_point_triangle(point, xyz[first])
        best_distance = float(np.linalg.norm(point - first_point))
        candidates = tree.query_ball_point(point, r=best_distance + float(np.max(radii)))
        if first not in candidates:
            candidates.append(first)
        best_id, best_point, best_bary = first, first_point, first_bary
        for candidate in candidates:
            candidate_point, candidate_bary = _closest_point_triangle(point, xyz[candidate])
            candidate_distance = float(np.linalg.norm(point - candidate_point))
            if candidate_distance < best_distance:
                best_id, best_point, best_bary, best_distance = int(candidate), candidate_point, candidate_bary, candidate_distance
        projected[row] = best_point
        triangle_id[row] = best_id
        barycentric[row] = best_bary
        distances[row] = best_distance
        mapped_normals[row] = topology["triangle_normals"][best_id]
    normal_angles = np.full(len(points), np.nan, dtype=np.float64)
    normal_bad = np.zeros(len(points), dtype=bool)
    if sample_normals is not None:
        cosine = np.einsum("ij,ij->i", sample_normals, mapped_normals)
        normal_angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        normal_bad = normal_angles > maximum_normal_angle_degrees
    missing = (distances > maximum_projection_distance_mm) | normal_bad
    return {
        "projected_points_mm": projected,
        "triangle_id": triangle_id,
        "parent_face_id": parent_ids[triangle_id],
        "barycentric_coordinates": barycentric,
        "projection_distance_mm": distances,
        "projection_normal_angle_degrees": normal_angles,
        "normal_bad_mask": normal_bad,
        "missing_mask": missing,
    }


def map_measurements_to_parent_faces(
    topology_path: str | Path,
    sample_points: Any,
    *,
    sample_normals: Any | None = None,
    sample_values: Any | None = None,
    expected_mesh_signature: str | None = None,
    expected_source_hash: str | None = None,
    artifact_set_id: str | None = None,
    transform_sha256: str | None = None,
    coordinate_frame: str = "topology_local_mm",
    require_artifact_identity: bool = False,
    maximum_projection_distance_mm: float = 2.0,
    maximum_normal_angle_degrees: float = 45.0,
) -> dict[str, Any]:
    """Project samples and return point-level plus parent-face-level arrays."""
    topology = load_topology_v3(
        topology_path,
        expected_mesh_signature=expected_mesh_signature,
        expected_source_hash=expected_source_hash,
    )
    if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
        raise ValidationError("coordinate_frame must be a non-empty string")
    for value, name in ((artifact_set_id, "artifact_set_id"), (transform_sha256, "transform_sha256")):
        if value is not None:
            _hex_hash(str(value), name)
    if require_artifact_identity and (artifact_set_id is None or transform_sha256 is None):
        raise ValidationError("measured mapping requires artifact_set_id and transform_sha256")
    points = finite_array(sample_points, "sample points", ndim=2)
    if points.shape[1] != 3 or len(points) == 0:
        raise ValidationError("sample points must have nonempty shape (n, 3)")
    if not np.isfinite(maximum_projection_distance_mm) or maximum_projection_distance_mm <= 0.0:
        raise ValidationError("maximum_projection_distance_mm must be finite and positive")
    if not 0.0 <= maximum_normal_angle_degrees <= 180.0:
        raise ValidationError("maximum_normal_angle_degrees must be in [0, 180]")
    normal_array = None if sample_normals is None else _validate_normals(sample_normals, "sample normals", len(points))
    value_array = None if sample_values is None else finite_array(sample_values, "sample values", ndim=1, shape=(len(points),))
    mapping = _map_samples(
        np.asarray(points, dtype=np.float64), topology, normal_array,
        maximum_projection_distance_mm, maximum_normal_angle_degrees,
    )
    n_faces = topology["n_parent_faces"]
    valid = ~mapping["missing_mask"]
    face_sample_count = np.bincount(mapping["parent_face_id"], minlength=n_faces).astype(np.int64)
    face_valid_count = np.bincount(mapping["parent_face_id"][valid], minlength=n_faces).astype(np.int64)
    face_missing_count = face_sample_count - face_valid_count
    face_distance_mean = np.full(n_faces, np.nan, dtype=np.float64)
    face_distance_p95 = np.full(n_faces, np.nan, dtype=np.float64)
    face_angle_mean = np.full(n_faces, np.nan, dtype=np.float64)
    face_value_mean = None if value_array is None else np.full(n_faces, np.nan, dtype=np.float64)
    face_quality = np.zeros(n_faces, dtype=np.float64)
    for face_id in range(n_faces):
        rows = np.flatnonzero((mapping["parent_face_id"] == face_id) & valid)
        all_rows = np.flatnonzero(mapping["parent_face_id"] == face_id)
        if len(rows):
            face_distance_mean[face_id] = float(np.mean(mapping["projection_distance_mm"][rows]))
            face_distance_p95[face_id] = float(np.percentile(mapping["projection_distance_mm"][rows], 95.0))
            if np.isfinite(mapping["projection_normal_angle_degrees"][rows]).all():
                face_angle_mean[face_id] = float(np.mean(mapping["projection_normal_angle_degrees"][rows]))
            if face_value_mean is not None:
                face_value_mean[face_id] = float(np.mean(value_array[rows]))
            face_quality[face_id] = float(len(rows) / max(len(all_rows), 1))
    quality = {
        "projection_coverage": float(np.mean(valid)),
        "missing_fraction": float(np.mean(mapping["missing_mask"])),
        "normal_available": bool(normal_array is not None),
        "normal_coverage": float(np.mean(~mapping["normal_bad_mask"])) if normal_array is not None else None,
        "projection_distance_p95_mm": float(np.percentile(mapping["projection_distance_mm"], 95.0)),
        "projection_distance_max_mm": float(np.max(mapping["projection_distance_mm"])),
        "parent_faces_with_valid_samples": int(np.sum(face_valid_count > 0)),
        "parent_faces_without_valid_samples": int(np.sum(face_valid_count == 0)),
        "n_samples": int(len(points)),
        "n_parent_faces": int(n_faces),
    }
    result: dict[str, Any] = {
        "version": PARENT_FACE_MAPPING_VERSION,
        "artifact_version": topology["artifact_version"],
        "mesh_signature": topology["mesh_signature"],
        "source_vtp_sha256": topology["source_vtp_sha256"],
        "topology_path": topology["path"],
        "artifact_set_id": None if artifact_set_id is None else str(artifact_set_id).lower(),
        "transform_sha256": None if transform_sha256 is None else str(transform_sha256).lower(),
        "coordinate_frame": coordinate_frame.strip(),
        "artifact_identity_required": bool(require_artifact_identity),
        "units": "mm",
        "point_parent_face_id": mapping["parent_face_id"],
        "point_triangle_id": mapping["triangle_id"],
        "point_projected_mm": mapping["projected_points_mm"],
        "point_barycentric": mapping["barycentric_coordinates"],
        "point_projection_distance_mm": mapping["projection_distance_mm"],
        "point_projection_normal_angle_degrees": mapping["projection_normal_angle_degrees"],
        "point_missing_mask": mapping["missing_mask"],
        "point_normal_bad_mask": mapping["normal_bad_mask"],
        "face_id": np.arange(n_faces, dtype=np.int64),
        "face_sample_count": face_sample_count,
        "face_valid_count": face_valid_count,
        "face_missing_count": face_missing_count,
        "face_projection_distance_mean_mm": face_distance_mean,
        "face_projection_distance_p95_mm": face_distance_p95,
        "face_projection_normal_angle_mean_degrees": face_angle_mean,
        "face_quality": face_quality,
        "quality_report": quality,
        "no_identity_fallback": True,
    }
    if face_value_mean is not None:
        result["face_value_mean"] = face_value_mean
    result["result_sha256"] = canonical_sha256(_hashable({key: value for key, value in result.items() if key != "result_sha256"}))
    return result


def _save_npz(result: dict[str, Any], path: Path) -> None:
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    metadata = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
    arrays["metadata_json"] = np.asarray(json.dumps(jsonable(metadata), ensure_ascii=True, sort_keys=True))
    np.savez_compressed(path, **arrays)


def _synthetic_topology(path: Path) -> tuple[str, str]:
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    parents = np.array([0, 1], dtype=np.int32)
    centers = points[triangles].mean(axis=1)
    vectors = 0.5 * np.cross(points[triangles][:, 1] - points[triangles][:, 0], points[triangles][:, 2] - points[triangles][:, 0])
    tri_area = np.linalg.norm(vectors, axis=1)
    normals = vectors / tri_area[:, None]
    face_area_m2 = np.array([2.0e-6, 2.0e-6])
    face_centers = np.array([[4.0 / 3.0, 2.0 / 3.0, 0.0], [2.0 / 3.0, 4.0 / 3.0, 0.0]])
    face_normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    radii = np.linalg.norm(points[triangles] - centers[:, None, :], axis=2).max(axis=1)
    signature = _mesh_signature(points, triangles, parents, "mm")
    source_hash = "a" * 64
    np.savez_compressed(
        path,
        points_local_mm=points,
        triangles_local_mm=triangles,
        triangle_parent_face_id=parents,
        face_centers_local_mm=face_centers,
        face_normals_local=face_normals,
        face_area_m2=face_area_m2,
        triangle_centers_local_mm=centers,
        triangle_normals_local=normals,
        triangle_radii_mm=radii,
        face_id=np.arange(2, dtype=np.int32),
        parent_face_id=np.arange(2, dtype=np.int32),
        mesh_signature=np.array(signature),
        mesh_signature_algorithm=np.array(TOPOLOGY_V3_SIGNATURE_ALGORITHM),
        source_vtp_sha256=np.array(source_hash),
        artifact_version=np.array(TOPOLOGY_V3_VERSION),
        source_coordinate_units=np.array("mm"),
        internal_coordinate_units=np.array("mm"),
        coordinate_scale_to_mm=np.array(1.0),
    )
    return signature, source_hash


def selftest() -> bool:
    with tempfile.TemporaryDirectory() as directory:
        artifact = Path(directory) / "topology_v3.npz"
        signature, source_hash = _synthetic_topology(artifact)
        points = np.array([[1.6, 0.4, 0.2], [0.4, 1.6, 0.2], [9.0, 9.0, 0.0]])
        normals = np.tile([0.0, 0.0, 1.0], (3, 1))
        result = map_measurements_to_parent_faces(
            artifact, points, sample_normals=normals,
            sample_values=np.array([1.0, 2.0, 3.0]),
            expected_mesh_signature=signature, expected_source_hash=source_hash,
            maximum_projection_distance_mm=0.5,
        )
        assert result["no_identity_fallback"] is True
        assert result["artifact_set_id"] is None and result["transform_sha256"] is None
        assert np.array_equal(result["point_parent_face_id"][:2], [0, 1])
        assert result["point_missing_mask"][-1]
        assert result["face_valid_count"].tolist() == [1, 1]
        with np.load(artifact, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["parent_face_id"] = np.array([1, 0], dtype=np.int32)
        np.savez_compressed(artifact, **arrays)
        try:
            load_topology_v3(artifact)
        except ValidationError:
            pass
        else:
            raise AssertionError("corrupt parent IDs were accepted")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step30 strict measurement-to-parent-face mapper")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--topology", type=Path)
    parser.add_argument("--points", type=Path)
    parser.add_argument("--normals", type=Path)
    parser.add_argument("--values", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-mesh-signature")
    parser.add_argument("--expected-source-hash")
    parser.add_argument("--artifact-set-id")
    parser.add_argument("--transform-sha256")
    parser.add_argument("--coordinate-frame", default="topology_local_mm")
    parser.add_argument("--require-artifact-identity", action="store_true")
    parser.add_argument("--maximum-distance-mm", type=float, default=2.0)
    parser.add_argument("--maximum-normal-angle-degrees", type=float, default=45.0)
    args = parser.parse_args()
    if args.selftest:
        selftest()
        print("step30_measurement_parent_face_mapping selftest: PASS")
    elif args.topology is not None and args.points is not None:
        result = map_measurements_to_parent_faces(
            args.topology,
            np.load(args.points, allow_pickle=False),
            sample_normals=None if args.normals is None else np.load(args.normals, allow_pickle=False),
            sample_values=None if args.values is None else np.load(args.values, allow_pickle=False),
            expected_mesh_signature=args.expected_mesh_signature,
            expected_source_hash=args.expected_source_hash,
            artifact_set_id=args.artifact_set_id,
            transform_sha256=args.transform_sha256,
            coordinate_frame=args.coordinate_frame,
            require_artifact_identity=args.require_artifact_identity,
            maximum_projection_distance_mm=args.maximum_distance_mm,
            maximum_normal_angle_degrees=args.maximum_normal_angle_degrees,
        )
        if args.output is None:
            print(json.dumps(jsonable(result["quality_report"]), indent=2, ensure_ascii=True))
        else:
            _save_npz(result, args.output)
    else:
        parser.print_help()
