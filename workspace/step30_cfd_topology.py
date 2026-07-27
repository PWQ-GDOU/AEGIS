"""
Native OpenFOAM bucket-patch topology for Step 30.

The VTP is produced by foamToVTK from the same OpenFOAM mesh/time as the
cell-centre velocity fields.  VTP polygon order is retained as a stable
patch-local parent face ID; triangles are only a collision representation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

TOPOLOGY_NPZ = Path(__file__).parent / "output" / "bucket_topology.npz"
TOPOLOGY_JSON = Path(__file__).parent / "output" / "bucket_topology_summary.json"


def _mesh_signature(points: np.ndarray, triangles: np.ndarray, parent_ids: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in (points, triangles, parent_ids):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _triangle_geometry(points: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = points[triangles]
    cross = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    valid = area > 1e-12
    if not np.all(valid):
        triangles = triangles[valid]
        cross = cross[valid]
        area = area[valid]
        xyz = xyz[valid]
    normals = cross / np.maximum(2.0 * area[:, None], 1e-30)
    centers = xyz.mean(axis=1)
    return triangles, normals, centers, area


def build_topology_from_vtp(
    vtp_path: Path,
    *,
    patch_name: str = "bucket",
    time_name: str = "1000",
    units: str = "mm",
    output_npz: Path = TOPOLOGY_NPZ,
    output_json: Path = TOPOLOGY_JSON,
) -> dict:
    """Read a native OpenFOAM VTP patch and save validated topology."""
    try:
        import vtk
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("vtk is required to parse OpenFOAM VTP topology") from exc

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetNumberOfPolys() == 0:
        raise ValueError(f"VTP contains no polygon patch: {vtp_path}")

    scale = {"mm": 1.0, "m": 1000.0}.get(units)
    if scale is None:
        raise ValueError("units must be 'mm' or 'm'")
    points = np.array([poly.GetPoint(i) for i in range(poly.GetNumberOfPoints())], dtype=np.float64) * scale

    parent_polygons: list[np.ndarray] = []
    id_list = vtk.vtkIdList()
    for cell_id in range(poly.GetNumberOfCells()):
        poly.GetCellPoints(cell_id, id_list)
        ids = np.array([id_list.GetId(j) for j in range(id_list.GetNumberOfIds())], dtype=np.int64)
        if len(ids) < 3:
            raise ValueError(f"polygon {cell_id} has fewer than 3 points")
        parent_polygons.append(ids)

    # Fan triangulation is valid for the planar convex OpenFOAM patch faces in
    # this case. Parent IDs remain the native patch-local polygon IDs.
    triangle_rows: list[list[int]] = []
    triangle_parent: list[int] = []
    for parent_id, polygon in enumerate(parent_polygons):
        for j in range(1, len(polygon) - 1):
            triangle_rows.append([int(polygon[0]), int(polygon[j]), int(polygon[j + 1])])
            triangle_parent.append(parent_id)
    triangles = np.asarray(triangle_rows, dtype=np.int64)
    parent_ids = np.asarray(triangle_parent, dtype=np.int64)
    triangles, triangle_normals, triangle_centers, triangle_area_mm2 = _triangle_geometry(points, triangles)
    # _triangle_geometry can remove degenerate triangles; mirror the parent ID
    # mask by recomputing from the original geometry when needed.
    if len(parent_ids) != len(triangles):
        raw_xyz = points[np.asarray(triangle_rows, dtype=np.int64)]
        raw_area = 0.5 * np.linalg.norm(np.cross(raw_xyz[:, 1] - raw_xyz[:, 0], raw_xyz[:, 2] - raw_xyz[:, 0]), axis=1)
        parent_ids = parent_ids[raw_area > 1e-12]

    parent_count = len(parent_polygons)
    parent_area_mm2 = np.bincount(parent_ids, weights=triangle_area_mm2, minlength=parent_count)
    parent_center = np.zeros((parent_count, 3), dtype=np.float64)
    np.add.at(parent_center, parent_ids, triangle_centers * triangle_area_mm2[:, None])
    parent_center /= np.maximum(parent_area_mm2[:, None], 1e-30)
    parent_normal_sum = np.zeros((parent_count, 3), dtype=np.float64)
    np.add.at(parent_normal_sum, parent_ids, triangle_normals * triangle_area_mm2[:, None])
    parent_normals = parent_normal_sum / np.maximum(np.linalg.norm(parent_normal_sum, axis=1, keepdims=True), 1e-30)

    signature = _mesh_signature(points, triangles, parent_ids)
    radii = np.linalg.norm(points[triangles] - triangle_centers[:, None, :], axis=2).max(axis=1)
    broadphase_radius = float(max(15.0, 3.0 * np.median(radii)))
    regular = radii <= broadphase_radius
    summary = {
        "version": "step30-native-openfoam-topology-v1",
        "source_vtp": str(vtp_path),
        "patch_name": patch_name,
        "time": str(time_name),
        "coordinate_units": units,
        "n_points": int(len(points)),
        "n_parent_faces": int(parent_count),
        "n_collision_triangles": int(len(triangles)),
        "surface_area_m2": float(parent_area_mm2.sum() * 1e-6),
        "bounds_mm": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        "mesh_signature": signature,
        "degenerate_triangles_removed": int(sum(len(p) - 2 for p in parent_polygons) - len(triangles)),
        "triangle_size_mm": {
            "median": float(np.median(radii)),
            "p95": float(np.percentile(radii, 95)),
            "max": float(radii.max()),
        },
    }
    validate_topology_arrays(
        points, triangles, parent_ids, parent_center, parent_normals, parent_area_mm2,
        expected_parent_count=parent_count,
    )
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_npz),
        points_local_mm=points.astype(np.float32),
        triangles_local_mm=triangles.astype(np.int32),
        triangle_parent_face_id=parent_ids.astype(np.int32),
        face_id=np.arange(parent_count, dtype=np.int32),
        parent_face_id=np.arange(parent_count, dtype=np.int32),
        face_centers_local_mm=parent_center.astype(np.float32),
        face_normals_local=parent_normals.astype(np.float32),
        face_area_m2=(parent_area_mm2 * 1e-6).astype(np.float32),
        triangle_centers_local_mm=triangle_centers.astype(np.float32),
        triangle_normals_local=triangle_normals.astype(np.float32),
        triangle_radii_mm=radii.astype(np.float32),
        mesh_signature=np.array(signature),
        patch_name=np.array(patch_name),
        time_name=np.array(str(time_name)),
        coordinate_units=np.array("mm"),
    )
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def validate_topology_arrays(
    points: np.ndarray,
    triangles: np.ndarray,
    triangle_parent_face_id: np.ndarray,
    face_centers: np.ndarray,
    face_normals: np.ndarray,
    face_area_m2: np.ndarray,
    *,
    expected_parent_count: int | None = None,
) -> None:
    """Fail closed on malformed or non-physical topology artifacts."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("topology points must have shape (N, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("topology collision triangles must have shape (M, 3)")
    if len(triangle_parent_face_id) != len(triangles):
        raise ValueError("triangle parent IDs do not match triangle count")
    if face_centers.shape != (len(face_area_m2), 3):
        raise ValueError("face centres and face areas have inconsistent shapes")
    if face_normals.shape != face_centers.shape:
        raise ValueError("face normals and face centres have inconsistent shapes")
    if not np.isfinite(points).all() or not np.isfinite(face_centers).all():
        raise ValueError("topology coordinates contain non-finite values")
    if not np.isfinite(face_normals).all() or not np.isfinite(face_area_m2).all():
        raise ValueError("topology normals/areas contain non-finite values")
    if np.any(face_area_m2 <= 0.0):
        raise ValueError("topology contains non-positive face areas")
    normal_norm = np.linalg.norm(face_normals, axis=1)
    if np.any(np.abs(normal_norm - 1.0) > 1e-4):
        raise ValueError("topology face normals are not unit vectors")
    if len(np.unique(np.arange(len(face_area_m2)))) != len(face_area_m2):
        raise ValueError("topology face IDs are not unique")
    if expected_parent_count is not None and len(face_area_m2) != expected_parent_count:
        raise ValueError("unexpected parent face count")
    if np.any(triangle_parent_face_id < 0) or np.any(triangle_parent_face_id >= len(face_area_m2)):
        raise ValueError("triangle parent ID out of range")


def load_topology_artifact(path: Path = TOPOLOGY_NPZ) -> dict:
    """Load a validated artifact and normalize it to Step-30 collision fields."""
    data = np.load(str(path), allow_pickle=False)
    required = {
        "points_local_mm", "triangles_local_mm", "triangle_parent_face_id",
        "face_centers_local_mm", "face_normals_local", "face_area_m2",
        "triangle_centers_local_mm", "triangle_normals_local", "triangle_radii_mm",
        "mesh_signature", "patch_name", "time_name",
    }
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"topology artifact missing fields: {missing}")
    points = data["points_local_mm"].astype(np.float64)
    triangles = data["triangles_local_mm"].astype(np.int64)
    parent_ids = data["triangle_parent_face_id"].astype(np.int64)
    centers = data["face_centers_local_mm"].astype(np.float64)
    normals = data["face_normals_local"].astype(np.float64)
    areas = data["face_area_m2"].astype(np.float64)
    validate_topology_arrays(points, triangles, parent_ids, centers, normals, areas)
    tri_normals = data["triangle_normals_local"].astype(np.float64)
    tri_centers = data["triangle_centers_local_mm"].astype(np.float64)
    radii = data["triangle_radii_mm"].astype(np.float64)
    if len(tri_normals) != len(triangles) or len(tri_centers) != len(triangles) or len(radii) != len(triangles):
        raise ValueError("topology triangle metadata length mismatch")
    regular = radii <= max(15.0, 3.0 * np.median(radii))
    signature = str(data["mesh_signature"].item())
    return {
        "backend": "cfd_topology",
        "face_id_namespace": "openfoam_bucket_patch",
        "mesh_signature": signature,
        "patch_name": str(data["patch_name"].item()),
        "time_name": str(data["time_name"].item()),
        "vertices_mm": points,
        "faces": triangles,
        "triangles_mm": points[triangles],
        "collision_triangles_mm": points[triangles],
        "face_centers_mm": centers,
        "face_normals": normals,
        "face_area_m2": areas,
        "face_area_mm2": areas * 1e6,
        "face_radii_mm": radii,
        "collision_face_centers_mm": tri_centers,
        "collision_face_normals": tri_normals,
        "collision_face_ids": parent_ids,
        "parent_face_ids": np.arange(len(areas), dtype=np.int64),
        "n_parent_faces": int(len(areas)),
        "collision_broadphase_radius_mm": float(max(15.0, 3.0 * np.median(radii))),
        "regular_face_indices": np.flatnonzero(regular),
        "large_face_indices": np.flatnonzero(~regular),
        "regular_face_tree": cKDTree(tri_centers[regular]),
        "large_face_tree": cKDTree(tri_centers[~regular]) if np.any(~regular) else None,
        "backend": "cfd_topology",
        "face_id_namespace": "openfoam_bucket_patch",
        "parent_face_id": np.arange(len(areas), dtype=np.int64),
        "bounds_mm": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        "surface_area_m2": float(areas.sum()),
    }
