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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mesh_signature(
    points: np.ndarray,
    triangles: np.ndarray,
    parent_ids: np.ndarray,
    source_units: str = "mm",
) -> str:
    digest = hashlib.sha256()
    digest.update(source_units.encode("ascii"))
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


def _newell_polygon_area_vector(points: np.ndarray) -> np.ndarray:
    shifted = np.roll(points, -1, axis=0)
    return 0.5 * np.sum(np.cross(points, shifted), axis=0)


def _newell_polygon_area(points: np.ndarray) -> float:
    return float(np.linalg.norm(_newell_polygon_area_vector(points)))


def _vtk_parent_triangulated_areas(poly, scale: float) -> np.ndarray:
    """Compute an independent per-cell area using each VTK cell triangulator."""
    try:
        import vtk
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("vtk is required to triangulate OpenFOAM patch faces") from exc
    areas = np.zeros(poly.GetNumberOfCells(), dtype=np.float64)
    triangle_ids = vtk.vtkIdList()
    triangle_points = vtk.vtkPoints()
    for cell_id in range(poly.GetNumberOfCells()):
        cell = poly.GetCell(cell_id)
        triangle_ids.Reset()
        triangle_points.Reset()
        if not cell.Triangulate(0, triangle_ids, triangle_points):
            raise ValueError(f"VTK failed to triangulate polygon {cell_id}")
        if triangle_points.GetNumberOfPoints() % 3:
            raise ValueError(f"VTK polygon {cell_id} produced incomplete triangles")
        for j in range(0, triangle_points.GetNumberOfPoints(), 3):
            xyz = np.asarray([triangle_points.GetPoint(j + k) for k in range(3)], dtype=np.float64) * scale
            areas[cell_id] += 0.5 * np.linalg.norm(np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0]))
    return areas


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

    parent_count = len(parent_polygons)
    parent_cell_ids = vtk.vtkIdTypeArray()
    parent_cell_ids.SetName("parentFaceId")
    parent_cell_ids.SetNumberOfValues(parent_count)
    for parent_id in range(parent_count):
        parent_cell_ids.SetValue(parent_id, parent_id)
    poly.GetCellData().AddArray(parent_cell_ids)

    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(poly)
    triangle_filter.PassVertsOff()
    triangle_filter.PassLinesOff()
    triangle_filter.Update()
    triangulated = triangle_filter.GetOutput()
    parent_array = triangulated.GetCellData().GetArray("parentFaceId")
    if parent_array is None:
        raise ValueError("VTK triangulation did not preserve parent face IDs")

    triangle_rows: list[list[int]] = []
    triangle_parent: list[int] = []
    for triangle_id in range(triangulated.GetNumberOfCells()):
        triangulated.GetCellPoints(triangle_id, id_list)
        ids = [id_list.GetId(j) for j in range(id_list.GetNumberOfIds())]
        if len(ids) != 3:
            raise ValueError("VTK triangle filter produced a non-triangle cell")
        triangle_rows.append(ids)
        triangle_parent.append(int(parent_array.GetTuple1(triangle_id)))
    triangles_raw = np.asarray(triangle_rows, dtype=np.int64)
    parent_ids_raw = np.asarray(triangle_parent, dtype=np.int64)
    raw_xyz = points[triangles_raw]
    raw_area_vectors = 0.5 * np.cross(
        raw_xyz[:, 1] - raw_xyz[:, 0], raw_xyz[:, 2] - raw_xyz[:, 0]
    )
    raw_area = np.linalg.norm(raw_area_vectors, axis=1)
    if np.any(raw_area <= 1e-12):
        raise ValueError("VTK triangulation produced degenerate collision triangles")
    triangles = triangles_raw
    parent_ids = parent_ids_raw
    triangles, triangle_normals, triangle_centers, triangle_area_mm2 = _triangle_geometry(points, triangles)
    triangle_keys = np.sort(triangles, axis=1)
    if len(np.unique(triangle_keys, axis=0)) != len(triangles):
        raise ValueError("VTK triangulation produced duplicate collision triangles")
    expected_triangle_count = np.asarray([len(polygon) - 2 for polygon in parent_polygons])
    actual_triangle_count = np.bincount(parent_ids, minlength=parent_count)
    if not np.array_equal(actual_triangle_count, expected_triangle_count):
        raise ValueError("VTK triangulation produced an incomplete parent face")

    source_area_vectors = np.asarray(
        [_newell_polygon_area_vector(points[polygon]) for polygon in parent_polygons], dtype=np.float64
    )
    parent_area_mm2 = np.linalg.norm(source_area_vectors, axis=1)
    if np.any(parent_area_mm2 <= 1e-12):
        raise ValueError("OpenFOAM patch contains a zero-area parent face")
    triangulated_area_vectors = np.zeros((parent_count, 3), dtype=np.float64)
    np.add.at(triangulated_area_vectors, parent_ids, raw_area_vectors)
    area_vector_error = np.linalg.norm(
        triangulated_area_vectors - source_area_vectors, axis=1
    ) / parent_area_mm2
    relative_area_error = np.abs(
        np.linalg.norm(triangulated_area_vectors, axis=1) - parent_area_mm2
    ) / parent_area_mm2
    total_area_relative_error = float(
        np.sum(np.linalg.norm(triangulated_area_vectors - source_area_vectors, axis=1))
        / parent_area_mm2.sum()
    )
    if total_area_relative_error > 1e-10 or float(area_vector_error.max()) > 1e-8:
        raise ValueError("VTK triangulation does not conserve OpenFOAM face area vectors")
    collision_parent_area_mm2 = np.bincount(
        parent_ids, weights=triangle_area_mm2, minlength=parent_count
    )
    polygon_warpage = (collision_parent_area_mm2 - parent_area_mm2) / parent_area_mm2
    if np.any(polygon_warpage < -1e-10):
        raise ValueError("collision triangle area is smaller than OpenFOAM parent face area")
    source_normals = source_area_vectors / parent_area_mm2[:, None]
    orientation_dot = np.einsum("ij,ij->i", triangle_normals, source_normals[parent_ids])
    inverted_triangles = int(np.sum(orientation_dot <= 0.0))
    if inverted_triangles:
        raise ValueError("VTK triangulation produced inverted triangles")
    parent_center = np.zeros((parent_count, 3), dtype=np.float64)
    np.add.at(parent_center, parent_ids, triangle_centers * triangle_area_mm2[:, None])
    parent_center /= collision_parent_area_mm2[:, None]
    parent_normal_sum = np.zeros((parent_count, 3), dtype=np.float64)
    np.add.at(parent_normal_sum, parent_ids, triangle_normals * triangle_area_mm2[:, None])
    parent_normals = parent_normal_sum / np.maximum(np.linalg.norm(parent_normal_sum, axis=1, keepdims=True), 1e-30)

    stored_points = points.astype(np.float64)
    stored_triangles = triangles.astype(np.int32)
    stored_parent_ids = parent_ids.astype(np.int32)
    signature = _mesh_signature(stored_points, stored_triangles, stored_parent_ids, units)
    radii = np.linalg.norm(points[triangles] - triangle_centers[:, None, :], axis=2).max(axis=1)
    source_vtp_sha256 = _file_sha256(vtp_path)
    summary = {
        "version": "step30-native-openfoam-topology-v3",
        "mesh_signature_algorithm": "sha256-vtk-triangulation-source-units-points-triangles-parent-ids-v3",
        "source_vtp": str(vtp_path),
        "source_vtp_sha256": source_vtp_sha256,
        "patch_name": patch_name,
        "time": str(time_name),
        "coordinate_units": "mm",
        "source_coordinate_units": units,
        "internal_coordinate_units": "mm",
        "coordinate_scale_to_mm": scale,
        "n_points": int(len(points)),
        "n_parent_faces": int(parent_count),
        "n_collision_triangles": int(len(triangles)),
        "surface_area_m2": float(parent_area_mm2.sum() * 1e-6),
        "bounds_mm": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        "mesh_signature": signature,
        "degenerate_triangles_removed": 0,
        "triangulation_method": "vtkTriangleFilter-parentFaceId-v3",
        "area_conservation_metric": "oriented-face-area-vector-v1",
        "total_area_relative_error": total_area_relative_error,
        "p99_parent_area_relative_error": float(np.percentile(relative_area_error, 99)),
        "max_parent_area_relative_error": float(relative_area_error.max()),
        "max_parent_area_vector_relative_error": float(area_vector_error.max()),
        "collision_scalar_area_m2": float(collision_parent_area_mm2.sum() * 1e-6),
        "collision_scalar_area_excess_relative": float(
            (collision_parent_area_mm2.sum() - parent_area_mm2.sum()) / parent_area_mm2.sum()
        ),
        "p99_polygon_warpage": float(np.percentile(polygon_warpage, 99)),
        "max_polygon_warpage": float(polygon_warpage.max()),
        "inverted_triangles": inverted_triangles,
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
        points_local_mm=stored_points,
        triangles_local_mm=stored_triangles,
        triangle_parent_face_id=stored_parent_ids,
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
        source_coordinate_units=np.array(units),
        internal_coordinate_units=np.array("mm"),
        coordinate_scale_to_mm=np.array(scale),
        artifact_version=np.array("step30-native-openfoam-topology-v3"),
        mesh_signature_algorithm=np.array("sha256-vtk-triangulation-source-units-points-triangles-parent-ids-v3"),
        source_vtp_sha256=np.array(source_vtp_sha256),
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
    strict_geometry: bool = True,
) -> None:
    """Fail closed on malformed or non-physical topology artifacts."""
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("topology points must have nonempty shape (N, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        raise ValueError("topology collision triangles must have nonempty shape (M, 3)")
    if len(triangle_parent_face_id) != len(triangles):
        raise ValueError("triangle parent IDs do not match triangle count")
    if face_centers.shape != (len(face_area_m2), 3):
        raise ValueError("face centres and face areas have inconsistent shapes")
    if face_normals.shape != face_centers.shape:
        raise ValueError("face normals and face centres have inconsistent shapes")
    if np.any(triangles < 0) or np.any(triangles >= len(points)):
        raise ValueError("topology triangle vertex index out of range")
    if not np.isfinite(points).all() or not np.isfinite(face_centers).all():
        raise ValueError("topology coordinates contain non-finite values")
    if not np.isfinite(face_normals).all() or not np.isfinite(face_area_m2).all():
        raise ValueError("topology normals/areas contain non-finite values")
    if np.any(face_area_m2 <= 0.0):
        raise ValueError("topology contains non-positive face areas")
    normal_norm = np.linalg.norm(face_normals, axis=1)
    if np.any(np.abs(normal_norm - 1.0) > 1e-4):
        raise ValueError("topology face normals are not unit vectors")
    if expected_parent_count is not None and len(face_area_m2) != expected_parent_count:
        raise ValueError("unexpected parent face count")
    if np.any(triangle_parent_face_id < 0) or np.any(triangle_parent_face_id >= len(face_area_m2)):
        raise ValueError("triangle parent ID out of range")
    if strict_geometry:
        parent_counts = np.bincount(triangle_parent_face_id, minlength=len(face_area_m2))
        if np.any(parent_counts == 0):
            raise ValueError("topology parent face has no collision triangles")
        triangle_keys = np.sort(triangles, axis=1)
        if len(np.unique(triangle_keys, axis=0)) != len(triangles):
            raise ValueError("topology contains duplicate collision triangles")
        xyz = points[triangles]
        triangle_area = 0.5 * np.linalg.norm(
            np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
        )
        if np.any(triangle_area <= 1e-12):
            raise ValueError("topology contains degenerate collision triangles")


def _topology_payload(
    data,
    points: np.ndarray,
    triangles: np.ndarray,
    parent_ids: np.ndarray,
    centers: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    tri_normals: np.ndarray,
    tri_centers: np.ndarray,
    radii: np.ndarray,
    *,
    signature: str,
    source_units: str,
    internal_units: str,
    scale_to_mm: float,
    artifact_version: str,
    signature_algorithm: str,
    source_vtp_sha256: str | None,
) -> dict:
    regular = radii <= max(15.0, 3.0 * np.median(radii))
    return {
        "backend": "cfd_topology",
        "face_id_namespace": "openfoam_bucket_patch",
        "mesh_signature": signature,
        "artifact_version": artifact_version,
        "mesh_signature_algorithm": signature_algorithm,
        "source_vtp_sha256": source_vtp_sha256,
        "patch_name": str(data["patch_name"].item()),
        "time_name": str(data["time_name"].item()),
        "source_coordinate_units": source_units,
        "internal_coordinate_units": internal_units,
        "coordinate_scale_to_mm": scale_to_mm,
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
        "parent_face_id": np.arange(len(areas), dtype=np.int64),
        "bounds_mm": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        "collision_bounds_mm": [
            triangles.min(axis=(0, 1)).tolist(),
            triangles.max(axis=(0, 1)).tolist(),
        ],
        "surface_area_m2": float(areas.sum()),
    }


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
    stored_points = data["points_local_mm"]
    stored_triangles = data["triangles_local_mm"]
    stored_parent_ids = data["triangle_parent_face_id"]
    points = stored_points.astype(np.float64)
    triangles = stored_triangles.astype(np.int64)
    parent_ids = stored_parent_ids.astype(np.int64)
    centers = data["face_centers_local_mm"].astype(np.float64)
    normals = data["face_normals_local"].astype(np.float64)
    areas = data["face_area_m2"].astype(np.float64)
    signature = str(data["mesh_signature"].item())
    source_units = str(data["source_coordinate_units"].item()) if "source_coordinate_units" in data.files else "mm"
    internal_units = str(data["internal_coordinate_units"].item()) if "internal_coordinate_units" in data.files else "mm"
    scale_to_mm = float(data["coordinate_scale_to_mm"].item()) if "coordinate_scale_to_mm" in data.files else 1.0
    if internal_units != "mm" or (source_units, scale_to_mm) not in {("mm", 1.0), ("m", 1000.0)}:
        raise ValueError("topology artifact has an invalid coordinate unit contract")
    artifact_version = str(data["artifact_version"].item()) if "artifact_version" in data.files else "step30-native-openfoam-topology-v1"
    signature_algorithm = str(data["mesh_signature_algorithm"].item()) if "mesh_signature_algorithm" in data.files else "sha256-normalized-arrays-v1"
    source_vtp_sha256 = str(data["source_vtp_sha256"].item()) if "source_vtp_sha256" in data.files else None
    is_v3 = artifact_version == "step30-native-openfoam-topology-v3"
    validate_topology_arrays(
        points,
        triangles,
        parent_ids,
        centers,
        normals,
        areas,
        strict_geometry=is_v3,
    )

    tri_normals = data["triangle_normals_local"].astype(np.float64)
    tri_centers = data["triangle_centers_local_mm"].astype(np.float64)
    radii = data["triangle_radii_mm"].astype(np.float64)
    if (
        tri_normals.shape != (len(triangles), 3)
        or tri_centers.shape != (len(triangles), 3)
        or radii.shape != (len(triangles),)
        or not np.isfinite(tri_normals).all()
        or not np.isfinite(tri_centers).all()
        or not np.isfinite(radii).all()
    ):
        raise ValueError("topology triangle metadata is malformed")
    if not is_v3:
        return _topology_payload(
            data,
            points,
            triangles,
            parent_ids,
            centers,
            normals,
            areas,
            tri_normals,
            tri_centers,
            radii,
            signature=signature,
            source_units=source_units,
            internal_units=internal_units,
            scale_to_mm=scale_to_mm,
            artifact_version=artifact_version,
            signature_algorithm=signature_algorithm,
            source_vtp_sha256=source_vtp_sha256,
        )

    triangle_xyz = points[triangles]
    triangle_area_vectors = 0.5 * np.cross(
        triangle_xyz[:, 1] - triangle_xyz[:, 0],
        triangle_xyz[:, 2] - triangle_xyz[:, 0],
    )
    triangle_area = np.linalg.norm(triangle_area_vectors, axis=1)
    computed_tri_normals = triangle_area_vectors / triangle_area[:, None]
    computed_tri_centers = triangle_xyz.mean(axis=1)
    computed_radii = np.linalg.norm(
        triangle_xyz - computed_tri_centers[:, None, :], axis=2
    ).max(axis=1)
    if not np.allclose(tri_normals, computed_tri_normals, rtol=1e-5, atol=1e-6):
        raise ValueError("topology triangle normals differ from points/indices")
    if not np.allclose(tri_centers, computed_tri_centers, rtol=1e-6, atol=2e-5):
        raise ValueError("topology triangle centres differ from points/indices")
    if not np.allclose(radii, computed_radii, rtol=1e-5, atol=2e-5):
        raise ValueError("topology triangle radii differ from points/indices")

    parent_area_vectors = np.zeros((len(areas), 3), dtype=np.float64)
    np.add.at(parent_area_vectors, parent_ids, triangle_area_vectors)
    computed_areas = np.linalg.norm(parent_area_vectors, axis=1) * 1e-6
    computed_normals = parent_area_vectors / np.maximum(
        np.linalg.norm(parent_area_vectors, axis=1, keepdims=True), 1e-30
    )
    parent_scalar_area = np.bincount(
        parent_ids, weights=triangle_area, minlength=len(areas)
    )
    computed_centers = np.zeros_like(centers)
    np.add.at(computed_centers, parent_ids, computed_tri_centers * triangle_area[:, None])
    computed_centers /= parent_scalar_area[:, None]
    if not np.allclose(areas, computed_areas, rtol=2e-5, atol=1e-12):
        raise ValueError("topology parent areas differ from collision geometry")
    if not np.allclose(normals, computed_normals, rtol=1e-5, atol=2e-6):
        raise ValueError("topology parent normals differ from collision geometry")
    if not np.allclose(centers, computed_centers, rtol=1e-6, atol=2e-5):
        raise ValueError("topology parent centres differ from collision geometry")
    orientation = np.einsum("ij,ij->i", computed_tri_normals, computed_normals[parent_ids])
    if np.any(orientation <= 0.0):
        raise ValueError("topology contains inverted collision triangles")

    if is_v3:
        required_v3 = {
            "face_id", "parent_face_id", "source_coordinate_units",
            "internal_coordinate_units", "coordinate_scale_to_mm", "artifact_version",
            "mesh_signature_algorithm", "source_vtp_sha256",
        }
        missing_v3 = sorted(required_v3 - set(data.files))
        if missing_v3:
            raise ValueError(f"topology v3 artifact missing fields: {missing_v3}")
        face_ids = data["face_id"].astype(np.int64)
        stored_face_ids = data["parent_face_id"].astype(np.int64)
        expected_face_ids = np.arange(len(areas), dtype=np.int64)
        if not np.array_equal(face_ids, expected_face_ids) or not np.array_equal(stored_face_ids, expected_face_ids):
            raise ValueError("topology v3 parent face IDs are not canonical and unique")
        expected_algorithm = "sha256-vtk-triangulation-source-units-points-triangles-parent-ids-v3"
        if signature_algorithm != expected_algorithm:
            raise ValueError("topology v3 signature algorithm is invalid")
        if not isinstance(source_vtp_sha256, str) or len(source_vtp_sha256) != 64:
            raise ValueError("topology v3 source VTP hash is invalid")
        computed_signature = _mesh_signature(
            stored_points, stored_triangles, stored_parent_ids, source_units
        )
        if signature != computed_signature:
            raise ValueError("topology v3 mesh signature differs from stored arrays")

    return _topology_payload(
        data,
        points,
        triangles,
        parent_ids,
        centers,
        normals,
        areas,
        computed_tri_normals,
        computed_tri_centers,
        computed_radii,
        signature=signature,
        source_units=source_units,
        internal_units=internal_units,
        scale_to_mm=scale_to_mm,
        artifact_version=artifact_version,
        signature_algorithm=signature_algorithm,
        source_vtp_sha256=source_vtp_sha256,
    )
