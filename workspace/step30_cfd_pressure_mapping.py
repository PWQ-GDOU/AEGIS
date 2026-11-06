"""Conservative CFD ``pMean`` to structural-surface pressure mapping.

Public entry points fail closed: identity, unit, coverage, orientation, or
conservation failures produce a ``blocked`` report and never a usable load.
Array helpers support software tests without representing production evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree


VERSION = "step30-cfd-pressure-mapping-v1"
_HEX = frozenset("0123456789abcdef")


class PressureMappingError(ValueError):
    """Raised when a pressure load cannot be established unambiguously."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in _HEX for char in value.lower())
    )


def _validated_surface_arrays(
    surface: Mapping[str, Any], name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if surface.get("coordinate_units") != "m":
        raise PressureMappingError(f"{name} surface coordinate units must be SI metres")
    if not is_sha256(surface.get("topology_sha256")):
        raise PressureMappingError(f"{name} surface has no valid topology SHA-256")
    centers = np.asarray(surface.get("centers_m"), dtype=np.float64)
    normals = np.asarray(surface.get("normals"), dtype=np.float64)
    areas = np.asarray(surface.get("area_m2"), dtype=np.float64)
    if (
        centers.ndim != 2
        or centers.shape[1:] != (3,)
        or normals.shape != centers.shape
        or areas.shape != (len(centers),)
        or len(centers) == 0
    ):
        raise PressureMappingError(f"{name} parent-face geometry shapes are inconsistent")
    if (
        not np.isfinite(centers).all()
        or not np.isfinite(normals).all()
        or not np.isfinite(areas).all()
    ):
        raise PressureMappingError(f"{name} parent-face geometry contains non-finite values")
    if np.any(areas <= 0.0):
        raise PressureMappingError(f"{name} parent-face areas must be positive")
    normal_length = np.linalg.norm(normals, axis=1)
    if not np.allclose(normal_length, 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise PressureMappingError(f"{name} parent-face normals must be unit vectors")
    return centers, normals, areas


def _array_hash(*values: np.ndarray, label: str) -> str:
    digest = hashlib.sha256(label.encode("ascii"))
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _coordinate_scale(units: str) -> float:
    normalized = str(units).strip().lower()
    if normalized == "m":
        return 1.0
    if normalized == "mm":
        return 1.0e-3
    raise PressureMappingError("coordinate units must be explicitly 'm' or 'mm'")


def kinematic_pressure_to_pa(
    values: Any, density_kg_m3: float, pressure_units: str = "m2/s2"
) -> np.ndarray:
    """Convert OpenFOAM kinematic pressure to dynamic pressure in Pa."""
    normalized = str(pressure_units).strip().lower().replace("^", "")
    normalized = normalized.replace(" ", "")
    if normalized not in {"m2/s2", "m2s-2"}:
        raise PressureMappingError(
            "pMean unit contract mismatch: expected kinematic pressure m2/s2"
        )
    try:
        rho = float(density_kg_m3)
    except (TypeError, ValueError) as exc:
        raise PressureMappingError("density_kg_m3 must be numeric") from exc
    pressure = np.asarray(values, dtype=np.float64)
    if not math.isfinite(rho) or rho <= 0.0:
        raise PressureMappingError("density_kg_m3 must be finite and positive")
    if pressure.ndim != 1 or len(pressure) == 0 or not np.isfinite(pressure).all():
        raise PressureMappingError("pMean must be a nonempty finite scalar array")
    return pressure * rho


def _polygon_geometry(
    points_m: np.ndarray, polygons: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    areas: list[float] = []
    for face_id, polygon in enumerate(polygons):
        ids = np.asarray(polygon, dtype=np.int64)
        if ids.ndim != 1 or len(ids) < 3:
            raise PressureMappingError(f"parent face {face_id} has fewer than 3 nodes")
        if np.any(ids < 0) or np.any(ids >= len(points_m)):
            raise PressureMappingError(f"parent face {face_id} has an invalid node ID")
        xyz = points_m[ids]
        anchor = xyz[0]
        area_vectors = 0.5 * np.cross(xyz[1:-1] - anchor, xyz[2:] - anchor)
        triangle_areas = np.linalg.norm(area_vectors, axis=1)
        scalar_area = float(np.sum(triangle_areas))
        area_vector = np.sum(area_vectors, axis=0)
        oriented_area = float(np.linalg.norm(area_vector))
        if scalar_area <= 1.0e-18 or oriented_area <= 1.0e-18:
            raise PressureMappingError(f"parent face {face_id} is degenerate")
        if oriented_area / scalar_area < 0.95:
            raise PressureMappingError(f"parent face {face_id} is excessively warped")
        triangle_centers = (anchor + xyz[1:-1] + xyz[2:]) / 3.0
        center = np.sum(triangle_centers * triangle_areas[:, None], axis=0) / scalar_area
        centers.append(center)
        normals.append(area_vector / oriented_area)
        areas.append(oriented_area)
    return (
        np.asarray(centers, dtype=np.float64),
        np.asarray(normals, dtype=np.float64),
        np.asarray(areas, dtype=np.float64),
    )


def build_surface_geometry(
    points: Any,
    faces: Sequence[Any],
    *,
    coordinate_units: str,
) -> dict[str, Any]:
    """Build oriented parent-face geometry in SI units."""
    scale = _coordinate_scale(coordinate_units)
    points_m = np.asarray(points, dtype=np.float64) * scale
    if points_m.ndim != 2 or points_m.shape[1] != 3 or len(points_m) == 0:
        raise PressureMappingError("surface points must have shape (N, 3)")
    if not np.isfinite(points_m).all():
        raise PressureMappingError("surface points contain non-finite values")
    polygons = [np.asarray(face, dtype=np.int64) for face in faces]
    if not polygons:
        raise PressureMappingError("surface must contain parent faces")
    centers, normals, areas = _polygon_geometry(points_m, polygons)
    offsets = np.cumsum([0, *[len(face) for face in polygons]], dtype=np.int64)
    connectivity = np.concatenate(polygons).astype(np.int64, copy=False)
    topology_sha256 = _array_hash(
        points_m,
        offsets,
        connectivity,
        label="step30-oriented-parent-surface-v1",
    )
    return {
        "points_m": points_m,
        "faces": polygons,
        "centers_m": centers,
        "normals": normals,
        "area_m2": areas,
        "coordinate_units": "m",
        "topology_sha256": topology_sha256,
    }


def load_vtp_pmean(
    vtp_path: str | Path,
    *,
    density_kg_m3: float,
    coordinate_units: str,
    pressure_units: str = "m2/s2",
    expected_vtp_sha256: str | None = None,
    expected_topology_sha256: str | None = None,
    pressure_array: str = "pMean",
) -> dict[str, Any]:
    """Load VTP parent polygons and cell-data ``pMean`` with identity checks."""
    path = Path(vtp_path).resolve()
    if not path.is_file():
        raise PressureMappingError(f"VTP file does not exist: {path}")
    actual_file_hash = sha256_file(path)
    if expected_vtp_sha256 is not None:
        if not is_sha256(expected_vtp_sha256) or actual_file_hash != expected_vtp_sha256.lower():
            raise PressureMappingError("VTP SHA-256 mismatch")
    try:
        import vtk
    except ImportError as exc:  # pragma: no cover
        raise PressureMappingError("vtk is required to read VTP pressure data") from exc
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetNumberOfPolys() <= 0:
        raise PressureMappingError("VTP contains no parent polygons")
    if poly.GetNumberOfCells() != poly.GetNumberOfPolys():
        raise PressureMappingError("VTP must contain polygons only")
    points = np.asarray(
        [poly.GetPoint(index) for index in range(poly.GetNumberOfPoints())],
        dtype=np.float64,
    )
    id_list = vtk.vtkIdList()
    polygons: list[np.ndarray] = []
    for cell_id in range(poly.GetNumberOfCells()):
        poly.GetCellPoints(cell_id, id_list)
        polygons.append(
            np.asarray(
                [id_list.GetId(index) for index in range(id_list.GetNumberOfIds())],
                dtype=np.int64,
            )
        )
    array = poly.GetCellData().GetArray(pressure_array)
    if array is None or array.GetNumberOfComponents() != 1:
        raise PressureMappingError(f"VTP cell data is missing scalar {pressure_array!r}")
    if array.GetNumberOfTuples() != len(polygons):
        raise PressureMappingError("pMean row count differs from parent-face count")
    parent_ids_array = poly.GetCellData().GetArray("parentFaceId")
    if parent_ids_array is not None:
        parent_ids = np.asarray(
            [int(parent_ids_array.GetTuple1(index)) for index in range(len(polygons))]
        )
        if not np.array_equal(parent_ids, np.arange(len(polygons))):
            raise PressureMappingError("VTP parentFaceId values must be canonical and ordered")
    geometry = build_surface_geometry(
        points, polygons, coordinate_units=coordinate_units
    )
    if expected_topology_sha256 is not None:
        if not is_sha256(expected_topology_sha256):
            raise PressureMappingError("expected topology SHA-256 is malformed")
        if geometry["topology_sha256"] != expected_topology_sha256.lower():
            raise PressureMappingError("VTP parent topology SHA-256 mismatch")
    pmean = np.asarray(
        [array.GetTuple1(index) for index in range(len(polygons))], dtype=np.float64
    )
    geometry.update(
        {
            "pressure_pa": kinematic_pressure_to_pa(
                pmean, density_kg_m3, pressure_units
            ),
            "pressure_units": "Pa",
            "source_pressure_units": "m2/s2",
            "density_kg_m3": float(density_kg_m3),
            "source_path": str(path),
            "source_vtp_sha256": actual_file_hash,
            "parent_face_id": np.arange(len(polygons), dtype=np.int64),
        }
    )
    return geometry


def _load_vector_matrix(
    centers_m: np.ndarray, normals: np.ndarray, area_m2: np.ndarray, origin_m: np.ndarray
) -> np.ndarray:
    force_per_pa = -normals * area_m2[:, None]
    moment_per_pa = np.cross(centers_m - origin_m, force_per_pa)
    return np.vstack((force_per_pa.T, moment_per_pa.T))


def resultant_force_moment(
    pressure_pa: Any,
    centers_m: Any,
    normals: Any,
    area_m2: Any,
    *,
    origin_m: Any = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    pressure = np.asarray(pressure_pa, dtype=np.float64)
    centers = np.asarray(centers_m, dtype=np.float64)
    directions = np.asarray(normals, dtype=np.float64)
    areas = np.asarray(area_m2, dtype=np.float64)
    origin = np.asarray(origin_m, dtype=np.float64)
    if pressure.shape != areas.shape or centers.shape != (len(areas), 3):
        raise PressureMappingError("pressure and parent geometry shapes differ")
    matrix = _load_vector_matrix(centers, directions, areas, origin)
    load = matrix @ pressure
    return load[:3], load[3:]


def pressure_load_sha256(
    pressure_pa: Any,
    target_surface: Mapping[str, Any],
    *,
    origin_m: Any = (0.0, 0.0, 0.0),
) -> str:
    """Return the immutable identity of one SI pressure load and its geometry."""
    centers, normals, areas = _validated_surface_arrays(target_surface, "target")
    pressure = np.asarray(pressure_pa, dtype=np.float64)
    origin = np.asarray(origin_m, dtype=np.float64)
    if pressure.shape != (len(centers),) or not np.isfinite(pressure).all():
        raise PressureMappingError(
            "pressure values must be finite with one value per target face"
        )
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise PressureMappingError("origin_m must contain three finite coordinates")
    return _array_hash(
        pressure,
        centers,
        normals,
        areas,
        origin,
        label="step30-structural-pressure-load-pa-v1",
    )


def conservative_map_arrays(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    origin_m: Any = (0.0, 0.0, 0.0),
    minimum_coverage: float = 0.995,
    minimum_normal_dot: float = 0.75,
    maximum_distance_m: float | None = None,
    conservation_rtol: float = 1.0e-8,
) -> dict[str, Any]:
    """Nearest-parent transfer followed by exact force/moment correction."""
    required = {"centers_m", "normals", "area_m2", "topology_sha256"}
    for name, surface in (("source", source), ("target", target)):
        missing = sorted(required - set(surface))
        if missing:
            raise PressureMappingError(f"{name} surface missing fields: {missing}")
    if source.get("pressure_units") != "Pa":
        raise PressureMappingError("source pressure must be converted to Pa before mapping")
    source_pressure = np.asarray(source.get("pressure_pa"), dtype=np.float64)
    source_centers, source_normals, source_area = _validated_surface_arrays(
        source, "source"
    )
    target_centers, target_normals, target_area = _validated_surface_arrays(
        target, "target"
    )
    if source_pressure.shape != source_area.shape:
        raise PressureMappingError("source pressure count differs from parent-face count")
    if source.get("software_validation_only") is not True and not is_sha256(
        source.get("source_vtp_sha256")
    ):
        raise PressureMappingError("production pressure source has no valid VTP SHA-256")
    for name, array in (
        ("source pressure", source_pressure),
        ("source geometry", source_centers),
        ("target geometry", target_centers),
    ):
        if array.size == 0 or not np.isfinite(array).all():
            raise PressureMappingError(f"{name} is empty or non-finite")
    if not 0.0 < minimum_coverage <= 1.0:
        raise PressureMappingError("minimum_coverage must be in (0, 1]")
    if not -1.0 <= minimum_normal_dot <= 1.0:
        raise PressureMappingError("minimum_normal_dot must be in [-1, 1]")
    characteristic = max(
        math.sqrt(float(np.median(source_area))),
        math.sqrt(float(np.median(target_area))),
    )
    distance_limit = (
        2.5 * characteristic if maximum_distance_m is None else float(maximum_distance_m)
    )
    if not math.isfinite(distance_limit) or distance_limit <= 0.0:
        raise PressureMappingError("maximum_distance_m must be finite and positive")

    source_tree = cKDTree(source_centers)
    target_distance, source_index = source_tree.query(target_centers, k=1)
    target_dot = np.einsum(
        "ij,ij->i", target_normals, source_normals[source_index]
    )
    target_supported = (target_distance <= distance_limit) & (
        target_dot >= minimum_normal_dot
    )
    target_coverage = float(np.sum(target_area[target_supported]) / np.sum(target_area))

    target_tree = cKDTree(target_centers)
    source_distance, target_index = target_tree.query(source_centers, k=1)
    source_dot = np.einsum(
        "ij,ij->i", source_normals, target_normals[target_index]
    )
    source_supported = (source_distance <= distance_limit) & (
        source_dot >= minimum_normal_dot
    )
    source_coverage = float(np.sum(source_area[source_supported]) / np.sum(source_area))
    if min(source_coverage, target_coverage) < minimum_coverage:
        raise PressureMappingError(
            "surface coverage/normal gate failed: "
            f"source={source_coverage:.6f}, target={target_coverage:.6f}"
        )
    if np.any(~target_supported):
        raise PressureMappingError("uncovered target faces cannot receive a conservative load")

    initial_pressure = source_pressure[source_index]
    origin = np.asarray(origin_m, dtype=np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise PressureMappingError("origin_m must contain three finite coordinates")
    source_matrix = _load_vector_matrix(
        source_centers, source_normals, source_area, origin
    )
    target_matrix = _load_vector_matrix(
        target_centers, target_normals, target_area, origin
    )
    desired = source_matrix @ source_pressure
    residual = desired - target_matrix @ initial_pressure
    correction = target_matrix.T @ np.linalg.pinv(
        target_matrix @ target_matrix.T, rcond=1.0e-13
    ) @ residual
    mapped_pressure = initial_pressure + correction
    pressure_scale = max(float(np.max(np.abs(source_pressure))), 1.0)
    actual = target_matrix @ mapped_pressure
    force_scale = max(np.linalg.norm(desired[:3]), pressure_scale * np.sum(source_area), 1.0)
    length_scale = max(
        float(np.linalg.norm(source_centers - origin, axis=1).max()), characteristic
    )
    moment_scale = max(np.linalg.norm(desired[3:]), force_scale * length_scale, 1.0)
    force_error = float(np.linalg.norm(actual[:3] - desired[:3]) / force_scale)
    moment_error = float(np.linalg.norm(actual[3:] - desired[3:]) / moment_scale)
    if force_error > conservation_rtol or moment_error > conservation_rtol:
        raise PressureMappingError(
            "conservative mapping failed resultant check: "
            f"force={force_error:.3e}, moment={moment_error:.3e}"
        )
    load_sha256 = pressure_load_sha256(mapped_pressure, target, origin_m=origin)
    return {
        "version": VERSION,
        "status": "mapped",
        "pressure_gate": "passed",
        "pressure_units": "Pa",
        "coordinate_units": "m",
        "pressure_pa": mapped_pressure,
        "target_face_id": np.arange(len(mapped_pressure), dtype=np.int64),
        "source_vtp_sha256": source.get("source_vtp_sha256"),
        "source_topology_sha256": source["topology_sha256"],
        "target_topology_sha256": target["topology_sha256"],
        "load_sha256": load_sha256,
        "origin_m": origin.tolist(),
        "coverage": {"source_area_fraction": source_coverage, "target_area_fraction": target_coverage},
        "normal_check": {
            "minimum_required_dot": float(minimum_normal_dot),
            "minimum_target_dot": float(np.min(target_dot)),
            "minimum_source_dot": float(np.min(source_dot)),
        },
        "resultants": {
            "source_force_n": desired[:3].tolist(),
            "mapped_force_n": actual[:3].tolist(),
            "source_moment_nm": desired[3:].tolist(),
            "mapped_moment_nm": actual[3:].tolist(),
            "force_relative_error": force_error,
            "moment_relative_error": moment_error,
        },
        "software_validation_only": bool(source.get("software_validation_only", False)),
    }


def map_vtp_pressure(
    vtp_path: str | Path,
    target_points: Any,
    target_faces: Sequence[Any],
    *,
    density_kg_m3: float,
    source_coordinate_units: str,
    target_coordinate_units: str,
    pressure_units: str = "m2/s2",
    expected_vtp_sha256: str | None = None,
    expected_source_topology_sha256: str | None = None,
    expected_target_topology_sha256: str | None = None,
    origin_m: Any = (0.0, 0.0, 0.0),
    minimum_coverage: float = 0.995,
    minimum_normal_dot: float = 0.75,
    maximum_distance_m: float | None = None,
) -> dict[str, Any]:
    """Map one VTP field, returning a fail-closed report."""
    try:
        source = load_vtp_pmean(
            vtp_path,
            density_kg_m3=density_kg_m3,
            coordinate_units=source_coordinate_units,
            pressure_units=pressure_units,
            expected_vtp_sha256=expected_vtp_sha256,
            expected_topology_sha256=expected_source_topology_sha256,
        )
        target = build_surface_geometry(
            target_points, target_faces, coordinate_units=target_coordinate_units
        )
        if expected_target_topology_sha256 is not None:
            if not is_sha256(expected_target_topology_sha256):
                raise PressureMappingError("expected target topology SHA-256 is malformed")
            if target["topology_sha256"] != expected_target_topology_sha256.lower():
                raise PressureMappingError("structural target topology SHA-256 mismatch")
        return conservative_map_arrays(
            source,
            target,
            origin_m=origin_m,
            minimum_coverage=minimum_coverage,
            minimum_normal_dot=minimum_normal_dot,
            maximum_distance_m=maximum_distance_m,
        )
    except (OSError, PressureMappingError, ValueError) as exc:
        return {
            "version": VERSION,
            "status": "blocked",
            "pressure_gate": "blocked",
            "reason": str(exc),
        }


def json_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe mapping report."""
    result: dict[str, Any] = {}
    for key, value in report.items():
        result[key] = value.tolist() if isinstance(value, np.ndarray) else value
    json.dumps(result, allow_nan=False)
    return result


__all__ = [
    "PressureMappingError",
    "VERSION",
    "build_surface_geometry",
    "conservative_map_arrays",
    "is_sha256",
    "json_summary",
    "kinematic_pressure_to_pa",
    "load_vtp_pmean",
    "map_vtp_pressure",
    "pressure_load_sha256",
    "resultant_force_moment",
    "sha256_file",
]
