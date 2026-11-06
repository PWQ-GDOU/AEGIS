"""Hermetic tests for native OpenFOAM topology v3 artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import vtk

from step30_cfd_topology import build_topology_from_vtp, load_topology_artifact


def _write_patch_vtp(path: Path) -> None:
    points = vtk.vtkPoints()
    for point in [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (2.0, 1.0, 0.0),
    ]:
        points.InsertNextPoint(point)

    polygons = vtk.vtkCellArray()
    quad = vtk.vtkPolygon()
    quad.GetPointIds().SetNumberOfIds(4)
    for index, point_id in enumerate([0, 1, 2, 3]):
        quad.GetPointIds().SetId(index, point_id)
    polygons.InsertNextCell(quad)

    triangle = vtk.vtkTriangle()
    for index, point_id in enumerate([4, 5, 6]):
        triangle.GetPointIds().SetId(index, point_id)
    polygons.InsertNextCell(triangle)

    patch = vtk.vtkPolyData()
    patch.SetPoints(points)
    patch.SetPolys(polygons)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(patch)
    assert writer.Write() == 1


def _write_duplicate_patch_vtp(path: Path) -> None:
    points = vtk.vtkPoints()
    for point in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]:
        points.InsertNextPoint(point)
    polygons = vtk.vtkCellArray()
    for point_ids in ([0, 1, 2], [2, 1, 0]):
        triangle = vtk.vtkTriangle()
        for index, point_id in enumerate(point_ids):
            triangle.GetPointIds().SetId(index, point_id)
        polygons.InsertNextCell(triangle)
    patch = vtk.vtkPolyData()
    patch.SetPoints(points)
    patch.SetPolys(polygons)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(patch)
    assert writer.Write() == 1


def test_topology_v3_round_trip_preserves_parent_faces(tmp_path: Path) -> None:
    source = tmp_path / "bucket.vtp"
    artifact = tmp_path / "bucket_topology.npz"
    summary_path = tmp_path / "bucket_topology.json"
    _write_patch_vtp(source)

    summary = build_topology_from_vtp(
        source,
        patch_name="bucket",
        time_name="7",
        units="m",
        output_npz=artifact,
        output_json=summary_path,
    )
    topology = load_topology_artifact(artifact)
    with np.load(artifact, allow_pickle=False) as stored:
        assert stored["points_local_mm"].dtype == np.float64

    assert summary["version"] == "step30-native-openfoam-topology-v3"
    assert summary["triangulation_method"] == "vtkTriangleFilter-parentFaceId-v3"
    assert summary["n_parent_faces"] == 2
    assert summary["n_collision_triangles"] == 3
    assert summary["inverted_triangles"] == 0
    assert summary["total_area_relative_error"] < 1e-12
    assert np.isclose(summary["surface_area_m2"], 1.5)

    assert topology["artifact_version"] == summary["version"]
    assert topology["mesh_signature"] == summary["mesh_signature"]
    assert topology["source_coordinate_units"] == "m"
    assert topology["coordinate_scale_to_mm"] == 1000.0
    assert topology["time_name"] == "7"
    assert topology["n_parent_faces"] == 2
    assert np.array_equal(np.bincount(topology["collision_face_ids"]), [2, 1])
    assert np.allclose(topology["face_area_m2"], [1.0, 0.5])
    assert np.allclose(topology["face_normals"], [[0.0, 0.0, 1.0]] * 2)


def test_topology_v3_rejects_duplicate_collision_triangles(tmp_path: Path) -> None:
    source = tmp_path / "duplicate_bucket.vtp"
    _write_duplicate_patch_vtp(source)
    with pytest.raises(ValueError, match="duplicate collision triangles"):
        build_topology_from_vtp(
            source,
            units="m",
            output_npz=tmp_path / "duplicate_topology.npz",
            output_json=tmp_path / "duplicate_topology.json",
        )


def test_legacy_topology_remains_readable_with_historical_duplicates(tmp_path: Path) -> None:
    artifact = tmp_path / "legacy_topology.npz"
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    triangles = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32)
    center = np.array([[1.0 / 3.0, 1.0 / 3.0, 0.0]], dtype=np.float32)
    np.savez_compressed(
        artifact,
        points_local_mm=points,
        triangles_local_mm=triangles,
        triangle_parent_face_id=np.array([0, 0], dtype=np.int32),
        face_centers_local_mm=center,
        face_normals_local=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        face_area_m2=np.array([1.0e-6], dtype=np.float32),
        triangle_centers_local_mm=np.repeat(center, 2, axis=0),
        triangle_normals_local=np.array([[0.0, 0.0, 1.0]] * 2, dtype=np.float32),
        triangle_radii_mm=np.array([np.sqrt(5.0) / 3.0] * 2, dtype=np.float32),
        mesh_signature=np.array("legacy-signature"),
        patch_name=np.array("bucket"),
        time_name=np.array("legacy"),
    )
    topology = load_topology_artifact(artifact)
    assert topology["artifact_version"] == "step30-native-openfoam-topology-v1"
    assert len(topology["collision_triangles_mm"]) == 2


def _built_artifact(tmp_path: Path) -> Path:
    source = tmp_path / "bucket.vtp"
    artifact = tmp_path / "bucket_topology.npz"
    _write_patch_vtp(source)
    build_topology_from_vtp(
        source,
        units="m",
        output_npz=artifact,
        output_json=tmp_path / "bucket_topology.json",
    )
    return artifact


def _rewrite_npz(path: Path, name: str, value: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays[name] = value
    np.savez_compressed(path, **arrays)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("mesh_signature", np.array("0" * 64), "mesh signature"),
        ("parent_face_id", np.array([1, 1], dtype=np.int32), "parent face IDs"),
        (
            "triangles_local_mm",
            np.array([[999, 1, 2], [0, 2, 3], [4, 5, 6]], dtype=np.int32),
            "vertex index out of range",
        ),
    ],
)
def test_topology_v3_loader_rejects_corrupt_identity_or_indices(
    tmp_path: Path,
    field: str,
    replacement: np.ndarray,
    message: str,
) -> None:
    artifact = _built_artifact(tmp_path)
    _rewrite_npz(artifact, field, replacement)
    with pytest.raises(ValueError, match=message):
        load_topology_artifact(artifact)
