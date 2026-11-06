"""Synthetic/analytic software tests for the Step30 structural toolchain.

These fixtures do not run CalculiX and do not constitute real FEM, erosion, or
protection evidence.  Tests only exercise mapping, contracts, and fail-closed
gates.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from typing import Any

import meshio
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from step30_calculix_case import (
    CalculixCaseError,
    generate_calculix_case,
    inspect_structural_mesh,
)
from step30_cfd_pressure_mapping import (
    PressureMappingError,
    build_surface_geometry,
    conservative_map_arrays,
    kinematic_pressure_to_pa,
    map_vtp_pressure,
    resultant_force_moment,
    sha256_file,
)
from step30_fem_results import (
    FEMResultsError,
    parse_calculix_results,
    project_ai5_stress,
    validate_calculix_results,
)
from step30_fem_validation import validate_fem_evidence, validate_three_mesh_convergence
from step30_multiobjective_tradeoff import build_pareto_tradeoff, validate_tradeoff
from step30_protection_design import classify_protection_zones, design_protection


_HASH = "a" * 64


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _write_vtp(
    path: pathlib.Path,
    points: np.ndarray,
    faces: list[list[int]],
    pressure: list[float],
) -> None:
    vtk = pytest.importorskip("vtk")
    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(*(float(value) for value in point))
    polygons = vtk.vtkCellArray()
    for face in faces:
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(len(face))
        for index, point_id in enumerate(face):
            polygon.GetPointIds().SetId(index, int(point_id))
        polygons.InsertNextCell(polygon)
    pmean = vtk.vtkDoubleArray()
    pmean.SetName("pMean")
    parent = vtk.vtkIntArray()
    parent.SetName("parentFaceId")
    for index, value in enumerate(pressure):
        pmean.InsertNextValue(float(value))
        parent.InsertNextValue(index)
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetPolys(polygons)
    poly.GetCellData().AddArray(pmean)
    poly.GetCellData().AddArray(parent)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    assert writer.Write() == 1


def _tetra10_mesh(*, include_ai5: bool = True) -> meshio.Mesh:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    tetra = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int64)
    # First face is P1 with outward -z orientation.  The second face is P2.
    triangles = np.asarray(
        [[0, 2, 1, 6, 5, 4], [0, 3, 1, 7, 8, 4]], dtype=np.int64
    )
    cell_sets: dict[str, list[np.ndarray]] = {
        "WETTED_SURFACE": [np.asarray([], dtype=np.int64), np.asarray([0])],
        "ROOT_INTERFACE": [np.asarray([], dtype=np.int64), np.asarray([1])],
        "STEEL": [np.asarray([0]), np.asarray([], dtype=np.int64)],
    }
    if include_ai5:
        cell_sets["AI5_04"] = [np.asarray([], dtype=np.int64), np.asarray([1])]
    return meshio.Mesh(
        points,
        [("tetra10", tetra), ("triangle6", triangles)],
        cell_sets=cell_sets,
    )


def _material() -> dict[str, dict[str, float]]:
    return {
        "STEEL": {
            "young_modulus_pa": 2.05e11,
            "poisson_ratio": 0.29,
            "density_kg_m3": 7850.0,
        }
    }


def _mapped_tetra_pressure(tmp_path: pathlib.Path) -> dict[str, Any]:
    mesh = _tetra10_mesh()
    vtp_path = tmp_path / "tetra_wetted.vtp"
    _write_vtp(vtp_path, np.asarray(mesh.points), [[0, 2, 1]], [2.0])
    return map_vtp_pressure(
        vtp_path,
        mesh.points,
        [[0, 2, 1]],
        density_kg_m3=1000.0,
        source_coordinate_units="m",
        target_coordinate_units="m",
        pressure_units="m2/s2",
        expected_vtp_sha256=sha256_file(vtp_path),
        origin_m=(0.0, 0.0, 0.0),
    )


def test_kinematic_pressure_units_convert_to_pa_and_mismatch_blocks() -> None:
    converted = kinematic_pressure_to_pa([1.25, -0.5], 1000.0, "m2/s2")
    assert converted.tolist() == pytest.approx([1250.0, -500.0])
    with pytest.raises(PressureMappingError, match="unit contract mismatch"):
        kinematic_pressure_to_pa([1.0], 1000.0, "Pa")
    with pytest.raises(PressureMappingError, match="coordinate units"):
        build_surface_geometry(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 1, 2]],
            coordinate_units="inch",
        )


def test_vtp_pmean_mapping_checks_hash_and_si_resultant(tmp_path: pathlib.Path) -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    path = tmp_path / "wall.vtp"
    _write_vtp(path, points, [[0, 1, 2, 3]], [2.0])
    report = map_vtp_pressure(
        path,
        points,
        [[0, 1, 2, 3]],
        density_kg_m3=1000.0,
        source_coordinate_units="m",
        target_coordinate_units="m",
        expected_vtp_sha256=sha256_file(path),
    )
    assert report["status"] == "mapped"
    assert report["pressure_pa"].tolist() == pytest.approx([2000.0])
    assert report["resultants"]["mapped_force_n"] == pytest.approx([0, 0, -2000])
    assert report["resultants"]["mapped_moment_nm"] == pytest.approx([-1000, 1000, 0])

    blocked = map_vtp_pressure(
        path,
        points,
        [[0, 1, 2, 3]],
        density_kg_m3=1000.0,
        source_coordinate_units="m",
        target_coordinate_units="m",
        expected_vtp_sha256="0" * 64,
    )
    assert blocked["status"] == "blocked"
    assert "SHA-256 mismatch" in blocked["reason"]


def test_conservative_mapping_preserves_force_and_same_origin_moment() -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    source = build_surface_geometry(points, [[0, 1, 2, 3]], coordinate_units="m")
    source.update(
        {
            "pressure_pa": np.asarray([1234.5]),
            "pressure_units": "Pa",
            "software_validation_only": True,
        }
    )
    target = build_surface_geometry(points, [[0, 1, 2], [0, 2, 3]], coordinate_units="m")
    origin = (0.25, -0.5, 1.0)
    mapped = conservative_map_arrays(source, target, origin_m=origin)
    source_force, source_moment = resultant_force_moment(
        source["pressure_pa"],
        source["centers_m"],
        source["normals"],
        source["area_m2"],
        origin_m=origin,
    )
    target_force, target_moment = resultant_force_moment(
        mapped["pressure_pa"],
        target["centers_m"],
        target["normals"],
        target["area_m2"],
        origin_m=origin,
    )
    assert target_force == pytest.approx(source_force, rel=1.0e-11, abs=1.0e-10)
    assert target_moment == pytest.approx(source_moment, rel=1.0e-11, abs=1.0e-10)
    assert mapped["resultants"]["force_relative_error"] <= 1.0e-8
    assert mapped["resultants"]["moment_relative_error"] <= 1.0e-8


def test_structural_mesh_missing_required_group_is_rejected() -> None:
    with pytest.raises(CalculixCaseError, match="AI5_04"):
        inspect_structural_mesh(
            _tetra10_mesh(include_ai5=False),
            coordinate_units="m",
            materials=_material(),
        )


def test_calculix_case_has_c3d10_loads_groups_and_no_solver_gate(
    tmp_path: pathlib.Path,
) -> None:
    pressure = _mapped_tetra_pressure(tmp_path)
    assert pressure["status"] == "mapped"
    report = generate_calculix_case(
        _tetra10_mesh(),
        pressure,
        tmp_path / "case",
        materials=_material(),
        coordinate_units="m",
        angular_speed_rad_s=100.0,
        rotation_origin_m=(0.0, 0.0, 0.0),
        rotation_axis=(0.0, 0.0, 1.0),
        ccx_executable="ccx-definitely-not-installed-step30",
    )
    assert report["status"] == "case_generated"
    assert report["fem_gate"] == "blocked"
    assert report["fem_gate_reason"] == "ccx_not_available"
    assert report["real_fem_evidence"] is False
    assert report["named_groups"] == ["ROOT_INTERFACE", "WETTED_SURFACE", "AI5_04"]
    deck = pathlib.Path(report["input_path"]).read_text(encoding="ascii")
    for required in (
        "*ELEMENT, TYPE=C3D10",
        "*NSET, NSET=ROOT_INTERFACE",
        "*NSET, NSET=AI5_04",
        "*SURFACE, TYPE=ELEMENT, NAME=WETTED_SURFACE",
        "ALL_SOLID, CENTRIF",
        "*MATERIAL, NAME=STEEL",
        ", P1,",
    ):
        assert required in deck

    fake_result = {
        "result_format": "calculix_json",
        "evidence_status": "real_solver",
        "real_fem_evidence": True,
    }
    blocked = validate_calculix_results(fake_result, report)
    assert blocked["fem_gate"] == "blocked"
    assert "ccx was not available" in blocked["reason"]


def test_centrifugal_only_s1_has_no_pressure_and_requires_375_rpm(
    tmp_path: pathlib.Path,
) -> None:
    report = generate_calculix_case(
        _tetra10_mesh(),
        None,
        tmp_path / "s1",
        materials=_material(),
        coordinate_units="m",
        angular_speed_rad_s=375.0 * 2.0 * np.pi / 60.0,
        rotation_origin_m=(0.0, 0.0, 0.0),
        rotation_axis=(0.0, 0.0, 1.0),
        case_name="s1_centrifugal",
        ccx_executable="ccx-definitely-not-installed-step30",
        load_mode="centrifugal_only_s1",
    )
    assert report["status"] == "case_generated"
    assert report["load_mode"] == "centrifugal_only_s1"
    assert report["pressure_moment_origin_m"] is None
    assert report["rotation_origin_m"] == [0.0, 0.0, 0.0]
    deck = pathlib.Path(report["input_path"]).read_text(encoding="ascii")
    assert "ALL_SOLID, CENTRIF" in deck
    assert not any(
        line[:1].isdigit() and ", P" in line
        for line in deck.splitlines()
    )
    assert "pressure_load_sha256=none" in deck

    blocked = generate_calculix_case(
        _tetra10_mesh(),
        None,
        tmp_path / "bad_s1",
        materials=_material(),
        coordinate_units="m",
        angular_speed_rad_s=300.0 * 2.0 * np.pi / 60.0,
        rotation_origin_m=(0.0, 0.0, 0.0),
        rotation_axis=(0.0, 0.0, 1.0),
        load_mode="centrifugal_only_s1",
    )
    assert blocked["status"] == "blocked"
    assert "375 rpm" in blocked["reason"]


def test_parametric_or_forged_fem_is_ineligible(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FEMResultsError, match="parametric/synthetic"):
        parse_calculix_results(
            {
                "result_format": "calculix_json",
                "evidence_status": "parametric",
                "real_fem_evidence": False,
            }
        )
    fake = {
        "status": "validated",
        "fem_gate": "passed",
        "real_fem_evidence": False,
        "evidence_status": "parametric",
    }
    validation = validate_fem_evidence([fake, fake, fake], {})
    assert validation["status"] == "blocked"
    assert validation["real_fem_evidence"] is False


def test_ai5_tensor_projection_and_back_calibration_rejected(
    tmp_path: pathlib.Path,
) -> None:
    tensor = [[10.0, 2.0, 0.0], [2.0, 30.0, 0.0], [0.0, 0.0, 5.0]]
    assert project_ai5_stress(tensor, [1.0, 1.0, 0.0]) == pytest.approx(22.0)

    output = tmp_path / "fake.dat"
    output.write_text("software fixture only\n", encoding="ascii")
    result: dict[str, Any] = {
        "status": "validated",
        "fem_gate": "passed",
        "real_fem_evidence": True,
        "evidence_status": "real_solver",
        "mesh_sha256": "1" * 64,
        "pressure_load_sha256": "2" * 64,
        "pressure_source_vtp_sha256": "3" * 64,
        "input_sha256": "4" * 64,
        "load_mode": "pressure_plus_centrifugal",
        "output_sha256": sha256_file(output),
        "case_provenance_sha256": "5" * 64,
        "ccx_executable_sha256": "6" * 64,
        "solver_output_path": str(output),
        "mesh_metrics": {"characteristic_size_m": 0.1},
        "ai5_04": {
            "projection_method": "tensor_direction",
            "projected_stress_pa": 22.0,
            "direction_unit": [2**-0.5, 2**-0.5, 0.0],
            "calibrated_from_validation_point": True,
        },
    }
    result["result_validation_sha256"] = _canonical_sha256(result)
    blocked = None
    try:
        validate_three_mesh_convergence([result, result, result])
    except Exception as exc:  # contract raises before any engineering result is released
        blocked = str(exc)
    assert blocked is not None
    assert "back-calibrate" in blocked


def test_three_mesh_convergence_accepts_centrifugal_only_s1(tmp_path: pathlib.Path) -> None:
    results = []
    for index, (size, stress_value) in enumerate(((0.3, 100.0), (0.2, 108.0), (0.1, 112.0)), start=1):
        output = tmp_path / f"centrifugal_{index}.dat"
        output.write_text(f"fixture {index}\n", encoding="ascii")
        result: dict[str, Any] = {
            "status": "validated",
            "fem_gate": "passed",
            "real_fem_evidence": True,
            "evidence_status": "real_solver",
            "mesh_sha256": f"{index}" * 64,
            "pressure_load_sha256": None,
            "pressure_source_vtp_sha256": None,
            "input_sha256": f"{index + 3}" * 64,
            "output_sha256": sha256_file(output),
            "case_provenance_sha256": f"{index + 6}" * 64,
            "ccx_executable_sha256": "a" * 64,
            "ccx_version": "CalculiX 2.19",
            "solver_output_path": str(output),
            "mesh_metrics": {"characteristic_size_m": size},
            "load_mode": "centrifugal_only_s1",
            "ai5_04": {
                "projection_method": "tensor_direction",
                "projected_stress_pa": stress_value,
                "direction_unit": [0.0, 1.0, 0.0],
                "calibrated_from_validation_point": False,
            },
            "result_validation_sha256": "",
        }
        unsigned = dict(result)
        unsigned.pop("result_validation_sha256", None)
        result["result_validation_sha256"] = _canonical_sha256(unsigned)
        results.append(result)
    report = validate_three_mesh_convergence(results)
    assert report["status"] == "passed"
    assert report["pressure_load_sha256"] == [None, None, None]
    assert report["pressure_source_vtp_sha256"] is None
    assert report["refinement_ratios"] == pytest.approx([1.5, 2.0])


def test_protection_software_fixture_respects_root_forbidden_zone() -> None:
    report = classify_protection_zones(
        parent_face_id=np.arange(6),
        erosion_rate_mm_h=[10.0, 9.0, 1.0, 1.0, 1.0, 20.0],
        von_mises_stress_pa=[1.0, 1.0, 5.0, 6.0, 8.0, 20.0],
        fatigue_damage=[0.01, 0.01, 0.2, 0.3, 0.5, 1.0],
        face_area_m2=np.full(6, 0.1),
        face_centers_m=np.asarray([[float(index), 0.0, 0.0] for index in range(6)]),
        adjacency=[[1], [0, 2], [1, 3], [2, 4], [3, 5], [4]],
        root_forbidden_mask=[False, False, False, False, False, True],
        high_erosion_quantile=0.6,
        high_stress_quantile=0.75,
        low_stress_quantile=0.5,
    )
    assert report["status"] == "software_validated"
    assert report["software_validation_only"] is True
    assert report["real_fem_evidence"] is False
    assert report["zone_labels"][5] == "root_forbidden"
    assert report["protected_mask"][5] is False
    assert not any(
        protected and forbidden
        for protected, forbidden in zip(
            report["protected_mask"], report["root_forbidden_mask"]
        )
    )


def test_disconnected_protection_and_fake_fem_are_blocked() -> None:
    disconnected = classify_protection_zones(
        parent_face_id=np.arange(5),
        erosion_rate_mm_h=[10.0, 1.0, 1.0, 1.0, 10.0],
        von_mises_stress_pa=[1.0, 5.0, 5.0, 5.0, 1.0],
        fatigue_damage=[0.0, 1.0, 1.0, 1.0, 0.0],
        face_area_m2=np.ones(5),
        adjacency=[[1], [0, 2], [1, 3], [2, 4], [3]],
        root_forbidden_mask=np.zeros(5, dtype=bool),
        high_erosion_quantile=0.7,
        low_stress_quantile=0.3,
        require_connected=True,
    )
    assert disconnected["status"] == "blocked"
    assert "disconnected" in disconnected["reason"]

    erosion: dict[str, Any] = {
        "status": "measured",
        "evaluation_status": "truth_built",
        "independent": True,
        "erosion_rate_units": "mm/h",
        "parent_face_id": [0],
        "erosion_rate_mm_h": [1.0],
        "topology_sha256": "6" * 64,
    }
    erosion["evidence_sha256"] = _canonical_sha256(erosion)
    forged_fem = {
        "status": "passed",
        "fem_gate": "passed",
        "real_fem_evidence": True,
        "surface_results": {},
    }
    blocked = design_protection(
        erosion,
        forged_fem,
        parent_face_id=[0],
        face_area_m2=[1.0],
        adjacency=[[]],
        root_forbidden_mask=[False],
    )
    assert blocked["status"] == "blocked"
    assert blocked["real_fem_evidence"] is False
    assert "validation_sha256" in blocked["reason"]


def _candidate(
    identifier: str,
    efficiency: float,
    stress: float,
    fatigue: float,
    erosion: float,
) -> dict[str, Any]:
    return {
        "candidate_id": identifier,
        "metrics": {
            "units": {
                "efficiency": "1",
                "stress": "Pa",
                "fatigue_damage": "1",
                "erosion_rate": "mm/h",
            },
            "efficiency": efficiency,
            "stress_pa": stress,
            "fatigue_damage": fatigue,
            "erosion_by_condition": {
                "rated": {"rate_mm_h": erosion, "weight": 0.8},
                "part_load": {"rate_mm_h": erosion * 0.5, "weight": 0.2},
            },
        },
        # Deliberately incomplete: this software fixture must keep the final gate blocked.
        "evidence": {"rerun_for_candidate": False},
    }


def test_pareto_is_deterministic_filters_dominated_and_stays_blocked() -> None:
    candidates = [
        _candidate("A", 0.90, 100.0, 0.10, 1.0),
        _candidate("B", 0.80, 120.0, 0.20, 2.0),
        _candidate("C", 0.92, 150.0, 0.08, 0.8),
    ]
    forward = build_pareto_tradeoff(candidates)
    reverse = build_pareto_tradeoff(list(reversed(candidates)))
    assert forward["status"] == "blocked"
    assert forward["tradeoff_gate"] == "blocked"
    assert forward["nondominated_candidate_ids"] == ["C", "A"]
    assert forward["dominated_by"]["B"] == ["A"]
    assert forward["nondominated_candidate_ids"] == reverse["nondominated_candidate_ids"]
    assert forward["dominated_by"] == reverse["dominated_by"]
    assert forward["tradeoff_sha256"] == reverse["tradeoff_sha256"]
    assert validate_tradeoff(forward)["tradeoff_gate"] == "blocked"
