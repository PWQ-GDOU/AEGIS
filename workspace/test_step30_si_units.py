"""Hermetic unit tests for Step30 SI artifact pairing and unit conversion."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import step30_lagrangian_particle_tracking as tracking
from step30_cfd_topology import load_topology_artifact


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scalar_field(path: Path, location: str, object_name: str, dimensions: str, value: float) -> None:
    path.write_text(
        f'''FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    location "{location}";
    object {object_name};
}}
dimensions {dimensions};
internalField nonuniform List<scalar>
1
(
{value}
)
;
''',
        encoding="utf-8",
    )


def _vector_field(path: Path, location: str, object_name: str, dimensions: str, value: tuple[float, float, float]) -> None:
    path.write_text(
        f'''FoamFile
{{
    version 2.0;
    format ascii;
    class volVectorField;
    location "{location}";
    object {object_name};
}}
dimensions {dimensions};
internalField nonuniform List<vector>
1
(
({value[0]} {value[1]} {value[2]})
)
;
''',
        encoding="utf-8",
    )


def _native_si_fixture(output: Path, field_time: str = "7", statistic: str = "snapshot") -> Path:
    native = output / "native_bucket_si"
    native.mkdir(parents=True)
    velocity_object = "U" if statistic == "snapshot" else "UMean"
    specifications = {
        "Cx": ("scalar", "[0 1 0 0 0 0 0]", 0.25),
        "Cy": ("scalar", "[0 1 0 0 0 0 0]", -0.01),
        "Cz": ("scalar", "[0 1 0 0 0 0 0]", 0.02),
        velocity_object: ("vector", "[0 1 -1 0 0 0 0]", (104.8, 0.0, 0.0)),
    }
    fields = {}
    for name, (kind, dimensions, value) in specifications.items():
        path = native / f"native_si_{field_time}_{name}.foam"
        if kind == "scalar":
            _scalar_field(path, field_time, name, dimensions, value)
        else:
            _vector_field(path, field_time, name, dimensions, value)
        fields[name] = {"file": path.name, "sha256": _sha256(path)}
    semantics = {
        "application": "simpleFoam" if statistic == "snapshot" else "pimpleFoam",
        "mode": "steady_rans" if statistic == "snapshot" else "urans",
        "field_statistic": statistic,
        "tracking_velocity_object": velocity_object,
        "averaging_window": (
            None
            if statistic == "snapshot"
            else {"start_time": "1", "end_time": field_time, "sample_count": 7}
        ),
    }
    manifest = {
        "version": "step30-native-si-manifest-v2",
        "field_time": field_time,
        "solver_semantics": semantics,
        "units": {
            "source_coordinate_units": "m",
            "internal_coordinate_units": "mm",
            "coordinate_scale_to_mm": 1000.0,
        },
        "fields": fields,
        "topology": {"mesh_signature": "synthetic"},
    }
    manifest_path = native / f"native_si_manifest_{field_time}.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_native_si_field_headers_hashes_and_scale(tmp_path: Path) -> None:
    _native_si_fixture(tmp_path)
    field = tracking.load_cfd_velocity_field(tmp_path, "native_si", "7")
    assert np.allclose(field["cell_centers_mm"], [[250.0, -10.0, 20.0]])
    assert np.allclose(field["U_ms"], [[104.8, 0.0, 0.0]])
    assert field["source_coordinate_units"] == "m"
    assert field["coordinate_scale_to_mm"] == 1000.0


def test_native_si_time_mean_selects_umean(tmp_path: Path) -> None:
    _native_si_fixture(tmp_path, statistic="time_mean")
    field = tracking.load_cfd_velocity_field(tmp_path, "native_si", "7")
    assert np.allclose(field["U_ms"], [[104.8, 0.0, 0.0]])
    assert field["artifact_manifest"]["solver_semantics"]["tracking_velocity_object"] == "UMean"


def test_native_si_rejects_renamed_wrong_time_field(tmp_path: Path) -> None:
    manifest_path = _native_si_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    path = tmp_path / "native_bucket_si" / manifest["fields"]["U"]["file"]
    _vector_field(path, "6", "U", "[0 1 -1 0 0 0 0]", (104.8, 0.0, 0.0))
    manifest["fields"]["U"]["sha256"] = _sha256(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="time mismatch"):
        tracking.load_cfd_velocity_field(tmp_path, "native_si", "7")


def test_native_si_rejects_hash_mismatch(tmp_path: Path) -> None:
    manifest_path = _native_si_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["fields"]["U"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        tracking.load_cfd_velocity_field(tmp_path, "native_si", "7")


def test_native_si_rejects_mixed_solver_semantics(tmp_path: Path) -> None:
    manifest_path = _native_si_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["solver_semantics"]["tracking_velocity_object"] = "UMean"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="steady snapshot semantics"):
        tracking.load_cfd_velocity_field(tmp_path, "native_si", "7")


def _topology_npz(path: Path, source_units: str, scale: float) -> None:
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    np.savez_compressed(
        path,
        points_local_mm=points,
        triangles_local_mm=triangles,
        triangle_parent_face_id=np.array([0], dtype=np.int32),
        face_centers_local_mm=np.array([[1 / 3, 1 / 3, 0]], dtype=np.float32),
        face_normals_local=np.array([[0, 0, 1]], dtype=np.float32),
        face_area_m2=np.array([0.5e-6], dtype=np.float32),
        triangle_centers_local_mm=np.array([[1 / 3, 1 / 3, 0]], dtype=np.float32),
        triangle_normals_local=np.array([[0, 0, 1]], dtype=np.float32),
        triangle_radii_mm=np.array([np.sqrt(5.0) / 3.0], dtype=np.float32),
        mesh_signature=np.array("synthetic"),
        patch_name=np.array("bucket"),
        time_name=np.array("7"),
        source_coordinate_units=np.array(source_units),
        internal_coordinate_units=np.array("mm"),
        coordinate_scale_to_mm=np.array(scale),
    )


def test_topology_rejects_invalid_unit_contract(tmp_path: Path) -> None:
    path = tmp_path / "topology.npz"
    _topology_npz(path, "m", 1.0)
    with pytest.raises(ValueError, match="unit contract"):
        load_topology_artifact(path)


def test_staging_maps_foam_to_vtk_series_index_to_physical_time(tmp_path: Path) -> None:
    module_path = (
        Path(__file__).parent
        / "of_case"
        / "pelton_bucket_open_si"
        / "stage_step30_artifacts.py"
    )
    spec = importlib.util.spec_from_file_location("open_si_staging", module_path)
    staging = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(staging)

    case = tmp_path / "case"
    vtp = case / "VTK" / "case_1" / "boundary" / "bucket.vtp"
    vtp.parent.mkdir(parents=True)
    vtp.write_text("physical-time-vtp", encoding="utf-8")
    series = {
        "file-series-version": "1.0",
        "files": [
            {"name": "case_0.vtm", "time": 0.0},
            {"name": "case_1.vtm", "time": 0.1173819136},
        ],
    }
    (case / "VTK" / "case.vtm.series").write_text(json.dumps(series), encoding="utf-8")
    assert staging._bucket_vtp(case, "0.1173819136") == vtp
    series["files"][1]["time"] = 0.11738191
    (case / "VTK" / "case.vtm.series").write_text(json.dumps(series), encoding="utf-8")
    assert staging._bucket_vtp(case, "0.1173819136") == vtp


def test_staging_rejects_self_declared_incomplete_urans_evidence() -> None:
    module_path = (
        Path(__file__).parent
        / "of_case"
        / "pelton_bucket_open_si"
        / "stage_step30_artifacts.py"
    )
    spec = importlib.util.spec_from_file_location("open_si_staging_gates", module_path)
    staging = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(staging)
    window = {"start_time": "1", "end_time": "7", "sample_count": 10}
    validation = {
        "version": "urans-time-mean-validation-v1",
        "field_time": "7",
        "averaging_window": window,
        "gates": {"max_co_below_0p5": True},
        "accepted": True,
    }
    assessment = {
        "version": "steady-rans-fixed-point-assessment-v1",
        "accepted_as_steady_fixed_point": False,
    }
    assert staging._accepted_urans_report(validation, window, "7") is False
    assert staging._rejected_steady_assessment(assessment) is False
    complete = {name: True for name in staging.URANS_VALIDATION_GATES}
    validation.update({"version": "urans-time-mean-validation-v2", "gates": complete})
    assert staging._accepted_urans_report(validation, window, "7") is True


def test_si_case_freezes_authoritative_geometry_and_metre_turbulence_scale() -> None:
    case = Path(__file__).parent / "of_case" / "pelton_bucket_open_si"
    assert "scale 0.001;" in (case / "system" / "blockMeshDict").read_text()
    assert "radius 0.12;" in (case / "system" / "topoSetDict").read_text()
    snappy = (case / "system" / "snappyHexMeshDict").read_text()
    assert "mergePatchFaces false;" in snappy
    assert 'file "single_bucket_target_authoritative_local_m.stl";' in snappy
    assert "level (3 3);" in snappy
    assert "maxBoundarySkewness 4;" in snappy
    assert "minDeterminant 0.01;" in snappy
    omega = (case / "0.orig" / "omega").read_text()
    assert "uniform 697.44025;" in omega
    assert "4118640" not in omega
    run_script = (case / "setup_and_run.sh").read_text()
    assert "181eb659091f187e1286ca27b689d654a83b5522b28d6f39aceaa4ebc3fc38ca" in run_script
    assert "surfaceTransformPoints" not in run_script
    assert "trap restore_dictionaries EXIT INT TERM" in run_script
    assert "--field-statistic snapshot" in run_script
    prepare_script = (case / "prepare_urans_restart.sh").read_text()
    assert "cmp -s" in prepare_script
    assert '[[ ! -e "$TARGET_DIR" ]]' in prepare_script
    assert 'for field in ("U", "k", "omega", "phi")' in prepare_script
    urans_script = (case / "run_urans_time_mean.sh").read_text()
    assert 'AVERAGING_START=0.1209875984' in urans_script
    assert 'AVERAGING_HARD_END=0.4094423824' in urans_script
    assert 'controlDict.urans_extended_burnin' in urans_script
    assert '--minimum-flow-throughs 20' in urans_script
    assert '--consecutive-required 2' in urans_script
    extended = (case / "system" / "controlDict.urans_extended_burnin").read_text()
    assert "startTime 0.0452682176;" in extended
    assert "endTime 0.1209875984;" in extended
    average = (case / "system" / "controlDict.urans_average").read_text()
    assert "startTime 0.1209875984;" in average
    assert "endTime 0.4094423824;" in average
    assert "maxCo 0.499;" in average
    assert "writeControl timeStep;" in average
    assert "writeInterval 100;" in average
    assert "restartOnRestart true;" in average
    watcher = (case / "watch_urans_acceptance.py").read_text()
    assert 'consecutive_required < 2' in watcher
    assert '"stopAt endTime;", "stopAt nextWrite;"' in watcher
    assert "--field-statistic time_mean" in urans_script
    assert "--application pimpleFoam" in urans_script
    assert "--averaging-start \"$AVERAGING_START\"" in urans_script
    assert "--steady-assessment \"$STEADY_ASSESSMENT\"" in urans_script
