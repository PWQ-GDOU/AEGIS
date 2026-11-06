"""Hermetic tests for Step30 SI provenance v3 and immutable comparison v3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from step30_artifact_identity import canonical_sha256, file_sha256
from step30_cfd_topology import _mesh_signature
from step30_si_comparison import run as run_comparison
from step30_si_provenance import (
    STEADY_VALIDATION_GATES,
    URANS_VALIDATION_GATES,
    run as run_provenance,
)


SNAPSHOT_SEMANTICS = {
    "application": "simpleFoam",
    "mode": "steady_rans",
    "field_statistic": "snapshot",
    "tracking_velocity_object": "U",
    "averaging_window": None,
}
MEAN_SEMANTICS = {
    "application": "pimpleFoam",
    "mode": "urans",
    "field_statistic": "time_mean",
    "tracking_velocity_object": "UMean",
    "averaging_window": {"start_time": "1", "end_time": "7", "sample_count": 61},
}


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _scalar_field(
    path: Path,
    location: str,
    object_name: str,
    dimensions: str,
    values: tuple[float, ...],
) -> dict:
    body = "\n".join(str(value) for value in values)
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
{len(values)}
(
{body}
)
;
''',
        encoding="utf-8",
    )
    return {
        "class": "volScalarField",
        "location": location,
        "object": object_name,
        "dimensions": dimensions,
    }


def _vector_field(
    path: Path,
    location: str,
    object_name: str,
    dimensions: str,
    values: tuple[tuple[float, float, float], ...],
    *,
    bucket_values: tuple[tuple[float, float, float], ...] | None = None,
) -> dict:
    body = "\n".join(f"({x} {y} {z})" for x, y, z in values)
    boundary = ""
    if bucket_values is not None:
        bucket_body = "\n".join(f"({x} {y} {z})" for x, y, z in bucket_values)
        boundary = f'''boundaryField
{{
    bucket
    {{
        type calculated;
        value nonuniform List<vector>
        {len(bucket_values)}
        (
{bucket_body}
        )
        ;
    }}
}}
'''
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
{len(values)}
(
{body}
)
;
{boundary}''',
        encoding="utf-8",
    )
    return {
        "class": "volVectorField",
        "location": location,
        "object": object_name,
        "dimensions": dimensions,
    }


def _source_files(native: Path, statistic: str) -> dict:
    source_text = {
        "bucket_vtp": "synthetic bucket VTP\n",
        "si_stl": "solid synthetic\nendsolid synthetic\n",
        "polyMesh_boundary": "synthetic polyMesh boundary\n",
        "controlDict": "application simpleFoam;\n" if statistic == "snapshot" else "application pimpleFoam;\n",
        "fvSolution": "solvers {}\n",
        "fvSchemes": "ddtSchemes {}\n",
        "blockMeshDict": "scale 0.001;\n",
        "snappyHexMeshDict": "castellatedMesh true;\n",
        "topoSetDict": "radius 0.12;\n",
        "transportProperties": "nu 1e-6;\n",
        "turbulenceProperties": "simulationType RAS;\n",
        "initial_U": "boundaryField\n{\njetInlet\n{\nvalue uniform (104.8 0 0);\n}\n}\n",
        "initial_k": "boundaryField\n{\njetInlet\n{\nvalue uniform 41.1864;\n}\n}\n",
        "initial_omega": "boundaryField\n{\njetInlet\n{\nvalue uniform 697.44025;\n}\n}\n",
        "checkMesh_log": (
            "points: 3\nfaces: 1\ncells: 2\nMax skewness = 0.1\nMesh OK\n"
        ),
    }
    if statistic == "snapshot":
        source_text["solver_log"] = """Time = 7
smoothSolver: Solving for Ux, Initial residual = 1e-05, Final residual = 1e-07, No Iterations 1
smoothSolver: Solving for Uy, Initial residual = 2e-05, Final residual = 1e-07, No Iterations 1
smoothSolver: Solving for Uz, Initial residual = 3e-05, Final residual = 1e-07, No Iterations 1
GAMG: Solving for p, Initial residual = 4e-05, Final residual = 1e-07, No Iterations 1
smoothSolver: Solving for omega, Initial residual = 5e-05, Final residual = 1e-07, No Iterations 1
smoothSolver: Solving for k, Initial residual = 6e-05, Final residual = 1e-07, No Iterations 1
continuity errors : sum local = 1e-08, global = 2e-09, cumulative = 3e-09
solution convergence criteria satisfied
"""
    else:
        source_text["solver_log"] = (
            "pimpleFoam URANS sampling completed\n"
            "solution convergence criteria satisfied\n"
        )

    entries = {}
    for name, text in source_text.items():
        path = native / f"{name}.txt"
        path.write_text(text, encoding="utf-8")
        entries[name] = {"file": path.name, "sha256": file_sha256(path)}
    return entries


def _topology(path: Path, source_vtp_sha256: str) -> dict:
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    parent_ids = np.array([0], dtype=np.int32)
    area = np.array([0.5e-6], dtype=np.float32)
    center = np.array([[1.0 / 3.0, 1.0 / 3.0, 0.0]], dtype=np.float32)
    signature = _mesh_signature(points, triangles, parent_ids, "m")
    np.savez_compressed(
        path,
        points_local_mm=points,
        triangles_local_mm=triangles,
        triangle_parent_face_id=parent_ids,
        face_id=np.array([0], dtype=np.int32),
        parent_face_id=np.array([0], dtype=np.int32),
        face_centers_local_mm=center,
        face_normals_local=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        face_area_m2=area,
        triangle_centers_local_mm=center,
        triangle_normals_local=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        triangle_radii_mm=np.array([np.sqrt(5.0) / 3.0], dtype=np.float32),
        mesh_signature=np.array(signature),
        patch_name=np.array("bucket"),
        time_name=np.array("7"),
        source_coordinate_units=np.array("m"),
        internal_coordinate_units=np.array("mm"),
        coordinate_scale_to_mm=np.array(1000.0),
        artifact_version=np.array("step30-native-openfoam-topology-v3"),
        mesh_signature_algorithm=np.array(
            "sha256-vtk-triangulation-source-units-points-triangles-parent-ids-v3"
        ),
        source_vtp_sha256=np.array(source_vtp_sha256),
    )
    return {
        "file": path.name,
        "sha256": file_sha256(path),
        "artifact_version": "step30-native-openfoam-topology-v3",
        "mesh_signature_algorithm": "sha256-vtk-triangulation-source-units-points-triangles-parent-ids-v3",
        "mesh_signature": signature,
        "source_vtp_sha256": source_vtp_sha256,
        "time": "7",
        "patch_name": "bucket",
        "n_parent_faces": 1,
        "surface_area_m2": float(area.sum()),
    }


def _provenance_fixture(root: Path, statistic: str) -> tuple[Path, Path, Path, Path]:
    native = root / "native"
    native.mkdir(parents=True)
    semantics = SNAPSHOT_SEMANTICS if statistic == "snapshot" else MEAN_SEMANTICS
    solution = (
        {
            "U": ("vector", "[0 1 -1 0 0 0 0]", ((100.0, 0.0, 0.0), (50.0, 0.0, 0.0))),
            "p": ("scalar", "[0 2 -2 0 0 0 0]", (10.0, 20.0)),
            "k": ("scalar", "[0 2 -2 0 0 0 0]", (1.0, 2.0)),
            "omega": ("scalar", "[0 0 -1 0 0 0 0]", (100.0, 200.0)),
            "wallShearStress": ("wall", "[0 2 -2 0 0 0 0]", ((2.0, 0.0, 0.0),)),
        }
        if statistic == "snapshot"
        else {
            "UMean": ("vector", "[0 1 -1 0 0 0 0]", ((90.0, 0.0, 0.0), (45.0, 0.0, 0.0))),
            "pMean": ("scalar", "[0 2 -2 0 0 0 0]", (9.0, 18.0)),
            "kMean": ("scalar", "[0 2 -2 0 0 0 0]", (0.9, 1.8)),
            "omegaMean": ("scalar", "[0 0 -1 0 0 0 0]", (90.0, 180.0)),
            "wallShearStressMean": ("wall", "[0 2 -2 0 0 0 0]", ((1.8, 0.0, 0.0),)),
        }
    )
    coordinates = {
        "Cx": ("[0 1 0 0 0 0 0]", (0.1, 0.2)),
        "Cy": ("[0 1 0 0 0 0 0]", (0.0, 0.0)),
        "Cz": ("[0 1 0 0 0 0 0]", (0.0, 0.0)),
    }
    fields = {}
    for name, (dimensions, values) in coordinates.items():
        path = native / f"native_si_7_{name}.foam"
        header = _scalar_field(path, "7", name, dimensions, values)
        fields[name] = {"file": path.name, "sha256": file_sha256(path), "header": header}
    for name, (kind, dimensions, values) in solution.items():
        path = native / f"native_si_7_{name}.foam"
        if kind == "scalar":
            header = _scalar_field(path, "7", name, dimensions, values)
        else:
            bucket = values if kind == "wall" else None
            header = _vector_field(path, "7", name, dimensions, values, bucket_values=bucket)
        fields[name] = {"file": path.name, "sha256": file_sha256(path), "header": header}

    previous_fields = {}
    for name, (kind, dimensions, values) in solution.items():
        if kind == "wall":
            continue
        path = native / f"native_si_6_{name}.foam"
        scaled = tuple(
            tuple(0.9 * component for component in row) if isinstance(row, tuple) else 0.9 * row
            for row in values
        )
        if kind == "vector":
            _vector_field(path, "6", name, dimensions, scaled)
        else:
            _scalar_field(path, "6", name, dimensions, scaled)
        previous_fields[name] = {"file": path.name, "sha256": file_sha256(path)}
    previous = {"field_time": "6", "fields": previous_fields}

    source_files = _source_files(native, statistic)
    if statistic == "time_mean":
        validation_path = native / "urans_validation_7.json"
        _write_json(validation_path, {
            "version": "urans-time-mean-validation-v1",
            "field_time": "7",
            "averaging_window": MEAN_SEMANTICS["averaging_window"],
            "gates": {name: True for name in URANS_VALIDATION_GATES},
            "accepted": True,
        })
        source_files["urans_validation"] = {
            "file": validation_path.name,
            "sha256": file_sha256(validation_path),
        }
        assessment_path = native / "steady_fixed_point_assessment.json"
        steady_gates = {name: True for name in STEADY_VALIDATION_GATES}
        steady_gates["all_field_relative_l2_below_1e3"] = False
        _write_json(assessment_path, {
            "version": "steady-rans-fixed-point-assessment-v1",
            "pressure_threshold": 1e-4,
            "strict_marker": False,
            "gates": steady_gates,
            "accepted_as_steady_fixed_point": False,
        })
        source_files["steady_assessment"] = {
            "file": assessment_path.name,
            "sha256": file_sha256(assessment_path),
        }
        rejection_path = native / "extended_burnin_rejected_gate.json"
        rejection_gates = {name: True for name in URANS_VALIDATION_GATES}
        rejection_gates["mean_field_relative_l2_below_1e3"] = False
        _write_json(rejection_path, {
            "version": "urans-time-mean-validation-v1",
            "field_time": "1",
            "averaging_window": {"start_time": "0", "end_time": "1", "sample_count": 61},
            "gates": rejection_gates,
            "accepted": False,
        })
        source_files["extended_burnin_rejection"] = {
            "file": rejection_path.name,
            "sha256": file_sha256(rejection_path),
        }
    topology_path = root / "topology.npz"
    topology = _topology(topology_path, source_files["bucket_vtp"]["sha256"])
    poly_mesh = {"algorithm": "synthetic-v1", "signature": "poly-synthetic"}
    si_stl = source_files["si_stl"]
    source_bundle = {
        "version": "step30-native-si-source-bundle-v2",
        "source_case": "/synthetic/native-si-case",
        "field_time": "7",
        "solver_semantics": semantics,
        "units": {
            "source_coordinate_units": "m",
            "velocity_units": "m/s",
            "kinematic_pressure_units": "m2/s2",
            "kinematic_wall_shear_stress_units": "m2/s2",
        },
        "polyMesh": poly_mesh,
        "si_stl": si_stl,
        "fields": {
            name: {"file": entry["file"], "sha256": entry["sha256"]}
            for name, entry in fields.items()
        },
        "previous_time": previous,
        "source_files": source_files,
    }
    source_bundle_path = native / "source_bundle_manifest_7.json"
    _write_json(source_bundle_path, source_bundle)
    manifest = {
        "version": "step30-native-si-manifest-v2",
        "source_case": source_bundle["source_case"],
        "source_bundle_manifest": {
            "file": source_bundle_path.name,
            "sha256": file_sha256(source_bundle_path),
        },
        "solver_semantics": semantics,
        "polyMesh": poly_mesh,
        "si_stl": si_stl,
        "field_time": "7",
        "units": {
            "source_coordinate_units": "m",
            "internal_coordinate_units": "mm",
            "coordinate_scale_to_mm": 1000.0,
        },
        "fields": fields,
        "previous_time": previous,
        "topology": topology,
        "source_files": source_files,
    }
    manifest_path = native / "native_si_manifest_7.json"
    _write_json(manifest_path, manifest)
    return manifest_path, native, topology_path, root / "provenance.json"


def test_snapshot_provenance_uses_steady_solver_evidence(tmp_path: Path) -> None:
    manifest, native, topology, output = _provenance_fixture(tmp_path, "snapshot")
    result = run_provenance(
        "7", manifest_path=manifest, native_dir=native, topology_path=topology, output_path=output
    )

    assert result["version"] == "si-unit-provenance-v3"
    assert result["fields"]["field_statistic"] == "snapshot"
    assert result["fields"]["objects"]["velocity"] == "U"
    assert result["fields"]["U_speed_ms"]["max"] == 100.0
    assert result["convergence"]["assessment_type"] == "steady_snapshot_convergence"
    assert result["convergence"]["final_initial_residuals"]["p_first_corrector"] == 4e-5
    assert result["convergence"]["strict_residual_control_met"] is True
    assert result["convergence"]["previous_snapshot_field_change"]["field_relative_l2"]["U"] > 0.0


def test_time_mean_provenance_uses_sampling_not_simple_marker(tmp_path: Path) -> None:
    manifest, native, topology, output = _provenance_fixture(tmp_path, "time_mean")
    validation = tmp_path / "urans_validation.json"
    _write_json(validation, {
        "version": "urans-time-mean-validation-v1",
        "field_time": "7",
        "averaging_window": MEAN_SEMANTICS["averaging_window"],
        "gates": {name: True for name in URANS_VALIDATION_GATES},
        "accepted": True,
    })
    result = run_provenance(
        "7",
        manifest_path=manifest,
        native_dir=native,
        topology_path=topology,
        output_path=output,
        urans_validation_path=validation,
    )

    assert result["fields"]["field_statistic"] == "time_mean"
    assert result["fields"]["objects"]["velocity"] == "UMean"
    assert result["fields"]["UMean_speed_ms"]["max"] == 90.0
    assert result["convergence"]["assessment_type"] == "urans_time_mean_sampling"
    assert result["convergence"]["averaging_window"] == MEAN_SEMANTICS["averaging_window"]
    assert result["convergence"]["simple_convergence_marker_applied"] is False
    assert "strict_residual_control_met" not in result["convergence"]
    change = result["convergence"]["previous_mean_field_change"]
    assert change["field_relative_l2"]["UMean"] > 0.0
    assert result["convergence"]["validation_report"]["gates"]["max_co_below_0p5"] is True


def test_time_mean_provenance_requires_accepted_validation(tmp_path: Path) -> None:
    manifest, native, topology, output = _provenance_fixture(tmp_path, "time_mean")
    validation = tmp_path / "urans_validation.json"
    _write_json(validation, {
        "version": "urans-time-mean-validation-v1",
        "field_time": "7",
        "averaging_window": MEAN_SEMANTICS["averaging_window"],
        "gates": {"max_co_below_0p5": False},
        "accepted": False,
    })
    with pytest.raises(ValueError, match="differs from the staged report"):
        run_provenance(
            "7",
            manifest_path=manifest,
            native_dir=native,
            topology_path=topology,
            output_path=output,
            urans_validation_path=validation,
        )


def _summary(
    artifact_set_id: str,
    particle_run_id: str,
    max_impacts: int,
    semantics: dict,
) -> dict:
    return {
        "version": "step30-cfd-driven-v2",
        "artifact_identity": {
            "artifact_set_id": artifact_set_id,
            "particle_run_id": particle_run_id,
            "layout": "output/step30_runs/<artifact_set_id>/<collision_policy>/<particle_run_id>",
        },
        "cfd": {
            "variant": "native_si",
            "field_time": "7",
            "n_cells": 2,
            "source_coordinate_units": "m",
            "internal_coordinate_units": "mm",
            "coordinate_scale_to_mm": 1000.0,
            "artifact_manifest_sha256": "native-manifest-hash",
            "speed_stats_ms": {"max": 100.0, "median": 75.0},
            "solver_semantics": semantics,
        },
        "geometry": {
            "mesh_signature": "mesh-synthetic",
            "source_time": "7",
            "field_time": "7",
            "n_parent_faces": 1,
            "n_collision_triangles": 1,
            "surface_area_m2": 0.5e-6,
            "bounds_mm": [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        },
        "parameters": {
            "n_parcels": 20,
            "collision_backend": "cfd_topology",
            "collision_policy": "strict_v2",
            "cfd_variant": "native_si",
            "max_impacts": max_impacts,
        },
        "tracking": {
            "n_impacts": 4,
            "hit_rate_parcels_pct": 20.0,
            "intersection_hits": 4,
            "fallback_hits": 0,
            "physical_strict_hit_ratio": 1.0,
            "impact_mass_kg": 0.01,
        },
        "face_erosion": {"total_target_mass_kg": 1e-6, "dhdt_max_mm_h": 2e-5},
        "validation": {
            "registration_status": "passed",
            "cfd_ER_vs_analytic_Mansouri": {"spearman_rho": 0.5, "top10_iou": 0.25},
        },
        "units": {
            "source_coordinate_units": "m",
            "internal_coordinate_units": "mm",
            "coordinate_scale_to_mm": 1000.0,
        },
    }


def _comparison_fixture(root: Path) -> dict:
    semantics = dict(SNAPSHOT_SEMANTICS)
    artifact = {
        "version": "step30-artifact-set-descriptor-v2",
        "cfd": {
            "variant": "native_si",
            "field_time": "7",
            "manifest_version": "step30-native-si-manifest-v2",
            "manifest_sha256": "native-manifest-hash",
            "polyMesh_signature": "poly-synthetic",
            "field_sha256": {"U": "u-hash"},
            "solver_semantics": semantics,
        },
        "topology": {
            "file_sha256": "topology-hash",
            "artifact_version": "step30-native-openfoam-topology-v3",
            "mesh_signature_algorithm": "sha256-v3",
            "mesh_signature": "mesh-synthetic",
            "source_vtp_sha256": "vtp-hash",
            "patch_name": "bucket",
            "time": "7",
            "n_parent_faces": 1,
        },
        "units": {
            "source_coordinate_units": "m",
            "internal_coordinate_units": "mm",
            "coordinate_scale_to_mm": 1000.0,
        },
    }
    artifact["artifact_set_id"] = canonical_sha256(artifact)
    artifact_root = root / "step30_runs" / artifact["artifact_set_id"]
    artifact_root.mkdir(parents=True)
    _write_json(artifact_root / "artifact_set_manifest.json", artifact)

    paths = {}
    for max_impacts, label in ((1, "i1"), (3, "i3")):
        descriptor = {
            "version": "step30-particle-run-descriptor-v2",
            "artifact_set_id": artifact["artifact_set_id"],
            "collision_backend": "cfd_topology",
            "implementation_hashes": {"particle_tracking": "code-hash"},
            "input_hashes": {"jet_rotation": "input-hash"},
            "config": {
                "cfd_variant": "native_si",
                "collision_policy": "strict_v2",
                "cfd_field_time": "7",
                "n_parcels": 20,
                "max_impacts": max_impacts,
            },
        }
        descriptor["particle_run_id"] = canonical_sha256(descriptor)
        run_dir = artifact_root / "strict_v2" / descriptor["particle_run_id"]
        run_dir.mkdir(parents=True)
        summary_path = run_dir / "summary.json"
        hits_path = run_dir / "particle_hits.npz"
        validation_path = run_dir / "validation.png"
        _write_json(
            summary_path,
            _summary(artifact["artifact_set_id"], descriptor["particle_run_id"], max_impacts, semantics),
        )
        parcel_id = np.array([0, 1] if max_impacts == 1 else [0, 1, 0, 1], dtype=np.int32)
        impact_index = np.array([0, 0] if max_impacts == 1 else [0, 0, 1, 1], dtype=np.int16)
        parent_face_id = np.array([10, 20] if max_impacts == 1 else [10, 20, 30, 40], dtype=np.int64)
        triangle_id = np.array([100, 200] if max_impacts == 1 else [100, 200, 300, 400], dtype=np.int64)
        count = len(parcel_id)
        np.savez_compressed(
            hits_path,
            parcel_id=parcel_id,
            impact_index=impact_index,
            parent_face_id=parent_face_id,
            collision_triangle_id=triangle_id,
            time_s=np.array([0.01, 0.02, 0.03, 0.04][:count]),
            position_local_mm=np.array(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]][:count]
            ),
            velocity_ms=np.array(
                [[40.0, 0.0, 0.0], [41.0, 0.0, 0.0], [30.0, 1.0, 0.0], [31.0, 1.0, 0.0]][:count]
            ),
            Vp_ms=np.array([40.0, 41.0, 30.0, 31.0][:count]),
            alpha_deg=np.array([5.0, 6.0, 7.0, 8.0][:count]),
        )
        validation_path.write_bytes(f"png-{label}".encode("ascii"))
        run_manifest = {
            **descriptor,
            "outputs": {
                "particle_hits": {"file": hits_path.name, "sha256": file_sha256(hits_path)},
                "summary": {"file": summary_path.name, "sha256": file_sha256(summary_path)},
                "validation": {"file": validation_path.name, "sha256": file_sha256(validation_path)},
            },
        }
        run_manifest_path = run_dir / "particle_run_manifest.json"
        _write_json(run_manifest_path, run_manifest)
        paths[label] = run_manifest_path

    provenance = {
        "version": "si-unit-provenance-v3",
        "source_time": "7",
        "solver_semantics": semantics,
        "artifact_manifest": {
            "version": "step30-native-si-manifest-v2",
            "sha256": "native-manifest-hash",
        },
        "topology": {
            "artifact_version": "step30-native-openfoam-topology-v3",
            "artifact_sha256": "topology-hash",
            "mesh_signature": "mesh-synthetic",
        },
        "convergence": {
            "assessment_type": "steady_snapshot_convergence",
            "strict_residual_control_met": True,
            "previous_snapshot_field_change": {"field_relative_l2": {"U": 0.01}},
        },
    }
    provenance_path = root / "provenance.json"
    _write_json(provenance_path, provenance)
    paths.update({
        "artifact": artifact_root / "artifact_set_manifest.json",
        "provenance": provenance_path,
        "output": root / "comparison.json",
    })
    return paths


def _rewrite_summary_and_hash(run_manifest_path: Path, mutate) -> None:
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    summary_path = run_manifest_path.parent / run_manifest["outputs"]["summary"]["file"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mutate(summary)
    _write_json(summary_path, summary)
    run_manifest["outputs"]["summary"]["sha256"] = file_sha256(summary_path)
    _write_json(run_manifest_path, run_manifest)


def _rewrite_hits_and_hash(run_manifest_path: Path, mutate) -> None:
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    hits_path = run_manifest_path.parent / run_manifest["outputs"]["particle_hits"]["file"]
    with np.load(hits_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    mutate(arrays)
    np.savez_compressed(hits_path, **arrays)
    run_manifest["outputs"]["particle_hits"]["sha256"] = file_sha256(hits_path)
    _write_json(run_manifest_path, run_manifest)


def test_valid_immutable_i1_i3_comparison(tmp_path: Path) -> None:
    paths = _comparison_fixture(tmp_path)
    result = run_comparison(paths["i1"], paths["i3"], paths["provenance"], paths["output"])

    assert result["version"] == "step30-native-si-comparison-v3"
    assert result["accepted"] is True
    assert result["artifact_identity"]["artifact_set_manifest_verified"] is True
    assert result["quality_gates"]["zero_fallback_hits"] is True
    assert result["quality_gates"]["physical_strict_hit_ratio_is_one"] is True
    assert result["quality_gates"]["i1_i3_first_impacts_identical"] is True
    assert result["first_impact_match"]["parent_face_agreement"] == 1.0
    assert result["quality_gates"]["solver_evidence"]["assessment_type"] == "steady_snapshot_convergence"
    assert "fallback_warning" not in result["quality_gates"]


def test_comparison_rejects_mismatched_first_impacts(tmp_path: Path) -> None:
    paths = _comparison_fixture(tmp_path)

    def change_first_parent(arrays: dict[str, np.ndarray]) -> None:
        arrays["parent_face_id"] = arrays["parent_face_id"].copy()
        arrays["parent_face_id"][0] = 999

    _rewrite_hits_and_hash(paths["i3"], change_first_parent)
    with pytest.raises(ValueError, match="first impacts differ for parent_face_id"):
        run_comparison(paths["i1"], paths["i3"], paths["provenance"], paths["output"])


def test_comparison_rejects_output_hash_mismatch(tmp_path: Path) -> None:
    paths = _comparison_fixture(tmp_path)
    manifest = json.loads(paths["i3"].read_text(encoding="utf-8"))
    output = paths["i3"].parent / manifest["outputs"]["particle_hits"]["file"]
    output.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="output hash mismatch"):
        run_comparison(paths["i1"], paths["i3"], paths["provenance"], paths["output"])


def test_comparison_rejects_failed_geometry_registration(tmp_path: Path) -> None:
    paths = _comparison_fixture(tmp_path)

    def fail_registration(summary: dict) -> None:
        summary["validation"]["registration_status"] = "failed"

    _rewrite_summary_and_hash(paths["i3"], fail_registration)
    with pytest.raises(ValueError, match="geometry registration must pass"):
        run_comparison(paths["i1"], paths["i3"], paths["provenance"], paths["output"])


def test_comparison_rejects_collision_event_limit_termination(tmp_path: Path) -> None:
    paths = _comparison_fixture(tmp_path)

    def add_event_limit(summary: dict) -> None:
        summary["tracking"]["collision_event_limit_count"] = 1

    _rewrite_summary_and_hash(paths["i3"], add_event_limit)
    with pytest.raises(ValueError, match="zero collision-event-limit"):
        run_comparison(paths["i1"], paths["i3"], paths["provenance"], paths["output"])


def test_comparison_rejects_missing_artifact_set_manifest(tmp_path: Path) -> None:
    paths = _comparison_fixture(tmp_path)
    paths["artifact"].unlink()
    with pytest.raises(FileNotFoundError, match="artifact-set manifest is missing"):
        run_comparison(paths["i1"], paths["i3"], paths["provenance"], paths["output"])


def test_comparison_rejects_speed_at_limit(tmp_path: Path) -> None:
    paths = _comparison_fixture(tmp_path)

    def set_limit_speed(summary: dict) -> None:
        summary["cfd"]["speed_stats_ms"]["max"] = 500.0

    _rewrite_summary_and_hash(paths["i3"], set_limit_speed)
    with pytest.raises(ValueError, match="speed must remain below 500"):
        run_comparison(paths["i1"], paths["i3"], paths["provenance"], paths["output"])


def test_comparison_rejects_failed_steady_solver_evidence(tmp_path: Path) -> None:
    paths = _comparison_fixture(tmp_path)
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    provenance["convergence"]["strict_residual_control_met"] = False
    _write_json(paths["provenance"], provenance)
    with pytest.raises(ValueError, match="rejected solver evidence"):
        run_comparison(paths["i1"], paths["i3"], paths["provenance"], paths["output"])


@pytest.mark.parametrize("mismatch", ["policy", "artifact", "solver"])
def test_comparison_rejects_mixed_policy_artifact_or_solver(tmp_path: Path, mismatch: str) -> None:
    paths = _comparison_fixture(tmp_path)
    if mismatch == "solver":
        def change_solver(summary: dict) -> None:
            summary["cfd"]["solver_semantics"] = dict(MEAN_SEMANTICS)

        _rewrite_summary_and_hash(paths["i3"], change_solver)
        match = "solver semantics"
    else:
        run_manifest = json.loads(paths["i3"].read_text(encoding="utf-8"))
        if mismatch == "policy":
            run_manifest["config"]["collision_policy"] = "legacy_proximity_v1"
            match = "strict_v2"
        else:
            run_manifest["artifact_set_id"] = "different-artifact"
            match = "artifact_set_id"
        unhashed = dict(run_manifest)
        outputs = unhashed.pop("outputs")
        unhashed.pop("particle_run_id")
        run_manifest["particle_run_id"] = canonical_sha256(unhashed)
        run_manifest["outputs"] = outputs
        _write_json(paths["i3"], run_manifest)

    with pytest.raises(ValueError, match=match):
        run_comparison(paths["i1"], paths["i3"], paths["provenance"], paths["output"])
