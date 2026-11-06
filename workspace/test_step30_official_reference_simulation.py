"""Hermetic tests for authenticated contest-reference simulation evidence."""
from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import vtk

from run_step30_acceptance_gates import aggregate
from step30_cfd_topology import build_topology_from_vtp, load_topology_artifact
from step30_official_reference_intake import build_intake, normalize_records, preflight_intake
from step30_official_reference_prediction import build_prediction
from step30_official_reference_simulation import validate_official_reference
from step30_validation_common import canonical_sha256, sha256_file


def _write_patch_vtp(path: Path) -> None:
    points = vtk.vtkPoints()
    for point in (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (2.0, 1.0, 0.0),
    ):
        points.InsertNextPoint(point)
    polygons = vtk.vtkCellArray()
    quad = vtk.vtkPolygon()
    quad.GetPointIds().SetNumberOfIds(4)
    for index, point_id in enumerate((0, 1, 2, 3)):
        quad.GetPointIds().SetId(index, point_id)
    polygons.InsertNextCell(quad)
    triangle = vtk.vtkTriangle()
    for index, point_id in enumerate((4, 5, 6)):
        triangle.GetPointIds().SetId(index, point_id)
    polygons.InsertNextCell(triangle)
    patch = vtk.vtkPolyData()
    patch.SetPoints(points)
    patch.SetPolys(polygons)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(patch)
    assert writer.Write() == 1


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    _write_json(path, manifest)


def _package(
    root: Path,
    prediction_rates: tuple[float, float] = (1.0, 2.0),
) -> dict[str, Any]:
    source = root / "bucket.vtp"
    topology_path = root / "topology.npz"
    _write_patch_vtp(source)
    build_topology_from_vtp(
        source,
        patch_name="bucket",
        time_name="0.8684190921",
        units="m",
        output_npz=topology_path,
        output_json=root / "topology.json",
    )
    topology = load_topology_artifact(topology_path)
    semantics = {
        "application": "pimpleFoam",
        "mode": "urans",
        "field_statistic": "time_mean",
        "tracking_velocity_object": "UMean",
        "averaging_window": {
            "start_time": "0.7",
            "end_time": "0.8684190921",
            "sample_count": 12,
        },
    }
    units = {
        "source_coordinate_units": "m",
        "internal_coordinate_units": "mm",
        "coordinate_scale_to_mm": 1000.0,
    }
    native = {
        "version": "step30-artifact-set-descriptor-v2",
        "cfd": {
            "variant": "native_si",
            "field_time": "0.8684190921",
            "manifest_version": "step30-native-si-manifest-v2",
            "manifest_sha256": "a" * 64,
            "polyMesh_signature": "b" * 64,
            "field_sha256": {"UMean": "c" * 64},
            "solver_semantics": semantics,
        },
        "topology": {
            "file_sha256": sha256_file(topology_path),
            "artifact_version": "step30-native-openfoam-topology-v3",
            "mesh_signature_algorithm": "sha256-vtk-triangulation-source-units-points-triangles-parent-ids-v3",
            "mesh_signature": topology["mesh_signature"],
            "source_vtp_sha256": topology["source_vtp_sha256"],
            "patch_name": "bucket",
            "time": "0.8684190921",
            "n_parent_faces": topology["n_parent_faces"],
        },
        "units": units,
    }
    native["artifact_set_id"] = canonical_sha256(native)
    native_path = root / "artifact_set_manifest.json"
    _write_json(native_path, native)
    (root / "official-description.txt").write_text(
        "Organizer reference simulation fixture definition", encoding="utf-8"
    )
    (root / "official-evaluation-definition.txt").write_text(
        "Area-weighted wear-depth integral and spatial hotspot metrics", encoding="utf-8"
    )
    (root / "raw-reference.csv").write_text(
        "parent_face_id,wear_um,unit\n0,1000,um\n1,2000,um\n",
        encoding="utf-8",
    )
    (root / "independent-split.csv").write_text(
        "parent_face_id\n0\n1\n", encoding="utf-8"
    )
    spec = {
        "version": "step30-official-reference-intake-spec-v1",
        "source_root": str(root),
        "contest": {
            "competition": "Step30 fixture",
            "organizer": "fixture organizer",
            "dataset_id": "fixture-reference",
            "dataset_version": "v2",
        },
        "official_description": "official-description.txt",
        "official_evaluation_definition": "official-evaluation-definition.txt",
        "native_artifact_set_manifest": native_path.name,
        "topology_v3": topology_path.name,
        "reference_source": {
            "path": "raw-reference.csv",
            "adapter": {
                "format": "csv",
                "parent_face_id_field": "parent_face_id",
                "value_field": "wear_um",
                "canonical_value_field": "wear_depth_mm",
                "unit_field": "unit",
                "source_units": "um",
                "canonical_units": "mm",
                "value_metric": "wear_depth",
            },
        },
        "independent_split": {
            "provider": "organizer_provided",
            "is_independent": True,
            "split_id": "organizer-independent-v2",
            "path": "independent-split.csv",
            "format": "csv",
            "parent_face_id_field": "parent_face_id",
        },
        "record_contract": {
            "parent_face_id_field": "parent_face_id",
            "value_field": "wear_depth_mm",
            "value_metric": "wear_depth",
            "value_units": "mm",
            "coverage": "full_parent_faces",
            "area_source": "topology_parent_face_area_m2",
        },
        "evaluation_contract": {
            "version": "step30-reference-evaluation-contract-v1",
            "definition_sha256": sha256_file(root / "official-evaluation-definition.txt"),
            "metric_name": "wear_depth_mm",
            "value_metric": "wear_depth",
            "canonical_units": "mm",
            "aggregation": "area_weighted_integral",
            "relative_error_denominator": "absolute_reference",
            "zero_reference_policy": "block",
            "hotspot": {
                "mode": "weighted_percentile",
                "percentile": 90.0,
                "area_source": "topology_parent_face_area_m2",
            },
            "top_k_tie_break": "parent_face_id_ascending",
        },
        "reference_solver_semantics": {
            "application": "organizerSolver",
            "mode": "organizer_reference",
            "field_statistic": "cumulative",
            "value_object": "wear_depth_mm",
        },
        "thresholds": {
            "total_quantity_relative_error": 0.05,
            "maximum_value_relative_error": 0.05,
            "hotspot_area_relative_error": 0.05,
            "hotspot_iou_error": 0.05,
            "top_k": [{"k": 1, "minimum_overlap": 1.0}],
        },
    }
    spec_path = root / "intake-spec.json"
    _write_json(spec_path, spec)
    assert preflight_intake(spec_path)["status"] == "passed"
    intake = build_intake(spec_path, root / "intake")
    assert intake["status"] == "passed"
    manifest_path = Path(intake["manifest"]["path"])
    observation_path = Path(intake["records"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    normalized_observations = json.loads(observation_path.read_text(encoding="utf-8"))

    # Build a frozen particle run and export predictions through the official exporter.
    # This replaces the former hand-authored predictions.json and satisfies the full
    # validate_prediction_document / build_prediction recomputation contract.
    artifact_set_id = native["artifact_set_id"]
    mesh_signature = topology["mesh_signature"]
    run_desc: dict[str, Any] = {
        "version": "step30-particle-run-descriptor-v2",
        "artifact_set_id": artifact_set_id,
        "collision_backend": "cfd_topology",
        "config": {"collision_policy": "strict_v2"},
    }
    run_desc["particle_run_id"] = canonical_sha256(run_desc)
    hits_path = root / "particle_hits.npz"
    dhdt = np.array(list(prediction_rates), dtype=np.float64)
    np.savez_compressed(
        str(hits_path),
        artifact_set_id=np.array(artifact_set_id),
        particle_run_id=np.array(run_desc["particle_run_id"]),
        mesh_signature=np.array(mesh_signature),
        source_coordinate_units=np.array("m"),
        internal_coordinate_units=np.array("mm"),
        coordinate_scale_to_mm=np.array(1000.0),
        dhdt_face_mm_h=dhdt,
    )
    hits_sha256 = sha256_file(hits_path)
    run_desc["outputs"] = {
        "particle_hits": {"file": "particle_hits.npz", "sha256": hits_sha256}
    }
    run_manifest_path = root / "run_manifest.json"
    _write_json(run_manifest_path, run_desc)
    prediction_path = root / "predictions.json"
    prediction_doc = build_prediction(
        run_manifest_path, manifest_path, exposure_hours=1.0
    )
    _write_json(prediction_path, prediction_doc)

    return {
        "spec": spec,
        "spec_path": spec_path,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "prediction_path": prediction_path,
        "observation_path": observation_path,
        "topology_path": topology_path,
        "native_path": native_path,
        "observations": normalized_observations,
    }


def _numerical_passes() -> dict[str, dict[str, str]]:
    return {
        name: {"status": "passed"}
        for name in ("artifact", "collision", "cfd", "sensitivity", "comparison")
    }


def test_csv_adapter_converts_explicit_units(tmp_path: Path) -> None:
    source = tmp_path / "reference.csv"
    source.write_text(
        "parent_face_id,wear_um,unit\n0,1000,um\n1,2500,um\n",
        encoding="utf-8",
    )
    result = normalize_records(
        source,
        {
            "format": "csv",
            "parent_face_id_field": "parent_face_id",
            "value_field": "wear_um",
            "canonical_value_field": "wear_depth_mm",
            "unit_field": "unit",
            "source_units": "um",
            "canonical_units": "mm",
            "value_metric": "wear_depth",
        },
    )
    assert [row["wear_depth_mm"] for row in result["records"]] == [1.0, 2.5]
    assert result["normalization"]["inferred_fields"] is False
    assert result["records_sha256"]


def test_json_adapter_requires_declared_columns(tmp_path: Path) -> None:
    source = tmp_path / "reference.json"
    _write_json(source, {"records": [{"record_id": 0, "value": 1.0}]})
    with pytest.raises(ValueError, match="missing declared columns"):
        normalize_records(
            source,
            {
                "format": "json",
                "records_key": "records",
                "parent_face_id_field": "parent_face_id",
                "value_field": "wear_depth_mm",
                "canonical_value_field": "wear_depth_mm",
                "source_units": "mm",
                "canonical_units": "mm",
                "value_metric": "wear_depth",
            },
        )


def test_adapter_rejects_duplicate_parent_face_ids(tmp_path: Path) -> None:
    source = tmp_path / "reference.csv"
    source.write_text("parent_face_id,value\n0,1\n0,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate parent_face_id"):
        normalize_records(
            source,
            {
                "format": "csv",
                "parent_face_id_field": "parent_face_id",
                "value_field": "value",
                "canonical_value_field": "wear_depth_mm",
                "source_units": "mm",
                "canonical_units": "mm",
                "value_metric": "wear_depth",
            },
        )


def test_preflight_blocks_missing_official_evaluation_definition(tmp_path: Path) -> None:
    package = _package(tmp_path)
    Path(tmp_path / "official-evaluation-definition.txt").unlink()
    report = preflight_intake(package["spec_path"])
    assert report["status"] == "blocked"
    assert report["official_reference_gate"] is False


def test_preflight_rejects_path_escape(tmp_path: Path) -> None:
    package = _package(tmp_path)
    spec = deepcopy(package["spec"])
    spec["official_description"] = "../outside.txt"
    escaped = tmp_path / "escaped-spec.json"
    _write_json(escaped, spec)
    report = preflight_intake(escaped)
    assert report["status"] == "blocked"
    assert "inside source_root" in report["reason"]


def test_complete_reference_requires_out_of_band_manifest_pin(tmp_path: Path) -> None:
    package = _package(tmp_path)
    report = validate_official_reference(
        package["prediction_path"], package["observation_path"], package["manifest_path"]
    )
    assert report["status"] == "passed"
    assert report["comparison"]["top_k"][0]["overlap"] == 1.0

    unpinned = aggregate(
        **_numerical_passes(),
        measurement=None,
        reference_simulation=report,
        production=True,
    )
    assert unpinned["status"] == "blocked"
    assert unpinned["official_reference_gate"] is False

    pinned_report = validate_official_reference(
        package["prediction_path"],
        package["observation_path"],
        package["manifest_path"],
        trusted_manifest_sha256=package["manifest"]["manifest_sha256"],
    )
    assert pinned_report["official_reference_gate"] is True
    accepted = aggregate(
        **_numerical_passes(),
        measurement=None,
        reference_simulation=pinned_report,
        trusted_manifest_sha256=package["manifest"]["manifest_sha256"],
        production=True,
    )
    assert accepted["status"] == "passed"
    assert accepted["official_reference_gate"] is True
    unsigned = dict(accepted)
    assert unsigned.pop("acceptance_sha256") == canonical_sha256(unsigned)


def test_observations_must_match_hash_bound_artifact(tmp_path: Path) -> None:
    package = _package(tmp_path)
    substituted = deepcopy(package["observations"])
    substituted["records"][0]["wear_depth_mm"] = 9.0
    result = validate_official_reference(
        package["prediction_path"], substituted, package["manifest_path"]
    )
    assert result["status"] == "blocked"
    assert "hash-bound reference artifact" in result["reason"]

    _write_json(package["observation_path"], substituted)
    result = validate_official_reference(
        package["prediction_path"], package["observation_path"], package["manifest_path"]
    )
    assert result["status"] == "blocked"
    assert "hash mismatch" in result["reason"]


@pytest.mark.parametrize(
    "case",
    [
        "artifact_set_id",
        "dataset_version",
        "solver_semantics",
        "units",
        "topology_count",
        "independent_split",
        "evaluation_contract",
        "record_provenance",
        "split_hash",
        "loose_threshold",
        "path_escape",
    ],
)
def test_manifest_contract_mismatch_blocks(tmp_path: Path, case: str) -> None:
    package = _package(tmp_path)
    manifest = deepcopy(package["manifest"])
    if case == "artifact_set_id":
        manifest["artifact_set_id"] = "0" * 64
    elif case == "dataset_version":
        del manifest["contest"]["dataset_version"]
    elif case == "solver_semantics":
        manifest["solver_semantics"]["tracking_velocity_object"] = "U"
    elif case == "units":
        manifest["units"]["internal_coordinate_units"] = "m"
    elif case == "topology_count":
        manifest["topology"]["n_parent_faces"] += 1
    elif case == "independent_split":
        manifest["independent_split"]["provider"] = "local_fixture"
    elif case == "evaluation_contract":
        manifest["evaluation_contract"]["aggregation"] = "explicit_sum"
    elif case == "record_provenance":
        manifest["record_provenance"]["inferred_fields"] = True
    elif case == "split_hash":
        manifest["independent_split"]["split_artifact_sha256"] = "0" * 64
    elif case == "loose_threshold":
        manifest["thresholds"]["maximum_value_relative_error"] = 0.06
    elif case == "path_escape":
        manifest["artifacts"][0]["path"] = "../artifact_set_manifest.json"
        manifest["native_artifact"]["path"] = "../artifact_set_manifest.json"
    _write_manifest(package["manifest_path"], manifest)
    result = validate_official_reference(
        package["prediction_path"], package["observation_path"], package["manifest_path"]
    )
    assert result["status"] == "blocked"
    assert result["official_reference_gate"] is False


@pytest.mark.parametrize(
    "records",
    [
        [
            {"parent_face_id": 0, "wear_depth_mm": 1.0},
            {"parent_face_id": 0, "wear_depth_mm": 2.0},
        ],
        [
            {"parent_face_id": 0, "wear_depth_mm": 1.0},
            {"parent_face_id": 2, "wear_depth_mm": 2.0},
        ],
        [{"parent_face_id": 0, "wear_depth_mm": 1.0}],
        [
            {"parent_face_id": 0, "wear_depth_mm": -1.0},
            {"parent_face_id": 1, "wear_depth_mm": 2.0},
        ],
        [
            {"parent_face_id": 0, "wear_depth_mm": 1.0, "value_units": "kg"},
            {"parent_face_id": 1, "wear_depth_mm": 2.0},
        ],
    ],
)
def test_parent_face_record_contract_blocks_invalid_predictions(
    tmp_path: Path, records: list[dict[str, Any]]
) -> None:
    package = _package(tmp_path)
    result = validate_official_reference(
        {"records": records}, package["observation_path"], package["manifest_path"]
    )
    assert result["status"] == "blocked"
    assert result["official_reference_gate"] is False


def test_valid_evidence_over_threshold_is_failed_not_blocked(tmp_path: Path) -> None:
    package = _package(tmp_path, prediction_rates=(1.0, 2.2))
    report = validate_official_reference(
        package["prediction_path"], package["observation_path"], package["manifest_path"]
    )
    assert report["status"] == "failed"
    assert report["official_reference"] is True
    assert report["official_reference_gate"] is False
    assert report["comparison"]["threshold_checks"]["maximum_value_relative_error"] is False


def test_tampered_report_is_recomputed_before_acceptance(tmp_path: Path) -> None:
    package = _package(tmp_path)
    report = validate_official_reference(
        package["prediction_path"], package["observation_path"], package["manifest_path"]
    )
    report["comparison"]["mae"] = 123.0
    report.pop("report_sha256")
    report["report_sha256"] = canonical_sha256(report)
    result = aggregate(
        **_numerical_passes(),
        measurement=None,
        reference_simulation=report,
        trusted_manifest_sha256=package["manifest"]["manifest_sha256"],
        production=True,
    )
    assert result["status"] == "blocked"
    assert result["official_reference_gate"] is False


def test_acceptance_cli_writes_blocked_report_for_malformed_json(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.json"
    output = tmp_path / "acceptance.json"
    malformed.write_text("{not-json", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("run_step30_acceptance_gates.py")),
            "--reference-simulation",
            str(malformed),
            "--output",
            str(output),
            "--production",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 3
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert report["official_reference_gate"] is False
