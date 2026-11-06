"""Build and validate the manifest that pairs native SI fields and topology."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from step30_cfd_topology import load_topology_artifact
from step30_lagrangian_particle_tracking import parse_foam_header

WORKSPACE = Path(__file__).parent
OUTPUT = WORKSPACE / "output"
NATIVE = OUTPUT / "native_bucket_si"
TOPOLOGY = OUTPUT / "bucket_topology_si.npz"
URANS_VALIDATION_GATES = {
    "max_co_below_0p5", "no_turbulence_bounding", "no_fatal_or_nonfinite_log",
    "twenty_flow_through_window", "mean_field_relative_l2_below_1e3",
    "force_late_mean_change_below_1pct", "flux_late_mean_change_below_1pct",
    "flux_imbalance_below_1e4", "probe_late_mean_change_below_1pct",
    "mean_speed_below_500_ms", "all_mean_fields_finite",
}
STEADY_VALIDATION_GATES = {
    "first_pressure_residual_below_1e4", "all_initial_residuals_below_1e4",
    "all_field_relative_l2_below_1e3", "force_change_below_1pct",
    "flux_imbalance_below_1e4", "no_turbulence_bounding",
}


def _accepted_urans_report(validation: dict, window: dict, field_time: str) -> bool:
    gates = validation.get("gates")
    return (
        validation.get("version") in {
            "urans-time-mean-validation-v1",
            "urans-time-mean-validation-v2",
        }
        and validation.get("averaging_window") == window
        and str(validation.get("field_time")) == field_time
        and validation.get("accepted") is True
        and isinstance(gates, dict)
        and set(gates) == URANS_VALIDATION_GATES
        and all(value is True for value in gates.values())
    )


def _rejected_urans_report(validation: dict, averaging_start: str) -> bool:
    gates = validation.get("gates")
    window = validation.get("averaging_window")
    return (
        validation.get("version") in {
            "urans-time-mean-validation-v1",
            "urans-time-mean-validation-v2",
        }
        and str(validation.get("field_time")) == averaging_start
        and isinstance(window, dict)
        and str(window.get("end_time")) == averaging_start
        and validation.get("accepted") is False
        and isinstance(gates, dict)
        and set(gates) == URANS_VALIDATION_GATES
        and all(isinstance(value, bool) for value in gates.values())
        and not all(gates.values())
    )


def _rejected_steady_assessment(assessment: dict) -> bool:
    gates = assessment.get("gates")
    strict_marker = assessment.get("strict_marker")
    return (
        assessment.get("version") == "steady-rans-fixed-point-assessment-v1"
        and assessment.get("pressure_threshold") == 1e-4
        and isinstance(strict_marker, bool)
        and isinstance(gates, dict)
        and set(gates) == STEADY_VALIDATION_GATES
        and all(isinstance(value, bool) for value in gates.values())
        and gates["first_pressure_residual_below_1e4"] is True
        and not all(gates.values())
        and assessment.get("accepted_as_steady_fixed_point") is (strict_marker and all(gates.values()))
        and assessment.get("accepted_as_steady_fixed_point") is False
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_semantics(semantics: dict) -> tuple[str, dict[str, tuple[str, str]]]:
    required = {
        "application", "mode", "field_statistic",
        "tracking_velocity_object", "averaging_window",
    }
    if set(semantics) != required:
        raise ValueError("source bundle has incomplete solver semantics")
    statistic = semantics["field_statistic"]
    if statistic == "snapshot":
        expected = {
            "application": "simpleFoam",
            "mode": "steady_rans",
            "field_statistic": "snapshot",
            "tracking_velocity_object": "U",
            "averaging_window": None,
        }
        if semantics != expected:
            raise ValueError("snapshot semantics require simpleFoam/steady_rans/U")
        objects = {
            "U": ("volVectorField", "[0 1 -1 0 0 0 0]"),
            "p": ("volScalarField", "[0 2 -2 0 0 0 0]"),
            "k": ("volScalarField", "[0 2 -2 0 0 0 0]"),
            "omega": ("volScalarField", "[0 0 -1 0 0 0 0]"),
            "wallShearStress": ("volVectorField", "[0 2 -2 0 0 0 0]"),
        }
    elif statistic == "time_mean":
        window = semantics["averaging_window"]
        if (
            semantics["application"] != "pimpleFoam"
            or semantics["mode"] != "urans"
            or semantics["tracking_velocity_object"] != "UMean"
            or not isinstance(window, dict)
            or set(window) != {"start_time", "end_time", "sample_count"}
            or int(window["sample_count"]) <= 0
        ):
            raise ValueError("time-mean semantics require pimpleFoam/urans/UMean and a valid averaging window")
        objects = {
            "UMean": ("volVectorField", "[0 1 -1 0 0 0 0]"),
            "pMean": ("volScalarField", "[0 2 -2 0 0 0 0]"),
            "kMean": ("volScalarField", "[0 2 -2 0 0 0 0]"),
            "omegaMean": ("volScalarField", "[0 0 -1 0 0 0 0]"),
            "wallShearStressMean": ("volVectorField", "[0 2 -2 0 0 0 0]"),
        }
    else:
        raise ValueError("field statistic must be snapshot or time_mean")
    return statistic, objects


def build_manifest(
    field_time: str,
    expected_source_case: str | None = None,
    *,
    native_dir: Path = NATIVE,
    topology_path: Path = TOPOLOGY,
    output_path: Path | None = None,
) -> dict:
    source_manifest_path = native_dir / f"source_bundle_manifest_{field_time}.json"
    if not source_manifest_path.exists():
        raise FileNotFoundError(f"missing source bundle manifest: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("version") != "step30-native-si-source-bundle-v2":
        raise ValueError("production manifest requires source bundle v2")
    if str(source_manifest.get("field_time")) != field_time:
        raise ValueError("source bundle time differs from requested time")
    source_case = source_manifest.get("source_case")
    if not source_case:
        raise ValueError("source bundle has no source case")
    if expected_source_case is not None and source_case != expected_source_case:
        raise ValueError("source bundle case differs from expected case")
    if source_manifest.get("units", {}).get("source_coordinate_units") != "m":
        raise ValueError("source bundle is not SI-coordinate data")
    semantics = source_manifest.get("solver_semantics", {})
    statistic, solution_objects = _validate_semantics(semantics)

    topology = load_topology_artifact(topology_path)
    if topology["time_name"] != field_time:
        raise ValueError("SI topology and field time differ")
    if (
        topology["source_coordinate_units"],
        topology["internal_coordinate_units"],
        topology["coordinate_scale_to_mm"],
    ) != ("m", "mm", 1000.0):
        raise ValueError("SI topology has an invalid coordinate unit contract")
    if topology["artifact_version"] != "step30-native-openfoam-topology-v3":
        raise ValueError("production manifest requires topology v3")

    expected = {
        "Cx": ("volScalarField", "[0 1 0 0 0 0 0]"),
        "Cy": ("volScalarField", "[0 1 0 0 0 0 0]"),
        "Cz": ("volScalarField", "[0 1 0 0 0 0 0]"),
        **solution_objects,
    }
    fields = {}
    for object_name, (expected_class, expected_dimensions) in expected.items():
        source_entry = source_manifest.get("fields", {}).get(object_name, {})
        file_name = source_entry.get("file")
        if not file_name or Path(file_name).name != file_name:
            raise ValueError(f"invalid staged field path for {object_name}")
        path = native_dir / file_name
        header = parse_foam_header(path)
        if header != {
            "class": expected_class,
            "location": field_time,
            "object": object_name,
            "dimensions": expected_dimensions,
        }:
            raise ValueError(f"unexpected OpenFOAM header for {path.name}: {header}")
        file_hash = _sha256(path)
        if source_entry.get("sha256") != file_hash:
            raise ValueError(f"staged field differs from source bundle: {path.name}")
        fields[object_name] = {"file": path.name, "sha256": file_hash, "header": header}

    required_sources = {
        "bucket_vtp", "si_stl", "polyMesh_boundary", "controlDict", "fvSolution",
        "fvSchemes", "blockMeshDict", "snappyHexMeshDict", "topoSetDict",
        "transportProperties", "turbulenceProperties", "initial_U", "initial_k",
        "initial_omega", "solver_log", "checkMesh_log",
    }
    if statistic == "time_mean":
        required_sources.update({"urans_validation", "steady_assessment", "extended_burnin_rejection"})
    source_files = {}
    source_entries = source_manifest.get("source_files", {})
    if set(source_entries) != required_sources:
        raise ValueError("source bundle has incomplete or unexpected source files")
    for key, entry in source_entries.items():
        file_name = entry.get("file")
        if not file_name or Path(file_name).name != file_name:
            raise ValueError(f"invalid staged source path for {key}")
        path = native_dir / file_name
        file_hash = _sha256(path)
        if entry.get("sha256") != file_hash:
            raise ValueError(f"staged source file differs from source bundle: {path.name}")
        source_files[key] = {"file": path.name, "sha256": file_hash}
    if statistic == "time_mean":
        validation = json.loads((native_dir / source_files["urans_validation"]["file"]).read_text(encoding="utf-8"))
        if not _accepted_urans_report(validation, semantics["averaging_window"], field_time):
            raise ValueError("staged URANS validation does not accept this mean field")
        assessment = json.loads((native_dir / source_files["steady_assessment"]["file"]).read_text(encoding="utf-8"))
        if not _rejected_steady_assessment(assessment):
            raise ValueError("staged steady assessment does not justify the URANS transition")
        rejection = json.loads((native_dir / source_files["extended_burnin_rejection"]["file"]).read_text(encoding="utf-8"))
        if not _rejected_urans_report(rejection, str(semantics["averaging_window"]["start_time"])):
            raise ValueError("staged extended-burnin rejection does not precede this averaging window")

    previous_time = source_manifest.get("previous_time")
    if previous_time is not None:
        for name, entry in previous_time.get("fields", {}).items():
            path = native_dir / entry["file"]
            if _sha256(path) != entry["sha256"]:
                raise ValueError(f"staged previous-time field hash mismatch: {path.name}")

    vtp_hash = source_files["bucket_vtp"]["sha256"]
    if topology["source_vtp_sha256"] != vtp_hash:
        raise ValueError("topology artifact was not built from the staged bucket VTP")
    stl_entry = source_manifest.get("si_stl", {})
    if (
        stl_entry.get("file") != source_files["si_stl"]["file"]
        or stl_entry.get("sha256") != source_files["si_stl"]["sha256"]
    ):
        raise ValueError("SI STL identity differs between source bundle entries")

    result = {
        "version": "step30-native-si-manifest-v2",
        "source_case": source_case,
        "source_bundle_manifest": {
            "file": source_manifest_path.name,
            "sha256": _sha256(source_manifest_path),
        },
        "solver_semantics": semantics,
        "polyMesh": source_manifest["polyMesh"],
        "si_stl": stl_entry,
        "field_time": field_time,
        "units": {
            "source_coordinate_units": "m",
            "internal_coordinate_units": "mm",
            "coordinate_scale_to_mm": 1000.0,
        },
        "fields": fields,
        "previous_time": previous_time,
        "topology": {
            "file": topology_path.name,
            "sha256": _sha256(topology_path),
            "artifact_version": topology["artifact_version"],
            "mesh_signature_algorithm": topology["mesh_signature_algorithm"],
            "mesh_signature": topology["mesh_signature"],
            "source_vtp_sha256": topology["source_vtp_sha256"],
            "time": topology["time_name"],
            "patch_name": topology["patch_name"],
            "n_parent_faces": topology["n_parent_faces"],
            "surface_area_m2": topology["surface_area_m2"],
        },
        "source_files": source_files,
    }
    output_path = output_path or native_dir / f"native_si_manifest_{field_time}.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to replace an existing production manifest: {output_path}")
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build native SI CFD/topology manifest v2")
    parser.add_argument("--time", required=True, help="OpenFOAM field time")
    parser.add_argument("--expect-source-case", help="optional exact source-case assertion")
    parser.add_argument("--native-dir", type=Path, default=NATIVE)
    parser.add_argument("--topology", type=Path, default=TOPOLOGY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    build_manifest(
        args.time,
        args.expect_source_case,
        native_dir=args.native_dir,
        topology_path=args.topology,
        output_path=args.output,
    )
