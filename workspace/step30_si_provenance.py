"""Write fail-closed provenance and SI checks for a native artifact manifest."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from step30_artifact_identity import file_sha256
from step30_cfd_topology import load_topology_artifact
from step30_lagrangian_particle_tracking import (
    parse_foam_boundary_vector_field,
    parse_foam_header,
    parse_foam_scalar_field,
    parse_foam_vector_field,
)

WORKSPACE = Path(__file__).parent
OUTPUT = WORKSPACE / "output"
NATIVE = OUTPUT / "native_bucket_si"
TOPOLOGY = OUTPUT / "bucket_topology_si.npz"
OUT_JSON = OUTPUT / "si_unit_provenance.json"

MANIFEST_VERSION = "step30-native-si-manifest-v2"
TOPOLOGY_VERSION = "step30-native-openfoam-topology-v3"
PROVENANCE_VERSION = "si-unit-provenance-v3"
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


_SEMANTIC_KEYS = {
    "application",
    "mode",
    "field_statistic",
    "tracking_velocity_object",
    "averaging_window",
}
_SOURCE_FILE_KEYS = {
    "bucket_vtp",
    "si_stl",
    "polyMesh_boundary",
    "controlDict",
    "fvSolution",
    "fvSchemes",
    "blockMeshDict",
    "snappyHexMeshDict",
    "topoSetDict",
    "transportProperties",
    "turbulenceProperties",
    "initial_U",
    "initial_k",
    "initial_omega",
    "solver_log",
    "checkMesh_log",
}
_COORDINATE_HEADERS = {
    "Cx": ("volScalarField", "[0 1 0 0 0 0 0]"),
    "Cy": ("volScalarField", "[0 1 0 0 0 0 0]"),
    "Cz": ("volScalarField", "[0 1 0 0 0 0 0]"),
}
_SNAPSHOT_HEADERS = {
    "U": ("volVectorField", "[0 1 -1 0 0 0 0]"),
    "p": ("volScalarField", "[0 2 -2 0 0 0 0]"),
    "k": ("volScalarField", "[0 2 -2 0 0 0 0]"),
    "omega": ("volScalarField", "[0 0 -1 0 0 0 0]"),
    "wallShearStress": ("volVectorField", "[0 2 -2 0 0 0 0]"),
}
_MEAN_HEADERS = {
    "UMean": ("volVectorField", "[0 1 -1 0 0 0 0]"),
    "pMean": ("volScalarField", "[0 2 -2 0 0 0 0]"),
    "kMean": ("volScalarField", "[0 2 -2 0 0 0 0]"),
    "omegaMean": ("volScalarField", "[0 0 -1 0 0 0 0]"),
    "wallShearStressMean": ("volVectorField", "[0 2 -2 0 0 0 0]"),
}


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot calculate statistics for an empty field")
    return {
        "min": float(values.min()),
        "p01": float(np.percentile(values, 1)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        "finite": bool(np.isfinite(values).all()),
        "n": int(values.size),
    }


def _steady_solver_metrics(path: Path, time_name: str) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.findall(
        rf"^Time = {re.escape(time_name)}\s*$\n(?P<body>.*?)(?=^Time = |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if not blocks:
        raise ValueError(f"solver log has no Time = {time_name} block: {path}")
    block = blocks[-1]
    residuals = {}
    for field in ("Ux", "Uy", "Uz", "p", "omega", "k"):
        match = re.search(rf"Solving for {field}, Initial residual = ([^,]+)", block)
        if not match:
            raise ValueError(f"final steady solver block has no {field} residual")
        residuals["p_first_corrector" if field == "p" else field] = float(match.group(1))
    continuity = re.search(
        r"continuity errors\s*:\s*sum local = ([^,]+), global = ([^,]+), cumulative = ([^\s]+)",
        block,
    )
    if not continuity:
        raise ValueError("final steady solver block has no continuity record")
    marker_found = "solution convergence criteria satisfied" in block
    return {
        "assessment_type": "steady_snapshot_convergence",
        "final_initial_residuals": residuals,
        "final_continuity": {
            "local": float(continuity.group(1)),
            "global": float(continuity.group(2)),
            "cumulative": float(continuity.group(3)),
        },
        "residual_control_marker_found": marker_found,
        "strict_residual_control_met": marker_found,
    }


def _mesh_metrics(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")

    def integer(label: str) -> int:
        match = re.search(rf"^\s*{label}:\s+(\d+)\s*$", text, re.MULTILINE)
        if not match:
            raise ValueError(f"checkMesh log has no {label} count")
        return int(match.group(1))

    skewness = re.search(r"Max skewness = ([^\s]+)", text)
    if not skewness:
        raise ValueError("checkMesh log has no maximum skewness")
    return {
        "status": "Mesh OK" if "Mesh OK" in text else "failed",
        "points": integer("points"),
        "faces": integer("faces"),
        "cells": integer("cells"),
        "max_skewness": float(skewness.group(1)),
    }


def _relative_l2(current: np.ndarray, previous: np.ndarray) -> float:
    if current.shape != previous.shape:
        raise ValueError(f"current and previous field shapes differ: {current.shape} vs {previous.shape}")
    return float(np.linalg.norm(current - previous) / max(np.linalg.norm(current), 1e-30))


def _patch_body(text: str, patch_name: str) -> str:
    match = re.search(rf"\b{re.escape(patch_name)}\s*\{{(?P<body>.*?)\n\s*\}}", text, re.DOTALL)
    if not match:
        raise ValueError(f"initial field has no {patch_name} patch")
    return match.group("body")


def _scalar_uniform(text: str, pattern: str, label: str) -> float:
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(f"cannot parse {label}")
    return float(match.group(1))


def _entry_path(root: Path, entry: dict, label: str) -> Path:
    if not isinstance(entry, dict):
        raise ValueError(f"{label} manifest entry is not an object")
    file_name = entry.get("file")
    if not file_name or Path(file_name).name != file_name:
        raise ValueError(f"invalid staged path for {label}")
    path = root / file_name
    if not path.is_file():
        raise FileNotFoundError(f"missing staged {label}: {path}")
    expected_hash = entry.get("sha256")
    if not isinstance(expected_hash, str) or file_sha256(path) != expected_hash:
        raise ValueError(f"staged {label} hash mismatch: {path.name}")
    return path


def _validate_semantics(semantics: dict) -> tuple[str, dict, dict[str, str]]:
    if not isinstance(semantics, dict) or set(semantics) != _SEMANTIC_KEYS:
        raise ValueError("native SI manifest has incomplete or unexpected solver semantics")
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
            raise ValueError("snapshot semantics require simpleFoam/steady_rans/snapshot/U/None")
        return statistic, _SNAPSHOT_HEADERS, {
            "velocity": "U",
            "pressure": "p",
            "turbulence_kinetic_energy": "k",
            "specific_dissipation_rate": "omega",
            "wall_shear_stress": "wallShearStress",
        }
    if statistic == "time_mean":
        window = semantics["averaging_window"]
        if (
            semantics["application"] != "pimpleFoam"
            or semantics["mode"] != "urans"
            or semantics["tracking_velocity_object"] != "UMean"
            or not isinstance(window, dict)
            or set(window) != {"start_time", "end_time", "sample_count"}
            or not isinstance(window["sample_count"], int)
            or isinstance(window["sample_count"], bool)
            or window["sample_count"] <= 0
        ):
            raise ValueError("time-mean semantics require pimpleFoam/urans/time_mean/UMean and a valid window")
        if not str(window["start_time"]) or not str(window["end_time"]):
            raise ValueError("time-mean averaging window contains invalid times")
        return statistic, _MEAN_HEADERS, {
            "velocity": "UMean",
            "pressure": "pMean",
            "turbulence_kinetic_energy": "kMean",
            "specific_dissipation_rate": "omegaMean",
            "wall_shear_stress": "wallShearStressMean",
        }
    raise ValueError("field_statistic must be snapshot or time_mean")


def _validate_field(
    native_dir: Path,
    entry: dict,
    object_name: str,
    expected: tuple[str, str],
    location: str,
) -> Path:
    path = _entry_path(native_dir, entry, f"field {object_name}")
    header = parse_foam_header(path)
    expected_header = {
        "class": expected[0],
        "location": location,
        "object": object_name,
        "dimensions": expected[1],
    }
    if header != expected_header:
        raise ValueError(f"unexpected OpenFOAM header for {path.name}: {header}")
    if entry.get("header") != header:
        raise ValueError(f"manifest header record differs from field header: {path.name}")
    return path


def _physical_inputs(manifest: dict, native_dir: Path) -> dict:
    sources = manifest["source_files"]

    def text(key: str) -> str:
        return (native_dir / sources[key]["file"]).read_text(encoding="utf-8", errors="replace")

    initial_u = text("initial_U")
    initial_k = text("initial_k")
    initial_omega = text("initial_omega")
    jet_u_match = re.search(r"value\s+uniform\s+\(([^)]+)\)", _patch_body(initial_u, "jetInlet"))
    if not jet_u_match:
        raise ValueError("cannot parse jet inlet velocity")
    jet_velocity = np.fromstring(jet_u_match.group(1), sep=" ")
    if jet_velocity.shape != (3,):
        raise ValueError("jet inlet velocity is not a vector")
    jet_speed = float(np.linalg.norm(jet_velocity))
    inlet_k = _scalar_uniform(_patch_body(initial_k, "jetInlet"), r"value\s+uniform\s+([^;]+);", "inlet k")
    inlet_omega = _scalar_uniform(
        _patch_body(initial_omega, "jetInlet"), r"value\s+uniform\s+([^;]+);", "inlet omega"
    )
    nu = _scalar_uniform(text("transportProperties"), r"\bnu\s+([^;]+);", "kinematic viscosity")
    radius = _scalar_uniform(text("topoSetDict"), r"\bradius\s+([^;]+);", "jet radius")
    block_scale = _scalar_uniform(text("blockMeshDict"), r"\bscale\s+([^;]+);", "blockMesh scale")
    diameter = 2.0 * radius
    turbulence_intensity = float(np.sqrt(2.0 * inlet_k / 3.0) / jet_speed)
    cmu = 0.09
    length_scale = float(np.sqrt(inlet_k) / (cmu**0.25 * inlet_omega))
    values = {
        "jet_velocity_ms": jet_velocity.tolist(),
        "jet_speed_ms": jet_speed,
        "jet_radius_m": radius,
        "jet_diameter_m": diameter,
        "water_kinematic_viscosity_m2_s": nu,
        "reynolds_number": jet_speed * diameter / nu,
        "dynamic_pressure_Pa_at_rho_1000": 0.5 * 1000.0 * jet_speed**2,
        "turbulence_intensity": turbulence_intensity,
        "turbulence_length_scale_m": length_scale,
        "inlet_k_m2_s2": inlet_k,
        "inlet_omega_s-1": inlet_omega,
        "blockMesh_scale": block_scale,
    }
    expected = {
        "jet_speed_ms": 104.8,
        "jet_diameter_m": 0.24,
        "water_kinematic_viscosity_m2_s": 1.0e-6,
        "turbulence_intensity": 0.05,
        "turbulence_length_scale_m": 0.0168,
        "inlet_k_m2_s2": 41.1864,
        "inlet_omega_s-1": 697.44025,
        "blockMesh_scale": 0.001,
    }
    for key, target in expected.items():
        if not np.isclose(values[key], target, rtol=1e-6, atol=1e-12):
            raise ValueError(f"SI physical input mismatch for {key}: {values[key]} vs {target}")
    values["status"] = "passed"
    return values


def _previous_field_change(
    manifest: dict,
    native_dir: Path,
    headers: dict,
    objects: dict[str, str],
    current: dict[str, np.ndarray],
) -> dict | None:
    previous = manifest.get("previous_time")
    if previous is None:
        return None
    if not isinstance(previous, dict) or set(previous) != {"field_time", "fields"}:
        raise ValueError("previous-time manifest has incomplete or unexpected fields")
    previous_time = str(previous["field_time"])
    logical_names = (
        "velocity",
        "pressure",
        "turbulence_kinetic_energy",
        "specific_dissipation_rate",
    )
    expected_names = {objects[key] for key in logical_names}
    entries = previous.get("fields")
    if not isinstance(entries, dict) or set(entries) != expected_names:
        raise ValueError("previous-time manifest has incomplete or unexpected solution fields")

    previous_values = {}
    for logical_name in logical_names:
        object_name = objects[logical_name]
        path = _entry_path(native_dir, entries[object_name], f"previous field {object_name}")
        expected_header = {
            "class": headers[object_name][0],
            "location": previous_time,
            "object": object_name,
            "dimensions": headers[object_name][1],
        }
        if parse_foam_header(path) != expected_header:
            raise ValueError(f"unexpected previous-field header for {path.name}")
        previous_values[logical_name] = (
            parse_foam_vector_field(path) if logical_name == "velocity" else parse_foam_scalar_field(path)
        )

    current_speed_stats = _stats(np.linalg.norm(current["velocity"], axis=1))
    previous_speed_stats = _stats(np.linalg.norm(previous_values["velocity"], axis=1))
    keys = ("median", "p95", "p99", "max")
    speed_change_pct = {
        key: float(
            100.0
            * abs(current_speed_stats[key] - previous_speed_stats[key])
            / max(abs(current_speed_stats[key]), 1e-30)
        )
        for key in keys
    }
    return {
        "previous_time": previous_time,
        "field_relative_l2": {
            objects[logical_name]: _relative_l2(current[logical_name], previous_values[logical_name])
            for logical_name in logical_names
        },
        "bulk_speed_relative_change_pct": speed_change_pct,
        "max_bulk_speed_relative_change_pct": max(speed_change_pct.values()),
    }


def _validate_source_bundle(manifest: dict, source_bundle: dict, time_name: str) -> None:
    expected_units = {
        "source_coordinate_units": "m",
        "velocity_units": "m/s",
        "kinematic_pressure_units": "m2/s2",
        "kinematic_wall_shear_stress_units": "m2/s2",
    }
    if (
        source_bundle.get("version") != "step30-native-si-source-bundle-v2"
        or source_bundle.get("units") != expected_units
        or str(source_bundle.get("field_time")) != time_name
        or source_bundle.get("source_case") != manifest.get("source_case")
        or source_bundle.get("solver_semantics") != manifest.get("solver_semantics")
        or source_bundle.get("polyMesh") != manifest.get("polyMesh")
        or source_bundle.get("si_stl") != manifest.get("si_stl")
        or source_bundle.get("source_files") != manifest.get("source_files")
        or source_bundle.get("previous_time") != manifest.get("previous_time")
    ):
        raise ValueError("source bundle identity/semantics differ from the native SI manifest")
    source_fields = source_bundle.get("fields")
    manifest_fields = manifest.get("fields")
    if not isinstance(source_fields, dict) or set(source_fields) != set(manifest_fields):
        raise ValueError("source bundle fields differ from the native SI manifest")
    for name, entry in source_fields.items():
        if {key: manifest_fields[name].get(key) for key in ("file", "sha256")} != entry:
            raise ValueError(f"source bundle field identity differs for {name}")


def run(
    time_name: str,
    *,
    manifest_path: Path | str | None = None,
    native_dir: Path | str = NATIVE,
    topology_path: Path | str = TOPOLOGY,
    output_path: Path | str = OUT_JSON,
    urans_validation_path: Path | str | None = None,
) -> dict:
    """Validate one production manifest and write provenance v3."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", time_name):
        raise ValueError("invalid OpenFOAM time name")
    native_dir = Path(native_dir)
    topology_path = Path(topology_path)
    output_path = Path(output_path)
    manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else native_dir / f"native_si_manifest_{time_name}.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != MANIFEST_VERSION or str(manifest.get("field_time")) != time_name:
        raise ValueError("production provenance requires the requested native SI manifest v2")
    if manifest.get("units") != {
        "source_coordinate_units": "m",
        "internal_coordinate_units": "mm",
        "coordinate_scale_to_mm": 1000.0,
    }:
        raise ValueError("native SI manifest has an invalid unit contract")

    statistic, solution_headers, objects = _validate_semantics(manifest.get("solver_semantics"))
    if statistic == "time_mean" and str(manifest["solver_semantics"]["averaging_window"]["end_time"]) != time_name:
        raise ValueError("time-mean averaging window end differs from the field time")
    expected_headers = {**_COORDINATE_HEADERS, **solution_headers}
    entries = manifest.get("fields")
    if not isinstance(entries, dict) or set(entries) != set(expected_headers):
        raise ValueError("native SI manifest has incomplete or unexpected fields")
    paths = {
        name: _validate_field(native_dir, entries[name], name, expected, time_name)
        for name, expected in expected_headers.items()
    }

    expected_source_keys = set(_SOURCE_FILE_KEYS)
    if statistic == "time_mean":
        expected_source_keys.update({"urans_validation", "steady_assessment", "extended_burnin_rejection"})
    source_entries = manifest.get("source_files")
    if not isinstance(source_entries, dict) or set(source_entries) != expected_source_keys:
        raise ValueError("native SI manifest has incomplete or unexpected source files")
    source_paths = {
        name: _entry_path(native_dir, source_entries[name], f"source file {name}")
        for name in sorted(expected_source_keys)
    }
    source_bundle_entry = manifest.get("source_bundle_manifest")
    source_bundle_path = _entry_path(native_dir, source_bundle_entry, "source bundle manifest")
    source_bundle = json.loads(source_bundle_path.read_text(encoding="utf-8"))
    _validate_source_bundle(manifest, source_bundle, time_name)

    topology_entry = manifest.get("topology")
    expected_topology_keys = {
        "file",
        "sha256",
        "artifact_version",
        "mesh_signature_algorithm",
        "mesh_signature",
        "source_vtp_sha256",
        "time",
        "patch_name",
        "n_parent_faces",
        "surface_area_m2",
    }
    if (
        not isinstance(topology_entry, dict)
        or set(topology_entry) != expected_topology_keys
        or topology_entry.get("file") != topology_path.name
    ):
        raise ValueError("native SI manifest topology contract/path is invalid")
    if file_sha256(topology_path) != topology_entry.get("sha256"):
        raise ValueError("SI topology artifact hash differs from the manifest")
    topology = load_topology_artifact(topology_path)
    topology_contract = {
        "artifact_version": topology["artifact_version"],
        "mesh_signature_algorithm": topology["mesh_signature_algorithm"],
        "mesh_signature": topology["mesh_signature"],
        "source_vtp_sha256": topology["source_vtp_sha256"],
        "time": topology["time_name"],
        "patch_name": topology["patch_name"],
        "n_parent_faces": topology["n_parent_faces"],
        "surface_area_m2": topology["surface_area_m2"],
    }
    for key, value in topology_contract.items():
        if topology_entry.get(key) != value:
            raise ValueError(f"SI topology manifest mismatch for {key}")
    if topology["artifact_version"] != TOPOLOGY_VERSION or topology["time_name"] != time_name:
        raise ValueError("production provenance requires topology v3 at the requested field time")
    if (
        topology["source_coordinate_units"],
        topology["internal_coordinate_units"],
        topology["coordinate_scale_to_mm"],
    ) != ("m", "mm", 1000.0):
        raise ValueError("SI topology has an invalid coordinate unit contract")
    if source_entries["bucket_vtp"]["sha256"] != topology["source_vtp_sha256"]:
        raise ValueError("SI topology was not built from the staged bucket VTP")
    if manifest.get("si_stl") != source_entries["si_stl"]:
        raise ValueError("SI STL identity differs between manifest entries")

    current = {
        "velocity": parse_foam_vector_field(paths[objects["velocity"]]),
        "pressure": parse_foam_scalar_field(paths[objects["pressure"]]),
        "turbulence_kinetic_energy": parse_foam_scalar_field(paths[objects["turbulence_kinetic_energy"]]),
        "specific_dissipation_rate": parse_foam_scalar_field(paths[objects["specific_dissipation_rate"]]),
    }
    wall_shear = parse_foam_boundary_vector_field(paths[objects["wall_shear_stress"]], "bucket")
    n_cells = len(current["velocity"])
    if any(len(values) != n_cells for values in current.values()):
        raise ValueError("native SI solution fields have inconsistent cell counts")
    speed = np.linalg.norm(current["velocity"], axis=1)
    wall_shear_magnitude = np.linalg.norm(wall_shear, axis=1)
    previous_change = _previous_field_change(manifest, native_dir, solution_headers, objects, current)

    solver_log = source_paths["solver_log"]
    if statistic == "snapshot":
        convergence = _steady_solver_metrics(solver_log, time_name)
        convergence.update({
            "solver_log": solver_log.name,
            "end_time": time_name,
            "residual_control_threshold": 1.0e-4,
            "previous_snapshot_field_change": previous_change,
        })
        convergence["status"] = (
            "strict_residual_control_met"
            if convergence["strict_residual_control_met"]
            else "bounded_statistical_plateau_strict_residual_control_not_met"
        )
    else:
        if urans_validation_path is None:
            raise ValueError("time-mean provenance requires an explicit URANS validation report")
        urans_validation_path = Path(urans_validation_path)
        if file_sha256(urans_validation_path) != source_entries["urans_validation"]["sha256"]:
            raise ValueError("explicit URANS validation differs from the staged report")
        urans_validation = json.loads(urans_validation_path.read_text(encoding="utf-8"))
        if not _accepted_urans_report(
            urans_validation,
            manifest["solver_semantics"]["averaging_window"],
            time_name,
        ):
            raise ValueError("URANS validation report does not accept this averaging window")
        steady_assessment = json.loads(source_paths["steady_assessment"].read_text(encoding="utf-8"))
        if not _rejected_steady_assessment(steady_assessment):
            raise ValueError("steady assessment does not justify the URANS transition")
        extended_burnin_rejection = json.loads(
            source_paths["extended_burnin_rejection"].read_text(encoding="utf-8")
        )
        averaging_start = str(manifest["solver_semantics"]["averaging_window"]["start_time"])
        if not _rejected_urans_report(extended_burnin_rejection, averaging_start):
            raise ValueError("extended-burnin rejection does not precede this averaging window")
        convergence = {
            "assessment_type": "urans_time_mean_sampling",
            "solver_log": solver_log.name,
            "end_time": time_name,
            "averaging_window": manifest["solver_semantics"]["averaging_window"],
            "previous_mean_field_change": previous_change,
            "simple_convergence_marker_applied": False,
            "validation_report": {
                "file": urans_validation_path.name,
                "sha256": file_sha256(urans_validation_path),
                "gates": urans_validation["gates"],
            },
            "steady_rejection_evidence": {
                "file": source_paths["steady_assessment"].name,
                "sha256": file_sha256(source_paths["steady_assessment"]),
                "accepted_as_steady_fixed_point": False,
            },
            "extended_burnin_rejection_evidence": {
                "file": source_paths["extended_burnin_rejection"].name,
                "sha256": file_sha256(source_paths["extended_burnin_rejection"]),
                "accepted_as_clean_window": False,
            },
            "status": "sampling_window_validated",
        }

    mesh = _mesh_metrics(source_paths["checkMesh_log"])
    if mesh["cells"] != n_cells:
        raise ValueError("checkMesh cell count differs from the native SI solution fields")
    physical_inputs = _physical_inputs(manifest, native_dir)
    all_finite = bool(
        all(np.isfinite(values).all() for values in current.values()) and np.isfinite(wall_shear).all()
    )
    field_stats = {
        "field_statistic": statistic,
        "objects": objects,
        "n_cells": n_cells,
        f"{objects['velocity']}_speed_ms": _stats(speed),
        f"{objects['pressure']}_kinematic_m2_s2": _stats(current["pressure"]),
        f"{objects['pressure']}_Pa_at_rho_1000": _stats(1000.0 * current["pressure"]),
        f"{objects['turbulence_kinetic_energy']}_m2_s2": _stats(current["turbulence_kinetic_energy"]),
        f"{objects['specific_dissipation_rate']}_s-1": _stats(current["specific_dissipation_rate"]),
        f"bucket_{objects['wall_shear_stress']}_kinematic_m2_s2": _stats(wall_shear_magnitude),
        f"bucket_{objects['wall_shear_stress']}_Pa_at_rho_1000": _stats(1000.0 * wall_shear_magnitude),
        "bucket_wall_faces": int(len(wall_shear)),
    }
    result = {
        "version": PROVENANCE_VERSION,
        "openfoam_version": "2412",
        "source_case": manifest["source_case"],
        "source_time": time_name,
        "solver_semantics": manifest["solver_semantics"],
        "artifact_manifest": {
            "file": manifest_path.name,
            "version": manifest["version"],
            "sha256": file_sha256(manifest_path),
        },
        "source_coordinate_units": "m",
        "internal_coordinate_units": "mm",
        "coordinate_scale_to_mm": 1000.0,
        "velocity_units": "m/s",
        "pressure_field_units": "m2/s2 (kinematic pressure)",
        "reference_density_kg_m3": 1000.0,
        "turbulence_units": {"k": "m2/s2", "omega": "1/s"},
        "wall_shear_stress_field_units": "m2/s2 (kinematic wall shear stress)",
        "topology": {
            "artifact": topology_path.name,
            "artifact_sha256": file_sha256(topology_path),
            "artifact_version": topology["artifact_version"],
            "mesh_signature_algorithm": topology["mesh_signature_algorithm"],
            "mesh_signature": topology["mesh_signature"],
            "patch_name": topology["patch_name"],
            "n_parent_faces": topology["n_parent_faces"],
            "n_collision_triangles": int(len(topology["collision_triangles_mm"])),
            "surface_area_m2": topology["surface_area_m2"],
            "bounds_mm": topology["bounds_mm"],
            "polyMesh_signature": manifest["polyMesh"],
        },
        "fields": field_stats,
        "physical_inputs": physical_inputs,
        "physical_checks": {
            "max_speed_quality_limit_ms": 500.0,
            "max_speed_quality_status": bool(speed.max() < 500.0),
            "bucket_face_count_matches_topology": bool(len(wall_shear) == topology["n_parent_faces"]),
            "all_field_values_finite": all_finite,
        },
        "convergence": convergence,
        "mesh": mesh,
        "source_files": manifest["source_files"],
        "si_stl": manifest["si_stl"],
    }
    if mesh["status"] != "Mesh OK":
        raise ValueError("final SI checkMesh did not report Mesh OK")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write SI OpenFOAM provenance v3")
    parser.add_argument("--time", required=True, help="validated OpenFOAM field time")
    parser.add_argument("--manifest", type=Path, help="native SI manifest v2 path")
    parser.add_argument("--native-dir", type=Path, default=NATIVE, help="staged native SI artifact directory")
    parser.add_argument("--topology", type=Path, default=TOPOLOGY, help="topology v3 artifact path")
    parser.add_argument("--output", type=Path, default=OUT_JSON, help="provenance JSON output path")
    parser.add_argument("--urans-validation", type=Path, help="accepted URANS time-mean validation report")
    args = parser.parse_args()
    run(
        args.time,
        manifest_path=args.manifest,
        native_dir=args.native_dir,
        topology_path=args.topology,
        output_path=args.output,
        urans_validation_path=args.urans_validation,
    )
