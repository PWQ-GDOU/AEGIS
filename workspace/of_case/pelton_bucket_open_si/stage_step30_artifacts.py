"""Stage one time-consistent OpenFOAM result bundle for Step 30."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path

MESH_FILES = ("boundary", "faces", "neighbour", "owner", "points")
STEADY_OBJECTS = ("U", "p", "k", "omega", "wallShearStress")
MEAN_OBJECTS = ("UMean", "pMean", "kMean", "omegaMean", "wallShearStressMean")
URANS_VALIDATION_GATES = {
    "max_co_below_0p5",
    "no_turbulence_bounding",
    "no_fatal_or_nonfinite_log",
    "twenty_flow_through_window",
    "mean_field_relative_l2_below_1e3",
    "force_late_mean_change_below_1pct",
    "flux_late_mean_change_below_1pct",
    "flux_imbalance_below_1e4",
    "probe_late_mean_change_below_1pct",
    "mean_speed_below_500_ms",
    "all_mean_fields_finite",
}
STEADY_VALIDATION_GATES = {
    "first_pressure_residual_below_1e4",
    "all_initial_residuals_below_1e4",
    "all_field_relative_l2_below_1e3",
    "force_change_below_1pct",
    "flux_imbalance_below_1e4",
    "no_turbulence_bounding",
}


def _accepted_urans_report(validation: dict, averaging_window: dict, time_name: str) -> bool:
    gates = validation.get("gates")
    return (
        validation.get("version") in {
            "urans-time-mean-validation-v1",
            "urans-time-mean-validation-v2",
        }
        and validation.get("averaging_window") == averaging_window
        and str(validation.get("field_time")) == time_name
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def poly_mesh_signature(poly_mesh: Path) -> str:
    digest = hashlib.sha256()
    for name in MESH_FILES:
        path = poly_mesh / name
        digest.update(name.encode("ascii"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def solver_semantics(
    application: str,
    mode: str,
    field_statistic: str,
    averaging_window: dict | None,
) -> dict:
    if field_statistic == "snapshot":
        if (application, mode, averaging_window) != ("simpleFoam", "steady_rans", None):
            raise ValueError("snapshot staging requires simpleFoam/steady_rans and no averaging window")
        velocity_object = "U"
    elif field_statistic == "time_mean":
        if application != "pimpleFoam" or mode != "urans" or not isinstance(averaging_window, dict):
            raise ValueError("time-mean staging requires pimpleFoam/urans and an averaging window")
        if set(averaging_window) != {"start_time", "end_time", "sample_count"}:
            raise ValueError("averaging window has unexpected fields")
        if int(averaging_window["sample_count"]) <= 0:
            raise ValueError("averaging sample count must be positive")
        velocity_object = "UMean"
    else:
        raise ValueError("field statistic must be snapshot or time_mean")
    return {
        "application": application,
        "mode": mode,
        "field_statistic": field_statistic,
        "tracking_velocity_object": velocity_object,
        "averaging_window": averaging_window,
    }


def _bucket_vtp(case: Path, field_time: str) -> Path:
    direct = case / "VTK" / f"{case.name}_{field_time}" / "boundary" / "bucket.vtp"
    if direct.is_file():
        return direct
    target = Decimal(field_time)
    matches = []
    for series_path in (case / "VTK").glob("*.vtm.series"):
        series = json.loads(series_path.read_text(encoding="utf-8"))
        for entry in series.get("files", []):
            try:
                mapped_time = Decimal(str(entry["time"]))
                name = Path(entry["name"])
            except (KeyError, InvalidOperation, TypeError):
                continue
            tolerance = max(Decimal("5e-9"), abs(target) * Decimal("5e-9"))
            if abs(mapped_time - target) <= tolerance:
                candidate = series_path.parent / name.stem / "boundary" / "bucket.vtp"
                if candidate.is_file():
                    matches.append(candidate)
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise FileNotFoundError(
            f"cannot identify exactly one bucket VTP for physical time {field_time}: {unique}"
        )
    return unique[0]


def _previous_time(case: Path, field_time: str) -> tuple[str, Path] | None:
    current = Decimal(field_time)
    numeric_times = []
    for candidate in case.iterdir():
        if not candidate.is_dir() or candidate.name == field_time:
            continue
        try:
            numeric_times.append((Decimal(candidate.name), candidate))
        except InvalidOperation:
            pass
    previous = max((item for item in numeric_times if item[0] < current), default=None)
    return (previous[1].name, previous[1]) if previous is not None else None


def stage(
    case: Path,
    time_name: str,
    output: Path,
    solver_log: Path,
    mesh_log: Path,
    *,
    application: str,
    mode: str,
    field_statistic: str,
    averaging_window: dict | None = None,
    validation_report: Path | None = None,
    steady_assessment: Path | None = None,
) -> dict:
    case = case.resolve()
    time_dir = case / time_name
    if not time_dir.is_dir():
        raise FileNotFoundError(f"OpenFOAM time directory is missing: {time_dir}")
    semantics = solver_semantics(application, mode, field_statistic, averaging_window)
    if field_statistic == "time_mean":
        if validation_report is None or not validation_report.is_file():
            raise ValueError("time-mean staging requires an accepted URANS validation report")
        validation = json.loads(validation_report.read_text(encoding="utf-8"))
        if not _accepted_urans_report(validation, averaging_window, time_name):
            raise ValueError("URANS validation report does not accept this staged mean field")
        if steady_assessment is None or not steady_assessment.is_file():
            raise ValueError("time-mean staging requires the rejected steady fixed-point assessment")
        assessment = json.loads(steady_assessment.read_text(encoding="utf-8"))
        if not _rejected_steady_assessment(assessment):
            raise ValueError("steady assessment does not justify the URANS transition")
        extended_burnin_rejection = case / "extended_burnin_rejected_gate.json"
        if not extended_burnin_rejection.is_file():
            raise ValueError("time-mean staging requires rejected extended-burnin evidence")
        rejection = json.loads(extended_burnin_rejection.read_text(encoding="utf-8"))
        if not _rejected_urans_report(rejection, str(averaging_window["start_time"])):
            raise ValueError("extended-burnin rejection does not precede this averaging window")
    elif validation_report is not None or steady_assessment is not None:
        raise ValueError("steady snapshot staging must not attach URANS transition reports")
    solution_objects = STEADY_OBJECTS if field_statistic == "snapshot" else MEAN_OBJECTS
    field_objects = ("Cx", "Cy", "Cz", *solution_objects)

    vtp = _bucket_vtp(case, time_name)

    stl = case / "constant" / "triSurface" / "single_bucket_target_authoritative_local_m.stl"
    previous = _previous_time(case, time_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="step30-export-", dir=output.parent) as temp_name:
        temp = Path(temp_name)
        fields: dict[str, dict] = {}
        for name in field_objects:
            source = time_dir / name
            target = temp / f"native_si_{time_name}_{name}.foam"
            shutil.copy2(source, target)
            fields[name] = {"file": target.name, "sha256": sha256(target)}

        fixed = {
            "bucket_vtp": (vtp, f"bucket_{time_name}_si.vtp"),
            "si_stl": (stl, stl.name),
            "polyMesh_boundary": (case / "constant" / "polyMesh" / "boundary", f"polyMesh_boundary_{time_name}.txt"),
            "controlDict": (case / "system" / "controlDict", f"controlDict_{time_name}"),
            "fvSolution": (case / "system" / "fvSolution", f"fvSolution_{time_name}"),
            "fvSchemes": (case / "system" / "fvSchemes", f"fvSchemes_{time_name}"),
            "blockMeshDict": (case / "system" / "blockMeshDict", "blockMeshDict"),
            "snappyHexMeshDict": (case / "system" / "snappyHexMeshDict", "snappyHexMeshDict"),
            "topoSetDict": (case / "system" / "topoSetDict", "topoSetDict"),
            "transportProperties": (case / "constant" / "transportProperties", "transportProperties"),
            "turbulenceProperties": (case / "constant" / "turbulenceProperties", "turbulenceProperties"),
            "initial_U": (case / "0.orig" / "U", "initial_U"),
            "initial_k": (case / "0.orig" / "k", "initial_k"),
            "initial_omega": (case / "0.orig" / "omega", "initial_omega"),
            "solver_log": (solver_log, f"{application}_{time_name}.log"),
            "checkMesh_log": (mesh_log, f"checkMesh_{time_name}.log"),
        }
        if validation_report is not None:
            fixed["urans_validation"] = (
                validation_report,
                f"urans_validation_{time_name}.json",
            )
            fixed["steady_assessment"] = (
                steady_assessment,
                "steady_fixed_point_assessment.json",
            )
            fixed["extended_burnin_rejection"] = (
                extended_burnin_rejection,
                "extended_burnin_rejected_gate.json",
            )
        source_files = {}
        for key, (source, target_name) in fixed.items():
            target = temp / target_name
            shutil.copy2(source, target)
            source_files[key] = {"file": target.name, "sha256": sha256(target)}

        previous_manifest = None
        if previous is not None:
            previous_time, previous_dir = previous
            previous_fields = {}
            previous_objects = ("U", "p", "k", "omega") if field_statistic == "snapshot" else solution_objects[:4]
            for name in previous_objects:
                source = previous_dir / name
                target = temp / f"native_si_{previous_time}_{name}.foam"
                shutil.copy2(source, target)
                previous_fields[name] = {"file": target.name, "sha256": sha256(target)}
            previous_manifest = {"field_time": previous_time, "fields": previous_fields}

        manifest = {
            "version": "step30-native-si-source-bundle-v2",
            "source_case": str(case),
            "field_time": time_name,
            "solver_semantics": semantics,
            "units": {
                "source_coordinate_units": "m",
                "velocity_units": "m/s",
                "kinematic_pressure_units": "m2/s2",
                "kinematic_wall_shear_stress_units": "m2/s2",
            },
            "polyMesh": {
                "algorithm": "sha256-filename-content-boundary-faces-neighbour-owner-points-v1",
                "signature": poly_mesh_signature(case / "constant" / "polyMesh"),
            },
            "si_stl": {"file": stl.name, "sha256": sha256(stl)},
            "fields": fields,
            "previous_time": previous_manifest,
            "source_files": source_files,
        }
        manifest_path = temp / f"source_bundle_manifest_{time_name}.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"refusing to replace an existing Step30 source bundle: {output}")
        Path(temp_name).replace(output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage Step30 SI OpenFOAM artifacts")
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--time", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--solver-log", type=Path, required=True)
    parser.add_argument("--mesh-log", type=Path, required=True)
    parser.add_argument("--application", choices=["simpleFoam", "pimpleFoam"], required=True)
    parser.add_argument("--mode", choices=["steady_rans", "urans"], required=True)
    parser.add_argument("--field-statistic", choices=["snapshot", "time_mean"], required=True)
    parser.add_argument("--averaging-start")
    parser.add_argument("--averaging-end")
    parser.add_argument("--averaging-samples", type=int)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--steady-assessment", type=Path)
    args = parser.parse_args()
    window = None
    averaging_values = (args.averaging_start, args.averaging_end, args.averaging_samples)
    if any(value is not None for value in averaging_values):
        if not all(value is not None for value in averaging_values):
            parser.error("all averaging-window arguments are required together")
        window = {
            "start_time": args.averaging_start,
            "end_time": args.averaging_end,
            "sample_count": args.averaging_samples,
        }
    output = args.output or args.case / f"step30_export_{args.time}_{args.field_statistic}"
    stage(
        args.case,
        args.time,
        output,
        args.solver_log,
        args.mesh_log,
        application=args.application,
        mode=args.mode,
        field_statistic=args.field_statistic,
        averaging_window=window,
        validation_report=args.validation_report,
        steady_assessment=args.steady_assessment,
    )
