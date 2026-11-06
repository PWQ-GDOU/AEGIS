"""Stage one time-consistent OpenFOAM result bundle for Step 30."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path

FIELD_NAMES = ("Cx", "Cy", "Cz", "U", "p", "k", "omega", "wallShearStress")
MESH_FILES = ("boundary", "faces", "neighbour", "owner", "points")


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


def stage(case: Path, time_name: str, output: Path, solver_log: Path, mesh_log: Path) -> dict:
    case = case.resolve()
    time_dir = case / time_name
    if not time_dir.is_dir():
        raise FileNotFoundError(f"OpenFOAM time directory is missing: {time_dir}")
    vtp = case / "VTK" / f"{case.name}_{time_name}" / "boundary" / "bucket.vtp"
    if not vtp.exists():
        matches = list((case / "VTK").glob(f"*_{time_name}/boundary/bucket.vtp"))
        if len(matches) != 1:
            raise FileNotFoundError(f"cannot identify bucket VTP for time {time_name}")
        vtp = matches[0]

    numeric_times = []
    for candidate in case.iterdir():
        if not candidate.is_dir() or candidate.name == time_name:
            continue
        try:
            numeric_times.append((Decimal(candidate.name), candidate))
        except InvalidOperation:
            pass
    previous = max((item for item in numeric_times if item[0] < Decimal(time_name)), default=None)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="step30-export-", dir=output.parent) as temp_name:
        temp = Path(temp_name)
        files: dict[str, dict] = {}
        for name in FIELD_NAMES:
            source = time_dir / name
            target = temp / f"native_si_{time_name}_{name}.foam"
            shutil.copy2(source, target)
            files[name] = {"file": target.name, "sha256": sha256(target)}

        fixed = {
            "bucket_vtp": (vtp, f"bucket_{time_name}_si.vtp"),
            "polyMesh_boundary": (case / "constant" / "polyMesh" / "boundary", f"polyMesh_boundary_{time_name}.txt"),
            "controlDict": (case / "system" / "controlDict", f"controlDict_{time_name}"),
            "fvSolution": (case / "system" / "fvSolution", f"fvSolution_{time_name}"),
            "blockMeshDict": (case / "system" / "blockMeshDict", "blockMeshDict"),
            "snappyHexMeshDict": (case / "system" / "snappyHexMeshDict", "snappyHexMeshDict"),
            "topoSetDict": (case / "system" / "topoSetDict", "topoSetDict"),
            "transportProperties": (case / "constant" / "transportProperties", "transportProperties"),
            "turbulenceProperties": (case / "constant" / "turbulenceProperties", "turbulenceProperties"),
            "initial_U": (case / "0" / "U", "initial_U"),
            "initial_k": (case / "0" / "k", "initial_k"),
            "initial_omega": (case / "0" / "omega", "initial_omega"),
            "solver_log": (solver_log, f"simpleFoam_{time_name}.log"),
            "checkMesh_log": (mesh_log, f"checkMesh_{time_name}.log"),
        }
        fixed_manifest = {}
        for key, (source, target_name) in fixed.items():
            target = temp / target_name
            shutil.copy2(source, target)
            fixed_manifest[key] = {"file": target.name, "sha256": sha256(target)}

        previous_manifest = None
        if previous is not None:
            previous_time = previous[1].name
            previous_fields = {}
            for name in ("U", "p", "k", "omega"):
                source = previous[1] / name
                target = temp / f"native_si_{previous_time}_{name}.foam"
                shutil.copy2(source, target)
                previous_fields[name] = {"file": target.name, "sha256": sha256(target)}
            previous_manifest = {"field_time": previous_time, "fields": previous_fields}

        stl = case / "constant" / "triSurface" / "dec_full_runner_local_si.stl"
        manifest = {
            "version": "step30-native-si-source-bundle-v1",
            "source_case": str(case),
            "field_time": time_name,
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
            "fields": files,
            "previous_time": previous_manifest,
            "source_files": fixed_manifest,
        }
        manifest_path = temp / f"source_bundle_manifest_{time_name}.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if output.exists():
            shutil.rmtree(output)
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
    args = parser.parse_args()
    output = args.output or args.case / "step30_export"
    stage(args.case, args.time, output, args.solver_log, args.mesh_log)
