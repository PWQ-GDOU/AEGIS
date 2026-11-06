"""Isolated CalculiX mathematical boundary-envelope harness.

This module is intentionally outside the formal FEM pipeline. Surface tags are
anonymous numerical hypotheses only. It neither imports formal selection
validators nor accepts CFD pressure, AI5 values, or named physical selections.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from step30_validation_common import canonical_sha256, safe_json_dump, sha256_file


VERSION = "step30-exploratory-fem-envelope-v1"
EXPECTED_STEP_SHA256 = "521fbaf7688ad1f0e063dedb08c360135026ce60e5e371b4a196294b18cfe21d"
RPM = 375.0
OMEGA_RAD_S = RPM * 2.0 * math.pi / 60.0
DENSITIES_KG_M3 = (7700.0, 7800.0, 7900.0)
UNIT_HYPOTHESES = {"m": 1.0, "mm": 1.0e-3}
AXIS_HYPOTHESES = {
    "global_x": (1.0, 0.0, 0.0),
    "global_y": (0.0, 1.0, 0.0),
    "global_z": (0.0, 0.0, 1.0),
}


class ExploratoryFEMEnvelopeError(ValueError):
    """Raised when an exploratory case violates its isolation contract."""


def build_case_matrix(surface_tags: Sequence[int]) -> list[dict[str, Any]]:
    """Construct all anonymous support/unit/density/axis combinations."""
    tags = sorted({int(tag) for tag in surface_tags})
    if len(tags) != 30 or tags != list(range(1, 31)):
        raise ExploratoryFEMEnvelopeError(
            "exploratory envelope requires exactly catalogued anonymous surface tags 1..30"
        )
    cases = []
    for support_tag in tags:
        for unit_hypothesis in UNIT_HYPOTHESES:
            for density in DENSITIES_KG_M3:
                for axis_name, axis in AXIS_HYPOTHESES.items():
                    case_id = (
                        f"support_{support_tag:02d}_{unit_hypothesis}_rho_{int(density)}_{axis_name}"
                    )
                    cases.append(
                        {
                            "case_id": case_id,
                            "fixed_support_hypothesis": f"fixed_support_surface_{support_tag}",
                            "anonymous_surface_tag": support_tag,
                            "coordinate_unit_hypothesis": unit_hypothesis,
                            "coordinate_scale_to_m": UNIT_HYPOTHESES[unit_hypothesis],
                            "density_kg_m3": density,
                            "rotation_axis_hypothesis": axis_name,
                            "rotation_axis_unit": list(axis),
                            "rotation_origin_native": [0.0, 0.0, 0.0],
                            "rpm": RPM,
                            "angular_speed_rad_s": OMEGA_RAD_S,
                        }
                    )
    return cases


def _load_surface_catalog(catalog_path: str | Path, step_path: Path) -> list[int]:
    catalog = Path(catalog_path).resolve()
    if not catalog.is_file():
        raise ExploratoryFEMEnvelopeError(f"surface catalog is missing: {catalog}")
    document = json.loads(catalog.read_text(encoding="utf-8"))
    if document.get("source_sha256") != sha256_file(step_path):
        raise ExploratoryFEMEnvelopeError("surface catalog is not bound to supplied STEP bytes")
    surfaces = document.get("surfaces")
    if not isinstance(surfaces, Sequence):
        raise ExploratoryFEMEnvelopeError("surface catalog has no surface list")
    return [int(item["surface_tag"]) for item in surfaces if isinstance(item, Mapping)]


def _dependency_status(ccx_executable: str | None) -> dict[str, Any]:
    try:
        import gmsh  # type: ignore

        gmsh_version = str(gmsh.__version__)
        gmsh_ready = True
    except (ImportError, AttributeError):
        gmsh_version = None
        gmsh_ready = False
    candidate = ccx_executable or "ccx"
    resolved_ccx = (
        str(Path(candidate).resolve())
        if Path(candidate).is_file()
        else shutil.which(candidate)
    )
    return {
        "gmsh_python_available": gmsh_ready,
        "gmsh_version": gmsh_version,
        "ccx_executable_requested": candidate,
        "ccx_executable_resolved": resolved_ccx,
        "ccx_available": resolved_ccx is not None,
    }


def build_blocked_envelope_report(
    step_path: str | Path,
    surface_catalog_path: str | Path,
    *,
    ccx_executable: str | None = None,
) -> dict[str, Any]:
    """Return an honest preflight result without claiming solver output."""
    step = Path(step_path).resolve()
    if not step.is_file():
        raise ExploratoryFEMEnvelopeError(f"STEP geometry is missing: {step}")
    step_hash = sha256_file(step)
    tags = _load_surface_catalog(surface_catalog_path, step)
    cases = build_case_matrix(tags)
    dependencies = _dependency_status(ccx_executable)
    blockers = []
    if not dependencies["gmsh_python_available"]:
        blockers.append("gmsh_python_module_unavailable")
    if not dependencies["ccx_available"]:
        blockers.append("calculix_executable_unavailable")
    status = "ready_to_run" if not blockers else "blocked"
    report = {
        "version": VERSION,
        "status": status,
        "evidence_status": "exploratory_solver_envelope",
        "formal_evidence_status": "ineligible",
        "fem_gate": "blocked",
        "official_accuracy_gate": False,
        "real_fem_evidence": False,
        "STEP": {
            "path": str(step),
            "sha256": step_hash,
            "expected_official_geometry_hash": EXPECTED_STEP_SHA256,
            "hash_matches_catalogued_geometry": step_hash == EXPECTED_STEP_SHA256,
        },
        "assumption_contract": {
            "surface_semantics": "anonymous numerical support hypotheses only",
            "forbidden_inputs": [
                "CFD pressure",
                "AI5 calibration",
                "AI5 patch or gauge direction",
                "ROOT_INTERFACE or WETTED_SURFACE physical selections",
            ],
            "load_mode": "centrifugal_only",
            "rpm": RPM,
            "coordinate_unit_hypotheses": list(UNIT_HYPOTHESES),
            "density_kg_m3": list(DENSITIES_KG_M3),
            "rotation_axes": AXIS_HYPOTHESES,
        },
        "dependency_status": dependencies,
        "case_count": len(cases),
        "case_matrix_sha256": canonical_sha256(cases),
        "case_matrix": cases,
        "blockers": blockers,
        "result_interpretation": (
            "No physical boundary-condition, stress-validation, material-certification, "
            "or competition-score meaning may be assigned to these cases."
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def _generate_mesh(step_path: Path, mesh_path: Path, characteristic_length: float) -> dict[str, Any]:
    """Mesh the one-volume STEP and retain surface-node sets by raw tag."""
    try:
        import gmsh  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ExploratoryFEMEnvelopeError("gmsh Python module is unavailable") from exc
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("exploratory_envelope")
        gmsh.merge(str(step_path))
        gmsh.model.occ.synchronize()
        volumes = gmsh.model.getEntities(3)
        surfaces = gmsh.model.getEntities(2)
        if len(volumes) != 1 or len(surfaces) != 30:
            raise ExploratoryFEMEnvelopeError("STEP is not the expected one-volume/30-surface catalog")
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", characteristic_length)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", characteristic_length)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(mesh_path))
        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        nodes = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
        node_ids = np.asarray(node_tags, dtype=np.int64)
        types, _, element_nodes = gmsh.model.mesh.getElements(3, volumes[0][1])
        tetra = None
        for element_type, connectivity in zip(types, element_nodes):
            properties = gmsh.model.mesh.getElementProperties(element_type)
            if int(properties[3]) == 4:
                tetra = np.asarray(connectivity, dtype=np.int64).reshape(-1, 4)
                break
        if tetra is None or len(tetra) == 0:
            raise ExploratoryFEMEnvelopeError("Gmsh did not create linear tetrahedral elements")
        surface_nodes: dict[int, np.ndarray] = {}
        for _, tag in surfaces:
            raw, _, _ = gmsh.model.mesh.getNodes(2, tag, includeBoundary=True)
            surface_nodes[int(tag)] = np.unique(np.asarray(raw, dtype=np.int64))
        return {
            "node_ids": node_ids,
            "nodes_native": nodes,
            "tetra_node_ids": tetra,
            "surface_nodes": surface_nodes,
            "mesh_sha256": sha256_file(mesh_path),
            "n_nodes": int(len(node_ids)),
            "n_tetra4": int(len(tetra)),
        }
    finally:
        gmsh.finalize()


def _write_case_input(
    mesh: Mapping[str, Any],
    case: Mapping[str, Any],
    input_path: Path,
) -> None:
    node_ids = np.asarray(mesh["node_ids"], dtype=np.int64)
    nodes = np.asarray(mesh["nodes_native"], dtype=np.float64)
    tetra = np.asarray(mesh["tetra_node_ids"], dtype=np.int64)
    support = int(case["anonymous_surface_tag"])
    support_nodes = np.asarray(mesh["surface_nodes"][support], dtype=np.int64)
    scale = float(case["coordinate_scale_to_m"])
    origin = np.asarray(case["rotation_origin_native"], dtype=np.float64) * scale
    axis = np.asarray(case["rotation_axis_unit"], dtype=np.float64)
    lines = [
        "** exploratory_solver_envelope only; not formal FEM evidence",
        f"** anonymous_support={case['fixed_support_hypothesis']}",
        f"** coordinate_unit_hypothesis={case['coordinate_unit_hypothesis']}",
        "*HEADING",
        "Exploratory anonymous-boundary centrifugal envelope",
        "*NODE, NSET=ALL_NODES",
    ]
    for node_id, xyz in zip(node_ids, nodes * scale):
        lines.append(f"{int(node_id)}, {xyz[0]:.16e}, {xyz[1]:.16e}, {xyz[2]:.16e}")
    lines.append("*ELEMENT, TYPE=C3D4, ELSET=ALL_SOLID")
    for element_id, members in enumerate(tetra, start=1):
        lines.append(f"{element_id}, " + ", ".join(str(int(node)) for node in members))
    lines.extend(["*NSET, NSET=ANON_SUPPORT"])
    for start in range(0, len(support_nodes), 16):
        lines.append(", ".join(str(int(node)) for node in support_nodes[start : start + 16]))
    lines.extend(
        [
            "*MATERIAL, NAME=NUMERIC_ELASTIC",
            "*ELASTIC",
            "2.0600000000000000e11, 3.0000000000000000e-01",
            "*DENSITY",
            f"{float(case['density_kg_m3']):.16e}",
            "*SOLID SECTION, ELSET=ALL_SOLID, MATERIAL=NUMERIC_ELASTIC",
            "*BOUNDARY",
            "ANON_SUPPORT, 1, 3, 0.0",
            "*STEP",
            "*STATIC",
            "*DLOAD",
            "ALL_SOLID, CENTRIF, "
            f"{float(case['angular_speed_rad_s']) ** 2:.16e}, "
            f"{origin[0]:.16e}, {origin[1]:.16e}, {origin[2]:.16e}, "
            f"{axis[0]:.16e}, {axis[1]:.16e}, {axis[2]:.16e}",
            "*NODE PRINT, NSET=ALL_NODES",
            "U",
            "*EL PRINT, ELSET=ALL_SOLID",
            "S",
            "*END STEP",
        ]
    )
    input_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _parse_dat_extrema(dat_path: Path) -> dict[str, float | None]:
    """Best-effort parser for CalculiX text output; preserves native output too."""
    if not dat_path.is_file():
        return {"max_displacement_m": None, "max_von_mises_pa": None}
    max_displacement = 0.0
    max_von_mises = 0.0
    mode = None
    for raw in dat_path.read_text(encoding="latin-1", errors="ignore").splitlines():
        lower = raw.lower()
        if "displacements" in lower:
            mode = "u"
            continue
        if "stresses" in lower:
            mode = "s"
            continue
        tokens = raw.split()
        try:
            values = [float(item.replace("D", "E")) for item in tokens]
        except ValueError:
            continue
        if mode == "u" and len(values) >= 4:
            max_displacement = max(max_displacement, float(np.linalg.norm(values[-3:])))
        elif mode == "s" and len(values) >= 7:
            sxx, syy, szz, sxy, sxz, syz = values[-6:]
            vm = math.sqrt(
                0.5
                * (
                    (sxx - syy) ** 2
                    + (syy - szz) ** 2
                    + (szz - sxx) ** 2
                    + 6.0 * (sxy**2 + sxz**2 + syz**2)
                )
            )
            max_von_mises = max(max_von_mises, vm)
    return {
        "max_displacement_m": max_displacement if max_displacement > 0.0 else None,
        "max_von_mises_pa": max_von_mises if max_von_mises > 0.0 else None,
    }


def run_exploratory_envelope(
    step_path: str | Path,
    surface_catalog_path: str | Path,
    output_directory: str | Path,
    *,
    ccx_executable: str | None = "ccx",
    characteristic_length_native: float = 25.0,
    max_cases: int | None = None,
) -> dict[str, Any]:
    """Run the isolated envelope only when local Gmsh and CalculiX exist."""
    preflight = build_blocked_envelope_report(
        step_path, surface_catalog_path, ccx_executable=ccx_executable
    )
    target = Path(output_directory).resolve()
    target.mkdir(parents=True, exist_ok=True)
    if preflight["status"] != "ready_to_run":
        preflight["status"] = "blocked"
        preflight["run_status"] = "not_started_no_solver_artifacts_fabricated"
        preflight["report_sha256"] = canonical_sha256(
            {key: value for key, value in preflight.items() if key != "report_sha256"}
        )
        safe_json_dump(preflight, target / "exploratory_envelope_report.json")
        return preflight
    cases = preflight["case_matrix"]
    if max_cases is not None:
        if max_cases <= 0:
            raise ExploratoryFEMEnvelopeError("max_cases must be positive when supplied")
        cases = cases[:max_cases]
    mesh_path = target / "anonymous_native_mesh.msh"
    mesh = _generate_mesh(
        Path(step_path).resolve(), mesh_path, characteristic_length_native
    )
    ccx = str(preflight["dependency_status"]["ccx_executable_resolved"])
    case_results = []
    for case in cases:
        case_dir = target / case["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        input_path = case_dir / "case.inp"
        _write_case_input(mesh, case, input_path)
        stdout_path = case_dir / "stdout.txt"
        stderr_path = case_dir / "stderr.txt"
        completed = subprocess.run(
            [ccx, "-i", input_path.stem],
            cwd=case_dir,
            text=True,
            capture_output=True,
            check=False,
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
        stderr_path.write_text(completed.stderr, encoding="utf-8", errors="replace")
        native = {
            suffix: case_dir / f"{input_path.stem}{suffix}"
            for suffix in (".dat", ".frd", ".sta", ".cvg")
        }
        extrema = _parse_dat_extrema(native[".dat"])
        native_hashes = {
            suffix: sha256_file(path) if path.is_file() else None
            for suffix, path in native.items()
        }
        native_results = {
            suffix: {
                "path": str(path),
                "sha256": native_hashes[suffix],
            }
            for suffix, path in native.items()
        }
        case_results.append(
            {
                **case,
                "status": "solver_completed" if completed.returncode == 0 else "solver_failed",
                "return_code": completed.returncode,
                "mesh_path": str(mesh_path),
                "mesh_sha256": mesh["mesh_sha256"],
                "input_path": str(input_path),
                "input_sha256": sha256_file(input_path),
                "stdout_path": str(stdout_path),
                "stdout_sha256": sha256_file(stdout_path),
                "stderr_path": str(stderr_path),
                "stderr_sha256": sha256_file(stderr_path),
                "native_result_hashes": native_hashes,
                "native_results": native_results,
                **extrema,
                "formal_evidence_status": "ineligible",
                "fem_gate": "blocked",
            }
        )
    complete = [item for item in case_results if item["status"] == "solver_completed"]
    stress = [item["max_von_mises_pa"] for item in complete if item["max_von_mises_pa"] is not None]
    displacement = [item["max_displacement_m"] for item in complete if item["max_displacement_m"] is not None]
    report = {
        **preflight,
        "status": "completed" if len(complete) == len(case_results) else "completed_with_solver_failures",
        "run_status": "solver_executed",
        "mesh": {
            "path": str(mesh_path),
            "sha256": mesh["mesh_sha256"],
            "n_nodes": mesh["n_nodes"],
            "n_tetra4": mesh["n_tetra4"],
        },
        "executed_case_count": len(case_results),
        "case_results": case_results,
        "envelope_bounds": {
            "max_von_mises_pa": None if not stress else {"min": min(stress), "max": max(stress)},
            "max_displacement_m": None
            if not displacement
            else {"min": min(displacement), "max": max(displacement)},
        },
    }
    report["report_sha256"] = canonical_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    safe_json_dump(report, target / "exploratory_envelope_report.json")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an ineligible anonymous-support FEM envelope")
    parser.add_argument("--step", type=Path, required=True)
    parser.add_argument("--surface-catalog", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--ccx", default="ccx")
    parser.add_argument("--characteristic-length-native", type=float, default=25.0)
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args(argv)
    report = run_exploratory_envelope(
        args.step,
        args.surface_catalog,
        args.output_directory,
        ccx_executable=args.ccx,
        characteristic_length_native=args.characteristic_length_native,
        max_cases=args.max_cases,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "completed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
