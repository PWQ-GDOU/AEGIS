"""Strict CalculiX C3D10 case generation for Step 30 structural analysis."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from step30_cfd_pressure_mapping import (
    build_surface_geometry,
    is_sha256,
    pressure_load_sha256,
    sha256_file,
)


VERSION = "step30-calculix-case-v1"
REQUIRED_GROUPS = ("ROOT_INTERFACE", "WETTED_SURFACE", "AI5_04")
_TETRA_TYPE = "tetra10"
_SURFACE_TYPE = "triangle6"
# CalculiX C3D10 corner-node definitions for P1..P4.
_C3D10_FACES = ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0))
_C3D10_FACE_NODES = (
    (0, 1, 2, 4, 5, 6),
    (0, 3, 1, 7, 8, 4),
    (1, 3, 2, 8, 9, 5),
    (2, 3, 0, 9, 7, 6),
)


class CalculixCaseError(ValueError):
    """Raised for a mesh, group, material, load, or unit contract failure."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _unit_scale(units: str) -> float:
    normalized = str(units).strip().lower()
    if normalized == "m":
        return 1.0
    if normalized == "mm":
        return 1.0e-3
    raise CalculixCaseError("mesh coordinate units must be explicitly 'm' or 'mm'")


def _mesh_content_sha256(mesh: Any) -> str:
    digest = hashlib.sha256(b"step30-meshio-content-v1")
    points = np.ascontiguousarray(np.asarray(mesh.points, dtype=np.float64))
    digest.update(np.asarray(points.shape, dtype=np.int64).tobytes())
    digest.update(points.tobytes())
    for cell_type, data in _cell_blocks(mesh):
        digest.update(cell_type.encode("ascii"))
        contiguous = np.ascontiguousarray(data, dtype=np.int64)
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    cell_sets = getattr(mesh, "cell_sets", {}) or {}
    for name in sorted(cell_sets):
        digest.update(str(name).encode("utf-8"))
        values = cell_sets[name]
        if isinstance(values, Mapping):
            iterator = sorted(values.items(), key=lambda item: str(item[0]))
        else:
            iterator = enumerate(values)
        for block, rows in iterator:
            digest.update(str(block).encode("utf-8"))
            if rows is not None:
                digest.update(
                    np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes()
                )
    return digest.hexdigest()


def _read_mesh(mesh_or_path: Any) -> tuple[Any, Path | None, str]:
    if hasattr(mesh_or_path, "points") and hasattr(mesh_or_path, "cells"):
        return mesh_or_path, None, _mesh_content_sha256(mesh_or_path)
    path = Path(mesh_or_path).resolve()
    if not path.is_file():
        raise CalculixCaseError(f"mesh file does not exist: {path}")
    if path.suffix.lower() not in {".msh", ".vtk", ".vtu", ".xdmf", ".med"}:
        raise CalculixCaseError("mesh must be a Gmsh/meshio-supported finite-element file")
    try:
        import meshio
    except ImportError as exc:  # pragma: no cover
        raise CalculixCaseError("meshio is required to read structural meshes") from exc
    try:
        mesh = meshio.read(path)
    except Exception as exc:
        raise CalculixCaseError(f"meshio could not read structural mesh: {exc}") from exc
    return mesh, path, sha256_file(path)


def _cell_blocks(mesh: Any) -> list[tuple[str, np.ndarray]]:
    blocks: list[tuple[str, np.ndarray]] = []
    for block in mesh.cells:
        if hasattr(block, "type"):
            cell_type = str(block.type)
            raw_data = block.data
        else:
            cell_type = str(block[0])
            raw_data = block[1]
        data = np.asarray(raw_data, dtype=np.int64)
        blocks.append((cell_type, data))
    return blocks


def _group_rows(mesh: Any, group: str, block_index: int) -> np.ndarray:
    blocks = _cell_blocks(mesh)
    cell_sets = getattr(mesh, "cell_sets", {}) or {}
    rows: Any = []
    group_found = group in cell_sets
    if group_found:
        values = cell_sets[group]
        if isinstance(values, Mapping):
            rows = values.get(blocks[block_index][0], [])
        else:
            rows = values[block_index] if block_index < len(values) else []
            if rows is None:
                rows = []
    else:
        field_data = getattr(mesh, "field_data", {}) or {}
        physical = (getattr(mesh, "cell_data", {}) or {}).get("gmsh:physical")
        if group in field_data and physical is not None:
            definition = np.asarray(field_data[group], dtype=np.int64).reshape(-1)
            if len(definition) < 1:
                raise CalculixCaseError(f"Gmsh physical group is malformed: {group}")
            tag = int(definition[0])
            if block_index < len(physical):
                block_tags = np.asarray(physical[block_index], dtype=np.int64).reshape(-1)
                rows = np.flatnonzero(block_tags == tag)
            group_found = True
    if not group_found:
        raise CalculixCaseError(f"required named mesh group is missing: {group}")
    result = np.asarray(rows, dtype=np.int64).reshape(-1)
    n_rows = len(blocks[block_index][1])
    if np.any(result < 0) or np.any(result >= n_rows):
        raise CalculixCaseError(f"mesh group {group} has an out-of-range cell index")
    return np.unique(result)


def _group_members(mesh: Any, group: str) -> list[tuple[int, str, np.ndarray]]:
    members: list[tuple[int, str, np.ndarray]] = []
    for block_index, (cell_type, _) in enumerate(_cell_blocks(mesh)):
        rows = _group_rows(mesh, group, block_index)
        if len(rows):
            members.append((block_index, cell_type, rows))
    if not members:
        raise CalculixCaseError(f"named mesh group is empty: {group}")
    return members


def _nodes_from_group(mesh: Any, group: str) -> np.ndarray:
    blocks = _cell_blocks(mesh)
    nodes = [blocks[index][1][rows].reshape(-1) for index, _, rows in _group_members(mesh, group)]
    return np.unique(np.concatenate(nodes)).astype(np.int64)


def _format_set(values: Sequence[int], width: int = 16) -> list[str]:
    rows: list[str] = []
    values_list = list(values)
    for start in range(0, len(values_list), width):
        rows.append(", ".join(str(value) for value in values_list[start : start + width]))
    return rows


def _validate_material(material: Mapping[str, Any], name: str) -> dict[str, float]:
    required = ("young_modulus_pa", "poisson_ratio", "density_kg_m3")
    missing = [field for field in required if field not in material]
    if missing:
        raise CalculixCaseError(f"material {name} missing fields: {missing}")
    result = {field: float(material[field]) for field in required}
    if not all(math.isfinite(value) for value in result.values()):
        raise CalculixCaseError(f"material {name} contains non-finite values")
    if result["young_modulus_pa"] <= 0.0 or result["density_kg_m3"] <= 0.0:
        raise CalculixCaseError(f"material {name} modulus and density must be positive")
    if not -1.0 < result["poisson_ratio"] < 0.5:
        raise CalculixCaseError(f"material {name} Poisson ratio is invalid")
    return result


def inspect_structural_mesh(
    mesh_or_path: Any,
    *,
    coordinate_units: str,
    materials: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate C3D10 topology, named groups, and complete material domains."""
    mesh, source_path, mesh_hash = _read_mesh(mesh_or_path)
    scale = _unit_scale(coordinate_units)
    points_m = np.asarray(mesh.points, dtype=np.float64)[:, :3] * scale
    if points_m.ndim != 2 or points_m.shape[1] != 3 or len(points_m) == 0:
        raise CalculixCaseError("mesh points must have shape (N, 3)")
    if not np.isfinite(points_m).all():
        raise CalculixCaseError("mesh points contain non-finite values")
    blocks = _cell_blocks(mesh)
    tetra_blocks = [(index, data) for index, (kind, data) in enumerate(blocks) if kind == _TETRA_TYPE]
    unsupported_solids = sorted(
        {kind for kind, _ in blocks if kind.startswith(("tetra", "hexa", "wedge", "pyramid")) and kind != _TETRA_TYPE}
    )
    if unsupported_solids:
        raise CalculixCaseError(f"only C3D10/tetra10 solid cells are accepted, found {unsupported_solids}")
    if not tetra_blocks:
        raise CalculixCaseError("mesh contains no tetra10 cells for C3D10")
    for group in REQUIRED_GROUPS:
        _group_members(mesh, group)
    wetted_members = _group_members(mesh, "WETTED_SURFACE")
    if any(kind != _SURFACE_TYPE for _, kind, _ in wetted_members):
        raise CalculixCaseError("WETTED_SURFACE must contain triangle6 faces")
    if not any(kind == _SURFACE_TYPE for _, kind, _ in _group_members(mesh, "ROOT_INTERFACE")):
        raise CalculixCaseError("ROOT_INTERFACE must contain triangle6 faces")
    if not materials:
        raise CalculixCaseError("at least one named material domain is required")

    normalized_materials: dict[str, dict[str, float]] = {}
    for raw_name, value in materials.items():
        name = str(raw_name).strip()
        if not name or not all(char.isalnum() or char == "_" for char in name):
            raise CalculixCaseError(
                "material/domain names may contain only letters, digits, and underscore"
            )
        if name.upper() in {group.upper() for group in REQUIRED_GROUPS}:
            raise CalculixCaseError("a material domain collides with a required group name")
        normalized_materials[name] = _validate_material(value, name)
    tetra_offsets: dict[int, int] = {}
    tetra_rows: list[np.ndarray] = []
    next_element = 1
    for block_index, data in tetra_blocks:
        tetra_offsets[block_index] = next_element
        tetra_rows.append(data)
        next_element += len(data)
    tetra = np.vstack(tetra_rows)
    material_elements: dict[str, np.ndarray] = {}
    coverage = np.zeros(len(tetra), dtype=np.int64)
    global_offset = 0
    block_global_offset: dict[int, int] = {}
    for block_index, data in tetra_blocks:
        block_global_offset[block_index] = global_offset
        global_offset += len(data)
    for material_name in sorted(normalized_materials):
        members = _group_members(mesh, material_name)
        invalid = [kind for _, kind, _ in members if kind != _TETRA_TYPE]
        if invalid:
            raise CalculixCaseError(f"material domain {material_name} must contain only tetra10 cells")
        ids: list[np.ndarray] = []
        for block_index, _, rows in members:
            if block_index not in block_global_offset:
                raise CalculixCaseError(f"material domain {material_name} does not reference a tetra10 block")
            ids.append(rows + block_global_offset[block_index])
        global_ids = np.unique(np.concatenate(ids)).astype(np.int64)
        if len(global_ids) == 0:
            raise CalculixCaseError(f"material domain is empty: {material_name}")
        coverage[global_ids] += 1
        material_elements[material_name] = global_ids + 1
    if np.any(coverage != 1):
        raise CalculixCaseError("material domains must cover every C3D10 exactly once without overlap")

    root_nodes = _nodes_from_group(mesh, "ROOT_INTERFACE")
    ai5_nodes = _nodes_from_group(mesh, "AI5_04")
    surface_records: list[tuple[int, np.ndarray]] = []
    for block_index, kind, rows in wetted_members:
        block_data = blocks[block_index][1]
        for row in rows:
            surface_records.append((block_index, block_data[row]))
    boundary_lookup: dict[frozenset[int], tuple[int, int]] = {}
    for global_tet_id, nodes in enumerate(tetra, start=1):
        for face_number, local in enumerate(_C3D10_FACE_NODES, start=1):
            key = frozenset(int(nodes[index]) for index in local)
            if key in boundary_lookup:
                boundary_lookup[key] = (-1, -1)
            else:
                boundary_lookup[key] = (global_tet_id, face_number)
    wetted_owner: list[tuple[int, int]] = []
    wetted_faces: list[np.ndarray] = []
    seen_surface_faces: set[frozenset[int]] = set()
    for _, face_nodes in surface_records:
        if len(face_nodes) != 6 or len(np.unique(face_nodes)) != 6:
            raise CalculixCaseError("WETTED_SURFACE must contain valid six-node faces")
        key = frozenset(int(value) for value in face_nodes)
        if key in seen_surface_faces:
            raise CalculixCaseError("WETTED_SURFACE contains a duplicate face")
        seen_surface_faces.add(key)
        owner = boundary_lookup.get(key)
        if owner is None or owner == (-1, -1):
            raise CalculixCaseError("WETTED_SURFACE face has no unique C3D10 boundary owner")
        wetted_owner.append(owner)
        wetted_faces.append(np.asarray(face_nodes, dtype=np.int64))
    target_geometry = build_surface_geometry(
        points_m,
        [face[:3] for face in wetted_faces],
        coordinate_units="m",
    )
    for row, (element_id, face_number) in enumerate(wetted_owner):
        tetra_nodes = tetra[element_id - 1]
        corner_ids = set(_C3D10_FACES[face_number - 1])
        opposite_local = next(index for index in range(4) if index not in corner_ids)
        opposite = points_m[tetra_nodes[opposite_local]]
        inward = opposite - target_geometry["centers_m"][row]
        if float(np.dot(target_geometry["normals"][row], inward)) >= -1.0e-14:
            raise CalculixCaseError("WETTED_SURFACE face normal is not outward from its C3D10 owner")
    return {
        "mesh": mesh,
        "source_path": None if source_path is None else str(source_path),
        "mesh_sha256": mesh_hash,
        "points_m": points_m,
        "tetra10": tetra,
        "material_elements": material_elements,
        "materials": normalized_materials,
        "root_nodes": root_nodes,
        "ai5_nodes": ai5_nodes,
        "wetted_faces": wetted_faces,
        "wetted_owner": wetted_owner,
        "wetted_geometry": target_geometry,
        "wetted_topology_sha256": target_geometry["topology_sha256"],
        "n_nodes": int(len(points_m)),
        "n_elements": int(len(tetra)),
        "n_wetted_faces": int(len(wetted_faces)),
    }


def _solver_version(executable: str | None) -> tuple[str | None, str | None]:
    if executable is None:
        return None, None
    resolved = shutil.which(executable) if not Path(executable).is_file() else str(Path(executable).resolve())
    if resolved is None:
        return None, None
    version = "unknown"
    for args in ((resolved, "-v"), (resolved, "--version")):
        try:
            completed = subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)
        except (OSError, subprocess.SubprocessError):
            continue
        text = "\n".join((completed.stdout, completed.stderr)).strip()
        if text:
            version = text.splitlines()[0].strip()
            break
    return resolved, version


def generate_calculix_case(
    mesh_or_path: Any,
    pressure_load: Mapping[str, Any] | None,
    output_directory: str | Path,
    *,
    materials: Mapping[str, Mapping[str, Any]],
    coordinate_units: str,
    angular_speed_rad_s: float,
    rotation_origin_m: Sequence[float],
    rotation_axis: Sequence[float],
    case_name: str = "step30_structural",
    ccx_executable: str | None = "ccx",
    software_validation_only: bool = False,
    load_mode: str = "pressure_plus_centrifugal",
) -> dict[str, Any]:
    """Generate a provenance-bound C3D10 input deck; this does not run FEM.

    ``centrifugal_only_s1`` is the pressure-free 375-rpm sanity case. The
    legacy pressure-bearing path remains available for deferred S2 diagnostics.
    """
    try:
        if load_mode not in {"pressure_plus_centrifugal", "centrifugal_only_s1"}:
            raise CalculixCaseError("unsupported structural load mode")
        model = inspect_structural_mesh(
            mesh_or_path, coordinate_units=coordinate_units, materials=materials
        )
        pressure_is_test_only = False
        pressure = np.zeros(model["n_wetted_faces"], dtype=np.float64)
        load_hash = None
        load_origin = np.asarray(rotation_origin_m, dtype=np.float64)
        pressure_source_vtp_sha256 = None
        pressure_source_topology_sha256 = None
        pressure_mapping_resultants = None
        if load_mode == "pressure_plus_centrifugal":
            if not isinstance(pressure_load, Mapping):
                raise CalculixCaseError("pressure load is required for pressure-bearing mode")
            if (
                pressure_load.get("status") != "mapped"
                or pressure_load.get("pressure_gate") != "passed"
                or pressure_load.get("pressure_units") != "Pa"
                or pressure_load.get("coordinate_units") != "m"
            ):
                raise CalculixCaseError("pressure load must be a successful SI mapping")
            pressure_is_test_only = pressure_load.get("software_validation_only") is True
            if pressure_is_test_only and not software_validation_only:
                raise CalculixCaseError(
                    "software-validation pressure cannot generate production FEM evidence"
                )
            if pressure_load.get("target_topology_sha256") != model["wetted_topology_sha256"]:
                raise CalculixCaseError("pressure target topology differs from WETTED_SURFACE")
            pressure = np.asarray(pressure_load.get("pressure_pa"), dtype=np.float64)
            if pressure.shape != (model["n_wetted_faces"],) or not np.isfinite(pressure).all():
                raise CalculixCaseError("pressure count/values differ from WETTED_SURFACE")
            load_hash = pressure_load.get("load_sha256")
            if not is_sha256(load_hash):
                raise CalculixCaseError("pressure load has no valid SHA-256 identity")
            load_origin = np.asarray(pressure_load.get("origin_m"), dtype=np.float64)
            if load_origin.shape != (3,) or not np.isfinite(load_origin).all():
                raise CalculixCaseError("pressure mapping has no valid moment origin")
            actual_load_hash = pressure_load_sha256(
                pressure, model["wetted_geometry"], origin_m=load_origin
            )
            if load_hash != actual_load_hash:
                raise CalculixCaseError("pressure load SHA-256/content mismatch")
            if not pressure_is_test_only and not is_sha256(
                pressure_load.get("source_vtp_sha256")
            ):
                raise CalculixCaseError("production pressure load has no VTP identity")
            pressure_source_vtp_sha256 = pressure_load.get("source_vtp_sha256")
            pressure_source_topology_sha256 = pressure_load.get("source_topology_sha256")
            pressure_mapping_resultants = pressure_load.get("resultants")
        if not isinstance(software_validation_only, bool):
            raise CalculixCaseError("software_validation_only must be boolean")
        omega = float(angular_speed_rad_s)
        origin = np.asarray(rotation_origin_m, dtype=np.float64)
        axis = np.asarray(rotation_axis, dtype=np.float64)
        if not math.isfinite(omega) or omega < 0.0:
            raise CalculixCaseError("angular_speed_rad_s must be finite and non-negative")
        expected_s1_omega = 375.0 * 2.0 * math.pi / 60.0
        if load_mode == "centrifugal_only_s1" and not math.isclose(
            omega, expected_s1_omega, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise CalculixCaseError("centrifugal_only_s1 requires exactly 375 rpm")
        if origin.shape != (3,) or axis.shape != (3,) or not np.isfinite(origin).all() or not np.isfinite(axis).all():
            raise CalculixCaseError("rotation origin/axis must have three finite coordinates")
        norm = float(np.linalg.norm(axis))
        if norm <= 0.0:
            raise CalculixCaseError("rotation axis must be nonzero")
        axis /= norm
        if (
            not isinstance(case_name, str)
            or not case_name
            or not all(char.isalnum() or char in {"_", "-"} for char in case_name)
        ):
            raise CalculixCaseError(
                "case_name may contain only letters, digits, underscore, and hyphen"
            )

        output = Path(output_directory).resolve()
        output.mkdir(parents=True, exist_ok=True)
        input_path = output / f"{case_name}.inp"
        provenance_path = output / f"{case_name}.provenance.json"
        lines = [
            "** STEP30 STRUCTURAL CASE - SI UNITS (m, N, Pa, kg, s)",
            f"** mesh_sha256={model['mesh_sha256']}",
            f"** load_mode={load_mode}",
            f"** pressure_load_sha256={load_hash or 'none'}",
            "*HEADING",
            "Step30 provenance-bound C3D10 pressure and centrifugal analysis",
            "*NODE, NSET=ALL_NODES",
        ]
        for node_id, xyz in enumerate(model["points_m"], start=1):
            lines.append(f"{node_id}, {xyz[0]:.16e}, {xyz[1]:.16e}, {xyz[2]:.16e}")
        lines.append("*ELEMENT, TYPE=C3D10, ELSET=ALL_SOLID")
        for element_id, nodes in enumerate(model["tetra10"], start=1):
            lines.append(f"{element_id}, " + ", ".join(str(int(node) + 1) for node in nodes))
        lines.append("*NSET, NSET=ROOT_INTERFACE")
        lines.extend(_format_set((model["root_nodes"] + 1).tolist()))
        lines.append("*NSET, NSET=AI5_04")
        lines.extend(_format_set((model["ai5_nodes"] + 1).tolist()))
        lines.append("*SURFACE, TYPE=ELEMENT, NAME=WETTED_SURFACE")
        for element_id, face_number in model["wetted_owner"]:
            lines.append(f"{element_id}, S{face_number}")
        for material_name, element_ids in model["material_elements"].items():
            set_name = f"MAT_{material_name}".upper()
            lines.append(f"*ELSET, ELSET={set_name}")
            lines.extend(_format_set(element_ids.tolist()))
            material = model["materials"][material_name]
            lines.extend(
                [
                    f"*MATERIAL, NAME={material_name}",
                    "*ELASTIC",
                    f"{material['young_modulus_pa']:.16e}, {material['poisson_ratio']:.16e}",
                    "*DENSITY",
                    f"{material['density_kg_m3']:.16e}",
                    f"*SOLID SECTION, ELSET={set_name}, MATERIAL={material_name}",
                ]
            )
        lines.extend(["*BOUNDARY", "ROOT_INTERFACE, 1, 3, 0.0", "*STEP", "*STATIC"])
        lines.extend(
            [
                "*DLOAD",
                "ALL_SOLID, CENTRIF, "
                f"{omega * omega:.16e}, {origin[0]:.16e}, {origin[1]:.16e}, {origin[2]:.16e}, "
                f"{axis[0]:.16e}, {axis[1]:.16e}, {axis[2]:.16e}",
            ]
        )
        if load_mode == "pressure_plus_centrifugal":
            for (element_id, face_number), value in zip(model["wetted_owner"], pressure):
                lines.append(f"{element_id}, P{face_number}, {value:.16e}")
        lines.extend(
            [
                "*NODE FILE, NSET=ALL_NODES",
                "U, RF",
                "*EL FILE, ELSET=ALL_SOLID",
                "S, E",
                "*NODE PRINT, NSET=ROOT_INTERFACE, TOTALS=ONLY",
                "RF",
                "*EL PRINT, ELSET=ALL_SOLID, TOTALS=ONLY",
                "S",
                "*END STEP",
            ]
        )
        input_path.write_text("\n".join(lines) + "\n", encoding="ascii")
        input_hash = sha256_file(input_path)
        solver_path, solver_version = _solver_version(ccx_executable)
        solver_hash = None if solver_path is None else sha256_file(solver_path)
        provenance = {
            "version": VERSION,
            "status": "case_generated",
            "fem_gate": "blocked",
            "fem_gate_reason": "ccx_not_available" if solver_path is None else "solver_not_run_or_validated",
            "real_fem_evidence": False,
            "software_validation_only": bool(software_validation_only),
            "case_name": case_name,
            "load_mode": load_mode,
            "mesh_path": model["source_path"],
            "mesh_sha256": model["mesh_sha256"],
            "mesh_units_input": coordinate_units,
            "analysis_units": {"length": "m", "force": "N", "stress": "Pa", "mass": "kg", "time": "s"},
            "wetted_topology_sha256": model["wetted_topology_sha256"],
            "pressure_source_vtp_sha256": pressure_source_vtp_sha256,
            "pressure_source_topology_sha256": pressure_source_topology_sha256,
            "pressure_load_sha256": load_hash,
            "pressure_moment_origin_m": (
                load_origin.tolist()
                if load_mode == "pressure_plus_centrifugal"
                else None
            ),
            "pressure_mapping_resultants": pressure_mapping_resultants,
            "input_path": str(input_path),
            "input_sha256": input_hash,
            "generator_sha256": sha256_file(Path(__file__)),
            "ccx_executable": solver_path,
            "ccx_executable_sha256": solver_hash,
            "ccx_version_at_generation": solver_version,
            "n_nodes": model["n_nodes"],
            "n_c3d10": model["n_elements"],
            "n_wetted_faces": model["n_wetted_faces"],
            "named_groups": list(REQUIRED_GROUPS),
            "material_domains": sorted(model["materials"]),
            "angular_speed_rad_s": omega,
            "rotation_origin_m": origin.tolist(),
            "rotation_axis": axis.tolist(),
        }
        provenance["provenance_sha256"] = _canonical_sha256(provenance)
        provenance_path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="ascii",
        )
        return provenance
    except (CalculixCaseError, OSError, ValueError) as exc:
        return {
            "version": VERSION,
            "status": "blocked",
            "fem_gate": "blocked",
            "real_fem_evidence": False,
            "reason": str(exc),
        }
