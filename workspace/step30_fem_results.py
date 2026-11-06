"""CalculiX result parsing with immutable evidence bindings."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


VERSION = "step30-fem-results-v1"
_NON_REAL = frozenset(
    {"synthetic", "parametric", "analytical", "simulation_proxy", "fake", "generated"}
)
_HEX = frozenset("0123456789abcdef")


class FEMResultsError(ValueError):
    """Raised when a CalculiX result package is malformed or not admissible."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise FEMResultsError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise FEMResultsError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise FEMResultsError(f"{name} must be finite")
    return number


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in _HEX for char in value.lower())
    ):
        raise FEMResultsError(f"{name} must be a valid SHA-256")
    return value.lower()


def _validated_provenance(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        provenance = dict(value)
    else:
        path = Path(value).resolve()
        if not path.is_file():
            raise FEMResultsError(f"case provenance file does not exist: {path}")
        provenance = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(provenance, Mapping):
        raise FEMResultsError("case provenance must be an object")
    provenance = dict(provenance)
    stored_hash = _sha256(provenance.get("provenance_sha256"), "provenance_sha256")
    unsigned = dict(provenance)
    unsigned.pop("provenance_sha256", None)
    if canonical_sha256(unsigned) != stored_hash:
        raise FEMResultsError("case provenance SHA-256/content mismatch")
    if provenance.get("status") != "case_generated":
        raise FEMResultsError("case provenance status is not case_generated")
    if provenance.get("fem_gate") != "blocked":
        raise FEMResultsError("generated case must remain blocked until result validation")
    if provenance.get("software_validation_only") is True:
        raise FEMResultsError("software-validation case is ineligible as real FEM evidence")
    if provenance.get("real_fem_evidence") is not False:
        raise FEMResultsError("case generation cannot claim real FEM evidence")
    input_path = Path(str(provenance.get("input_path", ""))).resolve()
    if not input_path.is_file():
        raise FEMResultsError("case input file is missing")
    input_hash = _sha256(provenance.get("input_sha256"), "input_sha256")
    if sha256_file(input_path) != input_hash:
        raise FEMResultsError("case input file hash mismatch")
    solver_path = provenance.get("ccx_executable")
    if not isinstance(solver_path, str) or not Path(solver_path).is_file():
        raise FEMResultsError("ccx was not available when the case was generated")
    solver_hash = _sha256(
        provenance.get("ccx_executable_sha256"), "ccx_executable_sha256"
    )
    if sha256_file(solver_path) != solver_hash:
        raise FEMResultsError("ccx executable hash mismatch")
    _sha256(provenance.get("mesh_sha256"), "mesh_sha256")
    load_mode = provenance.get("load_mode")
    if load_mode not in {"pressure_plus_centrifugal", "centrifugal_only_s1"}:
        raise FEMResultsError("case provenance has unsupported load_mode")
    if load_mode == "pressure_plus_centrifugal":
        _sha256(provenance.get("pressure_load_sha256"), "pressure_load_sha256")
        _sha256(
            provenance.get("pressure_source_vtp_sha256"),
            "pressure_source_vtp_sha256",
        )
        _reaction_vector(
            provenance.get("pressure_moment_origin_m"),
            "pressure_moment_origin_m",
        )
    elif provenance.get("pressure_load_sha256") is not None:
        raise FEMResultsError(
            "centrifugal_only_s1 provenance must not declare a pressure load"
        )
    if load_mode == "centrifugal_only_s1":
        if provenance.get("pressure_moment_origin_m") is not None:
            raise FEMResultsError(
                "centrifugal_only_s1 provenance must not declare a pressure moment origin"
            )
        expected_omega = 375.0 * 2.0 * math.pi / 60.0
        actual_omega = _finite(
            provenance.get("angular_speed_rad_s"),
            "angular_speed_rad_s",
        )
        if not math.isclose(actual_omega, expected_omega, rel_tol=0.0, abs_tol=1.0e-12):
            raise FEMResultsError("centrifugal_only_s1 provenance requires exactly 375 rpm")
        _reaction_vector(provenance.get("rotation_origin_m"), "rotation_origin_m")
    return provenance


def project_ai5_stress(stress_tensor: Any, direction: Any) -> float:
    """Project the Cauchy stress tensor onto the declared AI5-04 direction."""
    tensor = np.asarray(stress_tensor, dtype=np.float64)
    vector = np.asarray(direction, dtype=np.float64)
    if tensor.shape != (3, 3) or vector.shape != (3,) or not np.isfinite(tensor).all() or not np.isfinite(vector).all():
        raise FEMResultsError("AI5 stress tensor/direction must be finite 3x3/3-vector")
    length = float(np.linalg.norm(vector))
    if length <= 0.0:
        raise FEMResultsError("AI5 direction must be nonzero")
    normalized = vector / length
    symmetric = 0.5 * (tensor + tensor.T)
    return float(normalized @ symmetric @ normalized)


def _parse_dat(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lower = text.lower()
    convergence = "converged" in lower and "not converged" not in lower
    if "not converged" in lower or "no convergence" in lower:
        convergence = False
    node_stresses: list[dict[str, Any]] = []
    # Accept a deliberately documented simple export: node_id followed by six
    # stress components. Native .frd binary/blocked records must be converted by ccx.
    marker = re.compile(r"^\s*(\d+)\s+([-+0-9.eE]+(?:\s+[-+0-9.eE]+){5})\s*$", re.MULTILINE)
    for match in marker.finditer(text):
        values = [float(item) for item in match.group(2).split()]
        sxx, syy, szz, sxy, syz, sxz = values
        node_stresses.append({"node_id": int(match.group(1)), "tensor": [[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]]})
    return {
        "version": VERSION,
        "result_format": "calculix_dat",
        "solver_status": "converged" if convergence else "not_converged",
        "convergence": {"converged": convergence},
        "node_stresses": node_stresses,
        "source_path": str(path),
        "output_sha256": sha256_file(path),
        "real_fem_evidence": True,
        "evidence_status": "real_solver",
    }


def parse_calculix_results(results: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Parse a JSON export or simple CalculiX ``.dat`` result into one schema."""
    if isinstance(results, Mapping):
        document = dict(results)
        source_path = document.get("source_path")
        if source_path and Path(str(source_path)).is_file():
            document.setdefault("output_sha256", sha256_file(source_path))
    else:
        path = Path(results).resolve()
        if not path.is_file():
            raise FEMResultsError(f"CalculiX result file does not exist: {path}")
        if path.suffix.lower() == ".json":
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise FEMResultsError(f"cannot parse FEM JSON result: {path}") from exc
            if not isinstance(document, Mapping):
                raise FEMResultsError("FEM JSON result must be an object")
            document = dict(document)
            document["result_manifest_path"] = str(path)
            document["result_manifest_sha256"] = sha256_file(path)
        elif path.suffix.lower() == ".dat":
            document = _parse_dat(path)
        else:
            raise FEMResultsError("CalculiX results must be a JSON export or .dat file")
    if document.get("result_format") not in {None, "calculix_dat", "calculix_json"}:
        raise FEMResultsError("unsupported CalculiX result format")
    if document.get("evidence_status") in _NON_REAL or document.get("status") in _NON_REAL or document.get("fem_status") in _NON_REAL:
        raise FEMResultsError("parametric/synthetic FEM results are ineligible")
    return document


def _reaction_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise FEMResultsError(f"{name} must be a finite 3-vector")
    return array


def _validate_surface_results(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FEMResultsError("surface_results must be an object")
    parent = np.asarray(value.get("parent_face_id"), dtype=np.int64)
    stress = np.asarray(value.get("von_mises_stress_pa"), dtype=np.float64)
    fatigue = np.asarray(value.get("fatigue_damage"), dtype=np.float64)
    if (
        parent.ndim != 1
        or len(parent) == 0
        or stress.shape != parent.shape
        or fatigue.shape != parent.shape
    ):
        raise FEMResultsError("surface result arrays have inconsistent shapes")
    if not np.array_equal(parent, np.arange(len(parent), dtype=np.int64)):
        raise FEMResultsError(
            "surface parent-face IDs must be canonical, ordered, and complete"
        )
    if (
        not np.isfinite(stress).all()
        or not np.isfinite(fatigue).all()
        or np.any(stress < 0.0)
        or np.any(fatigue < 0.0)
    ):
        raise FEMResultsError("surface stress/fatigue fields are invalid")
    topology_hash = _sha256(value.get("topology_sha256"), "surface topology_sha256")
    return {
        "parent_face_id": parent.tolist(),
        "von_mises_stress_pa": stress.tolist(),
        "fatigue_damage": fatigue.tolist(),
        "topology_sha256": topology_hash,
    }


def _validate_mesh_metrics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FEMResultsError("mesh_metrics must be an object")
    metrics = dict(value)
    if "characteristic_size_m" in metrics:
        size = _finite(metrics["characteristic_size_m"], "characteristic_size_m")
        if size <= 0.0:
            raise FEMResultsError("characteristic_size_m must be positive")
        metrics["characteristic_size_m"] = size
    elif "n_elements" in metrics:
        count = _finite(metrics["n_elements"], "n_elements")
        if count <= 0.0 or not count.is_integer():
            raise FEMResultsError("n_elements must be a positive integer")
        metrics["n_elements"] = int(count)
    else:
        raise FEMResultsError("mesh_metrics needs characteristic_size_m or n_elements")
    return metrics


def _check_convergence(document: Mapping[str, Any]) -> dict[str, Any]:
    convergence = document.get("convergence")
    if not isinstance(convergence, Mapping):
        raise FEMResultsError("convergence evidence is missing")
    if convergence.get("converged") is not True:
        raise FEMResultsError("CalculiX convergence gate failed")
    if convergence.get("termination") != "normal":
        raise FEMResultsError("CalculiX did not report normal termination")
    increments = _finite(
        convergence.get("increments_completed"),
        "convergence.increments_completed",
    )
    if increments <= 0.0 or not increments.is_integer():
        raise FEMResultsError("convergence.increments_completed must be positive integer")
    nonconverged = _finite(
        convergence.get("nonconverged_increments", 0),
        "convergence.nonconverged_increments",
    )
    fatal_errors = _finite(
        convergence.get("fatal_errors", 0), "convergence.fatal_errors"
    )
    if nonconverged != 0.0 or fatal_errors != 0.0:
        raise FEMResultsError("CalculiX convergence history contains failures")
    result = dict(convergence)
    for key in ("iterations", "residual_norm", "increment_norm"):
        if key in convergence:
            value = _finite(convergence[key], f"convergence.{key}")
            if value < 0.0:
                raise FEMResultsError(f"convergence.{key} must be non-negative")
            result[key] = value
    result["increments_completed"] = int(increments)
    result["nonconverged_increments"] = int(nonconverged)
    result["fatal_errors"] = int(fatal_errors)
    return result


def _check_ai5(document: Mapping[str, Any]) -> dict[str, Any]:
    ai5 = document.get("ai5_04")
    if not isinstance(ai5, Mapping):
        raise FEMResultsError("AI5-04 direction projection evidence is missing")
    if ai5.get("calibrated_from_validation_point") is True or ai5.get("calibration_source") in {"AI5_04", "validation_point", "verification_point"}:
        raise FEMResultsError("AI5-04 validation point was used for back-calibration")
    if ai5.get("projection_method") not in {"tensor_direction", "n_dot_sigma_n"}:
        raise FEMResultsError("AI5-04 projection method must be tensor_direction")
    if "stress_tensor_pa" not in ai5:
        raise FEMResultsError("AI5-04 stress tensor is missing")
    computed = project_ai5_stress(ai5["stress_tensor_pa"], ai5.get("direction"))
    projected_number = _finite(
        ai5.get("projected_stress_pa"), "AI5-04 projected stress"
    )
    if not math.isclose(projected_number, computed, rel_tol=1.0e-10, abs_tol=1.0e-3):
        raise FEMResultsError("AI5-04 projected stress/tensor mismatch")
    direction = np.asarray(ai5.get("direction"), dtype=np.float64)
    if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) <= 0.0:
        raise FEMResultsError("AI5-04 direction is invalid")
    result = dict(ai5)
    result["projected_stress_pa"] = projected_number
    result["direction_unit"] = (direction / np.linalg.norm(direction)).tolist()
    return result


def validate_calculix_results(
    results: str | Path | Mapping[str, Any],
    provenance: str | Path | Mapping[str, Any],
    *,
    expected_ccx_version: str | None = None,
    expected_mesh_sha256: str | None = None,
    expected_load_sha256: str | None = None,
    expected_input_sha256: str | None = None,
    expected_output_sha256: str | None = None,
    reaction_tolerance: float = 1.0e-5,
) -> dict[str, Any]:
    """Validate solver results and return a gate report.

    A report only passes with explicit real-solver evidence. Parametric fields,
    including the existing ``step30_fem_stress_field`` output, are rejected.
    """
    try:
        document = parse_calculix_results(results)
        prov = _validated_provenance(provenance)
        bindings = [
            ("mesh_sha256", expected_mesh_sha256 or prov.get("mesh_sha256")),
            ("input_sha256", expected_input_sha256 or prov.get("input_sha256")),
        ]
        if prov.get("load_mode") == "pressure_plus_centrifugal":
            bindings.append(
                (
                    "pressure_load_sha256",
                    expected_load_sha256 or prov.get("pressure_load_sha256"),
                )
            )
        for field, expected in bindings:
            actual = _sha256(document.get(field), field)
            if expected is None or actual != _sha256(expected, f"expected {field}"):
                raise FEMResultsError(f"{field} mismatch")
        if prov.get("load_mode") == "centrifugal_only_s1":
            if document.get("pressure_load_sha256") not in {None, "none"}:
                raise FEMResultsError(
                    "centrifugal_only_s1 results must not declare a pressure load"
                )

        output_path_value = document.get("solver_output_path", document.get("source_path"))
        if not isinstance(output_path_value, str):
            raise FEMResultsError("solver output path is missing")
        output_path = Path(output_path_value).resolve()
        if not output_path.is_file():
            raise FEMResultsError("solver output file is missing")
        output_hash = _sha256(document.get("output_sha256"), "output_sha256")
        if sha256_file(output_path) != output_hash:
            raise FEMResultsError("solver output file hash mismatch")
        if expected_output_sha256 is not None and output_hash != _sha256(
            expected_output_sha256, "expected_output_sha256"
        ):
            raise FEMResultsError("output hash mismatch")

        ccx_version = document.get("ccx_version")
        if not isinstance(ccx_version, str) or not ccx_version.strip():
            raise FEMResultsError("ccx version evidence is missing")
        ccx_version = ccx_version.strip()
        declared_version = prov.get("ccx_version_at_generation")
        if not isinstance(declared_version, str) or declared_version in {"", "unknown"}:
            raise FEMResultsError("case provenance has no usable ccx version")
        if expected_ccx_version is not None and expected_ccx_version not in ccx_version:
            raise FEMResultsError("ccx version mismatch")
        if declared_version not in ccx_version and ccx_version not in declared_version:
            raise FEMResultsError("ccx version differs from case provenance")
        if document.get("solver_status") != "converged":
            raise FEMResultsError("solver status is not converged")
        convergence = _check_convergence(document)

        reactions = document.get("reactions")
        if not isinstance(reactions, Mapping):
            raise FEMResultsError("reaction balance evidence is missing")
        applied_force = _reaction_vector(
            reactions.get("applied_force_n"), "applied_force_n"
        )
        reaction_force = _reaction_vector(
            reactions.get("reaction_force_n"), "reaction_force_n"
        )
        applied_moment = _reaction_vector(
            reactions.get("applied_moment_nm"), "applied_moment_nm"
        )
        reaction_moment = _reaction_vector(
            reactions.get("reaction_moment_nm"), "reaction_moment_nm"
        )
        reaction_origin = _reaction_vector(reactions.get("origin_m"), "reaction origin_m")
        origin_field = (
            "pressure_moment_origin_m"
            if prov.get("load_mode") == "pressure_plus_centrifugal"
            else "rotation_origin_m"
        )
        case_origin = _reaction_vector(
            prov.get(origin_field), f"case {origin_field}"
        )
        if not np.allclose(reaction_origin, case_origin, rtol=0.0, atol=1.0e-12):
            raise FEMResultsError(
                f"reaction and {origin_field} moments use different origins"
            )
        force_error = float(
            np.linalg.norm(applied_force + reaction_force)
            / max(np.linalg.norm(applied_force), 1.0)
        )
        moment_error = float(
            np.linalg.norm(applied_moment + reaction_moment)
            / max(np.linalg.norm(applied_moment), 1.0)
        )
        if force_error > reaction_tolerance:
            raise FEMResultsError(f"reaction force balance failed: {force_error:.3e}")
        if moment_error > reaction_tolerance:
            raise FEMResultsError(f"reaction moment balance failed: {moment_error:.3e}")

        if document.get("stress_units") != "Pa":
            raise FEMResultsError("FEM stress units must be Pa")
        ai5 = _check_ai5(document)
        surface_results = _validate_surface_results(document.get("surface_results"))
        if surface_results["topology_sha256"] != prov.get("wetted_topology_sha256"):
            raise FEMResultsError("surface result topology differs from WETTED_SURFACE")
        if len(surface_results["parent_face_id"]) != int(prov.get("n_wetted_faces", -1)):
            raise FEMResultsError("surface results do not cover the full WETTED_SURFACE")
        mesh_metrics = _validate_mesh_metrics(document.get("mesh_metrics"))
        if (
            document.get("real_fem_evidence") is not True
            or document.get("evidence_status") != "real_solver"
        ):
            raise FEMResultsError("results are not marked as real CalculiX evidence")
        result = {
            "version": VERSION,
            "status": "validated",
            "fem_gate": "passed",
            "real_fem_evidence": True,
            "evidence_status": "real_solver",
            "ccx_version": ccx_version,
            "ccx_executable_sha256": prov.get("ccx_executable_sha256"),
            "load_mode": prov.get("load_mode"),
            "convergence": convergence,
            "reactions": {
                **dict(reactions),
                "relative_force_balance_error": force_error,
                "relative_moment_balance_error": moment_error,
            },
            "ai5_04": ai5,
            "mesh_metrics": mesh_metrics,
            "surface_results": surface_results,
            "mesh_sha256": prov.get("mesh_sha256"),
            "pressure_load_sha256": prov.get("pressure_load_sha256"),
            "pressure_source_vtp_sha256": prov.get("pressure_source_vtp_sha256"),
            "pressure_mapping_resultants": prov.get("pressure_mapping_resultants"),
            "input_sha256": prov.get("input_sha256"),
            "output_sha256": output_hash,
            "solver_output_path": str(output_path),
            "case_provenance_sha256": prov.get("provenance_sha256"),
        }
        result["result_validation_sha256"] = canonical_sha256(result)
        return result
    except (FEMResultsError, OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {
            "version": VERSION,
            "status": "blocked",
            "fem_gate": "blocked",
            "real_fem_evidence": False,
            "reason": str(exc),
        }


def save_result_report(report: Mapping[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(dict(report), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
