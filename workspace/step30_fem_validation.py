"""Three-mesh structural convergence and independent AI5-04 validation gates."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


VERSION = "step30-fem-validation-v1"
_INELIGIBLE = frozenset(
    {"synthetic", "parametric", "analytical", "simulation_proxy", "fake", "generated"}
)
_HEX = frozenset("0123456789abcdef")


class FEMValidationError(ValueError):
    """Raised when real FEM convergence/validation evidence is incomplete."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise FEMValidationError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise FEMValidationError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise FEMValidationError(f"{name} must be finite" + (" and positive" if positive else ""))
    return number


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in _HEX for char in value.lower())
    ):
        raise FEMValidationError(f"{name} must be a valid SHA-256")
    return value.lower()


def _eligible_real_fem(result: Mapping[str, Any], index: int) -> None:
    if result.get("fem_gate") != "passed" or result.get("status") not in {"validated", "passed"}:
        raise FEMValidationError(f"mesh result {index} has not passed its FEM gate")
    if result.get("real_fem_evidence") is not True:
        raise FEMValidationError(f"mesh result {index} is not real FEM evidence")
    declared = {
        str(result.get(field, "")).strip().lower()
        for field in ("evidence_status", "source_type", "fem_status", "status")
    }
    if declared & _INELIGIBLE:
        raise FEMValidationError(f"mesh result {index} is parametric/synthetic and ineligible")
    load_mode = str(result.get("load_mode", "")).strip()
    if load_mode not in {"pressure_plus_centrifugal", "centrifugal_only_s1"}:
        raise FEMValidationError(f"mesh result {index} has unsupported load_mode")
    for field in (
        "mesh_sha256",
        "input_sha256",
        "output_sha256",
        "case_provenance_sha256",
        "ccx_executable_sha256",
        "result_validation_sha256",
    ):
        _sha256(result.get(field), f"mesh result {index} {field}")
    if load_mode == "pressure_plus_centrifugal":
        _sha256(
            result.get("pressure_load_sha256"),
            f"mesh result {index} pressure_load_sha256",
        )
        _sha256(
            result.get("pressure_source_vtp_sha256"),
            f"mesh result {index} pressure_source_vtp_sha256",
        )
    elif result.get("pressure_load_sha256") not in {None, "none"}:
        raise FEMValidationError(
            "centrifugal_only_s1 FEM result must not declare a pressure load"
        )
    unsigned = dict(result)
    stored_validation_hash = unsigned.pop("result_validation_sha256")
    if _canonical_sha256(unsigned) != stored_validation_hash:
        raise FEMValidationError(
            f"mesh result {index} validation SHA-256/content mismatch"
        )
    if str(result.get("evidence_status", "")).lower() != "real_solver":
        raise FEMValidationError(f"mesh result {index} is not real solver evidence")
    output_path = Path(str(result.get("solver_output_path", ""))).resolve()
    if not output_path.is_file() or _file_sha256(output_path) != result.get(
        "output_sha256"
    ):
        raise FEMValidationError(f"mesh result {index} solver output hash is not verifiable")
    ai5 = result.get("ai5_04")
    if not isinstance(ai5, Mapping):
        raise FEMValidationError(f"mesh result {index} has no AI5-04 projection")
    if ai5.get("calibrated_from_validation_point") is True or ai5.get("calibration_source") in {"AI5_04", "validation_point", "verification_point"}:
        raise FEMValidationError("AI5-04 was used to back-calibrate an FEM result")
    if ai5.get("projection_method") not in {"tensor_direction", "n_dot_sigma_n"}:
        raise FEMValidationError("AI5-04 result is not a direction projection")


def validate_three_mesh_convergence(
    mesh_results: Sequence[Mapping[str, Any]],
    *,
    maximum_fine_medium_difference: float = 0.05,
    minimum_refinement_ratio: float = 1.2,
) -> dict[str, Any]:
    """Validate an ordered coarse/medium/fine triplet using AI5 projected stress.

    The triplet must be independent solver runs with three unique meshes,
    inputs, and outputs. The pressure-load identity must be identical so that
    the comparison isolates spatial discretization.
    """
    if len(mesh_results) != 3:
        raise FEMValidationError("three-mesh convergence requires exactly three results")
    normalized = [dict(result) for result in mesh_results]
    for index, result in enumerate(normalized):
        _eligible_real_fem(result, index)
    for field in ("mesh_sha256", "input_sha256", "output_sha256"):
        values = [str(result[field]) for result in normalized]
        if len(set(values)) != 3:
            raise FEMValidationError(f"three-mesh results require three unique {field} values")
    load_modes = {str(result.get("load_mode", "")) for result in normalized}
    if len(load_modes) != 1 or "" in load_modes:
        raise FEMValidationError("three-mesh results must use the same structural load mode")
    load_mode = next(iter(load_modes))
    load_hashes = [result.get("pressure_load_sha256") for result in normalized]
    if load_mode == "pressure_plus_centrifugal":
        if any(value is None for value in load_hashes):
            raise FEMValidationError("pressure-bearing runs require pressure_load_sha256")
        if len({str(value) for value in load_hashes}) != 1:
            raise FEMValidationError(
                "three-mesh pressure-bearing runs must use the same pressure load identity"
            )
        pressure_source_hashes = {
            _sha256(result.get("pressure_source_vtp_sha256"), "pressure source VTP SHA-256")
            for result in normalized
        }
        if len(pressure_source_hashes) != 1:
            raise FEMValidationError(
                "three-mesh results must use the same CFD pressure source identity"
            )
    elif load_mode == "centrifugal_only_s1":
        if any(value not in {None, "none"} for value in load_hashes):
            raise FEMValidationError(
                "centrifugal_only_s1 runs must not declare pressure loads"
            )
        pressure_source_hashes = {None}
    else:
        raise FEMValidationError("unsupported structural load mode")
    solver_hashes = {str(result["ccx_executable_sha256"]) for result in normalized}
    solver_versions = {str(result.get("ccx_version", "")) for result in normalized}
    if len(solver_hashes) != 1 or len(solver_versions) != 1 or "" in solver_versions:
        raise FEMValidationError(
            "three-mesh results must use the same CalculiX executable and version"
        )
    directions = [
        np.asarray(result["ai5_04"].get("direction_unit"), dtype=np.float64)
        for result in normalized
    ]
    if any(direction.shape != (3,) for direction in directions) or any(
        not np.allclose(directions[0], direction, rtol=0.0, atol=1.0e-12)
        for direction in directions[1:]
    ):
        raise FEMValidationError("three-mesh AI5-04 projection directions differ")

    sizes: list[float] = []
    stresses: list[float] = []
    for index, result in enumerate(normalized):
        mesh = result.get("mesh_metrics")
        if not isinstance(mesh, Mapping):
            raise FEMValidationError(f"mesh result {index} has no mesh_metrics")
        if "characteristic_size_m" in mesh:
            size = _number(mesh["characteristic_size_m"], "characteristic_size_m", positive=True)
        elif "n_elements" in mesh:
            size = _number(mesh["n_elements"], "n_elements", positive=True) ** (-1.0 / 3.0)
        else:
            raise FEMValidationError("mesh_metrics requires characteristic_size_m or n_elements")
        sizes.append(size)
        stresses.append(_number(result["ai5_04"].get("projected_stress_pa"), "AI5 stress"))
    if not sizes[0] > sizes[1] > sizes[2]:
        raise FEMValidationError("mesh results must be ordered coarse, medium, fine")
    r21 = sizes[0] / sizes[1]
    r32 = sizes[1] / sizes[2]
    if min(r21, r32) < minimum_refinement_ratio:
        raise FEMValidationError("mesh refinement ratio is too small")
    delta_coarse = stresses[1] - stresses[0]
    delta_fine = stresses[2] - stresses[1]
    monotonic = bool(delta_coarse == 0.0 or delta_fine == 0.0 or delta_coarse * delta_fine > 0.0)
    if not monotonic:
        raise FEMValidationError("AI5 projected stress does not converge monotonically")
    fine_medium = abs(stresses[2] - stresses[1]) / max(abs(stresses[2]), 1.0)
    if fine_medium > maximum_fine_medium_difference:
        raise FEMValidationError(
            f"fine/medium projected-stress difference exceeds limit: {fine_medium:.6f}"
        )
    denominator = stresses[1] - stresses[0]
    observed_order: float | None = None
    richardson_stress: float | None = None
    if abs(denominator) > 1.0e-30 and abs(delta_fine / denominator) > 0.0:
        average_ratio = math.sqrt(r21 * r32)
        observed_order = abs(math.log(abs(delta_fine / denominator)) / math.log(average_ratio))
        if math.isfinite(observed_order) and observed_order > 0.0:
            richardson_stress = stresses[2] + (stresses[2] - stresses[1]) / (average_ratio**observed_order - 1.0)
    return {
        "status": "passed",
        "mesh_convergence_gate": "passed",
        "mesh_hashes": [result["mesh_sha256"] for result in normalized],
        "input_hashes": [result["input_sha256"] for result in normalized],
        "output_hashes": [result["output_sha256"] for result in normalized],
        "pressure_load_sha256": load_hashes,
        "pressure_source_vtp_sha256": next(iter(pressure_source_hashes)),
        "characteristic_sizes": sizes,
        "ai5_projected_stress_pa": stresses,
        "refinement_ratios": [r21, r32],
        "fine_medium_relative_difference": fine_medium,
        "monotonic": monotonic,
        "observed_order": observed_order,
        "richardson_projected_stress_pa": richardson_stress,
    }


def validate_ai5_measurement(
    fine_result: Mapping[str, Any],
    measurement: Mapping[str, Any],
    *,
    maximum_relative_error: float = 0.10,
    minimum_direction_alignment: float = 0.999,
) -> dict[str, Any]:
    """Compare, never calibrate, an FEM direction projection to independent data."""
    _eligible_real_fem(fine_result, 2)
    if measurement.get("status") != "measured" or measurement.get("independent") is not True:
        raise FEMValidationError("AI5-04 measurement must be independent measured evidence")
    if measurement.get("used_for_calibration") is True:
        raise FEMValidationError("AI5-04 measurement was used for calibration")
    if measurement.get("units") != "Pa":
        raise FEMValidationError("AI5-04 measurement units must be Pa")
    measurement_hash = _sha256(
        measurement.get("measurement_sha256"), "measurement_sha256"
    )
    signed = dict(measurement)
    signed.pop("measurement_sha256", None)
    if _canonical_sha256(signed) != measurement_hash:
        raise FEMValidationError("AI5-04 measurement SHA-256/content mismatch")
    measured_direction = np.asarray(measurement.get("direction"), dtype=np.float64)
    predicted_direction = np.asarray(
        fine_result["ai5_04"].get("direction_unit"), dtype=np.float64
    )
    if (
        measured_direction.shape != (3,)
        or predicted_direction.shape != (3,)
        or not np.isfinite(measured_direction).all()
        or not np.isfinite(predicted_direction).all()
        or np.linalg.norm(measured_direction) <= 0.0
        or np.linalg.norm(predicted_direction) <= 0.0
    ):
        raise FEMValidationError("AI5-04 measured/predicted directions are invalid")
    measured_direction /= np.linalg.norm(measured_direction)
    predicted_direction /= np.linalg.norm(predicted_direction)
    direction_alignment = abs(float(np.dot(measured_direction, predicted_direction)))
    if direction_alignment < minimum_direction_alignment:
        raise FEMValidationError("AI5-04 measurement and FEM projection directions differ")
    measured = _number(
        measurement.get("projected_stress_pa"), "measured AI5 stress"
    )
    predicted = _number(
        fine_result["ai5_04"].get("projected_stress_pa"), "predicted AI5 stress"
    )
    error = abs(predicted - measured) / max(abs(measured), 1.0)
    return {
        "status": "passed" if error <= maximum_relative_error else "failed",
        "ai5_validation_gate": "passed" if error <= maximum_relative_error else "failed",
        "predicted_projected_stress_pa": predicted,
        "measured_projected_stress_pa": measured,
        "relative_error": error,
        "maximum_relative_error": float(maximum_relative_error),
        "projection_method": fine_result["ai5_04"].get("projection_method"),
        "direction_alignment": direction_alignment,
        "calibrated_from_validation_point": False,
        "measurement_sha256": measurement_hash,
    }


def validate_fem_evidence(
    mesh_results: Sequence[Mapping[str, Any]],
    measurement: Mapping[str, Any],
    *,
    maximum_fine_medium_difference: float = 0.05,
    maximum_ai5_relative_error: float = 0.10,
) -> dict[str, Any]:
    """Build the only FEM report eligible for downstream protection design."""
    try:
        convergence = validate_three_mesh_convergence(
            mesh_results,
            maximum_fine_medium_difference=maximum_fine_medium_difference,
        )
        ai5 = validate_ai5_measurement(
            mesh_results[-1], measurement, maximum_relative_error=maximum_ai5_relative_error
        )
        if ai5["ai5_validation_gate"] != "passed":
            raise FEMValidationError("AI5-04 independent validation failed")
        fine = dict(mesh_results[-1])
        surface_results = fine.get("surface_results")
        if not isinstance(surface_results, Mapping):
            raise FEMValidationError("fine-grid FEM surface field is missing")
        parent_ids = np.asarray(surface_results.get("parent_face_id"), dtype=np.int64)
        stress = np.asarray(surface_results.get("von_mises_stress_pa"), dtype=np.float64)
        fatigue = np.asarray(surface_results.get("fatigue_damage"), dtype=np.float64)
        if parent_ids.ndim != 1 or len(parent_ids) == 0 or stress.shape != parent_ids.shape or fatigue.shape != parent_ids.shape:
            raise FEMValidationError("fine-grid FEM surface field shapes are inconsistent")
        if len(np.unique(parent_ids)) != len(parent_ids) or np.any(parent_ids < 0):
            raise FEMValidationError("FEM parent-face IDs must be unique and non-negative")
        if not np.isfinite(stress).all() or not np.isfinite(fatigue).all() or np.any(stress < 0.0) or np.any(fatigue < 0.0):
            raise FEMValidationError("FEM surface stress/fatigue fields are invalid")
        report = {
            "version": VERSION,
            "status": "passed",
            "fem_gate": "passed",
            "real_fem_evidence": True,
            "mesh_convergence": convergence,
            "ai5_validation": ai5,
            "fine_result_output_sha256": fine["output_sha256"],
            "fine_mesh_sha256": fine["mesh_sha256"],
            "pressure_load_sha256": fine["pressure_load_sha256"],
            "surface_results": {
                "parent_face_id": parent_ids.tolist(),
                "von_mises_stress_pa": stress.tolist(),
                "fatigue_damage": fatigue.tolist(),
                "topology_sha256": surface_results.get("topology_sha256"),
            },
        }
        report["validation_sha256"] = _canonical_sha256(report)
        return report
    except (FEMValidationError, TypeError, ValueError) as exc:
        return {
            "version": VERSION,
            "status": "blocked",
            "fem_gate": "blocked",
            "real_fem_evidence": False,
            "reason": str(exc),
        }
