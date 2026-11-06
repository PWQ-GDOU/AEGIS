"""Rigid coordinate registration and geometry coverage diagnostics for Step30.

All transforms use the row-vector convention::

    target = source @ R + translation

Kabsch is implemented for this convention and rejects reflections, non-finite
inputs, unit mismatches, and insufficient correspondence.  Nearest-neighbor
and bidirectional checks are diagnostics, not substitutes for an independent
measured validation set.
"""

from __future__ import annotations

import argparse
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree

from step30_validation_common import (
    ValidationError,
    canonical_sha256,
    convert_units,
    finite_array,
    provenance_note,
    require_status,
    safe_json_dump,
    validate_units,
)


REGISTRATION_VERSION = "step30-coordinate-registration-v1"


def _validate_points(points: Any, name: str) -> np.ndarray:
    array = finite_array(points, name, ndim=2)
    if array.shape[1] != 3:
        raise ValidationError(f"{name} must have shape (n, 3)")
    if len(array) < 3:
        raise ValidationError(f"{name} requires at least 3 points")
    return array


def _validate_correspondences(source: Any, target: Any) -> tuple[np.ndarray, np.ndarray]:
    source_array = _validate_points(source, "source points")
    target_array = _validate_points(target, "target points")
    if source_array.shape != target_array.shape:
        raise ValidationError("Kabsch correspondences must have equal shapes")
    if len(source_array) < 3:
        raise ValidationError("Kabsch requires at least 3 correspondences")
    return source_array, target_array


def kabsch_row_vector(
    source: Any,
    target: Any,
    *,
    source_units: str = "mm",
    target_units: str = "mm",
    allow_unit_conversion: bool = False,
) -> dict[str, Any]:
    """Fit ``target = source @ R + translation`` using paired points."""
    source_array, target_array = _validate_correspondences(source, target)
    source_unit = validate_units(source_units, expected_dimension="length")
    target_unit = validate_units(target_units, expected_dimension="length")
    if source_unit != target_unit and not allow_unit_conversion:
        raise ValidationError("source and target coordinate units differ")
    if source_unit != target_unit:
        source_array = convert_units(source_array, source_unit, target_unit)
        source_unit = target_unit
    source_centroid = source_array.mean(axis=0)
    target_centroid = target_array.mean(axis=0)
    source_centered = source_array - source_centroid
    target_centered = target_array - target_centroid
    covariance = source_centered.T @ target_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    determinant = float(np.linalg.det(rotation))
    if determinant <= 0.0 or not np.isclose(determinant, 1.0, atol=1e-8):
        raise ValidationError("Kabsch produced an improper rotation")
    if singular_values[0] <= 0.0 or singular_values[1] <= max(singular_values[0], 1.0) * 1.0e-12:
        raise ValidationError("Kabsch correspondences are collinear or degenerate")
    translation = target_centroid - source_centroid @ rotation
    transformed = source_array @ rotation + translation
    residuals = np.linalg.norm(transformed - target_array, axis=1)
    if not np.isfinite(residuals).all():
        raise ValidationError("Kabsch residuals are non-finite")
    result = {
        "version": REGISTRATION_VERSION,
        "convention": "row-vector: target = source @ R + translation",
        "source_units": source_unit,
        "target_units": target_unit,
        "R": rotation,
        "translation": translation,
        "source_centroid": source_centroid,
        "target_centroid": target_centroid,
        "singular_values": singular_values,
        "det_R": determinant,
        "n_correspondences": int(len(source_array)),
        "residuals": residuals,
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "max_residual": float(residuals.max()),
        "registration_sha256": canonical_sha256({
            "convention": "row-vector: target = source @ R + translation",
            "source_units": source_unit,
            "target_units": target_unit,
            "R": rotation,
            "translation": translation,
        }),
    }
    return result


def apply_row_transform(points: Any, registration: Mapping[str, Any]) -> np.ndarray:
    """Apply a validated row-vector rigid transform."""
    array = _validate_points(points, "points")
    rotation = finite_array(registration.get("R"), "R", ndim=2, shape=(3, 3))
    translation = finite_array(registration.get("translation"), "translation", ndim=1, shape=(3,))
    determinant = float(np.linalg.det(rotation))
    if (
        not np.isclose(rotation.T @ rotation, np.eye(3), atol=1e-6).all()
        or determinant <= 0.0
        or not np.isclose(determinant, 1.0, atol=1e-6)
    ):
        raise ValidationError("registration R is not a proper rigid rotation")
    result = array @ rotation + translation
    return finite_array(result, "transformed points", ndim=2, shape=array.shape)


def _distance_summary(
    distances: np.ndarray,
    threshold: float | None,
    *,
    min_coverage: float = 0.95,
) -> dict[str, Any]:
    distances = finite_array(distances, "distances", ndim=1)
    if not 0.0 <= min_coverage <= 1.0:
        raise ValidationError("minimum coverage must be in [0, 1]")
    summary = {
        "n": int(len(distances)),
        "min": float(distances.min()),
        "median": float(np.median(distances)),
        "p95": float(np.percentile(distances, 95)),
        "max": float(distances.max()),
    }
    if threshold is not None:
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValidationError("distance threshold must be finite and non-negative")
        summary["threshold"] = float(threshold)
        summary["min_coverage"] = float(min_coverage)
        summary["coverage"] = float(np.mean(distances <= threshold))
        summary["passed"] = bool(summary["coverage"] >= min_coverage)
    return summary


def nearest_neighbor_diagnostics(
    source: Any,
    target: Any,
    *,
    threshold: float | None = None,
    min_coverage: float = 0.95,
    source_units: str = "mm",
    target_units: str = "mm",
    transform: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute one-way nearest-neighbor distances and coverage."""
    source_array = _validate_points(source, "source points")
    target_array = _validate_points(target, "target points")
    source_unit = validate_units(source_units, expected_dimension="length")
    target_unit = validate_units(target_units, expected_dimension="length")
    if source_unit != target_unit:
        raise ValidationError("nearest-neighbor coordinate units differ")
    if transform is not None:
        source_array = apply_row_transform(source_array, transform)
    tree = cKDTree(target_array)
    distances, indices = tree.query(source_array, k=1)
    result = {
        "version": REGISTRATION_VERSION,
        "direction": "source_to_target",
        "units": source_unit,
        "source_count": int(len(source_array)),
        "target_count": int(len(target_array)),
        "summary": _distance_summary(distances, threshold, min_coverage=min_coverage),
        "indices": indices.astype(np.int64),
        "distances": distances.astype(np.float64),
    }
    return result


def bidirectional_registration_diagnostics(
    source: Any,
    target: Any,
    *,
    threshold: float,
    min_coverage: float = 0.95,
    normal_source: Any | None = None,
    normal_target: Any | None = None,
    normal_angle_degrees: float = 30.0,
    source_units: str = "mm",
    target_units: str = "mm",
    transform: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Check source->target and target->source distance, coverage, and normals."""
    if threshold < 0.0 or not np.isfinite(threshold):
        raise ValidationError("threshold must be finite and non-negative")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValidationError("minimum coverage must be in [0, 1]")
    if normal_angle_degrees < 0.0 or normal_angle_degrees > 180.0:
        raise ValidationError("normal angle threshold must be in [0, 180]")
    source_array = _validate_points(source, "source points")
    target_array = _validate_points(target, "target points")
    source_unit = validate_units(source_units, expected_dimension="length")
    target_unit = validate_units(target_units, expected_dimension="length")
    if source_unit != target_unit:
        raise ValidationError("bidirectional coordinate units differ")
    if transform is not None:
        source_array = apply_row_transform(source_array, transform)
    forward = nearest_neighbor_diagnostics(
        source_array, target_array, threshold=threshold, min_coverage=min_coverage,
        source_units=source_unit, target_units=target_unit,
    )
    reverse = nearest_neighbor_diagnostics(
        target_array, source_array, threshold=threshold, min_coverage=min_coverage,
        source_units=target_unit, target_units=source_unit,
    )
    result: dict[str, Any] = {
        "version": REGISTRATION_VERSION,
        "units": source_unit,
        "threshold": float(threshold),
        "forward": {key: value for key, value in forward.items() if key not in {"indices", "distances"}},
        "reverse": {key: value for key, value in reverse.items() if key not in {"indices", "distances"}},
    }
    normal_check = {"available": False, "passed": False}
    if normal_source is not None or normal_target is not None:
        if normal_source is None or normal_target is None:
            raise ValidationError("both normal arrays are required for normal checks")
        source_normals = finite_array(normal_source, "source normals", ndim=2, shape=source_array.shape)
        target_normals = finite_array(normal_target, "target normals", ndim=2, shape=target_array.shape)
        if transform is not None:
            rotation = finite_array(transform.get("R"), "registration R", ndim=2, shape=(3, 3))
            source_normals = source_normals @ rotation
        source_norms = np.linalg.norm(source_normals, axis=1)
        target_norms = np.linalg.norm(target_normals, axis=1)
        if np.any(source_norms <= 0.0) or np.any(target_norms <= 0.0):
            raise ValidationError("normal vectors must be non-zero")
        source_normals = source_normals / source_norms[:, None]
        target_normals = target_normals / target_norms[:, None]
        forward_indices = forward["indices"]
        reverse_indices = reverse["indices"]
        forward_cos = np.einsum("ij,ij->i", source_normals, target_normals[forward_indices])
        reverse_cos = np.einsum("ij,ij->i", target_normals, source_normals[reverse_indices])
        cutoff = float(np.cos(np.deg2rad(normal_angle_degrees)))
        normal_check = {
            "available": True,
            "angle_threshold_degrees": float(normal_angle_degrees),
            "forward_median_angle_degrees": float(np.median(np.degrees(np.arccos(np.clip(forward_cos, -1.0, 1.0))))),
            "reverse_median_angle_degrees": float(np.median(np.degrees(np.arccos(np.clip(reverse_cos, -1.0, 1.0))))),
            "forward_coverage": float(np.mean(forward_cos >= cutoff)),
            "reverse_coverage": float(np.mean(reverse_cos >= cutoff)),
        }
        normal_check["passed"] = bool(
            normal_check["forward_coverage"] >= min_coverage
            and normal_check["reverse_coverage"] >= min_coverage
        )
    result["normal_check"] = normal_check
    result["passed"] = bool(
        result["forward"]["summary"].get("coverage", 0.0) >= min_coverage
        and result["reverse"]["summary"].get("coverage", 0.0) >= min_coverage
        and (not normal_check["available"] or normal_check["passed"])
    )
    result["diagnostics_sha256"] = canonical_sha256(result)
    return result


def register_and_diagnose(
    source: Any,
    target: Any,
    *,
    threshold: float,
    min_coverage: float = 0.95,
    status: str = "measured",
    normal_source: Any | None = None,
    normal_target: Any | None = None,
    source_units: str = "mm",
    target_units: str = "mm",
    allow_unit_conversion: bool = False,
) -> dict[str, Any]:
    """Fit Kabsch, run bidirectional checks, and mark blocked failures."""
    require_status(status)
    try:
        registration = kabsch_row_vector(
            source, target, source_units=source_units, target_units=target_units,
            allow_unit_conversion=allow_unit_conversion,
        )
        diagnostic_source = (
            convert_units(source, source_units, target_units)
            if source_units != target_units and allow_unit_conversion
            else source
        )
        diagnostics = bidirectional_registration_diagnostics(
            diagnostic_source, target, threshold=threshold, min_coverage=min_coverage,
            normal_source=normal_source, normal_target=normal_target,
            transform=registration, source_units=registration["source_units"],
            target_units=registration["target_units"],
        )
        passed = bool(diagnostics["passed"])
        error = None
    except (ValidationError, ValueError) as exc:
        registration = None
        diagnostics = None
        passed = False
        error = str(exc)
    result = {
        "version": REGISTRATION_VERSION,
        "status": status if passed else "blocked",
        "input_status": status,
        "passed": passed,
        "registration": registration,
        "diagnostics": diagnostics,
        "error": error,
        "provenance_note": provenance_note("synthetic" if status == "synthetic" else ("blocked" if not passed else status)),
    }
    return result


def save_registration(result: Mapping[str, Any], path: str) -> dict[str, Any]:
    """Save a registration result immutably as strict JSON."""
    return safe_json_dump(result, path)


def _selftest() -> None:
    rng = np.random.default_rng(42)
    source = rng.normal(size=(20, 3))
    angle = 0.4
    rotation = np.array([[np.cos(angle), np.sin(angle), 0.0], [-np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([3.0, -2.0, 0.5])
    target = source @ rotation + translation
    result = kabsch_row_vector(source, target)
    assert np.allclose(result["R"], rotation, atol=1e-10)
    assert np.allclose(result["translation"], translation, atol=1e-10)
    diagnostics = bidirectional_registration_diagnostics(source, target, threshold=1e-8, transform=result)
    assert diagnostics["passed"]
    source_normals = np.tile([0.0, 0.0, 1.0], (len(source), 1))
    target_normals = source_normals @ rotation
    normal_diagnostics = bidirectional_registration_diagnostics(
        source, target, threshold=1e-8, transform=result,
        normal_source=source_normals, normal_target=target_normals,
    )
    assert normal_diagnostics["normal_check"]["passed"]
    blocked = register_and_diagnose(source, target, threshold=1e-8, source_units="mm", target_units="m")
    assert blocked["status"] == "blocked"
    converted = register_and_diagnose(
        source, target * 0.001, threshold=1e-8, source_units="mm", target_units="m",
        allow_unit_conversion=True,
    )
    assert converted["status"] == "measured"
    collinear = np.column_stack([np.arange(5.0), np.zeros(5), np.zeros(5)])
    assert register_and_diagnose(collinear, collinear, threshold=1e-8)["status"] == "blocked"


def selftest() -> bool:
    _selftest()
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step30 coordinate registration selftest")
    parser.add_argument("--selftest", action="store_true", help="run the minimal selftest")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        print("step30_coordinate_registration selftest: PASS")
    else:
        parser.print_help()
