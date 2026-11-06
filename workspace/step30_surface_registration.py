"""Correspondence-free robust rigid registration for measured surfaces.

The public API uses the same row-vector convention as the existing Step30
geometry tools::

    reference = measured @ R + translation

Registration does not assume point order or equal point counts. Several PCA
initialisations are followed by trimmed nearest-neighbour ICP. Distances are
reported in both directions so that an apparently good one-way fit cannot hide
missing or occluded surface regions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree

from step30_validation_common import (
    ValidationError,
    canonical_sha256,
    convert_units,
    finite_array,
    jsonable,
    validate_units,
)


REGISTRATION_VERSION = "step30-surface-registration-v1"


def _points(value: Any, name: str) -> np.ndarray:
    array = finite_array(value, name, ndim=2)
    if array.shape[1] != 3 or len(array) < 3:
        raise ValidationError(f"{name} must have shape (n >= 3, 3)")
    return np.asarray(array, dtype=np.float64)


def _hashable(value: Any) -> Any:
    """Hash arrays by shape/dtype/bytes so diagnostic NaNs remain valid."""
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "dtype": str(contiguous.dtype),
            "shape": list(contiguous.shape),
            "sha256": __import__("hashlib").sha256(contiguous.tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(key): _hashable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_hashable(item) for item in value]
    return value


def _normals(value: Any, name: str, count: int) -> np.ndarray:
    array = finite_array(value, name, ndim=2, shape=(count, 3))
    lengths = np.linalg.norm(array, axis=1)
    if np.any(lengths <= 1.0e-12):
        raise ValidationError(f"{name} contains zero-length normals")
    return array / lengths[:, None]


def _proper_pca_frames(points: np.ndarray) -> list[np.ndarray]:
    """Return deterministic right-handed PCA frames, including permutations."""
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / max(len(points), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[-1] <= 1.0e-15 or eigenvalues[-2] <= eigenvalues[-1] * 1.0e-12:
        return [np.eye(3)]
    base = eigenvectors[:, ::-1]
    frames: list[np.ndarray] = []
    for permutation in (
        (0, 1, 2), (0, 2, 1), (1, 0, 2),
        (1, 2, 0), (2, 0, 1), (2, 1, 0),
    ):
        permuted = base[:, permutation]
        for signs in (
            (1.0, 1.0, 1.0), (1.0, 1.0, -1.0),
            (1.0, -1.0, 1.0), (1.0, -1.0, -1.0),
            (-1.0, 1.0, 1.0), (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0), (-1.0, -1.0, -1.0),
        ):
            frame = permuted @ np.diag(signs)
            if np.linalg.det(frame) > 0.0:
                frames.append(frame)
    unique: list[np.ndarray] = []
    for frame in frames:
        if not any(np.allclose(frame, old, atol=1.0e-12) for old in unique):
            unique.append(frame)
    return unique or [np.eye(3)]


def _validate_rotation(rotation: Any) -> np.ndarray:
    matrix = finite_array(rotation, "rotation", ndim=2, shape=(3, 3))
    determinant = float(np.linalg.det(matrix))
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=2.0e-6):
        raise ValidationError("rotation is not orthonormal")
    if determinant <= 0.0 or not np.isclose(determinant, 1.0, atol=2.0e-6):
        raise ValidationError("rotation is not a proper rigid rotation")
    return np.asarray(matrix, dtype=np.float64)


def _kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if source.shape != target.shape or len(source) < 3:
        raise ValidationError("Kabsch requires at least three paired points")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, singular, vt = np.linalg.svd(covariance, full_matrices=False)
    if singular[0] <= 1.0e-15 or singular[1] <= singular[0] * 1.0e-12:
        raise ValidationError("Kabsch correspondences are collinear or degenerate")
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    rotation = _validate_rotation(rotation)
    translation = target_center - source_center @ rotation
    return rotation, np.asarray(translation, dtype=np.float64)


def _apply(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return points @ rotation + translation


def _initial_transforms(source: np.ndarray, target: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    transforms: list[tuple[np.ndarray, np.ndarray]] = [(np.eye(3), target_center - source_center)]
    source_frames = _proper_pca_frames(source)
    target_frames = _proper_pca_frames(target)
    for source_frame in source_frames:
        for target_frame in target_frames:
            # Row-vector map: x @ source_frame @ target_frame.T.
            rotation = source_frame @ target_frame.T
            if np.linalg.det(rotation) > 0.0:
                transforms.append((rotation, target_center - source_center @ rotation))
    return transforms


def _trim_indices(
    distances: np.ndarray,
    trim_fraction: float,
    *,
    normal_cosines: np.ndarray | None = None,
    normal_min_cosine: float | None = None,
    min_inliers: int,
) -> np.ndarray:
    keep = max(min_inliers, int(np.ceil((1.0 - trim_fraction) * len(distances))))
    keep = min(keep, len(distances))
    score = np.asarray(distances, dtype=np.float64).copy()
    if normal_cosines is not None and normal_min_cosine is not None:
        score[normal_cosines < normal_min_cosine] = np.inf
    order = np.argsort(score, kind="mergesort")
    selected = order[:keep]
    selected = selected[np.isfinite(score[selected])]
    if len(selected) < min_inliers:
        selected = np.argsort(distances, kind="mergesort")[:keep]
    return np.asarray(selected, dtype=np.int64)


def _nearest_spacing(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    distances, _ = cKDTree(points).query(points, k=2)
    spacing = float(np.median(distances[:, 1]))
    return spacing if np.isfinite(spacing) and spacing > 0.0 else 0.0


def _normal_angles(
    source_normals: np.ndarray | None,
    target_normals: np.ndarray | None,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray | None:
    if source_normals is None or target_normals is None:
        return None
    transformed = source_normals @ rotation
    cosine = np.einsum("ij,ij->i", transformed[source_indices], target_normals[target_indices])
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _uncertainty(
    source: np.ndarray,
    transformed: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    inliers: np.ndarray,
) -> dict[str, Any]:
    residual = transformed[inliers] - target[indices[inliers]]
    if len(residual) == 0:
        return {"available": False, "reason": "no inliers"}
    sigma = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    centered = source[inliers] - source[inliers].mean(axis=0)
    radius = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    dof = max(1, 3 * len(inliers) - 6)
    residual_variance = float(np.sum(residual * residual) / dof)
    translation_sigma = float(np.sqrt(residual_variance / max(len(inliers), 1)))
    rotation_sigma = float(np.degrees(np.sqrt(residual_variance / max(len(inliers), 1)) / max(radius, 1.0e-12)))
    covariance_translation = np.eye(3) * translation_sigma ** 2
    return {
        "available": True,
        "residual_rms_mm": sigma,
        "residual_sigma_mm": float(np.sqrt(residual_variance)),
        "translation_sigma_mm": translation_sigma,
        "rotation_sigma_degrees": rotation_sigma,
        "translation_covariance_mm2": covariance_translation,
        "n_inliers": int(len(inliers)),
    }


def _summary(distances: np.ndarray, threshold: float, *, inlier_mask: np.ndarray) -> dict[str, Any]:
    return {
        "n": int(len(distances)),
        "min_mm": float(np.min(distances)),
        "median_mm": float(np.median(distances)),
        "p95_mm": float(np.percentile(distances, 95.0)),
        "max_mm": float(np.max(distances)),
        "threshold_mm": float(threshold),
        "coverage": float(np.mean(distances <= threshold)),
        "inlier_fraction": float(np.mean(inlier_mask)),
    }


def register_surfaces(
    measured_points: Any,
    reference_points: Any,
    *,
    measured_normals: Any | None = None,
    reference_normals: Any | None = None,
    measured_units: str = "mm",
    reference_units: str = "mm",
    allow_unit_conversion: bool = False,
    trim_fraction: float = 0.20,
    distance_threshold_mm: float | None = None,
    normal_angle_threshold_degrees: float = 45.0,
    min_coverage: float = 0.80,
    max_iterations: int = 60,
    tolerance_mm: float = 1.0e-6,
) -> dict[str, Any]:
    """Register two unordered, unequal-density point clouds.

    ``measured_points`` is transformed into the reference frame. The result
    contains arrays for both nearest-neighbour directions and explicit masks:
    ``missing_mask`` marks measured samples with no nearby reference support;
    ``occluded_mask`` marks reference samples not represented by the measured
    scan. Neither mask relies on array row order.
    """
    source = _points(measured_points, "measured points")
    target = _points(reference_points, "reference points")
    source_unit = validate_units(measured_units, expected_dimension="length")
    target_unit = validate_units(reference_units, expected_dimension="length")
    if source_unit != target_unit:
        if not allow_unit_conversion:
            raise ValidationError("measured and reference coordinate units differ")
        source = convert_units(source, source_unit, target_unit)
        source_unit = target_unit
    if not 0.0 <= trim_fraction < 1.0:
        raise ValidationError("trim_fraction must be in [0, 1)")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValidationError("min_coverage must be in [0, 1]")
    if max_iterations < 1:
        raise ValidationError("max_iterations must be positive")
    if normal_angle_threshold_degrees < 0.0 or normal_angle_threshold_degrees > 180.0:
        raise ValidationError("normal angle threshold must be in [0, 180]")
    source_normal_array = None if measured_normals is None else _normals(measured_normals, "measured normals", len(source))
    target_normal_array = None if reference_normals is None else _normals(reference_normals, "reference normals", len(target))
    if (source_normal_array is None) != (target_normal_array is None):
        raise ValidationError("both measured and reference normals are required together")
    spacing = max(_nearest_spacing(source), _nearest_spacing(target), 1.0e-9)
    threshold = float(distance_threshold_mm) if distance_threshold_mm is not None else max(3.0 * spacing, 1.0e-3)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValidationError("distance_threshold_mm must be finite and positive")
    normal_cosine_cutoff = float(np.cos(np.deg2rad(normal_angle_threshold_degrees)))
    min_inliers = max(3, min(len(source), len(target), 10))

    best: dict[str, Any] | None = None
    target_tree = cKDTree(target)
    for initial_rotation, initial_translation in _initial_transforms(source, target):
        rotation = initial_rotation.copy()
        translation = initial_translation.copy()
        previous_objective = np.inf
        converged = False
        final_indices = np.zeros(len(source), dtype=np.int64)
        final_distances = np.full(len(source), np.inf, dtype=np.float64)
        final_inliers = np.zeros(len(source), dtype=bool)
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            transformed = _apply(source, rotation, translation)
            distances, indices = target_tree.query(transformed, k=1)
            angles = _normal_angles(
                source_normal_array, target_normal_array,
                np.arange(len(source)), indices.astype(np.int64), rotation,
            )
            cosines = None if angles is None else np.cos(np.deg2rad(angles))
            selected = _trim_indices(
                distances, trim_fraction, normal_cosines=cosines,
                normal_min_cosine=normal_cosine_cutoff if cosines is not None else None,
                min_inliers=min_inliers,
            )
            if len(selected) < 3:
                break
            try:
                new_rotation, new_translation = _kabsch(source[selected], target[indices[selected]])
            except ValidationError:
                break
            objective = float(np.sqrt(np.mean(distances[selected] ** 2)))
            rotation_delta = float(np.linalg.norm(new_rotation - rotation))
            translation_delta = float(np.linalg.norm(new_translation - translation))
            rotation, translation = new_rotation, new_translation
            final_indices, final_distances = indices.astype(np.int64), distances.astype(np.float64)
            final_inliers[:] = False
            final_inliers[selected] = True
            if abs(previous_objective - objective) <= tolerance_mm and translation_delta <= tolerance_mm and rotation_delta <= tolerance_mm:
                converged = True
                break
            previous_objective = objective
        if not np.isfinite(final_distances).all():
            continue
        transformed = _apply(source, rotation, translation)
        final_distances, final_indices = target_tree.query(transformed, k=1)
        final_indices = final_indices.astype(np.int64)
        final_distances = final_distances.astype(np.float64)
        final_angles = _normal_angles(
            source_normal_array, target_normal_array,
            np.arange(len(source)), final_indices, rotation,
        )
        final_cosines = None if final_angles is None else np.cos(np.deg2rad(final_angles))
        selected = _trim_indices(
            final_distances, trim_fraction, normal_cosines=final_cosines,
            normal_min_cosine=normal_cosine_cutoff if final_cosines is not None else None,
            min_inliers=min_inliers,
        )
        final_inliers[:] = False
        final_inliers[selected] = True
        forward_rmse = float(np.sqrt(np.mean(final_distances[selected] ** 2)))
        reverse_distances, reverse_indices = cKDTree(transformed).query(target, k=1)
        reverse_rmse = float(np.sqrt(np.mean(reverse_distances ** 2)))
        score = forward_rmse + reverse_rmse + 0.25 * float(np.mean(final_distances > threshold)) * threshold
        candidate = {
            "rotation": rotation,
            "translation": translation,
            "transformed": transformed,
            "forward_distances": final_distances,
            "forward_indices": final_indices,
            "forward_inliers": final_inliers,
            "reverse_distances": reverse_distances.astype(np.float64),
            "reverse_indices": reverse_indices.astype(np.int64),
            "angles": final_angles,
            "score": score,
            "iterations": iterations,
            "converged": converged,
        }
        if best is None or candidate["score"] < best["score"]:
            best = candidate
    if best is None:
        raise ValidationError("all registration initialisations failed")

    transformed = best["transformed"]
    forward_distances = best["forward_distances"]
    reverse_distances = best["reverse_distances"]
    angles = best["angles"]
    forward_inliers = best["forward_inliers"]
    missing_mask = (forward_distances > threshold) | (~forward_inliers)
    occluded_mask = reverse_distances > threshold
    normal_bad_mask = np.zeros(len(source), dtype=bool)
    normal_summary: dict[str, Any] = {"available": False}
    reverse_angles = None
    if source_normal_array is not None and target_normal_array is not None:
        normal_bad_mask = angles > normal_angle_threshold_degrees
        reverse_source_indices = best["reverse_indices"]
        transformed_source_normals = source_normal_array @ best["rotation"]
        reverse_cosines = np.einsum("ij,ij->i", target_normal_array, transformed_source_normals[reverse_source_indices])
        reverse_angles = np.degrees(np.arccos(np.clip(reverse_cosines, -1.0, 1.0)))
        normal_summary = {
            "available": True,
            "threshold_degrees": float(normal_angle_threshold_degrees),
            "forward_median_degrees": float(np.median(angles)),
            "forward_p95_degrees": float(np.percentile(angles, 95.0)),
            "reverse_median_degrees": float(np.median(reverse_angles)),
            "reverse_p95_degrees": float(np.percentile(reverse_angles, 95.0)),
            "forward_coverage": float(np.mean(angles <= normal_angle_threshold_degrees)),
            "reverse_coverage": float(np.mean(reverse_angles <= normal_angle_threshold_degrees)),
        }
    uncertainty = _uncertainty(
        source, transformed, target, best["forward_indices"], np.flatnonzero(forward_inliers)
    )
    transform = {
        "R": best["rotation"],
        "translation": best["translation"],
        "convention": "row-vector: reference = measured @ R + translation",
        "units": source_unit,
    }
    transform_hash = canonical_sha256(transform)
    forward_summary = _summary(forward_distances, threshold, inlier_mask=forward_inliers)
    reverse_summary = _summary(reverse_distances, threshold, inlier_mask=~occluded_mask)
    result: dict[str, Any] = {
        "version": REGISTRATION_VERSION,
        "status": "measured",
        "transform": transform,
        "registered_measured_points": transformed,
        "transform_hash": transform_hash,
        "registration_sha256": transform_hash,
        "measured_units": source_unit,
        "reference_units": target_unit,
        "n_measured": int(len(source)),
        "n_reference": int(len(target)),
        "trim_fraction": float(trim_fraction),
        "iterations": int(best["iterations"]),
        "converged": bool(best["converged"]),
        "forward": {
            "distances_mm": forward_distances,
            "indices": best["forward_indices"],
            "inlier_mask": forward_inliers,
            "summary": forward_summary,
        },
        "reverse": {
            "distances_mm": reverse_distances,
            "indices": best["reverse_indices"],
            "occluded_mask": occluded_mask,
            "summary": reverse_summary,
        },
        "normal_angle_degrees": angles if angles is not None else np.full(len(source), np.nan),
        "reverse_normal_angle_degrees": reverse_angles if reverse_angles is not None else np.full(len(target), np.nan),
        "normal_bad_mask": normal_bad_mask,
        "missing_mask": missing_mask,
        "occluded_mask": occluded_mask,
        "uncertainty": uncertainty,
        "normal_summary": normal_summary,
        "quality": {
            "coverage_forward": forward_summary["coverage"],
            "coverage_reverse": reverse_summary["coverage"],
            "p95_forward_mm": forward_summary["p95_mm"],
            "p95_reverse_mm": reverse_summary["p95_mm"],
            "missing_fraction": float(np.mean(missing_mask)),
            "occluded_fraction": float(np.mean(occluded_mask)),
            "normal_passed": bool(not normal_summary["available"] or (
                normal_summary["forward_coverage"] >= min_coverage
                and normal_summary["reverse_coverage"] >= min_coverage
            )),
        },
    }
    result["passed"] = bool(
        result["quality"]["coverage_forward"] >= min_coverage
        and result["quality"]["coverage_reverse"] >= min_coverage
        and result["quality"]["normal_passed"]
    )
    result["result_sha256"] = canonical_sha256(_hashable({key: value for key, value in result.items() if key != "result_sha256"}))
    return result


def apply_transform(points: Any, registration: Mapping[str, Any]) -> np.ndarray:
    """Apply a registration result or its nested ``transform`` mapping."""
    values = _points(points, "points")
    transform = registration.get("transform", registration)
    rotation = _validate_rotation(transform.get("R"))
    translation = finite_array(transform.get("translation"), "translation", ndim=1, shape=(3,))
    return _apply(values, rotation, translation)


def _save_npz(result: Mapping[str, Any], path: Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
        elif isinstance(value, Mapping):
            nested: dict[str, Any] = {}
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, np.ndarray):
                    arrays[f"{key}__{nested_key}"] = nested_value
                else:
                    nested[nested_key] = nested_value
            metadata[key] = nested
        else:
            metadata[key] = value
    arrays["metadata_json"] = np.asarray(json.dumps(jsonable(metadata), ensure_ascii=True, sort_keys=True))
    np.savez_compressed(path, **arrays)


def selftest() -> bool:
    """Exercise shuffling, unequal density, occlusion and measurement noise."""
    rng = np.random.default_rng(30030)
    angles = rng.uniform(0.0, 2.0 * np.pi, 600)
    z = rng.uniform(-1.0, 1.0, len(angles))
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    reference = np.column_stack((2.4 * radial * np.cos(angles), 1.2 * radial * np.sin(angles), 0.8 * z))
    reference_normals = reference / np.array([2.4, 1.2, 0.8]) ** 2
    reference_normals /= np.linalg.norm(reference_normals, axis=1, keepdims=True)
    theta = 0.31
    rotation = np.array([[np.cos(theta), np.sin(theta), 0.0], [-np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([7.0, -3.0, 1.5])
    source_indices = rng.choice(len(reference), size=430, replace=False)
    source = (reference[source_indices] - translation) @ rotation.T
    source += rng.normal(0.0, 0.003, source.shape)
    source_permutation = rng.permutation(len(source))
    source = source[source_permutation]
    source_normals = (reference_normals[source_indices] @ rotation.T)[source_permutation]
    target_permutation = rng.permutation(len(reference))
    target = reference[target_permutation]
    target_normals = reference_normals[target_permutation]
    result = register_surfaces(
        source, target, measured_normals=source_normals,
        reference_normals=target_normals,
        trim_fraction=0.25, distance_threshold_mm=0.10, min_coverage=0.55,
        max_iterations=35,
    )
    assert result["transform_hash"] and len(result["transform_hash"]) == 64
    assert result["n_measured"] == 430 and result["n_reference"] == 600
    assert result["forward"]["summary"]["p95_mm"] < 0.10
    assert result["quality"]["coverage_forward"] > 0.55
    assert np.any(result["occluded_mask"])
    assert result["uncertainty"]["available"]
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step30 correspondence-free surface registration")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--measured", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold-mm", type=float, default=None)
    parser.add_argument("--trim-fraction", type=float, default=0.20)
    args = parser.parse_args()
    if args.selftest:
        selftest()
        print("step30_surface_registration selftest: PASS")
    elif args.measured is not None and args.reference is not None:
        measured = np.load(args.measured, allow_pickle=False)
        reference = np.load(args.reference, allow_pickle=False)
        result = register_surfaces(measured, reference, trim_fraction=args.trim_fraction, distance_threshold_mm=args.threshold_mm)
        if args.output is None:
            print(json.dumps(jsonable({key: value for key, value in result.items() if not isinstance(value, np.ndarray)}), indent=2, ensure_ascii=True))
        else:
            _save_npz(result, args.output)
    else:
        parser.print_help()
