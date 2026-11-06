"""Sparse thickness interpolation with explicit support and fail-closed masks.

Thickness change is defined physically as ``baseline_thickness -
measured_thickness``. The nearest and linear SciPy interpolators are available;
nearest is the robust default for surface samples that are coplanar in 3-D.
No method extrapolates by filling zero: unsupported estimates remain NaN and
are marked in ``missing_mask``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RBFInterpolator, griddata
from scipy.spatial import cKDTree

from step30_validation_common import ValidationError, canonical_sha256, finite_array, jsonable, validate_units


SPARSE_THICKNESS_VERSION = "step30-sparse-thickness-v1"


def _points(value: Any, name: str, count: int | None = None) -> np.ndarray:
    array = finite_array(value, name, ndim=2)
    if array.shape[1] != 3 or len(array) < 2:
        raise ValidationError(f"{name} must have shape (n >= 2, 3)")
    if count is not None and len(array) != count:
        raise ValidationError(f"{name} length does not match sample values")
    return np.asarray(array, dtype=np.float64)


def _values(value: Any, name: str, count: int) -> np.ndarray:
    array = finite_array(value, name, ndim=1, shape=(count,))
    return np.asarray(array, dtype=np.float64)


def _sample_spacing(points: np.ndarray, neighbours: int) -> tuple[np.ndarray, float]:
    tree = cKDTree(points)
    k = min(max(2, neighbours + 1), len(points))
    distances, _ = tree.query(points, k=k)
    local_radius = distances[:, -1]
    finite = local_radius[np.isfinite(local_radius) & (local_radius > 0.0)]
    if len(finite) == 0:
        raise ValidationError("sample points do not define a positive support radius")
    return local_radius, float(np.median(finite))


def _nearest_loocv(points: np.ndarray, delta: np.ndarray) -> np.ndarray:
    distances, indices = cKDTree(points).query(points, k=min(2, len(points)))
    if len(points) < 2:
        return np.full(len(points), np.nan)
    prediction = delta[indices[:, 1]]
    return delta - prediction


def _local_sigma(
    sample_points: np.ndarray,
    sample_delta: np.ndarray,
    loocv_residual: np.ndarray,
    query_points: np.ndarray,
    neighbours: int,
) -> np.ndarray:
    k = min(max(2, neighbours), len(sample_points))
    distances, indices = cKDTree(sample_points).query(query_points, k=k)
    distances = np.atleast_2d(distances)
    indices = np.atleast_2d(indices)
    absolute_residual = np.abs(loocv_residual)
    sigma = np.empty(len(query_points), dtype=np.float64)
    global_sigma = float(np.sqrt(np.mean(loocv_residual ** 2))) if np.isfinite(loocv_residual).any() else np.nan
    for row, local_indices in enumerate(indices):
        local = absolute_residual[np.asarray(local_indices, dtype=np.int64)]
        local = local[np.isfinite(local)]
        if len(local) < 2:
            sigma[row] = global_sigma
        else:
            median = float(np.median(local))
            sigma[row] = max(1.4826 * float(np.median(np.abs(local - median))), 1.0e-12)
    return sigma


def _intrinsic_coordinates(sample_points: np.ndarray, query_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Reduce a surface cloud to its numerical PCA dimension for SciPy."""
    center = sample_points.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(sample_points - center, full_matrices=False)
    if singular_values[0] <= 1.0e-12:
        raise ValidationError("sample points do not span an interpolation domain")
    rank = int(np.sum(singular_values > singular_values[0] * 1.0e-10))
    rank = max(1, min(rank, 3))
    basis = vt[:rank].T
    return (sample_points - center) @ basis, (query_points - center) @ basis, rank


def interpolate_sparse_thickness(
    sample_points: Any,
    baseline_thickness: Any,
    measured_thickness: Any,
    *,
    query_points: Any | None = None,
    method: str = "nearest",
    support_radius_mm: float | None = None,
    neighbours: int = 4,
    rbf_smoothing: float = 0.0,
    coordinate_units: str = "mm",
    thickness_units: str = "mm",
) -> dict[str, Any]:
    """Interpolate ``baseline - measured`` without unbounded extrapolation."""
    points = _points(sample_points, "sample points")
    baseline = _values(baseline_thickness, "baseline thickness", len(points))
    measured = _values(measured_thickness, "measured thickness", len(points))
    query = points.copy() if query_points is None else _points(query_points, "query points")
    coordinate_units = validate_units(coordinate_units, expected_dimension="length")
    thickness_units = validate_units(thickness_units, expected_dimension="length")
    if method not in {"nearest", "linear", "rbf"}:
        raise ValidationError("method must be nearest, linear, or rbf")
    if neighbours < 1:
        raise ValidationError("neighbours must be positive")
    if not np.isfinite(rbf_smoothing) or rbf_smoothing < 0.0:
        raise ValidationError("rbf_smoothing must be finite and non-negative")
    delta = baseline - measured
    local_radius, median_radius = _sample_spacing(points, neighbours)
    inferred_limit = max(3.0 * median_radius, float(np.percentile(local_radius, 95.0)))
    support_limit = inferred_limit if support_radius_mm is None else float(support_radius_mm)
    if not np.isfinite(support_limit) or support_limit <= 0.0:
        raise ValidationError("support_radius_mm must be finite and positive")

    query_tree = cKDTree(points)
    query_distances, query_nearest = query_tree.query(query, k=1)
    query_distances = query_distances.astype(np.float64)
    support_k = min(max(2, neighbours), len(points))
    query_support_neighbours, _ = query_tree.query(query, k=support_k)
    query_support_radius = np.asarray(query_support_neighbours[:, -1] if np.ndim(query_support_neighbours) == 2 else query_support_neighbours, dtype=np.float64)
    outside_support = query_support_radius > support_limit
    interpolation_outside = np.zeros(len(query), dtype=bool)
    if method == "nearest":
        estimate = delta[query_nearest.astype(np.int64)].astype(np.float64)
    else:
        intrinsic_points, intrinsic_query, intrinsic_rank = _intrinsic_coordinates(points, query)
        if intrinsic_rank == 1:
            order = np.argsort(intrinsic_points[:, 0])
            sorted_x = intrinsic_points[order, 0]
            sorted_delta = delta[order]
            interpolation_outside = (intrinsic_query[:, 0] < sorted_x[0]) | (intrinsic_query[:, 0] > sorted_x[-1])
        if method == "linear":
            if intrinsic_rank == 1:
                estimate = np.interp(intrinsic_query[:, 0], sorted_x, sorted_delta).astype(np.float64)
            else:
                estimate = np.asarray(griddata(intrinsic_points, delta, intrinsic_query, method="linear", fill_value=np.nan), dtype=np.float64)
            interpolation_outside |= ~np.isfinite(estimate)
        else:
            try:
                interpolator = RBFInterpolator(
                    intrinsic_points, delta, smoothing=rbf_smoothing,
                    neighbours=min(max(5, neighbours * 4), len(points)),
                )
                estimate = np.asarray(interpolator(intrinsic_query), dtype=np.float64).reshape(-1)
            except (ValueError, np.linalg.LinAlgError) as exc:
                raise ValidationError(f"RBF interpolation failed: {exc}") from exc
            # RBF is mathematically defined outside the convex hull; reject
            # those values explicitly rather than reporting extrapolation.
            if intrinsic_rank == 1:
                pass
            else:
                try:
                    from scipy.spatial import Delaunay
                    interpolation_outside |= Delaunay(intrinsic_points).find_simplex(intrinsic_query) < 0
                except (ValueError, np.linalg.LinAlgError):
                    interpolation_outside |= ~np.isfinite(estimate)

    loocv_residual = _nearest_loocv(points, delta)
    loocv_sigma = float(np.sqrt(np.mean(loocv_residual ** 2)))
    missing_mask = outside_support | interpolation_outside | ~np.isfinite(estimate)
    estimate = np.asarray(estimate, dtype=np.float64)
    estimate[missing_mask] = np.nan
    sigma = _local_sigma(points, delta, loocv_residual, query, neighbours)
    sigma[missing_mask] = np.nan
    support_radius = query_support_radius
    confidence = np.exp(-np.square(support_radius / support_limit))
    confidence *= 1.0 / (1.0 + np.nan_to_num(sigma, nan=0.0))
    confidence[missing_mask] = 0.0

    metadata = {
        "version": SPARSE_THICKNESS_VERSION,
        "method": method,
        "coordinate_units": coordinate_units,
        "thickness_units": thickness_units,
        "support_radius_limit_mm": support_limit,
        "n_samples": int(len(points)),
        "n_query": int(len(query)),
        "loocv_rmse_mm": loocv_sigma,
    }
    result: dict[str, Any] = {
        **metadata,
        "sample_points": points,
        "baseline_thickness": baseline,
        "measured_thickness": measured,
        "delta_thickness_samples_mm": delta,
        "query_points": query,
        "thickness_change_mm": estimate,
        "delta_thickness_mm": estimate,
        "support_radius_mm": support_radius,
        "nearest_sample_distance_mm": query_distances,
        "loocv_residual_mm": loocv_residual,
        "sigma_mm": sigma,
        "outside_mask": outside_support | interpolation_outside,
        "missing_mask": missing_mask,
        "confidence": confidence,
        "quality": {
            "coverage": float(np.mean(~missing_mask)),
            "missing_fraction": float(np.mean(missing_mask)),
            "outside_fraction": float(np.mean(outside_support | interpolation_outside)),
            "loocv_rmse_mm": loocv_sigma,
            "loocv_p95_mm": float(np.percentile(np.abs(loocv_residual), 95.0)),
            "median_sigma_mm": float(np.nanmedian(sigma)) if np.isfinite(sigma).any() else None,
            "p95_support_radius_mm": float(np.percentile(support_radius, 95.0)),
        },
    }
    hash_payload = {
        "version": SPARSE_THICKNESS_VERSION,
        "method": method,
        "sample_points": points,
        "delta": delta,
        "query_points": query,
        "support_radius_limit_mm": support_limit,
    }
    result["result_sha256"] = canonical_sha256(hash_payload)
    return result


def _save_npz(result: dict[str, Any], path: Path) -> None:
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    metadata = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
    arrays["metadata_json"] = np.asarray(json.dumps(jsonable(metadata), ensure_ascii=True, sort_keys=True))
    np.savez_compressed(path, **arrays)


def selftest() -> bool:
    rng = np.random.default_rng(30032)
    x, y = np.meshgrid(np.linspace(-1.0, 1.0, 9), np.linspace(-1.0, 1.0, 9))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    baseline = 8.0 + 0.2 * points[:, 0] - 0.1 * points[:, 1]
    measured = baseline - 0.45 - rng.normal(0.0, 0.01, len(points))
    query = np.vstack((points[rng.permutation(len(points))], [9.0, 9.0, 0.0]))
    result = interpolate_sparse_thickness(
        points, baseline, measured, query_points=query,
        method="nearest", support_radius_mm=0.5,
    )
    valid = ~result["missing_mask"]
    assert np.allclose(np.nanmedian(result["thickness_change_mm"][valid]), 0.45, atol=0.04)
    assert result["missing_mask"][-1]
    assert result["outside_mask"][-1]
    assert np.isnan(result["thickness_change_mm"][-1])
    assert result["loocv_residual_mm"].shape == baseline.shape
    assert np.isfinite(result["quality"]["loocv_rmse_mm"])
    # The subtraction is explicit and remains independent of input row order.
    shuffled = rng.permutation(len(points))
    reordered = interpolate_sparse_thickness(
        points[shuffled], baseline[shuffled], measured[shuffled],
        query_points=query[:20], method="nearest", support_radius_mm=0.5,
    )
    assert np.allclose(reordered["thickness_change_mm"], result["thickness_change_mm"][:20], equal_nan=True)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step30 sparse thickness interpolation")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--points", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--measured", type=Path)
    parser.add_argument("--query", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--method", choices=("nearest", "linear", "rbf"), default="nearest")
    parser.add_argument("--support-radius-mm", type=float, default=None)
    args = parser.parse_args()
    if args.selftest:
        selftest()
        print("step30_sparse_thickness selftest: PASS")
    elif args.points is not None and args.baseline is not None and args.measured is not None:
        result = interpolate_sparse_thickness(
            np.load(args.points, allow_pickle=False),
            np.load(args.baseline, allow_pickle=False),
            np.load(args.measured, allow_pickle=False),
            query_points=None if args.query is None else np.load(args.query, allow_pickle=False),
            method=args.method, support_radius_mm=args.support_radius_mm,
        )
        if args.output is None:
            print(json.dumps(jsonable(result["quality"]), indent=2, ensure_ascii=True))
        else:
            _save_npz(result, args.output)
    else:
        parser.print_help()
