"""Explainable roughness and pit morphology metrics for material trials.

Supports a 2-D height field (NPY/NPZ/CSV/JSON) or an N x 3 point cloud.
All heights and spacing are converted explicitly to millimetres before area
and volume calculations. Pit objects are connected components below a robust
threshold relative to the measured reference plane.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage

from step30_validation_common import ValidationError, canonical_sha256, finite_array, jsonable, safe_json_dump, sha256_file


MORPHOLOGY_VERSION = "step30-material-morphology-metrics-v1"


def _unit_factor(unit: str, target: str = "mm") -> float:
    normalized = str(unit).strip().lower().replace("μ", "u").replace("µ", "u").replace("²", "2").replace("³", "3").replace("^", "")
    factors = {"m": 1000.0, "cm": 10.0, "mm": 1.0, "um": 0.001, "u m": 0.001}
    if normalized not in factors:
        raise ValidationError(f"unsupported morphology length unit {unit!r}")
    return factors[normalized]


def _load_numeric(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        values = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            key = next((name for name in ("height", "height_field", "z", "points", "xyz") if name in data.files), None)
            if key is None:
                raise ValidationError("morphology NPZ requires height/height_field/z/points/xyz")
            values = data[key]
    elif suffix == ".csv":
        try:
            values = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
            if values.dtype.names:
                names = values.dtype.names
                if {"x", "y", "z"}.issubset(names):
                    values = np.column_stack([values[name] for name in ("x", "y", "z")])
                elif "height" in names:
                    values = np.asarray(values["height"])
                else:
                    values = np.asarray([values[name] for name in names], dtype=float).T
            else:
                values = np.genfromtxt(path, delimiter=",")
        except (OSError, ValueError) as exc:
            raise ValidationError(f"invalid morphology CSV: {path}") from exc
    elif suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValidationError(f"invalid morphology JSON: {path}") from exc
        if isinstance(data, dict):
            values = data.get("height_field", data.get("height", data.get("points", data.get("xyz"))))
        else:
            values = data
    else:
        raise ValidationError("morphology input must be NPY, NPZ, CSV, or JSON")
    return finite_array(values, "morphology input", nonempty=True)


def _grid_from_values(values: np.ndarray, *, height_unit: str, spacing_x: float, spacing_y: float, spacing_unit: str) -> tuple[np.ndarray, float, float, str]:
    heights = np.asarray(values, dtype=float)
    if heights.ndim == 2:
        grid = heights
    elif heights.ndim == 1:
        side = int(round(math.sqrt(len(heights))))
        if side * side != len(heights):
            raise ValidationError("one-dimensional height field must have a square sample count")
        grid = heights.reshape(side, side)
    elif heights.ndim == 2 and heights.shape[1] >= 3:
        raise ValidationError("ambiguous morphology array")
    else:
        raise ValidationError("height field must be 2-D")
    if min(grid.shape) < 3:
        raise ValidationError("height field needs at least 3 x 3 samples")
    return grid * _unit_factor(height_unit), float(spacing_x) * _unit_factor(spacing_unit), float(spacing_y) * _unit_factor(spacing_unit), "height_field"


def _grid_from_points(values: np.ndarray, *, coordinate_unit: str, spacing_x: float | None, spacing_y: float | None, spacing_unit: str) -> tuple[np.ndarray, float, float, str]:
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) < 9:
        raise ValidationError("point cloud must be N x 3 with at least 9 points")
    points = points[:, :3] * _unit_factor(coordinate_unit)
    unique_x = np.unique(points[:, 0])
    unique_y = np.unique(points[:, 1])
    if spacing_x is None and len(unique_x) > 1:
        spacing_x = float(np.median(np.diff(unique_x)))
    if spacing_y is None and len(unique_y) > 1:
        spacing_y = float(np.median(np.diff(unique_y)))
    if spacing_x is None or spacing_y is None or spacing_x <= 0 or spacing_y <= 0:
        raise ValidationError("point cloud needs positive spacing or at least two unique x/y coordinates")
    nx, ny = len(unique_x), len(unique_y)
    if nx * ny != len(points):
        raise ValidationError("point cloud must lie on a complete rectangular grid")
    x_index = {value: index for index, value in enumerate(unique_x)}
    y_index = {value: index for index, value in enumerate(unique_y)}
    grid = np.empty((nx, ny), dtype=float)
    for x, y, z in points:
        grid[x_index[x], y_index[y]] = z
    return grid, float(spacing_x), float(spacing_y), "point_cloud_rectilinear"


def _roughness(grid: np.ndarray) -> dict[str, float]:
    reference = float(np.mean(grid))
    residual = grid - reference
    absolute = np.abs(residual)
    return {
        "reference_plane_height_mm": reference,
        "Ra_mm": float(np.mean(absolute)), "Rq_mm": float(np.sqrt(np.mean(residual ** 2))),
        "Rz_mm": float(np.percentile(grid, 95) - np.percentile(grid, 5)),
        "Ra_um": float(np.mean(absolute) * 1000.0), "Rq_um": float(np.sqrt(np.mean(residual ** 2)) * 1000.0),
        "Rz_um": float((np.percentile(grid, 95) - np.percentile(grid, 5)) * 1000.0),
        "height_min_mm": float(np.min(grid)), "height_max_mm": float(np.max(grid)),
    }


def _pit_components(grid: np.ndarray, spacing_x: float, spacing_y: float, threshold_mm: float) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    reference = float(np.percentile(grid, 95))
    depth = np.clip(reference - grid, 0.0, None)
    mask = depth >= float(threshold_mm)
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    objects: list[dict[str, Any]] = []
    pixel_area = spacing_x * spacing_y
    for label in range(1, count + 1):
        where = labels == label
        n = int(np.sum(where))
        if n == 0:
            continue
        local_depth = depth[where]
        coordinates = np.argwhere(where)
        height_extent = max((coordinates[:, 0].max() - coordinates[:, 0].min() + 1) * spacing_x, spacing_x)
        width_extent = max((coordinates[:, 1].max() - coordinates[:, 1].min() + 1) * spacing_y, spacing_y)
        objects.append({
            "pit_id": len(objects) + 1, "pixel_count": n, "area_mm2": float(n * pixel_area),
            "max_depth_mm": float(np.max(local_depth)), "mean_depth_mm": float(np.mean(local_depth)),
            "max_depth_um": float(np.max(local_depth) * 1000.0), "mean_depth_um": float(np.mean(local_depth) * 1000.0),
            "width_x_mm": float(height_extent), "width_y_mm": float(width_extent), "equivalent_width_mm": float(math.sqrt(4.0 * n * pixel_area / math.pi)),
            "volume_mm3": float(np.sum(local_depth) * pixel_area), "centroid_index": [float(item) for item in np.mean(coordinates, axis=0)],
        })
    return labels, objects, reference


def calculate_morphology_metrics(values: Any, *, input_kind: str = "auto", height_unit: str = "mm", coordinate_unit: str = "mm", spacing_x: float = 1.0, spacing_y: float = 1.0, spacing_unit: str = "mm", threshold: float | None = None, threshold_unit: str = "mm", height_uncertainty: float = 0.0, uncertainty_unit: str = "mm", source: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Calculate roughness and connected pit metrics from a field or point cloud."""
    array = finite_array(values, "morphology values", nonempty=True)
    if input_kind not in {"auto", "height_field", "point_cloud"}:
        raise ValidationError("input_kind must be auto, height_field, or point_cloud")
    if input_kind == "point_cloud" or (input_kind == "auto" and array.ndim == 2 and array.shape[1] >= 3 and array.shape[0] != array.shape[1]):
        grid, dx, dy, kind = _grid_from_points(array, coordinate_unit=coordinate_unit, spacing_x=spacing_x if spacing_x != 1.0 else None, spacing_y=spacing_y if spacing_y != 1.0 else None, spacing_unit=spacing_unit)
    else:
        grid, dx, dy, kind = _grid_from_values(array, height_unit=height_unit, spacing_x=spacing_x, spacing_y=spacing_y, spacing_unit=spacing_unit)
    height_sigma = float(height_uncertainty) * _unit_factor(uncertainty_unit)
    if not math.isfinite(height_sigma) or height_sigma < 0.0:
        raise ValidationError("height_uncertainty must be finite and non-negative")
    threshold_mm = float(threshold) * _unit_factor(threshold_unit) if threshold is not None else max(3.0 * height_sigma, float(np.std(grid - np.mean(grid))) * 2.0)
    labels, pits, reference = _pit_components(grid, dx, dy, threshold_mm)
    roughness = _roughness(grid)
    total_volume = float(sum(pit["volume_mm3"] for pit in pits))
    volume_sigma = float(height_sigma * dx * dy * np.sqrt(max(1, int(np.sum(labels > 0)))))
    uncertainty = {
        "height_1sigma_mm": height_sigma, "threshold_mm": threshold_mm, "volume_1sigma_mm3": volume_sigma,
        "Ra_1sigma_um": height_sigma * 1000.0 / math.sqrt(grid.size), "Rq_1sigma_um": height_sigma * 1000.0 / math.sqrt(2.0),
        "propagation": "independent height samples; volume sigma = sigma_z * pixel_area * sqrt(n_pit_pixels)",
    }
    result = {
        "version": MORPHOLOGY_VERSION, "input_kind": kind, "input_units": {"height": "mm", "spacing_x": "mm", "spacing_y": "mm"},
        "grid_shape": [int(item) for item in grid.shape], "spacing_mm": {"x": dx, "y": dy}, "n_samples": int(grid.size),
        "roughness": roughness, "pit_detection": {"reference_height_mm": reference, "threshold_mm": threshold_mm, "connectivity": 8, "n_pits": len(pits)},
        "pits": pits, "pit_summary": {"count": len(pits), "max_depth_um": float(max((p["max_depth_um"] for p in pits), default=0.0)), "mean_depth_um": float(np.mean([p["mean_depth_um"] for p in pits])) if pits else 0.0, "total_volume_mm3": total_volume, "volume_uncertainty_mm3": volume_sigma},
        "uncertainty": uncertainty, "source": dict(source or {}),
    }
    result["metrics_sha256"] = canonical_sha256(result)
    return result


def morphology_from_file(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    target = Path(path).resolve()
    result = calculate_morphology_metrics(_load_numeric(target), source={"path": str(target), "sha256": sha256_file(target)}, **kwargs)
    return result


def selftest() -> bool:
    x = np.linspace(-1.0, 1.0, 21)
    y = np.linspace(-1.0, 1.0, 21)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    height = 0.01 * np.sin(xx * 3.0) + 0.01 * np.cos(yy * 2.0)
    height[8:12, 9:13] -= 0.2
    result = calculate_morphology_metrics(height, height_unit="mm", spacing_x=0.1, spacing_y=0.1, spacing_unit="mm", height_uncertainty=0.002, uncertainty_unit="mm")
    assert result["roughness"]["Ra_um"] > 0 and result["roughness"]["Rq_um"] >= result["roughness"]["Ra_um"]
    assert result["pit_summary"]["count"] >= 1 and result["pit_summary"]["total_volume_mm3"] > 0
    points = np.column_stack((xx.reshape(-1), yy.reshape(-1), height.reshape(-1)))
    point_result = calculate_morphology_metrics(points, input_kind="point_cloud", coordinate_unit="mm", height_uncertainty=0.001)
    assert point_result["input_kind"] == "point_cloud_rectilinear"
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute interpretable roughness and pit morphology metrics")
    parser.add_argument("--input", help="height NPY/NPZ/CSV/JSON or point-cloud file")
    parser.add_argument("--input-kind", choices=("auto", "height_field", "point_cloud"), default="auto")
    parser.add_argument("--height-unit", default="mm")
    parser.add_argument("--coordinate-unit", default="mm")
    parser.add_argument("--spacing-x", type=float, default=1.0)
    parser.add_argument("--spacing-y", type=float, default=1.0)
    parser.add_argument("--spacing-unit", default="mm")
    parser.add_argument("--height-uncertainty", type=float, default=0.0)
    parser.add_argument("--uncertainty-unit", default="mm")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--threshold-unit", default="mm")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        selftest()
        print("step30_material_morphology_metrics selftest: PASS")
        return 0
    if not args.input:
        build_parser().error("--input is required unless --selftest is used")
    result = morphology_from_file(args.input, input_kind=args.input_kind, height_unit=args.height_unit, coordinate_unit=args.coordinate_unit, spacing_x=args.spacing_x, spacing_y=args.spacing_y, spacing_unit=args.spacing_unit, height_uncertainty=args.height_uncertainty, uncertainty_unit=args.uncertainty_unit, threshold=args.threshold, threshold_unit=args.threshold_unit)
    if args.output:
        safe_json_dump(result, args.output)
    print(json.dumps(jsonable(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
