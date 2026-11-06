"""B-class field metadata and wear validation harness.

Stress context is retained for operating-condition interpretation only.  It is
never treated as erosion truth; erosion truth must come from a wear-depth
field or from original/worn geometry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from step30_experimental_validation import (
    MANIFEST_VERSION, add_common_cli, blocked_result, call_error_metrics,
    finalize_run, load_depth, load_points, new_output_dir, normalise_mode,
    read_records, support_status, temp_dir,
)
from step30_generate_synthetic_measurements import _bucket, _field
from step30_run_bucket_model_experiment import _normals

FIELDS = (
    "operating_point", "power_mw", "head_m", "flow_m3s",
    "stress_mean_mpa", "stress_amplitude_mpa", "dominant_frequency_hz",
    "wear_truth_role",
)


def _read_metadata(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("field metadata JSON must be an object")
        return value
    rows = read_records(source)
    if not rows:
        raise ValueError("field metadata table is empty")
    return {"rows": rows}


def _read_stress(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(value, list):
            return [dict(row) for row in value]
        if isinstance(value, dict):
            rows = value.get("rows", value.get("stress_context", value.get("data")))
            return [dict(row) for row in rows] if isinstance(rows, list) else [dict(value)]
        raise ValueError("stress context JSON has no rows")
    rows = read_records(source)
    if not rows:
        raise ValueError("stress context table is empty")
    return rows


def _stress_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"provided": False, "rows": 0, "role": "context_only_not_erosion_truth"}
    columns = {key for row in rows for key in row}
    if not columns.intersection({"stress_mean_mpa", "stress_amplitude_mpa", "stress_mpa", "sigma_a_mpa"}):
        raise ValueError("stress context lacks a stress mean/amplitude column")
    return {
        "provided": True,
        "rows": len(rows),
        "columns": sorted(columns),
        "role": "context_only_not_erosion_truth",
        "truth_exclusion": "stress values are not used to construct wear_depth_mm or erosion labels",
    }


def _context_rows(metadata: dict[str, Any], stress: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source = stress
    if not source and isinstance(metadata.get("rows"), list):
        source = [dict(row) for row in metadata["rows"]]
    if not source:
        source = [metadata] if metadata else [{"operating_point": "field"}]
    result = []
    for index, row in enumerate(source):
        item = dict(row)
        item.setdefault("operating_point", f"field-{index + 1:03d}")
        item["wear_truth_role"] = "context_only_not_erosion_truth"
        result.append(item)
    return result


def _measured(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]], tuple[str, ...]]:
    metadata = _read_metadata(args.metadata)
    stress = _read_stress(args.stress_context)
    if args.depth_field:
        depth, coordinates, depth_io = load_depth(args.depth_field, status="measured")
        if args.original_points:
            original, point_io = load_points(args.original_points, unit_scale=args.unit_scale, status="measured")
        elif coordinates is not None:
            original, point_io = np.asarray(coordinates, dtype=float), {"used": True, "source": "depth_field_coordinates"}
        else:
            raise ValueError("depth field requires --original-points or coordinates in the field")
        if len(depth) != len(original):
            raise ValueError("depth field length differs from original points")
        io = {"depth": depth_io, "points": point_io, "source": "provided_depth_field"}
    else:
        if not args.original_points or not args.worn_points:
            raise ValueError("field mode needs --depth-field or both --original-points and --worn-points")
        original, original_io = load_points(args.original_points, unit_scale=args.unit_scale, status="measured")
        worn, worn_io = load_points(args.worn_points, unit_scale=args.unit_scale, status="measured")
        if original.shape != worn.shape:
            raise ValueError("original and worn point clouds must have equal shape")
        depth = np.clip(-np.einsum("ij,ij->i", worn - original, _normals(original)), 0.0, None)
        io = {"points": {"original": original_io, "worn": worn_io}, "source": "original_worn_geometry"}
    return original, np.asarray(depth, dtype=float), metadata, stress, tuple(item for item in (args.depth_field, args.original_points, args.worn_points, args.metadata, args.stress_context) if item)


def run(args: argparse.Namespace) -> dict[str, Any]:
    mode = normalise_mode(args.mode)
    output = new_output_dir(args.output_dir, "field-validation")
    if mode == "blocked":
        return blocked_result(output, "field_validation", mode, args.seed, "blocked_by_request", required_inputs=("--depth-field or --original-points + --worn-points",))
    if mode == "synthetic":
        bucket = _bucket(args.seed)
        original, depth = bucket["original_xyz_mm"], bucket["wear_depth_mm"]
        field_rows, metadata = _field()
        stress = field_rows
        io = {"implementation": "fixed_seed_generator", "source": "known_synthetic_depth_field"}
        provided, status, truth_source = (), "synthetic", "fixed_seed_generator_wear_depth"
    else:
        try:
            original, depth, metadata, stress, provided = _measured(args)
            stress_summary = _stress_summary(stress)
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            provided = tuple(item for item in (args.depth_field, args.original_points, args.worn_points, args.metadata, args.stress_context) if item)
            return blocked_result(output, "field_validation", mode, args.seed, f"invalid measured field input: {exc}", required_inputs=("--depth-field or --original-points + --worn-points",), provided_inputs=provided)
        io, status, truth_source = {"implementation": "provided_real_input"}, "measured", "provided_real_wear_measurement"
    if mode == "synthetic":
        stress_summary = _stress_summary(stress)
    if len(depth) != len(original) or len(depth) == 0 or not np.isfinite(depth).all() or np.any(depth < 0):
        return blocked_result(output, "field_validation", mode, args.seed, "wear depth is invalid or does not match geometry", required_inputs=("wear_depth_mm",), provided_inputs=provided)
    x = (original[:, 0] - original[:, 0].min()) / max(np.ptp(original[:, 0]), 1e-12)
    y = (original[:, 1] - original[:, 1].min()) / max(np.ptp(original[:, 1]), 1e-12)
    prediction = 0.08 + 0.92 * np.exp(-((x - 0.55) ** 2 / 0.055 + (y - 0.48) ** 2 / 0.09))
    metrics, metric_io = call_error_metrics(prediction, depth, status=mode)
    rows = _context_rows(metadata, stress)
    summary = {
        "version": "step30-field-validation-v1", "manifest_version": MANIFEST_VERSION,
        "status": status, "mode": mode, "measurement_status": "known_synthetic_truth" if mode == "synthetic" else "real_input_loaded",
        "truth_source": truth_source, "point_count": int(len(original)), "field_metadata": metadata,
        "stress_context": stress_summary, "stress_used_as_erosion_truth": False,
        "erosion_truth_policy": "wear_depth_mm only; stress context is not erosion truth",
        "prediction": {"name": "coordinate_gaussian_baseline", "comparison": metrics, "error_metrics_interface": metric_io},
        "wear_depth_mm": {"min": float(depth.min()), "median": float(np.median(depth)), "max": float(depth.max()), "p95": float(np.percentile(depth, 95))},
        "measurement_io_interface": io,
    }
    return finalize_run(output, "field_validation", mode, args.seed, status, summary, {"field_metadata_stress_context": (rows, FIELDS)}, {"original_xyz_mm": original, "wear_depth_mm": depth, "predicted_depth_mm": prediction}, required_inputs=("--depth-field or --original-points + --worn-points",) if mode == "measured" else (), provided_inputs=provided, support=support_status(), evidence={"status": "synthetic", "source_type": "local_synthetic_fixture", "reference_role": "synthetic_pipeline_fixture", "pipeline_validation": True} if mode == "synthetic" else None, notes=("Metadata and stress are retained as context.", "Stress is explicitly excluded from erosion truth construction.", "Measured mode blocks on missing/invalid wear geometry."))


def selftest() -> None:
    import argparse as ap
    result = run(ap.Namespace(mode="synthetic", output_dir=str(temp_dir("field")), seed=19, depth_field=None, original_points=None, worn_points=None, metadata=None, stress_context=None, unit_scale=1.0))
    summary = json.loads((Path(result["output_dir"]) / "step30-field_validation-summary.json").read_text(encoding="utf-8"))
    assert result["status"] == "synthetic" and summary["stress_used_as_erosion_truth"] is False
    blocked = run(ap.Namespace(mode="measured", output_dir=str(temp_dir("field-blocked")), seed=19, depth_field=None, original_points=None, worn_points=None, metadata=None, stress_context=None, unit_scale=1.0))
    assert blocked["status"] == "blocked"
    print("step30_run_field_validation selftest: PASS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_common_cli(parser, "Run B-class field metadata and wear validation")
    parser.add_argument("--depth-field", default=None)
    parser.add_argument("--original-points", default=None)
    parser.add_argument("--worn-points", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--stress-context", default=None)
    parser.add_argument("--unit-scale", type=float, default=1.0)
    return parser


if __name__ == "__main__":
    parser = build_parser(); args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        result = run(args); print(json.dumps(result, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)); raise SystemExit(0 if result["status"] in {"synthetic", "measured"} else 2)
