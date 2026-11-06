"""B-class bucket point-cloud/depth-field validation harness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from step30_experimental_validation import (
    MANIFEST_VERSION, add_common_cli, apply_registration, blocked_result, call_error_metrics,
    call_registration, finalize_run, load_depth, load_points, new_output_dir,
    normalise_mode, read_records, support_status, temp_dir,
)
from step30_generate_synthetic_measurements import _bucket

FIELDS = (
    "point_id", "parent_face_id", "x_original_mm", "y_original_mm", "z_original_mm",
    "x_worn_mm", "y_worn_mm", "z_worn_mm", "wear_depth_mm", "predicted_depth_mm",
)


def _normals(points: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    normals = np.empty_like(points, dtype=float)
    k = min(12, len(points))
    centroid = points.mean(axis=0)
    for index, point in enumerate(points):
        _, indices = tree.query(point, k=k)
        local = points[np.atleast_1d(indices)] - point
        _, vectors = np.linalg.eigh(local.T @ local)
        normal = vectors[:, 0]
        if np.dot(normal, point - centroid) < 0:
            normal = -normal
        normals[index] = normal / max(np.linalg.norm(normal), 1e-12)
    return normals


def _mapping(path: str | None, count: int) -> tuple[np.ndarray, np.ndarray, str]:
    if not path:
        point = np.arange(count, dtype=np.int32)
        return point, point.copy(), "derived_identity_mapping"
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as data:
            if "topology_mapping" in data.files:
                values = np.asarray(data["topology_mapping"])
                point, parent = values[:, 0], values[:, 1]
            else:
                point_key = next((key for key in ("topology_point_id", "point_id", "vertex_id") if key in data.files), None)
                face_key = next((key for key in ("topology_parent_face_id", "parent_face_id", "face_id") if key in data.files), None)
                if point_key is None or face_key is None:
                    raise ValueError("topology NPZ needs topology_mapping or point/parent-face arrays")
                point, parent = data[point_key], data[face_key]
    elif source.suffix.lower() == ".json":
        document = json.loads(source.read_text(encoding="utf-8"))
        values = document.get("topology_mapping", document.get("mapping", document)) if isinstance(document, dict) else document
        if isinstance(values, dict):
            point, parent = values.get("point_id", values.get("vertex_id")), values.get("parent_face_id", values.get("face_id"))
        else:
            array = np.asarray(values); point, parent = array[:, 0], array[:, 1]
    elif source.suffix.lower() in (".csv", ".txt"):
        rows = read_records(source)
        if not rows:
            raise ValueError("topology mapping is empty")
        point_name = next((key for key in ("point_id", "topology_point_id", "vertex_id") if key in rows[0]), None)
        face_name = next((key for key in ("parent_face_id", "topology_parent_face_id", "face_id") if key in rows[0]), None)
        if point_name is None or face_name is None:
            raise ValueError("topology mapping needs point_id and parent_face_id")
        point = [int(row[point_name]) for row in rows]; parent = [int(row[face_name]) for row in rows]
    else:
        raise ValueError(f"unsupported topology mapping format: {source.suffix}")
    point, parent = np.asarray(point, dtype=np.int32).reshape(-1), np.asarray(parent, dtype=np.int32).reshape(-1)
    if len(point) != count or len(parent) != count or len(np.unique(point)) != count or np.any(parent < 0):
        raise ValueError("topology mapping does not match points or has invalid IDs")
    return point, parent, "provided_topology_mapping"


def _prediction(points: np.ndarray) -> np.ndarray:
    x = (points[:, 0] - points[:, 0].min()) / max(np.ptp(points[:, 0]), 1e-12)
    y = (points[:, 1] - points[:, 1].min()) / max(np.ptp(points[:, 1]), 1e-12)
    return 0.08 + 0.92 * np.exp(-((x - 0.55) ** 2 / 0.055 + (y - 0.48) ** 2 / 0.09))


def _measured(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any], tuple[str, ...]]:
    if args.depth_field:
        depth, coordinates, depth_io = load_depth(args.depth_field, status="measured")
        if args.original_points:
            original, point_io = load_points(args.original_points, unit_scale=args.unit_scale, status="measured")
        elif coordinates is not None:
            original, point_io = np.asarray(coordinates, dtype=float), {"used": True, "source": "depth_field_coordinates"}
        else:
            raise ValueError("depth-field input needs --original-points or coordinates in the field")
        if len(depth) != len(original):
            raise ValueError("depth field length differs from original points")
        worn = original - _normals(original) * depth[:, None]
        return original, worn, {"depth_source": "provided_depth_field", "depth_io": depth_io, "registration_status": "not_required"}, tuple(item for item in (args.depth_field, args.original_points, args.topology_map) if item)
    if not args.original_points or not args.worn_points:
        raise ValueError("measured mode needs --depth-field or both --original-points and --worn-points")
    original, original_io = load_points(args.original_points, unit_scale=args.unit_scale, status="measured")
    worn_raw, worn_io = load_points(args.worn_points, unit_scale=args.unit_scale, status="measured")
    registration, registration_io = call_registration(worn_raw, original, status="measured")
    if registration is None or not registration.get("passed"):
        raise ValueError(f"coordinate registration did not pass: {registration or registration_io}")
    worn = apply_registration(worn_raw, registration)
    if len(worn) != len(original):
        raise ValueError("registered worn point count differs from original point count")
    normals = _normals(original)
    depth = np.clip(-np.einsum("ij,ij->i", worn - original, normals), 0.0, None)
    provenance = {"depth_source": "registered_point_clouds", "registration": registration, "registration_io": registration_io, "point_io": {"original": original_io, "worn": worn_io}, "wear_depth_mm": depth}
    return original, worn, provenance, tuple(item for item in (args.original_points, args.worn_points, args.topology_map) if item)


def run(args: argparse.Namespace) -> dict[str, Any]:
    mode = normalise_mode(args.mode)
    output = new_output_dir(args.output_dir, "bucket-model")
    if mode == "blocked":
        return blocked_result(output, "bucket_model", mode, args.seed, "blocked_by_request", required_inputs=("--depth-field or --original-points + --worn-points",))
    if mode == "synthetic":
        data = _bucket(args.seed)
        original, worn, depth = data["original_xyz_mm"], data["worn_xyz_mm"], data["wear_depth_mm"]
        point, parent, mapping_source = data["topology_point_id"], data["topology_parent_face_id"], "fixed_seed_generator"
        provenance, provided, status, truth_source = {"depth_source": "known_synthetic_depth_field"}, (), "synthetic", "fixed_seed_generator"
    else:
        try:
            original, worn, provenance, provided = _measured(args)
            depth = np.asarray(provenance.get("wear_depth_mm"), dtype=float) if provenance["depth_source"] == "registered_point_clouds" else np.asarray(load_depth(args.depth_field, status="measured")[0])
            point, parent, mapping_source = _mapping(args.topology_map, len(original))
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            provided = tuple(item for item in (args.depth_field, args.original_points, args.worn_points, args.topology_map) if item)
            return blocked_result(output, "bucket_model", mode, args.seed, f"invalid measured bucket input: {exc}", required_inputs=("--depth-field or --original-points + --worn-points",), provided_inputs=provided)
        status, truth_source = "measured", "provided_real_wear_measurement"
    if mode == "synthetic":
        mapping_source = "fixed_seed_generator"
    predicted = _prediction(original)
    metrics, metric_io = call_error_metrics(predicted, depth, status=mode)
    rows = [{"point_id": int(point[i]), "parent_face_id": int(parent[i]), "x_original_mm": original[i, 0], "y_original_mm": original[i, 1], "z_original_mm": original[i, 2], "x_worn_mm": worn[i, 0], "y_worn_mm": worn[i, 1], "z_worn_mm": worn[i, 2], "wear_depth_mm": depth[i], "predicted_depth_mm": predicted[i]} for i in range(len(original))]
    arrays = {"original_xyz_mm": original, "worn_xyz_mm": worn, "wear_depth_mm": depth, "predicted_depth_mm": predicted, "topology_point_id": point, "topology_parent_face_id": parent, "topology_mapping": np.column_stack((point, parent))}
    summary = {
        "version": "step30-bucket-model-experiment-v1", "manifest_version": MANIFEST_VERSION,
        "status": status, "mode": mode, "measurement_status": "known_synthetic_truth" if mode == "synthetic" else "real_input_loaded", "truth_source": truth_source,
        "point_count": int(len(original)), "wear_depth_mm": {"min": float(depth.min()), "median": float(np.median(depth)), "max": float(depth.max()), "p95": float(np.percentile(depth, 95))},
        "prediction": {"name": "coordinate_gaussian_baseline", "formula": "0.08+0.92*exp(-((x_norm-.55)^2/.055+(y_norm-.48)^2/.09))", "comparison": metrics, "error_metrics_interface": metric_io},
        "topology": {"mapping_source": mapping_source, "mapping_columns": ["point_id", "parent_face_id"], "parent_face_count": int(len(np.unique(parent)))},
        "provenance": provenance, "stress_used_as_erosion_truth": False,
    }
    return finalize_run(output, "bucket_model", mode, args.seed, status, summary, {"bucket_measurements": (rows, FIELDS)}, arrays, required_inputs=("--depth-field or --original-points + --worn-points",) if mode == "measured" else (), provided_inputs=provided, support=support_status(), evidence={"status": "synthetic", "source_type": "local_synthetic_fixture", "reference_role": "synthetic_pipeline_fixture", "pipeline_validation": True} if mode == "synthetic" else None, notes=("Original/worn point clouds are registered before depth extraction.", "Topology mapping is retained as point_id to parent_face_id.", "Measured depth is never replaced by the prediction."))


def selftest() -> None:
    import argparse as ap
    result = run(ap.Namespace(mode="synthetic", output_dir=str(temp_dir("bucket")), seed=13, depth_field=None, original_points=None, worn_points=None, topology_map=None, unit_scale=1.0))
    assert result["status"] == "synthetic" and result["tables"]["bucket_measurements"]["rows"] == 140
    blocked = run(ap.Namespace(mode="measured", output_dir=str(temp_dir("bucket-blocked")), seed=13, depth_field=None, original_points=None, worn_points=None, topology_map=None, unit_scale=1.0))
    assert blocked["status"] == "blocked"
    print("step30_run_bucket_model_experiment selftest: PASS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_common_cli(parser, "Run B-class bucket model validation")
    parser.add_argument("--depth-field", default=None)
    parser.add_argument("--original-points", default=None)
    parser.add_argument("--worn-points", default=None)
    parser.add_argument("--topology-map", default=None)
    parser.add_argument("--unit-scale", type=float, default=1.0)
    return parser


if __name__ == "__main__":
    parser = build_parser(); args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        result = run(args); print(json.dumps(result, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)); raise SystemExit(0 if result["status"] in {"synthetic", "measured"} else 2)
