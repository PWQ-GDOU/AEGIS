"""Shared utilities and unified CLI for the Step30 B-class experiment harness."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

MANIFEST_VERSION = "step30-experiment-manifest-v1"
MODES = ("synthetic", "measured", "blocked")
SUPPORT_MODULES = (
    "step30_measurement_io",
    "step30_coordinate_registration",
    "step30_error_metrics",
)


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    if value not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    return value


def new_output_dir(requested: str | Path | None, label: str) -> Path:
    """Create a fresh run directory; an existing directory is never replaced."""
    base = Path(requested) if requested else Path.cwd() / "output" / "step30_experiments" / label
    candidate = base
    number = 1
    while candidate.exists():
        candidate = Path(f"{base}__run-{number:03d}")
        number += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def support_status() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in SUPPORT_MODULES:
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            result[name] = {"available": False, "implementation": "local_fallback", "error": type(exc).__name__}
        else:
            result[name] = {"available": True, "implementation": "external_module", "module": str(getattr(module, "__file__", name))}
    return result


def call_measurement_reader(path: str | Path, kind: str, *, status: str = "measured") -> tuple[Any | None, dict[str, Any]]:
    try:
        module = importlib.import_module("step30_measurement_io")
        function = getattr(module, "read_measurement")
        return function(path, kind=kind, status=status), {"used": True, "function": "read_measurement", "implementation": "external_module"}
    except Exception as exc:
        return None, {"used": False, "implementation": "local_fallback", "reason": type(exc).__name__}


def _local_registration(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Small row-vector ICP fallback for a temporarily unavailable support module."""
    from scipy.spatial import cKDTree
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != 3 or target.shape[1] != 3 or len(source) < 3 or len(target) < 3:
        raise ValueError("registration requires two N x 3 point clouds with N >= 3")
    initial_translation = target.mean(axis=0) - source.mean(axis=0)
    aligned = source + initial_translation
    tree = cKDTree(target)
    last_distance = float("inf")
    total_rotation = np.eye(3)
    total_translation = initial_translation.copy()
    for _ in range(40):
        distances, indices = tree.query(aligned, k=1)
        paired = target[indices]
        source_mean = aligned.mean(axis=0)
        target_mean = paired.mean(axis=0)
        covariance = (aligned - source_mean).T @ (paired - target_mean)
        u, _, vt = np.linalg.svd(covariance)
        correction = np.eye(3)
        if np.linalg.det(vt.T @ u.T) < 0.0:
            correction[-1, -1] = -1.0
        rotation = vt.T @ correction @ u.T
        translation = target_mean - source_mean @ rotation
        updated = aligned @ rotation + translation
        total_rotation = total_rotation @ rotation
        total_translation = total_translation @ rotation + translation
        mean_distance = float(np.mean(distances))
        aligned = updated
        if abs(last_distance - mean_distance) < 1e-8:
            break
        last_distance = mean_distance
    final_distances = tree.query(aligned, k=1)[0]
    passed = bool(float(np.percentile(final_distances, 95)) <= 5.0)
    return {
        "version": "step30-coordinate-registration-local-fallback-v1",
        "status": status if passed else "blocked",
        "input_status": status,
        "passed": passed,
        "registration": {"R": total_rotation, "translation": total_translation, "convention": "row-vector: target=source@R+translation"},
        "diagnostics": {"p95_distance_mm": float(np.percentile(final_distances, 95)), "max_distance_mm": float(np.max(final_distances))},
    }


def apply_registration(points: np.ndarray, registration: Mapping[str, Any]) -> np.ndarray:
    transform = registration.get("registration", registration)
    rotation = np.asarray(transform["R"], dtype=float)
    translation = np.asarray(transform["translation"], dtype=float)
    return np.asarray(points, dtype=float) @ rotation + translation


def call_registration(source: np.ndarray, target: np.ndarray, *, status: str = "measured") -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        module = importlib.import_module("step30_coordinate_registration")
        function = getattr(module, "register_and_diagnose")
        result = function(source, target, threshold=5.0, min_coverage=0.95, status=status)
        return result, {"used": True, "function": "register_and_diagnose", "implementation": "external_module"}
    except Exception as exc:
        try:
            return _local_registration(source, target), {"used": False, "implementation": "local_fallback", "reason": type(exc).__name__}
        except Exception as fallback_exc:
            return None, {"used": False, "implementation": "local_fallback", "reason": type(fallback_exc).__name__, "external_error": type(exc).__name__}


def call_error_metrics(predicted: Sequence[float], observed: Sequence[float], *, status: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        module = importlib.import_module("step30_error_metrics")
        result = module.calculate_error_metrics(predicted, observed, status=status, value_units="mm", independent_validation_set=(status == "measured"), quality_passed=True)
        return dict(result), {"used": True, "function": "calculate_error_metrics", "implementation": "external_module"}
    except Exception as exc:
        pred = np.asarray(predicted, dtype=float)
        obs = np.asarray(observed, dtype=float)
        residual = pred - obs
        total = max(float(np.sum(np.abs(obs))), 1e-30)
        return {
            "version": "local-error-metrics-v1",
            "status": status,
            "n": int(len(pred)),
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "max_abs_error": float(np.max(np.abs(residual))),
            "relative_l1_error": float(np.sum(np.abs(residual)) / total),
        }, {"used": False, "implementation": "local_fallback", "reason": type(exc).__name__}


def read_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            return [dict(row) for row in csv.DictReader(stream)]
    if source.suffix.lower() == ".json":
        document = json.loads(source.read_text(encoding="utf-8"))
        rows = document if isinstance(document, list) else document.get("rows", document.get("data", document.get("measurements")))
        if not isinstance(rows, list):
            rows = [document] if isinstance(document, dict) else []
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError("record JSON rows must be objects")
        return [dict(row) for row in rows]
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as data:
            if "rows_json" in data.files:
                rows = json.loads(str(data["rows_json"].item()))
                return [dict(row) for row in rows]
            arrays = {key: np.asarray(data[key]) for key in data.files}
        lengths = {len(value) for value in arrays.values() if value.ndim > 0}
        if len(lengths) != 1:
            raise ValueError("NPZ arrays have inconsistent lengths")
        length = lengths.pop() if lengths else 0
        return [{key: (value[index].item() if value.ndim == 1 else value[index].tolist()) for key, value in arrays.items()} for index in range(length)]
    raise ValueError(f"unsupported record format: {source.suffix}")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    records = [dict(row) for row in rows]
    names = list(columns or [])
    for row in records:
        for key in row:
            if key not in names:
                names.append(key)
    if not names:
        names = ["status"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for row in records:
            writer.writerow({name: row.get(name, "") for name in names})


def load_points(path: str | Path, *, unit_scale: float = 1.0, status: str = "measured") -> tuple[np.ndarray, dict[str, Any]]:
    record, interface = call_measurement_reader(path, "point_cloud", status=status)
    if record is not None:
        points = np.asarray(record["points"], dtype=float)
        return points * float(unit_scale), interface
    source = Path(path)
    if source.suffix.lower() == ".npy":
        points = np.load(source, allow_pickle=False)
    elif source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as data:
            key = next((name for name in ("xyz", "points", "points_mm", "vertices", "original_xyz_mm", "worn_xyz_mm") if name in data.files), None)
            if key is None:
                raise ValueError("point-cloud NPZ needs xyz/points/points_mm/vertices/original_xyz_mm/worn_xyz_mm")
            points = data[key]
    elif source.suffix.lower() in (".xyz", ".txt", ".csv"):
        points = np.loadtxt(source, delimiter="," if source.suffix.lower() == ".csv" else None, skiprows=1 if source.suffix.lower() == ".csv" else 0)
    elif source.suffix.lower() == ".json":
        document = json.loads(source.read_text(encoding="utf-8"))
        points = document.get("points", document.get("xyz")) if isinstance(document, dict) else document
    else:
        raise ValueError(f"fallback point reader does not support {source.suffix}")
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) < 3 or not np.isfinite(points[:, :3]).all():
        raise ValueError("points must be finite N x 3 with N >= 3")
    return points[:, :3] * float(unit_scale), interface


def load_depth(path: str | Path, *, status: str = "measured") -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    record, interface = call_measurement_reader(path, "depth_field", status=status)
    if record is not None:
        values = np.asarray(record.get("values", record.get("grid_values")), dtype=float).reshape(-1)
        coordinates = record.get("coordinates")
        coordinates = None if coordinates is None else np.asarray(coordinates, dtype=float)
        return values, coordinates, interface
    source = Path(path)
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as data:
            key = next((name for name in ("wear_depth_mm", "depth_mm", "depth", "values") if name in data.files), None)
            if key is None:
                raise ValueError("depth NPZ needs wear_depth_mm/depth_mm/depth/values")
            values = np.asarray(data[key], dtype=float).reshape(-1)
            coordinates = np.asarray(data["xyz"], dtype=float) if "xyz" in data.files else None
    else:
        values = np.asarray(np.load(source, allow_pickle=False) if source.suffix.lower() == ".npy" else [float(row.get("wear_depth_mm", row.get("depth_mm", row.get("depth")))) for row in read_records(source)], dtype=float).reshape(-1)
        coordinates = None
    if len(values) == 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("depth must be finite, non-negative, and non-empty")
    return values, coordinates, interface


def finalize_run(output_dir: Path, experiment: str, mode: str, seed: int, status: str, summary: Mapping[str, Any], tables: Mapping[str, tuple[Sequence[Mapping[str, Any]], Sequence[str] | None]], arrays: Mapping[str, Any], *, required_inputs: Sequence[str] = (), provided_inputs: Sequence[str | Path] = (), support: Mapping[str, Any] | None = None, notes: Sequence[str] = (), evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"step30-{experiment}"
    from step30_validation_common import canonical_sha256, evidence_metadata, safe_json_dump
    if evidence is None:
        evidence = (
            {"status": status, "source_type": "local_synthetic_fixture", "reference_role": "synthetic_pipeline_fixture", "pipeline_validation": True}
            if status == "synthetic"
            else {"status": status}
        )
    evidence_value = evidence_metadata(evidence, status=status)
    summary_value = dict(summary)
    summary_value.update({key: value for key, value in evidence_value.items() if key != "status"})
    artifacts: list[Path] = []
    summary_path = output_dir / f"{prefix}-summary.json"
    write_json(summary_path, summary_value)
    artifacts.append(summary_path)
    table_info: dict[str, Any] = {}
    for name, (rows, columns) in tables.items():
        path = output_dir / f"{prefix}-{name}.csv"
        write_csv(path, rows, columns)
        artifacts.append(path)
        table_info[name] = {"file": path.name, "rows": len(rows), "columns": list(columns or [])}
    npz_path = output_dir / f"{prefix}-arrays.npz"
    np.savez_compressed(npz_path, **{key: np.asarray(value) for key, value in arrays.items()})
    artifacts.append(npz_path)
    manifest = {
        "version": MANIFEST_VERSION,
        "experiment": experiment,
        "mode": mode,
        "status": status,
        **evidence_value,
        "seed": int(seed),
        "output_dir": str(output_dir),
        "required_inputs": list(required_inputs),
        "provided_inputs": [{"path": str(item), "exists": Path(item).is_file(), "sha256": sha256_file(Path(item)) if Path(item).is_file() else None} for item in provided_inputs],
        "synthetic_policy": "synthetic values only when mode=synthetic",
        "synthetic_truth": dict(summary.get("synthetic_truth", {})) if mode == "synthetic" else {"status": "not_applicable"},
        "measured_policy": "missing or invalid real data blocks; no fabricated measurements",
        "support_interfaces": dict(support or support_status()),
        "tables": table_info,
        "artifacts": [{"file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in artifacts],
        "notes": list(notes),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = output_dir / f"{MANIFEST_VERSION}.json"
    safe_json_dump(manifest, manifest_path)
    return manifest


def blocked_result(output_dir: Path, experiment: str, mode: str, seed: int, reason: str, *, required_inputs: Sequence[str], provided_inputs: Sequence[str | Path] = (), support: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return finalize_run(
        output_dir, experiment, mode, seed, "blocked",
        {"version": MANIFEST_VERSION, "experiment": experiment, "mode": mode, "status": "blocked", "measurement_status": "not_available", "truth_source": "none", "blocked_reason": reason},
        {"measurements": ([{"status": "blocked", "blocked_reason": reason}], ["status", "blocked_reason"])}
        , {"status_code": np.asarray([1], dtype=np.int8)}, required_inputs=required_inputs, provided_inputs=provided_inputs, support=support, notes=("Blocked output contains no fabricated measurement values.",)
    )


def add_common_cli(parser: argparse.ArgumentParser, description: str) -> None:
    parser.description = description
    parser.add_argument("--mode", choices=MODES, default="synthetic")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selftest", action="store_true")


def temp_dir(label: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"step30-{label}-"))


def unified_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unified Step30 B-class experimental validation")
    add_common_cli(parser, parser.description)
    parser.add_argument("--experiment", choices=("measurements", "material", "bucket", "field"), default="measurements")
    parser.add_argument("--input", default=None)
    parser.add_argument("--depth-field", default=None)
    parser.add_argument("--original-points", default=None)
    parser.add_argument("--worn-points", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--stress-context", default=None)
    args = parser.parse_args(argv)
    # The individual runners accept a few extra options; unified CLI keeps
    # their namespace total so dispatch is deterministic.
    for name in ("topology_map", "unit_scale"):
        if not hasattr(args, name):
            setattr(args, name, None if name == "topology_map" else 1.0)
    if args.selftest:
        modules = {
            "measurements": "step30_generate_synthetic_measurements",
            "material": "step30_run_material_erosion_experiment",
            "bucket": "step30_run_bucket_model_experiment",
            "field": "step30_run_field_validation",
        }
        module = importlib.import_module(modules[args.experiment])
        module.selftest()
        return 0
    modules = {
        "measurements": "step30_generate_synthetic_measurements",
        "material": "step30_run_material_erosion_experiment",
        "bucket": "step30_run_bucket_model_experiment",
        "field": "step30_run_field_validation",
    }
    module = importlib.import_module(modules[args.experiment])
    result = module.run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=json_default))
    return 0 if result.get("status") in {"synthetic", "measured"} else 2


if __name__ == "__main__":
    raise SystemExit(unified_main())
