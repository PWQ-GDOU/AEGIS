"""Validate predicted wear against independent measured/synthetic arrays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from step30_error_metrics import calculate_error_metrics
from step30_validation_common import file_sha256, safe_json_dump


def _load(path: Path, key: str) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if key not in data.files:
                raise ValueError(f"{path} has no array {key}")
            values = data[key]
    else:
        delimiter = "," if path.suffix.lower() == ".csv" else None
        values = np.loadtxt(path, delimiter=delimiter)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def run(predicted_path: Path, observed_path: Path, output: Path, *, predicted_key: str, observed_key: str, status: str, independent: bool, quality_passed: bool, source_manifest: Path | None = None, split_manifest: Path | None = None, quality_manifest: Path | None = None) -> dict:
    predicted = _load(predicted_path, predicted_key)
    observed = _load(observed_path, observed_key)
    if predicted.shape != observed.shape:
        raise ValueError("predicted and observed arrays differ in length")
    metrics = calculate_error_metrics(
        predicted, observed, status=status, value_units="mm",
        independent_validation_set={
            "is_independent": bool(independent), "n": int(len(observed)),
            "source_hashes": {"predicted": file_sha256(predicted_path), "observed": file_sha256(observed_path)},
        },
        quality_gates={"registration_quality": bool(quality_passed), "finite": bool(np.isfinite(predicted).all() and np.isfinite(observed).all())},
    )
    # The legacy array CLI is diagnostic-only unless it is bound to the full
    # measured evidence package. Caller flags and array hashes are insufficient.
    evidence_complete = all(path is not None for path in (source_manifest, split_manifest, quality_manifest))
    if status == "measured" and not evidence_complete:
        metrics["official_accuracy_gate"] = False
        metrics["provenance_note"] = "legacy array comparison blocked; use step30_measured_validation with source/split/quality manifests"
        metrics["legacy_gate_blocked"] = True
    error_below_5pct = all(metrics[name]["relative"] < 0.05 for name in ("total_error", "maximum_depth_error"))
    metrics["declared_error_below_5pct"] = bool(error_below_5pct)
    metrics["official_accuracy_gate"] = bool(metrics["official_accuracy_gate"] and error_below_5pct)
    metrics["input_files"] = {"predicted": str(predicted_path), "observed": str(observed_path)}
    metrics.pop("metrics_sha256", None)
    from step30_validation_common import canonical_sha256
    metrics["metrics_sha256"] = canonical_sha256(metrics)
    safe_json_dump(metrics, output)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--predicted-key", default="predicted_depth_mm")
    parser.add_argument("--observed-key", default="wear_depth_mm")
    parser.add_argument("--status", choices=("measured", "synthetic", "blocked"), required=True)
    parser.add_argument("--independent", action="store_true")
    parser.add_argument("--quality-passed", action="store_true")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--quality-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.predicted, args.observed, args.output, predicted_key=args.predicted_key, observed_key=args.observed_key, status=args.status, independent=args.independent, quality_passed=args.quality_passed, source_manifest=args.source_manifest, split_manifest=args.split_manifest, quality_manifest=args.quality_manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["official_accuracy_gate"] or args.status == "synthetic" else 3)
