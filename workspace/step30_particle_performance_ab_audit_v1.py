"""Audit semantic equality for the strict-v2 near-miss performance A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _canonical_array_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        array = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(repr(tuple(array.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key], copy=True) for key in archive.files}


def _summary_tracking(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    tracking = summary.get("tracking", {})
    return {
        "n_impacts": tracking.get("n_impacts"),
        "intersection_hits": tracking.get("intersection_hits"),
        "fallback_hits": tracking.get("fallback_hits"),
        "hit_parcels": tracking.get("hit_parcels"),
        "total_injection_mass_kg": tracking.get("total_injection_mass_kg"),
        "impact_mass_kg": tracking.get("impact_mass_kg"),
        "substep_count": tracking.get("substep_count"),
        "termination_counts": tracking.get("termination_counts"),
    }


def audit(baseline_dir: Path, optimized_dir: Path) -> dict[str, Any]:
    baseline_npz = baseline_dir / "particle_hits.npz"
    optimized_npz = optimized_dir / "particle_hits.npz"
    baseline_arrays = _load_npz(baseline_npz)
    optimized_arrays = _load_npz(optimized_npz)
    keys_equal = sorted(baseline_arrays) == sorted(optimized_arrays)
    array_comparison: dict[str, Any] = {}
    provenance_only_keys = {"particle_run_id"}
    arrays_equal = keys_equal
    physical_arrays_equal = keys_equal
    if keys_equal:
        for key in sorted(baseline_arrays):
            left = baseline_arrays[key]
            right = optimized_arrays[key]
            same = left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
            entry: dict[str, Any] = {
                "equal": bool(same),
                "dtype_baseline": str(left.dtype),
                "dtype_optimized": str(right.dtype),
                "shape_baseline": list(left.shape),
                "shape_optimized": list(right.shape),
            }
            if not same and np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number):
                delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
                entry["max_abs_difference"] = float(np.max(np.abs(delta))) if delta.size else 0.0
                entry["nonzero_difference_count"] = int(np.count_nonzero(delta))
            array_comparison[key] = entry
            arrays_equal = arrays_equal and bool(same)
            if key not in provenance_only_keys:
                physical_arrays_equal = physical_arrays_equal and bool(same)

    baseline_tracking = _summary_tracking(baseline_dir / "summary.json")
    optimized_tracking = _summary_tracking(optimized_dir / "summary.json")
    summary_equal = baseline_tracking == optimized_tracking
    baseline_summary = json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8"))
    optimized_summary = json.loads((optimized_dir / "summary.json").read_text(encoding="utf-8"))
    near_miss_baseline = baseline_summary.get("tracking", {}).get("near_miss_diagnostics")
    near_miss_optimized = optimized_summary.get("tracking", {}).get("near_miss_diagnostics")

    report = {
        "version": "step30-particle-performance-ab-audit-v1",
        "status": "passed" if physical_arrays_equal and summary_equal and near_miss_optimized == 0 else "failed",
        "baseline_dir": str(baseline_dir),
        "optimized_dir": str(optimized_dir),
        "container_byte_equal": _file_digest(baseline_npz) == _file_digest(optimized_npz),
        "container_sha256": {
            "baseline": _file_digest(baseline_npz),
            "optimized": _file_digest(optimized_npz),
        },
        "semantic_npz": {
            "keys_equal": keys_equal,
            "canonical_array_digest_equal": _canonical_array_digest(baseline_arrays)
            == _canonical_array_digest(optimized_arrays),
            "canonical_array_sha256": {
                "baseline": _canonical_array_digest(baseline_arrays),
                "optimized": _canonical_array_digest(optimized_arrays),
            },
            "arrays_equal": arrays_equal,
            "physical_arrays_equal": physical_arrays_equal,
            "provenance_only_keys": sorted(provenance_only_keys),
            "arrays": array_comparison,
        },
        "summary_tracking_equal": summary_equal,
        "tracking": {"baseline": baseline_tracking, "optimized": optimized_tracking},
        "near_miss_diagnostics": {
            "baseline": near_miss_baseline,
            "optimized": near_miss_optimized,
        },
        "interpretation": (
            "A container-byte difference is not a physical-result difference. "
            "The performance switch is semantically safe only when canonical arrays "
            "and accepted-impact tracking are equal; this audit does not establish spatial convergence."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--optimized-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.baseline_dir, args.optimized_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
