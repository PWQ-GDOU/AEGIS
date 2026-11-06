"""Validate and mass-conservatively merge independent particle-hit runs.

Each input is one Monte Carlo seed for the same exposure.  Concatenating the
hits without rescaling would multiply the sediment exposure by the number of
seeds, so every parcel mass is divided by the number of accepted sources.
The merged artifact is an exploratory ensemble input, never an official CFD
or spatial-accuracy reference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from step30_validation_common import canonical_sha256, safe_json_dump, safe_npz_dump, sha256_file


VERSION = "step30-particle-hits-ensemble-v1"
REQUIRED_KEYS = (
    "parent_face_id",
    "velocity_ms",
    "Vp_ms",
    "alpha_deg",
    "diameter_m",
    "parcel_mass_kg",
    "face_centers_local_mm",
    "face_area_m2",
)


class ParticleEnsembleError(ValueError):
    """Raised when independent hit artifacts cannot share one face domain."""


def _finite(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0 or not np.isfinite(array).all():
        raise ParticleEnsembleError(f"{name} must be non-empty and finite")
    return array


def _domain_fingerprint(centers: np.ndarray, areas: np.ndarray) -> str:
    payload = centers.astype("<f8", copy=False).tobytes() + areas.astype("<f8", copy=False).tobytes()
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _load_hits(path: str | Path, index: int) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise ParticleEnsembleError(f"hit artifact does not exist: {source}")
    with np.load(source, allow_pickle=False) as data:
        missing = sorted(set(REQUIRED_KEYS) - set(data.files))
        if missing:
            raise ParticleEnsembleError(f"{source} is missing fields: {missing}")
        arrays = {name: np.asarray(data[name]) for name in REQUIRED_KEYS}
    parent = arrays["parent_face_id"].astype(np.int64, copy=False)
    velocity = _finite(arrays["velocity_ms"], "velocity_ms")
    if velocity.ndim != 2 or velocity.shape[1] != 3:
        raise ParticleEnsembleError("velocity_ms must have shape (N, 3)")
    centers = _finite(arrays["face_centers_local_mm"], "face_centers_local_mm").astype(np.float64, copy=False)
    areas = _finite(arrays["face_area_m2"], "face_area_m2").astype(np.float64, copy=False)
    if centers.ndim != 2 or centers.shape[1] != 3 or areas.ndim != 1 or len(centers) != len(areas):
        raise ParticleEnsembleError("face centers/areas have incompatible shapes")
    n_hits = len(parent)
    for name in ("Vp_ms", "alpha_deg", "diameter_m", "parcel_mass_kg"):
        if len(arrays[name]) != n_hits:
            raise ParticleEnsembleError(f"{name} length differs from parent_face_id")
        _finite(arrays[name], name)
    if np.any(parent < 0) or np.any(parent >= len(areas)):
        raise ParticleEnsembleError(f"parent_face_id is outside the face domain: {source}")
    return {
        "path": source,
        "sha256": sha256_file(source),
        "index": int(index),
        "arrays": arrays,
        "parent": parent,
        "centers": centers,
        "areas": areas,
        "n_hits": n_hits,
        "domain_fingerprint": _domain_fingerprint(centers, areas),
        "mass_total_kg": float(np.sum(arrays["parcel_mass_kg"], dtype=np.float64)),
    }


def merge_particle_hits(
    hit_paths: Sequence[str | Path],
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    paths = [Path(path).resolve() for path in hit_paths]
    if len(paths) < 2:
        raise ParticleEnsembleError("at least two independent hit artifacts are required")
    if len(set(paths)) != len(paths):
        raise ParticleEnsembleError("hit artifact paths must be unique")
    sources = [_load_hits(path, index) for index, path in enumerate(paths)]
    reference = sources[0]
    for source in sources[1:]:
        if source["centers"].shape != reference["centers"].shape or source["areas"].shape != reference["areas"].shape:
            raise ParticleEnsembleError("seed face-domain shapes differ")
        if not np.array_equal(source["centers"], reference["centers"]) or not np.array_equal(source["areas"], reference["areas"]):
            raise ParticleEnsembleError("seed face centers or face areas differ; merge refused")
    count = len(sources)
    merged: dict[str, np.ndarray] = {
        "parent_face_id": np.concatenate([source["parent"] for source in sources]),
        "velocity_ms": np.concatenate([source["arrays"]["velocity_ms"] for source in sources]),
        "Vp_ms": np.concatenate([source["arrays"]["Vp_ms"] for source in sources]),
        "alpha_deg": np.concatenate([source["arrays"]["alpha_deg"] for source in sources]),
        "diameter_m": np.concatenate([source["arrays"]["diameter_m"] for source in sources]),
        "parcel_mass_kg": np.concatenate([source["arrays"]["parcel_mass_kg"] / count for source in sources]),
        "face_centers_local_mm": reference["centers"],
        "face_area_m2": reference["areas"],
        "seed_index": np.concatenate([np.full(source["n_hits"], source["index"], dtype=np.int16) for source in sources]),
    }
    source_total = float(sum(source["mass_total_kg"] for source in sources) / count)
    merged_total = float(np.sum(merged["parcel_mass_kg"], dtype=np.float64))
    if not np.isclose(source_total, merged_total, rtol=1.0e-12, atol=1.0e-18):
        raise ParticleEnsembleError("mass-conservation check failed")
    report: dict[str, Any] = {
        "version": VERSION,
        "status": "provisional_particle_ensemble",
        "official_accuracy_gate": False,
        "official_reference_gate": False,
        "fem_gate": "blocked",
        "claim_scope": "mass-conservative seed ensemble input for exploratory erosion priors",
        "merge_contract": {
            "n_sources": count,
            "mass_rule": "divide every source parcel_mass_kg by n_sources before concatenation",
            "face_domain_rule": "exact equality of face centers and areas is required",
            "seed_index_field": "seed_index",
            "domain_fingerprint": reference["domain_fingerprint"],
        },
        "sources": [
            {
                "index": source["index"],
                "path": str(source["path"]),
                "sha256": source["sha256"],
                "n_hits": source["n_hits"],
                "mass_total_kg": source["mass_total_kg"],
                "domain_fingerprint": source["domain_fingerprint"],
            }
            for source in sources
        ],
        "merged": {
            "n_hits": int(len(merged["parent_face_id"])),
            "n_faces": int(len(reference["areas"])),
            "mass_total_kg": merged_total,
            "expected_mean_source_mass_kg": source_total,
            "mass_conservation_passed": True,
        },
        "limitations": [
            "This ensemble reduces Monte Carlo seed sensitivity but does not validate the common particle driver.",
            "The merged artifact represents the mean exposure across sources, not five times the exposure.",
            "No baseline/worn scan or official spatial erosion truth is included.",
        ],
    }
    report["report_sha256"] = canonical_sha256(report)
    if output_path is not None:
        target = Path(output_path).resolve()
        safe_json_dump(report, target)
        safe_npz_dump(merged, target.with_suffix(".npz"), metadata={"report_sha256": report["report_sha256"], "version": VERSION})
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and merge independent Step30 particle-hit artifacts")
    parser.add_argument("--hits", action="append", required=True, help="one particle_hits.npz; repeat at least twice")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = merge_particle_hits(args.hits, output_path=args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
