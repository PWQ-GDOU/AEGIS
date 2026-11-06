"""Statistical convergence gates for immutable strict-v2 particle runs.

The evaluator compares runs on the same accepted URANS/UMean artifact. It does
not validate erosion accuracy against physical truth and never opens an official
accuracy or organizer-reference gate.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from step30_validation_common import canonical_sha256, jsonable, safe_json_dump, safe_json_load, sha256_file

VERSION = "step30-particle-seed-convergence-v3"
MIN_SEEDS = 3
PREREGISTERED_MAX_SCALE = 20_000
SPATIAL_PARTITION = {
    "version": "step30-particle-spatial-macro-bins-v1",
    "axes": ["x", "y", "z"],
    "bins_per_axis": [8, 8, 8],
    "bounds_source": "frozen_topology_face_centres",
}
GATES = {
    "pooled_erosion_ess_min": 200.0,
    "max_parcel_fraction_max": 0.05,
    "ci_relative_half_width_max": 0.10,
    "spearman_median_min": 0.90,
    "spearman_p10_min": 0.80,
    "hotspot_iou_median_min": 0.80,
    "scale_relative_change_max": 0.05,
    "spatial_js_divergence_max": 0.05,
}
_ALLOWED_CHANGES = {"n_parcels", "n_injected", "seed", "output_tag"}


class ParticleConvergenceError(ValueError):
    """Raised when run artifacts cannot be compared legitimately."""


def _finite(array: Any, name: str, *, ndim: int = 1) -> np.ndarray:
    result = np.asarray(array, dtype=np.float64)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ParticleConvergenceError(f"{name} must be finite with ndim={ndim}")
    return result


def _macro_spatial_field(
    face_centres_mm: np.ndarray,
    face_values: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    if face_centres_mm.ndim != 2 or face_centres_mm.shape != (len(face_values), 3):
        raise ParticleConvergenceError("face centres differ from face erosion field")
    bounds_min = np.min(face_centres_mm, axis=0)
    bounds_max = np.max(face_centres_mm, axis=0)
    span = bounds_max - bounds_min
    if np.any(span <= 0.0):
        raise ParticleConvergenceError("topology face-centre bounds are degenerate")
    bins = np.asarray(SPATIAL_PARTITION["bins_per_axis"], dtype=np.int64)
    scaled = (face_centres_mm - bounds_min) / span
    indices = np.floor(np.clip(scaled, 0.0, 1.0 - np.finfo(float).eps) * bins).astype(np.int64)
    flat = (indices[:, 0] * bins[1] + indices[:, 1]) * bins[2] + indices[:, 2]
    field = np.bincount(flat, weights=face_values, minlength=int(np.prod(bins))).astype(np.float64)
    contract = {
        **SPATIAL_PARTITION,
        "bounds_min_mm": bounds_min.tolist(),
        "bounds_max_mm": bounds_max.tolist(),
        "contract_sha256": canonical_sha256(
            {
                **SPATIAL_PARTITION,
                "bounds_min_mm": bounds_min.tolist(),
                "bounds_max_mm": bounds_max.tolist(),
            }
        ),
    }
    return field, contract


def _validate_artifact_set_manifest(
    run_directory: Path,
    identity: Mapping[str, Any],
    cfd: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> tuple[Path, str]:
    artifact_root = run_directory.parent.parent
    manifest_path = artifact_root / "artifact_set_manifest.json"
    if not manifest_path.is_file():
        raise ParticleConvergenceError("artifact-set manifest is missing")
    descriptor = safe_json_load(manifest_path)
    if not isinstance(descriptor, Mapping):
        raise ParticleConvergenceError("artifact-set manifest must be a JSON object")
    descriptor = dict(descriptor)
    stored_artifact_id = descriptor.pop("artifact_set_id", None)
    topology = descriptor.get("topology")
    descriptor_cfd = descriptor.get("cfd")
    if (
        descriptor.get("version") != "step30-artifact-set-descriptor-v2"
        or not isinstance(stored_artifact_id, str)
        or canonical_sha256(descriptor) != stored_artifact_id
        or artifact_root.name != stored_artifact_id
        or identity.get("artifact_set_id") != stored_artifact_id
        or not isinstance(topology, Mapping)
        or not isinstance(descriptor_cfd, Mapping)
        or str(descriptor_cfd.get("field_time")) != str(cfd.get("field_time"))
        or descriptor_cfd.get("solver_semantics") != cfd.get("solver_semantics")
        or topology.get("mesh_signature") != geometry.get("mesh_signature")
        or int(topology.get("n_parent_faces", -1))
        != int(geometry.get("n_parent_faces", -2))
    ):
        raise ParticleConvergenceError("artifact-set manifest identity is invalid")
    return manifest_path, sha256_file(manifest_path)


def _validate_immutable_manifest(
    npz_path: Path,
    summary_path: Path,
    identity: Mapping[str, Any],
    cfd: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, str, Path, str]:
    npz_path = npz_path.resolve()
    summary_path = summary_path.resolve()
    if (
        npz_path.parent != summary_path.parent
        or npz_path.name != "particle_hits.npz"
        or summary_path.name != "summary.json"
    ):
        raise ParticleConvergenceError(
            "immutable particle NPZ and summary must use the standard run layout"
        )
    artifact_manifest_path, artifact_manifest_sha256 = _validate_artifact_set_manifest(
        npz_path.parent, identity, cfd, geometry
    )
    manifest_path = npz_path.parent / "particle_run_manifest.json"
    if not manifest_path.is_file():
        raise ParticleConvergenceError("immutable particle run manifest is missing")
    manifest = safe_json_load(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ParticleConvergenceError("particle run manifest must be a JSON object")
    manifest = dict(manifest)
    outputs = manifest.pop("outputs", None)
    stored_run_id = manifest.pop("particle_run_id", None)
    if (
        manifest.get("version") != "step30-particle-run-descriptor-v2"
        or not isinstance(stored_run_id, str)
        or canonical_sha256(manifest) != stored_run_id
        or identity.get("particle_run_id") != stored_run_id
        or identity.get("artifact_set_id") != manifest.get("artifact_set_id")
        or manifest.get("collision_backend") != "cfd_topology"
        or npz_path.parent.name != stored_run_id
        or npz_path.parent.parent.name != "strict_v2"
    ):
        raise ParticleConvergenceError("immutable particle run identity is invalid")
    expected = {
        "particle_hits": (npz_path, "particle_hits.npz"),
        "summary": (summary_path, "summary.json"),
        "validation": (npz_path.parent / "validation.png", "validation.png"),
    }
    if not isinstance(outputs, Mapping) or set(outputs) != set(expected):
        raise ParticleConvergenceError("particle run manifest output set is invalid")
    for name, (path, filename) in expected.items():
        entry = outputs.get(name)
        if (
            not isinstance(entry, Mapping)
            or entry.get("file") != filename
            or not path.is_file()
            or sha256_file(path) != entry.get("sha256")
        ):
            raise ParticleConvergenceError(
                f"particle run manifest output hash mismatch: {name}"
            )
    manifest["particle_run_id"] = stored_run_id
    return (
        manifest,
        manifest_path,
        sha256_file(manifest_path),
        artifact_manifest_path,
        artifact_manifest_sha256,
    )


def _load_pair(npz_path: Path, summary_path: Path) -> dict[str, Any]:
    if not npz_path.is_file() or not summary_path.is_file():
        raise ParticleConvergenceError("particle NPZ and summary JSON must exist")
    summary = safe_json_load(summary_path)
    if not isinstance(summary, Mapping):
        raise ParticleConvergenceError("particle summary must be a JSON object")
    with np.load(npz_path, allow_pickle=False) as handle:
        required = {
            "parcel_id", "impact_index", "parent_face_id", "parcel_mass_kg",
            "ER_mansouri", "Vp_ms", "alpha_deg", "face_target_mass_kg",
            "dhdt_face_mm_h", "face_centers_local_mm", "artifact_set_id",
            "mesh_signature", "field_time", "collision_backend", "collision_policy",
        }
        missing = sorted(required - set(handle.files))
        if missing:
            raise ParticleConvergenceError(f"particle NPZ is missing arrays: {missing}")
        arrays = {name: np.asarray(handle[name]).copy() for name in required}
    params = summary.get("parameters")
    tracking = summary.get("tracking")
    cfd = summary.get("cfd")
    geometry = summary.get("geometry")
    identity = summary.get("artifact_identity")
    semantics = cfd.get("solver_semantics") if isinstance(cfd, Mapping) else None
    if (
        summary.get("version") != "step30-cfd-driven-v2"
        or not isinstance(identity, Mapping)
        or not isinstance(params, Mapping)
        or not isinstance(tracking, Mapping)
        or not isinstance(geometry, Mapping)
        or not isinstance(semantics, Mapping)
        or semantics.get("application") != "pimpleFoam"
        or semantics.get("mode") != "urans"
        or semantics.get("field_statistic") != "time_mean"
        or semantics.get("tracking_velocity_object") != "UMean"
        or not isinstance(semantics.get("averaging_window"), Mapping)
        or params.get("collision_policy") != "strict_v2"
        or params.get("collision_backend") != "cfd_topology"
        or params.get("cfd_variant") != "native_si"
        or params.get("n_parcels") != params.get("n_injected")
        or tracking.get("fallback_hits") != 0
        or tracking.get("collision_event_limit_count", 0) != 0
        or tracking.get("physical_strict_hit_ratio") != 1.0
    ):
        raise ParticleConvergenceError("run is not accepted immutable strict-v2 URANS/UMean evidence")
    (
        manifest,
        manifest_path,
        manifest_sha256,
        artifact_manifest_path,
        artifact_manifest_sha256,
    ) = _validate_immutable_manifest(npz_path, summary_path, identity, cfd, geometry)
    manifest_config = manifest.get("config")
    if not isinstance(manifest_config, Mapping):
        raise ParticleConvergenceError("particle run manifest config is missing")
    shared_config_fields = (
        "cfd_variant",
        "collision_backend",
        "collision_policy",
        "n_parcels",
        "n_injected",
        "max_impacts",
        "seed",
        "dt_s",
        "particle_cfl",
        "cfd_velocity_interpolation",
        "particle_integrator",
        "strict_max_substep_s",
        "min_substep_s",
        "max_collision_events_per_step",
        "include_gravity",
        "e_normal",
        "e_tangent",
        "mass_flow_mode",
        "psd_sampling",
    )
    if any(
        jsonable(manifest_config.get(field)) != jsonable(params.get(field))
        for field in shared_config_fields
        if field in manifest_config or field in params
    ) or str(manifest_config.get("cfd_field_time")) != str(cfd.get("field_time")):
        raise ParticleConvergenceError("particle manifest and summary config differ")
    if str(arrays["collision_policy"].item()) != "strict_v2" or str(arrays["collision_backend"].item()) != "cfd_topology":
        raise ParticleConvergenceError("NPZ collision contract mismatch")
    if str(arrays["artifact_set_id"].item()) != str(identity.get("artifact_set_id")):
        raise ParticleConvergenceError("artifact_set_id mismatch")
    if str(arrays["mesh_signature"].item()) != str(geometry.get("mesh_signature")):
        raise ParticleConvergenceError("mesh_signature mismatch")
    if str(arrays["field_time"].item()) != str(cfd.get("field_time")):
        raise ParticleConvergenceError("field_time mismatch")
    parcel_id = np.asarray(arrays["parcel_id"], dtype=np.int64)
    impact_index = np.asarray(arrays["impact_index"], dtype=np.int64)
    parent_face = np.asarray(arrays["parent_face_id"], dtype=np.int64)
    parcel_mass = _finite(arrays["parcel_mass_kg"], "parcel_mass_kg")
    er = _finite(arrays["ER_mansouri"], "ER_mansouri")
    speed = _finite(arrays["Vp_ms"], "Vp_ms")
    angle = _finite(arrays["alpha_deg"], "alpha_deg")
    if not (len(parcel_id) == len(impact_index) == len(parent_face) == len(parcel_mass) == len(er) == len(speed) == len(angle)):
        raise ParticleConvergenceError("impact arrays have inconsistent lengths")
    if len(parcel_id) == 0:
        raise ParticleConvergenceError("run has zero physical impacts")
    if np.any(parcel_id < 0) or np.any(parent_face < 0) or np.any(parcel_mass < 0) or np.any(er < 0):
        raise ParticleConvergenceError("impact arrays contain invalid values")
    first = impact_index == 0
    if not np.any(first):
        raise ParticleConvergenceError("run has zero first impacts")
    contribution = parcel_mass * er
    parcel_contribution: dict[int, float] = defaultdict(float)
    for pid, value in zip(parcel_id, contribution):
        parcel_contribution[int(pid)] += float(value)
    contributions = np.asarray(list(parcel_contribution.values()), dtype=np.float64)
    contribution_sum = float(contributions.sum())
    if contribution_sum <= 0.0:
        raise ParticleConvergenceError("run has zero erosion contribution")
    weights = parcel_mass
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        raise ParticleConvergenceError("run has zero impact mass")
    face_target = _finite(arrays["face_target_mass_kg"], "face_target_mass_kg")
    dhdt = _finite(arrays["dhdt_face_mm_h"], "dhdt_face_mm_h")
    face_centres = _finite(
        arrays["face_centers_local_mm"], "face_centers_local_mm", ndim=2
    )
    if face_target.shape != dhdt.shape or len(face_target) != int(geometry.get("n_parent_faces", -1)):
        raise ParticleConvergenceError("face erosion arrays differ from topology cardinality")
    macro_field, spatial_contract = _macro_spatial_field(face_centres, face_target)
    n_parcels = int(params["n_parcels"])
    first_mass = float(parcel_mass[first].sum())
    total_injection = float(tracking.get("total_injection_mass_kg", 0.0))
    if total_injection <= 0.0:
        raise ParticleConvergenceError("total injection mass is invalid")
    sorted_contribution = np.sort(contributions)[::-1]
    metric = {
        "n_parcels": n_parcels,
        "seed": int(params["seed"]),
        "n_impacts": int(len(parcel_id)),
        "first_hit_parcels": int(np.sum(first)),
        "first_hit_rate": float(np.sum(first) / n_parcels),
        "first_hit_mass_fraction": float(first_mass / total_injection),
        "target_mass_kg": float(face_target.sum()),
        "Vp_mass_weighted": float(np.sum(speed * weights) / weight_sum),
        "angle_mass_weighted": float(np.sum(angle * weights) / weight_sum),
        "hit_faces": int(np.sum(face_target > 0.0)),
        "direct_coverage": float(np.mean(face_target > 0.0)),
        "erosion_ess": float(contribution_sum**2 / np.sum(contributions**2)),
        "max_parcel_contribution_kg": float(sorted_contribution[0]),
        "max_parcel_fraction": float(sorted_contribution[0] / contribution_sum),
        "top5_fraction": float(sorted_contribution[:5].sum() / contribution_sum),
        "top10_fraction": float(sorted_contribution[:10].sum() / contribution_sum),
    }
    comparable_params = {key: jsonable(value) for key, value in params.items() if key not in _ALLOWED_CHANGES}
    comparable_identity = {
        "artifact_set_id": identity.get("artifact_set_id"),
        "mesh_signature": geometry.get("mesh_signature"),
        "field_time": str(cfd.get("field_time")),
        "solver_semantics": semantics,
        "implementation_hashes": manifest.get("implementation_hashes"),
        "input_hashes": manifest.get("input_hashes"),
        "parameters": comparable_params,
    }
    return {
        "npz_path": str(npz_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "artifact_manifest_path": str(artifact_manifest_path.resolve()),
        "npz_sha256": sha256_file(npz_path),
        "summary_sha256": sha256_file(summary_path),
        "manifest_sha256": manifest_sha256,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "metrics": metric,
        "identity": comparable_identity,
        "face_target_mass_kg": face_target,
        "macro_spatial_target_mass_kg": macro_field,
        "spatial_partition": spatial_contract,
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.bincount(inverse, weights=ranks)
        ranks = sums[inverse] / counts[inverse]
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    active = (left > 0.0) | (right > 0.0)
    if int(active.sum()) < 2:
        return 1.0 if np.array_equal(left, right) else 0.0
    a, b = _rank(left[active]), _rank(right[active])
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _top_iou(left: np.ndarray, right: np.ndarray, percentile: float = 90.0) -> float:
    active = (left > 0.0) | (right > 0.0)
    ids = np.flatnonzero(active)
    if len(ids) == 0:
        return 1.0
    n_top = max(1, int(math.ceil((100.0 - percentile) / 100.0 * len(ids))))
    left_ids = set(ids[np.argsort(left[ids], kind="mergesort")[-n_top:]].tolist())
    right_ids = set(ids[np.argsort(right[ids], kind="mergesort")[-n_top:]].tolist())
    return float(len(left_ids & right_ids) / len(left_ids | right_ids))


def _js(left: np.ndarray, right: np.ndarray) -> float:
    left_sum, right_sum = float(left.sum()), float(right.sum())
    if left_sum <= 0.0 or right_sum <= 0.0:
        raise ParticleConvergenceError("JS divergence requires positive spatial mass")
    p, q = left / left_sum, right / right_sum
    m = 0.5 * (p + q)
    def kl(x: np.ndarray) -> float:
        mask = x > 0.0
        return float(np.sum(x[mask] * np.log2(x[mask] / m[mask])))
    return 0.5 * (kl(p) + kl(q))


def _ci(values: Sequence[float]) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(data))
    result: dict[str, Any] = {
        "method": "empirical_central_95pct_seed_interval",
        "interpretation": "descriptive seed spread; not a confidence interval for the population mean",
        "sample_count": int(len(data)),
        "mean": mean,
        "low": None,
        "high": None,
        "relative_half_width": None,
    }
    if len(data) < MIN_SEEDS:
        return result
    low, high = np.percentile(data, [2.5, 97.5])
    result.update(
        {
            "low": float(low),
            "high": float(high),
            "relative_half_width": float(
                (high - low) / 2.0 / max(abs(mean), 1.0e-30)
            ),
        }
    )
    return result


def evaluate_runs(pairs: Sequence[tuple[Path, Path]]) -> dict[str, Any]:
    try:
        if not pairs:
            raise ParticleConvergenceError("at least one particle run is required")
        loaded = [_load_pair(npz, summary) for npz, summary in pairs]
        reference = loaded[0]["identity"]
        spatial_partition = loaded[0]["spatial_partition"]
        if any(item["identity"] != reference for item in loaded[1:]):
            raise ParticleConvergenceError("particle runs differ beyond n_parcels/seed/output identity")
        if any(item["spatial_partition"] != spatial_partition for item in loaded[1:]):
            raise ParticleConvergenceError("particle runs use different spatial macro partitions")
        key_set = {(item["metrics"]["n_parcels"], item["metrics"]["seed"]) for item in loaded}
        if len(key_set) != len(loaded):
            raise ParticleConvergenceError("duplicate particle scale/seed run")
        groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in loaded:
            groups[item["metrics"]["n_parcels"]].append(item)
        scale_reports: dict[str, Any] = {}
        all_failures: list[str] = []
        all_seed_complete = True
        for scale, runs in sorted(groups.items()):
            runs.sort(key=lambda item: item["metrics"]["seed"])
            pooled_contributions = []
            for run in runs:
                # Per-run ESS combines additively only through the equivalent
                # first two contribution moments reconstructed from sum and ESS.
                total = run["metrics"]["target_mass_kg"]
                ess = run["metrics"]["erosion_ess"]
                pooled_contributions.append((total, total**2 / ess))
            pooled_sum = sum(item[0] for item in pooled_contributions)
            pooled_sum_sq = sum(item[1] for item in pooled_contributions)
            pooled_ess = float(pooled_sum**2 / pooled_sum_sq) if pooled_sum_sq > 0.0 else 0.0
            pooled_max_parcel_fraction = max(
                run["metrics"]["max_parcel_contribution_kg"] for run in runs
            ) / max(pooled_sum, 1.0e-30)
            pair_stats = [
                {
                    "seeds": [left["metrics"]["seed"], right["metrics"]["seed"]],
                    "spearman": _spearman(
                        left["macro_spatial_target_mass_kg"],
                        right["macro_spatial_target_mass_kg"],
                    ),
                    "hotspot_iou": _top_iou(
                        left["macro_spatial_target_mass_kg"],
                        right["macro_spatial_target_mass_kg"],
                    ),
                }
                for left, right in itertools.combinations(runs, 2)
            ]
            spearman = np.asarray([row["spearman"] for row in pair_stats], dtype=float)
            iou = np.asarray([row["hotspot_iou"] for row in pair_stats], dtype=float)
            ci = {
                name: _ci([run["metrics"][name] for run in runs])
                for name in ("first_hit_mass_fraction", "target_mass_kg", "Vp_mass_weighted", "angle_mass_weighted")
            }
            complete = len(runs) >= MIN_SEEDS
            all_seed_complete &= complete
            gates = {
                "minimum_seeds": complete,
                "pooled_erosion_ess": pooled_ess >= GATES["pooled_erosion_ess_min"],
                "max_parcel_fraction": pooled_max_parcel_fraction <= GATES["max_parcel_fraction_max"],
                "ci_relative_half_width": complete and all(
                    item["relative_half_width"] is not None
                    and item["relative_half_width"] <= GATES["ci_relative_half_width_max"]
                    for item in ci.values()
                ),
                "spearman_median": complete and bool(len(spearman)) and float(np.median(spearman)) >= GATES["spearman_median_min"],
                "spearman_p10": complete and bool(len(spearman)) and float(np.percentile(spearman, 10)) >= GATES["spearman_p10_min"],
                "hotspot_iou_median": complete and bool(len(iou)) and float(np.median(iou)) >= GATES["hotspot_iou_median_min"],
            }
            failures = [name for name, passed in gates.items() if not passed]
            all_failures.extend(f"n={scale}:{name}" for name in failures if name != "minimum_seeds")
            scale_reports[str(scale)] = {
                "n_runs": len(runs),
                "seeds": [run["metrics"]["seed"] for run in runs],
                "runs": [run["metrics"] for run in runs],
                "pooled_erosion_ess": pooled_ess,
                "pooled_max_parcel_fraction": pooled_max_parcel_fraction,
                "pairwise": pair_stats,
                "spearman_median": float(np.median(spearman)) if len(spearman) else None,
                "spearman_p10": float(np.percentile(spearman, 10)) if len(spearman) else None,
                "hotspot_iou_median": float(np.median(iou)) if len(iou) else None,
                "confidence_intervals": ci,
                "gates": gates,
                "passed": all(gates.values()),
            }
        scales = sorted(groups)
        scale_comparisons = []
        final_scale_comparison = None
        if len(scales) >= 2:
            final_pair = [scales[-2], scales[-1]]
            for left_scale, right_scale in zip(scales, scales[1:]):
                left_by_seed = {run["metrics"]["seed"]: run for run in groups[left_scale]}
                right_by_seed = {run["metrics"]["seed"]: run for run in groups[right_scale]}
                common = sorted(set(left_by_seed) & set(right_by_seed))
                if len(common) < MIN_SEEDS:
                    continue
                changes, divergences = [], []
                for seed in common:
                    left, right = left_by_seed[seed], right_by_seed[seed]
                    lval, rval = left["metrics"]["target_mass_kg"], right["metrics"]["target_mass_kg"]
                    changes.append(abs(lval - rval) / max(abs(lval), abs(rval), 1.0e-30))
                    divergences.append(
                        _js(
                            left["macro_spatial_target_mass_kg"],
                            right["macro_spatial_target_mass_kg"],
                        )
                    )
                row = {
                    "scales": [left_scale, right_scale],
                    "seeds": common,
                    "target_mass_relative_change_max": float(max(changes)),
                    "spatial_js_divergence_max": float(max(divergences)),
                }
                row["gates"] = {
                    "target_mass_relative_change": row["target_mass_relative_change_max"] <= GATES["scale_relative_change_max"],
                    "spatial_js_divergence": row["spatial_js_divergence_max"] <= GATES["spatial_js_divergence_max"],
                }
                row["passed"] = all(row["gates"].values())
                scale_comparisons.append(row)
                if row["scales"] == final_pair:
                    final_scale_comparison = row
                if not row["passed"]:
                    all_failures.extend(f"scale={left_scale}->{right_scale}:{name}" for name, passed in row["gates"].items() if not passed)

        decision_scale = scales[-1]
        decision_failures = [
            f"n={decision_scale}:{name}"
            for name, passed in scale_reports[str(decision_scale)]["gates"].items()
            if name != "minimum_seeds" and not passed
        ]
        if final_scale_comparison is not None:
            decision_failures.extend(
                f"scale={final_scale_comparison['scales'][0]}->{decision_scale}:{name}"
                for name, passed in final_scale_comparison["gates"].items()
                if not passed
            )
        decision_seed_complete = scale_reports[str(decision_scale)]["gates"][
            "minimum_seeds"
        ]
        if not decision_seed_complete:
            status, action = "blocked", "collect_more_seeds"
        elif len(scales) >= 2 and final_scale_comparison is None:
            status, action = "blocked", "collect_matching_seeds_across_scales"
        elif decision_failures:
            heavy = any(
                "erosion_ess" in item or "max_parcel_fraction" in item
                for item in decision_failures
            )
            sampling = reference["parameters"].get("psd_sampling")
            if heavy and sampling != "mass_stratified_v1":
                action = "stratified_psd_required"
            elif decision_scale >= PREREGISTERED_MAX_SCALE:
                action = "stop_at_preregistered_max_scale_investigate_spatial_instability"
            else:
                action = "increase_scale_or_fix_spatial_instability"
            status = "failed"
        else:
            status = "passed"
            action = "scale_converged" if len(scales) >= 2 else "seed_converged_collect_next_scale"
        evaluator_path = Path(__file__).resolve()
        result = {
            "version": VERSION,
            "status": status,
            "evaluation_status": status,
            "action": action,
            "gates": dict(GATES),
            "evaluation_provenance": {
                "evaluator_file": evaluator_path.name,
                "evaluator_sha256": sha256_file(evaluator_path),
                "contract_sha256": canonical_sha256(
                    {
                        "version": VERSION,
                        "minimum_seeds": MIN_SEEDS,
                        "preregistered_max_scale": PREREGISTERED_MAX_SCALE,
                        "gates": GATES,
                        "spatial_partition": SPATIAL_PARTITION,
                    }
                ),
            },
            "statistical_methods": {
                "seed_interval": "empirical central 2.5/97.5 percentiles across seed estimates; descriptive, not a confidence interval for the population mean",
                "spatial_rank_and_hotspot": "pairwise comparisons on the frozen 8x8x8 topology macro partition",
                "scale_change": "maximum paired-seed relative change between adjacent scales",
            },
            "identity": reference,
            "spatial_partition": spatial_partition,
            "scales": scale_reports,
            "scale_comparisons": scale_comparisons,
            "decision_scale": decision_scale,
            "preregistered_max_scale": PREREGISTERED_MAX_SCALE,
            "decision_failures": decision_failures,
            "failures": all_failures,
            "n_runs": len(loaded),
            "artifact_inputs": [
                {
                    "npz_path": item["npz_path"],
                    "summary_path": item["summary_path"],
                    "manifest_path": item["manifest_path"],
                    "artifact_manifest_path": item["artifact_manifest_path"],
                    "npz_sha256": item["npz_sha256"],
                    "summary_sha256": item["summary_sha256"],
                    "manifest_sha256": item["manifest_sha256"],
                    "artifact_manifest_sha256": item[
                        "artifact_manifest_sha256"
                    ],
                }
                for item in loaded
            ],
            "official_accuracy_gate": False,
            "official_reference_gate": False,
            "provenance_note": "particle statistical convergence only; not physical erosion accuracy",
        }
        result["report_sha256"] = canonical_sha256(result)
        return result
    except (ParticleConvergenceError, OSError, ValueError, TypeError, KeyError) as exc:
        result = {
            "version": VERSION,
            "status": "blocked",
            "evaluation_status": "blocked",
            "action": "repair_evidence_contract",
            "reason": str(exc),
            "official_accuracy_gate": False,
            "official_reference_gate": False,
        }
        result["report_sha256"] = canonical_sha256(result)
        return result


def _run_dir(path: Path) -> tuple[Path, Path]:
    return path / "particle_hits.npz", path / "summary.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate immutable particle seed/scale convergence")
    parser.add_argument("--run", action="append", type=Path, default=[])
    parser.add_argument("--pair", action="append", nargs=2, metavar=("NPZ", "SUMMARY"), default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    pairs = [_run_dir(path) for path in args.run]
    pairs.extend((Path(item[0]), Path(item[1])) for item in args.pair)
    report = evaluate_runs(pairs)
    if args.output is not None:
        safe_json_dump(report, args.output)
    print(json.dumps(jsonable(report), ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
