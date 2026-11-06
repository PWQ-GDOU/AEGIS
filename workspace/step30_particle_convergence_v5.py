"""Discovery/confirmation evaluator for v5 coarse particle screening maps.

The evaluator separates method development from confirmation.  Existing
immutable v3/v4 particle runs may identify a candidate topology-only scale, but
the same runs can never be described as independent confirmation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from step30_particle_convergence_v4 import (
    js_divergence,
    mass_hotspot_iou,
    weighted_spearman,
)
from step30_particle_seed_convergence import _load_pair as _load_v3_immutable_pair
from step30_particle_spatial_estimator_v5 import (
    AGGREGATION_CONTRACT,
    ParticleSpatialEstimatorV5Error,
    estimate_immutable_particle_run_v5,
    save_estimate_v5,
)
from step30_validation_common import canonical_sha256, safe_json_dump, sha256_file


VERSION = "step30-particle-convergence-v5"
MIN_SEEDS = 5
CONFIRMATION_N_PARCELS = 20_000
GATES = {
    "weighted_spearman_median_min": 0.90,
    "weighted_spearman_p10_min": 0.80,
    "mass_hotspot_iou_median_min": 0.80,
    "spatial_js_divergence_max": 0.05,
}


class ParticleConvergenceV5Error(ValueError):
    """Raised for invalid v5 immutable-run comparisons."""


def _load_run(
    run_directory: str | Path,
    topology_path: str | Path,
    *,
    max_parent_faces_per_aggregate: int,
    topology_aggregate_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    directory = Path(run_directory).resolve()
    hits = directory / "particle_hits.npz"
    summary_path = directory / "summary.json"
    if not hits.is_file() or not summary_path.is_file():
        raise ParticleConvergenceV5Error(
            f"run must contain immutable particle_hits.npz and summary.json: {directory}"
        )
    estimate = estimate_immutable_particle_run_v5(
        hits,
        topology_path,
        max_parent_faces_per_aggregate=max_parent_faces_per_aggregate,
        topology_aggregate_context=topology_aggregate_context,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    parameters = summary.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ParticleConvergenceV5Error("run summary has no parameter object")
    n_parcels = int(parameters.get("n_parcels", -1))
    seed = int(parameters.get("seed", -1))
    if n_parcels <= 0 or seed < 0:
        raise ParticleConvergenceV5Error("run has invalid n_parcels or seed")
    # Reuse the legacy immutable-pair loader only as a strict provenance
    # validator.  It binds code/input/config identity but does not change any
    # v3 statistical conclusion or make its old spatial partition a v5 input.
    legacy_identity = _load_v3_immutable_pair(hits, summary_path)["identity"]
    estimate.update(
        {
            "run_directory": str(directory),
            "n_parcels": n_parcels,
            "seed": seed,
            "legacy_comparability_identity": legacy_identity,
        }
    )
    return estimate


def _identity(estimate: Mapping[str, Any]) -> dict[str, Any]:
    bindings = estimate["metadata"]["input_bindings"]
    return {
        "artifact_set_id": bindings["artifact_set_id"],
        "mesh_signature": bindings["mesh_signature"],
        "topology_sha256": bindings["topology_sha256"],
        "aggregation_contract_sha256": estimate["metadata"][
            "aggregation_contract_sha256"
        ],
        "partition_sha256": estimate["metadata"]["partition"]["partition_sha256"],
        "strict_immutable_run_identity": estimate["legacy_comparability_identity"],
    }


def _confirmation_identity_compatibility(
    discovery_identity: Mapping[str, Any],
    confirmation_identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Accept only documented non-physical provenance changes.

    The remote-only runner adds ``parameters.execution_target`` to immutable
    run descriptors. That field records where the computation ran and does not
    alter the physical model. The v2 AABB optimization changed both the
    particle-tracking and topology-collision implementation hashes; those two
    hashes are jointly allowed only as a registered performance-only change.
    """
    if discovery_identity == confirmation_identity:
        return {"status": "exact_match", "changed_fields": []}
    discovery = dict(discovery_identity)
    confirmation = dict(confirmation_identity)
    discovery_strict = dict(discovery.pop("strict_immutable_run_identity", {}))
    confirmation_strict = dict(confirmation.pop("strict_immutable_run_identity", {}))
    if discovery != confirmation:
        return None
    changed_fields: list[str] = []
    discovery_parameters = dict(discovery_strict.get("parameters", {}))
    confirmation_parameters = dict(confirmation_strict.get("parameters", {}))
    discovery_execution_target = discovery_parameters.pop("execution_target", None)
    confirmation_execution_target = confirmation_parameters.pop("execution_target", None)
    if discovery_execution_target != confirmation_execution_target:
        if discovery_execution_target is not None and confirmation_execution_target is not None:
            return None
        changed_fields.append("strict_immutable_run_identity.parameters.execution_target")
    if "parameters" in discovery_strict or "parameters" in confirmation_strict:
        discovery_strict["parameters"] = discovery_parameters
        confirmation_strict["parameters"] = confirmation_parameters
    discovery_impl = dict(discovery_strict.pop("implementation_hashes", {}))
    confirmation_impl = dict(confirmation_strict.pop("implementation_hashes", {}))
    if discovery_strict != confirmation_strict:
        return None
    implementation_changes = sorted(
        key
        for key in set(discovery_impl) | set(confirmation_impl)
        if discovery_impl.get(key) != confirmation_impl.get(key)
    )
    if implementation_changes not in (
        ["particle_tracking"],
        ["cfd_topology", "particle_tracking"],
    ):
        return None
    changed_fields.extend(
        f"strict_immutable_run_identity.implementation_hashes.{key}"
        for key in implementation_changes
    )
    return {
        "status": (
            "performance_only_tracker_hash_change"
            if implementation_changes == ["particle_tracking"]
            else "performance_only_collision_and_tracker_hash_change"
        ),
        "changed_fields": changed_fields,
        "interpretation": (
            "Discovery predates the registered AABB performance optimization and "
            "the confirmation descriptor adds execution-target provenance; CFD "
            "field inputs, topology, partition, physical inputs, parameters and "
            "solver semantics match."
        ),
    }


def _metrics(
    left: np.ndarray, right: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    return {
        "area_weighted_spearman": float(weighted_spearman(left, right, weights)),
        "mass_hotspot_iou": float(mass_hotspot_iou(left, right)),
        "js_divergence": float(js_divergence(left, right)),
    }


def _aggregate_pairwise(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    pairwise: list[dict[str, float]] = []
    for index, left in enumerate(rows):
        left_field = np.asarray(
            left["field"]["aggregate_target_mass_kg"], dtype=np.float64
        )
        aggregate_area = np.asarray(
            left["field"]["aggregate_area_m2"], dtype=np.float64
        )
        for right in rows[index + 1 :]:
            right_field = np.asarray(
                right["field"]["aggregate_target_mass_kg"], dtype=np.float64
            )
            pairwise.append(_metrics(left_field, right_field, aggregate_area))
    return pairwise


def _raw_pairwise(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    pairwise: list[dict[str, float]] = []
    for index, left in enumerate(rows):
        left_field = np.asarray(
            left["field"]["raw_face_target_mass_kg"], dtype=np.float64
        )
        face_area = np.asarray(left["topology"]["face_area_m2"], dtype=np.float64)
        for right in rows[index + 1 :]:
            right_field = np.asarray(
                right["field"]["raw_face_target_mass_kg"], dtype=np.float64
            )
            pairwise.append(_metrics(left_field, right_field, face_area))
    return pairwise


def _ensemble_block_rows(
    rows: Sequence[Mapping[str, Any]], block_size: int
) -> list[dict[str, Any]]:
    """Average disjoint immutable runs into independent ensemble blocks.

    For block b with k runs, the field is the arithmetic mean
    (\bar m_b = k^{-1}\sum_{j=1}^{k} m_{b,j}\).  The source runs remain
    immutable and their hashes are retained in the block provenance.
    """
    if block_size < 1:
        raise ParticleConvergenceV5Error("ensemble block size must be positive")
    if block_size == 1:
        return [dict(row) for row in rows]
    if len(rows) < MIN_SEEDS * block_size or len(rows) % block_size:
        raise ParticleConvergenceV5Error(
            "ensemble block confirmation requires at least five complete blocks"
        )
    result: list[dict[str, Any]] = []
    for block_index in range(0, len(rows), block_size):
        block = list(rows[block_index : block_index + block_size])
        representative = dict(block[0])
        field = dict(representative["field"])
        # Formula: m_bar[b] = (1/k) * sum_j m[b,j].
        for field_name in (
            "raw_face_target_mass_kg",
            "aggregate_target_mass_kg",
            "aggregate_local_support_mass_kg",
        ):
            values = [np.asarray(item["field"][field_name], dtype=np.float64) for item in block]
            field[field_name] = np.mean(np.stack(values, axis=0), axis=0)
        field["total_input_target_mass_kg"] = float(
            np.mean(
                [
                    float(item["field"]["total_input_target_mass_kg"])
                    for item in block
                ]
            )
        )
        representative["field"] = field
        representative["seed"] = f"block_{block_index // block_size + 1:02d}"
        representative["ensemble_block_seeds"] = [int(item["seed"]) for item in block]
        representative["ensemble_block_particle_hits_sha256"] = [
            str(item["metadata"]["input_bindings"]["particle_hits_sha256"])
            for item in block
        ]
        result.append(representative)
    return result


def _metric_summary(pairwise: Sequence[Mapping[str, float]], n_runs: int) -> dict[str, Any]:
    spearman = np.asarray(
        [row["area_weighted_spearman"] for row in pairwise], dtype=np.float64
    )
    iou = np.asarray([row["mass_hotspot_iou"] for row in pairwise], dtype=np.float64)
    js = np.asarray([row["js_divergence"] for row in pairwise], dtype=np.float64)
    complete = n_runs >= MIN_SEEDS
    gates = {
        "minimum_independent_seeds": complete,
        "area_weighted_spearman_median": complete
        and bool(len(spearman))
        and float(np.median(spearman)) >= GATES["weighted_spearman_median_min"],
        "area_weighted_spearman_p10": complete
        and bool(len(spearman))
        and float(np.percentile(spearman, 10)) >= GATES["weighted_spearman_p10_min"],
        "mass_hotspot_iou_median": complete
        and bool(len(iou))
        and float(np.median(iou)) >= GATES["mass_hotspot_iou_median_min"],
        "spatial_js_divergence_max": complete
        and bool(len(js))
        and float(np.max(js)) <= GATES["spatial_js_divergence_max"],
    }
    return {
        "pairwise": list(pairwise),
        "area_weighted_spearman_median": (
            None if not len(spearman) else float(np.median(spearman))
        ),
        "area_weighted_spearman_p10": (
            None if not len(spearman) else float(np.percentile(spearman, 10))
        ),
        "mass_hotspot_iou_median": (
            None if not len(iou) else float(np.median(iou))
        ),
        "spatial_js_divergence_max": None if not len(js) else float(np.max(js)),
        "gates": gates,
        "passed": all(gates.values()),
    }


def _scale_mass_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    values = np.asarray(
        [item["field"]["total_input_target_mass_kg"] for item in rows],
        dtype=np.float64,
    )
    mean = float(values.mean())
    sample_sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "n_runs": int(len(values)),
        "mean_target_mass_kg": mean,
        "sample_sd_target_mass_kg": sample_sd,
        "relative_sample_sd": sample_sd / max(abs(mean), 1.0e-30),
        "relative_standard_error": (
            sample_sd / math.sqrt(len(values)) / max(abs(mean), 1.0e-30)
        ),
    }


def _load_discovery_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path).resolve()
    if not report_path.is_file():
        raise ParticleConvergenceV5Error(
            f"discovery report does not exist: {report_path}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise ParticleConvergenceV5Error("discovery report must be a JSON object")
    stored_hash = report.get("report_sha256")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if (
        report.get("version") != VERSION
        or report.get("analysis_role") != "discovery"
        or report.get("status") != "exploratory_candidate_requires_confirmation"
        or stored_hash != canonical_sha256(unsigned)
    ):
        raise ParticleConvergenceV5Error(
            "discovery report is not a hash-valid v5 candidate record"
        )
    return dict(report)


def _validate_discovery_evaluator_provenance(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Require discovery and confirmation statistics to use identical code."""
    recorded = report.get("evaluation_provenance")
    if not isinstance(recorded, Mapping):
        raise ParticleConvergenceV5Error(
            "discovery report has no evaluation provenance"
        )
    estimator_path = Path(__file__).resolve().with_name(
        "step30_particle_spatial_estimator_v5.py"
    )
    current = {
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "estimator_sha256": sha256_file(estimator_path),
    }
    mismatches = {
        key: {"recorded": recorded.get(key), "current": value}
        for key, value in current.items()
        if recorded.get(key) != value
    }
    if mismatches:
        raise ParticleConvergenceV5Error(
            "discovery evaluator provenance differs from current code: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {"status": "exact_match", **current}


def _discovery_input_hashes(report: Mapping[str, Any]) -> set[str]:
    values = report.get("artifact_inputs")
    if not isinstance(values, Sequence):
        raise ParticleConvergenceV5Error("discovery report has no artifact input list")
    hashes = {
        str(row["particle_hits_sha256"])
        for row in values
        if isinstance(row, Mapping) and isinstance(row.get("particle_hits_sha256"), str)
    }
    if not hashes:
        raise ParticleConvergenceV5Error("discovery report has no particle input hashes")
    return hashes


def evaluate_runs_v5(
    run_directories: Sequence[str | Path],
    topology_path: str | Path,
    *,
    analysis_role: str = "discovery",
    discovery_report_path: str | Path | None = None,
    max_parent_faces_per_aggregate: int = int(
        AGGREGATION_CONTRACT["default_max_parent_faces_per_aggregate"]
    ),
    fields_directory: str | Path | None = None,
    ensemble_block_size: int = 1,
) -> dict[str, Any]:
    """Evaluate a v5 discovery or independent confirmation dataset."""
    try:
        if analysis_role not in {"discovery", "confirmation"}:
            raise ParticleConvergenceV5Error(
                "analysis_role must be 'discovery' or 'confirmation'"
            )
        if not run_directories:
            raise ParticleConvergenceV5Error("at least one immutable run is required")
        first, *remaining = run_directories
        first_estimate = _load_run(
            first,
            topology_path,
            max_parent_faces_per_aggregate=max_parent_faces_per_aggregate,
        )
        topology_aggregate_context = first_estimate.pop(
            "_runtime_topology_aggregate_context"
        )
        estimates = [first_estimate]
        for directory in remaining:
            estimate = _load_run(
                directory,
                topology_path,
                max_parent_faces_per_aggregate=max_parent_faces_per_aggregate,
                topology_aggregate_context=topology_aggregate_context,
            )
            estimate.pop("_runtime_topology_aggregate_context")
            estimates.append(estimate)
        reference = _identity(estimates[0])
        if any(_identity(item) != reference for item in estimates[1:]):
            raise ParticleConvergenceV5Error(
                "runs do not share topology and aggregation-partition identity"
            )
        keys = {(item["n_parcels"], item["seed"]) for item in estimates}
        if len(keys) != len(estimates):
            raise ParticleConvergenceV5Error(
                "duplicate n_parcels/seed immutable-run identity"
            )
        if fields_directory is not None:
            root = Path(fields_directory).resolve()
            root.mkdir(parents=True, exist_ok=True)
            for item in estimates:
                output = root / (
                    "particle_surface_aggregate_v5_"
                    f"{item['metadata']['input_bindings']['particle_run_id']}.npz"
                )
                item["saved_field"] = save_estimate_v5(item, output)

        discovery_report: dict[str, Any] | None = None
        identity_compatibility: dict[str, Any] | None = None
        provenance_compatibility: dict[str, Any] | None = None
        if analysis_role == "confirmation":
            if discovery_report_path is None:
                raise ParticleConvergenceV5Error(
                    "confirmation requires a hash-valid discovery report"
                )
            discovery_report = _load_discovery_report(discovery_report_path)
            provenance_compatibility = _validate_discovery_evaluator_provenance(
                discovery_report
            )
            identity_compatibility = _confirmation_identity_compatibility(
                discovery_report.get("identity", {}), reference
            )
            if identity_compatibility is None:
                raise ParticleConvergenceV5Error(
                    "confirmation topology/partition identity differs from discovery"
                )
            discovery_hashes = _discovery_input_hashes(discovery_report)
            current_hashes = {
                str(item["metadata"]["input_bindings"]["particle_hits_sha256"])
                for item in estimates
            }
            overlap = sorted(discovery_hashes & current_hashes)
            if overlap:
                raise ParticleConvergenceV5Error(
                    "confirmation reuses discovery particle runs: " + ", ".join(overlap)
                )

        groups: dict[int, list[dict[str, Any]]] = {}
        for item in estimates:
            groups.setdefault(item["n_parcels"], []).append(item)
        scale_reports: dict[str, Any] = {}
        for scale, rows in sorted(groups.items()):
            rows = sorted(rows, key=lambda item: item["seed"])
            comparison_rows = _ensemble_block_rows(rows, ensemble_block_size)
            aggregate = _metric_summary(
                _aggregate_pairwise(comparison_rows), len(comparison_rows)
            )
            raw = _metric_summary(_raw_pairwise(comparison_rows), len(comparison_rows))
            confidence_area = np.asarray(
                [
                    np.sum(
                        np.asarray(row["field"]["aggregate_area_m2"], dtype=np.float64)
                        * np.asarray(
                            row["field"]["aggregate_confidence_mask"], dtype=bool
                        )
                    )
                    / np.sum(
                        np.asarray(row["field"]["aggregate_area_m2"], dtype=np.float64)
                    )
                    for row in rows
                ],
                dtype=np.float64,
            )
            scale_reports[str(scale)] = {
                "seeds": [item["seed"] for item in rows],
                "n_runs": len(rows),
                "comparison_unit": (
                    "immutable_run" if ensemble_block_size == 1 else "ensemble_block"
                ),
                "ensemble_block_size": int(ensemble_block_size),
                "comparison_blocks": [
                    item.get("ensemble_block_seeds", [item["seed"]])
                    for item in comparison_rows
                ],
                "estimators": {
                    "raw_parent_face": raw,
                    "topology_aggregate": aggregate,
                },
                "mass_estimate": _scale_mass_summary(rows),
                "aggregate_support_diagnostics": {
                    "median_confident_area_fraction": float(
                        np.median(confidence_area)
                    ),
                    "minimum_confident_area_fraction": float(
                        np.min(confidence_area)
                    ),
                    "per_run_confident_area_fraction": confidence_area.tolist(),
                },
            }

        expected_scale = (
            int(
                discovery_report["confirmation_contract"][
                    "required_n_parcels"
                ]
            )
            if discovery_report is not None
            else CONFIRMATION_N_PARCELS
        )
        decision_scale = expected_scale
        decision = scale_reports.get(str(decision_scale))
        if decision is None:
            raise ParticleConvergenceV5Error(
                f"v5 requires n_parcels={decision_scale} for its decision scale"
            )
        aggregate_decision = decision["estimators"]["topology_aggregate"]
        discovery_candidate = bool(aggregate_decision["passed"])

        scale_diagnostics: list[dict[str, Any]] = []
        scales = sorted(groups)
        for left_scale, right_scale in zip(scales, scales[1:]):
            left_by_seed = {item["seed"]: item for item in groups[left_scale]}
            right_by_seed = {item["seed"]: item for item in groups[right_scale]}
            common = sorted(set(left_by_seed) & set(right_by_seed))
            if not common:
                continue
            pairwise = []
            relative_mass_change = []
            for seed in common:
                left = left_by_seed[seed]
                right = right_by_seed[seed]
                pairwise.append(
                    _metrics(
                        np.asarray(
                            left["field"]["aggregate_target_mass_kg"], dtype=np.float64
                        ),
                        np.asarray(
                            right["field"]["aggregate_target_mass_kg"], dtype=np.float64
                        ),
                        np.asarray(
                            left["field"]["aggregate_area_m2"], dtype=np.float64
                        ),
                    )
                )
                left_mass = float(left["field"]["total_input_target_mass_kg"])
                right_mass = float(right["field"]["total_input_target_mass_kg"])
                relative_mass_change.append(
                    abs(left_mass - right_mass)
                    / max(abs(left_mass), abs(right_mass), 1.0e-30)
                )
            scale_diagnostics.append(
                {
                    "scales": [left_scale, right_scale],
                    "shared_seed_count": len(common),
                    "aggregate_pairwise": pairwise,
                    "max_paired_relative_mass_change": float(
                        max(relative_mass_change)
                    ),
                    "interpretation": (
                        "diagnostic only: mass-stratified runs at different parcel "
                        "counts are not nested common-random-number samples, so "
                        "this is not an acceptance test for bias"
                    ),
                }
            )

        if analysis_role == "discovery":
            status = (
                "exploratory_candidate_requires_confirmation"
                if discovery_candidate
                else "failed"
            )
            action = (
                "freeze_v5_contract_then_run_nonoverlapping_confirmation_seeds"
                if discovery_candidate
                else "do_not_claim_convergence_increase_independent_particle_support"
            )
        else:
            expected_confirmation = decision_scale == CONFIRMATION_N_PARCELS
            status = (
                "numerically_stable_coarse_risk_map"
                if expected_confirmation and aggregate_decision["passed"]
                else "failed"
            )
            action = (
                "confirmed_coarse_numerical_screening_map"
                if status == "numerically_stable_coarse_risk_map"
                else "collect_or_repair_confirmation_runs"
            )

        result = {
            "version": VERSION,
            "analysis_role": analysis_role,
            "status": status,
            "numerical_stability_status": status,
            "spatial_accuracy_status": "blocked",
            "official_accuracy_gate": False,
            "official_reference_gate": False,
            "fem_gate": "blocked",
            "action": action,
            "aggregation_contract": AGGREGATION_CONTRACT,
            "aggregation_contract_sha256": canonical_sha256(AGGREGATION_CONTRACT),
            "evaluation_provenance": {
                "evaluator_file": Path(__file__).name,
                "evaluator_sha256": sha256_file(Path(__file__).resolve()),
                "estimator_file": "step30_particle_spatial_estimator_v5.py",
                "estimator_sha256": sha256_file(
                    Path(__file__).resolve().with_name(
                        "step30_particle_spatial_estimator_v5.py"
                    )
                ),
            },
            "gates": GATES,
            "identity": reference,
            "discovery_identity_compatibility": identity_compatibility,
            "discovery_evaluation_provenance_compatibility": provenance_compatibility,
            "decision_scale": decision_scale,
            "scales": scale_reports,
            "scale_diagnostics": scale_diagnostics,
            "confirmation_contract": {
                "required_n_parcels": CONFIRMATION_N_PARCELS,
                "minimum_nonoverlapping_runs": MIN_SEEDS,
                "run_overlap_rule": (
                    "particle_hits SHA-256 must not appear in the hash-valid "
                    "discovery report"
                ),
                "acceptance_interpretation": (
                    "numerically stable coarse risk map only; never a physical "
                    "erosion-accuracy or official competition-evidence claim"
                ),
            },
            "discovery_report_path": (
                None if discovery_report_path is None else str(Path(discovery_report_path).resolve())
            ),
            "discovery_report_sha256": (
                None
                if discovery_report is None
                else str(discovery_report["report_sha256"])
            ),
            "historical_records": {
                "v3": {
                    "path": str(
                        Path(__file__).resolve().parent
                        / "output"
                        / "step30_particle_convergence_5k_10k_20k_mass_stratified_v3.json"
                    ),
                    "status": "failed",
                    "preserved_as": "historical_spatial_instability_record",
                },
                "v4": {
                    "path": str(
                        Path(__file__).resolve().parent
                        / "output"
                        / "step30_particle_convergence_5k_10k_20k_surface_kernel_v4r2.json"
                    ),
                    "status": "failed",
                    "preserved_as": "historical_short-range_kernel_record",
                },
            },
            "artifact_inputs": [
                {
                    "run_directory": item["run_directory"],
                    "n_parcels": item["n_parcels"],
                    "seed": item["seed"],
                    "particle_run_id": item["metadata"]["input_bindings"][
                        "particle_run_id"
                    ],
                    "particle_hits_sha256": item["metadata"]["input_bindings"][
                        "particle_hits_sha256"
                    ],
                    "field_contract_sha256": item["metadata"][
                        "field_contract_sha256"
                    ],
                    "field_path": (
                        item["saved_field"]["field_path"]
                        if "saved_field" in item
                        else None
                    ),
                    "field_sha256": (
                        item["saved_field"]["field_sha256"]
                        if "saved_field" in item
                        else None
                    ),
                }
                for item in estimates
            ],
            "limitations": [
                "The discovery data identify a method candidate but cannot confirm it.",
                "Aggregate stability does not validate the parent-face field.",
                "No worn scan, baseline map, or independent spatial erosion truth is present.",
                "No v5 result may open official accuracy, reference, or FEM gates.",
            ],
        }
        result["report_sha256"] = canonical_sha256(result)
        return result
    except (
        OSError,
        ValueError,
        KeyError,
        ParticleSpatialEstimatorV5Error,
        ParticleConvergenceV5Error,
    ) as exc:
        result = {
            "version": VERSION,
            "analysis_role": analysis_role,
            "status": "blocked",
            "numerical_stability_status": "blocked",
            "spatial_accuracy_status": "blocked",
            "reason": str(exc),
            "official_accuracy_gate": False,
            "official_reference_gate": False,
            "fem_gate": "blocked",
        }
        result["report_sha256"] = canonical_sha256(result)
        return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate v5 topology-connected particle screening maps"
    )
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--analysis-role", choices=["discovery", "confirmation"], default="discovery"
    )
    parser.add_argument("--discovery-report", type=Path)
    parser.add_argument("--fields-directory", type=Path)
    parser.add_argument(
        "--max-parent-faces-per-aggregate",
        type=int,
        default=int(
            AGGREGATION_CONTRACT["default_max_parent_faces_per_aggregate"]
        ),
    )
    parser.add_argument(
        "--ensemble-block-size",
        type=int,
        default=1,
        help="average this many disjoint immutable runs per confirmation block",
    )
    args = parser.parse_args(argv)
    report = evaluate_runs_v5(
        args.run,
        args.topology,
        analysis_role=args.analysis_role,
        discovery_report_path=args.discovery_report,
        max_parent_faces_per_aggregate=args.max_parent_faces_per_aggregate,
        fields_directory=args.fields_directory,
        ensemble_block_size=args.ensemble_block_size,
    )
    safe_json_dump(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "numerically_stable_coarse_risk_map" else 3


if __name__ == "__main__":
    raise SystemExit(main())
