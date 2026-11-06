"""Auditable progress package for the eight-step Step30 optimization track.

This module only aggregates already-produced, hash-bound reports.  It never
fills missing particle-count/seed cells with synthetic data and never changes
the official evidence gates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from step30_validation_common import canonical_sha256, safe_json_dump, sha256_file


VERSION = "step30-eight-step-optimization-v1"
PARTICLE_COUNTS = (5000, 10000, 20000, 50000)
SEED_COUNTS = (5, 10, 20)


class EightStepOptimizationError(ValueError):
    """Raised when an optimization progress package is not auditable."""


def _load_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    target = Path(path).resolve()
    if not target.is_file():
        raise EightStepOptimizationError(f"{label} does not exist: {target}")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EightStepOptimizationError(f"invalid {label}: {target}") from exc
    if not isinstance(value, Mapping):
        raise EightStepOptimizationError(f"{label} must be a JSON object")
    return target, dict(value)


def _load_face_domain(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    if not target.is_file():
        raise EightStepOptimizationError(f"face domain does not exist: {target}")
    try:
        import numpy as np

        with np.load(target, allow_pickle=False) as data:
            required = {"face_centers_local_mm", "face_area_m2"}
            missing = sorted(required - set(data.files))
            if missing:
                raise EightStepOptimizationError(f"face domain lacks {missing}")
            centers = np.asarray(data["face_centers_local_mm"])
            areas = np.asarray(data["face_area_m2"])
    except EightStepOptimizationError:
        raise
    except Exception as exc:
        raise EightStepOptimizationError(f"invalid face domain: {target}") from exc
    if centers.ndim != 2 or centers.shape[1] != 3 or areas.ndim != 1 or len(centers) != len(areas):
        raise EightStepOptimizationError("face domain centers and areas have incompatible shapes")
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "n_faces": int(len(areas)),
    }


def _convergence_matrix(report: Mapping[str, Any]) -> dict[str, Any]:
    scales = report.get("scales")
    if not isinstance(scales, Mapping):
        raise EightStepOptimizationError("convergence report lacks scales")
    cells: list[dict[str, Any]] = []
    for particle_count in PARTICLE_COUNTS:
        item = scales.get(str(particle_count))
        for seed_count in SEED_COUNTS:
            cell: dict[str, Any] = {
                "particle_count": particle_count,
                "seed_count": seed_count,
                "status": "blocked_missing_runs",
                "source_scale_present": item is not None,
            }
            if isinstance(item, Mapping):
                n_runs = int(item.get("n_runs", 0) or 0)
                cell["available_runs"] = n_runs
                if n_runs >= seed_count:
                    estimator = item.get("estimators", {}).get("topology_aggregate", {})
                    cell.update(
                        {
                            "status": "available",
                            "estimator_status": estimator.get("passed", False),
                            "area_weighted_spearman_median": estimator.get("area_weighted_spearman_median"),
                            "mass_hotspot_iou_median": estimator.get("mass_hotspot_iou_median"),
                            "spatial_js_divergence_max": estimator.get("spatial_js_divergence_max"),
                        }
                    )
                else:
                    cell["status"] = "blocked_insufficient_seeds"
            cells.append(cell)
    confirmed = [
        cell
        for cell in cells
        if cell["status"] == "available" and cell.get("estimator_status") is True
    ]
    selected = max(confirmed, key=lambda cell: (cell["particle_count"], cell["seed_count"])) if confirmed else None
    return {
        "source_version": report.get("version"),
        "cells": cells,
        "confirmed_cells": confirmed,
        "selected_candidate": selected,
        "status": "numerically_stable_candidate" if selected else "blocked",
        "claim_boundary": "numerical stability only; not physical spatial accuracy",
    }


def _score_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    required = ("conservative_score", "nominal_score", "optimistic_score")
    if any(key not in result for key in required):
        raise EightStepOptimizationError("participant result lacks score fields")
    return {
        "version": result.get("version"),
        "status": result.get("status"),
        "conservative_score": float(result["conservative_score"]),
        "nominal_score": float(result["nominal_score"]),
        "optimistic_score": float(result["optimistic_score"]),
        "score_interval": result.get("competition_self_benchmark_score_interval"),
        "competition_self_benchmark_status": result.get("competition_self_benchmark_status"),
        "official_accuracy_gate": result.get("official_accuracy_gate"),
        "official_reference_gate": result.get("official_reference_gate"),
        "fem_gate": result.get("fem_gate"),
    }


def build_eight_step_package(
    convergence_report_path: str | Path,
    participant_result_path: str | Path,
    face_domain_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    convergence_path, convergence = _load_json(convergence_report_path, "convergence report")
    participant_path, participant = _load_json(participant_result_path, "participant result")
    domain = _load_face_domain(face_domain_path)
    mapping = participant.get("face_region_mapping")
    material = participant.get("material_amplitude_calibration")
    baseline = participant.get("baseline_worn_reference")
    combination = participant.get("model_combination")
    for name, value in (("face_region_mapping", mapping), ("material_amplitude_calibration", material), ("baseline_worn_reference", baseline), ("model_combination", combination)):
        if not isinstance(value, Mapping):
            raise EightStepOptimizationError(f"participant result lacks {name}")
    package: dict[str, Any] = {
        "version": VERSION,
        "status": "conditional_progress_package",
        "inputs": {
            "convergence_report": {"path": str(convergence_path), "sha256": sha256_file(convergence_path)},
            "participant_result": {"path": str(participant_path), "sha256": sha256_file(participant_path)},
            "face_domain": domain,
        },
        "step_status": {
            "particle_convergence_matrix": _convergence_matrix(convergence),
            "active_face_threshold_audit": {
                "status": "available",
                "selected_estimator": "topology_aggregate_v5",
                "source": "convergence matrix selected candidate",
                "physical_accuracy": "blocked_without_independent_measured_truth",
            },
            "dec_mapping_hypotheses": {
                "status": "participant_built",
                "primary": mapping.get("version"),
                "face_count": domain["n_faces"],
                "claim_boundary": mapping.get("claim_boundary"),
                "additional_CAD_semantic_hypotheses": "blocked_without official surface tags/normals",
            },
            "material_amplitude_calibration": {
                "status": material.get("status"),
                "material": material.get("material"),
                "measured_pilot": material.get("independent_measured_pilot", False),
                "claim_boundary": "synthetic/prior-based until physical pilot files are ingested",
            },
            "independent_baseline_worn_depth_field": {
                "status": baseline.get("status"),
                "measured_truth": baseline.get("measured_truth", False),
                "profile_accuracy_status": "blocked_without_registered_measured_profile",
            },
            "preregistered_model_combination": {
                "status": combination.get("status"),
                "weights": combination.get("weights"),
                "weight_sum": combination.get("weight_sum"),
                "posthoc_weight_change_allowed": combination.get("posthoc_weight_change_allowed"),
            },
            "score_recalculation": _score_summary(participant),
            "remote_verification": {
                "status": "completed_elsewhere",
                "local_numerical_execution": False,
                "execution_target": "dcu_cpu",
            },
        },
        "unresolved_blockers": [
            "50,000-parcel convergence cells are missing",
            "active-face seed CV remains high in the available five-seed ensemble",
            "no measured ZG0Cr13Ni4Mo pilot files are ingested",
            "no registered independent baseline/worn 3-D profile is ingested",
            "official_accuracy_gate, official_reference_gate, and fem_gate remain closed",
        ],
    }
    package["report_sha256"] = canonical_sha256(package)
    if output_path is not None:
        safe_json_dump(package, Path(output_path).resolve())
    return package


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Step30 eight-step optimization progress package")
    parser.add_argument("--convergence-report", type=Path, required=True)
    parser.add_argument("--participant-result", type=Path, required=True)
    parser.add_argument("--face-domain", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_eight_step_package(args.convergence_report, args.participant_result, args.face_domain, output_path=args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
