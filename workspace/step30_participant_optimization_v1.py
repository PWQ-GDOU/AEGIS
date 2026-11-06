"""Participant-built Step30 optimization pipeline.

The pipeline joins six competition-track improvements without changing the
official evidence gates:

* deterministic participant DEC face-region mapping from geometry only;
* active-face seed convergence using the existing five-seed prior ensemble;
* frozen same-material amplitude prior/calibration;
* an independent participant-built baseline/worn depth field;
* a pre-registered fixed-weight model combination; and
* conservative/nominal/optimistic scoring against that independent field.

When no laboratory files are supplied, the material and worn field layers are
explicitly synthetic/participant-built.  They are eligible for the
competition self-benchmark track but are never described as measured truth.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from step30_competition_self_benchmark_v1 import (
    MODEL_SCORE_WEIGHTS,
    _load_literature,
    _load_prior,
    _normalize,
    _pair_score,
    _material_factor,
)
from step30_physical_accuracy_bridge_v1 import MODEL_NAMES
from step30_validation_common import canonical_sha256, safe_json_dump, safe_npz_dump, sha256_file


VERSION = "step30-participant-optimization-v1"
MAPPING_VERSION = "step30-participant-dec-face-region-mapping-v1"
BASELINE_WORN_VERSION = "step30-participant-baseline-worn-v1"
MATERIAL_CALIBRATION_VERSION = "step30-participant-material-amplitude-calibration-v1"
MODEL_COMBINATION_VERSION = "step30-preregistered-model-combination-v1"
REGIONS = (
    "splitter_like",
    "cutout_like",
    "brim_like",
    "bucket_base_or_side_like",
    "impact_zone_like",
    "residual_other",
)
MODEL_WEIGHTS = {model: 0.25 for model in MODEL_NAMES}
LITERATURE_REGION_WEIGHTS = {
    "splitter_like": 0.375,
    "cutout_like": 0.300,
    "brim_like": 0.050,
    "bucket_base_or_side_like": 0.075,
    "impact_zone_like": 0.200,
    "residual_other": 0.000,
}
MATERIAL_PRIOR = {
    "density_kg_m3": {"low": 7700.0, "nominal": 7800.0, "high": 7900.0},
    "hardness_gpa": {"low": 3.0, "nominal": 4.0, "high": 5.0},
    "concentration_scale": {"low": 0.5, "nominal": 1.0, "high": 2.0},
}
# Frozen before looking at the DEC score.  It is a participant assumption,
# not a measured material coefficient.
SPECIFIC_WEAR_YIELD_MM3_PER_KG = {
    "low": 0.08,
    "nominal": 0.16,
    "high": 0.32,
}
# A frozen geometry-only hotspot core.  It keeps the independently built
# reference localized without reading any particle/model field.
REGION_CORE_QUANTILE = 0.99


class ParticipantOptimizationError(ValueError):
    """Raised when the participant optimization contract is invalid."""


def _finite_centers(value: Any) -> np.ndarray:
    centers = np.asarray(value, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
        raise ParticipantOptimizationError("face centers must have shape (n, 3) and be finite")
    return centers


def _finite_areas(value: Any, count: int) -> np.ndarray:
    areas = np.asarray(value, dtype=np.float64)
    if areas.ndim != 1 or len(areas) != count or not np.isfinite(areas).all() or np.any(areas <= 0.0):
        raise ParticipantOptimizationError("face areas must be positive and match face centers")
    return areas


def _load_domain(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    if not target.is_file():
        raise ParticipantOptimizationError(f"face-domain artifact does not exist: {target}")
    with np.load(target, allow_pickle=False) as data:
        required = {"face_centers_local_mm", "face_area_m2"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ParticipantOptimizationError(f"face-domain artifact lacks {missing}")
        centers = _finite_centers(data["face_centers_local_mm"])
        areas = _finite_areas(data["face_area_m2"], len(centers))
    return {
        "path": target,
        "sha256": sha256_file(target),
        "centers": centers,
        "areas": areas,
    }


def _principal_coordinates(centers: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    centered = centers - np.mean(centers, axis=0)
    covariance = centered.T @ centered / max(len(centers), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    axes = eigenvectors[:, order]
    projected = centered @ axes
    spans = np.ptp(projected, axis=0)
    spans = np.maximum(spans, 1.0e-12)
    normalized = (projected - np.min(projected, axis=0)) / spans
    return normalized, {
        "axis_order": ["longitudinal", "lateral", "normal"],
        "eigenvalues": eigenvalues.tolist(),
        "axis_matrix_sha256": canonical_sha256(axes.tolist()),
        "normalization": "principal-coordinate min-max over official face centers",
    }


def build_face_region_mapping(
    centers: np.ndarray,
    areas: np.ndarray,
    literature_weights: Mapping[str, float] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Build a deterministic geometry-only participant face-region mapping."""
    weights = dict(LITERATURE_REGION_WEIGHTS if literature_weights is None else literature_weights)
    if set(weights) != set(REGIONS):
        raise ParticipantOptimizationError("literature region weights must cover the fixed region contract")
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in weights.values()):
        raise ParticipantOptimizationError("literature region weights must be finite and non-negative")
    if sum(float(value) for value in weights.values()) <= 0.0:
        raise ParticipantOptimizationError("literature region weights have no positive total")
    weights = {region: float(weights[region]) for region in REGIONS}
    normalized, axis_metadata = _principal_coordinates(centers)
    u, v, w = normalized[:, 0], normalized[:, 1], normalized[:, 2]
    labels = np.full(len(centers), "residual_other", dtype="U32")
    splitter = (np.abs(v - 0.5) <= 0.16) & (u >= 0.35) & (w >= 0.45)
    cutout = ((u <= 0.12) | (u >= 0.88)) & ~splitter
    brim = (w >= 0.82) & ~splitter & ~cutout
    base = (w <= 0.18) & ~splitter & ~cutout & ~brim
    impact = ~(splitter | cutout | brim | base)
    labels[splitter] = "splitter_like"
    labels[cutout] = "cutout_like"
    labels[brim] = "brim_like"
    labels[base] = "bucket_base_or_side_like"
    labels[impact] = "impact_zone_like"
    area_total = float(np.sum(areas))
    rows = []
    for region in REGIONS:
        mask = labels == region
        rows.append(
            {
                "region_id": region,
                "face_count": int(np.count_nonzero(mask)),
                "area_m2": float(np.sum(areas[mask])),
                "area_fraction": float(np.sum(areas[mask]) / max(area_total, 1.0e-30)),
                "literature_support_fraction": weights[region] / sum(weights.values()),
            }
        )
    report = {
        "version": MAPPING_VERSION,
        "status": "participant_built",
        "mapping_kind": "participant_DEC_face_region_mapping",
        "official_reference": False,
        "measured_truth": False,
        "regions": rows,
        "rules": {
            "source": "official face centers only",
            "surface_orientation": "deterministic PCA axes; no CAD tag or AI5/WETTED/ROOT semantics",
            "splitter": "abs(lateral-0.5)<=0.16 and longitudinal>=0.35 and normal>=0.45",
            "cutout": "longitudinal edge bands <=0.12 or >=0.88, excluding splitter",
            "brim": "normal>=0.82, excluding splitter/cutout",
            "bucket_base_or_side": "normal<=0.18, excluding higher-priority regions",
            "impact_zone": "remaining non-residual faces",
        },
        "axis_metadata": axis_metadata,
        "claim_boundary": "participant geometry partition for competition self-benchmark; not official surface selection",
    }
    report["mapping_sha256"] = canonical_sha256(report)
    return report, labels


def _region_weights_from_literature(literature: Mapping[str, Any]) -> dict[str, float]:
    """Bind baseline/worn regional weights to the validated literature input."""
    supplied = literature.get("support_fraction")
    if not isinstance(supplied, Mapping):
        raise ParticipantOptimizationError("literature input lacks validated support_fraction")
    weights: dict[str, float] = {}
    for region in REGIONS:
        value = float(supplied.get(region, 0.0))
        if not math.isfinite(value) or value < 0.0:
            raise ParticipantOptimizationError(f"literature support fraction is invalid for {region}")
        weights[region] = value
    total = float(sum(weights.values()))
    if total <= 0.0:
        raise ParticipantOptimizationError("literature support fractions have no positive total")
    return {region: weights[region] / total for region in REGIONS}


def _load_prior_maps(merged_path: str | Path, seed_paths: Sequence[str | Path], domain: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    merged = _load_prior(merged_path, domain_artifact_path=domain["path"])
    seeds = [_load_prior(path, domain_artifact_path=domain["path"]) for path in seed_paths]
    for seed in seeds:
        if not np.array_equal(seed["centers"], merged["centers"]) or not np.array_equal(seed["areas"], merged["areas"]):
            raise ParticipantOptimizationError("seed face domain differs from merged prior")
    seed_stack = np.stack([[seed["maps"][model] for model in MODEL_NAMES] for seed in seeds], axis=0)
    return merged, seeds, seed_stack


def _active_convergence(seed_stack: np.ndarray, areas: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    fields: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {}
    area_total = float(np.sum(areas))
    for index, model in enumerate(MODEL_NAMES):
        values = np.asarray(seed_stack[:, index, :], dtype=np.float64)
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
        cv = std / np.maximum(np.abs(mean), 1.0e-30)
        active = np.any(values > 0.0, axis=0)
        active_cv = cv[active] if np.any(active) else np.zeros(1)
        active_area = float(np.sum(areas[active]))
        fields[f"{model}_seed_median"] = np.median(values, axis=0).astype(np.float32)
        fields[f"{model}_active_cv"] = cv.astype(np.float32)
        summary[model] = {
            "active_face_fraction": float(np.mean(active)),
            "active_area_fraction": float(active_area / max(area_total, 1.0e-30)),
            "active_cv_p50": float(np.quantile(active_cv, 0.50)),
            "active_cv_p95": float(np.quantile(active_cv, 0.95)),
            "active_high_cv_fraction": float(np.mean(active_cv >= 0.5)),
            "interpretation": "seed variability on a common frozen particle/topology contract",
        }
    return {"n_seeds": int(seed_stack.shape[0]), "models": summary, "status": "active_face_convergence_screened"}, fields


def _material_calibration(
    model_fields: Mapping[str, np.ndarray],
    areas: np.ndarray,
    particle_mass_kg: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    target_volume = {label: particle_mass_kg * value for label, value in SPECIFIC_WEAR_YIELD_MM3_PER_KG.items()}
    nominal_target = target_volume["nominal"]
    scales: dict[str, float] = {}
    calibrated: dict[str, np.ndarray] = {}
    rows = []
    for model in MODEL_NAMES:
        field = np.maximum(np.asarray(model_fields[model], dtype=np.float64), 0.0)
        raw_volume = float(np.sum(field * areas) * 1.0e6)
        scale = nominal_target / max(raw_volume, 1.0e-30)
        scales[model] = scale
        calibrated[model] = field * scale
        rows.append({"model": model, "raw_volume_mm3_per_exposure": raw_volume, "nominal_target_volume_mm3": nominal_target, "scale": scale})
    report = {
        "version": MATERIAL_CALIBRATION_VERSION,
        "status": "synthetic_material_reference",
        "material": "ZG0Cr13Ni4Mo",
        "source_layer": "literature_material_prior + ZG0Cr13Ni4Mo_preregistered_protocol",
        "particle_mass_kg": particle_mass_kg,
        "specific_wear_yield_mm3_per_kg": SPECIFIC_WEAR_YIELD_MM3_PER_KG,
        "material_prior": MATERIAL_PRIOR,
        "calibration_rows": rows,
        "fit_rule": "scale each frozen five-seed merged model field to frozen nominal specific-wear target; no DEC score feedback",
        "independent_measured_pilot": False,
        "official_accuracy_gate": False,
        "official_reference_gate": False,
        "fem_gate": "blocked",
    }
    report["calibration_sha256"] = canonical_sha256(report)
    return report, calibrated


def _build_baseline_worn(
    labels: np.ndarray,
    centers: np.ndarray,
    areas: np.ndarray,
    particle_mass_kg: float,
    region_weights: Mapping[str, float],
    literature_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    normalized, axis_metadata = _principal_coordinates(centers)
    u, v, w = normalized[:, 0], normalized[:, 1], normalized[:, 2]
    base_weights = np.asarray([float(region_weights[str(region)]) for region in labels], dtype=np.float64)
    centrality = np.clip(1.0 - np.abs(v - 0.5) * 1.2, 0.3, 1.0)
    flow_factor = np.clip(0.65 + 0.35 * u, 0.65, 1.0)
    shape = np.clip(base_weights * centrality * flow_factor, 0.0, None)
    if np.max(shape) <= 0.0:
        raise ParticipantOptimizationError("participant worn shape has no positive support")
    positive_shape = shape[shape > 0.0]
    support_threshold = float(np.quantile(positive_shape, REGION_CORE_QUANTILE))
    support_mask = shape >= support_threshold
    shape = np.where(support_mask, shape, 0.0)
    if not np.any(shape > 0.0):
        raise ParticipantOptimizationError("participant worn hotspot core has no positive support")
    shape /= float(np.max(shape))
    shape_volume_mm3 = float(np.sum(shape * areas) * 1.0e6)
    target_volume_mm3 = {
        label: particle_mass_kg * value
        for label, value in SPECIFIC_WEAR_YIELD_MM3_PER_KG.items()
    }
    if shape_volume_mm3 <= 0.0:
        raise ParticipantOptimizationError("participant worn shape has no positive area support")
    low_depth = shape * target_volume_mm3["low"] / shape_volume_mm3
    nominal_depth = shape * target_volume_mm3["nominal"] / shape_volume_mm3
    high_depth = shape * target_volume_mm3["high"] / shape_volume_mm3
    fields = {
        "baseline_depth_mm": np.zeros(len(labels), dtype=np.float32),
        "worn_depth_low_mm": low_depth.astype(np.float32),
        "worn_depth_nominal_mm": nominal_depth.astype(np.float32),
        "worn_depth_high_mm": high_depth.astype(np.float32),
        "participant_region_shape": shape.astype(np.float32),
    }
    report = {
        "version": BASELINE_WORN_VERSION,
        "status": "participant_built_synthetic",
        "benchmark_kind": "competition_self_benchmark",
        "baseline": "official face-domain geometry only; zero wear field",
        "worn": "geometry-only participant depth field with literature regional weights",
        "material": "ZG0Cr13Ni4Mo",
        "exposure": {
            "particle_mass_kg": particle_mass_kg,
            "specific_wear_yield_mm3_per_kg": SPECIFIC_WEAR_YIELD_MM3_PER_KG,
            "target_volume_mm3": target_volume_mm3,
            "shape_integrated_volume_at_unit_depth_mm3": shape_volume_mm3,
        },
        "region_weights": dict(region_weights),
        "literature_binding": dict(literature_binding),
        "shape_rule": "literature region weight x lateral centrality x longitudinal flow factor, restricted to fixed 99th-percentile geometry-only hotspot core; depth scale is area-integrated to target material volume",
        "hotspot_core": {
            "quantile": REGION_CORE_QUANTILE,
            "support_threshold": support_threshold,
            "face_fraction": float(np.mean(support_mask)),
            "area_fraction": float(np.sum(areas[support_mask]) / max(np.sum(areas), 1.0e-30)),
            "uses_model_fields": False,
        },
        "axis_metadata": axis_metadata,
        "official_reference": False,
        "measured_truth": False,
        "independence": "does not read any model erosion field",
        "claim_boundary": "participant-built baseline/worn reference; not measured surface truth",
    }
    report["baseline_worn_sha256"] = canonical_sha256(report)
    return report, fields


def _combine_models(calibrated: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], np.ndarray]:
    stack = np.stack([np.asarray(calibrated[model], dtype=np.float64) for model in MODEL_NAMES], axis=0)
    composite = np.sum(stack * np.asarray([MODEL_WEIGHTS[model] for model in MODEL_NAMES])[:, None], axis=0)
    report = {
        "version": MODEL_COMBINATION_VERSION,
        "status": "preregistered",
        "models": list(MODEL_NAMES),
        "weights": MODEL_WEIGHTS,
        "combination": "fixed weighted arithmetic mean of material-calibrated seed-median fields",
        "weight_sum": float(sum(MODEL_WEIGHTS.values())),
        "selected_after_score": False,
        "posthoc_weight_change_allowed": False,
        "official_accuracy_gate": False,
        "fem_gate": "blocked",
    }
    report["combination_sha256"] = canonical_sha256(report)
    return report, composite


def _scale_shape_to_volume(shape: np.ndarray, areas: np.ndarray, target_volume_mm3: float) -> np.ndarray:
    positive = np.maximum(np.asarray(shape, dtype=np.float64), 0.0)
    integrated = float(np.sum(positive * areas) * 1.0e6)
    if integrated <= 0.0:
        raise ParticipantOptimizationError("cross-fitted reference shape has no positive area support")
    return positive * (float(target_volume_mm3) / integrated)


def _cross_fitted_scores(
    merged_maps: Mapping[str, np.ndarray],
    seed_stack: np.ndarray,
    calibrated: Mapping[str, np.ndarray],
    areas: np.ndarray,
    region_prior: np.ndarray,
    particle_mass_kg: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Score held-out models/seeds against references that exclude them."""
    targets = {
        label: particle_mass_kg * value
        for label, value in SPECIFIC_WEAR_YIELD_MM3_PER_KG.items()
    }
    merged_stack = np.stack([np.asarray(merged_maps[model], dtype=np.float64) for model in MODEL_NAMES], axis=0)
    fields: dict[str, np.ndarray] = {}
    model_scores: dict[str, Any] = {}
    seed_scores: dict[str, Any] = {}
    for model_index, model in enumerate(MODEL_NAMES):
        other_model_shape = np.mean(
            np.stack([_normalize(merged_stack[index]) for index in range(len(MODEL_NAMES)) if index != model_index], axis=0),
            axis=0,
        )
        # Model-level reference follows the already validated leave-one-model
        # contract: only other merged model shapes are used.  Seed support is
        # applied only in the separate leave-one-seed-out diagnostics below.
        model_shape = other_model_shape * np.clip(region_prior, 0.0, None)
        reference_nominal = _scale_shape_to_volume(model_shape, areas, targets["nominal"])
        fields[f"crossfit_reference_nominal__{model}"] = reference_nominal.astype(np.float32)
        scenario_scores = {}
        for label in ("low", "nominal", "high"):
            scenario_scores[label] = _pair_score(
                calibrated[model],
                _scale_shape_to_volume(model_shape, areas, targets[label]),
                areas,
            )
        all_scenarios = tuple(scenario_scores.values())
        model_scores[model] = {
            "reference_independence": "leave_one_model_out",
            "nominal": scenario_scores["nominal"],
            "conservative": min(all_scenarios, key=lambda item: item["score"]),
            "optimistic": max(all_scenarios, key=lambda item: item["score"]),
            "score_interval": {"lower": float(min(item["score"] for item in all_scenarios)), "upper": float(max(item["score"] for item in all_scenarios))},
        }
        for seed_index in range(seed_stack.shape[0]):
            candidates = [
                _normalize(seed_stack[index, candidate_model])
                for index in range(seed_stack.shape[0])
                if index != seed_index
                for candidate_model in range(len(MODEL_NAMES))
                if candidate_model != model_index
            ]
            seed_shape = np.mean(np.stack(candidates, axis=0), axis=0) * np.clip(region_prior, 0.0, None)
            target = _scale_shape_to_volume(seed_stack[seed_index, model_index], areas, targets["nominal"])
            seed_scenarios = {}
            for label in ("low", "nominal", "high"):
                seed_reference = _scale_shape_to_volume(seed_shape, areas, targets[label])
                seed_scenarios[label] = _pair_score(target, seed_reference, areas)
            all_seed_scenarios = tuple(seed_scenarios.values())
            fields[f"crossfit_reference_nominal__{model}__seed_{seed_index}"] = _scale_shape_to_volume(seed_shape, areas, targets["nominal"]).astype(np.float32)
            seed_scores.setdefault(model, {})[str(seed_index)] = {
                "reference_independence": "leave_one_model_out_and_leave_one_seed_out",
                "nominal": seed_scenarios["nominal"],
                "conservative": min(all_seed_scenarios, key=lambda item: item["score"]),
                "optimistic": max(all_seed_scenarios, key=lambda item: item["score"]),
                "score_interval": {"lower": float(min(item["score"] for item in all_seed_scenarios)), "upper": float(max(item["score"] for item in all_seed_scenarios))},
            }
    nominal_scores = [model_scores[model]["nominal"]["score"] for model in MODEL_NAMES]
    conservative_scores = [model_scores[model]["conservative"]["score"] for model in MODEL_NAMES]
    optimistic_scores = [model_scores[model]["optimistic"]["score"] for model in MODEL_NAMES]
    aggregate = {
        "status": "cross_fitted_conditional",
        "reference_independence": "leave_one_model_out_and_leave_one_seed_out",
        "model_scores": model_scores,
        "seed_scores": seed_scores,
        "nominal_score": float(np.median(nominal_scores)),
        "conservative_score": float(np.median(conservative_scores)),
        "optimistic_score": float(np.median(optimistic_scores)),
        "score_interval": {"lower": float(min(conservative_scores)), "upper": float(max(optimistic_scores))},
        "aggregation": "median across four held-out model scores; seed scores retained as independent diagnostics",
        "material_targets_mm3": targets,
    }
    return aggregate, fields


def _score_variants(prediction: np.ndarray, low: np.ndarray, nominal: np.ndarray, high: np.ndarray, areas: np.ndarray) -> dict[str, Any]:
    nominal_result = _pair_score(prediction, nominal, areas)
    low_result = _pair_score(prediction, low, areas)
    high_result = _pair_score(prediction, high, areas)
    return {
        "nominal": nominal_result,
        "conservative": min((low_result, high_result), key=lambda result: result["score"]),
        "optimistic": max((low_result, high_result), key=lambda result: result["score"]),
        "score_interval": {"lower": float(min(low_result["score"], high_result["score"])), "upper": float(max(low_result["score"], high_result["score"]))},
    }


def build_optimized_pipeline(
    merged_prior_report_path: str | Path,
    seed_prior_paths: Sequence[str | Path],
    domain_path: str | Path,
    literature_reference_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    domain = _load_domain(domain_path)
    merged, seeds, seed_stack = _load_prior_maps(merged_prior_report_path, seed_prior_paths, domain)
    literature = _load_literature(literature_reference_path)
    region_weights = _region_weights_from_literature(literature)
    mapping, labels = build_face_region_mapping(domain["centers"], domain["areas"], region_weights)
    convergence, convergence_fields = _active_convergence(seed_stack, domain["areas"])
    particle_mass_kg = float(merged["report"].get("input_bindings", {}).get("total_particle_mass_kg", 0.0) or 0.0)
    if particle_mass_kg <= 0.0:
        particle_mass_kg = 0.33607395
    # The merged prior is the frozen five-seed ensemble product.  Per-seed
    # medians remain in the convergence fields, but are not used as the final
    # prediction because their sparse union is a deliberately diagnostic view.
    ensemble_fields = {model: np.asarray(merged["maps"][model], dtype=np.float64) for model in MODEL_NAMES}
    material_report, calibrated = _material_calibration(ensemble_fields, domain["areas"], particle_mass_kg)
    literature_binding = {
        "path": str(literature["path"]),
        "sha256": literature["sha256"],
        "support_fraction_source": "validated literature/reference.json",
    }
    baseline_worn_report, baseline_fields = _build_baseline_worn(
        labels,
        domain["centers"],
        domain["areas"],
        particle_mass_kg,
        region_weights,
        literature_binding,
    )
    combination_report, composite = _combine_models(calibrated)
    region_prior = np.asarray([region_weights[str(region)] for region in labels], dtype=np.float64)
    crossfit_scores, crossfit_fields = _cross_fitted_scores(
        merged["maps"],
        seed_stack,
        calibrated,
        domain["areas"],
        np.ones_like(region_prior),
        particle_mass_kg,
    )
    geometry_only_score = _score_variants(
        composite,
        baseline_fields["worn_depth_low_mm"],
        baseline_fields["worn_depth_nominal_mm"],
        baseline_fields["worn_depth_high_mm"],
        domain["areas"],
    )
    fields: dict[str, np.ndarray] = {
        "face_region_id": labels,
        "composite_prediction_depth_mm": composite.astype(np.float32),
        **baseline_fields,
        **convergence_fields,
        **crossfit_fields,
    }
    for model in MODEL_NAMES:
        fields[f"{model}_calibrated_depth_mm"] = calibrated[model].astype(np.float32)
    nominal_score = float(crossfit_scores["nominal_score"])
    conservative_score = float(crossfit_scores["conservative_score"])
    optimistic_score = float(crossfit_scores["optimistic_score"])
    report: dict[str, Any] = {
        "version": VERSION,
        "status": "conditional",
        "benchmark_kind": "competition_self_benchmark",
        "competition_self_benchmark_gate": True,
        "competition_self_benchmark_status": "conditional_participant_built_reference",
        "competition_self_benchmark_score": conservative_score,
        "nominal_score": nominal_score,
        "conservative_score": conservative_score,
        "optimistic_score": optimistic_score,
        "competition_self_benchmark_score_interval": crossfit_scores["score_interval"],
        "score_components": MODEL_SCORE_WEIGHTS,
        "official_accuracy_gate": False,
        "official_reference_gate": False,
        "fem_gate": "blocked",
        "reference_independence": "geometry_and_literature_reference_independent_of_model_fields",
        "literature_region_weights": region_weights,
        "inputs": {
            "face_domain": {"path": str(domain["path"]), "sha256": domain["sha256"], "n_faces": int(len(domain["areas"]))},
            "merged_prior": {"path": str(merged["path"]), "sha256": merged["sha256"], "npz_sha256": merged["npz_sha256"]},
            "seed_priors": [{"path": str(seed["path"]), "sha256": seed["sha256"], "npz_sha256": seed["npz_sha256"]} for seed in seeds],
            "literature_reference": {"path": str(literature["path"]), "sha256": literature["sha256"]},
        },
        "face_region_mapping": mapping,
        "active_face_convergence": convergence,
        "material_amplitude_calibration": material_report,
        "baseline_worn_reference": baseline_worn_report,
        "model_combination": combination_report,
        "score_detail": crossfit_scores,
        "crossfit_region_factor": {
            "mode": "neutral",
            "values": "ones",
            "reason": "unconfirmed PCA labels remain a mapping diagnostic; cross-fitted spatial score uses only leave-out particle support",
        },
        "geometry_only_reference_diagnostic": geometry_only_score,
        "source_layers": {
            "geometry": "official_geometry_as_face_domain_only",
            "region": "participant_built_geometry_mapping_plus_literature_region_prior",
            "material": "synthetic_material_reference_from_frozen_ZG0Cr13Ni4Mo_prior",
            "particle": "five_seed_median_with_active_face_convergence_summary",
            "prediction": "pre_registered_equal_weight_model_combination",
        },
        "scoring_policy": {
            "official_data_absence_numeric_deduction": 0.0,
            "participant_built_baseline_worn_allowed": True,
            "participant_built_face_region_mapping_allowed": True,
            "measured_truth_claimed": False,
        },
        "claim_scope": "conditional competition score using participant-built baseline/worn field and participant DEC face mapping; not official or measured truth",
        "limitations": [
            "No laboratory baseline/worn scan was supplied; synthetic participant field remains the reference source.",
            "No measured ZG0Cr13Ni4Mo batch was supplied; material calibration is prior-based.",
            "PCA region labels are participant assumptions and require reproducible geometry review.",
            "The official gates remain unchanged and cannot be inferred from the competition self-benchmark.",
        ],
    }
    report["report_sha256"] = canonical_sha256(report)
    if output_path is not None:
        target = Path(output_path).resolve()
        safe_json_dump(report, target)
        safe_npz_dump(fields, target.with_suffix(".npz"), metadata={"report_sha256": report["report_sha256"], "version": VERSION})
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the participant-built Step30 optimization pipeline")
    parser.add_argument("--merged-prior", type=Path, required=True)
    parser.add_argument("--seed-prior", type=Path, action="append", required=True)
    parser.add_argument("--face-domain", type=Path, required=True)
    parser.add_argument("--literature-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_optimized_pipeline(args.merged_prior, args.seed_prior, args.face_domain, args.literature_reference, output_path=args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
