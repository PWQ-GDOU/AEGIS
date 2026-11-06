"""Participant-built Step30 baseline/worn benchmark and conditional score.

This module is intentionally separate from the official gates.  It builds a
versioned, hash-bound reference from the official face domain, literature
regional evidence, and leave-one-out particle/model priors.  The result is a
conditional competition benchmark, never measured truth or official accuracy.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from step30_physical_accuracy_bridge_v1 import MODEL_NAMES
from step30_validation_common import canonical_sha256, safe_json_dump, safe_npz_dump, sha256_file


VERSION = "step30-competition-self-benchmark-v1"
BENCHMARK_KIND = "competition_self_benchmark"
REFERENCE_INDEPENDENCE = "leave_one_model_out_and_leave_one_seed_out"
MATERIAL_SOURCE_LAYERS = (
    "literature_material_prior",
    "ZG0Cr13Ni4Mo_preregistered_protocol",
    "synthetic_material_reference",
    "participant_measured_pilot",
)
DEFAULT_MATERIAL_PRIOR = {
    "density_kg_m3": {"low": 7700.0, "nominal": 7800.0, "high": 7900.0},
    "hardness_gpa": {"low": 3.0, "nominal": 4.0, "high": 5.0},
    "concentration_scale": {"low": 0.5, "nominal": 1.0, "high": 2.0},
}
MODEL_SCORE_WEIGHTS = {
    "hotspot_iou": 0.35,
    "ranking": 0.25,
    "js_distribution": 0.20,
    "absolute_amplitude": 0.20,
}


class CompetitionBenchmarkError(ValueError):
    """Raised when a self-built benchmark violates its audit contract."""


def _vector(value: Any, name: str, *, nonnegative: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise CompetitionBenchmarkError(f"{name} must be a non-empty finite vector")
    if nonnegative and np.any(array < 0.0):
        raise CompetitionBenchmarkError(f"{name} must be non-negative")
    return array


def _load_prior(path: str | Path, *, domain_artifact_path: str | Path | None = None) -> dict[str, Any]:
    report_path = Path(path).resolve()
    if not report_path.is_file():
        raise CompetitionBenchmarkError(f"prior report does not exist: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CompetitionBenchmarkError(f"invalid prior report: {report_path}") from exc
    if not isinstance(report, Mapping) or report.get("status") != "provisional":
        raise CompetitionBenchmarkError("prior report must have status=provisional")
    contract = report.get("prior_contract")
    if not isinstance(contract, Mapping) or contract.get("ai5_used_for_calibration") is not False:
        raise CompetitionBenchmarkError("prior report must prove AI5-free calibration")
    npz_path = report_path.with_suffix(".npz")
    if not npz_path.is_file():
        raise CompetitionBenchmarkError(f"prior array artifact does not exist: {npz_path}")
    domain_path = npz_path if domain_artifact_path is None else Path(domain_artifact_path).resolve()
    if not domain_path.is_file():
        raise CompetitionBenchmarkError(f"face-domain artifact does not exist: {domain_path}")
    with np.load(npz_path, allow_pickle=False) as data, np.load(domain_path, allow_pickle=False) as domain:
        if "face_centers_local_mm" in data.files and "face_area_m2" in data.files:
            centers = np.asarray(data["face_centers_local_mm"], dtype=np.float64)
            areas = _vector(data["face_area_m2"], "face_area_m2")
            domain_source = npz_path
        else:
            missing = [field for field in ("face_centers_local_mm", "face_area_m2") if field not in domain.files]
            if missing:
                raise CompetitionBenchmarkError(f"face-domain artifact lacks {missing}")
            centers = np.asarray(domain["face_centers_local_mm"], dtype=np.float64)
            areas = _vector(domain["face_area_m2"], "face_area_m2")
            domain_source = domain_path
        if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) != len(areas):
            raise CompetitionBenchmarkError("face centers and areas have incompatible shapes")
        if np.any(areas <= 0.0):
            raise CompetitionBenchmarkError("face_area_m2 must be positive")
        maps: dict[str, np.ndarray] = {}
        for model in MODEL_NAMES:
            field = f"{model}_p50_mm_h"
            if field not in data.files:
                raise CompetitionBenchmarkError(f"prior array artifact lacks {field}")
            values = _vector(data[field], field, nonnegative=True)
            if len(values) != len(areas):
                raise CompetitionBenchmarkError(f"{field} length differs from face domain")
            maps[model] = values
    return {
        "report": dict(report),
        "path": report_path,
        "npz_path": npz_path,
        "sha256": sha256_file(report_path),
        "npz_sha256": sha256_file(npz_path),
        "domain_path": domain_source,
        "domain_sha256": sha256_file(domain_source),
        "centers": centers,
        "areas": areas,
        "maps": maps,
    }


def _load_literature(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    if not target.is_file():
        raise CompetitionBenchmarkError(f"literature reference does not exist: {target}")
    try:
        report = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CompetitionBenchmarkError(f"invalid literature reference: {target}") from exc
    if not isinstance(report, Mapping) or report.get("reference_kind") != "literature_reference":
        raise CompetitionBenchmarkError("literature input must have reference_kind=literature_reference")
    if report.get("official_reference_gate") is True:
        raise CompetitionBenchmarkError("official literature reference cannot be substituted into self benchmark")
    regions = report.get("regions")
    if not isinstance(regions, Sequence) or isinstance(regions, (str, bytes)) or not regions:
        raise CompetitionBenchmarkError("literature reference must contain non-empty regions")
    scores: list[float] = []
    for item in regions:
        if not isinstance(item, Mapping) or not isinstance(item.get("region_id"), str):
            raise CompetitionBenchmarkError("literature region entries must contain region_id")
        score = float(item.get("literature_support_score", 0.0))
        if not math.isfinite(score) or score < 0.0:
            raise CompetitionBenchmarkError("literature support scores must be finite and non-negative")
        scores.append(score)
    total = float(sum(scores))
    if total <= 0.0:
        raise CompetitionBenchmarkError("literature support scores must have positive total")
    return {
        "path": target,
        "sha256": sha256_file(target),
        "report": dict(report),
        "regions": [dict(item) for item in regions],
        "support_fraction": {str(item["region_id"]): score / total for item, score in zip(regions, scores)},
    }


def _load_optional_region_field(path: str | Path | None, n_faces: int) -> tuple[np.ndarray | None, dict[str, Any]]:
    if path is None:
        return None, {"status": "unmapped", "reason": "no externally supplied face-region field"}
    target = Path(path).resolve()
    if not target.is_file():
        raise CompetitionBenchmarkError(f"literature face-region field does not exist: {target}")
    with np.load(target, allow_pickle=False) as data:
        key = "literature_region_prior" if "literature_region_prior" in data.files else "region_prior"
        if key not in data.files:
            raise CompetitionBenchmarkError("literature face-region field lacks literature_region_prior")
        field = _vector(data[key], key, nonnegative=True)
    if len(field) != n_faces:
        raise CompetitionBenchmarkError("literature face-region field length differs from face domain")
    if not np.any(field > 0.0):
        raise CompetitionBenchmarkError("literature face-region field has no positive support")
    return field / float(np.max(field)), {"status": "supplied", "path": str(target), "sha256": sha256_file(target)}


def _normalize(values: np.ndarray) -> np.ndarray:
    positive = values[values > 0.0]
    if positive.size == 0:
        return np.zeros_like(values, dtype=np.float64)
    scale = float(np.quantile(positive, 0.95))
    return np.clip(values / max(scale, 1.0e-30), 0.0, None)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    target = float(np.clip(quantile, 0.0, 1.0) * cumulative[-1])
    return float(ordered_values[min(int(np.searchsorted(cumulative, target, side="left")), len(values) - 1)])


def _weighted_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _weighted_spearman(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> float:
    if len(left) < 2 or float(np.sum(weights)) <= 0.0:
        return 1.0
    left_rank = _weighted_rank(left)
    right_rank = _weighted_rank(right)
    w = weights / float(np.sum(weights))
    left_centered = left_rank - float(np.sum(w * left_rank))
    right_centered = right_rank - float(np.sum(w * right_rank))
    denominator = math.sqrt(float(np.sum(w * left_centered**2)) * float(np.sum(w * right_centered**2)))
    return float(np.sum(w * left_centered * right_centered) / denominator) if denominator > 0.0 else 1.0


def _weighted_iou(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> float:
    union = float(np.sum(weights[(left) | (right)]))
    return float(np.sum(weights[left & right]) / union) if union > 0.0 else 1.0


def _active_js(left: np.ndarray, right: np.ndarray, areas: np.ndarray) -> float:
    active = (left > 0.0) | (right > 0.0)
    if not np.any(active):
        return 0.0
    left_mass = np.maximum(left[active], 0.0) * areas[active]
    right_mass = np.maximum(right[active], 0.0) * areas[active]
    left_total = float(np.sum(left_mass))
    right_total = float(np.sum(right_mass))
    if left_total <= 0.0 or right_total <= 0.0:
        return 1.0
    p = left_mass / left_total
    q = right_mass / right_total
    midpoint = 0.5 * (p + q)
    p_term = np.zeros_like(p)
    q_term = np.zeros_like(q)
    p_positive = p > 0.0
    q_positive = q > 0.0
    p_term[p_positive] = p[p_positive] * np.log(p[p_positive] / midpoint[p_positive])
    q_term[q_positive] = q[q_positive] * np.log(q[q_positive] / midpoint[q_positive])
    return float(0.5 * np.sum(p_term + q_term) / math.log(2.0))


def _top_area_mask(values: np.ndarray, areas: np.ndarray, fraction: float) -> np.ndarray:
    active = values > 0.0
    target = float(np.sum(areas[active]) * fraction)
    mask = np.zeros(len(values), dtype=bool)
    cumulative = 0.0
    for index in np.argsort(values, kind="mergesort")[::-1]:
        if not active[index]:
            continue
        mask[index] = True
        cumulative += float(areas[index])
        if cumulative >= target:
            break
    return mask


def _pair_score(prediction: np.ndarray, reference: np.ndarray, areas: np.ndarray) -> dict[str, Any]:
    prediction = np.maximum(np.asarray(prediction, dtype=np.float64), 0.0)
    reference = np.maximum(np.asarray(reference, dtype=np.float64), 0.0)
    active = (prediction > 0.0) | (reference > 0.0)
    if not np.any(active):
        raise CompetitionBenchmarkError("prediction/reference pair has no active face")
    p = prediction[active]
    r = reference[active]
    w = areas[active]
    thresholds = {
        "low_medium": (_weighted_quantile(p, w, 0.50), _weighted_quantile(r, w, 0.50)),
        "medium_high": (_weighted_quantile(p, w, 0.80), _weighted_quantile(r, w, 0.80)),
    }
    p_low, p_high = thresholds["low_medium"][0], thresholds["medium_high"][0]
    r_low, r_high = thresholds["low_medium"][1], thresholds["medium_high"][1]
    bands = {
        "low": ((p >= 0.0) & (p < p_low), (r >= 0.0) & (r < r_low)),
        "medium": ((p >= p_low) & (p < p_high), (r >= r_low) & (r < r_high)),
        "high": ((p >= p_high), (r >= r_high)),
    }
    band_iou = {name: _weighted_iou(pair[0], pair[1], w) for name, pair in bands.items()}
    hotspot_iou = float(np.mean(list(band_iou.values())))
    full_rank = _weighted_spearman(p, r, w)
    top_rank_values: dict[str, float] = {}
    for label, fraction in (("top_10_percent", 0.10), ("top_20_percent", 0.20)):
        p_top = _top_area_mask(p, w, fraction)
        r_top = _top_area_mask(r, w, fraction)
        subset = p_top | r_top
        top_rank_values[label] = _weighted_spearman(p[subset], r[subset], w[subset]) if np.any(subset) else 1.0
    rank_score = float(np.clip(np.mean([full_rank, top_rank_values["top_10_percent"], top_rank_values["top_20_percent"]]) * 0.5 + 0.5, 0.0, 1.0))
    js = _active_js(p, r, w)
    js_score = float(np.clip(1.0 - js, 0.0, 1.0))
    prediction_volume = float(np.sum(prediction * areas) * 1.0e6)
    reference_volume = float(np.sum(reference * areas) * 1.0e6)
    relative_volume_error = abs(prediction_volume - reference_volume) / max(reference_volume, 1.0e-30)
    amplitude_score = float(np.clip(1.0 - relative_volume_error, 0.0, 1.0))
    score = 100.0 * (
        MODEL_SCORE_WEIGHTS["hotspot_iou"] * hotspot_iou
        + MODEL_SCORE_WEIGHTS["ranking"] * rank_score
        + MODEL_SCORE_WEIGHTS["js_distribution"] * js_score
        + MODEL_SCORE_WEIGHTS["absolute_amplitude"] * amplitude_score
    )
    return {
        "score": score,
        "components": {
            "hotspot_iou": hotspot_iou,
            "ranking": rank_score,
            "js_distribution": js_score,
            "absolute_amplitude": amplitude_score,
        },
        "weights": dict(MODEL_SCORE_WEIGHTS),
        "band_iou": band_iou,
        "ranking_raw": {"full": full_rank, **top_rank_values},
        "active_face_fraction": float(np.mean(active)),
        "active_face_count": int(np.count_nonzero(active)),
        "active_js_divergence": js,
        "prediction_volume_mm3_h": prediction_volume,
        "reference_volume_mm3_h": reference_volume,
        "relative_volume_error": relative_volume_error,
    }


def _material_factor(density: float, hardness: float, concentration: float) -> float:
    return (density / 7800.0) * (4.0 / hardness) * concentration


def _reference_from_candidates(
    candidates: Sequence[np.ndarray],
    region_field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a raw-rate reference and its dimensionless regional support."""
    if not candidates:
        raise CompetitionBenchmarkError("at least one reference candidate is required")
    normalized = np.mean(np.stack([_normalize(values) for values in candidates], axis=0), axis=0)
    anchors = [float(np.quantile(values[values > 0.0], 0.95)) for values in candidates if np.any(values > 0.0)]
    if not anchors:
        raise CompetitionBenchmarkError("reference candidates have no positive amplitude")
    amplitude_anchor = float(np.median(anchors))
    regional_shape = np.clip(normalized * region_field, 0.0, None)
    return regional_shape * amplitude_anchor, regional_shape


def _scenario_reference(base: np.ndarray, region: np.ndarray, *, material_factor: float, region_expansion: float) -> np.ndarray:
    base = np.maximum(base, 0.0) * material_factor
    if region_expansion <= 1.0:
        return base * region_expansion
    positive = base[base > 0.0]
    floor = float(np.quantile(positive, 0.10)) if positive.size else 0.0
    return base + (region_expansion - 1.0) * floor * np.clip(region, 0.0, None)


def _scenario_grid() -> list[dict[str, float]]:
    scenarios: list[dict[str, float]] = []
    for density in (7700.0, 7800.0, 7900.0):
        for hardness in (3.0, 4.0, 5.0):
            for concentration in (0.5, 1.0, 2.0):
                for expansion in (0.85, 1.0, 1.15):
                    scenarios.append({
                        "density_kg_m3": density,
                        "hardness_gpa": hardness,
                        "concentration_scale": concentration,
                        "region_expansion": expansion,
                        "material_factor": _material_factor(density, hardness, concentration),
                    })
    return scenarios


def _descriptor(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(item["path"]),
        "sha256": item["sha256"],
        "npz_path": str(item["npz_path"]),
        "npz_sha256": item["npz_sha256"],
    }


def build_self_benchmark(
    merged_prior_report_path: str | Path,
    seed_prior_paths: Sequence[str | Path],
    literature_reference_path: str | Path,
    *,
    face_domain_path: str | Path | None = None,
    literature_region_field_path: str | Path | None = None,
    material_source_layer: str = "synthetic_material_reference",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the conditional participant benchmark and all score variants."""
    if material_source_layer not in MATERIAL_SOURCE_LAYERS:
        raise CompetitionBenchmarkError(f"unsupported material source layer: {material_source_layer}")
    if len(seed_prior_paths) < 2:
        raise CompetitionBenchmarkError("at least two independent seed priors are required")
    seed_paths = [Path(path).resolve() for path in seed_prior_paths]
    if len(set(seed_paths)) != len(seed_paths):
        raise CompetitionBenchmarkError("seed prior paths must be unique")
    merged = _load_prior(merged_prior_report_path, domain_artifact_path=face_domain_path)
    seeds = [_load_prior(path, domain_artifact_path=face_domain_path) for path in seed_paths]
    for item in seeds:
        if not np.array_equal(item["centers"], merged["centers"]) or not np.array_equal(item["areas"], merged["areas"]):
            raise CompetitionBenchmarkError("seed face domain differs from merged prior")
    literature = _load_literature(literature_reference_path)
    n_faces = len(merged["areas"])
    region_field, region_binding = _load_optional_region_field(literature_region_field_path, n_faces)
    if region_field is None:
        region_field = np.ones(n_faces, dtype=np.float64)
    seed_stack = np.stack([[seed["maps"][model] for model in MODEL_NAMES] for seed in seeds], axis=0)
    merged_maps = merged["maps"]
    all_model_stack = np.stack([merged_maps[model] for model in MODEL_NAMES], axis=0)
    fields: dict[str, np.ndarray] = {}
    model_scores: dict[str, Any] = {}
    seed_scores: dict[str, Any] = {}
    scenario_grid = _scenario_grid()

    # Every model target receives a reference made without that model.  Every
    # seed target additionally receives a reference made without that seed.
    model_references: dict[str, np.ndarray] = {}
    for model_index, model in enumerate(MODEL_NAMES):
        other_model = np.delete(all_model_stack, model_index, axis=0)
        model_base, model_shape = _reference_from_candidates(list(other_model), region_field)
        seed_support = np.mean([_normalize(seed_stack[i, j]) for i in range(len(seeds)) for j in range(len(MODEL_NAMES)) if j != model_index], axis=0)
        model_region = np.sqrt(np.clip(model_shape, 0.0, None) * np.clip(seed_support, 0.0, None))
        if np.max(model_region) > 0.0:
            model_region /= float(np.max(model_region))
        model_references[model] = model_base
        fields[f"reference_region_prior__{model}"] = model_region.astype(np.float32)
        fields[f"reference_nominal_worn_field__{model}"] = model_base.astype(np.float32)
        fields[f"reference_depth_envelope_low__{model}"] = (model_base * _material_factor(7700.0, 5.0, 0.5)).astype(np.float32)
        fields[f"reference_depth_envelope_high__{model}"] = (model_base * _material_factor(7900.0, 3.0, 2.0)).astype(np.float32)
        scenario_scores: list[dict[str, Any]] = []
        for scenario in scenario_grid:
            reference = _scenario_reference(model_base, model_region, material_factor=scenario["material_factor"], region_expansion=scenario["region_expansion"])
            score = _pair_score(merged_maps[model], reference, merged["areas"])
            scenario_scores.append({**scenario, **score})
        nominal = min(scenario_scores, key=lambda item: abs(item["material_factor"] - 1.0) + abs(item["region_expansion"] - 1.0))
        model_scores[model] = {
            "reference_independence": "leave_one_model_out",
            "nominal": nominal,
            "conservative": min(scenario_scores, key=lambda item: item["score"]),
            "optimistic": max(scenario_scores, key=lambda item: item["score"]),
            "score_interval": {"lower": float(min(item["score"] for item in scenario_scores)), "upper": float(max(item["score"] for item in scenario_scores))},
            "scenario_count": len(scenario_scores),
        }

        per_seed: dict[str, Any] = {}
        for seed_index in range(len(seeds)):
            leave_out = [seed_stack[i, j] for i in range(len(seeds)) if i != seed_index for j in range(len(MODEL_NAMES)) if j != model_index]
            seed_reference, seed_region = _reference_from_candidates(leave_out, region_field)
            fields[f"reference_nominal_worn_field__{model}__seed_{seed_index}"] = seed_reference.astype(np.float32)
            target = seed_stack[seed_index, model_index]
            target_scenarios = []
            for scenario in scenario_grid:
                reference = _scenario_reference(seed_reference, seed_region, material_factor=scenario["material_factor"], region_expansion=scenario["region_expansion"])
                target_scenarios.append({**scenario, **_pair_score(target, reference, merged["areas"])})
            per_seed[str(seed_index)] = {
                "reference_independence": "leave_one_model_out_and_leave_one_seed_out",
                "nominal": min(target_scenarios, key=lambda item: abs(item["material_factor"] - 1.0) + abs(item["region_expansion"] - 1.0)),
                "conservative": min(target_scenarios, key=lambda item: item["score"]),
                "optimistic": max(target_scenarios, key=lambda item: item["score"]),
                "score_interval": {"lower": float(min(item["score"] for item in target_scenarios)), "upper": float(max(item["score"] for item in target_scenarios))},
            }
        seed_scores[model] = per_seed

    generic_reference = np.mean(np.stack(list(model_references.values()), axis=0), axis=0)
    generic_region = np.mean(np.stack([fields[f"reference_region_prior__{model}"] for model in MODEL_NAMES], axis=0), axis=0)
    fields["reference_region_prior"] = generic_region.astype(np.float32)
    fields["reference_nominal_worn_field"] = generic_reference.astype(np.float32)
    reference_low = generic_reference * _material_factor(7700.0, 5.0, 0.5)
    reference_high = generic_reference * _material_factor(7900.0, 3.0, 2.0)
    fields["reference_depth_envelope_low"] = reference_low.astype(np.float32)
    fields["reference_depth_envelope_high"] = reference_high.astype(np.float32)
    fields["reference_depth_envelope"] = np.stack(
        [reference_low, generic_reference, reference_high], axis=0
    ).astype(np.float32)

    nominal_scores = [model_scores[model]["nominal"]["score"] for model in MODEL_NAMES]
    conservative_scores = [model_scores[model]["conservative"]["score"] for model in MODEL_NAMES]
    optimistic_scores = [model_scores[model]["optimistic"]["score"] for model in MODEL_NAMES]
    report: dict[str, Any] = {
        "version": VERSION,
        "status": "conditional",
        "benchmark_kind": BENCHMARK_KIND,
        "reference_independence": REFERENCE_INDEPENDENCE,
        "official_reference": False,
        "measured_truth": False,
        "competition_self_benchmark_gate": True,
        "competition_self_benchmark_status": "conditional",
        "competition_self_benchmark_score": float(np.median(conservative_scores)),
        "competition_self_benchmark_score_interval": {"lower": float(min(conservative_scores)), "upper": float(max(optimistic_scores))},
        "nominal_score": float(np.median(nominal_scores)),
        "conservative_score": float(np.median(conservative_scores)),
        "optimistic_score": float(np.median(optimistic_scores)),
        "score_components": dict(MODEL_SCORE_WEIGHTS),
        "scoring_policy": {
            "participant_built_baseline_worn_allowed": True,
            "participant_built_face_region_mapping_allowed": True,
            "official_data_absence_numeric_deduction": 0.0,
            "measured_truth_status": "not_claimed",
            "conditional_status_reason": "reference is participant-built and must remain distinguishable from measured truth",
        },
        "gate_semantics": "participant_built_reference_accepted; official_data_absence_is_not_a_numeric_deduction",
        "official_accuracy_gate": False,
        "official_reference_gate": False,
        "fem_gate": "blocked",
        "source_layers": {
            "official_geometry": "baseline_face_domain_only",
            "literature": "literature_material_prior_and_region_prior",
            "material": material_source_layer,
            "particle": "five_seed_mass_conservative_ensemble",
        },
        "inputs": {
            "merged_prior": _descriptor(merged),
            "seed_priors": [_descriptor(item) for item in seeds],
            "face_domain_artifact": {"path": str(merged["domain_path"]), "sha256": merged["domain_sha256"]},
            "literature_reference": {"path": str(literature["path"]), "sha256": literature["sha256"]},
            "literature_region_field": region_binding,
            "face_domain": {"n_faces": n_faces, "areas_sha256": canonical_sha256(merged["areas"].tolist()), "centers_sha256": canonical_sha256(merged["centers"].tolist())},
        },
        "material_prior": DEFAULT_MATERIAL_PRIOR,
        "region_prior": {
            "literature_support_fraction": literature["support_fraction"],
            "face_mapping_status": region_binding["status"],
            "interpretation": "regional evidence is a conditional prior; no unmapped region label is claimed as DEC face truth",
        },
        "reference_fields": {
            "region": "reference_region_prior",
            "depth_envelope": ["low", "nominal", "high"],
            "nominal_worn": "reference_nominal_worn_field",
            "units": "mm_per_h_prior_rate_scale",
        },
        "model_scores": model_scores,
        "seed_scores": seed_scores,
        "deductions_and_boundaries": [
            {"item": "participant_built_baseline_worn_reference", "effect": "no_numeric_deduction", "claim_boundary": "not_official_or_measured_truth"},
            {"item": "participant_built_DEC_face_region_mapping", "effect": "no_numeric_deduction", "claim_boundary": "mapping assumptions must remain hash-bound"},
            {"item": "material_is_not_measured_ZG0Cr13Ni4Mo_batch", "effect": "sensitivity_interval_only", "claim_boundary": "amplitude is prior-sensitive"},
            {"item": "official_reference_and_accuracy_gates_remain_closed", "effect": "no_numeric_deduction", "claim_boundary": "not_official_accuracy"},
        ],
        "claim_scope": "conditional competition score against a participant-built leave-one-out benchmark; not official accuracy and not measured wear truth",
        "limitations": [
            "The reference field is participant-built and is not presented as an official or measured physical truth.",
            "The score is eligible for the competition self-benchmark track; its conditional label describes evidence provenance, not a missing-data penalty.",
            "The ensemble's common particle support and high active-face seed CV remain visible in upstream evidence.",
        ],
    }
    report["report_sha256"] = canonical_sha256(report)
    if output_path is not None:
        target = Path(output_path).resolve()
        safe_json_dump(report, target)
        safe_npz_dump(fields, target.with_suffix(".npz"), metadata={"report_sha256": report["report_sha256"], "version": VERSION})
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the conditional Step30 participant self benchmark")
    parser.add_argument("--merged-prior", type=Path, required=True)
    parser.add_argument("--seed-prior", type=Path, action="append", required=True)
    parser.add_argument("--face-domain", type=Path)
    parser.add_argument("--literature-reference", type=Path, required=True)
    parser.add_argument("--literature-region-field", type=Path)
    parser.add_argument("--material-source-layer", default="synthetic_material_reference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_self_benchmark(
        args.merged_prior,
        args.seed_prior,
        args.literature_reference,
        face_domain_path=args.face_domain,
        literature_region_field_path=args.literature_region_field,
        material_source_layer=args.material_source_layer,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
