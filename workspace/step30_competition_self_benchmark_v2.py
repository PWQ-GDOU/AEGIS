"""Versioned participant self-benchmark score with preregistered model combination."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import step30_competition_self_benchmark_v1 as base
from step30_validation_common import canonical_sha256, safe_json_dump, safe_npz_dump

VERSION = "step30-competition-self-benchmark-v2-model-combination"
COMBINATION_WEIGHTS = {name: 0.25 for name in base.MODEL_NAMES}


def _p95(values: np.ndarray) -> float:
    positive = values[values > 0.0]
    return float(np.quantile(positive, 0.95)) if positive.size else 1.0


def _combine(maps: dict[str, np.ndarray]) -> np.ndarray:
    names = list(maps)
    normalized = np.mean(np.stack([base._normalize(maps[name]) * COMBINATION_WEIGHTS[name] / 0.25 for name in names], axis=0), axis=0)
    anchor = float(np.median([_p95(maps[name]) for name in names]))
    return normalized * anchor


def _scenario_summary(prediction: np.ndarray, reference_base: np.ndarray, region: np.ndarray, areas: np.ndarray) -> dict[str, Any]:
    rows = []
    for scenario in base._scenario_grid():
        reference = base._scenario_reference(reference_base, region, material_factor=scenario["material_factor"], region_expansion=scenario["region_expansion"])
        rows.append({**scenario, **base._pair_score(prediction, reference, areas)})
    nominal = min(rows, key=lambda item: abs(item["material_factor"] - 1.0) + abs(item["region_expansion"] - 1.0))
    conservative = min(rows, key=lambda item: item["score"])
    optimistic = max(rows, key=lambda item: item["score"])
    return {"nominal": nominal, "conservative": conservative, "optimistic": optimistic, "score_interval": {"lower": float(conservative["score"]), "upper": float(optimistic["score"])}, "scenario_count": len(rows)}


def build_self_benchmark_v2(merged_prior: str | Path, seeds: Sequence[str | Path], literature: str | Path, *, face_domain: str | Path | None = None, region_field: str | Path | None = None, output: str | Path) -> dict[str, Any]:
    output_path = Path(output).resolve()
    audit_path = output_path.with_name(output_path.stem + "_single_model_audit.json")
    audit = base.build_self_benchmark(merged_prior, seeds, literature, face_domain_path=face_domain, literature_region_field_path=region_field, material_source_layer="synthetic_material_reference", output_path=audit_path)
    merged = base._load_prior(merged_prior, domain_artifact_path=face_domain)
    seed_items = [base._load_prior(path, domain_artifact_path=face_domain) for path in seeds]
    region, region_binding = base._load_optional_region_field(region_field, len(merged["areas"]))
    if region is None:
        region = np.ones(len(merged["areas"]), dtype=np.float64)
    merged_maps = merged["maps"]
    prediction = _combine(merged_maps)
    model_loo = {}
    combination_references = {}
    for excluded in base.MODEL_NAMES:
        others = {name: merged_maps[name] for name in base.MODEL_NAMES if name != excluded}
        reference_base, reference_shape = base._reference_from_candidates(list(others.values()), region)
        seed_support = np.mean([base._normalize(item["maps"][name]) for item in seed_items for name in base.MODEL_NAMES if name != excluded], axis=0)
        combined_region = np.sqrt(np.clip(reference_shape, 0.0, None) * np.clip(seed_support, 0.0, None))
        if np.max(combined_region) > 0.0:
            combined_region /= float(np.max(combined_region))
        model_loo[excluded] = _scenario_summary(prediction, reference_base, combined_region, merged["areas"])
        combination_references[f"combined_reference__omit_{excluded}"] = reference_base.astype(np.float32)
    seed_loo = {}
    for index, target in enumerate(seed_items):
        other_seed_maps = []
        for seed_index, item in enumerate(seed_items):
            if seed_index == index:
                continue
            other_seed_maps.extend(item["maps"].values())
        reference_base, reference_shape = base._reference_from_candidates(other_seed_maps, region)
        target_prediction = _combine(target["maps"])
        seed_loo[str(index)] = _scenario_summary(target_prediction, reference_base, reference_shape, merged["areas"])
        combination_references[f"combined_seed_reference__leave_out_{index}"] = reference_base.astype(np.float32)
    model_nominal = float(np.median([v["nominal"]["score"] for v in model_loo.values()]))
    model_conservative = float(np.median([v["conservative"]["score"] for v in model_loo.values()]))
    model_optimistic = float(np.median([v["optimistic"]["score"] for v in model_loo.values()]))
    seed_nominal = float(np.median([v["nominal"]["score"] for v in seed_loo.values()]))
    seed_conservative = float(np.median([v["conservative"]["score"] for v in seed_loo.values()]))
    seed_optimistic = float(np.median([v["optimistic"]["score"] for v in seed_loo.values()]))
    report = {
        "version": VERSION,
        "status": "conditional",
        "benchmark_kind": base.BENCHMARK_KIND,
        "reference_independence": base.REFERENCE_INDEPENDENCE,
        "participant_built": True,
        "measured_truth": False,
        "competition_self_benchmark_gate": True,
        "competition_self_benchmark_status": "conditional_preregistered_model_combination",
        "model_combination": {"weights": COMBINATION_WEIGHTS, "weight_sum": 1.0, "normalization": "per-model p95 normalized then equal-weight mean with median p95 amplitude anchor", "posthoc_weight_change_allowed": False},
        "model_combination_score": {"model_leave_one_out": {"nominal": model_nominal, "conservative": model_conservative, "optimistic": model_optimistic}, "seed_leave_one_out": {"nominal": seed_nominal, "conservative": seed_conservative, "optimistic": seed_optimistic}},
        "competition_self_benchmark_score": model_conservative,
        "competition_self_benchmark_score_interval": {"lower": float(min(model_conservative, seed_conservative)), "upper": float(max(model_optimistic, seed_optimistic))},
        "nominal_score": model_nominal,
        "conservative_score": model_conservative,
        "optimistic_score": model_optimistic,
        "official_accuracy_gate": False,
        "official_reference_gate": False,
        "fem_gate": "blocked",
        "source_layers": {"official_geometry": "baseline_face_domain_only", "particle": "10 independent 60000-parcel runs with stable topology aggregate", "material": "synthetic_material_reference", "literature": "participant-built literature region prior"},
        "inputs": {"base_single_model_audit": {"path": str(audit_path), "sha256": base.sha256_file(audit_path)}, "merged_prior": {"path": str(Path(merged_prior).resolve()), "sha256": base.sha256_file(merged["path"])}, "seed_count": len(seed_items), "n_faces": len(merged["areas"]), "region_field": region_binding},
        "single_model_audit": {"nominal": audit["nominal_score"], "conservative": audit["conservative_score"], "optimistic": audit["optimistic_score"], "score_interval": audit["competition_self_benchmark_score_interval"]},
        "model_leave_one_out": model_loo,
        "seed_leave_one_out": seed_loo,
        "claim_scope": "conditional competition score against a participant-built independent benchmark; not official accuracy and not measured wear truth",
        "limitations": ["The reference field is participant-built and synthetic/prior-based.", "The combination score is independent by leave-one-model-out and leave-one-seed-out construction, but it is not measured physical accuracy.", "Official accuracy, official reference, and formal FEM gates remain closed."]
    }
    report["report_sha256"] = canonical_sha256(report)
    with np.load(audit_path.with_suffix(".npz"), allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files if name != "__metadata_json__"}
    arrays["combined_prediction"] = prediction.astype(np.float32)
    arrays.update(combination_references)
    safe_json_dump(report, output_path)
    safe_npz_dump(arrays, output_path.with_suffix(".npz"), metadata={"report_sha256": report["report_sha256"], "version": VERSION})
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build v2 participant benchmark with preregistered model combination")
    parser.add_argument("--merged-prior", type=Path, required=True)
    parser.add_argument("--seed-prior", type=Path, action="append", required=True)
    parser.add_argument("--literature-reference", type=Path, required=True)
    parser.add_argument("--face-domain", type=Path)
    parser.add_argument("--literature-region-field", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_self_benchmark_v2(args.merged_prior, args.seed_prior, args.literature_reference, face_domain=args.face_domain, region_field=args.literature_region_field, output=args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())