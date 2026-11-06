"""Separate particle-seed uncertainty from erosion-model uncertainty.

The input seed reports are p50 face maps generated with the same frozen prior
contract.  Seed statistics describe Monte Carlo/input sampling variability;
model statistics describe disagreement between the four erosion laws on the
mass-conservative merged input.  Neither statistic is a comparison to worn
material truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from step30_physical_accuracy_bridge_v1 import (
    MODEL_NAMES,
    _hotspot_iou,
    _load_prior,
    _normalized_maps,
    _spearman,
)
from step30_validation_common import canonical_sha256, safe_json_dump, safe_npz_dump, sha256_file


VERSION = "step30-physical-accuracy-ensemble-v1"


class PhysicalAccuracyEnsembleError(ValueError):
    """Raised when seed prior maps cannot be compared on one face domain."""


def _seed_summary(stack: np.ndarray) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    values = np.asarray(stack, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise PhysicalAccuracyEnsembleError("seed stack must have shape (n_seeds>=2, n_faces)")
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0, ddof=1)
    cv = std / np.maximum(np.abs(mean), 1.0e-12)
    active = np.any(values > 0.0, axis=0)
    active_std = std[active] if np.any(active) else np.zeros(1, dtype=np.float64)
    active_cv = cv[active] if np.any(active) else np.zeros(1, dtype=np.float64)
    summary = {
        "seed_mean_of_face_mean": float(np.mean(mean)),
        "seed_std_p50": float(np.quantile(std, 0.50)),
        "seed_std_p95": float(np.quantile(std, 0.95)),
        "seed_cv_p50": float(np.quantile(cv, 0.50)),
        "seed_cv_p95": float(np.quantile(cv, 0.95)),
        "high_cv_face_fraction": float(np.mean(cv >= 0.5)),
        "active_face_fraction": float(np.mean(active)),
        "active_seed_std_p50": float(np.quantile(active_std, 0.50)),
        "active_seed_std_p95": float(np.quantile(active_std, 0.95)),
        "active_seed_cv_p50": float(np.quantile(active_cv, 0.50)),
        "active_seed_cv_p95": float(np.quantile(active_cv, 0.95)),
        "active_high_cv_face_fraction": float(np.mean(active_cv >= 0.5)),
    }
    fields = {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "cv": cv.astype(np.float32),
        "p05": np.quantile(values, 0.05, axis=0).astype(np.float32),
        "p50": np.quantile(values, 0.50, axis=0).astype(np.float32),
        "p95": np.quantile(values, 0.95, axis=0).astype(np.float32),
    }
    return summary, fields


def build_ensemble(
    merged_prior_report_path: str | Path,
    seed_prior_paths: Sequence[str | Path],
    *,
    hotspot_quantile: float = 0.90,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if len(seed_prior_paths) < 2:
        raise PhysicalAccuracyEnsembleError("at least two seed prior reports are required")
    merged_report, merged_maps, merged_report_path, merged_npz_path = _load_prior(merged_prior_report_path)
    seed_data = [_load_prior(path) for path in seed_prior_paths]
    expected_faces = len(next(iter(merged_maps.values())))
    for _, maps, _, _ in seed_data:
        if any(len(values) != expected_faces for values in maps.values()):
            raise PhysicalAccuracyEnsembleError("seed prior face counts differ from merged prior")
    seed_stacks = {model: np.vstack([maps[model] for _, maps, _, _ in seed_data]) for model in MODEL_NAMES}
    seed_summaries: dict[str, Any] = {}
    fields: dict[str, np.ndarray] = {}
    for model in MODEL_NAMES:
        summary, model_fields = _seed_summary(seed_stacks[model])
        seed_summaries[model] = summary
        for name, values in model_fields.items():
            fields[f"{model}_seed_{name}"] = values

    normalized = _normalized_maps(merged_maps)
    stacked_models = np.vstack([normalized[model] for model in MODEL_NAMES])
    consensus = np.exp(np.mean(np.log(np.maximum(stacked_models, 1.0e-12)), axis=0))
    model_disagreement_cv = np.std(stacked_models, axis=0) / np.maximum(np.mean(stacked_models, axis=0), 1.0e-12)
    model_active = np.any(stacked_models > 0.0, axis=0)
    active_model_cv = model_disagreement_cv[model_active] if np.any(model_active) else np.zeros(1, dtype=np.float64)
    pairwise: list[dict[str, Any]] = []
    for index, left in enumerate(MODEL_NAMES):
        for right in MODEL_NAMES[index + 1 :]:
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "spearman": _spearman(normalized[left], normalized[right]),
                    "hotspot_iou": _hotspot_iou(normalized[left], normalized[right], hotspot_quantile),
                }
            )
    spearman = np.asarray([row["spearman"] for row in pairwise], dtype=np.float64)
    iou = np.asarray([row["hotspot_iou"] for row in pairwise], dtype=np.float64)
    for model in MODEL_NAMES:
        fields[f"{model}_merged_p50"] = merged_maps[model].astype(np.float32)
        fields[f"{model}_merged_normalized"] = normalized[model].astype(np.float32)
    fields["consensus_normalized"] = consensus.astype(np.float32)
    fields["model_disagreement_cv"] = model_disagreement_cv.astype(np.float32)
    report: dict[str, Any] = {
        "version": VERSION,
        "status": "provisional_physical_accuracy_ensemble",
        "physical_spatial_accuracy_status": "blocked_without_independent_measured_truth",
        "spatial_accuracy_status": "blocked",
        "official_accuracy_gate": False,
        "official_reference_gate": False,
        "fem_gate": "blocked",
        "claim_scope": "seed uncertainty and erosion-model disagreement for exploratory spatial screening",
        "merged_prior": {
            "path": str(merged_report_path),
            "sha256": sha256_file(merged_report_path),
            "npz_path": str(merged_npz_path),
            "npz_sha256": sha256_file(merged_npz_path),
            "n_faces": expected_faces,
        },
        "seed_priors": [
            {"path": str(report_path), "sha256": sha256_file(report_path), "npz_path": str(npz_path), "npz_sha256": sha256_file(npz_path)}
            for _, _, report_path, npz_path in seed_data
        ],
        "seed_uncertainty": {
            "n_seeds": len(seed_data),
            "models": seed_summaries,
            "interpretation": "variation across independent particle seeds on the same frozen CFD/topology contract",
        },
        "model_uncertainty": {
            "models": list(MODEL_NAMES),
            "pairwise": pairwise,
            "spearman_median": float(np.median(spearman)),
            "spearman_min": float(np.min(spearman)),
            "hotspot_iou_median": float(np.median(iou)),
            "hotspot_iou_min": float(np.min(iou)),
            "normalized_disagreement_cv_p50": float(np.quantile(model_disagreement_cv, 0.50)),
            "normalized_disagreement_cv_p95": float(np.quantile(model_disagreement_cv, 0.95)),
            "active_face_fraction": float(np.mean(model_active)),
            "active_normalized_disagreement_cv_p50": float(np.quantile(active_model_cv, 0.50)),
            "active_normalized_disagreement_cv_p95": float(np.quantile(active_model_cv, 0.95)),
            "interpretation": "model disagreement is conditional on one common particle-hit driver and is not physical validation",
        },
        "limitations": [
            "Seed uncertainty is not a measurement uncertainty and does not cover CFD model-form error.",
            "The four laws share the same hit locations; near-identical spatial rankings are expected under sparse common support.",
            "No baseline/worn scan or independent ZG0Cr13Ni4Mo spatial measurement is supplied.",
        ],
    }
    report["report_sha256"] = canonical_sha256(report)
    if output_path is not None:
        target = Path(output_path).resolve()
        safe_json_dump(report, target)
        safe_npz_dump(fields, target.with_suffix(".npz"), metadata={"report_sha256": report["report_sha256"], "version": VERSION})
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Separate seed and model uncertainty for Step30 physical screening")
    parser.add_argument("--merged-prior", type=Path, required=True)
    parser.add_argument("--seed-prior", type=Path, action="append", required=True)
    parser.add_argument("--hotspot-quantile", type=float, default=0.90)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_ensemble(args.merged_prior, args.seed_prior, hotspot_quantile=args.hotspot_quantile, output_path=args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
