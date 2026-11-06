"""Build a conservative physical-accuracy bridge from model priors.

This module compares the four frozen erosion-law maps on a common parent-face
domain and produces a consensus screening map plus an epistemic disagreement
field.  It is deliberately unable to open the official accuracy gate: only an
independent baseline/worn measurement package can do that.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from step30_validation_common import canonical_sha256, safe_json_dump, safe_npz_dump, sha256_file


VERSION = "step30-physical-accuracy-bridge-v1"
MODEL_NAMES = ("oka", "finnie", "tabakoff", "mansouri")
DEFAULT_HOTSPOT_QUANTILE = 0.90


class PhysicalAccuracyBridgeError(ValueError):
    """Raised when a prior map cannot support a conservative comparison."""


def _finite_vector(value: Any, name: str, *, positive: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise PhysicalAccuracyBridgeError(f"{name} must be a non-empty finite vector")
    if positive and np.any(array <= 0.0):
        raise PhysicalAccuracyBridgeError(f"{name} must be positive")
    return array


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = _rank(left)
    right_rank = _rank(right)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    return float(np.dot(left_centered, right_centered) / denominator) if denominator else 1.0


def _hotspot_iou(left: np.ndarray, right: np.ndarray, quantile: float) -> float:
    left_mask = left >= np.quantile(left, quantile)
    right_mask = right >= np.quantile(right, quantile)
    union = np.count_nonzero(left_mask | right_mask)
    return float(np.count_nonzero(left_mask & right_mask) / union) if union else 1.0


def _normalized_maps(maps: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    normalized: dict[str, np.ndarray] = {}
    for model, values in maps.items():
        scale = float(np.quantile(values, 0.95))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(np.max(values))
        if not np.isfinite(scale) or scale <= 0.0:
            raise PhysicalAccuracyBridgeError(f"{model} map has no positive scale")
        normalized[model] = np.clip(values / scale, 0.0, None)
    return normalized


def _load_prior(prior_report_path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray], Path, Path]:
    report_path = Path(prior_report_path).resolve()
    if not report_path.is_file():
        raise PhysicalAccuracyBridgeError(f"prior report does not exist: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhysicalAccuracyBridgeError(f"invalid prior report: {report_path}") from exc
    if not isinstance(report, Mapping) or report.get("status") != "provisional":
        raise PhysicalAccuracyBridgeError("prior report must have provisional status")
    contract = report.get("prior_contract")
    if not isinstance(contract, Mapping) or contract.get("ai5_used_for_calibration") is not False:
        raise PhysicalAccuracyBridgeError("prior report does not prove AI5-free calibration")
    npz_path = report_path.with_suffix(".npz")
    if not npz_path.is_file():
        raise PhysicalAccuracyBridgeError(f"prior array artifact does not exist: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        if "face_centers_local_mm" not in data.files:
            raise PhysicalAccuracyBridgeError("prior array artifact lacks face centers")
        centers = _finite_vector(data["face_centers_local_mm"].reshape(-1), "face centers")
        del centers
        maps: dict[str, np.ndarray] = {}
        for model in MODEL_NAMES:
            field = f"{model}_p50_mm_h"
            if field not in data.files:
                raise PhysicalAccuracyBridgeError(f"prior array artifact lacks {field}")
            maps[model] = _finite_vector(data[field], field)
    lengths = {len(values) for values in maps.values()}
    if len(lengths) != 1:
        raise PhysicalAccuracyBridgeError("prior model maps have different face counts")
    return dict(report), maps, report_path, npz_path


def build_bridge(
    prior_report_path: str | Path,
    *,
    measured_report_path: str | Path | None = None,
    hotspot_quantile: float = DEFAULT_HOTSPOT_QUANTILE,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if not 0.5 < hotspot_quantile < 1.0:
        raise PhysicalAccuracyBridgeError("hotspot_quantile must be between 0.5 and 1.0")
    report, maps, prior_report, prior_npz = _load_prior(prior_report_path)
    normalized = _normalized_maps(maps)
    model_list = list(MODEL_NAMES)
    pairwise: list[dict[str, Any]] = []
    for index, left_name in enumerate(model_list):
        for right_name in model_list[index + 1 :]:
            pairwise.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "spearman": _spearman(normalized[left_name], normalized[right_name]),
                    "hotspot_iou": _hotspot_iou(
                        normalized[left_name], normalized[right_name], hotspot_quantile
                    ),
                }
            )
    stacked = np.vstack([normalized[model] for model in model_list])
    consensus = np.exp(np.mean(np.log(np.maximum(stacked, 1.0e-12)), axis=0))
    disagreement_cv = np.std(stacked, axis=0) / np.maximum(np.mean(stacked, axis=0), 1.0e-12)
    rank_consensus = np.mean(np.vstack([_rank(normalized[model]) for model in model_list]), axis=0)
    rank_consensus = rank_consensus / max(len(rank_consensus) - 1, 1)
    raw_summary = {
        model: {
            "p50_map_mean": float(np.mean(maps[model])),
            "p50_map_max": float(np.max(maps[model])),
            "p50_positive_face_fraction": float(np.mean(maps[model] > 0.0)),
        }
        for model in model_list
    }
    positive_support = np.vstack([maps[model] > 0.0 for model in model_list])
    shared_support_fraction = float(np.mean(np.all(positive_support, axis=0)))
    union_support = np.any(positive_support, axis=0)
    support_iou = float(np.mean(np.all(positive_support, axis=0)) / np.mean(union_support)) if np.any(union_support) else 1.0
    spearman_values = np.asarray([row["spearman"] for row in pairwise], dtype=np.float64)
    iou_values = np.asarray([row["hotspot_iou"] for row in pairwise], dtype=np.float64)
    measured_status = "not_supplied"
    measured_report_hash = None
    if measured_report_path is not None:
        measured = Path(measured_report_path).resolve()
        if not measured.is_file():
            raise PhysicalAccuracyBridgeError(f"measured report does not exist: {measured}")
        document = json.loads(measured.read_text(encoding="utf-8"))
        if not isinstance(document, Mapping):
            raise PhysicalAccuracyBridgeError("measured report must be a JSON object")
        measured_status = str(document.get("status", "invalid"))
        measured_report_hash = sha256_file(measured)
    result: dict[str, Any] = {
        "version": VERSION,
        "status": "provisional_physical_accuracy_bridge",
        "physical_spatial_accuracy_status": "blocked_without_independent_measured_truth",
        "spatial_accuracy_status": "blocked",
        "official_accuracy_gate": False,
        "official_reference_gate": False,
        "fem_gate": "blocked",
        "claim_scope": "model-consensus screening and epistemic disagreement only",
        "input_bindings": {
            "prior_report": str(prior_report),
            "prior_report_sha256": sha256_file(prior_report),
            "prior_array_artifact": str(prior_npz),
            "prior_array_artifact_sha256": sha256_file(prior_npz),
            "measured_report": None if measured_report_path is None else str(Path(measured_report_path).resolve()),
            "measured_report_sha256": measured_report_hash,
        },
        "upstream_particle_input": {
            "path": report.get("input_bindings", {}).get("particle_hits_path"),
            "sha256": report.get("input_bindings", {}).get("particle_hits_sha256"),
            "n_hit_impacts": report.get("input_bindings", {}).get("n_hit_impacts"),
            "n_faces": report.get("input_bindings", {}).get("n_faces"),
            "interpretation": "fixed common particle-hit driver shared by all four erosion laws",
        },
        "model_contract": {
            "models": model_list,
            "normalization": "each map divided by its facewise p95, then clipped at zero",
            "consensus": "geometric mean of normalized model maps",
            "disagreement": "coefficient of variation across normalized model maps",
            "hotspot_quantile": float(hotspot_quantile),
            "ai5_used_for_calibration": False,
        },
        "model_agreement": {
            "pairwise": pairwise,
            "spearman_median": float(np.median(spearman_values)),
            "spearman_min": float(np.min(spearman_values)),
            "hotspot_iou_median": float(np.median(iou_values)),
            "hotspot_iou_min": float(np.min(iou_values)),
            "interpretation": "agreement is not validation against physical wear",
        },
        "model_diversity_diagnostic": {
            "status": "low_spatial_diversity_shared_particle_driver",
            "raw_p50_summary": raw_summary,
            "positive_support_iou": support_iou,
            "shared_positive_support_fraction": shared_support_fraction,
            "interpretation": "near-identical rankings mainly show common hit support; they do not demonstrate law-level spatial accuracy",
        },
        "field_summary": {
            "n_faces": int(len(consensus)),
            "consensus_p05": float(np.quantile(consensus, 0.05)),
            "consensus_p50": float(np.quantile(consensus, 0.50)),
            "consensus_p95": float(np.quantile(consensus, 0.95)),
            "disagreement_cv_p50": float(np.quantile(disagreement_cv, 0.50)),
            "disagreement_cv_p95": float(np.quantile(disagreement_cv, 0.95)),
            "high_disagreement_face_fraction": float(np.mean(disagreement_cv >= 0.5)),
        },
        "measured_truth": {
            "status": measured_status,
            "required_for_physical_accuracy": True,
            "interpretation": "literature priors and model consensus cannot replace baseline/worn scans",
        },
        "limitations": [
            "The four erosion laws share the same particle-hit input and are not independent physical measurements.",
            "Normalization removes absolute-scale differences; the consensus field is a ranking aid, not calibrated depth.",
            "No baseline/worn scan, independent material batch, or official AI5 spatial truth is asserted.",
            "This report cannot open official_accuracy_gate, official_reference_gate, or fem_gate.",
        ],
    }
    result["report_sha256"] = canonical_sha256(result)
    if output_path is not None:
        target = Path(output_path).resolve()
        safe_json_dump(result, target)
        safe_npz_dump(
            {
                "consensus_normalized": consensus.astype(np.float32),
                "disagreement_cv": disagreement_cv.astype(np.float32),
                "rank_consensus": rank_consensus.astype(np.float32),
                **{f"{model}_normalized": normalized[model].astype(np.float32) for model in model_list},
            },
            target.with_suffix(".npz"),
            metadata={"report_sha256": result["report_sha256"], "version": VERSION},
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Step30 physical-accuracy bridge")
    parser.add_argument("--prior-report", type=Path, required=True)
    parser.add_argument("--measured-report", type=Path)
    parser.add_argument("--hotspot-quantile", type=float, default=DEFAULT_HOTSPOT_QUANTILE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_bridge(
        args.prior_report,
        measured_report_path=args.measured_report,
        hotspot_quantile=args.hotspot_quantile,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
