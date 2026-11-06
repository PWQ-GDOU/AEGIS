"""v4 convergence evaluation for raw and topology-local particle risk fields."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from step30_particle_spatial_estimator_v4 import (
    KERNEL_CONTRACT,
    ParticleSpatialEstimatorError,
    estimate_immutable_particle_run,
    save_estimate,
)
from step30_validation_common import canonical_sha256, safe_json_dump, sha256_file


VERSION = "step30-particle-convergence-v4"
MIN_SEEDS = 3
GATES = {
    "weighted_spearman_median_min": 0.90,
    "weighted_spearman_p10_min": 0.80,
    "mass_hotspot_iou_median_min": 0.80,
    "spatial_js_divergence_max": 0.05,
    "scale_relative_mass_change_max": 0.05,
}


class ParticleConvergenceV4Error(ValueError):
    """Raised for invalid comparisons between immutable particle runs."""


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        ranks = np.bincount(inverse, weights=ranks)[inverse] / counts[inverse]
    return ranks


def weighted_spearman(
    left: np.ndarray, right: np.ndarray, face_area_m2: np.ndarray
) -> float:
    """Area-weighted rank correlation over faces active in either field."""
    mask = (left > 0.0) | (right > 0.0)
    if int(mask.sum()) < 2:
        return 1.0 if np.array_equal(left, right) else 0.0
    a, b = _rank(left[mask]), _rank(right[mask])
    weights = face_area_m2[mask]
    weights = weights / max(float(weights.sum()), np.finfo(float).tiny)
    mean_a, mean_b = float(np.sum(weights * a)), float(np.sum(weights * b))
    covariance = float(np.sum(weights * (a - mean_a) * (b - mean_b)))
    scale = math.sqrt(
        float(np.sum(weights * (a - mean_a) ** 2))
        * float(np.sum(weights * (b - mean_b) ** 2))
    )
    return 1.0 if scale == 0.0 and np.array_equal(a, b) else covariance / max(scale, 1.0e-30)


def mass_hotspot_iou(
    left: np.ndarray, right: np.ndarray, *, mass_fraction: float = 0.10
) -> float:
    """IoU of the smallest descending-mass face sets containing a fixed mass."""
    if not 0.0 < mass_fraction <= 1.0:
        raise ParticleConvergenceV4Error("mass_fraction must be in (0, 1]")

    def select(field: np.ndarray) -> set[int]:
        total = float(field.sum())
        if total <= 0.0:
            return set()
        selected: set[int] = set()
        cumulative = 0.0
        for index in np.argsort(field, kind="mergesort")[::-1]:
            if field[index] <= 0.0:
                break
            selected.add(int(index))
            cumulative += float(field[index])
            if cumulative >= mass_fraction * total:
                break
        return selected

    first, second = select(left), select(right)
    if not first and not second:
        return 1.0
    return float(len(first & second) / len(first | second))


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left_sum, right_sum = float(left.sum()), float(right.sum())
    if left_sum <= 0.0 or right_sum <= 0.0:
        raise ParticleConvergenceV4Error("JS divergence requires positive total mass")
    p, q = left / left_sum, right / right_sum
    middle = 0.5 * (p + q)

    def kl(value: np.ndarray) -> float:
        mask = value > 0.0
        return float(np.sum(value[mask] * np.log2(value[mask] / middle[mask])))

    return 0.5 * (kl(p) + kl(q))


def _pair_metrics(
    left: Mapping[str, Any], right: Mapping[str, Any], field_name: str
) -> dict[str, float]:
    left_field = np.asarray(left["field"][field_name], dtype=np.float64)
    right_field = np.asarray(right["field"][field_name], dtype=np.float64)
    area = np.asarray(left["topology"]["face_area_m2"], dtype=np.float64)
    return {
        "area_weighted_spearman": float(weighted_spearman(left_field, right_field, area)),
        "mass_hotspot_iou": float(mass_hotspot_iou(left_field, right_field)),
        "js_divergence": float(js_divergence(left_field, right_field)),
    }


def _load_run(
    run_directory: str | Path,
    topology_path: str | Path,
    *,
    bandwidth_hops: int,
) -> dict[str, Any]:
    directory = Path(run_directory).resolve()
    hits = directory / "particle_hits.npz"
    if not hits.is_file():
        raise ParticleConvergenceV4Error(f"run has no particle_hits.npz: {directory}")
    value = estimate_immutable_particle_run(
        hits, topology_path, bandwidth_hops=bandwidth_hops
    )
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    parameters = summary.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise ParticleConvergenceV4Error("run summary has no parameter object")
    value["run_directory"] = str(directory)
    value["n_parcels"] = int(parameters.get("n_parcels", -1))
    value["seed"] = int(parameters.get("seed", -1))
    if value["n_parcels"] <= 0 or value["seed"] < 0:
        raise ParticleConvergenceV4Error("run has invalid n_parcels or seed")
    return value


def _identity(estimate: Mapping[str, Any]) -> dict[str, Any]:
    binding = estimate["metadata"]["input_bindings"]
    return {
        "artifact_set_id": binding["artifact_set_id"],
        "mesh_signature": binding["mesh_signature"],
        "topology_sha256": binding["topology_sha256"],
        "kernel_contract_sha256": estimate["metadata"]["kernel_contract_sha256"],
    }


def evaluate_runs_v4(
    run_directories: Sequence[str | Path],
    topology_path: str | Path,
    *,
    bandwidth_hops: int = int(KERNEL_CONTRACT["default_bandwidth_hops"]),
    fields_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate raw and v4 kernel fields without altering v3 history."""
    try:
        if not run_directories:
            raise ParticleConvergenceV4Error("at least one immutable run is required")
        estimates = [
            _load_run(path, topology_path, bandwidth_hops=bandwidth_hops)
            for path in run_directories
        ]
        reference = _identity(estimates[0])
        if any(_identity(item) != reference for item in estimates[1:]):
            raise ParticleConvergenceV4Error("runs do not share topology/kernel identity")
        keys = {(item["n_parcels"], item["seed"]) for item in estimates}
        if len(keys) != len(estimates):
            raise ParticleConvergenceV4Error("duplicate n_parcels/seed run identity")
        if fields_directory is not None:
            root = Path(fields_directory).resolve()
            root.mkdir(parents=True, exist_ok=True)
            for item in estimates:
                output = root / (
                    "particle_surface_kernel_v4_"
                    f"{item['metadata']['input_bindings']['particle_run_id']}.npz"
                )
                item["saved_field"] = save_estimate(item, output)
        groups: dict[int, list[dict[str, Any]]] = {}
        for item in estimates:
            groups.setdefault(item["n_parcels"], []).append(item)
        reports: dict[str, Any] = {}
        estimator_names = {
            "raw": "raw_face_target_mass_kg",
            "surface_kernel": "surface_kernel_face_target_mass_kg",
        }
        all_failures: list[str] = []
        for scale, rows in sorted(groups.items()):
            rows = sorted(rows, key=lambda item: item["seed"])
            by_estimator: dict[str, Any] = {}
            for name, field_name in estimator_names.items():
                pairwise = [
                    _pair_metrics(left, right, field_name)
                    for offset, left in enumerate(rows)
                    for right in rows[offset + 1 :]
                ]
                spearman = [item["area_weighted_spearman"] for item in pairwise]
                iou = [item["mass_hotspot_iou"] for item in pairwise]
                js = [item["js_divergence"] for item in pairwise]
                complete = len(rows) >= MIN_SEEDS
                gates = {
                    "minimum_seeds": complete,
                    "area_weighted_spearman_median": complete
                    and float(np.median(spearman)) >= GATES["weighted_spearman_median_min"],
                    "area_weighted_spearman_p10": complete
                    and float(np.percentile(spearman, 10)) >= GATES["weighted_spearman_p10_min"],
                    "mass_hotspot_iou_median": complete
                    and float(np.median(iou)) >= GATES["mass_hotspot_iou_median_min"],
                    "spatial_js_divergence_max": complete
                    and max(js, default=float("inf")) <= GATES["spatial_js_divergence_max"],
                }
                by_estimator[name] = {
                    "pairwise": pairwise,
                    "area_weighted_spearman_median": None if not spearman else float(np.median(spearman)),
                    "area_weighted_spearman_p10": None if not spearman else float(np.percentile(spearman, 10)),
                    "mass_hotspot_iou_median": None if not iou else float(np.median(iou)),
                    "spatial_js_divergence_max": None if not js else float(max(js)),
                    "gates": gates,
                    "passed": all(gates.values()),
                }
                all_failures.extend(
                    f"n={scale}:{name}:{gate}"
                    for gate, passed in gates.items()
                    if gate != "minimum_seeds" and not passed
                )
            reports[str(scale)] = {
                "seeds": [item["seed"] for item in rows],
                "n_runs": len(rows),
                "estimator_comparison": by_estimator,
                "run_fields": [
                    {
                        "particle_run_id": item["metadata"]["input_bindings"]["particle_run_id"],
                        "particle_hits_sha256": item["metadata"]["input_bindings"]["particle_hits_sha256"],
                        "field_contract_sha256": item["metadata"]["field_contract_sha256"],
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
                        "field_report_sha256": (
                            item["saved_field"]["report_sha256"]
                            if "saved_field" in item
                            else None
                        ),
                    }
                    for item in rows
                ],
            }
        scale_comparisons: list[dict[str, Any]] = []
        scales = sorted(groups)
        for left_scale, right_scale in zip(scales, scales[1:]):
            left_by_seed = {item["seed"]: item for item in groups[left_scale]}
            right_by_seed = {item["seed"]: item for item in groups[right_scale]}
            common = sorted(set(left_by_seed) & set(right_by_seed))
            if len(common) < MIN_SEEDS:
                continue
            estimators: dict[str, Any] = {}
            for name, field_name in estimator_names.items():
                changes, divergences = [], []
                for seed in common:
                    left = np.asarray(left_by_seed[seed]["field"][field_name], dtype=np.float64)
                    right = np.asarray(right_by_seed[seed]["field"][field_name], dtype=np.float64)
                    changes.append(abs(left.sum() - right.sum()) / max(left.sum(), right.sum(), 1.0e-30))
                    divergences.append(js_divergence(left, right))
                gates = {
                    "scale_relative_mass_change": bool(
                        max(changes) <= GATES["scale_relative_mass_change_max"]
                    ),
                    "spatial_js_divergence": bool(
                        max(divergences) <= GATES["spatial_js_divergence_max"]
                    ),
                }
                estimators[name] = {
                    "max_relative_mass_change": float(max(changes)),
                    "max_js_divergence": float(max(divergences)),
                    "gates": gates,
                    "passed": all(gates.values()),
                }
                all_failures.extend(
                    f"scale={left_scale}->{right_scale}:{name}:{gate}"
                    for gate, passed in gates.items()
                    if not passed
                )
            scale_comparisons.append(
                {
                    "scales": [left_scale, right_scale],
                    "seeds": common,
                    "estimator_comparison": estimators,
                }
            )
        decision_scale = scales[-1]
        kernel_passed = bool(
            reports[str(decision_scale)]["estimator_comparison"]["surface_kernel"]["passed"]
        )
        if scale_comparisons:
            kernel_passed = kernel_passed and bool(
                scale_comparisons[-1]["estimator_comparison"]["surface_kernel"]["passed"]
            )
        status = "numerically_stable_risk_map" if kernel_passed else "failed"
        result = {
            "version": VERSION,
            "status": status,
            "numerical_stability_status": status,
            "spatial_accuracy_status": "blocked",
            "official_accuracy_gate": False,
            "official_reference_gate": False,
            "historical_v3": {
                "path": str(
                    Path(__file__).resolve().parent
                    / "output"
                    / "step30_particle_convergence_5k_10k_20k_mass_stratified_v3.json"
                ),
                "status": "failed",
                "preserved_as": "historical_spatial_instability_record",
            },
            "kernel_contract": KERNEL_CONTRACT,
            "kernel_contract_sha256": canonical_sha256(KERNEL_CONTRACT),
            "identity": reference,
            "gates": GATES,
            "scales": reports,
            "scale_comparisons": scale_comparisons,
            "decision_scale": decision_scale,
            "selected_estimator": "surface_kernel",
            "selected_estimator_interpretation": (
                "numerically stable risk map only when its v4 numerical gates pass; "
                "not a spatial-accuracy claim"
            ),
            "failures": sorted(set(all_failures)),
            "limitations": [
                "Raw and surface-kernel fields are compared on immutable particle simulations.",
                "The v4 kernel does not create worn/baseline truth or independent validation.",
                "No current v4 result opens official accuracy, official reference, or FEM gates.",
            ],
        }
        result["report_sha256"] = canonical_sha256(result)
        return result
    except (
        OSError,
        ValueError,
        KeyError,
        ParticleSpatialEstimatorError,
        ParticleConvergenceV4Error,
    ) as exc:
        result = {
            "version": VERSION,
            "status": "blocked",
            "numerical_stability_status": "blocked",
            "spatial_accuracy_status": "blocked",
            "reason": str(exc),
            "official_accuracy_gate": False,
            "official_reference_gate": False,
        }
        result["report_sha256"] = canonical_sha256(result)
        return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare v4 raw and topology-local particle fields")
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fields-directory", type=Path)
    parser.add_argument("--bandwidth-hops", type=int, default=2)
    args = parser.parse_args(argv)
    report = evaluate_runs_v4(
        args.run,
        args.topology,
        bandwidth_hops=args.bandwidth_hops,
        fields_directory=args.fields_directory,
    )
    safe_json_dump(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "numerically_stable_risk_map" else 3


if __name__ == "__main__":
    raise SystemExit(main())
