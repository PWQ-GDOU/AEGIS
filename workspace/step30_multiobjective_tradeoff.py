"""Deterministic, evidence-gated multiobjective trade-off analysis."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Mapping, Sequence


VERSION = "step30-multiobjective-tradeoff-v1"
_HEX = frozenset("0123456789abcdef")
_OBJECTIVE_KEYS = (
    "efficiency_max",
    "stress_pa_min",
    "fatigue_damage_min",
    "weighted_erosion_mm_h_min",
)


class TradeoffError(ValueError):
    """Raised for invalid objective or evidence contracts."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TradeoffError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TradeoffError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise TradeoffError(f"{name} must be finite")
    return result


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in _HEX for char in value.lower())
    ):
        raise TradeoffError(f"{name} must be a valid SHA-256")
    return value.lower()


def _rerun_evidence(
    candidate: Mapping[str, Any], identifier: str
) -> tuple[bool, list[str], dict[str, str]]:
    evidence = candidate.get("evidence")
    reasons: list[str] = []
    hashes: dict[str, str] = {}
    if not isinstance(evidence, Mapping):
        return False, ["evidence object is missing"], hashes
    if evidence.get("rerun_for_candidate") is not True:
        reasons.append("rerun_for_candidate is not true")
    checks = {
        "cfd": lambda item: item.get("status") == "passed"
        and item.get("real_cfd_evidence", item.get("real_solver")) is True,
        "fem": lambda item: item.get("status") in {"passed", "validated"}
        and item.get("fem_gate") == "passed"
        and item.get("real_fem_evidence") is True,
        "erosion": lambda item: item.get("status") in {"passed", "validated", "measured"}
        and item.get("real_rerun") is True
        and item.get("independent_validation_passed") is True,
    }
    for domain, predicate in checks.items():
        item = evidence.get(domain)
        if not isinstance(item, Mapping):
            reasons.append(f"{domain} rerun evidence is missing")
            continue
        if str(item.get("candidate_id", "")) != identifier:
            reasons.append(f"{domain} evidence is not bound to candidate_id")
        if item.get("rerun_for_candidate") is not True:
            reasons.append(f"{domain} is not a per-candidate rerun")
        if not predicate(item):
            reasons.append(f"{domain} real rerun gate has not passed")
        try:
            hashes[domain] = _sha256(
                item.get("artifact_sha256"), f"{domain} artifact_sha256"
            )
        except TradeoffError as exc:
            reasons.append(str(exc))
    return not reasons, reasons, hashes


def _objectives(
    candidate: Mapping[str, Any], weights: Mapping[str, Any]
) -> tuple[float, float, float, float]:
    metrics = candidate.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TradeoffError("candidate metrics object is required")
    units = metrics.get("units")
    expected_units = {
        "efficiency": "1",
        "stress": "Pa",
        "fatigue_damage": "1",
        "erosion_rate": "mm/h",
    }
    if not isinstance(units, Mapping) or any(
        units.get(key) != value for key, value in expected_units.items()
    ):
        raise TradeoffError(
            "metric units must be efficiency=1, stress=Pa, fatigue_damage=1, erosion_rate=mm/h"
        )
    efficiency = _number(metrics.get("efficiency"), "efficiency")
    stress = _number(
        metrics.get("stress_pa", metrics.get("max_stress_pa")), "stress_pa"
    )
    fatigue = _number(metrics.get("fatigue_damage"), "fatigue_damage")
    if not 0.0 <= efficiency <= 1.0:
        raise TradeoffError("efficiency must lie in [0, 1]")
    if stress < 0.0 or fatigue < 0.0:
        raise TradeoffError("stress and fatigue damage must be non-negative")
    erosion_by_condition = metrics.get("erosion_by_condition")
    if not isinstance(erosion_by_condition, Mapping) or not erosion_by_condition:
        raise TradeoffError("erosion_by_condition is required")
    weighted_erosion = 0.0
    total_weight = 0.0
    for condition in sorted(erosion_by_condition, key=str):
        row = erosion_by_condition[condition]
        if isinstance(row, Mapping):
            value = _number(
                row.get("rate_mm_h", row.get("erosion_rate_mm_h")),
                f"erosion {condition}",
            )
            weight = _number(
                row.get("weight", weights.get(str(condition), 1.0)),
                f"weight {condition}",
            )
        else:
            value = _number(row, f"erosion {condition}")
            weight = _number(weights.get(str(condition), 1.0), f"weight {condition}")
        if value < 0.0 or weight < 0.0:
            raise TradeoffError("erosion values and weights must be non-negative")
        weighted_erosion += weight * value
        total_weight += weight
    if total_weight <= 0.0:
        raise TradeoffError("erosion weights must have a positive total")
    return efficiency, stress, fatigue, weighted_erosion / total_weight


def _dominates(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    tolerance: float,
) -> bool:
    no_worse = (
        left[0] >= right[0] - tolerance
        and left[1] <= right[1] + tolerance
        and left[2] <= right[2] + tolerance
        and left[3] <= right[3] + tolerance
    )
    strictly_better = (
        left[0] > right[0] + tolerance
        or left[1] < right[1] - tolerance
        or left[2] < right[2] - tolerance
        or left[3] < right[3] - tolerance
    )
    return no_worse and strictly_better


def build_pareto_tradeoff(
    candidates: Sequence[Mapping[str, Any]],
    *,
    condition_weights: Mapping[str, Any] | None = None,
    tolerance: float = 1.0e-12,
    require_real_reruns: bool = True,
) -> dict[str, Any]:
    """Filter dominated candidates; incomplete per-candidate reruns block the gate."""
    try:
        if not candidates:
            raise TradeoffError("at least one candidate is required")
        if not isinstance(require_real_reruns, bool):
            raise TradeoffError("require_real_reruns must be boolean")
        weights = {
            str(key): _number(value, f"weight {key}")
            for key, value in sorted(
                dict(condition_weights or {}).items(), key=lambda item: str(item[0])
            )
        }
        if any(value < 0.0 for value in weights.values()):
            raise TradeoffError("condition weights must be non-negative")
        tol = _number(tolerance, "tolerance")
        if tol < 0.0:
            raise TradeoffError("tolerance must be non-negative")

        rows: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise TradeoffError("each candidate must be an object")
            identifier = str(candidate.get("candidate_id", "")).strip()
            if not identifier:
                raise TradeoffError("candidate_id is required")
            if identifier in seen_ids:
                raise TradeoffError("candidate_id values must be unique")
            seen_ids.add(identifier)
            objectives = _objectives(candidate, weights)
            complete, reasons, hashes = _rerun_evidence(candidate, identifier)
            rows.append(
                {
                    "candidate_id": identifier,
                    "objectives": dict(zip(_OBJECTIVE_KEYS, objectives)),
                    "evidence_complete": complete,
                    "evidence_reasons": sorted(reasons),
                    "artifact_hashes": hashes,
                }
            )
        rows.sort(key=lambda row: row["candidate_id"])

        # Reusing one domain artifact for different candidates is not a rerun.
        for domain in ("cfd", "fem", "erosion"):
            counts = Counter(
                row["artifact_hashes"].get(domain)
                for row in rows
                if domain in row["artifact_hashes"]
            )
            duplicate_hashes = {value for value, count in counts.items() if count > 1}
            for row in rows:
                if row["artifact_hashes"].get(domain) in duplicate_hashes:
                    row["evidence_complete"] = False
                    row["evidence_reasons"].append(
                        f"{domain} artifact is reused by another candidate"
                    )
                    row["evidence_reasons"].sort()

        nondominated: list[dict[str, Any]] = []
        dominated_by: dict[str, list[str]] = {
            row["candidate_id"]: [] for row in rows
        }
        for row in rows:
            current = tuple(row["objectives"][key] for key in _OBJECTIVE_KEYS)
            dominators: list[str] = []
            for other in rows:
                if other is row:
                    continue
                other_objectives = tuple(
                    other["objectives"][key] for key in _OBJECTIVE_KEYS
                )
                if _dominates(other_objectives, current, tol):
                    dominators.append(other["candidate_id"])
            if dominators:
                dominated_by[row["candidate_id"]] = sorted(dominators)
            else:
                nondominated.append(row)
        nondominated.sort(
            key=lambda row: (
                -row["objectives"]["efficiency_max"],
                row["objectives"]["stress_pa_min"],
                row["objectives"]["fatigue_damage_min"],
                row["objectives"]["weighted_erosion_mm_h_min"],
                row["candidate_id"],
            )
        )
        incomplete = [
            row["candidate_id"] for row in rows if not row["evidence_complete"]
        ]
        # This switch is retained for API compatibility only. Evidence remains
        # mandatory; disabling it marks an exploratory frontier, never a pass.
        blocked = bool(incomplete or not require_real_reruns)
        report = {
            "version": VERSION,
            "status": "blocked" if blocked else "passed",
            "tradeoff_gate": "blocked" if blocked else "passed",
            "reason": (
                "every candidate requires distinct real CFD, FEM, and independently validated erosion reruns"
                if incomplete
                else "real rerun enforcement cannot be disabled for a final tradeoff"
                if not require_real_reruns
                else None
            ),
            "objective_directions": {
                "efficiency": "maximize",
                "stress_pa": "minimize",
                "fatigue_damage": "minimize",
                "weighted_erosion_mm_h": "minimize",
            },
            "condition_weights": weights,
            "n_candidates": len(rows),
            "nondominated_candidate_ids": [
                row["candidate_id"] for row in nondominated
            ],
            "dominated_by": dominated_by,
            "incomplete_candidate_evidence": incomplete,
            "frontier": nondominated,
            "software_validation_only": blocked,
        }
        report["tradeoff_sha256"] = _canonical_sha256(report)
        return report
    except (TradeoffError, TypeError, ValueError) as exc:
        return {
            "version": VERSION,
            "status": "blocked",
            "tradeoff_gate": "blocked",
            "software_validation_only": True,
            "reason": str(exc),
        }


def validate_tradeoff(report: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("tradeoff_gate") not in {"passed", "blocked"}:
        raise TradeoffError("tradeoff_gate is invalid")
    if report.get("status") != report.get("tradeoff_gate"):
        raise TradeoffError("status and tradeoff_gate differ")
    stored = _sha256(report.get("tradeoff_sha256"), "tradeoff_sha256")
    unsigned = dict(report)
    unsigned.pop("tradeoff_sha256", None)
    if _canonical_sha256(unsigned) != stored:
        raise TradeoffError("tradeoff SHA-256/content mismatch")
    if report.get("tradeoff_gate") == "passed":
        if report.get("incomplete_candidate_evidence"):
            raise TradeoffError("a passed tradeoff contains incomplete evidence")
        if report.get("software_validation_only") is not False:
            raise TradeoffError("a passed tradeoff is marked software-only")
    return dict(report)


__all__ = [
    "TradeoffError",
    "VERSION",
    "build_pareto_tradeoff",
    "validate_tradeoff",
]
