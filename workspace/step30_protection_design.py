"""Evidence-gated erosion/structural protection zoning.

The array classifier is explicitly software-validation-only.  Production
protection design is only released after independent measured erosion and a
hash-bound, three-mesh, independently validated real FEM report both pass.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from typing import Any, Mapping

import numpy as np


VERSION = "step30-protection-design-v1"
_HEX = frozenset("0123456789abcdef")
_NON_REAL = ("synthetic", "parametric", "analytical", "cfd", "simulation", "generated", "fake")


class ProtectionDesignError(ValueError):
    """Raised for malformed protection evidence or geometric constraints."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in _HEX for char in value.lower())
    ):
        raise ProtectionDesignError(f"{name} must be a valid SHA-256")
    return value.lower()


def _content_hash(document: Mapping[str, Any], field: str) -> str:
    stored = _sha256(document.get(field), field)
    unsigned = dict(document)
    unsigned.pop(field, None)
    if _canonical_sha256(unsigned) != stored:
        raise ProtectionDesignError(f"{field}/content mismatch")
    return stored


def _array(value: Any, name: str, n: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1 or (n is not None and result.shape != (n,)):
        raise ProtectionDesignError(f"{name} must be a one-dimensional array")
    if result.size == 0:
        raise ProtectionDesignError(f"{name} must not be empty")
    return result


def _float_array(value: Any, name: str, n: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (n,) or not np.isfinite(result).all():
        raise ProtectionDesignError(f"{name} must be finite with length {n}")
    return result


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ProtectionDesignError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtectionDesignError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ProtectionDesignError(f"{name} must be {qualifier}")
    return result


def _build_adjacency(adjacency: Any, n: int) -> list[set[int]]:
    if adjacency is None:
        raise ProtectionDesignError("face adjacency is required to enforce connectivity")
    result = [set() for _ in range(n)]
    if isinstance(adjacency, Mapping):
        items = adjacency.items()
    else:
        rows = list(adjacency)
        if len(rows) != n:
            raise ProtectionDesignError("face adjacency must have one row per face")
        items = enumerate(rows)
    for raw_index, raw_neighbors in items:
        index = int(raw_index)
        neighbors = np.asarray(list(raw_neighbors), dtype=np.int64).reshape(-1)
        if index < 0 or index >= n or np.any(neighbors < 0) or np.any(neighbors >= n):
            raise ProtectionDesignError("face adjacency index is out of range")
        result[index].update(int(value) for value in neighbors if int(value) != index)
    for index in range(n):
        for neighbor in tuple(result[index]):
            result[neighbor].add(index)
    return result


def _components(mask: np.ndarray, adjacency: list[set[int]]) -> list[np.ndarray]:
    seen = np.zeros(len(mask), dtype=bool)
    output: list[np.ndarray] = []
    for start in np.flatnonzero(mask):
        if seen[start]:
            continue
        queue: deque[int] = deque([int(start)])
        seen[start] = True
        members: list[int] = []
        while queue:
            current = queue.popleft()
            members.append(current)
            for neighbor in sorted(adjacency[current]):
                if mask[neighbor] and not seen[neighbor]:
                    seen[neighbor] = True
                    queue.append(neighbor)
        output.append(np.asarray(sorted(members), dtype=np.int64))
    return output


def _validate_quantile(value: Any, name: str) -> float:
    result = _number(value, name)
    if not 0.0 <= result <= 1.0:
        raise ProtectionDesignError(f"{name} must lie in [0, 1]")
    return result


def _validated_real_erosion(evidence: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    if evidence.get("status") != "measured":
        raise ProtectionDesignError("independent measured erosion evidence is required")
    if evidence.get("evaluation_status") not in {"truth_built", "validated", "passed", "measured"}:
        raise ProtectionDesignError("independent erosion evidence has not passed")
    if evidence.get("independent") is not True and evidence.get("independent_erosion") is not True:
        raise ProtectionDesignError("erosion evidence is not declared independent")
    origin = " ".join(
        str(evidence.get(key, "")).lower()
        for key in ("source_type", "evidence_status", "origin", "data_origin")
    )
    if any(token in origin for token in _NON_REAL):
        raise ProtectionDesignError("erosion evidence is simulated or generated")
    evidence_hash = _content_hash(evidence, "evidence_sha256")
    if evidence.get("erosion_rate_units") != "mm/h":
        raise ProtectionDesignError("erosion rate units must be mm/h")
    parent_ids = _array(evidence.get("parent_face_id"), "erosion parent_face_id").astype(np.int64)
    rates = _float_array(evidence.get("erosion_rate_mm_h"), "erosion_rate_mm_h", len(parent_ids))
    if len(np.unique(parent_ids)) != len(parent_ids) or np.any(parent_ids < 0):
        raise ProtectionDesignError("erosion parent-face IDs must be unique and non-negative")
    if np.any(rates < 0.0):
        raise ProtectionDesignError("erosion rates must be non-negative")
    return parent_ids, rates, evidence_hash


def _validated_real_fem(evidence: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    if (
        evidence.get("status") != "passed"
        or evidence.get("fem_gate") != "passed"
        or evidence.get("real_fem_evidence") is not True
    ):
        raise ProtectionDesignError("real FEM validation evidence is required")
    validation_hash = _content_hash(evidence, "validation_sha256")
    surface = evidence.get("surface_results")
    if not isinstance(surface, Mapping):
        raise ProtectionDesignError("real FEM surface results are missing")
    _sha256(surface.get("topology_sha256"), "FEM surface topology_sha256")
    _sha256(evidence.get("fine_mesh_sha256"), "fine_mesh_sha256")
    _sha256(evidence.get("fine_result_output_sha256"), "fine_result_output_sha256")
    return dict(surface), validation_hash


def classify_protection_zones(
    *,
    parent_face_id: Any,
    erosion_rate_mm_h: Any,
    von_mises_stress_pa: Any,
    fatigue_damage: Any,
    face_area_m2: Any,
    adjacency: Any,
    root_forbidden_mask: Any,
    face_centers_m: Any | None = None,
    high_erosion_quantile: float = 0.75,
    high_stress_quantile: float = 0.75,
    low_stress_quantile: float = 0.50,
    minimum_feature_m: float = 0.0,
    minimum_transition_width_m: float = 0.0,
    maximum_area_fraction: float = 1.0,
    maximum_mass_kg: float | None = None,
    coating_density_kg_m3: float = 7800.0,
    coating_thickness_m: float = 0.001,
    require_connected: bool = True,
) -> dict[str, Any]:
    """Exercise zoning/geometry algorithms without creating engineering evidence."""
    try:
        ids = _array(parent_face_id, "parent_face_id").astype(np.int64)
        n = len(ids)
        if len(np.unique(ids)) != n or np.any(ids < 0):
            raise ProtectionDesignError("parent_face_id values must be unique and non-negative")
        erosion = _float_array(erosion_rate_mm_h, "erosion_rate_mm_h", n)
        stress = _float_array(von_mises_stress_pa, "von_mises_stress_pa", n)
        fatigue = _float_array(fatigue_damage, "fatigue_damage", n)
        area = _float_array(face_area_m2, "face_area_m2", n)
        if np.any(erosion < 0.0) or np.any(stress < 0.0) or np.any(fatigue < 0.0):
            raise ProtectionDesignError("erosion, stress, and fatigue must be non-negative")
        if np.any(area <= 0.0):
            raise ProtectionDesignError("face areas must be positive")
        root = np.asarray(root_forbidden_mask, dtype=bool)
        if root.shape != (n,):
            raise ProtectionDesignError("root_forbidden_mask must have one entry per face")
        graph = _build_adjacency(adjacency, n)
        centers: np.ndarray | None = None
        if face_centers_m is not None:
            centers = np.asarray(face_centers_m, dtype=np.float64)
            if centers.shape != (n, 3) or not np.isfinite(centers).all():
                raise ProtectionDesignError("face_centers_m must have shape (n, 3)")

        qe = _validate_quantile(high_erosion_quantile, "high_erosion_quantile")
        qh = _validate_quantile(high_stress_quantile, "high_stress_quantile")
        ql = _validate_quantile(low_stress_quantile, "low_stress_quantile")
        erosion_cut = float(np.quantile(erosion, qe))
        stress_high_cut = float(np.quantile(stress, qh))
        stress_low_cut = float(np.quantile(stress, ql))
        fatigue_high_cut = float(np.quantile(fatigue, qh))
        fatigue_low_cut = float(np.quantile(fatigue, ql))

        high_erosion = erosion >= erosion_cut
        high_structure = (stress >= stress_high_cut) | (fatigue >= fatigue_high_cut)
        low_structure = (stress <= stress_low_cut) & (fatigue <= fatigue_low_cut)
        coupled = high_erosion & high_structure & ~root
        erosion_low_stress = high_erosion & low_structure & ~root
        review = high_erosion & ~(coupled | erosion_low_stress | root)
        core = coupled | erosion_low_stress
        core_components = _components(core, graph)
        if require_connected and len(core_components) > 1:
            raise ProtectionDesignError("protective core is disconnected")

        transition = np.zeros(n, dtype=bool)
        for index in np.flatnonzero(core):
            for neighbor in graph[int(index)]:
                if not core[neighbor] and not root[neighbor]:
                    transition[neighbor] = True
        application = core | transition
        if np.any(application & root):
            raise ProtectionDesignError("protection intersects the root-forbidden region")

        min_feature = _number(minimum_feature_m, "minimum_feature_m")
        min_transition = _number(
            minimum_transition_width_m, "minimum_transition_width_m"
        )
        if min_feature < 0.0 or min_transition < 0.0:
            raise ProtectionDesignError("minimum dimensions must be non-negative")
        feature_sizes = [
            2.0 * math.sqrt(float(np.sum(area[component])) / math.pi)
            for component in core_components
        ]
        if any(size < min_feature for size in feature_sizes):
            raise ProtectionDesignError("minimum protective feature constraint failed")
        if min_transition > 0.0 and centers is None:
            raise ProtectionDesignError(
                "face centers are required for a transition-width constraint"
            )
        transition_widths: list[float] = []
        for component in core_components:
            if centers is None:
                width = 0.0
            else:
                component_transition = np.asarray(
                    sorted(
                        {
                            neighbor
                            for index in component
                            for neighbor in graph[int(index)]
                            if transition[neighbor]
                        }
                    ),
                    dtype=np.int64,
                )
                outside = np.flatnonzero(~application)
                if len(component_transition) == 0:
                    width = 0.0
                elif len(outside) == 0:
                    width = float("inf")
                else:
                    distances = np.linalg.norm(
                        centers[component_transition, None, :]
                        - centers[outside][None, :, :],
                        axis=2,
                    )
                    width = float(np.min(distances))
            transition_widths.append(width)
        if any(width < min_transition for width in transition_widths):
            raise ProtectionDesignError("protective transition-width constraint failed")

        total_area = float(np.sum(area))
        application_area = float(np.sum(area[application]))
        area_fraction = application_area / total_area
        area_limit = _number(maximum_area_fraction, "maximum_area_fraction", positive=True)
        if area_limit > 1.0:
            raise ProtectionDesignError("maximum_area_fraction must not exceed 1")
        if area_fraction > area_limit + 1.0e-12:
            raise ProtectionDesignError("protection area budget exceeded")
        thickness = _number(coating_thickness_m, "coating_thickness_m", positive=True)
        density = _number(coating_density_kg_m3, "coating_density_kg_m3", positive=True)
        mass = application_area * thickness * density
        mass_limit = None
        if maximum_mass_kg is not None:
            mass_limit = _number(maximum_mass_kg, "maximum_mass_kg", positive=True)
            if mass > mass_limit + 1.0e-12:
                raise ProtectionDesignError("protection mass budget exceeded")

        labels = np.full(n, "unprotected", dtype=object)
        labels[review] = "high_erosion_structural_review"
        labels[transition] = "transition"
        labels[erosion_low_stress] = "high_erosion_low_stress"
        labels[coupled] = "coupled_high_erosion_high_structural_demand"
        labels[root] = "root_forbidden"
        return {
            "version": VERSION,
            "status": "software_validated",
            "protection_gate": "not_applicable",
            "software_validation_only": True,
            "real_fem_evidence": False,
            "independent_erosion_evidence": False,
            "parent_face_id": ids.tolist(),
            "zone_labels": labels.tolist(),
            "core_protected_mask": core.tolist(),
            "transition_mask": transition.tolist(),
            "protected_mask": application.tolist(),
            "root_forbidden_mask": root.tolist(),
            "thresholds": {
                "erosion_rate_mm_h": erosion_cut,
                "stress_pa_high": stress_high_cut,
                "stress_pa_low": stress_low_cut,
                "fatigue_high": fatigue_high_cut,
                "fatigue_low": fatigue_low_cut,
            },
            "components": [
                {
                    "face_ids": ids[component].tolist(),
                    "feature_size_m": feature_sizes[index],
                    "transition_width_m": transition_widths[index],
                }
                for index, component in enumerate(core_components)
            ],
            "budgets": {
                "total_area_m2": total_area,
                "protected_area_m2": application_area,
                "area_fraction": area_fraction,
                "maximum_area_fraction": area_limit,
                "coating_thickness_m": thickness,
                "coating_density_kg_m3": density,
                "mass_kg": mass,
                "maximum_mass_kg": mass_limit,
            },
            "constraints": {
                "connected": len(core_components) <= 1,
                "minimum_feature_m": min_feature,
                "minimum_transition_width_m": min_transition,
                "root_forbidden_faces": int(np.sum(root)),
            },
        }
    except (ProtectionDesignError, TypeError, ValueError) as exc:
        return {
            "version": VERSION,
            "status": "blocked",
            "protection_gate": "blocked",
            "software_validation_only": True,
            "real_fem_evidence": False,
            "independent_erosion_evidence": False,
            "reason": str(exc),
        }


def design_protection(
    erosion_evidence: Mapping[str, Any],
    fem_validation: Mapping[str, Any],
    *,
    parent_face_id: Any | None = None,
    erosion_rate_mm_h: Any | None = None,
    von_mises_stress_pa: Any | None = None,
    fatigue_damage: Any | None = None,
    face_area_m2: Any | None = None,
    face_centers_m: Any | None = None,
    adjacency: Any | None = None,
    root_forbidden_mask: Any | None = None,
    high_erosion_quantile: float = 0.75,
    high_stress_quantile: float = 0.75,
    low_stress_quantile: float = 0.50,
    minimum_feature_m: float = 0.0,
    minimum_transition_width_m: float = 0.0,
    maximum_area_fraction: float = 1.0,
    maximum_mass_kg: float | None = None,
    coating_density_kg_m3: float = 7800.0,
    coating_thickness_m: float = 0.001,
    require_connected: bool = True,
) -> dict[str, Any]:
    """Create a production design only from independent erosion and real FEM."""
    try:
        erosion_ids, evidence_rates, erosion_hash = _validated_real_erosion(
            erosion_evidence
        )
        fem_surface, fem_hash = _validated_real_fem(fem_validation)
        fem_ids = _array(
            fem_surface.get("parent_face_id"), "FEM parent_face_id"
        ).astype(np.int64)
        if len(np.unique(fem_ids)) != len(fem_ids) or np.any(fem_ids < 0):
            raise ProtectionDesignError("FEM parent-face IDs must be unique and non-negative")
        if set(erosion_ids.tolist()) != set(fem_ids.tolist()):
            raise ProtectionDesignError("erosion and FEM parent-face coverage differs")
        if erosion_evidence.get("topology_sha256") != fem_surface.get("topology_sha256"):
            raise ProtectionDesignError("erosion and FEM surface topology differs")
        if parent_face_id is None:
            raise ProtectionDesignError(
                "parent_face_id is required to bind geometric constraints to FEM ordering"
            )
        geometry_ids = _array(parent_face_id, "parent_face_id", len(fem_ids)).astype(
            np.int64
        )
        if not np.array_equal(geometry_ids, fem_ids):
            raise ProtectionDesignError("geometric arrays are not in FEM parent-face order")

        if erosion_rate_mm_h is not None and not np.array_equal(
            _float_array(
                erosion_rate_mm_h, "erosion_rate_mm_h", len(erosion_ids)
            ),
            evidence_rates,
        ):
            raise ProtectionDesignError(
                "erosion_rate_mm_h override differs from hashed erosion evidence"
            )
        erosion_by_id = {
            int(face_id): float(rate)
            for face_id, rate in zip(erosion_ids, evidence_rates)
        }
        ordered_erosion = np.asarray(
            [erosion_by_id[int(face_id)] for face_id in fem_ids], dtype=np.float64
        )
        evidence_stress = _float_array(
            fem_surface.get("von_mises_stress_pa"),
            "FEM von_mises_stress_pa",
            len(fem_ids),
        )
        evidence_fatigue = _float_array(
            fem_surface.get("fatigue_damage"),
            "FEM fatigue_damage",
            len(fem_ids),
        )
        if von_mises_stress_pa is not None and not np.array_equal(
            _float_array(
                von_mises_stress_pa, "von_mises_stress_pa", len(fem_ids)
            ),
            evidence_stress,
        ):
            raise ProtectionDesignError(
                "von_mises_stress_pa override differs from hashed FEM evidence"
            )
        if fatigue_damage is not None and not np.array_equal(
            _float_array(fatigue_damage, "fatigue_damage", len(fem_ids)),
            evidence_fatigue,
        ):
            raise ProtectionDesignError(
                "fatigue_damage override differs from hashed FEM evidence"
            )
        if root_forbidden_mask is None:
            raise ProtectionDesignError("root_forbidden_mask is required")
        classified = classify_protection_zones(
            parent_face_id=fem_ids,
            erosion_rate_mm_h=ordered_erosion,
            von_mises_stress_pa=evidence_stress,
            fatigue_damage=evidence_fatigue,
            face_area_m2=face_area_m2,
            face_centers_m=face_centers_m,
            adjacency=adjacency,
            root_forbidden_mask=root_forbidden_mask,
            high_erosion_quantile=high_erosion_quantile,
            high_stress_quantile=high_stress_quantile,
            low_stress_quantile=low_stress_quantile,
            minimum_feature_m=minimum_feature_m,
            minimum_transition_width_m=minimum_transition_width_m,
            maximum_area_fraction=maximum_area_fraction,
            maximum_mass_kg=maximum_mass_kg,
            coating_density_kg_m3=coating_density_kg_m3,
            coating_thickness_m=coating_thickness_m,
            require_connected=require_connected,
        )
        if classified.get("status") == "blocked":
            raise ProtectionDesignError(str(classified.get("reason")))
        report = dict(classified)
        report.update(
            {
                "status": "designed",
                "protection_gate": "passed",
                "software_validation_only": False,
                "real_fem_evidence": True,
                "independent_erosion_evidence": True,
                "evidence": {
                    "erosion_evidence_sha256": erosion_hash,
                    "fem_validation_sha256": fem_hash,
                    "fem_mesh_sha256": fem_validation.get("fine_mesh_sha256"),
                    "pressure_load_sha256": fem_validation.get(
                        "pressure_load_sha256"
                    ),
                    "surface_topology_sha256": fem_surface.get(
                        "topology_sha256"
                    ),
                },
            }
        )
        report["design_sha256"] = _canonical_sha256(report)
        return report
    except (ProtectionDesignError, TypeError, ValueError) as exc:
        return {
            "version": VERSION,
            "status": "blocked",
            "protection_gate": "blocked",
            "software_validation_only": False,
            "real_fem_evidence": False,
            "independent_erosion_evidence": False,
            "reason": str(exc),
        }


def validate_protection_design(report: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("status") != "designed" or report.get("protection_gate") != "passed":
        raise ProtectionDesignError("protection design is not passed")
    if report.get("software_validation_only") is not False:
        raise ProtectionDesignError("software-validation zoning is not a production design")
    stored_hash = _content_hash(report, "design_sha256")
    labels = np.asarray(report.get("zone_labels"), dtype=object)
    protected = np.asarray(report.get("protected_mask"), dtype=bool)
    root = np.asarray(report.get("root_forbidden_mask"), dtype=bool)
    if labels.shape != protected.shape or protected.shape != root.shape:
        raise ProtectionDesignError("protection mask shapes differ")
    if np.any(protected & root):
        raise ProtectionDesignError("protected mask intersects root-forbidden region")
    if (
        report.get("real_fem_evidence") is not True
        or report.get("independent_erosion_evidence") is not True
    ):
        raise ProtectionDesignError("design evidence flags are incomplete")
    result = dict(report)
    result["design_sha256"] = stored_hash
    return result


__all__ = [
    "ProtectionDesignError",
    "VERSION",
    "classify_protection_zones",
    "design_protection",
    "validate_protection_design",
]
