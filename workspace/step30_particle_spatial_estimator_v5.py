"""Topology-connected coarse particle erosion estimator for Step 30.

This is an exploratory numerical-screening estimator for sparse strict-v2
particle impacts.  It intentionally does not alter the frozen v3/v4 evidence:
the parent-face field remains available, while a deterministic partition of the
native CFD topology reports a lower-resolution, mass-conservative surface map.

The aggregation scale is not a CAD selection, a worn-surface reference, or an
official spatial-accuracy claim.  It is a topology-only numerical device whose
stability must be confirmed on particle runs not used during development.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from step30_particle_spatial_estimator_v4 import (
    ParticleSpatialEstimatorError,
    build_parent_face_adjacency,
    estimate_immutable_particle_run,
)
from step30_validation_common import canonical_sha256, safe_json_dump, sha256_file


VERSION = "step30-particle-spatial-estimator-v5"
AGGREGATION_CONTRACT = {
    "version": "topology-connected-bfs-face-budget-v1",
    "adjacency_rule": "parent faces share at least one collision-triangle edge",
    "partition_rule": (
        "canonical ascending-parent-face seed; breadth-first expansion over "
        "unassigned topology neighbours; fixed maximum parent-face budget"
    ),
    "candidate_max_parent_faces_per_aggregate": [4096, 8192, 16384, 32768],
    "default_max_parent_faces_per_aggregate": 16384,
    "selected_scale_interpretation": (
        "coarse topology-connected numerical screening scale for sparse "
        "particle impacts; not a physical erosion-resolution claim"
    ),
    "normalization": (
        "sum target mass within each connected aggregate, then redistribute "
        "to member faces in proportion to face area; exact total-mass conservation"
    ),
    "confidence": {
        "minimum_support_mass_fraction_of_total": 1.0e-5,
        "minimum_aggregate_contribution_ess": 3.0,
    },
}

# Float32 particle-hit files can produce a small discrepancy when impact-level
# support and face-level target masses are summed independently. This tolerance
# applies only to that cross-representation audit, not to projected mass
# conservation below.
IMPACT_SUPPORT_SERIALIZATION_REL_TOL = 2.0e-8


class ParticleSpatialEstimatorV5Error(ValueError):
    """Raised when a v5 topology aggregate cannot be built audibly."""


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _validate_adjacency(adjacency: Sequence[np.ndarray]) -> None:
    n_faces = len(adjacency)
    if n_faces <= 0:
        raise ParticleSpatialEstimatorV5Error("topology adjacency must be nonempty")
    for face_id, neighbours in enumerate(adjacency):
        values = np.asarray(neighbours, dtype=np.int64)
        if (
            values.ndim != 1
            or np.any(values < 0)
            or np.any(values >= n_faces)
            or np.any(values == face_id)
            or (len(values) > 1 and np.any(values[:-1] >= values[1:]))
        ):
            raise ParticleSpatialEstimatorV5Error("topology adjacency is not canonical")


def _partition_is_connected(
    adjacency: Sequence[np.ndarray], aggregate_id: np.ndarray
) -> bool:
    """Check that every aggregate is a graph-connected subset."""
    labels = np.asarray(aggregate_id, dtype=np.int64)
    n_aggregates = int(labels.max()) + 1 if len(labels) else 0
    counts = np.bincount(labels, minlength=n_aggregates)
    first = np.full(n_aggregates, -1, dtype=np.int64)
    for face_id, label in enumerate(labels):
        if first[int(label)] < 0:
            first[int(label)] = face_id
    visited = np.zeros(len(labels), dtype=bool)
    for label, expected_count in enumerate(counts):
        seed = int(first[label])
        if seed < 0 or expected_count <= 0:
            return False
        queue: deque[int] = deque([seed])
        visited[seed] = True
        reached = 0
        while queue:
            current = queue.popleft()
            reached += 1
            for neighbour in adjacency[current]:
                item = int(neighbour)
                if not visited[item] and labels[item] == label:
                    visited[item] = True
                    queue.append(item)
        if reached != int(expected_count):
            return False
    return True


def build_connected_aggregate_partition(
    adjacency: Sequence[np.ndarray],
    face_area_m2: np.ndarray,
    *,
    max_parent_faces_per_aggregate: int,
) -> dict[str, Any]:
    """Build a deterministic, connected parent-face partition.

    The helper permits any positive face budget so small synthetic topologies
    can exercise it.  Immutable Step-30 estimates restrict the budget to the
    frozen candidate set in :data:`AGGREGATION_CONTRACT`.
    """
    _validate_adjacency(adjacency)
    area = np.asarray(face_area_m2, dtype=np.float64)
    if (
        area.ndim != 1
        or len(area) != len(adjacency)
        or not np.isfinite(area).all()
        or np.any(area <= 0.0)
        or max_parent_faces_per_aggregate <= 0
    ):
        raise ParticleSpatialEstimatorV5Error(
            "face areas or aggregate face budget are invalid"
        )

    n_faces = len(adjacency)
    aggregate_id = np.full(n_faces, -1, dtype=np.int32)
    queued_for = np.zeros(n_faces, dtype=np.int32)
    aggregate = 0
    for seed in range(n_faces):
        if aggregate_id[seed] >= 0:
            continue
        queue: deque[int] = deque([seed])
        queued_for[seed] = aggregate + 1
        assigned = 0
        while queue and assigned < max_parent_faces_per_aggregate:
            current = queue.popleft()
            if aggregate_id[current] >= 0:
                continue
            aggregate_id[current] = aggregate
            assigned += 1
            for neighbour in adjacency[current]:
                item = int(neighbour)
                if (
                    aggregate_id[item] < 0
                    and queued_for[item] != aggregate + 1
                ):
                    queued_for[item] = aggregate + 1
                    queue.append(item)
        aggregate += 1

    if np.any(aggregate_id < 0) or not _partition_is_connected(adjacency, aggregate_id):
        raise ParticleSpatialEstimatorV5Error(
            "canonical topology aggregate partition is not connected"
        )
    n_aggregates = int(aggregate_id.max()) + 1
    face_count = np.bincount(aggregate_id, minlength=n_aggregates).astype(np.int64)
    aggregate_area_m2 = np.bincount(
        aggregate_id, weights=area, minlength=n_aggregates
    ).astype(np.float64)
    if (
        np.any(face_count <= 0)
        or np.any(aggregate_area_m2 <= 0.0)
        or int(face_count.sum()) != n_faces
    ):
        raise ParticleSpatialEstimatorV5Error("aggregate partition accounting failed")
    partition_descriptor = {
        "aggregation_contract_sha256": canonical_sha256(AGGREGATION_CONTRACT),
        "max_parent_faces_per_aggregate": int(max_parent_faces_per_aggregate),
        "n_parent_faces": int(n_faces),
        "n_aggregates": n_aggregates,
        "aggregate_id_sha256": _array_sha256(aggregate_id),
        "aggregate_face_count_sha256": _array_sha256(face_count),
        "aggregate_area_m2_sha256": _array_sha256(aggregate_area_m2),
    }
    partition_descriptor["partition_sha256"] = canonical_sha256(partition_descriptor)
    return {
        "aggregate_id": aggregate_id,
        "aggregate_face_count": face_count,
        "aggregate_area_m2": aggregate_area_m2,
        "partition": partition_descriptor,
    }


def estimate_topology_aggregate(
    raw_face_target_mass_kg: np.ndarray,
    face_area_m2: np.ndarray,
    aggregate_id: np.ndarray,
    *,
    contribution_parent_face_id: np.ndarray | None = None,
    contribution_target_mass_kg: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | int]:
    """Aggregate a face field and preserve its total target mass exactly."""
    raw = np.asarray(raw_face_target_mass_kg, dtype=np.float64)
    area = np.asarray(face_area_m2, dtype=np.float64)
    labels = np.asarray(aggregate_id, dtype=np.int64)
    if (
        raw.ndim != 1
        or area.shape != raw.shape
        or labels.shape != raw.shape
        or not np.isfinite(raw).all()
        or not np.isfinite(area).all()
        or np.any(raw < 0.0)
        or np.any(area <= 0.0)
        or np.any(labels < 0)
    ):
        raise ParticleSpatialEstimatorV5Error("face field or aggregate labels are invalid")
    n_aggregates = int(labels.max()) + 1 if len(labels) else 0
    if n_aggregates <= 0 or not np.array_equal(
        np.unique(labels), np.arange(n_aggregates, dtype=np.int64)
    ):
        raise ParticleSpatialEstimatorV5Error("aggregate labels must be contiguous")
    aggregate_area = np.bincount(
        labels, weights=area, minlength=n_aggregates
    ).astype(np.float64)
    aggregate_mass = np.bincount(
        labels, weights=raw, minlength=n_aggregates
    ).astype(np.float64)
    if np.any(aggregate_area <= 0.0):
        raise ParticleSpatialEstimatorV5Error("aggregate has zero area")

    if contribution_parent_face_id is None or contribution_target_mass_kg is None:
        support_mass = aggregate_mass.copy()
        squared_contribution = np.bincount(
            labels, weights=raw**2, minlength=n_aggregates
        ).astype(np.float64)
        contribution_count = np.bincount(
            labels, weights=(raw > 0.0), minlength=n_aggregates
        ).astype(np.int64)
        contribution_resolution = "parent_face_aggregate_only"
        impact_support_mass_relative_error = 0.0
        impact_support_mass_relative_tolerance = None
    else:
        parent_id = np.asarray(contribution_parent_face_id, dtype=np.int64)
        contribution = np.asarray(contribution_target_mass_kg, dtype=np.float64)
        if (
            parent_id.ndim != 1
            or contribution.shape != parent_id.shape
            or not np.isfinite(contribution).all()
            or np.any(parent_id < 0)
            or np.any(parent_id >= len(raw))
            or np.any(contribution < 0.0)
        ):
            raise ParticleSpatialEstimatorV5Error(
                "impact-level aggregate support inputs are invalid"
            )
        support_mass = np.bincount(
            labels[parent_id], weights=contribution, minlength=n_aggregates
        ).astype(np.float64)
        squared_contribution = np.bincount(
            labels[parent_id], weights=contribution**2, minlength=n_aggregates
        ).astype(np.float64)
        contribution_count = np.bincount(
            labels[parent_id],
            weights=(contribution > 0.0),
            minlength=n_aggregates,
        ).astype(np.int64)
        support_total = float(support_mass.sum())
        aggregate_total = float(aggregate_mass.sum())
        impact_support_mass_relative_error = abs(support_total - aggregate_total) / max(
            abs(aggregate_total), np.finfo(np.float64).tiny
        )
        impact_support_mass_relative_tolerance = IMPACT_SUPPORT_SERIALIZATION_REL_TOL
        if impact_support_mass_relative_error > IMPACT_SUPPORT_SERIALIZATION_REL_TOL:
            raise ParticleSpatialEstimatorV5Error(
                "impact-level support mass differs from face target mass"
            )
        contribution_resolution = "impact_target_mass_kg"

    aggregate_ess = np.divide(
        support_mass**2,
        squared_contribution,
        out=np.zeros_like(support_mass),
        where=squared_contribution > 0.0,
    )
    aggregate_density = aggregate_mass / aggregate_area
    projected_mass = aggregate_density[labels] * area
    total_mass = float(raw.sum())
    projected_total = float(projected_mass.sum())
    if not np.isclose(projected_total, total_mass, rtol=1.0e-12, atol=1.0e-18):
        raise ParticleSpatialEstimatorV5Error(
            "topology aggregate failed target-mass conservation"
        )
    minimum_support = max(
        np.finfo(np.float64).tiny,
        total_mass
        * float(
            AGGREGATION_CONTRACT["confidence"][
                "minimum_support_mass_fraction_of_total"
            ]
        ),
    )
    aggregate_confidence_mask = (
        (support_mass >= minimum_support)
        & (
            aggregate_ess
            >= float(
                AGGREGATION_CONTRACT["confidence"][
                    "minimum_aggregate_contribution_ess"
                ]
            )
        )
    )
    return {
        "raw_face_target_mass_kg": raw,
        "raw_mass_per_area_kg_m2": raw / area,
        "aggregate_target_mass_kg": aggregate_mass,
        "aggregate_area_m2": aggregate_area,
        "aggregate_mass_per_area_kg_m2": aggregate_density,
        "aggregate_local_support_mass_kg": support_mass,
        "aggregate_local_ess": aggregate_ess,
        "aggregate_contribution_count": contribution_count,
        "aggregate_confidence_mask": aggregate_confidence_mask,
        "surface_aggregate_face_target_mass_kg": projected_mass,
        "surface_aggregate_mass_per_area_kg_m2": aggregate_density[labels],
        "local_support_mass_kg": support_mass[labels],
        "local_ess": aggregate_ess[labels],
        "confidence_mask": aggregate_confidence_mask[labels],
        "contribution_resolution": np.asarray(contribution_resolution),
        "total_input_target_mass_kg": total_mass,
        "total_aggregate_target_mass_kg": float(aggregate_mass.sum()),
        "total_projected_target_mass_kg": projected_total,
        "mass_conservation_error_kg": float(projected_total - total_mass),
        "impact_support_mass_relative_error": float(impact_support_mass_relative_error),
        "impact_support_mass_relative_tolerance": impact_support_mass_relative_tolerance,
    }


def _load_impact_support(particle_hits_path: str | Path, n_parent_faces: int) -> tuple[np.ndarray, np.ndarray]:
    path = Path(particle_hits_path).resolve()
    with np.load(path, allow_pickle=False) as handle:
        required = {"parent_face_id", "parcel_mass_kg", "ER_mansouri"}
        missing = sorted(required - set(handle.files))
        if missing:
            raise ParticleSpatialEstimatorV5Error(
                f"particle hit file misses impact support fields: {missing}"
            )
        parent_face_id = np.asarray(handle["parent_face_id"], dtype=np.int64)
        contribution = np.asarray(
            handle["parcel_mass_kg"], dtype=np.float64
        ) * np.asarray(handle["ER_mansouri"], dtype=np.float64)
    if (
        parent_face_id.ndim != 1
        or contribution.shape != parent_face_id.shape
        or not np.isfinite(contribution).all()
        or np.any(parent_face_id < 0)
        or np.any(parent_face_id >= n_parent_faces)
        or np.any(contribution < 0.0)
    ):
        raise ParticleSpatialEstimatorV5Error("particle impact support arrays are invalid")
    return parent_face_id, contribution


def _topology_context_key(
    topology: Mapping[str, Any],
    *,
    max_parent_faces_per_aggregate: int,
) -> dict[str, Any]:
    """Hash the immutable topology inputs that define a reusable context."""
    if (
        max_parent_faces_per_aggregate
        not in AGGREGATION_CONTRACT["candidate_max_parent_faces_per_aggregate"]
    ):
        raise ParticleSpatialEstimatorV5Error(
            "aggregate face budget is outside the frozen v5 candidate set"
        )
    area = np.asarray(topology["face_area_m2"], dtype=np.float64)
    faces = np.asarray(topology["faces"], dtype=np.int64)
    collision_face_ids = np.asarray(topology["collision_face_ids"], dtype=np.int64)
    n_parent_faces = int(topology["n_parent_faces"])
    return {
        "n_parent_faces": n_parent_faces,
        "face_area_m2_sha256": _array_sha256(area),
        "faces_sha256": _array_sha256(faces),
        "collision_face_ids_sha256": _array_sha256(collision_face_ids),
        "max_parent_faces_per_aggregate": int(max_parent_faces_per_aggregate),
    }


def build_topology_aggregate_context(
    topology: Mapping[str, Any],
    *,
    max_parent_faces_per_aggregate: int,
) -> dict[str, Any]:
    """Build reusable topology-only state for a fixed aggregate scale."""
    topology_key = _topology_context_key(
        topology,
        max_parent_faces_per_aggregate=max_parent_faces_per_aggregate,
    )
    area = np.asarray(topology["face_area_m2"], dtype=np.float64)
    faces = np.asarray(topology["faces"], dtype=np.int64)
    collision_face_ids = np.asarray(topology["collision_face_ids"], dtype=np.int64)
    n_parent_faces = int(topology["n_parent_faces"])
    adjacency, adjacency_sha256 = build_parent_face_adjacency(
        faces, collision_face_ids, n_parent_faces
    )
    partition = build_connected_aggregate_partition(
        adjacency,
        area,
        max_parent_faces_per_aggregate=max_parent_faces_per_aggregate,
    )
    return {
        "topology_key": topology_key,
        "adjacency_sha256": adjacency_sha256,
        "partition": partition,
    }


def _validated_aggregate_context(
    topology: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    max_parent_faces_per_aggregate: int,
) -> dict[str, Any]:
    """Reject cached state if the immutable topology or scale differs."""
    expected_key = _topology_context_key(
        topology,
        max_parent_faces_per_aggregate=max_parent_faces_per_aggregate,
    )
    partition = context.get("partition")
    if context.get("topology_key") != expected_key or not isinstance(partition, Mapping):
        raise ParticleSpatialEstimatorV5Error(
            "runtime aggregate context does not match immutable topology"
        )
    labels = np.asarray(partition.get("aggregate_id"), dtype=np.int64)
    face_count = np.asarray(partition.get("aggregate_face_count"), dtype=np.int64)
    if (
        labels.shape != (int(expected_key["n_parent_faces"]),)
        or len(face_count) <= 0
        or np.any(labels < 0)
        or np.any(labels >= len(face_count))
        or int(face_count.sum()) != len(labels)
    ):
        raise ParticleSpatialEstimatorV5Error(
            "runtime aggregate context partition is malformed"
        )
    return dict(context)


def estimate_immutable_particle_run_v5(
    particle_hits_path: str | Path,
    topology_path: str | Path,
    *,
    max_parent_faces_per_aggregate: int = int(
        AGGREGATION_CONTRACT["default_max_parent_faces_per_aggregate"]
    ),
    topology_aggregate_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a provenance-bound v5 aggregate field for one immutable run."""
    if (
        max_parent_faces_per_aggregate
        not in AGGREGATION_CONTRACT["candidate_max_parent_faces_per_aggregate"]
    ):
        raise ParticleSpatialEstimatorV5Error(
            "aggregate face budget is outside the frozen v5 candidate set"
        )
    # v4 is the immutable-run verifier.  Its estimator output is intentionally
    # not used as a v5 input; only its strict topology/run identity validation is.
    verified = estimate_immutable_particle_run(
        particle_hits_path, topology_path, bandwidth_hops=2
    )
    topology = verified["topology"]
    metadata_v4 = verified["metadata"]
    raw = np.asarray(verified["field"]["raw_face_target_mass_kg"], dtype=np.float64)
    area = np.asarray(topology["face_area_m2"], dtype=np.float64)
    context = (
        build_topology_aggregate_context(
            topology,
            max_parent_faces_per_aggregate=max_parent_faces_per_aggregate,
        )
        if topology_aggregate_context is None
        else _validated_aggregate_context(
            topology,
            topology_aggregate_context,
            max_parent_faces_per_aggregate=max_parent_faces_per_aggregate,
        )
    )
    adjacency_sha256 = str(context["adjacency_sha256"])
    partition = dict(context["partition"])
    parent_face_id, contribution = _load_impact_support(
        particle_hits_path, int(topology["n_parent_faces"])
    )
    field = estimate_topology_aggregate(
        raw,
        area,
        partition["aggregate_id"],
        contribution_parent_face_id=parent_face_id,
        contribution_target_mass_kg=contribution,
    )
    input_bindings = dict(metadata_v4["input_bindings"])
    metadata = {
        "version": VERSION,
        "evidence_status": "exploratory_method_development",
        "numerical_stability_status": "requires_independent_confirmation",
        "spatial_accuracy_status": "blocked",
        "official_accuracy_gate": False,
        "official_reference_gate": False,
        "fem_gate": "blocked",
        "aggregation_contract": AGGREGATION_CONTRACT,
        "aggregation_contract_sha256": canonical_sha256(AGGREGATION_CONTRACT),
        "implementation_provenance": {
            "estimator_file": Path(__file__).name,
            "estimator_sha256": sha256_file(Path(__file__).resolve()),
        },
        "topology_adjacency_sha256": adjacency_sha256,
        "partition": partition["partition"],
        "input_bindings": input_bindings,
        "field_hashes": {
            name: _array_sha256(np.asarray(value))
            for name, value in field.items()
            if isinstance(value, np.ndarray)
        },
        "limitations": [
            "The aggregate partition uses native CFD topology only; it does not infer CAD semantics.",
            "The map is a coarse numerical screening product, not a worn-surface spatial validation.",
            "The current development runs cannot also serve as independent confirmation data.",
        ],
    }
    metadata["field_contract_sha256"] = canonical_sha256(metadata)
    return {
        "metadata": metadata,
        "field": field,
        "topology": topology,
        "aggregate_id": partition["aggregate_id"],
        "aggregate_face_count": partition["aggregate_face_count"],
        "_runtime_topology_aggregate_context": context,
    }


def save_estimate_v5(
    estimate: Mapping[str, Any], output_path: str | Path
) -> dict[str, Any]:
    """Persist an aggregate field without modifying an immutable particle run."""
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(estimate["metadata"])
    field = estimate["field"]
    topology = estimate["topology"]
    arrays = {
        "face_id": np.arange(len(field["raw_face_target_mass_kg"]), dtype=np.int64),
        "face_area_m2": np.asarray(topology["face_area_m2"], dtype=np.float64),
        "aggregate_id": np.asarray(estimate["aggregate_id"], dtype=np.int32),
        "aggregate_face_count": np.asarray(
            estimate["aggregate_face_count"], dtype=np.int64
        ),
        "raw_face_target_mass_kg": np.asarray(
            field["raw_face_target_mass_kg"], dtype=np.float64
        ),
        "surface_aggregate_face_target_mass_kg": np.asarray(
            field["surface_aggregate_face_target_mass_kg"], dtype=np.float64
        ),
        "surface_aggregate_mass_per_area_kg_m2": np.asarray(
            field["surface_aggregate_mass_per_area_kg_m2"], dtype=np.float64
        ),
        "aggregate_target_mass_kg": np.asarray(
            field["aggregate_target_mass_kg"], dtype=np.float64
        ),
        "aggregate_area_m2": np.asarray(field["aggregate_area_m2"], dtype=np.float64),
        "aggregate_mass_per_area_kg_m2": np.asarray(
            field["aggregate_mass_per_area_kg_m2"], dtype=np.float64
        ),
        "aggregate_local_support_mass_kg": np.asarray(
            field["aggregate_local_support_mass_kg"], dtype=np.float64
        ),
        "aggregate_local_ess": np.asarray(
            field["aggregate_local_ess"], dtype=np.float64
        ),
        "aggregate_contribution_count": np.asarray(
            field["aggregate_contribution_count"], dtype=np.int64
        ),
        "aggregate_confidence_mask": np.asarray(
            field["aggregate_confidence_mask"], dtype=bool
        ),
        "confidence_mask": np.asarray(field["confidence_mask"], dtype=bool),
        "mesh_signature": np.asarray(
            str(metadata["input_bindings"]["mesh_signature"])
        ),
        "partition_sha256": np.asarray(str(metadata["partition"]["partition_sha256"])),
        "field_contract_sha256": np.asarray(str(metadata["field_contract_sha256"])),
    }
    np.savez_compressed(target, **arrays)
    metadata["field_path"] = str(target)
    metadata["field_sha256"] = sha256_file(target)
    metadata["report_sha256"] = canonical_sha256(
        {key: value for key, value in metadata.items() if key != "report_sha256"}
    )
    safe_json_dump(metadata, target.with_suffix(".json"))
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a v5 topology-connected coarse particle screening map"
    )
    parser.add_argument("--particle-hits", required=True, type=Path)
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--max-parent-faces-per-aggregate",
        type=int,
        default=int(
            AGGREGATION_CONTRACT["default_max_parent_faces_per_aggregate"]
        ),
    )
    args = parser.parse_args(argv)
    try:
        estimate = estimate_immutable_particle_run_v5(
            args.particle_hits,
            args.topology,
            max_parent_faces_per_aggregate=args.max_parent_faces_per_aggregate,
        )
        report = save_estimate_v5(estimate, args.output)
    except (
        OSError,
        ValueError,
        KeyError,
        ParticleSpatialEstimatorError,
        ParticleSpatialEstimatorV5Error,
    ) as exc:
        report = {
            "version": VERSION,
            "status": "blocked",
            "reason": str(exc),
            "official_accuracy_gate": False,
            "official_reference_gate": False,
            "fem_gate": "blocked",
        }
        safe_json_dump(report, args.output.with_suffix(".json"))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 3
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
