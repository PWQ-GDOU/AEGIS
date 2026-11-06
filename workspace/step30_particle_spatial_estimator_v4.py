"""Conservative, topology-local particle erosion estimator for Step 30.

This is a numerical risk-map estimator, not a spatial-accuracy validator.  It
uses only the native CFD topology identity attached to an immutable particle
run.  A finite, shared-edge parent-face neighbourhood prevents interpolation
through disconnected surface regions and conserves the total input target mass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from step30_cfd_topology import load_topology_artifact
from step30_validation_common import canonical_sha256, safe_json_dump, sha256_file


VERSION = "step30-particle-spatial-estimator-v4"
KERNEL_CONTRACT = {
    "version": "topology-local-area-normalized-gaussian-v1",
    "adjacency_rule": "parent faces share at least one collision-triangle edge",
    "kernel": "gaussian_hop_distance",
    "bandwidth_hop_candidates": [1, 2, 3],
    "default_bandwidth_hops": 2,
    "truncation_hops": 3,
    "receiver_weight": "face_area_m2 * exp(-0.5 * (hop_distance / bandwidth_hops)^2)",
    "normalization": "per-source finite neighbourhood; exact target-mass conservation",
    "confidence": {
        "minimum_support_mass_fraction_of_total": 1.0e-5,
        "minimum_local_ess": 1.5,
    },
}


class ParticleSpatialEstimatorError(ValueError):
    """Raised when the frozen input identity cannot support a v4 risk map."""


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _topology_identity(topology_path: str | Path) -> tuple[dict[str, Any], Path, str]:
    path = Path(topology_path).resolve()
    if not path.is_file():
        raise ParticleSpatialEstimatorError(f"topology artifact does not exist: {path}")
    topology = load_topology_artifact(path)
    return topology, path, sha256_file(path)


def build_parent_face_adjacency(
    triangles: np.ndarray,
    triangle_parent_face_id: np.ndarray,
    n_parent_faces: int,
) -> tuple[tuple[np.ndarray, ...], str]:
    """Build canonical parent-face adjacency from collision-triangle edges."""
    tri = np.asarray(triangles, dtype=np.int64)
    parent = np.asarray(triangle_parent_face_id, dtype=np.int64)
    if (
        tri.ndim != 2
        or tri.shape[1] != 3
        or parent.shape != (len(tri),)
        or n_parent_faces <= 0
        or np.any(parent < 0)
        or np.any(parent >= n_parent_faces)
    ):
        raise ParticleSpatialEstimatorError("topology triangle/parent arrays are invalid")
    edge_parents: dict[tuple[int, int], set[int]] = {}
    for row, face_id in zip(tri, parent):
        for first, second in ((0, 1), (1, 2), (2, 0)):
            left, right = int(row[first]), int(row[second])
            edge = (left, right) if left < right else (right, left)
            edge_parents.setdefault(edge, set()).add(int(face_id))
    neighbours: list[set[int]] = [set() for _ in range(n_parent_faces)]
    for parents in edge_parents.values():
        if len(parents) < 2:
            continue
        ordered = sorted(parents)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                neighbours[left].add(right)
                neighbours[right].add(left)
    result = tuple(
        np.asarray(sorted(values), dtype=np.int64) for values in neighbours
    )
    adjacency_hash = canonical_sha256(
        {
            "rule": KERNEL_CONTRACT["adjacency_rule"],
            "n_parent_faces": int(n_parent_faces),
            "neighbours": [value.tolist() for value in result],
        }
    )
    return result, adjacency_hash


def finite_neighbourhood(
    adjacency: Sequence[np.ndarray],
    source_face_id: int,
    *,
    truncation_hops: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted receiver IDs and graph-hop distance for one source face."""
    if (
        source_face_id < 0
        or source_face_id >= len(adjacency)
        or truncation_hops < 0
    ):
        raise ParticleSpatialEstimatorError("invalid source face or truncation hops")
    distance = {int(source_face_id): 0}
    queue: deque[int] = deque([int(source_face_id)])
    while queue:
        current = queue.popleft()
        current_distance = distance[current]
        if current_distance >= truncation_hops:
            continue
        for neighbour in adjacency[current]:
            item = int(neighbour)
            if item not in distance:
                distance[item] = current_distance + 1
                queue.append(item)
    ids = np.asarray(sorted(distance), dtype=np.int64)
    hops = np.asarray([distance[int(value)] for value in ids], dtype=np.int16)
    return ids, hops


def estimate_surface_kernel(
    raw_face_target_mass_kg: np.ndarray,
    face_area_m2: np.ndarray,
    adjacency: Sequence[np.ndarray],
    *,
    bandwidth_hops: int = int(KERNEL_CONTRACT["default_bandwidth_hops"]),
    truncation_hops: int = int(KERNEL_CONTRACT["truncation_hops"]),
) -> dict[str, np.ndarray | float | int]:
    """Distribute raw erosion mass over topology-local, area-weighted kernels."""
    raw = np.asarray(raw_face_target_mass_kg, dtype=np.float64)
    area = np.asarray(face_area_m2, dtype=np.float64)
    if (
        raw.ndim != 1
        or area.shape != raw.shape
        or len(adjacency) != len(raw)
        or not np.isfinite(raw).all()
        or not np.isfinite(area).all()
        or np.any(raw < 0.0)
        or np.any(area <= 0.0)
    ):
        raise ParticleSpatialEstimatorError("raw mass, area, and adjacency shapes are invalid")
    if bandwidth_hops not in KERNEL_CONTRACT["bandwidth_hop_candidates"]:
        raise ParticleSpatialEstimatorError("bandwidth_hops is outside the frozen candidate set")
    if truncation_hops != KERNEL_CONTRACT["truncation_hops"]:
        raise ParticleSpatialEstimatorError("truncation_hops differs from the frozen contract")

    kernel_mass = np.zeros(len(raw), dtype=np.float64)
    support_mass = np.zeros(len(raw), dtype=np.float64)
    squared_contribution = np.zeros(len(raw), dtype=np.float64)
    active_sources = np.flatnonzero(raw > 0.0)
    for source_id in active_sources:
        receiver_ids, hop_distance = finite_neighbourhood(
            adjacency, int(source_id), truncation_hops=truncation_hops
        )
        unnormalized = area[receiver_ids] * np.exp(
            -0.5 * (hop_distance.astype(np.float64) / float(bandwidth_hops)) ** 2
        )
        normalizer = float(unnormalized.sum())
        if not np.isfinite(normalizer) or normalizer <= 0.0:
            raise ParticleSpatialEstimatorError("local kernel normalizer is invalid")
        contribution = raw[source_id] * unnormalized / normalizer
        np.add.at(kernel_mass, receiver_ids, contribution)
        np.add.at(support_mass, receiver_ids, raw[source_id])
        np.add.at(squared_contribution, receiver_ids, contribution**2)
    kernel_ess = np.divide(
        kernel_mass**2,
        squared_contribution,
        out=np.zeros_like(kernel_mass),
        where=squared_contribution > 0.0,
    )
    total_mass = float(raw.sum())
    conservation_error = float(kernel_mass.sum() - total_mass)
    if not np.isclose(kernel_mass.sum(), total_mass, rtol=1.0e-12, atol=1.0e-18):
        raise ParticleSpatialEstimatorError("surface kernel failed target-mass conservation")
    min_support = max(
        np.finfo(np.float64).tiny,
        total_mass
        * float(KERNEL_CONTRACT["confidence"]["minimum_support_mass_fraction_of_total"]),
    )
    confidence_mask = (
        (support_mass >= min_support)
        & (kernel_ess >= float(KERNEL_CONTRACT["confidence"]["minimum_local_ess"]))
    )
    return {
        "raw_face_target_mass_kg": raw,
        "raw_mass_per_area_kg_m2": raw / area,
        "surface_kernel_face_target_mass_kg": kernel_mass,
        "surface_kernel_mass_per_area_kg_m2": kernel_mass / area,
        "local_support_mass_kg": support_mass,
        "local_ess": kernel_ess,
        "confidence_mask": confidence_mask,
        "bandwidth_hops": int(bandwidth_hops),
        "truncation_hops": int(truncation_hops),
        "total_input_target_mass_kg": total_mass,
        "total_kernel_target_mass_kg": float(kernel_mass.sum()),
        "mass_conservation_error_kg": conservation_error,
    }


def _load_immutable_run(
    particle_hits_path: str | Path,
    topology: Mapping[str, Any],
    topology_path: Path,
    topology_sha256: str,
) -> dict[str, Any]:
    npz_path = Path(particle_hits_path).resolve()
    summary_path = npz_path.parent / "summary.json"
    run_manifest_path = npz_path.parent / "particle_run_manifest.json"
    artifact_manifest_path = npz_path.parent.parent.parent / "artifact_set_manifest.json"
    for path, name in (
        (npz_path, "particle hits"),
        (summary_path, "particle summary"),
        (run_manifest_path, "particle manifest"),
        (artifact_manifest_path, "artifact-set manifest"),
    ):
        if not path.is_file():
            raise ParticleSpatialEstimatorError(f"{name} is missing: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    descriptor = dict(artifact_manifest)
    artifact_id = descriptor.pop("artifact_set_id", None)
    topo_descriptor = descriptor.get("topology")
    if (
        artifact_manifest.get("version") != "step30-artifact-set-descriptor-v2"
        or canonical_sha256(descriptor) != artifact_id
        or not isinstance(topo_descriptor, Mapping)
        or topo_descriptor.get("file_sha256") != topology_sha256
        or topo_descriptor.get("mesh_signature") != topology.get("mesh_signature")
        or int(topo_descriptor.get("n_parent_faces", -1))
        != int(topology.get("n_parent_faces", -2))
    ):
        raise ParticleSpatialEstimatorError(
            "topology path is not the topology identity bound to this artifact set"
        )
    with np.load(npz_path, allow_pickle=False) as handle:
        required = {
            "face_target_mass_kg",
            "face_area_m2",
            "face_centers_local_mm",
            "artifact_set_id",
            "mesh_signature",
            "collision_backend",
            "collision_policy",
        }
        missing = sorted(required - set(handle.files))
        if missing:
            raise ParticleSpatialEstimatorError(f"particle hit file misses fields: {missing}")
        values = {name: np.asarray(handle[name]).copy() for name in required}
    geometry = summary.get("geometry")
    identity = summary.get("artifact_identity")
    if (
        summary.get("version") != "step30-cfd-driven-v2"
        or not isinstance(geometry, Mapping)
        or not isinstance(identity, Mapping)
        or identity.get("artifact_set_id") != artifact_id
        or geometry.get("mesh_signature") != topology.get("mesh_signature")
        or int(geometry.get("n_parent_faces", -1)) != int(topology.get("n_parent_faces", -2))
        or str(values["artifact_set_id"].item()) != artifact_id
        or str(values["mesh_signature"].item()) != topology.get("mesh_signature")
        or str(values["collision_backend"].item()) != "cfd_topology"
        or str(values["collision_policy"].item()) != "strict_v2"
    ):
        raise ParticleSpatialEstimatorError("particle run is not bound to the supplied native topology")
    raw = np.asarray(values["face_target_mass_kg"], dtype=np.float64)
    area = np.asarray(values["face_area_m2"], dtype=np.float64)
    if raw.shape != (int(topology["n_parent_faces"]),) or area.shape != raw.shape:
        raise ParticleSpatialEstimatorError("particle face field cardinality differs from topology")
    if not np.allclose(area, topology["face_area_m2"], rtol=1.0e-6, atol=1.0e-12):
        raise ParticleSpatialEstimatorError("particle face areas differ from topology")
    return {
        "raw_face_target_mass_kg": raw,
        "face_area_m2": area,
        "particle_hits_path": str(npz_path),
        "particle_hits_sha256": sha256_file(npz_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "run_manifest_path": str(run_manifest_path),
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "artifact_manifest_path": str(artifact_manifest_path),
        "artifact_manifest_sha256": sha256_file(artifact_manifest_path),
        "artifact_set_id": artifact_id,
        "particle_run_id": identity.get("particle_run_id"),
        "mesh_signature": topology.get("mesh_signature"),
        "topology_path": str(topology_path),
        "topology_sha256": topology_sha256,
    }


def estimate_immutable_particle_run(
    particle_hits_path: str | Path,
    topology_path: str | Path,
    *,
    bandwidth_hops: int = int(KERNEL_CONTRACT["default_bandwidth_hops"]),
) -> dict[str, Any]:
    """Return a provenance-bound v4 field package for one immutable run."""
    topology, resolved_topology, topology_sha256 = _topology_identity(topology_path)
    run = _load_immutable_run(
        particle_hits_path, topology, resolved_topology, topology_sha256
    )
    adjacency, adjacency_sha256 = build_parent_face_adjacency(
        topology["faces"],
        topology["collision_face_ids"],
        int(topology["n_parent_faces"]),
    )
    field = estimate_surface_kernel(
        run["raw_face_target_mass_kg"],
        run["face_area_m2"],
        adjacency,
        bandwidth_hops=bandwidth_hops,
    )
    metadata = {
        "version": VERSION,
        "evidence_status": "numerical_risk_map_only",
        "spatial_accuracy_status": "blocked",
        "official_accuracy_gate": False,
        "official_reference_gate": False,
        "kernel_contract": KERNEL_CONTRACT,
        "kernel_contract_sha256": canonical_sha256(KERNEL_CONTRACT),
        "bandwidth_hops": int(bandwidth_hops),
        "topology_adjacency_sha256": adjacency_sha256,
        "input_bindings": {
            key: value
            for key, value in run.items()
            if key not in {"raw_face_target_mass_kg", "face_area_m2"}
        },
        "field_hashes": {
            name: _array_sha256(np.asarray(value))
            for name, value in field.items()
            if isinstance(value, np.ndarray)
        },
        "limitations": [
            "A topology-local kernel improves numerical regularity only.",
            "No baseline/worn scan or independent spatial erosion truth is present.",
            "No ROOT, WETTED, AI5 patch, gauge direction, or CAD semantic inference is used.",
        ],
    }
    # This hash binds the immutable inputs and all in-memory field arrays. It
    # stays stable when save_estimate later adds the output-file binding.
    metadata["field_contract_sha256"] = canonical_sha256(metadata)
    return {"metadata": metadata, "field": field, "topology": topology}


def save_estimate(estimate: Mapping[str, Any], output_path: str | Path) -> dict[str, Any]:
    """Persist a field package without modifying the immutable particle run."""
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    field = estimate["field"]
    topology = estimate["topology"]
    metadata = dict(estimate["metadata"])
    arrays = {
        "face_id": np.arange(len(field["raw_face_target_mass_kg"]), dtype=np.int64),
        "face_area_m2": np.asarray(topology["face_area_m2"], dtype=np.float64),
        "raw_face_target_mass_kg": np.asarray(field["raw_face_target_mass_kg"], dtype=np.float64),
        "raw_mass_per_area_kg_m2": np.asarray(field["raw_mass_per_area_kg_m2"], dtype=np.float64),
        "surface_kernel_face_target_mass_kg": np.asarray(field["surface_kernel_face_target_mass_kg"], dtype=np.float64),
        "surface_kernel_mass_per_area_kg_m2": np.asarray(field["surface_kernel_mass_per_area_kg_m2"], dtype=np.float64),
        "local_support_mass_kg": np.asarray(field["local_support_mass_kg"], dtype=np.float64),
        "local_ess": np.asarray(field["local_ess"], dtype=np.float64),
        "confidence_mask": np.asarray(field["confidence_mask"], dtype=bool),
        "mesh_signature": np.asarray(str(metadata["input_bindings"]["mesh_signature"])),
        "field_contract_sha256": np.asarray(str(metadata["field_contract_sha256"])),
    }
    np.savez_compressed(target, **arrays)
    metadata["field_path"] = str(target)
    metadata["field_sha256"] = sha256_file(target)
    metadata["report_sha256"] = canonical_sha256(
        {key: value for key, value in metadata.items() if key != "report_sha256"}
    )
    json_path = target.with_suffix(".json")
    safe_json_dump(metadata, json_path)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a v4 topology-local particle risk map")
    parser.add_argument("--particle-hits", required=True, type=Path)
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--bandwidth-hops",
        type=int,
        default=int(KERNEL_CONTRACT["default_bandwidth_hops"]),
    )
    args = parser.parse_args(argv)
    try:
        estimate = estimate_immutable_particle_run(
            args.particle_hits, args.topology, bandwidth_hops=args.bandwidth_hops
        )
        report = save_estimate(estimate, args.output)
    except (OSError, ValueError, KeyError, ParticleSpatialEstimatorError) as exc:
        report = {
            "version": VERSION,
            "status": "blocked",
            "reason": str(exc),
            "official_accuracy_gate": False,
            "official_reference_gate": False,
        }
        safe_json_dump(report, args.output.with_suffix(".json"))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 3
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
