"""Hermetic tests for Step30 immutable artifact and particle identities."""

from __future__ import annotations

from dataclasses import replace

import pytest

from step30_artifact_identity import (
    build_artifact_set_descriptor,
    build_particle_run_descriptor,
    canonical_sha256,
    validate_i1_i3_descriptors,
)
from step30_lagrangian_particle_tracking import TrackingConfig


def _topology(path):
    path.write_bytes(b"topology-v3")
    return {
        "artifact_version": "step30-native-openfoam-topology-v3",
        "mesh_signature_algorithm": "sha256-v3",
        "mesh_signature": "mesh-a",
        "source_vtp_sha256": "vtp-a",
        "patch_name": "bucket",
        "time_name": "7",
        "n_parent_faces": 2,
    }


def _cfd(statistic: str = "snapshot") -> dict:
    semantics = {
        "application": "simpleFoam" if statistic == "snapshot" else "pimpleFoam",
        "mode": "steady_rans" if statistic == "snapshot" else "urans",
        "field_statistic": statistic,
        "tracking_velocity_object": "U" if statistic == "snapshot" else "UMean",
        "averaging_window": (
            None
            if statistic == "snapshot"
            else {"start_time": "10", "end_time": "30", "sample_count": 2001}
        ),
    }
    manifest = {
        "version": "step30-native-si-manifest-v2",
        "polyMesh": {"signature": "poly-a"},
        "fields": {
            "Cx": {"sha256": "cx"},
            "Cy": {"sha256": "cy"},
            "Cz": {"sha256": "cz"},
            semantics["tracking_velocity_object"]: {"sha256": "velocity"},
        },
        "solver_semantics": semantics,
    }
    return {
        "variant": "native_si",
        "field_time": "7",
        "source_coordinate_units": "m",
        "internal_coordinate_units": "mm",
        "coordinate_scale_to_mm": 1000.0,
        "artifact_manifest": manifest,
        "artifact_manifest_sha256": "manifest-a",
    }


def test_canonical_hash_is_key_order_independent() -> None:
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})


def test_snapshot_and_time_mean_create_distinct_artifact_sets(tmp_path) -> None:
    topology_path = tmp_path / "topology.npz"
    topology = _topology(topology_path)
    snapshot = build_artifact_set_descriptor(_cfd("snapshot"), topology, topology_path)
    time_mean = build_artifact_set_descriptor(_cfd("time_mean"), topology, topology_path)
    assert snapshot["artifact_set_id"] != time_mean["artifact_set_id"]
    assert snapshot["cfd"]["solver_semantics"]["tracking_velocity_object"] == "U"
    assert time_mean["cfd"]["solver_semantics"]["tracking_velocity_object"] == "UMean"


def test_invalid_snapshot_and_mean_semantics_are_rejected(tmp_path) -> None:
    topology_path = tmp_path / "topology.npz"
    topology = _topology(topology_path)
    bad_snapshot = _cfd("snapshot")
    bad_snapshot["artifact_manifest"]["solver_semantics"]["tracking_velocity_object"] = "UMean"
    with pytest.raises(ValueError, match="snapshot"):
        build_artifact_set_descriptor(bad_snapshot, topology, topology_path)

    bad_mean = _cfd("time_mean")
    bad_mean["artifact_manifest"]["solver_semantics"]["averaging_window"] = None
    with pytest.raises(ValueError, match="time-mean"):
        build_artifact_set_descriptor(bad_mean, topology, topology_path)


def test_artifact_id_changes_when_topology_file_changes(tmp_path) -> None:
    topology_path = tmp_path / "topology.npz"
    topology = _topology(topology_path)
    first = build_artifact_set_descriptor(_cfd(), topology, topology_path)
    topology_path.write_bytes(b"topology-v3-modified")
    second = build_artifact_set_descriptor(_cfd(), topology, topology_path)
    assert first["artifact_set_id"] != second["artifact_set_id"]


def test_i1_i3_descriptors_may_only_differ_in_max_impacts() -> None:
    base = TrackingConfig(
        max_impacts=1,
        n_parcels=20,
        n_injected=20,
        cfd_field_time="7",
        output_tag="ignored",
    )
    single = build_particle_run_descriptor(
        "artifact-a", base, collision_backend="cfd_topology", implementation_sha256="code-a"
    )
    multiple = build_particle_run_descriptor(
        "artifact-a",
        replace(base, max_impacts=3, output_tag="also-ignored"),
        collision_backend="cfd_topology",
        implementation_sha256="code-a",
    )
    validate_i1_i3_descriptors(single, multiple)

    mixed_policy = build_particle_run_descriptor(
        "artifact-a",
        replace(base, max_impacts=3, collision_policy="legacy_proximity_v1"),
        collision_backend="cfd_topology",
        implementation_sha256="code-a",
    )
    with pytest.raises(ValueError, match="beyond max_impacts"):
        validate_i1_i3_descriptors(single, mixed_policy)


def test_particle_identity_is_independent_of_local_staging_paths() -> None:
    first = TrackingConfig(
        cfd_variant="native_si",
        cfd_field_time="7",
        topology_path="output/a/topology.npz",
        native_si_directory="output/a/native",
    )
    second = replace(
        first,
        topology_path="output/b/topology.npz",
        native_si_directory="output/b/native",
    )
    kwargs = {
        "collision_backend": "cfd_topology",
        "implementation_hashes": {"particle_tracking": "code-a"},
        "input_hashes": {"jet_rotation": "input-a"},
    }
    first_descriptor = build_particle_run_descriptor("artifact-a", first, **kwargs)
    second_descriptor = build_particle_run_descriptor("artifact-a", second, **kwargs)
    assert first_descriptor == second_descriptor
