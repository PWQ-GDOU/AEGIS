from copy import deepcopy

import numpy as np

from step30_particle_convergence_v5 import (
    _confirmation_identity_compatibility,
    _ensemble_block_rows,
)


def _identity():
    return {
        "artifact_set_id": "artifact",
        "mesh_signature": "mesh",
        "topology_sha256": "topology",
        "aggregation_contract_sha256": "contract",
        "partition_sha256": "partition",
        "strict_immutable_run_identity": {
            "artifact_set_id": "artifact",
            "field_time": "0.8684190921",
            "implementation_hashes": {
                "artifact_identity": "artifact-code",
                "cfd_topology": "topology-code",
                "particle_tracking": "old-tracker",
            },
            "parameters": {"rho_particle_kg_m3": 2650.0},
        },
    }


def test_execution_target_and_tracker_change_are_compatible():
    discovery = _identity()
    confirmation = deepcopy(discovery)
    confirmation["strict_immutable_run_identity"]["parameters"]["execution_target"] = "dcu_cpu"
    confirmation["strict_immutable_run_identity"]["implementation_hashes"]["particle_tracking"] = "new-tracker"

    result = _confirmation_identity_compatibility(discovery, confirmation)

    assert result is not None
    assert result["status"] == "performance_only_tracker_hash_change"
    assert "strict_immutable_run_identity.parameters.execution_target" in result["changed_fields"]
    assert "strict_immutable_run_identity.implementation_hashes.particle_tracking" in result["changed_fields"]


def test_physical_parameter_change_is_rejected():
    discovery = _identity()
    confirmation = deepcopy(discovery)
    confirmation["strict_immutable_run_identity"]["parameters"]["rho_particle_kg_m3"] = 2700.0
    confirmation["strict_immutable_run_identity"]["implementation_hashes"]["particle_tracking"] = "new-tracker"

    assert _confirmation_identity_compatibility(discovery, confirmation) is None


def test_registered_collision_and_tracker_performance_change_is_compatible():
    discovery = _identity()
    confirmation = deepcopy(discovery)
    confirmation["strict_immutable_run_identity"]["parameters"]["execution_target"] = "dcu_cpu"
    confirmation["strict_immutable_run_identity"]["implementation_hashes"]["cfd_topology"] = "new-collision"
    confirmation["strict_immutable_run_identity"]["implementation_hashes"]["particle_tracking"] = "new-tracker"

    result = _confirmation_identity_compatibility(discovery, confirmation)

    assert result is not None
    assert result["status"] == "performance_only_collision_and_tracker_hash_change"
    assert "strict_immutable_run_identity.implementation_hashes.cfd_topology" in result["changed_fields"]


def test_unregistered_implementation_change_is_rejected():
    discovery = _identity()
    confirmation = deepcopy(discovery)
    confirmation["strict_immutable_run_identity"]["parameters"]["execution_target"] = "dcu_cpu"
    confirmation["strict_immutable_run_identity"]["implementation_hashes"]["artifact_identity"] = "new-artifact"
    confirmation["strict_immutable_run_identity"]["implementation_hashes"]["particle_tracking"] = "new-tracker"

    assert _confirmation_identity_compatibility(discovery, confirmation) is None


def test_ensemble_block_rows_average_disjoint_runs_and_preserve_provenance():
    rows = []
    for seed, value in enumerate(range(15), start=1):
        rows.append(
            {
                "seed": seed,
                "metadata": {"input_bindings": {"particle_hits_sha256": f"h{seed}"}},
                "field": {
                    "raw_face_target_mass_kg": np.asarray([value, value + 1.0]),
                    "aggregate_target_mass_kg": np.asarray([value + 2.0]),
                    "aggregate_local_support_mass_kg": np.asarray([value + 3.0]),
                    "total_input_target_mass_kg": value + 4.0,
                },
            }
        )
    blocks = _ensemble_block_rows(rows, 3)
    assert len(blocks) == 5
    assert blocks[0]["ensemble_block_seeds"] == [1, 2, 3]
    assert blocks[0]["ensemble_block_particle_hits_sha256"] == ["h1", "h2", "h3"]
    assert np.allclose(blocks[0]["field"]["aggregate_target_mass_kg"], [3.0])
    assert blocks[4]["seed"] == "block_05"
