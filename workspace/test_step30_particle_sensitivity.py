"""Hermetic tests for fail-closed strict particle timestep sensitivity."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from step30_particle_sensitivity import run


def _artifact(root: Path, label: str, dt: float, faces: tuple[int, int]) -> tuple[str, Path, Path]:
    npz = root / f"{label}.npz"
    summary = root / f"{label}.json"
    np.savez_compressed(
        npz,
        parcel_id=np.array([0, 1], dtype=np.int32),
        impact_index=np.array([0, 0], dtype=np.int16),
        parent_face_id=np.array(faces, dtype=np.int64),
        parcel_mass_kg=np.array([1.0, 1.0]),
        Vp_ms=np.array([40.0, 42.0]),
        alpha_deg=np.array([5.0, 6.0]),
        face_target_mass_kg=np.array([1e-6, 2e-6]),
        dhdt_face_mm_h=np.array([0.1, 0.2]),
        collision_policy=np.array("strict_v2"),
    )
    summary.write_text(json.dumps({
        "version": "step30-cfd-driven-v2",
        "artifact_identity": {"artifact_set_id": "artifact-a"},
        "geometry": {"mesh_signature": "mesh-a"},
        "cfd": {
            "variant": "native_si",
            "field_time": "7",
            "speed_stats_ms": {"max": 100.0},
            "solver_semantics": {
                "application": "pimpleFoam",
                "mode": "urans",
                "field_statistic": "time_mean",
                "tracking_velocity_object": "UMean",
                "averaging_window": {"start_time": "1", "end_time": "7", "sample_count": 10},
            },
        },
        "parameters": {
            "dt_s": dt,
            "strict_max_substep_s": dt,
            "collision_policy": "strict_v2",
            "collision_backend": "cfd_topology",
            "cfd_variant": "native_si",
            "cfd_velocity_interpolation": "idw_k8_v1",
            "particle_integrator": "midpoint_exponential_v2",
            "particle_cfl": 0.05,
            "n_parcels": 500,
            "n_injected": 500,
            "seed": 42,
            "max_impacts": 3,
        },
        "tracking": {
            "n_impacts": 2,
            "fallback_hits": 0,
            "physical_strict_hit_ratio": 1.0,
            "collision_event_limit_count": 0,
        },
        "validation": {"registration_status": "passed"},
    }), encoding="utf-8")
    return label, npz, summary


def test_sensitivity_passes_only_when_all_declared_gates_pass(tmp_path: Path) -> None:
    runs = [
        _artifact(tmp_path, "coarse", 1e-5, (10, 20)),
        _artifact(tmp_path, "medium", 5e-6, (10, 20)),
        _artifact(tmp_path, "fine", 2.5e-6, (10, 20)),
    ]
    result = run(runs, tmp_path / "passed.json")
    assert result["status"] == "passed"
    assert all(pair["passed"] for pair in result["pairs"].values())


def test_sensitivity_cannot_report_passed_when_face_gate_fails(tmp_path: Path) -> None:
    runs = [
        _artifact(tmp_path, "coarse", 1e-5, (10, 20)),
        _artifact(tmp_path, "medium", 5e-6, (11, 20)),
        _artifact(tmp_path, "fine", 2.5e-6, (11, 20)),
    ]
    result = run(runs, tmp_path / "failed.json")
    pair = next(iter(result["pairs"].values()))
    assert pair["first_parent_face_agreement"] == 0.5
    assert pair["gates"]["first_parent_face_agreement"] is False
    assert pair["passed"] is False
    assert result["status"] == "failed"


def test_sensitivity_cli_returns_nonzero_for_failed_gates(tmp_path: Path) -> None:
    runs = [
        _artifact(tmp_path, "coarse", 1e-5, (10, 20)),
        _artifact(tmp_path, "medium", 5e-6, (11, 20)),
        _artifact(tmp_path, "fine", 2.5e-6, (11, 20)),
    ]
    command = [sys.executable, str(Path(__file__).with_name("step30_particle_sensitivity.py"))]
    for label, npz, summary in runs:
        command.extend(["--run", f"{label}={npz},{summary}"])
    command.extend(["--output", str(tmp_path / "failed.json")])
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 2
    assert json.loads((tmp_path / "failed.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_sensitivity_rejects_outer_dt_with_same_effective_substep(tmp_path: Path) -> None:
    coarse = _artifact(tmp_path, "coarse", 1e-5, (10, 20))
    summary = json.loads(coarse[2].read_text(encoding="utf-8"))
    summary["parameters"]["strict_max_substep_s"] = 2.5e-6
    coarse[2].write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="effective strict timestep"):
        run(
            [
                coarse,
                _artifact(tmp_path, "medium", 5e-6, (10, 20)),
                _artifact(tmp_path, "fine", 2.5e-6, (10, 20)),
            ],
            tmp_path / "failed.json",
        )


def test_sensitivity_rejects_zero_impact_artifact(tmp_path: Path) -> None:
    empty = _artifact(tmp_path, "empty", 1e-5, (10, 20))
    summary = json.loads(empty[2].read_text(encoding="utf-8"))
    summary["tracking"]["n_impacts"] = 0
    empty[2].write_text(json.dumps(summary), encoding="utf-8")
    np.savez_compressed(
        empty[1],
        parcel_id=np.array([], dtype=np.int32),
        impact_index=np.array([], dtype=np.int16),
        parent_face_id=np.array([], dtype=np.int64),
        parcel_mass_kg=np.array([], dtype=np.float64),
        Vp_ms=np.array([], dtype=np.float64),
        alpha_deg=np.array([], dtype=np.float64),
        face_target_mass_kg=np.array([], dtype=np.float64),
        dhdt_face_mm_h=np.array([], dtype=np.float64),
        collision_policy=np.array("strict_v2"),
    )
    with pytest.raises(ValueError, match="nonzero physical impacts"):
        run(
            [
                empty,
                _artifact(tmp_path, "medium", 5e-6, (10, 20)),
                _artifact(tmp_path, "fine", 2.5e-6, (10, 20)),
            ],
            tmp_path / "failed.json",
        )
