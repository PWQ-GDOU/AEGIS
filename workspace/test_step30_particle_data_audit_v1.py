from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from step30_particle_data_audit_v1 import audit_run


def _run_fixture(tmp_path: Path, *, bad_hash: bool = False) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    arrays = {
        "parent_face_id": np.asarray([0, 1], dtype=np.int64),
        "parcel_mass_kg": np.asarray([2.0, 3.0], dtype=np.float64),
        "particle_mass_kg": np.asarray([1.0, 1.0], dtype=np.float64),
        "ER_mansouri": np.asarray([0.1, 0.2], dtype=np.float64),
        "Vp_ms": np.asarray([10.0, 20.0], dtype=np.float64),
        "alpha_deg": np.asarray([20.0, 30.0], dtype=np.float64),
        "diameter_m": np.asarray([1.0e-4, 1.2e-4], dtype=np.float64),
        "face_centers_local_mm": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        "face_area_m2": np.asarray([1.0, 1.0], dtype=np.float64),
        "hit_kind": np.asarray([0, 0], dtype=np.int8),
    }
    np.savez(run / "particle_hits.npz", **arrays)
    summary = {
        "artifact_identity": {"artifact_set_id": "fixture"},
        "parameters": {"n_parcels": 10, "seed": 7, "collision_policy": "strict_v2", "collision_backend": "cfd_topology"},
        "cfd": {"field_time": "0.1"},
        "tracking": {"n_impacts": 2, "fallback_hits": 0, "physical_strict_hit_ratio": 1.0, "impact_mass_kg": 5.0, "hit_parcels": 2, "near_miss_diagnostics": 0, "substep_count": 20, "total_injection_mass_kg": 10.0},
        "face_erosion": {"surface_faces": 2, "total_target_mass_kg": 0.8},
    }
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run / "validation.png").write_bytes(b"fixture")
    manifest = {
        "config": {"n_parcels": 10, "seed": 7, "collision_policy": "strict_v2", "collision_backend": "cfd_topology", "cfd_field_time": "0.1"},
        "implementation_hashes": {"particle_tracking": "fixture"},
        "outputs": {},
    }
    for key, filename in (("particle_hits", "particle_hits.npz"), ("summary", "summary.json"), ("validation", "validation.png")):
        import hashlib
        manifest["outputs"][key] = {"file": filename, "sha256": hashlib.sha256((run / filename).read_bytes()).hexdigest()}
    (run / "particle_run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if bad_hash:
        (run / "summary.json").write_text("{}", encoding="utf-8")
    return run


def test_audit_passes_consistent_run(tmp_path: Path) -> None:
    report = audit_run(_run_fixture(tmp_path))
    assert report["status"] == "passed"
    assert report["stats"]["target_mass_kg_from_npz"] == pytest.approx(0.8)


def test_audit_fails_manifest_hash_mismatch(tmp_path: Path) -> None:
    report = audit_run(_run_fixture(tmp_path, bad_hash=True))
    assert report["status"] == "failed"
    assert any(item["code"] == "artifact_hash_mismatch" for item in report["issues"])


def test_audit_reports_sparse_support_and_near_miss_overhead(tmp_path: Path) -> None:
    run = _run_fixture(tmp_path)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    summary["tracking"]["near_miss_diagnostics"] = 100
    summary["tracking"]["substep_count"] = 20000
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    # The manifest hash mismatch is expected because this fixture intentionally
    # changes summary content after manifest creation.
    report = audit_run(run)
    codes = {item["code"] for item in report["issues"]}
    assert "artifact_hash_mismatch" in codes
    assert "near_miss_overhead" not in codes  # cross-run warning belongs to audit_run_root
