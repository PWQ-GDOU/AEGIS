from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from step30_particle_seed_convergence import evaluate_runs
from step30_validation_common import canonical_sha256, sha256_file


ARTIFACT = "a" * 64
MESH = "b" * 64
N_FACES = 100


def _write_run(
    root: Path,
    *,
    n: int,
    seed: int,
    face_shift: int = 0,
    heavy: bool = False,
    target_scale: float = 1.0,
    version: str = "step30-cfd-driven-v2",
    artifact: str = ARTIFACT,
    zero: bool = False,
) -> tuple[Path, Path]:
    solver_semantics = {
        "application": "pimpleFoam",
        "mode": "urans",
        "field_statistic": "time_mean",
        "tracking_velocity_object": "UMean",
        "averaging_window": {"start": 0.7, "end": 0.8},
    }
    artifact_descriptor = {
        "version": "step30-artifact-set-descriptor-v2",
        "cfd": {
            "field_time": "0.8684190921",
            "solver_semantics": solver_semantics,
            "fixture_marker": artifact,
        },
        "topology": {
            "mesh_signature": MESH,
            "n_parent_faces": N_FACES,
        },
        "units": {
            "source_coordinate_units": "m",
            "internal_coordinate_units": "mm",
            "coordinate_scale_to_mm": 1000.0,
        },
    }
    artifact_id = canonical_sha256(artifact_descriptor)
    artifact_descriptor["artifact_set_id"] = artifact_id
    artifact_root = root / artifact_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "artifact_set_manifest.json").write_text(
        json.dumps(artifact_descriptor), encoding="utf-8"
    )

    config = {
        "cfd_variant": "native_si",
        "collision_backend": "cfd_topology",
        "collision_policy": "strict_v2",
        "cfd_field_time": "0.8684190921",
        "n_parcels": n,
        "n_injected": n,
        "max_impacts": 3,
        "seed": seed,
        "dt_s": 1e-5,
        "strict_max_substep_s": 1e-5,
    }
    descriptor = {
        "version": "step30-particle-run-descriptor-v2",
        "artifact_set_id": artifact_id,
        "collision_backend": "cfd_topology",
        "implementation_hashes": {"particle_tracking": "d" * 64},
        "input_hashes": {"fixture_input": "e" * 64},
        "config": config,
    }
    run_id = canonical_sha256(descriptor)
    descriptor["particle_run_id"] = run_id
    run_root = artifact_root / "strict_v2" / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    count = 0 if zero else 300
    parcel_id = np.arange(count, dtype=np.int32)
    face_ids = (np.arange(count) % 20 + face_shift) % N_FACES
    masses = np.ones(count, dtype=np.float64)
    er = np.ones(count, dtype=np.float64) * target_scale
    if heavy and count:
        er[0] = 1000.0
    face_target = np.bincount(face_ids, weights=masses * er, minlength=N_FACES)
    npz = run_root / "particle_hits.npz"
    np.savez_compressed(
        npz,
        parcel_id=parcel_id,
        impact_index=np.zeros(count, dtype=np.int16),
        parent_face_id=face_ids.astype(np.int64),
        parcel_mass_kg=masses,
        ER_mansouri=er,
        Vp_ms=np.ones(count) * (40.0 + seed * 1e-5),
        alpha_deg=np.ones(count) * (15.0 + seed * 1e-5),
        face_target_mass_kg=face_target,
        dhdt_face_mm_h=face_target,
        face_centers_local_mm=np.column_stack(
            (
                np.arange(N_FACES, dtype=float),
                np.arange(N_FACES, dtype=float) % 10.0,
                np.arange(N_FACES, dtype=float) % 7.0,
            )
        ),
        artifact_set_id=np.array(artifact_id),
        mesh_signature=np.array(MESH),
        field_time=np.array("0.8684190921"),
        collision_backend=np.array("cfd_topology"),
        collision_policy=np.array("strict_v2"),
    )
    parameters = {
        "n_parcels": n,
        "n_injected": n,
        "seed": seed,
        "output_tag": None,
        "collision_policy": "strict_v2",
        "collision_backend": "cfd_topology",
        "cfd_variant": "native_si",
        "dt_s": 1e-5,
        "strict_max_substep_s": 1e-5,
        "max_impacts": 3,
    }
    summary = {
        "version": version,
        "artifact_identity": {"artifact_set_id": artifact_id, "particle_run_id": run_id},
        "cfd": {
            "field_time": "0.8684190921",
            "solver_semantics": solver_semantics,
        },
        "geometry": {"mesh_signature": MESH, "n_parent_faces": N_FACES},
        "parameters": parameters,
        "tracking": {
            "fallback_hits": 0,
            "collision_event_limit_count": 0,
            "physical_strict_hit_ratio": 1.0,
            "total_injection_mass_kg": float(max(n, 1)),
        },
    }
    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    validation_path = run_root / "validation.png"
    validation_path.write_bytes(b"particle convergence fixture")
    descriptor["outputs"] = {
        "particle_hits": {"file": npz.name, "sha256": sha256_file(npz)},
        "summary": {"file": summary_path.name, "sha256": sha256_file(summary_path)},
        "validation": {
            "file": validation_path.name,
            "sha256": sha256_file(validation_path),
        },
    }
    (run_root / "particle_run_manifest.json").write_text(
        json.dumps(descriptor), encoding="utf-8"
    )
    return npz, summary_path


def test_single_pilot_is_blocked_for_more_seeds(tmp_path: Path) -> None:
    report = evaluate_runs([_write_run(tmp_path / "run", n=5000, seed=42)])
    assert report["status"] == "blocked"
    assert report["action"] == "collect_more_seeds"
    assert report["official_accuracy_gate"] is False


def test_legacy_or_identity_mismatch_is_blocked(tmp_path: Path) -> None:
    legacy = _write_run(tmp_path / "legacy", n=5000, seed=1, version="step30-cfd-driven-v1")
    assert evaluate_runs([legacy])["status"] == "blocked"
    pairs = [
        _write_run(tmp_path / "a", n=5000, seed=1),
        _write_run(tmp_path / "b", n=5000, seed=2, artifact="d" * 64),
    ]
    assert evaluate_runs(pairs)["status"] == "blocked"


def test_zero_impacts_is_blocked(tmp_path: Path) -> None:
    report = evaluate_runs([_write_run(tmp_path / "zero", n=5000, seed=1, zero=True)])
    assert report["status"] == "blocked"
    assert "zero" in report["reason"]


def test_missing_or_tampered_immutable_manifest_is_blocked(tmp_path: Path) -> None:
    missing = _write_run(tmp_path / "missing", n=5000, seed=1)
    (missing[0].parent / "particle_run_manifest.json").unlink()
    report = evaluate_runs([missing])
    assert report["status"] == "blocked"
    assert "manifest is missing" in report["reason"]

    tampered = _write_run(tmp_path / "tampered", n=5000, seed=2)
    summary = json.loads(tampered[1].read_text(encoding="utf-8"))
    summary["parameters"]["seed"] = 999
    tampered[1].write_text(json.dumps(summary), encoding="utf-8")
    report = evaluate_runs([tampered])
    assert report["status"] == "blocked"
    assert "output hash mismatch" in report["reason"]


def test_tampered_artifact_set_identity_is_blocked(tmp_path: Path) -> None:
    pair = _write_run(tmp_path / "artifact", n=5000, seed=3)
    artifact_manifest_path = (
        pair[0].parent.parent.parent / "artifact_set_manifest.json"
    )
    manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    manifest["topology"]["n_parent_faces"] += 1
    artifact_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = evaluate_runs([pair])
    assert report["status"] == "blocked"
    assert "artifact-set manifest identity is invalid" in report["reason"]


def test_tampered_particle_descriptor_identity_is_blocked(tmp_path: Path) -> None:
    pair = _write_run(tmp_path / "descriptor", n=5000, seed=3)
    manifest_path = pair[0].parent / "particle_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"]["seed"] = 4
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = evaluate_runs([pair])
    assert report["status"] == "blocked"
    assert "identity is invalid" in report["reason"]


def test_low_ess_requests_stratified_psd(tmp_path: Path) -> None:
    pairs = [_write_run(tmp_path / str(seed), n=5000, seed=seed, heavy=True) for seed in (1, 2, 3)]
    report = evaluate_runs(pairs)
    assert report["status"] == "failed"
    assert report["action"] == "stratified_psd_required"


def test_spatial_instability_fails(tmp_path: Path) -> None:
    pairs = [
        _write_run(tmp_path / "a", n=5000, seed=1, face_shift=0),
        _write_run(tmp_path / "b", n=5000, seed=2, face_shift=30),
        _write_run(tmp_path / "c", n=5000, seed=3, face_shift=60),
    ]
    report = evaluate_runs(pairs)
    assert report["status"] == "failed"
    assert "spatial" in report["action"]


def test_stable_seed_and_scale_runs_pass(tmp_path: Path) -> None:
    pairs = []
    for n in (5000, 10000):
        for seed in (1, 2, 3):
            pairs.append(_write_run(tmp_path / f"n{n}s{seed}", n=n, seed=seed))
    report = evaluate_runs(pairs)
    assert report["status"] == "passed"
    assert report["action"] == "scale_converged"
    assert report["scale_comparisons"][0]["passed"] is True


def test_preregistered_max_scale_failure_stops_expansion(tmp_path: Path) -> None:
    pairs = [
        _write_run(
            tmp_path / f"n{n}s{seed}",
            n=n,
            seed=seed,
            face_shift=shift,
        )
        for n in (10000, 20000)
        for seed, shift in zip((1, 2, 3), (0, 30, 60))
    ]
    report = evaluate_runs(pairs)
    assert report["status"] == "failed"
    assert report["decision_scale"] == 20000
    assert report["preregistered_max_scale"] == 20000
    assert report["action"] == (
        "stop_at_preregistered_max_scale_investigate_spatial_instability"
    )


def test_final_scale_decision_can_supersede_failed_pilot(tmp_path: Path) -> None:
    pairs = [
        _write_run(
            tmp_path / f"n5000s{seed}",
            n=5000,
            seed=seed,
            face_shift=shift,
        )
        for seed, shift in zip((1, 2, 3), (0, 30, 60))
    ]
    for n in (10000, 20000):
        for seed in (1, 2, 3):
            pairs.append(_write_run(tmp_path / f"n{n}s{seed}", n=n, seed=seed))
    report = evaluate_runs(pairs)
    assert report["status"] == "passed"
    assert report["decision_scale"] == 20000
    assert report["decision_failures"] == []
    assert report["scales"]["5000"]["passed"] is False
    assert report["scale_comparisons"][-1]["scales"] == [10000, 20000]
    assert report["scale_comparisons"][-1]["passed"] is True
