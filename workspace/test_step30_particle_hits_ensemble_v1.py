from __future__ import annotations

import json

import numpy as np

from step30_particle_hits_ensemble_v1 import ParticleEnsembleError, merge_particle_hits


def _write_hit(path, seed: int, *, changed_domain: bool = False) -> None:
    centers = np.arange(12, dtype=np.float64).reshape(4, 3)
    if changed_domain:
        centers[0, 0] += 1.0
    np.savez(
        path,
        parent_face_id=np.array([0, 1, 2], dtype=np.int64),
        velocity_ms=np.full((3, 3), 10.0 + seed),
        Vp_ms=np.full(3, 4.0),
        alpha_deg=np.full(3, 30.0),
        diameter_m=np.full(3, 50.0e-6),
        parcel_mass_kg=np.array([1.0, 2.0, 3.0]) * 1.0e-6,
        face_centers_local_mm=centers,
        face_area_m2=np.full(4, 1.0e-4),
    )


def test_merge_is_mass_conservative_and_immutable(tmp_path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_hit(first, 1)
    _write_hit(second, 2)
    output = tmp_path / "ensemble.json"
    result = merge_particle_hits([first, second], output_path=output)
    assert result["merged"]["mass_conservation_passed"] is True
    assert result["merged"]["n_hits"] == 6
    assert output.is_file() and output.with_suffix(".npz").is_file()
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["report_sha256"] == result["report_sha256"]
    with np.load(output.with_suffix(".npz"), allow_pickle=False) as data:
        assert np.isclose(np.sum(data["parcel_mass_kg"]), 6.0e-6)
        assert set(data["seed_index"].tolist()) == {0, 1}


def test_merge_rejects_domain_mismatch(tmp_path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_hit(first, 1)
    _write_hit(second, 2, changed_domain=True)
    try:
        merge_particle_hits([first, second])
    except ParticleEnsembleError as exc:
        assert "face centers" in str(exc)
    else:
        raise AssertionError("domain mismatch must be rejected")
