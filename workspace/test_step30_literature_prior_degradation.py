from __future__ import annotations

from pathlib import Path

import numpy as np

from step30_literature_prior_degradation import MODEL_NAMES, run_prior_propagation


def _hits(path: Path) -> None:
    np.savez(
        path,
        parent_face_id=np.asarray([0, 1, 0, 2], dtype=np.int64),
        velocity_ms=np.asarray([[40.0, 0.0, 0.0]] * 4),
        Vp_ms=np.asarray([40.0] * 4),
        alpha_deg=np.asarray([25.0] * 4),
        diameter_m=np.asarray([200.0e-6] * 4),
        parcel_mass_kg=np.asarray([1.0e-7] * 4),
        face_centers_local_mm=np.zeros((3, 3), dtype=np.float32),
        face_area_m2=np.asarray([1.0e-4, 2.0e-4, 1.5e-4], dtype=np.float32),
    )


def test_prior_propagation_reports_nonzero_area_integrated_totals(tmp_path: Path) -> None:
    hits = tmp_path / "hits.npz"
    _hits(hits)
    report = run_prior_propagation(hits, n_draws=4, seed=7)
    assert report["status"] == "provisional"
    assert set(report["model_summary"]) == set(MODEL_NAMES)
    for model in MODEL_NAMES:
        summary = report["model_summary"][model]["draw_summary"]
        assert summary["total_area_weighted_mm3_h"] > 0.0
        assert summary["total_area_weighted_p95_mm3_h"] >= summary["total_area_weighted_mm3_h"]


def test_prior_propagation_records_exploratory_stress_summary(tmp_path: Path) -> None:
    hits = tmp_path / "hits.npz"
    stress = tmp_path / "stress.npz"
    _hits(hits)
    np.savez(
        stress,
        sigma_centrifugal_mpa=np.asarray([10.0, 20.0, 30.0]),
        sigma_hydraulic_mpa=np.asarray([5.0, 6.0, 7.0]),
    )
    report = run_prior_propagation(hits, stress_path=stress, n_draws=4, seed=7)
    assert report["exploratory_stress"]["status"] == "provisional"
    assert report["exploratory_stress"]["real_fem_evidence"] is False
    assert report["exploratory_stress"]["density_sensitivity"]["base"]["max_mpa"] == 37.0
