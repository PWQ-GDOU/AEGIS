from __future__ import annotations

import numpy as np

from step30_artifact_identity import build_particle_run_descriptor
from step30_lagrangian_particle_tracking import (
    TrackingConfig,
    compute_parcel_masses,
    sample_rosin_rammler,
)


def test_number_sampling_remains_mass_proportional() -> None:
    cfg = TrackingConfig(psd_sampling="number_random_v1")
    diam = np.array([10e-6, 20e-6, 40e-6])
    particle_mass, parcel_mass, total = compute_parcel_masses(diam, cfg)
    assert parcel_mass.sum() == total
    assert np.allclose(parcel_mass / parcel_mass.sum(), particle_mass / particle_mass.sum())


def test_mass_stratified_parcels_are_equal_mass_and_conserve_total() -> None:
    cfg = TrackingConfig(psd_sampling="mass_stratified_v1")
    diam = sample_rosin_rammler(np.random.default_rng(42), 5000, cfg)
    particle_mass, parcel_mass, total = compute_parcel_masses(diam, cfg)
    assert len(diam) == 5000
    assert np.isfinite(diam).all()
    assert np.all((diam >= cfg.d_min_m) & (diam <= cfg.d_max_m))
    assert np.isclose(parcel_mass.sum(), total, rtol=0.0, atol=1e-14)
    assert np.allclose(parcel_mass, total / 5000, rtol=0.0, atol=1e-12)
    assert np.all(particle_mass > 0.0)


def test_mass_stratified_quantiles_are_seed_stable() -> None:
    cfg = TrackingConfig(psd_sampling="mass_stratified_v1")
    first = sample_rosin_rammler(np.random.default_rng(1), 5000, cfg)
    second = sample_rosin_rammler(np.random.default_rng(2), 5000, cfg)
    for percentile in (10, 50, 90, 99):
        left = float(np.percentile(first, percentile))
        right = float(np.percentile(second, percentile))
        assert abs(left - right) / max(left, right) < 0.01


def test_sampling_contract_changes_immutable_run_identity() -> None:
    base = TrackingConfig(psd_sampling="number_random_v1")
    stratified = TrackingConfig(psd_sampling="mass_stratified_v1")
    kwargs = {
        "artifact_set_id": "a" * 64,
        "collision_backend": "cfd_topology",
        "implementation_hashes": {"particle_tracking": "b" * 64},
        "input_hashes": {"field": "c" * 64},
    }
    first = build_particle_run_descriptor(config=base, **kwargs)
    second = build_particle_run_descriptor(config=stratified, **kwargs)
    assert first["particle_run_id"] != second["particle_run_id"]
    assert first["config"]["psd_sampling"] == "number_random_v1"
    assert second["config"]["psd_sampling"] == "mass_stratified_v1"
