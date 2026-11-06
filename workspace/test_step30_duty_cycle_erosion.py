"""
Regression tests for step30_duty_cycle_erosion.py

Coverage:
  - fractions always sum to 1.0
  - empty power_steps raises
  - invalid rated_fraction raises
  - rated 150 MW correctly assigned rated_fraction
  - empty dhdt raises
  - invalid fraction sum raises
  - hotspot IoU guarded against empty union
  - np.load handle is closed after run_weighted_erosion
  - zero-erosion array produces IoU 0.0 safely
"""
from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest

import step30_duty_cycle_erosion as dc


# ---------------------------------------------------------------------------
# build_duty_cycle
# ---------------------------------------------------------------------------

class TestBuildDutyCycle:
    def test_fractions_sum_to_one_uniform_with_rated(self):
        d = dc.build_duty_cycle(mode="uniform_with_rated", rated_fraction=0.25)
        assert abs(d["fraction"].sum() - 1.0) < 1e-9

    def test_fractions_sum_to_one_uniform(self):
        d = dc.build_duty_cycle(mode="uniform")
        assert abs(d["fraction"].sum() - 1.0) < 1e-9

    def test_rated_fraction_assigned_correctly(self):
        d = dc.build_duty_cycle(mode="uniform_with_rated", rated_fraction=0.25)
        rated_idx = np.where(d["power_mw"] == 150.0)[0]
        if len(rated_idx):
            assert abs(d["fraction"][rated_idx[0]] - 0.25) < 1e-9

    def test_empty_power_steps_raises(self):
        with pytest.raises(ValueError, match="empty"):
            dc.build_duty_cycle(power_steps=[])

    def test_invalid_rated_fraction_zero_raises(self):
        with pytest.raises(ValueError, match="rated_fraction"):
            dc.build_duty_cycle(rated_fraction=0.0)

    def test_invalid_rated_fraction_one_raises(self):
        with pytest.raises(ValueError, match="rated_fraction"):
            dc.build_duty_cycle(rated_fraction=1.0)

    def test_custom_power_steps(self):
        d = dc.build_duty_cycle(power_steps=[60, 100, 150])
        assert abs(d["fraction"].sum() - 1.0) < 1e-9
        assert len(d["power_mw"]) == 3

    def test_all_rated_power_steps_normalise(self):
        """When all steps are rated (150 MW), fractions must still sum to 1."""
        d = dc.build_duty_cycle(
            mode="uniform_with_rated",
            rated_fraction=0.25,
            power_steps=[150, 150],
        )
        assert abs(d["fraction"].sum() - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# compute_weighted_erosion
# ---------------------------------------------------------------------------

class TestComputeWeightedErosion:
    def _duty(self):
        return dc.build_duty_cycle()

    def test_output_shape(self):
        rng = np.random.default_rng(0)
        dhdt = rng.exponential(2.0, 500).astype(np.float32)
        r = dc.compute_weighted_erosion(dhdt, self._duty())
        assert r["dhdt_weighted_mm_h"].shape == (500,)

    def test_weighted_less_than_rated(self):
        rng = np.random.default_rng(0)
        dhdt = rng.exponential(2.0, 500).astype(np.float32)
        r = dc.compute_weighted_erosion(dhdt, self._duty())
        assert r["weighted_to_rated_ratio"] < 1.0, \
            "weighted mean must be below rated (load < rated most of the time)"

    def test_empty_dhdt_raises(self):
        with pytest.raises(ValueError, match="empty"):
            dc.compute_weighted_erosion(np.array([]), self._duty())

    def test_invalid_fraction_sum_raises(self):
        duty = dc.build_duty_cycle()
        bad_duty = dict(duty)
        bad_duty["fraction"] = duty["fraction"] * 2.0   # sum ≠ 1
        with pytest.raises(ValueError, match="fractions sum"):
            dc.compute_weighted_erosion(np.ones(100), bad_duty)

    def test_hotspot_iou_in_range(self):
        rng = np.random.default_rng(3)
        dhdt = rng.exponential(2.0, 1000).astype(np.float32)
        r = dc.compute_weighted_erosion(dhdt, self._duty())
        assert 0.0 <= r["hotspot_iou_rated_vs_weighted"] <= 1.0

    def test_zero_erosion_no_crash(self):
        """All-zero erosion array must not raise division-by-zero."""
        dhdt = np.zeros(200, dtype=np.float32)
        r = dc.compute_weighted_erosion(dhdt, self._duty())
        assert r["hotspot_iou_rated_vs_weighted"] == 0.0
        assert r["weighted_to_rated_ratio"] == 0.0

    def test_hotspot_iou_defined_when_all_nonzero(self):
        rng = np.random.default_rng(9)
        dhdt = rng.uniform(0.1, 5.0, 200).astype(np.float32)
        r = dc.compute_weighted_erosion(dhdt, self._duty())
        assert r["hotspot_iou_rated_vs_weighted"] >= 0.0


# ---------------------------------------------------------------------------
# run_weighted_erosion — NPZ handle closure
# ---------------------------------------------------------------------------

class TestRunWeightedErosionHandleClosure:
    def _make_npz(self, tmp: pathlib.Path) -> pathlib.Path:
        rng = np.random.default_rng(5)
        dhdt_raw = rng.exponential(2.0, 800).astype(np.float32)
        path = tmp / "fake_hits.npz"
        np.savez_compressed(str(path), dhdt_face_mm_h=dhdt_raw)
        return path

    def test_npz_handle_closed_after_call(self, tmp_path):
        """File handle must be closed; deleting the file after the call must succeed."""
        npz_path = self._make_npz(tmp_path)
        dc.run_weighted_erosion(npz_path)
        # On Windows an open handle blocks deletion — this will raise if not closed
        npz_path.unlink()

    def test_result_contains_expected_keys(self, tmp_path):
        npz_path = self._make_npz(tmp_path)
        r = dc.run_weighted_erosion(npz_path)
        for key in ("dhdt_weighted_mm_h", "max_weighted_mm_h",
                    "weighted_to_rated_ratio", "hotspot_iou_rated_vs_weighted"):
            assert key in r, f"missing key: {key}"
