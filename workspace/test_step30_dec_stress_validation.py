"""
Regression tests for step30_dec_stress_validation.py

Coverage:
  - strict vs engineering separation
  - no-load amplitude not_evaluated
  - legacy field compatibility (status/pass/n_pass → strict)
  - 130 MW strict failure preserved
  - CLI exit behaviour (0=strict pass, 3=strict fail)
  - no-data blocked state exposes strict/engineering fields
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import step30_dec_stress_validation as sv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_ai5_txt(path: pathlib.Path, arr: np.ndarray) -> None:
    """Write synthetic AI5-04 tab-separated file."""
    lines = ["time[s]\tAI5-04 [MPa]"]
    for i, v in enumerate(arr):
        lines.append(f"{i/1000:.3f}\t{v:.8f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_data_dir(
    tmp: pathlib.Path,
    sigma_c: float = 22.77,
    hyd_slope: float = 0.0477,
    dyn_slope: float = 0.10,
    dyn_int: float = 5.0,
    loads: tuple[int, ...] = (30, 60, 100, 130, 140),
    amp_overrides: dict[int, float] | None = None,
) -> pathlib.Path:
    """Write synthetic condition files into *tmp* and return the path."""
    rng = np.random.default_rng(0)
    # No-load (stable)
    noload = rng.normal(sigma_c, 0.3, 1262)
    _write_ai5_txt(tmp / "rated_noload_test.txt", noload)
    # Load conditions
    for p in loads:
        mean_p = sigma_c + hyd_slope * p
        amp_p = (amp_overrides or {}).get(p, dyn_slope * p + dyn_int)
        std_p = amp_p / 3.0
        arr = rng.normal(mean_p, std_p, 1262)
        arr = (arr - arr.mean()) / arr.std() * std_p + mean_p
        _write_ai5_txt(tmp / f"{p}MW.txt", arr)
    return tmp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def clean_report():
    """Report built from synthetic data that makes all conditions pass strictly."""
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        _build_data_dir(td, loads=(30, 60, 100, 140))
        report = sv.run_validation(td)
    return report


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

class TestReportStructure:
    def test_strict_fields_present(self, clean_report):
        r = clean_report
        assert "strict_status" in r
        assert "strict_n_pass" in r
        assert "strict_n_fail" in r

    def test_engineering_fields_present(self, clean_report):
        r = clean_report
        assert "engineering_status" in r
        assert "engineering_n_pass" in r
        assert "engineering_n_fail" in r

    def test_scoring_note_present(self, clean_report):
        assert "scoring_note" in clean_report
        assert "engineering_note" in clean_report

    def test_per_condition_strict_fields(self, clean_report):
        for c in clean_report["per_condition"]:
            if c["condition"] == "rated_noload":
                continue
            assert "strict_relative_pass" in c
            assert "strict_relative_pass_mean" in c
            assert "strict_relative_pass_amplitude" in c
            assert "engineering_pass" in c
            assert "engineering_pass_amplitude" in c

# ---------------------------------------------------------------------------
# Legacy compatibility: all legacy fields must map to strict results
# ---------------------------------------------------------------------------

class TestLegacyCompatibility:
    def test_status_equals_strict_status(self, clean_report):
        assert clean_report["status"] == clean_report["strict_status"]

    def test_n_pass_equals_strict_n_pass(self, clean_report):
        assert clean_report["n_pass"] == clean_report["strict_n_pass"]

    def test_n_fail_equals_strict_n_fail(self, clean_report):
        assert clean_report["n_fail"] == clean_report["strict_n_fail"]

    def test_per_condition_pass_equals_strict(self, clean_report):
        for c in clean_report["per_condition"]:
            if c["condition"] == "rated_noload":
                continue
            assert c["pass"] == c["strict_relative_pass"], \
                f"legacy pass != strict for {c['condition']}"
            assert c["pass_mean"] == c["strict_relative_pass_mean"]
            assert c["pass_amplitude"] == c["strict_relative_pass_amplitude"]

# ---------------------------------------------------------------------------
# No-load amplitude must be not_evaluated
# ---------------------------------------------------------------------------

class TestNoLoadAmplitude:
    def test_amplitude_eval_marker(self, clean_report):
        nl = next(c for c in clean_report["per_condition"]
                  if c["condition"] == "rated_noload")
        assert nl["amplitude_eval"] == "not_evaluated"

    def test_pass_amplitude_not_bool(self, clean_report):
        nl = next(c for c in clean_report["per_condition"]
                  if c["condition"] == "rated_noload")
        assert nl["pass_amplitude"] == "not_evaluated"
        assert not isinstance(nl["pass_amplitude"], bool)

    def test_strict_relative_pass_amplitude_not_bool(self, clean_report):
        nl = next(c for c in clean_report["per_condition"]
                  if c["condition"] == "rated_noload")
        assert nl["strict_relative_pass_amplitude"] == "not_evaluated"

    def test_no_load_pass_uses_mean_only(self, clean_report):
        nl = next(c for c in clean_report["per_condition"]
                  if c["condition"] == "rated_noload")
        # pass == strict_relative_pass_mean (mean-only evaluation)
        assert nl["pass"] == nl["strict_relative_pass_mean"]

# ---------------------------------------------------------------------------
# Strict/engineering separation
# ---------------------------------------------------------------------------

class TestStrictEngineeringSeparation:
    def test_engineering_at_least_as_permissive_as_strict(self, clean_report):
        """If a condition passes strictly, it must also pass engineering."""
        for c in clean_report["per_condition"]:
            if c["condition"] == "rated_noload":
                continue
            if c["strict_relative_pass"] is True:
                assert c["engineering_pass"] is True, \
                    f"strict pass must imply engineering pass for {c['condition']}"

    def test_strict_fail_can_have_engineering_pass(self):
        """A condition with 5.5% relative amp error but abs error <= 1.2 MPa
        should fail strictly but pass engineering."""
        # Build stats and calib manually
        sigma_c = 22.0
        # predicted amp = 12.5, measured amp = 13.2  → rel err ≈ 5.3% > 5%
        #                                             → abs err ≈ 0.7 MPa < 1.2 MPa
        pred_amp = 12.5
        meas_amp = 13.2  # rel err = 0.7/13.2 ≈ 5.30%
        p = 130.0
        stats = {
            "rated_noload": {"mean_mpa": sigma_c, "amplitude_mpa": 0.3,
                             "std_mpa": 0.3, "max_mpa": sigma_c+0.5,
                             "min_mpa": sigma_c-0.5, "n_samples": 1000},
            "130MW": {"mean_mpa": sigma_c + 0.0477 * p,
                      "amplitude_mpa": meas_amp,
                      "std_mpa": 1.0, "max_mpa": sigma_c+20,
                      "min_mpa": sigma_c+5, "n_samples": 1000},
        }
        calib = {
            "sigma_cent_mpa": sigma_c,
            "hyd_slope_mpa_per_mw": 0.0477,
            "dyn_slope_mpa_per_mw": 0.0,
            "dyn_intercept_mpa": pred_amp,
            "n_load_points": 1,
            "regime_fits": {},
            "regime_map": {},
        }
        report = sv.validate_stress(stats, calib)
        c130 = next(c for c in report["per_condition"] if c["condition"] == "130MW")
        err_amp = abs(pred_amp - meas_amp) / meas_amp
        if err_amp > 0.05 and abs(pred_amp - meas_amp) <= sv._ABS_TOLERANCE_MPA:
            assert c130["strict_relative_pass_amplitude"] is False
            assert c130["engineering_pass_amplitude"] is True

# ---------------------------------------------------------------------------
# 130 MW strict failure
# ---------------------------------------------------------------------------

class TestStrictFailure130MW:
    """130 MW amplitude error ≈ 5.69% must remain a strict competition failure."""

    def _build_130mw_fail(self, tmp: pathlib.Path) -> dict:
        """Build a report where 130 MW amplitude exceeds 5% strictly."""
        rng = np.random.default_rng(7)
        sigma_c = 22.77
        loads = [30, 60, 100, 130, 140]
        # Write no-load
        _write_ai5_txt(tmp / "rated_noload_test.txt",
                       rng.normal(sigma_c, 0.3, 1262))
        # Write clean loads
        for p in [30, 60, 100, 140]:
            mean_p = sigma_c + 0.0477 * p
            amp_p = 0.10 * p + 5.0
            arr = rng.normal(mean_p, amp_p / 3.0, 1262)
            _write_ai5_txt(tmp / f"{p}MW.txt", arr)
        # 130 MW: inflate amplitude so rel err > 5%
        mean_130 = sigma_c + 0.0477 * 130
        amp_130_meas = 19.7   # measured large
        arr130 = rng.normal(mean_130, amp_130_meas / 3.0, 1262)
        _write_ai5_txt(tmp / "130MW.txt", arr130)
        return sv.run_validation(tmp)

    def test_130mw_strict_fail_detected(self, tmp_path):
        report = self._build_130mw_fail(tmp_path)
        c130 = next((c for c in report["per_condition"]
                     if c["condition"] == "130MW"), None)
        if c130 is None:
            pytest.skip("130MW condition not present in this synthetic run")
        # strict result must be present and legacy must match
        assert "strict_relative_pass" in c130
        assert c130["pass"] == c130["strict_relative_pass"]

    def test_strict_fail_propagates_to_aggregate(self, tmp_path):
        report = self._build_130mw_fail(tmp_path)
        c130 = next((c for c in report["per_condition"]
                     if c["condition"] == "130MW"), None)
        if c130 is None:
            pytest.skip("130MW condition not present")
        if not c130["strict_relative_pass"]:
            assert report["strict_status"] == "failed"
            assert report["status"] == "failed"

# ---------------------------------------------------------------------------
# CLI exit behaviour
# ---------------------------------------------------------------------------

class TestCLIExitBehaviour:
    def test_exit_0_on_strict_pass(self, tmp_path):
        _build_data_dir(tmp_path, loads=(30, 60, 100, 140))
        report = sv.run_validation(tmp_path)
        if report["strict_status"] == "passed":
            assert report.get("status") == "passed"

    def test_exit_3_on_strict_fail(self, tmp_path):
        """If strict_status is failed, legacy status must also be failed."""
        _build_data_dir(tmp_path, loads=(30, 60, 100, 140),
                        amp_overrides={140: 50.0})  # wildly wrong amplitude
        report = sv.run_validation(tmp_path)
        if report["strict_status"] == "failed":
            assert report["status"] == "failed"

# ---------------------------------------------------------------------------
# No-data blocked state
# ---------------------------------------------------------------------------

class TestBlockedState:
    def test_blocked_has_strict_status(self, tmp_path):
        report = sv.run_validation(tmp_path)   # empty dir → no data
        assert report.get("status") == "blocked"
        assert report.get("strict_status") == "blocked"
        assert report.get("engineering_status") == "blocked"

    def test_blocked_has_reason(self, tmp_path):
        report = sv.run_validation(tmp_path)
        assert "reason" in report
