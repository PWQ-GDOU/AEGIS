"""Regression tests for the Step30 experiment and validation extensions."""
from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

import numpy as np

from run_step30_acceptance_gates import aggregate
from step30_collision_validation import run as run_collision_validation
from step30_error_metrics import calculate_error_metrics
from step30_erosion_mechanism_classifier import ClassifierError, load_inputs
from step30_measurement_io import read_point_cloud
from step30_validation_common import ValidationError, safe_json_dump


def test_complex_values_are_rejected_before_cast() -> None:
    from step30_validation_common import finite_array
    try:
        finite_array(np.asarray([1 + 2j]), "complex")
    except ValidationError:
        pass
    else:
        raise AssertionError("complex values were silently accepted")


def test_binary_stl_attribute_bytes_are_skipped() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "triangle.stl"
        header = b"binary test".ljust(80, b" ")
        record = struct.pack("<3f", 0.0, 0.0, 1.0)
        record += struct.pack("<9f", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        record += struct.pack("<H", 0)
        path.write_bytes(header + struct.pack("<I", 1) + record)
        result = read_point_cloud(path, status="measured", units="mm")
        assert result["points"].shape == (3, 3)
        assert np.allclose(result["points"][1], [1.0, 0.0, 0.0])


def test_synthetic_error_never_passes_official_gate() -> None:
    result = calculate_error_metrics(
        [1.0, 2.0], [1.0, 2.0], status="synthetic", value_units="mm",
        independent_validation_set={"is_independent": True, "n": 2, "source_sha256": "a" * 64},
        quality_passed=True,
    )
    assert result["official_accuracy_gate"] is False


def test_collision_validation_passes_and_has_no_physical_fallback() -> None:
    result = run_collision_validation()
    assert result["status"] == "passed"
    assert result["physical_fallback_hits"] == 0


def test_acceptance_gate_remains_blocked_without_measured_validation() -> None:
    passed = {"status": "passed"}
    result = aggregate(
        artifact=passed, collision=passed, cfd=passed, sensitivity=passed,
        comparison=passed, measurement={"status": "blocked"}, production=True,
    )
    assert result["status"] == "blocked"
    assert result["numerical_status"] == "passed"
    assert result["official_accuracy_gate"] is False


def test_classifier_rejects_mismatched_summary_count() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        hits = root / "hits.npz"
        summary = root / "summary.json"
        np.savez(hits, position_local_mm=np.zeros((2, 3)), Vp_ms=np.ones(2), parent_face_id=np.zeros(2, dtype=np.int64))
        summary.write_text(json.dumps({"tracking": {"n_impacts": 3}}), encoding="utf-8")
        try:
            load_inputs(hits, summary)
        except ClassifierError:
            pass
        else:
            raise AssertionError("mismatched NPZ/summary count was accepted")


def test_immutable_json_writer_does_not_overwrite() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "record.json"
        safe_json_dump({"value": 1}, path)
        try:
            safe_json_dump({"value": 2}, path)
        except FileExistsError:
            pass
        else:
            raise AssertionError("immutable output was overwritten")
