from __future__ import annotations

import json

import numpy as np
import pytest

from step30_competition_self_benchmark_v1 import (
    CompetitionBenchmarkError,
    build_self_benchmark,
)


def _prior(root, name: str, offset: float) -> str:
    report = {
        "status": "provisional",
        "prior_contract": {"ai5_used_for_calibration": False},
    }
    path = root / f"{name}.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    centers = np.arange(18, dtype=np.float64).reshape(6, 3)
    areas = np.full(6, 1.0e-4, dtype=np.float64)
    base = np.array([0.0, 1.0, 2.0, 0.0, 3.0, 4.0], dtype=np.float64) + offset
    base[base < 0.0] = 0.0
    np.savez(
        path.with_suffix(".npz"),
        face_centers_local_mm=centers,
        face_area_m2=areas,
        oka_p50_mm_h=base,
        finnie_p50_mm_h=base * 0.9 + 0.01,
        tabakoff_p50_mm_h=base * 1.1,
        mansouri_p50_mm_h=base * 0.8 + 0.02,
    )
    return str(path)


def _literature(root) -> str:
    path = root / "literature.json"
    path.write_text(
        json.dumps(
            {
                "reference_kind": "literature_reference",
                "official_reference_gate": False,
                "regions": [
                    {"region_id": "splitter_like", "literature_support_score": 3},
                    {"region_id": "cutout_like", "literature_support_score": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def test_self_benchmark_is_conditional_and_preserves_official_gates(tmp_path) -> None:
    merged = _prior(tmp_path, "merged", 0.0)
    seeds = [_prior(tmp_path, f"seed_{index}", 0.02 * index) for index in range(3)]
    output = tmp_path / "benchmark.json"

    result = build_self_benchmark(merged, seeds, _literature(tmp_path), output_path=output)

    assert result["benchmark_kind"] == "competition_self_benchmark"
    assert result["reference_independence"] == "leave_one_model_out_and_leave_one_seed_out"
    assert result["competition_self_benchmark_status"] == "conditional"
    assert result["competition_self_benchmark_gate"] is True
    assert result["scoring_policy"]["official_data_absence_numeric_deduction"] == 0.0
    assert result["deductions_and_boundaries"][0]["effect"] == "no_numeric_deduction"
    assert result["official_accuracy_gate"] is False
    assert result["official_reference_gate"] is False
    assert result["fem_gate"] == "blocked"
    assert 0.0 <= result["conservative_score"] <= 100.0
    assert result["conservative_score"] <= result["optimistic_score"]
    assert set(result["model_scores"]) == {"oka", "finnie", "tabakoff", "mansouri"}
    assert output.is_file() and output.with_suffix(".npz").is_file()
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["report_sha256"] == result["report_sha256"]
    with np.load(output.with_suffix(".npz"), allow_pickle=False) as data:
        assert "reference_region_prior" in data.files
        assert data["reference_depth_envelope"].shape == (3, 6)
        assert "reference_depth_envelope_low__oka" in data.files
        assert "reference_nominal_worn_field__mansouri__seed_2" in data.files


def test_duplicate_seed_paths_are_rejected(tmp_path) -> None:
    merged = _prior(tmp_path, "merged", 0.0)
    seed = _prior(tmp_path, "seed", 0.0)
    with pytest.raises(CompetitionBenchmarkError, match="unique"):
        build_self_benchmark(merged, [seed, seed], _literature(tmp_path))


def test_domain_mismatch_is_rejected(tmp_path) -> None:
    merged = _prior(tmp_path, "merged", 0.0)
    seed = _prior(tmp_path, "seed", 0.0)
    seed_npz = tmp_path / "seed.npz"
    with np.load(seed_npz, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    arrays["face_centers_local_mm"] = arrays["face_centers_local_mm"].copy()
    arrays["face_centers_local_mm"][0, 0] += 1.0
    np.savez(seed_npz, **arrays)
    with pytest.raises(CompetitionBenchmarkError, match="face domain"):
        build_self_benchmark(merged, [seed, _prior(tmp_path, "seed2", 0.0)], _literature(tmp_path))


def test_official_reference_cannot_be_substituted(tmp_path) -> None:
    merged = _prior(tmp_path, "merged", 0.0)
    seeds = [_prior(tmp_path, "seed1", 0.0), _prior(tmp_path, "seed2", 0.0)]
    literature = tmp_path / "official.json"
    literature.write_text(
        json.dumps(
            {
                "reference_kind": "literature_reference",
                "official_reference_gate": True,
                "regions": [{"region_id": "splitter_like", "literature_support_score": 1}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CompetitionBenchmarkError, match="official"):
        build_self_benchmark(merged, seeds, literature)
