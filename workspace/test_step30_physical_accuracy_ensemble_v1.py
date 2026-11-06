from __future__ import annotations

import json

import numpy as np

from step30_physical_accuracy_ensemble_v1 import build_ensemble


def _prior(root, name: str, offset: float) -> str:
    report = {
        "status": "provisional",
        "prior_contract": {"ai5_used_for_calibration": False},
    }
    path = root / f"{name}.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    x = np.linspace(1.0 + offset, 8.0 + offset, 8)
    np.savez(
        path.with_suffix(".npz"),
        face_centers_local_mm=np.zeros((8, 3), dtype=np.float32),
        oka_p50_mm_h=x,
        finnie_p50_mm_h=x * 1.1,
        tabakoff_p50_mm_h=x * 0.9,
        mansouri_p50_mm_h=x * 1.2,
    )
    return str(path)


def test_ensemble_separates_seed_and_model_uncertainty(tmp_path) -> None:
    merged = _prior(tmp_path, "merged", 0.0)
    seeds = [_prior(tmp_path, "seed1", 0.0), _prior(tmp_path, "seed2", 0.2), _prior(tmp_path, "seed3", -0.1)]
    output = tmp_path / "ensemble.json"
    result = build_ensemble(merged, seeds, output_path=output)
    assert result["status"] == "provisional_physical_accuracy_ensemble"
    assert result["seed_uncertainty"]["n_seeds"] == 3
    assert "active_seed_cv_p95" in result["seed_uncertainty"]["models"]["oka"]
    assert result["model_uncertainty"]["models"] == ["oka", "finnie", "tabakoff", "mansouri"]
    assert result["official_accuracy_gate"] is False
    assert output.is_file() and output.with_suffix(".npz").is_file()
    with np.load(output.with_suffix(".npz"), allow_pickle=False) as data:
        assert "oka_seed_std" in data.files
        assert "model_disagreement_cv" in data.files
