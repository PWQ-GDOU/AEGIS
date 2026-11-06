from __future__ import annotations

import json

from step30_material_profile_compare_v1 import compare_profile


def test_profile_compare_blocks_without_measured_profile(tmp_path) -> None:
    prediction = tmp_path / "prediction.npz"
    prediction.write_bytes(b"prediction-artifact")
    output = tmp_path / "comparison.json"
    result = compare_profile(prediction, output_path=output)
    assert result["status"] == "blocked"
    assert result["official_accuracy_gate"] is False
    assert any("measured" in reason for reason in result["blocking_reasons"])
    assert json.loads(output.read_text(encoding="utf-8"))["report_sha256"] == result["report_sha256"]
