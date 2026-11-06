from __future__ import annotations

from pathlib import Path

import pytest

from step30_fem_evidence_intake import (
    FEMEvidenceIntakeError,
    discover_official_sources,
    main,
    validate_manifest,
)
from step30_validation_common import canonical_sha256, safe_json_dump


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_official_dec_fem_evidence_is_frozen_and_replayable() -> None:
    package_root = PROJECT_ROOT / "DecPelton2024"
    if not package_root.is_dir():
        pytest.skip("official DEC package is not present")
    manifest = discover_official_sources(PROJECT_ROOT)
    assert manifest["status"] == "blocked"
    assert manifest["fem_gate"] == "blocked"
    assert len(manifest["artifacts"]) == 16
    assert manifest["claims"]["ai5_channel"]["value"] == "AI5-04"
    assert manifest["claims"]["sampling"]["value"] == pytest.approx(1000.0)
    assert manifest["claims"]["young_modulus"]["value"] == pytest.approx(206.0e9)
    assert manifest["claims"]["poisson_ratio"]["value"] == pytest.approx(0.3)
    assert manifest["claims"]["material_density"]["status"] == "blocked"
    assert all(record["n_samples"] == 1262 for record in manifest["measurements"]["records"])
    assert all(record["sample_frequency_hz"] == pytest.approx(1000.0) for record in manifest["measurements"]["records"])
    assert {item["id"] for item in manifest["remaining_blockers"]} >= {
        "ROOT_INTERFACE",
        "WETTED_SURFACE",
        "AI5_04_CAD_PATCH",
        "AI5_04_GAUGE_DIRECTIONS",
        "MATERIAL_DENSITY",
        "STEP_COORDINATE_UNITS",
    }


def test_manifest_validation_rejects_tampered_source(tmp_path: Path) -> None:
    package_root = PROJECT_ROOT / "DecPelton2024"
    if not package_root.is_dir():
        pytest.skip("official DEC package is not present")
    manifest = discover_official_sources(PROJECT_ROOT)
    source = Path(manifest["official_package_root"]) / manifest["artifacts"][0]["path"]
    assert source.is_file()
    tampered = dict(manifest)
    tampered_artifacts = [dict(item) for item in manifest["artifacts"]]
    tampered_artifacts[0]["sha256"] = "0" * 64
    tampered["artifacts"] = tampered_artifacts
    with pytest.raises(FEMEvidenceIntakeError, match="hash mismatch"):
        validate_manifest(tampered)


def test_validate_cli_classifies_missing_source_as_blocked(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest = {
        "version": "step30-fem-evidence-intake-v1",
        "status": "blocked",
        "official_package_root": str(tmp_path / "missing-source"),
        "artifacts": [],
        "measurements": {"status": "measured", "n_conditions": 0, "records": []},
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    path = tmp_path / "manifest.json"
    safe_json_dump(manifest, path)

    assert main(["--validate", str(path)]) == 3
    assert '"status": "blocked"' in capsys.readouterr().out
