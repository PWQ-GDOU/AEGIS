from __future__ import annotations

import json
from pathlib import Path

import pytest

from step30_fem_selection_intake import (
    EXPECTED_STEP_SHA256,
    FEMSelectionIntakeError,
    validate_selection_package,
)
from step30_validation_common import canonical_sha256, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP_PATH = PROJECT_ROOT / "workspace" / "output" / "step30_fem_remote_export_v1" / "output" / "single_bucket_target_authoritative.step"


def _artifact(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _package(tmp_path: Path) -> dict[str, object]:
    if not STEP_PATH.is_file():
        pytest.skip("recovered authoritative STEP is not present")
    evidence = tmp_path / "cad_selection_report.json"
    evidence.write_text("independent CAD selection review\n", encoding="ascii")
    certificate = tmp_path / "material_certificate.pdf"
    certificate.write_bytes(b"material certificate fixture\n")
    selections = {
        "ROOT_INTERFACE": {
            "status": "passed",
            "confirmation_method": "cad_named_selection",
            "independent_confirmation": True,
            "entity_dimension": 2,
            "entity_tags": [1, 2],
            "evidence_files": [_artifact(evidence)],
        },
        "WETTED_SURFACE": {
            "status": "passed",
            "confirmation_method": "independent_cad_review",
            "independent_confirmation": True,
            "entity_dimension": 2,
            "entity_tags": [3, 4, 5],
            "excludes_root_interface": True,
            "evidence_files": [_artifact(evidence)],
        },
        "AI5_04": {
            "status": "passed",
            "confirmation_method": "independent_cad_review",
            "independent_confirmation": True,
            "entity_dimension": 2,
            "entity_tags": [6],
            "patch_center": {"coordinates": [1.0, 2.0, 3.0], "units": "m"},
            "direction_frame": "step_global",
            "gauge_directions": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "outside_fixed_root": True,
            "evidence_files": [_artifact(evidence)],
        },
        "MATERIAL_DOMAIN": {
            "status": "passed",
            "confirmation_method": "cad_named_selection",
            "independent_confirmation": True,
            "entity_dimension": 3,
            "entity_tags": [1],
            "evidence_files": [_artifact(evidence)],
        },
    }
    package: dict[str, object] = {
        "version": "step30-fem-selection-material-intake-v1",
        "status": "passed",
        "geometry": {"coordinate_units": "m", "artifact": _artifact(STEP_PATH)},
        "selections": selections,
        "material": {
            "grade": "fixture-grade",
            "heat_or_batch": "fixture-heat",
            "source_kind": "material_certificate",
            "density_kg_m3": 7850.0,
            "source_artifact": _artifact(certificate),
        },
    }
    package["package_sha256"] = canonical_sha256(package)
    return package


def test_valid_selection_and_material_package_is_ready_for_s0(tmp_path: Path) -> None:
    package = validate_selection_package(_package(tmp_path))
    assert package["package_sha256"]
    assert package["geometry"]["artifact"]["sha256"] == EXPECTED_STEP_SHA256
    assert package["material"]["density_kg_m3"] == pytest.approx(7850.0)
    assert package["selections"]["AI5_04"]["gauge_directions"][0] == pytest.approx([1.0, 0.0, 0.0])


def test_heuristic_surface_tags_are_rejected(tmp_path: Path) -> None:
    package = _package(tmp_path)
    package["selections"]["ROOT_INTERFACE"]["confirmation_method"] = "heuristic_surface_tag"
    unsigned = dict(package)
    unsigned.pop("package_sha256")
    package["package_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(FEMSelectionIntakeError, match="admissible CAD confirmation"):
        validate_selection_package(package)


def test_missing_density_certificate_is_rejected(tmp_path: Path) -> None:
    package = _package(tmp_path)
    package["material"]["density_kg_m3"] = None
    unsigned = dict(package)
    unsigned.pop("package_sha256")
    package["package_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(FEMSelectionIntakeError, match="density_kg_m3"):
        validate_selection_package(package)
