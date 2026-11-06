from __future__ import annotations

import json
from pathlib import Path

import pytest

from step30_supplemental_data_validator import SupplementalDataError, validate_manifest


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "input" / "supplemental_data" / "step30_supplemental_data_manifest_v1.json"


def _load() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_validates_source_hashes_and_gates() -> None:
    report = validate_manifest(_load(), root=ROOT)
    assert report["status"] == "valid"
    assert report["source_count"] == 7
    assert report["official_accuracy_gate"] is False
    assert report["official_reference_gate"] is False
    assert report["fem_gate"] == "blocked"


def test_manifest_rejects_hash_tampering() -> None:
    document = _load()
    document["source_files"][0]["sha256"] = "0" * 64
    with pytest.raises(SupplementalDataError, match="hash mismatch"):
        validate_manifest(document, root=ROOT)


def test_manifest_rejects_c_drive_source_path() -> None:
    document = _load()
    document["source_files"][0]["path"] = "C:/forbidden/source.pdf"
    with pytest.raises(SupplementalDataError, match="workspace-relative"):
        validate_manifest(document, root=ROOT)


def test_manifest_rejects_gate_promotion() -> None:
    document = _load()
    document["gate_effect"]["official_accuracy_gate"] = True
    with pytest.raises(SupplementalDataError, match="official_accuracy_gate"):
        validate_manifest(document, root=ROOT)
