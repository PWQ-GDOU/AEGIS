from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from run_step30_acceptance_gates import aggregate
from step30_external_measured_calibration import build_calibration_manifest, trend_report
from step30_official_reference_simulation import validate_official_reference
from step30_validate_against_measurements import run as legacy_validate
from step30_validation_common import canonical_sha256, evidence_metadata, sha256_file


def _numerical_passes() -> dict[str, dict]:
    return {name: {"status": "passed"} for name in ("artifact", "collision", "cfd", "sensitivity", "comparison")}


def test_evidence_taxonomy_keeps_local_synthetic_ineligible() -> None:
    metadata = evidence_metadata({
        "status": "synthetic",
        "source_type": "local_synthetic_fixture",
        "reference_role": "synthetic_pipeline_fixture",
        "pipeline_validation": True,
    })
    assert metadata["official_reference"] is False
    result = aggregate(**_numerical_passes(), measurement={"status": "synthetic", **metadata}, production=True)
    assert result["status"] == "blocked"
    assert result["official_reference_gate"] is False


def test_self_hashed_minimal_reference_report_cannot_pass() -> None:
    reference = {
        "status": "passed",
        "official_reference": True,
        "official_reference_gate": True,
        "reference_role": "official_reference_simulation",
    }
    reference["report_sha256"] = canonical_sha256(reference)
    result = aggregate(
        **_numerical_passes(),
        measurement=None,
        reference_simulation=reference,
        production=True,
    )
    assert result["status"] == "blocked"
    assert result["official_reference_gate"] is False
    assert result["official_accuracy_gate"] is False


def test_external_calibration_never_becomes_bucket_accuracy() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        data = root / "external.csv"
        data.write_text("speed_m_s,wear_loss_mg\n2.5,1.2\n3.0,1.8\n", encoding="utf-8")
        manifest = build_calibration_manifest(
            data,
            title="External slurry test",
            doi="10.0000/example",
            license="CC BY 4.0",
            material="AISI 1018",
            limitations=("no bucket geometry", "no parent-face mapping"),
        )
        report = trend_report(manifest)
        result = aggregate(**_numerical_passes(), measurement=report, production=True)
        assert report["official_accuracy_gate"] is False
        assert result["external_physical_validation_status"] == "calibration_only"
        assert result["status"] == "blocked"


def test_legacy_array_flags_cannot_pass_measured_gate() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        predicted = root / "predicted.npy"
        observed = root / "observed.npy"
        output = root / "report.json"
        np.save(predicted, np.asarray([1.0, 2.0, 3.0]))
        np.save(observed, np.asarray([1.0, 2.0, 3.0]))
        report = legacy_validate(
            predicted,
            observed,
            output,
            predicted_key="predicted_depth_mm",
            observed_key="wear_depth_mm",
            status="measured",
            independent=True,
            quality_passed=True,
        )
        assert report["official_accuracy_gate"] is False
        assert report["legacy_gate_blocked"] is True
        unsigned = dict(report)
        digest = unsigned.pop("metrics_sha256")
        assert digest == canonical_sha256(unsigned)


def test_reference_manifest_hash_tamper_blocks() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        artifact = root / "reference.json"
        artifact.write_text(json.dumps({"records": [{"record_id": "f1", "value": 1.0}]}), encoding="utf-8")
        manifest = {
            "version": "step30-official-reference-manifest-v1",
            "status": "synthetic",
            "source_type": "contest_provided_simulation",
            "reference_role": "official_reference_simulation",
            "pipeline_validation": False,
            "official_reference": True,
            "contest": {"competition": "test", "organizer": "test", "dataset_id": "d1"},
            "source_root": str(root),
            "artifacts": [{"path": artifact.name, "sha256": sha256_file(artifact)}],
            "solver_semantics": {"application": "pimpleFoam"},
            "topology": {"mesh_signature": "mesh"},
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        artifact.write_text("tampered", encoding="utf-8")
        result = validate_official_reference(
            [{"record_id": "f1", "value": 1.0}],
            [{"record_id": "f1", "value": 1.0}],
            manifest,
        )
        assert result["status"] == "blocked"
        assert result["official_reference_gate"] is False
