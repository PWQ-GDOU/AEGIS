from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from step30_measured_validation import validate_measured
from step30_measurement_acceptance import evaluate_acceptance, replay_acceptance
from step30_validation_common import canonical_sha256, sha256_file


TRANSFORM_SHA256 = "1" * 64
TOPOLOGY_SHA256 = "2" * 64


def _seal(document: dict[str, Any], field: str) -> dict[str, Any]:
    document[field] = canonical_sha256(document)
    return document


def _evidence(tmp_path: Path, *, scale: float = 1.01) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    entries: dict[str, Any] = {}
    source_hashes: dict[str, str] = {}
    for record_id in ("r-test", "r-train"):
        path = tmp_path / f"{record_id}.json"
        path.write_text(json.dumps({"record_id": record_id}), encoding="utf-8")
        digest = sha256_file(path)
        source_hashes[record_id] = digest
        entries[record_id] = {
            "path": path.name,
            "sha256": digest,
            "bytes": path.stat().st_size,
            "status": "measured",
            "metadata": {"record_id": record_id, "kind": "wear_depth", "units": "mm", "schema": "face-v1"},
        }
    source = _seal({
        "version": "source-v1",
        "status": "measured",
        "provenance": {
            "origin": "laboratory_measurement",
            "acquisition_method": "registered profilometry",
            "acquired_at": "2026-08-03T00:00:00Z",
            "organization": "independent laboratory",
        },
        "entries": entries,
    }, "manifest_sha256")
    split = _seal({
        "version": "split-v1",
        "status": "measured",
        "pre_registered_core_metrics": [
            {"metric": "wear_depth_mm", "total_relative_error_max": 0.05},
        ],
        "pre_registered_hotspot": {
            "metric": "wear_depth_mm",
            "percentile": 75.0,
            "area_relative_error_max": 0.05,
            "iou_error_max": 0.05,
        },
        "records": [
            {"record_id": "r-test", "split": "independent_test", "group_id": "g-test", "source_sha256": source_hashes["r-test"]},
            {"record_id": "r-train", "split": "train", "group_id": "g-train", "source_sha256": source_hashes["r-train"]},
        ],
    }, "manifest_sha256")

    def record(record_id: str, values: list[float]) -> dict[str, Any]:
        return {
            "record_id": record_id,
            "status": "measured",
            "source_sha256": source_hashes[record_id],
            "transform_sha256": TRANSFORM_SHA256,
            "topology_sha256": TOPOLOGY_SHA256,
            "specimen_id": f"s-{record_id}",
            "campaign_id": "campaign-1",
            "wear_depth_mm": {
                "values": values,
                "units": "mm",
                "parent_face_id": [0, 1, 2, 3],
                "n_parent_faces": 4,
                "cohort_id": "bucket-parent-faces-v3",
                "area_mm2": {"values": [1.0, 2.0, 3.0, 4.0], "units": "mm2"},
            },
        }

    observed = [1.0, 2.0, 3.0, 4.0]
    prediction = {
        "status": "measured",
        "records": [record("r-test", [value * scale for value in observed]), record("r-train", observed)],
    }
    observation = {
        "status": "measured",
        "records": [record("r-test", observed), record("r-train", observed)],
    }
    bindings = {
        "source_manifest_sha256": source["manifest_sha256"],
        "protocol_sha256": split["manifest_sha256"],
        "transform_sha256": TRANSFORM_SHA256,
        "topology_sha256": TOPOLOGY_SHA256,
    }
    gates: dict[str, Any] = {}
    for gate_name in ("registration", "geometry", "measurement"):
        gates[gate_name] = _seal({
            "version": f"{gate_name}-quality-v1",
            "status": "measured",
            "bindings": bindings,
            "thresholds": {"coverage": {"minimum": 0.95}},
            "actuals": {"coverage": 0.99},
            "passed": False,
        }, "report_sha256")
    quality = _seal({"version": "quality-bundle-v1", "status": "measured", **gates}, "quality_sha256")
    return {
        "prediction": prediction,
        "observation": observation,
        "source": source,
        "split": split,
        "quality": quality,
        "source_root": tmp_path,
    }


def _validate(evidence: dict[str, Any]) -> dict[str, Any]:
    return validate_measured(
        evidence["prediction"], evidence["observation"], evidence["source"],
        evidence["split"], evidence["quality"], source_root=evidence["source_root"],
        bootstrap_samples=0,
    )


def _replay(evidence: dict[str, Any]) -> dict[str, Any]:
    return replay_acceptance(
        evidence["prediction"], evidence["observation"], evidence["source"],
        evidence["split"], evidence["quality"], source_root=evidence["source_root"],
        bootstrap_samples=0,
    )


def test_parent_faces_are_aligned_by_id_not_array_position(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    metric = evidence["prediction"]["records"][0]["wear_depth_mm"]
    order = [2, 0, 3, 1]
    for key in ("values", "parent_face_id"):
        metric[key] = [metric[key][index] for index in order]
    metric["area_mm2"]["values"] = [metric["area_mm2"]["values"][index] for index in order]

    report = _validate(evidence)

    assert report["status"] == "passed"
    assert report["metrics"]["wear_depth_mm"]["total_relative_error"] == pytest.approx(0.01)


def test_parent_face_set_mismatch_is_blocked(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    metric = evidence["prediction"]["records"][0]["wear_depth_mm"]
    metric["parent_face_id"] = [0, 1, 2, 4]
    metric["n_parent_faces"] = 5

    report = _validate(evidence)

    assert report["status"] == "blocked"
    assert report["official_accuracy_gate"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda metric: metric.update(parent_face_id=[0, 1, 1, 3]),
        lambda metric: metric.update(parent_face_id=[0, 1, 2, 4]),
        lambda metric: metric.update(units="kg"),
        lambda metric: metric["area_mm2"].update(units="m2"),
        lambda metric: metric["area_mm2"]["values"].__setitem__(0, 9.0),
        lambda metric: metric.pop("area_mm2"),
    ],
)
def test_invalid_spatial_identity_units_or_area_is_blocked(tmp_path: Path, mutation: Any) -> None:
    evidence = _evidence(tmp_path)
    mutation(evidence["prediction"]["records"][0]["wear_depth_mm"])

    assert _validate(evidence)["status"] == "blocked"


def test_forged_quality_bool_is_blocked(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    evidence["quality"] = {
        "status": "measured",
        "passed": True,
        "registration": {"passed": True},
        "geometry": {"passed": True},
        "measurement": {"passed": True},
    }

    report = _validate(evidence)

    assert report["status"] == "blocked"
    assert report["official_accuracy_gate"] is False


def test_self_hashed_validation_report_is_diagnostic_only(tmp_path: Path) -> None:
    validation = _validate(_evidence(tmp_path))
    assert validation["status"] == "passed"

    acceptance = evaluate_acceptance(validation)

    assert acceptance["status"] == "blocked"
    assert acceptance["diagnostic_only"] is True
    assert acceptance["official_accuracy_gate"] is False
    assert acceptance["official_reference_gate"] is False


def test_valid_high_error_is_failed_not_blocked(tmp_path: Path) -> None:
    acceptance = _replay(_evidence(tmp_path, scale=1.20))

    assert acceptance["status"] == "failed"
    assert acceptance["evaluation_status"] == "failed"
    assert acceptance["blocked"] is False
    assert acceptance["official_accuracy_gate"] is False
    assert acceptance["official_reference_gate"] is False


def test_invalid_quality_hash_is_blocked(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    evidence["quality"]["registration"]["actuals"]["coverage"] = 0.98

    acceptance = _replay(evidence)

    assert acceptance["status"] == "blocked"
    assert acceptance["official_accuracy_gate"] is False


def test_valid_low_error_replay_passes_as_measured_not_reference(tmp_path: Path) -> None:
    acceptance = _replay(_evidence(tmp_path, scale=1.01))

    assert acceptance["status"] == "passed"
    assert acceptance["evaluation_status"] == "passed"
    assert acceptance["evidence_status"] == "measured"
    assert acceptance["accepted"] is True
    assert acceptance["official_accuracy_gate"] is True
    assert acceptance["official_reference_gate"] is False


def test_zero_reference_and_synthetic_status_fail_closed(tmp_path: Path) -> None:
    zero = _evidence(tmp_path / "zero")
    zero["observation"]["records"][0]["wear_depth_mm"]["values"][0] = 0.0
    assert _validate(zero)["status"] == "blocked"

    synthetic = _evidence(tmp_path / "synthetic")
    synthetic["prediction"]["status"] = "synthetic"
    synthetic["prediction"]["records"][0]["status"] = "synthetic"
    report = _validate(synthetic)
    assert report["status"] == "blocked"
    assert report["evidence_status"] == "synthetic"
