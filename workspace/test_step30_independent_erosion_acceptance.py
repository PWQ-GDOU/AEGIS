"""End-to-end acceptance tests for pinned independent erosion evidence."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from step30_independent_erosion_protocol import (
    build_prediction_commitment,
    build_protocol,
    recompute_commitment_hash,
    save_prediction_commitment,
    save_protocol,
    source_file_descriptor,
)
from step30_independent_erosion_truth import (
    build_independent_erosion_truth,
    build_reveal_receipt,
    recompute_reveal_receipt_hash,
    recompute_truth_hash,
    save_independent_erosion_truth,
    save_reveal_receipt,
)
from step30_measurement_acceptance import (
    build_truth_build_contract,
    replay_independent_erosion_acceptance,
)
from step30_measurement_parent_face_mapping import (
    TOPOLOGY_V3_SIGNATURE_ALGORITHM,
    TOPOLOGY_V3_VERSION,
    _mesh_signature,
)
from step30_validation_common import canonical_sha256, safe_json_dump, sha256_file


ARTIFACT_SET_ID = "a" * 64
SOURCE_VTP_SHA256 = "b" * 64
REGISTRATION_OPTIONS = {
    "distance_threshold_mm": 0.2,
    "min_coverage": 0.75,
    "trim_fraction": 0.0,
    "max_iterations": 20,
}
MAXIMUM_PROJECTION_DISTANCE_MM = 0.5
_DEFAULT_PIN = object()


def _topology(path: Path) -> tuple[str, str]:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 2], [4, 5, 6]], dtype=np.int32)
    parent_ids = np.array([0, 1], dtype=np.int32)
    xyz = points[triangles]
    centers = xyz.mean(axis=1)
    area_vectors = 0.5 * np.cross(
        xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]
    )
    area_mm2 = np.linalg.norm(area_vectors, axis=1)
    normals = area_vectors / area_mm2[:, None]
    radii = np.linalg.norm(xyz - centers[:, None, :], axis=2).max(axis=1)
    signature = _mesh_signature(points, triangles, parent_ids, "mm")
    np.savez_compressed(
        path,
        points_local_mm=points,
        triangles_local_mm=triangles,
        triangle_parent_face_id=parent_ids,
        face_centers_local_mm=centers,
        face_normals_local=normals,
        face_area_m2=area_mm2 * 1.0e-6,
        triangle_centers_local_mm=centers,
        triangle_normals_local=normals,
        triangle_radii_mm=radii,
        face_id=np.arange(2, dtype=np.int32),
        parent_face_id=np.arange(2, dtype=np.int32),
        mesh_signature=np.array(signature),
        mesh_signature_algorithm=np.array(TOPOLOGY_V3_SIGNATURE_ALGORITHM),
        source_vtp_sha256=np.array(SOURCE_VTP_SHA256),
        artifact_version=np.array(TOPOLOGY_V3_VERSION),
        source_coordinate_units=np.array("mm"),
        internal_coordinate_units=np.array("mm"),
        coordinate_scale_to_mm=np.array(1.0),
    )
    return signature, sha256_file(path)


def _write_thickness(path: Path, points_mm: np.ndarray, values_mm: np.ndarray) -> None:
    table = np.column_stack((points_mm, values_mm * 1000.0))
    np.savetxt(
        path,
        table,
        delimiter=",",
        header="x,y,z,thickness",
        comments="",
    )


def _rewrite(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, ensure_ascii=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _evidence(root: Path, *, prediction_scale: float = 1.02) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    topology_path = root / "topology-v3.npz"
    mesh_signature, topology_hash = _topology(topology_path)
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    baseline_values = np.full(len(points), 8.0)
    wear_values = np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
    assets: list[dict[str, Any]] = []
    raw_sources: dict[str, tuple[Path, Path]] = {}
    for index in range(3):
        record_id = f"record-{index}"
        baseline = root / f"baseline-{index}.csv"
        worn = root / f"worn-{index}.csv"
        _write_thickness(baseline, points, baseline_values)
        _write_thickness(worn, points, baseline_values - wear_values)
        raw_sources[record_id] = (baseline, worn)
        assets.append(
            {
                "record_id": record_id,
                "asset_id": f"asset-{index}",
                "component_id": "runner-bucket",
                "service_episode_id": f"episode-{index}",
                "campaign_id": f"campaign-{index}",
                "scan_id": f"scan-{index}",
                "status": "measured",
                "source": source_file_descriptor(baseline),
                "truth_source": source_file_descriptor(worn),
            }
        )

    protocol = build_protocol(assets, seed=30)
    record_id = protocol["splits"]["independent_test"][0]
    baseline, worn = raw_sources[record_id]
    truth_records = [
        {
            "record_id": record_id,
            "record_source": "sparse_thickness",
            "status": "measured",
            "source_type": "ultrasonic_thickness",
            "baseline_path": str(baseline),
            "worn_path": str(worn),
            "coordinate_units": "mm",
            "thickness_units": "um",
            # Only the first parent face is explicitly measured; face 1 stays uncovered.
            "support_radius_mm": 1.5,
        }
    ]
    build_config = build_truth_build_contract(
        topology_path=topology_path,
        topology_v3_sha256=topology_hash,
        mesh_signature=mesh_signature,
        artifact_set_id=ARTIFACT_SET_ID,
        registration_options=REGISTRATION_OPTIONS,
        maximum_projection_distance_mm=MAXIMUM_PROJECTION_DISTANCE_MM,
    )
    predictions = [
        {
            "record_id": record_id,
            "parent_face_id": face_id,
            "prediction_value": expected * prediction_scale,
            "value_units": "mm",
            "parent_face_area_mm2": 0.5,
            "area_units": "mm2",
        }
        for face_id, expected in ((0, 0.1),)
    ]
    commitment = build_prediction_commitment(
        protocol,
        {
            "model_id": "frozen-pre-reveal-model",
            "truth_build_contract_sha256": build_config[
                "truth_build_contract_sha256"
            ],
        },
        predictions,
        topology_binding={
            "topology_v3_sha256": topology_hash,
            "mesh_signature": mesh_signature,
            "artifact_set_id": ARTIFACT_SET_ID,
            "n_parent_faces": 2,
        },
    )

    protocol_path = root / "protocol.json"
    commitment_path = root / "commitment.json"
    truth_records_path = root / "truth-records.json"
    save_protocol(protocol, protocol_path)
    save_prediction_commitment(commitment, commitment_path, protocol=protocol)
    safe_json_dump({"records": truth_records}, truth_records_path)

    # Truth is opened only after the predictions and build contract are committed.
    truth = build_independent_erosion_truth(
        protocol,
        truth_records,
        topology_path=topology_path,
        topology_v3_sha256=topology_hash,
        mesh_signature=mesh_signature,
        artifact_set_id=ARTIFACT_SET_ID,
        registration_options=REGISTRATION_OPTIONS,
        maximum_projection_distance_mm=MAXIMUM_PROJECTION_DISTANCE_MM,
    )
    truth_path = root / "truth.json"
    save_independent_erosion_truth(truth, truth_path, protocol=protocol)
    receipt = build_reveal_receipt(
        protocol,
        commitment,
        truth,
        topology_path=topology_path,
    )
    receipt_path = root / "receipt.json"
    save_reveal_receipt(receipt, receipt_path)
    return {
        "protocol": protocol,
        "commitment": commitment,
        "truth": truth,
        "receipt": receipt,
        "protocol_path": protocol_path,
        "commitment_path": commitment_path,
        "truth_path": truth_path,
        "receipt_path": receipt_path,
        "truth_records_path": truth_records_path,
        "topology_path": topology_path,
        "build_config": build_config,
        "worn_path": worn,
    }


def _replay(
    evidence: dict[str, Any], *, trusted_pin: Any = _DEFAULT_PIN
) -> dict[str, Any]:
    if trusted_pin is _DEFAULT_PIN:
        trusted_pin = evidence["commitment"]["commitment_sha256"]
    return replay_independent_erosion_acceptance(
        evidence["protocol_path"],
        evidence["commitment_path"],
        evidence["truth_path"],
        evidence["receipt_path"],
        trusted_commitment_sha256=trusted_pin,
        trusted_protocol_sha256=evidence["protocol"]["protocol_sha256"],
        truth_records=evidence["truth_records_path"],
        build_config=evidence["build_config"],
    )


def test_persisted_pre_reveal_evidence_passes(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    assert all(
        evidence[name].is_file()
        for name in (
            "protocol_path",
            "commitment_path",
            "truth_path",
            "receipt_path",
        )
    )

    report = _replay(evidence)

    assert report["status"] == "passed"
    assert report["accepted"] is True
    assert report["independent_accuracy_gate"] is True
    assert report["official_accuracy_gate"] is False
    assert report["production_acceptance_status"] == "blocked"
    assert report["official_reference_gate"] is False
    assert report["n_explicit_measurements"] == 1
    assert report["relative_errors"]["wear_depth_mm"] == pytest.approx(0.02)
    assert report["relative_errors"]["total_volume_mm3"] == pytest.approx(0.02)
    assert set(report["artifact_hashes"]) == {
        "protocol_sha256",
        "commitment_sha256",
        "truth_sha256",
        "receipt_sha256",
    }


def test_mapping_artifacts_follow_the_same_formal_replay(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)

    report = replay_independent_erosion_acceptance(
        evidence["protocol"],
        evidence["commitment"],
        evidence["truth"],
        evidence["receipt"],
        trusted_commitment_sha256=evidence["commitment"]["commitment_sha256"],
        truth_records={
            "records": json.loads(
                evidence["truth_records_path"].read_text(encoding="utf-8")
            )["records"]
        },
        frozen_build_config=evidence["build_config"],
    )

    assert report["status"] == "passed"
    assert report["independent_accuracy_gate"] is True
    assert report["official_accuracy_gate"] is False
    assert report["production_acceptance_status"] == "blocked"
    assert report["official_reference_gate"] is False


def test_changed_truth_build_parameter_is_blocked(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    changed_options = {**REGISTRATION_OPTIONS, "max_iterations": 21}
    changed_config = build_truth_build_contract(
        topology_path=evidence["topology_path"],
        topology_v3_sha256=evidence["truth"]["topology_v3_sha256"],
        mesh_signature=evidence["truth"]["mesh_signature"],
        artifact_set_id=ARTIFACT_SET_ID,
        registration_options=changed_options,
        maximum_projection_distance_mm=MAXIMUM_PROJECTION_DISTANCE_MM,
    )

    report = replay_independent_erosion_acceptance(
        evidence["protocol_path"],
        evidence["commitment_path"],
        evidence["truth_path"],
        evidence["receipt_path"],
        trusted_commitment_sha256=evidence["commitment"]["commitment_sha256"],
        truth_records=evidence["truth_records_path"],
        build_config=changed_config,
    )

    assert report["status"] == "blocked"
    assert "pre-reveal commitment contract" in report["reasons"][0]


def test_valid_high_numeric_error_is_failed(tmp_path: Path) -> None:
    report = _replay(_evidence(tmp_path, prediction_scale=1.20))

    assert report["status"] == "failed"
    assert report["blocked"] is False
    assert report["official_accuracy_gate"] is False
    assert report["official_reference_gate"] is False


@pytest.mark.parametrize("trusted_pin", [None, "0" * 64])
def test_missing_or_wrong_external_commitment_pin_is_blocked(
    tmp_path: Path, trusted_pin: str | None
) -> None:
    report = _replay(_evidence(tmp_path), trusted_pin=trusted_pin)

    assert report["status"] == "blocked"
    assert report["official_accuracy_gate"] is False
    assert report["official_reference_gate"] is False


def test_self_consistent_truth_tamper_is_blocked_by_raw_rebuild(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    tampered = deepcopy(evidence["truth"])
    tampered["rows"][0]["wear_depth_mm"] *= 1.5
    tampered["truth_sha256"] = recompute_truth_hash(tampered)
    receipt = build_reveal_receipt(
        evidence["protocol"],
        evidence["commitment"],
        tampered,
        topology_path=evidence["topology_path"],
    )
    _rewrite(evidence["truth_path"], tampered)
    _rewrite(evidence["receipt_path"], receipt)

    report = _replay(evidence)

    assert report["status"] == "blocked"
    assert report["checks"]["truth_rebuilt_from_raw_sources"] is True
    assert "reconstruction" in report["reasons"][0]


def test_raw_worn_source_tamper_is_blocked(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    evidence["worn_path"].write_bytes(evidence["worn_path"].read_bytes() + b"\n")

    report = _replay(evidence)

    assert report["status"] == "blocked"
    assert "hash" in report["reasons"][0] or "byte count" in report["reasons"][0]
    assert report["official_accuracy_gate"] is False


def test_self_consistent_commitment_topology_mismatch_is_blocked(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path)
    commitment = deepcopy(evidence["commitment"])
    commitment["topology_binding"]["mesh_signature"] = "c" * 64
    commitment["topology_binding_sha256"] = canonical_sha256(
        commitment["topology_binding"]
    )
    commitment["commitment_sha256"] = recompute_commitment_hash(commitment)
    receipt = build_reveal_receipt(
        evidence["protocol"],
        commitment,
        evidence["truth"],
        topology_path=evidence["topology_path"],
    )
    assert receipt["evaluation_status"] == "ready"
    _rewrite(evidence["commitment_path"], commitment)
    _rewrite(evidence["receipt_path"], receipt)

    report = _replay(
        evidence, trusted_pin=commitment["commitment_sha256"]
    )

    assert report["status"] == "blocked"
    assert "topology_binding" in report["reasons"][0]


def test_self_consistent_receipt_binding_mismatch_is_blocked(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    receipt = deepcopy(evidence["receipt"])
    receipt["truth_sha256"] = "f" * 64
    receipt["receipt_sha256"] = recompute_reveal_receipt_hash(receipt)
    _rewrite(evidence["receipt_path"], receipt)

    report = _replay(evidence)

    assert report["status"] == "blocked"
    assert "same protocol, commitment, and truth" in report["reasons"][0]
    assert report["official_reference_gate"] is False
