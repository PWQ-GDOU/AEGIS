"""Hermetic contract tests for independent measured spatial erosion truth."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from step30_independent_erosion_protocol import (
    build_prediction_commitment,
    build_protocol,
    source_file_descriptor,
)
from step30_independent_erosion_truth import (
    IndependentErosionTruthError,
    build_independent_erosion_truth,
    build_reveal_receipt,
    recompute_truth_hash,
    validate_independent_erosion_truth,
)
from step30_measurement_parent_face_mapping import (
    TOPOLOGY_V3_SIGNATURE_ALGORITHM,
    TOPOLOGY_V3_VERSION,
    _mesh_signature,
)
from step30_validation_common import canonical_sha256, sha256_file


ARTIFACT_SET_ID = "a" * 64
SOURCE_VTP_SHA256 = "b" * 64
REGISTRATION_OPTIONS = {
    "distance_threshold_mm": 0.2,
    "min_coverage": 0.75,
    "trim_fraction": 0.0,
    "max_iterations": 20,
}


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
    area_vectors = 0.5 * np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
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


def _write_points(path: Path, points_mm: np.ndarray, units: str) -> None:
    scale = {"mm": 1.0, "m": 0.001}[units]
    np.savetxt(path, points_mm * scale)


def _write_thickness(
    path: Path,
    points_mm: np.ndarray,
    values_mm: np.ndarray,
    *,
    coordinate_units: str,
    thickness_units: str,
) -> None:
    coordinate_scale = {"mm": 1.0, "m": 0.001}[coordinate_units]
    thickness_scale = {"mm": 1.0, "um": 1000.0}[thickness_units]
    table = np.column_stack((points_mm * coordinate_scale, values_mm * thickness_scale))
    np.savetxt(path, table, delimiter=",", header="x,y,z,thickness", comments="")


def _protocol(
    root: Path,
    independent_source: Path,
    independent_truth_source: Path | None = None,
) -> tuple[dict, str]:
    assets = []
    if independent_truth_source is None:
        independent_truth_source = independent_source
    for index in range(3):
        source = independent_source if index == 2 else root / f"unused-{index}.csv"
        truth_source = (
            independent_truth_source
            if index == 2
            else root / f"unused-worn-{index}.csv"
        )
        if index != 2:
            source.write_text(f"unused measured bytes {index}\n", encoding="utf-8")
            truth_source.write_text(
                f"unused worn measured bytes {index}\n", encoding="utf-8"
            )
        assets.append(
            {
                "record_id": f"record-{index}",
                "asset_id": f"asset-{index}",
                "component_id": "runner-bucket",
                "service_episode_id": f"episode-{index}",
                "campaign_id": f"campaign-{index}",
                "scan_id": f"scan-{index}",
                "status": "measured",
                "source": source_file_descriptor(source),
                "truth_source": source_file_descriptor(truth_source),
            }
        )
    # Determine the identity-only split, then point the selected independent
    # record at the requested baseline/worn sources without changing identity.
    protocol = build_protocol(assets, seed=30)
    independent_id = protocol["splits"]["independent_test"][0]
    if independent_id != "record-2":
        index = int(independent_id.split("-")[-1])
        assets[index]["source"] = source_file_descriptor(independent_source)
        assets[index]["truth_source"] = source_file_descriptor(independent_truth_source)
        if index != 2:
            replacement = root / "replacement-unused.csv"
            replacement_worn = root / "replacement-unused-worn.csv"
            replacement.write_text("unused replacement bytes\n", encoding="utf-8")
            replacement_worn.write_text("unused replacement worn bytes\n", encoding="utf-8")
            assets[2]["source"] = source_file_descriptor(replacement)
            assets[2]["truth_source"] = source_file_descriptor(replacement_worn)
        protocol = build_protocol(assets, seed=30)
    return protocol, independent_id


def _surface_case(root: Path) -> tuple[dict, list[dict], Path, str, str]:
    topology_path = root / "topology-v3.npz"
    signature, topology_hash = _topology(topology_path)
    baseline = root / "baseline.xyz"
    worn = root / "worn.xyz"
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
        ]
    )
    _write_points(baseline, points, "m")
    worn_points = points.copy()
    worn_points[:4, 2] -= 0.10
    worn_points[4:, 2] -= 0.20
    _write_points(worn, worn_points, "m")
    protocol, record_id = _protocol(root, baseline, worn)
    records = [
        {
            "record_id": record_id,
            "record_source": "baseline_worn_surface",
            "status": "measured",
            "source_type": "laser_scan",
            "baseline_path": str(baseline),
            "worn_path": str(worn),
            "baseline_units": "m",
            "worn_units": "m",
        }
    ]
    return protocol, records, topology_path, signature, topology_hash


def _sparse_case(
    root: Path,
    *,
    worn_values_mm: np.ndarray | None = None,
) -> tuple[dict, list[dict], Path, str, str]:
    topology_path = root / "topology-v3.npz"
    signature, topology_hash = _topology(topology_path)
    baseline = root / "baseline-thickness.csv"
    worn = root / "worn-thickness.csv"
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
        ]
    )
    baseline_values = np.full(len(points), 8.0)
    if worn_values_mm is None:
        worn_values_mm = baseline_values - np.array(
            [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2]
        )
    _write_thickness(
        baseline,
        points,
        baseline_values,
        coordinate_units="mm",
        thickness_units="um",
    )
    _write_thickness(
        worn,
        points,
        worn_values_mm,
        coordinate_units="mm",
        thickness_units="um",
    )
    protocol, record_id = _protocol(root, baseline, worn)
    records = [
        {
            "record_id": record_id,
            "record_source": "sparse_thickness",
            "status": "measured",
            "source_type": "ultrasonic_thickness",
            "baseline_path": str(baseline),
            "worn_path": str(worn),
            "coordinate_units": "mm",
            "thickness_units": "um",
            "support_radius_mm": 2.0,
        }
    ]
    return protocol, records, topology_path, signature, topology_hash


def _build(case: tuple[dict, list[dict], Path, str, str]) -> dict:
    protocol, records, topology_path, signature, topology_hash = case
    return build_independent_erosion_truth(
        protocol,
        records,
        topology_path=topology_path,
        topology_v3_sha256=topology_hash,
        mesh_signature=signature,
        artifact_set_id=ARTIFACT_SET_ID,
        registration_options=REGISTRATION_OPTIONS,
        maximum_projection_distance_mm=0.5,
    )


def _commitment(protocol: dict, truth: dict) -> dict:
    predictions = [
        {
            "record_id": row["record_id"],
            "parent_face_id": row["parent_face_id"],
            "prediction_value": row["wear_depth_mm"],
            "value_units": "mm",
            "parent_face_area_mm2": row["area_mm2"],
            "area_units": "mm2",
        }
        for row in truth["rows"]
    ]
    return build_prediction_commitment(
        protocol, {"model_id": "pre-reveal-model"}, predictions
    )


def test_before_after_scans_build_sorted_mm_truth(tmp_path: Path) -> None:
    protocol, records, topology_path, signature, topology_hash = _surface_case(tmp_path)
    truth = _build((protocol, records, topology_path, signature, topology_hash))

    assert truth["status"] == "measured"
    assert truth["protocol_sha256"] == protocol["protocol_sha256"]
    assert truth["topology_v3_sha256"] == topology_hash
    assert truth["mesh_signature"] == signature
    assert truth["artifact_set_id"] == ARTIFACT_SET_ID
    assert truth["official_reference_gate"] is False
    assert [row["parent_face_id"] for row in truth["rows"]] == [0, 1]
    assert [row["wear_depth_mm"] for row in truth["rows"]] == pytest.approx(
        [0.1, 0.2], abs=1.0e-8
    )
    assert all(row["area_mm2"] == pytest.approx(0.5) for row in truth["rows"])
    assert all(row["valid_count"] > 0 for row in truth["rows"])
    assert all(row["uncertainty_mm"] >= 0.0 for row in truth["rows"])
    registration = truth["registration_bindings"][0]["topology_registration"]
    assert len(registration["transform_sha256"]) == 64
    assert len(registration["quality_sha256"]) == 64


def test_sparse_thickness_converts_um_and_does_not_fill_missing_faces(
    tmp_path: Path,
) -> None:
    protocol, records, topology_path, signature, topology_hash = _sparse_case(tmp_path)
    records[0]["support_radius_mm"] = 1.5
    truth = _build((protocol, records, topology_path, signature, topology_hash))

    assert [row["parent_face_id"] for row in truth["rows"]] == [0]
    assert [row["wear_depth_mm"] for row in truth["rows"]] == pytest.approx(
        [0.1], abs=1.0e-8
    )
    assert all(row["wear_depth_mm"] != 0.0 for row in truth["rows"])
    sparse_source = truth["record_sources"][0]
    assert sparse_source["baseline"]["bytes"] > 0
    assert len(sparse_source["worn"]["sha256"]) == 64


def test_unknown_units_nonfinite_and_negative_wear_are_rejected(
    tmp_path: Path,
) -> None:
    protocol, records, topology_path, signature, topology_hash = _sparse_case(tmp_path)
    records[0]["thickness_units"] = "furlong"
    with pytest.raises(IndependentErosionTruthError, match="unit|unsupported"):
        _build((protocol, records, topology_path, signature, topology_hash))

    negative_root = tmp_path / "negative"
    negative_root.mkdir()
    case = _sparse_case(negative_root, worn_values_mm=np.full(7, 8.1))
    with pytest.raises(IndependentErosionTruthError, match="negative wear"):
        _build(case)

    nonfinite_root = tmp_path / "nonfinite"
    nonfinite_root.mkdir()
    protocol, records, topology_path, signature, topology_hash = _sparse_case(
        nonfinite_root
    )
    worn_path = Path(records[0]["worn_path"])
    lines = worn_path.read_text(encoding="utf-8").splitlines()
    first_data = lines[1].split(",")
    first_data[-1] = "nan"
    lines[1] = ",".join(first_data)
    worn_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(IndependentErosionTruthError, match="hash|NaN|finite|parse"):
        _build((protocol, records, topology_path, signature, topology_hash))


def test_source_hash_tampering_and_empty_valid_set_fail_closed(
    tmp_path: Path,
) -> None:
    protocol, records, topology_path, signature, topology_hash = _sparse_case(tmp_path)
    baseline = Path(records[0]["baseline_path"])
    original = baseline.read_bytes()
    baseline.write_bytes(b"x" * len(original))
    with pytest.raises(IndependentErosionTruthError, match="hash|protocol validation"):
        _build((protocol, records, topology_path, signature, topology_hash))

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    protocol, records, topology_path, signature, topology_hash = _sparse_case(empty_root)
    records[0]["support_radius_mm"] = 1.0e-9
    with pytest.raises(IndependentErosionTruthError, match="empty valid"):
        _build((protocol, records, topology_path, signature, topology_hash))


def test_wrong_topology_identity_and_synthetic_provenance_are_rejected(
    tmp_path: Path,
) -> None:
    protocol, records, topology_path, signature, topology_hash = _sparse_case(tmp_path)
    with pytest.raises(IndependentErosionTruthError, match="mesh signature mismatch"):
        build_independent_erosion_truth(
            protocol,
            records,
            topology_path=topology_path,
            topology_v3_sha256=topology_hash,
            mesh_signature="0" * 64,
            artifact_set_id=ARTIFACT_SET_ID,
            registration_options=REGISTRATION_OPTIONS,
        )
    with pytest.raises(IndependentErosionTruthError, match="topology v3 sha256"):
        build_independent_erosion_truth(
            protocol,
            records,
            topology_path=topology_path,
            topology_v3_sha256="0" * 64,
            mesh_signature=signature,
            artifact_set_id=ARTIFACT_SET_ID,
            registration_options=REGISTRATION_OPTIONS,
        )
    records[0]["source_type"] = "CFD_simulation"
    with pytest.raises(IndependentErosionTruthError, match="cannot be classified"):
        _build((protocol, records, topology_path, signature, topology_hash))


def test_truth_hash_tampering_is_rejected(tmp_path: Path) -> None:
    protocol, records, topology_path, signature, topology_hash = _sparse_case(tmp_path)
    truth = _build((protocol, records, topology_path, signature, topology_hash))
    tampered = deepcopy(truth)
    tampered["registration_bindings"][0]["topology_registration"]["quality"][
        "coverage_forward"
    ] = 0.01
    tampered["truth_sha256"] = recompute_truth_hash(tampered)
    with pytest.raises(IndependentErosionTruthError, match="quality hash mismatch"):
        validate_independent_erosion_truth(tampered, protocol=protocol)


def test_reveal_receipt_binds_commitment_and_truth_and_blocks_failures(
    tmp_path: Path,
) -> None:
    protocol, records, topology_path, signature, topology_hash = _sparse_case(tmp_path)
    truth = _build((protocol, records, topology_path, signature, topology_hash))
    commitment = _commitment(protocol, truth)

    ready = build_reveal_receipt(
        protocol, commitment, truth, topology_path=topology_path
    )
    assert ready["evaluation_status"] == "ready"
    assert ready["prediction_commitment_sha256"] == commitment["commitment_sha256"]
    assert ready["truth_sha256"] == truth["truth_sha256"]
    assert ready["official_reference_gate"] is False
    assert ready["receipt_sha256"] == canonical_sha256(
        {key: value for key, value in ready.items() if key != "receipt_sha256"}
    )

    missing_commitment = build_reveal_receipt(protocol, truth=truth)
    assert missing_commitment["evaluation_status"] == "blocked"
    assert missing_commitment["official_reference_gate"] is False

    wrong = deepcopy(commitment)
    wrong["protocol_sha256"] = "0" * 64
    blocked_mismatch = build_reveal_receipt(protocol, wrong, truth)
    assert blocked_mismatch["evaluation_status"] == "blocked"
    assert "commitment" in blocked_mismatch["reason"]

    wrong_faces = _commitment(protocol, truth)
    wrong_faces["predictions"][0]["parent_face_id"] += 10
    wrong_faces["predictions"].sort(key=lambda row: row["parent_face_id"])
    wrong_faces["predictions_sha256"] = canonical_sha256(
        {"predictions": wrong_faces["predictions"]}
    )
    unsigned = dict(wrong_faces)
    unsigned.pop("commitment_sha256")
    wrong_faces["commitment_sha256"] = canonical_sha256(unsigned)
    blocked_faces = build_reveal_receipt(protocol, wrong_faces, truth)
    assert blocked_faces["evaluation_status"] == "blocked"
    assert "identities differ" in blocked_faces["reason"]

    missing_truth = build_reveal_receipt(
        protocol,
        commitment,
        truth_path=tmp_path / "missing-truth.json",
    )
    assert missing_truth["evaluation_status"] == "blocked"
    assert missing_truth["truth_sha256"] is None
    assert missing_truth["official_reference_gate"] is False

    Path(records[0]["worn_path"]).unlink()
    missing_raw_measurement = build_reveal_receipt(protocol, commitment, truth)
    assert missing_raw_measurement["evaluation_status"] == "blocked"
    assert "source file does not exist" in missing_raw_measurement["reason"]
    assert missing_raw_measurement["official_reference_gate"] is False
