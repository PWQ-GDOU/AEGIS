"""Hermetic tests for frozen independent erosion truth and commitments."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from step30_independent_erosion_protocol import (
    IndependentErosionProtocolError,
    PredictionCommitmentError,
    build_blocked_report,
    build_prediction_commitment,
    build_protocol,
    load_prediction_commitment,
    load_protocol,
    recompute_commitment_hash,
    recompute_protocol_hash,
    save_prediction_commitment,
    save_protocol,
    source_file_descriptor,
    validate_blocked_report,
    validate_prediction_commitment,
    validate_protocol,
)


def _assets(root: Path, count: int = 6) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index in range(count):
        source = root / f"truth-{index}.csv"
        source.write_text(
            f"parent_face_id,wear_depth_mm\n{index},{0.1 + index / 100}\n",
            encoding="utf-8",
        )
        records.append(
            {
                "record_id": f"record-{index}",
                "asset_id": f"asset-{index}",
                "component_id": "runner-bucket",
                "service_episode_id": f"episode-{index}",
                "campaign_id": f"campaign-{index // 2}",
                "scan_id": f"scan-{index}",
                "status": "measured",
                "source": source_file_descriptor(source),
            }
        )
    return records


def _predictions(protocol: dict[str, object]) -> list[dict[str, object]]:
    cohort = protocol["splits"]["independent_test"]
    return [
        {
            "record_id": record_id,
            "parent_face_id": 100 + index,
            "prediction_value": 0.2 + index / 100,
            "value_units": "mm",
            "parent_face_area_mm2": 12.0 + index,
            "area_units": "mm2",
        }
        for index, record_id in enumerate(cohort)
    ]


def test_labels_and_source_hashes_do_not_influence_split(tmp_path: Path) -> None:
    assets = _assets(tmp_path)
    labelled = deepcopy(assets)
    relabelled = deepcopy(assets)
    for index, record in enumerate(labelled):
        record.update(
            {
                "wear_depth_mm": index / 10,
                "label": f"high-{index}",
                "observation_value": index,
                "metric": "truth-a",
            }
        )
    for index, record in enumerate(relabelled):
        record.update(
            {
                "wear_depth_mm": 999.0 - index,
                "label": f"low-{index}",
                "observation_value": -index,
                "metric": "truth-b",
            }
        )

    first = build_protocol(labelled, seed=1701)
    for index, record in enumerate(relabelled):
        source = Path(record["source"]["path"])
        source.write_text(f"different frozen source bytes {index}\n", encoding="utf-8")
        record["source"] = source_file_descriptor(source)
    second = build_protocol(relabelled, seed=1701)

    assert first["splits"] == second["splits"]
    assert first["groups"] == second["groups"]
    assert all("wear_depth_mm" not in asset for asset in first["assets"])
    assert first["split_contract"]["source_hash_role"] == (
        "integrity_binding_only_not_partitioning"
    )


def test_fixed_seed_is_deterministic_across_input_order(tmp_path: Path) -> None:
    assets = _assets(tmp_path)
    first = build_protocol(assets, seed=731)
    second = build_protocol(list(reversed(assets)), seed=731)

    assert first == second
    assert first["split_contract"]["seed"] == 731
    assert all(first["splits"][name] for name in first["splits"])


def test_fewer_than_three_independent_groups_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(IndependentErosionProtocolError, match="at least three"):
        build_protocol(_assets(tmp_path, count=2), seed=30)


def test_same_asset_leakage_is_rejected_even_with_rehashed_protocol(
    tmp_path: Path,
) -> None:
    assets = _assets(tmp_path, count=5)
    extra_source = tmp_path / "truth-extra.csv"
    extra_source.write_text("parent_face_id,wear_depth_mm\n99,0.3\n", encoding="utf-8")
    assets.append(
        {
            "record_id": "record-extra",
            "asset_id": assets[0]["asset_id"],
            "component_id": "runner-bucket",
            "service_episode_id": assets[0]["service_episode_id"],
            "campaign_id": "campaign-extra",
            "scan_id": "scan-extra",
            "status": "measured",
            "source": source_file_descriptor(extra_source),
        }
    )
    protocol = build_protocol(assets, seed=19)
    tampered = deepcopy(protocol)
    shared = [
        asset
        for asset in tampered["assets"]
        if asset["asset_id"] == assets[0]["asset_id"]
    ]
    replacement = next(
        name for name in ("train", "calibration", "independent_test")
        if name != shared[0]["partition"]
    )
    shared[1]["partition"] = replacement
    tampered["protocol_sha256"] = recompute_protocol_hash(tampered)

    with pytest.raises(IndependentErosionProtocolError, match="leakage"):
        validate_protocol(tampered)


def test_truth_source_hash_tampering_fails_closed(tmp_path: Path) -> None:
    assets = _assets(tmp_path)
    protocol = build_protocol(assets, seed=30)
    source = Path(protocol["assets"][0]["path"])
    original = source.read_bytes()
    source.write_bytes(b"x" * len(original))

    with pytest.raises(IndependentErosionProtocolError, match="hash mismatch"):
        validate_protocol(protocol)


def test_protocol_metrics_are_preregistered_and_cannot_be_relaxed(
    tmp_path: Path,
) -> None:
    protocol = build_protocol(_assets(tmp_path), seed=30)
    assert protocol["version"] == "step30-independent-erosion-protocol-v1"
    assert protocol["schema_version"] == protocol["version"]
    assert set(protocol["metric_contract"]["metrics"]) == {
        "wear_depth_mm",
        "total_volume_mm3",
        "max_depth_mm",
        "hotspot_area_mm2",
        "hotspot_iou_error",
    }
    assert all(
        metric["maximum_error_fraction"] <= 0.05
        for metric in protocol["metric_contract"]["metrics"].values()
    )
    assert set(protocol["metric_contract"]["thresholds"].values()) == {0.05}

    relaxed = deepcopy(protocol)
    relaxed["metric_contract"]["metrics"]["max_depth_mm"][
        "maximum_error_fraction"
    ] = 0.051
    relaxed["protocol_sha256"] = recompute_protocol_hash(relaxed)
    with pytest.raises(IndependentErosionProtocolError, match="thresholds changed"):
        validate_protocol(relaxed)


def test_prediction_commitment_requires_complete_independent_cohort(
    tmp_path: Path,
) -> None:
    protocol = build_protocol(_assets(tmp_path), seed=30)
    predictions = _predictions(protocol)
    predictions[0]["record_id"] = "not-in-frozen-cohort"

    with pytest.raises(PredictionCommitmentError, match="cohort mismatch"):
        build_prediction_commitment(
            protocol, {"model_id": "erosion-model-v4"}, predictions
        )


def test_prediction_rejects_duplicate_parent_face_and_wrong_units(
    tmp_path: Path,
) -> None:
    protocol = build_protocol(_assets(tmp_path), seed=30)
    predictions = _predictions(protocol)
    duplicate = deepcopy(predictions[0])
    predictions.append(duplicate)
    with pytest.raises(PredictionCommitmentError, match="must be unique"): 
        build_prediction_commitment(
            protocol, {"model_id": "erosion-model-v4"}, predictions
        )

    wrong_units = _predictions(protocol)
    wrong_units[0]["value_units"] = "cm"
    with pytest.raises(PredictionCommitmentError, match="must be mm"):
        build_prediction_commitment(
            protocol, {"model_id": "erosion-model-v4"}, wrong_units
        )


def test_commitment_hash_is_recomputable_and_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    protocol = build_protocol(_assets(tmp_path), seed=30)
    commitment = build_prediction_commitment(
        protocol, {"model_id": "erosion-model-v4", "revision": "frozen"},
        list(reversed(_predictions(protocol))),
    )

    assert commitment["commitment_sha256"] == recompute_commitment_hash(commitment)
    assert commitment["predictions"] == sorted(
        commitment["predictions"], key=lambda row: row["parent_face_id"]
    )
    assert commitment["official_reference_gate"] is False

    tampered = deepcopy(commitment)
    tampered["predictions"][0]["prediction_value"] += 0.01
    tampered["commitment_sha256"] = recompute_commitment_hash(tampered)
    with pytest.raises(PredictionCommitmentError, match="predictions_sha256"):
        validate_prediction_commitment(tampered, protocol)


def test_prediction_commitment_does_not_read_truth_files(tmp_path: Path) -> None:
    protocol = build_protocol(_assets(tmp_path), seed=30)
    predictions = _predictions(protocol)
    for asset in protocol["assets"]:
        Path(asset["path"]).unlink()

    commitment = build_prediction_commitment(
        protocol, {"model_id": "blind-model-v1"}, predictions
    )
    assert commitment["protocol_sha256"] == protocol["protocol_sha256"]


def test_immutable_protocol_and_commitment_save_load(tmp_path: Path) -> None:
    protocol = build_protocol(_assets(tmp_path), seed=30)
    protocol_path = tmp_path / "protocol.json"
    save_protocol(protocol, protocol_path)
    assert load_protocol(protocol_path) == protocol
    with pytest.raises(FileExistsError):
        save_protocol(protocol, protocol_path)

    commitment = build_prediction_commitment(
        protocol, {"model_id": "erosion-model-v4"}, _predictions(protocol)
    )
    commitment_path = tmp_path / "commitment.json"
    save_prediction_commitment(commitment, commitment_path, protocol=protocol)
    assert load_prediction_commitment(
        commitment_path, protocol=protocol
    ) == commitment
    with pytest.raises(FileExistsError):
        save_prediction_commitment(
            commitment, commitment_path, protocol=protocol
        )


def test_blocked_report_never_sets_official_reference_gate() -> None:
    report = build_blocked_report(
        "truth source hash mismatch", details={"record_id": "record-3"}
    )
    assert validate_blocked_report(report) == report
    assert report["status"] == "blocked"
    assert report["official_reference_gate"] is False
