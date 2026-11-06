"""Delivery tests for independent measured erosion protocol v2."""
from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

import step30_independent_erosion_cli as delivery
from step30_external_commitment_pin import (
    ExternalCommitmentPinError,
    build_test_hmac_envelope,
    canonical_signature_payload,
    recompute_envelope_sha256,
    verify_pin_envelope,
)
from step30_independent_erosion_protocol import recompute_commitment_hash
from step30_validation_common import safe_json_dump, sha256_file


ZERO_HASH = "0" * 64
WORKSPACE = Path(__file__).resolve().parent
SCHEMA_PATH = (
    WORKSPACE / "schemas" / "step30-independent-erosion-protocol-v2.schema.json"
)
SCAN_TEMPLATE_PATH = (
    WORKSPACE / "templates" / "step30-scan-pair-manifest-v1.json"
)
PIN_TEMPLATE_PATH = (
    WORKSPACE / "templates" / "step30-prediction-pin-envelope-v1.json"
)
ACQUISITION_TEMPLATE_PATH = (
    WORKSPACE / "templates" / "step30-independent-baseline-worn-acquisition-v1.json"
)


def _descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _delivery_inputs(
    root: Path,
    *,
    count: int,
    formal_requested: bool = True,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    certificate = root / "calibration-certificate.txt"
    certificate.write_text("calibration certificate bytes\n", encoding="utf-8")
    topology = root / "topology-v3.npz"
    topology.write_bytes(b"frozen topology fixture bytes")
    topology_hash = sha256_file(topology)
    pairs = []
    episodes = []
    for index in range(count):
        baseline = root / f"baseline-{index}.csv"
        worn = root / f"worn-{index}.csv"
        baseline.write_text(f"real baseline scan {index}\n", encoding="utf-8")
        worn.write_text(f"real worn scan {index}\n", encoding="utf-8")
        record_id = f"record-{index}"
        episode_id = f"episode-{index}"
        common_scan = {
            "instrument_serial": f"serial-{index}",
            "calibration_certificate": _descriptor(certificate),
            "coordinate_frame": "fixture-local",
            "fiducial_id": f"fiducial-{index}",
            "fixture_id": f"fixture-{index}",
            "coordinate_units": "mm",
        }
        pairs.append(
            {
                "record_id": record_id,
                "asset_id": f"asset-{index}",
                "component_id": "runner-bucket",
                "campaign_id": f"campaign-{index}",
                "service_episode": {
                    "service_episode_id": episode_id,
                    "started_at": "2026-01-02T00:00:00Z",
                    "ended_at": "2026-01-03T00:00:00Z",
                },
                "baseline_scan": {
                    "scan_id": f"baseline-scan-{index}",
                    "acquired_at": "2026-01-01T00:00:00Z",
                    "source": _descriptor(baseline),
                    **common_scan,
                },
                "worn_scan": {
                    "scan_id": f"worn-scan-{index}",
                    "acquired_at": "2026-01-04T00:00:00Z",
                    "source": _descriptor(worn),
                    **common_scan,
                },
            }
        )
        episodes.append(
            {
                "record_id": record_id,
                "service_episode_id": episode_id,
                "parent_face_ids": [0, 1],
                "minimum_area_coverage_fraction": 0.8,
                "minimum_valid_count": 4,
            }
        )
    manifest = {
        "version": delivery.SCAN_MANIFEST_VERSION,
        "schema_version": delivery.SCAN_MANIFEST_VERSION,
        "status": "measured",
        "formal_requested": formal_requested,
        "created_at_utc": "2026-01-04T00:01:00Z",
        "minimum_independent_groups": 3,
        "recommended_formal_groups": 5,
        "blocked_reason": None,
        "scan_pairs": pairs,
    }
    mask = {
        "version": delivery.EVALUATION_MASK_VERSION,
        "status": "measured",
        "immutable": True,
        "topology_v3_sha256": topology_hash,
        "episodes": episodes,
    }
    mask["mask_sha256"] = delivery.recompute_evaluation_mask_sha256(mask)
    mask_path = root / "evaluation-mask.json"
    safe_json_dump(mask, mask_path)
    build_config = {
        "topology_path": str((root / "topology-v3.npz").resolve()),
        "topology_v3_sha256": topology_hash,
        "mesh_signature": "b" * 64,
        "artifact_set_id": "c" * 64,
        "registration_options": {
            "distance_threshold_mm": 0.2,
            "min_coverage": 0.8,
        },
        "maximum_projection_distance_mm": 0.5,
    }
    return manifest, mask_path, build_config


def _pin_bindings() -> dict[str, str]:
    return {
        "protocol_sha256": "1" * 64,
        "truth_build_contract_sha256": "2" * 64,
        "topology_v3_sha256": "3" * 64,
        "evaluation_mask_sha256": "4" * 64,
    }


def _commitment_file(path: Path) -> dict[str, Any]:
    commitment = {
        "model_descriptor": {
            "commitment_created_at_utc": "2026-02-01T00:00:00Z"
        },
    }
    commitment["commitment_sha256"] = recompute_commitment_hash(commitment)
    path.write_text(
        json.dumps(commitment, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return commitment


def _formal_envelope(
    commitment_path: Path,
    private_key: Any,
    *,
    received_at: str = "2026-02-01T00:01:00Z",
) -> dict[str, Any]:
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    envelope = {
        "version": "step30-prediction-pin-envelope-v1",
        "formal": True,
        "test_only": False,
        "canonical_commitment_digest": commitment["commitment_sha256"],
        "commitment_file_sha256": sha256_file(commitment_path),
        "bindings": _pin_bindings(),
        "issuer": {
            "name": "Independent metrology custodian",
            "organization": "External laboratory",
            "independent_third_party": True,
        },
        "received_at": received_at,
        "receipt_id": "external-receipt-001",
        "signature_algorithm": "ed25519",
        "signature": "pending",
        "certificate_pem": None,
        "public_key_url": "https://issuer.example/public-key.pem",
    }
    envelope["signature"] = base64.b64encode(
        private_key.sign(canonical_signature_payload(envelope))
    ).decode("ascii")
    envelope["envelope_sha256"] = recompute_envelope_sha256(envelope)
    return envelope


def test_cli_smoke_and_shipped_templates_are_valid(capsys: pytest.CaptureFixture[str]) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    scan_template = json.loads(SCAN_TEMPLATE_PATH.read_text(encoding="utf-8"))
    pin_template = json.loads(PIN_TEMPLATE_PATH.read_text(encoding="utf-8"))
    acquisition_template = json.loads(
        ACQUISITION_TEMPLATE_PATH.read_text(encoding="utf-8")
    )

    import jsonschema

    jsonschema.Draft202012Validator.check_schema(schema)
    assert delivery.main(["validate-template", str(SCAN_TEMPLATE_PATH)]) == 0
    assert delivery.main(["validate-template", str(PIN_TEMPLATE_PATH)]) == 0
    assert capsys.readouterr().out.count('"status": "valid_template"') == 2
    assert delivery._parser().parse_args(["status", "--protocol", "p.json"]).command == (
        "status"
    )
    subcommands = set(delivery._parser()._subparsers._group_actions[0].choices)
    assert subcommands == {
        "validate-template",
        "build-protocol",
        "commit",
        "build-truth",
        "reveal",
        "accept",
        "status",
    }
    assert scan_template["status"] == "blocked"
    assert pin_template["formal"] is False
    assert acquisition_template["status"] == "blocked"
    assert acquisition_template["official_accuracy_gate"] is False
    assert acquisition_template["records"] == []


def test_three_groups_build_pilot_and_five_build_formal(tmp_path: Path) -> None:
    pilot_root = tmp_path / "pilot"
    pilot_root.mkdir()
    manifest, mask_path, build = _delivery_inputs(pilot_root, count=3)
    pilot = delivery.build_protocol_v2(
        manifest,
        mask_path,
        build,
        created_at_utc="2026-01-05T00:00:00Z",
    )
    assert pilot["study_phase"] == "pilot"
    assert pilot["formal"] is False
    assert "fewer than five" in pilot["formal_eligibility_reason"]
    assert pilot["core_protocol"]["n_independent_groups"] == 3

    formal_root = tmp_path / "formal"
    formal_root.mkdir()
    manifest, mask_path, build = _delivery_inputs(formal_root, count=5)
    formal = delivery.build_protocol_v2(
        manifest,
        mask_path,
        build,
        created_at_utc="2026-01-05T00:00:00Z",
    )
    assert formal["study_phase"] == "formal_candidate"
    assert formal["formal"] is True
    assert formal["formal_eligibility_reason"] is None

    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(formal)


def test_time_order_is_fail_closed(tmp_path: Path) -> None:
    manifest, _, _ = _delivery_inputs(tmp_path, count=3)
    manifest["scan_pairs"][0]["worn_scan"]["acquired_at"] = (
        "2026-01-02T00:00:00Z"
    )
    with pytest.raises(delivery.IndependentErosionDeliveryError, match="time order"):
        delivery.validate_scan_manifest(manifest)

    utc_root = tmp_path / "utc-format"
    utc_root.mkdir()
    manifest, _, _ = _delivery_inputs(utc_root, count=3)
    manifest["scan_pairs"][0]["baseline_scan"]["acquired_at"] = (
        "2026-01-01T00:00:00+00:00"
    )
    with pytest.raises(delivery.IndependentErosionDeliveryError, match="UTC Z"):
        delivery.validate_scan_manifest(manifest)


def test_missing_real_scan_source_is_blocked(tmp_path: Path) -> None:
    manifest, mask_path, build = _delivery_inputs(tmp_path, count=3)
    missing = Path(manifest["scan_pairs"][0]["worn_scan"]["source"]["path"])
    missing.unlink()

    with pytest.raises(
        delivery.IndependentErosionDeliveryError,
        match="real measured source does not exist",
    ):
        delivery.build_protocol_v2(manifest, mask_path, build)


def test_coverage_below_preregistered_area_or_count_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = {
        "core_protocol": {"splits": {"independent_test": ["record-0"]}},
        "evaluation_mask": {
            "episodes": [
                {
                    "record_id": "record-0",
                    "service_episode_id": "episode-0",
                    "parent_face_ids": [0, 1],
                    "minimum_area_coverage_fraction": 0.8,
                    "minimum_valid_count": 4,
                }
            ]
        },
        "truth_build_contract": {
            "topology_path": "unused.npz",
            "mesh_signature": "b" * 64,
        },
    }
    truth = {
        "rows": [
            {
                "record_id": "record-0",
                "parent_face_id": 0,
                "valid_count": 3,
            }
        ]
    }
    monkeypatch.setattr(
        delivery, "validate_protocol_v2", lambda value, verify_sources=False: value
    )
    monkeypatch.setattr(
        delivery,
        "validate_independent_erosion_truth",
        lambda value, protocol, verify_sources=False: value,
    )
    monkeypatch.setattr(
        delivery,
        "load_topology_v3",
        lambda *args, **kwargs: {"face_area_m2": [1.0e-6, 1.0e-6]},
    )

    with pytest.raises(
        delivery.IndependentErosionDeliveryError,
        match="area coverage or valid_count is insufficient",
    ):
        delivery.validate_truth_coverage(protocol, truth)


def test_canonical_digest_is_not_file_bytes_hash(tmp_path: Path) -> None:
    commitment_path = tmp_path / "commitment.json"
    commitment = _commitment_file(commitment_path)
    envelope = build_test_hmac_envelope(
        commitment_path=commitment_path,
        bindings=_pin_bindings(),
        key="secret",
        received_at="2026-02-01T00:01:00Z",
    )
    assert envelope["canonical_commitment_digest"] == commitment["commitment_sha256"]
    assert envelope["commitment_file_sha256"] == sha256_file(commitment_path)
    assert envelope["canonical_commitment_digest"] != envelope[
        "commitment_file_sha256"
    ]

    confused = deepcopy(envelope)
    confused["commitment_file_sha256"] = confused["canonical_commitment_digest"]
    import hashlib
    import hmac

    confused["signature"] = base64.b64encode(
        hmac.new(
            b"secret", canonical_signature_payload(confused), hashlib.sha256
        ).digest()
    ).decode("ascii")
    confused["envelope_sha256"] = recompute_envelope_sha256(confused)
    with pytest.raises(ExternalCommitmentPinError, match="file bytes SHA-256"):
        verify_pin_envelope(
            confused,
            commitment=commitment,
            commitment_path=commitment_path,
            expected_bindings=_pin_bindings(),
            test_hmac_key="secret",
        )


def test_receipt_time_order_and_hmac_cannot_be_formal(tmp_path: Path) -> None:
    commitment_path = tmp_path / "commitment.json"
    commitment = _commitment_file(commitment_path)
    envelope = build_test_hmac_envelope(
        commitment_path=commitment_path,
        bindings=_pin_bindings(),
        key="secret",
        received_at="2026-01-31T23:59:59Z",
    )
    with pytest.raises(ExternalCommitmentPinError, match="predates commitment"):
        verify_pin_envelope(
            envelope,
            commitment=commitment,
            commitment_path=commitment_path,
            expected_bindings=_pin_bindings(),
            test_hmac_key="secret",
            not_before="2026-02-01T00:00:00Z",
        )

    envelope = build_test_hmac_envelope(
        commitment_path=commitment_path,
        bindings=_pin_bindings(),
        key="secret",
        received_at="2026-02-01T00:01:00Z",
    )
    with pytest.raises(ExternalCommitmentPinError, match="formal third-party"):
        verify_pin_envelope(
            envelope,
            commitment=commitment,
            commitment_path=commitment_path,
            expected_bindings=_pin_bindings(),
            test_hmac_key="secret",
            require_formal=True,
        )


def test_formal_public_key_receipt_verifies_offline(tmp_path: Path) -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        PublicFormat,
    )

    commitment_path = tmp_path / "commitment.json"
    commitment = _commitment_file(commitment_path)
    private_key = Ed25519PrivateKey.generate()
    public_path = tmp_path / "trusted-public-key.pem"
    public_path.write_bytes(
        private_key.public_key().public_bytes(
            Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
        )
    )
    envelope = _formal_envelope(commitment_path, private_key)

    result = verify_pin_envelope(
        envelope,
        commitment=commitment,
        commitment_path=commitment_path,
        expected_bindings=_pin_bindings(),
        trusted_public_key=public_path,
        not_before="2026-02-01T00:00:00Z",
        require_formal=True,
    )
    assert result["formal_verified"] is True
    assert result["offline_only"] is True
    assert result["external_requests_performed"] is False


def test_accept_requires_receipt_and_verifies_before_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commitment_path = tmp_path / "commitment.json"
    commitment = _commitment_file(commitment_path)
    frozen = {
        "formal": True,
        "core_protocol": {"protocol_sha256": "6" * 64},
        "truth_build_contract": {
            "truth_build_contract_sha256": "2" * 64,
            "topology_v3_sha256": "3" * 64,
        },
        "protocol_sha256": "1" * 64,
        "evaluation_mask_sha256": "4" * 64,
    }
    calls: list[str] = []
    monkeypatch.setattr(
        delivery, "validate_protocol_v2", lambda *args, **kwargs: frozen
    )
    monkeypatch.setattr(
        delivery, "validate_commitment_v2", lambda *args, **kwargs: commitment
    )
    monkeypatch.setattr(
        delivery,
        "external_pin_bindings",
        lambda value: _pin_bindings(),
    )
    monkeypatch.setattr(
        delivery,
        "validate_independent_erosion_truth",
        lambda *args, **kwargs: {"truth_sha256": "7" * 64},
    )
    monkeypatch.setattr(
        delivery,
        "validate_truth_coverage",
        lambda *args, **kwargs: {"passed": True, "episodes": []},
    )
    monkeypatch.setattr(
        delivery,
        "validate_reveal_receipt",
        lambda *args, **kwargs: {"receipt_sha256": "8" * 64},
    )
    monkeypatch.setattr(
        delivery,
        "replay_independent_erosion_acceptance",
        lambda *args, **kwargs: calls.append("replay")
        or {"status": "passed", "accepted": True, "reasons": []},
    )
    missing = delivery.accept_independent_erosion_delivery(
        {"protocol": "stub"},
        commitment_path,
        {"truth": "stub"},
        {"receipt": "stub"},
        None,
        truth_records={"records": []},
    )
    assert missing["status"] == "blocked"
    assert "third-party receipt" in missing["reasons"][0]
    assert calls == []

    def verified(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append("verify")
        return {
            "formal_verified": True,
            "canonical_commitment_digest": "5" * 64,
        }

    monkeypatch.setattr(delivery, "verify_pin_envelope", verified)
    accepted = delivery.accept_independent_erosion_delivery(
        {"protocol": "stub"},
        commitment_path,
        {"truth": "stub"},
        {"receipt": "stub"},
        {"envelope": "stub"},
        truth_records={"records": []},
        trusted_public_key=b"stub",
    )
    assert accepted["status"] == "passed"
    assert accepted["formal"] is True
    assert calls == ["verify", "replay"]
