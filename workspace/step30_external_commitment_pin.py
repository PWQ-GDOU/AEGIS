"""Offline verification for externally received prediction commitment pins.

The canonical commitment digest and the SHA-256 of the serialized commitment
file are deliberately separate fields. A receipt is never verified by making
an external request; a URL is an issuer identifier and requires a locally
supplied trusted public key.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from step30_independent_erosion_protocol import recompute_commitment_hash
from step30_validation_common import (
    canonical_sha256,
    jsonable,
    safe_json_load,
    sha256_file,
)


PIN_ENVELOPE_VERSION = "step30-prediction-pin-envelope-v1"
PUBLIC_SIGNATURE_ALGORITHMS = frozenset(
    {"ed25519", "rsa-pss-sha256", "ecdsa-sha256"}
)
TEST_SIGNATURE_ALGORITHM = "hmac-sha256-test-only"
_HEX = frozenset("0123456789abcdef")
_BINDING_FIELDS = frozenset(
    {
        "protocol_sha256",
        "truth_build_contract_sha256",
        "topology_v3_sha256",
        "evaluation_mask_sha256",
    }
)
_ENVELOPE_FIELDS = frozenset(
    {
        "version",
        "formal",
        "test_only",
        "canonical_commitment_digest",
        "commitment_file_sha256",
        "bindings",
        "issuer",
        "received_at",
        "receipt_id",
        "signature_algorithm",
        "signature",
        "certificate_pem",
        "public_key_url",
        "envelope_sha256",
    }
)
_ISSUER_FIELDS = frozenset({"name", "organization", "independent_third_party"})


class ExternalCommitmentPinError(ValueError):
    """Raised when an external pin cannot be verified offline."""


def _sha256(value: Any, name: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value.lower())
    ):
        raise ExternalCommitmentPinError(f"{name} must be a SHA-256 digest")
    return value.lower()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExternalCommitmentPinError(f"{name} must be a non-empty string")
    return value.strip()


def parse_utc(value: Any, name: str) -> datetime:
    """Parse an explicit RFC 3339 UTC timestamp."""
    text = _text(value, name)
    if not text.endswith("Z") or "T" not in text:
        raise ExternalCommitmentPinError(
            f"{name} must be an RFC 3339 timestamp with explicit UTC Z"
        )
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise ExternalCommitmentPinError(
            f"{name} is not a valid UTC timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ExternalCommitmentPinError(f"{name} must be UTC")
    return parsed


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def canonical_signature_payload(envelope: Mapping[str, Any]) -> bytes:
    """Return the exact canonical bytes covered by the receipt signature."""
    payload = dict(jsonable(envelope))
    payload.pop("signature", None)
    payload.pop("envelope_sha256", None)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def recompute_envelope_sha256(envelope: Mapping[str, Any]) -> str:
    unsigned = dict(jsonable(envelope))
    unsigned.pop("envelope_sha256", None)
    return canonical_sha256(unsigned)


def _load_public_key(material: bytes) -> tuple[Any, Any | None]:
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives.serialization import load_pem_public_key
    except ImportError as exc:
        raise ExternalCommitmentPinError(
            "cryptography is required for formal offline signature verification"
        ) from exc
    try:
        return load_pem_public_key(material), None
    except ValueError:
        try:
            certificate = x509.load_pem_x509_certificate(material)
        except ValueError as exc:
            raise ExternalCommitmentPinError(
                "trusted certificate/public key is not valid PEM"
            ) from exc
        return certificate.public_key(), certificate


def _trusted_public_material(
    envelope: Mapping[str, Any], trusted_public_key: str | Path | bytes | None
) -> bytes:
    if trusted_public_key is None:
        if envelope.get("public_key_url") is not None:
            raise ExternalCommitmentPinError(
                "public_key_url is never fetched; provide a local trusted public key"
            )
        raise ExternalCommitmentPinError(
            "formal verification requires a locally trusted certificate/public key"
        )
    if isinstance(trusted_public_key, bytes):
        return trusted_public_key
    path = Path(trusted_public_key)
    if not path.is_file():
        raise ExternalCommitmentPinError("trusted public key file does not exist")
    return path.read_bytes()


def _public_key_der(key: Any) -> bytes:
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    return key.public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)


def _certificate_valid_at(certificate: Any, received: datetime) -> None:
    not_before = getattr(certificate, "not_valid_before_utc", None)
    not_after = getattr(certificate, "not_valid_after_utc", None)
    if not_before is None or not_after is None:
        not_before = certificate.not_valid_before.replace(tzinfo=timezone.utc)
        not_after = certificate.not_valid_after.replace(tzinfo=timezone.utc)
    if not not_before <= received <= not_after:
        raise ExternalCommitmentPinError(
            "certificate was not valid at receipt time"
        )


def _verify_public_signature(
    algorithm: str,
    key: Any,
    signature: bytes,
    payload: bytes,
) -> None:
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa
    except ImportError as exc:
        raise ExternalCommitmentPinError(
            "cryptography is required for formal offline signature verification"
        ) from exc
    try:
        if algorithm == "ed25519":
            if not isinstance(key, ed25519.Ed25519PublicKey):
                raise ExternalCommitmentPinError("receipt key is not Ed25519")
            key.verify(signature, payload)
        elif algorithm == "rsa-pss-sha256":
            if not isinstance(key, rsa.RSAPublicKey):
                raise ExternalCommitmentPinError("receipt key is not RSA")
            key.verify(
                signature,
                payload,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH,
                ),
                hashes.SHA256(),
            )
        elif algorithm == "ecdsa-sha256":
            if not isinstance(key, ec.EllipticCurvePublicKey):
                raise ExternalCommitmentPinError("receipt key is not ECDSA")
            key.verify(signature, payload, ec.ECDSA(hashes.SHA256()))
        else:
            raise ExternalCommitmentPinError("unsupported public signature algorithm")
    except InvalidSignature as exc:
        raise ExternalCommitmentPinError(
            "external receipt signature is invalid"
        ) from exc


def _validate_structure(envelope: Any) -> dict[str, Any]:
    if not isinstance(envelope, Mapping) or set(envelope) != _ENVELOPE_FIELDS:
        raise ExternalCommitmentPinError(
            "prediction pin envelope fields are invalid"
        )
    if envelope.get("version") != PIN_ENVELOPE_VERSION:
        raise ExternalCommitmentPinError("prediction pin envelope version mismatch")
    if not isinstance(envelope.get("formal"), bool) or not isinstance(
        envelope.get("test_only"), bool
    ):
        raise ExternalCommitmentPinError("formal and test_only must be booleans")
    _sha256(
        envelope.get("canonical_commitment_digest"),
        "canonical commitment digest",
    )
    _sha256(envelope.get("commitment_file_sha256"), "commitment file SHA-256")
    bindings = envelope.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != _BINDING_FIELDS:
        raise ExternalCommitmentPinError("external receipt bindings are invalid")
    for field in sorted(_BINDING_FIELDS):
        _sha256(bindings.get(field), field)
    issuer = envelope.get("issuer")
    if not isinstance(issuer, Mapping) or set(issuer) != _ISSUER_FIELDS:
        raise ExternalCommitmentPinError("external receipt issuer fields are invalid")
    _text(issuer.get("name"), "issuer name")
    _text(issuer.get("organization"), "issuer organization")
    if not isinstance(issuer.get("independent_third_party"), bool):
        raise ExternalCommitmentPinError(
            "issuer independent_third_party must be a boolean"
        )
    parse_utc(envelope.get("received_at"), "received_at")
    _text(envelope.get("receipt_id"), "receipt_id")
    algorithm = _text(envelope.get("signature_algorithm"), "signature_algorithm")
    if algorithm not in PUBLIC_SIGNATURE_ALGORITHMS | {TEST_SIGNATURE_ALGORITHM}:
        raise ExternalCommitmentPinError("signature_algorithm is unsupported")
    signature_text = _text(envelope.get("signature"), "signature")
    try:
        base64.b64decode(signature_text, validate=True)
    except ValueError as exc:
        raise ExternalCommitmentPinError(
            "signature must be canonical base64"
        ) from exc
    certificate = envelope.get("certificate_pem")
    public_url = envelope.get("public_key_url")
    if certificate is not None and not isinstance(certificate, str):
        raise ExternalCommitmentPinError(
            "certificate_pem must be a string or null"
        )
    if public_url is not None:
        parsed = urlparse(_text(public_url, "public_key_url"))
        if parsed.scheme != "https" or not parsed.netloc:
            raise ExternalCommitmentPinError("public_key_url must be an HTTPS URL")
    if certificate is None and public_url is None and algorithm in PUBLIC_SIGNATURE_ALGORITHMS:
        raise ExternalCommitmentPinError(
            "formal receipt must declare certificate_pem or public_key_url"
        )
    if envelope.get("envelope_sha256") != recompute_envelope_sha256(envelope):
        raise ExternalCommitmentPinError("envelope_sha256/content mismatch")
    return dict(envelope)


def verify_pin_envelope(
    envelope: Mapping[str, Any],
    *,
    commitment: Mapping[str, Any] | None = None,
    commitment_path: str | Path | None = None,
    expected_bindings: Mapping[str, Any] | None = None,
    trusted_public_key: str | Path | bytes | None = None,
    test_hmac_key: str | bytes | None = None,
    not_before: str | datetime | None = None,
    require_formal: bool = False,
) -> dict[str, Any]:
    """Verify a receipt locally and return explicit formal-verification state."""
    frozen = _validate_structure(envelope)
    if commitment_path is None:
        raise ExternalCommitmentPinError(
            "commitment_path is required to verify serialized file bytes"
        )
    commitment_file = Path(commitment_path)
    if not commitment_file.is_file():
        raise ExternalCommitmentPinError("commitment file does not exist")
    if commitment is None:
        commitment = safe_json_load(commitment_file)
    if not isinstance(commitment, Mapping):
        raise ExternalCommitmentPinError("commitment must be a JSON object")
    canonical_digest = _sha256(
        commitment.get("commitment_sha256"), "commitment commitment_sha256"
    )
    if recompute_commitment_hash(commitment) != canonical_digest:
        raise ExternalCommitmentPinError(
            "canonical commitment digest does not match commitment content"
        )
    file_digest = sha256_file(commitment_file)
    if frozen["canonical_commitment_digest"] != canonical_digest:
        raise ExternalCommitmentPinError("canonical commitment digest mismatch")
    if frozen["commitment_file_sha256"] != file_digest:
        raise ExternalCommitmentPinError(
            "commitment file bytes SHA-256 mismatch"
        )
    if frozen["canonical_commitment_digest"] == frozen["commitment_file_sha256"]:
        raise ExternalCommitmentPinError(
            "canonical digest and file-bytes SHA-256 are distinct bindings"
        )
    if expected_bindings is not None:
        if set(expected_bindings) != _BINDING_FIELDS:
            raise ExternalCommitmentPinError(
                "expected receipt bindings are incomplete"
            )
        for field in _BINDING_FIELDS:
            expected = _sha256(expected_bindings[field], field)
            if frozen["bindings"][field] != expected:
                raise ExternalCommitmentPinError(
                    f"external receipt {field} mismatch"
                )
    received = parse_utc(frozen["received_at"], "received_at")
    if not_before is not None:
        earliest = (
            parse_utc(not_before, "not_before")
            if isinstance(not_before, str)
            else not_before
        )
        if earliest.tzinfo is None:
            raise ExternalCommitmentPinError("not_before must be timezone-aware")
        if received < earliest:
            raise ExternalCommitmentPinError(
                "external receipt predates commitment"
            )

    algorithm = frozen["signature_algorithm"]
    signature = base64.b64decode(frozen["signature"], validate=True)
    payload = canonical_signature_payload(frozen)
    if algorithm == TEST_SIGNATURE_ALGORITHM:
        if frozen["formal"] is not False or frozen["test_only"] is not True:
            raise ExternalCommitmentPinError("HMAC is test_only and formal=false")
        if test_hmac_key is None:
            raise ExternalCommitmentPinError("test HMAC key is required")
        key = (
            test_hmac_key.encode("utf-8")
            if isinstance(test_hmac_key, str)
            else test_hmac_key
        )
        expected = hmac.new(key, payload, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            raise ExternalCommitmentPinError("test HMAC signature is invalid")
        formal_verified = False
    else:
        if frozen["formal"] is not True or frozen["test_only"] is not False:
            raise ExternalCommitmentPinError(
                "public-key receipt must set formal=true and test_only=false"
            )
        if frozen["issuer"]["independent_third_party"] is not True:
            raise ExternalCommitmentPinError(
                "formal receipt issuer must be an independent third party"
            )
        material = _trusted_public_material(frozen, trusted_public_key)
        key, trusted_certificate = _load_public_key(material)
        if trusted_certificate is not None:
            _certificate_valid_at(trusted_certificate, received)
        declared_certificate = frozen.get("certificate_pem")
        if isinstance(declared_certificate, str) and declared_certificate.strip():
            declared_key, certificate = _load_public_key(
                declared_certificate.encode("ascii")
            )
            if certificate is None:
                raise ExternalCommitmentPinError(
                    "certificate_pem must contain an X.509 certificate"
                )
            _certificate_valid_at(certificate, received)
            if _public_key_der(declared_key) != _public_key_der(key):
                raise ExternalCommitmentPinError(
                    "declared certificate differs from locally trusted public key"
                )
        _verify_public_signature(algorithm, key, signature, payload)
        formal_verified = True
    if require_formal and not formal_verified:
        raise ExternalCommitmentPinError(
            "a verifiable formal third-party receipt is required"
        )
    return {
        "verified": True,
        "formal_verified": formal_verified,
        "offline_only": True,
        "external_requests_performed": False,
        "canonical_commitment_digest": canonical_digest,
        "commitment_file_sha256": file_digest,
        "receipt_id": frozen["receipt_id"],
        "received_at": frozen["received_at"],
        "envelope_sha256": frozen["envelope_sha256"],
    }


def build_test_hmac_envelope(
    *,
    commitment_path: str | Path,
    bindings: Mapping[str, Any],
    key: str | bytes,
    issuer_name: str = "test fixture",
    issuer_organization: str = "test only",
    received_at: str | None = None,
    receipt_id: str = "test-only-receipt",
) -> dict[str, Any]:
    """Build an explicitly non-formal HMAC envelope for hermetic tests."""
    path = Path(commitment_path)
    commitment = safe_json_load(path)
    if not isinstance(commitment, Mapping):
        raise ExternalCommitmentPinError("commitment must be a JSON object")
    envelope: dict[str, Any] = {
        "version": PIN_ENVELOPE_VERSION,
        "formal": False,
        "test_only": True,
        "canonical_commitment_digest": _sha256(
            commitment.get("commitment_sha256"),
            "commitment commitment_sha256",
        ),
        "commitment_file_sha256": sha256_file(path),
        "bindings": {
            field: _sha256(bindings.get(field), field)
            for field in _BINDING_FIELDS
        },
        "issuer": {
            "name": issuer_name,
            "organization": issuer_organization,
            "independent_third_party": False,
        },
        "received_at": received_at or utc_now(),
        "receipt_id": receipt_id,
        "signature_algorithm": TEST_SIGNATURE_ALGORITHM,
        "signature": "pending",
        "certificate_pem": None,
        "public_key_url": None,
    }
    parse_utc(envelope["received_at"], "received_at")
    secret = key.encode("utf-8") if isinstance(key, str) else key
    envelope["signature"] = base64.b64encode(
        hmac.new(
            secret,
            canonical_signature_payload(envelope),
            hashlib.sha256,
        ).digest()
    ).decode("ascii")
    envelope["envelope_sha256"] = recompute_envelope_sha256(envelope)
    _validate_structure(envelope)
    return envelope
