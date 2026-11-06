"""Fail-closed validation for the participant supplemental data register.

This validator checks provenance and file integrity only. It does not run any
erosion model, particle tracking, FEM solver, or scoring calculation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


VERSION = "step30-supplemental-data-validator-v1"
REQUIRED_TOP_LEVEL = {
    "version",
    "created_date",
    "status",
    "storage_policy",
    "scope",
    "source_files",
    "data_products",
    "not_yet_obtained",
    "gate_effect",
    "validation_status",
}
HEX = frozenset("0123456789abcdef")


class SupplementalDataError(ValueError):
    """Raised when the source register is incomplete or unverifiable."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SupplementalDataError(f"{name} must be a non-empty string")
    return value.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source(source: Mapping[str, Any], root: Path) -> dict[str, Any]:
    source_id = _text(source.get("source_id"), "source_id")
    path_value = _text(source.get("path"), f"{source_id}.path")
    relative = Path(path_value.replace("\\", "/"))
    if relative.is_absolute() or re.match(r"^[A-Za-z]:[\\/]", path_value) or ".." in relative.parts:
        raise SupplementalDataError(f"{source_id}.path must be workspace-relative")
    path = (root / relative).resolve()
    if not path.is_file():
        raise SupplementalDataError(f"{source_id} source file is missing: {path}")
    if path.drive.upper() == "C:":
        raise SupplementalDataError(f"{source_id} source file is on C: drive")
    expected_bytes = source.get("bytes")
    if not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise SupplementalDataError(f"{source_id}.bytes must be a positive integer")
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise SupplementalDataError(
            f"{source_id} byte mismatch: expected {expected_bytes}, got {actual_bytes}"
        )
    expected_hash = _text(source.get("sha256"), f"{source_id}.sha256").lower()
    if len(expected_hash) != 64 or not set(expected_hash) <= HEX:
        raise SupplementalDataError(f"{source_id}.sha256 is not SHA-256")
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise SupplementalDataError(
            f"{source_id} hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    uses = source.get("evidence_uses")
    if not isinstance(uses, list) or not uses:
        raise SupplementalDataError(f"{source_id}.evidence_uses must be non-empty")
    for index, use in enumerate(uses):
        if not isinstance(use, Mapping):
            raise SupplementalDataError(f"{source_id}.evidence_uses[{index}] must be an object")
        for key in ("claim_id", "evidence_level", "locator", "claim", "allowed_use", "forbidden_use"):
            _text(use.get(key), f"{source_id}.evidence_uses[{index}].{key}")
    return {
        "source_id": source_id,
        "path": str(path),
        "bytes": actual_bytes,
        "sha256": actual_hash,
        "evidence_use_count": len(uses),
    }


def validate_manifest(document: Mapping[str, Any], *, root: str | Path) -> dict[str, Any]:
    if not isinstance(document, Mapping):
        raise SupplementalDataError("manifest must be an object")
    missing = REQUIRED_TOP_LEVEL - set(document)
    if missing:
        raise SupplementalDataError(f"manifest is missing fields: {sorted(missing)}")
    if document.get("version") != "step30-supplemental-data-manifest-v1":
        raise SupplementalDataError("unsupported supplemental data manifest version")
    storage = document.get("storage_policy")
    if not isinstance(storage, Mapping):
        raise SupplementalDataError("storage_policy must be an object")
    if storage.get("local_c_drive_write_allowed") is not False:
        raise SupplementalDataError("local C: drive writes must remain disabled")
    sources = document.get("source_files")
    if not isinstance(sources, list) or not sources:
        raise SupplementalDataError("source_files must be a non-empty array")
    root_path = Path(root).resolve()
    checked = [_validate_source(source, root_path) for source in sources]
    gate_effect = document.get("gate_effect")
    if not isinstance(gate_effect, Mapping):
        raise SupplementalDataError("gate_effect must be an object")
    if gate_effect.get("official_accuracy_gate") is not False:
        raise SupplementalDataError("official_accuracy_gate must remain false")
    if gate_effect.get("official_reference_gate") is not False:
        raise SupplementalDataError("official_reference_gate must remain false")
    if gate_effect.get("fem_gate") != "blocked":
        raise SupplementalDataError("fem_gate must remain blocked")
    return {
        "validator_version": VERSION,
        "status": "valid",
        "source_count": len(checked),
        "sources": checked,
        "official_accuracy_gate": False,
        "official_reference_gate": False,
        "fem_gate": "blocked",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Step30 supplemental data provenance")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    document = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = validate_manifest(document, root=args.workspace_root)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
