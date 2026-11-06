"""Fail-closed source and measurement-authenticity audit for Step30.

A ``measured`` label is accepted only when a canonical source manifest binds the
input bytes, uses a measurement provenance origin, and contains explicit
acquisition metadata.  File names and table content are also inspected for
synthetic-generator evidence.  In particular, a synthetic CSV labelled
``measured`` is reported with the stable issue code
``legacy_status_mismatch``.

The audit does not rewrite source data or repair legacy labels.  Reports may be
written once with the immutable output helper used by other Step30 modules.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from step30_validation_common import (
    ValidationError,
    canonical_sha256,
    jsonable,
    require_status,
    safe_json_dump,
    sha256_file,
)
from step30_validation_split import MANIFEST_VERSION as SPLIT_MANIFEST_VERSION
from step30_validation_split import load_validation_manifest


VERSION = "step30-measurement-audit-v1"
AUDIT_REPORT_VERSION = "step30-measurement-audit-report-v1"
MEASURED_ORIGINS = frozenset({
    "laboratory_measurement", "field_measurement", "physical_measurement",
    "inspection_measurement", "metrology_measurement",
})
SYNTHETIC_ORIGINS = frozenset({
    "synthetic", "deterministic_simulation", "simulation", "generated",
    "known_synthetic_truth", "fixed_seed_generator",
})
_REQUIRED_MEASURED_PROVENANCE = (
    "origin", "acquisition_method", "acquired_at", "organization",
)
_SOURCE_ENTRY_NAMES = (
    "measurements", "measurement", "trials", "trial_data", "source", "data",
)
_SYNTHETIC_NAME_PATTERN = re.compile(
    r"(?:^|[-_.])(?:synthetic|simulated|generated|mock|dummy)(?:[-_.]|$)", re.IGNORECASE,
)
_SYNTHETIC_COLUMNS = frozenset({
    "synthetic", "is_synthetic", "synthetic_truth", "generator", "seed",
    "seed_policy", "truth_source", "measurement_status", "data_origin",
    "source_type", "provenance_origin",
})
_SYNTHETIC_VALUES = (
    "synthetic", "simulated", "simulation", "generated", "fixed_seed_generator",
    "known_synthetic_truth", "deterministic_simulation", "mock", "dummy",
)
_HEX = frozenset("0123456789abcdef")


class AuditError(ValidationError):
    """Raised for malformed audit inputs."""


def _issue(code: str, message: str, **context: Any) -> dict[str, Any]:
    result = {"code": code, "message": message}
    if context:
        result["context"] = jsonable(context)
    return result


def _load_json(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"invalid {label}: {target}") from exc
    if not isinstance(value, Mapping):
        raise AuditError(f"{label} must contain an object")
    return dict(value)


def _is_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value.lower())
    )


def _validate_canonical_hash(manifest: Mapping[str, Any]) -> str | None:
    stored = manifest.get("manifest_sha256")
    if not _is_hash(stored):
        return "source manifest has no valid manifest_sha256"
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if canonical_sha256(unsigned) != str(stored).lower():
        return "source manifest canonical hash mismatch"
    return None


def _entry_candidates(manifest: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    source = manifest.get("source")
    if isinstance(source, Mapping):
        candidates.append(("source", source))
    entries = manifest.get("entries")
    if isinstance(entries, Mapping):
        for name, entry in entries.items():
            if isinstance(entry, Mapping):
                candidates.append((str(name), entry))
    files = manifest.get("files")
    if isinstance(files, Mapping):
        for name, entry in files.items():
            if isinstance(entry, Mapping):
                candidates.append((str(name), entry))
    return candidates


def _entry_path(entry: Mapping[str, Any], manifest_path: Path) -> Path | None:
    value = entry.get("absolute_path") if entry.get("external") is True else entry.get("path", entry.get("file"))
    if not isinstance(value, str) or not value.strip():
        return None
    target = Path(value)
    if not target.is_absolute():
        target = manifest_path.parent / target
    return target.resolve()


def _match_source_entry(
    manifest: Mapping[str, Any], manifest_path: Path, source_path: Path,
) -> tuple[str | None, Mapping[str, Any] | None]:
    fallback: tuple[str, Mapping[str, Any]] | None = None
    for name, entry in _entry_candidates(manifest):
        path = _entry_path(entry, manifest_path)
        if path == source_path:
            return name, entry
        if fallback is None and name.lower() in _SOURCE_ENTRY_NAMES:
            fallback = (name, entry)
    if fallback is not None and _entry_path(fallback[1], manifest_path) == source_path:
        return fallback
    return None, None


def _status(value: Any) -> str | None:
    try:
        return require_status(value)
    except ValidationError:
        return None


def _text_is_synthetic(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return any(token in text for token in _SYNTHETIC_VALUES)


def _synthetic_manifest_evidence(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    evidence: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).strip().lower()
            item_path = path + (str(key),)
            if key_text in {
                "generator", "seed_policy", "synthetic_truth", "truth_source",
                "measurement_status", "origin", "data_origin", "source_type",
            } and _text_is_synthetic(item):
                evidence.append(".".join(item_path))
            evidence.extend(_synthetic_manifest_evidence(item, item_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            evidence.extend(_synthetic_manifest_evidence(item, path + (str(index),)))
    return sorted(set(evidence))


def _read_csv_evidence(path: Path) -> tuple[int, list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise AuditError("CSV requires a named header")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        if isinstance(exc, AuditError):
            raise
        raise AuditError(f"cannot inspect CSV: {path}") from exc
    fields = [str(field).strip().lower() for field in reader.fieldnames]
    evidence: list[str] = []
    for field in fields:
        if field in _SYNTHETIC_COLUMNS and field in {
            "synthetic_truth", "generator", "seed", "seed_policy",
        }:
            evidence.append(f"column:{field}")
    for row_number, row in enumerate(rows, 2):
        normalized = {str(key).strip().lower(): value for key, value in row.items()}
        for field in fields:
            value = normalized.get(field)
            if field in _SYNTHETIC_COLUMNS and _text_is_synthetic(value):
                evidence.append(f"row:{row_number}:{field}")
            if field == "is_synthetic" and str(value).strip().lower() in {"1", "true", "yes"}:
                evidence.append(f"row:{row_number}:is_synthetic")
    return len(rows), sorted(set(evidence))


def _source_evidence(path: Path, manifest: Mapping[str, Any]) -> tuple[int | None, list[str]]:
    evidence: list[str] = []
    if _SYNTHETIC_NAME_PATTERN.search(path.name):
        evidence.append("filename:synthetic_marker")
    if path.suffix.lower() == ".csv":
        row_count, csv_evidence = _read_csv_evidence(path)
        evidence.extend(csv_evidence)
    else:
        row_count = None
    evidence.extend(f"manifest:{item}" for item in _synthetic_manifest_evidence(manifest))
    return row_count, sorted(set(evidence))


def _record_statuses(records: Any) -> set[str]:
    if records is None:
        return set()
    if isinstance(records, (str, Path)):
        path = Path(records)
        if path.suffix.lower() == ".csv":
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as stream:
                    values: Any = list(csv.DictReader(stream))
            except (OSError, UnicodeError, csv.Error) as exc:
                raise AuditError(f"cannot read record CSV: {path}") from exc
        else:
            value = _load_json(path, "record input")
            values = value.get("records", value.get("rows", value.get("data", value)))
    else:
        values = records
    if isinstance(values, Mapping):
        values = values.get("records", values.get("rows", values.get("data", [])))
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise AuditError("record audit input must contain a record list")
    statuses: set[str] = set()
    for index, record in enumerate(values, 1):
        if not isinstance(record, Mapping):
            raise AuditError(f"record {index} is not an object")
        if "status" in record:
            status = _status(record.get("status"))
            if status is None:
                raise AuditError(f"record {index} has invalid status")
            statuses.add(status)
    return statuses


def _provenance_issues(
    manifest: Mapping[str, Any],
    status: str,
) -> tuple[list[dict[str, Any]], Mapping[str, Any] | None]:
    issues: list[dict[str, Any]] = []
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        issues.append(_issue("provenance_missing", "source manifest has no provenance object"))
        return issues, None
    if status == "measured":
        missing = [
            field for field in _REQUIRED_MEASURED_PROVENANCE
            if not str(provenance.get(field, "")).strip()
        ]
        if missing:
            issues.append(_issue(
                "measured_provenance_incomplete",
                "measured provenance requires explicit acquisition metadata",
                missing=missing,
            ))
        origin = str(provenance.get("origin", "")).strip().lower()
        if origin not in MEASURED_ORIGINS:
            issues.append(_issue(
                "measured_origin_invalid",
                "measured status requires a physical measurement origin",
                origin=origin,
            ))
        for forbidden in ("generator", "seed", "seed_policy", "synthetic_truth"):
            if provenance.get(forbidden) not in (None, "", [], {}):
                issues.append(_issue(
                    "measured_synthetic_provenance",
                    f"measured provenance contains synthetic field {forbidden}",
                ))
    elif status == "synthetic":
        origin = str(provenance.get("origin", "")).strip().lower()
        if origin and origin not in SYNTHETIC_ORIGINS:
            issues.append(_issue(
                "synthetic_origin_invalid",
                "synthetic status conflicts with provenance origin",
                origin=origin,
            ))
    return issues, provenance


def _split_manifest_issues(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    try:
        manifest = load_validation_manifest(path)
    except (OSError, ValidationError, ValueError) as exc:
        return [_issue("validation_manifest_invalid", str(exc), path=str(path))]
    if manifest.get("manifest_version") != SPLIT_MANIFEST_VERSION:
        return [_issue("validation_manifest_invalid", "validation manifest version mismatch")]
    return []


def audit_measurement(
    source_path: str | Path,
    source_manifest_path: str | Path,
    *,
    records: Any = None,
    validation_manifest_path: str | Path | None = None,
    expected_status: str | None = None,
) -> dict[str, Any]:
    """Audit source integrity and whether a ``measured`` claim is authentic."""
    source = Path(source_path).resolve()
    manifest_file = Path(source_manifest_path).resolve()
    issues: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}
    if expected_status is not None:
        try:
            expected_status = require_status(expected_status)
        except ValidationError as exc:
            raise AuditError(str(exc)) from exc
    if not source.is_file():
        issues.append(_issue("source_missing", "measurement source file is missing", path=str(source)))
        return _finalize_report(
            source, manifest_file, None, expected_status, issues, checks,
            row_count=None, synthetic_evidence=[], validation_manifest_path=validation_manifest_path,
        )
    try:
        manifest = _load_json(manifest_file, "source manifest")
    except AuditError as exc:
        issues.append(_issue("source_manifest_invalid", str(exc), path=str(manifest_file)))
        return _finalize_report(
            source, manifest_file, None, expected_status, issues, checks,
            row_count=None, synthetic_evidence=[], validation_manifest_path=validation_manifest_path,
        )

    canonical_error = _validate_canonical_hash(manifest)
    checks["source_manifest_canonical_hash"] = canonical_error is None
    if canonical_error:
        issues.append(_issue("source_manifest_hash_mismatch", canonical_error))
    manifest_status = _status(manifest.get("status"))
    checks["manifest_status_valid"] = manifest_status is not None
    if manifest_status is None:
        issues.append(_issue("source_manifest_status_invalid", "source manifest status is invalid"))
    if expected_status is not None and manifest_status != expected_status:
        issues.append(_issue(
            "expected_status_mismatch", "expected status differs from source manifest",
            expected=expected_status, actual=manifest_status,
        ))
    entry_name, entry = _match_source_entry(manifest, manifest_file, source)
    checks["source_entry_bound"] = entry is not None
    if entry is None:
        issues.append(_issue(
            "source_manifest_unbound", "source manifest has no entry bound to the requested source",
            source=str(source),
        ))
    else:
        actual_hash = sha256_file(source)
        checks["source_hash_matches"] = entry.get("sha256") == actual_hash
        if entry.get("sha256") != actual_hash:
            issues.append(_issue(
                "source_hash_mismatch", "source bytes differ from the manifest hash",
                entry=entry_name, expected=entry.get("sha256"), actual=actual_hash,
            ))
        entry_status = _status(entry.get("status", manifest_status))
        checks["entry_status_matches"] = entry_status == manifest_status and entry_status is not None
        if entry_status != manifest_status or entry_status is None:
            issues.append(_issue(
                "source_entry_status_mismatch",
                "source entry status differs from source manifest status",
                entry_status=entry_status, manifest_status=manifest_status,
            ))
    statuses = _record_statuses(records)
    checks["record_statuses_match"] = not statuses or statuses == {manifest_status}
    if statuses and statuses != {manifest_status}:
        issues.append(_issue(
            "record_status_mismatch", "record statuses differ from source manifest status",
            record_statuses=sorted(statuses), manifest_status=manifest_status,
        ))
    if manifest_status is not None:
        provenance_findings, provenance = _provenance_issues(manifest, manifest_status)
        issues.extend(provenance_findings)
        checks["provenance_consistent"] = not provenance_findings
    else:
        provenance = manifest.get("provenance") if isinstance(manifest.get("provenance"), Mapping) else None
        checks["provenance_consistent"] = False
    try:
        row_count, synthetic_evidence = _source_evidence(source, manifest)
    except AuditError as exc:
        issues.append(_issue("source_content_invalid", str(exc)))
        row_count, synthetic_evidence = None, []
    checks["synthetic_evidence_absent_for_measured"] = not (
        manifest_status == "measured" and synthetic_evidence
    )
    if manifest_status == "measured" and source.suffix.lower() == ".csv" and synthetic_evidence:
        issues.append(_issue(
            "legacy_status_mismatch",
            "synthetic CSV is labelled measured",
            evidence=synthetic_evidence,
        ))
    elif manifest_status == "measured" and synthetic_evidence:
        issues.append(_issue(
            "measured_synthetic_mismatch",
            "synthetic source evidence conflicts with measured status",
            evidence=synthetic_evidence,
        ))
    issues.extend(_split_manifest_issues(validation_manifest_path))
    if validation_manifest_path is not None:
        checks["validation_manifest_valid"] = not any(
            issue["code"] == "validation_manifest_invalid" for issue in issues
        )
    return _finalize_report(
        source, manifest_file, manifest, expected_status, issues, checks,
        row_count=row_count, synthetic_evidence=synthetic_evidence,
        validation_manifest_path=validation_manifest_path,
    )


def _finalize_report(
    source: Path,
    manifest_file: Path,
    manifest: Mapping[str, Any] | None,
    expected_status: str | None,
    issues: Sequence[Mapping[str, Any]],
    checks: Mapping[str, bool],
    *,
    row_count: int | None,
    synthetic_evidence: Iterable[str],
    validation_manifest_path: str | Path | None,
) -> dict[str, Any]:
    manifest_status = _status(manifest.get("status")) if manifest is not None else None
    issue_list = [dict(item) for item in issues]
    report: dict[str, Any] = {
        "version": VERSION,
        "report_version": AUDIT_REPORT_VERSION,
        "status": "passed" if not issue_list else "failed",
        "measurement_status": manifest_status,
        "measured_authentic": bool(manifest_status == "measured" and not issue_list),
        "source": {
            "path": str(source),
            "exists": source.is_file(),
            "sha256": sha256_file(source) if source.is_file() else None,
            "rows": row_count,
        },
        "source_manifest": {
            "path": str(manifest_file),
            "exists": manifest_file.is_file(),
            "sha256": sha256_file(manifest_file) if manifest_file.is_file() else None,
            "manifest_sha256": manifest.get("manifest_sha256") if manifest else None,
        },
        "expected_status": expected_status,
        "validation_manifest": str(Path(validation_manifest_path).resolve()) if validation_manifest_path else None,
        "checks": dict(checks),
        "synthetic_evidence": sorted(set(synthetic_evidence)),
        "issues": issue_list,
        "issue_codes": sorted({str(item["code"]) for item in issue_list}),
    }
    unsigned = dict(report)
    report["audit_sha256"] = canonical_sha256(unsigned)
    return report


def write_audit_report(report: Mapping[str, Any], output_path: str | Path) -> dict[str, Any]:
    candidate = dict(report)
    stored = candidate.pop("audit_sha256", None)
    if candidate.get("report_version") != AUDIT_REPORT_VERSION:
        raise AuditError("unsupported audit report")
    if not _is_hash(stored) or canonical_sha256(candidate) != str(stored).lower():
        raise AuditError("audit report canonical hash mismatch")
    candidate["audit_sha256"] = str(stored).lower()
    return safe_json_dump(candidate, output_path)


def _source_manifest(path: Path, source: Path, status: str, *, synthetic: bool = False) -> None:
    manifest: dict[str, Any] = {
        "version": "step30-measurement-source-manifest-v1",
        "status": status,
        "source": {"path": source.name, "sha256": sha256_file(source), "status": status},
        "provenance": {
            "origin": "deterministic_simulation" if synthetic else "laboratory_measurement",
            "acquisition_method": "fixed-seed generator" if synthetic else "coupon rig and profilometer",
            "acquired_at": "2026-07-30T00:00:00Z",
            "organization": "selftest",
            "campaign_id": "C-1",
        },
    }
    if synthetic:
        manifest["generator"] = "fixed_seed_generator"
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _selftest() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        measured = root / "measurements.csv"
        measured.write_text("specimen_id,value\nS-1,0.1\n", encoding="utf-8")
        measured_manifest = root / "measured-manifest.json"
        _source_manifest(measured_manifest, measured, "measured")
        report = audit_measurement(measured, measured_manifest, expected_status="measured")
        assert report["status"] == "passed" and report["measured_authentic"] is True
        synthetic = root / "synthetic-measurements.csv"
        synthetic.write_text(
            "specimen_id,truth_source,seed,value\nS-1,fixed_seed_generator,30,0.1\n",
            encoding="utf-8",
        )
        legacy_manifest = root / "legacy-manifest.json"
        _source_manifest(legacy_manifest, synthetic, "measured", synthetic=False)
        legacy = audit_measurement(synthetic, legacy_manifest)
        assert legacy["status"] == "failed"
        assert "legacy_status_mismatch" in legacy["issue_codes"]
        synthetic_manifest = root / "synthetic-manifest.json"
        _source_manifest(synthetic_manifest, synthetic, "synthetic", synthetic=True)
        synthetic_report = audit_measurement(synthetic, synthetic_manifest)
        assert synthetic_report["status"] == "passed"
        output = root / "audit.json"
        write_audit_report(report, output)
        try:
            write_audit_report(report, output)
        except FileExistsError:
            pass
        else:
            raise AssertionError("audit report was overwritten")


def selftest() -> bool:
    _selftest()
    return True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Step30 measurement source integrity and authenticity")
    parser.add_argument("--source", help="measurement CSV/JSON/NPZ/source file")
    parser.add_argument("--source-manifest", help="canonical hashed source manifest")
    parser.add_argument("--records", help="optional record CSV/JSON to check status consistency")
    parser.add_argument("--validation-manifest", help="optional immutable split manifest")
    parser.add_argument("--expect-status", choices=("measured", "synthetic", "blocked"))
    parser.add_argument("--output", type=Path, help="immutable audit JSON")
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.selftest:
        selftest()
        print("step30_measurement_audit selftest: PASS")
        return 0
    if not args.source or not args.source_manifest:
        _parser().error("--source and --source-manifest are required unless --selftest is used")
    report = audit_measurement(
        args.source,
        args.source_manifest,
        records=args.records,
        validation_manifest_path=args.validation_manifest,
        expected_status=args.expect_status,
    )
    if args.output:
        write_audit_report(report, args.output)
    print(json.dumps(jsonable(report), ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
