"""Manifest and trend-only adapter for public external measured datasets.

External coupon/slurry datasets can support model calibration and physical trend
checks, but they are never accepted as current bucket spatial ground truth.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from step30_validation_common import (
    EvidenceContractError,
    canonical_sha256,
    evidence_metadata,
    safe_json_dump,
    sha256_file,
)

VERSION = "step30-external-measured-calibration-v1"
REQUIRED_DATASET_FIELDS = ("title", "doi", "license", "material", "limitations")


class ExternalCalibrationError(ValueError):
    """Raised when an external measured calibration package is incomplete."""


def _hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value)


def _load(value: Any) -> tuple[Any, Path | None]:
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            raise ExternalCalibrationError(f"input does not exist: {path}")
        try:
            return json.loads(path.read_text(encoding="utf-8")), path.resolve()
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ExternalCalibrationError(f"invalid JSON input: {path}") from exc
    return value, None


def _rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = [dict(row) for row in csv.DictReader(stream)]
    elif path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value if isinstance(value, list) else value.get("records", value.get("rows", value.get("data", [])))
    else:
        raise ExternalCalibrationError("external calibration data must be CSV or JSON")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, Mapping) for row in rows):
        raise ExternalCalibrationError("external calibration data has no object records")
    return [dict(row) for row in rows]


def build_calibration_manifest(
    data_path: str | Path,
    *,
    title: str,
    doi: str,
    license: str,
    material: str,
    limitations: Sequence[str],
    conditions: Mapping[str, Any] | None = None,
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a hash-bound calibration-only manifest without claiming bucket truth."""
    path = Path(data_path).resolve()
    if not path.is_file():
        raise ExternalCalibrationError(f"data file does not exist: {path}")
    values = {"title": title, "doi": doi, "license": license, "material": material}
    if any(not isinstance(value, str) or not value.strip() for value in values.values()):
        raise ExternalCalibrationError("title, doi, license, and material are required")
    limitations_list = [str(item).strip() for item in limitations if str(item).strip()]
    if not limitations_list:
        raise ExternalCalibrationError("limitations must be explicit")
    rows = _rows(path)
    root = Path(source_root).resolve() if source_root is not None else path.parent
    try:
        relative = str(path.relative_to(root))
    except ValueError:
        relative = path.name
        root = path.parent
    result: dict[str, Any] = {
        "version": VERSION,
        "status": "measured",
        "source_type": "external_measured",
        "reference_role": "external_measured_calibration",
        "pipeline_validation": False,
        "official_reference": False,
        "dataset": {
            **values,
            "limitations": limitations_list,
            "conditions": dict(conditions or {}),
            "n_records": len(rows),
        },
        "source_root": str(root),
        "artifacts": [{"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size}],
        "official_spatial_accuracy_eligible": False,
        "provenance_note": "external measured calibration evidence; not a bucket spatial ground truth",
    }
    try:
        result.update(evidence_metadata(result, status="measured"))
    except EvidenceContractError as exc:
        raise ExternalCalibrationError(str(exc)) from exc
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def validate_calibration_manifest(value: Any) -> dict[str, Any]:
    """Validate a calibration manifest and verify all referenced artifact hashes."""
    document, origin = _load(value)
    if not isinstance(document, Mapping) or document.get("version") != VERSION:
        raise ExternalCalibrationError("unsupported external calibration manifest")
    document = dict(document)
    stored = document.get("manifest_sha256")
    unsigned = dict(document)
    unsigned.pop("manifest_sha256", None)
    if not _hash(stored) or canonical_sha256(unsigned) != str(stored).lower():
        raise ExternalCalibrationError("calibration manifest canonical hash mismatch")
    if document.get("status") != "measured" or document.get("source_type") != "external_measured" or document.get("reference_role") != "external_measured_calibration" or document.get("official_reference") is not False:
        raise ExternalCalibrationError("calibration evidence taxonomy is invalid")
    dataset = document.get("dataset")
    if not isinstance(dataset, Mapping) or any(not str(dataset.get(field, "")).strip() for field in REQUIRED_DATASET_FIELDS):
        raise ExternalCalibrationError("dataset metadata is incomplete")
    root = Path(document.get("source_root") or (origin.parent if origin else Path.cwd())).resolve()
    checked = []
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes, bytearray)) or not artifacts:
        raise ExternalCalibrationError("calibration artifacts are missing")
    for item in artifacts:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str) or not _hash(item.get("sha256")):
            raise ExternalCalibrationError("calibration artifact needs path and sha256")
        target = Path(item["path"])
        if not target.is_absolute():
            target = (root / target).resolve()
        if not target.is_file() or sha256_file(target) != str(item["sha256"]).lower():
            raise ExternalCalibrationError(f"calibration artifact hash mismatch: {item.get('path')}")
        checked.append({"path": str(target), "sha256": str(item["sha256"]).lower(), "bytes": int(target.stat().st_size)})
    document["checked_artifacts"] = checked
    document["calibration_only"] = True
    return document


def trend_report(manifest: Any) -> dict[str, Any]:
    """Return a non-official trend evidence report from a validated manifest."""
    try:
        document = validate_calibration_manifest(manifest)
        result = {
            "version": VERSION,
            "status": "passed",
            "source_type": "external_measured",
            "reference_role": "external_measured_calibration",
            "official_reference": False,
            "official_accuracy_gate": False,
            "official_spatial_accuracy_eligible": False,
            "dataset": document["dataset"],
            "manifest_sha256": document["manifest_sha256"],
            "provenance_note": "external measured calibration/trend evidence only; not eligible for bucket spatial accuracy",
        }
    except (ExternalCalibrationError, OSError, TypeError, ValueError, KeyError) as exc:
        result = {
            "version": VERSION,
            "status": "blocked",
            "source_type": "external_measured",
            "reference_role": "external_measured_calibration",
            "official_reference": False,
            "official_accuracy_gate": False,
            "reason": str(exc),
            "provenance_note": "blocked external calibration evidence",
        }
    result["report_sha256"] = canonical_sha256(result)
    return result


def _selftest() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        data = root / "external.csv"
        data.write_text("speed_m_s,wear_loss_mg\n2.5,1.2\n3.0,1.8\n", encoding="utf-8")
        manifest = build_calibration_manifest(data, title="External slurry test", doi="10.0000/example", license="CC BY 4.0", material="AISI 1018", limitations=("no bucket geometry", "no parent-face mapping"), conditions={"sand_diameter_mm": 1.05})
        assert trend_report(manifest)["official_accuracy_gate"] is False
        tampered = dict(manifest)
        tampered["artifacts"] = []
        assert trend_report(tampered)["status"] == "blocked"


def selftest() -> bool:
    _selftest()
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or validate external measured calibration evidence")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--title")
    parser.add_argument("--doi")
    parser.add_argument("--license")
    parser.add_argument("--material")
    parser.add_argument("--limitations", nargs="+", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        selftest()
        print("step30_external_measured_calibration selftest: PASS")
        return 0
    if args.manifest:
        report = trend_report(args.manifest)
    elif args.data and args.title and args.doi and args.license and args.material:
        report = build_calibration_manifest(args.data, title=args.title, doi=args.doi, license=args.license, material=args.material, limitations=args.limitations)
    else:
        parser.error("provide --manifest or --data with metadata")
    if args.output:
        safe_json_dump(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
