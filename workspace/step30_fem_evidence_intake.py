"""Freeze official DEC stress evidence and fail-closed FEM blockers.

This module records what the DEC package actually proves for the structural
workflow. It deliberately does not infer CAD entity selections, gauge vectors,
or runner-material density from figures, STL tags, or engineering defaults.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from step30_validation_common import canonical_sha256, safe_json_dump, safe_json_load, sha256_file


VERSION = "step30-fem-evidence-intake-v1"
_MANIFEST_NAME = "step30_fem_official_evidence_v1.json"


class FEMEvidenceIntakeError(ValueError):
    """Raised when official FEM evidence cannot be frozen safely."""


def _hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdefABCDEF" for char in value
    )


def _find_one(root: Path, pattern: str) -> Path:
    matches = sorted(path for path in root.rglob(pattern) if path.is_file())
    if len(matches) != 1:
        raise FEMEvidenceIntakeError(
            f"expected exactly one source matching {pattern!r}, found {len(matches)}"
        )
    return matches[0].resolve()


def _artifact(path: Path, root: Path, role: str) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise FEMEvidenceIntakeError(f"artifact is outside source root: {path}") from exc
    return {
        "role": role,
        "path": relative,
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def _parse_trace(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig", errors="strict")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines or lines[0].strip() != "time[s]\tAI5-04 [MPa]":
        raise FEMEvidenceIntakeError(f"unexpected official AI5 trace header: {path.name}")
    times: list[float] = []
    stress: list[float] = []
    for index, line in enumerate(lines[1:], start=2):
        fields = line.split("\t")
        if len(fields) != 2:
            raise FEMEvidenceIntakeError(f"malformed AI5 trace row {index}: {path.name}")
        try:
            time_s = float(fields[0])
            stress_mpa = float(fields[1])
        except ValueError as exc:
            raise FEMEvidenceIntakeError(f"non-numeric AI5 trace row {index}: {path.name}") from exc
        if not math.isfinite(time_s) or not math.isfinite(stress_mpa):
            raise FEMEvidenceIntakeError(f"non-finite AI5 trace row {index}: {path.name}")
        times.append(time_s)
        stress.append(stress_mpa)
    if len(times) < 2:
        raise FEMEvidenceIntakeError(f"AI5 trace is too short: {path.name}")
    intervals = [right - left for left, right in zip(times, times[1:])]
    if any(interval <= 0.0 for interval in intervals):
        raise FEMEvidenceIntakeError(f"AI5 trace time is not strictly increasing: {path.name}")
    interval = sum(intervals) / len(intervals)
    if max(abs(value - interval) for value in intervals) > 1.0e-9:
        raise FEMEvidenceIntakeError(f"AI5 trace sampling interval is not uniform: {path.name}")
    values = stress
    return {
        "condition": "rated_noload" if path.stem == "额定空转" else path.stem,
        "source_file": path.name,
        "n_samples": len(values),
        "time_units": "s",
        "stress_units": "MPa",
        "time_start_s": times[0],
        "time_end_s": times[-1],
        "sample_interval_s": interval,
        "sample_frequency_hz": 1.0 / interval,
        "mean_mpa": sum(values) / len(values),
        "min_mpa": min(values),
        "max_mpa": max(values),
        "raw_peak_to_peak_mpa": max(values) - min(values),
        "raw_half_peak_to_peak_mpa": (max(values) - min(values)) / 2.0,
    }


def discover_official_sources(project_root: str | Path | None = None) -> dict[str, Any]:
    """Locate the original DEC package and return its frozen evidence manifest."""
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[1]
    package_search_root = root / "DecPelton2024"
    if not package_search_root.is_dir():
        raise FEMEvidenceIntakeError(f"official package root does not exist: {package_search_root}")
    stress_report = _find_one(package_search_root, "3.2-转轮动应力试验报告.pdf")
    stress_note = _find_one(package_search_root, "3.3-转轮动应力试验数据说明.pdf")
    geometry_note = _find_one(package_search_root, "2.2-150MW级冲击式国产化真机转轮三维实体数据_文件说明.pdf")
    raw_dir = stress_report.parent / "3.1-现场动应力实测数据"
    if not raw_dir.is_dir():
        raise FEMEvidenceIntakeError(f"official AI5 trace directory does not exist: {raw_dir}")
    traces = sorted(raw_dir.glob("*.txt"))
    if len(traces) != 13:
        raise FEMEvidenceIntakeError(f"expected 13 official AI5 traces, found {len(traces)}")
    package_root = stress_report.parents[2]
    artifacts = [
        _artifact(stress_report, package_root, "official_dynamic_stress_report"),
        _artifact(stress_note, package_root, "official_dynamic_stress_data_note"),
        _artifact(geometry_note, package_root, "official_geometry_data_note"),
    ]
    artifacts.extend(_artifact(path, package_root, f"official_ai5_trace_{index:02d}") for index, path in enumerate(traces, start=1))
    trace_records = [_parse_trace(path) for path in traces]

    geometry_manifest_path = root / "workspace" / "output" / "step30_fem_remote_export_v1" / "fem_geometry_conversion_v2.json"
    geometry_manifest: dict[str, Any] | None = None
    if geometry_manifest_path.is_file():
        loaded = safe_json_load(geometry_manifest_path)
        if isinstance(loaded, Mapping):
            geometry_manifest = dict(loaded)
            geometry_manifest["path"] = str(geometry_manifest_path)
            geometry_manifest["sha256"] = sha256_file(geometry_manifest_path)

    blockers = [
        {
            "id": "ROOT_INTERFACE",
            "status": "blocked",
            "required": "independently confirmed CAD face/node selection for the bucket-runner attachment interface",
            "reason": "no named selection or independently confirmed face set is present in the official package or recovered COMSOL export",
        },
        {
            "id": "WETTED_SURFACE",
            "status": "blocked",
            "required": "independently confirmed wetted face set with root closure excluded",
            "reason": "the STEP catalog contains only geometric surface tags; tags are not physical selections",
        },
        {
            "id": "AI5_04_CAD_PATCH",
            "status": "blocked",
            "required": "numeric CAD patch/point bound to the source STEP",
            "reason": "the official report gives physical location semantics but no numeric CAD coordinates or face/node identity",
        },
        {
            "id": "AI5_04_GAUGE_DIRECTIONS",
            "status": "blocked",
            "required": "one or two frozen unit directions and their coordinate frame",
            "reason": "the report identifies a 90-degree strain gauge but does not provide a machine-readable direction vector",
        },
        {
            "id": "MATERIAL_DENSITY",
            "status": "blocked",
            "required": "official design/material certificate bound to grade/heat/batch and density in kg/m3",
            "reason": "the official package supports E=206 GPa and nu=0.3 but no admissible runner material density source",
        },
        {
            "id": "STEP_COORDINATE_UNITS",
            "status": "blocked",
            "required": "independent numeric unit probe for the recovered STEP coordinates",
            "reason": "COMSOL export declares meters, while the catalog magnitudes are mm-scale; the scale has not been independently frozen",
        },
    ]
    manifest: dict[str, Any] = {
        "version": VERSION,
        "status": "blocked",
        "fem_gate": "blocked",
        "evidence_taxonomy": {
            "source_type": "external_measured",
            "reference_role": "external_measured_calibration",
            "official_reference": False,
            "interpretation": "DEC field measurement package; admissible as independent measured AI5 evidence only after a CAD patch and gauge direction are supplied",
        },
        "official_package_root": str(package_root),
        "artifacts": artifacts,
        "claims": {
            "ai5_channel": {"status": "passed", "value": "AI5-04", "source_roles": ["official_ai5_trace_01", "official_dynamic_stress_data_note"]},
            "ai5_physical_location": {
                "status": "passed",
                "value": "4# bucket root, on the water-divider extension line, selected as the simulated maximum-stress location",
                "source_roles": ["official_dynamic_stress_report"],
                "report_pages": [3],
                "numeric_cad_coordinates": None,
            },
            "sampling": {"status": "passed", "value": 1000.0, "units": "Hz", "source_roles": ["official_dynamic_stress_data_note"]},
            "stress_units": {"status": "passed", "value": "MPa", "source_roles": ["official_ai5_trace_01"]},
            "runner_speed_s1": {"status": "passed", "value": 375.0, "units": "rpm", "source_roles": ["official_dynamic_stress_report"], "report_pages": [4]},
            "young_modulus": {"status": "passed", "value": 206.0e9, "units": "Pa", "source_roles": ["official_dynamic_stress_report", "official_dynamic_stress_data_note"], "report_pages": [3]},
            "poisson_ratio": {"status": "passed", "value": 0.3, "units": "1", "source_roles": ["official_dynamic_stress_report", "official_dynamic_stress_data_note"], "report_pages": [3]},
            "material_density": {"status": "blocked", "value": None, "units": "kg/m3"},
        },
        "measurements": {
            "status": "measured",
            "n_conditions": len(trace_records),
            "records": trace_records,
            "amplitude_definition": "raw_peak_to_peak is retained verbatim; raw_half_peak_to_peak is a derived descriptive statistic. The official report uses its own dynamic-amplitude convention and must be cited separately; neither field is an FEM calibration target.",
        },
        "geometry_provenance": geometry_manifest,
        "remaining_blockers": blockers,
        "acquisition_plan": {
            "cad_selection_package": "Export a named-selection/face-list report from the source STEP in COMSOL, NX, Ansys, or equivalent. Include source STEP SHA-256, coordinate units, entity dimension/tags, and a screenshot or CAD selection log for ROOT_INTERFACE, WETTED_SURFACE, AI5_04, and MATERIAL_DOMAIN.",
            "ai5_patch_package": "Provide numeric AI5 patch coordinates in the STEP frame, patch face/node IDs, gauge coordinate frame, and one/two unit direction vectors. Include the source report/CAD artifact hash and confirm AI5 nodes are outside fixed root nodes.",
            "density_package": "Provide the runner material certificate or design record with grade, heat/batch identity, density in kg/m3, and source hash. Do not use water density or the synthetic 7850 kg/m3 default.",
            "unit_probe": "Run an independent STEP bounding-box/volume probe with a known dimensional reference and freeze whether the solver mesh input is m or mm.",
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def validate_manifest(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        document = dict(value)
    else:
        path = Path(value).resolve()
        document = safe_json_load(path)
        if not isinstance(document, Mapping):
            raise FEMEvidenceIntakeError("FEM evidence manifest must be an object")
        document = dict(document)
    stored = document.get("manifest_sha256")
    unsigned = dict(document)
    unsigned.pop("manifest_sha256", None)
    if not _hash(stored) or canonical_sha256(unsigned) != str(stored).lower():
        raise FEMEvidenceIntakeError("FEM evidence manifest hash mismatch")
    if document.get("version") != VERSION or document.get("status") not in {"blocked", "passed", "failed"}:
        raise FEMEvidenceIntakeError("unsupported FEM evidence manifest")
    root = Path(str(document.get("official_package_root", ""))).resolve()
    if not root.is_dir():
        raise FEMEvidenceIntakeError(f"official package root does not exist: {root}")
    checked = []
    for item in document.get("artifacts", []):
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str) or not _hash(item.get("sha256")):
            raise FEMEvidenceIntakeError("FEM evidence artifact is incomplete")
        target = (root / item["path"]).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise FEMEvidenceIntakeError("FEM evidence artifact escapes package root") from exc
        if not target.is_file() or sha256_file(target) != str(item["sha256"]).lower() or int(item.get("bytes", -1)) != target.stat().st_size:
            raise FEMEvidenceIntakeError(f"FEM evidence artifact hash mismatch: {item.get('path')}")
        checked.append(str(target))
    measurements = document.get("measurements")
    if not isinstance(measurements, Mapping) or measurements.get("status") != "measured" or int(measurements.get("n_conditions", 0)) != len(measurements.get("records", [])):
        raise FEMEvidenceIntakeError("official AI5 measurements are incomplete")
    return {**document, "checked_artifacts": checked}


def build_and_write(project_root: str | Path | None = None, output_path: str | Path | None = None) -> dict[str, Any]:
    manifest = discover_official_sources(project_root)
    target = Path(output_path).resolve() if output_path is not None else Path(__file__).resolve().parent / "output" / _MANIFEST_NAME
    safe_json_dump(manifest, target)
    return {"path": str(target), "sha256": sha256_file(target), "status": manifest["status"], "manifest_sha256": manifest["manifest_sha256"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze official DEC FEM evidence and blockers")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate:
        try:
            result = validate_manifest(args.validate)
        except (FEMEvidenceIntakeError, OSError, ValueError, TypeError) as exc:
            print(json.dumps({"status": "blocked", "fem_gate": "blocked", "reason": str(exc)}, indent=2))
            return 3
        print(json.dumps({"status": result["status"], "manifest_sha256": result["manifest_sha256"], "n_checked_artifacts": len(result["checked_artifacts"])}, indent=2))
        return 0
    result = build_and_write(args.project_root, args.output)
    print(json.dumps(result, indent=2))
    return 3 if result["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
