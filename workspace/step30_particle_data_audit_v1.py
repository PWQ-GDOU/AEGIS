"""Server-side integrity, provenance, conservation, and performance audit.

The audit deliberately does not alter particle results.  It reads immutable
run directories, validates the four-piece artifact contract, compares summary
values with the hit arrays, and reports diagnostic ratios that can explain
poor spatial convergence.  It is a data audit, not a convergence pass.

Run only on an approved remote Linux host.  The local Windows workstation is
rejected even when the execution-target variable is present.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from step30_validation_common import canonical_sha256, safe_json_dump, sha256_file


VERSION = "step30-particle-data-audit-v1"
REQUIRED_FILES = ("particle_hits.npz", "summary.json", "validation.png", "particle_run_manifest.json")
ARRAY_FIELDS = (
    "parent_face_id",
    "parcel_mass_kg",
    "particle_mass_kg",
    "ER_mansouri",
    "Vp_ms",
    "alpha_deg",
    "diameter_m",
    "face_centers_local_mm",
    "face_area_m2",
)
OPTIONAL_ARRAY_FIELDS = ("collision_triangle_id", "impact_index", "hit_kind", "velocity_ms")
STRICT_POLICIES = {"strict_v2"}


class ParticleDataAuditError(ValueError):
    """Raised when an audit input cannot be interpreted safely."""


def _remote_guard() -> str:
    target = os.environ.get("STEP30_EXECUTION_TARGET", "").strip().lower()
    if os.name == "nt" or os.environ.get("STEP30_REMOTE_EXECUTION") != "1":
        raise ParticleDataAuditError(
            "data audit is server-only: require non-Windows host and "
            "STEP30_REMOTE_EXECUTION=1"
        )
    if target not in {"dcu_cpu", "user_gpu"}:
        raise ParticleDataAuditError(
            "STEP30_EXECUTION_TARGET must be dcu_cpu or user_gpu"
        )
    return target


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ParticleDataAuditError(f"invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ParticleDataAuditError(f"JSON object required: {path}")
    return dict(value)


def _float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ParticleDataAuditError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise ParticleDataAuditError(f"{label} is not finite")
    return result


def _issue(code: str, severity: str, message: str, **context: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"code": code, "severity": severity, "message": message}
    if context:
        result["context"] = context
    return result


def _record_issue(issues: list[dict[str, Any]], code: str, severity: str, message: str, **context: Any) -> None:
    issues.append(_issue(code, severity, message, **context))


def _required_file_checks(run_dir: Path, manifest: Mapping[str, Any], issues: list[dict[str, Any]]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        _record_issue(issues, "manifest_outputs_missing", "error", "particle_run_manifest.json has no outputs object")
        return hashes
    for name in REQUIRED_FILES:
        path = run_dir / name
        if not path.is_file():
            _record_issue(issues, "required_file_missing", "error", f"missing required artifact {name}", file=name)
            continue
        actual = sha256_file(path)
        hashes[name] = actual
        key = name.removesuffix(".npz").removesuffix(".json").removesuffix(".png")
        if name == "particle_run_manifest.json":
            # The manifest is the binding document itself and is not required
            # to contain a self-referential output record.
            continue
        record = outputs.get(key)
        if not isinstance(record, Mapping):
            _record_issue(issues, "manifest_output_record_missing", "error", f"manifest has no output record for {name}", file=name)
            continue
        if record.get("file") != name or str(record.get("sha256", "")).lower() != actual:
            _record_issue(
                issues,
                "artifact_hash_mismatch",
                "error",
                f"manifest hash binding does not match {name}",
                file=name,
                recorded=record.get("sha256"),
                actual=actual,
            )
    return hashes


def _load_arrays(path: Path, issues: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = [name for name in ARRAY_FIELDS if name not in data.files]
            if missing:
                _record_issue(issues, "npz_required_field_missing", "error", "NPZ is missing required arrays", missing=missing)
                return {}
            arrays = {name: np.asarray(data[name]) for name in data.files}
    except (OSError, ValueError) as exc:
        _record_issue(issues, "npz_read_failed", "error", "cannot read particle_hits.npz", error=str(exc))
        return {}
    for name, value in arrays.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            _record_issue(issues, "npz_nonfinite", "error", f"NPZ array {name} contains non-finite values")
    parent = arrays.get("parent_face_id")
    if parent is not None and (parent.ndim != 1 or not np.issubdtype(parent.dtype, np.integer)):
        _record_issue(issues, "parent_face_dtype_invalid", "error", "parent_face_id must be a one-dimensional integer array")
    for name in ("parcel_mass_kg", "particle_mass_kg", "ER_mansouri", "Vp_ms", "alpha_deg", "diameter_m"):
        value = arrays.get(name)
        if value is not None and value.ndim != 1:
            _record_issue(issues, "npz_shape_invalid", "error", f"{name} must be one-dimensional", field=name)
    if arrays.get("parcel_mass_kg") is not None and np.any(np.asarray(arrays["parcel_mass_kg"]) <= 0.0):
        _record_issue(issues, "nonpositive_parcel_mass", "error", "parcel_mass_kg must be strictly positive for accepted impacts")
    if arrays.get("particle_mass_kg") is not None and np.any(np.asarray(arrays["particle_mass_kg"]) <= 0.0):
        _record_issue(issues, "nonpositive_particle_mass", "error", "particle_mass_kg must be strictly positive")
    if arrays.get("ER_mansouri") is not None and np.any(np.asarray(arrays["ER_mansouri"]) < 0.0):
        _record_issue(issues, "negative_erosion_ratio", "error", "ER_mansouri cannot be negative")
    return arrays


def _summary_consistency(summary: Mapping[str, Any], arrays: Mapping[str, np.ndarray], issues: list[dict[str, Any]]) -> dict[str, Any]:
    tracking = summary.get("tracking") if isinstance(summary.get("tracking"), Mapping) else {}
    parameters = summary.get("parameters") if isinstance(summary.get("parameters"), Mapping) else {}
    erosion = summary.get("face_erosion") if isinstance(summary.get("face_erosion"), Mapping) else {}
    parent = np.asarray(arrays.get("parent_face_id", np.empty(0)), dtype=np.int64)
    parcel_mass = np.asarray(arrays.get("parcel_mass_kg", np.empty(0)), dtype=np.float64)
    er = np.asarray(arrays.get("ER_mansouri", np.empty(0)), dtype=np.float64)
    n_hits = len(parent)
    for name in ("parcel_mass_kg", "ER_mansouri", "Vp_ms", "alpha_deg", "diameter_m"):
        if name in arrays and len(arrays[name]) != n_hits:
            _record_issue(issues, "npz_length_mismatch", "error", f"{name} length differs from parent_face_id", field=name)
    expected_n = parameters.get("n_impacts", tracking.get("n_impacts"))
    if expected_n is not None and int(expected_n) != n_hits:
        _record_issue(issues, "summary_hit_count_mismatch", "error", "summary n_impacts differs from NPZ length", summary=int(expected_n), npz=n_hits)
    impact_mass = float(parcel_mass.sum(dtype=np.float64)) if len(parcel_mass) else 0.0
    target_mass = float(np.multiply(parcel_mass, er).sum(dtype=np.float64)) if len(parcel_mass) else 0.0
    recorded_impact_mass = tracking.get("impact_mass_kg")
    if recorded_impact_mass is not None:
        value = _float(recorded_impact_mass, "tracking.impact_mass_kg")
        if not math.isclose(value, impact_mass, rel_tol=2.0e-8, abs_tol=1.0e-18):
            _record_issue(issues, "impact_mass_mismatch", "error", "summary impact mass differs from NPZ parcel mass sum", summary=value, npz=impact_mass)
    recorded_target_mass = erosion.get("total_target_mass_kg")
    if recorded_target_mass is not None:
        value = _float(recorded_target_mass, "face_erosion.total_target_mass_kg")
        if not math.isclose(value, target_mass, rel_tol=2.0e-7, abs_tol=1.0e-18):
            _record_issue(issues, "target_mass_mismatch", "error", "summary target mass differs from NPZ parcel_mass*ER sum", summary=value, npz=target_mass)
    surface_faces = int(erosion.get("surface_faces", 0) or 0)
    if len(parent) and (np.any(parent < 0) or (surface_faces and np.any(parent >= surface_faces))):
        _record_issue(issues, "parent_face_out_of_range", "error", "parent_face_id is outside the declared face domain", max_parent=int(parent.max()), surface_faces=surface_faces)
    strict = parameters.get("collision_policy") in STRICT_POLICIES
    fallback = int(tracking.get("fallback_hits", 0) or 0)
    if strict and fallback != 0:
        _record_issue(issues, "strict_fallback_hits", "error", "strict_v2 run contains fallback hits", fallback_hits=fallback)
    ratio = tracking.get("physical_strict_hit_ratio")
    if strict and ratio is not None and not math.isclose(_float(ratio, "physical_strict_hit_ratio"), 1.0, abs_tol=1.0e-12):
        _record_issue(issues, "strict_hit_ratio_failed", "error", "strict_v2 physical hit ratio is not 1.0", ratio=ratio)
    hit_kind = arrays.get("hit_kind")
    if strict and hit_kind is not None and len(hit_kind) and np.any(np.asarray(hit_kind) != 0):
        _record_issue(issues, "strict_hit_kind_invalid", "error", "strict_v2 contains a non-intersection hit_kind")
    total_injected = tracking.get("total_injection_mass_kg")
    if total_injected is not None and _float(total_injected, "total_injection_mass_kg") <= 0.0:
        _record_issue(issues, "invalid_injection_mass", "error", "total injection mass must be positive")
    return {
        "n_hits": int(n_hits),
        "impact_mass_kg_from_npz": impact_mass,
        "target_mass_kg_from_npz": target_mass,
        "hit_parcels": int(tracking.get("hit_parcels", 0) or 0),
        "n_parcels": int(parameters.get("n_parcels", 0) or 0),
        "fallback_hits": fallback,
        "near_miss_count": int(tracking.get("near_miss_diagnostics", 0) or 0),
        "substep_count": int(tracking.get("substep_count", 0) or 0),
        "collision_event_limit_count": int(tracking.get("collision_event_limit_count", 0) or 0),
        "seed": parameters.get("seed"),
        "artifact_set_id": summary.get("artifact_identity", {}).get("artifact_set_id") if isinstance(summary.get("artifact_identity"), Mapping) else None,
        "collision_policy": parameters.get("collision_policy"),
        "collision_backend": parameters.get("collision_backend"),
        "field_time": (summary.get("cfd") or {}).get("field_time") if isinstance(summary.get("cfd"), Mapping) else None,
    }


def audit_run(run_dir: str | Path) -> dict[str, Any]:
    """Audit one immutable run directory without changing any artifact."""
    root = Path(run_dir).resolve()
    issues: list[dict[str, Any]] = []
    if not root.is_dir():
        raise ParticleDataAuditError(f"run directory does not exist: {root}")
    manifest_path = root / "particle_run_manifest.json"
    summary_path = root / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise ParticleDataAuditError(f"run must contain summary.json and particle_run_manifest.json: {root}")
    manifest = _json(manifest_path)
    summary = _json(summary_path)
    hashes = _required_file_checks(root, manifest, issues)
    arrays = _load_arrays(root / "particle_hits.npz", issues) if (root / "particle_hits.npz").is_file() else {}
    stats = _summary_consistency(summary, arrays, issues)
    stats["implementation_hashes"] = manifest.get("implementation_hashes")
    config = manifest.get("config") if isinstance(manifest.get("config"), Mapping) else {}
    # Exclude run identity fields so the signature represents physical/model
    # settings, while preserving all sampling, integration and collision knobs.
    physical_config = dict(config)
    physical_config.pop("seed", None)
    physical_config.pop("n_parcels", None)
    physical_config.pop("n_injected", None)
    physical_config.pop("output_tag", None)
    stats["physical_config_sha256"] = canonical_sha256(physical_config)
    if config:
        parameters = summary.get("parameters") if isinstance(summary.get("parameters"), Mapping) else {}
        for key in ("seed", "n_parcels", "collision_policy", "collision_backend", "cfd_field_time"):
            summary_key = "field_time" if key == "cfd_field_time" else key
            expected = config.get(key)
            actual = stats.get(summary_key, parameters.get(summary_key))
            if expected is not None and actual is not None and str(expected) != str(actual):
                _record_issue(issues, "manifest_summary_config_mismatch", "error", f"manifest config differs from summary for {key}", field=key, manifest=expected, summary=actual)
    status = "failed" if any(item["severity"] == "error" for item in issues) else ("warning" if issues else "passed")
    return {
        "version": VERSION,
        "status": status,
        "run_directory": str(root),
        "file_sha256": hashes,
        "stats": stats,
        "issues": issues,
        "audit_sha256": canonical_sha256({
            "version": VERSION,
            "status": status,
            "run_directory": str(root),
            "file_sha256": hashes,
            "stats": stats,
            "issues": issues,
        }),
    }


def audit_run_root(run_root: str | Path) -> dict[str, Any]:
    """Audit all immutable strict-v2 runs below an artifact root."""
    root = Path(run_root).resolve()
    if not root.is_dir():
        raise ParticleDataAuditError(f"run root does not exist: {root}")
    candidates = sorted(
        path.parent for path in root.rglob("particle_hits.npz") if path.parent.name != ""
    )
    reports: list[dict[str, Any]] = []
    for directory in candidates:
        if (directory / "summary.json").is_file() and (directory / "particle_run_manifest.json").is_file():
            reports.append(audit_run(directory))
    issues: list[dict[str, Any]] = []
    seeds: list[Any] = []
    identities: list[Any] = []
    implementation_hashes: list[str] = []
    physical_configs: list[tuple[Any, ...]] = []
    for report in reports:
        seeds.append(report["stats"].get("seed"))
        identities.append(report["stats"].get("artifact_set_id"))
        impl = report["stats"].get("implementation_hashes")
        if isinstance(impl, Mapping):
            implementation_hashes.append(canonical_sha256(dict(impl)))
        stats = report["stats"]
        physical_configs.append((stats.get("n_parcels"), stats.get("collision_policy"), stats.get("collision_backend"), stats.get("field_time")))
        issues.extend(report["issues"])
    if len(seeds) != len(set(seeds)):
        _record_issue(issues, "duplicate_seed", "error", "the audited run root contains duplicate seeds")
    nonempty_identity = {value for value in identities if value}
    if len(nonempty_identity) > 1:
        _record_issue(issues, "artifact_identity_mismatch", "error", "runs do not share one artifact_set_id", identities=sorted(nonempty_identity))
    nonempty_configs = {value for value in physical_configs if any(item is not None for item in value)}
    if len(nonempty_configs) > 1:
        _record_issue(issues, "physical_config_mismatch", "error", "runs do not share one physical tracking configuration")
    if len(set(implementation_hashes)) > 1:
        _record_issue(issues, "implementation_version_mismatch", "error", "runs mix tracker or collision implementation hashes")
    if not reports:
        _record_issue(issues, "no_immutable_runs", "error", "no auditable immutable run directories were found")

    # Cross-run comparisons are only meaningful within a homogeneous physical
    # configuration and implementation version.  Keep mixed historical runs
    # visible, but expose eligible groups so they can be audited independently.
    groups: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        stats = report["stats"]
        impl_key = canonical_sha256(stats.get("implementation_hashes") or {})
        key = json.dumps(
            {
                "n_parcels": stats.get("n_parcels"),
                "collision_policy": stats.get("collision_policy"),
                "collision_backend": stats.get("collision_backend"),
                "field_time": stats.get("field_time"),
                "implementation_hashes_sha256": impl_key,
                "physical_config_sha256": stats.get("physical_config_sha256"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        groups.setdefault(key, []).append(report)

    grouped_reports: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        group_stats = [item["stats"] for item in group]
        group_issues = [issue for item in group for issue in item["issues"]]
        group_hit_total = sum(int(item.get("n_hits", 0)) for item in group_stats)
        group_parcel_total = sum(int(item.get("n_parcels", 0)) for item in group_stats)
        group_hit_parcels = sum(int(item.get("hit_parcels", 0)) for item in group_stats)
        group_substeps = sum(int(item.get("substep_count", 0)) for item in group_stats)
        group_near_misses = sum(int(item.get("near_miss_count", 0)) for item in group_stats)
        grouped_reports.append(
            {
                "group_key": json.loads(key),
                "run_count": len(group),
                "seeds": sorted(item.get("seed") for item in group_stats),
                "individual_statuses": Counter(item["status"] for item in group),
                "error_issue_count": sum(issue.get("severity") == "error" for issue in group_issues),
                "eligible_for_same_configuration_replay": len(group) >= 2 and not any(issue.get("severity") == "error" for issue in group_issues),
                "performance": {
                    "hit_total": group_hit_total,
                    "parcel_total": group_parcel_total,
                    "hit_parcel_total": group_hit_parcels,
                    "hit_parcel_rate": group_hit_parcels / group_parcel_total if group_parcel_total else 0.0,
                    "substeps_per_parcel": group_substeps / group_parcel_total if group_parcel_total else 0.0,
                    "near_miss_to_impact_ratio": group_near_misses / max(group_hit_total, 1),
                },
            }
        )
    hit_total = sum(int(report["stats"].get("n_hits", 0)) for report in reports)
    parcel_total = sum(int(report["stats"].get("n_parcels", 0)) for report in reports)
    hit_parcels = sum(int(report["stats"].get("hit_parcels", 0)) for report in reports)
    substeps = sum(int(report["stats"].get("substep_count", 0)) for report in reports)
    near_misses = sum(int(report["stats"].get("near_miss_count", 0)) for report in reports)
    hit_rate = hit_parcels / parcel_total if parcel_total else 0.0
    near_miss_ratio = near_misses / max(hit_total, 1)
    substeps_per_parcel = substeps / max(parcel_total, 1)
    if hit_rate < 0.10:
        _record_issue(issues, "sparse_hit_support", "warning", "physical hit support is below 10 percent of parcels", hit_rate=hit_rate)
    if near_miss_ratio > 10.0:
        _record_issue(issues, "near_miss_overhead", "warning", "endpoint near-miss diagnostics greatly exceed true impacts", near_miss_ratio=near_miss_ratio)
    if substeps_per_parcel > 1000.0:
        _record_issue(issues, "high_substep_cost", "warning", "substep cost is high enough to threaten high-particle confirmation runtime", substeps_per_parcel=substeps_per_parcel)
    status = "failed" if any(item["severity"] == "error" for item in issues) else ("warning" if issues else "passed")
    unsigned = {
        "version": VERSION,
        "status": status,
        "run_root": str(root),
        "run_count": len(reports),
        "seed_values": seeds,
        "reports": reports,
        "cross_run": {
            "hit_total": hit_total,
            "parcel_total": parcel_total,
            "hit_parcel_total": hit_parcels,
            "hit_parcel_rate": hit_rate,
            "substeps_per_parcel": substeps_per_parcel,
            "near_miss_to_impact_ratio": near_miss_ratio,
        },
        "configuration_groups": grouped_reports,
        "interpretation": {
            "root_mixed_configuration_status": "mixed_historical_and_implementation_groups" if len(grouped_reports) > 1 else "single_group",
            "individual_run_integrity": "passed" if not any(item["status"] == "failed" for item in reports) else "failed",
            "group_comparisons_must_use_group_key": True,
            "performance_warnings_are_diagnostics": True,
        },
        "issues": issues,
    }
    unsigned["audit_sha256"] = canonical_sha256(unsigned)
    return unsigned


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Step30 particle artifacts on a remote server")
    parser.add_argument("--run-root", type=Path, required=True, help="artifact root containing strict_v2 immutable runs")
    parser.add_argument("--output", type=Path, required=True, help="write an immutable JSON audit report")
    parser.add_argument("--selftest", action="store_true", help="reserved for remote test harness")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _remote_guard()
    report = audit_run_root(args.run_root)
    safe_json_dump(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2 if report["status"] == "warning" else 3


if __name__ == "__main__":
    raise SystemExit(main())
