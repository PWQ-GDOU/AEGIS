"""Auditable geometry, unit, and physical-selection audit for the STEP export.

The audit records what the official geometry actually contains.  It does not
infer ROOT_INTERFACE or WETTED_SURFACE from Gmsh tags, bounding boxes, or
visual appearance, and it never turns an embedded unit declaration into a
formal FEM-ready unit pass when downstream scale evidence disagrees.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from step30_validation_common import canonical_sha256, safe_json_dump, sha256_file


VERSION = "step30-fem-geometry-audit-v1"
EXPECTED_STEP_SHA256 = "521fbaf7688ad1f0e063dedb08c360135026ce60e5e371b4a196294b18cfe21d"


class GeometryAuditError(ValueError):
    """Raised when the authoritative STEP cannot be audited."""


def _artifact(path: str | Path, *, role: str, required: bool = False) -> dict[str, Any]:
    target = Path(path).resolve()
    if not target.is_file():
        if required:
            raise GeometryAuditError(f"{role} does not exist: {target}")
        return {
            "role": role,
            "status": "missing",
            "path": str(target),
            "sha256": None,
            "bytes": None,
        }
    return {
        "role": role,
        "status": "present",
        "path": str(target),
        "sha256": sha256_file(target),
        "bytes": int(target.stat().st_size),
    }


def _parse_number(token: str) -> float:
    try:
        value = float(token.strip().replace("D", "E").replace("d", "e"))
    except ValueError as exc:
        raise GeometryAuditError(f"invalid STEP coordinate: {token!r}") from exc
    if not math.isfinite(value):
        raise GeometryAuditError("STEP coordinate is not finite")
    return value


def parse_step_coordinate_points(text: str) -> dict[str, Any]:
    """Parse Cartesian points and the 44 vertex-to-point references."""
    point_pattern = re.compile(
        r"#(\d+)\s*=\s*CARTESIAN_POINT\s*\(\s*'[^']*'\s*,\s*"
        r"\(\s*([^()]*)\)\s*\)\s*;",
        re.IGNORECASE,
    )
    points: dict[int, list[float]] = {}
    for match in point_pattern.finditer(text):
        values = [_parse_number(item) for item in match.group(2).split(",")]
        if len(values) != 3:
            raise GeometryAuditError("CARTESIAN_POINT must have three coordinates")
        points[int(match.group(1))] = values

    vertex_pattern = re.compile(
        r"#(\d+)\s*=\s*VERTEX_POINT\s*\(\s*'[^']*'\s*,\s*#(\d+)\s*\)\s*;",
        re.IGNORECASE,
    )
    vertex_point_ids = [int(match.group(2)) for match in vertex_pattern.finditer(text)]
    if not points or not vertex_point_ids:
        raise GeometryAuditError("STEP has no parseable Cartesian points/vertices")
    missing = sorted(set(vertex_point_ids) - set(points))
    if missing:
        raise GeometryAuditError(f"VERTEX_POINT references missing CARTESIAN_POINT ids: {missing[:5]}")

    all_values = np.asarray(list(points.values()), dtype=np.float64)
    vertex_values = np.asarray([points[index] for index in vertex_point_ids], dtype=np.float64)

    def bbox(values: np.ndarray) -> list[float]:
        return [float(value) for value in np.concatenate((values.min(axis=0), values.max(axis=0)))]

    return {
        "cartesian_point_count": int(len(points)),
        "vertex_count": int(len(vertex_point_ids)),
        "control_point_bbox_native": bbox(all_values),
        "vertex_point_bbox_native": bbox(vertex_values),
    }


def parse_step_length_unit(text: str) -> dict[str, Any]:
    """Return the embedded STEP length-unit declaration without interpreting scale."""
    pattern = re.compile(
        r"LENGTH_UNIT\s*\(\s*\)\s*NAMED_UNIT\s*\(\s*\*\s*\)\s*"
        r"SI_UNIT\s*\(\s*\$\s*,\s*\.([A-Z]+)\.\s*\)",
        re.IGNORECASE | re.DOTALL,
    )
    matches = [match.upper() for match in pattern.findall(text)]
    if not matches:
        return {
            "status": "blocked",
            "value": None,
            "raw_si_unit_names": [],
            "reason": "no embedded STEP LENGTH_UNIT/SI_UNIT declaration found",
        }
    unique = sorted(set(matches))
    values = {"METRE": "m", "METER": "m", "MILLIMETRE": "mm", "MILLIMETER": "mm"}
    if len(unique) != 1 or unique[0] not in values:
        return {
            "status": "blocked",
            "value": None,
            "raw_si_unit_names": unique,
            "reason": "embedded STEP length unit is absent, ambiguous, or unsupported",
        }
    return {
        "status": "declared",
        "value": values[unique[0]],
        "raw_si_unit_names": unique,
        "reason": "embedded STEP length-unit declaration was parsed",
    }


def _read_catalog_status(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    record = _artifact(target, role="surface_catalog", required=False)
    if record["status"] != "present":
        record["parse_status"] = "blocked"
        record["reason"] = "surface catalog artifact is missing"
        return record
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        record["parse_status"] = "blocked"
        record["reason"] = f"surface catalog is not valid JSON: {exc}"
        return record
    record["parse_status"] = "readable"
    record["catalog_version"] = data.get("version")
    record["catalog_source_sha256"] = data.get("source_sha256")
    record["catalog_surface_count"] = data.get("geometry", {}).get("surface_count")
    record["catalog_note"] = data.get("note")
    return record


def _gmsh_metrics(step_path: Path) -> dict[str, Any]:
    try:
        import gmsh
    except ModuleNotFoundError as exc:
        raise GeometryAuditError(
            "Gmsh is required for STEP topology audit; run with uv --with gmsh"
        ) from exc
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.open(str(step_path))
        volumes = gmsh.model.getEntities(3)
        surfaces = gmsh.model.getEntities(2)
        edges = gmsh.model.getEntities(1)
        vertices = gmsh.model.getEntities(0)
        if len(volumes) != 1:
            raise GeometryAuditError(f"expected one Gmsh volume, found {len(volumes)}")
        volume_tag = int(volumes[0][1])
        bbox = gmsh.model.getBoundingBox(3, volume_tag)
        volume_mass = float(gmsh.model.occ.getMass(3, volume_tag))
        surface_mass = float(sum(gmsh.model.occ.getMass(2, tag) for _, tag in surfaces))
        return {
            "status": "passed",
            "volume_count": int(len(volumes)),
            "surface_count": int(len(surfaces)),
            "edge_count": int(len(edges)),
            "vertex_count": int(len(vertices)),
            "volume_tag": volume_tag,
            "bbox_native": [float(value) for value in bbox],
            "volume_mass_native": volume_mass,
            "boundary_area_sum_native": surface_mass,
            "surface_tags": [int(tag) for _, tag in surfaces],
        }
    finally:
        gmsh.finalize()


def _bbox_spans(bbox: Sequence[float]) -> np.ndarray:
    values = np.asarray(bbox, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise GeometryAuditError("bbox must contain six finite values")
    spans = values[3:] - values[:3]
    if np.any(spans <= 0.0):
        raise GeometryAuditError("bbox spans must be positive")
    return spans


def _scale_consistency(step_points: dict[str, Any], gmsh_metrics: dict[str, Any]) -> dict[str, Any]:
    raw_bbox = np.asarray(step_points["control_point_bbox_native"], dtype=np.float64)
    gmsh_bbox = np.asarray(gmsh_metrics["bbox_native"], dtype=np.float64)
    raw_spans = _bbox_spans(raw_bbox)
    gmsh_spans = _bbox_spans(gmsh_bbox)
    ratios = gmsh_spans / raw_spans
    median_ratio = float(np.median(ratios))
    spread = float(np.max(ratios) / np.min(ratios))
    uniform = bool(np.allclose(ratios, median_ratio, rtol=0.05, atol=0.0))
    return {
        "status": "passed" if uniform else "blocked",
        "gmsh_bbox_to_step_control_span_ratios": [float(value) for value in ratios],
        "median_scale_ratio": median_ratio,
        "max_to_min_scale_ratio": spread,
        "uniform_scale_within_5_percent": uniform,
        "reason": (
            "Gmsh and STEP coordinate extents admit one uniform scale within 5%"
            if uniform
            else "Gmsh native extents do not admit one uniform scale within 5%; downstream unit semantics remain unresolved"
        ),
    }


def build_geometry_audit(
    step_path: str | Path,
    *,
    comsol_java_path: str | Path | None = None,
    export_log_path: str | Path | None = None,
    surface_catalog_path: str | Path | None = None,
) -> dict[str, Any]:
    step = Path(step_path).resolve()
    if not step.is_file():
        raise GeometryAuditError(f"authoritative STEP does not exist: {step}")
    step_text = step.read_text(encoding="utf-8", errors="replace")
    embedded_unit = parse_step_length_unit(step_text)
    step_points = parse_step_coordinate_points(step_text)
    gmsh_metrics = _gmsh_metrics(step)
    scale_check = _scale_consistency(step_points, gmsh_metrics)

    export_root = step.parent.parent
    comsol_java = (
        Path(comsol_java_path)
        if comsol_java_path is not None
        else export_root / "work" / "ComsolBucketExport.java"
    )
    export_log = (
        Path(export_log_path)
        if export_log_path is not None
        else export_root / "work" / "target_export.log"
    )
    surface_catalog = (
        Path(surface_catalog_path)
        if surface_catalog_path is not None
        else export_root / "step_surface_catalog.json"
    )

    artifacts = [
        _artifact(step, role="authoritative_step", required=True),
        _artifact(comsol_java, role="comsol_export_java"),
        _artifact(export_log, role="comsol_export_log"),
        _read_catalog_status(surface_catalog),
    ]
    blockers = [
        "ROOT_INTERFACE has no independently confirmed CAD named selection",
        "WETTED_SURFACE has no independently confirmed CAD named selection",
        "AI5_04 numeric patch and gauge direction are not contained in the STEP export",
        "Gmsh native geometry scale is not uniformly consistent with the parsed STEP coordinates",
    ]
    if any(item["status"] != "present" for item in artifacts[1:3]):
        blockers.append("COMSOL export script/log binding is incomplete")

    report: dict[str, Any] = {
        "version": VERSION,
        "status": "blocked",
        "fem_gate": "blocked",
        "real_fem_evidence": False,
        "source_artifacts": artifacts,
        "geometry": {
            "step_sha256_matches_expected": artifacts[0]["sha256"] == EXPECTED_STEP_SHA256,
            "step_sha256": artifacts[0]["sha256"],
            "embedded_length_unit": embedded_unit,
            "step_coordinate_points": step_points,
            "gmsh_metrics": gmsh_metrics,
            "scale_consistency": scale_check,
            "comsol_export_script_claim": {
                "status": "declared",
                "parasolid_length_unit": "m",
                "step_length_unit": "m",
                "basis": "ComsolBucketExport.java calls setLengthUnit('m') and setLengthUnitSTEP('m')",
            },
        },
        "physical_selection_audit": {
            "ROOT_INTERFACE": {
                "status": "blocked",
                "entity_dimension": 2,
                "entity_tags": [],
                "confirmation_method": None,
                "reason": "Geometry topology and raw Gmsh surface tags do not establish runner-interface semantics",
            },
            "WETTED_SURFACE": {
                "status": "blocked",
                "entity_dimension": 2,
                "entity_tags": [],
                "confirmation_method": None,
                "reason": "Geometry topology and raw Gmsh surface tags do not establish wetted-face semantics or root exclusion",
            },
            "AI5_04": {
                "status": "blocked",
                "entity_dimension": 2,
                "entity_tags": [],
                "patch_center": None,
                "gauge_directions": [],
                "reason": "STEP export contains no machine-readable AI5 patch or gauge direction",
            },
        },
        "selection_inference_policy": {
            "raw_surface_tags_are_physical_names": False,
            "bbox_heuristics_are_admissible": False,
            "visual_guess_is_admissible": False,
            "required_next_evidence": "independent CAD named-selection/face-list report bound to this STEP SHA-256",
        },
        "blockers": blockers,
        "limitations": [
            "The embedded STEP unit is recorded as declared evidence, not as a formal unit pass.",
            "Gmsh mass values are reported in native reader units and are not relabelled as m3 or m2.",
            "This audit does not generate a mesh, CalculiX case, or FEM result.",
            "No AI5 data are used to calibrate geometry, material, boundary conditions, or gauge directions.",
        ],
    }
    report["audit_sha256"] = canonical_sha256(report)
    return report


def selftest() -> bool:
    text = """
    #1=CARTESIAN_POINT('',(0.,0.,0.));
    #2=CARTESIAN_POINT('',(1.,2.,3.));
    #3=VERTEX_POINT('',#1);
    #4=VERTEX_POINT('',#2);
    #5=(
    LENGTH_UNIT()
    NAMED_UNIT(*)
    SI_UNIT($,.METRE.)
    );
    """
    unit = parse_step_length_unit(text)
    points = parse_step_coordinate_points(text)
    assert unit["status"] == "declared" and unit["value"] == "m"
    assert points["cartesian_point_count"] == 2 and points["vertex_count"] == 2
    assert points["control_point_bbox_native"] == [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
    print("step30_fem_geometry_audit selftest: PASS")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit STEP geometry, units, and physical-selection evidence")
    parser.add_argument("--step", type=Path, required=False)
    parser.add_argument("--comsol-java", type=Path)
    parser.add_argument("--export-log", type=Path)
    parser.add_argument("--surface-catalog", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if args.step is None:
        parser.error("--step is required unless --selftest is used")
    report = build_geometry_audit(
        args.step,
        comsol_java_path=args.comsol_java,
        export_log_path=args.export_log,
        surface_catalog_path=args.surface_catalog,
    )
    if args.output is not None:
        safe_json_dump(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
