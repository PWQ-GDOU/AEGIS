from __future__ import annotations

from pathlib import Path

import pytest

from step30_fem_geometry_audit import (
    EXPECTED_STEP_SHA256,
    build_geometry_audit,
    parse_step_coordinate_points,
    parse_step_length_unit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP_PATH = PROJECT_ROOT / "workspace" / "output" / "step30_fem_remote_export_v1" / "output" / "single_bucket_target_authoritative.step"


def test_step_unit_and_point_parser() -> None:
    text = """
    #1=CARTESIAN_POINT('',(0.,0.,0.));
    #2=CARTESIAN_POINT('',(1.,2.,3.));
    #3=VERTEX_POINT('',#1);
    #4=VERTEX_POINT('',#2);
    #5=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.));
    """
    assert parse_step_length_unit(text) == {
        "status": "declared",
        "value": "m",
        "raw_si_unit_names": ["METRE"],
        "reason": "embedded STEP length-unit declaration was parsed",
    }
    parsed = parse_step_coordinate_points(text)
    assert parsed["cartesian_point_count"] == 2
    assert parsed["vertex_count"] == 2
    assert parsed["vertex_point_bbox_native"] == [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]


def test_authoritative_geometry_audit_remains_blocked() -> None:
    if not STEP_PATH.is_file():
        pytest.skip("recovered authoritative STEP is not present")
    report = build_geometry_audit(STEP_PATH)
    assert report["status"] == "blocked"
    assert report["fem_gate"] == "blocked"
    assert report["geometry"]["step_sha256"] == EXPECTED_STEP_SHA256
    assert report["geometry"]["embedded_length_unit"]["value"] == "m"
    assert report["geometry"]["embedded_length_unit"]["status"] == "declared"
    assert report["geometry"]["gmsh_metrics"]["volume_count"] == 1
    assert report["geometry"]["gmsh_metrics"]["surface_count"] == 30
    assert report["physical_selection_audit"]["ROOT_INTERFACE"]["status"] == "blocked"
    assert report["physical_selection_audit"]["WETTED_SURFACE"]["status"] == "blocked"
    assert report["geometry"]["scale_consistency"]["status"] == "blocked"
