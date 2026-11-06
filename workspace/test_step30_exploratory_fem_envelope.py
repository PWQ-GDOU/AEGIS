from __future__ import annotations

import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import step30_exploratory_fem_envelope as envelope
from step30_exploratory_fem_envelope import (
    DENSITIES_KG_M3,
    ExploratoryFEMEnvelopeError,
    build_blocked_envelope_report,
    build_case_matrix,
)
from step30_validation_common import canonical_sha256


def test_case_matrix_has_all_anonymous_assumptions() -> None:
    cases = build_case_matrix(range(1, 31))
    assert len(cases) == 30 * 2 * len(DENSITIES_KG_M3) * 3
    assert all(item["fixed_support_hypothesis"].startswith("fixed_support_surface_") for item in cases)
    assert {item["coordinate_unit_hypothesis"] for item in cases} == {"m", "mm"}


def test_case_matrix_rejects_noncatalogue_surface_set() -> None:
    with pytest.raises(ExploratoryFEMEnvelopeError):
        build_case_matrix(range(1, 30))


def test_missing_solver_dependencies_make_report_ineligible_not_formal(
    tmp_path: pathlib.Path,
) -> None:
    step = tmp_path / "bucket.step"
    step.write_bytes(b"step fixture")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "source_sha256": __import__("hashlib").sha256(b"step fixture").hexdigest(),
                "surfaces": [{"surface_tag": index} for index in range(1, 31)],
            }
        ),
        encoding="utf-8",
    )
    report = build_blocked_envelope_report(step, catalog, ccx_executable="definitely_missing_ccx")
    assert report["formal_evidence_status"] == "ineligible"
    assert report["fem_gate"] == "blocked"
    assert report["report_sha256"] == canonical_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    assert report["official_accuracy_gate"] is False


def test_executed_case_records_mesh_and_native_result_paths(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    step = tmp_path / "bucket.step"
    step.write_bytes(b"step fixture")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "source_sha256": __import__("hashlib").sha256(b"step fixture").hexdigest(),
                "surfaces": [{"surface_tag": index} for index in range(1, 31)],
            }
        ),
        encoding="utf-8",
    )
    mesh = {
        "node_ids": np.asarray([1, 2, 3, 4], dtype=np.int64),
        "nodes_native": np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        "tetra_node_ids": np.asarray([[1, 2, 3, 4]], dtype=np.int64),
        "surface_nodes": {index: np.asarray([1, 2, 3], dtype=np.int64) for index in range(1, 31)},
        "mesh_sha256": "mesh-fixture",
        "n_nodes": 4,
        "n_tetra4": 1,
    }

    def fake_preflight(*_args: object, **_kwargs: object) -> dict:
        return {
            "status": "ready_to_run",
            "dependency_status": {"ccx_executable_resolved": "ccx-fixture"},
            "case_matrix": build_case_matrix(range(1, 31)),
        }

    def fake_mesh(_step: pathlib.Path, mesh_path: pathlib.Path, _length: float) -> dict:
        mesh_path.write_text("mesh fixture", encoding="ascii")
        return mesh

    monkeypatch.setattr(envelope, "build_blocked_envelope_report", fake_preflight)
    monkeypatch.setattr(envelope, "_generate_mesh", fake_mesh)
    monkeypatch.setattr(
        envelope.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    report = envelope.run_exploratory_envelope(
        step, catalog, tmp_path / "output", max_cases=1
    )
    case = report["case_results"][0]
    assert case["mesh_path"] == report["mesh"]["path"]
    assert pathlib.Path(case["input_path"]).is_file()
    assert case["native_results"][".dat"]["path"].endswith("case.dat")
    assert case["formal_evidence_status"] == "ineligible"
    assert case["fem_gate"] == "blocked"
