"""Regression tests for source-bound analytic reference transforms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import step30_lagrangian_particle_tracking as tracking


def _cfd(source_hash: str) -> dict:
    return {"artifact_manifest": {"si_stl": {"sha256": source_hash}}}


def _write_transform(path: Path, source_hash: str) -> None:
    path.write_text(
        json.dumps(
            {
                "version": "step30-analytic-reference-transform-v1",
                "convention": "row-vector: global_mm = local_mm @ R + translation_mm",
                "source_geometry_sha256": source_hash,
                "R": np.eye(3).tolist(),
                "translation_mm": [1.0, 2.0, 3.0],
            }
        ),
        encoding="utf-8",
    )


def test_reference_transform_loads_and_binds_source_hash(tmp_path: Path) -> None:
    path = tmp_path / "reference.json"
    _write_transform(path, "source-a")
    cfg = tracking.TrackingConfig(analytic_reference_transform_path=str(path))

    result = tracking._load_reference_transform(tmp_path, cfg, _cfd("source-a"))

    assert result["version"] == "step30-analytic-reference-transform-v1"
    assert np.allclose(result["R"], np.eye(3))
    assert np.allclose(result["nozzle_pos"], [1.0, 2.0, 3.0])
    assert result["sha256"]


def test_reference_transform_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "reference.json"
    _write_transform(path, "source-a")
    cfg = tracking.TrackingConfig(analytic_reference_transform_path=str(path))

    with pytest.raises(ValueError, match="source geometry hash mismatch"):
        tracking._load_reference_transform(tmp_path, cfg, _cfd("source-b"))


def test_reference_transform_rejects_reflection(tmp_path: Path) -> None:
    path = tmp_path / "reference.json"
    _write_transform(path, "source-a")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["R"][0][0] = -1.0
    path.write_text(json.dumps(data), encoding="utf-8")
    cfg = tracking.TrackingConfig(analytic_reference_transform_path=str(path))

    with pytest.raises(ValueError, match="proper rigid rotation"):
        tracking._load_reference_transform(tmp_path, cfg, _cfd("source-a"))


def test_reference_transform_maps_authoritative_bounds(tmp_path: Path) -> None:
    transform = {
        "version": "step30-analytic-reference-transform-v1",
        "convention": "row-vector: global_mm = local_mm @ R + translation_mm",
        "source_geometry_sha256": "source-a",
        "R": [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        "translation_mm": [10.0, 20.0, 30.0],
    }
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(transform), encoding="utf-8")
    cfg = tracking.TrackingConfig(analytic_reference_transform_path=str(path))
    loaded = tracking._load_reference_transform(tmp_path, cfg, _cfd("source-a"))
    local = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(local @ loaded["R"] + loaded["nozzle_pos"], [[8.0, 21.0, 33.0]])
