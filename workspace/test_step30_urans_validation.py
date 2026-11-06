"""Hermetic tests for production URANS time-mean validation gates."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import numpy as np

from step30_urans_validation import FLOW_THROUGH_TIME_S, _flux_relative_mean_change, run


def _vector_field(path: Path, values: list[tuple[float, float, float]], *, bucket=False) -> None:
    body = "\n".join(f"({x} {y} {z})" for x, y, z in values)
    boundary = ""
    if bucket:
        boundary = f'''boundaryField
{{
    bucket
    {{
        type calculated;
        value nonuniform List<vector>
        {len(values)}
        (
{body}
        )
        ;
    }}
}}
'''
    path.write_text(
        f'''internalField nonuniform List<vector>
{len(values)}
(
{body}
)
;
{boundary}''',
        encoding="utf-8",
    )


def _scalar_field(path: Path, values: list[float]) -> None:
    body = "\n".join(str(value) for value in values)
    path.write_text(
        f'''internalField nonuniform List<scalar>
{len(values)}
(
{body}
)
;
''',
        encoding="utf-8",
    )


def _fixture(root: Path, *, bounding=False) -> tuple[str, str, str, int]:
    start = Decimal("1")
    end = start + Decimal(20) * FLOW_THROUGH_TIME_S
    previous = end - FLOW_THROUGH_TIME_S
    field_time = str(end)
    previous_time = str(previous)
    for time_name in (previous_time, field_time):
        directory = root / time_name
        directory.mkdir(parents=True)
        _vector_field(directory / "UMean", [(100.0, 0.0, 0.0), (50.0, 0.0, 0.0)])
        _scalar_field(directory / "pMean", [10.0, 20.0])
        _scalar_field(directory / "kMean", [1.0, 2.0])
        _scalar_field(directory / "omegaMean", [100.0, 200.0])
        _vector_field(directory / "wallShearStressMean", [(2.0, 0.0, 0.0)], bucket=True)

    sample_count = 20
    times = [start + Decimal(index + 1) * (end - start) / sample_count for index in range(sample_count)]
    log_lines = []
    for value in times:
        log_lines.extend(["Courant Number mean: 0.01 max: 0.45", f"Time = {value}"])
    if bounding:
        log_lines.append("bounding k, min: -1e-8 max: 2 average: 1")
    (root / "log.urans.average").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    def monitor(function: str, name: str, rows: list[str]) -> None:
        path = root / "postProcessing" / function / str(start) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic\n" + "\n".join(rows) + "\n", encoding="utf-8")

    force_rows = [f"{value} (1 2 3) (1 2 3) (0 0 0)" for value in times]
    force_rows.insert(0, "0.5 (1e9 0 0) (1e9 0 0) (0 0 0)")
    monitor("bucketForces", "force.dat", force_rows)
    for name, flux in {
        "inletFlux": -6.5,
        "jetFlux": -4.5,
        "outletFlux": 8.7,
        "sideFlux": 2.3,
    }.items():
        monitor(
            name,
            "surfaceFieldValue.dat",
            ["0.5 1e9", *(f"{value} {flux}" for value in times)],
        )
    monitor(
        "flowProbes",
        "U",
        [
            "0.5 (1e9 0 0) (1e9 0 0) (1e9 0 0)",
            *(f"{value} (100 0 0) (50 0 0) (25 0 0)" for value in times),
        ],
    )
    monitor(
        "flowProbes",
        "p",
        ["0.5 1e9 1e9 1e9", *(f"{value} 10 20 30" for value in times)],
    )
    return field_time, previous_time, str(start), sample_count


def test_urans_validation_accepts_all_strict_gates(tmp_path: Path) -> None:
    field_time, previous_time, start, samples = _fixture(tmp_path)
    result = run(
        tmp_path,
        field_time,
        previous_time,
        start,
        samples,
        tmp_path / "validation.json",
    )
    assert result["version"] == "urans-time-mean-validation-v2"
    assert result["accepted"] is True
    assert all(result["gates"].values())
    assert result["metrics"]["max_courant_number"] == 0.45
    assert set(result["metrics"]["force_late_mean_relative_change"]) == {
        "pressure", "viscous", "total",
    }
    assert result["metrics"]["flux_normalized_imbalance"] < 1e-12


def test_flux_change_uses_principal_scale_for_near_zero_patch_flux() -> None:
    flux = {
        "inletFlux": np.full(20, -6.5),
        "jetFlux": np.full(20, -4.5),
        "outletFlux": np.full(20, 11.0),
        "sideFlux": np.concatenate((np.full(10, 0.8), np.full(10, 0.9))),
    }

    changes, recent = _flux_relative_mean_change(flux)

    assert recent["sideFlux"] == 0.9
    assert changes["sideFlux"] < 0.005


def test_flux_change_keeps_strict_relative_scale_for_principal_flux() -> None:
    flux = {
        "inletFlux": np.concatenate((np.full(10, -6.5), np.full(10, -6.0))),
        "jetFlux": np.full(20, -4.5),
        "outletFlux": np.full(20, 10.5),
        "sideFlux": np.zeros(20),
    }

    changes, _ = _flux_relative_mean_change(flux)

    assert changes["inletFlux"] == 0.5 / 6.0


def test_urans_validation_rejects_turbulence_bounding(tmp_path: Path) -> None:
    field_time, previous_time, start, samples = _fixture(tmp_path, bounding=True)
    result = run(
        tmp_path,
        field_time,
        previous_time,
        start,
        samples,
        tmp_path / "validation.json",
    )
    assert result["gates"]["no_turbulence_bounding"] is False
    assert result["accepted"] is False
