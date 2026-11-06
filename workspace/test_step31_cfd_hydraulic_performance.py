"""Hermetic tests for the Step31 direct CFD hydraulic performance toolchain."""

from __future__ import annotations

import math
import pathlib
import sys
from dataclasses import replace

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from step31_cfd_hydraulic_performance import (
    calculate_axis_torque,
    compute_cfd_hydraulic_performance,
)
from step31_dec_energy_alignment import DECEnergyRow, align_cfd_to_dec, assess_q11_quality
from step31_hydraulic_case_contract import (
    HydraulicCaseContract,
    MeasurementSurface,
    PressureDefinition,
    WaterPhase,
    validate_case_contract,
)
from step31_openfoam_monitor_io import (
    DuplicateTimeConflict,
    MonitorBundle,
    MonitorSeries,
    load_monitor_bundle,
    parse_force_dat,
    parse_moment_dat,
    parse_patch_flux,
    parse_solver_log,
    trapezoidal_time_average,
)


_HASH = "a" * 64
_TIMES = np.asarray([0.0, 0.5, 1.75, 3.0, 4.0])


def _contract(
    *,
    case_type: str = "full_runner_rotating_vof",
    rotating: bool = True,
    vof: bool = True,
    net_head: bool = True,
    energy: bool = True,
) -> HydraulicCaseContract:
    n_rpm = 600.0 if rotating else 0.0
    return HydraulicCaseContract(
        geometry_sha256="1" * 64,
        mesh_sha256="2" * 64,
        case_sha256="3" * 64,
        case_type=case_type,
        D1_m=1.0,
        H_m=100.0,
        n_rpm=n_rpm,
        omega_rad_s=n_rpm * 2.0 * math.pi / 60.0,
        axis=(0.0, 0.0, 2.0),
        CofR_m=(1.0, 2.0, 3.0),
        nozzle_count=6,
        opening_mm=18.0,
        measurement_surfaces=(
            MeasurementSurface(
                name="inlet plane",
                patch="inlet",
                role="inlet",
                flow_sign=-1,
                water_flux_monitor="waterFluxIn",
                total_head_monitor="totalHeadIn" if net_head else None,
                energy_flux_monitor="energyIn" if energy else None,
            ),
            MeasurementSurface(
                name="outlet plane",
                patch="outlet",
                role="outlet",
                flow_sign=1,
                water_flux_monitor="waterFluxOut",
                total_head_monitor="totalHeadOut" if net_head else None,
                energy_flux_monitor="energyOut" if energy else None,
            ),
        ),
        pressure_definition=PressureDefinition(field="p_rgh", kind="kinematic"),
        water_phase=WaterPhase(
            name="water",
            alpha_field="alpha.water",
            density_kg_m3=1000.0,
        ),
        dec_row_identity=None,
        rotating_domain=rotating,
        vof_enabled=vof,
        moment_monitor="runnerMoments",
    )


def _series(
    name: str,
    kind: str,
    units: str,
    values: list[float] | list[tuple[float, float, float]],
    *,
    times: np.ndarray = _TIMES,
) -> MonitorSeries:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        columns = ("value",)
    else:
        columns = ("x", "y", "z")
    return MonitorSeries(
        name=name,
        kind=kind,
        units=units,
        times=times.copy(),
        values=array,
        columns=columns,
    )


def _monitors(*, energy_out_kw: float = 234.2, q_out_lps: float = 2000.0) -> dict[str, MonitorSeries]:
    omega = 600.0 * 2.0 * math.pi / 60.0
    phyd_w = 1000.0 * 9.81 * 2.0 * 100.0
    torque_knm = 0.90 * phyd_w / omega / 1000.0
    count = len(_TIMES)
    return {
        "runnerMoments": _series(
            "runnerMoments",
            "moment",
            "kN*m",
            [(4.0, -3.0, torque_knm)] * count,
        ),
        "waterFluxIn": _series("waterFluxIn", "water_phase_flux", "L/s", [-2000.0] * count),
        "waterFluxOut": _series("waterFluxOut", "water_phase_flux", "L/s", [q_out_lps] * count),
        "totalHeadIn": _series("totalHeadIn", "total_head", "m2/s2", [981.0] * count),
        "totalHeadOut": _series("totalHeadOut", "total_head", "m2/s2", [0.0] * count),
        "energyIn": _series("energyIn", "energy", "kW", [-2000.0] * count),
        "energyOut": _series("energyOut", "energy", "kW", [energy_out_kw] * count),
    }


def _dec_row(*, q11: float = 200.0, eta: float = 90.0, row_number: int = 7) -> DECEnergyRow:
    return DECEnergyRow(
        source_file="autoD1276P21-B01-6_E1_6.txt",
        source_sha256=_HASH,
        row_number=row_number,
        nozzle_count=6,
        opening_mm=18.0,
        n11_rpm=60.0,
        q11_lps=q11,
        calibration_efficiency_pct=eta,
    )


def test_case_contract_freezes_identity_and_normalizes_axis() -> None:
    contract = _contract()
    rebuilt = validate_case_contract(contract.to_dict())
    assert rebuilt.axis == (0.0, 0.0, 1.0)
    assert rebuilt.CofR_m == (1.0, 2.0, 3.0)
    assert rebuilt.performance_eligible is True
    assert rebuilt.contract_sha256 == contract.contract_sha256
    assert len(contract.contract_sha256) == 64


def test_irregular_dt_uses_trapezoidal_time_weighting() -> None:
    times = np.asarray([0.0, 1.0, 3.0])
    values = np.asarray([0.0, 3.0, 3.0])
    weighted = trapezoidal_time_average(times, values)
    assert weighted == pytest.approx(2.5)
    assert weighted != pytest.approx(float(values.mean()))


def test_restart_duplicate_is_deduplicated_with_segment_lineage(tmp_path: pathlib.Path) -> None:
    first = tmp_path / "surfaceFieldValue.dat"
    second = tmp_path / "surfaceFieldValue_2.dat"
    first.write_text("# Time sum(alphaPhi)\n0 -2\n1 -2\n2 -2\n", encoding="utf-8")
    second.write_text("# restarted\n2 -2\n3 -2\n", encoding="utf-8")
    series = parse_patch_flux([first, second], name="waterFluxIn")
    assert series.times.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert len(series.segments) == 2
    assert series.segments[1].duplicate_rows == 1
    assert all(len(segment.sha256) == 64 for segment in series.segments)
    assert len(series.series_sha256) == 64


def test_restart_duplicate_conflict_blocks_bundle(tmp_path: pathlib.Path) -> None:
    first = tmp_path / "surfaceFieldValue.dat"
    second = tmp_path / "surfaceFieldValue_2.dat"
    first.write_text("0 -2\n1 -2\n2 -2\n", encoding="utf-8")
    second.write_text("2 -2.1\n3 -2\n", encoding="utf-8")
    with pytest.raises(DuplicateTimeConflict, match="conflicting duplicate"):
        parse_patch_flux([first, second], name="waterFluxIn")
    bundle = load_monitor_bundle({
        "waterFluxIn": {"parser": "flux", "paths": [first, second]}
    })
    assert bundle.status == "blocked"
    assert bundle.conflicts


def test_force_moment_parsers_and_axis_projection(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "moment.dat"
    path.write_text(
        "# Time pressure viscous porous\n"
        "0 (1 2 3) (4 5 6) (0 0 0)\n"
        "1 (1 2 3) (4 5 6) (0 0 0)\n",
        encoding="utf-8",
    )
    force = parse_force_dat(path, name="runnerForces")
    moment = parse_moment_dat(path, name="runnerMoments")
    assert force.values[0].tolist() == [5.0, 7.0, 9.0]
    assert moment.values[0].tolist() == [5.0, 7.0, 9.0]
    assert calculate_axis_torque(moment.values[0], (0.0, 0.0, 2.0)) == pytest.approx(9.0)
    assert calculate_axis_torque((1.0, 2.0, 3.0), (1.0, 1.0, 0.0)) == pytest.approx(3.0 / math.sqrt(2.0))


def test_solver_log_fallback_extracts_time_block_monitors(tmp_path: pathlib.Path) -> None:
    log = tmp_path / "log.interFoam"
    log.write_text(
        "Time = 0\nrunner moment = (1 2 3)\nwater flux = -2\n"
        "Time = 1\nrunner moment = (2 3 4)\nwater flux = -2\n",
        encoding="utf-8",
    )
    monitors = parse_solver_log(log, {
        "runnerMoments": {
            "pattern": r"runner moment = \(([^)]+)\)",
            "vector": True,
            "kind": "moment",
            "units": "N*m",
        },
        "waterFluxIn": {
            "pattern": r"water flux = ([-+0-9.eE]+)",
            "kind": "water_phase_flux",
            "units": "m3/s",
        },
    })
    assert monitors["runnerMoments"].values.tolist() == [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]
    assert monitors["waterFluxIn"].values[:, 0].tolist() == [-2.0, -2.0]
    assert monitors["runnerMoments"].source == "solver_log_fallback"
    assert len(monitors["runnerMoments"].segments[0].sha256) == 64


def test_full_runner_units_formulas_and_low_error_passed() -> None:
    result = compute_cfd_hydraulic_performance(_contract(), _monitors())
    metrics = result["metrics"]
    assert result["status"] == "passed"
    assert all(result["gates"].values())
    assert metrics["torque_axis_nm"] == pytest.approx(1_765_800.0 / (20.0 * math.pi), rel=1.0e-10)
    assert metrics["Pshaft_w"] == pytest.approx(1_765_800.0)
    assert metrics["Q_m3_s"] == pytest.approx(2.0)
    assert metrics["net_head_m"] == pytest.approx(100.0)
    assert metrics["Phyd_w"] == pytest.approx(1_962_000.0)
    assert metrics["eta"] == pytest.approx(0.9)
    assert metrics["n11_rpm"] == pytest.approx(60.0)
    assert metrics["Q11_lps"] == pytest.approx(200.0)
    assert metrics["mass_imbalance_fraction"] == pytest.approx(0.0)
    assert metrics["energy_imbalance_fraction"] == pytest.approx(0.0, abs=1.0e-12)
    assert set(result["block_confidence_intervals"]) >= {
        "torque_axis_nm", "shaft_power_w", "net_head_m", "hydraulic_power_w", "efficiency"
    }


def test_full_runner_high_mass_and_energy_error_failed() -> None:
    result = compute_cfd_hydraulic_performance(
        _contract(),
        _monitors(energy_out_kw=1000.0, q_out_lps=1500.0),
    )
    assert result["status"] == "failed"
    assert result["gates"]["mass_imbalance_below_limit"] is False
    assert result["gates"]["energy_imbalance_below_limit"] is False


def test_stationary_single_bucket_is_diagnostic_blocked_without_shaft_power() -> None:
    contract = _contract(
        case_type="stationary_single_bucket",
        rotating=False,
        vof=False,
    )
    result = compute_cfd_hydraulic_performance(contract, _monitors())
    assert result["status"] == "blocked"
    assert result["diagnostic_only"] is True
    assert result["metrics"]["moment_axis_nm"] > 0.0
    assert result["metrics"]["Pshaft_w"] is None
    assert result["metrics"]["shaft_power_w"] is None
    assert "stationary_single_bucket" in " ".join(result["diagnostic_reasons"])


@pytest.mark.parametrize(
    ("contract", "missing_name"),
    [
        (_contract(), "runnerMoments"),
        (_contract(energy=False), "energy balance monitor"),
        (_contract(net_head=False), "totalHeadIn"),
    ],
)
def test_missing_required_monitor_or_measurement_contract_is_blocked(
    contract: HydraulicCaseContract,
    missing_name: str,
) -> None:
    monitors = _monitors()
    if missing_name == "runnerMoments":
        monitors.pop("runnerMoments")
    result = compute_cfd_hydraulic_performance(contract, monitors)
    assert result["status"] == "blocked"
    if missing_name == "totalHeadIn":
        assert result["diagnostic_only"] is True
        assert "net head" in " ".join(result["diagnostic_reasons"])
    else:
        assert missing_name in result.get("missing_monitors", [])


def test_monitor_bundle_conflict_propagates_blocked() -> None:
    bundle = MonitorBundle(status="blocked", monitors={}, conflicts=("restart conflict",))
    result = compute_cfd_hydraulic_performance(_contract(), bundle)
    assert result["status"] == "blocked"
    assert result["monitor_conflicts"] == ["restart conflict"]


def test_incomplete_control_surface_contract_is_structurally_blocked() -> None:
    contract = replace(_contract(), measurement_surfaces=_contract().measurement_surfaces[:1])
    result = compute_cfd_hydraulic_performance(contract, _monitors())
    assert result["status"] == "blocked"
    assert result["diagnostic_only"] is True
    assert "outlet measurement surface" in result["reason"]


def test_dec_alignment_matches_nozzle_opening_n11_q11_and_keeps_roles_separate() -> None:
    cfd = compute_cfd_hydraulic_performance(_contract(), _monitors())
    result = align_cfd_to_dec(cfd, [_dec_row()])
    assert result["status"] == "passed"
    assert result["match_basis"] == ["nozzle_count", "opening_mm", "n11_rpm", "Q11_lps"]
    assert result["cfd"]["efficiency_role"] == "direct_cfd_computed"
    assert result["dec_calibration"]["efficiency_role"] == "parsed_calibration_only"
    assert result["dec_calibration"]["is_direct_cfd_efficiency"] is False
    assert result["cfd"]["efficiency_pct"] == pytest.approx(90.0)


def test_dec_efficiency_error_fails_without_substituting_calibration() -> None:
    cfd = compute_cfd_hydraulic_performance(_contract(), _monitors())
    result = align_cfd_to_dec(cfd, [_dec_row(eta=70.0)])
    assert result["status"] == "failed"
    assert result["efficiency_gate_passed"] is False
    assert result["cfd"]["efficiency_pct"] == pytest.approx(90.0)
    assert result["dec_calibration"]["efficiency_pct"] == pytest.approx(70.0)


def test_previously_rejected_dec_q11_cannot_be_rehabilitated() -> None:
    rejected_input = replace(
        _dec_row(),
        q11_quality="rejected",
        q11_reason="upstream instrument quality gate",
    )
    checked, rejected = assess_q11_quality([rejected_input])
    assert checked[0].q11_quality == "rejected"
    assert rejected[0].q11_reason == "upstream instrument quality gate"


def test_dec_anomalous_q11_row_is_rejected_and_blocks_alignment() -> None:
    rows = [
        _dec_row(q11=200.0, row_number=7),
        _dec_row(q11=201.0, row_number=8),
        _dec_row(q11=2000.0, row_number=9),
    ]
    checked, rejected = assess_q11_quality(rows)
    assert [row.row_number for row in rejected] == [9]
    assert next(row for row in checked if row.row_number == 9).q11_quality == "rejected"

    cfd = compute_cfd_hydraulic_performance(_contract(), _monitors())
    blocked = align_cfd_to_dec(cfd, [_dec_row(q11=-10.0)])
    assert blocked["status"] == "blocked"
    assert "anomalous Q11" in blocked["reason"]
    assert blocked["dec_efficiency_is_direct_cfd"] is False
