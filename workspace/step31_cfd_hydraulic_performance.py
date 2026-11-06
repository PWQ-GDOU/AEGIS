"""Direct CFD hydraulic performance metrics and fail-closed acceptance gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from step31_hydraulic_case_contract import (
    HydraulicCaseContract,
    MeasurementSurface,
    validate_case_contract,
)
from step31_openfoam_monitor_io import (
    MonitorBundle,
    MonitorParseError,
    MonitorSeries,
    block_confidence_interval,
    trapezoidal_time_average,
)


VERSION = "step31-cfd-hydraulic-performance-v1"
G = 9.81
DEFAULT_RHO_KG_M3 = 998.2
DEFAULT_RELATIVE_ERROR_GATE = 0.05
DEFAULT_MASS_IMBALANCE_GATE = 0.01
DEFAULT_ENERGY_IMBALANCE_GATE = 0.05


class PerformanceContractError(ValueError):
    """Raised when a performance calculation lacks a physical contract."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise PerformanceContractError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PerformanceContractError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise PerformanceContractError(f"{name} must be finite")
    return result


def _unit_factor(unit: str, dimension: str) -> float:
    value = unit.strip().lower().replace(" ", "")
    if dimension == "flow":
        factors = {"m3/s": 1.0, "m^3/s": 1.0, "l/s": 1.0e-3, "lps": 1.0e-3, "kg/s": 1.0}
    elif dimension == "torque":
        factors = {"n*m": 1.0, "nm": 1.0, "kn*m": 1.0e3, "knm": 1.0e3, "nmm": 1.0e-3}
    elif dimension == "power":
        factors = {"w": 1.0, "kw": 1.0e3, "mw": 1.0e6}
    elif dimension == "head":
        factors = {"m": 1.0, "mhead": 1.0}
    elif dimension == "energy_flux":
        factors = {"w": 1.0, "kw": 1.0e3, "mw": 1.0e6}
    else:
        factors = {}
    if value not in factors:
        raise PerformanceContractError(f"unsupported {dimension} unit {unit!r}")
    return factors[value]


def axis_project_vector(vector: Sequence[float], axis: Sequence[float]) -> float:
    """Project a vector onto a normalized axis, preserving the signed direction."""
    vector_array = np.asarray(vector, dtype=np.float64)
    axis_array = np.asarray(axis, dtype=np.float64)
    if vector_array.shape != (3,) or axis_array.shape != (3,):
        raise PerformanceContractError("axis projection requires two three-component vectors")
    if not np.isfinite(vector_array).all() or not np.isfinite(axis_array).all():
        raise PerformanceContractError("axis projection requires finite vectors")
    norm = float(np.linalg.norm(axis_array))
    if norm <= 1.0e-15:
        raise PerformanceContractError("axis projection requires a non-zero axis")
    return float(np.dot(vector_array, axis_array / norm))


def calculate_axis_torque(moment: Sequence[float], axis: Sequence[float], *, sign: int = 1) -> float:
    if isinstance(sign, bool) or sign not in (-1, 1):
        raise PerformanceContractError("torque sign must be -1 or 1")
    return float(sign) * axis_project_vector(moment, axis)


def _mean(series: MonitorSeries, *, start_time: float | None, end_time: float | None) -> np.ndarray:
    value = trapezoidal_time_average(series, start_time=start_time, end_time=end_time)
    result = np.asarray(value, dtype=np.float64)
    return result.reshape(-1)


def _single_mean(series: MonitorSeries, *, start_time: float | None, end_time: float | None) -> float:
    result = _mean(series, start_time=start_time, end_time=end_time)
    if result.size != 1:
        raise PerformanceContractError(f"monitor {series.name!r} must be scalar")
    return float(result[0])


def _convert_flow(value: float, units: str, rho: float) -> float:
    unit = units.strip().lower().replace(" ", "")
    if unit in {"kg/s"}:
        return value / rho
    return value * _unit_factor(units, "flow")


def _convert_torque(value: float, units: str) -> float:
    return value * _unit_factor(units, "torque")


def _convert_power(value: float, units: str) -> float:
    return value * _unit_factor(units, "power")


def _head_value(value: float, units: str, *, rho: float, g: float, pressure_kind: str) -> float:
    unit = units.strip().lower().replace(" ", "")
    if unit in {"m", "mhead"}:
        return value
    if unit in {"m2/s2", "m^2/s^2", "m2s-2", "m2/s2kinematic"}:
        return value / g
    if unit in {"pa", "kpa", "mpa"}:
        multiplier = {"pa": 1.0, "kpa": 1.0e3, "mpa": 1.0e6}[unit]
        return value * multiplier / (rho * g)
    if pressure_kind == "kinematic":
        raise PerformanceContractError(f"kinematic pressure head monitor has unsupported units {units!r}")
    raise PerformanceContractError(f"unsupported head units {units!r}")


def _convert_head_series(
    series: MonitorSeries,
    *,
    rho: float,
    g: float,
    pressure_kind: str,
    start_time: float | None,
    end_time: float | None,
) -> float:
    value = _single_mean(series, start_time=start_time, end_time=end_time)
    return _head_value(value, series.units, rho=rho, g=g, pressure_kind=pressure_kind)


def _resolve_monitor(monitors: Mapping[str, MonitorSeries], name: str) -> MonitorSeries:
    series = monitors.get(name)
    if not isinstance(series, MonitorSeries):
        raise PerformanceContractError(f"required monitor {name!r} is missing")
    return series


def _surface_map(contract: HydraulicCaseContract) -> dict[str, MeasurementSurface]:
    result: dict[str, MeasurementSurface] = {}
    for surface in contract.measurement_surfaces:
        if surface.role in result:
            raise PerformanceContractError(f"multiple {surface.role} measurement surfaces are unsupported")
        result[surface.role] = surface
    for role in ("inlet", "outlet"):
        if role not in result:
            raise PerformanceContractError(f"missing {role} measurement surface")
    return result


def _ci(
    series: MonitorSeries,
    *,
    values: np.ndarray | None = None,
    start_time: float | None,
    end_time: float | None,
    n_blocks: int,
) -> dict[str, Any]:
    try:
        if values is None:
            return block_confidence_interval(
                series,
                start_time=start_time,
                end_time=end_time,
                n_blocks=n_blocks,
            )
        return block_confidence_interval(
            series.times,
            values,
            start_time=start_time,
            end_time=end_time,
            n_blocks=n_blocks,
        )
    except MonitorParseError as exc:
        raise PerformanceContractError(str(exc)) from exc


def _percent_relative(numerator: float, denominator: float) -> float:
    return abs(numerator) / max(abs(denominator), 1.0e-30)


def _common_history_times(
    series: Sequence[MonitorSeries],
    *,
    start_time: float | None,
    end_time: float | None,
) -> np.ndarray:
    start = max(item.times[0] for item in series) if start_time is None else float(start_time)
    end = min(item.times[-1] for item in series) if end_time is None else float(end_time)
    if end <= start or any(start < item.times[0] or end > item.times[-1] for item in series):
        raise PerformanceContractError("required monitors do not share the requested averaging window")
    interior = np.concatenate([
        item.times[(item.times > start) & (item.times < end)] for item in series
    ])
    return np.unique(np.concatenate(([start], interior, [end])))


def compute_cfd_hydraulic_performance(
    contract: HydraulicCaseContract | Mapping[str, Any],
    monitors: Mapping[str, MonitorSeries] | MonitorBundle,
    *,
    rho_kg_m3: float | None = None,
    g: float = G,
    start_time: float | None = None,
    end_time: float | None = None,
    n_blocks: int = 4,
    mass_imbalance_gate: float = DEFAULT_MASS_IMBALANCE_GATE,
    energy_imbalance_gate: float = DEFAULT_ENERGY_IMBALANCE_GATE,
) -> dict[str, Any]:
    """Calculate direct CFD performance, retaining diagnostic and gate states."""
    case = validate_case_contract(contract)
    if rho_kg_m3 is None:
        rho_kg_m3 = case.water_phase.density_kg_m3
    rho = _finite(rho_kg_m3, "rho_kg_m3")
    gravity = _finite(g, "g")
    if rho <= 0.0 or gravity <= 0.0:
        raise PerformanceContractError("rho_kg_m3 and g must be positive")
    if isinstance(monitors, MonitorBundle):
        if monitors.blocked:
            return {
                "version": VERSION,
                "status": "blocked",
                "reason": "monitor bundle is blocked",
                "monitor_conflicts": list(monitors.conflicts),
                "missing_monitors": list(monitors.missing),
                "case_contract_sha256": case.contract_sha256,
            }
        monitor_map = dict(monitors.monitors)
    else:
        monitor_map = dict(monitors)
    diagnostic_reasons: list[str] = []
    if case.case_type == "stationary_single_bucket":
        diagnostic_reasons.append("stationary_single_bucket has no rotating runner shaft")
    if not case.rotating_domain or case.omega_rad_s <= 0.0:
        diagnostic_reasons.append("positive rotating-domain angular velocity is absent")
    if not case.vof_enabled:
        diagnostic_reasons.append("VOF water-phase transport is absent")
    if not case.has_net_head_measurement:
        diagnostic_reasons.append("net head requires both inlet and outlet total-head measurement surfaces")

    try:
        surfaces = _surface_map(case)
    except PerformanceContractError as exc:
        return {
            "version": VERSION,
            "status": "blocked",
            "reason": str(exc),
            "diagnostic_only": True,
            "diagnostic_reasons": diagnostic_reasons or ["complete inlet/outlet control surfaces are absent"],
            "case_contract_sha256": case.contract_sha256,
        }
    inlet = surfaces["inlet"]
    outlet = surfaces["outlet"]
    missing: list[str] = []
    try:
        moment = _resolve_monitor(monitor_map, case.moment_monitor)
    except PerformanceContractError:
        missing.append(case.moment_monitor)
        moment = None
    flux_series: dict[str, MonitorSeries] = {}
    head_series: dict[str, MonitorSeries] = {}
    for surface in (inlet, outlet):
        for monitor_name, target in (
            (surface.water_flux_monitor, flux_series),
            (surface.total_head_monitor, head_series),
        ):
            if monitor_name is None:
                continue
            if monitor_name not in monitor_map:
                missing.append(monitor_name)
            else:
                target[surface.role] = monitor_map[monitor_name]
    energy_names = (inlet.energy_flux_monitor, outlet.energy_flux_monitor)
    if any(energy_names) and not all(energy_names):
        missing.append("paired inlet/outlet energy flux monitor declaration")
    if all(energy_names):
        missing.extend(name for name in energy_names if name not in monitor_map)
    elif case.energy_balance_monitor:
        if case.energy_balance_monitor not in monitor_map:
            missing.append(case.energy_balance_monitor)
    else:
        missing.append("energy balance monitor")
    if missing:
        return {
            "version": VERSION,
            "status": "blocked",
            "reason": "required CFD monitors are missing",
            "missing_monitors": sorted(set(missing)),
            "diagnostic_only": bool(diagnostic_reasons),
            "diagnostic_reasons": diagnostic_reasons,
            "case_contract_sha256": case.contract_sha256,
        }
    assert moment is not None
    try:
        moment_mean = _mean(moment, start_time=start_time, end_time=end_time)
        if moment_mean.size != 3:
            raise PerformanceContractError("moment monitor must contain x/y/z components")
        moment_nm = moment_mean * _unit_factor(moment.units, "torque")
        torque_axis_nm = calculate_axis_torque(moment_nm, case.axis, sign=case.torque_sign)
        q_in = inlet.flow_sign * _convert_flow(
            _single_mean(flux_series["inlet"], start_time=start_time, end_time=end_time),
            flux_series["inlet"].units,
            rho,
        )
        q_out = outlet.flow_sign * _convert_flow(
            _single_mean(flux_series["outlet"], start_time=start_time, end_time=end_time),
            flux_series["outlet"].units,
            rho,
        )
        Q_m3_s = 0.5 * (q_in + q_out)
        mass_imbalance = _percent_relative(q_in - q_out, max(abs(q_in), abs(q_out)))
        if not head_series:
            raise PerformanceContractError("net head monitors are missing")
        h_in = _convert_head_series(
            head_series["inlet"], rho=rho, g=gravity,
            pressure_kind=case.pressure_definition.kind,
            start_time=start_time, end_time=end_time,
        )
        h_out = _convert_head_series(
            head_series["outlet"], rho=rho, g=gravity,
            pressure_kind=case.pressure_definition.kind,
            start_time=start_time, end_time=end_time,
        )
        net_head_m = h_in - h_out
        if net_head_m <= 0.0:
            raise PerformanceContractError("net hydraulic head must be positive")
        hydraulic_power_w = rho * gravity * Q_m3_s * net_head_m
        shaft_power_w = torque_axis_nm * case.omega_rad_s
        efficiency = shaft_power_w / hydraulic_power_w if hydraulic_power_w > 0.0 else float("nan")
        n11 = case.n_rpm * case.D1_m / math.sqrt(net_head_m)
        Q11 = Q_m3_s * 1000.0 / (case.D1_m**2 * math.sqrt(net_head_m))
        energy_in_w = energy_out_w = None
        energy_imbalance = None
        if inlet.energy_flux_monitor and outlet.energy_flux_monitor:
            energy_in_series = _resolve_monitor(monitor_map, inlet.energy_flux_monitor)
            energy_out_series = _resolve_monitor(monitor_map, outlet.energy_flux_monitor)
            energy_in_w = inlet.flow_sign * _convert_power(
                _single_mean(energy_in_series, start_time=start_time, end_time=end_time),
                energy_in_series.units,
            )
            energy_out_w = outlet.flow_sign * _convert_power(
                _single_mean(energy_out_series, start_time=start_time, end_time=end_time),
                energy_out_series.units,
            )
            energy_imbalance = _percent_relative(
                energy_in_w - energy_out_w - shaft_power_w,
                max(abs(energy_in_w), 1.0e-30),
            )
        elif case.energy_balance_monitor:
            balance_series = _resolve_monitor(monitor_map, case.energy_balance_monitor)
            balance_w = _convert_power(
                _single_mean(balance_series, start_time=start_time, end_time=end_time),
                balance_series.units,
            )
            energy_imbalance = _percent_relative(balance_w, max(abs(hydraulic_power_w), 1.0e-30))
        else:
            energy_imbalance = None
        axis = np.asarray(case.axis, dtype=np.float64)
        torque_history_nm = (
            case.torque_sign
            * (moment.values @ axis)
            * _unit_factor(moment.units, "torque")
        )
        inlet_flow_history = np.asarray([
            inlet.flow_sign * _convert_flow(value, flux_series["inlet"].units, rho)
            for value in flux_series["inlet"].values[:, 0]
        ])
        outlet_flow_history = np.asarray([
            outlet.flow_sign * _convert_flow(value, flux_series["outlet"].units, rho)
            for value in flux_series["outlet"].values[:, 0]
        ])
        history_series = (
            moment,
            flux_series["inlet"],
            flux_series["outlet"],
            head_series["inlet"],
            head_series["outlet"],
        )
        common_times = _common_history_times(
            history_series, start_time=start_time, end_time=end_time
        )
        torque_common = np.interp(common_times, moment.times, torque_history_nm)
        q_in_common = np.interp(common_times, flux_series["inlet"].times, inlet_flow_history)
        q_out_common = np.interp(common_times, flux_series["outlet"].times, outlet_flow_history)
        h_in_common = np.asarray([
            _head_value(
                value, head_series["inlet"].units, rho=rho, g=gravity,
                pressure_kind=case.pressure_definition.kind,
            )
            for value in np.interp(
                common_times, head_series["inlet"].times, head_series["inlet"].values[:, 0]
            )
        ])
        h_out_common = np.asarray([
            _head_value(
                value, head_series["outlet"].units, rho=rho, g=gravity,
                pressure_kind=case.pressure_definition.kind,
            )
            for value in np.interp(
                common_times, head_series["outlet"].times, head_series["outlet"].values[:, 0]
            )
        ])
        q_common = 0.5 * (q_in_common + q_out_common)
        net_head_common = h_in_common - h_out_common
        shaft_power_common = torque_common * case.omega_rad_s
        hydraulic_power_common = rho * gravity * q_common * net_head_common
        efficiency_common = shaft_power_common / hydraulic_power_common
        ci = {
            "torque_axis_nm": _ci(
                moment, values=torque_history_nm,
                start_time=start_time, end_time=end_time, n_blocks=n_blocks,
            ),
            "q_in_m3_s": _ci(
                flux_series["inlet"], values=inlet_flow_history,
                start_time=start_time, end_time=end_time, n_blocks=n_blocks,
            ),
            "q_out_m3_s": _ci(
                flux_series["outlet"], values=outlet_flow_history,
                start_time=start_time, end_time=end_time, n_blocks=n_blocks,
            ),
            "shaft_power_w": block_confidence_interval(
                common_times, shaft_power_common, n_blocks=n_blocks
            ),
            "net_head_m": block_confidence_interval(
                common_times, net_head_common, n_blocks=n_blocks
            ),
            "hydraulic_power_w": block_confidence_interval(
                common_times, hydraulic_power_common, n_blocks=n_blocks
            ),
            "efficiency": block_confidence_interval(
                common_times, efficiency_common, n_blocks=n_blocks
            ),
        }
    except (MonitorParseError, PerformanceContractError, KeyError) as exc:
        return {
            "version": VERSION,
            "status": "blocked",
            "reason": str(exc),
            "diagnostic_only": bool(diagnostic_reasons),
            "diagnostic_reasons": diagnostic_reasons,
            "case_contract_sha256": case.contract_sha256,
        }
    gates = {
        "full_runner_case": case.case_type == "full_runner_rotating_vof",
        "rotating_domain": case.rotating_domain and case.omega_rad_s > 0.0,
        "vof_water_phase": case.vof_enabled and case.water_phase.flux_is_water_phase_only,
        "net_head_measurement": case.has_net_head_measurement,
        "positive_torque": torque_axis_nm > 0.0,
        "positive_flow": Q_m3_s > 0.0,
        "physical_efficiency": 0.0 < efficiency <= 1.0,
        "mass_imbalance_below_limit": mass_imbalance <= mass_imbalance_gate,
        "energy_imbalance_available": energy_imbalance is not None,
        "energy_imbalance_below_limit": energy_imbalance is not None and energy_imbalance <= energy_imbalance_gate,
        "finite_metrics": bool(np.isfinite([torque_axis_nm, shaft_power_w, Q_m3_s, net_head_m, efficiency, n11, Q11]).all()),
    }
    gates = {name: bool(value) for name, value in gates.items()}
    accepted = bool(all(gates.values()) and not diagnostic_reasons)
    result = {
        "version": VERSION,
        "status": "passed" if accepted else ("blocked" if diagnostic_reasons else "failed"),
        "diagnostic_only": bool(diagnostic_reasons),
        "diagnostic_reasons": diagnostic_reasons,
        "case_contract_sha256": case.contract_sha256,
        "case_type": case.case_type,
        "operating_point": {
            "nozzle_count": case.nozzle_count,
            "opening_mm": case.opening_mm,
            "D1_m": case.D1_m,
            "declared_head_m": case.H_m,
            "n_rpm": case.n_rpm,
        },
        "metrics": {
            "axis": list(case.axis),
            "moment_axis_nm": torque_axis_nm,
            "torque_axis_nm": torque_axis_nm,
            "omega_rad_s": case.omega_rad_s,
            "Pshaft_w": shaft_power_w,
            "shaft_power_w": shaft_power_w,
            "Q_in_m3_s": q_in,
            "Q_out_m3_s": q_out,
            "Q_m3_s": Q_m3_s,
            "H_in_m": h_in,
            "H_out_m": h_out,
            "net_head_m": net_head_m,
            "net_total_specific_energy_j_kg": gravity * net_head_m,
            "total_energy_w": hydraulic_power_w,
            "Phyd_w": hydraulic_power_w,
            "eta": efficiency,
            "eta_pct": efficiency * 100.0,
            "n11_rpm": n11,
            "Q11_lps": Q11,
            "mass_imbalance_fraction": mass_imbalance,
            "energy_imbalance_fraction": energy_imbalance,
            "energy_in_w": energy_in_w,
            "energy_out_w": energy_out_w,
        },
        "gates": gates,
        "block_confidence_intervals": ci,
        "lineage": {
            name: series.to_lineage_dict() for name, series in sorted(monitor_map.items())
        },
        "thresholds": {
            "mass_imbalance_fraction": mass_imbalance_gate,
            "energy_imbalance_fraction": energy_imbalance_gate,
        },
        "acceptance_note": (
            "Direct CFD shaft-power result; DEC calibration remains a separate comparison record."
            if accepted else
            "Result is not eligible for a direct CFD performance claim until all gates pass."
        ),
    }
    if diagnostic_reasons:
        result["shaft_power_w"] = None
        result["Pshaft_w"] = None
        result["metrics"]["shaft_power_w"] = None
        result["metrics"]["Pshaft_w"] = None
        result["metrics"]["eta"] = None
        result["metrics"]["eta_pct"] = None
        result["block_confidence_intervals"]["shaft_power_w"] = None
        result["block_confidence_intervals"]["efficiency"] = None
    return result


def evaluate_cfd_performance(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility alias for :func:`compute_cfd_hydraulic_performance`."""
    return compute_cfd_hydraulic_performance(*args, **kwargs)


def calculate_performance(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return compute_cfd_hydraulic_performance(*args, **kwargs)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calculate direct CFD hydraulic performance")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--monitors", type=Path, required=True, help="JSON monitor specification")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    contract = validate_case_contract(_load_json(args.contract))
    specs = _load_json(args.monitors)
    bundle = __import__("step31_openfoam_monitor_io", fromlist=["load_monitor_bundle"]).load_monitor_bundle(specs)
    result = compute_cfd_hydraulic_performance(contract, bundle)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if result.get("status") == "passed" else 3


__all__ = [
    "DEFAULT_ENERGY_IMBALANCE_GATE",
    "DEFAULT_MASS_IMBALANCE_GATE",
    "DEFAULT_RELATIVE_ERROR_GATE",
    "G",
    "PerformanceContractError",
    "VERSION",
    "axis_project_vector",
    "calculate_axis_torque",
    "calculate_performance",
    "compute_cfd_hydraulic_performance",
    "evaluate_cfd_performance",
]


if __name__ == "__main__":
    raise SystemExit(main())
