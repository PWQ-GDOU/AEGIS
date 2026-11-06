"""Validate one production URANS time-mean window without steady-state semantics."""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal
from pathlib import Path

import numpy as np

VERSION = "urans-time-mean-validation-v2"
FLOW_THROUGH_TIME_S = Decimal("0.0036056848")


def _internal_field(path: Path, kind: str) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        rf"internalField\s+nonuniform\s+List<{kind}>\s+(\d+)\s*\((.*?)\)\s*;",
        text,
        re.DOTALL,
    )
    if not match:
        raise ValueError(f"cannot parse {kind} internal field: {path}")
    body = match.group(2)
    if kind == "vector":
        values = np.fromstring(body.replace("(", " ").replace(")", " "), sep=" ").reshape(-1, 3)
    else:
        values = np.fromstring(body, sep=" ")
    if len(values) != int(match.group(1)):
        raise ValueError(f"field count mismatch: {path}")
    return values


def _bucket_vector_field(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="replace")
    patch = re.search(r"\bbucket\s*\{(?P<body>.*?)\n\s*\}", text, re.DOTALL)
    if not patch:
        raise ValueError(f"bucket patch is missing: {path}")
    match = re.search(
        r"value\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;",
        patch.group("body"),
        re.DOTALL,
    )
    if not match:
        raise ValueError(f"bucket vector values are missing: {path}")
    values = np.fromstring(match.group(2).replace("(", " ").replace(")", " "), sep=" ").reshape(-1, 3)
    if len(values) != int(match.group(1)):
        raise ValueError(f"bucket field count mismatch: {path}")
    return values


def _numeric_rows(root: Path, relative_pattern: str) -> np.ndarray:
    rows_by_time = {}
    number = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
    for path in sorted(root.glob(relative_pattern)):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line or line.startswith("#"):
                continue
            values = np.asarray([float(value) for value in number.findall(line)], dtype=np.float64)
            if len(values):
                # OpenFOAM restart writes may create both `force.dat` and
                # `force_<restart-time>.dat`; later files supersede overlap.
                rows_by_time[values[0]] = values
    if not rows_by_time:
        raise ValueError(f"no monitor data found for {relative_pattern}")
    rows = list(rows_by_time.values())
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError(f"monitor data has inconsistent row widths: {relative_pattern}")
    result = np.asarray(rows)
    return result[np.argsort(result[:, 0])]


def _averaging_rows(
    rows: np.ndarray,
    averaging_start: Decimal,
    field_time: Decimal,
    label: str,
) -> np.ndarray:
    start = float(averaging_start)
    end = float(field_time)
    selected = rows[(rows[:, 0] >= start) & (rows[:, 0] <= end)]
    if len(selected) < 20:
        raise ValueError(f"{label} has too few samples inside the averaging window")
    if np.any(np.diff(selected[:, 0]) < 0.0):
        raise ValueError(f"{label} monitor times are not ordered")
    duration = end - start
    if duration <= 0.0 or selected[-1, 0] - selected[0, 0] < 0.9 * duration:
        raise ValueError(f"{label} does not cover the declared averaging window")
    return selected


def _averaging_values(
    times: np.ndarray,
    values: np.ndarray,
    averaging_start: Decimal,
    field_time: Decimal,
    label: str,
) -> np.ndarray:
    rows = np.column_stack((times, values.reshape(len(values), -1)))
    selected = _averaging_rows(rows, averaging_start, field_time, label)
    return selected[:, 1:].reshape((-1, *values.shape[1:]))


def _probe_vectors(root: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for path in root.glob("postProcessing/flowProbes/*/U"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line or line.startswith("#"):
                continue
            time_match = re.match(r"\s*([^\s]+)", line)
            vectors = re.findall(r"\(([^)]+)\)", line)
            if not time_match or not vectors:
                continue
            values = np.asarray([[float(value) for value in item.split()] for item in vectors])
            rows.append((float(time_match.group(1)), values))
    if not rows:
        raise ValueError("no vector probe data found")
    rows.sort(key=lambda item: item[0])
    return (
        np.asarray([time_value for time_value, _ in rows], dtype=np.float64),
        np.asarray([values for _, values in rows]),
    )


def _window_slices(length: int) -> tuple[slice, slice]:
    width = max(10, length // 5)
    if length < 2 * width:
        raise ValueError("monitor history is too short for two late-window means")
    return slice(length - 2 * width, length - width), slice(length - width, length)


def _relative_mean_change(values: np.ndarray) -> float:
    previous_slice, current_slice = _window_slices(len(values))
    previous = values[previous_slice].mean(axis=0)
    current = values[current_slice].mean(axis=0)
    return float(np.linalg.norm(current - previous) / max(np.linalg.norm(current), 1e-30))


def _flux_relative_mean_change(
    flux: dict[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, float]]:
    recent_slice = _window_slices(len(next(iter(flux.values()))))[1]
    recent_flux = {name: float(values[recent_slice].mean()) for name, values in flux.items()}
    principal_flux_scale = max(
        sum(abs(recent_flux[name]) for name in ("inletFlux", "jetFlux", "outletFlux")),
        1e-30,
    )
    changes = {}
    for name, values in flux.items():
        previous_slice, current_slice = _window_slices(len(values))
        previous = float(values[previous_slice].mean())
        current = float(values[current_slice].mean())
        reference = (
            abs(current)
            if abs(current) >= 0.1 * principal_flux_scale
            else principal_flux_scale
        )
        changes[name] = abs(current - previous) / reference
    return changes, recent_flux


def _field_relative_l2(current: np.ndarray, previous: np.ndarray) -> float:
    if current.shape != previous.shape:
        raise ValueError(f"mean field shapes differ: {current.shape} vs {previous.shape}")
    return float(np.linalg.norm(current - previous) / max(np.linalg.norm(current), 1e-30))


def run(
    case: Path,
    field_time: str,
    previous_time: str,
    averaging_start: str,
    sample_count: int,
    output: Path,
) -> dict:
    case = case.resolve()
    field_time_value = Decimal(field_time)
    averaging_start_value = Decimal(averaging_start)
    log_path = case / "log.urans.average"
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    times = [Decimal(value) for value in re.findall(r"^Time = ([^\s]+)", log_text, re.MULTILINE)]
    if not times or times[-1] != field_time_value:
        raise ValueError("URANS log end time differs from the requested field time")
    if sample_count != len(times):
        raise ValueError("declared averaging sample count differs from the URANS log")
    max_co_values = [float(value) for value in re.findall(r"Courant Number mean: [^\s]+ max: ([^\s]+)", log_text)]
    if not max_co_values:
        raise ValueError("URANS log contains no Courant-number records")

    objects = {
        "UMean": "vector",
        "pMean": "scalar",
        "kMean": "scalar",
        "omegaMean": "scalar",
    }
    mean_l2 = {}
    current_fields = {}
    for name, kind in objects.items():
        current = _internal_field(case / field_time / name, kind)
        previous = _internal_field(case / previous_time / name, kind)
        current_fields[name] = current
        mean_l2[name] = _field_relative_l2(current, previous)
    current_wss = _bucket_vector_field(case / field_time / "wallShearStressMean")
    previous_wss = _bucket_vector_field(case / previous_time / "wallShearStressMean")
    mean_l2["wallShearStressMean"] = _field_relative_l2(current_wss, previous_wss)

    force_rows = _averaging_rows(
        _numeric_rows(case, "postProcessing/bucketForces/*/force*.dat"),
        averaging_start_value,
        field_time_value,
        "bucket forces",
    )
    if force_rows.shape[1] < 7:
        raise ValueError("bucket force monitor must contain pressure and viscous vectors")
    force_components = {
        "pressure": force_rows[:, 1:4],
        "viscous": force_rows[:, 4:7],
    }
    force_components["total"] = force_components["pressure"] + force_components["viscous"]
    flux = {}
    for name in ("inletFlux", "jetFlux", "outletFlux", "sideFlux"):
        rows = _averaging_rows(
            _numeric_rows(case, f"postProcessing/{name}/*/surfaceFieldValue*.dat"),
            averaging_start_value,
            field_time_value,
            name,
        )
        flux[name] = rows[:, -1]
    probe_u_times, probe_u_values = _probe_vectors(case)
    probe_u = _averaging_values(
        probe_u_times,
        probe_u_values,
        averaging_start_value,
        field_time_value,
        "U probes",
    )
    probe_p_rows = _averaging_rows(
        _numeric_rows(case, "postProcessing/flowProbes/*/p"),
        averaging_start_value,
        field_time_value,
        "p probes",
    )
    probe_p = probe_p_rows[:, 1:]
    force_change = {
        name: _relative_mean_change(values)
        for name, values in force_components.items()
    }
    flux_change, recent_flux = _flux_relative_mean_change(flux)
    probe_u_change = _relative_mean_change(probe_u)
    probe_p_change = _relative_mean_change(probe_p)

    net_flux = sum(recent_flux.values())
    flux_scale = sum(abs(value) for value in recent_flux.values())
    flux_imbalance = float(abs(net_flux) / max(flux_scale, 1e-30))
    speed = np.linalg.norm(current_fields["UMean"], axis=1)
    all_finite = all(np.isfinite(value).all() for value in current_fields.values()) and np.isfinite(current_wss).all()
    duration = field_time_value - averaging_start_value
    averaging_window = {
        "start_time": averaging_start,
        "end_time": field_time,
        "sample_count": sample_count,
    }
    gates = {
        "max_co_below_0p5": max(max_co_values) <= 0.5,
        "no_turbulence_bounding": (
            "bounding k," not in log_text and "bounding omega," not in log_text
        ),
        "no_fatal_or_nonfinite_log": not re.search(
            r"FOAM FATAL|\b(?:nan|[-+]?inf)\b", log_text, re.IGNORECASE
        ),
        "twenty_flow_through_window": duration >= Decimal(20) * FLOW_THROUGH_TIME_S,
        "mean_field_relative_l2_below_1e3": max(mean_l2.values()) < 1e-3,
        "force_late_mean_change_below_1pct": max(force_change.values()) < 0.01,
        "flux_late_mean_change_below_1pct": max(flux_change.values()) < 0.01,
        "flux_imbalance_below_1e4": flux_imbalance < 1e-4,
        "probe_late_mean_change_below_1pct": max(probe_u_change, probe_p_change) < 0.01,
        "mean_speed_below_500_ms": float(speed.max()) < 500.0,
        "all_mean_fields_finite": all_finite,
    }
    gates = {name: bool(value) for name, value in gates.items()}
    result = {
        "version": VERSION,
        "case": str(case),
        "field_time": field_time,
        "previous_time": previous_time,
        "averaging_window": averaging_window,
        "flow_through_time_s": float(FLOW_THROUGH_TIME_S),
        "metrics": {
            "max_courant_number": max(max_co_values),
            "bounding_k_count": log_text.count("bounding k,"),
            "bounding_omega_count": log_text.count("bounding omega,"),
            "mean_field_relative_l2": mean_l2,
            "force_late_mean_relative_change": force_change,
            "flux_late_mean_relative_change": flux_change,
            "flux_recent_mean": recent_flux,
            "flux_normalized_imbalance": flux_imbalance,
            "probe_U_late_mean_relative_change": probe_u_change,
            "probe_p_late_mean_relative_change": probe_p_change,
            "mean_speed_ms": {
                "min": float(speed.min()),
                "median": float(np.median(speed)),
                "p99": float(np.percentile(speed, 99)),
                "max": float(speed.max()),
            },
        },
        "gates": gates,
        "accepted": all(gates.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate URANS time-mean stability")
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--field-time", required=True)
    parser.add_argument("--previous-time", required=True)
    parser.add_argument("--averaging-start", required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.case,
        args.field_time,
        args.previous_time,
        args.averaging_start,
        args.sample_count,
        args.output,
    )
    if not result["accepted"]:
        raise SystemExit(2)
