"""Request a clean URANS stop after consecutive accepted averaging checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import time
from decimal import Decimal
from pathlib import Path

import numpy as np


def _load_validator(path: Path):
    spec = importlib.util.spec_from_file_location("step30_urans_validation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load URANS validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_times(case: Path, start: Decimal) -> list[tuple[Decimal, str]]:
    values = []
    for path in (case / "processor0").iterdir():
        if not path.is_dir():
            continue
        try:
            value = Decimal(path.name)
        except Exception:
            continue
        if value >= start:
            values.append((value, path.name))
    return sorted(values)


def _parallel_field_metrics(case: Path, validator, previous: str, current: str):
    metrics = {}
    speed_max = 0.0
    all_finite = True
    processors = sorted(case.glob("processor*"))
    if not processors:
        raise ValueError("no decomposed URANS fields")
    for name, kind in {
        "UMean": "vector",
        "pMean": "scalar",
        "kMean": "scalar",
        "omegaMean": "scalar",
    }.items():
        numerator = 0.0
        denominator = 0.0
        for processor in processors:
            current_values = validator._internal_field(processor / current / name, kind)
            previous_values = validator._internal_field(processor / previous / name, kind)
            all_finite = all_finite and bool(np.isfinite(current_values).all())
            numerator += float(np.sum((current_values - previous_values) ** 2))
            denominator += float(np.sum(current_values ** 2))
            if name == "UMean":
                speed_max = max(speed_max, float(np.linalg.norm(current_values, axis=1).max()))
        metrics[name] = float(np.sqrt(numerator / max(denominator, 1e-30)))

    numerator = 0.0
    denominator = 0.0
    found = False
    for processor in processors:
        try:
            current_values = validator._bucket_vector_field(
                processor / current / "wallShearStressMean"
            )
            previous_values = validator._bucket_vector_field(
                processor / previous / "wallShearStressMean"
            )
        except ValueError:
            continue
        found = True
        all_finite = all_finite and bool(np.isfinite(current_values).all())
        numerator += float(np.sum((current_values - previous_values) ** 2))
        denominator += float(np.sum(current_values ** 2))
    if not found:
        raise ValueError("no decomposed bucket wallShearStressMean values")
    metrics["wallShearStressMean"] = float(np.sqrt(numerator / max(denominator, 1e-30)))
    return metrics, speed_max, all_finite


def _monitor_metrics(case: Path, validator, start: Decimal, field_time: Decimal) -> dict:
    force_rows = validator._averaging_rows(
        validator._numeric_rows(case, "postProcessing/bucketForces/*/force*.dat"),
        start,
        field_time,
        "bucket forces",
    )
    force = {
        "pressure": force_rows[:, 1:4],
        "viscous": force_rows[:, 4:7],
    }
    force["total"] = force["pressure"] + force["viscous"]
    force_change = {
        name: validator._relative_mean_change(values) for name, values in force.items()
    }

    flux = {}
    for name in ("inletFlux", "jetFlux", "outletFlux", "sideFlux"):
        rows = validator._averaging_rows(
            validator._numeric_rows(case, f"postProcessing/{name}/*/surfaceFieldValue*.dat"),
            start,
            field_time,
            name,
        )
        flux[name] = rows[:, -1]
    flux_change, recent_flux = validator._flux_relative_mean_change(flux)
    flux_imbalance = abs(sum(recent_flux.values())) / max(
        sum(abs(value) for value in recent_flux.values()), 1e-30
    )

    probe_u_times, probe_u_values = validator._probe_vectors(case)
    probe_u = validator._averaging_values(
        probe_u_times, probe_u_values, start, field_time, "U probes"
    )
    probe_p_rows = validator._averaging_rows(
        validator._numeric_rows(case, "postProcessing/flowProbes/*/p"),
        start,
        field_time,
        "p probes",
    )
    return {
        "force_change": force_change,
        "flux_change": flux_change,
        "flux_imbalance": float(flux_imbalance),
        "probe_u_change": validator._relative_mean_change(probe_u),
        "probe_p_change": validator._relative_mean_change(probe_p_rows[:, 1:]),
    }


def evaluate(
    case: Path,
    validator,
    start: Decimal,
    minimum_end: Decimal,
    previous: str,
    current: str,
) -> dict:
    current_time = Decimal(current)
    mean_l2, speed_max, all_finite = _parallel_field_metrics(
        case, validator, previous, current
    )
    monitors = _monitor_metrics(case, validator, start, current_time)
    log_text = (case / "log.urans.average").read_text(encoding="utf-8", errors="replace")
    max_co = max(
        float(value)
        for value in re.findall(
            r"Courant Number mean: [^\s]+ max: ([^\s]+)", log_text
        )
    )
    gates = {
        "max_co_below_0p5": max_co <= 0.5,
        "no_turbulence_bounding": (
            "bounding k," not in log_text and "bounding omega," not in log_text
        ),
        "no_fatal_or_nonfinite_log": not bool(
            re.search(r"FOAM FATAL|\b(?:nan|[-+]?inf)\b", log_text, re.IGNORECASE)
        ),
        "twenty_flow_through_window": current_time >= minimum_end,
        "mean_field_relative_l2_below_1e3": max(mean_l2.values()) < 1e-3,
        "force_late_mean_change_below_1pct": max(monitors["force_change"].values()) < 0.01,
        "flux_late_mean_change_below_1pct": max(monitors["flux_change"].values()) < 0.01,
        "flux_imbalance_below_1e4": monitors["flux_imbalance"] < 1e-4,
        "probe_late_mean_change_below_1pct": max(
            monitors["probe_u_change"], monitors["probe_p_change"]
        ) < 0.01,
        "mean_speed_below_500_ms": speed_max < 500.0,
        "all_mean_fields_finite": all_finite,
    }
    return {
        "previous_time": previous,
        "field_time": current,
        "mean_field_relative_l2": mean_l2,
        "max_speed_ms": speed_max,
        "monitors": monitors,
        "max_courant_number": max_co,
        "gates": {name: bool(value) for name, value in gates.items()},
        "accepted": all(gates.values()),
    }


def run(
    case: Path,
    validator_path: Path,
    averaging_start: str,
    flow_through_time: str,
    minimum_flow_throughs: int,
    consecutive_required: int,
    poll_seconds: int,
    solver_pid: int,
) -> None:
    case = case.resolve()
    validator = _load_validator(validator_path.resolve())
    start = Decimal(averaging_start)
    minimum_end = start + Decimal(minimum_flow_throughs) * Decimal(flow_through_time)
    last_checked = None
    consecutive = 0
    while True:
        times = _write_times(case, start)
        if len(times) >= 2 and times[-1][0] >= minimum_end and times[-1][1] != last_checked:
            previous = times[-2][1]
            current = times[-1][1]
            try:
                result = evaluate(case, validator, start, minimum_end, previous, current)
            except (OSError, ValueError, EOFError) as error:
                print(f"checkpoint {current} is not ready: {error}", flush=True)
                time.sleep(poll_seconds)
                continue
            result["minimum_flow_throughs"] = minimum_flow_throughs
            result["flow_throughs"] = float(
                (Decimal(current) - start) / Decimal(flow_through_time)
            )
            output = case / f"urans_live_gate_{current}.json"
            output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result, indent=2), flush=True)
            last_checked = current
            consecutive = consecutive + 1 if result["accepted"] else 0
            print(f"consecutive acceptances: {consecutive}", flush=True)
            if consecutive >= consecutive_required:
                control = case / "system" / "controlDict"
                text = control.read_text(encoding="utf-8")
                if text.count("stopAt endTime;") != 1:
                    raise RuntimeError("cannot switch controlDict to nextWrite")
                control.write_text(
                    text.replace("stopAt endTime;", "stopAt nextWrite;"),
                    encoding="utf-8",
                )
                print(
                    f"accepted {consecutive} times through {current}; requested nextWrite stop",
                    flush=True,
                )
                return
        try:
            os.kill(solver_pid, 0)
        except ProcessLookupError:
            print("URANS solver exited before live acceptance", flush=True)
            return
        time.sleep(poll_seconds)


if __name__ == "__main__":
    default_case = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Stop URANS after consecutive accepted means")
    parser.add_argument("--case", type=Path, default=default_case)
    parser.add_argument(
        "--validator",
        type=Path,
        default=default_case / ".." / ".." / "step30_urans_validation.py",
    )
    parser.add_argument("--averaging-start", default="0.0452682176")
    parser.add_argument("--flow-through-time", default="0.0036056848")
    parser.add_argument("--minimum-flow-throughs", type=int, default=20)
    parser.add_argument("--consecutive-required", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--solver-pid", type=int, required=True)
    args = parser.parse_args()
    if args.minimum_flow_throughs < 20 or args.consecutive_required < 2 or args.poll_seconds < 1:
        parser.error("watcher limits are below the production contract")
    run(
        args.case,
        args.validator,
        args.averaging_start,
        args.flow_through_time,
        args.minimum_flow_throughs,
        args.consecutive_required,
        args.poll_seconds,
        args.solver_pid,
    )
