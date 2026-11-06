"""B-class material erosion experiment harness."""
from __future__ import annotations

import argparse
from typing import Any, Mapping

import numpy as np

from step30_experimental_validation import (
    MANIFEST_VERSION, add_common_cli, blocked_result, call_error_metrics,
    finalize_run, new_output_dir, normalise_mode, read_records, support_status,
    temp_dir,
)
from step30_generate_synthetic_measurements import material_erosion_law

FIELDS = (
    "case_id", "speed_ms", "impact_angle_deg", "particle_size_um",
    "sediment_mass_kg", "target_area_mm2", "observed_mass_loss_mg",
    "observed_wear_depth_mm", "predicted_erosion_mm3kg",
    "predicted_mass_loss_mg", "predicted_wear_depth_mm", "observation_kind",
)


def _number(row: Mapping[str, Any], names: tuple[str, ...], required: bool = True) -> float | None:
    for name in names:
        if name in row and str(row[name]).strip():
            value = float(row[name])
            if not np.isfinite(value):
                raise ValueError(f"{name} is not finite")
            return value
    if required:
        raise ValueError(f"missing required column; expected one of {names}")
    return None


def _normalise(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    if not rows:
        raise ValueError("material table is empty")
    result: list[dict[str, Any]] = []
    kinds: set[str] = set()
    for index, row in enumerate(rows):
        speed = _number(row, ("speed_ms", "speed", "velocity_ms", "velocity"))
        angle = _number(row, ("impact_angle_deg", "angle_deg", "impact_angle", "angle"))
        size = _number(row, ("particle_size_um", "particle_diameter_um", "particle_size", "dp_um"))
        mass = _number(row, ("mass_loss_mg", "mass_loss"), False)
        depth = _number(row, ("wear_depth_mm", "depth_mm", "depth"), False)
        if mass is None and depth is None:
            raise ValueError("each row needs mass_loss_mg or wear_depth_mm")
        if speed <= 0 or size <= 0 or not 0 < angle <= 90:
            raise ValueError("speed and particle size must be positive; angle must be in (0,90]")
        if mass is not None and mass < 0 or depth is not None and depth < 0:
            raise ValueError("observations must be non-negative")
        sediment = _number(row, ("sediment_mass_kg", "particle_mass_kg", "mass_kg"), False) or 0.01
        area = _number(row, ("target_area_mm2", "area_mm2", "area"), False) or 2500.0
        erosion = float(material_erosion_law(speed, angle, size))
        volume = erosion * sediment
        kind = "mass_loss_mg" if mass is not None else "wear_depth_mm"
        kinds.add(kind)
        result.append({
            "case_id": str(row.get("case_id", f"material-{index + 1:03d}")),
            "speed_ms": speed, "impact_angle_deg": angle, "particle_size_um": size,
            "sediment_mass_kg": sediment, "target_area_mm2": area,
            "observed_mass_loss_mg": "" if mass is None else mass,
            "observed_wear_depth_mm": "" if depth is None else depth,
            "predicted_erosion_mm3kg": erosion,
            "predicted_mass_loss_mg": volume * 7.85,
            "predicted_wear_depth_mm": volume / area,
            "observation_kind": kind,
        })
    return result, next(iter(kinds)) if len(kinds) == 1 else "mixed"


def _synthetic_rows(seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    for index, (speed, angle, size) in enumerate(zip([30, 42, 55, 68, 82, 96, 108, 120], [20, 30, 42, 55, 65, 72, 80, 86], [35, 50, 75, 100, 125, 150, 180, 220])):
        sediment = 0.008 + 0.001 * index
        volume = float(material_erosion_law(speed, angle, size)) * sediment
        rows.append({
            "case_id": f"synthetic-{index + 1:02d}", "speed_ms": speed,
            "impact_angle_deg": angle, "particle_size_um": size,
            "sediment_mass_kg": sediment, "target_area_mm2": 2500.0,
            "mass_loss_mg": volume * 7.85 * (1.0 + rng.normal(0, 0.012)),
        })
    return rows


def _load_measured(path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Use the strict planned reader first.  Its named-column table is then
    # normalized through the same validation path as the fallback reader.
    try:
        from step30_measurement_io import read_material_test_table
        record = read_material_test_table(path, status="measured")
        columns = list(record["columns"])
        rows = [dict(zip(columns, values)) for values in np.asarray(record["values"])]
        return rows, {"used": True, "function": "read_material_test_table", "implementation": "external_module"}
    except Exception as exc:
        rows = read_records(path)
        return rows, {"used": False, "implementation": "local_fallback", "reason": type(exc).__name__}


def run(args: argparse.Namespace) -> dict[str, Any]:
    mode = normalise_mode(args.mode)
    output = new_output_dir(args.output_dir, "material-erosion")
    if mode == "blocked":
        return blocked_result(output, "material_erosion", mode, args.seed, "blocked_by_request", required_inputs=("--input",))
    if mode == "measured":
        if not args.input:
            return blocked_result(output, "material_erosion", mode, args.seed, "measured mode requires --input", required_inputs=("--input",))
        try:
            raw, io_info = _load_measured(args.input)
            rows, kind = _normalise(raw)
        except (OSError, ValueError, TypeError) as exc:
            return blocked_result(output, "material_erosion", mode, args.seed, f"invalid measured material input: {exc}", required_inputs=("--input",), provided_inputs=(args.input,))
        status, truth_source, provided = "measured", "provided_real_material_measurement", (args.input,)
    else:
        rows, kind = _normalise(_synthetic_rows(args.seed))
        io_info = {"used": False, "implementation": "fixed_seed_generator"}
        status, truth_source, provided = "synthetic", "fixed_seed_generator", ()
    metric_results: dict[str, Any] = {}
    metric_info: dict[str, Any] = {}
    for observation, prediction in (("observed_mass_loss_mg", "predicted_mass_loss_mg"), ("observed_wear_depth_mm", "predicted_wear_depth_mm")):
        selected = [row for row in rows if row[observation] != ""]
        if selected:
            metric_results[observation], metric_info[observation] = call_error_metrics(
                [float(row[prediction]) for row in selected], [float(row[observation]) for row in selected], status=mode,
            )
    arrays = {name: np.asarray([row[name] if row[name] != "" else np.nan for row in rows], dtype=float) for name in (
        "speed_ms", "impact_angle_deg", "particle_size_um", "predicted_erosion_mm3kg", "predicted_mass_loss_mg", "predicted_wear_depth_mm", "observed_mass_loss_mg", "observed_wear_depth_mm")}
    summary = {
        "version": "step30-material-erosion-experiment-v1", "manifest_version": MANIFEST_VERSION,
        "status": status, "mode": mode, "measurement_status": "known_synthetic_truth" if mode == "synthetic" else "real_input_loaded",
        "truth_source": truth_source, "n_rows": len(rows), "observation_kind": kind,
        "condition_columns": ["speed_ms", "impact_angle_deg", "particle_size_um"],
        "prediction_model": {"name": "interpretable_power_angle_law", "formula": "E=12*(V/50)^2.2*(Dp/50)^0.35*sin(alpha)^1.2*(1+0.25*cos(alpha)^2)", "density_mg_per_mm3": 7.85},
        "comparison": metric_results, "error_metrics_interfaces": metric_info,
        "measurement_io_interface": io_info, "stress_used_as_erosion_truth": False,
    }
    return finalize_run(output, "material_erosion", mode, args.seed, status, summary, {"material_comparison": (rows, FIELDS)}, arrays, required_inputs=("--input",) if mode == "measured" else (), provided_inputs=provided, support=support_status(), evidence={"status": "synthetic", "source_type": "local_synthetic_fixture", "reference_role": "synthetic_pipeline_fixture", "pipeline_validation": True} if mode == "synthetic" else None, notes=("Measured mode requires speed, angle, particle size, and mass loss or depth.", "No missing observation is imputed."))


def selftest() -> None:
    import argparse as ap
    result = run(ap.Namespace(mode="synthetic", output_dir=str(temp_dir("material")), seed=21, input=None))
    assert result["status"] == "synthetic" and result["tables"]["material_comparison"]["rows"] == 8
    blocked = run(ap.Namespace(mode="measured", output_dir=str(temp_dir("material-blocked")), seed=21, input=None))
    assert blocked["status"] == "blocked"
    print("step30_run_material_erosion_experiment selftest: PASS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_common_cli(parser, "Run B-class material erosion experiment")
    parser.add_argument("--input", default=None, help="measured material CSV/JSON/NPZ")
    return parser


if __name__ == "__main__":
    parser = build_parser(); args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        result = run(args); import json
        print(json.dumps(result, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x))
        raise SystemExit(0 if result["status"] in {"synthetic", "measured"} else 2)
