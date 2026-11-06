"""Deterministic synthetic B-class measurement package generator."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from step30_experimental_validation import (
    MANIFEST_VERSION, add_common_cli, blocked_result, finalize_run,
    new_output_dir, normalise_mode, read_records, support_status, temp_dir,
)

VERSION = "step30-synthetic-measurements-v1"
MATERIAL_COLUMNS = (
    "case_id", "speed_ms", "impact_angle_deg", "particle_size_um",
    "sediment_mass_kg", "target_area_mm2", "material_erosion_mm3kg",
    "wear_volume_mm3", "mass_loss_mg", "wear_depth_mm",
)
FIELD_COLUMNS = (
    "operating_point", "power_mw", "head_m", "flow_m3s",
    "stress_mean_mpa", "stress_amplitude_mpa", "dominant_frequency_hz",
    "wear_truth_role",
)


def material_erosion_law(speed_ms: Any, angle_deg: Any, particle_size_um: Any) -> np.ndarray:
    speed = np.asarray(speed_ms, dtype=float)
    angle = np.deg2rad(np.clip(np.asarray(angle_deg, dtype=float), 0.1, 89.9))
    size = np.asarray(particle_size_um, dtype=float)
    return 12.0 * (speed / 50.0) ** 2.2 * (size / 50.0) ** 0.35 * np.sin(angle) ** 1.2 * (1.0 + 0.25 * np.cos(angle) ** 2)


def _material(seed: int) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    speed = np.asarray([30, 42, 55, 68, 82, 96, 108, 120], dtype=float)
    angle = np.asarray([20, 30, 42, 55, 65, 72, 80, 86], dtype=float)
    size = np.asarray([35, 50, 75, 100, 125, 150, 180, 220], dtype=float)
    sediment = np.asarray([0.008 + i * 0.001 for i in range(8)])
    area = np.full(8, 2500.0)
    erosion = material_erosion_law(speed, angle, size)
    volume = erosion * sediment
    mass_loss = volume * 7.85 * (1.0 + rng.normal(0.0, 0.012, 8))
    depth = volume / area * (1.0 + rng.normal(0.0, 0.008, 8))
    rows = []
    for i in range(8):
        rows.append({
            "case_id": f"mat-{i + 1:02d}", "speed_ms": speed[i],
            "impact_angle_deg": angle[i], "particle_size_um": size[i],
            "sediment_mass_kg": sediment[i], "target_area_mm2": area[i],
            "material_erosion_mm3kg": erosion[i], "wear_volume_mm3": volume[i],
            "mass_loss_mg": mass_loss[i], "wear_depth_mm": depth[i],
        })
    arrays = {
        "speed_ms": speed, "impact_angle_deg": angle, "particle_size_um": size,
        "sediment_mass_kg": sediment, "target_area_mm2": area,
        "material_erosion_mm3kg": erosion, "wear_volume_mm3": volume,
        "mass_loss_mg": mass_loss, "wear_depth_mm": depth,
    }
    return rows, arrays


def _bucket(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 1001)
    u = np.linspace(0, 1, 14)
    v = np.linspace(0, 1, 10)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    xyz = np.stack((120 * (uu - 0.5), 80 * (vv - 0.5), 8 * np.sin(np.pi * uu) * np.sin(np.pi * vv)), axis=-1).reshape(-1, 3)
    normals = np.stack((-0.06 * np.pi * np.cos(np.pi * uu), -0.10 * np.pi * np.cos(np.pi * vv), np.ones_like(uu)), axis=-1).reshape(-1, 3)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    risk = np.exp(-((uu - 0.55) ** 2 / 0.055 + (vv - 0.48) ** 2 / 0.09)).reshape(-1)
    depth = np.clip(0.08 + 0.92 * risk + rng.normal(0, 0.008, len(xyz)), 0.01, None)
    worn = xyz - normals * depth[:, None]
    point = np.arange(len(xyz), dtype=np.int32)
    # Stable parent-face IDs make the point-to-topology contract explicit.
    parent = ((point // 2) % 117).astype(np.int32)
    return {
        "original_xyz_mm": xyz, "worn_xyz_mm": worn, "normals": normals,
        "wear_depth_mm": depth, "topology_point_id": point,
        "topology_parent_face_id": parent,
        "topology_mapping": np.column_stack((point, parent)),
    }


def _field() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [
        {"operating_point": "rated", "power_mw": 143.6, "head_m": 595.0, "flow_m3s": 27.0, "stress_mean_mpa": 28.1, "stress_amplitude_mpa": 12.6, "dominant_frequency_hz": 37.2, "wear_truth_role": "context_only_not_erosion_truth"},
        {"operating_point": "50MW", "power_mw": 50.0, "head_m": 570.0, "flow_m3s": 11.2, "stress_mean_mpa": 25.8, "stress_amplitude_mpa": 10.4, "dominant_frequency_hz": 37.2, "wear_truth_role": "context_only_not_erosion_truth"},
        {"operating_point": "30MW", "power_mw": 30.0, "head_m": 540.0, "flow_m3s": 7.4, "stress_mean_mpa": 24.5, "stress_amplitude_mpa": 8.9, "dominant_frequency_hz": 12.7, "wear_truth_role": "context_only_not_erosion_truth"},
    ]
    metadata = {
        "asset": "synthetic_pelton_bucket", "site": "synthetic_test_rig",
        "component": "bucket", "material": "ZG0Cr13Ni4Mo",
        "coordinate_units": "mm", "service_hours": 120.0,
        "wear_measurement_method": "known synthetic depth field",
        "stress_context_role": "context_only_not_erosion_truth",
        "truth_source": "wear_depth_mm; never stress",
    }
    return rows, metadata


def run(args: argparse.Namespace) -> dict[str, Any]:
    mode = normalise_mode(args.mode)
    output = new_output_dir(args.output_dir, "synthetic-measurements")
    if mode == "blocked":
        return blocked_result(output, "synthetic_measurements", mode, args.seed, "blocked_by_request", required_inputs=("none",))
    if mode == "measured":
        if not getattr(args, "input", None):
            return blocked_result(output, "synthetic_measurements", mode, args.seed, "measured mode requires --input; no real input was supplied", required_inputs=("--input",))
        try:
            rows = read_records(args.input)
            if not rows:
                raise ValueError("input contains no rows")
        except (OSError, ValueError) as exc:
            return blocked_result(output, "synthetic_measurements", mode, args.seed, f"cannot read measured input: {exc}", required_inputs=("--input",), provided_inputs=(args.input,))
        fields = sorted({key for row in rows for key in row})
        arrays = {field: np.asarray([row.get(field, "") for row in rows]) for field in fields}
        return finalize_run(output, "synthetic_measurements", mode, args.seed, "measured", {
            "version": VERSION, "manifest_version": MANIFEST_VERSION, "status": "measured",
            "measurement_status": "real_input_loaded", "truth_source": "provided_real_input", "n_rows": len(rows),
        }, {"measurements": (rows, fields)}, arrays, required_inputs=("--input",), provided_inputs=(args.input,))
    material_rows, material_arrays = _material(args.seed)
    bucket = _bucket(args.seed)
    field_rows, field_metadata = _field()
    arrays = {**material_arrays, **bucket}
    summary = {
        "version": VERSION, "manifest_version": MANIFEST_VERSION, "status": "synthetic", "mode": mode,
        "measurement_status": "known_synthetic_truth", "truth_source": "fixed_seed_generator",
        "seed_policy": "all random terms derive from np.random.default_rng(--seed)",
        "material_model": {
            "name": "interpretable_power_angle_law",
            "formula": "E=12*(V/50)^2.2*(Dp/50)^0.35*sin(alpha)^1.2*(1+0.25*cos(alpha)^2)",
            "density_mg_per_mm3": 7.85, "wear_depth_formula": "wear_volume_mm3/target_area_mm2",
        },
        "material_rows": len(material_rows), "bucket_points": int(len(bucket["original_xyz_mm"])),
        "field_metadata": field_metadata, "stress_used_as_erosion_truth": False,
        "synthetic_truth": {
            "material_table": "step30-synthetic_measurements-material_measurements.csv:material_erosion_mm3kg,mass_loss_mg,wear_depth_mm",
            "wear_depth_array": "step30-synthetic_measurements-arrays.npz:wear_depth_mm",
            "material_erosion_array": "step30-synthetic_measurements-arrays.npz:material_erosion_mm3kg",
            "bucket_original_array": "step30-synthetic_measurements-arrays.npz:original_xyz_mm",
            "bucket_worn_array": "step30-synthetic_measurements-arrays.npz:worn_xyz_mm",
            "topology_mapping_array": "step30-synthetic_measurements-arrays.npz:topology_mapping",
            "stress_is_context_only": True,
        },
    }
    return finalize_run(output, "synthetic_measurements", mode, args.seed, "synthetic", summary, {
        "material_measurements": (material_rows, MATERIAL_COLUMNS), "field_stress_context": (field_rows, FIELD_COLUMNS),
    }, arrays, support=support_status(), evidence={"status": "synthetic", "source_type": "local_synthetic_fixture", "reference_role": "synthetic_pipeline_fixture", "pipeline_validation": True}, notes=(
        "wear_depth_mm and material_erosion_mm3kg are known synthetic truth arrays and table columns.",
        "topology_mapping is point_id to parent_face_id.",
        "field stress context is not erosion truth.",
    ))


def selftest() -> None:
    import argparse as _argparse
    first = run(_argparse.Namespace(mode="synthetic", output_dir=str(temp_dir("gen-a")), seed=17, input=None))
    second = run(_argparse.Namespace(mode="synthetic", output_dir=str(temp_dir("gen-b")), seed=17, input=None))
    assert first["status"] == second["status"] == "synthetic"
    with np.load(Path(first["output_dir"]) / "step30-synthetic_measurements-arrays.npz") as left, np.load(Path(second["output_dir"]) / "step30-synthetic_measurements-arrays.npz") as right:
        assert np.array_equal(left["wear_depth_mm"], right["wear_depth_mm"])
        assert np.array_equal(left["material_erosion_mm3kg"], right["material_erosion_mm3kg"])
    blocked = run(_argparse.Namespace(mode="measured", output_dir=str(temp_dir("gen-blocked")), seed=17, input=None))
    assert blocked["status"] == "blocked"
    print("step30_generate_synthetic_measurements selftest: PASS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_common_cli(parser, "Generate deterministic synthetic B-class measurement artifacts")
    parser.add_argument("--input", default=None, help="real CSV/JSON/NPZ input for measured mode")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        result = run(args)
        print(__import__("json").dumps(result, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x))
        raise SystemExit(0 if result["status"] in {"synthetic", "measured"} else 2)
