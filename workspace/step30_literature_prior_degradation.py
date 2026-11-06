"""Literature-prior erosion comparison and exploratory structural sensitivity.

This module is deliberately separate from the real-FEM and official-accuracy
gates. It compares the existing Oka/Finnie/Tabakoff/Mansouri implementations
on the immutable particle-hit artifact, propagates material-density and
hardness priors, and optionally scales the existing analytical stress field
for an exploratory centrifugal sensitivity study.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from step30_material_model_calibration import (
    finnie_volume_mm3_per_kg,
    mansouri_mass_ratio_kg_per_kg,
    oka_volume_mm3_per_kg,
    tabakoff_volume_mm3_per_kg,
)
from step30_validation_common import canonical_sha256, safe_json_dump, sha256_file


VERSION = "step30-literature-prior-degradation-v1"
MODEL_NAMES = ("oka", "finnie", "tabakoff", "mansouri")
DEFAULT_DENSITY_KG_M3 = {"low": 7700.0, "base": 7800.0, "high": 7900.0}
DEFAULT_HARDNESS_GPA = {"low": 3.0, "base": 4.0, "high": 5.0}
DEFAULT_CONCENTRATION_SCALE = {"low": 0.5, "base": 1.0, "high": 2.0}

LITERATURE_SOURCES = (
    {
        "id": "rai_2017_wear",
        "citation": "Rai, Kumar and Staubli, Hydro-abrasive erosion in Pelton buckets: Classification and field study",
        "journal": "Wear",
        "year": 2017,
        "doi": "10.1016/j.wear.2017.08.016",
        "role": "field erosion classification and mechanism context",
    },
    {
        "id": "rai_2019_analytical",
        "citation": "Rai, Kumar and Staubli, Analytical modelling and mechanism of hydro-abrasive erosion in pelton buckets",
        "journal": "Wear",
        "year": 2019,
        "doi": "10.1016/j.wear.2019.203003",
        "role": "analytical erosion mechanism context",
    },
    {
        "id": "leguizamon_2019_multiscale",
        "citation": "Leguizamon et al., Multiscale Simulation of the Hydroabrasive Erosion of a Pelton Bucket: Bridging Scales to Improve the Accuracy",
        "journal": "International Journal of Turbomachinery, Propulsion and Power",
        "year": 2019,
        "doi": "10.3390/ijtpp4020009",
        "role": "prototype-scale multiscale particle/sediment model",
    },
    {
        "id": "messa_2019_injector",
        "citation": "Messa, Mandelli and Malavasi, Hydro-abrasive erosion in Pelton turbine injectors: A numerical study",
        "journal": "Renewable Energy",
        "year": 2019,
        "doi": "10.1016/j.renene.2018.06.064",
        "role": "Eulerian-Lagrangian erosion parameter sensitivity",
    },
    {
        "id": "ge_2021_bucket",
        "citation": "Ge et al., Experimental and Numerical studies on Opening and Velocity Influence on Sediment Erosion of Pelton Turbine Buckets",
        "journal": "Renewable Energy",
        "year": 2021,
        "doi": "10.1016/j.renene.2021.04.072",
        "role": "bucket erosion experiment/numerical comparison",
    },
    {
        "id": "fahrni_2025_defects",
        "citation": "Fahrni, Staubli and Casartelli, Efficiency Testing of Pelton Turbines with Artificial Defects—Part 1: Buckets",
        "journal": "Energies",
        "year": 2025,
        "doi": "10.3390/en18112716",
        "role": "defect geometry to efficiency-loss sensitivity",
    },
)


class LiteraturePriorError(ValueError):
    """Raised when exploratory inputs violate the prior-model contract."""


def _finite_array(value: Any, name: str, *, positive: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise LiteraturePriorError(f"{name} must be a non-empty finite array")
    if positive and np.any(array <= 0.0):
        raise LiteraturePriorError(f"{name} must be positive")
    return array


def _load_hits(path: str | Path) -> dict[str, np.ndarray]:
    source = Path(path).resolve()
    if not source.is_file():
        raise LiteraturePriorError(f"particle-hit artifact does not exist: {source}")
    with np.load(source, allow_pickle=False) as data:
        required = {
            "parent_face_id",
            "velocity_ms",
            "Vp_ms",
            "alpha_deg",
            "diameter_m",
            "parcel_mass_kg",
            "face_centers_local_mm",
            "face_area_m2",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise LiteraturePriorError(f"particle-hit artifact is missing fields: {missing}")
        parent = np.asarray(data["parent_face_id"], dtype=np.int64)
        velocity = _finite_array(data["velocity_ms"], "velocity_ms")
        if velocity.ndim != 2 or velocity.shape[1] != 3:
            raise LiteraturePriorError("velocity_ms must have shape (N, 3)")
        speed = np.linalg.norm(velocity, axis=1)
        normal_speed = _finite_array(data["Vp_ms"], "Vp_ms", positive=True)
        speed = np.where(speed > 1.0e-9, speed, normal_speed)
        angle = np.clip(_finite_array(data["alpha_deg"], "alpha_deg"), 0.1, 89.9)
        d50_um = _finite_array(data["diameter_m"], "diameter_m", positive=True) * 1.0e6
        parcel_mass = _finite_array(data["parcel_mass_kg"], "parcel_mass_kg", positive=True)
        centers = _finite_array(data["face_centers_local_mm"], "face_centers_local_mm")
        areas = _finite_array(data["face_area_m2"], "face_area_m2", positive=True)
        if not (len(parent) == len(speed) == len(angle) == len(d50_um) == len(parcel_mass)):
            raise LiteraturePriorError("particle-hit arrays have inconsistent lengths")
        if np.any(parent < 0) or np.any(parent >= len(areas)):
            raise LiteraturePriorError("parent_face_id is outside the face domain")
        return {
            "parent_face_id": parent,
            "speed_m_s": speed,
            "normal_speed_m_s": normal_speed,
            "angle_deg": angle,
            "d50_um": d50_um,
            "parcel_mass_kg": parcel_mass,
            "face_centers_local_mm": centers,
            "face_area_m2": areas,
            "source_path": np.asarray(str(source)),
            "source_sha256": np.asarray(sha256_file(source)),
        }


def _model_volume_mm3_per_kg(
    model: str,
    speed_m_s: np.ndarray,
    angle_deg: np.ndarray,
    d50_um: np.ndarray,
    hardness_gpa: float,
) -> np.ndarray:
    if model == "oka":
        return np.asarray(
            oka_volume_mm3_per_kg(
                speed_m_s, angle_deg, d50_um, hardness_gpa=hardness_gpa
            ),
            dtype=np.float64,
        )
    if model == "finnie":
        return np.asarray(finnie_volume_mm3_per_kg(speed_m_s, angle_deg), dtype=np.float64)
    if model == "tabakoff":
        return np.asarray(tabakoff_volume_mm3_per_kg(speed_m_s, angle_deg), dtype=np.float64)
    if model == "mansouri":
        return np.asarray(
            mansouri_mass_ratio_kg_per_kg(speed_m_s, angle_deg), dtype=np.float64
        )
    raise LiteraturePriorError(f"unsupported erosion model: {model}")


def _accumulate_depth(
    parent_face_id: np.ndarray,
    removed_volume_mm3: np.ndarray,
    face_area_m2: np.ndarray,
    *,
    exposure_h: float,
) -> np.ndarray:
    if exposure_h <= 0.0 or not math.isfinite(exposure_h):
        raise LiteraturePriorError("exposure_h must be finite and positive")
    area_mm2 = face_area_m2 * 1.0e6
    depth = np.bincount(
        parent_face_id,
        weights=removed_volume_mm3 / area_mm2[parent_face_id],
        minlength=len(face_area_m2),
    )
    return depth / exposure_h


def _model_map(
    hits: Mapping[str, np.ndarray],
    model: str,
    *,
    density_kg_m3: float,
    hardness_gpa: float,
    concentration_scale: float,
    exposure_h: float,
) -> np.ndarray:
    if density_kg_m3 <= 0.0 or hardness_gpa <= 0.0 or concentration_scale <= 0.0:
        raise LiteraturePriorError("prior parameters must be positive")
    base = _model_volume_mm3_per_kg(
        model,
        hits["speed_m_s"],
        hits["angle_deg"],
        hits["d50_um"],
        hardness_gpa,
    )
    mass = hits["parcel_mass_kg"] * concentration_scale
    if model == "mansouri":
        removed_volume_mm3 = base * mass / density_kg_m3 * 1.0e9
    else:
        removed_volume_mm3 = base * mass
    if not np.isfinite(removed_volume_mm3).all() or np.any(removed_volume_mm3 < 0.0):
        raise LiteraturePriorError(f"{model} produced invalid removed volume")
    return _accumulate_depth(
        hits["parent_face_id"],
        removed_volume_mm3,
        hits["face_area_m2"],
        exposure_h=exposure_h,
    ).astype(np.float32)


def _summary(values: np.ndarray, face_area_m2: np.ndarray) -> dict[str, float]:
    """Summarize face rates and area-integrated removed volume.

    ``values`` may be one face map or a draw-by-face matrix.  The previous
    implementation flattened the map and returned a hard-coded zero for the
    area-integrated quantity, which made that field unusable for comparison.
    """
    field = np.asarray(values, dtype=np.float64)
    areas = _finite_array(face_area_m2, "face_area_m2", positive=True)
    if field.ndim == 1:
        if len(field) != len(areas):
            raise LiteraturePriorError("summary face count differs from face_area_m2")
        draw_totals = np.asarray([np.sum(field * areas * 1.0e6)], dtype=np.float64)
    elif field.ndim == 2:
        if field.shape[1] != len(areas):
            raise LiteraturePriorError("summary face count differs from face_area_m2")
        draw_totals = np.sum(field * areas[None, :] * 1.0e6, axis=1)
    else:
        raise LiteraturePriorError("summary values must be one- or two-dimensional")
    return {
        "p05_mm_h": float(np.quantile(field, 0.05)),
        "p50_mm_h": float(np.quantile(field, 0.50)),
        "p95_mm_h": float(np.quantile(field, 0.95)),
        "mean_mm_h": float(np.mean(field)),
        "max_mm_h": float(np.max(field)),
        "total_area_weighted_p05_mm3_h": float(np.quantile(draw_totals, 0.05)),
        "total_area_weighted_mm3_h": float(np.quantile(draw_totals, 0.50)),
        "total_area_weighted_p95_mm3_h": float(np.quantile(draw_totals, 0.95)),
    }


def _scalar_field_summary(values: np.ndarray) -> dict[str, float]:
    field = np.asarray(values, dtype=np.float64)
    if field.ndim != 2 or field.shape[0] < 1 or field.shape[1] < 1:
        raise LiteraturePriorError("scalar field draws must have shape (n_draws, n_faces)")
    return {
        "p05_mpa": float(np.quantile(field, 0.05)),
        "p50_mpa": float(np.quantile(field, 0.50)),
        "p95_mpa": float(np.quantile(field, 0.95)),
        "mean_mpa": float(np.mean(field)),
        "max_mpa": float(np.max(field)),
    }


def _load_exploratory_stress(path: str | Path | None, n_faces: int) -> dict[str, np.ndarray] | None:
    if path is None:
        return None
    source = Path(path).resolve()
    if not source.is_file():
        raise LiteraturePriorError(f"exploratory stress field does not exist: {source}")
    with np.load(source, allow_pickle=False) as data:
        required = {"sigma_centrifugal_mpa", "sigma_hydraulic_mpa"}
        missing = sorted(required - set(data.files))
        if missing:
            raise LiteraturePriorError(f"stress field is missing fields: {missing}")
        centrifugal = _finite_array(data["sigma_centrifugal_mpa"], "sigma_centrifugal_mpa")
        hydraulic = _finite_array(data["sigma_hydraulic_mpa"], "sigma_hydraulic_mpa")
        if len(centrifugal) != n_faces or len(hydraulic) != n_faces:
            raise LiteraturePriorError("stress field face count differs from particle artifact")
        return {
            "centrifugal_mpa": centrifugal,
            "hydraulic_mpa": hydraulic,
            "source_sha256": np.asarray(sha256_file(source)),
        }


def run_prior_propagation(
    hits_path: str | Path,
    *,
    stress_path: str | Path | None = None,
    output_path: str | Path | None = None,
    n_draws: int = 32,
    seed: int = 20260810,
    exposure_h: float = 1.0,
) -> dict[str, Any]:
    """Run deterministic literature-prior model comparison and propagation."""
    if isinstance(n_draws, bool) or int(n_draws) < 4:
        raise LiteraturePriorError("n_draws must be an integer >= 4")
    hits = _load_hits(hits_path)
    stress = _load_exploratory_stress(stress_path, len(hits["face_area_m2"]))
    rng = np.random.default_rng(int(seed))
    n_faces = len(hits["face_area_m2"])
    maps: dict[str, np.ndarray] = {
        model: np.zeros((int(n_draws), n_faces), dtype=np.float32)
        for model in MODEL_NAMES
    }
    stress_draws = np.zeros((int(n_draws), n_faces), dtype=np.float32) if stress else None
    density_draws: list[float] = []
    hardness_draws: list[float] = []
    concentration_draws: list[float] = []

    for index in range(int(n_draws)):
        density = float(rng.uniform(DEFAULT_DENSITY_KG_M3["low"], DEFAULT_DENSITY_KG_M3["high"]))
        hardness = float(rng.uniform(DEFAULT_HARDNESS_GPA["low"], DEFAULT_HARDNESS_GPA["high"]))
        concentration = float(
            np.exp(
                rng.uniform(
                    math.log(DEFAULT_CONCENTRATION_SCALE["low"]),
                    math.log(DEFAULT_CONCENTRATION_SCALE["high"]),
                )
            )
        )
        density_draws.append(density)
        hardness_draws.append(hardness)
        concentration_draws.append(concentration)
        for model in MODEL_NAMES:
            maps[model][index] = _model_map(
                hits,
                model,
                density_kg_m3=density,
                hardness_gpa=hardness,
                concentration_scale=concentration,
                exposure_h=exposure_h,
            )
        if stress is not None and stress_draws is not None:
            stress_draws[index] = (
                stress["hydraulic_mpa"]
                + stress["centrifugal_mpa"] * density / DEFAULT_DENSITY_KG_M3["base"]
            ).astype(np.float32)

    density_sensitivity: dict[str, dict[str, float]] = {}
    for label, density in DEFAULT_DENSITY_KG_M3.items():
        field = _model_map(
            hits,
            "mansouri",
            density_kg_m3=density,
            hardness_gpa=DEFAULT_HARDNESS_GPA["base"],
            concentration_scale=DEFAULT_CONCENTRATION_SCALE["base"],
            exposure_h=exposure_h,
        )
        density_sensitivity[label] = {
            "density_kg_m3": density,
            "max_mansouri_mm_h": float(np.max(field)),
            "mean_mansouri_mm_h": float(np.mean(field)),
        }

    model_summary = {
        model: {
            "status": "provisional",
            "units": "mm/h",
            "draw_summary": _summary(maps[model], hits["face_area_m2"]),
            "p50_map_max_mm_h": float(np.max(np.quantile(maps[model], 0.50, axis=0))),
            "p95_map_max_mm_h": float(np.max(np.quantile(maps[model], 0.95, axis=0))),
        }
        for model in MODEL_NAMES
    }
    exploratory_stress: dict[str, Any] | None = None
    if stress_draws is not None and stress is not None:
        stress_density_sensitivity: dict[str, dict[str, float]] = {}
        for label, density in DEFAULT_DENSITY_KG_M3.items():
            field = (
                stress["hydraulic_mpa"]
                + stress["centrifugal_mpa"] * density / DEFAULT_DENSITY_KG_M3["base"]
            )
            stress_density_sensitivity[label] = {
                "density_kg_m3": density,
                "p95_face_mpa": float(np.quantile(field, 0.95)),
                "mean_mpa": float(np.mean(field)),
                "max_mpa": float(np.max(field)),
            }
        exploratory_stress = {
            "status": "provisional",
            "units": "MPa",
            "real_fem_evidence": False,
            "source_kind": "analytical_parametric_stress_field",
            "draw_summary": _scalar_field_summary(stress_draws),
            "p50_map_max_mpa": float(np.max(np.quantile(stress_draws, 0.50, axis=0))),
            "p95_map_max_mpa": float(np.max(np.quantile(stress_draws, 0.95, axis=0))),
            "density_sensitivity": stress_density_sensitivity,
        }
    report: dict[str, Any] = {
        "version": VERSION,
        "status": "provisional",
        "fem_gate": "blocked",
        "real_fem_evidence": False,
        "evidence_status": "literature_prior_exploratory",
        "literature_sources": [dict(item) for item in LITERATURE_SOURCES],
        "input_bindings": {
            "particle_hits_path": str(Path(hits_path).resolve()),
            "particle_hits_sha256": str(hits["source_sha256"].item()),
            "stress_field_path": None if stress_path is None else str(Path(stress_path).resolve()),
            "stress_field_sha256": None if stress is None else str(stress["source_sha256"].item()),
            "n_hit_impacts": int(len(hits["parent_face_id"])),
            "n_faces": int(n_faces),
        },
        "prior_contract": {
            "density_kg_m3": DEFAULT_DENSITY_KG_M3,
            "hardness_gpa": DEFAULT_HARDNESS_GPA,
            "concentration_scale": DEFAULT_CONCENTRATION_SCALE,
            "exposure_h": float(exposure_h),
            "density_role": "exploratory_sensitivity_only",
            "ai5_used_for_calibration": False,
            "official_material_certificate": False,
            "official_cad_named_selections": False,
        },
        "propagation": {
            "seed": int(seed),
            "n_draws": int(n_draws),
            "density_draw_min_kg_m3": float(min(density_draws)),
            "density_draw_max_kg_m3": float(max(density_draws)),
            "hardness_draw_min_gpa": float(min(hardness_draws)),
            "hardness_draw_max_gpa": float(max(hardness_draws)),
            "concentration_draw_min": float(min(concentration_draws)),
            "concentration_draw_max": float(max(concentration_draws)),
        },
        "density_sensitivity": density_sensitivity,
        "model_summary": model_summary,
        "exploratory_stress": exploratory_stress,
        "limitations": [
            "Literature priors are not DEC heat/batch material evidence.",
            "The particle-hit artifact is reused as a fixed exploratory input; no AI5 trace is used for calibration.",
            "The optional stress field is the existing analytical/parametric field and is not a CalculiX result.",
            "No official erosion spatial truth or independent worn/baseline scan is asserted.",
        ],
    }
    report["report_sha256"] = canonical_sha256(report)

    if output_path is not None:
        target = Path(output_path).resolve()
        safe_json_dump(report, target)
        arrays: dict[str, np.ndarray] = {
            "face_centers_local_mm": hits["face_centers_local_mm"].astype(np.float32),
        }
        for model in MODEL_NAMES:
            arrays[f"{model}_p05_mm_h"] = np.quantile(maps[model], 0.05, axis=0).astype(np.float32)
            arrays[f"{model}_p50_mm_h"] = np.quantile(maps[model], 0.50, axis=0).astype(np.float32)
            arrays[f"{model}_p95_mm_h"] = np.quantile(maps[model], 0.95, axis=0).astype(np.float32)
        if stress_draws is not None:
            arrays["exploratory_stress_p05_mpa"] = np.quantile(stress_draws, 0.05, axis=0).astype(np.float32)
            arrays["exploratory_stress_p50_mpa"] = np.quantile(stress_draws, 0.50, axis=0).astype(np.float32)
            arrays["exploratory_stress_p95_mpa"] = np.quantile(stress_draws, 0.95, axis=0).astype(np.float32)
        np.savez_compressed(target.with_suffix(".npz"), **arrays)
    return report


def selftest() -> bool:
    rng = np.random.default_rng(3)
    n_faces = 16
    n_hits = 12
    root = Path(__file__).resolve().parent / "output" / "_literature_prior_selftest"
    root.mkdir(parents=True, exist_ok=True)
    hits_path = root / "hits.npz"
    velocity = rng.normal(size=(n_hits, 3)) * 50.0
    np.savez(
        hits_path,
        parent_face_id=rng.integers(0, n_faces, n_hits),
        velocity_ms=velocity,
        Vp_ms=np.full(n_hits, 20.0),
        alpha_deg=np.full(n_hits, 25.0),
        diameter_m=np.full(n_hits, 200.0e-6),
        parcel_mass_kg=np.full(n_hits, 1.0e-8),
        face_centers_local_mm=rng.normal(size=(n_faces, 3)),
        face_area_m2=np.full(n_faces, 1.0e-4),
    )
    report = run_prior_propagation(hits_path, n_draws=4, seed=9)
    assert report["status"] == "provisional"
    assert report["fem_gate"] == "blocked"
    assert set(report["model_summary"]) == set(MODEL_NAMES)
    assert report["prior_contract"]["ai5_used_for_calibration"] is False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run literature-prior erosion and exploratory FEM sensitivity")
    parser.add_argument("--hits", type=Path, required=False)
    parser.add_argument("--stress-field", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--exposure-h", type=float, default=1.0)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        selftest()
        print("step30_literature_prior_degradation selftest: PASS")
        return 0
    if args.hits is None:
        parser.error("--hits is required unless --selftest is used")
    report = run_prior_propagation(
        args.hits,
        stress_path=args.stress_field,
        output_path=args.output,
        n_draws=args.draws,
        seed=args.seed,
        exposure_h=args.exposure_h,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
