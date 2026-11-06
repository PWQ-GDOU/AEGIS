"""Fail-closed timestep sensitivity evaluation for strict Step30 particle runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VERSION = "step30-particle-timestep-sensitivity-v1"
EXPECTED_TIMESTEPS_S = (1.0e-5, 5.0e-6, 2.5e-6)
EXPECTED_PARCELS = 500
GATES = {
    "first_parcel_jaccard_min": 0.99,
    "first_parent_face_agreement_min": 0.99,
    "hit_count_relative_max": 0.01,
    "mass_weighted_speed_relative_max": 0.01,
    "mass_weighted_angle_relative_max": 0.01,
    "target_mass_relative_max": 0.05,
    "erosion_p99_relative_max": 0.05,
}


def _relative(left: float, right: float) -> float:
    return float(abs(left - right) / max(abs(left), abs(right), 1e-30))


def _run_metrics(npz_path: Path, summary_path: Path) -> tuple[dict, dict, dict[int, int]]:
    data = np.load(npz_path, allow_pickle=False)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required = {
        "parcel_id", "impact_index", "parent_face_id", "parcel_mass_kg", "Vp_ms",
        "alpha_deg", "face_target_mass_kg", "dhdt_face_mm_h", "collision_policy",
    }
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"particle sensitivity artifact is missing arrays: {missing}")
    policy = str(data["collision_policy"].item())
    tracking = summary.get("tracking", {})
    parameters = summary.get("parameters", {})
    cfd = summary.get("cfd", {})
    identity = summary.get("artifact_identity")
    semantics = cfd.get("solver_semantics")
    if (
        summary.get("version") != "step30-cfd-driven-v2"
        or not isinstance(identity, dict)
        or not identity.get("artifact_set_id")
        or policy != "strict_v2"
        or parameters.get("collision_policy") != "strict_v2"
        or parameters.get("collision_backend") != "cfd_topology"
        or parameters.get("cfd_variant") != "native_si"
        or parameters.get("n_parcels") != EXPECTED_PARCELS
        or parameters.get("n_injected") != EXPECTED_PARCELS
        or cfd.get("variant") != "native_si"
        or semantics is None
        or semantics.get("application") != "pimpleFoam"
        or semantics.get("mode") != "urans"
        or semantics.get("field_statistic") != "time_mean"
        or semantics.get("tracking_velocity_object") != "UMean"
        or not isinstance(semantics.get("averaging_window"), dict)
        or summary.get("validation", {}).get("registration_status") != "passed"
        or float(cfd.get("speed_stats_ms", {}).get("max", np.inf)) >= 500.0
        or tracking.get("fallback_hits") != 0
        or tracking.get("physical_strict_hit_ratio") != 1.0
        or tracking.get("collision_event_limit_count", 0) != 0
    ):
        raise ValueError("sensitivity evaluation requires accepted strict_v2 URANS time-mean runs")

    impact_length = len(data["parcel_id"])
    for name in ("impact_index", "parent_face_id", "parcel_mass_kg", "Vp_ms", "alpha_deg"):
        if len(data[name]) != impact_length:
            raise ValueError("particle sensitivity impact arrays have inconsistent lengths")
    if tracking.get("n_impacts") != impact_length:
        raise ValueError("particle sensitivity summary impact count differs from NPZ")
    first_mask = data["impact_index"] == 0
    if len(np.unique(data["parcel_id"][first_mask])) != int(first_mask.sum()):
        raise ValueError("particle sensitivity artifact has duplicate first impacts")
    first_faces = {
        int(parcel): int(face)
        for parcel, face in zip(data["parcel_id"][first_mask], data["parent_face_id"][first_mask])
    }
    weights = data["parcel_mass_kg"].astype(np.float64)
    weight_sum = max(float(weights.sum()), 1e-30)
    exposed_erosion = data["dhdt_face_mm_h"][data["dhdt_face_mm_h"] > 0.0]
    dt_s = float(summary["parameters"]["dt_s"])
    strict_max_substep_s = float(summary["parameters"].get("strict_max_substep_s", -1.0))
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("sensitivity timestep must be finite and positive")
    if not np.isclose(strict_max_substep_s, dt_s, rtol=1e-12, atol=0.0):
        raise ValueError("sensitivity runs must vary the effective strict timestep")
    metrics = {
        "dt_s": dt_s,
        "strict_max_substep_s": strict_max_substep_s,
        "impacts": int(len(data["parcel_id"])),
        "first_hits": int(len(first_faces)),
        "hit_parcels": int(len(np.unique(data["parcel_id"]))),
        "target_mass_kg": float(data["face_target_mass_kg"].sum()),
        "erosion_p99_mm_h": float(np.percentile(exposed_erosion, 99)) if len(exposed_erosion) else 0.0,
        "impact_mass_kg": float(weights.sum()),
        "Vp_mass_weighted": float(np.sum(data["Vp_ms"] * weights) / weight_sum),
        "angle_mass_weighted": float(np.sum(data["alpha_deg"] * weights) / weight_sum),
        "fallback_hits": int(tracking["fallback_hits"]),
        "physical_strict_hit_ratio": float(tracking["physical_strict_hit_ratio"]),
    }
    if (
        metrics["impacts"] <= 0
        or metrics["first_hits"] <= 0
        or metrics["target_mass_kg"] <= 0.0
        or metrics["erosion_p99_mm_h"] <= 0.0
    ):
        raise ValueError("sensitivity runs require nonzero physical impacts and erosion")
    parameters = dict(summary["parameters"])
    parameters.pop("dt_s", None)
    parameters.pop("strict_max_substep_s", None)
    parameters.pop("output_tag", None)
    identity = {
        "artifact_set_id": (summary.get("artifact_identity") or {}).get("artifact_set_id"),
        "mesh_signature": summary.get("geometry", {}).get("mesh_signature"),
        "field_time": str(summary.get("cfd", {}).get("field_time")),
        "solver_semantics": summary.get("cfd", {}).get("solver_semantics"),
        "parameters_except_dt": parameters,
    }
    return metrics, identity, first_faces


def _pair(
    left_metrics: dict,
    right_metrics: dict,
    left_faces: dict[int, int],
    right_faces: dict[int, int],
) -> dict:
    left_ids = set(left_faces)
    right_ids = set(right_faces)
    union = left_ids | right_ids
    common = left_ids & right_ids
    jaccard = float(len(common) / len(union)) if union else 1.0
    face_agreement = (
        float(sum(left_faces[item] == right_faces[item] for item in common) / len(common))
        if common else 1.0
    )
    result = {
        "first_parcel_jaccard": jaccard,
        "first_parent_face_agreement": face_agreement,
        "hit_count_relative": _relative(left_metrics["impacts"], right_metrics["impacts"]),
        "mass_weighted_speed_relative": _relative(
            left_metrics["Vp_mass_weighted"], right_metrics["Vp_mass_weighted"]
        ),
        "mass_weighted_angle_relative": _relative(
            left_metrics["angle_mass_weighted"], right_metrics["angle_mass_weighted"]
        ),
        "target_mass_relative": _relative(
            left_metrics["target_mass_kg"], right_metrics["target_mass_kg"]
        ),
        "erosion_p99_relative": _relative(
            left_metrics["erosion_p99_mm_h"], right_metrics["erosion_p99_mm_h"]
        ),
    }
    result["gates"] = {
        "first_parcel_jaccard": result["first_parcel_jaccard"] >= GATES["first_parcel_jaccard_min"],
        "first_parent_face_agreement": result["first_parent_face_agreement"] >= GATES["first_parent_face_agreement_min"],
        "hit_count_relative": result["hit_count_relative"] < GATES["hit_count_relative_max"],
        "mass_weighted_speed_relative": result["mass_weighted_speed_relative"] < GATES["mass_weighted_speed_relative_max"],
        "mass_weighted_angle_relative": result["mass_weighted_angle_relative"] < GATES["mass_weighted_angle_relative_max"],
        "target_mass_relative": result["target_mass_relative"] < GATES["target_mass_relative_max"],
        "erosion_p99_relative": result["erosion_p99_relative"] < GATES["erosion_p99_relative_max"],
    }
    result["passed"] = all(result["gates"].values())
    return result


def run(runs: list[tuple[str, Path, Path]], output: Path) -> dict:
    if len(runs) != len(EXPECTED_TIMESTEPS_S):
        raise ValueError("production sensitivity requires exactly three timestep runs")
    loaded = {}
    identities = {}
    first_faces = {}
    for label, npz_path, summary_path in runs:
        if label in loaded:
            raise ValueError(f"duplicate sensitivity run label: {label}")
        loaded[label], identities[label], first_faces[label] = _run_metrics(npz_path, summary_path)
    reference_identity = next(iter(identities.values()))
    if any(identity != reference_identity for identity in identities.values()):
        raise ValueError("sensitivity runs differ beyond timestep/output identity")
    ordered = sorted(loaded, key=lambda label: loaded[label]["dt_s"], reverse=True)
    actual_timesteps = tuple(loaded[label]["dt_s"] for label in ordered)
    if not np.allclose(actual_timesteps, EXPECTED_TIMESTEPS_S, rtol=1e-12, atol=0.0):
        raise ValueError("sensitivity runs do not use the required effective timesteps")
    pairs = {}
    for left, right in zip(ordered, ordered[1:]):
        pairs[f"{left}__{right}"] = _pair(
            loaded[left], loaded[right], first_faces[left], first_faces[right]
        )
    result = {
        "version": VERSION,
        "status": "passed" if all(pair["passed"] for pair in pairs.values()) else "failed",
        "gates": GATES,
        "shared_identity": reference_identity,
        "runs": loaded,
        "pairs": pairs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _parse_run(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 1)
    paths = parts[1].split(",") if len(parts) == 2 else []
    if not parts[0] or len(paths) != 2:
        raise argparse.ArgumentTypeError("run must be LABEL=NPZ,SUMMARY_JSON")
    return parts[0], Path(paths[0]), Path(paths[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate strict Step30 timestep sensitivity")
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.run, args.output)
    if result["status"] != "passed":
        raise SystemExit(2)
