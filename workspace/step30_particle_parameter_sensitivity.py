"""Evaluate declared particle-property sensitivity arms without mixing identities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _metrics(npz_path: Path, summary_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=False)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tracking = summary.get("tracking", {})
    params = summary.get("parameters", {})
    mass = data["parcel_mass_kg"].astype(float)
    weight = max(float(mass.sum()), 1e-30)
    return {
        "artifact_set_id": str(summary.get("artifact_identity", {}).get("artifact_set_id", "")),
        "mesh_signature": str(summary.get("geometry", {}).get("mesh_signature", "")),
        "collision_policy": params.get("collision_policy"),
        "tracking_velocity": summary.get("cfd", {}).get("solver_semantics", {}).get("tracking_velocity_object"),
        "n_parcels": params.get("n_parcels"),
        "seed": params.get("seed"),
        "impacts": int(len(data["parcel_id"])),
        "first_hits": int(np.sum(data["impact_index"] == 0)),
        "hit_rate_pct": float(tracking.get("hit_rate_parcels_pct", 0.0)),
        "mass_weighted_speed_ms": float(np.sum(data["Vp_ms"] * mass) / weight),
        "mass_weighted_angle_deg": float(np.sum(data["alpha_deg"] * mass) / weight),
        "target_mass_kg": float(data["face_target_mass_kg"].sum()),
        "erosion_p99_mm_h": float(np.percentile(data["dhdt_face_mm_h"][data["dhdt_face_mm_h"] > 0], 99)) if np.any(data["dhdt_face_mm_h"] > 0) else 0.0,
        "fallback_hits": int(tracking.get("fallback_hits", -1)),
    }


def run(arms: list[tuple[str, Path, Path]], output: Path) -> dict:
    rows = [{"arm_id": label, **_metrics(npz, summary)} for label, npz, summary in arms]
    errors = []
    if len(rows) < 2:
        errors.append("at least two independent particle-property arms are required")
    for row in rows:
        if not row["artifact_set_id"] or row["collision_policy"] != "strict_v2" or row["fallback_hits"] != 0:
            errors.append(f"arm {row['arm_id']} is not an accepted strict_v2 run")
    fixed = ("artifact_set_id", "mesh_signature", "collision_policy", "tracking_velocity", "n_parcels", "seed")
    for key in fixed:
        if len({row[key] for row in rows}) != 1:
            errors.append(f"arms differ in fixed identity field {key}")
    baseline = rows[0]
    for row in rows:
        row["relative_to_baseline"] = {key: float(abs(row[key] - baseline[key]) / max(abs(row[key]), abs(baseline[key]), 1e-30)) for key in ("hit_rate_pct", "mass_weighted_speed_ms", "mass_weighted_angle_deg", "target_mass_kg", "erosion_p99_mm_h")}
    result = {"version": "step30-particle-parameter-sensitivity-v1", "status": "passed" if not errors else "failed", "variable": "particle_properties", "arms": rows, "errors": errors, "interpretation": "Only arms with identical CFD/topology/seed/parcel identity are comparable."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _parse(value: str) -> tuple[str, Path, Path]:
    label, paths = value.split("=", 1)
    npz, summary = paths.split(",", 1)
    return label, Path(npz), Path(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", type=_parse, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.arm, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 2)
