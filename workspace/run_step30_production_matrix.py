"""Declare and optionally execute a reproducible Step30 production matrix."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from step30_validation_common import canonical_sha256, write_json_immutable


def build_matrix(args: argparse.Namespace) -> dict:
    variables = {
        "collision_policy": args.collision_policy,
        "collision_backend": "cfd_topology",
        "cfd_variant": "native_si",
        "tracking_velocity": "UMean",
        "seed": args.seed,
        "n_parcels": args.parcels,
        "dt_s": args.dt,
        "max_impacts": args.max_impacts,
    }
    if args.mode == "mesh_independence":
        arms = [{"arm_id": label, "input_status": "blocked", "reason": "coarse/medium/fine native CFD bundles were not supplied"} for label in ("coarse", "medium", "fine")]
    elif args.mode == "flow_sensitivity":
        arms = [{"arm_id": str(value), "input_status": "blocked", "reason": "independent native SI flow artifacts were not supplied", "jet_speed_ms": value} for value in args.values]
    else:
        arms = [{"arm_id": str(value), "input_status": "blocked", "reason": "independent native SI field artifact was not supplied", "value": value} for value in args.values]
    descriptor = {
        "version": "step30-production-matrix-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": args.mode,
        "status": "blocked" if any(arm["input_status"] == "blocked" for arm in arms) else "planned",
        "fixed_identity": variables,
        "allowed_change": args.mode,
        "arms": arms,
    }
    descriptor["experiment_id"] = canonical_sha256({k: v for k, v in descriptor.items() if k != "created_at_utc"})
    return descriptor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("mesh_independence", "flow_sensitivity", "particle", "collision", "mean_window"), required=True)
    parser.add_argument("--values", nargs="+", type=float, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collision-policy", default="strict_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parcels", type=int, default=500)
    parser.add_argument("--dt", type=float, default=1e-5)
    parser.add_argument("--max-impacts", type=int, default=3)
    args = parser.parse_args()
    report = build_matrix(args)
    write_json_immutable(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "planned" else 3


if __name__ == "__main__":
    raise SystemExit(main())
