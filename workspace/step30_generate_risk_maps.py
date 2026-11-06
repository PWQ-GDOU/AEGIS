"""Generate Step30 candidate mechanism risk CSV/JSON/PNG artifacts.

The command consumes the immutable particle hits and paired summary, runs the
candidate classifier, and renders face/spatial maps.  It does not modify the
inputs or claim that a modelled hit is measured material damage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from step30_erosion_mechanism_classifier import classify_and_write
from step30_risk_map import render_risk_maps


def generate_risk_maps(
    hits_path: str | Path,
    summary_path: str | Path,
    output_dir: str | Path,
    *,
    spatial_bins: int = 24,
) -> dict[str, Any]:
    """Classify inputs, write tables, and render PNG maps."""
    result = classify_and_write(hits_path, summary_path, output_dir, spatial_bins=spatial_bins)
    result["outputs"].update(render_risk_maps(result, output_dir))
    return result


def selftest() -> bool:
    """Run both classifier and renderer selftests through their public APIs."""
    from step30_erosion_mechanism_classifier import selftest as classifier_selftest
    from step30_risk_map import selftest as risk_map_selftest

    assert classifier_selftest()
    assert risk_map_selftest()
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Step30 candidate mechanism risk maps")
    parser.add_argument("--hits", help="immutable particle_hits.npz")
    parser.add_argument("--summary", help="paired immutable summary.json")
    parser.add_argument("--output-dir", help="CSV, JSON, and PNG output directory")
    parser.add_argument("--spatial-bins", type=int, default=24)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        selftest()
        print("step30_generate_risk_maps selftest: PASS")
        return 0
    if not (args.hits and args.summary and args.output_dir):
        parser.error("--hits, --summary, and --output-dir are required unless --selftest is used")
    result = generate_risk_maps(args.hits, args.summary, args.output_dir, spatial_bins=args.spatial_bins)
    print(json.dumps({
        "status": result["provenance"]["status"],
        "candidate": result["provenance"]["candidate"],
        "measured_damage_claim": result["provenance"]["measured_damage_claim"],
        "outputs": result["outputs"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
