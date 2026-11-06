"""Compatibility wrapper for immutable Step30 i1/i3 comparison."""
from __future__ import annotations

import argparse
from pathlib import Path

from step30_si_comparison import run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--i1-run-manifest", type=Path, required=True)
    parser.add_argument("--i3-run-manifest", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-reference", type=Path)
    args = parser.parse_args()
    result = run(args.i1_run_manifest, args.i3_run_manifest, args.provenance, args.output, legacy_reference_path=args.legacy_reference)
    return 0 if result.get("accepted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
