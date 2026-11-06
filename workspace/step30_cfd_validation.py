"""Compatibility entry point for Step30 CFD validation evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate_cfd_convergence import inspect_solver_log, validate_report

__all__ = ["inspect_solver_log", "validate_report"]


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--solver-log", type=Path)
    group.add_argument("--validation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect_solver_log(args.solver_log) if args.solver_log else validate_report(args.validation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
