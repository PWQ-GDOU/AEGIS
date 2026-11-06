"""Machine-readable CFD convergence/time-mean evidence adapter."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def inspect_solver_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    max_co = [float(v) for v in re.findall(r"Courant Number mean:\s*[^\s]+\s+max:\s*([^\s]+)", text)]
    residual_lines = len(re.findall(r"Solving for (?:U|p|k|omega)", text))
    fatal = bool(re.search(r"FOAM FATAL|FOAM abort|\b(?:nan|inf)\b", text, re.I))
    bounding = {
        "k": text.count("bounding k,"),
        "omega": text.count("bounding omega,"),
    }
    return {
        "path": str(path),
        "status": "passed" if max_co and not fatal and not any(bounding.values()) else "failed",
        "metrics": {"max_courant_number": max(max_co) if max_co else None, "courant_samples": len(max_co), "residual_records": residual_lines, "fatal": fatal, "bounding": bounding},
        "gates": {"courant_below_0p5": bool(max_co and max(max_co) <= 0.5), "no_fatal": not fatal, "no_turbulence_bounding": not any(bounding.values())},
    }


def validate_report(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    gates = data.get("gates", {})
    return {"path": str(path), "version": data.get("version"), "status": "passed" if data.get("accepted") and all(gates.values()) else "failed", "accepted": bool(data.get("accepted")), "gates": gates, "metrics": data.get("metrics", {})}


if __name__ == "__main__":
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
    raise SystemExit(0 if result["status"] == "passed" else 2)
