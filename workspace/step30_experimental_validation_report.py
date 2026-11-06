"""Aggregate B-class material, bucket, and field experiment outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def run(inputs: list[Path], output: Path) -> dict:
    reports = []
    for path in inputs:
        summary = next(path.glob("*-summary.json"), None) if path.is_dir() else path
        if summary is None or not summary.exists():
            reports.append({"path": str(path), "status": "blocked", "reason": "summary missing"}); continue
        data = json.loads(summary.read_text(encoding="utf-8"))
        reports.append({"path": str(summary), "status": data.get("status", "blocked"), "experiment": data.get("experiment", data.get("version")), "mode": data.get("mode"), "comparison": data.get("comparison", data.get("prediction", {}).get("comparison"))})
    measured = [item for item in reports if item["status"] == "measured"]
    official = bool(measured) and all(bool((item.get("comparison") or {}).get("official_accuracy_gate", False)) for item in measured)
    result = {"version": "step30-experimental-validation-report-v1", "status": "passed" if reports and all(item["status"] in {"synthetic", "measured"} for item in reports) else "blocked", "reports": reports, "official_accuracy_gate": official, "note": "Synthetic outputs validate the software path only; measured accuracy requires independent measured data and all <5% gates."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 3)
