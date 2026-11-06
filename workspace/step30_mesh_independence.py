"""Fail-closed mesh-independence evaluator for Step30 arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def run(arms: list[dict], output: Path) -> dict:
    errors = []
    if len(arms) < 3:
        errors.append("coarse, medium and fine CFD artifacts are required")
    labels = {str(arm.get("label")) for arm in arms}
    if not {"coarse", "medium", "fine"}.issubset(labels):
        errors.append("required labels coarse/medium/fine are missing")
    result = {"version": "step30-mesh-independence-v1", "status": "passed" if not errors else "blocked", "arms": arms, "errors": errors, "official_accuracy_gate": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", default=[], help="JSON object per arm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = [json.loads(value) for value in args.arm]
    result = run(arms, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 3)
