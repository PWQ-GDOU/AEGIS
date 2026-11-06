"""Fail-closed evaluator for URANS averaging-window arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def run(arms: list[dict], output: Path) -> dict:
    errors = []
    if len(arms) < 2:
        errors.append("at least two independent averaging windows are required")
    identities = [arm.get("artifact_set_id") for arm in arms]
    if any(not value for value in identities) or len(set(identities)) != 1:
        errors.append("mean-window arms must share artifact identity")
    for arm in arms:
        if arm.get("tracking_velocity") != "UMean":
            errors.append(f"arm {arm.get('label')} does not use UMean")
    result = {"version": "step30-mean-window-sensitivity-v1", "status": "passed" if not errors else "blocked", "arms": arms, "errors": errors, "official_accuracy_gate": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", default=[], help="JSON object per arm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run([json.loads(value) for value in args.arm], args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 3)
