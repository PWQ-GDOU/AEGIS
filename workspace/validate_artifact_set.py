"""Validate one artifact set and all selected immutable particle runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate_step30_manifest import validate_artifact_manifest, validate_particle_run


def validate(root: Path, run_dirs: list[Path] | None = None) -> dict:
    root = root.resolve()
    artifact = validate_artifact_manifest(root / "artifact_set_manifest.json")
    if run_dirs is None:
        run_dirs = sorted((root / "strict_v2").glob("*/"))
    runs = [validate_particle_run(path) for path in run_dirs]
    errors = list(artifact["errors"])
    errors.extend(error for run in runs for error in run["errors"])
    return {
        "version": "step30-artifact-set-validation-v1",
        "status": "passed" if not errors else "failed",
        "artifact_set_id": artifact.get("artifact_set_id"),
        "artifact": artifact,
        "runs": runs,
        "errors": errors,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", action="append", type=Path, default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.root, args.run)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    raise SystemExit(0 if report["status"] == "passed" else 2)
