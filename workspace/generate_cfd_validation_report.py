"""Combine staged CFD validation evidence into JSON, CSV, and PNG artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def run(inputs: list[Path], output_dir: Path) -> dict:
    reports = []
    for path in inputs:
        value = json.loads(path.read_text(encoding="utf-8"))
        reports.append({"file": str(path), "status": value.get("status", "passed" if value.get("accepted") else "failed"), "version": value.get("version"), "accepted": bool(value.get("accepted", value.get("status") == "passed")), "gates": value.get("gates", {})})
    required = [item for item in reports if item["version"] in {"urans-time-mean-validation-v2", "step30-artifact-set-validation-v1", "step30-mean-field-validation-v1"}]
    status = "passed" if required and all(item["accepted"] or item["status"] == "passed" for item in required) else "failed"
    result = {"version": "step30-cfd-validation-report-v1", "status": status, "reports": reports, "interpretation": "URANS time-mean evidence is kept separate from steady fixed-point convergence."}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cfd_validation_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "cfd_validation_gates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["source", "gate", "passed"])
        for report in reports:
            for gate, passed in report["gates"].items(): writer.writerow([report["file"], gate, passed])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = [(Path(report["file"]).name, sum(bool(v) for v in report["gates"].values()), len(report["gates"])) for report in reports]
        fig, axis = plt.subplots(figsize=(max(7, len(rows) * 1.6), 4.5), constrained_layout=True)
        axis.bar([row[0] for row in rows], [row[1] for row in rows], color="#26734d")
        axis.scatter([row[0] for row in rows], [row[2] for row in rows], color="#a33", marker="_")
        axis.set_ylabel("Passed gates (red marker = total)"); axis.tick_params(axis="x", rotation=25)
        axis.set_title("Step30 CFD validation evidence")
        fig.savefig(output_dir / "cfd_validation.png", dpi=160); plt.close(fig)
    except ImportError:
        pass
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 2)
