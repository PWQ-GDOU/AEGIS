"""Sweep the registered v5 aggregate scales on the DCU only.

For every candidate aggregate size, the old five-run discovery set is
re-evaluated at that exact partition, then the non-overlapping confirmation
set is evaluated against the matching discovery identity. No thresholds are
changed and no run is modified.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


BASE = Path("output")
DISCOVERY = BASE / "step30_particle_convergence_5k_10k_20k_topology_aggregate_v5_discovery_r2.json"
STATUS = BASE / "step30_v5_confirmation_parallel_v1/run_status.json"
TOPOLOGY = BASE / "step30_acceptance_0.8684190921/bucket_topology_v3_0.8684190921.npz"
CONF_FIELDS = BASE / "step30_particle_scale_sweep_confirmation_fields_20260830_v1"


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def discovery_runs(report: dict) -> list[str]:
    rows = report.get("artifact_inputs")
    if not isinstance(rows, list):
        raise RuntimeError("discovery report has no artifact_inputs")
    result = [
        normalize_remote_path(row["run_directory"])
        for row in rows
        if isinstance(row, dict)
        and row.get("run_directory")
        and int(row.get("n_parcels", -1)) == 20_000
    ]
    if len(result) < 5:
        raise RuntimeError("discovery report has fewer than five run directories")
    return result


def confirmation_runs(status: dict) -> list[str]:
    rows = status.get("seeds")
    if not isinstance(rows, dict):
        raise RuntimeError("confirmation status has no seeds")
    result = [
        normalize_remote_path(row["run_directory"])
        for _, row in sorted(rows.items(), key=lambda item: int(item[0]))
        if isinstance(row, dict) and row.get("status") == "completed"
    ]
    if len(result) < 5:
        raise RuntimeError("confirmation status has fewer than five completed runs")
    return result


def normalize_remote_path(value: str) -> str:
    """Map a historical H: path in a copied report to the DCU workspace."""
    text = str(value).replace("\\", "/")
    marker = "/workspace/"
    if marker in text:
        suffix = text.split(marker, 1)[1]
        return "remote_artifacts/PWQ/step30_cpu_v1/workspace/" + suffix
    if re.match(r"^[A-Za-z]:/", text):
        raise RuntimeError(f"cannot map non-workspace Windows path: {value}")
    return text


def run_evaluator(runs: list[str], output: Path, scale: int, *, role: str, discovery_report: Path | None = None) -> None:
    command = [
        sys.executable,
        "step30_particle_convergence_v5.py",
        "--analysis-role",
        role,
        "--topology",
        str(TOPOLOGY),
        "--output",
        str(output),
        "--max-parent-faces-per-aggregate",
        str(scale),
    ]
    if discovery_report is not None:
        command.extend(["--discovery-report", str(discovery_report)])
    for run in runs:
        command.extend(["--run", run])
    completed = subprocess.run(command, check=False)
    if completed.returncode not in (0, 3):
        raise RuntimeError(f"evaluator invocation failed with code {completed.returncode}: {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", nargs="+", type=int, default=[4096, 8192, 16384])
    parser.add_argument("--discovery", type=Path, default=DISCOVERY)
    parser.add_argument("--status", type=Path, default=STATUS)
    parser.add_argument("--output-directory", type=Path, default=BASE / "step30_particle_scale_sweep_20260830_v1")
    args = parser.parse_args()
    old_discovery = load(args.discovery)
    status = load(args.status)
    d_runs = discovery_runs(old_discovery)
    c_runs = confirmation_runs(status)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    summary = {"version": "step30-particle-confirmation-scale-sweep-v1", "scales": {}}
    for scale in args.scales:
        discovery_out = args.output_directory / f"discovery_{scale}.json"
        confirmation_out = args.output_directory / f"confirmation_{scale}.json"
        run_evaluator(d_runs, discovery_out, scale, role="discovery")
        run_evaluator(c_runs, confirmation_out, scale, role="confirmation", discovery_report=discovery_out)
        d = load(discovery_out)
        c = load(confirmation_out)
        c_metrics = c.get("scales", {}).get("20000", {}).get("estimators", {}).get("topology_aggregate", {})
        summary["scales"][str(scale)] = {
            "discovery_status": d.get("status"),
            "confirmation_status": c.get("status"),
            "confirmation_metrics": {
                "area_weighted_spearman_median": c_metrics.get("area_weighted_spearman_median"),
                "area_weighted_spearman_p10": c_metrics.get("area_weighted_spearman_p10"),
                "mass_hotspot_iou_median": c_metrics.get("mass_hotspot_iou_median"),
                "spatial_js_divergence_max": c_metrics.get("spatial_js_divergence_max"),
                "passed": c_metrics.get("passed"),
                "gates": c_metrics.get("gates"),
            },
            "discovery_report": str(discovery_out),
            "confirmation_report": str(confirmation_out),
        }
    summary_path = args.output_directory / "sweep_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
