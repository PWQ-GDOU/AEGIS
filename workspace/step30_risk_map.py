"""PNG risk-map rendering for Step30 candidate mechanism results.

Plots are visual summaries of candidate/model risk.  They are explicitly
annotated as synthetic, measured, or blocked and never claim measured damage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLAIM_SCOPE = "candidate mechanism only; not measured damage"
VERSION = "step30-risk-map-v1"


def _status(result: Mapping[str, Any]) -> str:
    return str((result.get("provenance") or {}).get("status", "synthetic"))


def _annotation(result: Mapping[str, Any]) -> str:
    status = _status(result)
    return f"status={status} | candidate labels | no measured-damage claim"


def _records(result: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = result.get(key, [])
    return [row for row in value if isinstance(row, Mapping)]


def _values(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row.get(key, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        values.append(value if np.isfinite(value) else 0.0)
    return np.asarray(values, dtype=np.float64)


def _labels(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    labels = []
    for row in rows:
        value = row.get("candidate_labels", [])
        if isinstance(value, str):
            labels.extend(value.split("|"))
        elif isinstance(value, (list, tuple)):
            labels.extend(str(item) for item in value)
    return sorted(set(labels))


def _empty_axis(axis: Any, title: str) -> None:
    axis.text(0.5, 0.5, "No candidate records", ha="center", va="center", transform=axis.transAxes)
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])


def render_risk_maps(result: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Render face and spatial PNG maps from an in-memory classifier result."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    face_rows = _records(result, "face_records")
    spatial_rows = _records(result, "spatial_records")
    hit_rows = _records(result, "hit_records")
    status = _status(result)
    annotation = _annotation(result)
    pngs = {
        "face_png": root / "face_risk.png",
        "spatial_png": root / "spatial_risk.png",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle(f"Step30 candidate erosion risk maps\n{annotation}", fontsize=12)
    if face_rows:
        x = _values(face_rows, "x")
        y = _values(face_rows, "y")
        risk = _values(face_rows, "risk_score")
        valid = np.asarray([row.get("x") is not None and row.get("y") is not None for row in face_rows], dtype=bool)
        if valid.any():
            scatter = axes[0].scatter(x[valid], y[valid], c=risk[valid], s=4, cmap="magma", vmin=0, vmax=100, linewidths=0)
            fig.colorbar(scatter, ax=axes[0], label="candidate risk score (0-100)")
            axes[0].set_xlabel("face x [input coordinates]")
            axes[0].set_ylabel("face y [input coordinates]")
            axes[0].set_aspect("equal", adjustable="datalim")
        else:
            _empty_axis(axes[0], "Face risk (no coordinates)")
        axes[0].set_title(f"Face candidate risk ({len(face_rows):,} faces)")
    else:
        _empty_axis(axes[0], "Face risk")
    axes[0].text(0.01, 0.01, f"{status} | candidate | not measured damage", transform=axes[0].transAxes, fontsize=8)

    if spatial_rows:
        x = _values(spatial_rows, "x_center")
        y = _values(spatial_rows, "y_center")
        risk = _values(spatial_rows, "risk_score")
        scatter = axes[1].scatter(x, y, c=risk, s=55, cmap="magma", vmin=0, vmax=100, marker="s")
        fig.colorbar(scatter, ax=axes[1], label="candidate risk score (0-100)")
        axes[1].set_xlabel("spatial x [input coordinates]")
        axes[1].set_ylabel("spatial y [input coordinates]")
        axes[1].set_aspect("equal", adjustable="datalim")
        axes[1].set_title(f"Spatial candidate risk ({len(spatial_rows):,} occupied cells)")
    elif hit_rows:
        x = _values(hit_rows, "x")
        y = _values(hit_rows, "y")
        risk = _values(hit_rows, "risk_score")
        scatter = axes[1].scatter(x, y, c=risk, s=20, cmap="magma", vmin=0, vmax=100)
        fig.colorbar(scatter, ax=axes[1], label="candidate risk score (0-100)")
        axes[1].set_title("Spatial candidate risk (hit points)")
        axes[1].set_xlabel("x [input coordinates]")
        axes[1].set_ylabel("y [input coordinates]")
    else:
        _empty_axis(axes[1], "Spatial risk")
    axes[1].text(0.01, 0.01, f"{status} | candidate | not measured damage", transform=axes[1].transAxes, fontsize=8)

    for axis in axes:
        axis.grid(True, alpha=0.22)
    fig.savefig(pngs["face_png"], dpi=150)
    plt.close(fig)

    # Keep a separate spatial artifact for workflows that publish one map per
    # aggregation level, while retaining the same explicit disclaimer.
    fig, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    fig.suptitle(f"Step30 spatial candidate risk\n{annotation}", fontsize=12)
    if spatial_rows:
        x = _values(spatial_rows, "x_center")
        y = _values(spatial_rows, "y_center")
        risk = _values(spatial_rows, "risk_score")
        scatter = axis.scatter(x, y, c=risk, s=65, cmap="magma", vmin=0, vmax=100, marker="s")
        fig.colorbar(scatter, ax=axis, label="candidate risk score (0-100)")
        axis.set_xlabel("spatial x [input coordinates]")
        axis.set_ylabel("spatial y [input coordinates]")
        axis.set_aspect("equal", adjustable="datalim")
    elif hit_rows:
        x = _values(hit_rows, "x")
        y = _values(hit_rows, "y")
        risk = _values(hit_rows, "risk_score")
        scatter = axis.scatter(x, y, c=risk, s=25, cmap="magma", vmin=0, vmax=100)
        fig.colorbar(scatter, ax=axis, label="candidate risk score (0-100)")
        axis.set_xlabel("x [input coordinates]")
        axis.set_ylabel("y [input coordinates]")
    else:
        _empty_axis(axis, "Spatial risk")
    axis.text(0.01, 0.01, f"{status} | candidate | not measured damage", transform=axis.transAxes, fontsize=8)
    axis.grid(True, alpha=0.22)
    fig.savefig(pngs["spatial_png"], dpi=150)
    plt.close(fig)
    return {key: str(path) for key, path in pngs.items()}


def render_from_json(result_path: str | Path, output_dir: str | Path) -> dict[str, str]:
    document = json.loads(Path(result_path).read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("classification JSON must be an object")
    return render_risk_maps(document, output_dir)


def selftest() -> bool:
    import tempfile
    with tempfile.TemporaryDirectory(prefix="step30_risk_map_selftest_") as directory:
        result = {
            "version": VERSION,
            "provenance": {"status": "synthetic", "candidate": True, "claim_scope": CLAIM_SCOPE},
            "hit_records": [{"x": 0, "y": 0, "risk_score": 20, "candidate_labels": ["sliding"]}],
            "face_records": [{"x": 0, "y": 0, "risk_score": 20, "candidate_labels": ["sliding"]}],
            "spatial_records": [{"x_center": 0, "y_center": 0, "risk_score": 20, "candidate_labels": ["sliding"]}],
        }
        paths = render_risk_maps(result, directory)
        assert all(Path(path).is_file() and Path(path).stat().st_size > 100 for path in paths.values())
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render Step30 candidate risk PNG maps")
    parser.add_argument("--result", help="mechanism_classification.json")
    parser.add_argument("--output-dir", help="PNG output directory")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        selftest()
        print("step30_risk_map selftest: PASS")
        return 0
    if not args.result or not args.output_dir:
        parser.error("--result and --output-dir are required unless --selftest is used")
    print(json.dumps(render_from_json(args.result, args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
