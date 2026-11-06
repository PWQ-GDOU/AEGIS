"""Auditable material-trial report and the sole official accuracy gate.

A report is eligible for ``official_accuracy_gate`` only when status is
measured, calibration/test groups are independent, registration and
measurement quality pass, a pre-registered metric list is present, and every
pre-registered test metric is below 5 percent. Synthetic status is always
ineligible, even when its numbers are perfect.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from step30_material_model_calibration import CALIBRATION_VERSION, calibrate_material_model
from step30_material_morphology_metrics import MORPHOLOGY_VERSION
from step30_material_trial_ingest import INGEST_VERSION, ingest_material_trials
from step30_validation_common import ValidationError, canonical_sha256, jsonable, require_status, safe_json_dump, sha256_file


REPORT_VERSION = "step30-material-trial-report-v1"
DEFAULT_PREREGISTERED_METRICS = ("relative_volume_error", "relative_mass_error", "relative_depth_error")


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"JSON artifact must be an object: {path}")
    return value


def _artifact(value: Mapping[str, Any] | str | Path, version: str) -> dict[str, Any]:
    data = dict(value) if isinstance(value, Mapping) else _read_json(value)
    if data.get("version") != version:
        raise ValidationError(f"expected artifact version {version}")
    return data


def _record_lookup(data: Mapping[str, Any]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    rows = data.get("records")
    if not isinstance(rows, list):
        raise ValidationError("ingest artifact has no records")
    lookup: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["campaign_id"]), str(row["run_id"]), str(row["specimen_id"]))
        if key in lookup:
            raise ValidationError("duplicate specimen identity in report input")
        lookup[key] = row
    return lookup


def _prediction_lookup(calibration: Mapping[str, Any]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    model = calibration.get("predicted_model", {})
    rows = model.get("predictions")
    if not isinstance(rows, list):
        raise ValidationError("calibration artifact has no predictions")
    lookup = {}
    for row in rows:
        key = (str(row["campaign_id"]), str(row["run_id"]), str(row["specimen_id"]))
        lookup[key] = row
    return lookup


def _relative_error(predicted: float, observed: float) -> float:
    return abs(float(predicted) - float(observed)) / max(abs(float(observed)), 1e-30)


def _metric_rows(ingest: Mapping[str, Any], calibration: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    records = _record_lookup(ingest)
    predictions = _prediction_lookup(calibration)
    model_name = str(calibration["model_name"])
    result_rows: list[dict[str, Any]] = []
    errors: dict[str, list[float]] = {name: [] for name in DEFAULT_PREREGISTERED_METRICS}
    for key, row in records.items():
        if row["split"] != "test":
            continue
        if key not in predictions:
            raise ValidationError(f"test prediction missing for {key}")
        prediction = predictions[key]
        predicted_unit = prediction["calibrated_prediction_unit"]
        if model_name == "mansouri":
            # Mansouri is kg/kg. Convert predicted eroded mass to volume:
            # kg/kg * kg / (kg/m3) = m3, then m3 -> mm3.
            predicted_volume = float(prediction["calibrated_prediction"]) * float(row["sediment_mass_kg"]) * 1.0e9 / float(row["material_density_kg_m3"])
            predicted_mass_g = float(prediction["calibrated_prediction"]) * float(row["sediment_mass_kg"]) * 1000.0
        else:
            predicted_volume = float(prediction["calibrated_prediction"]) * float(row["sediment_mass_kg"])
            # density * volume gives mg when density is converted by 1e-3;
            # divide by 1000 again to report grams.
            predicted_mass_g = predicted_volume * float(row["material_density_kg_m3"]) * 1e-6
        predicted_depth = predicted_volume / float(row["exposed_area_mm2"])
        volume_error = _relative_error(predicted_volume, row["wear_volume_mm3"])
        mass_error = _relative_error(predicted_mass_g, row["mass_loss_g"])
        depth_error = _relative_error(predicted_depth, row["wear_depth_mm"])
        errors["relative_volume_error"].append(volume_error)
        errors["relative_mass_error"].append(mass_error)
        errors["relative_depth_error"].append(depth_error)
        result_rows.append({
            "campaign_id": row["campaign_id"], "run_id": row["run_id"], "specimen_id": row["specimen_id"], "group_id": row["group_id"],
            "model": model_name, "prediction_unit": predicted_unit, "predicted_volume_mm3": predicted_volume, "observed_volume_mm3": row["wear_volume_mm3"],
            "predicted_mass_g": predicted_mass_g, "observed_mass_g": row["mass_loss_g"], "predicted_depth_mm": predicted_depth, "observed_depth_mm": row["wear_depth_mm"],
            "relative_volume_error": volume_error, "relative_mass_error": mass_error, "relative_depth_error": depth_error,
            "volume_uncertainty_mm3": row["volume_uncertainty_mm3"], "mass_uncertainty_g": row["mass_uncertainty_g"], "depth_uncertainty_mm": row["depth_uncertainty_mm"],
        })
    if not result_rows:
        raise ValidationError("independent test split has no rows")
    summary = {name: float(np.mean(values)) for name, values in errors.items()}
    summary.update({f"{name}_max": float(np.max(values)) for name, values in errors.items()})
    return result_rows, summary


def _quality_gates(ingest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], calibration: Mapping[str, Any]) -> dict[str, Any]:
    records = _record_lookup(ingest)
    test_keys = {(str(row["campaign_id"]), str(row["run_id"]), str(row["specimen_id"])) for row in rows}
    quality_values = [records[key] for key in test_keys]
    registration = bool(quality_values) and all(row.get("registration_passed") is True for row in quality_values)
    measurement = bool(quality_values) and all(row.get("measurement_quality_passed") is True for row in quality_values)
    split = calibration.get("group_split", {})
    independent = bool(split.get("disjoint") is True and split.get("calibration_groups") and split.get("test_groups"))
    labels_frozen = calibration.get("test_labels_frozen") is True and isinstance(calibration.get("test_label_hash"), str)
    return {
        "registration_quality_passed": registration,
        "measurement_quality_passed": measurement,
        "independent_group_split": independent,
        "test_labels_frozen": labels_frozen,
        "source_manifest_bound": bool(ingest.get("source_manifest", {}).get("manifest_sha256")),
        "status_is_measured": ingest.get("status") == "measured",
    }


def official_accuracy_gate(*, status: str, quality_gates: Mapping[str, Any], metric_summary: Mapping[str, Any], preregistered_metrics: Sequence[str] = DEFAULT_PREREGISTERED_METRICS, threshold: float = 0.05) -> bool:
    """Return true only for measured, independent, preregistered <5% evidence."""
    require_status(status)
    names = tuple(preregistered_metrics)
    if not names or set(names) != set(DEFAULT_PREREGISTERED_METRICS):
        return False
    if not math.isclose(float(threshold), 0.05, rel_tol=0.0, abs_tol=1e-12):
        return False
    if status != "measured" or not all(value is True for value in quality_gates.values()):
        return False
    # Every pre-registered metric is an acceptance criterion. A partial pass
    # cannot be promoted to the official accuracy claim.
    return all(
        isinstance(metric_summary.get(f"{name}_max"), (int, float))
        and math.isfinite(float(metric_summary[f"{name}_max"]))
        and float(metric_summary[f"{name}_max"]) < threshold
        for name in names
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    columns: list[str] = []
    for row in rows:
        for name in row:
            if name not in columns:
                columns.append(name)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in columns})
    return {"path": str(path), "sha256": sha256_file(path), "rows": len(rows), "columns": columns}


def _write_plot(path: Path, metric_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    values = [float(row["relative_volume_error"]) * 100.0 for row in metric_rows]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(np.arange(1, len(values) + 1), values, marker="o", color="#1d6670")
    axis.axhline(5.0, color="#b33a3a", linestyle="--", linewidth=1.0)
    axis.set(xlabel="Independent test specimen", ylabel="Relative volume error (%)", title="Material trial independent test")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return {"path": str(path), "sha256": sha256_file(path), "optional": True}


def build_material_trial_report(ingest: Mapping[str, Any] | str | Path, calibration: Mapping[str, Any] | str | Path, *, output_dir: str | Path | None = None, preregistered_metrics: Sequence[str] = DEFAULT_PREREGISTERED_METRICS, make_png: bool = True) -> dict[str, Any]:
    data = _artifact(ingest, INGEST_VERSION)
    fit = _artifact(calibration, CALIBRATION_VERSION)
    status = require_status(data.get("status"))
    if fit.get("status") != status:
        raise ValidationError("ingest and calibration statuses differ")
    metric_rows, metric_summary = _metric_rows(data, fit)
    quality_gates = _quality_gates(data, metric_rows, fit)
    gate = official_accuracy_gate(status=status, quality_gates=quality_gates, metric_summary=metric_summary, preregistered_metrics=preregistered_metrics)
    report: dict[str, Any] = {
        "version": REPORT_VERSION, "status": status, "source_ingest_sha256": data.get("dataset_sha256"), "calibration_sha256": fit.get("calibration_sha256"),
        "model": {"name": fit.get("model_name"), "prediction_unit": fit.get("fit", {}).get("fit_units"), "fit_hash": fit.get("fit_hash")},
        "group_split": fit.get("group_split"), "test_labels_frozen": fit.get("test_labels_frozen"), "test_label_hash": fit.get("test_label_hash"),
        "preregistered_metrics": list(preregistered_metrics), "threshold_relative_error": 0.05, "metric_summary": metric_summary,
        "quality_gates": quality_gates, "official_accuracy_gate": gate, "n_test": len(metric_rows),
        "provenance_note": "synthetic result; not official accuracy" if status == "synthetic" else ("blocked result; not eligible for accuracy claims" if status == "blocked" else "measured result; official accuracy requires every declared gate"),
    }
    artifacts: dict[str, Any] = {}
    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        summary_path = root / "step30-material-trial-report.json"
        table_path = root / "step30-material-trial-metrics.csv"
        artifacts["csv"] = _write_csv(table_path, metric_rows)
        if make_png:
            plot = _write_plot(root / "step30-material-trial-metrics.png", metric_rows)
            if plot is not None:
                artifacts["png"] = plot
        # The JSON file cannot embed its own byte hash without a circular
        # dependency. Embed its stable path and canonical report hash; callers
        # can audit the file bytes with sha256_file(path) independently.
        artifacts["json"] = {"path": str(summary_path), "byte_hash_embedded": False}
        report["artifacts"] = artifacts
        report["report_sha256"] = canonical_sha256(report)
        safe_json_dump(report, summary_path)
    else:
        report["metric_rows"] = metric_rows
        report["report_sha256"] = canonical_sha256(report)
    return report


def selftest() -> bool:
    from step30_material_trial_ingest import _selftest_row, _write_selftest_manifest
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "trials.json"
        test_row = _selftest_row("test", "S-2", "R-2", "G-2")
        # Deliberately hold out a different measured response so the selftest
        # proves that a valid measured report can still fail the official gate.
        test_row.update({
            "after_mass_g": 99.986655,
            "wear_volume_mm3": 1.7,
            "wear_depth_mm": 0.017,
        })
        source.write_text(json.dumps({"rows": [_selftest_row("calibration", "S-1", "R-1", "G-1"), test_row]}), encoding="utf-8")
        manifest = root / "source-manifest.json"
        _write_selftest_manifest(manifest, source, "measured")
        ingested = ingest_material_trials(source, manifest, requested_mode="measured")
        # Exact formula-generated test values are not needed: the gate must stay
        # false until a pre-registered independent measurement package is used.
        calibrated = calibrate_material_model(ingested, model_name="oka")
        report = build_material_trial_report(ingested, calibrated, output_dir=root / "report", make_png=False)
        assert report["status"] == "measured" and report["official_accuracy_gate"] is False
        stored = json.loads((root / "report" / "step30-material-trial-report.json").read_text(encoding="utf-8"))
        unsigned = dict(stored)
        report_hash = unsigned.pop("report_sha256")
        assert report_hash == canonical_sha256(unsigned) == report["report_sha256"]
        synthetic_manifest = root / "synthetic-manifest.json"
        _write_selftest_manifest(synthetic_manifest, source, "synthetic")
        synthetic = ingest_material_trials(source, synthetic_manifest, requested_mode="synthetic")
        assert synthetic["status"] == "synthetic"
        synthetic_fit = calibrate_material_model(synthetic, model_name="oka")
        synthetic_report = build_material_trial_report(synthetic, synthetic_fit, make_png=False)
        assert synthetic_report["status"] == "synthetic" and synthetic_report["official_accuracy_gate"] is False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build material trial JSON/CSV/optional PNG report and official gate")
    parser.add_argument("--ingest", required=False, help="normalized material ingest JSON")
    parser.add_argument("--calibration", required=False, help="calibration JSON")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        selftest()
        print("step30_material_trial_report selftest: PASS")
        return 0
    if not args.ingest or not args.calibration:
        build_parser().error("--ingest and --calibration are required unless --selftest is used")
    report = build_material_trial_report(args.ingest, args.calibration, output_dir=args.output_dir, make_png=not args.no_png)
    print(json.dumps(jsonable(report), ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"measured", "synthetic"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
