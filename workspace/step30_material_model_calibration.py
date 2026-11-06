"""Group-held-out calibration for material erosion trial records.

Oka, Finnie and Tabakoff are reported as specific eroded volume in mm3/kg.
Mansouri is reported independently as mass ratio kg/kg.  These quantities are
never summed or fitted in a mixed unit space.  The fit is a positive
multiplicative correction learned only from rows explicitly assigned to the
calibration split; test labels are copied and hashed before predictions exist.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from step30_material_trial_ingest import INGEST_VERSION, ingest_material_trials
from step30_validation_common import ValidationError, canonical_sha256, jsonable, require_status, safe_json_dump


CALIBRATION_VERSION = "step30-material-model-calibration-v1"
VOLUME_UNIT = "mm3/kg"
MASS_RATIO_UNIT = "kg/kg"


def oka_volume_mm3_per_kg(speed_m_s: Any, angle_deg: Any, d50_um: Any, *, hardness_gpa: float = 4.0) -> np.ndarray:
    speed = np.asarray(speed_m_s, dtype=float)
    angle = np.deg2rad(np.asarray(angle_deg, dtype=float))
    d50 = np.asarray(d50_um, dtype=float)
    hv = float(hardness_gpa)
    n1 = 0.71 * hv**0.14
    n2 = 2.4 * hv**-0.94
    k2 = 2.3 * hv**0.038
    angle_function = np.sin(angle) ** n1 * (1.0 + hv * (1.0 - np.sin(angle))) ** n2
    return 65.0 * hv**-0.12 * (speed / 104.0) ** k2 * (d50 / 326.0) ** 0.19 * angle_function


def finnie_volume_mm3_per_kg(speed_m_s: Any, angle_deg: Any) -> np.ndarray:
    speed = np.asarray(speed_m_s, dtype=float)
    angle = np.deg2rad(np.asarray(angle_deg, dtype=float))
    low = np.sin(2.0 * angle) - 3.0 * np.sin(angle) ** 2
    high = np.cos(angle) ** 2 / 3.0
    return np.maximum(0.0, 2.0e-5 * speed**2 * np.where(np.tan(angle) > 1.0 / 3.0, high, low))


def tabakoff_volume_mm3_per_kg(speed_m_s: Any, angle_deg: Any) -> np.ndarray:
    speed = np.asarray(speed_m_s, dtype=float)
    degrees = np.asarray(angle_deg, dtype=float)
    angle = np.deg2rad(degrees)
    rt = np.clip(1.0 - 0.0023 * speed * np.sin(angle), 0.0, 1.0)
    fb = (1.0 + 0.585 * np.sin(np.pi / 2.0 * degrees / 25.0)) ** 2
    cutting = 1.0e-5 * fb * speed**2 * np.cos(angle) ** 2 * (1.0 - rt**2)
    deformation = 5.0e-12 * (speed * np.sin(angle)) ** 4
    return np.maximum(0.0, cutting + deformation)


def mansouri_mass_ratio_kg_per_kg(speed_m_s: Any, angle_deg: Any) -> np.ndarray:
    """Mansouri output in kg/kg; it is intentionally not a volume output."""
    speed = np.asarray(speed_m_s, dtype=float)
    angle = np.deg2rad(np.clip(np.asarray(angle_deg, dtype=float), 0.1, 89.9))
    sine = np.clip(np.sin(angle), 1e-12, 1.0)
    angle_factor = 228.5 * sine**7.91 * np.clip(1.5 - sine, 1e-12, None) ** 12.81
    return 2.57e-9 * speed**2.4 * angle_factor


def _records_from_input(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read ingested trial JSON: {path}") from exc
    if not isinstance(data, dict) or data.get("version") != INGEST_VERSION:
        raise ValidationError("calibration requires a step30 material ingest artifact")
    return data


def _finite_positive(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} is not numeric") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValidationError(f"{name} must be finite and positive")
    return number


def _physical_predictions(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    speed = np.asarray([_finite_positive(row["speed_m_s"], "speed_m_s") for row in records])
    angle = np.asarray([_finite_positive(row["impact_angle_deg"], "impact_angle_deg") for row in records])
    d50 = np.asarray([_finite_positive(row["psd_d50_um"], "psd_d50_um") for row in records])
    density = np.asarray([_finite_positive(row["material_density_kg_m3"], "material_density_kg_m3") for row in records])
    oka = oka_volume_mm3_per_kg(speed, angle, d50)
    finnie = finnie_volume_mm3_per_kg(speed, angle)
    tabakoff = tabakoff_volume_mm3_per_kg(speed, angle)
    mansouri = mansouri_mass_ratio_kg_per_kg(speed, angle)
    return [
        {
            "oka_volume_mm3_per_kg": float(oka[i]),
            "finnie_volume_mm3_per_kg": float(finnie[i]),
            "tabakoff_volume_mm3_per_kg": float(tabakoff[i]),
            "mansouri_mass_kg_per_kg": float(mansouri[i]),
            "mass_to_volume_density_kg_m3": float(density[i]),
        }
        for i in range(len(records))
    ]


def _target(row: Mapping[str, Any], model_name: str) -> float:
    if model_name == "mansouri":
        return _finite_positive(row["derived"]["erosion_mass_kg_per_kg"], "erosion_mass_kg_per_kg")
    return _finite_positive(row["derived"]["erosion_volume_mm3_per_kg"], "erosion_volume_mm3_per_kg")


def _baseline(prediction: Mapping[str, Any], model_name: str) -> float:
    key = "mansouri_mass_kg_per_kg" if model_name == "mansouri" else f"{model_name}_volume_mm3_per_kg"
    return _finite_positive(prediction[key], key)


def _test_label(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "specimen_id": row["specimen_id"], "campaign_id": row["campaign_id"], "run_id": row["run_id"], "group_id": row["group_id"],
        "split": row["split"], "wear_volume_mm3": row["wear_volume_mm3"], "volume_uncertainty_mm3": row["volume_uncertainty_mm3"],
        "mass_loss_g": row["mass_loss_g"], "mass_uncertainty_g": row["mass_uncertainty_g"], "wear_depth_mm": row["wear_depth_mm"],
        "depth_uncertainty_mm": row["depth_uncertainty_mm"], "derived": row["derived"],
    }


def calibrate_material_model(ingest: Mapping[str, Any] | str | Path, *, model_name: str = "oka", output_path: str | Path | None = None) -> dict[str, Any]:
    """Fit a positive scale on calibration groups and freeze independent labels."""
    if model_name not in {"oka", "finnie", "tabakoff", "mansouri"}:
        raise ValidationError("model_name must be oka, finnie, tabakoff, or mansouri")
    data = _records_from_input(ingest)
    status = require_status(data.get("status"))
    if status == "blocked":
        result = {"version": CALIBRATION_VERSION, "status": "blocked", "blocked_reason": data.get("blocked_reason", "ingest is blocked"), "model_name": model_name, "n_records": 0}
        if output_path is not None:
            safe_json_dump(result, output_path)
        return result
    rows = data.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValidationError("ingest artifact has no records")
    if any(not isinstance(row, Mapping) or row.get("status") != status for row in rows):
        raise ValidationError("record statuses do not agree with ingest status")
    predictions = _physical_predictions(rows)
    calibration_indices = [i for i, row in enumerate(rows) if row.get("split") == "calibration"]
    test_indices = [i for i, row in enumerate(rows) if row.get("split") == "test"]
    if not calibration_indices or not test_indices:
        raise ValidationError("calibration and test splits must both be non-empty")
    calibration_groups = sorted({str(rows[i]["group_id"]) for i in calibration_indices})
    test_groups = sorted({str(rows[i]["group_id"]) for i in test_indices})
    if set(calibration_groups) & set(test_groups):
        raise ValidationError("calibration/test group overlap is forbidden")
    # Freeze labels before any fit or prediction is constructed.
    frozen_test_labels = [_test_label(rows[i]) for i in test_indices]
    frozen_test_hash = canonical_sha256(frozen_test_labels)
    ratios = np.asarray([_target(rows[i], model_name) / _baseline(predictions[i], model_name) for i in calibration_indices], dtype=float)
    if not np.isfinite(ratios).all() or np.any(ratios <= 0.0):
        raise ValidationError("calibration target/baseline ratios must be finite and positive")
    fit_scale = float(np.exp(np.mean(np.log(ratios))))
    unit = MASS_RATIO_UNIT if model_name == "mansouri" else VOLUME_UNIT
    prediction_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        base = predictions[i]
        calibrated = _baseline(base, model_name) * fit_scale
        entry = {
            "specimen_id": row["specimen_id"], "campaign_id": row["campaign_id"], "run_id": row["run_id"], "group_id": row["group_id"], "split": row["split"],
            "physical_prediction": base, "calibrated_model": model_name, "calibrated_prediction": calibrated, "calibrated_prediction_unit": unit,
            "label_frozen": _test_label(row) if row["split"] == "test" else None,
        }
        prediction_rows.append(entry)
    fit_contract = {
        "model_name": model_name, "fit_scale": fit_scale, "fit_transform": "geometric_mean(target / physical_baseline)",
        "fit_units": unit, "calibration_indices": calibration_indices, "calibration_groups": calibration_groups,
        "source_dataset_sha256": data.get("dataset_sha256"), "physical_formula_version": "step30-material-physical-formulas-v1",
    }
    result = {
        "version": CALIBRATION_VERSION, "status": status, "model_name": model_name, "source_dataset_sha256": data.get("dataset_sha256"),
        "n_records": len(rows), "n_calibration": len(calibration_indices), "n_test": len(test_indices),
        "group_split": {"scheme": "explicit_group_holdout", "calibration_groups": calibration_groups, "test_groups": test_groups, "disjoint": True,
                        "calibration_indices": calibration_indices, "test_indices": test_indices},
        "fit": fit_contract, "fit_hash": canonical_sha256(fit_contract),
        "test_labels_frozen": True, "test_label_hash": frozen_test_hash, "test_labels": frozen_test_labels,
        "predicted_model": {"name": model_name, "unit": unit, "predictions": prediction_rows},
        "unit_contract": {"oka": VOLUME_UNIT, "finnie": VOLUME_UNIT, "tabakoff": VOLUME_UNIT, "mansouri": MASS_RATIO_UNIT, "no_mixed_addition": True},
        "provenance_note": "synthetic result; not official accuracy" if status == "synthetic" else "measured fit; official accuracy requires independent report gate",
    }
    result["calibration_sha256"] = canonical_sha256(result)
    if output_path is not None:
        safe_json_dump(result, output_path)
    return result


def load_calibration(path: str | Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid calibration artifact: {path}") from exc
    if not isinstance(data, dict) or data.get("version") != CALIBRATION_VERSION:
        raise ValidationError("unsupported calibration artifact")
    return data


def selftest() -> bool:
    from step30_material_trial_ingest import _selftest_row, _write_selftest_manifest
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "trials.json"
        source.write_text(json.dumps({"rows": [_selftest_row("calibration", "S-1", "R-1", "G-1"), _selftest_row("test", "S-2", "R-2", "G-2")]}), encoding="utf-8")
        manifest = root / "source-manifest.json"
        _write_selftest_manifest(manifest, source, "measured")
        ingested = ingest_material_trials(source, manifest, requested_mode="measured")
        result = calibrate_material_model(ingested, model_name="oka")
        assert result["status"] == "measured" and result["test_labels_frozen"] is True
        assert result["group_split"]["disjoint"] is True and len(result["predicted_model"]["predictions"]) == 2
        assert result["fit_hash"] and result["test_label_hash"]
        mansouri = calibrate_material_model(ingested, model_name="mansouri")
        assert mansouri["fit"]["fit_units"] == MASS_RATIO_UNIT
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate material erosion model on calibration groups only")
    parser.add_argument("--ingest", required=False, help="step30 material ingest JSON")
    parser.add_argument("--model", choices=("oka", "finnie", "tabakoff", "mansouri"), default="oka")
    parser.add_argument("--output", type=Path, help="immutable calibration JSON")
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        selftest()
        print("step30_material_model_calibration selftest: PASS")
        return 0
    if not args.ingest:
        build_parser().error("--ingest is required unless --selftest is used")
    result = calibrate_material_model(args.ingest, model_name=args.model, output_path=args.output)
    print(json.dumps(jsonable(result), ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"measured", "synthetic"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
