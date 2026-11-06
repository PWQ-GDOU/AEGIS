"""Fail-closed ingestion for auditable material erosion coupon trials.

The requested CLI mode is never treated as evidence. ``measured`` or
``synthetic`` status comes only from a source manifest whose hash binds the
named CSV/JSON/NPZ input and whose provenance block identifies its origin.
Accepted records are converted to the canonical units in ``FIELD_UNITS``.
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

from step30_validation_common import ValidationError, canonical_sha256, jsonable, require_status, safe_json_dump, sha256_file


INGEST_VERSION = "step30-material-trial-ingest-v1"
SOURCE_MANIFEST_VERSION = "step30-material-trial-source-manifest-v1"

FIELD_UNITS = {
    "speed_m_s": "m/s", "impact_angle_deg": "deg", "psd_d10_um": "um", "psd_d50_um": "um", "psd_d90_um": "um",
    "sediment_mass_kg": "kg", "exposure_s": "s", "before_mass_g": "g", "after_mass_g": "g", "mass_loss_g": "g",
    "mass_uncertainty_g": "g", "material_density_kg_m3": "kg/m3", "exposed_area_mm2": "mm2", "wear_depth_mm": "mm",
    "depth_uncertainty_mm": "mm", "wear_volume_mm3": "mm3", "volume_uncertainty_mm3": "mm3", "roughness_ra_um": "um",
    "roughness_rq_um": "um", "roughness_rz_um": "um", "roughness_uncertainty_um": "um", "pit_max_depth_um": "um",
    "pit_mean_depth_um": "um", "pit_mean_width_um": "um", "pit_volume_mm3": "mm3", "pit_uncertainty_mm3": "mm3",
}

CANONICAL_FIELDS = (
    "specimen_id", "campaign_id", "run_id", "material", "split", "group_id", "speed_m_s", "impact_angle_deg",
    "psd_d10_um", "psd_d50_um", "psd_d90_um", "sediment_mass_kg", "exposure_s", "before_mass_g", "after_mass_g",
    "mass_loss_g", "mass_uncertainty_g", "material_density_kg_m3", "exposed_area_mm2", "wear_depth_mm",
    "depth_uncertainty_mm", "wear_volume_mm3", "volume_uncertainty_mm3", "roughness_ra_um", "roughness_rq_um",
    "roughness_rz_um", "roughness_uncertainty_um", "pit_count", "pit_max_depth_um", "pit_mean_depth_um",
    "pit_mean_width_um", "pit_volume_mm3", "pit_uncertainty_mm3", "registration_passed", "measurement_quality_passed", "status",
)

# (source field, implied unit). A missing implied unit requires a manifest unit entry.
_ALIASES: dict[str, tuple[tuple[str, str | None], ...]] = {
    "specimen_id": (("specimen_id", None), ("specimen", None), ("coupon_id", None)),
    "campaign_id": (("campaign_id", None), ("campaign", None)),
    "run_id": (("run_id", None), ("run", None), ("trial_id", None)),
    "material": (("material", None), ("material_name", None), ("alloy", None)),
    "split": (("split", None), ("data_split", None)),
    "group_id": (("group_id", None), ("split_group", None), ("batch_id", None)),
    "speed_m_s": (("speed_m_s", "m/s"), ("speed_ms", "m/s"), ("velocity_m_s", "m/s"), ("speed", None)),
    "impact_angle_deg": (("impact_angle_deg", "deg"), ("angle_deg", "deg"), ("angle", None)),
    "psd_d10_um": (("psd_d10_um", "um"), ("d10_um", "um"), ("psd_d10", None)),
    "psd_d50_um": (("psd_d50_um", "um"), ("d50_um", "um"), ("particle_size_um", "um"), ("psd_d50", None)),
    "psd_d90_um": (("psd_d90_um", "um"), ("d90_um", "um"), ("psd_d90", None)),
    "sediment_mass_kg": (("sediment_mass_kg", "kg"), ("sediment_mass_g", "g"), ("sediment_mass", None)),
    "exposure_s": (("exposure_s", "s"), ("exposure_min", "min"), ("exposure_h", "h"), ("exposure", None)),
    "before_mass_g": (("before_mass_g", "g"), ("mass_before_g", "g"), ("before_mass_mg", "mg"), ("mass_before_mg", "mg"), ("before_mass", None)),
    "after_mass_g": (("after_mass_g", "g"), ("mass_after_g", "g"), ("after_mass_mg", "mg"), ("mass_after_mg", "mg"), ("after_mass", None)),
    "mass_loss_g": (("mass_loss_g", "g"), ("mass_loss_mg", "mg"), ("mass_loss", None)),
    "mass_uncertainty_g": (("mass_uncertainty_g", "g"), ("mass_uncertainty_mg", "mg"), ("mass_uncertainty", None)),
    "material_density_kg_m3": (("material_density_kg_m3", "kg/m3"), ("density_kg_m3", "kg/m3"), ("density_g_cm3", "g/cm3"), ("material_density", None)),
    "exposed_area_mm2": (("exposed_area_mm2", "mm2"), ("target_area_mm2", "mm2"), ("exposed_area", None)),
    "wear_depth_mm": (("wear_depth_mm", "mm"), ("depth_mm", "mm"), ("wear_depth_um", "um"), ("wear_depth", None)),
    "depth_uncertainty_mm": (("depth_uncertainty_mm", "mm"), ("depth_uncertainty_um", "um"), ("depth_uncertainty", None)),
    "wear_volume_mm3": (("wear_volume_mm3", "mm3"), ("volume_loss_mm3", "mm3"), ("wear_volume", None)),
    "volume_uncertainty_mm3": (("volume_uncertainty_mm3", "mm3"), ("volume_uncertainty", None)),
    "roughness_ra_um": (("roughness_ra_um", "um"), ("ra_um", "um"), ("ra", None)),
    "roughness_rq_um": (("roughness_rq_um", "um"), ("rq_um", "um"), ("rq", None)),
    "roughness_rz_um": (("roughness_rz_um", "um"), ("rz_um", "um"), ("rz", None)),
    "roughness_uncertainty_um": (("roughness_uncertainty_um", "um"), ("roughness_uncertainty", None)),
    "pit_count": (("pit_count", None), ("pits", None)),
    "pit_max_depth_um": (("pit_max_depth_um", "um"), ("max_pit_depth_um", "um"), ("pit_max_depth", None)),
    "pit_mean_depth_um": (("pit_mean_depth_um", "um"), ("mean_pit_depth_um", "um"), ("pit_mean_depth", None)),
    "pit_mean_width_um": (("pit_mean_width_um", "um"), ("mean_pit_width_um", "um"), ("pit_mean_width", None)),
    "pit_volume_mm3": (("pit_volume_mm3", "mm3"), ("total_pit_volume_mm3", "mm3"), ("pit_volume", None)),
    "pit_uncertainty_mm3": (("pit_uncertainty_mm3", "mm3"), ("pit_uncertainty", None)),
    "registration_passed": (("registration_passed", None), ("registration_quality_passed", None)),
    "measurement_quality_passed": (("measurement_quality_passed", None), ("measurement_passed", None)),
}

_UNIT_INFO = {
    "kg": ("mass", 1.0), "g": ("mass", 1e-3), "mg": ("mass", 1e-6), "ug": ("mass", 1e-9),
    "m": ("length", 1.0), "cm": ("length", 1e-2), "mm": ("length", 1e-3), "um": ("length", 1e-6),
    "s": ("time", 1.0), "min": ("time", 60.0), "h": ("time", 3600.0),
    "m/s": ("speed", 1.0), "mm/s": ("speed", 1e-3), "rad": ("angle", 1.0), "deg": ("angle", math.pi / 180.0),
    "m2": ("area", 1.0), "cm2": ("area", 1e-4), "mm2": ("area", 1e-6),
    "m3": ("volume", 1.0), "cm3": ("volume", 1e-6), "mm3": ("volume", 1e-9), "um3": ("volume", 1e-18),
    "kg/m3": ("density", 1.0), "g/cm3": ("density", 1000.0), "mg/mm3": ("density", 1000.0),
    "mm3/kg": ("specific_volume", 1e-9), "cm3/kg": ("specific_volume", 1e-6), "m3/kg": ("specific_volume", 1.0), "mm3/g": ("specific_volume", 1e-6),
    "kg/kg": ("mass_ratio", 1.0), "g/kg": ("mass_ratio", 1e-3), "mg/kg": ("mass_ratio", 1e-6), "g/g": ("mass_ratio", 1.0),
}


def normalize_unit(unit: str) -> str:
    if not isinstance(unit, str) or not unit.strip():
        raise ValidationError("every dimensional source field needs an explicit unit")
    value = unit.strip().lower().replace("μ", "u").replace("µ", "u").replace("³", "3").replace("²", "2").replace("^", "").replace(" ", "")
    value = {"sec": "s", "second": "s", "seconds": "s", "hour": "h", "hours": "h", "degree": "deg", "degrees": "deg", "mps": "m/s"}.get(value, value)
    if value not in _UNIT_INFO:
        raise ValidationError(f"unsupported material-trial unit {unit!r}")
    return value


def convert_value(value: Any, from_unit: str, to_unit: str) -> float:
    source = normalize_unit(from_unit)
    target = normalize_unit(to_unit)
    source_dimension, source_factor = _UNIT_INFO[source]
    target_dimension, target_factor = _UNIT_INFO[target]
    if source_dimension != target_dimension:
        raise ValidationError(f"incompatible units: {from_unit!r} and {to_unit!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"non-numeric value {value!r}") from exc
    if not math.isfinite(number):
        raise ValidationError("material-trial value is not finite")
    return number * source_factor / target_factor


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or any(not str(name).strip() for name in reader.fieldnames):
                raise ValidationError("CSV requires a non-empty named header")
            rows = [dict(row) for row in reader]
    elif suffix == ".json":
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValidationError(f"invalid trial JSON: {path}") from exc
        rows = document.get("rows", document.get("records", document.get("data"))) if isinstance(document, dict) else document
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValidationError("JSON trial input requires a named object row list")
        rows = [dict(row) for row in rows]
    elif suffix == ".npz":
        try:
            with np.load(path, allow_pickle=False) as loaded:
                names = [name for name in loaded.files if name != "__metadata_json__"]
                if not names:
                    raise ValidationError("NPZ contains no named trial arrays")
                arrays = {name: np.asarray(loaded[name]) for name in names}
        except (OSError, ValueError) as exc:
            if isinstance(exc, ValidationError):
                raise
            raise ValidationError(f"invalid trial NPZ: {path}") from exc
        if any(array.dtype == object or array.ndim != 1 for array in arrays.values()):
            raise ValidationError("NPZ trial fields must be safe one-dimensional named arrays")
        lengths = {len(array) for array in arrays.values()}
        if len(lengths) != 1:
            raise ValidationError("NPZ trial arrays have inconsistent lengths")
        count = lengths.pop()
        rows = [{name: arrays[name][index].item() for name in names} for index in range(count)]
    else:
        raise ValidationError("material trial input must be named CSV, JSON, or NPZ")
    if not rows:
        raise ValidationError("material trial input contains no rows")
    return rows


def _source_entry(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    source = manifest.get("source")
    if isinstance(source, Mapping):
        return source
    entries = manifest.get("entries")
    if isinstance(entries, Mapping):
        for name in ("trials", "trial_data", "source"):
            if isinstance(entries.get(name), Mapping):
                return entries[name]
    raise ValidationError("source manifest has no trial-data source entry")


def validate_source_manifest(manifest_path: str | Path, input_path: str | Path) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve()
    source_file = Path(input_path).resolve()
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid source manifest: {manifest_file}") from exc
    if not isinstance(manifest, dict) or manifest.get("version") != SOURCE_MANIFEST_VERSION:
        raise ValidationError(f"source manifest must use {SOURCE_MANIFEST_VERSION}")
    status = require_status(manifest.get("status"))
    stored_hash = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if not isinstance(stored_hash, str) or canonical_sha256(unsigned) != stored_hash:
        raise ValidationError("source manifest canonical hash mismatch")
    entry = _source_entry(manifest)
    entry_path = entry.get("path")
    if not isinstance(entry_path, str) or not entry_path:
        raise ValidationError("source manifest trial entry has no path")
    bound_path = Path(entry_path)
    if not bound_path.is_absolute():
        bound_path = manifest_file.parent / bound_path
    if bound_path.resolve() != source_file:
        raise ValidationError("source manifest is not bound to the requested trial input")
    if entry.get("sha256") != sha256_file(source_file):
        raise ValidationError("source manifest trial-data hash mismatch")
    if entry.get("status", status) != status:
        raise ValidationError("source entry status differs from manifest status")
    provenance = manifest.get("provenance")
    required_provenance = {"origin", "campaign_id", "acquisition_method", "acquired_at", "organization"}
    if not isinstance(provenance, Mapping) or any(not str(provenance.get(name, "")).strip() for name in required_provenance):
        raise ValidationError(f"source manifest provenance requires {sorted(required_provenance)}")
    expected_origin = {"measured": "laboratory_measurement", "synthetic": "deterministic_simulation", "blocked": "blocked"}[status]
    if provenance.get("origin") != expected_origin:
        raise ValidationError(f"{status} source requires provenance origin={expected_origin!r}")
    if not isinstance(manifest.get("units"), Mapping):
        raise ValidationError("source manifest requires an explicit units mapping")
    return manifest


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _bool(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "yes", "passed", "1"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"false", "no", "failed", "0"}:
        return False
    raise ValidationError(f"{name} must be an explicit boolean")


def _extract(row: Mapping[str, Any], field: str, units: Mapping[str, Any]) -> Any:
    for source_name, implied_unit in _ALIASES[field]:
        if source_name not in row or _blank(row[source_name]):
            continue
        value = row[source_name]
        if field in FIELD_UNITS:
            source_unit = units.get(source_name, units.get(field, implied_unit))
            return convert_value(value, str(source_unit) if source_unit is not None else "", FIELD_UNITS[field])
        if field in {"registration_passed", "measurement_quality_passed"}:
            return _bool(value, field)
        if field == "pit_count":
            number = float(value)
            if not number.is_integer() or number < 0:
                raise ValidationError("pit_count must be a non-negative integer")
            return int(number)
        return str(value).strip()
    return None


def _positive(record: Mapping[str, Any], fields: Sequence[str], *, allow_zero: bool = False) -> None:
    for field in fields:
        value = record[field]
        if value is None or (value < 0 if allow_zero else value <= 0):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValidationError(f"{field} must be present and {qualifier}")


def normalize_record(row: Mapping[str, Any], manifest: Mapping[str, Any], index: int) -> dict[str, Any]:
    units = manifest["units"]
    status = require_status(manifest["status"])
    record = {field: _extract(row, field, units) for field in CANONICAL_FIELDS if field != "status"}
    record["status"] = status
    for field in ("specimen_id", "campaign_id", "run_id", "material", "split"):
        if not record[field]:
            raise ValidationError(f"row {index}: missing {field}")
    row_status = row.get("status")
    if not _blank(row_status) and row_status != status:
        raise ValidationError(f"row {index}: row status differs from source manifest")
    if record["campaign_id"] != str(manifest["provenance"]["campaign_id"]):
        raise ValidationError(f"row {index}: campaign differs from source provenance")
    if record["split"] not in {"calibration", "test"}:
        raise ValidationError(f"row {index}: split must be calibration or test")
    record["group_id"] = record["group_id"] or f"{record['campaign_id']}::{record['specimen_id']}"
    _positive(record, ("speed_m_s", "psd_d10_um", "psd_d50_um", "psd_d90_um", "sediment_mass_kg", "exposure_s", "before_mass_g", "after_mass_g", "material_density_kg_m3", "exposed_area_mm2"))
    if not 0.0 < record["impact_angle_deg"] <= 90.0:
        raise ValidationError(f"row {index}: impact_angle_deg must be in (0, 90]")
    if not record["psd_d10_um"] <= record["psd_d50_um"] <= record["psd_d90_um"]:
        raise ValidationError(f"row {index}: PSD quantiles must satisfy d10 <= d50 <= d90")
    derived_loss = record["before_mass_g"] - record["after_mass_g"]
    if derived_loss < 0.0:
        raise ValidationError(f"row {index}: after mass exceeds before mass")
    if record["mass_loss_g"] is None:
        record["mass_loss_g"] = derived_loss
    elif abs(record["mass_loss_g"] - derived_loss) > max(record["mass_uncertainty_g"] or 0.0, 1e-9):
        raise ValidationError(f"row {index}: stated mass loss conflicts with before/after mass")
    _positive(record, ("mass_loss_g", "mass_uncertainty_g", "wear_depth_mm", "depth_uncertainty_mm", "wear_volume_mm3", "volume_uncertainty_mm3", "roughness_ra_um", "roughness_rq_um", "roughness_rz_um", "roughness_uncertainty_um", "pit_max_depth_um", "pit_mean_depth_um", "pit_mean_width_um", "pit_volume_mm3", "pit_uncertainty_mm3"), allow_zero=True)
    if record["pit_count"] is None:
        raise ValidationError(f"row {index}: missing pit_count")
    if record["registration_passed"] is None or record["measurement_quality_passed"] is None:
        raise ValidationError(f"row {index}: explicit registration and measurement quality flags are required")
    density_mg_mm3 = record["material_density_kg_m3"] * 1e-3
    volume_from_mass = record["mass_loss_g"] * 1000.0 / density_mg_mm3
    propagated_volume_uncertainty = record["mass_uncertainty_g"] * 1000.0 / density_mg_mm3
    if abs(record["wear_volume_mm3"] - volume_from_mass) > max(record["volume_uncertainty_mm3"], propagated_volume_uncertainty, 1e-9):
        raise ValidationError(f"row {index}: mass-derived and stated wear volume disagree beyond uncertainty")
    depth_from_volume = record["wear_volume_mm3"] / record["exposed_area_mm2"]
    if abs(record["wear_depth_mm"] - depth_from_volume) > max(record["depth_uncertainty_mm"], record["volume_uncertainty_mm3"] / record["exposed_area_mm2"], 1e-9):
        raise ValidationError(f"row {index}: volume/area and stated depth disagree beyond uncertainty")
    record["derived"] = {
        "volume_from_mass_mm3": volume_from_mass,
        "volume_from_mass_uncertainty_mm3": propagated_volume_uncertainty,
        "depth_from_volume_mm": depth_from_volume,
        "erosion_volume_mm3_per_kg": record["wear_volume_mm3"] / record["sediment_mass_kg"],
        "erosion_volume_uncertainty_mm3_per_kg": record["volume_uncertainty_mm3"] / record["sediment_mass_kg"],
        "erosion_mass_kg_per_kg": (record["mass_loss_g"] * 1e-3) / record["sediment_mass_kg"],
        "erosion_mass_uncertainty_kg_per_kg": (record["mass_uncertainty_g"] * 1e-3) / record["sediment_mass_kg"],
    }
    return record


def ingest_material_trials(input_path: str | Path | None, source_manifest_path: str | Path | None, *, requested_mode: str | None = None, output_path: str | Path | None = None) -> dict[str, Any]:
    """Ingest trials, returning ``blocked`` instead of inventing missing truth."""
    if requested_mode is not None:
        require_status(requested_mode)
    try:
        if input_path is None or source_manifest_path is None:
            raise ValidationError("real/synthetic input and its source manifest are both required")
        source = Path(input_path).resolve()
        manifest_file = Path(source_manifest_path).resolve()
        manifest = validate_source_manifest(manifest_file, source)
        status = require_status(manifest["status"])
        if requested_mode is not None and requested_mode != status:
            raise ValidationError("--mode cannot override source-manifest status")
        if status == "blocked":
            raise ValidationError("source manifest marks the trial input blocked")
        records = [normalize_record(row, manifest, index + 1) for index, row in enumerate(_load_rows(source))]
        identities = [(row["campaign_id"], row["run_id"], row["specimen_id"]) for row in records]
        if len(set(identities)) != len(identities):
            raise ValidationError("campaign/run/specimen identity must be unique")
        calibration_groups = {row["group_id"] for row in records if row["split"] == "calibration"}
        test_groups = {row["group_id"] for row in records if row["split"] == "test"}
        if not calibration_groups or not test_groups or calibration_groups & test_groups:
            raise ValidationError("calibration/test require non-empty disjoint group_id sets")
        result = {
            "version": INGEST_VERSION, "status": status, "requested_mode": requested_mode,
            "source": {"path": str(source), "sha256": sha256_file(source)},
            "source_manifest": {"path": str(manifest_file), "sha256": sha256_file(manifest_file), "manifest_sha256": manifest["manifest_sha256"]},
            "provenance": jsonable(manifest["provenance"]), "field_units": FIELD_UNITS, "canonical_fields": list(CANONICAL_FIELDS),
            "n_records": len(records), "records": records,
        }
        result["dataset_sha256"] = canonical_sha256(result)
    except (OSError, ValidationError, ValueError, TypeError) as exc:
        result = {
            "version": INGEST_VERSION, "status": "blocked", "requested_mode": requested_mode, "blocked_reason": str(exc),
            "source": {"path": str(input_path) if input_path is not None else None, "exists": bool(input_path is not None and Path(input_path).is_file())},
            "source_manifest": {"path": str(source_manifest_path) if source_manifest_path is not None else None, "exists": bool(source_manifest_path is not None and Path(source_manifest_path).is_file())},
            "n_records": 0, "records": [],
        }
        result["dataset_sha256"] = canonical_sha256(result)
    if output_path is not None:
        safe_json_dump(result, output_path)
    return result


def _selftest_row(split: str, specimen: str, run: str, group: str) -> dict[str, Any]:
    return {
        "specimen": specimen, "campaign": "C-1", "run": run, "material": "steel", "split": split, "group_id": group,
        "speed_ms": 50, "angle_deg": 45, "d10_um": 40, "d50_um": 50, "d90_um": 70, "sediment_mass_g": 100,
        "exposure_min": 10, "before_mass_g": 100.0, "after_mass_g": 99.99215, "mass_uncertainty_mg": 0.2,
        "density_g_cm3": 7.85, "target_area_mm2": 100, "wear_depth_mm": 0.01, "depth_uncertainty_um": 1,
        "wear_volume_mm3": 1.0, "volume_uncertainty_mm3": 0.03, "ra_um": 1.0, "rq_um": 1.2, "rz_um": 5.0,
        "roughness_uncertainty_um": 0.1, "pit_count": 2, "max_pit_depth_um": 8, "mean_pit_depth_um": 4,
        "mean_pit_width_um": 20, "pit_volume_mm3": 0.1, "pit_uncertainty_mm3": 0.01,
        "registration_passed": True, "measurement_quality_passed": True,
    }


def _write_selftest_manifest(path: Path, source: Path, status: str) -> None:
    manifest = {
        "version": SOURCE_MANIFEST_VERSION, "status": status,
        "source": {"path": source.name, "sha256": sha256_file(source), "status": status},
        "provenance": {"origin": "laboratory_measurement" if status == "measured" else "deterministic_simulation", "campaign_id": "C-1", "acquisition_method": "coupon rig and profilometer", "acquired_at": "2026-07-30T00:00:00Z", "organization": "selftest"},
        "units": {},
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def selftest() -> bool:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "trials.json"
        source.write_text(json.dumps({"rows": [_selftest_row("calibration", "S-1", "R-1", "G-1"), _selftest_row("test", "S-2", "R-2", "G-2")]}), encoding="utf-8")
        manifest = root / "source-manifest.json"
        _write_selftest_manifest(manifest, source, "measured")
        result = ingest_material_trials(source, manifest, requested_mode="measured")
        assert result["status"] == "measured" and result["n_records"] == 2
        assert math.isclose(result["records"][0]["sediment_mass_kg"], 0.1)
        assert math.isclose(result["records"][0]["derived"]["erosion_volume_mm3_per_kg"], 10.0)
        assert ingest_material_trials(source, None, requested_mode="measured")["status"] == "blocked"
        assert ingest_material_trials(source, manifest, requested_mode="synthetic")["status"] == "blocked"
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest auditable material coupon trial CSV/JSON/NPZ")
    parser.add_argument("--input", help="named material trial CSV/JSON/NPZ")
    parser.add_argument("--source-manifest", help="hashed provenance/source manifest")
    parser.add_argument("--mode", choices=("measured", "synthetic", "blocked"), help="assert, never establish, source status")
    parser.add_argument("--output", type=Path, help="immutable normalized JSON output")
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        selftest()
        print("step30_material_trial_ingest selftest: PASS")
        return 0
    result = ingest_material_trials(args.input, args.source_manifest, requested_mode=args.mode, output_path=args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"measured", "synthetic"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
