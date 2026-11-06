"""Candidate erosion-mechanism classification for Step30 particle hits.

This module reads an immutable ``particle_hits.npz`` and its paired
``summary.json``.  It produces conservative candidate labels only.  It never
turns particle impacts into a claim of measured material damage.

The implementation intentionally accepts both the legacy point-cloud fields
(``hit_pos_local``, ``Vp``, ``ER_cfd``) and the CFD tracking fields such as
``position_local_mm``, ``Vp_ms``, ``parent_face_id`` and face aggregates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

LABELS = ("cutting", "deformation", "impact_pitting", "fatigue_candidate", "sliding")
STATUSES = ("synthetic", "measured", "blocked")
VERSION = "step30-mechanism-candidate-v1"
CLAIM_SCOPE = "candidate mechanism only; not measured damage"


class ClassifierError(ValueError):
    """Raised when an input cannot satisfy the classifier contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_array(value: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as exc:  # pragma: no cover - NumPy normally raises ValueError
        raise ClassifierError(f"{name} is not an array") from exc
    if array.dtype.kind in "OUSV":
        raise ClassifierError(f"{name} must be numeric")
    try:
        array = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ClassifierError(f"{name} must be numeric") from exc
    if ndim is not None and array.ndim != ndim:
        raise ClassifierError(f"{name} must have ndim={ndim}, got {array.ndim}")
    if not np.isfinite(array).all():
        raise ClassifierError(f"{name} contains NaN or infinite values")
    return array


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim != 0:
        return value
    item = array.item()
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    return item


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ClassifierError("non-finite value cannot be written to JSON")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise ClassifierError(f"unsupported JSON value: {type(value).__name__}")


def _write_json(path: Path, value: Any, *, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        _json_safe(value), ensure_ascii=False, allow_nan=False,
        indent=2 if pretty else None, separators=None if pretty else (",", ":"),
    )
    path.write_text(text + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]], columns: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(columns)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in columns})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, set)):
        return "|".join(str(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _find_array(data: Mapping[str, np.ndarray], names: Iterable[str], n: int | None = None) -> tuple[np.ndarray | None, str | None]:
    for name in names:
        if name not in data:
            continue
        array = np.asarray(data[name])
        if array.ndim == 0:
            continue
        if n is not None and len(array) != n:
            raise ClassifierError(f"field {name} has length {len(array)}; expected {n}")
        return array, name
    return None, None


def _find_nested_status(value: Any) -> tuple[str | None, str | None]:
    """Find an explicit status without treating validation 'passed' as status."""
    status_keys = {
        "status", "provenance_status", "data_status", "source_status",
        "record_status", "input_status", "measurement_status",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_norm = str(key).lower()
            if key_norm in status_keys and isinstance(item, str) and item in STATUSES:
                return item, str(key)
        for key, item in value.items():
            found, source = _find_nested_status(item)
            if found:
                return found, source or str(key)
    elif isinstance(value, (list, tuple)):
        for item in value:
            found, source = _find_nested_status(item)
            if found:
                return found, source
    return None, None


def _infer_status(summary: Mapping[str, Any], metadata: Mapping[str, Any]) -> tuple[str, str]:
    status, source = _find_nested_status(metadata)
    if status:
        return status, f"npz:{source}"
    status, source = _find_nested_status(summary)
    if status:
        return status, f"summary:{source}"
    # A missing status is deliberately treated as synthetic.  CFD particle
    # impacts and model outputs are not measurements of material loss.
    return "synthetic", "default:simulation-or-unlabelled-input"


def load_inputs(hits_path: str | Path, summary_path: str | Path) -> dict[str, Any]:
    """Read the paired inputs without writing or mutating either file."""
    hits = Path(hits_path)
    summary_file = Path(summary_path)
    if not hits.is_file():
        raise FileNotFoundError(hits)
    if not summary_file.is_file():
        raise FileNotFoundError(summary_file)
    try:
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ClassifierError(f"invalid summary JSON: {summary_file}") from exc
    if not isinstance(summary, Mapping):
        raise ClassifierError("summary.json must contain a JSON object")
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    try:
        with np.load(str(hits), allow_pickle=False) as loaded:
            for key in loaded.files:
                value = np.asarray(loaded[key])
                if key == "__metadata_json__":
                    if value.ndim != 0:
                        raise ClassifierError("invalid NPZ metadata")
                    try:
                        parsed = json.loads(str(_scalar(value)))
                    except json.JSONDecodeError as exc:
                        raise ClassifierError("invalid embedded NPZ metadata JSON") from exc
                    if isinstance(parsed, Mapping):
                        metadata.update(parsed)
                    continue
                if value.dtype.kind in "OUSV":
                    # Existing CFD hits store scalar provenance such as
                    # collision_backend as 0-D Unicode arrays.  Preserve
                    # those values as metadata, while rejecting per-hit text
                    # arrays that cannot be used safely as numeric fields.
                    if value.ndim == 0:
                        metadata[key] = _scalar(value)
                        continue
                    raise ClassifierError(f"unsupported object/string array in NPZ: {key}")
                if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                    raise ClassifierError(f"NPZ field contains NaN or infinite values: {key}")
                arrays[key] = value
    except (OSError, ValueError) as exc:
        if isinstance(exc, ClassifierError):
            raise
        raise ClassifierError(f"invalid particle hits NPZ: {hits}") from exc
    status, status_source = _infer_status(summary, metadata)
    validation = summary.get("validation", {})
    if isinstance(validation, Mapping) and validation.get("registration_status") not in (None, "passed"):
        status = "blocked"
        status_source = "summary:validation.registration_status"
    n_hits = _determine_hit_count(arrays)
    if n_hits <= 0:
        raise ClassifierError("particle hits NPZ has no per-hit arrays")
    declared_hits = summary.get("tracking", {}).get("n_impacts")
    if declared_hits is not None and int(declared_hits) != n_hits:
        raise ClassifierError(f"summary n_impacts={declared_hits} does not match NPZ hit count={n_hits}")
    identity = summary.get("artifact_identity", {})
    for key, summary_value in {
        "artifact_set_id": identity.get("artifact_set_id"),
        "particle_run_id": identity.get("particle_run_id"),
        "mesh_signature": summary.get("geometry", {}).get("mesh_signature"),
    }.items():
        if summary_value is None or key not in metadata:
            continue
        if str(_scalar(metadata[key])) != str(summary_value):
            raise ClassifierError(f"NPZ/summary identity mismatch for {key}")
    return {
        "hits_path": hits,
        "summary_path": summary_file,
        "arrays": arrays,
        "metadata": metadata,
        "summary": dict(summary),
        "status": status,
        "status_source": status_source,
        "input_hashes": {"hits_sha256": _sha256(hits), "summary_sha256": _sha256(summary_file)},
    }


def _determine_hit_count(data: Mapping[str, np.ndarray]) -> int:
    for names in (
        ("position_local_mm", "hit_pos_local", "position_global_mm", "hit_pos_global", "position_local", "position_global"),
        ("Vp_ms", "Vp", "speed_ms", "speed"),
        ("alpha_deg", "impact_angle_deg", "angle_deg"),
        ("parent_face_id", "face_id", "collision_face_id"),
    ):
        array, _ = _find_array(data, names)
        if array is not None and array.ndim >= 1:
            return int(len(array))
    return 0


def _coordinates(data: Mapping[str, np.ndarray], n: int) -> tuple[np.ndarray, str]:
    for names, space in (
        (("hit_pos_global_mm", "hit_pos_global", "position_global_mm", "position_global"), "global"),
        (("position_local_mm", "hit_pos_local", "position_local", "local"), "local"),
    ):
        array, source = _find_array(data, names, n)
        if array is not None:
            array = _finite_array(array, source or "position", ndim=2)
            if array.shape != (n, 3):
                raise ClassifierError(f"{source} must have shape ({n}, 3)")
            return array, space
    raise ClassifierError("particle hits require a 3D position field")


def _vector_field(data: Mapping[str, np.ndarray], n: int) -> np.ndarray | None:
    array, source = _find_array(data, ("velocity_ms", "velocity", "impact_velocity_ms"), n)
    if array is None:
        return None
    array = _finite_array(array, source or "velocity", ndim=2)
    if array.shape != (n, 3):
        raise ClassifierError(f"{source} must have shape ({n}, 3)")
    return array


def _hit_face_ids(data: Mapping[str, np.ndarray], n: int) -> tuple[np.ndarray, str]:
    array, source = _find_array(data, ("parent_face_id", "face_id", "collision_face_id", "surface_face_id"), n)
    if array is None:
        return np.arange(n, dtype=np.int64), "synthetic_row_id"
    return np.asarray(array, dtype=np.int64), source or "face_id"


def _repeat_counts(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = Counter(int(x) for x in ids)
    return np.asarray([counts[int(x)] for x in ids], dtype=np.int64), np.asarray(list(counts.values()), dtype=np.int64)


def _percentile_score(values: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    if len(values) == 0:
        return np.empty(0, dtype=np.float64)
    reference = float(np.percentile(values, percentile))
    if reference <= 0.0:
        return np.ones(len(values), dtype=np.float64)
    return np.clip(values / reference, 0.0, 1.0)


def _normalize_rank(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    return order.astype(np.float64) / max(len(values) - 1, 1)


def _face_normals_for_hits(data: Mapping[str, np.ndarray], face_ids: np.ndarray, n: int) -> np.ndarray | None:
    direct, _ = _find_array(data, ("normal", "hit_normal", "face_normal"), n)
    if direct is not None:
        direct = _finite_array(direct, "hit normals", ndim=2)
        if direct.shape == (n, 3):
            return direct
    table, _ = _find_array(data, ("face_normals", "surface_normals"))
    if table is None:
        return None
    table = _finite_array(table, "face_normals", ndim=2)
    if table.shape[1] != 3:
        return None
    ids = np.asarray(face_ids, dtype=np.int64)
    if len(table) and ids.min(initial=0) >= 0 and ids.max(initial=0) < len(table):
        return table[ids]
    return None


def _mechanism_labels(alpha: float, speed: float, p20_speed: float, normal_speed: float, repeated: int, impact_index: int) -> tuple[str, ...]:
    """Use interpretable kinematic heuristics; every result remains candidate-only."""
    labels: list[str] = []
    tangential = speed * max(0.0, math.cos(math.radians(alpha)))
    normal = speed * max(0.0, math.sin(math.radians(alpha)))
    threshold = max(1.0, p20_speed * 0.25)
    if alpha <= 20.0 and tangential >= threshold:
        labels.append("sliding")
    if alpha <= 38.0 and tangential >= threshold:
        labels.append("cutting")
    if normal >= max(5.0, normal_speed * 0.65) and 18.0 <= alpha < 58.0:
        labels.append("deformation")
    if alpha >= 45.0 and normal >= max(10.0, normal_speed * 0.8):
        labels.append("impact_pitting")
    if repeated >= 2 or impact_index > 0:
        labels.append("fatigue_candidate")
    if not labels:
        labels.append("impact_pitting" if alpha >= 45.0 else "deformation")
    return tuple(label for label in LABELS if label in labels)


def classify_hits(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Classify hit-level candidate mechanisms and calculate dimensionless risk."""
    data = inputs["arrays"]
    n = _determine_hit_count(data)
    if n == 0:
        return {"records": [], "n_hits": 0, "position_space": "none", "face_id_source": "none", "source_fields": {}}
    coordinates, position_space = _coordinates(data, n)
    face_ids, face_id_source = _hit_face_ids(data, n)
    speed_field, speed_source = _find_array(data, ("Vp_ms", "Vp", "speed_ms", "speed"), n)
    velocity = _vector_field(data, n)
    if speed_field is not None:
        speed = _finite_array(speed_field, speed_source or "speed", ndim=1)
    elif velocity is not None:
        speed = np.linalg.norm(velocity, axis=1)
        speed_source = "norm(velocity)"
    else:
        speed = np.zeros(n, dtype=np.float64)
        speed_source = "unavailable_default_zero"
    alpha_field, alpha_source = _find_array(data, ("alpha_deg", "impact_angle_deg", "angle_deg"), n)
    normals = _face_normals_for_hits(data, face_ids, n)
    if alpha_field is not None:
        alpha = np.clip(_finite_array(alpha_field, alpha_source or "alpha_deg", ndim=1), 0.0, 90.0)
        angle_source = alpha_source or "alpha_deg"
    elif velocity is not None and normals is not None:
        vnorm = np.maximum(np.linalg.norm(velocity, axis=1), 1e-12)
        normal_component = np.abs(np.sum(velocity * normals, axis=1)) / vnorm
        alpha = np.degrees(np.arcsin(np.clip(normal_component, 0.0, 1.0)))
        angle_source = "derived_from_velocity_and_normal"
    else:
        alpha = np.full(n, 45.0, dtype=np.float64)
        angle_source = "unavailable_default_45deg"
    mass_field, mass_source = _find_array(data, ("parcel_mass_kg", "particle_mass_kg", "mass_kg", "mass"), n)
    mass = _finite_array(mass_field, mass_source or "mass", ndim=1) if mass_field is not None else np.ones(n, dtype=np.float64)
    er_field, er_source = _find_array(data, ("ER_mansouri", "ER_cfd", "erosion_rate_proxy"), n)
    erosion_proxy = _finite_array(er_field, er_source or "erosion_proxy", ndim=1) if er_field is not None else np.zeros(n, dtype=np.float64)
    parcel_ids, parcel_source = _find_array(data, ("parcel_id", "particle_id"), n)
    if parcel_ids is None:
        parcel_ids = np.arange(n, dtype=np.int64)
        parcel_source = "synthetic_row_id"
    else:
        parcel_ids = np.asarray(parcel_ids, dtype=np.int64)
    impact_field, impact_source = _find_array(data, ("impact_index", "impact_number"), n)
    impact_index = np.asarray(impact_field, dtype=np.int64) if impact_field is not None else np.zeros(n, dtype=np.int64)
    time_field, time_source = _find_array(data, ("time_s", "impact_time_s", "time"), n)
    times = _finite_array(time_field, time_source or "time", ndim=1) if time_field is not None else np.full(n, np.nan)
    repeat_face, _ = _repeat_counts(face_ids)
    repeat_parcel, _ = _repeat_counts(parcel_ids)
    speed_rank = _normalize_rank(speed)
    energy = np.maximum(mass, 0.0) * np.square(np.maximum(speed, 0.0))
    energy_rank = _normalize_rank(energy)
    erosion_rank = _normalize_rank(np.maximum(erosion_proxy, 0.0))
    repeat_score = np.clip((repeat_face.astype(float) - 1.0) / max(float(repeat_face.max(initial=1)), 1.0), 0.0, 1.0)
    normal_speed = speed * np.sin(np.radians(alpha))
    p20_speed = float(np.percentile(speed, 20.0)) if len(speed) else 0.0
    status = str(inputs.get("status", "synthetic"))
    records: list[dict[str, Any]] = []
    for i in range(n):
        labels = _mechanism_labels(float(alpha[i]), float(speed[i]), p20_speed, float(normal_speed[i]), int(repeat_face[i]), int(impact_index[i]))
        risk = 100.0 * (0.40 * speed_rank[i] + 0.25 * energy_rank[i] + 0.20 * erosion_rank[i] + 0.15 * repeat_score[i])
        if status == "blocked":
            risk = 0.0
        record = {
            "hit_index": i,
            "face_id": int(face_ids[i]),
            "parcel_id": int(parcel_ids[i]),
            "impact_index": int(impact_index[i]),
            "time_s": None if not np.isfinite(times[i]) else float(times[i]),
            "x": float(coordinates[i, 0]), "y": float(coordinates[i, 1]), "z": float(coordinates[i, 2]),
            "position_space": position_space,
            "speed_ms": float(speed[i]), "alpha_deg": float(alpha[i]),
            "normal_speed_ms": float(normal_speed[i]),
            "tangential_speed_ms": float(speed[i] * np.cos(np.radians(alpha[i]))),
            "mass_kg": float(mass[i]), "erosion_model_proxy": float(erosion_proxy[i]),
            "repeat_face_count": int(repeat_face[i]), "repeat_parcel_count": int(repeat_parcel[i]),
            "risk_score": float(np.clip(risk, 0.0, 100.0)),
            "candidate_labels": list(labels), "candidate": True,
            "label_state": "blocked_candidate" if status == "blocked" else "candidate",
            "status": status, "claim_scope": CLAIM_SCOPE,
        }
        records.append(record)
    return {
        "records": records, "n_hits": n, "position_space": position_space,
        "face_id_source": face_id_source,
        "source_fields": {"speed": speed_source, "angle": angle_source, "mass": mass_source or "default_unit_mass", "erosion_proxy": er_source or "unavailable", "parcel_id": parcel_source, "impact_index": impact_source or "default_zero"},
    }


def _labels_text(labels: Iterable[str]) -> str:
    return "|".join(label for label in LABELS if label in set(labels))


def _aggregate_labels(records: list[Mapping[str, Any]]) -> list[str]:
    found = {label for row in records for label in row.get("candidate_labels", [])}
    return [label for label in LABELS if label in found]


def _face_table(data: Mapping[str, np.ndarray], face_ids: np.ndarray, hit_records: list[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, np.ndarray]]:
    centers, _ = _find_array(data, ("face_centers_local_mm", "face_centers_mm", "face_centers"))
    centers = None if centers is None else _finite_array(centers, "face centers", ndim=2)
    if centers is not None and centers.shape[1] != 3:
        centers = None
    nface = len(centers) if centers is not None else 0
    if nface:
        all_ids = np.arange(nface, dtype=np.int64)
        # Existing Step30 face arrays are indexed by parent_face_id.
        if len(face_ids) and (int(face_ids.min()) < 0 or int(face_ids.max()) >= nface):
            all_ids = np.unique(face_ids)
            centers = None
    else:
        all_ids = np.unique(face_ids) if len(face_ids) else np.empty(0, dtype=np.int64)
    area, _ = _find_array(data, ("face_area_m2", "area_m2"))
    face_area = _finite_array(area, "face_area_m2", ndim=1) if area is not None and len(area) == nface else None
    provided: dict[str, np.ndarray] = {}
    for output, names in (("ER_mass_ratio_face", ("ER_mass_ratio_face",)), ("dhdt_face_mm_h", ("dhdt_face_mm_h",)), ("sediment_mass_flux_face_kg_m2_s", ("sediment_mass_flux_face_kg_m2_s",))):
        value, _ = _find_array(data, names)
        if value is not None and len(value) == nface:
            provided[output] = _finite_array(value, output, ndim=1)
    return all_ids, centers, face_area, provided


def aggregate_face_risk(inputs: Mapping[str, Any], hit_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    data = inputs["arrays"]
    records = list(hit_result["records"])
    face_ids = np.asarray([row["face_id"] for row in records], dtype=np.int64)
    all_ids, centers, face_area, provided = _face_table(data, face_ids, records)
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[int(row["face_id"])].append(row)
    status = str(inputs.get("status", "synthetic"))
    result: list[dict[str, Any]] = []
    for row_index, face_id in enumerate(all_ids):
        hits = grouped.get(int(face_id), [])
        scores = [float(row["risk_score"]) for row in hits]
        labels = _aggregate_labels(hits)
        output: dict[str, Any] = {
            "face_id": int(face_id), "face_row_index": row_index,
            "x": None, "y": None, "z": None,
            "hit_count": len(hits),
            "risk_score": float(max(scores) if scores else 0.0),
            "risk_mean": float(np.mean(scores)) if scores else 0.0,
            "candidate_labels": labels, "candidate": True,
            "label_state": "blocked_candidate" if status == "blocked" else ("candidate" if hits else "candidate_unexposed"),
            "status": status, "claim_scope": CLAIM_SCOPE,
        }
        if centers is not None and row_index < len(centers):
            output.update({"x": float(centers[row_index, 0]), "y": float(centers[row_index, 1]), "z": float(centers[row_index, 2])})
        elif hits:
            output.update({key: float(np.mean([row[key] for row in hits])) for key in ("x", "y", "z")})
        if face_area is not None:
            output["face_area_m2"] = float(face_area[row_index])
        for key, values in provided.items():
            output[key] = float(values[row_index])
        result.append(output)
    return result


def aggregate_spatial_risk(hit_result: Mapping[str, Any], *, bins: int = 24) -> list[dict[str, Any]]:
    records = list(hit_result["records"])
    if not records:
        return []
    bins = max(1, int(bins))
    points = np.asarray([[row["x"], row["y"]] for row in records], dtype=np.float64)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    x_edges = np.linspace(x_min - 1e-9 * x_span, x_max + 1e-9 * x_span, bins + 1)
    y_edges = np.linspace(y_min - 1e-9 * y_span, y_max + 1e-9 * y_span, bins + 1)
    ix = np.clip(np.searchsorted(x_edges, points[:, 0], side="right") - 1, 0, bins - 1)
    iy = np.clip(np.searchsorted(y_edges, points[:, 1], side="right") - 1, 0, bins - 1)
    groups: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for k, row in enumerate(records):
        groups[(int(ix[k]), int(iy[k]))].append(row)
    status = str(records[0].get("status", "synthetic"))
    result: list[dict[str, Any]] = []
    for (x_bin, y_bin), group in sorted(groups.items()):
        scores = [float(row["risk_score"]) for row in group]
        result.append({
            "x_bin": x_bin, "y_bin": y_bin,
            "x_center": float((x_edges[x_bin] + x_edges[x_bin + 1]) / 2.0),
            "y_center": float((y_edges[y_bin] + y_edges[y_bin + 1]) / 2.0),
            "hit_count": len(group),
            "risk_score": float(max(scores)), "risk_mean": float(np.mean(scores)),
            "candidate_labels": _aggregate_labels(group), "candidate": True,
            "label_state": "blocked_candidate" if status == "blocked" else "candidate",
            "status": status, "claim_scope": CLAIM_SCOPE,
        })
    return result


HIT_COLUMNS = ("hit_index", "face_id", "parcel_id", "impact_index", "time_s", "x", "y", "z", "position_space", "speed_ms", "alpha_deg", "normal_speed_ms", "tangential_speed_ms", "mass_kg", "erosion_model_proxy", "repeat_face_count", "repeat_parcel_count", "risk_score", "candidate_labels", "candidate", "label_state", "status", "claim_scope")
FACE_COLUMNS = ("face_id", "face_row_index", "x", "y", "z", "face_area_m2", "hit_count", "risk_score", "risk_mean", "ER_mass_ratio_face", "dhdt_face_mm_h", "sediment_mass_flux_face_kg_m2_s", "candidate_labels", "candidate", "label_state", "status", "claim_scope")
SPATIAL_COLUMNS = ("x_bin", "y_bin", "x_center", "y_center", "hit_count", "risk_score", "risk_mean", "candidate_labels", "candidate", "label_state", "status", "claim_scope")


def build_result(inputs: Mapping[str, Any], *, spatial_bins: int = 24) -> dict[str, Any]:
    hit_result = classify_hits(inputs)
    faces = aggregate_face_risk(inputs, hit_result)
    spatial = aggregate_spatial_risk(hit_result, bins=spatial_bins)
    records = hit_result["records"]
    label_counts = Counter(label for row in records for label in row["candidate_labels"])
    return {
        "version": VERSION,
        "provenance": {
            "status": inputs["status"], "status_source": inputs["status_source"],
            "input_status_explicit": inputs["status_source"] != "default:simulation-or-unlabelled-input",
            "candidate": True, "claim_scope": CLAIM_SCOPE,
            "measured_damage_claim": False,
            "note": "Labels are kinematic/model-based candidates. They do not establish measured erosion, pitting, deformation, fatigue, cutting, or sliding damage.",
        },
        "labels": list(LABELS),
        "inputs": {
            "hits_file": str(inputs["hits_path"]), "summary_file": str(inputs["summary_path"]),
            "hits_sha256": inputs["input_hashes"]["hits_sha256"], "summary_sha256": inputs["input_hashes"]["summary_sha256"],
            "summary_version": inputs["summary"].get("version"), "immutable_inputs_read_only": True,
        },
        "summary": {
            "n_hits": int(hit_result["n_hits"]), "n_faces": len(faces), "n_spatial_cells": len(spatial),
            "candidate_label_counts": {label: int(label_counts.get(label, 0)) for label in LABELS},
            "position_space": hit_result["position_space"], "face_id_source": hit_result["face_id_source"],
            "source_fields": hit_result["source_fields"],
        },
        "hit_records": records, "face_records": faces, "spatial_records": spatial,
    }


def write_outputs(result: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write CSV/JSON artifacts; the input paths are never opened for writing."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    hit_rows = list(result["hit_records"])
    face_rows = list(result["face_records"])
    spatial_rows = list(result["spatial_records"])
    paths = {
        "mechanism_csv": root / "mechanism_labels.csv",
        "mechanism_json": root / "mechanism_classification.json",
        "face_csv": root / "face_risk.csv",
        "face_json": root / "face_risk.json",
        "spatial_csv": root / "spatial_risk.csv",
        "spatial_json": root / "spatial_risk.json",
    }
    _write_csv(paths["mechanism_csv"], hit_rows, HIT_COLUMNS)
    _write_csv(paths["face_csv"], face_rows, FACE_COLUMNS)
    _write_csv(paths["spatial_csv"], spatial_rows, SPATIAL_COLUMNS)
    # Keep all aggregation levels in the primary JSON so the standalone map
    # renderer cannot silently downgrade to hit-point rendering.
    _write_json(paths["mechanism_json"], result)
    _write_json(paths["face_json"], {"version": VERSION, "provenance": result["provenance"], "inputs": result["inputs"], "records": face_rows}, pretty=False)
    _write_json(paths["spatial_json"], {"version": VERSION, "provenance": result["provenance"], "inputs": result["inputs"], "records": spatial_rows}, pretty=False)
    return {key: str(value) for key, value in paths.items()}


def classify_and_write(hits_path: str | Path, summary_path: str | Path, output_dir: str | Path, *, spatial_bins: int = 24) -> dict[str, Any]:
    inputs = load_inputs(hits_path, summary_path)
    result = build_result(inputs, spatial_bins=spatial_bins)
    result["outputs"] = write_outputs(result, output_dir)
    return result


def selftest() -> bool:
    """Exercise legacy fields, CFD fields, provenance labels, and all outputs."""
    with tempfile.TemporaryDirectory(prefix="step30_mechanism_selftest_") as directory:
        root = Path(directory)
        hits = root / "particle_hits.npz"
        summary = root / "summary.json"
        out = root / "out"
        np.savez_compressed(
            hits,
            hit_pos_global=np.asarray([[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 2, 0]], dtype=np.float64),
            Vp=np.asarray([10, 20, 40, 80], dtype=np.float64),
            alpha_deg=np.asarray([5, 25, 55, 75], dtype=np.float64),
            parent_face_id=np.asarray([0, 0, 1, 2], dtype=np.int64),
            parcel_id=np.asarray([0, 0, 1, 1], dtype=np.int64),
            impact_index=np.asarray([0, 1, 0, 1], dtype=np.int64),
        )
        summary.write_text(json.dumps({"version": "selftest", "status": "synthetic"}), encoding="utf-8")
        result = classify_and_write(hits, summary, out, spatial_bins=4)
        assert result["provenance"]["status"] == "synthetic"
        assert result["provenance"]["measured_damage_claim"] is False
        found = {label for row in result["hit_records"] for label in row["candidate_labels"]}
        assert {"sliding", "cutting", "deformation", "impact_pitting", "fatigue_candidate"}.issubset(found)
        for path in result["outputs"].values():
            assert Path(path).is_file()
        summary.write_text(json.dumps({"version": "selftest", "provenance": {"status": "blocked"}}), encoding="utf-8")
        blocked = build_result(load_inputs(hits, summary), spatial_bins=2)
        assert blocked["provenance"]["status"] == "blocked"
        assert all(row["status"] == "blocked" and row["candidate"] for row in blocked["hit_records"])
    return True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify candidate erosion mechanisms from immutable particle hits")
    parser.add_argument("--hits", help="immutable particle_hits.npz")
    parser.add_argument("--summary", help="paired immutable summary.json")
    parser.add_argument("--output-dir", help="directory for CSV and JSON outputs")
    parser.add_argument("--spatial-bins", type=int, default=24)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.selftest:
        selftest()
        print("step30_erosion_mechanism_classifier selftest: PASS")
        return 0
    if not (args.hits and args.summary and args.output_dir):
        _parser().error("--hits, --summary, and --output-dir are required unless --selftest is used")
    result = classify_and_write(args.hits, args.summary, args.output_dir, spatial_bins=args.spatial_bins)
    print(json.dumps({"status": result["provenance"]["status"], "candidate": True, "outputs": result["outputs"]}, ensure_ascii=False, indent=2))
    return 0


# Short aliases keep the module convenient for downstream Step30 scripts.
load_hits = load_inputs
classify_mechanisms = classify_hits
classify = classify_and_write


if __name__ == "__main__":
    raise SystemExit(main())
