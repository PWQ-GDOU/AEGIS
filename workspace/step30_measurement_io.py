"""Measured/synthetic input readers and immutable structured records for Step30.

Supported inputs are point clouds (NPY, XYZ/TXT/CSV, PLY, STL), gridded or
scattered depth fields, sparse thickness/erosion tables, and material-test
tables.  Readers retain source hashes and never infer official accuracy from
synthetic inputs.
"""

from __future__ import annotations

import argparse
import csv
import re
import struct
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.io import loadmat

from step30_validation_common import (
    ValidationError,
    build_manifest,
    canonical_sha256,
    finite_array,
    jsonable,
    manifest_entry,
    normalize_unit,
    require_status,
    safe_json_dump,
    safe_json_load,
    safe_npz_dump,
    safe_npz_load,
    sha256_file,
    validate_units,
)


MEASUREMENT_VERSION = "step30-measurement-record-v1"
_POINT_EXTENSIONS = {".npy", ".xyz", ".txt", ".csv", ".ply", ".stl"}


def _source_record(path: str | Path, *, status: str, kind: str, units: Mapping[str, str] | None) -> dict[str, Any]:
    target = Path(path)
    require_status(status)
    if not target.is_file():
        raise FileNotFoundError(target)
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "bytes": int(target.stat().st_size),
        "status": status,
        "kind": kind,
        "units": {name: normalize_unit(unit) for name, unit in (units or {}).items()},
    }


def _check_columns(values: np.ndarray, name: str, n_columns: int) -> np.ndarray:
    array = finite_array(values, name, ndim=2)
    if array.shape[1] < n_columns:
        raise ValidationError(f"{name} requires at least {n_columns} columns")
    return array


def _load_text_table(path: Path, *, delimiter: str | None = None, skiprows: int = 0) -> np.ndarray:
    try:
        values = np.genfromtxt(
            str(path), delimiter=delimiter, comments="#", skip_header=skiprows,
            dtype=np.float64, invalid_raise=True,
        )
    except (OSError, ValueError) as exc:
        raise ValidationError(f"cannot parse numeric table: {path}") from exc
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return finite_array(values, f"table {path.name}", ndim=2)


def _read_npy(path: Path) -> np.ndarray:
    try:
        values = np.load(str(path), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValidationError(f"cannot read NPY: {path}") from exc
    return _check_columns(values, path.name, 3)


def _read_ply(path: Path) -> np.ndarray:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValidationError(f"cannot read PLY: {path}") from exc
    marker = data.find(b"end_header\n")
    line_ending = b"\n"
    if marker < 0:
        marker = data.find(b"end_header\r\n")
        line_ending = b"\r\n"
    if marker < 0:
        raise ValidationError("PLY has no end_header")
    header_bytes = data[: marker + len(b"end_header")]
    header = header_bytes.decode("ascii", errors="strict").splitlines()
    if not header or header[0].strip() != "ply":
        raise ValidationError("invalid PLY header")
    format_line = next((line for line in header if line.startswith("format ")), "")
    vertex_line = next((line for line in header if line.startswith("element vertex ")), "")
    if not vertex_line:
        raise ValidationError("PLY has no vertex element")
    n_vertices = int(vertex_line.split()[2])
    vertex_header_index = header.index(vertex_line)
    next_element_index = next(
        (index for index in range(vertex_header_index + 1, len(header)) if header[index].startswith("element ")),
        len(header),
    )
    properties = [
        line.split()
        for line in header[vertex_header_index + 1:next_element_index]
        if line.startswith("property ")
    ]
    xyz_indices = []
    for coordinate in ("x", "y", "z"):
        matches = [index for index, parts in enumerate(properties) if len(parts) >= 3 and parts[-1] == coordinate]
        if not matches:
            raise ValidationError(f"PLY has no {coordinate} vertex property")
        xyz_indices.append(matches[0])
    body_offset = marker + len(b"end_header") + len(line_ending)
    if format_line.startswith("format ascii"):
        lines = data[body_offset:].decode("ascii", errors="strict").splitlines()
        if len(lines) < n_vertices:
            raise ValidationError("PLY has fewer vertices than declared")
        rows = [[float(lines[row].split()[index]) for index in xyz_indices] for row in range(n_vertices)]
        return finite_array(rows, "PLY vertices", ndim=2, shape=(n_vertices, 3))
    if format_line.startswith("format binary_little_endian"):
        type_sizes = {"char": ("b", 1), "uchar": ("B", 1), "short": ("h", 2), "ushort": ("H", 2), "int": ("i", 4), "uint": ("I", 4), "float": ("f", 4), "double": ("d", 8)}
        if any(len(parts) < 3 or parts[1] not in type_sizes for parts in properties):
            raise ValidationError("unsupported binary PLY vertex property")
        formats = ["<" + type_sizes[parts[1]][0] for parts in properties]
        stride = sum(type_sizes[parts[1]][1] for parts in properties)
        rows = []
        for row in range(n_vertices):
            start = body_offset + row * stride
            values = [struct.unpack_from(formats[index], data, start + sum(type_sizes[item[1]][1] for item in properties[:index]))[0] for index in xyz_indices]
            rows.append(values)
        return finite_array(rows, "PLY vertices", ndim=2, shape=(n_vertices, 3))
    raise ValidationError(f"unsupported PLY format: {format_line}")


def _read_stl(path: Path) -> np.ndarray:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValidationError(f"cannot read STL: {path}") from exc
    # Binary STL records are exactly 50 bytes: normal (12), three vertices
    # (36), and a 2-byte attribute field.  Never reshape across that field.
    if len(data) >= 84:
        try:
            n_triangles = struct.unpack_from("<I", data, 80)[0]
        except struct.error as exc:
            raise ValidationError(f"invalid STL header: {path}") from exc
        if 84 + 50 * n_triangles == len(data):
            vertices = []
            for triangle in range(n_triangles):
                offset = 84 + triangle * 50 + 12
                try:
                    vertices.extend(struct.unpack_from("<9f", data, offset))
                except struct.error as exc:
                    raise ValidationError(f"truncated binary STL: {path}") from exc
            values = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
            return finite_array(values, "STL vertices", ndim=2, shape=(n_triangles * 3, 3))
    try:
        text = data.decode("utf-8", errors="replace")
    except UnicodeError as exc:
        raise ValidationError(f"cannot decode ASCII STL: {path}") from exc
    rows = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            rows.append([float(item) for item in parts[1:]])
    if not rows or len(rows) % 3:
        raise ValidationError(f"cannot parse STL triangles: {path}")
    return finite_array(rows, "STL vertices", ndim=2, shape=(len(rows), 3))


def read_point_cloud(
    path: str | Path,
    *,
    status: str = "measured",
    units: str = "mm",
    delimiter: str | None = None,
    skiprows: int = 0,
    columns: tuple[int, int, int] = (0, 1, 2),
) -> dict[str, Any]:
    """Read a point cloud and return normalized coordinates plus provenance."""
    target = Path(path)
    require_status(status)
    coordinate_units = validate_units(units, expected_dimension="length")
    extension = target.suffix.lower()
    if extension not in _POINT_EXTENSIONS:
        raise ValidationError(f"unsupported point-cloud extension: {target.suffix}")
    if extension == ".npy":
        values = _read_npy(target)
        if len(columns) != 3 or max(columns) >= values.shape[1]:
            raise ValidationError("point-cloud columns are outside the input array")
        values = values[:, columns]
    elif extension == ".ply":
        values = _read_ply(target)
    elif extension == ".stl":
        values = _read_stl(target)
    else:
        try:
            values = _load_text_table(target, delimiter=delimiter or ("," if extension == ".csv" else None), skiprows=skiprows)
            if len(columns) != 3 or max(columns) >= values.shape[1]:
                raise ValidationError("point-cloud columns are outside the input table")
            values = values[:, columns]
        except ValidationError:
            headers, named_values = _parse_named_table(target, delimiter=delimiter)
            xyz_columns = [
                _find_column(headers, ("x", "x_mm", "x_m")),
                _find_column(headers, ("y", "y_mm", "y_m")),
                _find_column(headers, ("z", "z_mm", "z_m")),
            ]
            if any(column is None for column in xyz_columns):
                raise
            values = named_values[:, xyz_columns]
    points = finite_array(values, "points", ndim=2, shape=(len(values), 3))
    if len(points) < 3:
        raise ValidationError("point cloud requires at least 3 points")
    return {
        "version": MEASUREMENT_VERSION,
        "kind": "point_cloud",
        "status": status,
        "units": {"coordinates": coordinate_units},
        "points": points,
        "n_points": int(len(points)),
        "source": _source_record(target, status=status, kind="point_cloud", units={"coordinates": coordinate_units}),
        "provenance_note": "synthetic result; not official accuracy" if status == "synthetic" else None,
    }


def _parse_named_table(path: Path, *, delimiter: str | None = None) -> tuple[list[str], np.ndarray]:
    if delimiter is None:
        delimiter = "," if path.suffix.lower() == ".csv" else None
    try:
        if delimiter is None:
            text = path.read_text(encoding="utf-8")
            rows = [line.split() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        else:
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.reader(stream, delimiter=delimiter)
                rows = [row for row in reader if row and not row[0].lstrip().startswith("#")]
        if not rows:
            raise ValidationError(f"empty table: {path}")
        try:
            numeric_rows = [[float(item) for item in row] for row in rows]
        except ValueError:
            headers = [item.strip() for item in rows[0]]
            data_rows = rows[1:]
        else:
            headers = [f"column_{index}" for index in range(len(rows[0]))]
            data_rows = numeric_rows
        if not data_rows:
            raise ValidationError(f"table has no data rows: {path}")
        values = np.asarray(
            [[float(item) for item in row] for row in data_rows], dtype=np.float64
        )
    except (OSError, ValueError) as exc:
        raise ValidationError(f"cannot parse named table: {path}") from exc
    if values.ndim != 2 or values.shape[1] != len(headers):
        raise ValidationError("table rows do not match header columns")
    return headers, finite_array(values, f"table {path.name}", ndim=2)


def _find_column(headers: Iterable[str], names: Iterable[str]) -> int | None:
    normalized = {re.sub(r"[^a-z0-9]", "", name.lower()): index for index, name in enumerate(headers)}
    for name in names:
        if re.sub(r"[^a-z0-9]", "", name.lower()) in normalized:
            return normalized[re.sub(r"[^a-z0-9]", "", name.lower())]
    return None


def read_depth_field(
    path: str | Path,
    *,
    status: str = "measured",
    coordinate_units: str = "mm",
    depth_units: str = "mm",
    delimiter: str | None = None,
    depth_column: int | None = None,
) -> dict[str, Any]:
    """Read a depth field from NPY or a numeric table as x,y,z,depth."""
    target = Path(path)
    require_status(status)
    coordinate_units = validate_units(coordinate_units, expected_dimension="length")
    depth_units = validate_units(depth_units, expected_dimension="length")
    if target.suffix.lower() == ".npy":
        try:
            values = np.asarray(np.load(str(target), allow_pickle=False), dtype=np.float64)
        except (OSError, ValueError) as exc:
            raise ValidationError(f"cannot read depth NPY: {target}") from exc
        if values.ndim == 1 or (values.ndim == 2 and values.shape[1] != 4):
            grid_values = finite_array(values, "depth grid", nonempty=True)
            return {
                "version": MEASUREMENT_VERSION,
                "kind": "depth_field",
                "status": status,
                "units": {"depth": depth_units, "grid": "dimensionless"},
                "grid_values": grid_values,
                "grid_shape": list(grid_values.shape),
                "n_samples": int(grid_values.size),
                "source": _source_record(target, status=status, kind="depth_field", units={"depth": depth_units}),
            }
        values = _check_columns(values, "depth field", 4)
        if depth_column is None:
            depth_column = 3
        if isinstance(depth_column, bool) or not isinstance(depth_column, (int, np.integer)):
            raise ValidationError("depth_column must be an integer")
        if depth_column < 0 or depth_column >= values.shape[1]:
            raise ValidationError("depth_column is outside the depth array")
        coordinates = values[:, :3]
        depths = values[:, depth_column]
    else:
        headers, values = _parse_named_table(target, delimiter=delimiter)
        x_column = _find_column(headers, ("x", "x_mm", "x_m"))
        y_column = _find_column(headers, ("y", "y_mm", "y_m"))
        z_column = _find_column(headers, ("z", "z_mm", "z_m"))
        if depth_column is None:
            depth_column = _find_column(headers, ("depth", "depth_mm", "depth_m", "erosion", "value"))
        elif isinstance(depth_column, bool) or not isinstance(depth_column, (int, np.integer)):
            raise ValidationError("depth_column must be an integer")
        if depth_column is not None and (depth_column < 0 or depth_column >= values.shape[1]):
            raise ValidationError("depth_column is outside the depth table")
        if None in (x_column, y_column, z_column, depth_column):
            if values.shape[1] < 4:
                raise ValidationError("depth table requires x,y,z,depth columns")
            x_column, y_column, z_column, depth_column = 0, 1, 2, 3
        coordinates = values[:, [x_column, y_column, z_column]]
        depths = values[:, depth_column]
    coordinates = finite_array(coordinates, "depth coordinates", ndim=2, shape=(len(coordinates), 3))
    depths = finite_array(depths, "depth values", ndim=1, shape=(len(coordinates),), nonempty=True)
    return {
        "version": MEASUREMENT_VERSION,
        "kind": "depth_field",
        "status": status,
        "units": {"coordinates": coordinate_units, "depth": depth_units},
        "coordinates": coordinates,
        "values": depths,
        "n_samples": int(len(depths)),
        "source": _source_record(target, status=status, kind="depth_field", units={"coordinates": coordinate_units, "depth": depth_units}),
    }


def read_sparse_measurement_table(
    path: str | Path,
    *,
    kind: str = "erosion",
    status: str = "measured",
    coordinate_units: str = "mm",
    value_units: str = "mm",
    delimiter: str | None = None,
) -> dict[str, Any]:
    """Read sparse thickness or erosion observations as x,y,z,value."""
    if kind not in {"thickness", "erosion"}:
        raise ValidationError("kind must be thickness or erosion")
    target = Path(path)
    require_status(status)
    coordinate_units = validate_units(coordinate_units, expected_dimension="length")
    value_units = validate_units(value_units, expected_dimension="length")
    headers, table = _parse_named_table(target, delimiter=delimiter)
    columns = [_find_column(headers, names) for names in (("x", "x_mm", "x_m"), ("y", "y_mm", "y_m"), ("z", "z_mm", "z_m"), (kind, "value", "measurement", "thickness_mm", "erosion_mm"))]
    if any(column is None for column in columns):
        if table.shape[1] < 4:
            raise ValidationError(f"{kind} table requires x,y,z,value columns")
        columns = [0, 1, 2, 3]
    coordinates = finite_array(table[:, columns[:3]], "measurement coordinates", ndim=2, shape=(len(table), 3))
    values = finite_array(table[:, columns[3]], "measurement values", ndim=1, shape=(len(table),), nonempty=True)
    return {
        "version": MEASUREMENT_VERSION,
        "kind": kind,
        "status": status,
        "units": {"coordinates": coordinate_units, "value": value_units},
        "coordinates": coordinates,
        "values": values,
        "n_samples": int(len(values)),
        "source": _source_record(target, status=status, kind=kind, units={"coordinates": coordinate_units, "value": value_units}),
    }


def read_material_test_table(
    path: str | Path,
    *,
    status: str = "measured",
    delimiter: str | None = None,
    units: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Read a material test table with named columns and explicit units."""
    target = Path(path)
    require_status(status)
    headers, table = _parse_named_table(target, delimiter=delimiter)
    if table.shape[0] == 0:
        raise ValidationError("material test table has no observations")
    table_units = {name: normalize_unit(unit) for name, unit in (units or {}).items()}
    unknown = sorted(set(table_units) - set(headers))
    if unknown:
        raise ValidationError(f"material unit columns are absent: {unknown}")
    return {
        "version": MEASUREMENT_VERSION,
        "kind": "material_test",
        "status": status,
        "units": table_units,
        "columns": headers,
        "values": table,
        "n_samples": int(len(table)),
        "source": _source_record(target, status=status, kind="material_test", units=table_units),
    }


def read_measurement(
    path: str | Path,
    *,
    kind: str,
    status: str = "measured",
    **kwargs: Any,
) -> dict[str, Any]:
    """Dispatch to one of the supported measurement readers."""
    if kind == "point_cloud":
        return read_point_cloud(path, status=status, **kwargs)
    if kind == "depth_field":
        return read_depth_field(path, status=status, **kwargs)
    if kind in {"thickness", "erosion"}:
        return read_sparse_measurement_table(path, kind=kind, status=status, **kwargs)
    if kind == "material_test":
        return read_material_test_table(path, status=status, **kwargs)
    raise ValidationError(f"unsupported measurement kind: {kind}")


def _record_arrays(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    arrays = {}
    for key in ("points", "coordinates", "values", "grid_values"):
        if key in record:
            arrays[key] = np.asarray(record[key])
    if "columns" in record:
        # Header metadata remains JSON; values are the only material array.
        pass
    return arrays


def write_measurement_record(
    record: Mapping[str, Any],
    output_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    """Persist an input record as immutable NPZ plus optional JSON metadata."""
    status = require_status(record.get("status"))
    if record.get("version") != MEASUREMENT_VERSION:
        raise ValidationError("unsupported measurement record version")
    source = record.get("source")
    if not isinstance(source, Mapping) or not isinstance(source.get("sha256"), str):
        raise ValidationError("measurement record must retain source hash")
    arrays = _record_arrays(record)
    if not arrays:
        raise ValidationError("measurement record has no numeric arrays")
    metadata = {key: value for key, value in record.items() if key not in arrays}
    metadata["status"] = status
    metadata["record_sha256"] = canonical_sha256(record)
    result = safe_npz_dump(arrays, output_path, metadata=metadata)
    if metadata_path is not None:
        result["metadata"] = safe_json_dump(metadata, metadata_path)
    return result


def load_measurement_record(path: str | Path) -> dict[str, Any]:
    arrays, metadata = safe_npz_load(path)
    if metadata.get("version") != MEASUREMENT_VERSION:
        raise ValidationError("unsupported measurement record version")
    require_status(metadata.get("status"))
    result = dict(metadata)
    result.update(arrays)
    stored_hash = result.pop("record_sha256", None)
    if not isinstance(stored_hash, str) or len(stored_hash) != 64:
        raise ValidationError("measurement record has no valid record_sha256")
    if canonical_sha256(result) != stored_hash:
        raise ValidationError("measurement record canonical hash mismatch")
    result["record_sha256"] = stored_hash
    return result


def write_measurement_manifest(
    records: Mapping[str, Mapping[str, Any]],
    output_path: str | Path,
    *,
    status: str,
) -> dict[str, Any]:
    """Build and immutably save a manifest over structured records."""
    entries = {}
    for name, record in records.items():
        record_status = require_status(record.get("status"))
        if record_status != status:
            raise ValidationError(f"record {name} status differs from manifest status")
        source = record.get("source")
        if not isinstance(source, Mapping) or not source.get("sha256"):
            raise ValidationError(f"record {name} has no source hash")
        source_entry = dict(source)
        source_path = Path(str(source_entry.get("path", ""))).resolve()
        manifest_root = Path(output_path).resolve().parent
        try:
            source_entry["path"] = str(source_path.relative_to(manifest_root))
        except ValueError:
            # External measurements remain auditable without pretending that a
            # basename identifies the source.  The common validator checks the
            # absolute path and hash only for explicitly external entries.
            source_entry["path"] = source_path.name
            source_entry["external"] = True
            source_entry["absolute_path"] = str(source_path)
        source_entry.update({
            "kind": record.get("kind"),
            "status": record_status,
            "n_samples": record.get("n_samples", record.get("n_points")),
        })
        entries[name] = source_entry
    manifest = build_manifest(version="step30-measurement-manifest-v1", entries=entries, status=status)
    safe_json_dump(manifest, output_path)
    return manifest


def _selftest() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        points_path = root / "points.xyz"
        points_path.write_text("0 0 0\n1 0 0\n0 1 0\n", encoding="utf-8")
        record = read_point_cloud(points_path, status="measured", units="mm")
        assert record["n_points"] == 3
        assert len(record["source"]["sha256"]) == 64
        saved = root / "record.npz"
        write_measurement_record(record, saved)
        loaded = load_measurement_record(saved)
        assert loaded["status"] == "measured"
        assert np.allclose(loaded["points"], record["points"])
        try:
            write_measurement_record(record, saved)
        except FileExistsError:
            pass
        else:
            raise AssertionError("measurement record was overwritten")
        table = root / "erosion.csv"
        table.write_text("x,y,z,erosion\n0,0,0,0.1\n", encoding="utf-8")
        sparse = read_sparse_measurement_table(table, kind="erosion")
        assert sparse["n_samples"] == 1


def selftest() -> bool:
    _selftest()
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step30 measurement I/O selftest")
    parser.add_argument("--selftest", action="store_true", help="run the minimal selftest")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        print("step30_measurement_io selftest: PASS")
    else:
        parser.print_help()
