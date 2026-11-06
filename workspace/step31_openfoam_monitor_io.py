"""Restart-aware OpenFOAM monitor parsing and time-weighted statistics."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


VERSION = "step31-openfoam-monitor-io-v1"
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_TIME = re.compile(r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")
_VECTOR = re.compile(r"\(([^()]*)\)")


class MonitorParseError(ValueError):
    """Raised when a monitor cannot be parsed without guessing semantics."""


class DuplicateTimeConflict(MonitorParseError):
    """Raised when restart segments provide different values at one timestamp."""

    def __init__(self, label: str, time: str, first_segment: str, second_segment: str):
        super().__init__(
            f"{label} has conflicting duplicate values at time {time}: "
            f"{first_segment} vs {second_segment}"
        )
        self.label = label
        self.time = time
        self.first_segment = first_segment
        self.second_segment = second_segment


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _finite_array(value: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MonitorParseError(f"{name} is not numeric") from exc
    if ndim is not None and array.ndim != ndim:
        raise MonitorParseError(f"{name} must have ndim={ndim}")
    if not np.isfinite(array).all():
        raise MonitorParseError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class SegmentLineage:
    path: str
    sha256: str
    bytes: int
    first_time: float
    last_time: float
    rows_parsed: int
    duplicate_rows: int = 0


@dataclass(frozen=True)
class MonitorSeries:
    """One deduplicated SI monitor series with immutable source lineage."""

    name: str
    kind: str
    units: str
    times: np.ndarray
    values: np.ndarray
    columns: tuple[str, ...]
    segments: tuple[SegmentLineage, ...] = field(default_factory=tuple)
    source: str = "postProcessing"
    series_sha256: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise MonitorParseError("monitor name must not be empty")
        if not isinstance(self.kind, str) or not self.kind:
            raise MonitorParseError("monitor kind must not be empty")
        if not isinstance(self.units, str) or not self.units:
            raise MonitorParseError("monitor units must not be empty")
        times = _finite_array(self.times, f"{self.name}.times", ndim=1)
        values = _finite_array(self.values, f"{self.name}.values")
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or len(values) != len(times) or len(times) == 0:
            raise MonitorParseError(f"{self.name} has inconsistent or empty times and values")
        if np.any(np.diff(times) <= 0.0):
            raise MonitorParseError(f"{self.name} times must be strictly increasing after deduplication")
        columns = tuple(self.columns)
        if len(columns) != values.shape[1] or len(set(columns)) != len(columns):
            raise MonitorParseError(f"{self.name} columns do not match monitor width")
        times.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "columns", columns)
        identity = {
            "version": VERSION,
            "name": self.name,
            "kind": self.kind,
            "units": self.units,
            "times": times.tolist(),
            "values": values.tolist(),
            "columns": list(columns),
            "segments": [segment.__dict__ for segment in self.segments],
            "source": self.source,
        }
        object.__setattr__(self, "series_sha256", _canonical_sha256(identity))

    def column(self, name: str) -> np.ndarray:
        try:
            index = self.columns.index(name)
        except ValueError as exc:
            raise MonitorParseError(f"{self.name} has no {name!r} column") from exc
        return self.values[:, index]

    def to_lineage_dict(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "name": self.name,
            "kind": self.kind,
            "units": self.units,
            "source": self.source,
            "series_sha256": self.series_sha256,
            "segments": [segment.__dict__ for segment in self.segments],
            "sample_count": int(len(self.times)),
            "start_time": float(self.times[0]),
            "end_time": float(self.times[-1]),
        }


@dataclass(frozen=True)
class MonitorBundle:
    status: str
    monitors: Mapping[str, MonitorSeries]
    conflicts: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    fallback_used: tuple[str, ...] = ()

    @property
    def blocked(self) -> bool:
        return self.status == "blocked"


def _paths(value: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(value, (str, Path)):
        result = [Path(value)]
    else:
        result = [Path(item) for item in value]
    if not result:
        raise MonitorParseError("no monitor paths were supplied")
    missing = [str(path) for path in result if not path.is_file()]
    if missing:
        raise MonitorParseError(f"monitor files do not exist: {missing}")
    return result


def _decimal_time(line: str) -> tuple[Decimal, str] | None:
    match = _TIME.match(line)
    if not match:
        return None
    try:
        value = Decimal(match.group(1))
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    return value, match.group(1)


def _vectors_after_time(line: str) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for body in _VECTOR.findall(line):
        values = [float(item) for item in _NUMBER.findall(body)]
        if len(values) == 3 and np.isfinite(values).all():
            result.append(np.asarray(values, dtype=np.float64))
    return result


def _vector_total(line: str, *, vector_group: int = 0) -> np.ndarray:
    vectors = _vectors_after_time(line)
    if not vectors:
        raise MonitorParseError("vector monitor row has no 3-component vector")
    start = 3 * vector_group
    if len(vectors) >= start + 3:
        return vectors[start] + vectors[start + 1] + vectors[start + 2]
    if vector_group == 0:
        return np.sum(vectors[:3], axis=0)
    raise MonitorParseError("combined force/moment row does not contain requested vector group")


def _scalar_last(line: str) -> np.ndarray:
    values = [float(item) for item in _NUMBER.findall(line)]
    if len(values) < 2:
        raise MonitorParseError("scalar monitor row has no value after time")
    return np.asarray([values[-1]], dtype=np.float64)


def _all_values_after_time(line: str) -> np.ndarray:
    values = [float(item) for item in _NUMBER.findall(line)]
    if len(values) < 2:
        raise MonitorParseError("monitor row has no values after time")
    return np.asarray(values[1:], dtype=np.float64)


def _parse_segments(
    paths: Sequence[Path],
    *,
    label: str,
    row_parser: Callable[[str], np.ndarray],
    rtol: float,
    atol: float,
) -> tuple[np.ndarray, np.ndarray, tuple[SegmentLineage, ...]]:
    rows: dict[Decimal, tuple[np.ndarray, str]] = {}
    lineages: list[SegmentLineage] = []
    width: int | None = None
    for path in paths:
        segment_rows: list[tuple[Decimal, np.ndarray]] = []
        duplicate_rows = 0
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parsed_time = _decimal_time(line)
            if parsed_time is None:
                continue
            time_value, _ = parsed_time
            try:
                value = _finite_array(row_parser(line), f"{label} row", ndim=1)
            except MonitorParseError:
                continue
            if width is None:
                width = len(value)
            if len(value) != width:
                raise MonitorParseError(f"{label} has inconsistent row widths")
            segment_rows.append((time_value, value))
            if time_value in rows:
                previous, previous_path = rows[time_value]
                if not np.allclose(previous, value, rtol=rtol, atol=atol):
                    raise DuplicateTimeConflict(label, str(time_value), previous_path, str(path))
                duplicate_rows += 1
                continue
            rows[time_value] = (value, str(path))
        if not segment_rows:
            raise MonitorParseError(f"no numeric monitor rows found in {path}")
        lineages.append(
            SegmentLineage(
                path=str(path),
                sha256=_sha256_file(path),
                bytes=int(path.stat().st_size),
                first_time=float(min(item[0] for item in segment_rows)),
                last_time=float(max(item[0] for item in segment_rows)),
                rows_parsed=len(segment_rows),
                duplicate_rows=duplicate_rows,
            )
        )
    if not rows:
        raise MonitorParseError(f"no rows found for {label}")
    ordered = sorted(rows.items(), key=lambda item: item[0])
    times = np.asarray([float(time) for time, _ in ordered], dtype=np.float64)
    values = np.vstack([item[0] for _, item in ordered])
    if np.any(np.diff(times) <= 0.0):
        raise MonitorParseError(f"{label} timestamps are not uniquely representable as float64")
    return times, values, tuple(lineages)


def parse_vector_monitor(
    paths: str | Path | Iterable[str | Path],
    *,
    name: str,
    kind: str,
    units: str,
    vector_group: int = 0,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
) -> MonitorSeries:
    source_paths = _paths(paths)
    times, values, lineage = _parse_segments(
        source_paths,
        label=name,
        row_parser=lambda line: _vector_total(line, vector_group=vector_group),
        rtol=rtol,
        atol=atol,
    )
    return MonitorSeries(
        name=name,
        kind=kind,
        units=units,
        times=times,
        values=values,
        columns=("x", "y", "z"),
        segments=lineage,
    )


def parse_force_dat(
    paths: str | Path | Iterable[str | Path],
    *,
    name: str = "forces",
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
) -> MonitorSeries:
    return parse_vector_monitor(paths, name=name, kind="force", units="N", rtol=rtol, atol=atol)


def parse_moment_dat(
    paths: str | Path | Iterable[str | Path],
    *,
    name: str = "moments",
    combined_force_file: bool = False,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
) -> MonitorSeries:
    return parse_vector_monitor(
        paths,
        name=name,
        kind="moment",
        units="N*m",
        vector_group=1 if combined_force_file else 0,
        rtol=rtol,
        atol=atol,
    )


def parse_scalar_monitor(
    paths: str | Path | Iterable[str | Path],
    *,
    name: str,
    kind: str,
    units: str,
    columns: Sequence[str] | None = None,
    keep_all_columns: bool = False,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
) -> MonitorSeries:
    source_paths = _paths(paths)
    times, values, lineage = _parse_segments(
        source_paths,
        label=name,
        row_parser=_all_values_after_time if keep_all_columns else _scalar_last,
        rtol=rtol,
        atol=atol,
    )
    if columns is None:
        columns = ("value",) if values.shape[1] == 1 else tuple(f"value_{index}" for index in range(values.shape[1]))
    return MonitorSeries(
        name=name,
        kind=kind,
        units=units,
        times=times,
        values=values,
        columns=tuple(columns),
        segments=lineage,
    )


def parse_patch_flux(
    paths: str | Path | Iterable[str | Path],
    *,
    name: str,
    units: str = "m3/s",
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
) -> MonitorSeries:
    return parse_scalar_monitor(
        paths,
        name=name,
        kind="water_phase_flux",
        units=units,
        rtol=rtol,
        atol=atol,
    )


def parse_energy_monitor(
    paths: str | Path | Iterable[str | Path],
    *,
    name: str,
    units: str = "W",
    columns: Sequence[str] | None = None,
    keep_all_columns: bool = False,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
) -> MonitorSeries:
    return parse_scalar_monitor(
        paths,
        name=name,
        kind="energy",
        units=units,
        columns=columns,
        keep_all_columns=keep_all_columns,
        rtol=rtol,
        atol=atol,
    )


def parse_solver_log(
    path: str | Path,
    monitor_patterns: Mapping[str, Mapping[str, Any]],
) -> dict[str, MonitorSeries]:
    """Parse named monitor values from solver ``Time =`` blocks as a fallback.

    Each spec requires ``pattern`` and may set ``kind``, ``units``, ``columns``,
    and ``vector=True``. The regular expression is searched only inside the
    corresponding time block and must capture either one scalar or three vector
    components. Later duplicate time blocks must agree exactly within tolerance.
    """
    target = Path(path)
    if not target.is_file():
        raise MonitorParseError(f"solver log does not exist: {target}")
    text = target.read_text(encoding="utf-8", errors="replace")
    blocks = list(
        re.finditer(
            r"^Time\s*=\s*([^\s]+)\s*$\n(?P<body>.*?)(?=^Time\s*=|\Z)",
            text,
            re.MULTILINE | re.DOTALL,
        )
    )
    if not blocks:
        raise MonitorParseError(f"solver log has no Time blocks: {target}")
    file_hash = _sha256_file(target)
    result: dict[str, MonitorSeries] = {}
    for name, raw_spec in monitor_patterns.items():
        spec = dict(raw_spec)
        pattern = spec.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise MonitorParseError(f"solver fallback spec {name!r} has no pattern")
        regex = re.compile(pattern, re.MULTILINE)
        rows: dict[Decimal, np.ndarray] = {}
        first_time: Decimal | None = None
        last_time: Decimal | None = None
        duplicates = 0
        for block in blocks:
            try:
                time_value = Decimal(block.group(1))
            except InvalidOperation as exc:
                raise MonitorParseError("solver log contains an invalid Time value") from exc
            match = regex.search(block.group("body"))
            if not match:
                continue
            captured = match.groups()
            if spec.get("vector"):
                if len(captured) == 1:
                    values = [float(value) for value in _NUMBER.findall(captured[0])]
                else:
                    values = [float(value) for value in captured]
                if len(values) != 3:
                    raise MonitorParseError(f"solver fallback {name!r} did not capture three components")
            else:
                if not captured:
                    values = [float(match.group(0))]
                else:
                    values = [float(value) for value in captured]
            value = _finite_array(values, f"solver fallback {name}", ndim=1)
            if time_value in rows:
                if not np.allclose(rows[time_value], value, rtol=1.0e-12, atol=1.0e-12):
                    raise DuplicateTimeConflict(name, str(time_value), str(target), str(target))
                duplicates += 1
            else:
                rows[time_value] = value
            first_time = time_value if first_time is None else min(first_time, time_value)
            last_time = time_value if last_time is None else max(last_time, time_value)
        if not rows or first_time is None or last_time is None:
            continue
        ordered = sorted(rows.items(), key=lambda item: item[0])
        values = np.vstack([value for _, value in ordered])
        columns = spec.get("columns")
        if columns is None:
            columns = ("x", "y", "z") if values.shape[1] == 3 else tuple(
                "value" if values.shape[1] == 1 else f"value_{index}" for index in range(values.shape[1])
            )
        lineage = SegmentLineage(
            path=str(target),
            sha256=file_hash,
            bytes=int(target.stat().st_size),
            first_time=float(first_time),
            last_time=float(last_time),
            rows_parsed=len(ordered) + duplicates,
            duplicate_rows=duplicates,
        )
        result[name] = MonitorSeries(
            name=name,
            kind=str(spec.get("kind", "solver_log_fallback")),
            units=str(spec.get("units", "1")),
            times=np.asarray([float(time) for time, _ in ordered]),
            values=values,
            columns=tuple(columns),
            segments=(lineage,),
            source="solver_log_fallback",
        )
    return result


def load_monitor_bundle(
    specs: Mapping[str, Mapping[str, Any]],
    *,
    solver_log: str | Path | None = None,
    solver_fallback_patterns: Mapping[str, Mapping[str, Any]] | None = None,
) -> MonitorBundle:
    """Load declared monitors and return an explicit blocked bundle on conflicts."""
    monitors: dict[str, MonitorSeries] = {}
    conflicts: list[str] = []
    missing: list[str] = []
    for name, raw_spec in specs.items():
        spec = dict(raw_spec)
        parser_name = spec.pop("parser", "scalar")
        paths = spec.pop("paths", spec.pop("path", None))
        if paths is None:
            missing.append(name)
            continue
        try:
            if parser_name == "force":
                monitors[name] = parse_force_dat(paths, name=name, **spec)
            elif parser_name == "moment":
                monitors[name] = parse_moment_dat(paths, name=name, **spec)
            elif parser_name == "flux":
                monitors[name] = parse_patch_flux(paths, name=name, **spec)
            elif parser_name == "energy":
                monitors[name] = parse_energy_monitor(paths, name=name, **spec)
            elif parser_name == "scalar":
                monitors[name] = parse_scalar_monitor(paths, name=name, **spec)
            else:
                raise MonitorParseError(f"unknown monitor parser {parser_name!r}")
        except DuplicateTimeConflict as exc:
            conflicts.append(str(exc))
        except (MonitorParseError, OSError) as exc:
            missing.append(f"{name}: {exc}")
    fallback_used: list[str] = []
    if solver_log is not None and solver_fallback_patterns:
        needed_patterns = {
            name: spec
            for name, spec in solver_fallback_patterns.items()
            if name not in monitors
        }
        if needed_patterns:
            try:
                fallback = parse_solver_log(solver_log, needed_patterns)
            except DuplicateTimeConflict as exc:
                conflicts.append(str(exc))
                fallback = {}
            except (MonitorParseError, OSError) as exc:
                missing.append(f"solver log fallback: {exc}")
                fallback = {}
            for name, series in fallback.items():
                monitors[name] = series
                fallback_used.append(name)
            missing = [item for item in missing if item.split(":", 1)[0] not in fallback]
    unresolved = sorted(set(specs) - set(monitors))
    missing.extend(name for name in unresolved if name not in {item.split(":", 1)[0] for item in missing})
    status = "blocked" if conflicts or missing else "passed"
    return MonitorBundle(
        status=status,
        monitors=monitors,
        conflicts=tuple(conflicts),
        missing=tuple(missing),
        fallback_used=tuple(sorted(fallback_used)),
    )


def _window_arrays(
    times: np.ndarray,
    values: np.ndarray,
    start_time: float | None,
    end_time: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    start = float(times[0] if start_time is None else start_time)
    end = float(times[-1] if end_time is None else end_time)
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise MonitorParseError("averaging window must have finite end_time > start_time")
    if start < times[0] or end > times[-1]:
        raise MonitorParseError("monitor does not cover the requested averaging window")
    interior = (times > start) & (times < end)
    selected_times = np.concatenate(([start], times[interior], [end]))
    columns = [np.interp(selected_times, times, values[:, index]) for index in range(values.shape[1])]
    return selected_times, np.column_stack(columns)


def trapezoidal_time_average(
    series_or_times: MonitorSeries | Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray | None = None,
    *,
    start_time: float | None = None,
    end_time: float | None = None,
) -> np.ndarray | float:
    """Return the continuous-time trapezoidal mean for irregular monitor steps."""
    if isinstance(series_or_times, MonitorSeries):
        times_array = series_or_times.times
        values_array = series_or_times.values
    else:
        times_array = _finite_array(series_or_times, "times", ndim=1)
        if values is None:
            raise MonitorParseError("values are required when passing raw times")
        values_array = _finite_array(values, "values")
        if values_array.ndim == 1:
            values_array = values_array[:, None]
        if values_array.ndim != 2 or len(values_array) != len(times_array):
            raise MonitorParseError("times and values have inconsistent shapes")
        if np.any(np.diff(times_array) <= 0.0):
            raise MonitorParseError("times must be strictly increasing")
    if len(times_array) < 2:
        raise MonitorParseError("at least two monitor samples are required for time averaging")
    selected_times, selected_values = _window_arrays(
        times_array, values_array, start_time, end_time
    )
    duration = selected_times[-1] - selected_times[0]
    integrate = getattr(np, "trapezoid", np.trapz)
    mean = integrate(selected_values, selected_times, axis=0) / duration
    return float(mean[0]) if mean.shape == (1,) else mean


_T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}


def block_confidence_interval(
    series_or_times: MonitorSeries | Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray | None = None,
    *,
    start_time: float | None = None,
    end_time: float | None = None,
    n_blocks: int = 4,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Calculate an equal-duration block CI using trapezoidal block means."""
    if confidence != 0.95:
        raise MonitorParseError("only a 95% block confidence interval is supported")
    if isinstance(n_blocks, bool) or not 2 <= int(n_blocks) <= 8:
        raise MonitorParseError("n_blocks must be an integer in [2, 8]")
    n_blocks = int(n_blocks)
    if isinstance(series_or_times, MonitorSeries):
        times = series_or_times.times
        raw_values = series_or_times.values
    else:
        times = _finite_array(series_or_times, "times", ndim=1)
        if values is None:
            raise MonitorParseError("values are required when passing raw times")
        raw_values = _finite_array(values, "values")
        if raw_values.ndim == 1:
            raw_values = raw_values[:, None]
    start = float(times[0] if start_time is None else start_time)
    end = float(times[-1] if end_time is None else end_time)
    boundaries = np.linspace(start, end, n_blocks + 1)
    block_means = np.asarray(
        [
            np.atleast_1d(
                trapezoidal_time_average(
                    times,
                    raw_values,
                    start_time=float(boundaries[index]),
                    end_time=float(boundaries[index + 1]),
                )
            )
            for index in range(n_blocks)
        ]
    )
    mean = np.asarray(trapezoidal_time_average(times, raw_values, start_time=start, end_time=end))
    mean = np.atleast_1d(mean)
    standard_error = np.std(block_means, axis=0, ddof=1) / math.sqrt(n_blocks)
    half_width = _T95[n_blocks] * standard_error
    return {
        "method": "equal-duration block means with trapezoidal integration",
        "confidence": confidence,
        "n_blocks": n_blocks,
        "start_time": start,
        "end_time": end,
        "mean": float(mean[0]) if len(mean) == 1 else mean.tolist(),
        "half_width": float(half_width[0]) if len(half_width) == 1 else half_width.tolist(),
        "lower": float((mean - half_width)[0]) if len(mean) == 1 else (mean - half_width).tolist(),
        "upper": float((mean + half_width)[0]) if len(mean) == 1 else (mean + half_width).tolist(),
        "block_means": block_means[:, 0].tolist() if block_means.shape[1] == 1 else block_means.tolist(),
    }


__all__ = [
    "VERSION",
    "DuplicateTimeConflict",
    "MonitorBundle",
    "MonitorParseError",
    "MonitorSeries",
    "SegmentLineage",
    "block_confidence_interval",
    "load_monitor_bundle",
    "parse_energy_monitor",
    "parse_force_dat",
    "parse_moment_dat",
    "parse_patch_flux",
    "parse_scalar_monitor",
    "parse_solver_log",
    "parse_vector_monitor",
    "trapezoidal_time_average",
]
