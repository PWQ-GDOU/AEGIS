"""Explicit Step30 unit normalization and conversion.

The conversion contract is deliberately small and explicit.  A value is never
converted without both a source and target unit, and conversion across
physical dimensions fails closed.  Canonical units are SI-compatible where
possible; area, volume, and erosion outputs retain the units used by the
Step30 measurement records.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - scalar-only use remains available
    np = None  # type: ignore[assignment]


VERSION = "step30-units-v1"


class UnitError(ValueError):
    """Raised for an unknown unit or an incompatible conversion."""


@dataclass(frozen=True)
class UnitSpec:
    name: str
    dimension: str
    factor: float


# Factors convert a quantity to the base unit of its dimension.
_SPECS: dict[str, UnitSpec] = {
    "m": UnitSpec("m", "length", 1.0),
    "cm": UnitSpec("cm", "length", 1.0e-2),
    "mm": UnitSpec("mm", "length", 1.0e-3),
    "um": UnitSpec("um", "length", 1.0e-6),
    "s": UnitSpec("s", "time", 1.0),
    "h": UnitSpec("h", "time", 3600.0),
    "kg": UnitSpec("kg", "mass", 1.0),
    "g": UnitSpec("g", "mass", 1.0e-3),
    "mg": UnitSpec("mg", "mass", 1.0e-6),
    "m2": UnitSpec("m2", "area", 1.0),
    "mm2": UnitSpec("mm2", "area", 1.0e-6),
    "mm3": UnitSpec("mm3", "volume", 1.0e-9),
    "mm3/kg": UnitSpec("mm3/kg", "specific_volume", 1.0e-9),
    "kg/kg": UnitSpec("kg/kg", "mass_ratio", 1.0),
    # Both are length/time and can therefore be converted explicitly.
    "mm/h": UnitSpec("mm/h", "velocity", 1.0e-3 / 3600.0),
    "m/s": UnitSpec("m/s", "velocity", 1.0),
    "deg": UnitSpec("deg", "angle", 1.0),
}

# Input aliases normalize to the exact public units above.  No unrequested
# units are added implicitly (for example, minutes or radians).
_ALIASES: dict[str, str] = {
    "meter": "m", "metre": "m", "meters": "m", "metres": "m",
    "centimeter": "cm", "centimetre": "cm", "centimeters": "cm", "centimetres": "cm",
    "millimeter": "mm", "millimetre": "mm", "millimeters": "mm", "millimetres": "mm",
    "micrometer": "um", "micrometre": "um", "micron": "um", "microns": "um",
    "μm": "um", "µm": "um", "micro m": "um",
    "sec": "s", "second": "s", "seconds": "s",
    "hour": "h", "hours": "h",
    "kilogram": "kg", "kilograms": "kg", "gram": "g", "grams": "g",
    "milligram": "mg", "milligrams": "mg",
    "m^2": "m2", "m²": "m2", "square meter": "m2", "square metre": "m2",
    "mm^2": "mm2", "mm²": "mm2", "square millimeter": "mm2", "square millimetre": "mm2",
    "mm^3": "mm3", "mm³": "mm3", "cubic millimeter": "mm3", "cubic millimetre": "mm3",
    "mm^3/kg": "mm3/kg", "mm³/kg": "mm3/kg",
    "kg/kg": "kg/kg", "mm^3 kg^-1": "mm3/kg", "mm3 kg-1": "mm3/kg",
    "mm/h": "mm/h", "mm hr-1": "mm/h", "mm/hour": "mm/h",
    "m/s": "m/s", "mps": "m/s", "m s^-1": "m/s", "m s-1": "m/s",
    "degree": "deg", "degrees": "deg", "°": "deg",
}


def _compact(unit: str) -> str:
    return re.sub(r"\s+", " ", unit.strip().lower().replace("²", "^2").replace("³", "^3"))


def normalize_unit(unit: str) -> str:
    """Return a canonical unit name, rejecting missing or unsupported units."""
    if not isinstance(unit, str) or not unit.strip():
        raise UnitError("unit must be a non-empty string")
    value = _compact(unit)
    value = _ALIASES.get(value, value)
    value = value.replace("^2", "2").replace("^3", "3")
    value = _ALIASES.get(value, value)
    if value not in _SPECS:
        raise UnitError(f"unsupported unit {unit!r}; supported units: {', '.join(sorted(_SPECS))}")
    return value


def unit_spec(unit: str) -> UnitSpec:
    return _SPECS[normalize_unit(unit)]


def unit_dimension(unit: str) -> str:
    return unit_spec(unit).dimension


def _finite_scalar(value: Any) -> float:
    if isinstance(value, bool):
        raise UnitError("boolean is not a numeric value")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise UnitError(f"value is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise UnitError("value must be finite")
    return number


def convert(value: Any, from_unit: str, to_unit: str) -> Any:
    """Convert scalar or array-like values with an explicit dimensional check."""
    source = unit_spec(from_unit)
    target = unit_spec(to_unit)
    if source.dimension != target.dimension:
        raise UnitError(
            f"incompatible units: {source.name!r} ({source.dimension}) and "
            f"{target.name!r} ({target.dimension})"
        )
    scale = source.factor / target.factor
    if np is not None:
        try:
            raw = np.asarray(value)
            if np.iscomplexobj(raw):
                raise UnitError("complex values are not supported")
            array = np.asarray(raw, dtype=float)
            if not np.isfinite(array).all():
                raise UnitError("value must contain only finite numbers")
            result = array * scale
            if result.ndim == 0:
                return float(result)
            return result
        except (TypeError, ValueError) as exc:
            raise UnitError(f"value is not numeric: {value!r}") from exc
    return _finite_scalar(value) * scale


def convert_value(value: Any, from_unit: str, to_unit: str) -> Any:
    """Compatibility spelling for :func:`convert`."""
    return convert(value, from_unit, to_unit)


def convert_units(value: Any, from_unit: str, to_unit: str) -> Any:
    """Public plural spelling used by other Step30 validation modules."""
    return convert(value, from_unit, to_unit)


SUPPORTED_UNITS = frozenset(_SPECS)


def conversion_record(value: Any, from_unit: str, to_unit: str) -> dict[str, Any]:
    """Return a JSON-friendly conversion result with explicit unit metadata."""
    source = normalize_unit(from_unit)
    target = normalize_unit(to_unit)
    converted = convert(value, source, target)
    if np is not None and isinstance(converted, np.ndarray):
        converted = converted.tolist()
    return {
        "version": VERSION,
        "value": converted,
        "from_unit": source,
        "to_unit": target,
        "dimension": unit_dimension(source),
    }


def _selftest() -> None:
    assert normalize_unit("m^2") == "m2"
    assert normalize_unit("μm") == "um"
    assert math.isclose(convert(100.0, "cm", "m"), 1.0)
    assert math.isclose(convert(1.0, "h", "s"), 3600.0)
    assert math.isclose(convert(1.0, "kg", "mg"), 1.0e6)
    assert math.isclose(convert(1.0, "m2", "mm2"), 1.0e6)
    assert math.isclose(convert(1.0, "mm3/kg", "mm3/kg"), 1.0)
    assert math.isclose(convert(1.0, "kg/kg", "kg/kg"), 1.0)
    assert math.isclose(convert(1.0, "mm/h", "m/s"), 1.0e-3 / 3600.0)
    assert math.isclose(convert(180.0, "deg", "deg"), 180.0)
    if np is not None:
        assert np.allclose(convert([1000.0, 2000.0], "mm", "m"), [1.0, 2.0])
    try:
        convert(1.0, "m", "s")
    except UnitError:
        pass
    else:
        raise AssertionError("incompatible dimensions were accepted")
    try:
        normalize_unit("min")
    except UnitError:
        pass
    else:
        raise AssertionError("unrequested unit was accepted")


def selftest() -> bool:
    _selftest()
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explicit Step30 unit conversion")
    parser.add_argument("--value", required=False, help="JSON scalar or array")
    parser.add_argument("--from-unit", dest="from_unit")
    parser.add_argument("--to-unit", dest="to_unit")
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        selftest()
        print("step30_units selftest: PASS")
        return 0
    if args.value is None or args.from_unit is None or args.to_unit is None:
        build_parser().error("--value, --from-unit, and --to-unit are required unless --selftest is used")
    try:
        value = json.loads(args.value)
        result = conversion_record(value, args.from_unit, args.to_unit)
    except (json.JSONDecodeError, UnitError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
