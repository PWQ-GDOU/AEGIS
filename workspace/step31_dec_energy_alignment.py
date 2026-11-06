"""Fail-closed alignment of direct CFD results to DEC energy-test rows."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from step31_hydraulic_case_contract import DECRowIdentity, HydraulicCaseContract, validate_case_contract


VERSION = "step31-dec-energy-alignment-v1"
DEFAULT_OPENING_TOLERANCE_MM = 0.05
DEFAULT_N11_RELATIVE_TOLERANCE = 0.02
DEFAULT_Q11_RELATIVE_TOLERANCE = 0.05
DEFAULT_EFFICIENCY_RELATIVE_TOLERANCE = 0.05


class DECAlignmentError(ValueError):
    """Raised when DEC data cannot be parsed or aligned without ambiguity."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise DECAlignmentError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DECAlignmentError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise DECAlignmentError(f"{name} must be {qualifier}")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise DECAlignmentError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise DECAlignmentError(f"{name} must be a positive integer") from exc
    if result <= 0 or float(value) != result:
        raise DECAlignmentError(f"{name} must be a positive integer")
    return result


@dataclass(frozen=True)
class DECEnergyRow:
    """One parsed DEC calibration row with source identity and Q11 quality."""

    source_file: str
    source_sha256: str
    row_number: int
    nozzle_count: int
    opening_mm: float
    n11_rpm: float
    q11_lps: float
    calibration_efficiency_pct: float
    q11_quality: str = "unchecked"
    q11_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_file, str) or not self.source_file:
            raise DECAlignmentError("source_file must not be empty")
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in self.source_sha256)
        ):
            raise DECAlignmentError("source_sha256 must be a hexadecimal SHA-256")
        object.__setattr__(self, "source_sha256", self.source_sha256.lower())
        object.__setattr__(self, "row_number", _positive_int(self.row_number, "row_number"))
        object.__setattr__(self, "nozzle_count", _positive_int(self.nozzle_count, "nozzle_count"))
        object.__setattr__(self, "opening_mm", _finite(self.opening_mm, "opening_mm", positive=True))
        object.__setattr__(self, "n11_rpm", _finite(self.n11_rpm, "n11_rpm", positive=True))
        object.__setattr__(self, "q11_lps", _finite(self.q11_lps, "q11_lps"))
        efficiency = _finite(self.calibration_efficiency_pct, "calibration_efficiency_pct")
        if not 0.0 <= efficiency <= 100.0:
            raise DECAlignmentError("calibration_efficiency_pct must be in [0, 100]")
        object.__setattr__(self, "calibration_efficiency_pct", efficiency)
        if self.q11_quality not in {"unchecked", "passed", "rejected"}:
            raise DECAlignmentError("q11_quality must be unchecked, passed, or rejected")
        if self.q11_quality == "rejected" and not self.q11_reason:
            raise DECAlignmentError("a rejected Q11 row requires q11_reason")

    @property
    def identity(self) -> DECRowIdentity:
        return DECRowIdentity(
            source_file=self.source_file,
            source_sha256=self.source_sha256,
            row_number=self.row_number,
            nozzle_count=self.nozzle_count,
            opening_mm=self.opening_mm,
            n11_rpm=self.n11_rpm,
            q11_lps=self.q11_lps,
            calibration_efficiency_pct=self.calibration_efficiency_pct,
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["efficiency_role"] = "parsed_calibration_only"
        result["is_direct_cfd_efficiency"] = False
        return result


def _infer_nozzle_count(path: Path, text: str) -> int:
    header = re.search(r"投运喷嘴\s*:\s*([^\r\n]+)", text)
    if header:
        count = len(re.findall(r"\d+\s*#", header.group(1)))
        if count > 0:
            return count
    match = re.search(r"_(\d+)\.txt$", path.name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    raise DECAlignmentError(f"cannot infer nozzle count from {path}")


def parse_dec_energy_file(path: str | Path) -> list[DECEnergyRow]:
    """Parse one DEC ``S n11 Q11 Eff`` file and retain source row identity."""
    target = Path(path)
    if not target.is_file():
        raise DECAlignmentError(f"DEC energy file does not exist: {target}")
    raw = target.read_bytes()
    text = raw.decode("gbk", errors="replace")
    nozzle_count = _infer_nozzle_count(target, text)
    source_hash = hashlib.sha256(raw).hexdigest()
    rows: list[DECEnergyRow] = []
    header_seen = False
    numeric = re.compile(
        r"^\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s+"
        r"([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s+"
        r"([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s+"
        r"([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*$"
    )
    for line_number, line in enumerate(text.splitlines(), start=1):
        if all(label in line for label in ("S", "n11", "Q11", "Eff")):
            header_seen = True
            continue
        if not header_seen:
            continue
        match = numeric.match(line)
        if not match:
            continue
        opening, n11, q11, efficiency = (float(value) for value in match.groups())
        rows.append(
            DECEnergyRow(
                source_file=target.name,
                source_sha256=source_hash,
                row_number=line_number,
                nozzle_count=nozzle_count,
                opening_mm=opening,
                n11_rpm=n11,
                q11_lps=q11,
                calibration_efficiency_pct=efficiency,
            )
        )
    if not header_seen or not rows:
        raise DECAlignmentError(f"no DEC energy rows found in {target}")
    checked, _ = assess_q11_quality(rows)
    return checked


def _row_from_value(value: DECEnergyRow | DECRowIdentity | Mapping[str, Any]) -> DECEnergyRow:
    if isinstance(value, DECEnergyRow):
        return value
    if isinstance(value, DECRowIdentity):
        if value.calibration_efficiency_pct is None:
            raise DECAlignmentError("DEC row identity has no calibration efficiency")
        return DECEnergyRow(
            source_file=value.source_file,
            source_sha256=value.source_sha256,
            row_number=value.row_number,
            nozzle_count=value.nozzle_count,
            opening_mm=value.opening_mm,
            n11_rpm=value.n11_rpm,
            q11_lps=value.q11_lps,
            calibration_efficiency_pct=value.calibration_efficiency_pct,
        )
    if not isinstance(value, Mapping):
        raise DECAlignmentError("DEC rows must be mappings or DECEnergyRow values")
    data = dict(value)
    if "S_mm" in data and "opening_mm" not in data:
        data["opening_mm"] = data.pop("S_mm")
    if "S" in data and "opening_mm" not in data:
        data["opening_mm"] = data.pop("S")
    if "n11" in data and "n11_rpm" not in data:
        data["n11_rpm"] = data.pop("n11")
    if "Q11" in data and "q11_lps" not in data:
        data["q11_lps"] = data.pop("Q11")
    if "eta_pct" in data and "calibration_efficiency_pct" not in data:
        data["calibration_efficiency_pct"] = data.pop("eta_pct")
    if "efficiency_pct" in data and "calibration_efficiency_pct" not in data:
        data["calibration_efficiency_pct"] = data.pop("efficiency_pct")
    allowed = {
        "source_file", "source_sha256", "row_number", "nozzle_count",
        "opening_mm", "n11_rpm", "q11_lps", "calibration_efficiency_pct",
        "q11_quality", "q11_reason",
    }
    return DECEnergyRow(**{key: item for key, item in data.items() if key in allowed})


def assess_q11_quality(
    rows: Iterable[DECEnergyRow | DECRowIdentity | Mapping[str, Any]],
    *,
    minimum_q11_lps: float = 1.0,
    maximum_q11_lps: float = 5000.0,
    minimum_group_relative_tolerance: float = 0.10,
) -> tuple[list[DECEnergyRow], list[DECEnergyRow]]:
    """Reject impossible Q11 and robust outliers within nozzle/opening groups."""
    values = [_row_from_value(row) for row in rows]
    if not values:
        return [], []
    state: dict[int, tuple[str, str | None]] = {}
    for index, row in enumerate(values):
        if row.q11_quality == "rejected":
            state[index] = ("rejected", row.q11_reason or "Q11 was rejected by the source quality record")
        elif not minimum_q11_lps <= row.q11_lps <= maximum_q11_lps:
            state[index] = ("rejected", "Q11 is outside the accepted positive physical range")
    groups: dict[tuple[int, float], list[int]] = {}
    for index, row in enumerate(values):
        groups.setdefault((row.nozzle_count, round(row.opening_mm, 6)), []).append(index)
    for indices in groups.values():
        eligible = [index for index in indices if index not in state]
        if len(eligible) < 3:
            continue
        q11 = np.asarray([values[index].q11_lps for index in eligible], dtype=np.float64)
        median = float(np.median(q11))
        mad = float(np.median(np.abs(q11 - median)))
        robust_relative_limit = max(
            minimum_group_relative_tolerance,
            6.0 * 1.4826 * mad / max(abs(median), 1.0e-30),
        )
        for index in eligible:
            relative = abs(values[index].q11_lps - median) / max(abs(median), 1.0e-30)
            if relative > robust_relative_limit:
                state[index] = (
                    "rejected",
                    f"Q11 robust group outlier: relative deviation {relative:.6g} exceeds {robust_relative_limit:.6g}",
                )
    checked: list[DECEnergyRow] = []
    rejected: list[DECEnergyRow] = []
    for index, row in enumerate(values):
        quality, reason = state.get(index, ("passed", None))
        checked_row = replace(row, q11_quality=quality, q11_reason=reason)
        checked.append(checked_row)
        if quality == "rejected":
            rejected.append(checked_row)
    return checked, rejected


def load_dec_energy_rows(root: str | Path) -> list[DECEnergyRow]:
    target = Path(root)
    paths = [target] if target.is_file() else sorted(target.rglob("autoD1276P21-B01-*_E*.txt"))
    if not paths:
        raise DECAlignmentError(f"no DEC energy files found in {target}")
    rows: list[DECEnergyRow] = []
    for path in paths:
        rows.extend(parse_dec_energy_file(path))
    checked, _ = assess_q11_quality(rows)
    return checked


def _relative_difference(first: float, second: float) -> float:
    return abs(first - second) / max(abs(second), 1.0e-30)


def _identity_matches(row: DECEnergyRow, frozen: DECRowIdentity) -> bool:
    return bool(
        row.source_file == frozen.source_file
        and row.source_sha256 == frozen.source_sha256
        and row.row_number == frozen.row_number
        and row.nozzle_count == frozen.nozzle_count
        and math.isclose(row.opening_mm, frozen.opening_mm, rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(row.n11_rpm, frozen.n11_rpm, rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(row.q11_lps, frozen.q11_lps, rel_tol=0.0, abs_tol=1.0e-12)
        and (
            frozen.calibration_efficiency_pct is None
            or math.isclose(
                row.calibration_efficiency_pct,
                frozen.calibration_efficiency_pct,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        )
    )


def align_cfd_to_dec(
    cfd_result: Mapping[str, Any],
    rows: Iterable[DECEnergyRow | DECRowIdentity | Mapping[str, Any]],
    *,
    contract: HydraulicCaseContract | Mapping[str, Any] | None = None,
    nozzle_count: int | None = None,
    opening_mm: float | None = None,
    opening_tolerance_mm: float = DEFAULT_OPENING_TOLERANCE_MM,
    n11_relative_tolerance: float = DEFAULT_N11_RELATIVE_TOLERANCE,
    q11_relative_tolerance: float = DEFAULT_Q11_RELATIVE_TOLERANCE,
    efficiency_relative_tolerance: float = DEFAULT_EFFICIENCY_RELATIVE_TOLERANCE,
) -> dict[str, Any]:
    """Match CFD to DEC by nozzle/S/n11/Q11 and compare separate efficiencies."""
    if not isinstance(cfd_result, Mapping):
        raise DECAlignmentError("cfd_result must be a mapping")
    if cfd_result.get("status") != "passed" or cfd_result.get("diagnostic_only") is True:
        return {
            "version": VERSION,
            "status": "blocked",
            "reason": "only a passed direct full-runner CFD result can be aligned to DEC",
            "dec_efficiency_role": "parsed_calibration_only",
            "dec_efficiency_is_direct_cfd": False,
        }
    case = validate_case_contract(contract) if contract is not None else None
    if case is not None:
        nozzle_count = case.nozzle_count
        opening_mm = case.opening_mm
    operating_point = cfd_result.get("operating_point", {})
    if nozzle_count is None:
        nozzle_count = operating_point.get("nozzle_count")
    if opening_mm is None:
        opening_mm = operating_point.get("opening_mm")
    if nozzle_count is None or opening_mm is None:
        raise DECAlignmentError("nozzle_count and opening_mm are required for DEC matching")
    nozzle = _positive_int(nozzle_count, "nozzle_count")
    opening = _finite(opening_mm, "opening_mm", positive=True)
    metrics = cfd_result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise DECAlignmentError("CFD result has no metrics")
    n11 = _finite(metrics.get("n11_rpm"), "CFD n11_rpm", positive=True)
    q11 = _finite(metrics.get("Q11_lps"), "CFD Q11_lps", positive=True)
    eta_pct = _finite(metrics.get("eta_pct"), "CFD eta_pct")
    checked, rejected = assess_q11_quality(rows)
    rejected_candidates = [
        row for row in rejected
        if row.nozzle_count == nozzle and abs(row.opening_mm - opening) <= opening_tolerance_mm
    ]
    candidates: list[tuple[float, DECEnergyRow, dict[str, float]]] = []
    for row in checked:
        if row.q11_quality != "passed" or row.nozzle_count != nozzle:
            continue
        opening_error = abs(row.opening_mm - opening)
        n11_error = _relative_difference(n11, row.n11_rpm)
        q11_error = _relative_difference(q11, row.q11_lps)
        if (
            opening_error <= opening_tolerance_mm
            and n11_error <= n11_relative_tolerance
            and q11_error <= q11_relative_tolerance
        ):
            score = (
                opening_error / max(opening_tolerance_mm, 1.0e-30)
                + n11_error / max(n11_relative_tolerance, 1.0e-30)
                + q11_error / max(q11_relative_tolerance, 1.0e-30)
            )
            candidates.append((score, row, {
                "opening_absolute_error_mm": opening_error,
                "n11_relative_error": n11_error,
                "Q11_relative_error": q11_error,
            }))
    if not candidates:
        reason = "no DEC row matches nozzle/S/n11/Q11 within tolerance"
        if rejected_candidates:
            reason = "matching DEC opening contains anomalous Q11 rows; alignment rejected"
        return {
            "version": VERSION,
            "status": "blocked",
            "reason": reason,
            "rejected_q11_rows": [row.to_dict() for row in rejected_candidates],
            "dec_efficiency_role": "parsed_calibration_only",
            "dec_efficiency_is_direct_cfd": False,
        }
    candidates.sort(key=lambda item: (item[0], item[1].source_file, item[1].row_number))
    _, matched, errors = candidates[0]
    if len(candidates) > 1 and math.isclose(candidates[0][0], candidates[1][0], rel_tol=0.0, abs_tol=1.0e-12):
        return {
            "version": VERSION,
            "status": "blocked",
            "reason": "DEC match is ambiguous",
            "candidate_rows": [item[1].to_dict() for item in candidates[:2]],
            "dec_efficiency_role": "parsed_calibration_only",
            "dec_efficiency_is_direct_cfd": False,
        }
    if case is not None and case.dec_row_identity is not None and not _identity_matches(matched, case.dec_row_identity):
        return {
            "version": VERSION,
            "status": "blocked",
            "reason": "matched DEC row differs from the row identity frozen in the case contract",
            "matched_dec_row": matched.to_dict(),
            "frozen_dec_row_identity": asdict(case.dec_row_identity),
            "dec_efficiency_role": "parsed_calibration_only",
            "dec_efficiency_is_direct_cfd": False,
        }
    efficiency_error = _relative_difference(eta_pct, matched.calibration_efficiency_pct)
    passed = efficiency_error <= efficiency_relative_tolerance
    return {
        "version": VERSION,
        "status": "passed" if passed else "failed",
        "match_basis": ["nozzle_count", "opening_mm", "n11_rpm", "Q11_lps"],
        "match_errors": errors,
        "matched_dec_row": matched.to_dict(),
        "cfd": {
            "efficiency_pct": eta_pct,
            "efficiency_role": "direct_cfd_computed",
            "n11_rpm": n11,
            "Q11_lps": q11,
        },
        "dec_calibration": {
            "efficiency_pct": matched.calibration_efficiency_pct,
            "efficiency_role": "parsed_calibration_only",
            "is_direct_cfd_efficiency": False,
        },
        "efficiency_relative_error": efficiency_error,
        "efficiency_relative_tolerance": efficiency_relative_tolerance,
        "efficiency_gate_passed": passed,
        "dec_efficiency_role": "parsed_calibration_only",
        "dec_efficiency_is_direct_cfd": False,
        "interpretation": (
            "DEC efficiency is comparison/calibration evidence and is not substituted for the CFD efficiency."
        ),
    }


def align_dec_energy(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return align_cfd_to_dec(*args, **kwargs)


__all__ = [
    "DECAlignmentError",
    "DECEnergyRow",
    "DEFAULT_EFFICIENCY_RELATIVE_TOLERANCE",
    "DEFAULT_N11_RELATIVE_TOLERANCE",
    "DEFAULT_OPENING_TOLERANCE_MM",
    "DEFAULT_Q11_RELATIVE_TOLERANCE",
    "VERSION",
    "align_cfd_to_dec",
    "align_dec_energy",
    "assess_q11_quality",
    "load_dec_energy_rows",
    "parse_dec_energy_file",
]
