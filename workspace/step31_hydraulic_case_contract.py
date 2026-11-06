"""Frozen physical and provenance contract for direct CFD performance cases."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


VERSION = "step31-hydraulic-case-contract-v1"
CASE_TYPES = frozenset({"full_runner_rotating_vof", "stationary_single_bucket"})
PRESSURE_KINDS = frozenset({"kinematic", "static_pa", "total_pa"})
SURFACE_ROLES = frozenset({"inlet", "outlet"})


class CaseContractError(ValueError):
    """Raised when case identity or physical semantics are incomplete."""


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise CaseContractError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CaseContractError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise CaseContractError(f"{name} must be {qualifier}")
    return result


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise CaseContractError(f"{name} must be a SHA-256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise CaseContractError(f"{name} must be a 64-character hexadecimal SHA-256")
    return normalized


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CaseContractError(f"{label} must be a non-empty string")
    return value.strip()


def _vector3(value: Sequence[Any], name: str, *, normalize: bool = False) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise CaseContractError(f"{name} must contain three components")
    vector = tuple(_finite(component, f"{name}[{index}]") for index, component in enumerate(value))
    if normalize:
        magnitude = math.sqrt(sum(component * component for component in vector))
        if magnitude <= 1.0e-15:
            raise CaseContractError(f"{name} must be non-zero")
        vector = tuple(component / magnitude for component in vector)
    return vector  # type: ignore[return-value]


def canonical_json(value: Any) -> str:
    """Return deterministic strict JSON used for immutable contract identity."""
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class PressureDefinition:
    """OpenFOAM pressure semantics used by the total-head monitors."""

    field: str
    kind: str
    reference_pressure_pa: float = 0.0
    includes_hydrostatic_head: bool = False
    total_head_definition: str = "p/(rho*g) + |U|^2/(2*g) + z"

    def __post_init__(self) -> None:
        object.__setattr__(self, "field", _name(self.field, "pressure.field"))
        if self.kind not in PRESSURE_KINDS:
            raise CaseContractError(f"pressure.kind must be one of {sorted(PRESSURE_KINDS)}")
        object.__setattr__(
            self,
            "reference_pressure_pa",
            _finite(self.reference_pressure_pa, "pressure.reference_pressure_pa"),
        )
        if not isinstance(self.includes_hydrostatic_head, bool):
            raise CaseContractError("pressure.includes_hydrostatic_head must be boolean")
        object.__setattr__(
            self,
            "total_head_definition",
            _name(self.total_head_definition, "pressure.total_head_definition"),
        )


@dataclass(frozen=True)
class WaterPhase:
    """VOF water-phase identity and density used for phase flux and power."""

    name: str
    alpha_field: str
    density_kg_m3: float
    flux_is_water_phase_only: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "water_phase.name"))
        object.__setattr__(self, "alpha_field", _name(self.alpha_field, "water_phase.alpha_field"))
        object.__setattr__(
            self,
            "density_kg_m3",
            _finite(self.density_kg_m3, "water_phase.density_kg_m3", positive=True),
        )
        if not isinstance(self.flux_is_water_phase_only, bool):
            raise CaseContractError("water_phase.flux_is_water_phase_only must be boolean")


@dataclass(frozen=True)
class MeasurementSurface:
    """One oriented control surface and its explicitly named monitors."""

    name: str
    patch: str
    role: str
    flow_sign: int
    water_flux_monitor: str
    total_head_monitor: str | None
    energy_flux_monitor: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "measurement_surface.name"))
        object.__setattr__(self, "patch", _name(self.patch, "measurement_surface.patch"))
        if self.role not in SURFACE_ROLES:
            raise CaseContractError(f"measurement_surface.role must be one of {sorted(SURFACE_ROLES)}")
        if isinstance(self.flow_sign, bool) or self.flow_sign not in (-1, 1):
            raise CaseContractError("measurement_surface.flow_sign must be -1 or 1")
        object.__setattr__(
            self,
            "water_flux_monitor",
            _name(self.water_flux_monitor, "measurement_surface.water_flux_monitor"),
        )
        if self.total_head_monitor is not None:
            object.__setattr__(
                self,
                "total_head_monitor",
                _name(self.total_head_monitor, "measurement_surface.total_head_monitor"),
            )
        if self.energy_flux_monitor is not None:
            object.__setattr__(
                self,
                "energy_flux_monitor",
                _name(self.energy_flux_monitor, "measurement_surface.energy_flux_monitor"),
            )


@dataclass(frozen=True)
class DECRowIdentity:
    """Identity of one DEC PELTON energy-test row, never a CFD result."""

    source_file: str
    source_sha256: str
    row_number: int
    nozzle_count: int
    opening_mm: float
    n11_rpm: float
    q11_lps: float
    calibration_efficiency_pct: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_file", _name(self.source_file, "dec_row.source_file"))
        object.__setattr__(self, "source_sha256", _sha256(self.source_sha256, "dec_row.source_sha256"))
        if isinstance(self.row_number, bool) or int(self.row_number) <= 0:
            raise CaseContractError("dec_row.row_number must be a positive integer")
        object.__setattr__(self, "row_number", int(self.row_number))
        if isinstance(self.nozzle_count, bool) or int(self.nozzle_count) <= 0:
            raise CaseContractError("dec_row.nozzle_count must be a positive integer")
        object.__setattr__(self, "nozzle_count", int(self.nozzle_count))
        object.__setattr__(self, "opening_mm", _finite(self.opening_mm, "dec_row.opening_mm", positive=True))
        object.__setattr__(self, "n11_rpm", _finite(self.n11_rpm, "dec_row.n11_rpm", positive=True))
        object.__setattr__(self, "q11_lps", _finite(self.q11_lps, "dec_row.q11_lps", positive=True))
        if self.calibration_efficiency_pct is not None:
            efficiency = _finite(
                self.calibration_efficiency_pct,
                "dec_row.calibration_efficiency_pct",
            )
            if not 0.0 <= efficiency <= 100.0:
                raise CaseContractError("dec_row.calibration_efficiency_pct must be in [0, 100]")
            object.__setattr__(self, "calibration_efficiency_pct", efficiency)


@dataclass(frozen=True)
class HydraulicCaseContract:
    """Immutable identity and physical semantics for one CFD operating point."""

    geometry_sha256: str
    mesh_sha256: str
    case_sha256: str
    case_type: str
    D1_m: float
    H_m: float
    n_rpm: float
    omega_rad_s: float
    axis: tuple[float, float, float]
    CofR_m: tuple[float, float, float]
    nozzle_count: int
    opening_mm: float
    measurement_surfaces: tuple[MeasurementSurface, ...]
    pressure_definition: PressureDefinition
    water_phase: WaterPhase
    dec_row_identity: DECRowIdentity | None
    rotating_domain: bool
    vof_enabled: bool
    moment_monitor: str = "runnerMoments"
    energy_balance_monitor: str | None = None
    torque_sign: int = 1
    version: str = field(default=VERSION, init=False)
    contract_sha256: str = field(default="", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "geometry_sha256", _sha256(self.geometry_sha256, "geometry_sha256"))
        object.__setattr__(self, "mesh_sha256", _sha256(self.mesh_sha256, "mesh_sha256"))
        object.__setattr__(self, "case_sha256", _sha256(self.case_sha256, "case_sha256"))
        if self.case_type not in CASE_TYPES:
            raise CaseContractError(f"case_type must be one of {sorted(CASE_TYPES)}")
        object.__setattr__(self, "D1_m", _finite(self.D1_m, "D1_m", positive=True))
        object.__setattr__(self, "H_m", _finite(self.H_m, "H_m", positive=True))
        object.__setattr__(self, "n_rpm", _finite(self.n_rpm, "n_rpm"))
        object.__setattr__(self, "omega_rad_s", _finite(self.omega_rad_s, "omega_rad_s"))
        object.__setattr__(self, "axis", _vector3(self.axis, "axis", normalize=True))
        object.__setattr__(self, "CofR_m", _vector3(self.CofR_m, "CofR_m"))
        if isinstance(self.nozzle_count, bool) or int(self.nozzle_count) <= 0:
            raise CaseContractError("nozzle_count must be a positive integer")
        object.__setattr__(self, "nozzle_count", int(self.nozzle_count))
        object.__setattr__(self, "opening_mm", _finite(self.opening_mm, "opening_mm", positive=True))
        if not isinstance(self.rotating_domain, bool) or not isinstance(self.vof_enabled, bool):
            raise CaseContractError("rotating_domain and vof_enabled must be booleans")
        object.__setattr__(self, "moment_monitor", _name(self.moment_monitor, "moment_monitor"))
        if self.energy_balance_monitor is not None:
            object.__setattr__(
                self,
                "energy_balance_monitor",
                _name(self.energy_balance_monitor, "energy_balance_monitor"),
            )
        if isinstance(self.torque_sign, bool) or self.torque_sign not in (-1, 1):
            raise CaseContractError("torque_sign must be -1 or 1")
        surfaces = tuple(self.measurement_surfaces)
        if not surfaces:
            raise CaseContractError("measurement_surfaces must not be empty")
        if not all(isinstance(surface, MeasurementSurface) for surface in surfaces):
            raise CaseContractError("measurement_surfaces must contain MeasurementSurface values")
        names = [surface.name for surface in surfaces]
        if len(names) != len(set(names)):
            raise CaseContractError("measurement surface names must be unique")
        object.__setattr__(self, "measurement_surfaces", surfaces)
        expected_omega = self.n_rpm * 2.0 * math.pi / 60.0
        if not math.isclose(self.omega_rad_s, expected_omega, rel_tol=1.0e-8, abs_tol=1.0e-10):
            raise CaseContractError("n_rpm and omega_rad_s are inconsistent")
        if self.case_type == "stationary_single_bucket" and self.rotating_domain:
            raise CaseContractError("stationary_single_bucket cannot claim a rotating domain")
        if self.dec_row_identity is not None:
            row = self.dec_row_identity
            if row.nozzle_count != self.nozzle_count:
                raise CaseContractError("DEC row nozzle count differs from case contract")
        payload = asdict(self)
        payload.pop("contract_sha256", None)
        object.__setattr__(self, "contract_sha256", canonical_sha256(payload))

    @property
    def has_net_head_measurement(self) -> bool:
        roles = {
            surface.role
            for surface in self.measurement_surfaces
            if surface.total_head_monitor is not None
        }
        return roles == SURFACE_ROLES

    @property
    def performance_eligible(self) -> bool:
        return bool(
            self.case_type == "full_runner_rotating_vof"
            and self.rotating_domain
            and self.omega_rad_s > 0.0
            and self.vof_enabled
            and self.water_phase.flux_is_water_phase_only
            and self.has_net_head_measurement
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _surface_from_mapping(value: Mapping[str, Any]) -> MeasurementSurface:
    return MeasurementSurface(**dict(value))


def _dec_from_mapping(value: Mapping[str, Any] | None) -> DECRowIdentity | None:
    return None if value is None else DECRowIdentity(**dict(value))


def build_case_contract(value: Mapping[str, Any]) -> HydraulicCaseContract:
    """Build and validate a frozen contract from a JSON-like mapping."""
    data = dict(value)
    data.pop("version", None)
    stored_contract_hash = data.pop("contract_sha256", None)
    try:
        data["pressure_definition"] = PressureDefinition(**dict(data["pressure_definition"]))
        data["water_phase"] = WaterPhase(**dict(data["water_phase"]))
        data["measurement_surfaces"] = tuple(
            _surface_from_mapping(item) for item in data["measurement_surfaces"]
        )
        data["dec_row_identity"] = _dec_from_mapping(data.get("dec_row_identity"))
        contract = HydraulicCaseContract(**data)
    except KeyError as exc:
        raise CaseContractError(f"case contract is missing {exc.args[0]}") from exc
    except TypeError as exc:
        raise CaseContractError(f"invalid case contract fields: {exc}") from exc
    if stored_contract_hash is not None and contract.contract_sha256 != stored_contract_hash:
        raise CaseContractError("contract_sha256 does not match the canonical contract")
    return contract


def validate_case_contract(value: HydraulicCaseContract | Mapping[str, Any]) -> HydraulicCaseContract:
    if isinstance(value, HydraulicCaseContract):
        rebuilt = build_case_contract(value.to_dict())
        if rebuilt.contract_sha256 != value.contract_sha256:
            raise CaseContractError("case contract identity changed after construction")
        return value
    if not isinstance(value, Mapping):
        raise CaseContractError("case contract must be a mapping or HydraulicCaseContract")
    return build_case_contract(value)


__all__ = [
    "CASE_TYPES",
    "VERSION",
    "CaseContractError",
    "DECRowIdentity",
    "HydraulicCaseContract",
    "MeasurementSurface",
    "PressureDefinition",
    "WaterPhase",
    "build_case_contract",
    "canonical_sha256",
    "validate_case_contract",
]
