"""
step30_lagrangian_particle_tracking.py
=======================================
CFD-driven particle-wall erosion post-processor.

The OpenFOAM fields used here are the converged continuous-phase cell
velocity field. This is therefore a one-way-coupled CFD + particle
post-processing model, not an OpenFOAM kinematicCloud/DPM run. Parcels are
mass-weighted explicitly so the hit cloud can be converted to a wall
mass-flux and a Mansouri erosion field.

Outputs:
  output/particle_hits_cfd.npz
  output/step30_cfd_summary.json
  output/particle_tracking_cfd_validation.png

The previous unweighted point-cloud result remains in particle_hits.npz and
is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

WORKSPACE = Path(__file__).parent
OUTPUT_DIR = WORKSPACE / "output"
OUT_NPZ = OUTPUT_DIR / "particle_hits_cfd.npz"
OUT_JSON = OUTPUT_DIR / "step30_cfd_summary.json"
OUT_PNG = OUTPUT_DIR / "particle_tracking_cfd_validation.png"

# Mansouri (2015), SS316 coefficients used by steps 21 and 27.
K_M, N_M, A1, B1, B2, FS = 2.57e-9, 2.4, 228.5, 7.91, 12.81, 1.0
RHO_WALL = 7800.0  # kg/m3; SS316 engineering value used by step12
REMOTE_EXECUTION_TARGET_ENV = "STEP30_EXECUTION_TARGET"
REMOTE_EXECUTION_TARGETS = {"dcu_cpu", "user_gpu"}
REMOTE_EXECUTION_MODE_ENV = "STEP30_REMOTE_EXECUTION"


def require_remote_execution_target() -> str:
    """Refuse simulation unless it is explicitly launched on an approved server."""
    target = os.environ.get(REMOTE_EXECUTION_TARGET_ENV, "").strip().lower()
    if target not in REMOTE_EXECUTION_TARGETS:
        raise RuntimeError(
            "Step30 simulation is server-only; set "
            f"{REMOTE_EXECUTION_TARGET_ENV}=dcu_cpu or user_gpu on the remote server"
        )
    if os.name == "nt" or os.environ.get(REMOTE_EXECUTION_MODE_ENV) != "1":
        raise RuntimeError(
            "Step30 numerical execution is server-only; set "
            f"{REMOTE_EXECUTION_MODE_ENV}=1 on a non-Windows remote server"
        )
    return target


@dataclass(frozen=True)
class TrackingConfig:
    """Numerical and physical settings for one reproducible run."""

    cfd_variant: str = "native_baseline"
    collision_backend: str = "auto"
    collision_policy: str = "strict_v2"
    topology_path: str = "output/bucket_topology.npz"
    native_si_directory: str = "output/native_bucket_si"
    cfd_field_time: str | None = None
    output_tag: str = "latest"
    immutable_output: bool = False
    n_parcels: int = 2000
    n_injected: int = 2000
    max_impacts: int = 3
    seed: int = 42
    dt_s: float = 1.0e-5
    max_time_s: float = 0.05
    max_query_distance_mm: float = 80.0
    face_fallback_distance_mm: float = 1.5
    hit_epsilon_mm: float = 0.05
    particle_cfl: float = 0.5
    cfd_velocity_interpolation: str = "idw_k8_v1"
    particle_integrator: str = "midpoint_exponential_v2"
    strict_max_substep_s: float = 2.5e-6
    strict_near_miss_distance_diagnostics: bool = True
    collision_profile: bool = False
    analytic_reference_transform_path: str = "jet_rotation.npz"
    min_substep_s: float = 1.0e-8
    max_collision_events_per_step: int = 8
    min_speed_ms: float = 0.5
    e_normal: float = 0.8
    e_tangent: float = 0.9
    include_gravity: bool = False
    mass_flow_mode: str = "nominal"
    injection_window_s: float = 1.0
    observation_window_s: float = 1.0
    concentration_kg_m3: float = 1.31
    rated_flow_m3_s: float = 27.0
    nozzle_count: int = 6
    jet_speed_ms: float = 104.8
    max_accepted_cfd_speed_ms: float = 500.0
    jet_diameter_mm: float = 240.0
    rho_particle_kg_m3: float = 2650.0
    d_min_m: float = 1.0e-6
    d_max_m: float = 2.3e-4
    d50_m: float = 5.0e-5
    rr_exponent: float = 1.5
    psd_sampling: str = "number_random_v1"

    @property
    def max_steps(self) -> int:
        return max(1, int(np.ceil(self.max_time_s / self.dt_s)))

    @property
    def nozzle_flow_m3_s(self) -> float:
        if self.mass_flow_mode == "geometric":
            area = np.pi * (self.jet_diameter_mm * 1e-3) ** 2 / 4.0
            return area * self.jet_speed_ms
        if self.mass_flow_mode != "nominal":
            raise ValueError("mass_flow_mode must be 'nominal' or 'geometric'")
        return self.rated_flow_m3_s / self.nozzle_count

    @property
    def sediment_flow_kg_s(self) -> float:
        return self.concentration_kg_m3 * self.nozzle_flow_m3_s


def mansouri_er(Vp: np.ndarray, alpha_deg: np.ndarray) -> np.ndarray:
    """Mansouri mass erosion ratio [kg target / kg sediment]."""
    a = np.deg2rad(np.clip(alpha_deg, 0.5, 89.5))
    s = np.clip(np.sin(a), 1e-6, 1.0)
    fa = A1 * s**B1 * np.clip(1.5 - s, 1e-6, None) ** B2
    return K_M * FS * np.clip(Vp, 0.1, None) ** N_M * fa


def parse_foam_header(path: Path) -> dict:
    """Read the OpenFOAM header fields required for artifact pairing."""
    text = path.read_text(encoding="utf-8", errors="replace")
    header = re.search(r"FoamFile\s*\{(?P<body>.*?)\}", text, re.DOTALL)
    if not header:
        raise ValueError(f"Missing FoamFile header: {path}")
    body = header.group("body")

    def required(pattern: str, label: str) -> str:
        match = re.search(pattern, body)
        if not match:
            raise ValueError(f"Missing OpenFOAM {label}: {path}")
        return match.group(1).strip()

    dimensions = re.search(r"\bdimensions\s+(\[[^;]+\])\s*;", text)
    return {
        "class": required(r"\bclass\s+([^;]+)\s*;", "class"),
        "location": required(r'\blocation\s+"?([^";]+)"?\s*;', "location"),
        "object": required(r"\bobject\s+([^;]+)\s*;", "object"),
        "dimensions": dimensions.group(1) if dimensions else None,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_reference_transform_path(output_dir: Path, cfg: TrackingConfig, cfd: dict) -> Path:
    """Choose a source-geometry-bound analytic reference transform."""
    requested = Path(cfg.analytic_reference_transform_path)
    if not requested.is_absolute():
        requested = output_dir / requested
    manifest = cfd.get("artifact_manifest") or {}
    source_hash = manifest.get("si_stl", {}).get("sha256")
    authoritative_hash = "181eb659091f187e1286ca27b689d654a83b5522b28d6f39aceaa4ebc3fc38ca"
    if (
        cfg.cfd_variant == "native_si"
        and source_hash == authoritative_hash
        and requested.name == "jet_rotation.npz"
    ):
        requested = output_dir / "authoritative_bucket_reference_transform_v1.json"
    return requested


def _load_reference_transform(output_dir: Path, cfg: TrackingConfig, cfd: dict) -> dict:
    """Load and validate the analytic local-to-reference transform."""
    path = _resolve_reference_transform_path(output_dir, cfg, cfd)
    if not path.is_file():
        raise FileNotFoundError(f"analytic reference transform is missing: {path}")
    source_hash = (cfd.get("artifact_manifest") or {}).get("si_stl", {}).get("sha256")
    if path.suffix.lower() == ".json":
        document = json.loads(path.read_text(encoding="utf-8"))
        R = np.asarray(document.get("R"), dtype=np.float64)
        translation = np.asarray(document.get("translation_mm"), dtype=np.float64)
        declared_source_hash = document.get("source_geometry_sha256")
        if source_hash and declared_source_hash != source_hash:
            raise ValueError("analytic reference transform source geometry hash mismatch")
        version = document.get("version")
        convention = document.get("convention")
    elif path.suffix.lower() == ".npz":
        data = dict(np.load(str(path)))
        R = np.asarray(data.get("R"), dtype=np.float64)
        translation = np.asarray(data.get("nozzle_pos"), dtype=np.float64)
        declared_source_hash = None
        version = "legacy-jet-rotation-v1"
        convention = "row-vector: global_mm = local_mm @ R + nozzle_pos"
    else:
        raise ValueError("analytic reference transform must be JSON or NPZ")
    if R.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(translation).all():
        raise ValueError("analytic reference transform has invalid numeric shapes")
    if not np.allclose(R.T @ R, np.eye(3), rtol=0.0, atol=1e-8) or np.linalg.det(R) <= 0.0:
        raise ValueError("analytic reference transform rotation is not a proper rigid rotation")
    return {
        "R": R,
        "nozzle_pos": translation,
        "path": path,
        "sha256": _sha256(path),
        "source_geometry_sha256": declared_source_hash,
        "version": version,
        "convention": convention,
    }


def parse_foam_scalar_field(path: Path) -> np.ndarray:
    """Parse an OpenFOAM ASCII volScalarField internalField."""
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\n\s*\(\n(.*?)\n\s*\)",
        text,
        re.DOTALL,
    )
    if not match:
        raise ValueError(f"Cannot parse scalar internalField: {path}")
    expected = int(match.group(1))
    data = np.fromstring(match.group(2), sep="\n", dtype=np.float64)
    if len(data) != expected:
        raise ValueError(f"{path.name}: expected {expected} values, parsed {len(data)}")
    return data


def parse_foam_vector_field(path: Path) -> np.ndarray:
    """Parse an OpenFOAM ASCII volVectorField internalField."""
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\n\s*\(\n(.*?)\n\s*\)",
        text,
        re.DOTALL,
    )
    if not match:
        raise ValueError(f"Cannot parse vector internalField: {path}")
    expected = int(match.group(1))
    return _parse_vector_list(match.group(2), expected, path)


def _parse_vector_list(body: str, expected: int, path: Path) -> np.ndarray:
    raw = re.findall(r"\(([^)]+)\)", body)
    data = np.array([[float(x) for x in row.split()] for row in raw], dtype=np.float64)
    if data.shape != (expected, 3):
        raise ValueError(f"{path.name}: expected ({expected}, 3), parsed {data.shape}")
    return data


def parse_foam_boundary_vector_field(path: Path, patch_name: str) -> np.ndarray:
    """Parse a nonuniform vector list from one OpenFOAM boundary patch."""
    text = path.read_text(encoding="utf-8", errors="replace")
    patch = re.search(rf"\b{re.escape(patch_name)}\s*\{{(?P<body>.*?)\n\s*\}}", text, re.DOTALL)
    if not patch:
        raise ValueError(f"Boundary patch {patch_name!r} is missing: {path}")
    match = re.search(
        r"value\s+nonuniform\s+List<vector>\s+(\d+)\s*\n\s*\(\n(?P<body>.*?)\n\s*\)",
        patch.group("body"),
        re.DOTALL,
    )
    if not match:
        raise ValueError(f"Boundary patch {patch_name!r} has no nonuniform vector value: {path}")
    return _parse_vector_list(match.group("body"), int(match.group(1)), path)


def _load_native_si_manifest(native_dir: Path, field_time: str) -> tuple[dict, dict[str, Path]]:
    manifest_path = native_dir / f"native_si_manifest_{field_time}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing native SI artifact manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest.get("version")
    if version not in {"step30-native-si-manifest-v1", "step30-native-si-manifest-v2"}:
        raise ValueError("unsupported native SI artifact manifest version")
    if str(manifest.get("field_time")) != field_time:
        raise ValueError("native SI manifest field time differs from requested time")
    units = manifest.get("units", {})
    expected_units = {
        "source_coordinate_units": "m",
        "internal_coordinate_units": "mm",
        "coordinate_scale_to_mm": 1000.0,
    }
    if units != expected_units:
        raise ValueError(f"invalid native SI unit contract: {units}")

    velocity_object = "U"
    if version == "step30-native-si-manifest-v2":
        semantics = manifest.get("solver_semantics", {})
        required = {
            "application", "mode", "field_statistic",
            "tracking_velocity_object", "averaging_window",
        }
        if set(semantics) != required:
            raise ValueError("native SI v2 manifest has incomplete solver semantics")
        statistic = semantics["field_statistic"]
        velocity_object = semantics["tracking_velocity_object"]
        if statistic == "snapshot":
            if semantics != {
                "application": "simpleFoam",
                "mode": "steady_rans",
                "field_statistic": "snapshot",
                "tracking_velocity_object": "U",
                "averaging_window": None,
            }:
                raise ValueError("steady snapshot semantics require simpleFoam/steady_rans/U")
        elif statistic == "time_mean":
            window = semantics["averaging_window"]
            if (
                semantics["application"] != "pimpleFoam"
                or semantics["mode"] != "urans"
                or velocity_object != "UMean"
                or not isinstance(window, dict)
                or set(window) != {"start_time", "end_time", "sample_count"}
                or int(window["sample_count"]) <= 0
            ):
                raise ValueError("URANS time-mean semantics require pimpleFoam/urans/UMean and a valid averaging window")
        else:
            raise ValueError("native SI field statistic must be snapshot or time_mean")

    expected_headers = {
        "Cx": ("Cx", "volScalarField", "[0 1 0 0 0 0 0]"),
        "Cy": ("Cy", "volScalarField", "[0 1 0 0 0 0 0]"),
        "Cz": ("Cz", "volScalarField", "[0 1 0 0 0 0 0]"),
        "U": (velocity_object, "volVectorField", "[0 1 -1 0 0 0 0]"),
    }
    paths: dict[str, Path] = {}
    fields = manifest.get("fields", {})
    for logical_name, (object_name, expected_class, expected_dimensions) in expected_headers.items():
        entry = fields.get(object_name, {})
        file_name = entry.get("file")
        if not file_name or Path(file_name).name != file_name:
            raise ValueError(f"invalid native SI manifest path for {object_name}")
        path = native_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing native SI field: {path}")
        if _sha256(path) != entry.get("sha256"):
            raise ValueError(f"native SI field hash mismatch: {path.name}")
        header = parse_foam_header(path)
        if header["location"] != field_time:
            raise ValueError(f"native SI field time mismatch: {path.name}")
        if header["class"] != expected_class or header["object"] != object_name:
            raise ValueError(f"native SI field header mismatch: {path.name}")
        if header["dimensions"] != expected_dimensions:
            raise ValueError(f"native SI field dimensions mismatch: {path.name}")
        paths[logical_name] = path
    mesh_signature = manifest.get("topology", {}).get("mesh_signature")
    if not mesh_signature:
        raise ValueError("native SI manifest has no topology mesh signature")
    return manifest, paths


def load_cfd_velocity_field(
    output_dir: Path,
    variant: str = "baseline",
    field_time: str | None = None,
    native_si_dir: Path | None = None,
) -> dict:
    """Load and validate cell centres and velocity for baseline or MRF CFD."""
    coordinate_scale_to_mm = 1.0
    source_coordinate_units = "mm"
    artifact_manifest = None
    artifact_manifest_file = None
    artifact_manifest_sha256 = None
    if variant == "baseline":
        prefix = "cfd_"
        field_time = "500"
        paths = {key: output_dir / f"{prefix}{key}.foam" for key in ("Cx", "Cy", "Cz", "U")}
    elif variant == "mrf":
        prefix = "mrf_cfd_"
        field_time = "500"
        paths = {key: output_dir / f"{prefix}{key}.foam" for key in ("Cx", "Cy", "Cz", "U")}
    elif variant == "native":
        prefix = "native_1000_"
        field_time = "1000"
        native_dir = output_dir / "native_bucket"
        paths = {key: native_dir / f"{prefix}{key}.foam" for key in ("Cx", "Cy", "Cz", "U")}
    elif variant == "native_baseline":
        prefix = "native_baseline_500_"
        field_time = "500"
        native_dir = output_dir / "native_bucket"
        paths = {key: native_dir / f"{prefix}{key}.foam" for key in ("Cx", "Cy", "Cz", "U")}
    elif variant == "native_si":
        if field_time is None:
            raise ValueError("native_si requires an explicit OpenFOAM field time")
        field_time = str(field_time)
        prefix = f"native_si_{field_time}_"
        native_dir = native_si_dir or output_dir / "native_bucket_si"
        artifact_manifest, paths = _load_native_si_manifest(native_dir, field_time)
        artifact_manifest_path = native_dir / f"native_si_manifest_{field_time}.json"
        artifact_manifest_file = artifact_manifest_path.name
        artifact_manifest_sha256 = _sha256(artifact_manifest_path)
        coordinate_scale_to_mm = 1000.0
        source_coordinate_units = "m"
    else:
        raise ValueError("cfd_variant must be 'baseline', 'mrf', 'native', 'native_baseline', or 'native_si'")
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing CFD field(s): " + ", ".join(missing))

    Cx = parse_foam_scalar_field(paths["Cx"])
    Cy = parse_foam_scalar_field(paths["Cy"])
    Cz = parse_foam_scalar_field(paths["Cz"])
    U = parse_foam_vector_field(paths["U"])
    if not (len(Cx) == len(Cy) == len(Cz) == len(U)):
        raise ValueError("CFD coordinate and velocity fields have different cell counts")

    cell_centers = np.column_stack([Cx, Cy, Cz]) * coordinate_scale_to_mm
    if not np.isfinite(cell_centers).all() or not np.isfinite(U).all():
        raise ValueError("CFD fields contain non-finite values")
    speeds = np.linalg.norm(U, axis=1)
    return {
        "cell_centers_mm": cell_centers,
        "U_ms": U,
        "variant": variant,
        "prefix": prefix,
        "field_time": field_time,
        "source_coordinate_units": source_coordinate_units,
        "internal_coordinate_units": "mm",
        "coordinate_scale_to_mm": coordinate_scale_to_mm,
        "artifact_manifest": artifact_manifest,
        "artifact_manifest_file": artifact_manifest_file,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "n_cells": int(len(cell_centers)),
        "center_bounds_mm": [cell_centers.min(axis=0).tolist(), cell_centers.max(axis=0).tolist()],
        "speed_stats_ms": {
            "min": float(speeds.min()),
            "max": float(speeds.max()),
            "median": float(np.median(speeds)),
        },
    }


def load_wall_points(path: Path) -> np.ndarray:
    """Load CFD wall sample coordinates; extra columns are diagnostic only."""
    data = np.loadtxt(str(path), comments="#")
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Wall sample file must have at least x/y/z columns: {path}")
    xyz = np.asarray(data[:, :3], dtype=np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError(f"Wall sample coordinates contain non-finite values: {path}")
    return xyz


def load_local_stl(path: Path) -> dict:
    """Load local bucket STL and precompute face geometry."""
    import trimesh

    mesh = trimesh.load(str(path), process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    xyz = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    tri = xyz[faces]
    edge1 = tri[:, 1] - tri[:, 0]
    edge2 = tri[:, 2] - tri[:, 0]
    cross = np.cross(edge1, edge2)
    area_mm2 = 0.5 * np.linalg.norm(cross, axis=1)
    valid = area_mm2 > 1e-12
    if not valid.all():
        faces = faces[valid]
        tri = tri[valid]
        cross = cross[valid]
        area_mm2 = area_mm2[valid]
    normals = cross / np.maximum(2.0 * area_mm2[:, None], 1e-12)
    centers = tri.mean(axis=1)
    radii = np.linalg.norm(tri - centers[:, None, :], axis=2).max(axis=1)
    broadphase_radius = float(max(15.0, 3.0 * np.median(radii)))
    regular = radii <= broadphase_radius
    regular_idx = np.flatnonzero(regular)
    large_idx = np.flatnonzero(~regular)
    return {
        "vertices_mm": xyz,
        "faces": faces,
        "triangles_mm": tri,
        "collision_triangles_mm": tri,
        "face_centers_mm": centers,
        "face_normals": normals,
        "face_area_mm2": area_mm2,
        "face_area_m2": area_mm2 * 1e-6,
        "face_radii_mm": radii,
        "collision_face_centers_mm": centers,
        "collision_face_normals": normals,
        "collision_face_ids": np.arange(len(tri), dtype=np.int64),
        "parent_face_ids": np.arange(len(tri), dtype=np.int64),
        "n_parent_faces": int(len(tri)),
        "backend": "stl",
        "face_id_namespace": "stl_processed_face",
        "mesh_signature": "stl_processed",
        "patch_name": "bucket_stl",
        "time_name": "unknown",
        "source_coordinate_units": "mm",
        "internal_coordinate_units": "mm",
        "coordinate_scale_to_mm": 1.0,
        "collision_broadphase_radius_mm": broadphase_radius,
        "regular_face_indices": regular_idx,
        "large_face_indices": large_idx,
        "regular_face_tree": cKDTree(centers[regular_idx]) if len(regular_idx) else None,
        "large_face_tree": cKDTree(centers[large_idx]) if len(large_idx) else None,
        "bounds_mm": [xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist()],
        "collision_bounds_mm": [tri.min(axis=(0, 1)).tolist(), tri.max(axis=(0, 1)).tolist()],
        "surface_area_m2": float(area_mm2.sum() * 1e-6),
    }


def load_collision_geometry(cfg: TrackingConfig) -> dict:
    """Load native CFD topology or the explicitly requested STL backend."""
    backend = cfg.collision_backend
    topology_path = WORKSPACE / cfg.topology_path
    if backend == "auto":
        backend = "cfd_topology" if topology_path.exists() else "stl"
    if backend == "cfd_topology":
        if not topology_path.exists():
            raise FileNotFoundError(f"native topology artifact is missing: {topology_path}")
        from step30_cfd_topology import load_topology_artifact
        return load_topology_artifact(topology_path)
    if backend == "stl":
        return load_local_stl(OUTPUT_DIR / "dec_single_bucket_local.stl")
    raise ValueError("collision_backend must be 'auto', 'cfd_topology', or 'stl'")


def geometry_preflight(stl: dict, wall_xyz: np.ndarray, tolerance_mm: float = 50.0) -> dict:
    """Check coverage without pretending point samples are STL topology."""
    bounds = np.asarray(stl["bounds_mm"])
    inside = np.all((wall_xyz >= bounds[0] - tolerance_mm) & (wall_xyz <= bounds[1] + tolerance_mm), axis=1)
    tree = cKDTree(stl["face_centers_mm"])
    nearest_face_dist, _ = tree.query(wall_xyz, k=1)
    report = {
        "wall_points": int(len(wall_xyz)),
        "bbox_tolerance_mm": float(tolerance_mm),
        "wall_points_in_stl_bbox_plus_tolerance_pct": float(100.0 * inside.mean()),
        "nearest_face_center_distance_mm": {
            "median": float(np.median(nearest_face_dist)),
            "p95": float(np.percentile(nearest_face_dist, 95)),
            "max": float(nearest_face_dist.max()),
        },
        "warning": (
            "CFD wall samples are independent surface samples; nearest face-centre distance "
            "is a coverage diagnostic, not a topology mapping."
        ),
    }
    if inside.mean() < 0.95:
        raise ValueError("CFD wall sample coverage is inconsistent with local STL bounding box")
    return report


def _rosin_rammler_quantile(u: np.ndarray, cfg: TrackingConfig) -> np.ndarray:
    """Return bounded Rosin-Rammler number-distribution quantiles."""
    d_char = cfg.d50_m / (-np.log(0.5)) ** (1.0 / cfg.rr_exponent)
    diam = d_char * (-np.log1p(-np.clip(u, 1e-12, 1.0 - 1e-12))) ** (
        1.0 / cfg.rr_exponent
    )
    return np.clip(diam, cfg.d_min_m, cfg.d_max_m)


def sample_rosin_rammler(rng: np.random.Generator, n: int, cfg: TrackingConfig) -> np.ndarray:
    """Sample diameters under the frozen number or mass-stratified contract."""
    if n <= 0:
        raise ValueError("particle sample size must be positive")
    if cfg.psd_sampling == "number_random_v1":
        return _rosin_rammler_quantile(rng.uniform(1e-8, 1.0 - 1e-8, n), cfg)
    if cfg.psd_sampling != "mass_stratified_v1":
        raise ValueError(
            "psd_sampling must be 'number_random_v1' or 'mass_stratified_v1'"
        )

    # Target mass CDF: q(d) is proportional to d^3 times the frozen number
    # distribution. Numerically invert it on a deterministic number-quantile
    # grid, retaining the configured lower/upper clipping atoms.
    grid_size = max(200_001, 40 * n + 1)
    u_number = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    diam_grid = _rosin_rammler_quantile(u_number, cfg)
    mass_density = diam_grid**3
    cumulative_mass = np.empty(grid_size, dtype=np.float64)
    cumulative_mass[0] = 0.0
    cumulative_mass[1:] = np.cumsum(
        0.5 * (mass_density[:-1] + mass_density[1:]) * np.diff(u_number)
    )
    total = float(cumulative_mass[-1])
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Rosin-Rammler mass CDF is invalid")
    cumulative_mass /= total
    mass_quantiles = (np.arange(n, dtype=np.float64) + rng.uniform(0.0, 1.0, n)) / n
    diam = np.interp(mass_quantiles, cumulative_mass, diam_grid)
    # Decouple stratified diameter order from injection-position order.
    return diam[rng.permutation(n)]


def compute_parcel_masses(diam_m: np.ndarray, cfg: TrackingConfig) -> tuple[np.ndarray, np.ndarray, float]:
    """Return physical particle masses, parcel weights, and total mass."""
    particle_mass = cfg.rho_particle_kg_m3 * (np.pi / 6.0) * diam_m**3
    total_mass = cfg.sediment_flow_kg_s * cfg.injection_window_s
    if cfg.psd_sampling == "mass_stratified_v1":
        parcel_mass = np.full(len(diam_m), total_mass / max(len(diam_m), 1))
        if len(parcel_mass):
            parcel_mass[-1] = total_mass - float(parcel_mass[:-1].sum())
    elif cfg.psd_sampling == "number_random_v1":
        # Number-distribution samples represent equal physical counts, so their
        # represented sediment mass is proportional to d^3.
        weights = particle_mass / max(float(particle_mass.sum()), 1e-30)
        parcel_mass = total_mass * weights
    else:
        raise ValueError(
            "psd_sampling must be 'number_random_v1' or 'mass_stratified_v1'"
        )
    return particle_mass, parcel_mass, float(total_mass)


def segment_triangle_intersection(
    start: np.ndarray,
    end: np.ndarray,
    triangles: np.ndarray,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Moller-Trumbore segment/triangle intersections."""
    direction = end - start
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    pvec = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    det = np.einsum("ij,ij->i", edge1, pvec)
    non_parallel = np.abs(det) > eps
    inv_det = np.zeros_like(det)
    inv_det[non_parallel] = 1.0 / det[non_parallel]
    tvec = start[None, :] - triangles[:, 0]
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    qvec = np.cross(tvec, edge1)
    v = np.einsum("ij,ij->i", np.broadcast_to(direction, qvec.shape), qvec) * inv_det
    t = np.einsum("ij,ij->i", edge2, qvec) * inv_det
    hit = non_parallel & (u >= -eps) & (v >= -eps) & (u + v <= 1.0 + eps) & (t >= -eps) & (t <= 1.0 + eps)
    return hit, t


def _closest_point_on_triangle(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Closest point on one triangle, used to place fallback impacts on-face."""
    a, b, c = triangle
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = np.dot(ab, ap), np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a.copy()
    bp = point - b
    d3, d4 = np.dot(ab, bp), np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b.copy()
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return a + (d1 / max(d1 - d3, 1e-30)) * ab
    cp = point - c
    d5, d6 = np.dot(ab, cp), np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c.copy()
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return a + (d2 / max(d2 - d6, 1e-30)) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        return b + ((d4 - d3) / max((d4 - d3) + (d5 - d6), 1e-30)) * (c - b)
    denom = max(va + vb + vc, 1e-30)
    return a + ab * (vb / denom) + ac * (vc / denom)


def _point_triangle_distance(point: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Squared distance from one point to many triangles for fallback hits."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac, ap = b - a, c - a, point[None, :] - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    best = np.full(len(triangles), np.inf, dtype=np.float64)
    mask_a = (d1 <= 0) & (d2 <= 0)
    best[mask_a] = np.einsum("ij,ij->i", ap[mask_a], ap[mask_a])

    bp = point[None, :] - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    mask_b = (d3 >= 0) & (d4 <= d3)
    best[mask_b] = np.einsum("ij,ij->i", bp[mask_b], bp[mask_b])

    vc = d1 * d4 - d3 * d2
    mask_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    v = d1[mask_ab] / np.maximum(d1[mask_ab] - d3[mask_ab], 1e-30)
    q = a[mask_ab] + v[:, None] * ab[mask_ab]
    delta = point[None, :] - q
    best[mask_ab] = np.einsum("ij,ij->i", delta, delta)

    cp = point[None, :] - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    mask_c = (d6 >= 0) & (d5 <= d6)
    best[mask_c] = np.einsum("ij,ij->i", cp[mask_c], cp[mask_c])

    vb = d5 * d2 - d1 * d6
    mask_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    w = d2[mask_ac] / np.maximum(d2[mask_ac] - d6[mask_ac], 1e-30)
    q = a[mask_ac] + w[:, None] * ac[mask_ac]
    delta = point[None, :] - q
    best[mask_ac] = np.einsum("ij,ij->i", delta, delta)

    va = d3 * d6 - d5 * d4
    mask_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    w = (d4[mask_bc] - d3[mask_bc]) / np.maximum((d4[mask_bc] - d3[mask_bc]) + (d5[mask_bc] - d6[mask_bc]), 1e-30)
    q = b[mask_bc] + w[:, None] * (c[mask_bc] - b[mask_bc])
    delta = point[None, :] - q
    best[mask_bc] = np.einsum("ij,ij->i", delta, delta)

    mask_face = ~(mask_a | mask_b | mask_ab | mask_c | mask_ac | mask_bc)
    denom = np.maximum(va[mask_face] + vb[mask_face] + vc[mask_face], 1e-30)
    v = vb[mask_face] / denom
    w = vc[mask_face] / denom
    q = a[mask_face] + ab[mask_face] * v[:, None] + ac[mask_face] * w[:, None]
    delta = point[None, :] - q
    best[mask_face] = np.einsum("ij,ij->i", delta, delta)
    return best


def find_segment_hit(
    start: np.ndarray,
    end: np.ndarray,
    mesh: dict,
    face_tree: cKDTree,
    fallback_distance_mm: float,
    collision_policy: str = "legacy_proximity_v1",
    suppressed_parent_face_id: int | None = None,
    diagnostics: Counter | None = None,
    strict_near_miss_distance_diagnostics: bool = True,
) -> tuple[int, float, np.ndarray, str, int] | None:
    """Return the deterministic first swept hit under the requested policy."""
    if collision_policy not in {"strict_v2", "legacy_proximity_v1"}:
        raise ValueError("collision_policy must be 'strict_v2' or 'legacy_proximity_v1'")
    segment = end - start
    profile_enabled = diagnostics is not None and diagnostics.get("_profile_enabled", 0) > 0
    if profile_enabled:
        diagnostics["profile_segment_query_count"] += 1
    # Safe broad rejection: a segment whose axis-aligned bounds do not overlap
    # the actual collision-triangle bounds cannot intersect any triangle.
    collision_bounds = mesh.get("collision_bounds_mm")
    if collision_bounds is not None:
        margin = max(float(fallback_distance_mm), 0.0)
        bounds_min = np.asarray(collision_bounds[0], dtype=np.float64) - margin
        bounds_max = np.asarray(collision_bounds[1], dtype=np.float64) + margin
        segment_min = np.minimum(start, end)
        segment_max = np.maximum(start, end)
        if np.any(segment_max < bounds_min) or np.any(segment_min > bounds_max):
            if profile_enabled:
                diagnostics["profile_axis_bbox_rejection_count"] += 1
            return None
    mid = 0.5 * (start + end)
    regular_idx = mesh.get("regular_face_indices")
    large_idx = mesh.get("large_face_indices")
    if regular_idx is None or large_idx is None:
        query_radius = 0.5 * np.linalg.norm(segment) + float(mesh["face_radii_mm"].max())
        candidates = face_tree.query_ball_point(mid, query_radius)
    else:
        query_radius = 0.5 * np.linalg.norm(segment) + mesh["collision_broadphase_radius_mm"]
        regular_tree = mesh.get("regular_face_tree", face_tree)
        candidates = regular_idx[regular_tree.query_ball_point(mid, query_radius)].tolist()
        large_tree = mesh.get("large_face_tree")
        if large_tree is not None and len(large_idx):
            large_radius = 0.5 * np.linalg.norm(segment) + float(mesh["face_radii_mm"][large_idx].max())
            candidates.extend(large_idx[large_tree.query_ball_point(mid, large_radius)].tolist())
    if not candidates:
        if profile_enabled:
            diagnostics["profile_empty_candidate_count"] += 1
        return None
    if profile_enabled:
        diagnostics["profile_candidate_raw_count"] += len(candidates)
    candidates = np.unique(np.asarray(candidates, dtype=np.int64))
    if profile_enabled:
        diagnostics["profile_candidate_unique_count"] += len(candidates)
    collision_face_ids = mesh.get("collision_face_ids", mesh.get("parent_face_ids"))
    if suppressed_parent_face_id is not None:
        candidates = candidates[collision_face_ids[candidates] != suppressed_parent_face_id]
        if not len(candidates):
            if profile_enabled:
                diagnostics["profile_suppressed_empty_count"] += 1
            return None
    tri = mesh["collision_triangles_mm"][candidates]
    if profile_enabled:
        diagnostics["profile_triangle_test_count"] += len(tri)
    hit, t = segment_triangle_intersection(start, end, tri)
    if hit.any():
        if profile_enabled:
            diagnostics["profile_intersection_query_count"] += 1
            diagnostics["profile_intersection_hit_count"] += int(hit.sum())
        local_hits = np.flatnonzero(hit)
        order = np.lexsort((candidates[local_hits], t[local_hits]))
        local = int(local_hits[order[0]])
        collision_index = int(candidates[local])
        face_id = int(collision_face_ids[collision_index])
        return face_id, float(t[local]), start + float(t[local]) * segment, "intersection", collision_index

    distance_candidates = candidates
    distance_triangles = tri
    if collision_policy == "strict_v2":
        centers = np.asarray(mesh["collision_face_centers_mm"])[candidates]
        radii = np.asarray(mesh["face_radii_mm"])[candidates]
        sphere_lower_bound = np.maximum(np.linalg.norm(centers - end, axis=1) - radii, 0.0)
        near_endpoint = sphere_lower_bound <= fallback_distance_mm
        if not near_endpoint.any():
            if profile_enabled:
                diagnostics["profile_near_endpoint_empty_count"] += 1
            return None
        # Strict-v2 never accepts proximity fallbacks.  The exact point-to-
        # triangle distance below is diagnostic only; make it optional so a
        # performance A/B can verify that this bookkeeping is not part of the
        # physical acceptance path.
        if not strict_near_miss_distance_diagnostics:
            return None
        distance_candidates = candidates[near_endpoint]
        distance_triangles = tri[near_endpoint]
        if profile_enabled:
            diagnostics["profile_near_endpoint_candidate_count"] += len(distance_candidates)

    d2 = _point_triangle_distance(end, distance_triangles)
    if profile_enabled:
        diagnostics["profile_distance_triangle_test_count"] += len(distance_triangles)
    nearest = int(np.argmin(d2))
    endpoint_distance = float(np.sqrt(d2[nearest]))
    if endpoint_distance <= fallback_distance_mm and diagnostics is not None:
        diagnostics["endpoint_near_miss"] += 1
    if collision_policy == "legacy_proximity_v1" and endpoint_distance <= fallback_distance_mm:
        collision_index = int(distance_candidates[nearest])
        face_id = int(collision_face_ids[collision_index])
        projected = _closest_point_on_triangle(end, mesh["collision_triangles_mm"][collision_index])
        return face_id, 1.0, projected, "fallback", collision_index
    return None


def reflect_velocity(velocity: np.ndarray, normal: np.ndarray, e_n: float, e_t: float) -> np.ndarray:
    """Reflect normal/tangential velocity components with restitution factors."""
    n = normal / max(np.linalg.norm(normal), 1e-30)
    vn = np.dot(velocity, n) * n
    vt = velocity - vn
    return e_t * vt - e_n * vn


def aggregate_face_impacts(
    n_faces: int,
    face_area_m2: np.ndarray,
    face_centers_mm: np.ndarray,
    face_normals: np.ndarray,
    face_ids: np.ndarray,
    er: np.ndarray,
    parcel_mass_kg: np.ndarray,
    observation_window_s: float,
    rho_wall: float = RHO_WALL,
) -> dict:
    """Aggregate ER, sediment flux and thickness loss per STL face."""
    mass = np.asarray(np.bincount(face_ids, weights=parcel_mass_kg, minlength=n_faces), dtype=np.float64)
    target_mass = np.asarray(np.bincount(face_ids, weights=parcel_mass_kg * er, minlength=n_faces), dtype=np.float64)
    er_face = np.divide(target_mass, mass, out=np.zeros_like(target_mass), where=mass > 0)
    mass_flux = np.divide(mass, face_area_m2 * observation_window_s, out=np.zeros_like(mass), where=face_area_m2 > 0)
    dhdt = 3.6e6 * np.divide(target_mass, rho_wall * face_area_m2 * observation_window_s, out=np.zeros_like(target_mass), where=face_area_m2 > 0)
    return {
        "face_mass_kg": mass,
        "face_target_mass_kg": target_mass,
        "ER_mass_ratio_face": er_face,
        "sediment_mass_flux_face_kg_m2_s": mass_flux,
        "dhdt_face_mm_h": dhdt,
        "face_centers_mm": face_centers_mm,
        "face_normals": face_normals,
        "face_area_m2": face_area_m2,
    }


def _metric_pair(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> dict:
    if mask.sum() < 3 or np.all(a[mask] == a[mask][0]) or np.all(b[mask] == b[mask][0]):
        return {"n": int(mask.sum()), "spearman_rho": None, "pearson_r": None, "nrmse": None, "top10_iou": None}
    av, bv = a[mask], b[mask]
    srho, sp = spearmanr(av, bv)
    pr, pp = pearsonr(av, bv)
    an = av / max(float(av.max()), 1e-30)
    bn = bv / max(float(bv.max()), 1e-30)
    nrmse = float(np.sqrt(np.mean((an - bn) ** 2)))
    at, bt = np.percentile(av, 90), np.percentile(bv, 90)
    union = np.sum((av >= at) | (bv >= bt))
    iou = float(np.sum((av >= at) & (bv >= bt)) / max(union, 1))
    return {
        "n": int(mask.sum()),
        "spearman_rho": round(float(srho), 5),
        "spearman_p": float(sp),
        "pearson_r": round(float(pr), 5),
        "pearson_p": float(pp),
        "nrmse_normalized": round(nrmse, 5),
        "top10_iou": round(iou, 5),
    }


def _sample_model_on_faces(mesh: dict, gpr: dict, rotation: dict) -> dict:
    """Map analytic global model values to local STL face centres."""
    R = rotation["R"]
    nozzle_pos = rotation["nozzle_pos"]
    face_global = mesh["face_centers_mm"] @ R + nozzle_pos
    tree = cKDTree(np.asarray(gpr["xyz"], dtype=np.float64))
    dist, idx = tree.query(face_global, k=1)
    e_final = np.asarray(gpr["E_final"], dtype=np.float64)[idx]
    er_truth = np.asarray(gpr.get("ER_mansouri_truth", gpr["E_final"]), dtype=np.float64)[idx]
    vp = np.asarray(gpr.get("Vp", np.ones(len(gpr["xyz"]))), dtype=np.float64)[idx]
    return {
        "face_global_mm": face_global,
        "model_idx": idx,
        "model_dist_mm": dist,
        "E_final": e_final,
        "ER_truth": er_truth,
        "Vp": vp,
    }


def _run_tracking_legacy(
    cfg: TrackingConfig,
    output_dir: Path,
    stl: dict,
    cfd: dict,
    rng: np.random.Generator,
    rotation: dict | None = None,
) -> dict:
    """Reproduce the legacy semi-implicit endpoint-proximity tracker."""
    if rotation is None:
        data = dict(np.load(str(output_dir / "jet_rotation.npz")))
        rotation = {"R": np.asarray(data["R"], dtype=np.float64), "nozzle_pos": np.asarray(data["nozzle_pos"], dtype=np.float64)}
    centers = cfd["cell_centers_mm"]
    U = cfd["U_ms"]
    tree_u = cKDTree(centers)
    tree_face = stl["regular_face_tree"]
    bounds = np.asarray(cfd["center_bounds_mm"], dtype=np.float64)
    R = rotation["R"]
    gravity_local = np.array([0.0, 0.0, -9.81]) @ R.T

    R_jet = cfg.jet_diameter_mm / 2.0
    r = R_jet * np.sqrt(rng.uniform(0.0, 1.0, cfg.n_parcels))
    theta = rng.uniform(0.0, 2.0 * np.pi, cfg.n_parcels)
    pos0 = np.zeros((cfg.n_parcels, 3), dtype=np.float64)
    pos0[:, 1] = r * np.cos(theta)
    pos0[:, 2] = r * np.sin(theta)
    vel0 = np.zeros_like(pos0)
    vel0[:, 0] = cfg.jet_speed_ms
    diam = sample_rosin_rammler(rng, cfg.n_parcels, cfg)
    particle_mass, parcel_mass, total_injection_mass = compute_parcel_masses(diam, cfg)

    impact_rows = []
    terminations = Counter()
    fallback_count = 0
    intersection_count = 0
    max_speed_seen = 0.0
    hit_parcels = set()

    for parcel_id in range(cfg.n_parcels):
        pos = pos0[parcel_id].copy()
        vel = vel0[parcel_id].copy()
        tau_p = cfg.rho_particle_kg_m3 * diam[parcel_id] ** 2 / (18.0 * 1000.0e-6)
        n_impacts = 0
        termination_reason = "max_time"
        for step in range(cfg.max_steps):
            nearest_dist, cell_idx = tree_u.query(pos)
            if nearest_dist > cfg.max_query_distance_mm:
                termination_reason = "outside_cfd_field"
                break
            uf = U[cell_idx]
            slip = uf - vel
            slip_mag = float(np.linalg.norm(slip))
            re_p = 1000.0 * slip_mag * diam[parcel_id] / (1000.0e-6) + 1e-12
            cd_corr = 1.0 + 0.15 * re_p**0.687
            # Semi-implicit drag update. Explicit Euler is unstable when the
            # sampled Rosin-Rammler tail contains d << 50 um (dt >> tau_p).
            tau_eff = max(tau_p / cd_corr, 1e-12)
            weight = cfg.dt_s / tau_eff
            gravity_term = gravity_local * cfg.dt_s if cfg.include_gravity else 0.0
            vel_new = (vel + weight * uf + gravity_term) / (1.0 + weight)
            start = pos.copy()
            end = pos + vel_new * cfg.dt_s * 1000.0
            max_speed_seen = max(max_speed_seen, float(np.linalg.norm(vel_new)))
            hit = find_segment_hit(
                start,
                end,
                stl,
                tree_face,
                cfg.face_fallback_distance_mm,
                collision_policy="legacy_proximity_v1",
            )
            if hit is not None:
                face_id, t_hit, hit_pos, hit_kind, collision_index = hit
                speed = float(np.linalg.norm(vel_new))
                normal = stl["collision_face_normals"][collision_index].copy()
                incident = vel_new / max(speed, 1e-30)
                # Orient the normal against incoming motion before reflecting;
                # this keeps the post-impact parcel on the fluid side.
                if np.dot(incident, normal) > 0.0:
                    normal = -normal
                alpha = float(np.clip(
                    np.degrees(np.arcsin(np.clip(abs(np.dot(incident, normal)), 0.0, 1.0))),
                    0.5,
                    89.5,
                ))
                er = float(mansouri_er(np.array([speed]), np.array([alpha]))[0])
                impact_rows.append({
                    "parcel_id": parcel_id,
                    "impact_index": n_impacts,
                    "time_s": (step + t_hit) * cfg.dt_s,
                    "face_id": face_id,
                    "parent_face_id": face_id,
                    "collision_triangle_id": collision_index,
                    "position_local_mm": hit_pos,
                    "velocity_ms": vel_new.copy(),
                    "Vp_ms": speed,
                    "alpha_deg": alpha,
                    "diameter_m": diam[parcel_id],
                    "particle_mass_kg": particle_mass[parcel_id],
                    "parcel_mass_kg": parcel_mass[parcel_id],
                    "ER_mansouri": er,
                    "hit_kind": 0 if hit_kind == "intersection" else 1,
                })
                hit_parcels.add(parcel_id)
                n_impacts += 1
                if hit_kind == "intersection":
                    intersection_count += 1
                else:
                    fallback_count += 1
                if n_impacts >= cfg.max_impacts:
                    termination_reason = "max_impacts"
                    break
                vel = reflect_velocity(vel_new, normal, cfg.e_normal, cfg.e_tangent)
                if np.linalg.norm(vel) < cfg.min_speed_ms:
                    termination_reason = "low_speed_after_reflect"
                    break
                pos = hit_pos + normal * cfg.hit_epsilon_mm
                continue

            pos, vel = end, vel_new
            if np.linalg.norm(vel) < cfg.min_speed_ms:
                termination_reason = "low_speed"
                break
            if np.any(pos < bounds[0] - 50.0) or np.any(pos > bounds[1] + 50.0):
                termination_reason = "outside_domain"
                break
        else:
            termination_reason = "max_time"
        if parcel_id not in hit_parcels and termination_reason == "max_time":
            termination_reason = "no_wall_hit"
        terminations[termination_reason] += 1

    return {
        "impact_rows": impact_rows,
        "parcel_diameter_m": diam,
        "parcel_mass_kg": parcel_mass,
        "total_injection_mass_kg": total_injection_mass,
        "terminations": terminations,
        "intersection_count": intersection_count,
        "fallback_count": fallback_count,
        "near_miss_count": 0,
        "substep_count": cfg.n_parcels * cfg.max_steps,
        "max_speed_seen_ms": max_speed_seen,
        "hit_parcels": hit_parcels,
    }


def _drag_state(
    position_mm: np.ndarray,
    velocity_ms: np.ndarray,
    fluid_velocity_ms: np.ndarray,
    tau_eff_s: float,
    dt_s: float,
    gravity_ms2: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance constant-fluid drag analytically over one substep."""
    tau = max(float(tau_eff_s), 1e-12)
    decay = float(np.exp(-dt_s / tau))
    if gravity_ms2 is None:
        equilibrium = fluid_velocity_ms
    else:
        equilibrium = fluid_velocity_ms + gravity_ms2 * tau
    new_velocity = equilibrium + (velocity_ms - equilibrium) * decay
    displacement_m = equilibrium * dt_s + (velocity_ms - equilibrium) * tau * (1.0 - decay)
    return position_mm + displacement_m * 1000.0, new_velocity


def _effective_drag_tau_s(
    particle_tau_s: float,
    diameter_m: float,
    particle_velocity_ms: np.ndarray,
    fluid_velocity_ms: np.ndarray,
) -> float:
    """Apply the Schiller-Naumann finite-Re correction to particle response time."""
    slip_mag = float(np.linalg.norm(fluid_velocity_ms - particle_velocity_ms))
    re_p = 1000.0 * slip_mag * diameter_m / (1000.0e-6) + 1e-12
    return max(particle_tau_s / (1.0 + 0.15 * re_p**0.687), 1e-12)


def _strict_particle_state(
    tree_u: cKDTree,
    velocity_field_ms: np.ndarray,
    position_mm: np.ndarray,
    particle_velocity_ms: np.ndarray,
    particle_tau_s: float,
    diameter_m: float,
    dt_s: float,
    gravity_ms2: np.ndarray | None,
    interpolation: str,
    integrator: str,
    fluid_start_ms: np.ndarray | None = None,
    profile_counter: Counter | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one strict-tracking interval with an explicit integrator policy."""
    if integrator not in {"frozen_exponential_v1", "midpoint_exponential_v2"}:
        raise ValueError(
            "particle_integrator must be 'frozen_exponential_v1' or 'midpoint_exponential_v2'"
        )
    # Reuse the start-of-substep CFD sample already obtained for CFL control.
    if fluid_start_ms is None:
        _, _, fluid_start = _sample_cfd_velocity(
            tree_u, velocity_field_ms, position_mm, interpolation, profile_counter
        )
    else:
        fluid_start = np.asarray(fluid_start_ms, dtype=np.float64)
    tau_start = _effective_drag_tau_s(
        particle_tau_s, diameter_m, particle_velocity_ms, fluid_start
    )
    if integrator == "frozen_exponential_v1" or dt_s <= 0.0:
        return _drag_state(
            position_mm, particle_velocity_ms, fluid_start, tau_start, dt_s, gravity_ms2
        )
    midpoint_position, midpoint_velocity = _drag_state(
        position_mm,
        particle_velocity_ms,
        fluid_start,
        tau_start,
        0.5 * dt_s,
        gravity_ms2,
    )
    _, _, fluid_midpoint = _sample_cfd_velocity(
        tree_u, velocity_field_ms, midpoint_position, interpolation, profile_counter
    )
    tau_midpoint = _effective_drag_tau_s(
        particle_tau_s, diameter_m, midpoint_velocity, fluid_midpoint
    )
    return _drag_state(
        position_mm,
        particle_velocity_ms,
        fluid_midpoint,
        tau_midpoint,
        dt_s,
        gravity_ms2,
    )


def _local_cell_spacing_mm(tree_u: cKDTree, position_mm: np.ndarray) -> float:
    """Estimate local tracking length from nearest cell-centre spacing."""
    distances, _ = tree_u.query(position_mm, k=2)
    values = np.atleast_1d(distances)
    spacing = float(values[-1]) if len(values) > 1 else float(values[0])
    return max(spacing, 0.1)


def _sample_cfd_velocity(
    tree_u: cKDTree,
    velocity_ms: np.ndarray,
    position_mm: np.ndarray,
    interpolation: str,
    profile_counter: Counter | None = None,
) -> tuple[float, float, np.ndarray]:
    """Return nearest-field distance, local spacing, and sampled velocity."""
    if interpolation not in {"nearest_cell_v1", "idw_k8_v1"}:
        raise ValueError("cfd_velocity_interpolation must be 'nearest_cell_v1' or 'idw_k8_v1'")
    n_neighbors = min(8 if interpolation == "idw_k8_v1" else 2, len(velocity_ms))
    if profile_counter is not None:
        profile_counter["profile_cfd_query_count"] += 1
        profile_counter["profile_cfd_neighbor_count"] += n_neighbors
    distances, indices = tree_u.query(position_mm, k=n_neighbors)
    distances = np.atleast_1d(distances).astype(np.float64, copy=False)
    indices = np.atleast_1d(indices).astype(np.int64, copy=False)
    order = np.lexsort((indices, distances))
    distances = distances[order]
    indices = indices[order]
    nearest_distance = float(distances[0])
    spacing = max(float(distances[1] if len(distances) > 1 else distances[0]), 0.1)
    if interpolation == "nearest_cell_v1" or nearest_distance <= 1e-12:
        return nearest_distance, spacing, np.asarray(velocity_ms[indices[0]], dtype=np.float64)
    weights = 1.0 / np.maximum(distances, 1e-12) ** 2
    sampled = np.average(velocity_ms[indices], axis=0, weights=weights)
    return nearest_distance, spacing, np.asarray(sampled, dtype=np.float64)


def _surface_tracking_length_mm(
    mesh: dict,
    position_mm: np.ndarray,
    profile_counter: Counter | None = None,
) -> float:
    """Return a conservative near-wall length without a proximity hit shell."""
    radii = np.asarray(mesh["face_radii_mm"], dtype=np.float64)
    floor = mesh.get("surface_tracking_floor_mm")
    if floor is None:
        floor = max(0.05, 0.25 * float(np.median(radii)))
        mesh["surface_tracking_floor_mm"] = floor
    clearances: list[float] = []
    regular_idx = mesh.get("regular_face_indices")
    if regular_idx is None:
        regular_idx = np.arange(len(radii), dtype=np.int64)
        mesh["regular_face_indices"] = regular_idx
    else:
        regular_idx = np.asarray(regular_idx, dtype=np.int64)
    regular_tree = mesh.get("regular_face_tree")
    if regular_tree is not None and len(regular_idx):
        if profile_counter is not None:
            profile_counter["profile_surface_tree_query_count"] += 1
        distance, local_idx = regular_tree.query(position_mm)
        global_idx = int(regular_idx[int(local_idx)])
        clearances.append(float(distance) - float(radii[global_idx]))
    large_idx = np.asarray(mesh.get("large_face_indices", []), dtype=np.int64)
    large_tree = mesh.get("large_face_tree")
    if large_tree is not None and len(large_idx):
        if profile_counter is not None:
            profile_counter["profile_surface_tree_query_count"] += 1
        distance, local_idx = large_tree.query(position_mm)
        global_idx = int(large_idx[int(local_idx)])
        clearances.append(float(distance) - float(radii[global_idx]))
    if not clearances:
        return floor
    return max(floor, min(clearances))


def _strict_impact_row(
    parcel_id: int,
    impact_index: int,
    time_s: float,
    face_id: int,
    collision_index: int,
    hit_pos: np.ndarray,
    velocity: np.ndarray,
    normal: np.ndarray,
    diameter_m: float,
    particle_mass_kg: float,
    parcel_mass_kg: float,
) -> tuple[dict, np.ndarray]:
    speed = float(np.linalg.norm(velocity))
    incident = velocity / max(speed, 1e-30)
    oriented_normal = normal / max(float(np.linalg.norm(normal)), 1e-30)
    if np.dot(incident, oriented_normal) > 0.0:
        oriented_normal = -oriented_normal
    alpha = float(np.clip(
        np.degrees(np.arcsin(np.clip(abs(np.dot(incident, oriented_normal)), 0.0, 1.0))),
        0.5,
        89.5,
    ))
    er = float(mansouri_er(np.array([speed]), np.array([alpha]))[0])
    return {
        "parcel_id": parcel_id,
        "impact_index": impact_index,
        "time_s": time_s,
        "face_id": face_id,
        "parent_face_id": face_id,
        "collision_triangle_id": collision_index,
        "position_local_mm": hit_pos,
        "velocity_ms": velocity.copy(),
        "Vp_ms": speed,
        "alpha_deg": alpha,
        "diameter_m": diameter_m,
        "particle_mass_kg": particle_mass_kg,
        "parcel_mass_kg": parcel_mass_kg,
        "ER_mansouri": er,
        "hit_kind": 0,
    }, oriented_normal


def _run_tracking_strict(
    cfg: TrackingConfig,
    output_dir: Path,
    stl: dict,
    cfd: dict,
    rng: np.random.Generator,
    rotation: dict | None = None,
) -> dict:
    """Trace parcels with adaptive strict swept-segment collision events."""
    if rotation is None:
        data = dict(np.load(str(output_dir / "jet_rotation.npz")))
        rotation = {"R": np.asarray(data["R"], dtype=np.float64), "nozzle_pos": np.asarray(data["nozzle_pos"], dtype=np.float64)}
    centers = cfd["cell_centers_mm"]
    U = cfd["U_ms"]
    tree_u = cKDTree(centers)
    tree_face = stl["regular_face_tree"]
    bounds = np.asarray(cfd["center_bounds_mm"], dtype=np.float64)
    R = rotation["R"]
    gravity_local = np.array([0.0, 0.0, -9.81]) @ R.T if cfg.include_gravity else None

    radius = cfg.jet_diameter_mm / 2.0
    radial = radius * np.sqrt(rng.uniform(0.0, 1.0, cfg.n_parcels))
    theta = rng.uniform(0.0, 2.0 * np.pi, cfg.n_parcels)
    pos0 = np.zeros((cfg.n_parcels, 3), dtype=np.float64)
    pos0[:, 1] = radial * np.cos(theta)
    pos0[:, 2] = radial * np.sin(theta)
    vel0 = np.zeros_like(pos0)
    vel0[:, 0] = cfg.jet_speed_ms
    diam = sample_rosin_rammler(rng, cfg.n_parcels, cfg)
    particle_mass, parcel_mass, total_injection_mass = compute_parcel_masses(diam, cfg)

    impact_rows: list[dict] = []
    terminations = Counter()
    diagnostics = Counter()
    if cfg.collision_profile:
        diagnostics["_profile_enabled"] = 1
    max_speed_seen = 0.0
    hit_parcels: set[int] = set()
    substep_count = 0

    for parcel_id in range(cfg.n_parcels):
        pos = pos0[parcel_id].copy()
        vel = vel0[parcel_id].copy()
        tau_p = cfg.rho_particle_kg_m3 * diam[parcel_id] ** 2 / (18.0 * 1000.0e-6)
        n_impacts = 0
        elapsed_s = 0.0
        termination_reason = "max_time"
        suppressed_face: int | None = None
        suppressed_normal: np.ndarray | None = None
        suppressed_hit_position: np.ndarray | None = None

        while elapsed_s < cfg.max_time_s - 1e-15:
            nearest_dist, cell_spacing, uf = _sample_cfd_velocity(
                tree_u,
                U,
                pos,
                cfg.cfd_velocity_interpolation,
                diagnostics if cfg.collision_profile else None,
            )
            if nearest_dist > cfg.max_query_distance_mm:
                termination_reason = "outside_cfd_field"
                break
            frozen_tau_eff = _effective_drag_tau_s(
                tau_p, diam[parcel_id], vel, uf
            )
            local_spacing = min(
                cell_spacing,
                _surface_tracking_length_mm(
                    stl,
                    pos,
                    diagnostics if cfg.collision_profile else None,
                ),
            )
            speed_scale = max(float(np.linalg.norm(vel)), float(np.linalg.norm(uf)), cfg.min_speed_ms)
            cfl_dt = cfg.particle_cfl * local_spacing / (speed_scale * 1000.0)
            remaining_total = cfg.max_time_s - elapsed_s
            sub_dt = min(
                cfg.dt_s,
                cfg.strict_max_substep_s,
                max(cfg.min_substep_s, cfl_dt),
                remaining_total,
            )

            remaining_substep = sub_dt
            collision_events = 0
            while remaining_substep > 1e-15:
                substep_count += 1
                start = pos.copy()
                if cfg.particle_integrator == "frozen_exponential_v1":
                    end, vel_end = _drag_state(
                        start,
                        vel,
                        uf,
                        frozen_tau_eff,
                        remaining_substep,
                        gravity_local,
                    )
                else:
                    end, vel_end = _strict_particle_state(
                        tree_u,
                        U,
                        start,
                        vel,
                        tau_p,
                        diam[parcel_id],
                        remaining_substep,
                        gravity_local,
                        cfg.cfd_velocity_interpolation,
                        cfg.particle_integrator,
                        fluid_start_ms=uf,
                        profile_counter=diagnostics if cfg.collision_profile else None,
                    )
                hit = find_segment_hit(
                    start,
                    end,
                    stl,
                    tree_face,
                    cfg.face_fallback_distance_mm,
                    collision_policy="strict_v2",
                    suppressed_parent_face_id=suppressed_face,
                    diagnostics=diagnostics,
                    strict_near_miss_distance_diagnostics=(
                        cfg.strict_near_miss_distance_diagnostics
                    ),
                )
                if hit is None:
                    pos, vel = end, vel_end
                    max_speed_seen = max(max_speed_seen, float(np.linalg.norm(vel)))
                    elapsed_s += remaining_substep
                    remaining_substep = 0.0
                    if (
                        suppressed_face is not None
                        and suppressed_normal is not None
                        and suppressed_hit_position is not None
                    ):
                        signed_exit = float(np.dot(pos - suppressed_hit_position, suppressed_normal))
                        if signed_exit > 2.0 * cfg.hit_epsilon_mm:
                            suppressed_face = None
                            suppressed_normal = None
                            suppressed_hit_position = None
                    break

                face_id, t_hit, hit_pos, hit_kind, collision_index = hit
                if hit_kind != "intersection":
                    raise RuntimeError("strict_v2 accepted a non-intersection impact")
                hit_dt = max(0.0, min(1.0, t_hit)) * remaining_substep
                if cfg.particle_integrator == "frozen_exponential_v1":
                    _, hit_velocity = _drag_state(
                        start,
                        vel,
                        uf,
                        frozen_tau_eff,
                        hit_dt,
                        gravity_local,
                    )
                else:
                    _, hit_velocity = _strict_particle_state(
                        tree_u,
                        U,
                        start,
                        vel,
                        tau_p,
                        diam[parcel_id],
                        hit_dt,
                        gravity_local,
                        cfg.cfd_velocity_interpolation,
                        cfg.particle_integrator,
                        fluid_start_ms=uf,
                        profile_counter=diagnostics if cfg.collision_profile else None,
                    )
                row, normal = _strict_impact_row(
                    parcel_id,
                    n_impacts,
                    elapsed_s + hit_dt,
                    face_id,
                    collision_index,
                    hit_pos,
                    hit_velocity,
                    stl["collision_face_normals"][collision_index],
                    diam[parcel_id],
                    particle_mass[parcel_id],
                    parcel_mass[parcel_id],
                )
                impact_rows.append(row)
                hit_parcels.add(parcel_id)
                n_impacts += 1
                max_speed_seen = max(max_speed_seen, row["Vp_ms"])
                elapsed_s += hit_dt
                remaining_substep -= hit_dt
                if n_impacts >= cfg.max_impacts:
                    termination_reason = "max_impacts"
                    remaining_substep = 0.0
                    break
                vel = reflect_velocity(hit_velocity, normal, cfg.e_normal, cfg.e_tangent)
                if np.linalg.norm(vel) < cfg.min_speed_ms:
                    termination_reason = "low_speed_after_reflect"
                    remaining_substep = 0.0
                    break
                epsilon = min(max(cfg.hit_epsilon_mm, 1e-6), 0.1 * float(stl["face_radii_mm"][collision_index]))
                pos = hit_pos + normal * epsilon
                suppressed_face = face_id
                suppressed_normal = normal
                suppressed_hit_position = hit_pos.copy()
                collision_events += 1
                if collision_events >= cfg.max_collision_events_per_step:
                    diagnostics["collision_event_limit"] += 1
                    termination_reason = "collision_event_limit"
                    remaining_substep = 0.0
                    break
                if remaining_substep <= cfg.min_substep_s:
                    elapsed_s += remaining_substep
                    remaining_substep = 0.0

            if termination_reason in {"max_impacts", "low_speed_after_reflect", "collision_event_limit"}:
                break
            if np.linalg.norm(vel) < cfg.min_speed_ms:
                termination_reason = "low_speed"
                break
            if np.any(pos < bounds[0] - 50.0) or np.any(pos > bounds[1] + 50.0):
                termination_reason = "outside_domain"
                break
        else:
            termination_reason = "max_time"
        if parcel_id not in hit_parcels and termination_reason == "max_time":
            termination_reason = "no_wall_hit"
        terminations[termination_reason] += 1

    return {
        "impact_rows": impact_rows,
        "parcel_diameter_m": diam,
        "parcel_mass_kg": parcel_mass,
        "total_injection_mass_kg": total_injection_mass,
        "terminations": terminations,
        "intersection_count": len(impact_rows),
        "fallback_count": 0,
        "near_miss_count": int(diagnostics["endpoint_near_miss"]),
        "collision_event_limit_count": int(diagnostics["collision_event_limit"]),
        "collision_profile": {
            key.removeprefix("profile_"): int(value)
            for key, value in diagnostics.items()
            if key.startswith("profile_")
        } if cfg.collision_profile else None,
        "substep_count": substep_count,
        "max_speed_seen_ms": max_speed_seen,
        "hit_parcels": hit_parcels,
    }


def _run_tracking(
    cfg: TrackingConfig,
    output_dir: Path,
    stl: dict,
    cfd: dict,
    rng: np.random.Generator,
    rotation: dict,
) -> dict:
    if cfg.collision_policy == "strict_v2":
        return _run_tracking_strict(cfg, output_dir, stl, cfd, rng, rotation)
    if cfg.collision_policy == "legacy_proximity_v1":
        return _run_tracking_legacy(cfg, output_dir, stl, cfd, rng, rotation)
    raise ValueError("collision_policy must be 'strict_v2' or 'legacy_proximity_v1'")


def _rows_to_arrays(rows: list[dict]) -> dict:
    if not rows:
        return {
            "parcel_id": np.empty(0, dtype=np.int32),
            "impact_index": np.empty(0, dtype=np.int16),
            "time_s": np.empty(0, dtype=np.float64),
            "face_id": np.empty(0, dtype=np.int64),
            "parent_face_id": np.empty(0, dtype=np.int64),
            "collision_triangle_id": np.empty(0, dtype=np.int64),
            "position_local_mm": np.empty((0, 3), dtype=np.float64),
            "velocity_ms": np.empty((0, 3), dtype=np.float64),
            "Vp_ms": np.empty(0, dtype=np.float64),
            "alpha_deg": np.empty(0, dtype=np.float64),
            "diameter_m": np.empty(0, dtype=np.float64),
            "particle_mass_kg": np.empty(0, dtype=np.float64),
            "parcel_mass_kg": np.empty(0, dtype=np.float64),
            "ER_mansouri": np.empty(0, dtype=np.float64),
            "hit_kind": np.empty(0, dtype=np.int8),
        }
    return {
        "parcel_id": np.array([r["parcel_id"] for r in rows], dtype=np.int32),
        "impact_index": np.array([r["impact_index"] for r in rows], dtype=np.int16),
        "time_s": np.array([r["time_s"] for r in rows], dtype=np.float64),
        "face_id": np.array([r["face_id"] for r in rows], dtype=np.int64),
        "parent_face_id": np.array([r["parent_face_id"] for r in rows], dtype=np.int64),
        "collision_triangle_id": np.array([r["collision_triangle_id"] for r in rows], dtype=np.int64),
        "position_local_mm": np.array([r["position_local_mm"] for r in rows], dtype=np.float64),
        "velocity_ms": np.array([r["velocity_ms"] for r in rows], dtype=np.float64),
        "Vp_ms": np.array([r["Vp_ms"] for r in rows], dtype=np.float64),
        "alpha_deg": np.array([r["alpha_deg"] for r in rows], dtype=np.float64),
        "diameter_m": np.array([r["diameter_m"] for r in rows], dtype=np.float64),
        "particle_mass_kg": np.array([r["particle_mass_kg"] for r in rows], dtype=np.float64),
        "parcel_mass_kg": np.array([r["parcel_mass_kg"] for r in rows], dtype=np.float64),
        "ER_mansouri": np.array([r["ER_mansouri"] for r in rows], dtype=np.float64),
        "hit_kind": np.array([r["hit_kind"] for r in rows], dtype=np.int8),
    }


def run(cfg: TrackingConfig | None = None) -> dict:
    execution_target = require_remote_execution_target()
    cfg = cfg or TrackingConfig()
    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", cfg.output_tag).strip("._") or "latest"
    out_npz = OUT_NPZ if safe_tag == "latest" else OUTPUT_DIR / f"particle_hits_cfd_{safe_tag}.npz"
    out_json = OUT_JSON if safe_tag == "latest" else OUTPUT_DIR / f"step30_cfd_summary_{safe_tag}.json"
    out_png = OUT_PNG if safe_tag == "latest" else OUTPUT_DIR / f"particle_tracking_cfd_validation_{safe_tag}.png"
    print("=" * 70)
    print("AEGIS Step-30 CFD-driven particle-wall Mansouri erosion")
    print("=" * 70)
    print(
        f"  CFD={cfg.cfd_variant}  parcels={cfg.n_parcels}  max_impacts={cfg.max_impacts} "
        f" mass_flow={cfg.mass_flow_mode}  collision={cfg.collision_policy}"
    )

    if cfg.collision_policy not in {"strict_v2", "legacy_proximity_v1"}:
        raise ValueError("unsupported collision policy")
    if cfg.strict_max_substep_s <= 0.0 or cfg.strict_max_substep_s < cfg.min_substep_s:
        raise ValueError("strict_max_substep_s must be positive and not below min_substep_s")
    native_variants = {"native", "native_baseline", "native_si"}
    if cfg.cfd_variant == "native_si" and cfg.cfd_field_time is None:
        raise ValueError("native_si requires an explicit OpenFOAM --field-time")
    if cfg.collision_backend == "cfd_topology" and cfg.cfd_variant not in native_variants:
        raise ValueError("cfd_topology backend requires a native same-mesh CFD variant")
    if cfg.collision_backend == "auto" and (WORKSPACE / cfg.topology_path).exists() and cfg.cfd_variant not in native_variants:
        raise ValueError("native topology artifact exists; use a native CFD variant or explicitly select --collision-backend stl")
    native_si_dir = WORKSPACE / cfg.native_si_directory
    cfd = load_cfd_velocity_field(
        OUTPUT_DIR,
        cfg.cfd_variant,
        cfg.cfd_field_time,
        native_si_dir=native_si_dir if cfg.cfd_variant == "native_si" else None,
    )
    if cfd["speed_stats_ms"]["max"] >= cfg.max_accepted_cfd_speed_ms:
        raise ValueError(
            f"{cfg.cfd_variant} CFD field max speed {cfd['speed_stats_ms']['max']:.1f} m/s "
            f"does not satisfy the strict quality limit <{cfg.max_accepted_cfd_speed_ms:.1f} m/s; "
            "inspect MRF export before particle tracking."
        )
    stl = load_collision_geometry(cfg)
    if stl["backend"] == "cfd_topology":
        if stl["source_coordinate_units"] != cfd["source_coordinate_units"]:
            raise ValueError(
                "CFD field and topology source coordinate units differ: "
                f"{cfd['source_coordinate_units']} vs {stl['source_coordinate_units']}"
            )
        if cfg.cfd_variant == "native_si":
            if stl["time_name"] != cfd["field_time"]:
                raise ValueError("native_si CFD field and topology must use the same OpenFOAM time")
            manifest_topology = cfd["artifact_manifest"]["topology"]
            topology_path = WORKSPACE / cfg.topology_path
            if stl["mesh_signature"] != manifest_topology["mesh_signature"]:
                raise ValueError("native_si topology signature differs from the field manifest")
            if _sha256(topology_path) != manifest_topology["sha256"]:
                raise ValueError("native_si topology artifact hash differs from the field manifest")
        wall_xyz = stl["face_centers_mm"]
        preflight = {
            "status": "native_topology",
            "wall_points": int(len(wall_xyz)),
            "wall_points_in_stl_bbox_plus_tolerance_pct": 100.0,
            "warning": "collision and face aggregation use the same native OpenFOAM bucket patch topology",
        }
    else:
        wall_xyz = load_wall_points(OUTPUT_DIR / "bucket_wss_turb.raw")
        preflight = geometry_preflight(stl, wall_xyz)
    reference_transform = _load_reference_transform(OUTPUT_DIR, cfg, cfd)
    artifact_descriptor = None
    particle_descriptor = None
    immutable_run_dir = None
    immutable_temp_dir = None
    immutable_temp_context = None
    if cfg.immutable_output:
        if cfg.cfd_variant != "native_si" or stl["backend"] != "cfd_topology":
            raise ValueError("immutable output requires native_si CFD with cfd_topology")
        if cfd["artifact_manifest"].get("version") != "step30-native-si-manifest-v2":
            raise ValueError("immutable output requires a native SI manifest v2")
        from step30_artifact_identity import (
            build_artifact_set_descriptor,
            build_particle_run_descriptor,
            file_sha256,
        )

        topology_path = WORKSPACE / cfg.topology_path
        artifact_descriptor = build_artifact_set_descriptor(cfd, stl, topology_path)
        particle_descriptor = build_particle_run_descriptor(
            artifact_descriptor["artifact_set_id"],
            cfg,
            collision_backend=stl["backend"],
            implementation_hashes={
                "particle_tracking": file_sha256(Path(__file__)),
                "cfd_topology": file_sha256(WORKSPACE / "step30_cfd_topology.py"),
                "artifact_identity": file_sha256(WORKSPACE / "step30_artifact_identity.py"),
            },
            input_hashes={
                "analytic_reference_transform": reference_transform["sha256"],
                "vertices_full_gpr": file_sha256(OUTPUT_DIR / "vertices_full_gpr.npz"),
            },
        )
        artifact_root = OUTPUT_DIR / "step30_runs" / artifact_descriptor["artifact_set_id"]
        immutable_run_dir = (
            artifact_root / cfg.collision_policy / particle_descriptor["particle_run_id"]
        )
        if immutable_run_dir.exists():
            raise FileExistsError(f"immutable Step30 run already exists: {immutable_run_dir}")
        artifact_root.mkdir(parents=True, exist_ok=True)
        artifact_manifest_path = artifact_root / "artifact_set_manifest.json"
        artifact_manifest_text = json.dumps(artifact_descriptor, ensure_ascii=False, indent=2)
        if artifact_manifest_path.exists():
            if json.loads(artifact_manifest_path.read_text(encoding="utf-8")) != artifact_descriptor:
                raise ValueError("existing artifact-set manifest differs from computed identity")
        else:
            artifact_manifest_path.write_text(artifact_manifest_text, encoding="utf-8")
        policy_root = artifact_root / cfg.collision_policy
        policy_root.mkdir(exist_ok=True)
        immutable_temp_context = tempfile.TemporaryDirectory(
            prefix=f".{particle_descriptor['particle_run_id']}.",
            dir=policy_root,
        )
        immutable_temp_dir = Path(immutable_temp_context.name)
        out_npz = immutable_temp_dir / "particle_hits.npz"
        out_json = immutable_temp_dir / "summary.json"
        out_png = immutable_temp_dir / "validation.png"

    print(f"[PT] CFD cells={cfd['n_cells']:,}, backend={stl['backend']}, parent faces={stl['n_parent_faces']:,}, area={stl['surface_area_m2']:.3f} m2")

    rng = np.random.default_rng(cfg.seed)
    tracked = _run_tracking(cfg, OUTPUT_DIR, stl, cfd, rng, reference_transform)
    arrays = _rows_to_arrays(tracked["impact_rows"])
    n_impacts = len(arrays["face_id"])
    print(f"[PT] impacts={n_impacts:,} (intersection={tracked['intersection_count']:,}, fallback={tracked['fallback_count']:,})")
    if cfg.collision_policy == "strict_v2" and tracked["fallback_count"] != 0:
        raise RuntimeError("strict_v2 cannot aggregate proximity fallback impacts")
    if cfg.collision_policy == "strict_v2" and tracked.get("collision_event_limit_count", 0) != 0:
        raise RuntimeError("strict_v2 reached the per-substep collision-event limit")

    aggregate = aggregate_face_impacts(
        stl["n_parent_faces"], stl["face_area_m2"], stl["face_centers_mm"], stl["face_normals"],
        arrays["face_id"], arrays["ER_mansouri"], arrays["parcel_mass_kg"], cfg.observation_window_s,
    )

    rotation = reference_transform
    gpr = dict(np.load(str(OUTPUT_DIR / "vertices_full_gpr.npz")))
    model = _sample_model_on_faces(stl, gpr, rotation)
    model_dhdt = 3.6e6 * model["ER_truth"] * cfg.concentration_kg_m3 * model["Vp"] / RHO_WALL
    theta = np.degrees(np.arctan2(model["face_global_mm"][:, 1], model["face_global_mm"][:, 0])) % 360.0
    bucket_width = 360.0 / 21.0
    bucket_margin = bucket_width * 1.3 / 2.0
    bucket0 = (theta <= bucket_margin) | (theta >= 360.0 - bucket_margin)
    distance_median = float(np.median(model["model_dist_mm"]))
    distance_p95 = float(np.percentile(model["model_dist_mm"], 95))
    registration_ok = distance_median <= 10.0 and distance_p95 <= 15.0
    common = bucket0 & (model["model_dist_mm"] <= 15.0)
    exposed = aggregate["face_mass_kg"] > 0
    compare_mask = common & exposed if registration_ok else np.zeros_like(exposed)
    metrics_er = _metric_pair(aggregate["ER_mass_ratio_face"], model["ER_truth"], compare_mask)
    metrics_flux = _metric_pair(aggregate["dhdt_face_mm_h"], model_dhdt, compare_mask)

    R = rotation["R"]
    nozzle_pos = rotation["nozzle_pos"]
    impact_global = arrays["position_local_mm"] @ R + nozzle_pos if n_impacts else np.empty((0, 3))
    np.savez_compressed(
        str(out_npz),
        **arrays,
        hit_pos_global_mm=impact_global.astype(np.float32),
        face_centers_local_mm=stl["face_centers_mm"].astype(np.float32),
        face_normals=stl["face_normals"].astype(np.float32),
        face_area_m2=stl["face_area_m2"].astype(np.float32),
        face_mass_kg=aggregate["face_mass_kg"].astype(np.float32),
        face_target_mass_kg=aggregate["face_target_mass_kg"].astype(np.float32),
        ER_mass_ratio_face=aggregate["ER_mass_ratio_face"].astype(np.float32),
        sediment_mass_flux_face_kg_m2_s=aggregate["sediment_mass_flux_face_kg_m2_s"].astype(np.float32),
        dhdt_face_mm_h=aggregate["dhdt_face_mm_h"].astype(np.float32),
        model_ER_truth_face=model["ER_truth"].astype(np.float32),
        model_E_final_face=model["E_final"].astype(np.float32),
        model_dhdt_face_mm_h=model_dhdt.astype(np.float32),
        gpr_index=model["model_idx"].astype(np.int64),
        registration_distance_mm=model["model_dist_mm"].astype(np.float32),
        model_dist_mm=model["model_dist_mm"].astype(np.float32),
        bucket0_face=bucket0,
        compare_mask=compare_mask,
        collision_backend=np.array(stl["backend"]),
        collision_policy=np.array(cfg.collision_policy),
        collision_algorithm_version=np.array("v2" if cfg.collision_policy == "strict_v2" else "v1"),
        cfd_velocity_interpolation=np.array(
            cfg.cfd_velocity_interpolation if cfg.collision_policy == "strict_v2" else "nearest_cell_v1"
        ),
        particle_integrator=np.array(
            cfg.particle_integrator if cfg.collision_policy == "strict_v2" else "legacy_semi_implicit_v1"
        ),
        face_id_namespace=np.array(stl["face_id_namespace"]),
        mesh_signature=np.array(stl["mesh_signature"]),
        field_time=np.array(cfd["field_time"]),
        source_coordinate_units=np.array(cfd["source_coordinate_units"]),
        internal_coordinate_units=np.array(cfd["internal_coordinate_units"]),
        coordinate_scale_to_mm=np.array(cfd["coordinate_scale_to_mm"]),
        artifact_manifest_file=np.array(cfd["artifact_manifest_file"] or ""),
        artifact_manifest_sha256=np.array(cfd["artifact_manifest_sha256"] or ""),
        polyMesh_signature=np.array(
            cfd["artifact_manifest"].get("polyMesh", {}).get("signature", "")
            if cfd["artifact_manifest"] else ""
        ),
        analytic_reference_transform_sha256=np.array(reference_transform["sha256"]),
        analytic_reference_transform_version=np.array(reference_transform["version"] or ""),
        artifact_set_id=np.array(
            artifact_descriptor["artifact_set_id"] if artifact_descriptor else ""
        ),
        particle_run_id=np.array(
            particle_descriptor["particle_run_id"] if particle_descriptor else ""
        ),
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("CFD-driven particle impacts and Mansouri face erosion", fontsize=12)
    if n_impacts:
        sc = axes[0].scatter(arrays["Vp_ms"], arrays["alpha_deg"], c=np.log10(arrays["ER_mansouri"] + 1e-30), s=8, alpha=0.5, cmap="hot")
        axes[0].set_xlabel("impact Vp [m/s]")
        axes[0].set_ylabel("impact alpha [deg]")
        fig.colorbar(sc, ax=axes[0], label="log10 Mansouri ER")
    else:
        axes[0].text(0.5, 0.5, "No impacts", ha="center", va="center")
    axes[0].set_title(f"{n_impacts:,} wall impacts")
    axes[0].grid(True, alpha=0.3)
    face = stl["face_centers_mm"]
    sc2 = axes[1].scatter(face[:, 0], np.linalg.norm(face[:, 1:3], axis=1), c=np.log10(aggregate["dhdt_face_mm_h"] + 1e-30), s=1, cmap="hot")
    axes[1].set_xlabel("local x [mm]")
    axes[1].set_ylabel("r_yz [mm]")
    axes[1].set_title("face dh/dt [mm/h]")
    fig.colorbar(sc2, ax=axes[1], label="log10 dh/dt")
    axes[1].grid(True, alpha=0.3)
    if compare_mask.sum() >= 3:
        axes[2].scatter(aggregate["ER_mass_ratio_face"][compare_mask], model["ER_truth"][compare_mask], s=2, alpha=0.35)
        axes[2].set_xlabel("CFD particle Mansouri ER [kg/kg]")
        axes[2].set_ylabel("analytic Mansouri ER [kg/kg]")
        axes[2].set_title(f"bucket0 exposed faces, rho={metrics_er['spearman_rho']}")
    else:
        axes[2].text(0.5, 0.5, "Insufficient exposed faces", ha="center", va="center")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_png), dpi=150)
    plt.close(fig)

    hit_stats = {}
    if n_impacts:
        for key in ["Vp_ms", "alpha_deg", "diameter_m", "ER_mansouri"]:
            val = arrays[key]
            hit_stats[key] = {"min": float(val.min()), "max": float(val.max()), "median": float(np.median(val))}

    exposed_er = aggregate["ER_mass_ratio_face"][exposed]
    speed_warning_limit = 3.0 * cfg.jet_speed_ms
    cfd_summary = {
        "variant": cfd["variant"],
        "prefix": cfd["prefix"],
        "field_time": cfd["field_time"],
        "n_cells": cfd["n_cells"],
        "center_bounds_mm": cfd["center_bounds_mm"],
        "source_coordinate_units": cfd["source_coordinate_units"],
        "internal_coordinate_units": cfd["internal_coordinate_units"],
        "coordinate_scale_to_mm": cfd["coordinate_scale_to_mm"],
        "artifact_manifest_file": cfd["artifact_manifest_file"],
            "artifact_manifest_sha256": cfd["artifact_manifest_sha256"],
            "analytic_reference_transform": {
                "file": reference_transform["path"].name,
                "sha256": reference_transform["sha256"],
                "version": reference_transform["version"],
                "convention": reference_transform["convention"],
                "source_geometry_sha256": reference_transform["source_geometry_sha256"],
            },
            "polyMesh_signature": (

            cfd["artifact_manifest"].get("polyMesh", {}).get("signature")
            if cfd["artifact_manifest"] else None
        ),
        "speed_stats_ms": cfd["speed_stats_ms"],
        "quality_status": "warning_velocity_outliers" if cfd["speed_stats_ms"]["max"] > speed_warning_limit else "passed",
        "velocity_warning_limit_ms": speed_warning_limit,
        "solver_semantics": (
            artifact_descriptor["cfd"]["solver_semantics"] if artifact_descriptor else None
        ),
    }
    summary = {
        "version": "step30-cfd-driven-v2" if artifact_descriptor else "step30-cfd-driven-v1",
        "artifact_identity": (
            {
                "artifact_set_id": artifact_descriptor["artifact_set_id"],
                "particle_run_id": particle_descriptor["particle_run_id"],
                "layout": "output/step30_runs/<artifact_set_id>/<collision_policy>/<particle_run_id>",
            }
            if artifact_descriptor
            else None
        ),
        "model_scope": "single-phase OpenFOAM continuous velocity field + one-way particle post-processing; no kinematicCloud wall log",
        "cfd": cfd_summary,
        "geometry": {
            "collision_backend": stl["backend"],
            "face_id_namespace": stl["face_id_namespace"],
            "mesh_signature": stl["mesh_signature"],
            "patch_name": stl["patch_name"],
            "source_time": stl["time_name"],
            "field_time": cfd["field_time"],
            "source_coordinate_units": stl["source_coordinate_units"],
            "internal_coordinate_units": stl["internal_coordinate_units"],
            "coordinate_scale_to_mm": stl["coordinate_scale_to_mm"],
            "n_vertices": int(len(stl["vertices_mm"])),
            "n_collision_triangles": int(len(stl["collision_triangles_mm"])),
            "n_parent_faces": int(stl["n_parent_faces"]),
            "surface_area_m2": stl["surface_area_m2"],
            "bounds_mm": stl["bounds_mm"],
            "preflight": preflight,
        },
        "parameters": {
            "execution_target": execution_target,
            "n_parcels": cfg.n_parcels,
            "n_injected": cfg.n_injected,
            "collision_backend": stl["backend"],
            "collision_policy": cfg.collision_policy,
            "collision_algorithm_version": "v2" if cfg.collision_policy == "strict_v2" else "v1",
            "cfd_variant": cfg.cfd_variant,
            "max_impacts": cfg.max_impacts,
            "seed": cfg.seed,
            "dt_s": cfg.dt_s,
            "max_time_s": cfg.max_time_s,
            "particle_cfl": cfg.particle_cfl,
            "cfd_velocity_interpolation": (
                cfg.cfd_velocity_interpolation if cfg.collision_policy == "strict_v2" else "nearest_cell_v1"
            ),
            "particle_integrator": (
                cfg.particle_integrator if cfg.collision_policy == "strict_v2" else "legacy_semi_implicit_v1"
            ),
            "strict_max_substep_s": cfg.strict_max_substep_s,
            "strict_near_miss_distance_diagnostics": (
                cfg.strict_near_miss_distance_diagnostics
            ),
            "collision_profile": cfg.collision_profile,
            "min_substep_s": cfg.min_substep_s,
            "max_query_distance_mm": cfg.max_query_distance_mm,
            "face_fallback_distance_mm": cfg.face_fallback_distance_mm,
            "hit_epsilon_mm": cfg.hit_epsilon_mm,
            "min_speed_ms": cfg.min_speed_ms,
            "max_collision_events_per_step": cfg.max_collision_events_per_step,
            "output_tag": None if cfg.immutable_output else safe_tag,
            "include_gravity": cfg.include_gravity,
            "e_normal": cfg.e_normal,
            "e_tangent": cfg.e_tangent,
            "diameter_distribution": {
                "type": "Rosin-Rammler",
                "min_m": cfg.d_min_m,
                "max_m": cfg.d_max_m,
                "d50_m": cfg.d50_m,
                "exponent": cfg.rr_exponent,
                "sampling_contract": cfg.psd_sampling,
            },
            "psd_sampling": cfg.psd_sampling,
            "rho_particle_kg_m3": cfg.rho_particle_kg_m3,
            "max_accepted_cfd_speed_ms": cfg.max_accepted_cfd_speed_ms,
            "mass_flow_mode": cfg.mass_flow_mode,
            "nozzle_flow_m3_s": cfg.nozzle_flow_m3_s,
            "sediment_flow_kg_s": cfg.sediment_flow_kg_s,
            "injection_window_s": cfg.injection_window_s,
            "observation_window_s": cfg.observation_window_s,
            "rho_wall_kg_m3": RHO_WALL,
            "rho_wall_alternative_kg_m3": 7850.0,
        },
        "tracking": {
            "n_impacts": n_impacts,
            "intersection_hits": tracked["intersection_count"],
            "fallback_hits": tracked["fallback_count"],
            "near_miss_diagnostics": tracked.get("near_miss_count", 0),
            "collision_event_limit_count": tracked.get("collision_event_limit_count", 0),
            "collision_profile": tracked.get("collision_profile"),
            "substep_count": tracked.get("substep_count", 0),
            "physical_strict_hit_ratio": (
                float(tracked["intersection_count"] / n_impacts) if n_impacts else 1.0
            ),
            "hit_parcels": len(tracked["hit_parcels"]),
            "hit_rate_parcels_pct": float(100.0 * len(tracked["hit_parcels"]) / max(cfg.n_parcels, 1)),
            "max_speed_seen_ms": tracked["max_speed_seen_ms"],
            "termination_counts": dict(tracked["terminations"]),
            "hit_stats": hit_stats,
            "total_injection_mass_kg": tracked["total_injection_mass_kg"],
            "impact_mass_kg": float(arrays["parcel_mass_kg"].sum()) if n_impacts else 0.0,
        },
        "face_erosion": {
            "exposed_faces": int(exposed.sum()),
            "surface_faces": int(stl["n_parent_faces"]),
            "total_target_mass_kg": float(aggregate["face_target_mass_kg"].sum()),
            "ER_face_stats": {"max": float(aggregate["ER_mass_ratio_face"].max()), "median_exposed": float(np.median(exposed_er)) if exposed.any() else 0.0},
            "mass_flux_max_kg_m2_s": float(aggregate["sediment_mass_flux_face_kg_m2_s"].max()),
            "dhdt_max_mm_h": float(aggregate["dhdt_face_mm_h"].max()),
        },
        "validation": {
            "registration_status": "passed" if registration_ok else "failed",
            "registration_distance_mm": {
                "median": distance_median,
                "p95": distance_p95,
                "max": float(model["model_dist_mm"].max()),
            },
            "registration_threshold_mm": {"median": 10.0, "p95": 15.0},
            "mask": "bucket0 angular window + registration distance<=15mm + exposed native parent faces",
            "n_faces_compared": int(compare_mask.sum()),
            "cfd_ER_vs_analytic_Mansouri": metrics_er,
            "cfd_dhdt_vs_analytic_Mansouri_rate": metrics_flux,
        },
        "units": {
            "position": "mm local / global",
            "velocity": "m/s",
            "ER": "kg target per kg sediment",
            "face_area": "m2",
            "dhdt": "mm/h",
            "source_coordinate_units": cfd["source_coordinate_units"],
            "internal_coordinate_units": cfd["internal_coordinate_units"],
            "coordinate_scale_to_mm": cfd["coordinate_scale_to_mm"],
        },
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if immutable_run_dir is not None:
        run_manifest = dict(particle_descriptor)
        run_manifest["outputs"] = {
            "particle_hits": {"file": out_npz.name, "sha256": file_sha256(out_npz)},
            "summary": {"file": out_json.name, "sha256": file_sha256(out_json)},
            "validation": {"file": out_png.name, "sha256": file_sha256(out_png)},
        }
        (immutable_temp_dir / "particle_run_manifest.json").write_text(
            json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if immutable_run_dir.exists():
            raise FileExistsError(f"immutable Step30 run already exists: {immutable_run_dir}")
        immutable_temp_dir.replace(immutable_run_dir)
        immutable_temp_context.cleanup()
        immutable_temp_context = None
        out_npz = immutable_run_dir / out_npz.name
        out_json = immutable_run_dir / out_json.name
        out_png = immutable_run_dir / out_png.name
    print(f"[PT] -> {out_npz}, {out_json}, {out_png}")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CFD-driven particle-wall Mansouri erosion")
    parser.add_argument("--cfd", choices=["baseline", "mrf", "native", "native_baseline", "native_si"], default="native_baseline")
    parser.add_argument("--collision-backend", choices=["auto", "cfd_topology", "stl"], default="auto")
    parser.add_argument(
        "--collision-policy",
        choices=["strict_v2", "legacy_proximity_v1"],
        default="strict_v2",
    )
    parser.add_argument("--topology", default="output/bucket_topology.npz")
    parser.add_argument("--native-si-directory", default="output/native_bucket_si")
    parser.add_argument("--field-time", help="OpenFOAM time for native_si fields")
    parser.add_argument("--tag", default="latest", help="output tag; latest uses canonical filenames")
    parser.add_argument(
        "--immutable-output",
        action="store_true",
        help="write an ID-addressed run directory and reject overwrites",
    )
    parser.add_argument("--parcels", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-impacts", type=int, default=3)
    parser.add_argument("--mass-flow", choices=["nominal", "geometric"], default="nominal")
    parser.add_argument(
        "--psd-sampling",
        choices=["number_random_v1", "mass_stratified_v1"],
        default="number_random_v1",
    )
    parser.add_argument("--max-time", type=float, default=0.05)
    parser.add_argument("--dt", type=float, default=1e-5)
    parser.add_argument("--particle-cfl", type=float, default=0.5)
    parser.add_argument("--strict-max-substep", type=float, default=2.5e-6)
    parser.add_argument(
        "--disable-strict-near-miss-distance-diagnostics",
        action="store_true",
        help=(
            "strict-v2 performance A/B: skip exact point-to-triangle distance "
            "after a segment has already missed; never accepts proximity hits"
        ),
    )
    parser.add_argument(
        "--collision-profile",
        action="store_true",
        help="collect strict collision broadphase and triangle-test counters",
    )
    parser.add_argument(
        "--analytic-reference-transform",
        default="jet_rotation.npz",
        help="source-bound local-to-analytic reference transform JSON or legacy NPZ",
    )
    parser.add_argument(
        "--cfd-interpolation",
        choices=["nearest_cell_v1", "idw_k8_v1"],
        default="idw_k8_v1",
    )
    parser.add_argument(
        "--particle-integrator",
        choices=["frozen_exponential_v1", "midpoint_exponential_v2"],
        default="midpoint_exponential_v2",
    )
    parser.add_argument("--max-cfd-speed", type=float, default=500.0)
    parser.add_argument("--gravity", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(TrackingConfig(
        cfd_variant=args.cfd,
        collision_backend=args.collision_backend,
        collision_policy=args.collision_policy,
        topology_path=args.topology,
        native_si_directory=args.native_si_directory,
        cfd_field_time=args.field_time,
        output_tag=args.tag,
        immutable_output=args.immutable_output,
        n_parcels=args.parcels,
        n_injected=args.parcels,
        seed=args.seed,
        max_impacts=args.max_impacts,
        mass_flow_mode=args.mass_flow,
        psd_sampling=args.psd_sampling,
        max_time_s=args.max_time,
        dt_s=args.dt,
        particle_cfl=args.particle_cfl,
        strict_max_substep_s=args.strict_max_substep,
        strict_near_miss_distance_diagnostics=(
            not args.disable_strict_near_miss_distance_diagnostics
        ),
        collision_profile=args.collision_profile,
        analytic_reference_transform_path=args.analytic_reference_transform,
        cfd_velocity_interpolation=args.cfd_interpolation,
        particle_integrator=args.particle_integrator,
        max_accepted_cfd_speed_ms=args.max_cfd_speed,
        include_gravity=args.gravity,
    ))
