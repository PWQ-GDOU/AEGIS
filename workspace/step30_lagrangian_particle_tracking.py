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
import json
import re
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


@dataclass(frozen=True)
class TrackingConfig:
    """Numerical and physical settings for one reproducible run."""

    cfd_variant: str = "native_baseline"
    collision_backend: str = "auto"
    topology_path: str = "output/bucket_topology.npz"
    output_tag: str = "latest"
    n_parcels: int = 2000
    n_injected: int = 2000
    max_impacts: int = 3
    seed: int = 42
    dt_s: float = 1.0e-5
    max_time_s: float = 0.05
    max_query_distance_mm: float = 80.0
    face_fallback_distance_mm: float = 1.5
    hit_epsilon_mm: float = 0.05
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
    raw = re.findall(r"\(([^)]+)\)", match.group(2))
    data = np.array([[float(x) for x in row.split()] for row in raw], dtype=np.float64)
    if data.shape != (expected, 3):
        raise ValueError(f"{path.name}: expected ({expected}, 3), parsed {data.shape}")
    return data


def load_cfd_velocity_field(output_dir: Path, variant: str = "baseline") -> dict:
    """Load and validate cell centres and velocity for baseline or MRF CFD."""
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
    else:
        raise ValueError("cfd_variant must be 'baseline', 'mrf', 'native', or 'native_baseline'")
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing CFD field(s): " + ", ".join(missing))

    Cx = parse_foam_scalar_field(paths["Cx"])
    Cy = parse_foam_scalar_field(paths["Cy"])
    Cz = parse_foam_scalar_field(paths["Cz"])
    U = parse_foam_vector_field(paths["U"])
    if not (len(Cx) == len(Cy) == len(Cz) == len(U)):
        raise ValueError("CFD coordinate and velocity fields have different cell counts")

    cell_centers = np.column_stack([Cx, Cy, Cz])
    if not np.isfinite(cell_centers).all() or not np.isfinite(U).all():
        raise ValueError("CFD fields contain non-finite values")
    speeds = np.linalg.norm(U, axis=1)
    return {
        "cell_centers_mm": cell_centers,
        "U_ms": U,
        "variant": variant,
        "prefix": prefix,
        "field_time": field_time,
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
        "collision_broadphase_radius_mm": broadphase_radius,
        "regular_face_indices": regular_idx,
        "large_face_indices": large_idx,
        "regular_face_tree": cKDTree(centers[regular_idx]) if len(regular_idx) else None,
        "large_face_tree": cKDTree(centers[large_idx]) if len(large_idx) else None,
        "bounds_mm": [xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist()],
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


def sample_rosin_rammler(rng: np.random.Generator, n: int, cfg: TrackingConfig) -> np.ndarray:
    """Sample bounded Rosin-Rammler diameters in metres."""
    u = rng.uniform(1e-8, 1.0 - 1e-8, n)
    d_char = cfg.d50_m / (-np.log(0.5)) ** (1.0 / cfg.rr_exponent)
    diam = d_char * (-np.log1p(-u)) ** (1.0 / cfg.rr_exponent)
    return np.clip(diam, cfg.d_min_m, cfg.d_max_m)


def compute_parcel_masses(diam_m: np.ndarray, cfg: TrackingConfig) -> tuple[np.ndarray, np.ndarray, float]:
    """Return physical particle masses, parcel weights, and total mass."""
    particle_mass = cfg.rho_particle_kg_m3 * (np.pi / 6.0) * diam_m**3
    total_mass = cfg.sediment_flow_kg_s * cfg.injection_window_s
    # Samples follow the number distribution; each sampled physical particle
    # represents an equal count, so the parcel mass is proportional to d^3.
    weights = particle_mass / max(float(particle_mass.sum()), 1e-30)
    parcel_mass = total_mass * weights
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
) -> tuple[int, float, np.ndarray, str, int] | None:
    """Find the first triangle hit; label proximity fallback separately."""
    segment = end - start
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
        return None
    candidates = np.asarray(candidates, dtype=np.int64)
    tri = mesh["collision_triangles_mm"][candidates]
    hit, t = segment_triangle_intersection(start, end, tri)
    if hit.any():
        local = np.flatnonzero(hit)[np.argmin(t[hit])]
        collision_index = int(candidates[local])
        face_id = int(mesh.get("collision_face_ids", mesh.get("parent_face_ids"))[collision_index])
        return face_id, float(t[local]), start + float(t[local]) * segment, "intersection", collision_index

    # A parcel can step over a very thin/open STL triangle. Keep this count
    # separate from strict intersections so diagnostics cannot hide it.
    d2 = _point_triangle_distance(end, tri)
    local = int(np.argmin(d2))
    if np.sqrt(d2[local]) <= fallback_distance_mm:
        collision_index = int(candidates[local])
        face_id = int(mesh.get("collision_face_ids", mesh.get("parent_face_ids"))[collision_index])
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


def _run_tracking(cfg: TrackingConfig, output_dir: Path, stl: dict, cfd: dict, rng: np.random.Generator) -> dict:
    """Trace parcels and return impact records plus termination diagnostics."""
    centers = cfd["cell_centers_mm"]
    U = cfd["U_ms"]
    tree_u = cKDTree(centers)
    tree_face = stl["regular_face_tree"]
    bounds = np.asarray(cfd["center_bounds_mm"], dtype=np.float64)
    R = np.asarray(np.load(str(output_dir / "jet_rotation.npz"))["R"], dtype=np.float64)
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
            hit = find_segment_hit(start, end, stl, tree_face, cfg.face_fallback_distance_mm)
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
        "max_speed_seen_ms": max_speed_seen,
        "hit_parcels": hit_parcels,
    }


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
    cfg = cfg or TrackingConfig()
    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", cfg.output_tag).strip("._") or "latest"
    out_npz = OUT_NPZ if safe_tag == "latest" else OUTPUT_DIR / f"particle_hits_cfd_{safe_tag}.npz"
    out_json = OUT_JSON if safe_tag == "latest" else OUTPUT_DIR / f"step30_cfd_summary_{safe_tag}.json"
    out_png = OUT_PNG if safe_tag == "latest" else OUTPUT_DIR / f"particle_tracking_cfd_validation_{safe_tag}.png"
    print("=" * 70)
    print("AEGIS Step-30 CFD-driven particle-wall Mansouri erosion")
    print("=" * 70)
    print(f"  CFD={cfg.cfd_variant}  parcels={cfg.n_parcels}  max_impacts={cfg.max_impacts}  mass_flow={cfg.mass_flow_mode}")

    native_variants = {"native", "native_baseline"}
    if cfg.collision_backend == "cfd_topology" and cfg.cfd_variant not in native_variants:
        raise ValueError("cfd_topology backend requires a native same-mesh CFD variant")
    if cfg.collision_backend == "auto" and (WORKSPACE / cfg.topology_path).exists() and cfg.cfd_variant not in native_variants:
        raise ValueError("native topology artifact exists; use a native CFD variant or explicitly select --collision-backend stl")
    cfd = load_cfd_velocity_field(OUTPUT_DIR, cfg.cfd_variant)
    if cfd["speed_stats_ms"]["max"] > cfg.max_accepted_cfd_speed_ms:
        raise ValueError(
            f"{cfg.cfd_variant} CFD field max speed {cfd['speed_stats_ms']['max']:.1f} m/s "
            f"exceeds quality limit {cfg.max_accepted_cfd_speed_ms:.1f} m/s; "
            "inspect MRF export before particle tracking."
        )
    stl = load_collision_geometry(cfg)
    if stl["backend"] == "cfd_topology":
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
    print(f"[PT] CFD cells={cfd['n_cells']:,}, backend={stl['backend']}, parent faces={stl['n_parent_faces']:,}, area={stl['surface_area_m2']:.3f} m2")

    rng = np.random.default_rng(cfg.seed)
    tracked = _run_tracking(cfg, OUTPUT_DIR, stl, cfd, rng)
    arrays = _rows_to_arrays(tracked["impact_rows"])
    n_impacts = len(arrays["face_id"])
    print(f"[PT] impacts={n_impacts:,} (intersection={tracked['intersection_count']:,}, fallback={tracked['fallback_count']:,})")

    aggregate = aggregate_face_impacts(
        stl["n_parent_faces"], stl["face_area_m2"], stl["face_centers_mm"], stl["face_normals"],
        arrays["face_id"], arrays["ER_mansouri"], arrays["parcel_mass_kg"], cfg.observation_window_s,
    )

    rotation = dict(np.load(str(OUTPUT_DIR / "jet_rotation.npz")))
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
        face_id_namespace=np.array(stl["face_id_namespace"]),
        mesh_signature=np.array(stl["mesh_signature"]),
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
        "speed_stats_ms": cfd["speed_stats_ms"],
        "quality_status": "warning_velocity_outliers" if cfd["speed_stats_ms"]["max"] > speed_warning_limit else "passed",
        "velocity_warning_limit_ms": speed_warning_limit,
    }
    summary = {
        "version": "step30-cfd-driven-v1",
        "model_scope": "steady single-phase OpenFOAM continuous velocity field + one-way particle post-processing; no kinematicCloud wall log",
        "cfd": cfd_summary,
        "geometry": {
            "collision_backend": stl["backend"],
            "face_id_namespace": stl["face_id_namespace"],
            "mesh_signature": stl["mesh_signature"],
            "patch_name": stl["patch_name"],
            "source_time": stl["time_name"],
            "field_time": cfd["field_time"],
            "n_vertices": int(len(stl["vertices_mm"])),
            "n_collision_triangles": int(len(stl["collision_triangles_mm"])),
            "n_parent_faces": int(stl["n_parent_faces"]),
            "surface_area_m2": stl["surface_area_m2"],
            "bounds_mm": stl["bounds_mm"],
            "preflight": preflight,
        },
        "parameters": {
            "n_parcels": cfg.n_parcels,
            "n_injected": cfg.n_injected,
            "collision_backend": stl["backend"],
            "cfd_variant": cfg.cfd_variant,
            "max_impacts": cfg.max_impacts,
            "seed": cfg.seed,
            "dt_s": cfg.dt_s,
            "max_time_s": cfg.max_time_s,
            "output_tag": safe_tag,
            "include_gravity": cfg.include_gravity,
            "e_normal": cfg.e_normal,
            "e_tangent": cfg.e_tangent,
            "diameter_distribution": {"type": "Rosin-Rammler", "min_m": cfg.d_min_m, "max_m": cfg.d_max_m, "d50_m": cfg.d50_m, "exponent": cfg.rr_exponent},
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
        "units": {"position": "mm local / global", "velocity": "m/s", "ER": "kg target per kg sediment", "face_area": "m2", "dhdt": "mm/h"},
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[PT] -> {out_npz.name}, {out_json.name}, {out_png.name}")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CFD-driven particle-wall Mansouri erosion")
    parser.add_argument("--cfd", choices=["baseline", "mrf", "native", "native_baseline"], default="native_baseline")
    parser.add_argument("--collision-backend", choices=["auto", "cfd_topology", "stl"], default="auto")
    parser.add_argument("--topology", default="output/bucket_topology.npz")
    parser.add_argument("--tag", default="latest", help="output tag; latest uses canonical filenames")
    parser.add_argument("--parcels", type=int, default=2000)
    parser.add_argument("--max-impacts", type=int, default=3)
    parser.add_argument("--mass-flow", choices=["nominal", "geometric"], default="nominal")
    parser.add_argument("--max-time", type=float, default=0.05)
    parser.add_argument("--dt", type=float, default=1e-5)
    parser.add_argument("--max-cfd-speed", type=float, default=500.0)
    parser.add_argument("--gravity", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(TrackingConfig(
        cfd_variant=args.cfd,
        collision_backend=args.collision_backend,
        topology_path=args.topology,
        output_tag=args.tag,
        n_parcels=args.parcels,
        n_injected=args.parcels,
        max_impacts=args.max_impacts,
        mass_flow_mode=args.mass_flow,
        max_time_s=args.max_time,
        dt_s=args.dt,
        max_accepted_cfd_speed_ms=args.max_cfd_speed,
        include_gravity=args.gravity,
    ))
