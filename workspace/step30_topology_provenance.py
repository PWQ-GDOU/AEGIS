"""Create a compact provenance report for the native OpenFOAM bucket topology."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from step30_cfd_topology import TOPOLOGY_NPZ, TOPOLOGY_JSON, load_topology_artifact

WORKSPACE = Path(__file__).parent
OUTPUT_DIR = WORKSPACE / "output"
OUT_JSON = OUTPUT_DIR / "native_bucket_provenance.json"


def _boundary_metadata(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"\bbucket\s*\{(?P<body>.*?)\n\s*\}", text, re.DOTALL
    )
    if not match:
        raise ValueError(f"bucket patch is missing from {path}")
    body = match.group("body")
    nf = re.search(r"\bnFaces\s+(\d+)\s*;", body)
    sf = re.search(r"\bstartFace\s+(\d+)\s*;", body)
    if not nf or not sf:
        raise ValueError("bucket patch has no nFaces/startFace")
    return {
        "patch_name": "bucket",
        "n_faces": int(nf.group(1)),
        "start_face": int(sf.group(1)),
        "boundary_file": str(path),
    }


def run() -> dict:
    topo = load_topology_artifact(TOPOLOGY_NPZ)
    boundary_path = OUTPUT_DIR / "native_bucket" / "polyMesh_boundary.txt"
    boundary = _boundary_metadata(boundary_path)
    if boundary["n_faces"] != topo["n_parent_faces"]:
        raise ValueError("native topology and OpenFOAM boundary face count differ")

    result = {
        "version": "native-bucket-provenance-v1",
        "openfoam_version": "2412",
        "source_case": "/public/home/wenquan/pelton_bucket_jet",
        "source_time": topo["time_name"],
        "patch": boundary,
        "topology": {
            "artifact": str(TOPOLOGY_NPZ),
            "summary": str(TOPOLOGY_JSON),
            "mesh_signature": topo["mesh_signature"],
            "n_points": int(len(topo["vertices_mm"])),
            "n_parent_faces": int(topo["n_parent_faces"]),
            "n_collision_triangles": int(len(topo["collision_triangles_mm"])),
            "surface_area_m2": float(topo["surface_area_m2"]),
            "bounds_mm": topo["bounds_mm"],
        },
        "same_mesh_velocity_fields": {
            "variant": "native",
            "directory": str(OUTPUT_DIR / "native_bucket"),
            "n_cells": 424161,
            "fields": ["native_1000_Cx.foam", "native_1000_Cy.foam", "native_1000_Cz.foam", "native_1000_U.foam"],
        },
        "gpr_registration": {},
        "raw_coordinate_match": {},
        "status": "passed",
    }

    rotation = dict(np.load(str(OUTPUT_DIR / "jet_rotation.npz")))
    gpr = np.load(str(OUTPUT_DIR / "vertices_full_gpr.npz"))
    centers_global = topo["face_centers_mm"] @ rotation["R"] + rotation["nozzle_pos"]
    distances, _ = cKDTree(gpr["xyz"].astype(np.float64)).query(centers_global)
    result["gpr_registration"] = {
        "status": "passed" if np.median(distances) <= 10.0 and np.percentile(distances, 95) <= 15.0 else "failed",
        "median_distance_mm": float(np.median(distances)),
        "p95_distance_mm": float(np.percentile(distances, 95)),
        "max_distance_mm": float(distances.max()),
    }

    raw_path = OUTPUT_DIR / "mrf2_p_bucket.raw"
    if raw_path.exists():
        raw = np.loadtxt(str(raw_path), comments="#")[:, :3]
        raw_distance, _ = cKDTree(raw).query(topo["face_centers_mm"])
        result["raw_coordinate_match"] = {
            "file": str(raw_path),
            "rows": int(len(raw)),
            "same_row_count": bool(len(raw) == topo["n_parent_faces"]),
            "median_distance_mm": float(np.median(raw_distance)),
            "p95_distance_mm": float(np.percentile(raw_distance, 95)),
            "max_distance_mm": float(raw_distance.max()),
            "status": "passed" if len(raw) == topo["n_parent_faces"] and np.percentile(raw_distance, 95) < 0.1 else "failed",
        }
    else:
        result["raw_coordinate_match"] = {"status": "not_available"}

    if result["gpr_registration"]["status"] != "passed":
        result["status"] = "failed_gpr_registration"
    if result["raw_coordinate_match"].get("status") == "failed":
        result["status"] = "failed_raw_coordinate_match"
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
