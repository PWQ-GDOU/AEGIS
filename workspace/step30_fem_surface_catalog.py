"""Catalog authoritative FEM STEP surfaces for manual selection review.

This tool does not create structural evidence by itself. It binds a STEP/BREP
solid to immutable surface statistics so ROOT_INTERFACE, WETTED_SURFACE, and
AI5_04 can be reviewed and confirmed explicitly instead of inferred from raw
Gmsh tag numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gmsh

from step30_validation_common import canonical_sha256, jsonable, safe_json_dump, sha256_file

VERSION = "step30-fem-surface-catalog-v1"


class SurfaceCatalogError(ValueError):
    """Raised when a STEP/BREP surface catalog cannot be created."""


def build_surface_catalog(step_path: str | Path) -> dict[str, Any]:
    step_path = Path(step_path).resolve()
    if not step_path.is_file():
        raise SurfaceCatalogError(f"STEP/BREP file does not exist: {step_path}")
    if step_path.suffix.lower() not in {".step", ".stp", ".brep"}:
        raise SurfaceCatalogError("surface catalog requires STEP/STP/BREP input")

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    try:
        gmsh.open(str(step_path))
        volumes = gmsh.model.getEntities(3)
        surfaces = gmsh.model.getEntities(2)
        if len(volumes) != 1:
            raise SurfaceCatalogError(
                f"expected exactly one solid volume, found {len(volumes)}"
            )
        if not surfaces:
            raise SurfaceCatalogError("STEP/BREP has no boundary surfaces")

        volume_id = volumes[0][1]
        volume_bbox = gmsh.model.getBoundingBox(3, volume_id)
        surface_rows = []
        for _, tag in surfaces:
            bbox = gmsh.model.getBoundingBox(2, tag)
            area = gmsh.model.occ.getMass(2, tag)
            center = gmsh.model.occ.getCenterOfMass(2, tag)
            surface_rows.append(
                {
                    "surface_tag": int(tag),
                    "area_m2": float(area),
                    "center_m": [float(value) for value in center],
                    "bbox_m": [float(value) for value in bbox],
                }
            )
        surface_rows.sort(key=lambda row: row["surface_tag"])
    finally:
        gmsh.finalize()

    catalog = {
        "version": VERSION,
        "status": "catalogued",
        "source_path": str(step_path),
        "source_sha256": sha256_file(step_path),
        "geometry": {
            "volume_count": 1,
            "surface_count": len(surface_rows),
            "volume_bbox_m": [float(value) for value in volume_bbox],
        },
        "surfaces": surface_rows,
        "note": "Surface tags are catalog identifiers only; production ROOT/WETTED/AI5 selections still require independent confirmation.",
    }
    catalog["catalog_sha256"] = canonical_sha256(catalog)
    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description="Catalog STEP/BREP surfaces for FEM selection review")
    parser.add_argument("step", type=Path, help="Authoritative STEP/STP/BREP solid")
    parser.add_argument("--output", type=Path, required=True, help="Immutable JSON output path")
    args = parser.parse_args()
    report = build_surface_catalog(args.step)
    safe_json_dump(report, args.output)
    print(json.dumps(jsonable(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
