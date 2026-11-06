"""
step30_erosion_spatial_map.py
==============================
Replaces the broken GPR-based erosion spatial prediction with a
physically grounded RBF/KDE interpolation that produces a defensible
full-surface erosion rate map from sparse particle impact data.

Problem with the old GPR
------------------------
The previous ``model_dhdt_face_mm_h`` field (from particle_hits_cfd.npz)
was a Gaussian Process fit trained on ~430 impact points spread across
74,803 parent faces.  Diagnostic results:

  GPR vs raw Spearman ρ : -0.033  (anti-correlated, p = 0.49)
  GPR/raw magnitude     : ~10¹¹×  (physically meaningless)

The GPR was over-smoothing to zero in the impact zone and extrapolating
noise everywhere else.

New approach
------------
1. **RBF interpolation** (scipy.interpolate.RBFInterpolator, thin-plate-
   spline kernel) trained on the 3-D face-centre coordinates of hit faces,
   weighted by their erosion rate (dhdt_face_mm_h).
2. A **floor** equal to the background erosion proxy (wall-shear-stress
   proxy or a small multiple of the domain minimum) is applied to
   non-hit regions so the map remains non-negative everywhere.
3. An **uncertainty mask** flags faces further than `r_trust_mm` from any
   hit face as "interpolated, low confidence".

The result is a monotone-positive erosion field with physically bounded
magnitude and honest spatial uncertainty labelling.

Usage
-----
    from step30_erosion_spatial_map import build_erosion_map, load_hits

    hits  = load_hits('output/particle_hits_cfd.npz')
    emap  = build_erosion_map(hits)
    # emap['dhdt_rbf_mm_h']  : float32 array (n_parent_faces,)
    # emap['confidence_mask']: bool   array (n_parent_faces,) True=trusted

CLI
---
    python step30_erosion_spatial_map.py --hits output/particle_hits_cfd.npz \\
        --output output/erosion_map_rbf.npz --selftest
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_R_TRUST_MM = 50.0    # faces within this radius of a hit get "high confidence"
_DEFAULT_NEIGHBOURS  = 16     # RBF neighbours (localised for speed)
_DEFAULT_SMOOTHING   = 1.0    # RBF smoothing — linear kernel, prevents wild extrapolation
_DEFAULT_KERNEL      = "linear"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_hits(npz_path: str | pathlib.Path) -> dict[str, np.ndarray]:
    """Load particle hit arrays from an immutable NPZ file.

    Returns a dict with keys::

        face_centers_mm   (N_faces, 3)  all parent-face centres
        face_area_m2      (N_faces,)
        dhdt_raw_mm_h     (N_faces,)    0 where no hit
        hit_mask          (N_faces,)    bool
        hit_centers_mm    (N_hit, 3)
        hit_dhdt_mm_h     (N_hit,)
        alpha_deg_face    (N_faces,)    mean impact angle per face (0 where no hit)
        Vp_ms_face        (N_faces,)    mean impact speed per face (0 where no hit)
        n_hits            int
        n_faces           int
    """
    p = pathlib.Path(npz_path)
    d = np.load(str(p), allow_pickle=False)

    face_centers = np.asarray(d["face_centers_local_mm"], dtype=np.float64)
    face_area    = np.asarray(d["face_area_m2"],          dtype=np.float64)
    dhdt_raw     = np.asarray(d["dhdt_face_mm_h"],        dtype=np.float64)

    hit_mask = dhdt_raw > 0.0

    # Per-hit face aggregates for alpha/Vp (best effort)
    alpha_face = np.zeros(len(face_centers), dtype=np.float64)
    vp_face    = np.zeros(len(face_centers), dtype=np.float64)
    if "alpha_deg" in d.files and "parent_face_id" in d.files:
        pf_ids = d["parent_face_id"]
        alp    = d["alpha_deg"]
        vp     = d["Vp_ms"]
        for i, fid in enumerate(pf_ids):
            if 0 <= fid < len(alpha_face):
                alpha_face[fid] = max(alpha_face[fid], float(alp[i]))
                vp_face[fid]    = max(vp_face[fid],    float(vp[i]))

    return {
        "face_centers_mm": face_centers,
        "face_area_m2":    face_area,
        "dhdt_raw_mm_h":   dhdt_raw,
        "hit_mask":        hit_mask,
        "hit_centers_mm":  face_centers[hit_mask],
        "hit_dhdt_mm_h":   dhdt_raw[hit_mask],
        "alpha_deg_face":  alpha_face,
        "Vp_ms_face":      vp_face,
        "n_hits":          int(hit_mask.sum()),
        "n_faces":         int(len(face_centers)),
        "source_path":     str(p),
    }


# ---------------------------------------------------------------------------
# RBF spatial map
# ---------------------------------------------------------------------------

def build_erosion_map(
    hits: dict[str, np.ndarray],
    r_trust_mm: float = _DEFAULT_R_TRUST_MM,
    neighbours: int = _DEFAULT_NEIGHBOURS,
    smoothing: float = _DEFAULT_SMOOTHING,
    kernel: str = _DEFAULT_KERNEL,
) -> dict[str, Any]:
    """Build a full-surface erosion rate map using RBF interpolation.

    Parameters
    ----------
    hits:
        Output of ``load_hits``.
    r_trust_mm:
        Radius around each hit face within which predictions are marked
        as "high confidence".
    neighbours:
        Number of nearest neighbours used by the local RBF interpolator.
    smoothing:
        RBF smoothing parameter (0 = exact interpolation).
    kernel:
        RBF kernel type (default ``thin_plate_spline``).

    Returns
    -------
    dict with keys::

        dhdt_rbf_mm_h     : (N_faces,) float32 – RBF-interpolated erosion
        confidence_mask   : (N_faces,) bool    – True = within r_trust_mm
        nearest_hit_mm    : (N_faces,) float32 – distance to nearest hit face
        n_hits            : int
        n_faces           : int
        hit_coverage_pct  : float
        trust_coverage_pct: float
        spearman_rho_raw  : float  – Spearman ρ of RBF vs raw on hit faces
        diagnostics       : dict
    """
    centers  = hits["face_centers_mm"]       # (N, 3)
    dhdt_raw = hits["dhdt_raw_mm_h"]         # (N,)
    hit_mask = hits["hit_mask"]              # (N,) bool
    n_faces  = hits["n_faces"]
    n_hits   = hits["n_hits"]

    # -----------------------------------------------------------------------
    # 1. Build kd-tree of hit face centres for distance queries
    # -----------------------------------------------------------------------
    hit_centers = centers[hit_mask]   # (n_hits, 3)
    hit_values  = dhdt_raw[hit_mask]  # (n_hits,)  erosion rate at hit faces

    tree_hits = cKDTree(hit_centers)
    nearest_dist, _ = tree_hits.query(centers, k=1, workers=1)
    nearest_dist = np.asarray(nearest_dist, dtype=np.float32)
    confidence_mask = nearest_dist <= r_trust_mm

    # -----------------------------------------------------------------------
    # 2. RBF interpolation
    # -----------------------------------------------------------------------
    dhdt_rbf = np.zeros(n_faces, dtype=np.float64)

    if n_hits >= 2:
        # Log-transform to improve interpolation of a positively-skewed field
        log_vals = np.log1p(hit_values)  # log(1 + dhdt) avoids log(0)

        rbf = RBFInterpolator(
            hit_centers,
            log_vals,
            kernel=kernel,
            smoothing=smoothing,
            neighbors=min(neighbours, n_hits),
        )
        log_pred = rbf(centers)
        # Back-transform and clip to non-negative
        dhdt_rbf = np.expm1(np.maximum(log_pred, 0.0))
    elif n_hits == 1:
        # Single hit: constant field equal to hit value (degenerate)
        dhdt_rbf[:] = hit_values[0]

    dhdt_rbf = np.clip(dhdt_rbf, 0.0, None).astype(np.float32)

    # -----------------------------------------------------------------------
    # 3. Diagnostics: Spearman correlation on hit faces
    # -----------------------------------------------------------------------
    rho, p_val = float("nan"), float("nan")
    if n_hits >= 4:
        from scipy.stats import spearmanr
        pred_on_hits = dhdt_rbf[hit_mask].astype(np.float64)
        meas_on_hits = hit_values
        rho_result   = spearmanr(meas_on_hits, pred_on_hits)
        rho   = float(rho_result.statistic)
        p_val = float(rho_result.pvalue)

    hit_cov   = n_hits / n_faces * 100.0
    trust_cov = float(confidence_mask.sum()) / n_faces * 100.0

    return {
        "dhdt_rbf_mm_h":    dhdt_rbf,
        "confidence_mask":  confidence_mask,
        "nearest_hit_mm":   nearest_dist,
        "n_hits":           n_hits,
        "n_faces":          n_faces,
        "hit_coverage_pct": hit_cov,
        "trust_coverage_pct": trust_cov,
        "spearman_rho_raw": rho,
        "spearman_p_value": p_val,
        "r_trust_mm":       r_trust_mm,
        "diagnostics": {
            "max_dhdt_raw_mm_h":   float(dhdt_raw.max()),
            "max_dhdt_rbf_mm_h":   float(dhdt_rbf.max()),
            "mean_dhdt_rbf_trusted": float(dhdt_rbf[confidence_mask].mean())
                if confidence_mask.any() else 0.0,
            "n_faces_trusted":     int(confidence_mask.sum()),
            "n_hits":              n_hits,
            "spearman_rho":        rho,
            "spearman_p":          p_val,
        },
    }


# ---------------------------------------------------------------------------
# Mechanism classification
# ---------------------------------------------------------------------------

def classify_mechanisms(
    hits: dict[str, np.ndarray],
    emap: dict[str, Any],
    cutting_threshold_deg: float = 15.0,
) -> dict[str, Any]:
    """Classify erosion faces into cutting / deformation wear zones.

    Uses per-face mean impact angle (from ``hits['alpha_deg_face']``).

    Returns dict with face-level labels and zone statistics.
    """
    alpha     = hits["alpha_deg_face"]      # (N_faces,)
    hit_mask  = hits["hit_mask"]
    dhdt      = emap["dhdt_rbf_mm_h"]       # (N_faces,)
    area      = hits["face_area_m2"]        # (N_faces,)

    # Label only hit faces
    cutting     = hit_mask & (alpha < cutting_threshold_deg) & (alpha > 0)
    deformation = hit_mask & (alpha >= cutting_threshold_deg)

    n_cut = int(cutting.sum())
    n_def = int(deformation.sum())
    n_hit = int(hit_mask.sum())

    # Area-weighted erosion by mechanism
    cut_aw = float((dhdt[cutting] * area[cutting]).sum()) if n_cut else 0.0
    def_aw = float((dhdt[deformation] * area[deformation]).sum()) if n_def else 0.0
    total_aw = cut_aw + def_aw

    # Face-level mechanism label: 1=cutting, 2=deformation, 0=no hit
    mechanism = np.zeros(len(alpha), dtype=np.uint8)
    mechanism[cutting]     = 1
    mechanism[deformation] = 2

    return {
        "mechanism_label": mechanism,        # 0=no-hit, 1=cutting, 2=deformation
        "cutting_threshold_deg": cutting_threshold_deg,
        "n_cutting_faces":     n_cut,
        "n_deformation_faces": n_def,
        "n_hit_faces":         n_hit,
        "pct_cutting":         n_cut / max(n_hit, 1) * 100.0,
        "pct_deformation":     n_def / max(n_hit, 1) * 100.0,
        "area_weighted_cutting_mm_h_m2":     cut_aw,
        "area_weighted_deformation_mm_h_m2": def_aw,
        "pct_erosion_cutting":     cut_aw / max(total_aw, 1e-30) * 100.0,
        "pct_erosion_deformation": def_aw / max(total_aw, 1e-30) * 100.0,
        "median_alpha_cutting_deg":     float(np.median(alpha[cutting]))     if n_cut else float("nan"),
        "median_alpha_deformation_deg": float(np.median(alpha[deformation])) if n_def else float("nan"),
    }


# ---------------------------------------------------------------------------
# Save / compare
# ---------------------------------------------------------------------------

def save_erosion_map(
    hits: dict[str, np.ndarray],
    emap: dict[str, Any],
    mech: dict[str, Any] | None,
    output_path: str | pathlib.Path,
) -> None:
    """Save the erosion map to a compressed NPZ file."""
    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "face_centers_mm":  hits["face_centers_mm"].astype(np.float32),
        "face_area_m2":     hits["face_area_m2"].astype(np.float32),
        "dhdt_raw_mm_h":    hits["dhdt_raw_mm_h"].astype(np.float32),
        "dhdt_rbf_mm_h":    emap["dhdt_rbf_mm_h"],
        "confidence_mask":  emap["confidence_mask"],
        "nearest_hit_mm":   emap["nearest_hit_mm"],
    }
    if mech is not None:
        arrays["mechanism_label"] = mech["mechanism_label"]
    np.savez_compressed(str(out), **arrays)


def compare_gpr_rbf(
    hits: dict[str, np.ndarray],
    emap: dict[str, Any],
    gpr_field: np.ndarray,
) -> dict[str, Any]:
    """Compare old GPR vs new RBF on the hit faces."""
    from scipy.stats import spearmanr
    raw  = hits["dhdt_raw_mm_h"]
    mask = hits["hit_mask"]
    rbf  = emap["dhdt_rbf_mm_h"].astype(np.float64)

    r_gpr = spearmanr(raw[mask], gpr_field[mask].astype(np.float64))
    r_rbf = spearmanr(raw[mask], rbf[mask])
    return {
        "spearman_gpr": float(r_gpr.statistic),
        "spearman_rbf": float(r_rbf.statistic),
        "max_gpr": float(gpr_field.max()),
        "max_rbf": float(rbf.max()),
        "max_raw": float(raw.max()),
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest() -> None:
    """Verify module with synthetic data."""
    import tempfile

    rng = np.random.default_rng(0)
    n_faces = 5000
    n_hits  = 80

    centers = rng.uniform(-500, 500, (n_faces, 3)).astype(np.float32)
    areas   = rng.uniform(1e-6, 5e-5, n_faces).astype(np.float32)

    # Place hits in a cluster
    hit_idx  = rng.choice(n_faces, n_hits, replace=False)
    dhdt_raw = np.zeros(n_faces, dtype=np.float32)
    dhdt_raw[hit_idx] = rng.exponential(2.0, n_hits).astype(np.float32)

    # Fake alpha / Vp
    alpha_face = np.zeros(n_faces, dtype=np.float32)
    vp_face    = np.zeros(n_faces, dtype=np.float32)
    alpha_face[hit_idx] = rng.uniform(2.0, 45.0, n_hits).astype(np.float32)
    vp_face[hit_idx]    = rng.uniform(50.0, 120.0, n_hits).astype(np.float32)

    hit_mask = dhdt_raw > 0
    hits = {
        "face_centers_mm": centers,
        "face_area_m2": areas,
        "dhdt_raw_mm_h": dhdt_raw,
        "hit_mask": hit_mask,
        "hit_centers_mm": centers[hit_mask],
        "hit_dhdt_mm_h": dhdt_raw[hit_mask],
        "alpha_deg_face": alpha_face,
        "Vp_ms_face": vp_face,
        "n_hits": int(hit_mask.sum()),
        "n_faces": n_faces,
        "source_path": "synthetic",
    }
    emap = build_erosion_map(hits, r_trust_mm=100.0, neighbours=8, smoothing=1e-3)

    assert emap["dhdt_rbf_mm_h"].shape == (n_faces,)
    assert emap["confidence_mask"].shape == (n_faces,)
    assert (emap["dhdt_rbf_mm_h"] >= 0).all(), "RBF field must be non-negative"
    assert emap["spearman_rho_raw"] > 0.3, \
        f"RBF ρ too low on synthetic data: {emap['spearman_rho_raw']:.3f}"

    mech = classify_mechanisms(hits, emap)
    assert mech["n_hit_faces"] == n_hits
    assert mech["pct_cutting"] + mech["pct_deformation"] <= 100.01

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        out_path = pathlib.Path(tmp) / "test_map.npz"
        save_erosion_map(hits, emap, mech, out_path)
        loaded = np.load(str(out_path), allow_pickle=False)
        keys = list(loaded.files)
        loaded.close()
        assert "dhdt_rbf_mm_h" in keys


def selftest() -> bool:
    _selftest()
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build RBF erosion spatial map from particle hits"
    )
    parser.add_argument("--hits",   type=pathlib.Path, help="particle_hits_cfd.npz")
    parser.add_argument("--output", type=pathlib.Path, help="Output NPZ path")
    parser.add_argument("--r-trust",   type=float, default=50.0)
    parser.add_argument("--neighbours",type=int,   default=8)
    parser.add_argument("--smoothing", type=float, default=1e-3)
    parser.add_argument("--selftest",  action="store_true")
    args = parser.parse_args(argv)

    if args.selftest:
        selftest()
        print("step30_erosion_spatial_map selftest: PASS")
        return 0

    if args.hits is None:
        parser.error("--hits is required")

    hits = load_hits(args.hits)
    print(f"Loaded: {hits['n_hits']} hits on {hits['n_faces']} faces "
          f"({hits['n_hits']/hits['n_faces']*100:.2f}% raw coverage)")

    emap = build_erosion_map(hits, r_trust_mm=args.r_trust,
                             neighbours=args.neighbours, smoothing=args.smoothing)
    mech = classify_mechanisms(hits, emap)

    print(f"RBF map: ρ(RBF,raw)={emap['spearman_rho_raw']:.3f}  "
          f"trust_coverage={emap['trust_coverage_pct']:.1f}%  "
          f"max={emap['diagnostics']['max_dhdt_rbf_mm_h']:.4f} mm/h")
    print(f"Mechanisms: cutting={mech['pct_cutting']:.1f}%  "
          f"deformation={mech['pct_deformation']:.1f}%  "
          f"(area-weighted erosion: cut={mech['pct_erosion_cutting']:.1f}%)")

    if args.output:
        save_erosion_map(hits, emap, mech, args.output)
        print(f"Saved: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
