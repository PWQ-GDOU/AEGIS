"""
step30_duty_cycle_erosion.py
==============================
Computes duty-cycle-weighted erosion rate for the Pelton bucket by
combining the CFD-based erosion rate field (dhdt_face_mm_h at rated
load) with the operating load distribution inferred from the DEC
PELTON 2024 stress time-series data.

Physics basis
-------------
At constant head H, the jet velocity u₁ and runner speed u are fixed.
The only variable with load change is the jet flow rate Q (∝ nozzle
opening S), which scales as:

    Q(P) ≈ P / (ρ g H η)     →    Q ∝ P  (at fixed H, η)

The sediment mass flux through the nozzles scales with Q:

    ṁ_sed = c_sed × Q ∝ P

The Mansouri erosion rate ER [kg_wall / kg_sed] depends on Vp and α,
which are set by the jet speed (constant) and bucket geometry (fixed).
So the erosion RATE per unit time scales with sediment mass flux:

    dhdt(P) ≈ dhdt_rated × (P / P_rated) × (n_nozzles_rated / n_nozzles(P))

where n_nozzles(P) accounts for the nozzle-switching schedule
(2→4→6 nozzles as load increases).

Duty cycle estimation
---------------------
Without a real operating log, we use a uniform distribution over
30–140 MW (the measurement range) plus 25% of time at rated 150 MW.
This is a conservative engineering assumption that can be replaced
with actual operating hours when available.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np

_RATED_POWER_MW   = 150.0
_RATED_NOZZLES    = 6
_NOZZLE_SCHEDULE  = {  # MW → n_nozzles
    30: 2, 40: 2,
    50: 4, 60: 4, 70: 4, 80: 4, 90: 4,
    100: 6, 110: 6, 120: 6, 130: 6, 140: 6, 150: 6,
}


def _nozzle_count(power_mw: float) -> int:
    """Return nozzle count for the given power level."""
    keys = sorted(_NOZZLE_SCHEDULE.keys())
    # nearest lower key
    for k in reversed(keys):
        if power_mw >= k:
            return _NOZZLE_SCHEDULE[k]
    return _NOZZLE_SCHEDULE[keys[0]]


def build_duty_cycle(
    mode: str = "uniform_with_rated",
    rated_fraction: float = 0.25,
    power_steps: list[float] | None = None,
) -> dict[str, Any]:
    """Build a duty-cycle distribution over operating power levels.

    Parameters
    ----------
    mode:
        ``uniform_with_rated`` — equal weight on each measured step
        (30–140 MW) plus extra fraction at rated 150 MW.
        ``uniform`` — equal weight on all steps including 150 MW.
    rated_fraction:
        Fraction of time spent at rated power (only used when
        ``mode='uniform_with_rated'``).
    power_steps:
        Custom list of power levels in MW.  Defaults to
        [30, 40, 50, …, 140, 150].

    Returns
    -------
    dict with ``power_mw``, ``fraction``, ``nozzle_count`` arrays.
    """
    if power_steps is None:
        power_steps = list(range(30, 150, 10)) + [150]

    if not power_steps:
        raise ValueError("power_steps must not be empty")
    if not (0.0 < rated_fraction < 1.0):
        raise ValueError(f"rated_fraction must be in (0, 1), got {rated_fraction}")

    powers   = np.array(power_steps, dtype=np.float64)
    nozzles  = np.array([_nozzle_count(p) for p in powers], dtype=np.int32)
    fractions = np.ones(len(powers), dtype=np.float64)

    if mode == "uniform_with_rated":
        # rated gets `rated_fraction`; rest share (1 - rated_fraction) equally
        non_rated = powers != _RATED_POWER_MW
        n_non_rated = int(non_rated.sum())
        if not non_rated.any():
            # all steps are rated — uniform share
            fractions[:] = 1.0
        else:
            fractions[~non_rated] = rated_fraction
            fractions[non_rated]  = (1.0 - rated_fraction) / max(n_non_rated, 1)
    else:
        pass   # fractions all-ones; will normalise below

    total = fractions.sum()
    if total <= 0.0:
        raise ValueError(f"duty-cycle fractions sum to {total}; cannot normalise")
    fractions /= total   # normalise to exactly 1.0

    return {
        "power_mw":     powers,
        "fraction":     fractions,
        "nozzle_count": nozzles,
        "mode":         mode,
    }


def compute_weighted_erosion(
    dhdt_rated_mm_h: np.ndarray,
    duty_cycle: dict[str, Any],
    rated_power_mw: float = _RATED_POWER_MW,
    rated_nozzles:  int   = _RATED_NOZZLES,
) -> dict[str, Any]:
    """Scale dhdt_rated by load distribution and accumulate weighted mean.

    Erosion rate scaling:
        dhdt(P) = dhdt_rated × (P/P_rated) × (n_rated/n_nozzles(P))

    This accounts for the fact that at lower load, each nozzle carries
    less flow (less sediment), AND fewer nozzles are active.

    Returns a dict with the weighted erosion field and per-load breakdown.
    """
    powers   = duty_cycle["power_mw"]
    frac     = duty_cycle["fraction"]
    nozzles  = duty_cycle["nozzle_count"]

    if len(dhdt_rated_mm_h) == 0:
        raise ValueError("dhdt_rated_mm_h must not be empty")
    if len(powers) == 0:
        raise ValueError("duty_cycle must contain at least one power step")
    if abs(float(frac.sum()) - 1.0) > 1e-6:
        raise ValueError(f"duty-cycle fractions sum to {frac.sum():.6f}, expected 1.0")

    n_faces = len(dhdt_rated_mm_h)
    dhdt_weighted = np.zeros(n_faces, dtype=np.float64)
    per_load: list[dict[str, Any]] = []

    for p, f, n_noz in zip(powers, frac, nozzles):
        # Scale factor relative to rated condition
        scale = (p / rated_power_mw) * (rated_nozzles / n_noz)
        dhdt_p = dhdt_rated_mm_h * scale
        dhdt_weighted += f * dhdt_p

        per_load.append({
            "power_mw":      float(p),
            "fraction":      float(f),
            "nozzle_count":  int(n_noz),
            "scale_factor":  float(scale),
            "max_dhdt_mm_h": float(dhdt_p.max()),
        })

    # Summary statistics
    rated_max = float(dhdt_rated_mm_h.max())
    weighted_max = float(dhdt_weighted.max())
    # Hot-spot consistency: top-10% overlap between rated and weighted maps.
    # When rated erosion is all-zero the comparison is meaningless → IoU = 0.0.
    active_faces = int((dhdt_rated_mm_h > 0).sum())
    if active_faces == 0 or rated_max == 0.0:
        top_iou = 0.0
    else:
        n_top        = max(1, int(0.10 * active_faces))
        top_rated    = set(np.argsort(dhdt_rated_mm_h)[-n_top:])
        top_weighted = set(np.argsort(dhdt_weighted)[-n_top:])
        union_size   = len(top_rated | top_weighted)
        top_iou      = len(top_rated & top_weighted) / union_size if union_size > 0 else 0.0

    return {
        "dhdt_weighted_mm_h":      dhdt_weighted.astype(np.float32),
        "dhdt_rated_mm_h":         dhdt_rated_mm_h.astype(np.float32),
        "max_rated_mm_h":          rated_max,
        "max_weighted_mm_h":       weighted_max,
        "weighted_to_rated_ratio": weighted_max / max(rated_max, 1e-12),
        "hotspot_iou_rated_vs_weighted": top_iou,
        "per_load":                per_load,
        "duty_cycle_mode":         duty_cycle["mode"],
    }


def run_weighted_erosion(
    hits_path: str | pathlib.Path,
    output_path: str | pathlib.Path | None = None,
    duty_mode: str = "uniform_with_rated",
    rated_fraction: float = 0.25,
) -> dict[str, Any]:
    """End-to-end pipeline: load hits → duty cycle → weighted erosion."""
    p = pathlib.Path(hits_path)
    with np.load(str(p), allow_pickle=False) as d:
        # Use RBF field if available, else fall back to raw
        if "dhdt_rbf_mm_h" in d.files:
            dhdt = np.asarray(d["dhdt_rbf_mm_h"], dtype=np.float64)
            source = "rbf"
        else:
            dhdt = np.asarray(d["dhdt_face_mm_h"], dtype=np.float64)
            source = "raw"

    duty = build_duty_cycle(mode=duty_mode, rated_fraction=rated_fraction)
    result = compute_weighted_erosion(dhdt, duty)
    result["source"] = source
    result["hits_path"] = str(p)

    if output_path is not None:
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Save arrays separately as NPZ
        np.savez_compressed(
            str(out),
            dhdt_weighted_mm_h=result["dhdt_weighted_mm_h"],
            dhdt_rated_mm_h=result["dhdt_rated_mm_h"],
        )
        # Save summary as JSON
        summary = {k: v for k, v in result.items()
                   if not isinstance(v, np.ndarray)}
        out.with_suffix(".json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return result


def _selftest() -> None:
    rng = np.random.default_rng(1)
    dhdt = rng.exponential(2.0, 1000).astype(np.float32)
    duty = build_duty_cycle()
    assert abs(duty["fraction"].sum() - 1.0) < 1e-9
    result = compute_weighted_erosion(dhdt, duty)
    assert result["dhdt_weighted_mm_h"].shape == (1000,)
    assert result["weighted_to_rated_ratio"] < 1.0, \
        "weighted mean must be < rated (load < rated most of the time)"
    assert result["hotspot_iou_rated_vs_weighted"] > 0.5, \
        "hotspot spatial pattern should be stable across load levels"


def selftest() -> bool:
    _selftest()
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Duty-cycle weighted erosion for Pelton bucket"
    )
    parser.add_argument("--hits", type=pathlib.Path,
                        help="particle_hits_cfd.npz or erosion_map_rbf.npz")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--duty-mode", default="uniform_with_rated")
    parser.add_argument("--rated-fraction", type=float, default=0.25)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)

    if args.selftest:
        selftest()
        print("step30_duty_cycle_erosion selftest: PASS")
        return 0

    if args.hits is None:
        parser.error("--hits is required")

    result = run_weighted_erosion(
        args.hits, output_path=args.output,
        duty_mode=args.duty_mode, rated_fraction=args.rated_fraction,
    )
    print(f"Source field    : {result['source']}")
    print(f"Max rated       : {result['max_rated_mm_h']:.4f} mm/h")
    print(f"Max weighted    : {result['max_weighted_mm_h']:.4f} mm/h")
    print(f"Weighted/Rated  : {result['weighted_to_rated_ratio']:.3f}")
    print(f"Hotspot IoU     : {result['hotspot_iou_rated_vs_weighted']:.3f}")
    print(f"Duty-cycle mode : {result['duty_cycle_mode']}")
    if args.output:
        print(f"Saved           : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
