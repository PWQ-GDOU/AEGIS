"""Diagnose dynamic particle erosion against correctly paired references.

This module separates formula self-consistency from dynamic/static spatial
comparisons.  It never treats a correlation value as an accuracy gate.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

from step30_lagrangian_particle_tracking import mansouri_er
from step30_validation_common import canonical_sha256, file_sha256

VERSION = "step30-correlation-diagnostics-v1"


def _finite(values: Any, name: str, *, length: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if length is not None and len(array) != length:
        raise ValueError(f"{name} length {len(array)} differs from {length}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _metric(a: np.ndarray, b: np.ndarray, mask: np.ndarray, *, bootstrap: int = 200, seed: int = 42) -> dict[str, Any]:
    if a.shape != b.shape or a.shape != mask.shape:
        raise ValueError("correlation arrays and mask must have equal shapes")
    av, bv = a[mask], b[mask]
    result: dict[str, Any] = {"n": int(len(av)), "mask_fraction": float(np.mean(mask)) if len(mask) else 0.0}
    if len(av) < 3 or np.allclose(av, av[0]) or np.allclose(bv, bv[0]):
        result.update({"spearman_rho": None, "spearman_p": None, "kendall_tau": None, "kendall_p": None, "pearson_r": None, "pearson_p": None, "nrmse_log1p": None, "top10_iou": None, "bootstrap_ci95": None})
        return result
    sr, sp = spearmanr(av, bv)
    kt, kp = kendalltau(av, bv)
    pr, pp = pearsonr(av, bv)
    scale = max(float(np.std(np.log1p(np.maximum(bv, 0.0)))), 1e-30)
    nrmse = float(np.sqrt(np.mean((np.log1p(np.maximum(av, 0.0)) - np.log1p(np.maximum(bv, 0.0))) ** 2)) / scale)
    at, bt = np.percentile(av, 90.0), np.percentile(bv, 90.0)
    union = (av >= at) | (bv >= bt)
    result.update({"spearman_rho": float(sr), "spearman_p": float(sp), "kendall_tau": float(kt), "kendall_p": float(kp), "pearson_r": float(pr), "pearson_p": float(pp), "nrmse_log1p": nrmse, "top10_iou": float(np.sum((av >= at) & (bv >= bt)) / max(int(np.sum(union)), 1))})
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(max(0, int(bootstrap))):
        indices = rng.integers(0, len(av), len(av))
        if np.allclose(av[indices], av[indices][0]) or np.allclose(bv[indices], bv[indices][0]):
            continue
        values.append(float(spearmanr(av[indices], bv[indices]).statistic))
    result["bootstrap_ci95"] = [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if values else None
    return result


def _aggregate_hits(data: dict[str, np.ndarray], n_faces: int) -> dict[str, np.ndarray]:
    parent = np.asarray(data.get("parent_face_id", data.get("face_id")), dtype=np.int64)
    if parent.ndim != 1 or len(parent) == 0 or parent.min() < 0 or parent.max() >= n_faces:
        raise ValueError("parent_face_id is outside the declared face topology")
    mass = _finite(data.get("parcel_mass_kg"), "parcel_mass_kg", length=len(parent))
    vp = _finite(data.get("Vp_ms"), "Vp_ms", length=len(parent))
    alpha = _finite(data.get("alpha_deg"), "alpha_deg", length=len(parent))
    stored = _finite(data.get("ER_mansouri"), "ER_mansouri", length=len(parent))
    mass_sum = np.bincount(parent, weights=mass, minlength=n_faces)
    er_sum = np.bincount(parent, weights=mass * stored, minlength=n_faces)
    formula = mansouri_er(vp, alpha)
    formula_sum = np.bincount(parent, weights=mass * formula, minlength=n_faces)
    vp_sum = np.bincount(parent, weights=mass * vp, minlength=n_faces)
    alpha_sum = np.bincount(parent, weights=mass * alpha, minlength=n_faces)
    return {
        "mass": mass_sum,
        "dynamic_er": np.divide(er_sum, mass_sum, out=np.zeros(n_faces), where=mass_sum > 0),
        "dynamic_formula_er": np.divide(formula_sum, mass_sum, out=np.zeros(n_faces), where=mass_sum > 0),
        "dynamic_vp": np.divide(vp_sum, mass_sum, out=np.zeros(n_faces), where=mass_sum > 0),
        "dynamic_alpha": np.divide(alpha_sum, mass_sum, out=np.zeros(n_faces), where=mass_sum > 0),
    }


def run(hits_path: Path, summary_path: Path, output: Path, *, bootstrap: int = 200, seed: int = 42) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("version") != "step30-cfd-driven-v2":
        raise ValueError("correlation diagnostics require a Step30 v2 summary")
    with np.load(hits_path, allow_pickle=False) as loaded:
        data = {key: np.asarray(loaded[key]) for key in loaded.files}
    required = {"Vp_ms", "alpha_deg", "ER_mansouri", "parcel_mass_kg", "parent_face_id"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"hits artifact is missing fields: {missing}")
    n_impacts = int(summary.get("tracking", {}).get("n_impacts", -1))
    if n_impacts != len(data["Vp_ms"]):
        raise ValueError("summary and hits impact counts differ")
    face_er = data.get("ER_mass_ratio_face")
    if face_er is not None:
        n_faces = len(face_er)
    else:
        n_faces = int(summary.get("geometry", {}).get("n_parent_faces", int(np.max(data["parent_face_id"])) + 1))
    aggregates = _aggregate_hits(data, n_faces)
    stored_event = _finite(data["ER_mansouri"], "ER_mansouri")
    recalculated_event = mansouri_er(data["Vp_ms"], data["alpha_deg"])
    event_mask = np.ones(len(stored_event), dtype=bool)
    event_formula = _metric(stored_event, recalculated_event, event_mask, bootstrap=bootstrap, seed=seed)
    rows = []
    exposed = aggregates["mass"] > 0.0
    rows.append({"comparison": "event_stored_vs_recalculated", **event_formula})
    if face_er is not None:
        static_er = _finite(face_er, "ER_mass_ratio_face", length=n_faces)
        rows.append({"comparison": "dynamic_face_vs_stored_dynamic_face", **_metric(aggregates["dynamic_er"], static_er, np.ones(n_faces, dtype=bool), bootstrap=bootstrap, seed=seed)})
    else:
        static_er = None
    if "model_ER_truth_face" in data:
        analytic = _finite(data["model_ER_truth_face"], "model_ER_truth_face", length=n_faces)
        rows.append({"comparison": "dynamic_face_vs_static_analytic_all_faces", **_metric(aggregates["dynamic_er"], analytic, np.ones(n_faces, dtype=bool), bootstrap=bootstrap, seed=seed)})
        rows.append({"comparison": "dynamic_face_vs_static_analytic_exposed_only", **_metric(aggregates["dynamic_er"], analytic, exposed, bootstrap=bootstrap, seed=seed)})
        rows.append({"comparison": "dynamic_formula_face_vs_static_analytic_all_faces", **_metric(aggregates["dynamic_formula_er"], analytic, np.ones(n_faces, dtype=bool), bootstrap=bootstrap, seed=seed)})
    result = {
        "version": VERSION,
        "status": "passed" if event_formula["n"] >= 3 and event_formula["nrmse_log1p"] is not None and event_formula["nrmse_log1p"] < 1e-10 else "failed",
        "accuracy_gate": False,
        "inputs": {"hits": str(hits_path), "hits_sha256": file_sha256(hits_path), "summary": str(summary_path), "summary_sha256": file_sha256(summary_path), "n_faces": n_faces, "n_impacts": n_impacts},
        "counts": {"exposed_faces": int(np.sum(exposed)), "zero_faces": int(np.sum(~exposed)), "exposed_fraction": float(np.mean(exposed))},
        "formula_self_consistency": event_formula,
        "comparisons": rows,
        "interpretation": "Dynamic collision quantities and static analytic fields are reported separately. Correlation diagnostics do not establish measured erosion accuracy and do not pass the official <5% gate.",
    }
    result["report_sha256"] = canonical_sha256(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return result


def selftest() -> bool:
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        hits = root / "hits.npz"; summary = root / "summary.json"; output = root / "report.json"
        vp = np.asarray([10.0, 20.0, 30.0, 40.0]); alpha = np.asarray([10.0, 25.0, 40.0, 60.0]); er = mansouri_er(vp, alpha)
        np.savez(hits, parent_face_id=np.asarray([0, 0, 1, 2]), parcel_mass_kg=np.ones(4), Vp_ms=vp, alpha_deg=alpha, ER_mansouri=er, ER_mass_ratio_face=np.zeros(3), model_ER_truth_face=np.asarray([er[0], er[2], er[3]]))
        summary.write_text(json.dumps({"version": "step30-cfd-driven-v2", "tracking": {"n_impacts": 4}, "geometry": {"n_parent_faces": 3}}), encoding="utf-8")
        report = run(hits, summary, output, bootstrap=10)
        assert report["status"] == "passed" and report["formula_self_consistency"]["nrmse_log1p"] < 1e-10
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hits", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest(); print("step30_correlation_diagnostics selftest: PASS")
    else:
        if not args.hits or not args.summary or not args.output: parser.error("--hits, --summary and --output are required")
        report = run(args.hits, args.summary, args.output, bootstrap=args.bootstrap, seed=args.seed)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(0 if report["status"] == "passed" else 2)
