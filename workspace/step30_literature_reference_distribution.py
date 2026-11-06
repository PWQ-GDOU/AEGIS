"""Literature-derived regional erosion reference distribution.

This module freezes a conservative, region-level reference prior from public
Pelton erosion studies.  It intentionally does not manufacture a face-wise
DEC truth field: literature observations are stored as qualitative support
and, where explicitly reported, geometry-change anchors.  A comparison with
the current model maps is only enabled when an independently supplied,
externally confirmed face-region mapping is present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from step30_validation_common import canonical_sha256, safe_json_dump, sha256_file


VERSION = "step30-literature-reference-distribution-v1"
REFERENCE_KIND = "literature_reference"
MANIFEST_VERSION = "step30-literature-fem-provisional-manifest-v2"
MODEL_MAP_KEYS = ("oka", "finnie", "tabakoff", "mansouri")
REGION_IDS = (
    "splitter_like",
    "cutout_like",
    "brim_like",
    "bucket_base_or_side_like",
    "impact_zone_like",
    "residual_other",
)


class LiteratureReferenceError(ValueError):
    """Raised when the literature-reference contract is invalid."""


# The evidence level describes what the cited source supports.  It is not a
# probability of damage on the DEC geometry.
LITERATURE_SOURCE_EVIDENCE = (
    {
        "source_id": "rai_2017_wear",
        "citation": (
            "Rai, Kumar and Staubli, Hydro-abrasive erosion in Pelton "
            "buckets: Classification and field study"
        ),
        "journal": "Wear",
        "year": 2017,
        "doi": "10.1016/j.wear.2017.08.016",
        "source_role": "field geometry-change anchor",
        "regional_evidence": {
            "splitter_like": "high",
            "cutout_like": "high",
        },
        "reported_anchors": [
            {
                "region_id": "splitter_like",
                "metric": "average_splitter_height_reduction_fraction_of_bucket_width",
                "value": 0.03,
                "units": "fraction_of_bucket_width",
                "scope": "reported field-study average for uncoated buckets",
            },
            {
                "region_id": "cutout_like",
                "metric": "cutout_abrasion_fraction_of_bucket_width",
                "value": 0.05,
                "units": "fraction_of_bucket_width",
                "scope": "reported field-study average for uncoated buckets",
            },
        ],
    },
    {
        "source_id": "rai_2019_analytical",
        "citation": (
            "Rai, Kumar and Staubli, Analytical modelling and mechanism of "
            "hydro-abrasive erosion in pelton buckets"
        ),
        "journal": "Wear",
        "year": 2019,
        "doi": "10.1016/j.wear.2019.203003",
        "source_role": "mechanism and analytical-model context",
        "regional_evidence": {
            "splitter_like": "medium",
            "cutout_like": "medium",
            "impact_zone_like": "medium",
        },
        "reported_anchors": [],
    },
    {
        "source_id": "leguizamon_2019_multiscale",
        "citation": (
            "Leguizamon et al., Multiscale Simulation of the Hydroabrasive "
            "Erosion of a Pelton Bucket: Bridging Scales to Improve the Accuracy"
        ),
        "journal": "International Journal of Turbomachinery, Propulsion and Power",
        "year": 2019,
        "doi": "10.3390/ijtpp4020009",
        "source_role": "full-scale multiscale impact and erosion distribution",
        "regional_evidence": {
            "splitter_like": "high",
            "impact_zone_like": "medium",
        },
        "reported_anchors": [
            {
                "region_id": "splitter_like",
                "metric": "hotspot_order",
                "value": 2,
                "units": "ordinal_hotspot_rank",
                "scope": "splitter reported as the second sediment-impact hotspot",
            }
        ],
    },
    {
        "source_id": "ge_2021_bucket",
        "citation": (
            "Ge et al., Experimental and Numerical studies on Opening and "
            "Velocity Influence on Sediment Erosion of Pelton Turbine Buckets"
        ),
        "journal": "Renewable Energy",
        "year": 2021,
        "doi": "10.1016/j.renene.2021.04.072",
        "source_role": "bucket erosion experiment and numerical sensitivity",
        "regional_evidence": {
            "impact_zone_like": "medium",
            "splitter_like": "medium",
            "cutout_like": "medium",
        },
        "reported_anchors": [],
    },
    {
        "source_id": "fahrni_2025_defects",
        "citation": (
            "Fahrni, Staubli and Casartelli, Efficiency Testing of Pelton "
            "Turbines with Artificial Defects—Part 1: Buckets"
        ),
        "journal": "Energies",
        "year": 2025,
        "doi": "10.3390/en18112716",
        "source_role": "load-dependent defect-location sensitivity",
        "regional_evidence": {
            "splitter_like": "high",
            "cutout_like": "high",
            "bucket_base_or_side_like": "high",
        },
        "reported_anchors": [
            {
                "region_id": "splitter_like",
                "metric": "partial_load_defect_sensitivity",
                "value": 1,
                "units": "qualitative",
                "scope": "splitter defects dominate the reported partial-load losses",
            },
            {
                "region_id": "cutout_like",
                "metric": "full_load_defect_sensitivity",
                "value": 1,
                "units": "qualitative",
                "scope": "cutout defects dominate the reported full-load losses",
            },
        ],
    },
    {
        "source_id": "pelton_erosion_review_2023",
        "citation": "Sediment Erosion on Pelton Turbines: A Review",
        "journal": "Chinese Journal of Mechanical Engineering",
        "year": 2023,
        "doi": "10.1186/s10033-023-00880-y",
        "source_role": "review-level component and mechanism context",
        "regional_evidence": {
            "splitter_like": "medium",
            "cutout_like": "medium",
            "brim_like": "medium",
            "impact_zone_like": "medium",
        },
        "reported_anchors": [],
    },
)

_EVIDENCE_SCORE = {"high": 3, "medium": 2, "low": 1, "not_reported": 0}


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LiteratureReferenceError(f"{name} must be a non-empty string")
    return value.strip()


def _region_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    total_score = 0
    for region_id in REGION_IDS:
        supporting: list[dict[str, Any]] = []
        anchors: list[dict[str, Any]] = []
        score = 0
        for source in LITERATURE_SOURCE_EVIDENCE:
            level = source["regional_evidence"].get(region_id, "not_reported")
            if level == "not_reported":
                continue
            source_score = _EVIDENCE_SCORE[level]
            score += source_score
            supporting.append(
                {
                    "source_id": source["source_id"],
                    "evidence_level": level,
                    "evidence_score": source_score,
                }
            )
            anchors.extend(
                dict(anchor)
                for anchor in source["reported_anchors"]
                if anchor["region_id"] == region_id
            )
        total_score += score
        records.append(
            {
                "region_id": region_id,
                "literature_support_score": score,
                "supporting_sources": supporting,
                "reported_geometry_anchors": anchors,
            }
        )
    for record in records:
        record["literature_support_fraction"] = (
            float(record["literature_support_score"] / total_score)
            if total_score
            else 0.0
        )
        record["interpretation"] = (
            "ordinal literature support share; not a physical damage probability"
        )
    return records


def _load_model_maps(
    prior_report_path: str | Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, Path]:
    report_path = Path(prior_report_path).resolve()
    if not report_path.is_file():
        raise LiteratureReferenceError(f"prior report does not exist: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    input_bindings = report.get("input_bindings", {})
    hits_path = Path(_text(input_bindings.get("particle_hits_path"), "particle_hits_path"))
    npz_path = report_path.with_suffix(".npz")
    if not npz_path.is_file():
        raise LiteratureReferenceError(f"prior map artifact does not exist: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        missing = [
            f"{model}_p50_mm_h"
            for model in MODEL_MAP_KEYS
            if f"{model}_p50_mm_h" not in data.files
        ]
        if missing:
            raise LiteratureReferenceError(f"prior map artifact is missing fields: {missing}")
        centers = np.asarray(data["face_centers_local_mm"], dtype=np.float64)
        maps = {
            model: np.asarray(data[f"{model}_p50_mm_h"], dtype=np.float64)
            for model in MODEL_MAP_KEYS
        }
    if not hits_path.is_file():
        raise LiteratureReferenceError(f"particle-hit artifact does not exist: {hits_path}")
    with np.load(hits_path, allow_pickle=False) as hits:
        if "face_area_m2" not in hits.files:
            raise LiteratureReferenceError("particle-hit artifact is missing face_area_m2")
        areas = np.asarray(hits["face_area_m2"], dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) != len(areas):
        raise LiteratureReferenceError("prior face centers and areas have inconsistent lengths")
    for model, values in maps.items():
        if values.shape != (len(areas),) or not np.isfinite(values).all():
            raise LiteratureReferenceError(f"{model} prior map is invalid")
    return maps, areas, npz_path


def _load_region_labels(
    path: str | Path,
    *,
    expected_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = Path(path).resolve()
    if not source.is_file():
        raise LiteratureReferenceError(f"face-region mapping does not exist: {source}")
    with np.load(source, allow_pickle=False) as data:
        field = "region_id" if "region_id" in data.files else "region_ids"
        if field not in data.files:
            raise LiteratureReferenceError(
                "face-region mapping must contain region_id or region_ids"
            )
        labels = np.asarray(data[field]).astype(str)
        status = (
            str(np.asarray(data["mapping_status"]).item())
            if "mapping_status" in data.files
            else "unknown"
        )
    if labels.shape != (expected_count,):
        raise LiteratureReferenceError("face-region mapping length differs from face count")
    if status != "externally_confirmed":
        raise LiteratureReferenceError(
            "face-region mapping must be explicitly marked externally_confirmed"
        )
    unknown = sorted(set(labels) - set(REGION_IDS))
    if unknown:
        raise LiteratureReferenceError(f"face-region mapping has unsupported regions: {unknown}")
    return labels, {
        "path": str(source),
        "sha256": sha256_file(source),
        "mapping_status": status,
    }


def _model_comparison(
    maps: Mapping[str, np.ndarray],
    areas_m2: np.ndarray,
    labels: np.ndarray | None,
    mapping_binding: Mapping[str, Any] | None,
    reference: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if labels is None or mapping_binding is None:
        return {
            "status": "blocked",
            "reason": (
                "No externally confirmed face-region mapping is available; "
                "raw surface tags, bounding boxes, and visual guesses are inadmissible."
            ),
            "official_accuracy_gate": False,
            "models": {},
        }
    reference_order = [
        item["region_id"]
        for item in sorted(
            reference,
            key=lambda item: (
                -float(item["literature_support_score"]),
                item["region_id"],
            ),
        )
    ]
    results: dict[str, Any] = {}
    for model, values in maps.items():
        # A top-10% area-weighted diagnostic is useful for prioritization, but
        # is not the preregistered independent accuracy metric.
        order = np.argsort(values)[::-1]
        total_area = float(np.sum(areas_m2))
        cumulative = 0.0
        selected: list[int] = []
        target = 0.10 * total_area
        for index in order:
            selected.append(int(index))
            cumulative += float(areas_m2[index])
            if cumulative >= target:
                break
        area_by_region = {
            region: float(np.sum(areas_m2[np.isin(labels, region)]))
            for region in REGION_IDS
        }
        selected_area_by_region = {
            region: float(np.sum(areas_m2[np.asarray(selected)[labels[selected] == region]]))
            if selected
            else 0.0
            for region in REGION_IDS
        }
        ranked = sorted(
            REGION_IDS,
            key=lambda region: (-selected_area_by_region[region], region),
        )
        results[model] = {
            "top_10_area_region_order": ranked,
            "top_10_area_region_share": {
                region: (
                    selected_area_by_region[region] / max(cumulative, 1.0e-30)
                )
                for region in REGION_IDS
            },
            "whole_face_region_area_share": {
                region: area_by_region[region] / max(total_area, 1.0e-30)
                for region in REGION_IDS
            },
            "reference_priority_order": reference_order,
            "status": "provisional_diagnostic",
            "not_an_accuracy_metric": True,
        }
    return {
        "status": "provisional_diagnostic",
        "reason": (
            "Region-level ranking only; no literature prior is used as a "
            "DEC face-wise truth field."
        ),
        "official_accuracy_gate": False,
        "mapping": dict(mapping_binding),
        "models": results,
    }


def _artifact(path: str | Path, role: str) -> dict[str, str]:
    target = Path(path).resolve()
    if not target.is_file():
        raise LiteratureReferenceError(f"{role} does not exist: {target}")
    return {
        "role": role,
        "path": str(target),
        "sha256": sha256_file(target),
    }


def recompute_provisional_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    return canonical_sha256(unsigned)


def build_provisional_manifest(
    reference_report_path: str | Path,
    *,
    prior_report_path: str | Path,
    geometry_audit_path: str | Path,
    scan_pair_template_path: str | Path,
    acquisition_template_path: str | Path,
    acquisition_readme_path: str | Path,
) -> dict[str, Any]:
    """Bind the new literature/scan deliverables without opening formal gates."""
    reference_report = Path(reference_report_path).resolve()
    prior_report = Path(prior_report_path).resolve()
    artifacts = [
        _artifact(reference_report, "literature reference report"),
        _artifact(prior_report, "literature prior report"),
        _artifact(prior_report.with_suffix(".npz"), "literature prior maps"),
        _artifact(geometry_audit_path, "geometry audit report"),
        _artifact(scan_pair_template_path, "scan-pair manifest template"),
        _artifact(acquisition_template_path, "baseline/worn acquisition template"),
        _artifact(acquisition_readme_path, "baseline/worn acquisition guide"),
    ]
    result: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "status": "provisional",
        "immutable": True,
        "fem_gate": "blocked",
        "official_reference_gate": False,
        "official_accuracy_gate": False,
        "independent_accuracy_gate": False,
        "ai5_used_for_calibration": False,
        "artifacts": artifacts,
        "interpretation": {
            "literature_reference": (
                "qualitative region-level prior; not measured DEC erosion truth"
            ),
            "model_region_comparison": (
                "blocked until externally confirmed face-region mapping is supplied"
            ),
            "baseline_worn_scans": (
                "blocked until raw independent measured scan pairs are supplied"
            ),
            "stress_field": "analytical/parametric only; not real FEM",
            "root_wetted_ai5_selection": "blocked",
        },
        "limitations": [
            "The manifest does not promote literature anchors to DEC dimensions.",
            "The manifest does not classify synthetic data, CFD fields, or AI5 traces as erosion truth.",
            "No real FEM solver result or independent measured scan is included.",
        ],
    }
    result["manifest_sha256"] = recompute_provisional_manifest_sha256(result)
    return result


def build_reference_distribution(
    *,
    prior_report_path: str | Path | None = None,
    face_region_mapping_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the frozen qualitative literature reference package."""
    reference = _region_records()
    maps: dict[str, np.ndarray] = {}
    areas = np.empty(0, dtype=np.float64)
    prior_map_artifact: Path | None = None
    labels: np.ndarray | None = None
    mapping_binding: dict[str, Any] | None = None
    blockers: list[str] = []

    if prior_report_path is not None:
        try:
            maps, areas, prior_map_artifact = _load_model_maps(prior_report_path)
        except (OSError, ValueError, KeyError, LiteratureReferenceError) as exc:
            blockers.append(f"model-prior comparison input blocked: {exc}")
    if face_region_mapping_path is not None:
        if not maps:
            blockers.append("face-region mapping supplied without a valid model-prior map")
        else:
            try:
                labels, mapping_binding = _load_region_labels(
                    face_region_mapping_path,
                    expected_count=len(areas),
                )
            except (OSError, ValueError, LiteratureReferenceError) as exc:
                blockers.append(f"face-region mapping blocked: {exc}")
    if prior_report_path is None or face_region_mapping_path is None:
        blockers.append(
            "No externally confirmed face-region mapping was supplied for model comparison."
        )

    report: dict[str, Any] = {
        "version": VERSION,
        "status": "provisional",
        "reference_kind": REFERENCE_KIND,
        "official_reference_gate": False,
        "official_accuracy_gate": False,
        "ai5_used_for_calibration": False,
        "distribution_semantics": {
            "type": "qualitative_region_prior",
            "probability_claim": False,
            "literature_support_fraction_is": (
                "normalized source-support score for prioritization only"
            ),
            "facewise_DEC_truth": False,
        },
        "literature_sources": [dict(item) for item in LITERATURE_SOURCE_EVIDENCE],
        "regions": reference,
        "model_comparison": _model_comparison(
            maps,
            areas,
            labels,
            mapping_binding,
            reference,
        ),
        "input_bindings": {
            "prior_report_path": (
                None
                if prior_report_path is None
                else str(Path(prior_report_path).resolve())
            ),
            "prior_map_artifact_path": (
                None if prior_map_artifact is None else str(prior_map_artifact)
            ),
            "prior_map_artifact_sha256": (
                None if prior_map_artifact is None else sha256_file(prior_map_artifact)
            ),
            "face_region_mapping_path": (
                None
                if mapping_binding is None
                else mapping_binding["path"]
            ),
            "face_region_mapping_sha256": (
                None
                if mapping_binding is None
                else mapping_binding["sha256"]
            ),
        },
        "blockers": sorted(set(blockers)),
        "limitations": [
            "The regional package is a literature_reference, not measured DEC truth.",
            "Reported 3% and 5% values are source-specific geometry-change anchors and are not transferred to DEC dimensions.",
            "No AI5 trace, CAD surface tag, bounding-box heuristic, or visual guess is used to label regions.",
            "Independent baseline/worn scans remain required for spatial accuracy and remain blocked when absent.",
        ],
    }
    report["report_sha256"] = canonical_sha256(report)
    if output_path is not None:
        safe_json_dump(report, output_path)
    return report


def selftest() -> bool:
    reference = _region_records()
    assert reference[0]["literature_support_score"] > 0
    assert any(
        anchor["value"] == 0.03
        for region in reference
        for anchor in region["reported_geometry_anchors"]
    )
    report = build_reference_distribution()
    assert report["status"] == "provisional"
    assert report["model_comparison"]["status"] == "blocked"
    assert report["official_accuracy_gate"] is False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a conservative literature-reference erosion distribution"
    )
    parser.add_argument("--prior-report", type=Path)
    parser.add_argument("--face-region-mapping", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        selftest()
        print("step30_literature_reference_distribution selftest: PASS")
        return 0
    report = build_reference_distribution(
        prior_report_path=args.prior_report,
        face_region_mapping_path=args.face_region_mapping,
        output_path=args.output,
    )
    display: dict[str, Any] = dict(report)
    if args.manifest_output is not None:
        reference_report_path = args.reference_report or args.output
        if reference_report_path is None or args.prior_report is None:
            parser.error(
                "--manifest-output requires --prior-report and "
                "--reference-report or --output"
            )
        root = Path(__file__).resolve().parent
        manifest = build_provisional_manifest(
            reference_report_path,
            prior_report_path=args.prior_report,
            geometry_audit_path=root / "output" / "step30_fem_geometry_audit_v1.json",
            scan_pair_template_path=root
            / "templates"
            / "step30-scan-pair-manifest-v1.json",
            acquisition_template_path=root
            / "templates"
            / "step30-independent-baseline-worn-acquisition-v1.json",
            acquisition_readme_path=root / "INDEPENDENT_BASELINE_WORN_ACQUISITION.md",
        )
        safe_json_dump(manifest, args.manifest_output)
        display = {
            "reference_report": report,
            "provisional_manifest": {
            "path": str(args.manifest_output.resolve()),
            "sha256": sha256_file(args.manifest_output),
            },
        }
    print(json.dumps(display, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
