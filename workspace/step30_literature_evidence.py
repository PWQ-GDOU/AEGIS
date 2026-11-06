"""Claim-level literature intake and fail-closed evidence validation.

The validator deliberately separates a DOI/citation record from accessible
original text.  Quantitative values, including apparent geometry fractions,
are unusable unless an original source file is hash-bound and a page plus
figure/table/section locator identifies the supporting material.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from step30_validation_common import canonical_sha256, safe_json_dump, sha256_file


VERSION = "step30-literature-evidence-v1"
MANIFEST_VERSION = "step30-literature-fem-provisional-manifest-v3"
EVIDENCE_LEVELS = {
    "quantitative": "supports a source-specific numerical claim only",
    "qualitative": "supports an ordinal/contextual claim only",
    "bibliographic_only": "citation metadata only; no claim is admissible",
}
CANONICAL_SOURCES = {
    "rai_2017_wear": {
        "title": "Hydro-abrasive erosion in Pelton buckets: Classification and field study",
        "journal": "Wear",
        "year": 2017,
        "doi": "10.1016/j.wear.2017.08.016",
    },
    "rai_2019_analytical": {
        "title": "Analytical modelling and mechanism of hydro-abrasive erosion in pelton buckets",
        "journal": "Wear",
        "year": 2019,
        "doi": "10.1016/j.wear.2019.203003",
    },
    "leguizamon_2019_multiscale": {
        "title": "Multiscale Simulation of the Hydroabrasive Erosion of a Pelton Bucket: Bridging Scales to Improve the Accuracy",
        "journal": "International Journal of Turbomachinery, Propulsion and Power",
        "year": 2019,
        "doi": "10.3390/ijtpp4020009",
    },
    "messa_2019_injector": {
        "title": "Hydro-abrasive erosion in Pelton turbine injectors: A numerical study",
        "journal": "Renewable Energy",
        "year": 2019,
        "doi": "10.1016/j.renene.2018.06.064",
    },
    "ge_2021_bucket": {
        "title": "Experimental and Numerical studies on Opening and Velocity Influence on Sediment Erosion of Pelton Turbine Buckets",
        "journal": "Renewable Energy",
        "year": 2021,
        "doi": "10.1016/j.renene.2021.04.072",
    },
    "fahrni_2025_defects": {
        "title": "Efficiency Testing of Pelton Turbines with Artificial Defects—Part 1: Buckets",
        "journal": "Energies",
        "year": 2025,
        "doi": "10.3390/en18112716",
    },
    "pelton_erosion_review_2023": {
        "title": "Sediment Erosion on Pelton Turbines: A Review",
        "journal": "Chinese Journal of Mechanical Engineering",
        "year": 2023,
        "doi": "10.1186/s10033-023-00880-y",
    },
}
MODEL_PRIOR_BINDINGS = {
    "oka": {
        "implementation": "step30_material_model_calibration.oka_volume_mm3_per_kg",
        "evidence_status": "bibliographic_only_pending_original_source_page_binding",
    },
    "finnie": {
        "implementation": "step30_material_model_calibration.finnie_volume_mm3_per_kg",
        "evidence_status": "bibliographic_only_pending_original_source_page_binding",
    },
    "tabakoff": {
        "implementation": "step30_material_model_calibration.tabakoff_volume_mm3_per_kg",
        "evidence_status": "bibliographic_only_pending_original_source_page_binding",
    },
    "mansouri": {
        "implementation": "step30_material_model_calibration.mansouri_mass_ratio_kg_per_kg",
        "evidence_status": "bibliographic_only_pending_original_source_page_binding",
    },
}


class LiteratureEvidenceError(ValueError):
    """Raised when an intake attempts to support an untraceable claim."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LiteratureEvidenceError(f"{name} must be a non-empty string")
    return value.strip()


def _source_binding(item: Mapping[str, Any]) -> dict[str, Any]:
    kind = _text(item.get("container_kind"), "source container_kind")
    if kind not in {"original_full_text", "bibliographic_index"}:
        raise LiteratureEvidenceError(
            "source container_kind must be original_full_text or bibliographic_index"
        )
    path_value = item.get("path")
    if path_value is None:
        url = _text(item.get("url"), "source url")
        return {
            "container_kind": kind,
            "path": None,
            "url": url,
            "sha256": None,
            "bytes": None,
            "hash_verified": False,
        }
    path = Path(_text(path_value, "source path")).resolve()
    digest = _text(item.get("sha256"), "source sha256").lower()
    if len(digest) != 64:
        raise LiteratureEvidenceError("source sha256 must contain 64 hex characters")
    if not path.is_file():
        raise LiteratureEvidenceError(f"source file is missing: {path}")
    actual = sha256_file(path)
    if actual != digest:
        raise LiteratureEvidenceError(f"source hash mismatch: {path}")
    expected_bytes = item.get("bytes")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
        raise LiteratureEvidenceError("source bytes must be an integer")
    if path.stat().st_size != expected_bytes:
        raise LiteratureEvidenceError(f"source byte count mismatch: {path}")
    return {
        "container_kind": kind,
        "path": str(path),
        "url": item.get("url"),
        "sha256": digest,
        "bytes": expected_bytes,
        "hash_verified": True,
    }


def _validate_citation(source_id: str, item: Mapping[str, Any]) -> dict[str, Any]:
    canonical = CANONICAL_SOURCES.get(source_id)
    if canonical is None:
        raise LiteratureEvidenceError(f"source_id is not registered: {source_id}")
    citation = {
        "source_id": source_id,
        "title": _text(item.get("title"), "title"),
        "journal": _text(item.get("journal"), "journal"),
        "year": int(item.get("year")),
        "doi": _text(item.get("doi"), "doi").lower(),
    }
    if citation["year"] != canonical["year"] or any(
        citation[field] != canonical[field] for field in ("title", "journal", "doi")
    ):
        raise LiteratureEvidenceError(
            f"citation metadata mismatch for {source_id}; use the canonical DOI/title/year record"
        )
    return citation


def _validated_claims(
    claims: Sequence[Mapping[str, Any]], source: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    results: list[dict[str, Any]] = []
    blockers: list[str] = []
    for raw in claims:
        claim_id = _text(raw.get("claim_id"), "claim_id")
        text = _text(raw.get("claim"), "claim")
        requested = _text(raw.get("evidence_level"), "claim evidence_level")
        if requested not in {"quantitative", "qualitative"}:
            raise LiteratureEvidenceError("claim evidence_level must be quantitative or qualitative")
        if raw.get("dec_dimension_transfer") is True or raw.get("dec_facewise_truth") is True:
            raise LiteratureEvidenceError(
                "literature claims may not be transferred into DEC dimensions or face-wise truth"
            )
        page = raw.get("page")
        locator = raw.get("locator")
        source_is_original = (
            source["container_kind"] == "original_full_text"
            and source["hash_verified"] is True
        )
        location_ok = isinstance(page, int) and page > 0 and isinstance(locator, str) and locator.strip()
        eligible = bool(source_is_original and location_ok)
        reason = None
        if not source_is_original:
            reason = "accessible source is not a hash-bound original full text"
        elif not location_ok:
            reason = "original text claim lacks a page and figure/table/section locator"
        level = requested if eligible else "bibliographic_only"
        if reason is not None:
            blockers.append(f"{claim_id}: {reason}")
        results.append(
            {
                "claim_id": claim_id,
                "claim": text,
                "requested_evidence_level": requested,
                "evidence_level": level,
                "page": page if location_ok else None,
                "locator": locator.strip() if isinstance(locator, str) and locator.strip() else None,
                "quantitative_calibration_eligible": bool(
                    eligible and requested == "quantitative"
                ),
                "qualitative_prior_eligible": bool(
                    eligible and requested == "qualitative"
                ),
                "dec_dimension_transfer": False,
                "dec_facewise_truth": False,
                "applicability_boundary": _text(
                    raw.get("applicability_boundary", "source-specific geometry/material/operating conditions"),
                    "claim applicability_boundary",
                ),
                "downgrade_reason": reason,
            }
        )
    return results, blockers


def validate_literature_intake(
    intake: Mapping[str, Any],
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate source-level evidence and downgrade unsupported claims."""
    if intake.get("version") != VERSION:
        raise LiteratureEvidenceError(f"intake version must be {VERSION}")
    sources = intake.get("sources")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise LiteratureEvidenceError("intake sources must be an array")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    blockers: list[str] = []
    for raw in sources:
        if not isinstance(raw, Mapping):
            raise LiteratureEvidenceError("each literature source must be an object")
        source_id = _text(raw.get("source_id"), "source_id")
        if source_id in seen:
            raise LiteratureEvidenceError("source_id must be unique")
        seen.add(source_id)
        citation = _validate_citation(source_id, raw)
        source = _source_binding(raw.get("source", {}))
        claims, claim_blockers = _validated_claims(raw.get("claims", []), source)
        blockers.extend(f"{source_id}: {item}" for item in claim_blockers)
        source_status = (
            "claim_level_evidence"
            if any(
                claim["quantitative_calibration_eligible"]
                or claim["qualitative_prior_eligible"]
                for claim in claims
            )
            else "bibliographic_only"
        )
        normalized.append(
            {
                "citation": citation,
                "source": source,
                "claims": claims,
                "source_status": source_status,
                "source_sha256": source["sha256"],
            }
        )
    result = {
        "version": VERSION,
        "status": "provisional",
        "sources": normalized,
        "source_count": len(normalized),
        "claim_count": sum(len(item["claims"]) for item in normalized),
        "quantitative_claim_count": sum(
            int(claim["quantitative_calibration_eligible"])
            for item in normalized
            for claim in item["claims"]
        ),
        "qualitative_claim_count": sum(
            int(claim["qualitative_prior_eligible"])
            for item in normalized
            for claim in item["claims"]
        ),
        "bibliographic_only_source_count": sum(
            int(item["source_status"] == "bibliographic_only") for item in normalized
        ),
        "model_prior_bindings": MODEL_PRIOR_BINDINGS,
        "blockers": sorted(blockers),
        "quantitative_anchor_policy": (
            "No 3%/5% geometry fraction, model coefficient, material range, or concentration range "
            "is usable without hash-bound original text plus page and figure/table/section locator."
        ),
        "dec_transfer_policy": (
            "Literature evidence cannot be converted into DEC dimensions, DEC face-wise truth, "
            "or an official spatial-accuracy assertion."
        ),
        "official_accuracy_gate": False,
        "official_reference_gate": False,
        "fem_gate": "blocked",
    }
    result["evidence_sha256"] = canonical_sha256(result)
    if output_path is not None:
        safe_json_dump(result, output_path)
    return result


def default_bibliographic_intake(index_path: str | Path) -> dict[str, Any]:
    """Create a conservative intake from a local bibliography, not original text."""
    index = Path(index_path).resolve()
    if not index.is_file():
        raise LiteratureEvidenceError(f"bibliographic index is missing: {index}")
    source = {
        "container_kind": "bibliographic_index",
        "path": str(index),
        "sha256": sha256_file(index),
        "bytes": int(index.stat().st_size),
    }
    return {
        "version": VERSION,
        "sources": [
            {
                "source_id": source_id,
                **citation,
                "source": dict(source),
                "claims": [],
            }
            for source_id, citation in CANONICAL_SOURCES.items()
        ],
    }


def build_v3_provisional_manifest(
    evidence_path: str | Path,
    *,
    prior_report_path: str | Path,
    reference_report_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Bind evidence and priors while preventing unsupported calibration."""
    evidence = Path(evidence_path).resolve()
    prior = Path(prior_report_path).resolve()
    reference = Path(reference_report_path).resolve()
    for path, name in (
        (evidence, "literature evidence"),
        (prior, "literature prior report"),
        (prior.with_suffix(".npz"), "literature prior maps"),
        (reference, "literature reference distribution"),
    ):
        if not path.is_file():
            raise LiteratureEvidenceError(f"{name} artifact is missing: {path}")
    evidence_report = json.loads(evidence.read_text(encoding="utf-8"))
    if evidence_report.get("version") != VERSION:
        raise LiteratureEvidenceError("unsupported literature evidence artifact")
    manifest = {
        "version": MANIFEST_VERSION,
        "status": "provisional",
        "evidence_status": "literature_prior_reproducibility_only",
        "artifacts": [
            {"role": "literature evidence report", "path": str(evidence), "sha256": sha256_file(evidence)},
            {"role": "literature prior degradation report", "path": str(prior), "sha256": sha256_file(prior)},
            {"role": "literature prior degradation maps", "path": str(prior.with_suffix(".npz")), "sha256": sha256_file(prior.with_suffix(".npz"))},
            {"role": "legacy qualitative reference distribution", "path": str(reference), "sha256": sha256_file(reference)},
        ],
        "evidence_summary": {
            "evidence_sha256": evidence_report.get("evidence_sha256"),
            "quantitative_claim_count": evidence_report.get("quantitative_claim_count"),
            "qualitative_claim_count": evidence_report.get("qualitative_claim_count"),
            "bibliographic_only_source_count": evidence_report.get("bibliographic_only_source_count"),
            "model_prior_bindings": evidence_report.get("model_prior_bindings"),
        },
        "parameter_prior_status": {
            model: "blocked_pending_claim_level_original_source_evidence"
            for model in MODEL_PRIOR_BINDINGS
        },
        "regional_reference_status": (
            "legacy qualitative distribution retained for history; not promoted without claim-level original evidence"
        ),
        "quantitative_literature_calibration": False,
        "DEC_dimension_transfer": False,
        "DEC_facewise_truth": False,
        "official_reference_gate": False,
        "official_accuracy_gate": False,
        "independent_accuracy_gate": False,
        "fem_gate": "blocked",
        "limitations": [
            "Model comparisons can be reproduced, but their coefficient priors are not quantitatively anchored by this manifest.",
            "The 3% and 5% source-specific geometry values are not DEC dimensions or spatial truth.",
            "No literature package can replace independent baseline/worn scans for spatial accuracy.",
        ],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    if output_path is not None:
        safe_json_dump(manifest, output_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Step30 claim-level literature evidence")
    parser.add_argument("--intake", type=Path)
    parser.add_argument("--bibliographic-index", type=Path)
    parser.add_argument("--output-evidence", type=Path, required=True)
    parser.add_argument("--prior-report", type=Path)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    args = parser.parse_args(argv)
    if (args.intake is None) == (args.bibliographic_index is None):
        parser.error("supply exactly one of --intake or --bibliographic-index")
    intake = (
        json.loads(args.intake.read_text(encoding="utf-8"))
        if args.intake is not None
        else default_bibliographic_intake(args.bibliographic_index)
    )
    report = validate_literature_intake(intake, output_path=args.output_evidence)
    display: dict[str, Any] = {"evidence": report}
    if args.output_manifest is not None:
        if args.prior_report is None or args.reference_report is None:
            parser.error("--output-manifest requires --prior-report and --reference-report")
        manifest = build_v3_provisional_manifest(
            args.output_evidence,
            prior_report_path=args.prior_report,
            reference_report_path=args.reference_report,
            output_path=args.output_manifest,
        )
        display["manifest"] = manifest
    print(json.dumps(display, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
