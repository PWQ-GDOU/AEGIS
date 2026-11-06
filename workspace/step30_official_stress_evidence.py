"""Auditable DEC official stress-table evidence and 130 MW leave-one diagnostic.

The legacy AI5 time-series report remains unchanged.  This module records the
competition report's Table 3-1 and Table 3-3 transcription separately, uses
the report's definitions (static = mean, dynamic = peak-to-peak), and keeps
strict qualification blocked until the transcription has two independent
reviews.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from step30_dec_stress_validation import compute_stress_statistics, load_dec_stress_data
from step30_validation_common import canonical_sha256, safe_json_dump, sha256_file


VERSION = "step30-official-stress-evidence-v1"
STRICT_RELATIVE_THRESHOLD = 0.05
OFFICIAL_TABLE_3_3 = (
    ("rated_noload", 0, 1, 22.31, 2.21),
    ("30MW", 30, 2, 24.19, 16.81),
    ("40MW", 40, 2, 24.69, 25.62),
    ("50MW", 50, 4, 25.24, 14.06),
    ("60MW", 60, 4, 25.79, 16.89),
    ("70MW", 70, 4, 26.33, 20.60),
    ("80MW", 80, 4, 26.88, 24.16),
    ("90MW", 90, 4, 27.48, 28.78),
    ("100MW", 100, 6, 28.04, 17.62),
    ("110MW", 110, 6, 28.56, 20.12),
    ("120MW", 120, 6, 29.10, 23.63),
    ("130MW", 130, 6, 29.57, 25.31),
    ("140MW", 140, 6, 29.92, 29.89),
)
OFFICIAL_TABLE_3_1_OPENING = {
    30: 39.70,
    40: 57.505,
    50: 30.5325,
    60: 37.8625,
    70: 45.975,
    80: 54.8325,
    90: 65.9675,
    100: 42.1767,
    110: 48.25,
    120: 54.495,
    130: 62.0675,
    140: 70.38,
}


class OfficialStressEvidenceError(ValueError):
    """Raised when source binding or review state cannot support a report."""


def official_table_records() -> list[dict[str, Any]]:
    """Return the frozen single-transcription table with page/table bindings."""
    rows = []
    for condition, power, nozzle_count, static, dynamic_p2p in OFFICIAL_TABLE_3_3:
        opening = None if power == 0 else OFFICIAL_TABLE_3_1_OPENING[power]
        rows.append(
            {
                "condition": condition,
                "power_mw": power,
                "nozzle_count": nozzle_count,
                "mean_needle_opening_pct": opening,
                "static_mean_mpa": static,
                "dynamic_peak_to_peak_mpa": dynamic_p2p,
                "source_binding": {
                    "static_and_dynamic": {
                        "page": 5,
                        "table": "3-3",
                        "row": condition,
                        "definition": (
                            "static stress is mean; dynamic stress amplitude is peak-to-peak"
                        ),
                    },
                    "needle_opening": (
                        None
                        if power == 0
                        else {"page": 4, "table": "3-1", "row": condition}
                    ),
                },
            }
        )
    return rows


def _validate_reviews(reviews: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    reviewer_ids: set[str] = set()
    for item in reviews:
        reviewer = str(item.get("reviewer_id", "")).strip()
        status = str(item.get("status", "")).strip()
        if not reviewer or status not in {"completed_match", "pending", "mismatch"}:
            raise OfficialStressEvidenceError("review records require reviewer_id and valid status")
        if reviewer in reviewer_ids:
            raise OfficialStressEvidenceError("reviewer IDs must be distinct")
        reviewer_ids.add(reviewer)
        normalized.append(
            {
                "reviewer_id": reviewer,
                "status": status,
                "reviewed_at": str(item.get("reviewed_at", "")).strip() or None,
                "scope": str(item.get("scope", "table_3_1_and_3_3")).strip(),
                "notes": str(item.get("notes", "")).strip() or None,
            }
        )
    return normalized


def official_table_is_double_reviewed(reviews: Sequence[Mapping[str, Any]]) -> bool:
    completed = {
        str(item.get("reviewer_id"))
        for item in reviews
        if item.get("status") == "completed_match"
    }
    return len(completed) >= 2


def build_official_stress_evidence(
    official_pdf_path: str | Path,
    *,
    transcription_reviews: Sequence[Mapping[str, Any]],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a table-level evidence package from an official report PDF."""
    pdf = Path(official_pdf_path).resolve()
    if not pdf.is_file():
        raise OfficialStressEvidenceError(f"official stress report is missing: {pdf}")
    reviews = _validate_reviews(transcription_reviews)
    table = official_table_records()
    result = {
        "version": VERSION,
        "status": "provisional_transcription",
        "formal_evidence_status": "blocked_pending_double_transcription_review",
        "official_pdf": {
            "path": str(pdf),
            "sha256": sha256_file(pdf),
            "pages_used": [4, 5],
            "tables_used": ["3-1", "3-3"],
        },
        "measurement_definition": {
            "static_stress": "mean_mpa",
            "dynamic_stress": "peak_to_peak_mpa",
            "source_page": 5,
            "source_table": "3-3",
        },
        "transcription_reviews": reviews,
        "double_review_completed": official_table_is_double_reviewed(reviews),
        "table_3_3_rows": table,
        "table_3_3_sha256": canonical_sha256(table),
        "ai5_raw_timeseries_role": (
            "recomputable diagnostic only; raw peak-to-peak values do not replace official table statistics"
        ),
        "strict_scoring_eligible": False,
        "official_accuracy_gate": False,
        "fem_gate": "blocked",
    }
    result["strict_scoring_eligible"] = bool(result["double_review_completed"])
    result["formal_evidence_status"] = (
        "official_table_transcription_double_reviewed"
        if result["double_review_completed"]
        else "blocked_pending_double_transcription_review"
    )
    result["evidence_sha256"] = canonical_sha256(result)
    if output_path is not None:
        safe_json_dump(result, output_path)
    return result


def raw_ai5_diagnostic(
    ai5_data_directory: str | Path | None,
    official_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare raw AI5 recomputation against table transcription without scoring."""
    if ai5_data_directory is None:
        return {"status": "not_run", "reason": "no raw AI5 directory supplied"}
    source = Path(ai5_data_directory).resolve()
    if not source.is_dir():
        return {"status": "blocked", "reason": f"AI5 data directory is missing: {source}"}
    stats = compute_stress_statistics(load_dec_stress_data(source))
    by_condition = {str(row["condition"]): row for row in official_rows}
    records = []
    for condition, row in sorted(by_condition.items()):
        raw = stats.get(condition)
        if raw is None:
            records.append({"condition": condition, "status": "missing_raw_ai5"})
            continue
        raw_p2p = 2.0 * float(raw["amplitude_mpa"])
        records.append(
            {
                "condition": condition,
                "status": "diagnostic_only",
                "raw_mean_mpa": float(raw["mean_mpa"]),
                "raw_peak_to_peak_mpa": raw_p2p,
                "official_static_mean_mpa": float(row["static_mean_mpa"]),
                "official_dynamic_peak_to_peak_mpa": float(row["dynamic_peak_to_peak_mpa"]),
                "mean_difference_mpa": float(raw["mean_mpa"] - row["static_mean_mpa"]),
                "peak_to_peak_difference_mpa": float(
                    raw_p2p - row["dynamic_peak_to_peak_mpa"]
                ),
            }
        )
    return {
        "status": "diagnostic_only",
        "ai5_data_directory": str(source),
        "records": records,
        "raw_data_used_for_strict_scoring": False,
    }


def fit_dynamic_by_nozzle_and_opening(
    rows: Sequence[Mapping[str, Any]],
    *,
    hold_out_condition: str = "130MW",
) -> dict[str, Any]:
    """Fit independent linear models by nozzle count and mean needle opening."""
    train_rows = [
        row
        for row in rows
        if row["condition"] not in {"rated_noload", hold_out_condition}
    ]
    held_out = next(
        (row for row in rows if row["condition"] == hold_out_condition), None
    )
    if held_out is None:
        raise OfficialStressEvidenceError("held-out condition is not present in official table")
    models: dict[str, Any] = {}
    for nozzle_count in sorted({int(row["nozzle_count"]) for row in train_rows}):
        group = [row for row in train_rows if int(row["nozzle_count"]) == nozzle_count]
        x = np.asarray([float(row["mean_needle_opening_pct"]) for row in group])
        y = np.asarray([float(row["dynamic_peak_to_peak_mpa"]) for row in group])
        if len(group) < 2:
            models[str(nozzle_count)] = {
                "status": "blocked_insufficient_training_conditions",
                "training_conditions": [row["condition"] for row in group],
            }
            continue
        slope, intercept = np.polyfit(x, y, 1)
        models[str(nozzle_count)] = {
            "status": "fit",
            "feature": "mean_needle_opening_pct",
            "training_conditions": [row["condition"] for row in group],
            "training_power_mw": [row["power_mw"] for row in group],
            "slope_mpa_per_opening_pct": float(slope),
            "intercept_mpa": float(intercept),
            "fit_sha256": canonical_sha256(
                {
                    "feature": "mean_needle_opening_pct",
                    "nozzle_count": nozzle_count,
                    "training_conditions": [row["condition"] for row in group],
                    "x": x.tolist(),
                    "y": y.tolist(),
                }
            ),
        }
    held_model = models.get(str(held_out["nozzle_count"]))
    if not isinstance(held_model, Mapping) or held_model.get("status") != "fit":
        raise OfficialStressEvidenceError("held-out nozzle regime cannot be fitted")
    predicted = (
        float(held_model["slope_mpa_per_opening_pct"])
        * float(held_out["mean_needle_opening_pct"])
        + float(held_model["intercept_mpa"])
    )
    measured = float(held_out["dynamic_peak_to_peak_mpa"])
    relative_error = abs(predicted - measured) / measured
    return {
        "model_scope": "official_table_leave_one_condition_out_diagnostic",
            "held_out_condition": hold_out_condition,
            "held_out_not_in_training": hold_out_condition
        not in held_model["training_conditions"],
        "dynamic_definition": "peak_to_peak_mpa",
        "models_by_nozzle_count": models,
        "prediction": {
                "condition": hold_out_condition,
            "nozzle_count": int(held_out["nozzle_count"]),
            "mean_needle_opening_pct": float(held_out["mean_needle_opening_pct"]),
            "predicted_peak_to_peak_mpa": predicted,
            "official_peak_to_peak_mpa": measured,
            "relative_error": relative_error,
            "strict_relative_threshold": STRICT_RELATIVE_THRESHOLD,
            "strict_candidate_pass": relative_error <= STRICT_RELATIVE_THRESHOLD,
        },
        "whole_table_independent_validation": False,
        "qualification_note": (
            "The same official table supplies model training conditions; this is a "
            "leave-one-condition-out diagnostic, not independent competition validation."
        ),
    }


def build_130mw_leave_one_report(
    evidence: Mapping[str, Any],
    *,
    ai5_data_directory: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the isolated 130 MW result with formal eligibility kept honest."""
    rows = evidence.get("table_3_3_rows")
    if not isinstance(rows, Sequence):
        raise OfficialStressEvidenceError("official evidence has no table rows")
    model = fit_dynamic_by_nozzle_and_opening(rows)
    raw_diagnostic = raw_ai5_diagnostic(ai5_data_directory, rows)
    eligible = evidence.get("double_review_completed") is True
    candidate_pass = model["prediction"]["strict_candidate_pass"] is True
    report = {
        "version": "step30-130mw-leave-one-stress-diagnostic-v1",
        "status": "passed" if eligible and candidate_pass else "blocked",
        "strict_scoring_status": (
            "candidate_pass_pending_transcription_review"
            if candidate_pass and not eligible
            else ("passed" if candidate_pass else "failed")
        ),
        "official_table_evidence_sha256": evidence.get("evidence_sha256"),
        "official_table_double_review_completed": eligible,
        "model": model,
        "raw_ai5_diagnostic": raw_diagnostic,
        "formal_evidence_status": (
            "eligible_official_table_diagnostic"
            if eligible
            else "ineligible_pending_double_transcription_review"
        ),
        "official_accuracy_gate": False,
        "fem_gate": "blocked",
        "limitations": [
            "130 MW is excluded from the six-nozzle model fit.",
            "This is not whole-table blind validation because neighbouring official rows train the fit.",
            "Raw AI5 time-series extrema remain diagnostic-only.",
        ],
    }
    report["report_sha256"] = canonical_sha256(report)
    if output_path is not None:
        safe_json_dump(report, output_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build official-table stress evidence and 130 MW diagnostic")
    parser.add_argument("--official-pdf", type=Path, required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
    parser.add_argument("--output-130mw", type=Path, required=True)
    parser.add_argument("--ai5-data-directory", type=Path)
    parser.add_argument(
        "--review",
        action="append",
        default=[],
        metavar="REVIEWER:STATUS",
        help="Status is completed_match, pending, or mismatch.",
    )
    args = parser.parse_args(argv)
    reviews = []
    for value in args.review:
        reviewer, separator, status = value.partition(":")
        if not separator:
            parser.error("--review must use REVIEWER:STATUS")
        reviews.append({"reviewer_id": reviewer, "status": status})
    evidence = build_official_stress_evidence(
        args.official_pdf, transcription_reviews=reviews, output_path=args.output_evidence
    )
    leave_one = build_130mw_leave_one_report(
        evidence,
        ai5_data_directory=args.ai5_data_directory,
        output_path=args.output_130mw,
    )
    print(json.dumps({"evidence": evidence, "130mw": leave_one}, ensure_ascii=False, indent=2))
    return 0 if leave_one["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
