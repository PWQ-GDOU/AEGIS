"""Fail-closed comparison contract for predicted maps and measured profiles.

The current coupon protocol does not provide a turbine-face registration map,
so a coupon profile cannot be silently compared to CFD face IDs.  This harness
records the missing evidence and only computes a spatial comparison when a
measured profile and an explicit prediction/profile registration are supplied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from step30_material_morphology_metrics import morphology_from_file
from step30_validation_common import canonical_sha256, safe_json_dump, sha256_file


VERSION = "step30-material-profile-compare-v1"


def compare_profile(
    prediction_path: str | Path,
    *,
    measured_profile_path: str | Path | None = None,
    registration_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    prediction = Path(prediction_path).resolve()
    if not prediction.is_file():
        raise FileNotFoundError(prediction)
    report: dict[str, Any] = {
        "version": VERSION,
        "status": "blocked",
        "physical_spatial_accuracy_status": "blocked_without_registered_measured_profile",
        "official_accuracy_gate": False,
        "official_reference_gate": False,
        "fem_gate": "blocked",
        "prediction": {"path": str(prediction), "sha256": sha256_file(prediction)},
        "measured_profile": None,
        "registration": None,
        "comparison": None,
        "blocking_reasons": [],
    }
    if measured_profile_path is None:
        report["blocking_reasons"].append("measured 3-D wear profile was not supplied")
    else:
        measured = Path(measured_profile_path).resolve()
        if not measured.is_file():
            report["blocking_reasons"].append(f"measured profile does not exist: {measured}")
        else:
            report["measured_profile"] = {"path": str(measured), "sha256": sha256_file(measured)}
            try:
                report["measured_profile"]["morphology_metrics"] = morphology_from_file(measured)
            except Exception as exc:  # fail closed and preserve the reason in the evidence package
                report["blocking_reasons"].append(f"measured profile parsing failed: {exc}")
    if registration_path is None:
        report["blocking_reasons"].append("prediction-to-profile registration was not supplied")
    else:
        registration = Path(registration_path).resolve()
        if not registration.is_file():
            report["blocking_reasons"].append(f"registration artifact does not exist: {registration}")
        else:
            report["registration"] = {"path": str(registration), "sha256": sha256_file(registration)}
    if not report["blocking_reasons"]:
        report["status"] = "blocked_registration_contract_only"
        report["blocking_reasons"].append("registered face-level prediction-to-profile metric is not implemented for coupon-to-turbine transfer")
    report["report_sha256"] = canonical_sha256(report)
    if output_path is not None:
        safe_json_dump(report, Path(output_path).resolve())
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed predicted-map and measured-profile comparison")
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--measured-profile", type=Path)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare_profile(args.prediction, measured_profile_path=args.measured_profile, registration_path=args.registration, output_path=args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
