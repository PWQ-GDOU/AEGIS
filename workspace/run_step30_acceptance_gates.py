"""Aggregate Step30 numerical and measurement gates into one fail-closed report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from step30_official_reference_simulation import validate_official_reference
from step30_validation_common import canonical_sha256, safe_json_dump, safe_json_load, sha256_file


def _read(path: Path | None) -> dict | None:
    if path is None:
        return None
    if not path.is_file():
        return {"status": "blocked", "reason": f"missing input: {path}"}
    try:
        value = safe_json_load(path)
    except (OSError, UnicodeError, ValueError):
        return {"status": "blocked", "reason": f"invalid JSON input: {path}"}
    return value if isinstance(value, dict) else {"status": "blocked", "reason": f"not an object: {path}"}


def _report_hash_valid(report: Mapping[str, Any]) -> bool:
    stored = report.get("report_sha256")
    if not isinstance(stored, str) or len(stored) != 64:
        return False
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    return canonical_sha256(unsigned) == stored.lower()


def _reference_report_valid(
    report: Mapping[str, Any],
    *,
    trusted_manifest_sha256: str | None = None,
) -> bool:
    """Validate the content contract before accepting an official report."""
    required = {
        "version", "status", "numerical_status", "official_reference",
        "official_reference_gate", "source_type", "reference_role",
        "pipeline_validation", "reference_manifest_sha256", "artifact_set_id",
        "solver_semantics", "reference_solver_semantics", "units", "topology",
        "independent_split", "record_contract", "evaluation_contract",
        "record_provenance", "source_artifact_hashes",
        "prediction_records_sha256", "observation_records_sha256",
        "comparison", "trusted_manifest_sha256", "report_sha256",
    }
    if not required <= set(report) or report.get("version") != "step30-official-reference-simulation-v2":
        return False
    if report.get("status") != "passed" or report.get("numerical_status") != "passed":
        return False
    if report.get("official_reference") is not True or report.get("official_reference_gate") is not True:
        return False
    if report.get("source_type") != "contest_provided_simulation":
        return False
    if report.get("reference_role") != "official_reference_simulation" or report.get("pipeline_validation") is not False:
        return False
    manifest_digest = report.get("reference_manifest_sha256")
    if not isinstance(manifest_digest, str) or len(manifest_digest) != 64:
        return False
    if (
        trusted_manifest_sha256 is None
        or not isinstance(trusted_manifest_sha256, str)
        or len(trusted_manifest_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in trusted_manifest_sha256)
        or manifest_digest.lower() != trusted_manifest_sha256.lower()
        or report.get("trusted_manifest_sha256") != trusted_manifest_sha256.lower()
    ):
        return False
    if not isinstance(report.get("artifact_set_id"), str) or len(report["artifact_set_id"]) != 64:
        return False
    for field in ("prediction_records_sha256", "observation_records_sha256"):
        digest = report.get(field)
        if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
            return False
    solver = report.get("solver_semantics")
    if not isinstance(solver, Mapping) or solver.get("application") != "pimpleFoam" or solver.get("mode") != "urans" or solver.get("field_statistic") != "time_mean" or solver.get("tracking_velocity_object") != "UMean":
        return False
    if report.get("units") != {"source_coordinate_units": "m", "internal_coordinate_units": "mm", "coordinate_scale_to_mm": 1000.0}:
        return False
    topology = report.get("topology")
    if not isinstance(topology, Mapping) or topology.get("artifact_version") != "step30-native-openfoam-topology-v3" or not isinstance(topology.get("mesh_signature"), str) or len(topology["mesh_signature"]) != 64 or topology.get("n_parent_faces", 0) <= 0 or topology.get("n_collision_triangles", 0) <= 0:
        return False
    split = report.get("independent_split")
    if (
        not isinstance(split, Mapping)
        or split.get("is_independent") is not True
        or not split.get("record_ids")
        or not isinstance(split.get("split_artifact_sha256"), str)
        or not isinstance(split.get("reference_source_sha256"), str)
    ):
        return False
    record_contract = report.get("record_contract")
    evaluation = report.get("evaluation_contract")
    if not isinstance(record_contract, Mapping) or not isinstance(evaluation, Mapping):
        return False
    if (
        evaluation.get("version") != "step30-reference-evaluation-contract-v1"
        or evaluation.get("value_metric") != record_contract.get("value_metric")
        or evaluation.get("canonical_units") != record_contract.get("value_units")
        or evaluation.get("zero_reference_policy") != "block"
    ):
        return False
    source_hashes = report.get("source_artifact_hashes")
    required_roles = {
        "official_description", "official_evaluation_definition",
        "official_reference_raw", "official_independent_split",
        "official_reference_observations", "native_artifact_set_manifest",
        "topology_v3",
    }
    if not isinstance(source_hashes, Mapping) or set(source_hashes) != required_roles:
        return False
    if not all(isinstance(value, str) and len(value) == 64 for value in source_hashes.values()):
        return False
    comparison = report.get("comparison")
    if (
        not isinstance(comparison, Mapping)
        or comparison.get("under_thresholds") is not True
        or comparison.get("threshold_checks") is None
        or comparison.get("value_metric") != evaluation.get("value_metric")
        or comparison.get("value_units") != evaluation.get("canonical_units")
        or comparison.get("aggregation") != evaluation.get("aggregation")
    ):
        return False
    checks = comparison.get("threshold_checks")
    if not isinstance(checks, Mapping) or not checks or not all(value is True for value in checks.values()):
        return False
    top_k = comparison.get("top_k", [])
    if not isinstance(top_k, list) or not all(isinstance(item, Mapping) and item.get("passed") is True for item in top_k):
        return False
    if not _report_hash_valid(report):
        return False
    manifest_path = report.get("reference_manifest_path")
    prediction = report.get("prediction_artifact")
    observation = report.get("reference_observation_artifact")
    if not isinstance(manifest_path, str) or not isinstance(prediction, Mapping) or not isinstance(observation, Mapping):
        return False
    prediction_path = prediction.get("path")
    observation_path = observation.get("path")
    if not isinstance(prediction_path, str) or not isinstance(observation_path, str):
        return False
    paths = [Path(manifest_path), Path(prediction_path), Path(observation_path)]
    if not all(path.is_file() for path in paths):
        return False
    if prediction.get("sha256") != sha256_file(paths[1]) or observation.get("sha256") != sha256_file(paths[2]):
        return False
    recomputed = validate_official_reference(
        paths[1],
        paths[2],
        paths[0],
        trusted_manifest_sha256=trusted_manifest_sha256,
    )
    return recomputed == dict(report)


def _status(name: str, report: dict | None, *, required: bool = True) -> dict[str, Any]:
    if report is None:
        return {"status": "blocked", "required": required, "reason": "not supplied"}
    status = report.get("status")
    if status in {"passed", "PASS"} and report.get("accepted") is not False:
        return {"status": "passed", "required": required, "source": report}
    if status == "accepted" and report.get("accepted") is True:
        return {"status": "passed", "required": required, "source": report}
    if status == "planned":
        return {"status": "blocked", "required": required, "reason": "planned matrix has not been executed", "source": report}
    if status == "blocked":
        return {"status": "blocked", "required": required, "source": report}
    return {"status": "failed", "required": required, "source": report}


def aggregate(*, artifact: dict | None, collision: dict | None, cfd: dict | None, sensitivity: dict | None, comparison: dict | None, measurement: dict | None, reference_simulation: dict | None = None, trusted_manifest_sha256: str | None = None, production: bool) -> dict:
    checks = {
        "artifact_set": _status("artifact_set", artifact),
        "strict_collision": _status("strict_collision", collision),
        "cfd_mean_field": _status("cfd_mean_field", cfd),
        "timestep_sensitivity": _status("timestep_sensitivity", sensitivity),
        "i1_i3_comparison": _status("i1_i3_comparison", comparison),
        "measurement_validation": _status("measurement_validation", measurement, required=False),
        "reference_simulation": _status("reference_simulation", reference_simulation, required=False),
    }
    required = [value for value in checks.values() if value["required"]]
    failed = any(value["status"] == "failed" for value in required)
    blocked = any(value["status"] == "blocked" for value in required)
    measurement_status = checks["measurement_validation"]["status"]
    reference_status = checks["reference_simulation"]["status"]
    official_accuracy_gate = measurement_status == "passed" and bool(
        (measurement or {}).get("official_accuracy_gate", False)
    )
    official_reference_gate = reference_status == "passed" and _reference_report_valid(
        reference_simulation or {},
        trusted_manifest_sha256=trusted_manifest_sha256,
    )
    if reference_simulation is not None and not official_reference_gate:
        checks["reference_simulation"] = {
            "status": "blocked",
            "required": False,
            "reason": "official reference report failed content/source revalidation",
            "source": reference_simulation,
        }
        reference_status = "blocked"
    external_physical_validation_status = "blocked"
    if measurement is not None and (measurement or {}).get("reference_role") == "external_measured_calibration":
        external_physical_validation_status = "calibration_only"
    if measurement is not None and not official_accuracy_gate and (measurement or {}).get("reference_role") != "external_measured_calibration":
        checks["measurement_validation"] = {
            "status": "blocked",
            "required": False,
            "reason": "no independent measured accuracy gate; synthetic or non-independent evidence is ineligible",
            "source": measurement,
        }
        measurement_status = "blocked"
    numerical_status = "failed" if failed else ("blocked" if blocked else "passed")
    production_reference_gate = official_reference_gate or official_accuracy_gate
    if production:
        if numerical_status == "failed":
            status = "failed"
        elif numerical_status == "blocked" or not production_reference_gate:
            status = "blocked"
        else:
            status = "passed"
    else:
        status = numerical_status
    result = {
        "version": "step30-acceptance-gates-v2",
        "status": status,
        "numerical_status": numerical_status,
        "production": production,
        "checks": checks,
        "official_accuracy_gate": official_accuracy_gate,
        "official_reference_gate": official_reference_gate,
        "external_physical_validation_status": external_physical_validation_status,
        "official_accuracy_note": "Measured physical accuracy requires independent measured validation; contest reference accuracy requires authenticated contest simulation provenance and is reported separately.",
    }
    result["acceptance_sha256"] = canonical_sha256(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--collision", type=Path)
    parser.add_argument("--cfd", type=Path)
    parser.add_argument("--sensitivity", type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--measurement", type=Path)
    parser.add_argument("--reference-simulation", type=Path)
    parser.add_argument(
        "--trusted-reference-manifest-sha256",
        help="out-of-band SHA-256 pin for the organizer reference manifest",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--production", action="store_true")
    args = parser.parse_args()
    report = aggregate(
        artifact=_read(args.artifact), collision=_read(args.collision), cfd=_read(args.cfd),
        sensitivity=_read(args.sensitivity), comparison=_read(args.comparison), measurement=_read(args.measurement),
        reference_simulation=_read(args.reference_simulation),
        trusted_manifest_sha256=args.trusted_reference_manifest_sha256,
        production=args.production,
    )
    safe_json_dump(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
