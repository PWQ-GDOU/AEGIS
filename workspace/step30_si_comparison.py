"""Compare immutable native-SI Step-30 i1/i3 runs with fail-closed identity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from step30_artifact_identity import (
    ARTIFACT_DESCRIPTOR_VERSION,
    PARTICLE_DESCRIPTOR_VERSION,
    canonical_sha256,
    file_sha256,
    validate_i1_i3_descriptors,
)

COMPARISON_VERSION = "step30-native-si-comparison-v3"
PROVENANCE_VERSION = "si-unit-provenance-v3"
SUMMARY_VERSION = "step30-cfd-driven-v2"
MANIFEST_VERSION = "step30-native-si-manifest-v2"
TOPOLOGY_VERSION = "step30-native-openfoam-topology-v3"
STRICT_POLICY = "strict_v2"
EXPECTED_UNITS = {
    "source_coordinate_units": "m",
    "internal_coordinate_units": "mm",
    "coordinate_scale_to_mm": 1000.0,
}
EXPECTED_OUTPUTS = {"particle_hits", "summary", "validation"}


def _load_json(path: Path, label: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return data


def _descriptor_from_run_manifest(run_manifest: dict) -> dict:
    descriptor = dict(run_manifest)
    descriptor.pop("outputs", None)
    return descriptor


def _validate_particle_descriptor(descriptor: dict) -> None:
    expected_keys = {
        "version",
        "artifact_set_id",
        "collision_backend",
        "implementation_hashes",
        "input_hashes",
        "config",
        "particle_run_id",
    }
    if set(descriptor) != expected_keys:
        raise ValueError("particle run descriptor has incomplete or unexpected fields")
    if descriptor.get("version") != PARTICLE_DESCRIPTOR_VERSION:
        raise ValueError("unsupported particle run descriptor version")
    if not isinstance(descriptor.get("artifact_set_id"), str) or not descriptor["artifact_set_id"]:
        raise ValueError("particle run descriptor has no artifact_set_id")
    implementation_hashes = descriptor.get("implementation_hashes")
    input_hashes = descriptor.get("input_hashes")
    if not isinstance(implementation_hashes, dict) or not implementation_hashes or not all(implementation_hashes.values()):
        raise ValueError("particle run descriptor has invalid implementation hashes")
    if not isinstance(input_hashes, dict) or not all(input_hashes.values()):
        raise ValueError("particle run descriptor has invalid input hashes")
    particle_run_id = descriptor.get("particle_run_id")
    unhashed = dict(descriptor)
    unhashed.pop("particle_run_id", None)
    if not isinstance(particle_run_id, str) or canonical_sha256(unhashed) != particle_run_id:
        raise ValueError("particle run descriptor ID/content mismatch")
    config = descriptor.get("config")
    if not isinstance(config, dict):
        raise ValueError("particle run descriptor has no config object")
    if descriptor.get("collision_backend") != "cfd_topology":
        raise ValueError("immutable SI comparison requires the cfd_topology collision backend")
    if config.get("collision_policy") != STRICT_POLICY:
        raise ValueError("immutable SI comparison requires strict_v2 collision policy")
    if config.get("cfd_variant") != "native_si":
        raise ValueError("immutable SI comparison requires native_si CFD")


def _validate_output_hashes(run_manifest_path: Path, run_manifest: dict) -> dict[str, Path]:
    outputs = run_manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != EXPECTED_OUTPUTS:
        raise ValueError("particle run manifest has incomplete or unexpected outputs")
    paths = {}
    for name in sorted(EXPECTED_OUTPUTS):
        entry = outputs[name]
        if not isinstance(entry, dict) or set(entry) != {"file", "sha256"}:
            raise ValueError(f"particle run output entry is invalid: {name}")
        file_name = entry.get("file")
        if not file_name or Path(file_name).name != file_name:
            raise ValueError(f"particle run output path is invalid: {name}")
        path = run_manifest_path.parent / file_name
        if not path.is_file():
            raise FileNotFoundError(f"particle run output is missing: {path}")
        if file_sha256(path) != entry.get("sha256"):
            raise ValueError(f"particle run output hash mismatch: {name}")
        paths[name] = path
    return paths


def _first_impacts(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    required = {
        "parcel_id", "impact_index", "parent_face_id", "collision_triangle_id",
        "time_s", "position_local_mm", "velocity_ms", "Vp_ms", "alpha_deg",
    }
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"particle hit artifact is missing first-impact arrays: {missing}")
    impact_count = len(data["parcel_id"])
    if any(len(data[name]) != impact_count for name in required):
        raise ValueError("particle hit arrays have inconsistent lengths")
    mask = data["impact_index"] == 0
    if not np.any(mask):
        raise ValueError("i1/i3 comparison requires physical first impacts")
    parcel_ids = data["parcel_id"][mask].astype(np.int64)
    if len(np.unique(parcel_ids)) != len(parcel_ids):
        raise ValueError("particle hit artifact contains duplicate first impacts")
    order = np.argsort(parcel_ids)
    return {name: data[name][mask][order] for name in required}


def _validate_matching_first_impacts(i1_path: Path, i3_path: Path) -> dict:
    i1 = _first_impacts(i1_path)
    i3 = _first_impacts(i3_path)
    for name in ("parcel_id", "parent_face_id", "collision_triangle_id"):
        if not np.array_equal(i1[name], i3[name]):
            raise ValueError(f"i1/i3 first impacts differ for {name}")
    for name in ("time_s", "position_local_mm", "velocity_ms", "Vp_ms", "alpha_deg"):
        if not np.allclose(i1[name], i3[name], rtol=1e-12, atol=1e-12):
            raise ValueError(f"i1/i3 first impacts differ for {name}")
    return {
        "matching_first_impact_parcels": int(len(i1["parcel_id"])),
        "parent_face_agreement": 1.0,
        "collision_triangle_agreement": 1.0,
        "max_position_delta_mm": float(
            np.linalg.norm(i1["position_local_mm"] - i3["position_local_mm"], axis=1).max()
        ),
    }


def _artifact_manifest_path(run_manifest_path: Path, descriptor: dict) -> Path | None:
    run_dir = run_manifest_path.parent
    config = descriptor["config"]
    if (
        run_dir.name == descriptor["particle_run_id"]
        and run_dir.parent.name == config["collision_policy"]
        and run_dir.parent.parent.name == descriptor["artifact_set_id"]
    ):
        return run_dir.parent.parent / "artifact_set_manifest.json"
    return None


def _validate_artifact_descriptor(artifact: dict) -> None:
    if artifact.get("version") != ARTIFACT_DESCRIPTOR_VERSION:
        raise ValueError("unsupported artifact-set descriptor version")
    artifact_set_id = artifact.get("artifact_set_id")
    unhashed = dict(artifact)
    unhashed.pop("artifact_set_id", None)
    if not isinstance(artifact_set_id, str) or canonical_sha256(unhashed) != artifact_set_id:
        raise ValueError("artifact-set descriptor ID/content mismatch")
    cfd = artifact.get("cfd")
    topology = artifact.get("topology")
    if not isinstance(cfd, dict) or not isinstance(topology, dict):
        raise ValueError("artifact-set descriptor is incomplete")
    if cfd.get("variant") != "native_si" or cfd.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("artifact set is not native_si manifest v2 data")
    if topology.get("artifact_version") != TOPOLOGY_VERSION:
        raise ValueError("artifact set does not use topology v3")
    if artifact.get("units") != EXPECTED_UNITS:
        raise ValueError("artifact set has an invalid SI unit contract")
    _validate_semantics(cfd.get("solver_semantics"))


def _load_identifiable_artifact_manifest(
    run_manifest_path: Path,
    descriptor: dict,
) -> tuple[dict | None, Path | None]:
    path = _artifact_manifest_path(run_manifest_path, descriptor)
    if path is None:
        return None, None
    if not path.is_file():
        raise FileNotFoundError(f"identifiable artifact-set manifest is missing: {path}")
    artifact = _load_json(path, "artifact-set manifest")
    _validate_artifact_descriptor(artifact)
    if artifact["artifact_set_id"] != descriptor["artifact_set_id"]:
        raise ValueError("particle and artifact-set descriptors use different artifact_set_id values")
    return artifact, path


def _validate_semantics(semantics: dict) -> str:
    required = {
        "application",
        "mode",
        "field_statistic",
        "tracking_velocity_object",
        "averaging_window",
    }
    if not isinstance(semantics, dict) or set(semantics) != required:
        raise ValueError("solver semantics are incomplete or unexpected")
    statistic = semantics["field_statistic"]
    if statistic == "snapshot":
        expected = {
            "application": "simpleFoam",
            "mode": "steady_rans",
            "field_statistic": "snapshot",
            "tracking_velocity_object": "U",
            "averaging_window": None,
        }
        if semantics != expected:
            raise ValueError("snapshot semantics require simpleFoam/steady_rans/snapshot/U/None")
    elif statistic == "time_mean":
        window = semantics["averaging_window"]
        if (
            semantics["application"] != "pimpleFoam"
            or semantics["mode"] != "urans"
            or semantics["tracking_velocity_object"] != "UMean"
            or not isinstance(window, dict)
            or set(window) != {"start_time", "end_time", "sample_count"}
            or not isinstance(window["sample_count"], int)
            or isinstance(window["sample_count"], bool)
            or window["sample_count"] <= 0
        ):
            raise ValueError("time-mean semantics require pimpleFoam/urans/time_mean/UMean and a valid window")
    else:
        raise ValueError("field statistic must be snapshot or time_mean")
    return statistic


def _validate_summary(
    summary: dict,
    descriptor: dict,
    artifact: dict | None,
    provenance: dict,
    expected_max_impacts: int,
) -> None:
    if summary.get("version") != SUMMARY_VERSION:
        raise ValueError("immutable SI comparison requires Step30 summary v2")
    identity = summary.get("artifact_identity")
    if not isinstance(identity, dict):
        raise ValueError("Step30 summary has no artifact identity")
    if identity.get("artifact_set_id") != descriptor["artifact_set_id"]:
        raise ValueError("summary artifact_set_id differs from its run manifest")
    if identity.get("particle_run_id") != descriptor["particle_run_id"]:
        raise ValueError("summary particle_run_id differs from its run manifest")

    config = descriptor["config"]
    parameters = summary.get("parameters", {})
    cfd = summary.get("cfd", {})
    geometry = summary.get("geometry", {})
    tracking = summary.get("tracking", {})
    if descriptor.get("artifact_set_id") is None or descriptor.get("particle_run_id") is None:
        raise ValueError("particle run descriptor lacks immutable IDs")
    if config.get("max_impacts") != expected_max_impacts or parameters.get("max_impacts") != expected_max_impacts:
        raise ValueError("SI comparison requires immutable i1 and i3 runs")
    if parameters.get("collision_policy") != STRICT_POLICY or config.get("collision_policy") != STRICT_POLICY:
        raise ValueError("SI comparison received a mixed or non-strict collision policy")
    if parameters.get("collision_backend") != "cfd_topology":
        raise ValueError("SI comparison requires cfd_topology particle runs")
    if cfd.get("variant") != "native_si" or parameters.get("cfd_variant") != "native_si":
        raise ValueError("SI comparison received a non-native_si run")
    if parameters.get("n_parcels") != config.get("n_parcels"):
        raise ValueError("summary and particle descriptor parcel counts differ")
    if cfd.get("artifact_manifest_sha256") != provenance["artifact_manifest"].get("sha256"):
        raise ValueError("particle run and provenance native manifest hashes differ")
    if tracking.get("fallback_hits") != 0:
        raise ValueError("SI comparison requires zero fallback hits")
    if tracking.get("physical_strict_hit_ratio") != 1.0:
        raise ValueError("SI comparison requires physical_strict_hit_ratio=1")
    if tracking.get("collision_event_limit_count", 0) != 0:
        raise ValueError("SI comparison requires zero collision-event-limit terminations")
    if summary.get("units", {}).get("source_coordinate_units") != "m" or (
        summary.get("units", {}).get("internal_coordinate_units"),
        summary.get("units", {}).get("coordinate_scale_to_mm"),
    ) != ("mm", 1000.0):
        raise ValueError("SI comparison received an invalid unit contract")
    if (
        cfd.get("source_coordinate_units"),
        cfd.get("internal_coordinate_units"),
        cfd.get("coordinate_scale_to_mm"),
    ) != ("m", "mm", 1000.0):
        raise ValueError("SI comparison received invalid CFD coordinate units")

    semantics = cfd.get("solver_semantics")
    _validate_semantics(semantics)
    if semantics != provenance.get("solver_semantics"):
        raise ValueError("particle run and provenance solver semantics/statistics differ")
    field_time = str(provenance["source_time"])
    if (
        str(cfd.get("field_time")) != field_time
        or str(geometry.get("source_time")) != field_time
        or str(geometry.get("field_time")) != field_time
        or str(config.get("cfd_field_time")) != field_time
    ):
        raise ValueError("SI comparison field/topology/provenance times differ")
    if geometry.get("mesh_signature") != provenance["topology"]["mesh_signature"]:
        raise ValueError("SI comparison topology differs from provenance")

    if artifact is not None:
        artifact_cfd = artifact["cfd"]
        artifact_topology = artifact["topology"]
        if semantics != artifact_cfd.get("solver_semantics"):
            raise ValueError("summary and artifact-set solver semantics differ")
        if str(artifact_cfd.get("field_time")) != field_time:
            raise ValueError("artifact-set and provenance field times differ")
        if artifact_cfd.get("manifest_sha256") != provenance["artifact_manifest"].get("sha256"):
            raise ValueError("artifact-set and provenance native manifest hashes differ")
        if artifact_topology.get("file_sha256") != provenance["topology"].get("artifact_sha256"):
            raise ValueError("artifact-set and provenance topology hashes differ")
        if artifact_topology.get("mesh_signature") != geometry.get("mesh_signature"):
            raise ValueError("artifact-set and summary mesh signatures differ")


def _metrics(data: dict) -> dict:
    tracking = data["tracking"]
    erosion = data["face_erosion"]
    validation = data["validation"]
    return {
        "field_time": data["cfd"]["field_time"],
        "max_impacts": data["parameters"]["max_impacts"],
        "n_parcels": data["parameters"]["n_parcels"],
        "impacts": tracking["n_impacts"],
        "hit_rate_pct": tracking["hit_rate_parcels_pct"],
        "strict_intersections": tracking["intersection_hits"],
        "fallback_hits": tracking["fallback_hits"],
        "physical_strict_hit_ratio": tracking["physical_strict_hit_ratio"],
        "impact_mass_kg": tracking["impact_mass_kg"],
        "total_target_mass_kg": erosion["total_target_mass_kg"],
        "max_dhdt_mm_h": erosion["dhdt_max_mm_h"],
        "spearman_er_vs_analytic": validation["cfd_ER_vs_analytic_Mansouri"]["spearman_rho"],
        "top10_iou": validation["cfd_ER_vs_analytic_Mansouri"]["top10_iou"],
    }


def _solver_quality_gate(provenance: dict) -> dict:
    semantics = provenance["solver_semantics"]
    statistic = _validate_semantics(semantics)
    assessment = provenance.get("convergence", {})
    if statistic == "snapshot":
        if assessment.get("assessment_type") != "steady_snapshot_convergence":
            raise ValueError("snapshot provenance has the wrong convergence assessment")
        if assessment.get("strict_residual_control_met") is not True:
            raise ValueError("steady snapshot provenance has rejected solver evidence")
        return {
            "field_statistic": statistic,
            "assessment_type": assessment["assessment_type"],
            "strict_residual_control_met": True,
            "previous_snapshot_field_change_available": assessment.get("previous_snapshot_field_change") is not None,
        }
    window = semantics["averaging_window"]
    validation = assessment.get("validation_report", {})
    validation_gates = validation.get("gates")
    if (
        assessment.get("assessment_type") != "urans_time_mean_sampling"
        or assessment.get("averaging_window") != window
        or assessment.get("simple_convergence_marker_applied") is not False
        or not isinstance(validation.get("sha256"), str)
        or not isinstance(validation_gates, dict)
        or not validation_gates
        or not all(validation_gates.values())
    ):
        raise ValueError("time-mean provenance has invalid or rejected sampling evidence")
    return {
        "field_statistic": statistic,
        "assessment_type": assessment["assessment_type"],
        "averaging_window": window,
        "sample_count_positive": int(window["sample_count"]) > 0,
        "validation_report_gates": validation_gates,
        "previous_mean_field_change_available": assessment.get("previous_mean_field_change") is not None,
        "simple_convergence_marker_applied": False,
    }


def _legacy_reference(path: Path | None) -> dict:
    if path is None:
        return {"status": "not_requested"}
    legacy = _load_json(path, "legacy reference summary")
    return {
        "status": "isolated_reference_only",
        "source": str(path),
        "summary_version": legacy.get("version"),
        "comparison_policy": "Excluded from SI identity checks and absolute physical comparison.",
        "metrics": _metrics(legacy),
    }


def run(
    i1_run_manifest_path: Path | str,
    i3_run_manifest_path: Path | str,
    provenance_path: Path | str,
    output_path: Path | str,
    *,
    legacy_reference_path: Path | str | None = None,
) -> dict:
    """Validate and compare one immutable strict-v2 native-SI i1/i3 pair."""
    i1_run_manifest_path = Path(i1_run_manifest_path)
    i3_run_manifest_path = Path(i3_run_manifest_path)
    provenance_path = Path(provenance_path)
    output_path = Path(output_path)
    legacy_reference_path = Path(legacy_reference_path) if legacy_reference_path is not None else None

    provenance = _load_json(provenance_path, "SI provenance")
    if provenance.get("version") != PROVENANCE_VERSION:
        raise ValueError("SI comparison requires provenance v3")
    _validate_semantics(provenance.get("solver_semantics"))
    if provenance.get("artifact_manifest", {}).get("version") != MANIFEST_VERSION:
        raise ValueError("SI comparison provenance does not identify native manifest v2")
    if provenance.get("topology", {}).get("artifact_version") != TOPOLOGY_VERSION:
        raise ValueError("SI comparison provenance does not identify topology v3")

    i1_manifest = _load_json(i1_run_manifest_path, "i1 particle run manifest")
    i3_manifest = _load_json(i3_run_manifest_path, "i3 particle run manifest")
    i1_descriptor = _descriptor_from_run_manifest(i1_manifest)
    i3_descriptor = _descriptor_from_run_manifest(i3_manifest)
    _validate_particle_descriptor(i1_descriptor)
    _validate_particle_descriptor(i3_descriptor)
    validate_i1_i3_descriptors(i1_descriptor, i3_descriptor)
    if i1_descriptor["artifact_set_id"] != i3_descriptor["artifact_set_id"]:
        raise ValueError("i1/i3 artifact_set_id values differ")

    i1_outputs = _validate_output_hashes(i1_run_manifest_path, i1_manifest)
    i3_outputs = _validate_output_hashes(i3_run_manifest_path, i3_manifest)
    first_impact_match = _validate_matching_first_impacts(
        i1_outputs["particle_hits"], i3_outputs["particle_hits"]
    )
    i1_summary = _load_json(i1_outputs["summary"], "i1 summary")
    i3_summary = _load_json(i3_outputs["summary"], "i3 summary")

    i1_artifact, i1_artifact_path = _load_identifiable_artifact_manifest(
        i1_run_manifest_path, i1_descriptor
    )
    i3_artifact, i3_artifact_path = _load_identifiable_artifact_manifest(
        i3_run_manifest_path, i3_descriptor
    )
    if i1_artifact is None or i3_artifact is None:
        raise ValueError("immutable comparison requires canonical artifact-set manifests")
    if i1_artifact != i3_artifact:
        raise ValueError("i1/i3 artifact-set manifest content differs")
    artifact = i1_artifact

    _validate_summary(i1_summary, i1_descriptor, artifact, provenance, 1)
    _validate_summary(i3_summary, i3_descriptor, artifact, provenance, 3)
    if i1_summary["cfd"]["solver_semantics"] != i3_summary["cfd"]["solver_semantics"]:
        raise ValueError("i1/i3 solver semantics/statistics differ")
    if i1_summary["geometry"]["mesh_signature"] != i3_summary["geometry"]["mesh_signature"]:
        raise ValueError("i1/i3 mesh signatures differ")
    if (
        i1_summary["validation"].get("registration_status") != "passed"
        or i3_summary["validation"].get("registration_status") != "passed"
    ):
        raise ValueError("i1/i3 geometry registration must pass")
    if max(
        i1_summary["cfd"]["speed_stats_ms"]["max"],
        i3_summary["cfd"]["speed_stats_ms"]["max"],
    ) >= 500.0:
        raise ValueError("i1/i3 CFD speed must remain below 500 m/s")

    semantics = provenance["solver_semantics"]
    solver_gate = _solver_quality_gate(provenance)
    quality_gates = {
        "zero_fallback_hits": (
            i1_summary["tracking"]["fallback_hits"] == 0
            and i3_summary["tracking"]["fallback_hits"] == 0
        ),
        "physical_strict_hit_ratio_is_one": (
            i1_summary["tracking"]["physical_strict_hit_ratio"] == 1.0
            and i3_summary["tracking"]["physical_strict_hit_ratio"] == 1.0
        ),
        "zero_collision_event_limit_terminations": True,
        "strict_v2_policy": True,
        "native_si": True,
        "same_immutable_artifact_set": True,
        "output_hashes_verified": True,
        "i1_i3_first_impacts_identical": True,
        "max_cfd_speed_below_500_ms": i3_summary["cfd"]["speed_stats_ms"]["max"] < 500.0,
        "registration_passed_i1": True,
        "registration_passed_i3": True,
        "solver_evidence": solver_gate,
    }
    result = {
        "version": COMPARISON_VERSION,
        "accepted": True,
        "artifact_identity": {
            "artifact_set_id": i1_descriptor["artifact_set_id"],
            "artifact_set_manifest": (
                str(i1_artifact_path or i3_artifact_path)
                if (i1_artifact_path or i3_artifact_path) is not None
                else None
            ),
            "artifact_set_manifest_verified": artifact is not None,
            "i1_particle_run_id": i1_descriptor["particle_run_id"],
            "i3_particle_run_id": i3_descriptor["particle_run_id"],
        },
        "final_cfd": {
            "variant": "native_si",
            "field_time": str(provenance["source_time"]),
            "solver_semantics": semantics,
            "n_cells": i3_summary["cfd"]["n_cells"],
            "source_coordinate_units": "m",
            "internal_coordinate_units": "mm",
            "coordinate_scale_to_mm": 1000.0,
            "speed_stats_ms": i3_summary["cfd"]["speed_stats_ms"],
            "solver_assessment": provenance["convergence"],
        },
        "topology": {
            "artifact_version": provenance["topology"]["artifact_version"],
            "mesh_signature": i3_summary["geometry"]["mesh_signature"],
            "n_parent_faces": i3_summary["geometry"]["n_parent_faces"],
            "n_collision_triangles": i3_summary["geometry"]["n_collision_triangles"],
            "surface_area_m2": i3_summary["geometry"]["surface_area_m2"],
            "bounds_mm": i3_summary["geometry"]["bounds_mm"],
        },
        "runs": {
            "strict_native_si_i1": _metrics(i1_summary),
            "strict_native_si_i3": _metrics(i3_summary),
        },
        "first_impact_match": first_impact_match,
        "quality_gates": quality_gates,
        "legacy_reference": _legacy_reference(legacy_reference_path),
        "interpretation": (
            "The compared runs use the same immutable native-SI CFD/topology artifact and strict swept "
            "intersections. Solver evidence is interpreted according to steady-snapshot or URANS time-mean semantics."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare immutable validated SI Step30 runs")
    parser.add_argument("--i1-run-manifest", type=Path, required=True)
    parser.add_argument("--i3-run-manifest", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-reference", type=Path)
    args = parser.parse_args()
    run(
        args.i1_run_manifest,
        args.i3_run_manifest,
        args.provenance,
        args.output,
        legacy_reference_path=args.legacy_reference,
    )
