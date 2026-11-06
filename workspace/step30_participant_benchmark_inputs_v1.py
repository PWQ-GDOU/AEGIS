"""Build auditable participant-built benchmark inputs on the stable face domain."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from step30_material_model_calibration import (
    finnie_volume_mm3_per_kg,
    mansouri_mass_ratio_kg_per_kg,
    oka_volume_mm3_per_kg,
    tabakoff_volume_mm3_per_kg,
)
from step30_validation_common import canonical_sha256, safe_json_dump, safe_npz_dump, sha256_file

VERSION = "step30-participant-benchmark-inputs-v1"
MODELS = ("oka", "finnie", "tabakoff", "mansouri")


class BenchmarkInputError(ValueError):
    pass


def _load_pair(hit_path: Path, field_path: Path) -> dict[str, Any]:
    if not hit_path.is_file() or not field_path.is_file():
        raise BenchmarkInputError(f"missing pair: {hit_path} / {field_path}")
    with np.load(hit_path, allow_pickle=False) as hit, np.load(field_path, allow_pickle=False) as field:
        required_hit = {"parent_face_id", "velocity_ms", "Vp_ms", "alpha_deg", "diameter_m", "parcel_mass_kg", "face_centers_local_mm", "face_area_m2"}
        missing = sorted(required_hit - set(hit.files))
        if missing:
            raise BenchmarkInputError(f"hit artifact lacks {missing}: {hit_path}")
        required_field = {"aggregate_id", "aggregate_area_m2", "face_area_m2", "surface_aggregate_mass_per_area_kg_m2"}
        missing = sorted(required_field - set(field.files))
        if missing:
            raise BenchmarkInputError(f"field artifact lacks {missing}: {field_path}")
        centers = np.asarray(hit["face_centers_local_mm"], dtype=np.float64)
        areas = np.asarray(hit["face_area_m2"], dtype=np.float64)
        field_areas = np.asarray(field["face_area_m2"], dtype=np.float64)
        aggregate_id = np.asarray(field["aggregate_id"], dtype=np.int64)
        aggregate_area = np.asarray(field["aggregate_area_m2"], dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) != len(areas):
            raise BenchmarkInputError("face domain has invalid shape")
        if not np.array_equal(areas, field_areas) or len(aggregate_id) != len(areas):
            raise BenchmarkInputError("hit and topology field domains differ")
        if np.any(~np.isfinite(areas)) or np.any(areas <= 0.0):
            raise BenchmarkInputError("face areas must be finite and positive")
        parent = np.asarray(hit["parent_face_id"], dtype=np.int64)
        if parent.ndim != 1 or np.any(parent < 0) or np.any(parent >= len(areas)):
            raise BenchmarkInputError("parent_face_id is outside the face domain")
        velocity = np.asarray(hit["velocity_ms"], dtype=np.float64)
        speed = np.linalg.norm(velocity, axis=1)
        speed = np.where(speed > 1.0e-9, speed, np.asarray(hit["Vp_ms"], dtype=np.float64))
        angle = np.clip(np.asarray(hit["alpha_deg"], dtype=np.float64), 0.1, 89.9)
        d50_um = np.asarray(hit["diameter_m"], dtype=np.float64) * 1.0e6
        mass = np.asarray(hit["parcel_mass_kg"], dtype=np.float64)
        if not (len(parent) == len(speed) == len(angle) == len(d50_um) == len(mass)):
            raise BenchmarkInputError("hit arrays have inconsistent lengths")
        if np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
            raise BenchmarkInputError("parcel masses must be finite and positive")
        return {"centers": centers, "areas": areas, "parent": parent, "speed": speed, "angle": angle, "d50_um": d50_um, "mass": mass, "aggregate_id": aggregate_id, "aggregate_area": aggregate_area, "support": np.asarray(field["surface_aggregate_mass_per_area_kg_m2"], dtype=np.float64), "hit_sha256": sha256_file(hit_path), "field_sha256": sha256_file(field_path), "hit_path": str(hit_path.resolve()), "field_path": str(field_path.resolve())}


def _map_for_model(data: dict[str, Any], model: str) -> np.ndarray:
    speed, angle, d50_um = data["speed"], data["angle"], data["d50_um"]
    if model == "oka":
        removed_volume = np.asarray(oka_volume_mm3_per_kg(speed, angle, d50_um, hardness_gpa=4.0), dtype=np.float64) * data["mass"]
    elif model == "finnie":
        removed_volume = np.asarray(finnie_volume_mm3_per_kg(speed, angle), dtype=np.float64) * data["mass"]
    elif model == "tabakoff":
        removed_volume = np.asarray(tabakoff_volume_mm3_per_kg(speed, angle), dtype=np.float64) * data["mass"]
    elif model == "mansouri":
        removed_volume = np.asarray(mansouri_mass_ratio_kg_per_kg(speed, angle), dtype=np.float64) * data["mass"] / 7800.0 * 1.0e9
    else:
        raise BenchmarkInputError(f"unsupported model: {model}")
    if np.any(~np.isfinite(removed_volume)) or np.any(removed_volume < 0.0):
        raise BenchmarkInputError(f"{model} produced invalid volume")
    aggregate_id = data["aggregate_id"]
    n_aggregates = int(np.max(aggregate_id)) + 1
    aggregate_volume = np.bincount(aggregate_id[data["parent"]], weights=removed_volume, minlength=n_aggregates)
    depth = aggregate_volume[aggregate_id] / (data["aggregate_area"] * 1.0e6)[aggregate_id]
    return np.asarray(depth, dtype=np.float32)


def _write_prior(path: Path, data: dict[str, Any], maps: dict[str, np.ndarray], *, role: str, index: int) -> dict[str, Any]:
    report = {"version": VERSION, "status": "provisional", "prior_contract": {"ai5_used_for_calibration": False, "official_material_certificate": False, "official_reference": False, "reference_role": role, "topology_aggregation": "stable_surface_aggregate_v5", "depth_units": "mm_per_h_prior_rate_scale"}, "source": {"source_type": "contest_provided_simulation", "participant_built_reference": True, "run_index": index, "hit_path": data["hit_path"], "hit_sha256": data["hit_sha256"], "field_path": data["field_path"], "field_sha256": data["field_sha256"]}, "domain": {"n_faces": int(len(data["areas"])), "centers_sha256": canonical_sha256(data["centers"].tolist()), "areas_sha256": canonical_sha256(data["areas"].tolist())}, "model_maps": {name: {"field": f"{name}_p50_mm_h", "construction": "hit-level erosion law then topology aggregate area redistribution"} for name in MODELS}, "independence": {"model_evaluation": "leave_one_model_out_in_consumer", "seed_evaluation": "leave_one_seed_out_in_consumer", "measured_truth": False}}
    report["report_sha256"] = canonical_sha256(report)
    arrays = {"face_centers_local_mm": data["centers"].astype(np.float32), "face_area_m2": data["areas"].astype(np.float32)}
    arrays.update({f"{name}_p50_mm_h": maps[name] for name in MODELS})
    safe_json_dump(report, path)
    safe_npz_dump(arrays, path.with_suffix(".npz"), metadata={"report_sha256": report["report_sha256"], "version": VERSION})
    return report


def build_inputs(pairs: Sequence[tuple[str | Path, str | Path]], output_dir: str | Path) -> dict[str, Any]:
    if len(pairs) < 4:
        raise BenchmarkInputError("at least four independent pairs are required")
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    loaded = [_load_pair(Path(hit), Path(field)) for hit, field in pairs]
    first = loaded[0]
    for item in loaded[1:]:
        if not np.array_equal(item["centers"], first["centers"]) or not np.array_equal(item["areas"], first["areas"]):
            raise BenchmarkInputError("all pairs must share identical face centers and areas")
    seed_maps = []
    for index, item in enumerate(loaded):
        maps = {model: _map_for_model(item, model) for model in MODELS}
        seed_maps.append(maps)
        _write_prior(target / f"seed_{index:02d}.json", item, maps, role="participant_seed_prior", index=index)
    merged_maps = {model: np.mean(np.stack([maps[model] for maps in seed_maps], axis=0), axis=0).astype(np.float32) for model in MODELS}
    merged_data = dict(first)
    merged_data["hit_path"] = "participant-built mean of independent seed priors"
    merged_data["field_path"] = "participant-built mean of stable topology fields"
    merged_data["hit_sha256"] = canonical_sha256([item["hit_sha256"] for item in loaded])
    merged_data["field_sha256"] = canonical_sha256([item["field_sha256"] for item in loaded])
    _write_prior(target / "merged.json", merged_data, merged_maps, role="participant_merged_prior", index=len(loaded))
    support = np.mean(np.stack([item["support"] for item in loaded], axis=0), axis=0)
    hit_counts = np.zeros(len(first["areas"]), dtype=np.float64)
    for item in loaded:
        hit_counts += np.bincount(item["parent"], minlength=len(first["areas"]))
    region = np.maximum(support, 0.0)
    if not np.any(region > 0.0):
        region = hit_counts
    if not np.any(region > 0.0):
        raise BenchmarkInputError("participant DEC activity prior has no positive support")
    region /= float(np.max(region))
    region_path = target / "participant_dec_region_field.npz"
    safe_npz_dump({"literature_region_prior": region.astype(np.float32), "active_face_mask": (region > 0.0).astype(np.uint8)}, region_path, metadata={"version": VERSION, "source": "participant_particle_activity_union"})
    literature = {"version": VERSION, "reference_kind": "literature_reference", "official_reference_gate": False, "measured_truth": False, "participant_built": True, "regions": [{"region_id": "particle_supported_DEC_activity", "literature_support_score": 1.0, "evidence_level": "qualitative_prior", "quantitative_anchor": False}, {"region_id": "unresolved_nonimpact_surface", "literature_support_score": 0.05, "evidence_level": "uncertain_background", "quantitative_anchor": False}], "sources": [{"id": "rai_2017_wear", "doi": "10.1016/j.wear.2017.08.016", "role": "Pelton bucket hydro-abrasive mechanism context", "evidence_level": "bibliographic_only"}, {"id": "rai_2019_analytical", "doi": "10.1016/j.wear.2019.203003", "role": "analytical erosion mechanism context", "evidence_level": "bibliographic_only"}, {"id": "leguizamon_2019_multiscale", "doi": "10.3390/ijtpp4020009", "role": "multiscale Pelton erosion context", "evidence_level": "bibliographic_only"}, {"id": "messa_2019_injector", "doi": "10.1016/j.renene.2018.06.064", "role": "Eulerian-Lagrangian sensitivity context", "evidence_level": "bibliographic_only"}, {"id": "ge_2021_bucket", "doi": "10.1016/j.renene.2021.04.072", "role": "Pelton bucket velocity/opening context", "evidence_level": "bibliographic_only"}], "region_field": {"path": str(region_path), "sha256": sha256_file(region_path), "construction": "mean stable particle support with nonzero background floor", "official_face_tags": False}, "claim_scope": "participant-built regional prior for conditional competition benchmark; not official DEC truth"}
    literature["report_sha256"] = canonical_sha256(literature)
    literature_path = target / "literature_reference.json"
    safe_json_dump(literature, literature_path)
    manifest = {"version": VERSION, "status": "conditional_participant_benchmark_inputs_ready", "n_independent_pairs": len(pairs), "n_faces": int(len(first["areas"])), "domain": {"centers_sha256": canonical_sha256(first["centers"].tolist()), "areas_sha256": canonical_sha256(first["areas"].tolist())}, "seed_reports": [{"path": str(target / f"seed_{i:02d}.json"), "sha256": sha256_file(target / f"seed_{i:02d}.json")} for i in range(len(pairs))], "merged_report": {"path": str(target / "merged.json"), "sha256": sha256_file(target / "merged.json")}, "literature_reference": {"path": str(literature_path), "sha256": sha256_file(literature_path)}, "region_field": {"path": str(region_path), "sha256": sha256_file(region_path)}, "official_accuracy_gate": False, "official_reference_gate": False, "fem_gate": "blocked", "measured_truth": False, "claim_scope": "conditional participant-built benchmark only"}
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    safe_json_dump(manifest, target / "input_manifest.json")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build participant benchmark inputs")
    parser.add_argument("--pair", action="append", required=True, help="hit_npz::aggregate_field_npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    pairs = []
    for text in args.pair:
        if "::" not in text:
            parser.error("--pair must use hit_npz::aggregate_field_npz")
        hit, field = text.split("::", 1)
        pairs.append((hit, field))
    print(json.dumps(build_inputs(pairs, args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())