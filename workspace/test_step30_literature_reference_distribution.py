from __future__ import annotations

from pathlib import Path

import numpy as np

from step30_literature_reference_distribution import (
    MODEL_MAP_KEYS,
    build_provisional_manifest,
    build_reference_distribution,
    recompute_provisional_manifest_sha256,
)


def _prior_fixture(root: Path) -> tuple[Path, Path]:
    hits = root / "hits.npz"
    prior = root / "prior.json"
    np.savez_compressed(
        hits,
        face_area_m2=np.asarray([1.0e-4, 2.0e-4, 3.0e-4, 4.0e-4]),
    )
    report = {
        "input_bindings": {"particle_hits_path": str(hits)},
    }
    prior.write_text(__import__("json").dumps(report), encoding="utf-8")
    arrays = {"face_centers_local_mm": np.zeros((4, 3))}
    for index, model in enumerate(MODEL_MAP_KEYS):
        arrays[f"{model}_p50_mm_h"] = np.asarray(
            [float(index + 1), 0.0, 0.0, 0.0]
        )
    np.savez_compressed(prior.with_suffix(".npz"), **arrays)
    return prior, hits


def test_reference_package_is_qualitative_and_blocks_missing_region_mapping(
    tmp_path: Path,
) -> None:
    prior, _ = _prior_fixture(tmp_path)
    report = build_reference_distribution(prior_report_path=prior)
    assert report["status"] == "provisional"
    assert report["reference_kind"] == "literature_reference"
    assert report["distribution_semantics"]["probability_claim"] is False
    assert report["official_reference_gate"] is False
    assert report["model_comparison"]["status"] == "blocked"
    assert any(
        anchor["value"] == 0.05
        for region in report["regions"]
        for anchor in region["reported_geometry_anchors"]
    )


def test_confirmed_face_mapping_only_produces_a_diagnostic_rank(
    tmp_path: Path,
) -> None:
    prior, _ = _prior_fixture(tmp_path)
    mapping = tmp_path / "region-mapping.npz"
    np.savez_compressed(
        mapping,
        region_id=np.asarray(
            ["splitter_like", "cutout_like", "residual_other", "brim_like"]
        ),
        mapping_status=np.asarray("externally_confirmed"),
    )
    report = build_reference_distribution(
        prior_report_path=prior,
        face_region_mapping_path=mapping,
    )
    assert report["model_comparison"]["status"] == "provisional_diagnostic"
    assert report["model_comparison"]["official_accuracy_gate"] is False
    assert report["model_comparison"]["models"]["oka"]["not_an_accuracy_metric"] is True


def test_provisional_manifest_binds_only_nonacceptance_artifacts(
    tmp_path: Path,
) -> None:
    prior, _ = _prior_fixture(tmp_path)
    reference = tmp_path / "reference.json"
    build_reference_distribution(prior_report_path=prior, output_path=reference)
    geometry = tmp_path / "geometry.json"
    scan = tmp_path / "scan.json"
    acquisition = tmp_path / "acquisition.json"
    guide = tmp_path / "acquisition.md"
    for path in (geometry, scan, acquisition, guide):
        path.write_text(path.name, encoding="utf-8")
    manifest = build_provisional_manifest(
        reference,
        prior_report_path=prior,
        geometry_audit_path=geometry,
        scan_pair_template_path=scan,
        acquisition_template_path=acquisition,
        acquisition_readme_path=guide,
    )
    assert manifest["status"] == "provisional"
    assert manifest["fem_gate"] == "blocked"
    assert manifest["official_accuracy_gate"] is False
    assert manifest["manifest_sha256"] == recompute_provisional_manifest_sha256(
        manifest
    )
