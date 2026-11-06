from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from step30_literature_evidence import (
    CANONICAL_SOURCES,
    LiteratureEvidenceError,
    default_bibliographic_intake,
    validate_literature_intake,
)
from step30_validation_common import sha256_file


def _source(path: pathlib.Path, kind: str = "original_full_text") -> dict:
    return {
        "container_kind": kind,
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _intake(path: pathlib.Path, *, page: int | None, locator: str | None) -> dict:
    citation = CANONICAL_SOURCES["rai_2017_wear"]
    return {
        "version": "step30-literature-evidence-v1",
        "sources": [
            {
                "source_id": "rai_2017_wear",
                **citation,
                "source": _source(path),
                "claims": [
                    {
                        "claim_id": "source_specific_anchor",
                        "claim": "A source-specific geometry fraction is reported.",
                        "evidence_level": "quantitative",
                        "page": page,
                        "locator": locator,
                    }
                ],
            }
        ],
    }


def test_original_text_page_and_locator_are_required_for_quantitative_claim(
    tmp_path: pathlib.Path,
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source fixture")
    report = validate_literature_intake(_intake(source, page=None, locator=None))
    claim = report["sources"][0]["claims"][0]
    assert claim["evidence_level"] == "bibliographic_only"
    assert claim["quantitative_calibration_eligible"] is False


def test_hash_mismatch_fails_closed(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source fixture")
    intake = _intake(source, page=4, locator="Figure 2")
    intake["sources"][0]["source"]["sha256"] = "0" * 64
    with pytest.raises(LiteratureEvidenceError, match="hash mismatch"):
        validate_literature_intake(intake)


def test_quantitative_claim_cannot_transfer_to_dec_dimensions(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source fixture")
    intake = _intake(source, page=4, locator="Figure 2")
    intake["sources"][0]["claims"][0]["dec_dimension_transfer"] = True
    with pytest.raises(LiteratureEvidenceError, match="DEC dimensions"):
        validate_literature_intake(intake)


def test_bibliographic_index_is_downgraded_even_when_hash_bound(
    tmp_path: pathlib.Path,
) -> None:
    index = tmp_path / "bibliography.pdf"
    index.write_bytes(b"bibliography fixture")
    intake = default_bibliographic_intake(index)
    report = validate_literature_intake(intake)
    assert report["bibliographic_only_source_count"] == len(CANONICAL_SOURCES)
    assert report["quantitative_claim_count"] == 0


def test_canonical_citations_include_corrected_crossref_metadata() -> None:
    assert CANONICAL_SOURCES["rai_2019_analytical"]["doi"] == "10.1016/j.wear.2019.203003"
    assert CANONICAL_SOURCES["leguizamon_2019_multiscale"]["title"].endswith(
        "Bridging Scales to Improve the Accuracy"
    )
    assert CANONICAL_SOURCES["fahrni_2025_defects"]["title"].endswith(
        "Part 1: Buckets"
    )
