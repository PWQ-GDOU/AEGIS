"""Public CLI and compatibility API for Step30 mechanism candidates.

The implementation lives in :mod:`step30_erosion_mechanism_classifier`; this
small module preserves a clear step30_mechanism_classifier entry point without
duplicating classification logic.
"""

from __future__ import annotations

from step30_erosion_mechanism_classifier import (
    CLAIM_SCOPE,
    LABELS,
    VERSION,
    ClassifierError,
    aggregate_face_risk,
    aggregate_spatial_risk,
    build_result,
    classify,
    classify_and_write,
    classify_hits,
    classify_mechanisms,
    load_hits,
    load_inputs,
    main,
    selftest,
    write_outputs,
)

__all__ = [
    "CLAIM_SCOPE", "LABELS", "VERSION", "ClassifierError",
    "aggregate_face_risk", "aggregate_spatial_risk", "build_result",
    "classify", "classify_and_write", "classify_hits", "classify_mechanisms",
    "load_hits", "load_inputs", "selftest", "write_outputs",
]


if __name__ == "__main__":
    raise SystemExit(main())
