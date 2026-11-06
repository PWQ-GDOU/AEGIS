from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from step30_particle_convergence_v4 import (
    js_divergence,
    mass_hotspot_iou,
    weighted_spearman,
)


def test_area_weighted_spearman_is_one_for_identical_fields() -> None:
    values = np.asarray([0.0, 2.0, 1.0, 4.0])
    area = np.asarray([1.0, 1.0, 2.0, 3.0])
    assert weighted_spearman(values, values, area) == pytest.approx(1.0)


def test_mass_hotspot_iou_is_one_for_identical_fields() -> None:
    values = np.asarray([0.0, 1.0, 4.0, 2.0])
    assert mass_hotspot_iou(values, values) == pytest.approx(1.0)


def test_js_divergence_is_zero_for_identical_fields() -> None:
    values = np.asarray([0.0, 1.0, 4.0, 2.0])
    assert js_divergence(values, values) == pytest.approx(0.0)


def test_js_divergence_rejects_zero_mass() -> None:
    with pytest.raises(ValueError):
        js_divergence(np.zeros(3), np.ones(3))


def test_scale_gate_values_are_json_serializable_python_bools() -> None:
    gate = bool(np.float64(0.01) <= 0.05)
    assert json.loads(json.dumps({"scale_relative_mass_change": gate})) == {
        "scale_relative_mass_change": True
    }
