"""Fail-closed quantitative error metrics for measured Step30 validation.

The module reports total quantity error, maximum depth error, area error,
MAE/RMSE/P50/P95/max, and hotspot IoU.  Synthetic results are explicitly
labelled and can never satisfy ``official_accuracy_gate``.
"""

from __future__ import annotations

import argparse
from typing import Any, Mapping

import numpy as np

from step30_validation_common import (
    ValidationError,
    canonical_sha256,
    finite_array,
    independent_validation_ok,
    official_accuracy_gate,
    require_status,
    safe_json_dump,
    validate_units,
)


ERROR_METRICS_VERSION = "step30-error-metrics-v1"


def _paired_values(predicted: Any, observed: Any) -> tuple[np.ndarray, np.ndarray]:
    prediction = finite_array(predicted, "predicted", ndim=1)
    observation = finite_array(observed, "observed", ndim=1)
    if prediction.shape != observation.shape:
        raise ValidationError("predicted and observed arrays must have equal shapes")
    if len(prediction) == 0:
        raise ValidationError("metric arrays must not be empty")
    return prediction, observation


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = finite_array(values, "absolute errors", ndim=1)
    return {
        "mae": float(np.mean(values)),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _validate_weights(weights: Any | None, n: int) -> np.ndarray:
    if weights is None:
        return np.ones(n, dtype=np.float64)
    values = finite_array(weights, "weights", ndim=1, shape=(n,))
    if np.any(values < 0.0) or float(values.sum()) <= 0.0:
        raise ValidationError("weights must be non-negative with positive total")
    return values


def weighted_total(values: Any, weights: Any | None = None) -> float:
    array = finite_array(values, "values", ndim=1)
    weight_array = _validate_weights(weights, len(array))
    return float(np.sum(array * weight_array))


def area_from_mask(mask: Any, area_weights: Any | None = None) -> float:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1 or values.size == 0:
        raise ValidationError("area mask must be a non-empty one-dimensional array")
    if area_weights is None:
        return float(np.sum(values))
    weights = finite_array(area_weights, "area weights", ndim=1, shape=values.shape)
    if np.any(weights < 0.0):
        raise ValidationError("area weights must be non-negative")
    return float(np.sum(weights[values]))


def hotspot_iou(
    predicted: Any,
    observed: Any,
    *,
    percentile: float = 90.0,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Compare high-value regions using identical percentile or explicit thresholds."""
    prediction, observation = _paired_values(predicted, observed)
    if not 0.0 <= percentile <= 100.0:
        raise ValidationError("hotspot percentile must be in [0, 100]")
    if threshold is None:
        prediction_threshold = float(np.percentile(prediction, percentile))
        observation_threshold = float(np.percentile(observation, percentile))
    else:
        threshold = float(threshold)
        if not np.isfinite(threshold):
            raise ValidationError("hotspot threshold must be finite")
        prediction_threshold = observation_threshold = threshold
    predicted_hot = prediction >= prediction_threshold
    observed_hot = observation >= observation_threshold
    intersection = int(np.sum(predicted_hot & observed_hot))
    union = int(np.sum(predicted_hot | observed_hot))
    return {
        "percentile": float(percentile),
        "predicted_threshold": prediction_threshold,
        "observed_threshold": observation_threshold,
        "predicted_count": int(np.sum(predicted_hot)),
        "observed_count": int(np.sum(observed_hot)),
        "intersection": intersection,
        "union": union,
        "iou": float(intersection / union) if union else 1.0,
    }


def calculate_error_metrics(
    predicted: Any,
    observed: Any,
    *,
    status: str = "measured",
    value_units: str = "mm",
    predicted_total: float | None = None,
    observed_total: float | None = None,
    predicted_area: Any | None = None,
    observed_area: Any | None = None,
    area_weights: Any | None = None,
    weights: Any | None = None,
    hotspot_percentile: float = 90.0,
    independent_validation_set: Any = False,
    quality_gates: Mapping[str, Any] | None = None,
    quality_passed: bool | None = None,
) -> dict[str, Any]:
    """Calculate metrics and apply the measured/independent/quality gate."""
    require_status(status)
    units = validate_units(value_units, expected_dimension="length")
    prediction, observation = _paired_values(predicted, observed)
    weight_array = _validate_weights(weights, len(prediction))
    residual = prediction - observation
    absolute = np.abs(residual)
    metrics: dict[str, Any] = {
        "version": ERROR_METRICS_VERSION,
        "status": status,
        "value_units": units,
        "n": int(len(prediction)),
        "residual_distribution": _distribution(absolute),
        "signed_bias": float(np.average(residual, weights=weight_array)),
        "predicted": {
            "total": float(predicted_total) if predicted_total is not None else weighted_total(prediction, weight_array),
            "maximum_depth": float(np.max(prediction)),
        },
        "observed": {
            "total": float(observed_total) if observed_total is not None else weighted_total(observation, weight_array),
            "maximum_depth": float(np.max(observation)),
        },
        "hotspot": hotspot_iou(prediction, observation, percentile=hotspot_percentile),
    }
    max_depth_signed = metrics["predicted"]["maximum_depth"] - metrics["observed"]["maximum_depth"]
    metrics["maximum_depth_error"] = {
        "signed": float(max_depth_signed),
        "absolute": float(abs(max_depth_signed)),
        "relative": float(abs(max_depth_signed) / max(abs(metrics["observed"]["maximum_depth"]), 1e-30)),
    }
    if predicted_total is not None or observed_total is not None:
        if predicted_total is None or observed_total is None:
            raise ValidationError("both predicted_total and observed_total are required")
        pred_total = float(predicted_total)
        obs_total = float(observed_total)
        if not np.isfinite([pred_total, obs_total]).all():
            raise ValidationError("total quantities must be finite")
        metrics["total_error"] = {
            "signed": pred_total - obs_total,
            "absolute": abs(pred_total - obs_total),
            "relative": abs(pred_total - obs_total) / max(abs(obs_total), 1e-30),
        }
    else:
        pred_total = metrics["predicted"]["total"]
        obs_total = metrics["observed"]["total"]
        metrics["total_error"] = {
            "signed": float(pred_total - obs_total),
            "absolute": float(abs(pred_total - obs_total)),
            "relative": float(abs(pred_total - obs_total) / max(abs(obs_total), 1e-30)),
        }
    metrics["total_quantity_error"] = dict(metrics["total_error"])
    if predicted_area is not None or observed_area is not None:
        if predicted_area is None or observed_area is None:
            raise ValidationError("both predicted_area and observed_area are required")
        pred_area_array = np.asarray(predicted_area)
        obs_area_array = np.asarray(observed_area)
        if pred_area_array.ndim == 0 and obs_area_array.ndim == 0:
            area_pred = float(pred_area_array)
            area_obs = float(obs_area_array)
            if not np.isfinite([area_pred, area_obs]).all() or area_pred < 0.0 or area_obs < 0.0:
                raise ValidationError("scalar areas must be finite and non-negative")
            area_result = {
                "predicted": area_pred,
                "observed": area_obs,
                "absolute": abs(area_pred - area_obs),
                "relative": abs(area_pred - area_obs) / max(abs(area_obs), 1e-30),
            }
        else:
            pred_mask = np.asarray(predicted_area, dtype=bool)
            obs_mask = np.asarray(observed_area, dtype=bool)
            if pred_mask.shape != obs_mask.shape or pred_mask.ndim != 1 or pred_mask.size == 0:
                raise ValidationError("predicted_area and observed_area must be equal non-empty 1D masks")
            area_pred = area_from_mask(pred_mask, area_weights)
            area_obs = area_from_mask(obs_mask, area_weights)
            area_result = {
                "predicted": area_pred,
                "observed": area_obs,
                "absolute": abs(area_pred - area_obs),
                "relative": abs(area_pred - area_obs) / max(abs(area_obs), 1e-30),
                "iou": float(np.sum(pred_mask & obs_mask) / np.sum(pred_mask | obs_mask)) if np.any(pred_mask | obs_mask) else 1.0,
            }
        metrics["area_error"] = area_result
    quality_ok = bool(quality_passed) if quality_passed is not None else bool(
        isinstance(quality_gates, Mapping) and quality_gates and all(value is True for value in quality_gates.values())
    )
    metrics["quality"] = {
        "gates": dict(quality_gates) if quality_gates is not None else None,
        "passed": quality_ok,
    }
    metrics["independent_validation_set"] = independent_validation_set
    metrics["official_accuracy_gate"] = official_accuracy_gate(
        status=status,
        independent_validation_set=independent_validation_set,
        quality_passed=quality_ok,
    )
    metrics["provenance_note"] = (
        "synthetic result; not official accuracy" if status == "synthetic" else
        "blocked result; not eligible for accuracy claims" if status == "blocked" else
        "measured result; official accuracy requires the independent-validation gate"
    )
    metrics["metrics_sha256"] = canonical_sha256(metrics)
    return metrics


def calculate_from_records(
    predicted_record: Mapping[str, Any],
    observed_record: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Calculate metrics from measurement records while enforcing status agreement."""
    predicted_status = require_status(predicted_record.get("status"))
    observed_status = require_status(observed_record.get("status"))
    status = kwargs.pop("status", observed_status)
    require_status(status)
    if predicted_status != status or observed_status != status:
        raise ValidationError("predicted, observed, and metric statuses must agree")
    value_column = kwargs.pop("value_column", None)

    def _values(record: Mapping[str, Any], label: str) -> np.ndarray:
        if "values" in record:
            raw = np.asarray(record["values"])
        elif "depths" in record:
            raw = np.asarray(record["depths"])
        elif "grid_values" in record:
            raw = np.asarray(record["grid_values"])
        elif "points" in record:
            raw = np.asarray(record["points"], dtype=np.float64)
            if raw.ndim != 2 or raw.shape[1] != 3:
                raise ValidationError(f"{label} points must have shape (n, 3)")
            return raw
        else:
            raise ValidationError(f"{label} record contains no comparable values")
        if np.iscomplexobj(raw) or not np.isfinite(raw).all():
            raise ValidationError(f"{label} values are not finite real numbers")
        if raw.ndim == 1:
            return raw.astype(np.float64)
        if raw.ndim == 2:
            if value_column is None:
                if raw.shape[1] == 1:
                    return raw[:, 0].astype(np.float64)
                # Material tables require an explicit or last-column value
                # convention; the final column is the documented fallback.
                column = raw.shape[1] - 1
            else:
                column = int(value_column)
            if column < 0 or column >= raw.shape[1]:
                raise ValidationError(f"{label} value_column is outside the record")
            return raw[:, column].astype(np.float64)
        return raw.reshape(-1).astype(np.float64)

    predicted_values = _values(predicted_record, "predicted")
    observed_values = _values(observed_record, "observed")
    if predicted_values.ndim == 2 or observed_values.ndim == 2:
        if predicted_values.shape != observed_values.shape:
            raise ValidationError("point-cloud records must have equal shapes")
        distances = np.linalg.norm(predicted_values - observed_values, axis=1)
        result = calculate_error_metrics(
            distances, np.zeros_like(distances), status=status,
            value_units=kwargs.pop("value_units", "mm"), **kwargs,
        )
        result["comparison_mode"] = "pointwise_euclidean_registration_distance"
        return result
    return calculate_error_metrics(predicted_values, observed_values, status=status, **kwargs)


def save_metrics(metrics: Mapping[str, Any], path: str) -> dict[str, Any]:
    return safe_json_dump(metrics, path)


def _selftest() -> None:
    predicted = np.array([0.0, 1.0, 2.0, 4.0])
    observed = np.array([0.0, 1.5, 2.0, 3.0])
    result = calculate_error_metrics(
        predicted, observed, status="measured", value_units="mm",
        independent_validation_set={"is_independent": True, "n": 4, "source_sha256": "a" * 64},
        quality_gates={"finite": True},
    )
    assert result["official_accuracy_gate"] is True
    assert result["residual_distribution"]["mae"] == 0.375
    assert result["maximum_depth_error"]["absolute"] == 1.0
    assert result["total_error"]["absolute"] == 0.5
    synthetic = calculate_error_metrics(predicted, observed, status="synthetic", quality_passed=True, independent_validation_set={"is_independent": True, "n": 4, "source_sha256": "a" * 64})
    assert synthetic["official_accuracy_gate"] is False
    assert synthetic["provenance_note"] == "synthetic result; not official accuracy"
    assert hotspot_iou(predicted, observed)["iou"] >= 0.0


def selftest() -> bool:
    _selftest()
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step30 error metrics selftest")
    parser.add_argument("--selftest", action="store_true", help="run the minimal selftest")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        print("step30_error_metrics selftest: PASS")
    else:
        parser.print_help()
