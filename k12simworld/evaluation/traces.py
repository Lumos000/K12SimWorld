"""Compare externally collected simulator traces with expert references."""

from __future__ import annotations

import math
from statistics import mean
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .metrics import event_f1, numeric_state_accuracy


def trajectory_rmse(
    expected: Sequence[Mapping[str, Any]], observed: Sequence[Mapping[str, Any]]
) -> float | None:
    """Nearest-time normalized RMSE for scalar or vector trace samples."""
    if not expected or not observed:
        return None
    errors: List[float] = []
    scales: List[float] = []
    for gold in expected:
        time = float(gold.get("t", 0.0))
        pred = min(observed, key=lambda item: abs(float(item.get("t", 0.0)) - time))
        gold_value = gold.get("value")
        pred_value = pred.get("value")
        gold_vector = list(gold_value) if isinstance(gold_value, list) else [gold_value]
        pred_vector = list(pred_value) if isinstance(pred_value, list) else [pred_value]
        if len(gold_vector) != len(pred_vector):
            continue
        for left, right in zip(gold_vector, pred_vector):
            if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                continue
            errors.append((float(left) - float(right)) ** 2)
            scales.append(abs(float(left)))
    if not errors:
        return None
    scale = max(1.0, max(scales, default=1.0))
    return math.sqrt(mean(errors)) / scale


def score_trace(reference: Mapping[str, Any], observed: Mapping[str, Any]) -> Dict[str, float | None]:
    expected_trajectories = reference.get("trajectories") or {}
    observed_trajectories = observed.get("trajectories") or {}
    trajectory_errors = []
    for key, samples in expected_trajectories.items():
        error = trajectory_rmse(samples, observed_trajectories.get(key, []))
        if error is not None:
            trajectory_errors.append(error)
    violations = observed.get("constraint_violations") or []
    constraint_score = 100.0
    if violations:
        failed = sum(bool(item.get("violated", item)) if isinstance(item, Mapping) else bool(item) for item in violations)
        constraint_score = 100.0 * (1.0 - failed / len(violations))
    rmse = mean(trajectory_errors) if trajectory_errors else None
    return {
        "initial_state_match": numeric_state_accuracy(
            reference.get("initial_state") or {}, observed.get("initial_state") or {}
        ),
        "key_event_accuracy": event_f1(
            reference.get("expected_events") or [], observed.get("events") or []
        ),
        "final_state_accuracy": numeric_state_accuracy(
            reference.get("final_state") or {}, observed.get("final_state") or {}
        ),
        "constraint_satisfaction": constraint_score,
        "trajectory_nrmse": rmse,
    }
