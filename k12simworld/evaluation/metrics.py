"""Success-aware educational and simulation metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


RUBRIC_DIMENSIONS = (
    "correctness_completeness",
    "logical_coherence",
    "pedagogical_effectiveness",
    "typographic_clarity",
    "simulation_problem_alignment",
    "element_layout_quality",
    "temporal_visual_consistency",
    "text_simulation_coordination",
)
OBJECTIVE_DIMENSIONS = (
    "initial_state_match",
    "key_event_accuracy",
    "final_state_accuracy",
    "constraint_satisfaction",
)


def _normalise_score(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    number = float(value)
    if 0.0 <= number <= 1.0:
        number *= 100.0
    if not 0.0 <= number <= 100.0:
        raise ValueError(f"score outside [0, 100]: {number}")
    return number


def geometric_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    if any(value <= 0 for value in clean):
        return 0.0
    return math.exp(sum(math.log(value) for value in clean) / len(clean))


def event_f1(expected: Sequence[Mapping[str, Any]], observed: Sequence[Mapping[str, Any]]) -> float:
    def signature(item: Mapping[str, Any]) -> Tuple[str, str, Tuple[str, ...]]:
        participants = tuple(sorted(str(value) for value in item.get("participants", [])))
        return str(item.get("id") or ""), str(item.get("type") or ""), participants

    gold = {signature(item) for item in expected}
    pred = {signature(item) for item in observed}
    if not gold and not pred:
        return 100.0
    if not gold or not pred:
        return 0.0
    true_positive = len(gold & pred)
    precision = true_positive / len(pred)
    recall = true_positive / len(gold)
    return 0.0 if precision + recall == 0 else 200.0 * precision * recall / (precision + recall)


def numeric_state_accuracy(
    expected: Mapping[str, Any], observed: Mapping[str, Any], tolerance: float = 0.05
) -> float:
    """Compare flat/nested numeric states with relative tolerance; nonnumeric fields exact."""
    checks: List[float] = []

    def walk(gold: Any, pred: Any) -> None:
        if isinstance(gold, Mapping):
            if not isinstance(pred, Mapping):
                checks.append(0.0)
                return
            for key, value in gold.items():
                if key not in pred:
                    checks.append(0.0)
                else:
                    walk(value, pred[key])
        elif isinstance(gold, (int, float)) and not isinstance(gold, bool):
            if not isinstance(pred, (int, float)) or isinstance(pred, bool):
                checks.append(0.0)
                return
            scale = max(1.0, abs(float(gold)))
            checks.append(1.0 if abs(float(gold) - float(pred)) <= tolerance * scale else 0.0)
        else:
            checks.append(1.0 if gold == pred else 0.0)

    walk(expected, observed)
    return 100.0 * mean(checks) if checks else 100.0


def score_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    success = bool(record.get("success", False))
    supplied = dict(record.get("scores") or {})
    scores: Dict[str, Optional[float]] = {}
    for name in (*RUBRIC_DIMENSIONS, *OBJECTIVE_DIMENSIONS):
        scores[name] = _normalise_score(supplied.get(name))
    if not success:
        method = str(record.get("method") or "")
        if method == "text_cot":
            applicable = {
                "correctness_completeness",
                "logical_coherence",
                "pedagogical_effectiveness",
                "typographic_clarity",
            }
            scores = {name: (0.0 if name in applicable else None) for name in scores}
        elif method == "static_manim":
            scores = {
                name: (0.0 if name in RUBRIC_DIMENSIONS else None) for name in scores
            }
        else:
            scores = {name: 0.0 for name in scores}

    solution = geometric_mean(
        scores[name]
        for name in ("correctness_completeness", "logical_coherence", "typographic_clarity")
    )
    simulation = geometric_mean(
        scores[name]
        for name in (
            "simulation_problem_alignment",
            "initial_state_match",
            "key_event_accuracy",
            "final_state_accuracy",
            "constraint_satisfaction",
        )
    )
    pedagogy = geometric_mean(
        scores[name]
        for name in (
            "pedagogical_effectiveness",
            "element_layout_quality",
            "temporal_visual_consistency",
            "text_simulation_coordination",
        )
    )
    # Never make an incomplete modality look competitive by silently averaging
    # only the available families. Text-only baselines retain their solution
    # score, but have no cross-modal overall score.
    overall = (
        geometric_mean([solution, simulation, pedagogy])
        if all(value is not None for value in (solution, simulation, pedagogy))
        else None
    )
    return {
        **dict(record),
        "scores": scores,
        "solution_quality": solution,
        "simulation_correctness": simulation,
        "pedagogical_quality": pedagogy,
        "overall": overall,
    }


def aggregate_records(
    records: Iterable[Mapping[str, Any]], group_by: Sequence[str] = ("model", "method")
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for raw in records:
        scored = score_record(raw)
        grouped[tuple(scored.get(key, "unknown") for key in group_by)].append(scored)
    output: List[Dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        result = {name: value for name, value in zip(group_by, key)}
        result["n"] = len(rows)
        result["success_rate"] = 100.0 * sum(bool(row.get("success")) for row in rows) / len(rows)
        for metric in (
            "solution_quality",
            "simulation_correctness",
            "pedagogical_quality",
            "overall",
        ):
            values = [row[metric] for row in rows if row[metric] is not None]
            result[metric] = mean(values) if values else None
        for metric in ("latency_seconds", "input_tokens", "output_tokens", "estimated_cost_usd"):
            values = [float(row.get(metric, 0.0) or 0.0) for row in rows]
            result[f"mean_{metric}"] = mean(values)
        output.append(result)
    return output
