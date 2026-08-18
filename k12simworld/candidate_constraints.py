"""Candidate-answer-conditioned simulation contracts and deterministic checks.

The gold answer is deliberately absent from this module.  A contract is built
only from the candidate model's solution and its EduWorldSpec, then checked
against traces emitted by trusted domain solvers.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .domain_solvers import _SafeExpression
from .evaluation.metrics import numeric_state_accuracy
from .models import ContractError, EduWorldSpec


def _mapping_list(value: Any, field_name: str) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ContractError(f"{field_name} must be an array of JSON objects")
    return [dict(item) for item in value]


@dataclass(frozen=True)
class CandidateSolution:
    """A gold-free, auditable solution produced before simulation planning."""

    problem_id: str
    analysis: str
    final_answer: str
    givens: List[Dict[str, Any]] = field(default_factory=list)
    derived_values: List[Dict[str, Any]] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    schema_version: str = "1.0"

    @classmethod
    def from_story(cls, problem_id: str, story: Mapping[str, Any]) -> "CandidateSolution":
        structured = story.get("solution") or {}
        if not isinstance(structured, Mapping):
            raise ContractError("storyboard.solution must be a JSON object")
        top_answer = str(story.get("final_answer") or "").strip()
        nested_answer = str(structured.get("final_answer") or "").strip()
        # ``solution.final_answer`` belongs to the structured candidate solution and
        # is therefore the canonical value when both representations are present.
        # A disagreement is retained by the pipeline as an audit warning instead of
        # preventing an otherwise renderable storyboard from progressing.
        final_answer = nested_answer or top_answer
        if not final_answer:
            raise ContractError("candidate solution requires final_answer")
        analysis = str(structured.get("analysis") or story.get("analysis") or "").strip()
        if not analysis:
            raise ContractError("candidate solution requires analysis")
        raw_assumptions = structured.get("assumptions") or []
        if isinstance(raw_assumptions, str):
            assumptions = [raw_assumptions]
        elif isinstance(raw_assumptions, Sequence):
            assumptions = [str(item) for item in raw_assumptions]
        else:
            raise ContractError("solution.assumptions must be an array or string")
        confidence = structured.get("confidence")
        if confidence is not None:
            confidence = float(confidence)
            if not 0.0 <= confidence <= 1.0:
                raise ContractError("solution.confidence must be within [0, 1]")
        return cls(
            problem_id=problem_id,
            analysis=analysis,
            final_answer=final_answer,
            givens=_mapping_list(structured.get("givens"), "solution.givens"),
            derived_values=_mapping_list(
                structured.get("derived_values"), "solution.derived_values"
            ),
            assumptions=assumptions,
            confidence=confidence,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def canonical_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SimulationContract:
    """Executable constraints derived only from a candidate solution."""

    problem_id: str
    candidate_solution_sha256: str
    initial_state: Dict[str, Any]
    final_state: Dict[str, Any]
    terminal_event: Dict[str, Any]
    expected_events: List[Dict[str, Any]]
    target_observables: List[Dict[str, Any]]
    invariants: List[Dict[str, Any]]
    source: str = "candidate_solution"
    schema_version: str = "1.0"

    @classmethod
    def from_world_spec(
        cls, solution: CandidateSolution, spec: EduWorldSpec
    ) -> "SimulationContract":
        return cls(
            problem_id=spec.problem_id,
            candidate_solution_sha256=solution.canonical_hash(),
            initial_state=dict(spec.initial_state),
            final_state=dict(spec.final_state),
            terminal_event=dict(spec.terminal_event),
            expected_events=[dict(item) for item in spec.expected_events],
            target_observables=[dict(item) for item in spec.target_observables],
            invariants=[dict(item) for item in spec.invariants],
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def evaluable(self) -> bool:
        return bool(self.target_observables or self.invariants)


def _series(trace: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    value = trace.get("time_series") or trace.get("frames") or []
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _canonical_snapshot(frame: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "time": frame.get("t", frame.get("time", 0.0)),
        "objects": dict(frame.get("objects") or {}),
    }


def _path_parts(path: str) -> List[str]:
    return [part for part in path.replace("[", ".").replace("]", "").split(".") if part]


def _resolve_path(root: Any, path: str) -> Any:
    current = root
    for part in _path_parts(path):
        if isinstance(current, Mapping):
            if part not in current:
                raise KeyError(path)
            current = current[part]
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            aliases = {"x": 0, "y": 1, "z": 2}
            index = aliases.get(part, int(part) if part.lstrip("-").isdigit() else -1)
            if index < 0 or index >= len(current):
                raise KeyError(path)
            current = current[index]
        else:
            raise KeyError(path)
    return current


def _compile_formula(
    item: Mapping[str, Any], location: str
) -> Tuple[_SafeExpression, Dict[str, str]]:
    expression = str(item.get("expression") or "").strip()
    raw_bindings = item.get("bindings")
    if not expression or not isinstance(raw_bindings, Mapping):
        raise ValueError(f"{location} requires expression and bindings")
    bindings = {str(alias): str(path).strip() for alias, path in raw_bindings.items()}
    return _SafeExpression(expression, bindings, f"{location}.expression"), bindings


def _evaluate_formula(
    evaluator: _SafeExpression,
    bindings: Mapping[str, str],
    frame: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> float | bool:
    values: Dict[str, float] = {}
    for alias, raw_path in bindings.items():
        if raw_path.startswith("trace."):
            root, path = trace, raw_path[len("trace.") :]
        else:
            root, path = frame, raw_path
        observed = _resolve_path(root, path)
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            raise ValueError(
                f"formula binding {alias!r} must resolve to a numeric scalar, "
                f"got {type(observed).__name__}"
            )
        numeric = float(observed)
        if not math.isfinite(numeric):
            raise ValueError(f"formula binding {alias!r} must resolve to a finite number")
        values[alias] = numeric
    return evaluator.evaluate(values)


def _event_matches(candidate: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    expected_id = str(expected.get("id") or "")
    candidate_id = str(candidate.get("id") or "")
    expected_type = str(expected.get("type") or "")
    candidate_type = str(candidate.get("type") or "")
    type_aliases = {
        "contact": {"contact", "contact_begin"},
        "collision": {"collision", "contact_begin"},
    }
    type_ok = not expected_type or candidate_type == expected_type
    if expected_type in type_aliases:
        type_ok = candidate_type in type_aliases[expected_type]
    id_ok = not expected_id or not candidate_id or expected_id == candidate_id
    expected_participants = set(str(item) for item in expected.get("participants", []))
    candidate_participants = set(str(item) for item in candidate.get("participants", []))
    return type_ok and id_ok and expected_participants.issubset(candidate_participants)


def _select_frame(
    trace: Mapping[str, Any], at: Any, terminal_event: Mapping[str, Any]
) -> Tuple[Mapping[str, Any], Optional[Mapping[str, Any]]]:
    frames = _series(trace)
    if not frames:
        raise KeyError("trace has no time_series")
    if at in (None, "final"):
        return frames[-1], None
    if at == "initial":
        return frames[0], None
    expected_event: Mapping[str, Any]
    if at == "terminal_event":
        expected_event = terminal_event
    elif isinstance(at, Mapping):
        expected_event = at
    else:
        expected_event = {"id": str(at).removeprefix("event:")}
    events = trace.get("events") or []
    event = next(
        (
            item
            for item in events
            if isinstance(item, Mapping) and _event_matches(item, expected_event)
        ),
        None,
    )
    if event is None:
        raise KeyError(f"event not observed: {dict(expected_event)}")
    event_time = float(event.get("t", event.get("time", 0.0)))
    event_snapshot = event.get("snapshot")
    if isinstance(event_snapshot, Mapping):
        return event_snapshot, event
    frame = min(frames, key=lambda item: abs(float(item.get("t", 0.0)) - event_time))
    return frame, event


def _numeric_close(observed: float, expected: float, absolute: float, relative: float) -> bool:
    return abs(observed - expected) <= max(absolute, relative * max(1.0, abs(expected)))


def _compare_values(
    observed: Any,
    expected: Any,
    operator: str,
    absolute: float,
    relative: float,
) -> Tuple[bool, Optional[float]]:
    if operator == "in_range":
        if (
            not isinstance(observed, (int, float))
            or isinstance(observed, bool)
            or not isinstance(expected, Sequence)
            or isinstance(expected, (str, bytes))
            or len(expected) != 2
        ):
            return False, None
        low, high = float(expected[0]), float(expected[1])
        value = float(observed)
        return low - absolute <= value <= high + absolute, 0.0
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        if not isinstance(observed, Sequence) or isinstance(observed, (str, bytes)):
            return False, None
        if len(observed) != len(expected):
            return False, None
        comparisons = [
            _compare_values(left, right, operator, absolute, relative)
            for left, right in zip(observed, expected)
        ]
        return all(item[0] for item in comparisons), max(
            (item[1] or 0.0 for item in comparisons), default=0.0
        )
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(observed, (int, float)) or isinstance(observed, bool):
            return False, None
        left, right = float(observed), float(expected)
        error = left - right
        if operator in {"approximately_equal", "approx", "equal", "=="}:
            return _numeric_close(left, right, absolute, relative), error
        if operator in {"less_than_or_equal", "lte", "<="}:
            return left <= right + absolute, error
        if operator in {"greater_than_or_equal", "gte", ">="}:
            return left >= right - absolute, error
        return False, error
    return observed == expected, None


def _observable_check(
    target: Mapping[str, Any],
    trace: Mapping[str, Any],
    terminal_event: Mapping[str, Any],
) -> Dict[str, Any]:
    check_id = str(
        target.get("id") or target.get("path") or target.get("expression") or "target"
    )
    result: Dict[str, Any] = {
        "id": check_id,
        "kind": "target_observable",
        "required": bool(target.get("required", True)),
        "passed": False,
    }
    try:
        frame, event = _select_frame(trace, target.get("at", "final"), terminal_event)
        raw_path = str(target.get("path") or "").strip()
        expression = str(target.get("expression") or "").strip()
        if expression:
            evaluator, bindings = _compile_formula(target, f"target {check_id!r}")
            observed = _evaluate_formula(evaluator, bindings, frame, trace)
        else:
            if raw_path.startswith("trace."):
                root, path = trace, raw_path[len("trace.") :]
            else:
                root, path = frame, raw_path
            if not path:
                object_id = str(target.get("object_id") or "")
                quantity = str(target.get("quantity") or "")
                if quantity in {"time", "t"}:
                    path = "t"
                elif object_id and quantity:
                    path = f"objects.{object_id}.{quantity}"
                else:
                    raise KeyError("target needs path or expression+bindings")
            observed = _resolve_path(root, path)
        expected = target.get("expected", target.get("value"))
        operator = str(target.get("operator") or "approximately_equal")
        absolute = float(target.get("absolute_tolerance", target.get("tolerance", 1e-6)))
        relative = float(target.get("relative_tolerance", 0.02))
        passed, error = _compare_values(observed, expected, operator, absolute, relative)
        result.update(
            {
                "path": (raw_path or path) if not expression else None,
                "expression": expression or None,
                "bindings": dict(target.get("bindings") or {}) if expression else None,
                "display_formula": target.get("display_formula"),
                "at": target.get("at", "final"),
                "expected": expected,
                "observed": observed,
                "operator": operator,
                "unit": target.get("result_unit", target.get("unit")),
                "absolute_tolerance": absolute,
                "relative_tolerance": relative,
                "error": error,
                "event": dict(event) if event else None,
                "passed": passed,
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        result["error_message"] = str(exc)
    return result


def _invariant_check(invariant: Mapping[str, Any], trace: Mapping[str, Any]) -> Dict[str, Any]:
    check_id = str(
        invariant.get("id") or invariant.get("path") or invariant.get("expression") or "invariant"
    )
    result: Dict[str, Any] = {
        "id": check_id,
        "kind": "invariant",
        "required": bool(invariant.get("required", True)),
        "passed": False,
    }
    try:
        path = str(invariant.get("path") or "").strip()
        expression = str(invariant.get("expression") or "").strip()
        if expression:
            evaluator, bindings = _compile_formula(invariant, f"invariant {check_id!r}")
            values = [
                _evaluate_formula(evaluator, bindings, frame, trace)
                for frame in _series(trace)
            ]
        elif not path:
            object_id = str(invariant.get("object_id") or "")
            quantity = str(invariant.get("quantity") or "")
            if not object_id or not quantity:
                raise KeyError("invariant needs path or expression+bindings")
            path = f"objects.{object_id}.{quantity}"
            values = [_resolve_path(frame, path) for frame in _series(trace)]
        else:
            values = [_resolve_path(frame, path) for frame in _series(trace)]
        if not values:
            raise KeyError("trace has no values for invariant")
        kind = str(invariant.get("type") or "constant")
        tolerance = float(invariant.get("tolerance", 0.02))
        if kind == "constant":
            baseline = float(invariant.get("value", values[0]))
            numeric = [float(value) for value in values]
            maximum_error = max(abs(value - baseline) for value in numeric)
            passed = maximum_error <= tolerance * max(1.0, abs(baseline))
        elif kind == "nondecreasing":
            numeric = [float(value) for value in values]
            maximum_error = max((left - right for left, right in zip(numeric, numeric[1:])), default=0.0)
            passed = maximum_error <= tolerance
        elif kind == "nonincreasing":
            numeric = [float(value) for value in values]
            maximum_error = max((right - left for left, right in zip(numeric, numeric[1:])), default=0.0)
            passed = maximum_error <= tolerance
        else:
            raise ValueError(f"unsupported invariant type {kind!r}")
        result.update(
            {
                "path": path or None,
                "expression": expression or None,
                "bindings": dict(invariant.get("bindings") or {}) if expression else None,
                "display_formula": invariant.get("display_formula"),
                "unit": invariant.get("result_unit", invariant.get("unit")),
                "type": kind,
                "samples": len(values),
                "maximum_error": maximum_error,
                "tolerance": tolerance,
                "passed": passed,
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        result["error_message"] = str(exc)
    return result


def _select_scene_trace(
    scenes: Sequence[Mapping[str, Any]], scene_id: Optional[str]
) -> Optional[Mapping[str, Any]]:
    if scene_id:
        for scene in scenes:
            if str(scene.get("scene_id") or "") == scene_id:
                trace = scene.get("trace")
                return trace if isinstance(trace, Mapping) else None
        return None
    for scene in reversed(scenes):
        trace = scene.get("trace")
        if isinstance(trace, Mapping):
            return trace
    return None


def validate_candidate_contract(
    contract: SimulationContract, program: Mapping[str, Any]
) -> Dict[str, Any]:
    """Validate candidate targets without consulting benchmark gold answers."""

    scenes = [item for item in program.get("scenes", []) if isinstance(item, Mapping)]
    checks: List[Dict[str, Any]] = []
    for target in contract.target_observables:
        scene_id = str(target.get("scene_id") or "") or None
        trace = _select_scene_trace(scenes, scene_id)
        if trace is None:
            checks.append(
                {
                    "id": str(target.get("id") or "target"),
                    "kind": "target_observable",
                    "required": bool(target.get("required", True)),
                    "passed": False,
                    "error_message": f"scene trace not found: {scene_id or 'last'}",
                }
            )
        else:
            checks.append(_observable_check(target, trace, contract.terminal_event))
    for invariant in contract.invariants:
        scene_id = str(invariant.get("scene_id") or "") or None
        trace = _select_scene_trace(scenes, scene_id)
        if trace is None:
            checks.append(
                {
                    "id": str(invariant.get("id") or "invariant"),
                    "kind": "invariant",
                    "required": bool(invariant.get("required", True)),
                    "passed": False,
                    "error_message": f"scene trace not found: {scene_id or 'last'}",
                }
            )
        else:
            checks.append(_invariant_check(invariant, trace))

    traces = [scene.get("trace") for scene in scenes if isinstance(scene.get("trace"), Mapping)]
    first_trace = traces[0] if traces else {}
    last_trace = traces[-1] if traces else {}
    first_frames = _series(first_trace)
    last_frames = _series(last_trace)
    initial_score = None
    final_score = None
    if contract.initial_state and first_frames:
        initial_score = numeric_state_accuracy(
            contract.initial_state, _canonical_snapshot(first_frames[0])
        )
    if contract.final_state and last_frames:
        final_score = numeric_state_accuracy(
            contract.final_state, _canonical_snapshot(last_frames[-1])
        )

    observed_events = [
        event
        for trace in traces
        for event in trace.get("events", [])
        if isinstance(event, Mapping)
    ]
    required_events = [
        event for event in contract.expected_events if bool(event.get("required_for_validation", False))
    ]
    matched_events = sum(
        any(_event_matches(observed, expected) for observed in observed_events)
        for expected in required_events
    )
    event_score = 100.0 if not required_events else 100.0 * matched_events / len(required_events)
    required_checks = [check for check in checks if check.get("required", True)]
    passed_checks = sum(bool(check.get("passed")) for check in required_checks)
    constraint_score = (
        100.0 * passed_checks / len(required_checks) if required_checks else None
    )
    evaluable = bool(required_checks)
    passed = evaluable and passed_checks == len(required_checks) and event_score == 100.0
    status = "passed" if passed else ("failed" if evaluable else "not_evaluable")
    return {
        "schema_version": "1.0",
        "problem_id": contract.problem_id,
        "source": contract.source,
        "status": status,
        "passed": passed,
        "checks": checks,
        "scores": {
            "candidate_initial_state_match": initial_score,
            "candidate_key_event_accuracy": event_score,
            "candidate_final_state_match": final_score,
            "candidate_constraint_satisfaction": constraint_score,
        },
    }


def build_observed_trace(problem_id: str, program: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize trusted solver traces for later expert-reference evaluation."""

    scene_records: List[Dict[str, Any]] = []
    trajectories: Dict[str, List[Dict[str, Any]]] = {}
    events: List[Dict[str, Any]] = []
    for scene in program.get("scenes", []):
        if not isinstance(scene, Mapping) or not isinstance(scene.get("trace"), Mapping):
            continue
        scene_id = str(scene.get("scene_id") or f"scene_{len(scene_records) + 1}")
        trace = scene["trace"]
        frames = _series(trace)
        scene_records.append(
            {
                "scene_id": scene_id,
                "engine": trace.get("engine"),
                "domain_model": trace.get("domain_model"),
                "initial_state": _canonical_snapshot(frames[0]) if frames else {},
                "final_state": _canonical_snapshot(frames[-1]) if frames else {},
                "summary": trace.get("summary") or {},
            }
        )
        for event in trace.get("events", []):
            if isinstance(event, Mapping):
                events.append({"scene_id": scene_id, **dict(event)})
        for frame in frames:
            time_value = float(frame.get("t", frame.get("time", 0.0)))
            for object_id, state in (frame.get("objects") or {}).items():
                if not isinstance(state, Mapping):
                    continue
                for quantity, value in state.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        trajectories.setdefault(f"{scene_id}.{object_id}.{quantity}", []).append(
                            {"t": time_value, "value": value}
                        )
                    elif isinstance(value, list) and all(
                        isinstance(item, (int, float)) and not isinstance(item, bool)
                        for item in value
                    ):
                        trajectories.setdefault(f"{scene_id}.{object_id}.{quantity}", []).append(
                            {"t": time_value, "value": value}
                        )
    return {
        "schema_version": "1.0",
        "problem_id": problem_id,
        "engine": program.get("engine"),
        "scenes": scene_records,
        "initial_state": scene_records[0]["initial_state"] if scene_records else {},
        "final_state": scene_records[-1]["final_state"] if scene_records else {},
        "events": events,
        "trajectories": trajectories,
    }


def validation_error_messages(report: Mapping[str, Any]) -> List[str]:
    messages = []
    for check in report.get("checks", []):
        if check.get("required", True) and not check.get("passed"):
            detail = check.get("error_message")
            if not detail:
                detail = (
                    f"expected {check.get('expected')!r}, observed {check.get('observed')!r}, "
                    f"error={check.get('error')!r}"
                )
            messages.append(f"candidate target {check.get('id')!r} failed: {detail}")
    if report.get("status") == "not_evaluable":
        messages.append("candidate simulation contract has no executable target_observables or invariants")
    return messages
