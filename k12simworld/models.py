"""Dependency-free data contracts used by K12SimWorld.

The original K12Vista reference solution is deliberately represented separately
from ``model_payload`` so generation code cannot accidentally leak it to a
candidate model.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .domain_common import DomainSimulationError, _SafeExpression


SUBJECT_ALIASES = {
    "math": "mathematics",
    "数学": "mathematics",
    "physics": "physics",
    "物理": "physics",
    "chemistry": "chemistry",
    "化学": "chemistry",
    "biology": "biology",
    "生物": "biology",
    "geography": "geography",
    "地理": "geography",
}

QUESTION_TYPE_ALIASES = {
    "选择题": "multiple_choice",
    "multiple choice": "multiple_choice",
    "multiple_choice": "multiple_choice",
    "填空题": "fill_blank",
    "fill-in-blank": "fill_blank",
    "fill_blank": "fill_blank",
    "解答题": "free_response",
    "简答题": "free_response",
    "问答题": "free_response",
    "free-response": "free_response",
    "free_response": "free_response",
}


class ContractError(ValueError):
    """Raised when an artifact violates a public K12SimWorld contract."""


def _normalise_subject(raw: str) -> tuple[str, Optional[int]]:
    value = (raw or "").strip().lower()
    grade_match = re.search(r"(?:-|_)g(?:rade)?\s*(\d{1,2})\b", value)
    grade = int(grade_match.group(1)) if grade_match else None
    base = re.split(r"(?:-|_)g(?:rade)?\s*\d{1,2}", value, maxsplit=1)[0]
    return SUBJECT_ALIASES.get(base, base or "unknown"), grade


def _normalise_question_type(raw: str) -> str:
    value = (raw or "").strip().lower()
    return QUESTION_TYPE_ALIASES.get(value, value or "unknown")


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[;；]", value) if item.strip()]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, Mapping)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


@dataclass(frozen=True)
class K12Problem:
    problem_id: str
    question: str
    subject: str
    grade: Optional[int]
    question_type: str
    knowledge_points: List[str] = field(default_factory=list)
    difficulty: str = "unknown"
    image: Optional[str] = None
    image_caption: Optional[str] = None
    reference_solution: List[str] = field(default_factory=list)
    ground_truth: List[str] = field(default_factory=list)
    simulation_type: Optional[str] = None
    dynamic_suitability: Optional[bool] = None
    source_metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "K12Problem":
        answer = record.get("format_answer") or record.get("answer") or {}
        if not isinstance(answer, Mapping):
            answer = {"ground_truth": answer}
        subject, inferred_grade = _normalise_subject(
            str(record.get("subject") or record.get("discipline") or "")
        )
        raw_grade = record.get("grade", inferred_grade)
        try:
            grade = int(raw_grade) if raw_grade not in (None, "") else inferred_grade
        except (TypeError, ValueError):
            grade = inferred_grade
        pid = str(
            record.get("problem_id")
            or record.get("hash_id")
            or record.get("id")
            or record.get("index")
            or ""
        ).strip()
        question = str(record.get("question") or record.get("problem") or "").strip()
        if not pid:
            pid = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
        if not question:
            raise ContractError(f"problem {pid!r} has no question text")
        return cls(
            problem_id=pid,
            question=question,
            subject=subject,
            grade=grade,
            question_type=_normalise_question_type(
                str(record.get("question_type") or record.get("type") or "")
            ),
            knowledge_points=_string_list(
                record.get("knowledge_points") or record.get("knowledge_point")
            ),
            difficulty=str(record.get("difficulty") or "unknown"),
            image=record.get("image") or record.get("img") or record.get("image_path"),
            image_caption=record.get("image_caption") or record.get("img_caption"),
            reference_solution=_string_list(
                record.get("reference_solution")
                or answer.get("format_solution")
                or answer.get("solution")
            ),
            ground_truth=_string_list(
                record.get("ground_truth") or answer.get("ground_truth")
            ),
            simulation_type=record.get("simulation_type"),
            dynamic_suitability=record.get("dynamic_suitability"),
            source_metadata={
                **dict(record.get("source_metadata") or {}),
                **{
                    key: record[key]
                    for key in ("split", "expert_reference")
                    if key in record
                },
            },
        )

    def model_payload(self) -> Dict[str, Any]:
        """Return generation input with gold answers intentionally omitted."""
        return {
            "problem_id": self.problem_id,
            "question": self.question,
            "image_caption": self.image_caption,
            "metadata": {
                "subject": self.subject,
                "grade": self.grade,
                "question_type": self.question_type,
                "knowledge_points": self.knowledge_points,
                "difficulty": self.difficulty,
                "simulation_type": self.simulation_type,
            },
        }

    def benchmark_record(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StoryBlock:
    block_id: str
    kind: str
    content: str
    learning_goal: Optional[str] = None
    highlights: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StoryBlock":
        kind = str(value.get("kind") or "").lower()
        if kind not in {"text", "sim"}:
            raise ContractError(f"story block kind must be text or sim, got {kind!r}")
        block_id = str(value.get("block_id") or value.get("id") or "").strip()
        content = str(value.get("content") or value.get("specification") or "").strip()
        if not block_id or not content:
            raise ContractError("story blocks require block_id and content")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", block_id):
            raise ContractError(f"unsafe or invalid storyboard block id: {block_id!r}")
        return cls(
            block_id=block_id,
            kind=kind,
            content=content,
            learning_goal=value.get("learning_goal"),
            highlights=_string_list(value.get("highlights")),
        )


@dataclass(frozen=True)
class RenderSpec:
    engine: str
    fps: int = 30
    duration: float = 8.0
    width: int = 512
    height: int = 512
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RenderSpec":
        result = cls(
            engine=str(value.get("engine") or "").strip().lower(),
            fps=int(value.get("fps", 30)),
            duration=float(value.get("duration", 8.0)),
            width=int(value.get("width", 512)),
            height=int(value.get("height", 512)),
            checkpoints=[dict(item) for item in value.get("checkpoints", [])],
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.engine not in {
            "threejs-cannon", "p5js", "manim", "mechanics-2d",
            "equation-solver", "circuit-solver", "ray-optics",
        }:
            raise ContractError(f"unsupported engine {self.engine!r}")
        if not 1 <= self.fps <= 60:
            raise ContractError("fps must be between 1 and 60")
        if not 1.0 <= self.duration <= 30.0:
            raise ContractError("duration must be between 1 and 30 seconds")
        if not (128 <= self.width <= 1920 and 128 <= self.height <= 1080):
            raise ContractError("render dimensions are outside the supported range")


def _validate_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(f"non-finite number at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite(item, f"{path}[{index}]")


_LOOKUP_PATH_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_-]*(?:\.(?:[A-Za-z_][A-Za-z0-9_-]*|[0-9]+))*$"
)
_FORMULA_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def _validate_formula_contract(item: Mapping[str, Any], location: str) -> None:
    """Validate the structural boundary between lookups and safe formulas."""
    raw_path = item.get("path")
    raw_expression = item.get("expression")
    has_path = isinstance(raw_path, str) and bool(raw_path.strip())
    has_expression = isinstance(raw_expression, str) and bool(raw_expression.strip())
    if has_path == has_expression:
        raise ContractError(f"{location} requires exactly one of path or expression")
    if has_path and not _LOOKUP_PATH_RE.fullmatch(raw_path.strip()):
        raise ContractError(
            f"{location}.path must be a dotted lookup path, not an arithmetic expression"
        )
    if has_expression:
        expression = raw_expression.strip()
        if len(expression) > 512:
            raise ContractError(f"{location}.expression exceeds 512 characters")
        bindings = item.get("bindings")
        if not isinstance(bindings, Mapping) or not 1 <= len(bindings) <= 32:
            raise ContractError(
                f"{location}.bindings must map 1 to 32 variable names to lookup paths"
            )
        for alias, binding_path in bindings.items():
            if not isinstance(alias, str) or not _FORMULA_NAME_RE.fullmatch(alias):
                raise ContractError(f"{location}.bindings contains unsafe variable name {alias!r}")
            if not isinstance(binding_path, str) or not _LOOKUP_PATH_RE.fullmatch(
                binding_path.strip()
            ):
                raise ContractError(
                    f"{location}.bindings.{alias} must be a dotted lookup path"
                )
        try:
            _SafeExpression(expression, bindings.keys(), f"{location}.expression")
        except DomainSimulationError as exc:
            raise ContractError(str(exc)) from exc
    elif item.get("bindings") not in (None, {}):
        raise ContractError(f"{location}.bindings is valid only with expression")
    display_formula = item.get("display_formula")
    if display_formula is not None and (
        not isinstance(display_formula, str) or len(display_formula) > 2000
    ):
        raise ContractError(f"{location}.display_formula must be a string up to 2000 characters")


@dataclass(frozen=True)
class EduWorldSpec:
    problem_id: str
    coordinate_system: Dict[str, Any]
    objects: List[Dict[str, Any]]
    parameters: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    initial_state: Dict[str, Any]
    expected_events: List[Dict[str, Any]]
    learning_goals: List[str]
    visual_conventions: Dict[str, Any]
    final_state: Dict[str, Any] = field(default_factory=dict)
    terminal_event: Dict[str, Any] = field(default_factory=dict)
    target_observables: List[Dict[str, Any]] = field(default_factory=list)
    invariants: List[Dict[str, Any]] = field(default_factory=list)
    schema_version: str = "1.0"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EduWorldSpec":
        object_fields = (
            "coordinate_system", "initial_state", "final_state", "terminal_event",
            "visual_conventions"
        )
        list_fields = (
            "objects", "parameters", "constraints", "expected_events", "target_observables",
            "invariants", "learning_goals"
        )
        for field_name in object_fields:
            field_value = value.get(field_name, {})
            if field_value is not None and not isinstance(field_value, Mapping):
                raise ContractError(
                    f"EduWorldSpec.{field_name} must be a JSON object, got {type(field_value).__name__}"
                )
        for field_name in list_fields:
            field_value = value.get(field_name, [])
            if field_value is not None and not isinstance(field_value, list):
                raise ContractError(
                    f"EduWorldSpec.{field_name} must be a JSON array, got {type(field_value).__name__}"
                )
        result = cls(
            problem_id=str(value.get("problem_id") or "").strip(),
            coordinate_system=dict(value.get("coordinate_system") or {}),
            objects=[dict(item) for item in value.get("objects", [])],
            parameters=[dict(item) for item in value.get("parameters", [])],
            constraints=[dict(item) for item in value.get("constraints", [])],
            initial_state=dict(value.get("initial_state") or {}),
            expected_events=[dict(item) for item in value.get("expected_events", [])],
            learning_goals=_string_list(value.get("learning_goals")),
            visual_conventions=dict(value.get("visual_conventions") or {}),
            final_state=dict(value.get("final_state") or {}),
            terminal_event=dict(value.get("terminal_event") or {}),
            target_observables=[dict(item) for item in value.get("target_observables", [])],
            invariants=[dict(item) for item in value.get("invariants", [])],
            schema_version=str(value.get("schema_version") or "1.0"),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not self.problem_id:
            raise ContractError("world spec requires problem_id")
        object_ids = [str(item.get("id") or "").strip() for item in self.objects]
        if not object_ids or any(not item for item in object_ids):
            raise ContractError("world spec requires objects with non-empty ids")
        if len(object_ids) != len(set(object_ids)):
            raise ContractError("world object ids must be unique")
        invalid_object_ids = [
            item for item in object_ids if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", item)
        ]
        if invalid_object_ids:
            raise ContractError(f"world object ids must be safe identifiers: {invalid_object_ids}")
        event_ids = [str(item.get("id") or "").strip() for item in self.expected_events]
        if any(not item for item in event_ids) or len(event_ids) != len(set(event_ids)):
            raise ContractError("expected event ids must be present and unique")
        known = set(object_ids)
        for event in self.expected_events:
            unknown = set(_string_list(event.get("participants"))) - known
            if unknown:
                raise ContractError(
                    f"event {event.get('id')!r} references unknown objects: {sorted(unknown)}"
                )
        terminal_unknown = set(_string_list(self.terminal_event.get("participants"))) - known
        if terminal_unknown:
            raise ContractError(
                f"terminal event references unknown objects: {sorted(terminal_unknown)}"
            )
        target_ids = [str(item.get("id") or "").strip() for item in self.target_observables]
        if any(not item for item in target_ids) or len(target_ids) != len(set(target_ids)):
            raise ContractError("target observable ids must be present and unique")
        invariant_ids = [str(item.get("id") or "").strip() for item in self.invariants]
        if any(not item for item in invariant_ids) or len(invariant_ids) != len(set(invariant_ids)):
            raise ContractError("invariant ids must be present and unique")
        for index, target in enumerate(self.target_observables):
            _validate_formula_contract(target, f"target_observables[{index}]")
            if "expected" not in target:
                raise ContractError(f"target_observables[{index}] requires expected")
        for index, invariant in enumerate(self.invariants):
            _validate_formula_contract(invariant, f"invariants[{index}]")
            if invariant.get("type", "constant") not in {
                "constant", "nondecreasing", "nonincreasing"
            }:
                raise ContractError(f"invariants[{index}] has unsupported type")
        _validate_finite(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def canonical_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class ArtifactManifest:
    problem_id: str
    model: str
    method: str
    success: bool
    engine: Optional[str] = None
    storyboard_path: Optional[str] = None
    solution_spec_path: Optional[str] = None
    world_spec_path: Optional[str] = None
    simulation_contract_path: Optional[str] = None
    program_path: Optional[str] = None
    observed_trace_path: Optional[str] = None
    target_validation_path: Optional[str] = None
    document_path: Optional[str] = None
    video_paths: List[str] = field(default_factory=list)
    trace_paths: List[str] = field(default_factory=list)
    attempts: int = 1
    repaired: bool = False
    error: Optional[str] = None
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    scores: Dict[str, Optional[float]] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
