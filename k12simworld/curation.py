"""K12Vista-to-K12SimBench curation and leakage-resistant split logic."""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .models import K12Problem
from .routing import MECHANICS_TERMS, STATE_TERMS


DYNAMIC_TERMS = MECHANICS_TERMS | STATE_TERMS | {
    "变化", "移动", "旋转", "传播", "加热", "冷却", "开关", "调节", "实验",
    "change", "move", "rotate", "propagate", "switch", "adjust", "experiment",
}
STATIC_ONLY_TERMS = {"作者", "诗歌", "历史人物", "拼写", "definition only"}
CORE_SUBJECT = "physics"
EXTENSION_SUBJECTS = {"mathematics", "chemistry", "biology", "geography"}


@dataclass(frozen=True)
class SelectionDecision:
    suitable: bool
    score: float
    reasons: List[str]


@dataclass
class CurationResult:
    selected: List[K12Problem]
    expert_subset_ids: List[str]
    rejected: List[Tuple[K12Problem, SelectionDecision]]
    summary: Dict[str, object]


def assess_suitability(problem: K12Problem) -> SelectionDecision:
    if problem.dynamic_suitability is not None:
        return SelectionDecision(
            bool(problem.dynamic_suitability),
            1.0 if problem.dynamic_suitability else 0.0,
            ["manual suitability annotation"],
        )
    evidence = " ".join(
        [problem.question, problem.image_caption or "", *problem.knowledge_points]
    ).lower()
    reasons: List[str] = []
    score = 0.0
    if problem.image or problem.image_caption:
        score += 0.25
        reasons.append("multimodal prompt")
    if any(term in evidence for term in DYNAMIC_TERMS):
        score += 0.45
        reasons.append("dynamic concept or state transition")
    if problem.reference_solution and problem.ground_truth:
        score += 0.2
        reasons.append("answer and stepwise reference available")
    if problem.simulation_type:
        score += 0.25
        reasons.append("simulation type annotated")
    if any(term in evidence for term in STATIC_ONLY_TERMS):
        score -= 0.6
        reasons.append("primarily non-simulatable recall content")
    if not problem.reference_solution or not problem.ground_truth:
        reasons.append("missing reference solution or answer")
        score -= 0.35
    if problem.subject not in {CORE_SUBJECT, *EXTENSION_SUBJECTS}:
        score -= 0.5
        reasons.append("outside benchmark subjects")
    score = max(0.0, min(1.0, score))
    return SelectionDecision(score >= 0.55, score, reasons)


def _stable_key(problem: K12Problem, seed: int) -> str:
    raw = f"{seed}:{problem.problem_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _round_robin_stratified(
    candidates: Sequence[Tuple[K12Problem, SelectionDecision]], target: int, seed: int
) -> List[K12Problem]:
    buckets: Dict[Tuple[str, str, str, str], List[Tuple[K12Problem, SelectionDecision]]] = defaultdict(list)
    for problem, decision in candidates:
        band = "middle" if problem.grade and problem.grade <= 9 else "high"
        route_type = problem.simulation_type or "untyped"
        buckets[(problem.subject, band, problem.question_type, route_type)].append((problem, decision))
    for bucket in buckets.values():
        bucket.sort(key=lambda item: (-item[1].score, _stable_key(item[0], seed)))
    ordered_keys = sorted(buckets)
    selected: List[K12Problem] = []
    while ordered_keys and len(selected) < target:
        next_keys = []
        for key in ordered_keys:
            if buckets[key] and len(selected) < target:
                selected.append(buckets[key].pop(0)[0])
            if buckets[key]:
                next_keys.append(key)
        ordered_keys = next_keys
    return selected


def build_benchmark(
    problems: Iterable[K12Problem],
    physics_target: int = 180,
    extension_target: int = 60,
    expert_target: int = 60,
    seed: int = 2026,
) -> CurationResult:
    assessed = [(problem, assess_suitability(problem)) for problem in problems]
    eligible = [(p, d) for p, d in assessed if d.suitable]
    rejected = [(p, d) for p, d in assessed if not d.suitable]
    physics = [(p, d) for p, d in eligible if p.subject == CORE_SUBJECT]
    extensions = [(p, d) for p, d in eligible if p.subject in EXTENSION_SUBJECTS]
    selected_physics = _round_robin_stratified(physics, physics_target, seed)
    selected_extension = _round_robin_stratified(extensions, extension_target, seed + 1)
    selected = selected_physics + selected_extension
    selected.sort(key=lambda item: _stable_key(item, seed))
    expert_subset = _round_robin_stratified(
        [(p, assess_suitability(p)) for p in selected], min(expert_target, len(selected)), seed + 2
    )
    expert_ids = [problem.problem_id for problem in expert_subset]
    subject_counts = Counter(problem.subject for problem in selected)
    type_counts = Counter(problem.question_type for problem in selected)
    grade_counts = Counter(str(problem.grade or "unknown") for problem in selected)
    warnings = []
    if len(selected_physics) < physics_target:
        warnings.append(f"physics target underfilled: {len(selected_physics)}/{physics_target}")
    if len(selected_extension) < extension_target:
        warnings.append(f"extension target underfilled: {len(selected_extension)}/{extension_target}")
    summary: Dict[str, object] = {
        "selected": len(selected),
        "expert_subset": len(expert_ids),
        "rejected": len(rejected),
        "by_subject": dict(sorted(subject_counts.items())),
        "by_question_type": dict(sorted(type_counts.items())),
        "by_grade": dict(sorted(grade_counts.items())),
        "warnings": warnings,
        "seed": seed,
    }
    return CurationResult(selected, expert_ids, rejected, summary)


def assign_knowledge_disjoint_splits(
    problems: Sequence[K12Problem],
    ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 2026,
) -> Dict[str, str]:
    """Assign all items sharing a primary knowledge point to the same split."""
    if abs(sum(ratios) - 1.0) > 1e-9 or any(value < 0 for value in ratios):
        raise ValueError("split ratios must be non-negative and sum to one")
    groups: Dict[str, List[K12Problem]] = defaultdict(list)
    for problem in problems:
        primary = problem.knowledge_points[0] if problem.knowledge_points else f"id:{problem.problem_id}"
        groups[f"{problem.subject}:{primary}"].append(problem)
    names = ("development", "validation", "test")
    targets = [len(problems) * ratio for ratio in ratios]
    counts = [0, 0, 0]
    assignment: Dict[str, str] = {}
    ordered = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest()),
    )
    for _, members in ordered:
        deficits = [targets[index] - counts[index] for index in range(3)]
        chosen = max(range(3), key=lambda index: (deficits[index], -counts[index]))
        for problem in members:
            assignment[problem.problem_id] = names[chosen]
        counts[chosen] += len(members)
    return assignment
