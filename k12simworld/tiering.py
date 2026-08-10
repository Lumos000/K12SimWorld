"""Deterministic execution-readiness tiers for human-selected physics tasks."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .io import read_records, write_json, write_jsonl
from .models import K12Problem


TIER_VERSION = "physics-execution-tiers-v1.2-equation-ode"

NATIVE_FAMILIES = {
    "kinematics", "projectile", "force_and_motion", "inclined_plane",
    "friction", "lever", "pulley", "spring", "collision",
    "circular_motion", "energy", "buoyancy",
}
EQUATION_FAMILIES = {"charged_particle"}
SPECIALIZED_FAMILIES = {"circuit", "ray_optics"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_physics_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Assign exactly one tier, with human groups overriding noisy model labels."""
    family = str(record.get("simulation_type") or "other")
    metadata = record.get("source_metadata") or {}
    groups = [str(value) for value in metadata.get("human_selection_groups") or []]

    # Specialized semantic solvers take precedence over generic mechanics or
    # equation labels when an item appears in multiple human groups.
    if family == "circuit" or "电路" in groups:
        return {
            "tier": "specialized",
            "subtype": "circuit",
            "current_engine": "circuit-solver",
            "required_backend": "circuit_solver",
            "implementation_status": "deterministic_dc_solver_available",
            "ready_for_main_experiment": True,
            "reason": "The deterministic DC backend solves topology, meter readings, lamp power, and switch actions.",
        }
    if family == "ray_optics" or "光学" in groups:
        return {
            "tier": "specialized",
            "subtype": "ray_optics",
            "current_engine": "ray-optics",
            "required_backend": "ray_optics",
            "implementation_status": "deterministic_ray_tracer_available",
            "ready_for_main_experiment": True,
            "reason": "The geometric backend computes intersections, reflection/refraction, TIR, and thin-lens rays.",
        }
    if family in EQUATION_FAMILIES or "电磁场" in groups:
        return {
            "tier": "equation",
            "subtype": "electromagnetic_dynamics",
            "current_engine": "equation-solver",
            "required_backend": "deterministic_equation_integrator",
            "implementation_status": "boris_and_safe_rk4_integrators_available",
            "ready_for_main_experiment": True,
            "reason": (
                "The equation backend supports charged-particle Lorentz motion with Boris integration "
                "and explicit coupled first-order systems with a restricted-expression RK4 solver."
            ),
        }
    if family in NATIVE_FAMILIES:
        return {
            "tier": "native",
            "subtype": family,
            "current_engine": "threejs-cannon",
            "required_backend": "threejs-cannon",
            "implementation_status": "backend_available_dependencies_required",
            "ready_for_main_experiment": True,
            "reason": "The primary process is covered by the current rigid-body/mechanics renderer.",
        }
    # Preserve total coverage without silently calling an unknown process native.
    return {
        "tier": "equation",
        "subtype": "manual_rule_engine",
        "current_engine": "p5js",
        "required_backend": "custom_rule_engine",
        "implementation_status": "manual_engine_review_required",
        "ready_for_main_experiment": False,
        "reason": f"Unrecognized simulation family {family!r}; deterministic rule implementation needs review.",
    }


def partition_physics_tiers(
    benchmark_path: str | Path, output_dir: str | Path
) -> Dict[str, Any]:
    source = Path(benchmark_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = list(read_records(source))
    if not records:
        raise ValueError("physics benchmark is empty")

    tier_records: Dict[str, List[Dict[str, Any]]] = {
        "native": [], "equation": [], "specialized": [],
    }
    assignments: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in records:
        problem = K12Problem.from_record(row)
        if problem.subject != "physics":
            raise ValueError(f"non-physics record in physics benchmark: {problem.problem_id}")
        if problem.problem_id in seen:
            raise ValueError(f"duplicate problem_id in physics benchmark: {problem.problem_id}")
        seen.add(problem.problem_id)
        decision = classify_physics_record(row)
        enriched = dict(row)
        enriched["source_metadata"] = {
            **dict(row.get("source_metadata") or {}),
            "execution_tier": {"version": TIER_VERSION, **decision},
        }
        tier_records[decision["tier"]].append(enriched)
        assignments.append({
            "problem_id": problem.problem_id,
            "grade": problem.grade,
            "question_type": problem.question_type,
            "knowledge_points": problem.knowledge_points,
            "simulation_type": problem.simulation_type,
            "human_selection_groups": (row.get("source_metadata") or {}).get("human_selection_groups") or [],
            **decision,
        })

    assigned = sum(len(values) for values in tier_records.values())
    if assigned != len(records) or len({item["problem_id"] for item in assignments}) != len(records):
        raise RuntimeError("tier partition is not mutually exclusive and collectively exhaustive")

    paths = {
        tier: output / f"physics_{tier}_v1.jsonl" for tier in tier_records
    }
    for tier, path in paths.items():
        write_jsonl(path, tier_records[tier])
    assignment_path = output / "tier_assignments.jsonl"
    write_jsonl(assignment_path, assignments)

    csv_buffer = io.StringIO()
    fields = [
        "problem_id", "tier", "subtype", "simulation_type", "grade",
        "question_type", "current_engine", "required_backend",
        "implementation_status", "ready_for_main_experiment",
        "human_selection_groups", "knowledge_points", "reason",
    ]
    writer = csv.DictWriter(csv_buffer, fieldnames=fields)
    writer.writeheader()
    for item in assignments:
        writer.writerow({
            **item,
            "human_selection_groups": "；".join(item["human_selection_groups"]),
            "knowledge_points": "；".join(item["knowledge_points"]),
        })
    (output / "tier_assignments.csv").write_text(csv_buffer.getvalue(), encoding="utf-8")

    report: Dict[str, Any] = {
        "version": TIER_VERSION,
        "source_benchmark": str(source.resolve()),
        "total": len(records),
        "covered": assigned,
        "unique_ids": len(seen),
        "mutually_exclusive": True,
        "counts": {tier: len(values) for tier, values in tier_records.items()},
        "ready_for_main_experiment": sum(bool(item["ready_for_main_experiment"]) for item in assignments),
        "by_subtype": dict(sorted(Counter(item["subtype"] for item in assignments).items())),
        "by_required_backend": dict(sorted(Counter(item["required_backend"] for item in assignments).items())),
        "by_tier_and_grade": {
            tier: dict(sorted(Counter(str(row.get("grade") or "unknown") for row in values).items()))
            for tier, values in tier_records.items()
        },
        "by_tier_and_question_type": {
            tier: dict(sorted(Counter(str(row.get("question_type") or "unknown") for row in values).items()))
            for tier, values in tier_records.items()
        },
        "unknown_family_ids": [
            item["problem_id"] for item in assignments if item["subtype"] == "manual_rule_engine"
        ],
    }
    write_json(output / "tier_report.json", report)
    checksums = {path.name: _sha256(path) for path in [*paths.values(), assignment_path]}
    checksums[source.name] = _sha256(source)
    write_json(output / "checksums.json", checksums)
    (output / "README.md").write_text(
        "# Physics execution tiers\n\n"
        "The three JSONL files form a mutually exclusive and collectively exhaustive partition "
        "of the human-selected physics benchmark. `native` uses Three.js/Cannon, `equation` uses "
        "Boris charged-particle or restricted-expression RK4 equation solvers, and `specialized` uses deterministic "
        "DC-circuit or geometric ray-optics solvers. All browser renderers still require "
        "Node.js, Puppeteer, and FFmpeg; solver traces can be generated and validated offline.\n",
        encoding="utf-8",
    )
    return report
