"""Deterministic subject/simulation-type to execution-engine routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .models import K12Problem


MECHANICS_TERMS = {
    "mechanics", "collision", "balance", "lever", "pulley", "incline",
    "friction", "projectile", "buoyancy", "kinematics", "force_and_motion",
    "inclined_plane", "spring", "circular_motion", "energy",
    "力学", "碰撞", "平衡", "杠杆",
    "滑轮", "斜面", "摩擦", "浮力", "运动",
}
STATE_TERMS = {
    "circuit", "electric", "magnetic", "optics", "wave", "heat", "fluid",
    "reaction", "process", "cycle", "电路", "电磁", "光学", "波", "热学",
    "反应", "过程", "循环",
}
MANIM_TERMS = {
    "geometry", "function", "graph", "construction", "几何", "函数", "作图",
}
CIRCUIT_TERMS = {"circuit", "电路", "ammeter", "voltmeter", "lamp", "电流表", "电压表", "灯泡"}
OPTICS_TERMS = {"ray_optics", "ray optics", "光路", "透镜", "反射", "折射", "平面镜"}
EQUATION_TERMS = {
    "charged_particle", "electromagnetic_dynamics", "lorentz", "charged particle",
    "带电粒子", "洛伦兹", "电场", "磁场", "电磁场",
}
SUPPORTED_ENGINES = {
    "threejs-cannon", "p5js", "manim",
    "equation-solver", "circuit-solver", "ray-optics",
}


@dataclass(frozen=True)
class RouteDecision:
    engine: str
    simulation_type: str
    reason: str
    physics_native: bool


class EngineRouter:
    def route(self, problem: K12Problem, requested: Optional[str] = None) -> RouteDecision:
        if requested:
            engine = requested.strip().lower()
            if engine not in SUPPORTED_ENGINES:
                raise ValueError(f"unsupported requested engine: {requested}")
            return RouteDecision(engine, problem.simulation_type or "manual", "explicit override", engine == "threejs-cannon")

        execution_tier = problem.source_metadata.get("execution_tier") or {}
        tier_subtype = str(execution_tier.get("subtype") or "").lower()
        evidence = " ".join(
            [tier_subtype, problem.simulation_type or "", problem.question, *problem.knowledge_points]
        ).lower()
        # A frozen human execution tier is authoritative.  Only when it is
        # absent may a family label or question wording decide the domain.
        if tier_subtype == "circuit":
            return RouteDecision("circuit-solver", "circuit", "Kirchhoff/Ohm-law circuit state solving", False)
        if tier_subtype == "ray_optics":
            return RouteDecision("ray-optics", "ray_optics", "deterministic geometric ray tracing", False)
        if tier_subtype == "electromagnetic_dynamics":
            return RouteDecision(
                "equation-solver", "equation_dynamics",
                "Boris charged-particle or restricted-expression RK4 integration", False,
            )
        # Domain solvers take precedence over broad words such as “运动”, which
        # otherwise misroute charged-particle motion to a rigid-body engine.
        if problem.simulation_type == "circuit":
            return RouteDecision("circuit-solver", "circuit", "Kirchhoff/Ohm-law circuit state solving", False)
        if problem.simulation_type == "ray_optics":
            return RouteDecision("ray-optics", "ray_optics", "deterministic geometric ray tracing", False)
        if problem.simulation_type == "charged_particle":
            return RouteDecision(
                "equation-solver", "charged_particle",
                "fixed-step Lorentz-force equation integration", False,
            )
        if self._contains(evidence, CIRCUIT_TERMS):
            return RouteDecision("circuit-solver", "circuit", "Kirchhoff/Ohm-law circuit state solving", False)
        if self._contains(evidence, OPTICS_TERMS):
            return RouteDecision("ray-optics", "ray_optics", "deterministic geometric ray tracing", False)
        if self._contains(evidence, EQUATION_TERMS):
            return RouteDecision(
                "equation-solver", "equation_dynamics",
                "Boris charged-particle or restricted-expression RK4 integration", False,
            )
        if problem.subject == "mathematics" or self._contains(evidence, MANIM_TERMS):
            return RouteDecision("manim", problem.simulation_type or "symbolic_geometry", "symbolic or geometric construction", False)
        if self._contains(evidence, MECHANICS_TERMS):
            return RouteDecision("threejs-cannon", problem.simulation_type or "rigid_body", "mechanics requires native constraints", True)
        if problem.subject in {"chemistry", "biology", "geography"}:
            return RouteDecision("p5js", problem.simulation_type or "rule_based_process", "cross-subject deterministic state transition", False)
        if self._contains(evidence, STATE_TERMS) or problem.subject == "physics":
            return RouteDecision("p5js", problem.simulation_type or "state_machine", "non-rigid physics process", False)
        return RouteDecision("p5js", problem.simulation_type or "rule_based_process", "safe deterministic fallback", False)

    @staticmethod
    def _contains(text: str, terms: Iterable[str]) -> bool:
        return any(term in text for term in terms)
