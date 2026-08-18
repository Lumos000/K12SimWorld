"""Deterministic engine routing with a conservative, auditable 3-D gate."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

from .models import K12Problem

MECHANICS_TERMS = {
    "mechanics", "collision", "balance", "lever", "pulley", "incline", "friction",
    "projectile", "buoyancy", "kinematics", "force_and_motion", "inclined_plane",
    "spring", "circular_motion", "energy", "力学", "碰撞", "平衡", "杠杆", "滑轮",
    "斜面", "摩擦", "浮力", "运动",
}
STATE_TERMS = {"circuit", "electric", "magnetic", "optics", "wave", "heat", "fluid", "reaction", "process", "cycle", "电路", "电磁", "光学", "波", "热学", "反应", "过程", "循环"}
MANIM_TERMS = {"geometry", "function", "graph", "construction", "几何", "函数", "作图"}
CIRCUIT_TERMS = {"circuit", "电路", "ammeter", "voltmeter", "lamp", "电流表", "电压表", "灯泡"}
OPTICS_TERMS = {"ray_optics", "ray optics", "光路", "透镜", "反射", "折射", "平面镜"}
EQUATION_TERMS = {"charged_particle", "electromagnetic_dynamics", "lorentz", "charged particle", "带电粒子", "洛伦兹", "电场", "磁场", "电磁场"}
NATIVE_SUBTYPES = {"kinematics", "projectile", "force_and_motion", "inclined_plane", "friction", "lever", "pulley", "spring", "collision", "circular_motion", "energy", "buoyancy"}
SUPPORTED_ENGINES = {"mechanics-2d", "threejs-cannon", "p5js", "manim", "equation-solver", "circuit-solver", "ray-optics"}

# A 3-D request is accepted only when its exact evidence quote occurs in the
# model-visible problem and contains a criterion-specific spatial cue.
SPATIAL_CRITERIA = {
    "non_coplanar_motion": {"不共面", "空间运动", "三维运动", "out of plane", "non-coplanar", "3d motion"},
    "depth_dependent_collision": {"深度方向", "前后碰撞", "空间碰撞", "depth", "spatial collision"},
    "spatial_rotation_axis": {"空间转轴", "三维转轴", "绕空间", "axis in space", "spatial axis"},
    "perspective_geometry_required": {"透视", "空间几何", "立体结构", "三维坐标", "perspective", "spatial geometry", "3d coordinates"},
    "occlusion_is_physics": {"遮挡", "遮蔽", "occlusion"},
    "multi_view_spatial_structure": {"俯视图", "侧视图", "正视图", "多视图", "three views", "multi-view"},
}


@dataclass(frozen=True)
class RouteDecision:
    engine: str
    simulation_type: str
    reason: str
    physics_native: bool
    visualization_mode: str = "schematic_2d"
    spatial_audit: Dict[str, Any] = field(default_factory=dict)


def audit_spatial_request(problem: K12Problem, decision: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Verify a model's 3-D request against exact, model-visible evidence."""
    raw = dict(decision or {})
    requested = str(raw.get("mode") or "schematic_2d").strip().lower()
    criterion = str(raw.get("criterion") or "").strip().lower()
    evidence = str(raw.get("evidence_quote") or "").strip()
    audit: Dict[str, Any] = {
        "requested_mode": requested,
        "criterion": criterion,
        "evidence_quote": evidence,
        "approved": False,
        "effective_mode": "schematic_2d",
        "reason": "2-D is the safe default",
    }
    if requested != "spatial_3d":
        audit["reason"] = "model did not request spatial_3d"
        return audit
    if criterion not in SPATIAL_CRITERIA:
        audit["reason"] = "3-D criterion is missing or not in the controlled vocabulary"
        return audit
    visible = (problem.question + "\n" + (problem.image_caption or "")).lower()
    if len(evidence) < 4 or evidence.lower() not in visible:
        audit["reason"] = "evidence_quote is not an exact quote from the question or image caption"
        return audit
    lowered = evidence.lower()
    if not any(cue in lowered for cue in SPATIAL_CRITERIA[criterion]):
        audit["reason"] = f"evidence_quote does not substantiate criterion {criterion!r}"
        return audit
    audit.update({"approved": True, "effective_mode": "spatial_3d", "reason": "exact spatial evidence passed the controlled 3-D gate"})
    return audit


class EngineRouter:
    def route(
        self,
        problem: K12Problem,
        requested: Optional[str] = None,
        visualization_decision: Optional[Mapping[str, Any]] = None,
    ) -> RouteDecision:
        if requested:
            engine = requested.strip().lower()
            if engine not in SUPPORTED_ENGINES:
                raise ValueError(f"unsupported requested engine: {requested}")
            mode = "spatial_3d" if engine == "threejs-cannon" else "schematic_2d"
            return RouteDecision(engine, problem.simulation_type or "manual", "explicit user/CLI override", engine in {"mechanics-2d", "threejs-cannon"}, mode, {"approved": True, "effective_mode": mode, "reason": "explicit user/CLI override"})

        execution_tier = problem.source_metadata.get("execution_tier") or {}
        tier_subtype = str(execution_tier.get("subtype") or "").lower()
        evidence = " ".join([tier_subtype, problem.simulation_type or "", problem.question, *problem.knowledge_points]).lower()
        if tier_subtype == "circuit":
            return RouteDecision("circuit-solver", "circuit", "Kirchhoff/Ohm-law circuit state solving", False)
        if tier_subtype == "ray_optics":
            return RouteDecision("ray-optics", "ray_optics", "deterministic geometric ray tracing", False)
        if tier_subtype == "electromagnetic_dynamics":
            return RouteDecision("equation-solver", "equation_dynamics", "Boris charged-particle or restricted-expression RK4 integration", False)
        if problem.simulation_type == "circuit":
            return RouteDecision("circuit-solver", "circuit", "Kirchhoff/Ohm-law circuit state solving", False)
        if problem.simulation_type == "ray_optics":
            return RouteDecision("ray-optics", "ray_optics", "deterministic geometric ray tracing", False)
        if problem.simulation_type == "charged_particle":
            return RouteDecision("equation-solver", "charged_particle", "fixed-step Lorentz-force equation integration", False)
        if self._contains(evidence, CIRCUIT_TERMS):
            return RouteDecision("circuit-solver", "circuit", "Kirchhoff/Ohm-law circuit state solving", False)
        if self._contains(evidence, OPTICS_TERMS):
            return RouteDecision("ray-optics", "ray_optics", "deterministic geometric ray tracing", False)
        if self._contains(evidence, EQUATION_TERMS):
            return RouteDecision("equation-solver", "equation_dynamics", "Boris charged-particle or restricted-expression RK4 integration", False)
        if problem.subject == "mathematics" or self._contains(evidence, MANIM_TERMS):
            return RouteDecision("manim", problem.simulation_type or "symbolic_geometry", "symbolic or geometric construction", False)
        if tier_subtype in NATIVE_SUBTYPES or self._contains(evidence, MECHANICS_TERMS):
            audit = audit_spatial_request(problem, visualization_decision)
            if audit["approved"]:
                return RouteDecision("threejs-cannon", problem.simulation_type or "rigid_body", "verified spatial necessity", True, "spatial_3d", audit)
            return RouteDecision("mechanics-2d", problem.simulation_type or "rigid_body", "declarative 2-D mechanics default; 3-D gate not approved", True, "schematic_2d", audit)
        if problem.subject in {"chemistry", "biology", "geography"}:
            return RouteDecision("p5js", problem.simulation_type or "rule_based_process", "cross-subject deterministic state transition", False)
        if self._contains(evidence, STATE_TERMS) or problem.subject == "physics":
            return RouteDecision("p5js", problem.simulation_type or "state_machine", "non-rigid physics process", False)
        return RouteDecision("p5js", problem.simulation_type or "rule_based_process", "safe deterministic fallback", False)

    @staticmethod
    def _contains(text: str, terms: Iterable[str]) -> bool:
        return any(term in text for term in terms)
