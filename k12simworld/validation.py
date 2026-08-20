"""Validation and lightweight safety checks for generated simulator programs."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from .domain_compiler import validate_compiled_domain_scene
from .domain_solvers import DOMAIN_ENGINES, domain_entity_ids
from .models import ContractError, EduWorldSpec, RenderSpec, StoryBlock


MAX_DOCUMENT_CHARS = 500_000
MAX_TRUSTED_DOCUMENT_CHARS = 5_000_000
JS_DENY_PATTERNS = {
    r"\bfetch\s*\(": "network fetch is not allowed",
    r"\bXMLHttpRequest\b": "network requests are not allowed",
    r"\bWebSocket\b": "network sockets are not allowed",
    r"\beval\s*\(": "eval is not allowed",
    r"\bnew\s+Function\b": "dynamic Function is not allowed",
    r"\brequire\s*\(": "Node require is not allowed in browser scenes",
    r"\bprocess\s*\.": "process access is not allowed",
    r"<(?:script|img|iframe)[^>]+src\s*=": "generated external resource sources are not allowed",
    r"\b(?:window\.)?location\s*=": "page navigation is not allowed",
    r"\.(?:position|rotation|scale|quaternion)\s*=\s*\[": (
        "Three.js vector/quaternion properties must not be replaced with arrays; use .set() or .copy()"
    ),
}
PY_DENY_NODES = (ast.Import, ast.ImportFrom)
PY_DENY_CALLS = {"eval", "exec", "compile", "open", "__import__"}
PY_ALLOW_MODULES = {"manim", "math", "numpy"}
UNIT_EXPECTATIONS = {
    "mass": {"kg", "g"},
    "radius": {"m", "cm", "mm"},
    "width": {"m", "cm", "mm"},
    "height": {"m", "cm", "mm"},
    "depth": {"m", "cm", "mm"},
}
AUXILIARY_OBJECT_TYPES = {"arrow", "bar", "bar_group", "curve", "label", "line", "point", "text", "text_label"}


@dataclass
class ValidationReport:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def require_valid(self) -> None:
        if not self.valid:
            raise ContractError("; ".join(self.errors))


def validate_storyboard(blocks: List[StoryBlock]) -> ValidationReport:
    errors: List[str] = []
    warnings: List[str] = []
    ids = [block.block_id for block in blocks]
    if not blocks:
        errors.append("storyboard is empty")
    if len(ids) != len(set(ids)):
        errors.append("storyboard block ids must be unique")
    sim_count = sum(block.kind == "sim" for block in blocks)
    if sim_count == 0:
        errors.append("storyboard must include at least one simulation block")
    for left, right in zip(blocks, blocks[1:]):
        if left.kind == right.kind:
            warnings.append(
                f"blocks {left.block_id} and {right.block_id} do not alternate; "
                "render order will follow the declared storyboard order"
            )
    return ValidationReport(not errors, errors, warnings)


def validate_world_spec(world_spec: EduWorldSpec) -> ValidationReport:
    """Check lightweight dimensional consistency before program generation."""
    errors: List[str] = []
    parameter_units = {
        str(item.get("id") or ""): str(item.get("unit") or "").strip()
        for item in world_spec.parameters
        if isinstance(item, Mapping)
    }
    for obj in world_spec.objects:
        object_id = str(obj.get("id") or "<unknown>")
        object_type = str(obj.get("type") or "").lower()
        properties = obj.get("properties")
        if not isinstance(properties, Mapping):
            continue
        shared_unit = str(properties.get("unit") or "").strip()
        for quantity, expected_units in UNIT_EXPECTATIONS.items():
            if quantity not in properties:
                continue
            if object_type in AUXILIARY_OBJECT_TYPES and quantity != "mass":
                continue
            explicit_unit = str(properties.get(f"{quantity}_unit") or shared_unit).strip()
            if explicit_unit and explicit_unit not in expected_units:
                errors.append(
                    f"object {object_id!r} {quantity} uses incompatible unit {explicit_unit!r}; "
                    f"expected one of {sorted(expected_units)}"
                )
        mass_parameter = parameter_units.get("m") or parameter_units.get("mass")
        if "mass" in properties and mass_parameter and shared_unit and shared_unit != mass_parameter:
            errors.append(
                f"object {object_id!r} mass unit {shared_unit!r} disagrees with parameter unit "
                f"{mass_parameter!r}"
            )
    return ValidationReport(not errors, errors)


def _browser_visibility_errors(document: str) -> List[str]:
    errors: List[str] = []
    backgrounds = {
        value.lower()
        for value in re.findall(
            r"(?:setClearColor|new\s+THREE\.Color)\s*\(\s*0x([0-9a-fA-F]{6})",
            document,
        )
    }
    materials = {
        value.lower()
        for value in re.findall(
            r"Material\s*\(\s*\{[^{}]*?color\s*:\s*0x([0-9a-fA-F]{6})",
            document,
            re.DOTALL,
        )
    }
    invisible = sorted(backgrounds & materials)
    if invisible:
        errors.append(f"foreground material matches background color: {invisible}")

    canvas_sizes: Dict[str, tuple[int, int]] = {}
    for name, width in re.findall(r"(\w+)\.width\s*=\s*(\d+)", document):
        canvas_sizes[name] = (int(width), canvas_sizes.get(name, (0, 0))[1])
    for name, height in re.findall(r"(\w+)\.height\s*=\s*(\d+)", document):
        canvas_sizes[name] = (canvas_sizes.get(name, (0, 0))[0], int(height))
    contexts = {
        context: canvas
        for context, canvas in re.findall(
            r"const\s+(\w+)\s*=\s*(\w+)\.getContext\s*\(\s*[\047\042]2d[\047\042]\s*\)",
            document,
        )
    }
    for context, x, y in re.findall(
        r"(\w+)\.fillText\s*\([^,]+,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)",
        document,
    ):
        canvas = contexts.get(context)
        width, height = canvas_sizes.get(canvas or "", (0, 0))
        if width and not 0 <= float(x) <= width:
            errors.append(f"text x={x} lies outside canvas {canvas} width {width}")
        if height and not 0 <= float(y) <= height:
            errors.append(f"text y={y} lies outside canvas {canvas} height {height}")
    return errors


def validate_program_payload(
    payload: Mapping[str, Any],
    world_spec: EduWorldSpec,
    storyboard: List[StoryBlock],
) -> ValidationReport:
    errors: List[str] = []
    warnings: List[str] = []
    expected_hash = world_spec.canonical_hash()
    if payload.get("world_spec_sha256") != expected_hash:
        errors.append("program does not declare the canonical EduWorldSpec hash")
    engine = str(payload.get("engine") or "")
    if engine not in {"threejs-cannon", "p5js", "manim", *DOMAIN_ENGINES}:
        errors.append(f"unsupported program engine {engine!r}")
    try:
        render_spec = RenderSpec.from_dict(payload.get("render_spec") or {})
        if render_spec.engine != engine:
            errors.append("render_spec.engine does not match program engine")
    except Exception as exc:
        errors.append(f"invalid render_spec: {exc}")
    scenes = payload.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        errors.append("program payload requires a non-empty scenes list")
        scenes = []
    expected_scene_ids = {block.block_id for block in storyboard if block.kind == "sim"}
    required_highlights = {
        block.block_id: set(block.highlights) for block in storyboard if block.kind == "sim"
    }
    simulation_narratives = {
        block.block_id: block.content.lower() for block in storyboard if block.kind == "sim"
    }
    actual_scene_id_list = [
        str(scene.get("scene_id")) for scene in scenes if isinstance(scene, Mapping)
    ]
    actual_scene_ids = set(actual_scene_id_list)
    if len(actual_scene_id_list) != len(actual_scene_ids):
        errors.append("scene ids must be unique so rendered files are not overwritten")
    missing_scene_ids = expected_scene_ids - actual_scene_ids
    extra_scene_ids = actual_scene_ids - expected_scene_ids
    if missing_scene_ids:
        warnings.append(
            "storyboard simulation blocks without a matching program scene: "
            f"{sorted(missing_scene_ids)}"
        )
    if extra_scene_ids:
        warnings.append(
            "program contains additional renderable scenes not declared by the storyboard: "
            f"{sorted(extra_scene_ids)}"
        )
    object_ids = [str(obj.get("id")) for obj in world_spec.objects]
    core_object_ids = [str(obj.get("id")) for obj in world_spec.objects if str(obj.get("type") or "").lower() not in AUXILIARY_OBJECT_TYPES]
    covered_object_ids: set[str] = set()
    for scene in scenes:
        if not isinstance(scene, Mapping):
            errors.append("scene entries must be objects")
            continue
        document = str(scene.get("document") or "")
        if engine in DOMAIN_ENGINES:
            errors.extend(
                f"{scene.get('scene_id')}: {item}"
                for item in validate_compiled_domain_scene(engine, scene)
            )
            simulation_spec = scene.get("simulation_spec")
            if isinstance(simulation_spec, Mapping):
                unknown_entities = domain_entity_ids(engine, simulation_spec) - set(object_ids)
                auxiliary_entities: set[str] = set()
                if engine == "mechanics-2d":
                    auxiliary_entities = {
                        str(item.get("id"))
                        for item in simulation_spec.get("static_geometry") or []
                        if isinstance(item, Mapping) and item.get("id")
                    }
                unknown_auxiliary = unknown_entities & auxiliary_entities
                unknown_core = unknown_entities - unknown_auxiliary
                if unknown_core:
                    errors.append(
                        f"{scene.get('scene_id')}: domain entities absent from EduWorldSpec: "
                        f"{sorted(unknown_core)}"
                    )
                if unknown_auxiliary:
                    warnings.append(
                        f"{scene.get('scene_id')}: auxiliary environment entities absent "
                        f"from EduWorldSpec: {sorted(unknown_auxiliary)}"
                    )
                if engine == "mechanics-2d":
                    scene_id = str(scene.get("scene_id") or "")
                    narrative = simulation_narratives.get(scene_id, "")
                    action_types = {
                        str(item.get("type") or "")
                        for item in simulation_spec.get("actions") or []
                        if isinstance(item, Mapping)
                    }
                    break_cues = (
                        "断绳", "剪断绳", "绳断", "解除约束", "脱离绳",
                        "cut the string", "string breaks", "rope breaks",
                        "remove the constraint", "detach from the rope",
                    )
                    if (
                        simulation_spec.get("distance_constraints")
                        and any(cue in narrative for cue in break_cues)
                        and "remove_distance_constraint" not in action_types
                    ):
                        warnings.append(
                            f"{scene_id}: storyboard describes release/break but the physical "
                            "trajectory never removes a distance constraint"
                        )
                    strategy = str(
                        simulation_spec.get("visual_strategy") or "continuous_process"
                    )
                    component_cues = (
                        "分解", "分量", "投影", "motion component", "vector component",
                        "coordinate projection",
                    )
                    if (
                        strategy == "component_decomposition"
                        and not any(cue in narrative for cue in component_cues)
                    ):
                        warnings.append(
                            f"{scene_id}: component panels were requested without an explicit "
                            "component-decomposition teaching goal"
                        )
                    if action_types and not simulation_spec.get("phases"):
                        warnings.append(
                            f"{scene_id}: physical interventions are present but no phases label "
                            "the complete before/after process"
                        )
        report = validate_document(
            document, engine, trusted_compiled=engine in DOMAIN_ENGINES
        )
        errors.extend(f"{scene.get('scene_id')}: {item}" for item in report.errors)
        warnings.extend(f"{scene.get('scene_id')}: {item}" for item in report.warnings)
        if engine not in DOMAIN_ENGINES:
            covered_object_ids.update(object_id for object_id in object_ids if object_id in document)
            missing = [object_id for object_id in core_object_ids if object_id not in document]
            if missing:
                warnings.append(
                    f"{scene.get('scene_id')}: object ids absent from source: {missing[:8]}"
                )
            scene_id = str(scene.get("scene_id"))
            missing_highlights = [
                item for item in required_highlights.get(scene_id, set()) if item not in document
            ]
            if missing_highlights:
                warnings.append(
                    f"{scene_id}: highlighted object ids absent from source: {sorted(missing_highlights)[:8]}"
                )
            errors.extend(_browser_visibility_errors(document))
    if engine not in DOMAIN_ENGINES:
        missing_anywhere = [object_id for object_id in object_ids if object_id not in covered_object_ids]
        if missing_anywhere:
            warnings.append(f"WorldSpec object ids absent from every scene: {missing_anywhere[:8]}")
    return ValidationReport(not errors, errors, warnings)


def validate_document(
    document: str, engine: str, *, trusted_compiled: bool = False
) -> ValidationReport:
    errors: List[str] = []
    warnings: List[str] = []
    if not document.strip():
        return ValidationReport(False, ["empty program document"])
    document_limit = (
        MAX_TRUSTED_DOCUMENT_CHARS if trusted_compiled else MAX_DOCUMENT_CHARS
    )
    if len(document) > document_limit:
        errors.append(f"document exceeds {document_limit} characters")
    elif trusted_compiled and len(document) > MAX_DOCUMENT_CHARS:
        warnings.append(
            f"trusted compiler document exceeds the general {MAX_DOCUMENT_CHARS}-character limit"
        )
    if engine in {"threejs-cannon", "p5js", *DOMAIN_ENGINES}:
        if "<canvas" not in document and "createCanvas" not in document and "WebGLRenderer" not in document:
            errors.append("browser simulation does not create a canvas")
        for pattern, message in JS_DENY_PATTERNS.items():
            if re.search(pattern, document):
                errors.append(message)
        if engine == "threejs-cannon" and "CANNON" not in document:
            warnings.append("Three.js mechanics scene does not reference CANNON")
    elif engine == "manim":
        try:
            tree = ast.parse(document)
        except SyntaxError as exc:
            return ValidationReport(False, [f"invalid Python syntax: {exc.msg} at line {exc.lineno}"])
        for node in ast.walk(tree):
            if isinstance(node, PY_DENY_NODES):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif node.module:
                    names = [node.module.split(".")[0]]
                blocked = set(names) - PY_ALLOW_MODULES
                if blocked:
                    errors.append(f"non-whitelisted Python module import: {sorted(blocked)}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in PY_DENY_CALLS:
                errors.append(f"blocked Python call: {node.func.id}")
        valid_scene = any(
            isinstance(node, ast.ClassDef)
            and node.name == "GeneratedScene"
            and any(
                (isinstance(base, ast.Name) and base.id == "Scene")
                or (isinstance(base, ast.Attribute) and base.attr == "Scene")
                for base in node.bases
            )
            for node in ast.walk(tree)
        )
        if not valid_scene:
            errors.append("Manim program must define class GeneratedScene")
    return ValidationReport(not errors, errors, warnings)


def parse_json_object(text: str) -> Dict[str, Any]:
    """Parse a JSON object from raw or fenced model output."""
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ContractError("model output must be a JSON object")
    return value
