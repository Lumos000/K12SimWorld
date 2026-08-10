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
}
PY_DENY_NODES = (ast.Import, ast.ImportFrom)
PY_DENY_CALLS = {"eval", "exec", "compile", "open", "__import__"}
PY_ALLOW_MODULES = {"manim", "math", "numpy"}


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
            errors.append(f"blocks {left.block_id} and {right.block_id} do not alternate")
    return ValidationReport(not errors, errors)


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
    actual_scene_ids = {str(scene.get("scene_id")) for scene in scenes if isinstance(scene, Mapping)}
    if actual_scene_ids != expected_scene_ids:
        errors.append(
            f"scene ids differ from storyboard: expected {sorted(expected_scene_ids)}, got {sorted(actual_scene_ids)}"
        )
    object_ids = [str(obj.get("id")) for obj in world_spec.objects]
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
                if unknown_entities:
                    errors.append(
                        f"{scene.get('scene_id')}: domain entities absent from EduWorldSpec: "
                        f"{sorted(unknown_entities)}"
                    )
        report = validate_document(document, engine)
        errors.extend(f"{scene.get('scene_id')}: {item}" for item in report.errors)
        warnings.extend(f"{scene.get('scene_id')}: {item}" for item in report.warnings)
        missing = [object_id for object_id in object_ids if object_id not in document]
        if missing:
            warnings.append(
                f"{scene.get('scene_id')}: object ids absent from source: {missing[:8]}"
            )
    return ValidationReport(not errors, errors, warnings)


def validate_document(document: str, engine: str) -> ValidationReport:
    errors: List[str] = []
    warnings: List[str] = []
    if not document.strip():
        return ValidationReport(False, ["empty program document"])
    if len(document) > MAX_DOCUMENT_CHARS:
        errors.append(f"document exceeds {MAX_DOCUMENT_CHARS} characters")
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
