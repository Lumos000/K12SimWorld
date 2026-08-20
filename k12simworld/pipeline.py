"""State-anchored K12SimWorld generation pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .adapters import ModelAdapter, ModelResponse, VisPhyRendererAdapter
from .candidate_constraints import (
    CandidateSolution,
    SimulationContract,
    build_observed_trace,
    validate_candidate_contract,
    validation_error_messages,
)
from .domain_compiler import compile_domain_program
from .domain_solvers import DOMAIN_ENGINES
from .formula_normalization import normalize_world_spec_formulas
from .io import safe_artifact_name, write_json
from .models import ArtifactManifest, EduWorldSpec, K12Problem, RenderSpec, StoryBlock
from .prompts import (
    execution_repair_prompt,
    program_prompt,
    repair_prompt,
    storyboard_prompt,
    target_repair_prompt,
    world_spec_repair_prompt,
    world_spec_prompt,
)
from .routing import EngineRouter
from .validation import (
    parse_json_object,
    validate_program_payload,
    validate_storyboard,
    validate_world_spec,
)


class K12SimWorldPipeline:
    def __init__(
        self,
        model: ModelAdapter,
        output_dir: str | Path,
        *,
        render: bool = False,
        router: Optional[EngineRouter] = None,
    ) -> None:
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.render_enabled = render
        self.router = router or EngineRouter()
        self._call_index = 0

    def generate(self, problem: K12Problem, requested_engine: Optional[str] = None) -> ArtifactManifest:
        started = time.monotonic()
        run_dir = self.output_dir / safe_artifact_name(problem.problem_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = ArtifactManifest(
            problem_id=problem.problem_id,
            model=self.model.model_name,
            method="k12simworld_candidate_target",
            success=False,
            metadata={
                "subject": problem.subject,
                "grade": problem.grade,
                "question_type": problem.question_type,
                "knowledge_points": problem.knowledge_points,
                "simulation_type": problem.simulation_type,
                "split": problem.source_metadata.get("split"),
                "generation_mode": "candidate_target_conditioned",
            },
        )
        responses: List[ModelResponse] = []

        def pre_render_warning(code: str, message: str) -> None:
            manifest.diagnostics.setdefault("pre_render_warnings", []).append(
                {"code": code, "message": message}
            )

        try:
            story_response = self._call(storyboard_prompt(problem), problem, 0)
            responses.append(story_response)
            story_payload = parse_json_object(story_response.text)
            model_requires_simulation = bool(story_payload.get("requires_simulation", False))
            manifest.diagnostics["model_requires_simulation"] = model_requires_simulation
            if not model_requires_simulation:
                pre_render_warning(
                    "model_simulation_preference_overridden",
                    "the model marked simulation as unnecessary; generation continues by policy",
                )
            raw_story_blocks = story_payload.get("blocks") or []
            if not isinstance(raw_story_blocks, list):
                pre_render_warning("invalid_story_blocks_replaced", "storyboard blocks was not a list")
                raw_story_blocks = []
            blocks = [StoryBlock.from_dict(item) for item in raw_story_blocks]
            solution = CandidateSolution.from_story(problem.problem_id, story_payload)
            top_answer = str(story_payload.get("final_answer") or "").strip()
            if top_answer and top_answer != solution.final_answer:
                pre_render_warning(
                    "final_answer_disagreement",
                    "top-level final_answer differed from solution.final_answer; "
                    "the structured solution answer was selected",
                )
            story_payload["final_answer"] = solution.final_answer
            if not any(block.kind == "sim" for block in blocks):
                used_ids = {block.block_id for block in blocks}
                fallback_index = 1
                fallback_id = "SIM_AUTO_1"
                while fallback_id in used_ids:
                    fallback_index += 1
                    fallback_id = f"SIM_AUTO_{fallback_index}"
                fallback_block = StoryBlock(
                    block_id=fallback_id,
                    kind="sim",
                    content=(
                        "用声明式仿真展示题目初始条件、关键物理过程以及候选答案约束的最终状态。"
                    ),
                    learning_goal="将题目条件、物理过程与候选答案连接起来",
                )
                blocks.append(fallback_block)
                raw_story_blocks.append(
                    {
                        "block_id": fallback_block.block_id,
                        "kind": fallback_block.kind,
                        "content": fallback_block.content,
                        "learning_goal": fallback_block.learning_goal,
                    }
                )
                pre_render_warning(
                    "simulation_block_synthesized",
                    f"storyboard had no simulation block; added {fallback_id}",
                )
            story_payload["blocks"] = raw_story_blocks
            story_payload["solution"] = solution.to_dict()
            solution_path = run_dir / "solution_spec.json"
            write_json(solution_path, solution.to_dict())
            manifest.solution_spec_path = str(solution_path)
            storyboard_report = validate_storyboard(blocks)
            storyboard_report.require_valid()
            for warning in storyboard_report.warnings:
                pre_render_warning("storyboard_structure", warning)
            route = self.router.route(
                problem,
                requested_engine,
                visualization_decision=story_payload.get("visualization_decision"),
            )
            story_payload["visualization_audit"] = {
                "effective_mode": route.visualization_mode,
                "selected_engine": route.engine,
                **dict(route.spatial_audit or {}),
            }
            story_path = run_dir / "storyboard.json"
            write_json(story_path, story_payload)
            manifest.storyboard_path = str(story_path)
            manifest.engine = route.engine
            manifest.metadata["visualization_mode"] = route.visualization_mode
            manifest.metadata["spatial_audit"] = dict(route.spatial_audit or {})
            spec_prompt = world_spec_prompt(problem, solution.to_dict(), blocks, route)
            spec_response = self._call(
                spec_prompt,
                problem,
                0,
            )
            responses.append(spec_response)
            self._write_raw_attempt(run_dir, "world_spec", 1, spec_response.text)
            spec_changes: List[str] = []
            spec_repair_response: Optional[ModelResponse] = None
            try:
                first_payload = parse_json_object(spec_response.text)
                normalized_payload, spec_changes = normalize_world_spec_formulas(first_payload)
                spec = self._validated_world_spec_payload(
                    normalized_payload, problem.problem_id
                )
            except Exception as first_spec_error:
                spec_repair_response = self._call(
                    world_spec_repair_prompt(
                        spec_prompt,
                        spec_response.text,
                        f"{type(first_spec_error).__name__}: {first_spec_error}",
                        spec_changes,
                    ),
                    problem,
                    1,
                )
                responses.append(spec_repair_response)
                self._write_raw_attempt(run_dir, "world_spec", 2, spec_repair_response.text)
                manifest.repaired = True
                manifest.attempts = 2
                replacement_payload = parse_json_object(spec_repair_response.text)
                normalized_replacement, replacement_changes = normalize_world_spec_formulas(
                    replacement_payload
                )
                spec_changes.extend(replacement_changes)
                spec = self._validated_world_spec_payload(
                    normalized_replacement, problem.problem_id
                )
            manifest.diagnostics["world_spec_attempts"] = (
                2 if spec_repair_response is not None else 1
            )
            manifest.diagnostics["world_spec_formula_normalizations"] = spec_changes
            spec_path = run_dir / "world_spec.json"
            write_json(spec_path, spec.to_dict())
            manifest.world_spec_path = str(spec_path)
            contract = SimulationContract.from_world_spec(solution, spec)
            contract_path = run_dir / "simulation_contract.json"
            write_json(contract_path, contract.to_dict())
            manifest.simulation_contract_path = str(contract_path)
            if route.engine in DOMAIN_ENGINES and not contract.evaluable:
                pre_render_warning(
                    "candidate_contract_not_evaluable",
                    "domain simulation has no target_observables or invariants; rendering continues",
                )

            code_prompt = program_prompt(
                problem, blocks, spec, route, contract.to_dict()
            )
            program_response = self._call(code_prompt, problem, 0)
            responses.append(program_response)
            self._write_raw_attempt(run_dir, "program", 1, program_response.text)
            program_payload, program_repaired, repair_response = self._validated_program(
                program_response.text, code_prompt, problem, spec, blocks, route.engine
            )
            if repair_response:
                responses.append(repair_response)
                self._write_raw_attempt(run_dir, "program", 2, repair_response.text)
            manifest.repaired = manifest.repaired or program_repaired
            manifest.attempts = max(manifest.attempts, 2 if program_repaired else 1)
            manifest.diagnostics["program_validation_attempts"] = (
                2 if program_repaired else 1
            )
            program_path = run_dir / "program.json"
            write_json(program_path, program_payload)
            manifest.program_path = str(program_path)
            manifest.trace_paths = self._write_domain_traces(program_payload, run_dir)
            if route.engine in DOMAIN_ENGINES:
                effective_raw = repair_response.text if repair_response else program_response.text
                program_payload, target_response, target_attempts = (
                    self._validated_candidate_targets(
                        program_payload,
                        effective_raw,
                        code_prompt,
                        problem,
                        spec,
                        blocks,
                        contract,
                    )
                )
                if target_response:
                    responses.append(target_response)
                    self._write_raw_attempt(run_dir, "target_repair", 1, target_response.text)
                    manifest.repaired = True
                    manifest.attempts = max(manifest.attempts, 2)
                    write_json(program_path, program_payload)
                    manifest.trace_paths = self._write_domain_traces(program_payload, run_dir)
                manifest.diagnostics["candidate_target_attempts"] = len(target_attempts)

                observed = build_observed_trace(problem.problem_id, program_payload)
                observed_path = run_dir / "observed_trace.json"
                write_json(observed_path, observed)
                manifest.observed_trace_path = str(observed_path)

                final_target_report = dict(target_attempts[-1])
                final_target_report["attempts"] = target_attempts
                target_path = run_dir / "target_validation.json"
                write_json(target_path, final_target_report)
                manifest.target_validation_path = str(target_path)
                manifest.diagnostics["candidate_target_status"] = final_target_report["status"]
                manifest.diagnostics["candidate_constraint_scores"] = dict(
                    final_target_report.get("scores") or {}
                )
                if not final_target_report.get("passed"):
                    messages = validation_error_messages(final_target_report)
                    pre_render_warning(
                        "candidate_target_validation_failed",
                        "candidate target validation did not pass; rendering continues: "
                        + "; ".join(messages),
                    )
            else:
                target_path = run_dir / "target_validation.json"
                write_json(
                    target_path,
                    {
                        "problem_id": problem.problem_id,
                        "source": "candidate_solution",
                        "status": "not_supported",
                        "passed": None,
                        "reason": (
                            "the selected free-code renderer does not emit a trusted state trace"
                        ),
                    },
                )
                manifest.target_validation_path = str(target_path)
                manifest.diagnostics["candidate_target_status"] = "not_supported"

            if self.render_enabled:
                manifest.diagnostics["render_attempts"] = 1
                try:
                    manifest.video_paths = self._render(program_payload, run_dir)
                except Exception as render_error:
                    first_feedback = self._execution_feedback(render_error, run_dir)
                    self._write_raw_attempt(run_dir, "render_feedback", 1, first_feedback)
                    try:
                        execution_response = self._call(
                            execution_repair_prompt(
                                code_prompt,
                                json.dumps(program_payload, ensure_ascii=False),
                                first_feedback,
                            ),
                            problem,
                            1,
                        )
                    except Exception as repair_error:
                        raise RuntimeError(
                            "rendering failed before execution-aware repair: "
                            f"{first_feedback}; repair model call failed: "
                            f"{type(repair_error).__name__}: {repair_error}"
                        ) from render_error
                    responses.append(execution_response)
                    self._write_raw_attempt(
                        run_dir, "execution_repair", 1, execution_response.text
                    )
                    replacement = compile_domain_program(
                        parse_json_object(execution_response.text), spec
                    )
                    if str(replacement.get("engine") or "") != route.engine:
                        raise ValueError(
                            f"execution-repaired program engine must be {route.engine!r}"
                        )
                    report = validate_program_payload(replacement, spec, blocks)
                    report.require_valid()
                    replacement["validation_warnings"] = report.warnings

                    if route.engine in DOMAIN_ENGINES:
                        execution_target_report = validate_candidate_contract(
                            contract, replacement
                        )
                        execution_target_path = (
                            run_dir / "target_validation_execution_repair.json"
                        )
                        write_json(execution_target_path, execution_target_report)
                        manifest.target_validation_path = str(execution_target_path)
                        if not execution_target_report.get("passed"):
                            messages = validation_error_messages(execution_target_report)
                            pre_render_warning(
                                "execution_repair_target_validation_failed",
                                "execution repair violated candidate targets; rendering continues: "
                                + "; ".join(messages),
                            )
                        observed = build_observed_trace(problem.problem_id, replacement)
                        observed_path = run_dir / "observed_trace.json"
                        write_json(observed_path, observed)
                        manifest.observed_trace_path = str(observed_path)

                    program_payload = replacement
                    manifest.repaired = True
                    manifest.attempts = max(manifest.attempts, 2)
                    manifest.diagnostics["execution_repair_attempts"] = 1
                    write_json(program_path, program_payload)
                    manifest.trace_paths = self._write_domain_traces(program_payload, run_dir)
                    manifest.diagnostics["render_attempts"] = 2
                    try:
                        manifest.video_paths = self._render(program_payload, run_dir)
                    except Exception as second_render_error:
                        second_feedback = self._execution_feedback(
                            second_render_error, run_dir
                        )
                        self._write_raw_attempt(
                            run_dir, "render_feedback", 2, second_feedback
                        )
                        raise RuntimeError(
                            "rendering failed after one execution-aware repair; "
                            f"first failure: {first_feedback}; "
                            f"second failure: {second_feedback}"
                        ) from second_render_error
            document_path = run_dir / "explanation.md"
            document_path.write_text(
                self._assemble_document(problem, story_payload, blocks, manifest.video_paths),
                encoding="utf-8",
            )
            manifest.document_path = str(document_path)
            manifest.success = True
        except Exception as exc:
            manifest.error = f"{type(exc).__name__}: {exc}"
        finally:
            manifest.latency_seconds = time.monotonic() - started
            manifest.input_tokens = sum(response.input_tokens for response in responses)
            manifest.output_tokens = sum(response.output_tokens for response in responses)
            manifest.estimated_cost_usd = sum(response.estimated_cost_usd for response in responses)
            manifest.diagnostics["model_calls"] = len(responses)
            manifest_path = run_dir / "manifest.json"
            write_json(manifest_path, manifest.to_dict())
        return manifest

    def _call(self, prompt: str, problem: K12Problem, attempt: int) -> ModelResponse:
        self._call_index += 1
        return self.model.generate(prompt, problem, self._call_index, attempt)

    @staticmethod
    def _validated_world_spec_payload(
        payload: Dict[str, Any], problem_id: str
    ) -> EduWorldSpec:
        spec = EduWorldSpec.from_dict(payload)
        if spec.problem_id != problem_id:
            raise ValueError("EduWorldSpec problem_id does not match input")
        validate_world_spec(spec).require_valid()
        return spec

    @staticmethod
    def _write_raw_attempt(
        run_dir: Path, stage: str, attempt: int, content: str
    ) -> str:
        safe_stage = safe_artifact_name(stage)
        target = run_dir / "attempts" / f"{safe_stage}_attempt_{attempt}.txt"
        target.parent.mkdir(exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return str(target)

    @staticmethod
    def _execution_feedback(error: Exception, run_dir: Path) -> str:
        """Build bounded feedback from an exception and its trusted renderer log."""
        parts = [f"{type(error).__name__}: {error}"]
        root = run_dir.resolve()
        seen_paths: set[Path] = set()
        current: Optional[BaseException] = error
        for _ in range(3):
            if current is None:
                break
            raw_log_path = getattr(current, "log_path", None)
            if raw_log_path:
                try:
                    log_path = Path(str(raw_log_path)).resolve()
                    if log_path not in seen_paths and root in log_path.parents:
                        seen_paths.add(log_path)
                        with log_path.open("rb") as handle:
                            handle.seek(0, 2)
                            size = handle.tell()
                            handle.seek(max(0, size - 65536))
                            tail = handle.read().decode("utf-8", errors="replace")
                        tail_lines = tail.splitlines()[-80:]
                        if tail_lines:
                            parts.append(
                                f"renderer log tail ({log_path.name}):\n"
                                + "\n".join(tail_lines)
                            )
                except OSError as log_error:
                    parts.append(f"renderer log unavailable: {log_error}")
            current = current.__cause__ or current.__context__
        return "\n\n".join(parts)[:30000]

    def _validated_program(
        self,
        raw: str,
        original_prompt: str,
        problem: K12Problem,
        spec: EduWorldSpec,
        blocks: List[StoryBlock],
        expected_engine: str,
    ) -> Tuple[Dict[str, Any], bool, Optional[ModelResponse]]:
        try:
            payload = compile_domain_program(parse_json_object(raw), spec)
            if str(payload.get("engine") or "") != expected_engine:
                raise ValueError(f"program engine must be {expected_engine!r}")
            report = validate_program_payload(payload, spec, blocks)
            report.require_valid()
            payload["validation_warnings"] = report.warnings
            return payload, False, None
        except Exception as first_error:
            response = self._call(
                repair_prompt(original_prompt, raw, [str(first_error)]), problem, 1
            )
            payload = compile_domain_program(parse_json_object(response.text), spec)
            if str(payload.get("engine") or "") != expected_engine:
                raise ValueError(f"repaired program engine must be {expected_engine!r}")
            report = validate_program_payload(payload, spec, blocks)
            report.require_valid()
            payload["validation_warnings"] = report.warnings
            return payload, True, response

    def _validated_candidate_targets(
        self,
        payload: Dict[str, Any],
        raw: str,
        original_prompt: str,
        problem: K12Problem,
        spec: EduWorldSpec,
        blocks: List[StoryBlock],
        contract: SimulationContract,
    ) -> Tuple[Dict[str, Any], Optional[ModelResponse], List[Dict[str, Any]]]:
        first_report = validate_candidate_contract(contract, payload)
        process_warnings = [
            str(item) for item in payload.get("validation_warnings") or [] if str(item).strip()
        ]
        if process_warnings:
            first_report["process_fidelity_warnings"] = process_warnings
        attempts = [first_report]
        if first_report.get("passed") or first_report.get("status") == "not_evaluable":
            return payload, None, attempts

        response = self._call(
            target_repair_prompt(original_prompt, raw, first_report), problem, 1
        )
        try:
            replacement = compile_domain_program(parse_json_object(response.text), spec)
            if str(replacement.get("engine") or "") != str(payload.get("engine") or ""):
                raise ValueError("target repair changed the selected engine")
            structural = validate_program_payload(replacement, spec, blocks)
            structural.require_valid()
            replacement["validation_warnings"] = structural.warnings
            attempts.append(validate_candidate_contract(contract, replacement))
            return replacement, response, attempts
        except Exception as exc:
            attempts.append(
                {
                    "schema_version": "1.0",
                    "problem_id": contract.problem_id,
                    "source": contract.source,
                    "status": "failed",
                    "passed": False,
                    "checks": [
                        {
                            "id": "target_repair",
                            "kind": "repair",
                            "required": True,
                            "passed": False,
                            "error_message": f"{type(exc).__name__}: {exc}",
                        }
                    ],
                    "scores": dict(first_report.get("scores") or {}),
                }
            )
            return payload, response, attempts

    @staticmethod
    def _write_domain_traces(payload: Dict[str, Any], run_dir: Path) -> List[str]:
        if str(payload.get("engine") or "") not in DOMAIN_ENGINES:
            return []
        trace_dir = run_dir / "traces"
        trace_dir.mkdir(exist_ok=True)
        paths: List[str] = []
        for scene in payload.get("scenes") or []:
            if not isinstance(scene, dict) or not isinstance(scene.get("trace"), dict):
                continue
            target = trace_dir / f"{safe_artifact_name(str(scene.get('scene_id') or 'scene'))}.json"
            write_json(target, scene["trace"])
            paths.append(str(target))
        return paths

    def _render(self, payload: Dict[str, Any], run_dir: Path) -> List[str]:
        videos_dir = run_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        render_spec = RenderSpec.from_dict(payload["render_spec"])
        adapter = VisPhyRendererAdapter(
            str(payload["engine"]), str(run_dir / "renderer"), render_spec
        )
        results = []
        for scene in payload["scenes"]:
            target = videos_dir / f"{scene['scene_id']}.mp4"
            results.append(adapter.render(str(scene["document"]), str(target)))
        return results

    @staticmethod
    def _assemble_document(
        problem: K12Problem,
        story_payload: Dict[str, Any],
        blocks: List[StoryBlock],
        video_paths: List[str],
    ) -> str:
        videos_by_scene = {Path(video).stem: video for video in video_paths}
        used_video_ids: set[str] = set()
        lines = [
            f"# K12SimWorld explanation: {problem.problem_id}",
            "",
            f"**Problem:** {problem.question}",
            "",
        ]
        for block in blocks:
            if block.kind == "text":
                lines.extend([block.content, ""])
            else:
                lines.extend([f"**Simulation {block.block_id}:** {block.content}", ""])
                video = videos_by_scene.get(block.block_id)
                if video:
                    used_video_ids.add(block.block_id)
                    lines.extend([f"<video controls src=\"{video}\"></video>", ""])
        for scene_id, video in videos_by_scene.items():
            if scene_id in used_video_ids:
                continue
            lines.extend(
                [
                    f"**Additional simulation {scene_id}:**",
                    f"<video controls src=\"{video}\"></video>",
                    "",
                ]
            )
        lines.extend([f"**Candidate answer:** {story_payload.get('final_answer', '')}", ""])
        return "\n".join(lines)
