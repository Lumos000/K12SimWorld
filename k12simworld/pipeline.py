"""State-anchored K12SimWorld generation pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .adapters import ModelAdapter, ModelResponse, VisPhyRendererAdapter
from .domain_compiler import compile_domain_program
from .domain_solvers import DOMAIN_ENGINES
from .io import safe_artifact_name, write_json
from .models import ArtifactManifest, EduWorldSpec, K12Problem, RenderSpec, StoryBlock
from .prompts import program_prompt, repair_prompt, storyboard_prompt, world_spec_prompt
from .routing import EngineRouter
from .validation import parse_json_object, validate_program_payload, validate_storyboard


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
            method="k12simworld_state_anchored",
            success=False,
            metadata={
                "subject": problem.subject,
                "grade": problem.grade,
                "question_type": problem.question_type,
                "knowledge_points": problem.knowledge_points,
                "simulation_type": problem.simulation_type,
                "split": problem.source_metadata.get("split"),
            },
        )
        responses: List[ModelResponse] = []
        try:
            story_response = self._call(storyboard_prompt(problem), problem, 0)
            responses.append(story_response)
            story_payload = parse_json_object(story_response.text)
            if not story_payload.get("requires_simulation", False):
                raise ValueError("model judged simulation not pedagogically useful")
            blocks = [StoryBlock.from_dict(item) for item in story_payload.get("blocks", [])]
            validate_storyboard(blocks).require_valid()
            story_path = run_dir / "storyboard.json"
            write_json(story_path, story_payload)
            manifest.storyboard_path = str(story_path)

            route = self.router.route(problem, requested_engine)
            manifest.engine = route.engine
            spec_response = self._call(
                world_spec_prompt(problem, str(story_payload.get("analysis") or ""), blocks, route),
                problem,
                0,
            )
            responses.append(spec_response)
            spec = EduWorldSpec.from_dict(parse_json_object(spec_response.text))
            if spec.problem_id != problem.problem_id:
                raise ValueError("EduWorldSpec problem_id does not match input")
            spec_path = run_dir / "world_spec.json"
            write_json(spec_path, spec.to_dict())
            manifest.world_spec_path = str(spec_path)

            code_prompt = program_prompt(problem, blocks, spec, route)
            program_response = self._call(code_prompt, problem, 0)
            responses.append(program_response)
            program_payload, repaired, repair_response = self._validated_program(
                program_response.text, code_prompt, problem, spec, blocks, route.engine
            )
            if repair_response:
                responses.append(repair_response)
            manifest.repaired = repaired
            manifest.attempts = 2 if repaired else 1
            program_path = run_dir / "program.json"
            write_json(program_path, program_payload)
            manifest.program_path = str(program_path)
            manifest.trace_paths = self._write_domain_traces(program_payload, run_dir)

            if self.render_enabled:
                try:
                    manifest.video_paths = self._render(program_payload, run_dir)
                except Exception as render_error:
                    # Trusted domain HTML cannot be repaired by asking the model
                    # to rewrite it; renderer/dependency failures must stay
                    # explicit and must not consume another API call.
                    if repaired or str(program_payload.get("engine")) in DOMAIN_ENGINES:
                        raise
                    repair_response = self._call(
                        repair_prompt(
                            code_prompt,
                            json.dumps(program_payload, ensure_ascii=False),
                            [f"execution failed: {type(render_error).__name__}: {render_error}"],
                        ),
                        problem,
                        1,
                    )
                    responses.append(repair_response)
                    replacement = compile_domain_program(
                        parse_json_object(repair_response.text), spec
                    )
                    if str(replacement.get("engine") or "") != route.engine:
                        raise ValueError(
                            f"repaired program engine must be {route.engine!r}"
                        )
                    report = validate_program_payload(replacement, spec, blocks)
                    report.require_valid()
                    replacement["validation_warnings"] = report.warnings
                    program_payload = replacement
                    manifest.repaired = True
                    manifest.attempts = 2
                    write_json(program_path, program_payload)
                    manifest.trace_paths = self._write_domain_traces(program_payload, run_dir)
                    manifest.video_paths = self._render(program_payload, run_dir)
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
        videos = iter(video_paths)
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
                video = next(videos, None)
                if video:
                    lines.extend([f"<video controls src=\"{video}\"></video>", ""])
        lines.extend([f"**Candidate answer:** {story_payload.get('final_answer', '')}", ""])
        return "\n".join(lines)
