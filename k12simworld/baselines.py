"""Controlled baseline generation with the same model and manifest contracts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

from .adapters import ModelAdapter, VisPhyRendererAdapter
from .domain_solvers import DOMAIN_ENGINES
from .io import safe_artifact_name, write_json
from .models import ArtifactManifest, K12Problem, RenderSpec
from .routing import EngineRouter
from .validation import parse_json_object, validate_document


BASELINE_METHODS = {"text_cot", "static_manim", "direct_code", "unanchored"}


def baseline_prompt(problem: K12Problem, method: str, engine: str) -> str:
    data = json.dumps(problem.model_payload(), ensure_ascii=False, indent=2)
    if method == "text_cot":
        contract = """Return JSON only: {"analysis":"step-by-step explanation","final_answer":"..."}.
Do not generate diagrams, code, or a world specification."""
    elif method == "static_manim":
        contract = """Return JSON only with analysis, final_answer, and scenes. Each scene has
scene_id and document containing a complete Python Manim Scene. Produce static/progressive
diagram explanation scenes in EduIllustrate style; every document must define a class named
GeneratedScene; do not create an EduWorldSpec."""
    elif method == "direct_code":
        contract = f"""Return JSON only with analysis, final_answer, engine="{engine}", and scenes.
Each scene has scene_id and a complete executable document. Generate code directly from the
problem in one pass; do not create an EduWorldSpec or cross-scene anchor."""
    elif method == "unanchored":
        contract = f"""Return JSON only with analysis, final_answer, engine="{engine}", and scenes.
Each scene has scene_id and a complete executable document. Plan every scene independently;
do not inherit object ids, layout, style, parameter values, or code from another scene."""
    else:
        raise ValueError(f"unknown baseline method: {method}")
    return f"""You are a K-12 educational content generator implementing baseline `{method}`.
Use only the problem input; the gold solution is unavailable. Keep language grade appropriate.
No network, file access, subprocess, dynamic eval, or external assets are allowed. Browser
baselines must use the standard Canvas API; Three.js/Cannon.js are injected only when the
selected engine is threejs-cannon.

{contract}

Problem:
{data}
"""


class BaselinePipeline:
    def __init__(
        self,
        model: ModelAdapter,
        output_dir: str | Path,
        *,
        method: str,
        render: bool = False,
    ) -> None:
        if method not in BASELINE_METHODS:
            raise ValueError(f"method must be one of {sorted(BASELINE_METHODS)}")
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.method = method
        self.render_enabled = render
        self.router = EngineRouter()
        self.call_index = 0

    def generate(self, problem: K12Problem, requested_engine: Optional[str] = None) -> ArtifactManifest:
        started = time.monotonic()
        run_dir = self.output_dir / safe_artifact_name(problem.problem_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        engine = "manim" if self.method == "static_manim" else self.router.route(problem, requested_engine).engine
        # Direct-code baselines must not inherit the trusted domain solver; they
        # remain generic Canvas code so the comparison measures its contribution.
        if engine in DOMAIN_ENGINES:
            engine = "p5js"
        manifest = ArtifactManifest(
            problem_id=problem.problem_id,
            model=self.model.model_name,
            method=self.method,
            success=False,
            engine=None if self.method == "text_cot" else engine,
            metadata={
                "subject": problem.subject,
                "grade": problem.grade,
                "question_type": problem.question_type,
                "knowledge_points": problem.knowledge_points,
                "simulation_type": problem.simulation_type,
                "split": problem.source_metadata.get("split"),
            },
        )
        responses = []
        try:
            self.call_index += 1
            response = self.model.generate(
                baseline_prompt(problem, self.method, engine), problem, self.call_index, 0
            )
            responses.append(response)
            payload = parse_json_object(response.text)
            if not str(payload.get("analysis") or "").strip() or not str(payload.get("final_answer") or "").strip():
                raise ValueError("baseline output requires analysis and final_answer")
            scenes = payload.get("scenes", [])
            if self.method != "text_cot":
                if not isinstance(scenes, list) or not scenes:
                    raise ValueError("visual baseline requires at least one scene")
                for scene in scenes:
                    report = validate_document(str(scene.get("document") or ""), engine)
                    report.require_valid()
            output_path = run_dir / "baseline_output.json"
            write_json(output_path, payload)
            manifest.program_path = str(output_path) if self.method != "text_cot" else None
            videos: List[str] = []
            if self.render_enabled and self.method != "text_cot":
                adapter = VisPhyRendererAdapter(
                    engine, str(run_dir / "renderer"), RenderSpec(engine=engine)
                )
                videos_dir = run_dir / "videos"
                videos_dir.mkdir(exist_ok=True)
                for index, scene in enumerate(scenes, 1):
                    scene_id = str(scene.get("scene_id") or f"SCENE_{index}")
                    videos.append(
                        adapter.render(str(scene["document"]), str(videos_dir / f"{scene_id}.mp4"))
                    )
            manifest.video_paths = videos
            document_path = run_dir / "explanation.md"
            lines = [f"# {self.method}: {problem.problem_id}", "", str(payload["analysis"]), ""]
            lines.extend(f"<video controls src=\"{video}\"></video>\n" for video in videos)
            lines.extend([f"**Candidate answer:** {payload['final_answer']}", ""])
            document_path.write_text("\n".join(lines), encoding="utf-8")
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
            write_json(run_dir / "manifest.json", manifest.to_dict())
        return manifest
