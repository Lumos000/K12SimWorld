"""Model and renderer adapters. Imports existing VisPhyWorld code lazily."""

from __future__ import annotations

import base64
import mimetypes
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from .models import K12Problem, RenderSpec


@dataclass
class ModelResponse:
    text: str
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    logs: List[Dict[str, Any]] = field(default_factory=list)


class ModelAdapter(Protocol):
    model_name: str

    def generate(self, prompt: str, problem: K12Problem, call_index: int, attempt: int) -> ModelResponse:
        ...


class StaticResponseAdapter:
    """Deterministic adapter for tests and protocol debugging."""

    def __init__(self, responses: List[str], model_name: str = "static-test-model") -> None:
        self.responses = list(responses)
        self.model_name = model_name

    def generate(self, prompt: str, problem: K12Problem, call_index: int, attempt: int) -> ModelResponse:
        if not self.responses:
            raise RuntimeError("StaticResponseAdapter has no response left")
        return ModelResponse(self.responses.pop(0))


class VisPhyLLMAdapter:
    """Use the repository's existing multi-provider LLMClient without changing it."""

    def __init__(self, model_name: str) -> None:
        from src.llm_client import LLMClient

        self.model_name = model_name
        self.client = LLMClient(model_name)

    def generate(self, prompt: str, problem: K12Problem, call_index: int, attempt: int) -> ModelResponse:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        image_reference: List[Dict[str, Any]] = []
        image = self._image_source(problem.image)
        if image:
            content.append({"type": "image", "source": image})
            image_reference.append({"problem_id": problem.problem_id, "media_type": image["media_type"]})
        started = time.monotonic()
        text, logs = self.client.call(
            messages=[{"role": "user", "content": content}],
            prompt=prompt,
            call_index=call_index,
            pipeline_attempt=attempt,
            request_context={"problem_id": problem.problem_id, "task": "k12simworld"},
            image_references=image_reference,
        )
        input_tokens = self._sum_usage(logs, ("prompt_tokens", "input_tokens"))
        output_tokens = self._sum_usage(logs, ("completion_tokens", "output_tokens"))
        return ModelResponse(
            text=text,
            latency_seconds=time.monotonic() - started,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            logs=logs,
        )

    @staticmethod
    def _sum_usage(logs: List[Dict[str, Any]], keys: tuple[str, ...]) -> int:
        total = 0
        for entry in logs:
            candidates = [entry, entry.get("usage", {}), entry.get("response", {}).get("usage", {})]
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                for key in keys:
                    value = candidate.get(key)
                    if isinstance(value, (int, float)):
                        total += int(value)
                        break
        return total

    @staticmethod
    def _image_source(raw: Optional[str]) -> Optional[Dict[str, str]]:
        if not raw:
            return None
        # Avoid treating a raw base64 image as a filesystem path (which can exceed
        # the OS filename limit before ``is_file`` returns).
        if len(raw) < 4096 and not raw.startswith("data:image/"):
            path = Path(raw)
            try:
                if path.is_file():
                    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
                    data = base64.b64encode(path.read_bytes()).decode("ascii")
                    return {"type": "base64", "media_type": media_type, "data": data}
            except OSError:
                pass
        if raw.startswith("data:image/") and ";base64," in raw:
            header, data = raw.split(",", 1)
            return {"type": "base64", "media_type": header[5:].split(";", 1)[0], "data": data}
        # K12Vista/EduIllustrate commonly stores raw JPEG base64 in `img`.
        if len(raw) > 256 and all(char.isalnum() or char in "+/=\n\r" for char in raw[:256]):
            return {"type": "base64", "media_type": "image/jpeg", "data": raw}
        return None


class VisPhyRendererAdapter:
    def __init__(self, engine: str, output_dir: str, render_spec: Optional[RenderSpec] = None) -> None:
        self.render_spec = render_spec or RenderSpec(engine=engine)
        if engine == "threejs-cannon":
            from src.threejs_renderer import ThreeJSRenderer

            self.renderer = ThreeJSRenderer(output_dir)
            self.content_type = "html"
        elif engine == "p5js":
            from src.p5js_renderer import P5JSRenderer

            self.renderer = P5JSRenderer(output_dir)
            self.content_type = "html"
        elif engine == "manim":
            from src.manim_renderer import ManimRenderer

            self.renderer = ManimRenderer(output_dir)
            self.content_type = "python"
        elif engine in {"equation-solver", "circuit-solver", "ray-optics"}:
            from src.domain_canvas_renderer import DomainCanvasRenderer

            self.renderer = DomainCanvasRenderer(output_dir)
            self.content_type = "html"
        else:
            raise ValueError(f"unsupported engine: {engine}")
        if hasattr(self.renderer, "target"):
            from src.video_normalizer import VideoTarget

            self.renderer.target = VideoTarget(
                width=self.render_spec.width,
                height=self.render_spec.height,
                fps=float(self.render_spec.fps),
                duration_s=float(self.render_spec.duration),
            )

    def render(self, document: str, output_path: str) -> str:
        if self.content_type == "html":
            document = self._inject_browser_runtime(document)
        return self.renderer.render(document, output_path, content_type=self.content_type)

    def _inject_browser_runtime(self, document: str) -> str:
        """Inject only repository-local dependencies and the recording guard."""
        scripts = []
        label = getattr(self.renderer, "label", "")
        if label == "Three.js":
            scripts.extend(
                ['<script src="three.min.js"></script>', '<script src="cannon.min.js"></script>']
            )
        else:
            # P5.js routes may use the raw Canvas API; Cannon remains available for
            # deterministic non-visual state updates without a CDN dependency.
            if label != "Domain Canvas":
                scripts.append('<script src="cannon.min.js"></script>')
        scripts.append('<script src="recording.js"></script>')
        scripts.append(
            """<script>
window.addEventListener('load', () => {
  const start = () => {
    if (window.__k12simRecording || typeof setupRecording !== 'function') return;
    const canvas = document.querySelector('canvas');
    if (!canvas) return;
    window.__k12simRecording = true;
    setupRecording(canvas, %d);
  };
  [0, 250, 750, 1500].forEach(delay => setTimeout(start, delay));
});
</script>""" % int(self.render_spec.duration * 1000)
        )
        injection = "\n".join(scripts)
        if "</head>" in document.lower():
            index = document.lower().find("</head>")
            return document[:index] + injection + "\n" + document[index:]
        html_tag = re.search(r"<html\b[^>]*>", document, flags=re.IGNORECASE)
        if html_tag:
            index = html_tag.end()
            return document[:index] + f"<head>{injection}</head>" + document[index:]
        return f"<!doctype html><html><head>{injection}</head><body>{document}</body></html>"
