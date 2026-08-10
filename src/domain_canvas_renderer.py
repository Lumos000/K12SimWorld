"""Renderer wrapper for trusted K12 equation/circuit/optics Canvas scenes."""

from __future__ import annotations

import os

from .canvas_html_renderer import CanvasHtmlRenderer


class DomainCanvasRenderer(CanvasHtmlRenderer):
    def __init__(self, output_dir: str = "./output") -> None:
        assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets", "threejs")
        super().__init__(output_dir=output_dir, assets_dir=assets_dir, label="Domain Canvas")
