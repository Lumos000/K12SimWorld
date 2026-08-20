import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from k12simworld.adapters import VisPhyRendererAdapter
from k12simworld.models import RenderSpec
from src.canvas_html_renderer import CanvasHtmlRenderer


class CompositeCanvasTest(unittest.TestCase):
    def test_splits_mixed_webgl_and_2d_canvas(self):
        document = """<!doctype html><html><body><canvas id="canvas"></canvas><script>
const canvas = document.getElementById("canvas");
const renderer = new THREE.WebGLRenderer({canvas, antialias: true});
const ctx = canvas.getContext("2d");
ctx.fillText("force", 20, 20);
</script></body></html>"""

        result = VisPhyRendererAdapter._prepare_composite_canvas(document)

        self.assertIn("preserveDrawingBuffer: true", result)
        self.assertIn("data-k12-overlay", result)
        self.assertIn("data-k12-recording", result)
        self.assertIn("drawImage(canvas, 0, 0)", result)
        self.assertNotIn("const ctx = canvas.getContext", result)
        self.assertRegex(result, r"const ctx = __k12Overlay_canvas_\d+\.getContext")

    def test_keeps_separate_2d_overlay_unchanged(self):
        document = """<canvas id="webgl"></canvas><canvas id="overlay"></canvas><script>
const canvas = document.getElementById("webgl");
new THREE.WebGLRenderer({canvas});
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
</script>"""
        self.assertEqual(
            document,
            VisPhyRendererAdapter._prepare_composite_canvas(document),
        )

    def test_keeps_plain_2d_canvas_unchanged(self):
        document = """<canvas id="canvas"></canvas><script>
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
</script>"""
        self.assertEqual(
            document,
            VisPhyRendererAdapter._prepare_composite_canvas(document),
        )

    def test_recording_guard_prefers_composite_canvas(self):
        adapter = object.__new__(VisPhyRendererAdapter)
        adapter.renderer = SimpleNamespace(label="Three.js")
        adapter.render_spec = RenderSpec(engine="threejs-cannon", duration=2)

        result = adapter._inject_browser_runtime("<html><head></head><body></body></html>")

        composite_selector = "document.querySelector('[data-k12-recording=\"true\"]')"
        self.assertIn(composite_selector, result)
        self.assertIn("window.__k12simFastCaptureRequested", result)
        self.assertLess(result.index(composite_selector), result.index("document.querySelector('canvas')"))


class RenderModeTest(unittest.TestCase):
    def test_domain_canvas_uses_offline_frames_by_default(self):
        with tempfile.TemporaryDirectory() as output_dir:
            renderer = CanvasHtmlRenderer(output_dir, label="Domain Canvas")
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("K12SIMWORLD_RENDER_MODE", None)
                self.assertEqual(renderer._render_mode("html"), "frames")
                self.assertEqual(renderer._capture_fps(), 5.0)

    def test_free_code_canvas_keeps_realtime_recording(self):
        with tempfile.TemporaryDirectory() as output_dir:
            renderer = CanvasHtmlRenderer(output_dir, label="Three.js")
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("K12SIMWORLD_RENDER_MODE", None)
                self.assertEqual(renderer._render_mode("html"), "realtime")

    def test_capture_fps_is_configurable(self):
        with tempfile.TemporaryDirectory() as output_dir:
            renderer = CanvasHtmlRenderer(output_dir, label="Domain Canvas")
            with mock.patch.dict(os.environ, {"K12SIMWORLD_CAPTURE_FPS": "10"}):
                self.assertEqual(renderer._capture_fps(), 10.0)


class VideoProbeTest(unittest.TestCase):
    def test_node_environment_includes_current_python_bin(self):
        with tempfile.TemporaryDirectory() as output_dir:
            renderer = CanvasHtmlRenderer(output_dir)
            with mock.patch.dict(os.environ, {"PATH": "/usr/bin"}):
                path_entries = renderer._node_env()["PATH"].split(os.pathsep)

            self.assertEqual(path_entries[0], os.path.dirname(os.path.abspath(sys.executable)))
            self.assertIn("/usr/bin", path_entries)

    def test_probe_rejects_empty_file_without_running_ffprobe(self):
        with tempfile.TemporaryDirectory() as output_dir:
            path = os.path.join(output_dir, "empty.webm")
            open(path, "wb").close()
            renderer = CanvasHtmlRenderer(output_dir)
            with mock.patch.object(renderer, "run_and_log") as run:
                self.assertFalse(renderer._probe_video(path, os.path.join(output_dir, "x.log")))
                run.assert_not_called()

    def test_probe_uses_first_video_stream(self):
        with tempfile.TemporaryDirectory() as output_dir:
            path = os.path.join(output_dir, "video.webm")
            with open(path, "wb") as handle:
                handle.write(b"video")
            renderer = CanvasHtmlRenderer(output_dir)
            with mock.patch.object(renderer, "run_and_log", return_value=0) as run:
                self.assertTrue(renderer._probe_video(path, os.path.join(output_dir, "x.log")))
            command = run.call_args.args[0]
            self.assertEqual(command[0], "ffprobe")
            self.assertIn("v:0", command)


if __name__ == "__main__":
    unittest.main()
