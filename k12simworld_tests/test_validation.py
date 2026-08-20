import unittest

from k12simworld.domain_compiler import compile_domain_program
from k12simworld.models import EduWorldSpec, StoryBlock
from k12simworld.validation import (
    validate_document,
    validate_program_payload,
    validate_storyboard,
    validate_world_spec,
)


class ValidationTest(unittest.TestCase):
    @staticmethod
    def _spec():
        return EduWorldSpec.from_dict({
            "problem_id": "p1", "coordinate_system": {},
            "objects": [{"id": "ball", "properties": {"mass": 1, "unit": "m"}}],
            "parameters": [{"id": "m", "value": 1, "unit": "kg"}],
            "constraints": [], "initial_state": {}, "expected_events": [],
            "learning_goals": [], "visual_conventions": {},
        })

    def test_allows_adjacent_storyboard_block_kinds_with_warning(self):
        blocks = [
            StoryBlock("TEXT_1", "text", "intro"),
            StoryBlock("TEXT_2", "text", "explain"),
            StoryBlock("SIM_1", "sim", "show"),
            StoryBlock("SIM_2", "sim", "compare"),
        ]
        report = validate_storyboard(blocks)
        self.assertTrue(report.valid, report.errors)
        self.assertTrue(any("do not alternate" in item for item in report.warnings))

    def test_scene_id_mismatch_is_audited_without_blocking_render(self):
        spec = self._spec()
        payload = {
            "engine": "p5js",
            "render_spec": {"engine": "p5js"},
            "world_spec_sha256": spec.canonical_hash(),
            "scenes": [{"scene_id": "SIM_EXTRA", "document": "<canvas></canvas><script>const ball=1;</script>"}],
        }
        report = validate_program_payload(
            payload, spec, [StoryBlock("SIM_1", "sim", "show")]
        )
        self.assertTrue(report.valid, report.errors)
        self.assertTrue(any("without a matching" in item for item in report.warnings))
        self.assertTrue(any("additional renderable" in item for item in report.warnings))

    def test_trusted_compiler_document_uses_larger_bounded_limit(self):
        document = "<canvas></canvas>" + (" " * 500_001)
        self.assertFalse(validate_document(document, "mechanics-2d").valid)
        trusted = validate_document(document, "mechanics-2d", trusted_compiled=True)
        self.assertTrue(trusted.valid, trusted.errors)
        self.assertTrue(any("trusted compiler" in item for item in trusted.warnings))

    def test_process_fidelity_warnings_do_not_block_render(self):
        spec = EduWorldSpec.from_dict({
            "problem_id": "pendulum", "coordinate_system": {},
            "objects": [{"id": "bob", "type": "point_mass"}],
            "parameters": [], "constraints": [], "initial_state": {},
            "expected_events": [], "learning_goals": [], "visual_conventions": {},
        })
        raw = {
            "engine": "mechanics-2d",
            "render_spec": {"engine": "mechanics-2d", "duration": 1},
            "world_spec_sha256": spec.canonical_hash(),
            "scenes": [{"scene_id": "SIM_1", "simulation_spec": {
                "domain_model": "mechanics_2d", "duration": 1, "dt": .01,
                "gravity": [0, 0],
                "bodies": [{
                    "id": "bob", "shape": "circle", "position": [1, 0],
                    "velocity": [0, 1],
                }],
                "distance_constraints": [{
                    "id": "rope", "anchor_a": [0, 0], "body_b": "bob", "length": 1,
                }],
                "visual_strategy": "component_decomposition",
                "visual_instances": [{
                    "id": "bob_h", "source_object_id": "bob",
                    "view": "horizontal_projection", "panel": "right",
                }],
            }}],
        }
        compiled = compile_domain_program(raw, spec)
        report = validate_program_payload(
            compiled, spec, [StoryBlock("SIM_1", "sim", "摆动到最低点后断绳")]
        )
        self.assertTrue(report.valid, report.errors)
        self.assertTrue(any("never removes" in item for item in report.warnings))
        self.assertTrue(any("component panels" in item for item in report.warnings))

    def test_rejects_inconsistent_object_unit(self):
        report = validate_world_spec(self._spec())
        self.assertFalse(report.valid)
        self.assertTrue(any("mass" in error and "kg" in error for error in report.errors))

    def test_allows_energy_driven_bar_height(self):
        spec = EduWorldSpec.from_dict({
            "problem_id": "p1", "coordinate_system": {},
            "objects": [{"id": "energy", "type": "bar", "properties": {"height": 218, "height_unit": "J"}}],
            "parameters": [], "constraints": [], "initial_state": {}, "expected_events": [],
            "learning_goals": [], "visual_conventions": {},
        })
        self.assertTrue(validate_world_spec(spec).valid)

    def test_rejects_missing_object_and_invisible_material(self):
        spec = self._spec()
        document = "<canvas></canvas><script>renderer.setClearColor(0x000000); new THREE.MeshLambertMaterial({color:0x000000}); CANNON.World();</script>"
        payload = {
            "engine": "threejs-cannon", "render_spec": {"engine": "threejs-cannon"},
            "world_spec_sha256": spec.canonical_hash(),
            "scenes": [{"scene_id": "SIM_1", "document": document}],
        }
        report = validate_program_payload(payload, spec, [StoryBlock("SIM_1", "sim", "show")])
        self.assertFalse(report.valid)
        self.assertTrue(any("absent" in warning for warning in report.warnings))
        self.assertTrue(any("background" in error for error in report.errors))

    def test_rejects_text_outside_backing_canvas(self):
        spec = EduWorldSpec.from_dict({
            "problem_id": "p1", "coordinate_system": {}, "objects": [{"id": "ball"}],
            "parameters": [], "constraints": [], "initial_state": {}, "expected_events": [],
            "learning_goals": [], "visual_conventions": {},
        })
        document = "<canvas></canvas><script>const labelCanvas=document.createElement(\"canvas\"); labelCanvas.width=512; labelCanvas.height=64; const labelCtx=labelCanvas.getContext(\"2d\"); labelCtx.fillText(\"ball\",640,20);</script>"
        payload = {
            "engine": "p5js", "render_spec": {"engine": "p5js"},
            "world_spec_sha256": spec.canonical_hash(),
            "scenes": [{"scene_id": "SIM_1", "document": document}],
        }
        report = validate_program_payload(payload, spec, [StoryBlock("SIM_1", "sim", "show")])
        self.assertFalse(report.valid)
        self.assertTrue(any("outside" in error for error in report.errors))

    def test_allows_auxiliary_object_in_only_one_relevant_scene(self):
        spec = EduWorldSpec.from_dict({
            "problem_id": "p1", "coordinate_system": {},
            "objects": [{"id": "ball", "type": "sphere"}, {"id": "impact", "type": "point"}],
            "parameters": [], "constraints": [], "initial_state": {}, "expected_events": [],
            "learning_goals": [], "visual_conventions": {},
        })
        payload = {
            "engine": "p5js", "render_spec": {"engine": "p5js"},
            "world_spec_sha256": spec.canonical_hash(),
            "scenes": [
                {"scene_id": "SIM_1", "document": "<canvas></canvas><script>const ball=1;</script>"},
                {"scene_id": "SIM_2", "document": "<canvas></canvas><script>const ball=1, impact=2;</script>"},
            ],
        }
        blocks = [StoryBlock("SIM_1", "sim", "motion"), StoryBlock("SIM_2", "sim", "impact")]
        report = validate_program_payload(payload, spec, blocks)
        self.assertTrue(report.valid, report.errors)

    def test_requires_storyboard_highlight_in_its_scene(self):
        spec = EduWorldSpec.from_dict({
            "problem_id": "p1", "coordinate_system": {},
            "objects": [{"id": "ball", "type": "sphere"}, {"id": "impact", "type": "point"}],
            "parameters": [], "constraints": [], "initial_state": {}, "expected_events": [],
            "learning_goals": [], "visual_conventions": {},
        })
        payload = {
            "engine": "p5js", "render_spec": {"engine": "p5js"},
            "world_spec_sha256": spec.canonical_hash(),
            "scenes": [{"scene_id": "SIM_1", "document": "<canvas></canvas><script>const ball=1;</script>"}],
        }
        block = StoryBlock("SIM_1", "sim", "impact", highlights=["impact"])
        report = validate_program_payload(payload, spec, [block])
        self.assertTrue(report.valid, report.errors)
        self.assertTrue(any("highlighted" in warning for warning in report.warnings))
    def test_blocks_networked_browser_code(self):
        report = validate_document("<canvas></canvas><script>fetch('https://x')</script>", "p5js")
        self.assertFalse(report.valid)
        self.assertTrue(any("network" in error for error in report.errors))

    def test_blocks_remote_script(self):
        report = validate_document(
            '<canvas></canvas><script src="https://cdn.example/p5.js"></script>', "p5js"
        )
        self.assertFalse(report.valid)

    def test_blocks_replacing_three_vector_with_array(self):
        report = validate_document(
            "<canvas></canvas><script>CANNON.World(); ball.position = [0, 20, 0];</script>",
            "threejs-cannon",
        )
        self.assertFalse(report.valid)
        self.assertTrue(any("arrays" in error for error in report.errors))

    def test_blocks_non_whitelisted_manim_import(self):
        report = validate_document(
            "import sys\nfrom manim import *\nclass GeneratedScene(Scene):\n    pass\n", "manim"
        )
        self.assertFalse(report.valid)

    def test_accepts_minimal_manim_scene(self):
        report = validate_document("from manim import *\nclass GeneratedScene(Scene):\n    def construct(self):\n        pass\n", "manim")
        self.assertTrue(report.valid, report.errors)

    def test_blocks_unsafe_manim_import(self):
        report = validate_document("import subprocess\nclass GeneratedScene(Scene):\n    pass\n", "manim")
        self.assertFalse(report.valid)


if __name__ == "__main__":
    unittest.main()
