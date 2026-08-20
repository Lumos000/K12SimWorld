import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from k12simworld.adapters import StaticResponseAdapter
from k12simworld.models import EduWorldSpec, K12Problem, StoryBlock
from k12simworld.pipeline import K12SimWorldPipeline


class RecordingStaticResponseAdapter(StaticResponseAdapter):
    def __init__(self, responses):
        super().__init__(responses)
        self.prompts = []

    def generate(self, prompt, problem, call_index, attempt):
        self.prompts.append(prompt)
        return super().generate(prompt, problem, call_index, attempt)


class PipelineTest(unittest.TestCase):
    def test_offline_generation_writes_auditable_artifacts(self):
        problem = K12Problem.from_record(
            {
                "hash_id": "p1",
                "question": "两个状态之间如何变化？",
                "subject": "physics-g9",
                "type": "选择题",
                "knowledge_point": ["状态变化"],
                "format_answer": {"format_solution": ["gold"], "ground_truth": ["A"]},
                "dynamic_suitability": True,
            }
        )
        story = {
            "analysis": "closing the switch completes the circuit",
            "requires_simulation": False,
            "final_answer": "灯亮",
            "solution": {
                "analysis": "closing the switch completes the circuit",
                "final_answer": "灯会点亮",
            },
            "blocks": [
                {"block_id": "TEXT_1", "kind": "text", "content": "观察开关。"},
                {"block_id": "TEXT_2", "kind": "text", "content": "说明完整回路。"},
            ],
        }
        spec_dict = {
            "schema_version": "1.0",
            "problem_id": "p1",
            "coordinate_system": {"origin": "top_left"},
            "objects": [{"id": "switch", "type": "switch"}, {"id": "lamp", "type": "lamp"}],
            "parameters": [],
            "constraints": [],
            "initial_state": {"switch": "open", "lamp": "off"},
            "expected_events": [
                {"id": "close", "type": "state_change", "participants": ["switch", "lamp"]}
            ],
            "final_state": {"switch": "closed", "lamp": "on"},
            "learning_goals": ["understand a closed circuit"],
            "visual_conventions": {"background": "white"},
        }
        spec = EduWorldSpec.from_dict(spec_dict)
        document = "<html><body><canvas id='c'></canvas><script>const ids=['switch','lamp'];</script></body></html>"
        program = {
            "engine": "p5js",
            "render_spec": {"engine": "p5js", "fps": 30, "duration": 8},
            "world_spec_sha256": spec.canonical_hash(),
            "scenes": [{"scene_id": "SIM_AUTO_1", "document": document}],
        }
        adapter = StaticResponseAdapter(
            [json.dumps(story), json.dumps(spec_dict), json.dumps(program)]
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = K12SimWorldPipeline(adapter, directory, render=False).generate(problem)
            self.assertTrue(manifest.success, manifest.error)
            self.assertTrue(manifest.world_spec_path)
            self.assertTrue(manifest.program_path)
            self.assertEqual(manifest.model, "static-test-model")
            warning_codes = {
                item["code"] for item in manifest.diagnostics["pre_render_warnings"]
            }
            self.assertIn("model_simulation_preference_overridden", warning_codes)
            self.assertIn("final_answer_disagreement", warning_codes)
            self.assertIn("simulation_block_synthesized", warning_codes)
            self.assertIn("storyboard_structure", warning_codes)
            storyboard = json.loads(Path(manifest.storyboard_path).read_text(encoding="utf-8"))
            self.assertEqual(storyboard["final_answer"], "灯会点亮")

    def test_additional_scene_video_is_not_mislabeled_as_storyboard_scene(self):
        problem = K12Problem.from_record(
            {
                "hash_id": "scene-map",
                "question": "展示运动",
                "subject": "physics-g9",
            }
        )
        document = K12SimWorldPipeline._assemble_document(
            problem, {"final_answer": "ok"},
            [StoryBlock("SIM_1", "sim", "expected scene")],
            ["/tmp/SIM_EXTRA.mp4"],
        )
        self.assertIn("Additional simulation SIM_EXTRA", document)

    def test_preserves_render_error_when_automatic_repair_is_unavailable(self):
        problem = K12Problem.from_record(
            {"hash_id": "p2", "question": "物体如何运动？", "subject": "physics-g9"}
        )
        story = {
            "analysis": "motion",
            "requires_simulation": True,
            "final_answer": "moves",
            "blocks": [{"block_id": "SIM_1", "kind": "sim", "content": "show ball"}],
        }
        spec_dict = {
            "problem_id": "p2",
            "coordinate_system": {},
            "objects": [{"id": "ball", "type": "sphere"}],
            "parameters": [],
            "constraints": [],
            "initial_state": {},
            "expected_events": [],
            "learning_goals": [],
            "visual_conventions": {},
        }
        spec = EduWorldSpec.from_dict(spec_dict)
        program = {
            "engine": "p5js",
            "render_spec": {"engine": "p5js", "fps": 5, "duration": 1},
            "world_spec_sha256": spec.canonical_hash(),
            "scenes": [
                {"scene_id": "SIM_1", "document": "<canvas></canvas><script>const ball=1;</script>"}
            ],
        }
        adapter = StaticResponseAdapter(
            [json.dumps(story), json.dumps(spec_dict), json.dumps(program)]
        )
        with tempfile.TemporaryDirectory() as directory:
            pipeline = K12SimWorldPipeline(adapter, directory, render=True)
            with mock.patch.object(pipeline, "_render", side_effect=ValueError("renderer exploded")):
                manifest = pipeline.generate(problem, requested_engine="p5js")

        self.assertFalse(manifest.success)
        self.assertIn("ValueError: renderer exploded", manifest.error)
        self.assertIn("StaticResponseAdapter has no response left", manifest.error)

    def test_world_spec_gets_one_targeted_repair_after_normalization(self):
        problem = K12Problem.from_record(
            {"hash_id": "p3", "question": "球的速度如何变化？", "subject": "physics-g9"}
        )
        story = {
            "analysis": "speed is represented by a trace scalar",
            "requires_simulation": True,
            "final_answer": "1 m/s",
            "blocks": [{"block_id": "SIM_1", "kind": "sim", "content": "show ball"}],
        }
        bad_spec = {
            "problem_id": "p3",
            "coordinate_system": {},
            "objects": [{"id": "ball", "type": "sphere"}],
            "parameters": [],
            "constraints": [],
            "initial_state": {},
            "expected_events": [],
            "learning_goals": [],
            "visual_conventions": {},
            "invariants": [
                {
                    "id": "bad_symbolic_invariant",
                    "expression": "n + speed",
                    "bindings": {"speed": "objects.ball.speed"},
                    "type": "constant",
                }
            ],
        }
        good_spec = {
            **bad_spec,
            "invariants": [],
            "target_observables": [
                {
                    "id": "final_speed",
                    "path": "objects.ball.speed",
                    "expected": 1.0,
                    "unit": "m/s",
                }
            ],
        }
        spec = EduWorldSpec.from_dict(good_spec)
        program = {
            "engine": "p5js",
            "render_spec": {"engine": "p5js", "fps": 5, "duration": 1},
            "world_spec_sha256": spec.canonical_hash(),
            "scenes": [
                {
                    "scene_id": "SIM_1",
                    "document": "<canvas></canvas><script>const ball=1;</script>",
                }
            ],
        }
        adapter = RecordingStaticResponseAdapter(
            [
                json.dumps(story),
                json.dumps(bad_spec),
                json.dumps(good_spec),
                json.dumps(program),
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            manifest = K12SimWorldPipeline(adapter, directory).generate(
                problem, requested_engine="p5js"
            )
            run_dir = Path(directory) / "p3"
            self.assertTrue((run_dir / "attempts/world_spec_attempt_1.txt").is_file())
            self.assertTrue((run_dir / "attempts/world_spec_attempt_2.txt").is_file())

        self.assertTrue(manifest.success, manifest.error)
        self.assertEqual(manifest.diagnostics["world_spec_attempts"], 2)
        self.assertEqual(manifest.diagnostics["model_calls"], 4)
        self.assertIn("unknown name 'n'", adapter.prompts[2])

    def test_render_retry_regenerates_program_with_real_log_tail(self):
        problem = K12Problem.from_record(
            {"hash_id": "p4", "question": "球如何运动？", "subject": "physics-g9"}
        )
        story = {
            "analysis": "the ball moves in the rendered scene",
            "requires_simulation": True,
            "final_answer": "moves",
            "blocks": [{"block_id": "SIM_1", "kind": "sim", "content": "show ball"}],
        }
        spec_dict = {
            "problem_id": "p4",
            "coordinate_system": {},
            "objects": [{"id": "ball", "type": "sphere"}],
            "parameters": [],
            "constraints": [],
            "initial_state": {},
            "expected_events": [],
            "learning_goals": [],
            "visual_conventions": {},
        }
        spec = EduWorldSpec.from_dict(spec_dict)
        program = {
            "engine": "p5js",
            "render_spec": {"engine": "p5js", "fps": 5, "duration": 1},
            "world_spec_sha256": spec.canonical_hash(),
            "scenes": [
                {
                    "scene_id": "SIM_1",
                    "document": "<canvas></canvas><script>const ball=1;</script>",
                }
            ],
        }
        repaired_program = {
            **program,
            "scenes": [
                {
                    "scene_id": "SIM_1",
                    "document": (
                        "<canvas></canvas><script>"
                        "const ball=1; const repaired_runtime=true;"
                        "</script>"
                    ),
                }
            ],
        }
        adapter = RecordingStaticResponseAdapter(
            [
                json.dumps(story),
                json.dumps(spec_dict),
                json.dumps(program),
                json.dumps(repaired_program),
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "p4"
            log_path = run_dir / "renderer" / "SIM_1.log"
            log_path.parent.mkdir(parents=True)
            log_path.write_text(
                "browser start\nReferenceError: missingCanvas is not defined\n",
                encoding="utf-8",
            )
            render_error = ValueError("renderer process returned 1")
            render_error.log_path = str(log_path)
            pipeline = K12SimWorldPipeline(adapter, directory, render=True)
            with mock.patch.object(
                pipeline,
                "_render",
                side_effect=[render_error, [str(run_dir / "videos/SIM_1.mp4")]],
            ):
                manifest = pipeline.generate(problem, requested_engine="p5js")
            feedback_path = run_dir / "attempts/render_feedback_attempt_1.txt"
            self.assertIn("missingCanvas", feedback_path.read_text(encoding="utf-8"))

        self.assertTrue(manifest.success, manifest.error)
        self.assertEqual(manifest.diagnostics["render_attempts"], 2)
        self.assertEqual(manifest.diagnostics["execution_repair_attempts"], 1)
        self.assertIn("missingCanvas", adapter.prompts[-1])


if __name__ == "__main__":
    unittest.main()
