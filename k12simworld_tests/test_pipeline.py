import json
import tempfile
import unittest

from k12simworld.adapters import StaticResponseAdapter
from k12simworld.models import EduWorldSpec, K12Problem
from k12simworld.pipeline import K12SimWorldPipeline


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
            "requires_simulation": True,
            "final_answer": "灯亮",
            "blocks": [
                {"block_id": "TEXT_1", "kind": "text", "content": "观察开关。"},
                {"block_id": "SIM_1", "kind": "sim", "content": "闭合开关。"},
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
            "scenes": [{"scene_id": "SIM_1", "document": document}],
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


if __name__ == "__main__":
    unittest.main()
