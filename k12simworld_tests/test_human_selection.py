import base64
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from k12simworld.human_selection import prepare_human_physics_selection
from k12simworld.io import read_records
from k12simworld.models import K12Problem


class HumanSelectionTest(unittest.TestCase):
    def test_normalizes_k12vista_question_answer_type(self):
        problem = K12Problem.from_record({
            "hash_id": "a" * 64,
            "question": "求物体速度。",
            "subject": "physics-g12",
            "type": "问答题",
        })
        self.assertEqual(problem.question_type, "free_response")

    def test_prepares_deduplicated_gold_free_model_inputs(self):
        image = Image.new("RGB", (8, 8), "white")
        buffer = io.BytesIO(); image.save(buffer, "JPEG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        ids = [character * 64 for character in ("a", "b")]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw.jsonl"
            selection = root / "selection.json"
            screening = root / "screening.jsonl"
            output = root / "out"
            raw.write_text("\n".join(json.dumps({
                "hash_id": item_id, "question": "物体碰撞后如何运动？",
                "subject": "physics-g9", "type": "选择题",
                "knowledge_point": ["碰撞"], "img": encoded,
                "format_answer": {"ground_truth": ["A"], "format_solution": ["step"]},
            }) for item_id in ids) + "\n", encoding="utf-8")
            selection.write_text(json.dumps({
                "source": {"type": "test"},
                "groups": {"运动相关": [ids[0]], "碰撞": [ids[0], ids[1]]},
            }), encoding="utf-8")
            screening.write_text("\n".join(json.dumps({
                "id": item_id, "final_category": "A_CORE",
                "simulation_family": "collision", "recommended_backend": "matterjs",
                "manual_review_required": False, "conflict_flag": False, "confidence": 0.99,
            }) for item_id in ids) + "\n", encoding="utf-8")

            report = prepare_human_physics_selection(
                raw, selection, output, screening_path=screening, smoke_target=2
            )
            benchmark = list(read_records(output / "physics_k12simbench.jsonl"))
            previews = list(read_records(output / "physics_model_input_preview.jsonl"))
            self.assertEqual(report["selection_occurrences"], 3)
            self.assertEqual(report["selection_unique_ids"], 2)
            self.assertEqual(report["matched_physics"], 2)
            self.assertEqual(len(benchmark), 2)
            self.assertEqual(len(report["smoke_ids"]), 2)
            self.assertEqual(benchmark[0]["source_metadata"]["human_selection_groups"], ["运动相关", "碰撞"])
            self.assertNotIn("ground_truth", previews[0])
            self.assertNotIn("reference_solution", previews[0])
            self.assertNotIn("image", previews[0])
            self.assertTrue(previews[0]["image_present"])
            self.assertNotIn("ground_truth", K12Problem.from_record(benchmark[0]).model_payload())


if __name__ == "__main__":
    unittest.main()
