import base64
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from k12simworld.screening_pipeline import (
    TEXT_VALIDATOR, VISION_VALIDATOR, Config, decode_image, enforce_vision_policy,
    generate_gallery, normalize_record, validate_payload,
)


class ScreeningPipelineTest(unittest.TestCase):
    def test_normalizes_real_k12vista_shape(self):
        row = normalize_record({
            "hash_id": "x", "question": "如图", "subject": "physics-g9", "type": "选择题",
            "knowledge_point": ["运动"],
            "format_answer": {"ground_truth": ["A"], "format_solution": ["step"]},
        }, 1)
        self.assertEqual(row["id"], "x")
        self.assertEqual(row["grade"], "9")
        self.assertEqual(row["ground_truth"], ["A"])

    def test_decodes_base64_image(self):
        image = Image.new("RGB", (16, 16), "red")
        buffer = io.BytesIO(); image.save(buffer, "PNG")
        raw = base64.b64encode(buffer.getvalue()).decode("ascii")
        data, media, meta = decode_image(raw)
        self.assertTrue(data)
        self.assertTrue(meta["decodable"])
        self.assertEqual(meta["width"], 16)

    def test_core_policy_is_non_compensatory(self):
        value = {
            "final_category": "A_CORE", "simulation_family": "kinematics",
            "manual_review_required": False, "selection_reason": "x",
            "scores": {name: 3 for name in (
                "dynamic_process", "explicit_rule", "initial_condition", "outcome_verifiable",
                "engine_feasibility", "visual_dependency", "educational_value", "data_quality")},
        }
        value["scores"]["initial_condition"] = 1
        result = enforce_vision_policy(value)
        self.assertNotEqual(result["final_category"], "A_CORE")
        self.assertTrue(result["manual_review_required"])

    def test_paginated_gallery_uses_external_images_and_deduplicates(self):
        image = Image.new("RGB", (20, 20), "blue")
        buffer = io.BytesIO(); image.save(buffer, "PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / "raw.jsonl"; output = root / "selected"
            output.mkdir()
            raw_rows = []
            results = []
            for index, category in enumerate(("A_CORE", "B_EXTENSION", "C_STATIC"), 1):
                item_id = f"item-{index}"
                raw_rows.append({
                    "hash_id": item_id, "question": f"question {index}", "subject": "physics-g9",
                    "type": "选择题", "knowledge_point": ["motion"], "img": encoded,
                    "format_answer": {"ground_truth": ["A"], "format_solution": ["solution"]},
                })
                results.append({
                    "id": item_id, "final_category": category, "subject": "physics-g9", "grade": "9",
                    "question_type": "选择题", "knowledge_points": ["motion"],
                    "simulation_family": "kinematics", "recommended_backend": "p5js",
                    "manual_review_required": category == "C_STATIC", "conflict_flag": False,
                    "confidence": 0.9, "selection_reason": "test", "scores": {},
                })
            source.write_text("\n".join(json.dumps(row) for row in raw_rows) + "\n", encoding="utf-8")
            cfg = Config(source, output, "", "", "", "", "", "", 2026, 1, 1, 1)
            generate_gallery(cfg, results)
            launcher = (output / "gallery.html").read_text(encoding="utf-8")
            page = (output / "review/pages/page_001.html").read_text(encoding="utf-8")
            index = (output / "review/index.html").read_text(encoding="utf-8")
            search_index = (output / "review/assets/search-index.js").read_text(encoding="utf-8")
            review_script = (output / "review/assets/review.js").read_text(encoding="utf-8")
            manifest = json.loads((output / "review/manifest.json").read_text(encoding="utf-8"))
            self.assertNotIn("base64,", launcher)
            self.assertNotIn("base64,", page)
            self.assertEqual(manifest["unique_items"], 3)
            self.assertEqual(manifest["page_count"], 1)
            self.assertEqual(len(list((output / "review/images").iterdir())), 3)
            self.assertEqual(page.count("data-review-card"), 3)
            self.assertIn("<th>题图</th>", index)
            self.assertIn('"image":', search_index)
            self.assertIn('class="list-thumb"', review_script)
            self.assertIn('loading="lazy"', review_script)


if __name__ == "__main__":
    unittest.main()
