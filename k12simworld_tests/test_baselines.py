import json
import tempfile
import unittest

from k12simworld.adapters import StaticResponseAdapter
from k12simworld.baselines import BaselinePipeline
from k12simworld.models import K12Problem


class BaselineTest(unittest.TestCase):
    def test_text_baseline_uses_common_manifest(self):
        problem = K12Problem.from_record(
            {"id": "p", "question": "为什么？", "subject": "physics-g9", "type": "解答题"}
        )
        adapter = StaticResponseAdapter([json.dumps({"analysis": "因为闭合。", "final_answer": "灯亮"})])
        with tempfile.TemporaryDirectory() as directory:
            manifest = BaselinePipeline(adapter, directory, method="text_cot").generate(problem)
        self.assertTrue(manifest.success, manifest.error)
        self.assertEqual(manifest.method, "text_cot")
        self.assertIsNone(manifest.program_path)


if __name__ == "__main__":
    unittest.main()
