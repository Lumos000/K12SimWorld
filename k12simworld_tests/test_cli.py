import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from k12simworld.cli import _run_generation
from k12simworld.io import read_records, write_json
from k12simworld.models import ArtifactManifest


class FakePipeline:
    def __init__(self, output):
        self.output = Path(output)
        self.calls = []

    def generate(self, problem, requested_engine=None):
        self.calls.append(problem.problem_id)
        manifest = ArtifactManifest(
            problem_id=problem.problem_id,
            model="fake",
            method="k12simworld_state_anchored",
            success=True,
        )
        write_json(self.output / problem.problem_id / "manifest.json", manifest.to_dict())
        return manifest


class CliResumeTest(unittest.TestCase):
    def test_resume_skips_completed_item_and_keeps_unique_index(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            benchmark = root / "benchmark.jsonl"
            output = root / "runs"
            rows = [
                {"problem_id": item_id, "question": "物体如何运动？", "subject": "physics-g9"}
                for item_id in ("p1", "p2")
            ]
            benchmark.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            first = SimpleNamespace(
                benchmark=str(benchmark), output_dir=str(output), limit=1,
                engine=None, resume=False, retry_failed=False,
            )
            pipeline = FakePipeline(output)
            self.assertEqual(_run_generation(first, pipeline), 0)
            self.assertEqual(pipeline.calls, ["p1"])

            resumed = SimpleNamespace(
                benchmark=str(benchmark), output_dir=str(output), limit=2,
                engine=None, resume=True, retry_failed=False,
            )
            resumed_pipeline = FakePipeline(output)
            self.assertEqual(_run_generation(resumed, resumed_pipeline), 0)
            self.assertEqual(resumed_pipeline.calls, ["p2"])
            manifests = list(read_records(output / "manifests.jsonl"))
            self.assertEqual([row["problem_id"] for row in manifests], ["p1", "p2"])


if __name__ == "__main__":
    unittest.main()
