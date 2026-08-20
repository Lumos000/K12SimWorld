import json
import tempfile
import unittest
from pathlib import Path

from k12simworld.cli import main
from k12simworld.io import read_records


class EvaluationCliTest(unittest.TestCase):
    def test_manifest_trace_scoring_and_score_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observed = root / "observed.json"
            observed.write_text(
                json.dumps(
                    {
                        "problem_id": "p1",
                        "initial_state": {},
                        "final_state": {},
                        "events": [],
                        "trajectories": {},
                    }
                ),
                encoding="utf-8",
            )
            references = root / "references.jsonl"
            references.write_text(
                json.dumps(
                    {
                        "problem_id": "p1",
                        "initial_state": {},
                        "expected_events": [],
                        "final_state": {},
                        "trajectories": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifests = root / "manifests.jsonl"
            manifests.write_text(
                json.dumps(
                    {
                        "problem_id": "p1",
                        "model": "m",
                        "method": "k12simworld_candidate_target",
                        "success": True,
                        "observed_trace_path": str(observed),
                        "scores": {},
                        "diagnostics": {
                            "candidate_target_status": "passed",
                            "candidate_constraint_scores": {
                                "candidate_constraint_satisfaction": 100
                            },
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            trace_scores = root / "trace_scores.jsonl"
            self.assertEqual(
                main(
                    [
                        "score-traces",
                        "--references",
                        str(references),
                        "--manifests",
                        str(manifests),
                        "--output",
                        str(trace_scores),
                        "--strict",
                    ]
                ),
                0,
            )
            self.assertEqual(list(read_records(trace_scores))[0]["scores"]["final_state_accuracy"], 100)

            report = root / "report"
            self.assertEqual(
                main(
                    [
                        "evaluate",
                        "--manifests",
                        str(manifests),
                        "--scores",
                        str(trace_scores),
                        "--output-dir",
                        str(report),
                    ]
                ),
                0,
            )
            aggregate = json.loads((report / "aggregate.json").read_text(encoding="utf-8"))[0]
            self.assertEqual(aggregate["candidate_target_pass_rate"], 100)
            scored = json.loads((report / "scored_records.json").read_text(encoding="utf-8"))[0]
            self.assertEqual(scored["scores"]["final_state_accuracy"], 100)


if __name__ == "__main__":
    unittest.main()
