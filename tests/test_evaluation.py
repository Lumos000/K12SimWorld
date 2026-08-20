import math
import unittest

from k12simworld.evaluation.metrics import aggregate_records, event_f1, score_record
from k12simworld.evaluation.statistics import krippendorff_alpha, spearman_rho


class EvaluationTest(unittest.TestCase):
    def test_failed_generation_is_zero(self):
        scored = score_record({"success": False, "scores": {"correctness_completeness": 100}})
        self.assertEqual(scored["overall"], 0.0)
        self.assertTrue(all(value == 0.0 for value in scored["scores"].values()))

    def test_aggregate_does_not_drop_failure(self):
        rows = [
            {"model": "m", "method": "ours", "success": True, "scores": {name: 100 for name in (
                "correctness_completeness", "logical_coherence", "pedagogical_effectiveness",
                "typographic_clarity", "simulation_problem_alignment", "element_layout_quality",
                "temporal_visual_consistency", "text_simulation_coordination", "initial_state_match",
                "key_event_accuracy", "final_state_accuracy", "constraint_satisfaction")}},
            {"model": "m", "method": "ours", "success": False, "scores": {}},
        ]
        result = aggregate_records(rows)[0]
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["success_rate"], 50.0)
        self.assertAlmostEqual(result["overall"], 50.0)

    def test_event_f1_and_agreement(self):
        event = {"id": "e", "type": "hit", "participants": ["a", "b"]}
        self.assertEqual(event_f1([event], [event]), 100.0)
        self.assertAlmostEqual(spearman_rho([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(krippendorff_alpha([[1, 1, 1], [2, 2, 2]]), 1.0)

    def test_incomplete_modality_has_no_overall_score(self):
        scored = score_record(
            {
                "success": True,
                "scores": {
                    "correctness_completeness": 90,
                    "logical_coherence": 90,
                    "typographic_clarity": 90,
                },
            }
        )
        self.assertEqual(scored["solution_quality"], 90.0)
        self.assertIsNone(scored["overall"])


if __name__ == "__main__":
    unittest.main()
