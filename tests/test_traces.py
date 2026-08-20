import unittest

from k12simworld.evaluation.traces import score_trace, trajectory_rmse


class TraceTest(unittest.TestCase):
    def test_identical_trace_scores_perfectly(self):
        reference = {
            "initial_state": {"x": 0},
            "expected_events": [{"id": "e", "type": "move", "participants": ["ball"]}],
            "final_state": {"x": 1},
            "trajectories": {"ball.x": [{"t": 0, "value": 0}, {"t": 1, "value": 1}]},
        }
        observed = {
            "initial_state": {"x": 0},
            "events": [{"id": "e", "type": "move", "participants": ["ball"]}],
            "final_state": {"x": 1},
            "trajectories": {"ball.x": [{"t": 0, "value": 0}, {"t": 1, "value": 1}]},
            "constraint_violations": [],
        }
        scores = score_trace(reference, observed)
        self.assertEqual(scores["key_event_accuracy"], 100.0)
        self.assertEqual(scores["trajectory_nrmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
