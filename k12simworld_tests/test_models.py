import unittest

from k12simworld.models import ContractError, EduWorldSpec, K12Problem


class ModelsTest(unittest.TestCase):
    def test_reference_solution_is_not_in_model_payload(self):
        problem = K12Problem.from_record(
            {
                "hash_id": "p1",
                "question": "What happens?",
                "subject": "physics-g9",
                "type": "选择题",
                "format_answer": {"format_solution": ["secret step"], "ground_truth": ["A"]},
            }
        )
        self.assertEqual(problem.grade, 9)
        self.assertEqual(problem.subject, "physics")
        self.assertNotIn("secret", str(problem.model_payload()))
        self.assertNotIn("ground_truth", problem.model_payload())

    def test_world_spec_rejects_unknown_event_participant(self):
        with self.assertRaises(ContractError):
            EduWorldSpec.from_dict(
                {
                    "problem_id": "p1",
                    "coordinate_system": {},
                    "objects": [{"id": "ball", "type": "circle"}],
                    "parameters": [],
                    "constraints": [],
                    "initial_state": {},
                    "expected_events": [
                        {"id": "e1", "type": "hit", "participants": ["missing"]}
                    ],
                    "learning_goals": [],
                    "visual_conventions": {},
                }
            )

    def test_world_spec_reports_field_type_errors(self):
        with self.assertRaisesRegex(
            ContractError, "EduWorldSpec.coordinate_system must be a JSON object"
        ):
            EduWorldSpec.from_dict(
                {
                    "problem_id": "p1",
                    "coordinate_system": "2-D Cartesian coordinates",
                    "objects": [{"id": "ball"}],
                    "parameters": [],
                    "constraints": [],
                    "initial_state": {},
                    "expected_events": [],
                    "learning_goals": [],
                    "visual_conventions": {},
                }
            )


if __name__ == "__main__":
    unittest.main()
