import unittest

from k12simworld.formula_normalization import normalize_world_spec_formulas
from k12simworld.models import EduWorldSpec


class FormulaNormalizationTest(unittest.TestCase):
    @staticmethod
    def _base_spec():
        return {
            "problem_id": "p1",
            "coordinate_system": {},
            "objects": [{"id": "ball"}, {"id": "other"}],
            "parameters": [{"id": "m", "value": 2.0, "unit": "kg"}],
            "constraints": [],
            "initial_state": {},
            "expected_events": [],
            "learning_goals": [],
            "visual_conventions": {},
        }

    def test_extracts_dotted_paths_and_inlines_numeric_parameters(self):
        payload = self._base_spec()
        payload["target_observables"] = [
            {
                "id": "kinetic_energy",
                "expression": "0.5 * m * objects.ball.speed ^ 2",
                "bindings": {},
                "expected": 4.0,
            }
        ]

        normalized, changes = normalize_world_spec_formulas(payload)
        target = normalized["target_observables"][0]

        self.assertIn("**", target["expression"])
        self.assertNotIn("objects.ball.speed", target["expression"])
        self.assertNotIn(" m ", target["expression"])
        self.assertIn("objects.ball.speed", target["bindings"].values())
        self.assertTrue(changes)
        EduWorldSpec.from_dict(normalized)

    def test_inlines_arithmetic_accidentally_placed_in_bindings(self):
        payload = self._base_spec()
        payload["invariants"] = [
            {
                "id": "energy_balance",
                "expression": "ke_total - reference_ke",
                "bindings": {
                    "ke_total": (
                        "objects.ball.kinetic_energy + "
                        "objects.other.kinetic_energy"
                    ),
                    "reference_ke": "objects.ball.mechanical_energy",
                },
                "type": "constant",
            }
        ]

        normalized, changes = normalize_world_spec_formulas(payload)
        invariant = normalized["invariants"][0]

        self.assertNotIn("ke_total", invariant["bindings"])
        self.assertIn("objects.ball.kinetic_energy", invariant["bindings"].values())
        self.assertIn("objects.other.kinetic_energy", invariant["bindings"].values())
        self.assertTrue(any("inlined derived expression" in item for item in changes))
        EduWorldSpec.from_dict(normalized)

    def test_extracts_negated_dotted_lookup(self):
        payload = self._base_spec()
        payload["target_observables"] = [{
            "id": "downward_velocity",
            "expression": "-objects.ball.velocity.1",
            "bindings": {},
            "expected": 3.0,
        }]

        normalized, _ = normalize_world_spec_formulas(payload)
        target = normalized["target_observables"][0]

        self.assertNotIn("objects.ball.velocity.1", target["expression"])
        self.assertTrue(target["expression"].startswith("-"))
        self.assertIn("objects.ball.velocity.1", target["bindings"].values())
        EduWorldSpec.from_dict(normalized)

    def test_safely_expands_negated_arithmetic_binding(self):
        payload = self._base_spec()
        payload["invariants"] = [{
            "id": "signed_momentum",
            "expression": "net_vertical - reference",
            "bindings": {
                "net_vertical": (
                    "-objects.ball.velocity.1 * m + "
                    "objects.other.velocity.1 * m"
                ),
                "reference": "objects.ball.position.1",
            },
            "type": "constant",
        }]

        normalized, changes = normalize_world_spec_formulas(payload)
        invariant = normalized["invariants"][0]

        self.assertNotIn("net_vertical", invariant["bindings"])
        self.assertNotIn("objects.", invariant["expression"])
        self.assertIn("objects.ball.velocity.1", invariant["bindings"].values())
        self.assertIn("objects.other.velocity.1", invariant["bindings"].values())
        self.assertTrue(any("inlined derived expression" in item for item in changes))
        EduWorldSpec.from_dict(normalized)


if __name__ == "__main__":
    unittest.main()
