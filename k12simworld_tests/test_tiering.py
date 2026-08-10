import unittest

from k12simworld.tiering import classify_physics_record


def record(family, groups):
    return {
        "simulation_type": family,
        "source_metadata": {"human_selection_groups": groups},
    }


class TieringTest(unittest.TestCase):
    def test_native_mechanics(self):
        result = classify_physics_record(record("collision", ["碰撞"]))
        self.assertEqual(result["tier"], "native")
        self.assertTrue(result["ready_for_main_experiment"])

    def test_electromagnetic_group_overrides_mechanics_label(self):
        result = classify_physics_record(record("force_and_motion", ["运动相关", "电磁场"]))
        self.assertEqual(result["tier"], "equation")
        self.assertEqual(result["required_backend"], "deterministic_equation_integrator")

    def test_circuit_has_precedence_over_electromagnetic_group(self):
        result = classify_physics_record(record("charged_particle", ["电磁场", "电路"]))
        self.assertEqual(result["tier"], "specialized")
        self.assertEqual(result["subtype"], "circuit")

    def test_optics_is_specialized(self):
        result = classify_physics_record(record("ray_optics", ["光学"]))
        self.assertEqual(result["tier"], "specialized")
        self.assertEqual(result["required_backend"], "ray_optics")


if __name__ == "__main__":
    unittest.main()
