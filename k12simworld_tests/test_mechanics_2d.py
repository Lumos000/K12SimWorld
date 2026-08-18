import json
import math
import tempfile
import unittest
from pathlib import Path

from k12simworld.domain_compiler import compile_domain_program
from k12simworld.domain_solvers import DomainSimulationError, simulate_domain
from k12simworld.models import EduWorldSpec, K12Problem, StoryBlock
from k12simworld.routing import EngineRouter, audit_spatial_request
from k12simworld.validation import validate_program_payload


def world(problem_id="m1"):
    return EduWorldSpec.from_dict({
        "problem_id": problem_id, "coordinate_system": {"type": "cartesian", "plane": "xy"},
        "objects": [{"id": "ball", "type": "circle"}, {"id": "ground", "type": "segment"}],
        "parameters": [], "constraints": [], "initial_state": {}, "expected_events": [],
        "final_state": {}, "learning_goals": ["observe motion"], "visual_conventions": {},
    })


class Mechanics2DSolverTest(unittest.TestCase):
    def test_free_flight_matches_constant_acceleration(self):
        spec = {
            "domain_model": "mechanics_2d", "duration": 1, "dt": .001,
            "gravity": [0, -10], "bounds": {"x_min": -2, "x_max": 5, "y_min": -10, "y_max": 5},
            "bodies": [{"id": "ball", "shape": "circle", "mass": 2, "radius": .1,
                        "position": [0, 0], "velocity": [3, 4]}],
        }
        trace = simulate_domain("mechanics-2d", spec)
        final = trace["time_series"][-1]["objects"]["ball"]
        self.assertAlmostEqual(final["position"][0], 3, places=8)
        # Constant acceleration is integrated exactly within each fixed step.
        self.assertAlmostEqual(final["position"][1], -1.0, places=8)
        self.assertAlmostEqual(final["velocity"][1], -6, places=8)

    def test_ground_contact_prevents_fall_through_and_emits_event(self):
        spec = {
            "domain_model": "mechanics_2d", "duration": 1, "dt": .005, "gravity": [0, -9.8],
            "bodies": [{"id": "ball", "shape": "circle", "mass": 1, "radius": .2,
                        "position": [0, 1], "velocity": [0, 0], "restitution": 0}],
            "static_geometry": [{"id": "ground", "type": "segment", "p1": [-2, 0], "p2": [2, 0], "normal": [0, 1], "restitution": 0}],
        }
        trace = simulate_domain("mechanics-2d", spec)
        self.assertGreaterEqual(trace["summary"]["ball"]["final_position"][1], .2 - 1e-9)
        self.assertTrue(any(event["type"] == "contact_begin" for event in trace["events"]))

    def test_distance_constraint_stays_near_declared_length(self):
        spec = {
            "domain_model": "mechanics_2d", "duration": 1, "dt": .002, "gravity": [0, -9.8],
            "bodies": [{"id": "ball", "shape": "circle", "mass": 1, "radius": .1,
                        "position": [1, 0], "velocity": [0, 1]}],
            "distance_constraints": [{"id": "rope", "anchor_a": [0, 0], "body_b": "ball", "length": 1}],
        }
        trace = simulate_domain("mechanics-2d", spec)
        for frame in trace["time_series"]:
            position = frame["objects"]["ball"]["position"]
            self.assertAlmostEqual(math.hypot(*position), 1, places=7)

    def test_dynamic_body_collides_with_static_body(self):
        trace = simulate_domain("mechanics-2d", {
            "domain_model": "mechanics_2d", "duration": 1, "dt": .002, "gravity": [0, 0],
            "bounds": {"x_min": -2, "x_max": 2, "y_min": -2, "y_max": 2},
            "bodies": [
                {"id": "moving", "shape": "circle", "mass": 1, "radius": .2,
                 "position": [-1, 0], "velocity": [2, 0], "restitution": 1},
                {"id": "fixed", "shape": "circle", "motion_type": "static", "mass": 1,
                 "radius": .2, "position": [0, 0], "velocity": [0, 0], "restitution": 1},
            ],
        })
        self.assertLess(trace["summary"]["moving"]["final_velocity"][0], -1.9)
        self.assertEqual(trace["summary"]["fixed"]["final_position"], [0.0, 0.0])

    def test_unknown_entities_are_rejected(self):
        with self.assertRaises(DomainSimulationError):
            simulate_domain("mechanics-2d", {
                "domain_model": "mechanics_2d", "bodies": [{"id": "ball", "shape": "circle", "position": [0, 0]}],
                "springs": [{"id": "spring", "anchor_a": [0, 0], "body_b": "missing", "stiffness": 10}],
            })

    def test_invalid_cosmetic_annotation_is_ignored_but_audited(self):
        trace = simulate_domain("mechanics-2d", {
            "domain_model": "mechanics_2d", "duration": 1, "dt": .01,
            "gravity": [0, 0],
            "bodies": [{"id": "ball", "shape": "circle", "position": [0, 0]}],
            "annotations": [
                {"type": "trail", "target": "ball"},
                {"type": "label", "target": "ball trajectory"},
            ],
        })
        self.assertEqual(trace["annotations"], [{"type": "trail", "target": "ball"}])
        self.assertEqual(trace["ignored_annotations"][0]["target"], "ball trajectory")

    def test_trace_exposes_initial_dynamics_and_pre_contact_energy(self):
        trace = simulate_domain("mechanics-2d", {
            "domain_model": "mechanics_2d", "duration": 8, "dt": .0083333333,
            "gravity": [0, -10],
            "bodies": [{
                "id": "ball", "shape": "circle", "mass": 1, "radius": .05,
                "collision_radius": 0, "position": [0, 20], "velocity": [6, 0], "restitution": 0,
            }],
            "static_geometry": [{
                "id": "ground", "type": "segment", "p1": [-1, 0],
                "p2": [20, 0], "normal": [0, 1], "restitution": 0,
            }],
            "terminal_contact": ["ball", "ground"],
        })
        initial = trace["time_series"][0]["objects"]["ball"]
        self.assertEqual(initial["acceleration"], [0.0, -10.0])
        self.assertEqual(initial["kinetic_energy"], 18.0)
        self.assertEqual(initial["potential_energy"], 200.0)
        self.assertEqual(initial["mechanical_energy"], 218.0)
        self.assertEqual(trace["time_series"][0]["energies"]["mechanical_total"], 218.0)
        impact = next(event for event in trace["events"] if event["type"] == "contact_begin")
        impact_ball = impact["snapshot"]["objects"]["ball"]
        self.assertAlmostEqual(impact_ball["kinetic_energy"], 218.0, places=6)
        self.assertAlmostEqual(impact_ball["potential_energy"], 0.0, places=6)
        self.assertAlmostEqual(impact_ball["mechanical_energy"], 218.0, places=6)
        self.assertAlmostEqual(impact["snapshot"]["energies"]["mechanical_total"], 218.0, places=6)
        self.assertEqual(impact_ball["velocity"], [6.0, -20.0])
        self.assertAlmostEqual(impact["t"], 2.0, places=8)
        self.assertAlmostEqual(impact_ball["position"][0], 12.0, places=8)
        self.assertAlmostEqual(impact_ball["position"][1], 0.0, places=8)
        self.assertEqual(trace["duration"], impact["t"])

    def test_visual_instance_reuses_body_state_without_becoming_physical(self):
        trace = simulate_domain("mechanics-2d", {
            "domain_model": "mechanics_2d", "duration": 1, "dt": .01,
            "gravity": [0, 0],
            "bodies": [{"id": "ball", "shape": "circle", "position": [0, 0],
                        "velocity": [1, 0]}],
            "visual_instances": [{
                "id": "ball_h", "source_object_id": "ball",
                "view": "horizontal_projection", "panel": "right",
                "label": "horizontal motion", "show_trail": True,
            }],
        })
        self.assertEqual(trace["visual_instances"][0]["source_object_id"], "ball")
        self.assertNotIn("ball_h", trace["time_series"][0]["objects"])
        self.assertNotIn("ball_h", trace["summary"])

    def test_visual_instance_requires_a_canonical_body(self):
        with self.assertRaises(DomainSimulationError):
            simulate_domain("mechanics-2d", {
                "domain_model": "mechanics_2d", "duration": 1, "dt": .01,
                "bodies": [{"id": "ball", "shape": "circle", "position": [0, 0]}],
                "visual_instances": [{
                    "id": "ball_h", "source_object_id": "missing",
                    "view": "horizontal_projection", "panel": "right",
                }],
            })


class Mechanics2DIntegrationTest(unittest.TestCase):
    def test_compiler_owns_html_and_validator_replays_trace(self):
        spec = world()
        raw = {
            "engine": "mechanics-2d",
            "render_spec": {"engine": "mechanics-2d", "fps": 30, "duration": 2},
            "world_spec_sha256": spec.canonical_hash(),
            "scenes": [{"scene_id": "SIM_1", "simulation_spec": {
                "domain_model": "mechanics_2d", "duration": 1, "dt": .01,
                "bodies": [{"id": "ball", "shape": "circle", "mass": 1, "radius": .1,
                            "position": [0, 1], "velocity": [1, 0]}],
                "static_geometry": [{"id": "ground", "type": "segment", "p1": [-2, 0], "p2": [2, 0], "normal": [0, 1]}],
                "annotations": [{"type": "trail", "target": "ball"}],
                "visual_instances": [
                    {"id": "ball_v", "source_object_id": "ball", "view": "vertical_projection", "panel": "left"},
                    {"id": "ball_h", "source_object_id": "ball", "view": "horizontal_projection", "panel": "right"},
                ],
            }}],
        }
        compiled = compile_domain_program(raw, spec)
        report = validate_program_payload(compiled, spec, [StoryBlock("SIM_1", "sim", "show ball")])
        self.assertTrue(report.valid, report.errors)
        self.assertIn("声明式二维力学", compiled["scenes"][0]["document"])
        self.assertIn("visualInstances", compiled["scenes"][0]["document"])
        self.assertNotIn("ball_h", compiled["scenes"][0]["trace"]["time_series"][0]["objects"])
        compiled["scenes"][0]["trace"]["duration"] = 99
        self.assertFalse(validate_program_payload(compiled, spec, [StoryBlock("SIM_1", "sim", "show")]).valid)

    def test_compiler_maps_world_point_mass_and_terminal_contact(self):
        spec = EduWorldSpec.from_dict({
            "problem_id": "projectile",
            "coordinate_system": {"type": "cartesian", "plane": "xy"},
            "objects": [
                {"id": "ball", "type": "point_mass"},
                {"id": "ground", "type": "static_plane"},
            ],
            "parameters": [], "constraints": [],
            "initial_state": {}, "expected_events": [], "final_state": {},
            "learning_goals": ["projectile motion"], "visual_conventions": {},
            "terminal_event": {
                "id": "impact", "type": "contact",
                "participants": ["ball", "ground"],
            },
        })
        raw = {
            "engine": "mechanics-2d",
            "render_spec": {"engine": "mechanics-2d", "fps": 30, "duration": 8},
            "world_spec_sha256": spec.canonical_hash(),
            "scenes": [{"scene_id": "SIM_1", "simulation_spec": {
                "domain_model": "mechanics_2d", "duration": 2, "dt": .001,
                "gravity": [0, -10],
                "bodies": [{
                    "id": "ball", "shape": "circle", "mass": 1,
                    "radius": .05, "position": [0, 20], "velocity": [6, 0],
                }],
                "static_geometry": [{
                    "id": "ground", "type": "segment",
                    "p1": [-1, 0], "p2": [20, 0], "normal": [0, 1],
                }],
            }}],
        }
        scene = compile_domain_program(raw, spec)["scenes"][0]
        self.assertEqual(scene["simulation_spec"]["bodies"][0]["collision_radius"], 0)
        self.assertEqual(scene["simulation_spec"]["terminal_contact"], ["ball", "ground"])
        impact = scene["trace"]["events"][-1]
        self.assertAlmostEqual(impact["t"], 2.0, places=8)
        self.assertAlmostEqual(impact["snapshot"]["objects"]["ball"]["position"][0], 12, places=8)
        self.assertAlmostEqual(impact["snapshot"]["objects"]["ball"]["kinetic_energy"], 218, places=6)
        self.assertEqual(scene["trace"]["duration"], 2.0)

    def test_cloned_teaching_body_is_still_rejected(self):
        spec_payload = world().to_dict()
        spec_payload["objects"] = [{"id": "ball", "type": "circle"}]
        spec = EduWorldSpec.from_dict(spec_payload)
        raw = {
            "engine": "mechanics-2d",
            "render_spec": {"engine": "mechanics-2d", "fps": 30, "duration": 1},
            "world_spec_sha256": spec.canonical_hash(),
            "scenes": [{"scene_id": "SIM_1", "simulation_spec": {
                "domain_model": "mechanics_2d", "duration": 1, "dt": .01,
                "gravity": [0, 0],
                "bodies": [
                    {"id": "ball", "shape": "circle", "position": [0, 0]},
                    {"id": "ball_h", "shape": "circle", "position": [0, 0]},
                ],
                "static_geometry": [{"id": "ground", "type": "segment", "p1": [-2, 0], "p2": [2, 0]}],
            }}],
        }
        compiled = compile_domain_program(raw, spec)
        report = validate_program_payload(
            compiled, spec, [StoryBlock("SIM_1", "sim", "show ball")]
        )
        self.assertFalse(report.valid)
        self.assertTrue(any("ball_h" in error for error in report.errors))
        self.assertTrue(
            any("ground" in warning and "auxiliary" in warning for warning in report.warnings)
        )


class SpatialGateTest(unittest.TestCase):
    def problem(self, question="小球在竖直平面内运动"):
        return K12Problem.from_record({"problem_id": "p", "question": question,
                                      "subject": "physics-g12", "simulation_type": "projectile"})

    def test_native_defaults_to_declarative_2d(self):
        route = EngineRouter().route(self.problem())
        self.assertEqual(route.engine, "mechanics-2d")
        self.assertEqual(route.visualization_mode, "schematic_2d")

    def test_unverifiable_3d_request_falls_back_to_2d(self):
        route = EngineRouter().route(self.problem(), visualization_decision={
            "mode": "spatial_3d", "criterion": "non_coplanar_motion",
            "evidence_quote": "3D看起来更漂亮",
        })
        self.assertEqual(route.engine, "mechanics-2d")
        self.assertFalse(route.spatial_audit["approved"])

    def test_exact_spatial_evidence_allows_3d(self):
        problem = self.problem("小球做不共面三维运动，求空间轨迹")
        request = {"mode": "spatial_3d", "criterion": "non_coplanar_motion",
                   "evidence_quote": "不共面三维运动"}
        audit = audit_spatial_request(problem, request)
        self.assertTrue(audit["approved"])
        self.assertEqual(EngineRouter().route(problem, visualization_decision=request).engine, "threejs-cannon")

    def test_cli_override_remains_explicit_authority(self):
        route = EngineRouter().route(self.problem(), requested="threejs-cannon")
        self.assertEqual(route.engine, "threejs-cannon")
        self.assertEqual(route.spatial_audit["reason"], "explicit user/CLI override")


if __name__ == "__main__":
    unittest.main()
