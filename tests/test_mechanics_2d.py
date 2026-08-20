import json
import math
import tempfile
import unittest
from pathlib import Path

from k12simworld.domain_compiler import compile_domain_program
from k12simworld.domain_solvers import DomainSimulationError, simulate_domain
from k12simworld.models import EduWorldSpec, K12Problem, StoryBlock
from k12simworld.prompts import program_prompt, storyboard_prompt
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

    def test_pendulum_constraint_preserves_tangent_motion_and_energy(self):
        trace = simulate_domain("mechanics-2d", {
            "domain_model": "mechanics_2d", "duration": 2, "dt": 1 / 240,
            "gravity": [0, -9.8],
            "bodies": [{
                "id": "bob", "shape": "circle", "mass": .1, "radius": .05,
                "position": [-.8, -.6], "velocity": [0, 0],
            }],
            "distance_constraints": [{
                "id": "rope", "anchor_a": [0, 0], "body_b": "bob", "length": 1,
            }],
        })
        self.assertGreater(trace["summary"]["bob"]["x_range"][1], .75)
        energies = [
            frame["objects"]["bob"]["mechanical_energy"]
            for frame in trace["time_series"]
        ]
        self.assertLess(max(energies) - min(energies), .01)
        for frame in trace["time_series"]:
            self.assertAlmostEqual(
                math.hypot(*frame["objects"]["bob"]["position"]), 1, places=7
            )

    def test_timed_string_break_preserves_state_and_releases_body(self):
        trace = simulate_domain("mechanics-2d", {
            "domain_model": "mechanics_2d", "duration": 1, "dt": .001,
            "gravity": [0, 0],
            "bodies": [{
                "id": "bob", "shape": "circle", "mass": 1, "radius": .05,
                "position": [1, 0], "velocity": [0, 1],
            }],
            "distance_constraints": [{
                "id": "rope", "anchor_a": [0, 0], "body_b": "bob", "length": 1,
            }],
            "actions": [{
                "time": .5, "type": "remove_distance_constraint", "target": "rope",
                "event_id": "break", "event_type": "string_break", "label": "断绳",
            }],
        })
        event = next(item for item in trace["events"] if item.get("id") == "break")
        self.assertEqual(event["type"], "string_break")
        before = max(
            (frame for frame in trace["time_series"] if frame["t"] < event["t"]),
            key=lambda frame: frame["t"],
        )
        after = event["snapshot"]
        self.assertIn("rope", before["active_constraints"])
        self.assertNotIn("rope", after["active_constraints"])
        self.assertLess(
            math.dist(before["objects"]["bob"]["velocity"], after["objects"]["bob"]["velocity"]),
            .01,
        )
        self.assertGreater(math.hypot(*trace["summary"]["bob"]["final_position"]), 1.05)

    def test_nested_time_trigger_alias_is_canonicalized(self):
        trace = simulate_domain("mechanics-2d", {
            "domain_model": "mechanics_2d", "duration": 1, "dt": .01,
            "gravity": [0, 0],
            "bodies": [{
                "id": "ball", "shape": "circle", "mass": 1, "radius": .05,
                "position": [1, 0], "velocity": [0, 1],
            }],
            "distance_constraints": [{
                "id": "rope", "anchor_a": [0, 0], "body_b": "ball", "length": 1,
            }],
            "actions": [{
                "trigger": {"type": "time", "value": 0.0},
                "type": "remove_distance_constraint", "target": "rope",
                "event_id": "break_at_A", "event_type": "string_break",
            }],
            "phases": [
                {"id": "initial", "start_time": 0, "end_time": 0, "label": "初始瞬间"},
                {"id": "free", "start_time": 0, "end_time": 1, "label": "自由运动"},
            ],
        })
        event = next(item for item in trace["events"] if item.get("id") == "break_at_A")
        self.assertEqual(event["t"], 0.0)
        self.assertNotIn("rope", event["snapshot"]["active_constraints"])
        self.assertEqual(trace["phases"][0]["start_time"], trace["phases"][0]["end_time"])

    def test_conflicting_nested_and_top_level_times_are_rejected(self):
        with self.assertRaisesRegex(DomainSimulationError, "conflicts"):
            simulate_domain("mechanics-2d", {
                "domain_model": "mechanics_2d", "duration": 1, "dt": .01,
                "bodies": [{"id": "ball", "shape": "circle", "position": [0, 0]}],
                "actions": [{
                    "time": .2, "trigger": {"type": "time", "value": .3},
                    "type": "set_velocity", "target": "ball", "value": [1, 0],
                }],
            })

    def test_position_crossing_can_trigger_string_break(self):
        trace = simulate_domain("mechanics-2d", {
            "domain_model": "mechanics_2d", "duration": 1.2, "dt": 1 / 240,
            "gravity": [0, -9.8],
            "bodies": [{
                "id": "bob", "shape": "circle", "mass": .1, "radius": .05,
                "position": [-.8, -.6], "velocity": [0, 0],
            }],
            "distance_constraints": [{
                "id": "rope", "anchor_a": [0, 0], "body_b": "bob", "length": 1,
            }],
            "actions": [{
                "type": "remove_distance_constraint", "target": "rope",
                "trigger": {
                    "type": "position_crossing", "body": "bob", "axis": "x",
                    "value": 0, "direction": "positive", "after_time": .1,
                },
                "event_id": "break_at_C", "event_type": "string_break",
            }],
            "phases": [
                {"id": "swing", "start_time": 0, "end_time": .53, "label": "摆动"},
                {"id": "flight", "start_time": .53, "end_time": 1.2, "label": "抛体"},
            ],
        })
        event = next(item for item in trace["events"] if item.get("id") == "break_at_C")
        self.assertAlmostEqual(event["t"], .53, delta=.03)
        event_frame = min(trace["time_series"], key=lambda frame: abs(frame["t"] - event["t"]))
        self.assertNotIn("rope", event_frame["active_constraints"])
        self.assertGreater(event_frame["objects"]["bob"]["velocity"][0], 2.5)
        self.assertEqual([phase["id"] for phase in trace["phases"]], ["swing", "flight"])

    def test_emit_event_records_state_without_changing_motion(self):
        trace = simulate_domain("mechanics-2d", {
            "domain_model": "mechanics_2d", "duration": 1, "dt": .01,
            "gravity": [0, 0],
            "bodies": [{
                "id": "ball", "shape": "circle", "mass": 1, "radius": .05,
                "position": [0, 0], "velocity": [1, 0],
            }],
            "actions": [{
                "type": "emit_event", "target": "ball",
                "trigger": {
                    "type": "position_crossing", "body": "ball", "axis": "x",
                    "value": .5, "direction": "positive", "after_time": 0,
                },
                "event_id": "midpoint", "event_type": "bottom_reached",
                "participants": ["ball"], "label": "到达底端",
            }],
        })

        event = next(item for item in trace["events"] if item.get("id") == "midpoint")
        self.assertEqual(event["action_type"], "emit_event")
        self.assertAlmostEqual(event["t"], .5, delta=.011)
        self.assertAlmostEqual(event["snapshot"]["objects"]["ball"]["velocity"][0], 1)
        self.assertAlmostEqual(trace["summary"]["ball"]["final_position"][0], 1)

    def test_circular_path_releases_at_endpoint_with_tangent_velocity(self):
        trace = simulate_domain("mechanics-2d", {
            "domain_model": "mechanics_2d", "duration": 4, "dt": .002,
            "gravity": [0, 0],
            "bounds": {"x_min": -2, "x_max": 3, "y_min": -2, "y_max": 2},
            "bodies": [{
                "id": "ball", "shape": "circle", "mass": 1, "radius": .05,
                "position": [-1, 0], "velocity": [0, 1],
            }],
            "path_constraints": [{
                "id": "arc_track", "body": "ball", "type": "circular_arc",
                "center": [0, 0], "radius": 1,
                "start_angle": math.pi, "end_angle": 0,
                "auto_release": "end", "release_event_id": "leave_track",
                "release_event_type": "path_release", "label": "离开轨道",
            }],
        })

        event = next(item for item in trace["events"] if item.get("id") == "leave_track")
        state = event["snapshot"]["objects"]["ball"]
        self.assertEqual(event["action_type"], "automatic_path_release")
        self.assertEqual(event["endpoint"], "end")
        self.assertAlmostEqual(event["t"], math.pi, delta=.004)
        self.assertAlmostEqual(state["position"][0], 1, places=6)
        self.assertAlmostEqual(state["position"][1], 0, places=6)
        self.assertAlmostEqual(state["velocity"][0], 0, places=6)
        self.assertAlmostEqual(state["velocity"][1], -1, places=6)
        self.assertNotIn("arc_track", event["snapshot"]["active_constraints"])
        self.assertLess(trace["summary"]["ball"]["final_position"][1], -.8)
        self.assertEqual(trace["path_constraints"][0]["type"], "circular_arc")

    def test_curve_constraints_alias_is_canonicalized(self):
        trace = simulate_domain("mechanics-2d", {
            "domain_model": "mechanics_2d", "duration": 1, "dt": .01,
            "gravity": [0, 0],
            "bodies": [{"id": "ball", "shape": "circle", "position": [0, 0], "velocity": [1, 1]}],
            "curve_constraints": [{
                "id": "curve", "body": "ball", "type": "bezier",
                "points": [[0, 0], [.5, .5], [1, 0]], "auto_release": "none",
            }],
        })
        self.assertEqual(trace["path_constraints"][0]["type"], "bezier")
        self.assertIn("curve", trace["time_series"][-1]["active_constraints"])

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
        self.assertIn("const offsetX=(W-worldWidth*scale)/2", compiled["scenes"][0]["document"])
        self.assertNotIn("(x-b.x_min)/(b.x_max-b.x_min)*(W-2*pad)", compiled["scenes"][0]["document"])
        self.assertIn("(TRACE.path_constraints||[]).forEach(drawPath)", compiled["scenes"][0]["document"])
        self.assertIn("visualInstances", compiled["scenes"][0]["document"])
        self.assertIn("visualStrategy==='component_decomposition'", compiled["scenes"][0]["document"])
        self.assertEqual(
            compiled["scenes"][0]["trace"]["visual_strategy"], "continuous_process"
        )
        self.assertNotIn("ball_h", compiled["scenes"][0]["trace"]["time_series"][0]["objects"])
        compiled["scenes"][0]["trace"]["duration"] = 99
        self.assertFalse(validate_program_payload(compiled, spec, [StoryBlock("SIM_1", "sim", "show")]).valid)

    def test_compiler_persists_canonical_top_level_action_time(self):
        spec = world()
        raw = {
            "engine": "mechanics-2d",
            "render_spec": {"engine": "mechanics-2d", "fps": 5, "duration": 1},
            "world_spec_sha256": spec.canonical_hash(),
            "scenes": [{"scene_id": "SIM_1", "simulation_spec": {
                "domain_model": "mechanics_2d", "duration": 1, "dt": .01,
                "gravity": [0, 0],
                "bodies": [{
                    "id": "ball", "shape": "circle", "position": [1, 0],
                    "velocity": [0, 1],
                }],
                "distance_constraints": [{
                    "id": "rope", "anchor_a": [0, 0], "body_b": "ball", "length": 1,
                }],
                "actions": [{
                    "trigger": {"type": "at_time", "time": 0},
                    "type": "remove_distance_constraint", "target": "rope",
                }],
            }}],
        }
        scene = compile_domain_program(raw, spec)["scenes"][0]
        action = scene["simulation_spec"]["actions"][0]
        self.assertEqual(action["time"], 0)
        self.assertNotIn("trigger", action)
        self.assertIn("converted trigger.type=at_time", scene["simulation_spec_normalizations"][0])

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

    def test_prompts_default_to_complete_process_not_projection_panels(self):
        problem = self.problem("单摆到最低点时剪断绳子，观察后续运动")
        route = EngineRouter().route(problem)
        prompt = program_prompt(
            problem,
            [StoryBlock("SIM_1", "sim", "从释放、摆动到剪断后的完整过程")],
            world(),
            route,
            {},
        )
        self.assertIn('"visual_strategy":"continuous_process"', prompt)
        self.assertIn('"visual_instances":[]', prompt)
        self.assertNotIn('"id":"ball_v"', prompt)
        story = storyboard_prompt(problem)
        self.assertIn("mutually exclusive interventions", story)
        self.assertIn("complete physical world", story)

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
