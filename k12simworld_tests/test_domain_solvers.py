import json
import math
import tempfile
import unittest
from pathlib import Path

from k12simworld.adapters import StaticResponseAdapter
from k12simworld.domain_compiler import compile_domain_program
from k12simworld.domain_solvers import (
    DomainSimulationError,
    simulate_charged_particles,
    simulate_dc_circuit,
    simulate_ode_system,
    trace_geometric_rays,
)
from k12simworld.models import EduWorldSpec, K12Problem, StoryBlock
from k12simworld.pipeline import K12SimWorldPipeline
from k12simworld.routing import EngineRouter
from k12simworld.validation import validate_program_payload


def world(problem_id, objects):
    return EduWorldSpec.from_dict(
        {
            "problem_id": problem_id,
            "coordinate_system": {"type": "cartesian"},
            "objects": [{"id": item, "type": "physics_object"} for item in objects],
            "parameters": [],
            "constraints": [],
            "initial_state": {},
            "expected_events": [],
            "final_state": {},
            "learning_goals": ["execute the governing rule"],
            "visual_conventions": {"background": "white"},
        }
    )


class ChargedParticleSolverTest(unittest.TestCase):
    def test_uniform_magnetic_field_preserves_speed_and_closes_orbit(self):
        trace = simulate_charged_particles(
            {
                "domain_model": "charged_particle_2d",
                "duration": 2 * math.pi,
                "dt": 0.005,
                "electric_field": [0, 0],
                "magnetic_field": [0, 0, 1],
                "bounds": {"x_min": -3, "x_max": 3, "y_min": -3, "y_max": 3},
                "particles": [
                    {"id": "particle", "mass": 1, "charge": 1, "position": [0, 0], "velocity": [1, 0]}
                ],
            }
        )
        summary = trace["summary"]["particle"]
        self.assertAlmostEqual(summary["initial_speed"], summary["final_speed"], places=9)
        self.assertLess(math.hypot(*summary["final_position"]), 0.02)

    def test_uniform_electric_field_accelerates_particle(self):
        trace = simulate_charged_particles(
            {
                "domain_model": "charged_particle_2d",
                "duration": 1,
                "dt": 0.01,
                "electric_field": [1, 0],
                "magnetic_field": [0, 0, 0],
                "particles": [
                    {"id": "particle", "mass": 1, "charge": 1, "position": [0, 0], "velocity": [0, 0]}
                ],
            }
        )
        final = trace["time_series"][-1]["objects"]["particle"]
        self.assertAlmostEqual(final["velocity"][0], 1.0, places=9)
        self.assertAlmostEqual(final["position"][0], 0.505, places=3)

    def test_bounded_field_region_emits_entry_event(self):
        trace = simulate_charged_particles(
            {
                "domain_model": "charged_particle_2d",
                "duration": 2,
                "dt": 0.1,
                "electric_field": [0, 0],
                "magnetic_field": [0, 0, 0],
                "field_regions": [
                    {
                        "id": "zone",
                        "bounds": {"x_min": 0, "x_max": 2, "y_min": -1, "y_max": 1},
                        "electric_field": [0, 0],
                        "magnetic_field": [0, 0, 0],
                        "mode": "override",
                    }
                ],
                "particles": [
                    {"id": "particle", "mass": 1, "charge": 1, "position": [-1, 0], "velocity": [1, 0]}
                ],
            }
        )
        entries = [event for event in trace["events"] if event["type"] == "field_region_entry"]
        self.assertEqual(entries[0]["participants"], ["particle", "zone"])


class EquationSystemSolverTest(unittest.TestCase):
    def test_rk4_harmonic_oscillator_returns_near_initial_state(self):
        trace = simulate_ode_system(
            {
                "domain_model": "ode_system",
                "duration": 2 * math.pi,
                "dt": 0.01,
                "variables": [
                    {"id": "x", "initial": 1},
                    {"id": "v", "initial": 0},
                ],
                "derivatives": {"x": "v", "v": "-x"},
                "observables": {"energy": "0.5*(x*x+v*v)"},
                "plot_channels": ["x", "v", "energy"],
            }
        )
        final = trace["time_series"][-1]
        self.assertAlmostEqual(final["state"]["x"], 1.0, places=7)
        self.assertAlmostEqual(final["state"]["v"], 0.0, places=7)
        self.assertAlmostEqual(final["observables"]["energy"], 0.5, places=8)

    def test_electromagnetic_induction_couples_speed_current_and_power(self):
        trace = simulate_ode_system(
            {
                "domain_model": "ode_system",
                "duration": 8,
                "dt": 0.01,
                "objects": [{"id": "rod", "kind": "rod"}],
                "variables": [{"id": "x", "initial": 0}, {"id": "v", "initial": 2}],
                "parameters": {"m": 1, "B": 1, "L": 1, "R": 2},
                "derivatives": {"x": "v", "v": "-(B*L)**2*v/(m*R)"},
                "observables": {
                    "current": "B*L*v/R",
                    "power": "(B*L*v/R)**2*R",
                },
                "plot_channels": ["v", "current", "power"],
                "visual_bindings": [
                    {"object_id": "rod", "type": "slider", "channel": "x"},
                    {"object_id": "rod", "type": "gauge", "channel": "current"},
                ],
                "event_conditions": [
                    {"id": "nearly_stopped", "expression": "abs(v)<0.05"}
                ],
            }
        )
        final = trace["time_series"][-1]
        self.assertAlmostEqual(final["state"]["v"], 2 * math.exp(-4), places=8)
        self.assertAlmostEqual(final["observables"]["current"], math.exp(-4), places=8)
        self.assertTrue(any(event.get("id") == "nearly_stopped" for event in trace["events"]))

    def test_expression_language_blocks_python_execution(self):
        for expression in ("__import__('os').system('id')", "2**1000000000"):
            with self.subTest(expression=expression), self.assertRaises(DomainSimulationError):
                simulate_ode_system(
                    {
                        "domain_model": "ode_system",
                        "duration": 1,
                        "dt": 0.1,
                        "variables": [{"id": "x", "initial": 0}],
                        "derivatives": {"x": expression},
                    }
                )


class CircuitSolverTest(unittest.TestCase):
    def test_ohms_law_and_meter_readings(self):
        trace = simulate_dc_circuit(
            {
                "domain_model": "dc_circuit",
                "duration": 1,
                "dt": 0.5,
                "ground": "gnd",
                "nodes": ["gnd", "n1", "n2"],
                "components": [
                    {"id": "battery", "type": "voltage_source", "node_a": "n1", "node_b": "gnd", "voltage": 6},
                    {"id": "A1", "type": "ammeter", "node_a": "n1", "node_b": "n2"},
                    {"id": "lamp", "type": "lamp", "node_a": "n2", "node_b": "gnd", "resistance": 6, "rated_power": 6},
                    {"id": "V1", "type": "voltmeter", "node_a": "n2", "node_b": "gnd"},
                ],
            }
        )
        state = trace["time_series"][0]["components"]
        self.assertAlmostEqual(state["A1"]["reading"], 1.0, places=5)
        self.assertAlmostEqual(state["V1"]["reading"], 6.0, places=5)
        self.assertAlmostEqual(state["lamp"]["power"], 6.0, places=5)
        self.assertAlmostEqual(state["lamp"]["brightness"], 1.0, places=5)

    def test_switch_action_changes_lamp_brightness(self):
        trace = simulate_dc_circuit(
            {
                "domain_model": "dc_circuit",
                "duration": 2,
                "dt": 1,
                "ground": "gnd",
                "nodes": ["gnd", "n1", "n2"],
                "components": [
                    {"id": "battery", "type": "voltage_source", "node_a": "n1", "node_b": "gnd", "voltage": 6},
                    {"id": "lamp", "type": "lamp", "node_a": "n1", "node_b": "n2", "resistance": 6, "rated_power": 6},
                    {"id": "S1", "type": "switch", "node_a": "n2", "node_b": "gnd", "closed": False},
                ],
                "actions": [{"time": 1, "target": "S1", "property": "closed", "value": True}],
            }
        )
        self.assertLess(trace["time_series"][0]["components"]["lamp"]["brightness"], 1e-6)
        self.assertGreater(trace["time_series"][-1]["components"]["lamp"]["brightness"], 0.99)


class RayOpticsSolverTest(unittest.TestCase):
    def test_mirror_reflects_ray(self):
        trace = trace_geometric_rays(
            {
                "domain_model": "geometric_ray_2d",
                "sources": [{"id": "ray", "origin": [0, 0], "direction": [1, 0]}],
                "elements": [{"id": "mirror", "type": "mirror", "p1": [1, -1], "p2": [1, 1]}],
                "max_interactions": 2,
                "max_distance": 5,
            }
        )
        event = trace["paths"][0]["interactions"][0]
        self.assertAlmostEqual(event["outgoing_direction"][0], -1.0, places=9)
        self.assertAlmostEqual(event["outgoing_direction"][1], 0.0, places=9)

    def test_parallel_ray_passes_through_lens_focus(self):
        trace = trace_geometric_rays(
            {
                "domain_model": "geometric_ray_2d",
                "sources": [{"id": "ray", "origin": [-2, 1], "direction": [1, 0]}],
                "elements": [
                    {"id": "lens", "type": "thin_lens", "x": 0, "optical_axis_y": 0, "aperture": 4, "focal_length": 2},
                    {"id": "screen", "type": "screen", "p1": [2, -2], "p2": [2, 2]},
                ],
            }
        )
        screen_hit = trace["paths"][0]["interactions"][-1]
        self.assertEqual(screen_hit["type"], "screen")
        self.assertAlmostEqual(screen_hit["point"][0], 2.0, places=8)
        self.assertAlmostEqual(screen_hit["point"][1], 0.0, places=8)

    def test_snell_refraction_bends_toward_normal(self):
        incoming = [math.cos(math.pi / 6), math.sin(math.pi / 6)]
        trace = trace_geometric_rays(
            {
                "domain_model": "geometric_ray_2d",
                "sources": [{"id": "ray", "origin": [-1, 0], "direction": incoming}],
                "elements": [
                    {"id": "glass", "type": "refractive_interface", "p1": [0, -2], "p2": [0, 2], "n1": 1, "n2": 1.5}
                ],
                "max_interactions": 1,
            }
        )
        outgoing = trace["paths"][0]["interactions"][0]["outgoing_direction"]
        angle = abs(math.atan2(outgoing[1], outgoing[0]))
        self.assertAlmostEqual(angle, math.asin(1 / 3), places=8)


class DomainIntegrationTest(unittest.TestCase):
    def test_router_uses_execution_tier_before_mechanics_words(self):
        problem = K12Problem.from_record(
            {
                "problem_id": "em1",
                "question": "带电粒子如何运动？",
                "subject": "physics-g12",
                "simulation_type": "force_and_motion",
                "source_metadata": {"execution_tier": {"subtype": "electromagnetic_dynamics"}},
            }
        )
        self.assertEqual(EngineRouter().route(problem).engine, "equation-solver")

    def test_frozen_equation_tier_beats_incidental_circuit_word(self):
        problem = K12Problem.from_record(
            {
                "problem_id": "em2",
                "question": "带电粒子附近的电路图仅用于说明加速电压，粒子如何运动？",
                "subject": "physics-g12",
                "simulation_type": "force_and_motion",
                "source_metadata": {"execution_tier": {"subtype": "electromagnetic_dynamics"}},
            }
        )
        self.assertEqual(EngineRouter().route(problem).engine, "equation-solver")

    def test_compiler_executes_and_validator_replays_trace(self):
        spec = world("em1", ["particle"])
        storyboard = [
            StoryBlock("TEXT_1", "text", "观察"), StoryBlock("SIM_1", "sim", "轨迹")
        ]
        raw = {
            "engine": "equation-solver",
            "render_spec": {"engine": "equation-solver", "fps": 30, "duration": 2},
            "world_spec_sha256": spec.canonical_hash(),
            "scenes": [
                {
                    "scene_id": "SIM_1",
                    "simulation_spec": {
                        "domain_model": "charged_particle_2d",
                        "duration": 2,
                        "dt": 0.02,
                        "electric_field": [0, 0],
                        "magnetic_field": [0, 0, 1],
                        "particles": [
                            {"id": "particle", "mass": 1, "charge": 1, "position": [0, 0], "velocity": [1, 0]}
                        ],
                    },
                }
            ],
        }
        compiled = compile_domain_program(raw, spec)
        report = validate_program_payload(compiled, spec, storyboard)
        self.assertTrue(report.valid, report.errors)
        self.assertIn("<canvas", compiled["scenes"][0]["document"])
        compiled["scenes"][0]["trace"]["duration"] = 999
        self.assertFalse(validate_program_payload(compiled, spec, storyboard).valid)

    def test_compiler_renders_equation_system_trace(self):
        spec = world("ode1", ["rod"])
        raw = {
            "engine": "equation-solver",
            "render_spec": {"engine": "equation-solver", "fps": 30, "duration": 2},
            "world_spec_sha256": spec.canonical_hash(),
            "scenes": [
                {
                    "scene_id": "SIM_1",
                    "simulation_spec": {
                        "domain_model": "ode_system",
                        "duration": 2,
                        "dt": 0.02,
                        "objects": [{"id": "rod"}],
                        "variables": [{"id": "x", "initial": 0}, {"id": "v", "initial": 1}],
                        "derivatives": {"x": "v", "v": "-v"},
                        "observables": {"speed": "abs(v)"},
                        "plot_channels": ["x", "speed"],
                        "visual_bindings": [
                            {"object_id": "rod", "type": "slider", "channel": "x"}
                        ],
                    },
                }
            ],
        }
        compiled = compile_domain_program(raw, spec)
        scene = compiled["scenes"][0]
        self.assertEqual(scene["trace"]["domain_model"], "ode_system")
        self.assertIn("受限表达式 + RK4", scene["document"])

    def test_offline_pipeline_writes_circuit_trace(self):
        problem = K12Problem.from_record(
            {
                "problem_id": "c1",
                "question": "闭合开关后灯泡如何变化？",
                "subject": "physics-g9",
                "simulation_type": "circuit",
            }
        )
        story = {
            "analysis": "闭合后形成通路",
            "requires_simulation": True,
            "final_answer": "灯泡变亮",
            "blocks": [
                {"block_id": "TEXT_1", "kind": "text", "content": "观察电路"},
                {"block_id": "SIM_1", "kind": "sim", "content": "闭合开关"},
            ],
        }
        spec_dict = world("c1", ["battery", "lamp", "S1"]).to_dict()
        spec_dict["target_observables"] = [
            {
                "id": "lamp_bright",
                "scene_id": "SIM_1",
                "at": "final",
                "path": "trace.summary.components.lamp.brightness",
                "expected": 1.0,
                "operator": "approximately_equal",
                "unit": "1",
                "absolute_tolerance": 0.01,
                "relative_tolerance": 0.01,
                "required": True,
            }
        ]
        spec = EduWorldSpec.from_dict(spec_dict)
        program = {
            "engine": "circuit-solver",
            "render_spec": {"engine": "circuit-solver", "fps": 30, "duration": 2},
            "world_spec_sha256": spec.canonical_hash(),
            "scenes": [
                {
                    "scene_id": "SIM_1",
                    "simulation_spec": {
                        "domain_model": "dc_circuit",
                        "duration": 2,
                        "dt": 1,
                        "ground": "gnd",
                        "nodes": ["gnd", "n1", "n2"],
                        "components": [
                            {"id": "battery", "type": "voltage_source", "node_a": "n1", "node_b": "gnd", "voltage": 6},
                            {"id": "lamp", "type": "lamp", "node_a": "n1", "node_b": "n2", "resistance": 6, "rated_power": 6},
                            {"id": "S1", "type": "switch", "node_a": "n2", "node_b": "gnd", "closed": False},
                        ],
                        "actions": [{"time": 1, "target": "S1", "property": "closed", "value": True}],
                    },
                }
            ],
        }
        adapter = StaticResponseAdapter([json.dumps(story), json.dumps(spec.to_dict()), json.dumps(program)])
        with tempfile.TemporaryDirectory() as directory:
            manifest = K12SimWorldPipeline(adapter, directory, render=False).generate(problem)
            self.assertTrue(manifest.success, manifest.error)
            self.assertEqual(len(manifest.trace_paths), 1)
            trace = json.loads(Path(manifest.trace_paths[0]).read_text(encoding="utf-8"))
            self.assertGreater(trace["summary"]["components"]["lamp"]["brightness"], 0.99)


if __name__ == "__main__":
    unittest.main()
