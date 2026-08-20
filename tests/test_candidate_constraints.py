import json
import tempfile
import unittest
from pathlib import Path

from k12simworld.adapters import StaticResponseAdapter
from k12simworld.candidate_constraints import (
    CandidateSolution,
    SimulationContract,
    build_observed_trace,
    validate_candidate_contract,
)
from k12simworld.domain_solvers import simulate_domain
from k12simworld.models import ContractError, EduWorldSpec, K12Problem
from k12simworld.pipeline import K12SimWorldPipeline


def _problem():
    return K12Problem.from_record(
        {
            "hash_id": "candidate-target-1",
            "question": "A ball starts at x=0 with speed 1 m/s. Where is it after 1 s?",
            "subject": "physics-g9",
            "simulation_type": "projectile",
            "format_answer": {
                "format_solution": ["gold must remain hidden"],
                "ground_truth": ["1 m"],
            },
        }
    )


def _story():
    return {
        "analysis": "Uniform motion gives x=vt=1 m.",
        "requires_simulation": True,
        "final_answer": "1 m",
        "solution": {
            "analysis": "Uniform motion gives x=vt=1 m.",
            "givens": [
                {"id": "speed", "value": 1, "unit": "m/s", "source": "problem"},
                {"id": "time", "value": 1, "unit": "s", "source": "problem"},
            ],
            "derived_values": [
                {"id": "position_x", "expression": "speed*time", "value": 1, "unit": "m"}
            ],
            "assumptions": ["uniform motion"],
            "final_answer": "1 m",
            "confidence": 1.0,
        },
        "visualization_decision": {"mode": "schematic_2d", "reason": "planar motion"},
        "blocks": [{"block_id": "SIM_1", "kind": "sim", "content": "show the motion"}],
    }


def _world_spec():
    return {
        "schema_version": "1.0",
        "problem_id": "candidate-target-1",
        "coordinate_system": {"axes": "xy", "units": {"length": "m", "time": "s"}},
        "objects": [{"id": "ball", "type": "ball"}],
        "parameters": [
            {"id": "speed", "value": 1, "unit": "m/s", "justification": "problem"}
        ],
        "constraints": [{"type": "uniform_motion", "description": "no net force"}],
        "initial_state": {
            "time": 0,
            "objects": {"ball": {"position": [0, 0], "velocity": [1, 0]}},
        },
        "expected_events": [],
        "final_state": {"objects": {"ball": {"position": [1, 0]}}},
        "terminal_event": {},
        "target_observables": [
            {
                "id": "final_x",
                "scene_id": "SIM_1",
                "at": "final",
                "path": "objects.ball.position.0",
                "expected": 1,
                "operator": "approximately_equal",
                "unit": "m",
                "absolute_tolerance": 0.02,
                "relative_tolerance": 0.01,
                "required": True,
            }
        ],
        "invariants": [
            {
                "id": "constant_speed",
                "scene_id": "SIM_1",
                "path": "objects.ball.speed",
                "type": "constant",
                "value": 1,
                "tolerance": 0.01,
                "required": True,
            }
        ],
        "learning_goals": ["connect speed, time, and displacement"],
        "visual_conventions": {"colors": {"ball": "#2563eb"}, "labels": True},
    }


def _program(velocity):
    return {
        "engine": "mechanics-2d",
        "render_spec": {"engine": "mechanics-2d", "fps": 30, "duration": 1},
        "scenes": [
            {
                "scene_id": "SIM_1",
                "simulation_spec": {
                    "domain_model": "mechanics_2d",
                    "duration": 1,
                    "dt": 0.01,
                    "playback_duration": 1,
                    "gravity": [0, 0],
                    "bounds": {"x_min": -1, "x_max": 2, "y_min": -1, "y_max": 1},
                    "units": {"length": "m", "time": "s", "mass": "kg"},
                    "bodies": [
                        {
                            "id": "ball",
                            "shape": "circle",
                            "motion_type": "dynamic",
                            "mass": 1,
                            "radius": 0.1,
                            "position": [0, 0],
                            "velocity": [velocity, 0],
                            "linear_damping": 0,
                            "label": "ball",
                            "color": "#2563eb",
                        }
                    ],
                    "static_geometry": [],
                    "springs": [],
                    "distance_constraints": [],
                    "forces": [],
                    "actions": [],
                    "annotations": [{"type": "trail", "target": "ball"}],
                },
            }
        ],
    }


class CandidateConstraintTest(unittest.TestCase):
    def test_structured_final_answer_wins_without_contract_failure(self):
        story = _story()
        story["final_answer"] = "conflicting summary"
        solution = CandidateSolution.from_story(_problem().problem_id, story)
        self.assertEqual(solution.final_answer, "1 m")

    def test_solution_contract_and_observed_trace_are_gold_free(self):
        solution = CandidateSolution.from_story(_problem().problem_id, _story())
        spec = EduWorldSpec.from_dict(_world_spec())
        contract = SimulationContract.from_world_spec(solution, spec)
        self.assertTrue(contract.evaluable)
        self.assertNotIn("gold", json.dumps(contract.to_dict()))

    def test_terminal_event_uses_pre_contact_snapshot(self):
        contract = SimulationContract(
            problem_id="impact",
            candidate_solution_sha256="candidate",
            initial_state={},
            final_state={},
            terminal_event={
                "type": "contact",
                "participants": ["ball", "ground"],
            },
            expected_events=[],
            target_observables=[{
                "id": "impact_energy",
                "scene_id": "SIM_1",
                "at": "terminal_event",
                "path": "objects.ball.kinetic_energy",
                "expected": 218,
                "absolute_tolerance": .01,
                "relative_tolerance": 0,
                "required": True,
            }],
            invariants=[],
        )
        program = {
            "scenes": [{
                "scene_id": "SIM_1",
                "trace": {
                    "time_series": [
                        {"t": 0, "objects": {"ball": {"kinetic_energy": 18}}},
                        {"t": 2, "objects": {"ball": {"kinetic_energy": 18}}},
                    ],
                    "events": [{
                        "type": "contact_begin",
                        "participants": ["ball", "ground"],
                        "t": 2,
                        "snapshot": {
                            "t": 2,
                            "objects": {"ball": {"kinetic_energy": 218}},
                        },
                    }],
                },
            }],
        }
        report = validate_candidate_contract(contract, program)
        self.assertTrue(report["passed"], report)

    def test_pipeline_repairs_a_trace_that_misses_candidate_target(self):
        spec = EduWorldSpec.from_dict(_world_spec())
        first_program = _program(0)
        repaired_program = _program(1)
        first_program["world_spec_sha256"] = spec.canonical_hash()
        repaired_program["world_spec_sha256"] = spec.canonical_hash()
        responses = [
            json.dumps(_story()),
            json.dumps(_world_spec()),
            json.dumps(first_program),
            json.dumps(repaired_program),
        ]
        with tempfile.TemporaryDirectory() as directory:
            manifest = K12SimWorldPipeline(
                StaticResponseAdapter(responses), directory, render=False
            ).generate(_problem(), requested_engine="mechanics-2d")

            self.assertTrue(manifest.success, manifest.error)
            self.assertTrue(manifest.repaired)
            self.assertEqual(manifest.diagnostics["model_calls"], 4)
            self.assertEqual(manifest.diagnostics["candidate_target_status"], "passed")
            validation = json.loads(
                Path(manifest.target_validation_path).read_text(encoding="utf-8")
            )
            self.assertEqual(len(validation["attempts"]), 2)
            self.assertEqual(validation["attempts"][0]["status"], "failed")
            self.assertEqual(validation["attempts"][1]["status"], "passed")
            observed = json.loads(
                Path(manifest.observed_trace_path).read_text(encoding="utf-8")
            )
            self.assertIn("SIM_1.ball.position", observed["trajectories"])


    def test_failed_candidate_target_is_reported_without_blocking_generation(self):
        spec = EduWorldSpec.from_dict(_world_spec())
        bad_program = _program(0)
        bad_program["world_spec_sha256"] = spec.canonical_hash()
        responses = [
            json.dumps(_story()),
            json.dumps(_world_spec()),
            json.dumps(bad_program),
            json.dumps(bad_program),
        ]
        with tempfile.TemporaryDirectory() as directory:
            manifest = K12SimWorldPipeline(
                StaticResponseAdapter(responses), directory, render=False
            ).generate(_problem(), requested_engine="mechanics-2d")

        self.assertTrue(manifest.success, manifest.error)
        self.assertEqual(manifest.diagnostics["candidate_target_status"], "failed")
        warning_codes = {
            item["code"] for item in manifest.diagnostics["pre_render_warnings"]
        }
        self.assertIn("candidate_target_validation_failed", warning_codes)

    def test_bound_formula_invariant_uses_safe_scalar_lookups(self):
        trace = simulate_domain("mechanics-2d", {
            "domain_model": "mechanics_2d", "duration": 1, "dt": .01,
            "gravity": [0, -10],
            "bodies": [{"id": "ball", "shape": "circle", "mass": 2,
                        "radius": .1, "position": [0, 3], "velocity": [2, 4]}],
        })
        contract = SimulationContract(
            problem_id="energy", candidate_solution_sha256="candidate",
            initial_state={}, final_state={}, terminal_event={}, expected_events=[],
            target_observables=[], invariants=[{
                "id": "energy_constant", "scene_id": "SIM_1",
                "expression": "ke + pe",
                "bindings": {
                    "ke": "objects.ball.kinetic_energy",
                    "pe": "objects.ball.potential_energy",
                },
                "display_formula": "E_k + E_p = constant",
                "result_unit": "J", "type": "constant", "tolerance": 1e-8,
            }],
        )
        report = validate_candidate_contract(
            contract, {"scenes": [{"scene_id": "SIM_1", "trace": trace}]}
        )
        self.assertTrue(report["passed"], report)
        check = report["checks"][0]
        self.assertEqual(check["expression"], "ke + pe")
        self.assertEqual(check["unit"], "J")

    def test_world_spec_rejects_arithmetic_hidden_in_path(self):
        value = _world_spec()
        value["invariants"][0]["path"] = (
            "objects.ball.kinetic_energy + objects.ball.potential_energy"
        )
        with self.assertRaisesRegex(ContractError, "dotted lookup path"):
            EduWorldSpec.from_dict(value)

    def test_unsafe_formula_is_reported_without_execution(self):
        contract = SimulationContract(
            problem_id="unsafe", candidate_solution_sha256="candidate",
            initial_state={}, final_state={}, terminal_event={}, expected_events=[],
            target_observables=[{
                "id": "unsafe", "scene_id": "SIM_1", "at": "final",
                "expression": "__import__(name)",
                "bindings": {"name": "objects.ball.speed"}, "expected": 1,
            }], invariants=[],
        )
        report = validate_candidate_contract(contract, {"scenes": [{
            "scene_id": "SIM_1", "trace": {"time_series": [
                {"t": 0, "objects": {"ball": {"speed": 1}}}
            ]}
        }]})
        self.assertFalse(report["passed"])
        self.assertIn("unknown name", report["checks"][0]["error_message"])


if __name__ == "__main__":
    unittest.main()
