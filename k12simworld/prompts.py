"""Versioned prompts for the three-stage K12SimWorld generation protocol."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .models import EduWorldSpec, K12Problem, StoryBlock
from .routing import RouteDecision


PROMPT_VERSION = "k12simworld-v1.1-domain-solvers"


def _domain_program_contract(route: RouteDecision) -> str:
    if route.engine == "equation-solver":
        return """Do not write HTML or an integrator. Choose exactly one equation domain model.

For a charged particle in prescribed electric/magnetic fields, return:
{
  "domain_model":"charged_particle_2d", "duration":8, "dt":0.0166666667,"playback_duration":8,
  "electric_field":[Ex,Ey], "magnetic_field":[0,0,Bz],
  "field_regions":[{"id":"field_zone","bounds":{"x_min":0,"x_max":5,"y_min":-3,"y_max":3},
    "electric_field":[0,0],"magnetic_field":[0,0,1],"mode":"override"}],
  "bounds":{"x_min":-10,"x_max":10,"y_min":-10,"y_max":10},
  "units":{"position":"m","time":"s","mass":"kg","charge":"C"},
  "particles":[{"id":"world_object_id","mass":1,"charge":1,
    "position":[x,y],"velocity":[vx,vy],"label":"...","color":"#2563eb"}]
}
The trusted runtime integrates q(E+v×B) using a fixed-step Boris solver. Bx and By must be zero.
Optional rectangular field_regions support entry into or exit from bounded uniform-field zones.
Choose dimensional or normalized values only when justified by the problem and WorldSpec.

For coupled mechanics/electromagnetic induction or another explicit first-order equation system,
return a safe declarative ODE (never Python/JavaScript):
{
  "domain_model":"ode_system", "duration":8, "dt":0.01, "playback_duration":8,
  "objects":[{"id":"world_object_id","label":"conducting rod","kind":"rod","color":"#2563eb"}],
  "variables":[
    {"id":"x","initial":0,"label":"position","unit":"m","color":"#2563eb"},
    {"id":"v","initial":2,"label":"speed","unit":"m/s","color":"#dc2626"}
  ],
  "parameters":{"m":1,"B":1,"L":1,"R":2,"drive":1},
  "derivatives":{"x":"v","v":"drive-(B*L)*(B*L)*v/(m*R)"},
  "observables":{"emf":"B*L*v","current":"B*L*v/R","power":"(B*L*v/R)**2*R"},
  "plot_channels":["v","current","power"],
  "visual_bindings":[
    {"object_id":"world_object_id","type":"slider","channel":"x","label":"rod position"},
    {"object_id":"world_object_id","type":"gauge","channel":"current","label":"ammeter"}
  ],
  "actions":[{"time":4,"target":"drive","value":0}],
  "event_conditions":[{"id":"nearly_stopped","expression":"abs(v)<0.01","terminal":false}]
}
The trusted runtime parses a restricted arithmetic language and integrates with fixed-step RK4.
Allowed functions are abs, min, max, sqrt, sin, cos, tan, exp, log, floor, and ceil; arbitrary
code, attribute access, indexing, imports, and observable-to-observable references are forbidden.
Visual binding types are slider, gauge, rotor, and lamp. Every object id must also occur in
WorldSpec. Use SI or clearly stated normalized units and encode only equations justified by the
problem; do not invent missing constitutive laws or parameters."""
    if route.engine == "circuit-solver":
        return """Do not write HTML or solve the circuit yourself. For every scene return simulation_spec:
{
  "domain_model":"dc_circuit", "duration":8, "dt":0.1, "ground":"gnd",
  "nodes":[{"id":"gnd","x":-1,"y":-1,"label":"GND"}, ...],
  "components":[
    {"id":"battery","type":"voltage_source","node_a":"n1","node_b":"gnd","voltage":6},
    {"id":"lamp","type":"lamp","node_a":"n1","node_b":"gnd","resistance":6,"rated_power":6},
    {"id":"A1","type":"ammeter","node_a":"n1","node_b":"n2","resistance":0.000001},
    {"id":"V1","type":"voltmeter","node_a":"n1","node_b":"gnd","resistance":1000000000},
    {"id":"S1","type":"switch","node_a":"n2","node_b":"gnd","closed":false}
  ],
  "actions":[{"time":2,"target":"S1","property":"closed","value":true}]
}
Supported component types are resistor, lamp, ammeter, voltmeter, switch, wire,
voltage_source, and current_source. The trusted runtime applies KCL/KVL/Ohm's law and derives
node voltages, branch currents, meter readings, lamp power, and normalized brightness. Nodes
must express the topology visible in the actual problem image; do not replace topology with a
visually similar but electrically different circuit."""
    if route.engine == "ray-optics":
        return """Do not write HTML or draw guessed rays. For every scene return simulation_spec:
{
  "domain_model":"geometric_ray_2d", "max_interactions":12, "max_distance":30,
  "sources":[{"id":"ray1","origin":[x,y],"direction":[dx,dy],
    "refractive_index":1,"label":"incident ray","color":"#dc2626"}],
  "elements":[
    {"id":"M1","type":"mirror","p1":[x1,y1],"p2":[x2,y2]},
    {"id":"L1","type":"thin_lens","x":5,"optical_axis_y":0,"aperture":8,"focal_length":3},
    {"id":"I1","type":"refractive_interface","p1":[x1,y1],"p2":[x2,y2],"n1":1,"n2":1.5},
    {"id":"screen","type":"screen","p1":[x1,y1],"p2":[x2,y2]}
  ]
}
Supported optical elements are mirror, refractive_interface, thin_lens, screen, and absorber.
The trusted runtime computes intersections, vector reflection, Snell refraction/TIR, and the
paraxial thin-lens direction change. Coordinates and ray directions must match the actual image."""
    return ""


def storyboard_prompt(problem: K12Problem) -> str:
    payload = json.dumps(problem.model_payload(), ensure_ascii=False, indent=2)
    return f"""You are designing a grade-appropriate K-12 explanation.
Prompt protocol: {PROMPT_VERSION}

Use only the problem and image evidence below. Do not assume access to a gold answer.
First solve the problem. Then decide whether executable motion/state change materially aids
understanding. This benchmark has already been curated for dynamic suitability, so a refusal
must include a concrete pedagogical reason.

Return one JSON object only:
{{
  "analysis": "short solution analysis",
  "requires_simulation": true,
  "final_answer": "candidate answer",
  "blocks": [
    {{"block_id":"TEXT_1","kind":"text","content":"...","learning_goal":"..."}},
    {{"block_id":"SIM_1","kind":"sim","content":"observable state and change","highlights":["object_id_hint"]}}
  ]
}}
Blocks must alternate text/sim, use grade-appropriate language, and include only simulations
that contribute to a reasoning step. Do not include implementation code.

Problem input:
{payload}
"""


def world_spec_prompt(
    problem: K12Problem, analysis: str, blocks: List[StoryBlock], route: RouteDecision
) -> str:
    block_payload = [block.__dict__ for block in blocks]
    return f"""Convert a solved K-12 problem and storyboard into one canonical EduWorldSpec.
Prompt protocol: {PROMPT_VERSION}
Selected engine: {route.engine}; route reason: {route.reason}.

The same object ids, coordinate system, parameters, units, camera, colors, and initial state
will anchor every scene. Physical quantities must use explicit units. Do not invent exact
numeric values when the problem only establishes an ordering; encode a justified normalized
value and state that convention in a constraint.

Return one JSON object only with keys:
schema_version, problem_id, coordinate_system, objects, parameters, constraints,
initial_state, expected_events, final_state, learning_goals, visual_conventions.
Each object needs a unique id. Each expected event needs id, type, participants, condition,
and storyboard_step. Event participants must be declared object ids.

Problem (gold answer omitted):
{json.dumps(problem.model_payload(), ensure_ascii=False, indent=2)}

Candidate analysis:
{analysis}

Storyboard:
{json.dumps(block_payload, ensure_ascii=False, indent=2)}
"""


def program_prompt(
    problem: K12Problem,
    blocks: List[StoryBlock],
    spec: EduWorldSpec,
    route: RouteDecision,
) -> str:
    scene_ids = [block.block_id for block in blocks if block.kind == "sim"]
    domain_contract = _domain_program_contract(route)
    if domain_contract:
        return f"""Generate declarative, deterministically executable teaching scenes from one immutable EduWorldSpec.
Prompt protocol: {PROMPT_VERSION}
Engine: {route.engine}; physics family: {route.simulation_type}.

Return one JSON object only:
{{
  "engine":"{route.engine}",
  "render_spec":{{"engine":"{route.engine}","fps":30,"duration":8,"width":512,"height":512,"checkpoints":[]}},
  "world_spec_sha256":"{spec.canonical_hash()}",
  "scenes":[{{"scene_id":"SIM_1","simulation_spec":{{...}}}}]
}}

Required scene ids: {json.dumps(scene_ids)}. Never include document, HTML, JavaScript, trace, or
solver output: the trusted compiler creates them. Use the same WorldSpec object ids across
scenes. A later scene may change only a parameter or action requested by the storyboard.

{domain_contract}

Problem:
{json.dumps(problem.model_payload(), ensure_ascii=False, indent=2)}
Storyboard:
{json.dumps([block.__dict__ for block in blocks], ensure_ascii=False, indent=2)}
EduWorldSpec:
{json.dumps(spec.to_dict(), ensure_ascii=False, indent=2)}
"""
    contract = "HTML with one canvas and local/global THREE and CANNON" if route.engine == "threejs-cannon" else (
        "HTML/JavaScript with a deterministic canvas animation" if route.engine == "p5js" else "a complete Python Manim class named GeneratedScene"
    )
    return f"""Generate deterministic executable teaching scenes from one immutable EduWorldSpec.
Prompt protocol: {PROMPT_VERSION}
Engine: {route.engine}. Each document must be {contract}.

Return one JSON object only:
{{
  "engine": "{route.engine}",
  "render_spec": {{"engine":"{route.engine}","fps":30,"duration":8,"width":512,"height":512,"checkpoints":[]}},
  "world_spec_sha256": "{spec.canonical_hash()}",
  "scenes": [{{"scene_id":"SIM_1","document":"complete executable source"}}]
}}

Required scene ids: {json.dumps(scene_ids)}.
Every scene must declare/reference the same object ids and numerical values from EduWorldSpec.
Later scenes may advance time, highlight objects, or vary a parameter explicitly requested by
the storyboard, but must not recreate a different world. Use a fixed time step and no network,
file, eval, subprocess, randomness, or external asset access. Show units and educational labels.
For browser scenes, Three.js/Cannon.js and recording helpers are injected locally by the
renderer. P5.js routes must use the standard Canvas API (Cannon.js is available) so execution
never depends on a CDN. Do not use script src URLs.

Problem:
{json.dumps(problem.model_payload(), ensure_ascii=False, indent=2)}
Storyboard:
{json.dumps([block.__dict__ for block in blocks], ensure_ascii=False, indent=2)}
EduWorldSpec:
{json.dumps(spec.to_dict(), ensure_ascii=False, indent=2)}
"""


def repair_prompt(original_prompt: str, invalid_output: str, errors: List[str]) -> str:
    return f"""{original_prompt}

Your previous JSON failed validation. Make exactly one repair attempt and return the complete
replacement JSON object, not a patch.
Validation errors:
{json.dumps(errors, ensure_ascii=False, indent=2)}
Previous output:
{invalid_output[:120000]}
"""
