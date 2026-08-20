"""Versioned prompts for the three-stage K12SimWorld generation protocol."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping

from .models import EduWorldSpec, K12Problem, StoryBlock
from .routing import RouteDecision


PROMPT_VERSION = "k12simworld-v2.0-path-constraints"


def _domain_program_contract(route: RouteDecision) -> str:
    if route.engine == "mechanics-2d":
        return """Do not write HTML, JavaScript, or an integrator. Return one trusted declarative 2-D mechanics spec per scene:
{
  "domain_model":"mechanics_2d", "duration":8, "dt":0.0083333333,
  "playback_duration":8, "gravity":[0,-9.8],
  "potential_reference":[0,0],
  "bounds":{"x_min":-5,"x_max":5,"y_min":-2,"y_max":5},
  "units":{"length":"m","time":"s","mass":"kg"},
  "bodies":[
    {"id":"world_object_id","shape":"circle","motion_type":"dynamic","mass":1,
     "radius":0.2,"collision_radius":0,"position":[0,2],"velocity":[2,0],"restitution":0.1,
     "friction":0.2,"linear_damping":0,"label":"ball","color":"#2563eb"}
  ],
  "static_geometry":[
    {"id":"declared_world_object_id","type":"segment","p1":[-5,0],"p2":[5,0],
     "normal":[0,1],"friction":0.3,"restitution":0.1,"label":"ground"}
  ],
  "springs":[
    {"id":"spring_link","anchor_a":[-2,1],"body_b":"world_object_id",
     "stiffness":20,"damping":0.5,"rest_length":1}
  ],
  "distance_constraints":[
    {"id":"rope_link","anchor_a":[0,3],"body_b":"world_object_id","length":2}
  ],
  "forces":[
    {"id":"push","target":"world_object_id","force":[2,0],"start_time":0,"end_time":1}
  ],
  "actions":[
    {"trigger":{"type":"position_crossing","body":"world_object_id","axis":"x",
                "value":0,"direction":"positive","after_time":0.1},
     "type":"remove_distance_constraint","target":"rope_link",
     "event_id":"rope_break","event_type":"string_break",
     "participants":["world_object_id","declared_rope_object_id"],
     "label":"绳子在事件点断开"},
    {"trigger":{"type":"position_crossing","body":"world_object_id","axis":"x",
                "value":0,"direction":"positive","after_time":0.1},
     "type":"emit_event","target":"world_object_id","event_id":"bottom_reached",
     "event_type":"bottom_reached","participants":["world_object_id"],"label":"到达底端"}
  ],
  "phases":[
    {"id":"constrained_motion","start_time":0,"end_time":2,
     "label":"阶段 1：受约束运动","description":"显示完整物理世界和约束"},
    {"id":"free_motion","start_time":2,"end_time":8,
     "label":"阶段 2：约束解除后的运动","description":"保留断开瞬间的速度"}
  ],
  "visual_strategy":"continuous_process",
  "annotations":[
    {"type":"trail","target":"world_object_id"},
    {"type":"velocity_arrow","target":"world_object_id"},
    {"type":"force_arrow","target":"world_object_id"}
  ],
  "visual_instances":[]
}
Supported body shapes: circle, box, rod. Supported motion types: dynamic, static, kinematic.
Static geometry currently supports finite non-zero-length line segments. Segment normal must point
toward the allowed body side. Never encode a pivot/point as p1==p2 static geometry; use a
distance-constraint anchor_a/anchor_b, which the trusted renderer draws as the pivot. Springs and
distance constraints connect body_a/body_b or anchor_a/anchor_b.
For a body constrained to a finite track, use path_constraints instead of inventing a zero-length
distance constraint. A path constraint owns exactly one body and is one of:
{"id":"track","body":"world_object_id","type":"polyline","points":[[0,2],[1,1],[2,1]],
 "auto_release":"end","release_event_id":"leave_track","release_event_type":"path_release"}
{"id":"track","body":"world_object_id","type":"bezier","points":[[0,2],[1,0],[2,1]],
 "auto_release":"end"}
{"id":"track","body":"world_object_id","type":"circular_arc","center":[0,0],"radius":2,
 "start_angle":3.141592653589793,"end_angle":0,"auto_release":"end"}
Angles are radians; decreasing angles traverse clockwise. The solver projects the initial state to
the path, keeps velocity tangent, integrates tangential acceleration, and automatically releases at
the configured start/end/either endpoint while emitting a snapshot event. Do not put the same body
in both distance_constraints and path_constraints.
Supported body actions: impulse, set_position, set_velocity, set_angular_velocity. Supported
constraint actions: remove_distance_constraint, restore_distance_constraint,
remove_path_constraint, and restore_path_constraint. emit_event records a real teaching event
without modifying state; target is the observed body, and event_id/event_type describe what
happened. Never fabricate a distance constraint merely to report arrival at the bottom or departure
from a track. An action uses
exactly one scheduling form: a TOP-LEVEL numeric time field such as {"time":0.5,...}, or
trigger={type:position_crossing, body, axis:x|y, value,
direction:positive|negative|either, after_time}, or trigger={type:speed_below, body, threshold,
after_time}. Never emit trigger.type=time/at_time/time_reached; timed actions use the top-level
time field. Constraint actions target the matching constraint id and omit value. Use event_id,
event_type, participants (canonical WorldSpec object ids), and label to expose the physical
intervention in the trace, candidate-event validation, and video.
The trusted runtime owns gravity,
forces, fixed-step integration, localized contacts, collisions, spring forces, distance projection,
trace generation, Canvas drawing, and recording. Use one consistent SI or explicitly normalized
scale justified by WorldSpec. Every body and static_geometry id must exist in WorldSpec. Link and
force ids are auxiliary and need not be WorldSpec objects. Every body frame exposes position,
velocity, speed, acceleration, kinetic_energy, gravitational_potential_energy, potential_energy,
and mechanical_energy. The frame-level energies object exposes kinetic, gravitational_potential,
elastic_potential, potential_total, and mechanical_total. potential_reference defines zero
gravitational potential. Every annotations[].target MUST be an
exact, character-for-character reference to one existing bodies[].id; never use a label, phrase,
comma-separated list, dotted path, static_geometry id, or invented id as an annotation target.
Annotations are optional, so omit any annotation whose target is not a body. Encode only physical
relations justified by the problem; never invent a third dimension or decorative depth.

For a WorldSpec particle, point, or point_mass, radius is visual size only and the canonical
position is the mathematical point used by equations. Set collision_radius=0 so a ground-contact
event occurs when the canonical y coordinate reaches the ground, not one drawing radius earlier.
For a finite-sized sphere, omit collision_radius (or set it equal to radius). The compiler also
enforces this distinction from the WorldSpec object type. When the immutable WorldSpec declares a
two-participant contact/collision terminal_event, the compiler binds it as terminal_contact and the
solver stops at the localized pre-impulse contact snapshot. Never shorten the starting height or
alter gravity, velocity, mass, radius, or duration merely to force candidate targets to pass.

Default to visual_strategy="continuous_process" and visual_instances=[] so the viewer sees one
complete physical world: supports, constraints, bodies, trajectories, interactions, and phase
changes together. Use phases to label meaningful consecutive parts of the same causal trajectory.
Do not turn ordinary 2-D motion into horizontal/vertical projection panels. Only when the problem
explicitly asks learners to decompose vector components may you set
visual_strategy="component_decomposition" and declare visual_instances. Each visual instance must
reference an existing bodies[].id and is render-only: never target it with forces, actions,
annotations, events, invariants, or target_observables. Mutually exclusive counterfactuals cannot
share one state trace; implement them as separate storyboard scenes that start from the same
canonical initial state and differ only by the requested intervention."""
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
  "solution": {{
    "analysis": "derivation used to obtain the candidate answer",
    "givens": [{{"id":"safe_id","value":1,"unit":"...","source":"problem"}}],
    "derived_values": [{{"id":"safe_id","expression":"...","value":1,"unit":"..."}}],
    "assumptions": ["only assumptions justified by the problem"],
    "final_answer": "exactly the same candidate answer as the top-level final_answer",
    "confidence": 0.9
  }},
  "visualization_decision": {{
    "mode": "schematic_2d",
    "criterion": "",
    "evidence_quote": "",
    "reason": "why 2-D is sufficient"
  }},
  "blocks": [
    {{"block_id":"TEXT_1","kind":"text","content":"...","learning_goal":"..."}},
    {{"block_id":"SIM_1","kind":"sim","content":"observable state and change","highlights":["object_id_hint"]}}
  ]
}}
Order blocks by the teaching argument; adjacent text or simulation blocks are allowed. Every SIM
block must describe one physically coherent trajectory from its initial condition through the key
interaction to the observable outcome, not a static diagram or a coordinate-component projection.
When the explanation compares mutually exclusive interventions or outcomes, create separate SIM
blocks (for example SIM_1 for breaking a string at A and SIM_2 for breaking it at C). Each such SIM
must show the complete physical world and full causal process. Use grade-appropriate language and
do not include implementation code.

Visualization policy: default to mode=schematic_2d. You may request mode=spatial_3d only when
physical understanding genuinely depends on depth. Then criterion must be exactly one of
non_coplanar_motion, depth_dependent_collision, spatial_rotation_axis,
perspective_geometry_required, occlusion_is_physics, or multi_view_spatial_structure; and
evidence_quote must be an exact quotation from the visible problem or image caption that contains
the spatial cue. "3-D looks better", ordinary perspective artwork, and a generic 3-D object are
not evidence. The runtime independently verifies the quote and criterion and otherwise forces 2-D.

Problem input:
{payload}
"""


def world_spec_prompt(
    problem: K12Problem,
    candidate_solution: Mapping[str, Any],
    blocks: List[StoryBlock],
    route: RouteDecision,
) -> str:
    block_payload = [block.__dict__ for block in blocks]
    return f"""Convert a solved K-12 problem and storyboard into one canonical EduWorldSpec.
Prompt protocol: {PROMPT_VERSION}
Selected engine: {route.engine}; route reason: {route.reason}.

The same object ids, coordinate system, parameters, units, camera, colors, and initial state
will anchor every scene. Physical quantities must use explicit units. Do not invent exact
numeric values when the problem only establishes an ordering; encode a justified normalized
value and state that convention in a constraint.

Return one JSON object only using this exact type structure:
{{
  "schema_version": "1.0",
  "problem_id": "{problem.problem_id}",
  "coordinate_system": {{"axes": "...", "origin": "...", "units": {{"length": "m", "time": "s"}}}},
  "objects": [{{"id": "safe_identifier", "type": "...", "label": "...", "properties": {{}}}}],
  "parameters": [{{"id": "safe_identifier", "value": 1, "unit": "...", "justification": "..."}}],
  "constraints": [{{"type": "...", "description": "..."}}],
  "initial_state": {{"time": 0, "objects": {{}}}},
  "expected_events": [{{"id": "safe_identifier", "type": "...", "participants": ["declared_object_id"], "condition": "...", "storyboard_step": "SIM_1"}}],
  "final_state": {{"objects": {{}}}},
  "learning_goals": ["..."],
  "visual_conventions": {{"colors": {{}}, "labels": true}},
  "terminal_event": {{"id":"safe_identifier","type":"contact","participants":["declared_object_id"]}},
  "target_observables": [
    {{"id":"safe_identifier","scene_id":"SIM_1","at":"final",
      "path":"objects.object_id.speed","expected":1.0,"operator":"approximately_equal",
      "unit":"m/s","absolute_tolerance":0.01,"relative_tolerance":0.02,"required":true}}
  ],
  "invariants": [
    {{"id":"mechanical_energy_constant","scene_id":"SIM_1",
      "expression":"ke + pe",
      "bindings":{{"ke":"objects.object_id.kinetic_energy",
                  "pe":"objects.object_id.potential_energy"}},
      "display_formula":"E_k + E_p = constant","result_unit":"J",
      "type":"constant","tolerance":0.02,"required":true}}
  ]
}}
coordinate_system, initial_state, final_state, terminal_event, and visual_conventions must be
JSON objects; objects, parameters, constraints, expected_events, target_observables, invariants,
and learning_goals must be JSON arrays.
Do not attach one ambiguous unit to an object with differently dimensioned properties. Use
property-specific units such as mass_unit="kg" and radius_unit="m". Units for the same named
quantity must agree everywhere in the WorldSpec.
Each object needs a unique id. Each expected event needs id, type, participants, condition,
and storyboard_step. Event participants must be declared object ids. Mark a decisive event with
required_for_validation=true when the selected trusted solver can emit it. For mechanics scenes,
the program must copy that expected event id/type/participants into the causal action so the event
is auditable rather than merely narrated.

When the Storyboard contains mutually exclusive SIM scenes, do not merge their outcomes into one
global final_state or terminal_event. Keep only genuinely shared facts there; attach each numerical
outcome to a target_observable with its own scene_id. A target at one scene's C point must never be
placed at=final for a different A-point break scene. Use one terminal_event only when it is common
to the relevant scenes and can be emitted by the selected solver.

Problem (gold answer omitted):
{json.dumps(problem.model_payload(), ensure_ascii=False, indent=2)}
Treat the CandidateSolution below as immutable for this run: do not recompute or silently change
its givens, derived values, assumptions, or final answer. Convert its numerical conclusions into
machine-checkable target_observables. A trusted domain engine requires at least one required target.
Use paths relative to a solver frame, for example objects.ball.position.0, objects.ball.speed,
objects.switch.closed, or t. Use trace.summary... only for values stored in the solver summary.
path is strictly a dotted lookup path, never an arithmetic expression. Prefer a trusted standard
observable whenever one exists. For mechanics-2d these include objects.ball.kinetic_energy,
objects.ball.potential_energy, objects.ball.mechanical_energy, energies.mechanical_total, and
objects.ball.acceleration.1.

Only when no standard observable exists, define a derived scalar with expression plus bindings.
expression may contain numbers, bound variable names, +, -, *, /, **, %, comparisons, boolean
operators, a conditional expression, pi/e, and abs, min, max, sqrt, sin, cos, tan, exp, log,
floor, or ceil. Use ** for powers, never ^. Each bindings key must be a short ASCII identifier;
each value must be one dotted lookup path resolving to a finite numeric scalar. Never place dotted
paths directly inside expression and never use attributes, indexing, imports, lambdas, or arbitrary
function calls. Example: expression="ke + pe" with
bindings={{"ke":"objects.ball.kinetic_energy","pe":"objects.ball.potential_energy"}}.
result_unit records the derived result unit. Addition/subtraction operands must have compatible
units; multiplication/division/powers must yield result_unit. display_formula is optional
human-readable teaching notation only and is never executed. Put the already computed candidate
value in expected.
Use at=initial, final, or terminal_event. A terminal event must be observable by the selected
domain solver. Use explicit absolute and relative tolerances justified by numerical precision. For fixed-step
mechanics targets, do not demand micro-unit agreement: normally use absolute_tolerance at least
1e-4 for time/position and 1e-3 for speed/energy, plus a small relative_tolerance, unless the
observable is produced by an exact algebraic solver. Tolerances represent numerical resolution,
not permission to change the physical answer.
Invariants may use only constant, nondecreasing, or nonincreasing and must point to a scalar trace
quantity or evaluate a scalar expression. Never alter a problem-given parameter merely to force a
target match.

CandidateSolution (model-derived, not gold):
{json.dumps(dict(candidate_solution), ensure_ascii=False, indent=2)}



Storyboard:
{json.dumps(block_payload, ensure_ascii=False, indent=2)}
"""


def program_prompt(
    problem: K12Problem,
    blocks: List[StoryBlock],
    spec: EduWorldSpec,
    route: RouteDecision,
    simulation_contract: Mapping[str, Any],
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

Process fidelity rules:
- Each scene is a complete causal trajectory, not a collection of disconnected diagrams.
- Start from the canonical initial condition, execute the physical interaction, and continue until
  the outcome needed by that SIM block is visibly established.
- Use visual_strategy=continuous_process and an empty visual_instances list by default.
- Use phases for consecutive stages such as constrained motion, release/collision, and free motion.
- Use remove_distance_constraint at the real break/release time or event; never leave a rope active
  while narrating free flight.
- Preserve instantaneous position and velocity across an intervention unless the problem supplies
  an impulse. Do not fake a transition with set_position or set_velocity.
- Put mutually exclusive interventions in different scenes, never in one impossible state trace.

{domain_contract}

The SimulationContract is immutable. Choose only equations, declared unknown parameters, event
conditions, and duration needed for the trusted forward solver to satisfy it. Never interpolate
states to the target and never modify a problem-given value to make the endpoint fit.

Problem:
{json.dumps(problem.model_payload(), ensure_ascii=False, indent=2)}
Storyboard:
{json.dumps([block.__dict__ for block in blocks], ensure_ascii=False, indent=2)}
EduWorldSpec:
{json.dumps(spec.to_dict(), ensure_ascii=False, indent=2)}
SimulationContract:
{json.dumps(dict(simulation_contract), ensure_ascii=False, indent=2)}
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
Every core physical EduWorldSpec object id must appear literally in every scene source. Auxiliary
points, arrows, curves, labels, and bars may appear only in relevant scenes, but every object id
must appear in at least one scene. Foreground objects must contrast with the background. All
Canvas text coordinates must lie inside their backing canvas.
Never assign arrays directly to Three.js position, rotation, scale, or quaternion properties;
preserve those objects and update them with .set(...) or .copy(...).
Later scenes may advance time, highlight objects, or vary a parameter explicitly requested by
the storyboard, but must not recreate a different world. Use a fixed time step and no network,
file, eval, subprocess, randomness, or external asset access. Show units and educational labels.
For browser scenes, Three.js/Cannon.js and recording helpers are injected locally by the
renderer. P5.js routes must use the standard Canvas API (Cannon.js is available) so execution
never depends on a CDN. Do not use script src URLs.

Problem:
{json.dumps(problem.model_payload(), ensure_ascii=False, indent=2)}
The SimulationContract is immutable. The animation must evolve from initial_state according to
explicit physics; do not linearly interpolate to the target or change problem-given values merely
to make the final frame fit.

Storyboard:
{json.dumps([block.__dict__ for block in blocks], ensure_ascii=False, indent=2)}
EduWorldSpec:
{json.dumps(spec.to_dict(), ensure_ascii=False, indent=2)}
SimulationContract:
{json.dumps(dict(simulation_contract), ensure_ascii=False, indent=2)}
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


def world_spec_repair_prompt(
    original_prompt: str,
    invalid_output: str,
    error: str,
    normalization_changes: List[str],
) -> str:
    return f"""{original_prompt}

Your previous EduWorldSpec still failed its strict contract after safe deterministic formula
normalization. Make exactly one targeted repair and return the complete replacement EduWorldSpec
JSON object, not a patch. Preserve the CandidateSolution, object identities, problem givens,
units, target expected values, tolerances, selected engine, and storyboard scene ids.

Formula repair rules:
- path is one dotted scalar lookup only; it never contains arithmetic.
- Composite quantities use expression plus bindings.
- Every bindings value is one dotted scalar lookup path, never a number or formula.
- Dotted paths never appear directly inside expression. Bind them to short ASCII aliases first.
- Every name used by expression is either a binding alias, pi/e, or an allowed function.
- Symbolic thresholds that are not trace values are not invariants; express them as a numerical
  candidate target or a parameter-sweep teaching scene without inventing a trace variable.

Deterministic normalization already attempted:
{json.dumps(normalization_changes, ensure_ascii=False, indent=2)}
Remaining contract error:
{error}
Previous output:
{invalid_output[:120000]}
"""


def execution_repair_prompt(
    original_prompt: str,
    invalid_output: str,
    execution_feedback: str,
) -> str:
    return f"""{original_prompt}

The previous program passed static JSON validation but failed during a real execution/render
attempt. Make exactly one execution-aware repair and return the complete replacement JSON object,
not a patch. Keep the selected engine, immutable EduWorldSpec hash, object ids, scene ids, problem
givens, and physical conclusions unchanged. Fix only code/runtime behavior that is supported by
the concrete exception and renderer log below. Do not hide the failure with a blank scene,
hard-coded final frame, removed object, or weakened target.

Real execution/render feedback:
{execution_feedback[:30000]}
Previous program:
{invalid_output[:120000]}
"""


def target_repair_prompt(
    original_prompt: str,
    invalid_output: str,
    validation_report: Mapping[str, Any],
) -> str:
    return f"""{original_prompt}

Your previous declarative simulation executed, but its trusted trace violated the immutable
candidate-derived SimulationContract. Return one complete replacement JSON object, not a patch.
Do not change problem givens, CandidateSolution, target values, or tolerances. Correct only the
simulation mapping, equations, event condition, allowed unknown parameter, or duration. Repair the
causal process, not just the endpoint: if an expected break/release/collision is absent, add the
supported timed or event-triggered action and preserve the state through that event. Keep the
complete world visible with visual_strategy=continuous_process unless the storyboard explicitly
requires vector-component decomposition.
Target validation report:
{json.dumps(dict(validation_report), ensure_ascii=False, indent=2)}
Previous output:
{invalid_output[:120000]}
"""
