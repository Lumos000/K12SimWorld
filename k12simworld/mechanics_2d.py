"""Trusted declarative two-dimensional mechanics solver.

The model supplies data, never executable code. This module owns integration,
contacts, springs, constraints, events, and trace output. It deliberately
implements an auditable K-12 subset rather than a general rigid-body engine.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .domain_solvers import (
    DomainSimulationError, _bounded_number, _finite_number, _round, _safe_id, _vec2,
)


def _radius(body: Mapping[str, Any], normal: Tuple[float, float] | None = None) -> float:
    """Return collision extent; drawing continues to use radius/size."""
    if body["shape"] == "circle":
        return float(body.get("collision_radius", body["radius"]))
    width, height = body["size"]
    if normal is None:
        return math.hypot(width, height) / 2
    angle = float(body["angle"])
    c, s = math.cos(angle), math.sin(angle)
    local_x = normal[0] * c + normal[1] * s
    local_y = -normal[0] * s + normal[1] * c
    return abs(local_x) * width / 2 + abs(local_y) * height / 2


def _endpoint(link: Mapping[str, Any], bodies: Mapping[str, Mapping[str, Any]], suffix: str):
    body_key, anchor_key = f"body_{suffix}", f"anchor_{suffix}"
    if link.get(body_key):
        body_id = str(link[body_key])
        if body_id not in bodies:
            raise DomainSimulationError(f"{link.get('id')} references unknown body {body_id!r}")
        return tuple(bodies[body_id]["position"]), body_id
    if link.get(anchor_key) is not None:
        return _vec2(link[anchor_key], f"{link.get('id')}.{anchor_key}"), None
    raise DomainSimulationError(f"{link.get('id')} requires {body_key} or {anchor_key}")


def _normalise_body(raw: Mapping[str, Any], index: int) -> Dict[str, Any]:
    body_id = _safe_id(raw.get("id"), f"bodies[{index}].id")
    shape = str(raw.get("shape") or "circle").lower()
    if shape not in {"circle", "box", "rod"}:
        raise DomainSimulationError(f"unsupported mechanics body shape {shape!r}")
    motion = str(raw.get("motion_type") or "dynamic").lower()
    if motion not in {"dynamic", "static", "kinematic"}:
        raise DomainSimulationError(f"unsupported motion_type {motion!r}")
    mass = _bounded_number(raw.get("mass", 1), f"{body_id}.mass", 1e-12, 1e12, inclusive_minimum=False)
    body: Dict[str, Any] = {
        "id": body_id, "shape": shape, "motion_type": motion, "mass": mass,
        "_inv_mass": 1 / mass if motion == "dynamic" else 0,
        "position": list(_vec2(raw.get("position", [0, 0]), f"{body_id}.position")),
        "velocity": list(_vec2(raw.get("velocity", [0, 0]), f"{body_id}.velocity")),
        "angle": _finite_number(raw.get("angle", 0), f"{body_id}.angle"),
        "angular_velocity": _finite_number(raw.get("angular_velocity", 0), f"{body_id}.angular_velocity"),
        "restitution": _bounded_number(raw.get("restitution", .15), f"{body_id}.restitution", 0, 1),
        "friction": _bounded_number(raw.get("friction", .25), f"{body_id}.friction", 0, 2),
        "linear_damping": _bounded_number(raw.get("linear_damping", 0), f"{body_id}.linear_damping", 0, 100),
        "label": str(raw.get("label") or body_id), "color": str(raw.get("color") or "#2563eb"),
    }
    if shape == "circle":
        body["radius"] = _bounded_number(raw.get("radius", .25), f"{body_id}.radius", 1e-9, 1e9, inclusive_minimum=False)
        body["collision_radius"] = _bounded_number(
            raw.get("collision_radius", body["radius"]),
            f"{body_id}.collision_radius",
            0,
            body["radius"],
        )
    else:
        size = _vec2(raw.get("size", [1, .3]), f"{body_id}.size")
        if min(size) <= 0:
            raise DomainSimulationError(f"{body_id}.size values must be positive")
        body["size"] = list(size)
    return body


def _normalise_links(spec: Mapping[str, Any], name: str, bodies: Mapping[str, Mapping[str, Any]]):
    raw_values = spec.get(name) or []
    if not isinstance(raw_values, list) or len(raw_values) > 32:
        raise DomainSimulationError(f"{name} must contain at most 32 entries")
    result, seen = [], set()
    for index, raw in enumerate(raw_values):
        if not isinstance(raw, Mapping):
            raise DomainSimulationError(f"{name}[{index}] must be an object")
        item = dict(raw)
        item["id"] = _safe_id(item.get("id"), f"{name}[{index}].id")
        if item["id"] in seen:
            raise DomainSimulationError(f"duplicate {name} id {item['id']!r}")
        seen.add(item["id"])
        _endpoint(item, bodies, "a"); _endpoint(item, bodies, "b")
        result.append(item)
    return result


def simulate_mechanics_2d(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Execute a bounded mechanics_2d declarative specification."""
    duration = _bounded_number(spec.get("duration", 8), "duration", .01, 30)
    requested_dt = _bounded_number(spec.get("dt", 1 / 120), "dt", 1e-5, .1)
    steps = int(math.ceil(duration / requested_dt))
    if steps > 6000:
        raise DomainSimulationError("mechanics-2d trace exceeds 6000 integration steps")
    dt = duration / steps
    playback = _bounded_number(spec.get("playback_duration", duration), "playback_duration", 1, 30)
    gravity = _vec2(spec.get("gravity", [0, -9.8]), "gravity")
    potential_reference = _vec2(
        spec.get("potential_reference", [0, 0]), "potential_reference"
    )
    raw_units = spec.get("units") or {"length": "m", "time": "s", "mass": "kg"}
    if not isinstance(raw_units, Mapping):
        raise DomainSimulationError("units must be an object")
    trace_units = dict(raw_units)
    if "energy" not in trace_units:
        trace_units["energy"] = (
            "J" if trace_units.get("mass") == "kg"
            and trace_units.get("length") == "m"
            and trace_units.get("time") == "s"
            else "mass*length^2/time^2"
        )
    raw_bounds = spec.get("bounds") or {"x_min": -5, "x_max": 5, "y_min": -3, "y_max": 5}
    if not isinstance(raw_bounds, Mapping):
        raise DomainSimulationError("bounds must be an object")
    bounds = {key: _finite_number(raw_bounds.get(key), f"bounds.{key}") for key in ("x_min", "x_max", "y_min", "y_max")}
    if bounds["x_min"] >= bounds["x_max"] or bounds["y_min"] >= bounds["y_max"]:
        raise DomainSimulationError("bounds must have positive width and height")

    raw_bodies = spec.get("bodies")
    if not isinstance(raw_bodies, list) or not raw_bodies or len(raw_bodies) > 64:
        raise DomainSimulationError("mechanics_2d requires 1 to 64 bodies")
    bodies: Dict[str, Dict[str, Any]] = {}
    for index, raw in enumerate(raw_bodies):
        if not isinstance(raw, Mapping):
            raise DomainSimulationError(f"bodies[{index}] must be an object")
        body = _normalise_body(raw, index)
        if body["id"] in bodies:
            raise DomainSimulationError(f"duplicate mechanics body {body['id']!r}")
        bodies[body["id"]] = body

    geometry, entity_ids = [], set(bodies)
    raw_geometry = spec.get("static_geometry") or []
    if not isinstance(raw_geometry, list) or len(raw_geometry) > 64:
        raise DomainSimulationError("static_geometry must contain at most 64 entries")
    for index, raw in enumerate(raw_geometry):
        if not isinstance(raw, Mapping):
            raise DomainSimulationError(f"static_geometry[{index}] must be an object")
        item_id = _safe_id(raw.get("id"), f"static_geometry[{index}].id")
        if item_id in entity_ids:
            raise DomainSimulationError(f"duplicate mechanics entity {item_id!r}")
        entity_ids.add(item_id)
        if str(raw.get("type") or "segment").lower() != "segment":
            raise DomainSimulationError("mechanics-2d static geometry supports segment only")
        p1, p2 = _vec2(raw.get("p1"), f"{item_id}.p1"), _vec2(raw.get("p2"), f"{item_id}.p2")
        if math.dist(p1, p2) <= 1e-12:
            raise DomainSimulationError(f"static segment {item_id!r} has zero length")
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        normal = _vec2(raw.get("normal", [-dy, dx]), f"{item_id}.normal")
        length = math.hypot(*normal)
        if length <= 1e-12:
            raise DomainSimulationError(f"{item_id}.normal must be non-zero")
        geometry.append({
            "id": item_id, "type": "segment", "p1": list(p1), "p2": list(p2),
            "normal": [normal[0] / length, normal[1] / length],
            "friction": _bounded_number(raw.get("friction", .3), f"{item_id}.friction", 0, 2),
            "restitution": _bounded_number(raw.get("restitution", .1), f"{item_id}.restitution", 0, 1),
            "label": str(raw.get("label") or item_id), "color": str(raw.get("color") or "#475569"),
        })

    terminal_contact: Tuple[str, str] | None = None
    raw_terminal_contact = spec.get("terminal_contact")
    if raw_terminal_contact is not None:
        if (
            not isinstance(raw_terminal_contact, Sequence)
            or isinstance(raw_terminal_contact, (str, bytes))
            or len(raw_terminal_contact) != 2
        ):
            raise DomainSimulationError("terminal_contact must contain exactly two entity ids")
        terminal_contact = tuple(
            _safe_id(item, f"terminal_contact[{index}]")
            for index, item in enumerate(raw_terminal_contact)
        )
        if not set(terminal_contact).issubset(entity_ids):
            raise DomainSimulationError("terminal_contact references unknown physical entities")

    springs = _normalise_links(spec, "springs", bodies)
    for spring in springs:
        pa, _ = _endpoint(spring, bodies, "a"); pb, _ = _endpoint(spring, bodies, "b")
        spring["stiffness"] = _bounded_number(spring.get("stiffness"), f"{spring['id']}.stiffness", 0, 1e7)
        spring["damping"] = _bounded_number(spring.get("damping", 0), f"{spring['id']}.damping", 0, 1e5)
        spring["rest_length"] = _bounded_number(spring.get("rest_length", math.dist(pa, pb)), f"{spring['id']}.rest_length", 0, 1e9)
        spring["color"] = str(spring.get("color") or "#7c3aed")
    constraints = _normalise_links(spec, "distance_constraints", bodies)
    for constraint in constraints:
        pa, _ = _endpoint(constraint, bodies, "a"); pb, _ = _endpoint(constraint, bodies, "b")
        constraint["length"] = _bounded_number(constraint.get("length", math.dist(pa, pb)), f"{constraint['id']}.length", 0, 1e9)
        constraint["color"] = str(constraint.get("color") or "#0f172a")

    forces = []
    raw_forces = spec.get("forces") or []
    if not isinstance(raw_forces, list) or len(raw_forces) > 64:
        raise DomainSimulationError("forces must contain at most 64 entries")
    for index, raw in enumerate(raw_forces):
        if not isinstance(raw, Mapping):
            raise DomainSimulationError(f"forces[{index}] must be an object")
        target = _safe_id(raw.get("target"), f"forces[{index}].target")
        if target not in bodies:
            raise DomainSimulationError(f"force references unknown body {target!r}")
        start = _bounded_number(raw.get("start_time", 0), f"forces[{index}].start_time", 0, duration)
        end = _bounded_number(raw.get("end_time", duration), f"forces[{index}].end_time", 0, duration)
        if end < start:
            raise DomainSimulationError("force end_time must not precede start_time")
        forces.append({"id": _safe_id(raw.get("id"), f"forces[{index}].id"), "target": target,
                       "force": list(_vec2(raw.get("force"), f"forces[{index}].force")),
                       "start_time": start, "end_time": end})

    actions = []
    raw_actions = spec.get("actions") or []
    if not isinstance(raw_actions, list) or len(raw_actions) > 64:
        raise DomainSimulationError("actions must contain at most 64 entries")
    for index, raw in enumerate(raw_actions):
        if not isinstance(raw, Mapping):
            raise DomainSimulationError(f"actions[{index}] must be an object")
        kind = str(raw.get("type") or "").lower()
        if kind not in {"impulse", "set_velocity", "set_angular_velocity"}:
            raise DomainSimulationError(f"unsupported mechanics action {kind!r}")
        target = _safe_id(raw.get("target"), f"actions[{index}].target")
        if target not in bodies:
            raise DomainSimulationError(f"action references unknown body {target!r}")
        value = (_finite_number(raw.get("value"), f"actions[{index}].value") if kind == "set_angular_velocity"
                 else list(_vec2(raw.get("value"), f"actions[{index}].value")))
        actions.append({"time": _bounded_number(raw.get("time"), f"actions[{index}].time", 0, duration),
                        "type": kind, "target": target, "value": value})
    actions.sort(key=lambda item: item["time"])

    # Annotations are optional rendering hints, not part of the physical model.
    # Models occasionally emit descriptive targets such as "ball trajectory"
    # instead of an exact bodies[].id. Dropping that cosmetic hint is safer than
    # aborting an otherwise valid simulation. Forces and actions remain strict
    # because silently changing either would alter the physics.
    annotations = []
    raw_annotations = spec.get("annotations") or []
    if not isinstance(raw_annotations, list) or len(raw_annotations) > 64:
        raise DomainSimulationError("annotations must contain at most 64 entries")
    ignored_annotations = []
    for index, raw in enumerate(raw_annotations):
        if not isinstance(raw, Mapping):
            raise DomainSimulationError(f"annotations[{index}] must be an object")
        kind = str(raw.get("type") or "").lower()
        target = str(raw.get("target") or "").strip()
        if (
            kind not in {"trail", "velocity_arrow", "force_arrow", "label"}
            or target not in bodies
        ):
            ignored_annotations.append({
                "index": index,
                "type": kind,
                "target": target,
                "reason": "annotation target must exactly reference an existing body id",
            })
            continue
        annotations.append({"type": kind, "target": target})

    # A visual instance is another view of one canonical physical body. It has
    # no mass, force, collision shape, integration state, or event identity.
    # This lets a teaching scene show horizontal/vertical projections without
    # inventing bodies such as ball_h and ball_v in the physical world.
    visual_instances = []
    raw_instances = spec.get("visual_instances") or []
    if not isinstance(raw_instances, list) or len(raw_instances) > 32:
        raise DomainSimulationError("visual_instances must contain at most 32 entries")
    visual_ids: set[str] = set()
    for index, raw in enumerate(raw_instances):
        if not isinstance(raw, Mapping):
            raise DomainSimulationError(f"visual_instances[{index}] must be an object")
        instance_id = _safe_id(raw.get("id"), f"visual_instances[{index}].id")
        if instance_id in entity_ids or instance_id in visual_ids:
            raise DomainSimulationError(
                f"duplicate physical/visual entity id {instance_id!r}"
            )
        source_id = _safe_id(
            raw.get("source_object_id"),
            f"visual_instances[{index}].source_object_id",
        )
        if source_id not in bodies:
            raise DomainSimulationError(
                f"visual instance {instance_id!r} references unknown body {source_id!r}"
            )
        view = str(raw.get("view") or "full").strip().lower()
        if view not in {"full", "horizontal_projection", "vertical_projection"}:
            raise DomainSimulationError(
                f"unsupported visual instance view {view!r}"
            )
        panel = str(raw.get("panel") or "main").strip().lower()
        if panel not in {"main", "left", "right", "top", "bottom"}:
            raise DomainSimulationError(
                f"unsupported visual instance panel {panel!r}"
            )
        show_trail = raw.get("show_trail", True)
        if not isinstance(show_trail, bool):
            raise DomainSimulationError(
                f"visual_instances[{index}].show_trail must be boolean"
            )
        visual_ids.add(instance_id)
        visual_instances.append({
            "id": instance_id, "source_object_id": source_id,
            "view": view, "panel": panel,
            "label": str(raw.get("label") or instance_id),
            "color": str(raw.get("color") or bodies[source_id]["color"]),
            "show_trail": show_trail,
        })

    frames: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    active_contacts: set[Tuple[str, str]] = set()
    action_index = 0

    def body_energies(
        body_id: str, position: Sequence[float], velocity: Sequence[float]
    ) -> Dict[str, float]:
        body = bodies[body_id]
        kinetic = 0.5 * body["mass"] * (
            velocity[0] * velocity[0] + velocity[1] * velocity[1]
        )
        gravitational = -body["mass"] * (
            gravity[0] * (position[0] - potential_reference[0])
            + gravity[1] * (position[1] - potential_reference[1])
        )
        return {
            "kinetic_energy": _round(kinetic),
            "gravitational_potential_energy": _round(gravitational),
            "potential_energy": _round(gravitational),
            "mechanical_energy": _round(kinetic + gravitational),
        }

    def frame_energies(objects: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
        kinetic = sum(
            float(objects[body_id]["kinetic_energy"])
            for body_id, body in bodies.items()
            if body["motion_type"] == "dynamic"
        )
        gravitational = sum(
            float(objects[body_id]["gravitational_potential_energy"])
            for body_id, body in bodies.items()
            if body["motion_type"] == "dynamic"
        )
        elastic = 0.0
        for spring in springs:
            point_a = (
                objects[str(spring["body_a"])]["position"]
                if spring.get("body_a") else spring["anchor_a"]
            )
            point_b = (
                objects[str(spring["body_b"])]["position"]
                if spring.get("body_b") else spring["anchor_b"]
            )
            extension = math.dist(point_a, point_b) - spring["rest_length"]
            elastic += 0.5 * spring["stiffness"] * extension * extension
        return {
            "kinetic": _round(kinetic),
            "gravitational_potential": _round(gravitational),
            "elastic_potential": _round(elastic),
            "potential_total": _round(gravitational + elastic),
            "mechanical_total": _round(kinetic + gravitational + elastic),
        }

    def frame_at(
        time_value: float, net: Mapping[str, Sequence[float]]
    ) -> Dict[str, Any]:
        objects = {}
        for body_id, body in bodies.items():
            vx, vy = body["velocity"]
            force = list(net.get(body_id, [0, 0]))
            acceleration = (
                [force[0] * body["_inv_mass"], force[1] * body["_inv_mass"]]
                if body["motion_type"] == "dynamic"
                else [0.0, 0.0]
            )
            speed = math.hypot(vx, vy)
            objects[body_id] = {
                "position": [_round(value) for value in body["position"]],
                "velocity": [_round(vx), _round(vy)],
                "speed": _round(speed),
                "acceleration": [_round(value) for value in acceleration],
                **body_energies(body_id, body["position"], body["velocity"]),
                "angle": _round(body["angle"]), "angular_velocity": _round(body["angular_velocity"]),
                "net_force": [_round(value) for value in force],
            }
        return {
            "t": _round(time_value), "objects": objects,
            "energies": frame_energies(objects),
        }

    def snapshot(time_value: float, net: Mapping[str, Sequence[float]]) -> None:
        frames.append(frame_at(time_value, net))

    def interpolated_contact_frame(
        time_value: float,
        body_id: str,
        elapsed: float,
        previous: Mapping[str, Mapping[str, Sequence[float]]],
        net: Mapping[str, Sequence[float]],
    ) -> Dict[str, Any]:
        """Reconstruct the pre-contact state using the integrator equation."""
        result = frame_at(time_value, net)
        state = result["objects"][body_id]
        prior = previous[body_id]
        body = bodies[body_id]
        acceleration = [
            net[body_id][axis] * body["_inv_mass"] for axis in range(2)
        ]
        position = [
            prior["position"][axis]
            + prior["velocity"][axis] * elapsed
            + 0.5 * acceleration[axis] * elapsed * elapsed
            for axis in range(2)
        ]
        velocity = [
            prior["velocity"][axis] + acceleration[axis] * elapsed
            for axis in range(2)
        ]
        damping = max(0, 1 - body["linear_damping"] * elapsed)
        velocity = [component * damping for component in velocity]
        speed = math.hypot(*velocity)
        state["position"] = [_round(value) for value in position]
        state["velocity"] = [_round(value) for value in velocity]
        state["speed"] = _round(speed)
        state.update(body_energies(body_id, position, velocity))
        result["energies"] = frame_energies(result["objects"])
        return result

    def net_forces(time_value: float) -> Dict[str, List[float]]:
        net = {
            body_id: (
                [gravity[0] * body["mass"], gravity[1] * body["mass"]]
                if body["motion_type"] == "dynamic"
                else [0.0, 0.0]
            )
            for body_id, body in bodies.items()
        }
        for force in forces:
            if force["start_time"] <= time_value <= force["end_time"]:
                net[force["target"]][0] += force["force"][0]
                net[force["target"]][1] += force["force"][1]
        for spring in springs:
            pa, ida = _endpoint(spring, bodies, "a")
            pb, idb = _endpoint(spring, bodies, "b")
            dx, dy = pb[0] - pa[0], pb[1] - pa[1]
            length = math.hypot(dx, dy)
            if length <= 1e-12:
                continue
            ux, uy = dx / length, dy / length
            va = bodies[ida]["velocity"] if ida else [0, 0]
            vb = bodies[idb]["velocity"] if idb else [0, 0]
            relative = (vb[0] - va[0]) * ux + (vb[1] - va[1]) * uy
            magnitude = spring["stiffness"] * (length - spring["rest_length"]) + spring["damping"] * relative
            fx, fy = magnitude * ux, magnitude * uy
            if ida: net[ida][0] += fx; net[ida][1] += fy
            if idb: net[idb][0] -= fx; net[idb][1] -= fy
        return net

    snapshot(0, net_forces(0))
    for step in range(1, steps + 1):
        current_time = (step - 1) * dt
        while action_index < len(actions) and actions[action_index]["time"] <= current_time + 1e-12:
            action = actions[action_index]; body = bodies[action["target"]]
            if action["type"] == "impulse" and body["_inv_mass"]:
                body["velocity"][0] += action["value"][0] * body["_inv_mass"]
                body["velocity"][1] += action["value"][1] * body["_inv_mass"]
            elif action["type"] == "set_velocity":
                body["velocity"] = list(action["value"])
            else:
                body["angular_velocity"] = float(action["value"])
            events.append({"type": action["type"], "t": _round(current_time), "participants": [body["id"]]})
            action_index += 1

        net = net_forces(current_time)

        previous_states = {
            body_id: {
                "position": list(body["position"]),
                "velocity": list(body["velocity"]),
            }
            for body_id, body in bodies.items()
        }
        for body_id, body in bodies.items():
            if body["motion_type"] == "dynamic":
                ax = net[body_id][0] * body["_inv_mass"]
                ay = net[body_id][1] * body["_inv_mass"]
                body["position"][0] += body["velocity"][0] * dt + 0.5 * ax * dt * dt
                body["position"][1] += body["velocity"][1] * dt + 0.5 * ay * dt * dt
                body["velocity"][0] += ax * dt
                body["velocity"][1] += ay * dt
                damping = max(0, 1 - body["linear_damping"] * dt)
                body["velocity"][0] *= damping; body["velocity"][1] *= damping
            elif body["motion_type"] == "kinematic":
                body["position"][0] += body["velocity"][0] * dt
                body["position"][1] += body["velocity"][1] * dt
            if body["motion_type"] in {"dynamic", "kinematic"}:
                body["angle"] += body["angular_velocity"] * dt

        # Preserve the unconstrained state at the instant of a newly detected
        # contact. Terminal-event targets such as impact speed and kinetic
        # energy refer to this pre-impulse state, not the rebound frame.
        pre_contact_frame = frame_at(step * dt, net)
        contacts: set[Tuple[str, str]] = set()
        contact_frames: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for body in bodies.values():
            if body["motion_type"] != "dynamic":
                continue
            for segment in geometry:
                x1, y1 = segment["p1"]; x2, y2 = segment["p2"]; dx, dy = x2 - x1, y2 - y1
                fraction = ((body["position"][0] - x1) * dx + (body["position"][1] - y1) * dy) / (dx * dx + dy * dy)
                if fraction < 0 or fraction > 1:
                    continue
                nx, ny = segment["normal"]
                signed = (body["position"][0] - (x1 + fraction * dx)) * nx + (body["position"][1] - (y1 + fraction * dy)) * ny
                previous_position = previous_states[body["id"]]["position"]
                previous_signed = (
                    (previous_position[0] - x1) * nx
                    + (previous_position[1] - y1) * ny
                )
                radius = _radius(body, (nx, ny))
                contact_tolerance = max(1e-12, radius * 1e-12)
                crossed = previous_signed > radius + contact_tolerance and signed <= radius + contact_tolerance
                normal_speed = abs(
                    body["velocity"][0] * nx + body["velocity"][1] * ny
                )
                penetration_window = max(
                    2 * radius, normal_speed * dt + math.hypot(*gravity) * dt * dt
                )
                if (
                    signed > radius + contact_tolerance
                    or (not crossed and signed < radius - penetration_window - 1e-12)
                ):
                    continue
                contact = (body["id"], segment["id"])
                if crossed and previous_signed - signed > 1e-15:
                    prior_velocity = previous_states[body["id"]]["velocity"]
                    normal_velocity = prior_velocity[0] * nx + prior_velocity[1] * ny
                    normal_acceleration = (
                        net[body["id"]][0] * body["_inv_mass"] * nx
                        + net[body["id"]][1] * body["_inv_mass"] * ny
                    )
                    constant = previous_signed - radius
                    roots = []
                    if abs(normal_acceleration) <= 1e-15:
                        if abs(normal_velocity) > 1e-15:
                            roots = [-constant / normal_velocity]
                    else:
                        discriminant = (
                            normal_velocity * normal_velocity
                            - 2 * normal_acceleration * constant
                        )
                        if discriminant >= -1e-12:
                            square_root = math.sqrt(max(0.0, discriminant))
                            roots = [
                                (-normal_velocity - square_root) / normal_acceleration,
                                (-normal_velocity + square_root) / normal_acceleration,
                            ]
                    valid_roots = [value for value in roots if -1e-12 <= value <= dt + 1e-12]
                    if valid_roots:
                        elapsed = max(0.0, min(dt, min(valid_roots)))
                    else:
                        elapsed = dt * max(
                            0.0,
                            min(1.0, constant / (previous_signed - signed)),
                        )
                    contact_time = current_time + elapsed
                    contact_frames[contact] = interpolated_contact_frame(
                        contact_time, body["id"], elapsed, previous_states, net
                    )
                else:
                    contact_frames[contact] = pre_contact_frame
                penetration = radius - signed
                body["position"][0] += nx * penetration; body["position"][1] += ny * penetration
                vn = body["velocity"][0] * nx + body["velocity"][1] * ny
                if vn < 0:
                    bounce = 1 + min(body["restitution"], segment["restitution"])
                    body["velocity"][0] -= bounce * vn * nx; body["velocity"][1] -= bounce * vn * ny
                    tx, ty = -ny, nx; vt = body["velocity"][0] * tx + body["velocity"][1] * ty
                    removed = vt * min(1, min(body["friction"], segment["friction"]) * 20 * dt)
                    body["velocity"][0] -= removed * tx; body["velocity"][1] -= removed * ty
                contacts.add(contact)

        all_bodies = list(bodies.values())
        for index, left in enumerate(all_bodies):
            for right in all_bodies[index + 1:]:
                if left["_inv_mass"] + right["_inv_mass"] <= 0:
                    continue
                dx, dy = right["position"][0] - left["position"][0], right["position"][1] - left["position"][1]
                distance, minimum = math.hypot(dx, dy), _radius(left) + _radius(right)
                if distance <= 1e-12 or distance >= minimum:
                    continue
                nx, ny = dx / distance, dy / distance; inv_sum = left["_inv_mass"] + right["_inv_mass"]
                correction = (minimum - distance) / inv_sum
                left["position"][0] -= nx * correction * left["_inv_mass"]; left["position"][1] -= ny * correction * left["_inv_mass"]
                right["position"][0] += nx * correction * right["_inv_mass"]; right["position"][1] += ny * correction * right["_inv_mass"]
                relative = (right["velocity"][0] - left["velocity"][0]) * nx + (right["velocity"][1] - left["velocity"][1]) * ny
                if relative < 0:
                    impulse = -(1 + min(left["restitution"], right["restitution"])) * relative / inv_sum
                    left["velocity"][0] -= impulse * nx * left["_inv_mass"]; left["velocity"][1] -= impulse * ny * left["_inv_mass"]
                    right["velocity"][0] += impulse * nx * right["_inv_mass"]; right["velocity"][1] += impulse * ny * right["_inv_mass"]
                contacts.add(tuple(sorted((left["id"], right["id"]))))

        for _ in range(4):
            for constraint in constraints:
                pa, ida = _endpoint(constraint, bodies, "a"); pb, idb = _endpoint(constraint, bodies, "b")
                dx, dy = pb[0] - pa[0], pb[1] - pa[1]; distance = math.hypot(dx, dy)
                inv_a = bodies[ida]["_inv_mass"] if ida else 0; inv_b = bodies[idb]["_inv_mass"] if idb else 0
                if distance <= 1e-12 or inv_a + inv_b <= 0:
                    continue
                scale = (distance - constraint["length"]) / distance; cx, cy = dx * scale, dy * scale
                if ida:
                    bodies[ida]["position"][0] += cx * inv_a / (inv_a + inv_b); bodies[ida]["position"][1] += cy * inv_a / (inv_a + inv_b)
                if idb:
                    bodies[idb]["position"][0] -= cx * inv_b / (inv_a + inv_b); bodies[idb]["position"][1] -= cy * inv_b / (inv_a + inv_b)

        terminal_snapshot = None
        for contact in sorted(contacts - active_contacts):
            event_snapshot = contact_frames.get(contact, pre_contact_frame)
            events.append({
                "type": "contact_begin", "t": event_snapshot["t"],
                "participants": list(contact), "snapshot": event_snapshot,
            })
            if terminal_contact and set(contact) == set(terminal_contact):
                terminal_snapshot = event_snapshot
        for contact in sorted(active_contacts - contacts):
            events.append({"type": "contact_end", "t": _round(step * dt), "participants": list(contact)})
        active_contacts = contacts
        if terminal_snapshot is not None:
            frames.append(terminal_snapshot)
            break
        snapshot(step * dt, net)

    summary = {}
    for body_id in bodies:
        states = [frame["objects"][body_id] for frame in frames]
        summary[body_id] = {
            "initial_position": states[0]["position"], "final_position": states[-1]["position"],
            "initial_velocity": states[0]["velocity"], "final_velocity": states[-1]["velocity"],
            "minimum_speed": min(state["speed"] for state in states), "maximum_speed": max(state["speed"] for state in states),
            "x_range": [min(state["position"][0] for state in states), max(state["position"][0] for state in states)],
            "y_range": [min(state["position"][1] for state in states), max(state["position"][1] for state in states)],
        }
    public_bodies = [{key: value for key, value in body.items() if key != "_inv_mass"} for body in bodies.values()]
    return {
        "schema_version": "1.0", "engine": "mechanics-2d", "domain_model": "mechanics_2d",
        "units": trace_units,
        "duration": frames[-1]["t"] if terminal_contact and frames else _round(duration),
        "requested_duration": _round(duration), "playback_duration": _round(playback), "dt": _round(dt),
        "gravity": list(gravity), "potential_reference": list(potential_reference),
        "bounds": bounds, "bodies": public_bodies, "static_geometry": geometry,
        "springs": springs, "distance_constraints": constraints, "annotations": annotations,
        "ignored_annotations": ignored_annotations, "visual_instances": visual_instances,
        "time_series": frames, "events": events, "summary": summary,
    }
