"""Deterministic solvers for equation and specialized K12 physics scenes.

The candidate model supplies a small declarative ``simulation_spec``.  These
trusted, dependency-free solvers execute that specification and return an
auditable state trace.  Rendering consumes the trace; it never reimplements
the governing equations.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

from .domain_common import (
    DomainSimulationError,
    _EXPRESSION_CONSTANTS,
    _EXPRESSION_FUNCTIONS,
    _SafeExpression,
    _bounded_number,
    _expression_id,
    _finite_number,
    _normalise,
    _round,
    _safe_id,
    _vec2,
    _vec3,
)


DOMAIN_ENGINES = {"mechanics-2d", "equation-solver", "circuit-solver", "ray-optics"}
DOMAIN_MODELS = {
    "mechanics-2d": {"mechanics_2d"},
    "equation-solver": {"charged_particle_2d", "ode_system"},
    "circuit-solver": {"dc_circuit"},
    "ray-optics": {"geometric_ray_2d"},
}


def simulate_domain(engine: str, spec: Mapping[str, Any]) -> Dict[str, Any]:
    expected_models = DOMAIN_MODELS.get(engine)
    if expected_models is None:
        raise DomainSimulationError(f"unsupported domain engine {engine!r}")
    if not isinstance(spec, Mapping):
        raise DomainSimulationError("simulation_spec must be an object")
    model = str(spec.get("domain_model") or "").strip()
    if model not in expected_models:
        raise DomainSimulationError(
            f"engine {engine!r} requires one of {sorted(expected_models)!r}, got {model!r}"
        )
    if engine == "mechanics-2d":
        from .mechanics_2d import simulate_mechanics_2d
        return simulate_mechanics_2d(spec)
    if engine == "equation-solver":
        return simulate_charged_particles(spec) if model == "charged_particle_2d" else simulate_ode_system(spec)
    if engine == "circuit-solver":
        return simulate_dc_circuit(spec)
    return trace_geometric_rays(spec)


def domain_entity_ids(engine: str, spec: Mapping[str, Any]) -> set[str]:
    """Return model-controlled physical entity ids that must exist in WorldSpec."""
    if engine == "mechanics-2d":
        values = [*(spec.get("bodies") or []), *(spec.get("static_geometry") or [])]
    elif engine == "equation-solver":
        if spec.get("domain_model") == "charged_particle_2d":
            values = spec.get("particles") or []
        else:
            values = spec.get("objects") or []
    elif engine == "circuit-solver":
        values = spec.get("components") or []
    elif engine == "ray-optics":
        values = [*(spec.get("sources") or []), *(spec.get("elements") or [])]
    else:
        return set()
    result: set[str] = set()
    for value in values:
        if isinstance(value, Mapping) and value.get("id"):
            result.add(str(value["id"]))
    return result


def simulate_charged_particles(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Integrate q(E + v x B) with the fixed-step Boris method in 2D."""
    duration = _bounded_number(spec.get("duration", 8.0), "duration", 1e-15, 1e6)
    requested_dt = _bounded_number(spec.get("dt", duration / 480), "dt", 1e-18, 1e5)
    playback_duration = _bounded_number(
        spec.get("playback_duration", 8.0), "playback_duration", 1.0, 30.0
    )
    steps = int(math.ceil(duration / requested_dt))
    if steps > 6000:
        raise DomainSimulationError("charged-particle trace exceeds 6000 integration steps")
    dt = duration / steps
    electric = _vec2(spec.get("electric_field", [0.0, 0.0]), "electric_field")
    magnetic_raw = spec.get("magnetic_field", [0.0, 0.0, 0.0])
    magnetic = _vec3(magnetic_raw, "magnetic_field")
    if abs(magnetic[0]) > 1e-12 or abs(magnetic[1]) > 1e-12:
        raise DomainSimulationError("charged_particle_2d supports only an out-of-plane Bz field")

    raw_particles = spec.get("particles")
    if not isinstance(raw_particles, list) or not raw_particles:
        raise DomainSimulationError("charged_particle_2d requires particles")
    if len(raw_particles) > 32:
        raise DomainSimulationError("charged_particle_2d supports at most 32 particles")
    particles: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_particles):
        if not isinstance(raw, Mapping):
            raise DomainSimulationError(f"particles[{index}] must be an object")
        particle_id = _safe_id(raw.get("id"), f"particles[{index}].id")
        if particle_id in seen:
            raise DomainSimulationError(f"duplicate particle id {particle_id!r}")
        seen.add(particle_id)
        particles.append(
            {
                "id": particle_id,
                "mass": _bounded_number(
                    raw.get("mass"), f"particles[{index}].mass", 1e-40, 1e40,
                    inclusive_minimum=False,
                ),
                "charge": _finite_number(raw.get("charge", 0.0), f"particles[{index}].charge"),
                "position": list(_vec2(raw.get("position", [0.0, 0.0]), f"particles[{index}].position")),
                "velocity": list(_vec2(raw.get("velocity", [0.0, 0.0]), f"particles[{index}].velocity")),
                "color": str(raw.get("color") or "#2563eb"),
                "label": str(raw.get("label") or particle_id),
            }
        )

    bounds_raw = spec.get("bounds") or {}
    if not isinstance(bounds_raw, Mapping):
        raise DomainSimulationError("bounds must be an object")
    bounds = {
        "x_min": _finite_number(bounds_raw.get("x_min", -10.0), "bounds.x_min"),
        "x_max": _finite_number(bounds_raw.get("x_max", 10.0), "bounds.x_max"),
        "y_min": _finite_number(bounds_raw.get("y_min", -10.0), "bounds.y_min"),
        "y_max": _finite_number(bounds_raw.get("y_max", 10.0), "bounds.y_max"),
    }
    return _finish_charged_particles(
        spec=spec,
        duration=duration,
        playback_duration=playback_duration,
        steps=steps,
        dt=dt,
        electric=electric,
        magnetic=magnetic,
        particles=particles,
        bounds=bounds,
    )




def simulate_ode_system(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Execute a safe declarative first-order ODE system with fixed-step RK4."""
    duration = _bounded_number(spec.get("duration", 8.0), "duration", 1e-12, 1e6)
    requested_dt = _bounded_number(spec.get("dt", duration / 480), "dt", 1e-15, 1e5)
    playback_duration = _bounded_number(
        spec.get("playback_duration", 8.0), "playback_duration", 1.0, 30.0
    )
    steps = int(math.ceil(duration / requested_dt))
    if steps > 6000:
        raise DomainSimulationError("ode_system exceeds 6000 integration steps")
    dt = duration / steps
    raw_variables = spec.get("variables")
    if not isinstance(raw_variables, list) or not raw_variables or len(raw_variables) > 32:
        raise DomainSimulationError("ode_system requires 1 to 32 variables")
    variables: List[Dict[str, Any]] = []
    state: Dict[str, float] = {}
    for index, raw in enumerate(raw_variables):
        if not isinstance(raw, Mapping):
            raise DomainSimulationError(f"variables[{index}] must be an object")
        variable_id = _expression_id(raw.get("id"), f"variables[{index}].id")
        if variable_id in state or variable_id in _EXPRESSION_FUNCTIONS or variable_id in _EXPRESSION_CONSTANTS:
            raise DomainSimulationError(f"duplicate or reserved variable id {variable_id!r}")
        initial = _finite_number(raw.get("initial"), f"{variable_id}.initial")
        state[variable_id] = initial
        variables.append(
            {
                "id": variable_id,
                "initial": initial,
                "label": str(raw.get("label") or variable_id),
                "unit": str(raw.get("unit") or "1"),
                "color": str(raw.get("color") or "#2563eb"),
            }
        )
    raw_parameters = spec.get("parameters") or {}
    if not isinstance(raw_parameters, Mapping) or len(raw_parameters) > 64:
        raise DomainSimulationError("parameters must be an object with at most 64 values")
    parameters: Dict[str, float] = {}
    for raw_id, raw_value in raw_parameters.items():
        parameter_id = _expression_id(raw_id, "parameter id")
        if parameter_id in state or parameter_id in _EXPRESSION_FUNCTIONS or parameter_id in _EXPRESSION_CONSTANTS:
            raise DomainSimulationError(f"duplicate or reserved parameter id {parameter_id!r}")
        parameters[parameter_id] = _finite_number(raw_value, f"parameters.{parameter_id}")
    names = {*state, *parameters, "t"}
    raw_derivatives = spec.get("derivatives")
    if not isinstance(raw_derivatives, Mapping) or set(raw_derivatives) != set(state):
        raise DomainSimulationError("derivatives must define exactly one expression per variable")
    derivatives = {
        variable_id: _SafeExpression(raw_derivatives[variable_id], names, f"derivatives.{variable_id}")
        for variable_id in state
    }
    raw_observables = spec.get("observables") or {}
    if not isinstance(raw_observables, Mapping) or len(raw_observables) > 64:
        raise DomainSimulationError("observables must be an object with at most 64 expressions")
    observables: Dict[str, _SafeExpression] = {}
    for raw_id, expression in raw_observables.items():
        observable_id = _expression_id(raw_id, "observable id")
        if observable_id in state or observable_id in parameters:
            raise DomainSimulationError(f"observable id conflicts with state/parameter {observable_id!r}")
        observables[observable_id] = _SafeExpression(expression, names, f"observables.{observable_id}")

    raw_objects = spec.get("objects") or []
    if not isinstance(raw_objects, list) or len(raw_objects) > 32:
        raise DomainSimulationError("objects must be a list with at most 32 entries")
    objects: List[Dict[str, Any]] = []
    object_ids: set[str] = set()
    for index, raw in enumerate(raw_objects):
        if not isinstance(raw, Mapping):
            raise DomainSimulationError(f"objects[{index}] must be an object")
        object_id = _safe_id(raw.get("id"), f"objects[{index}].id")
        if object_id in object_ids:
            raise DomainSimulationError(f"duplicate object id {object_id!r}")
        object_ids.add(object_id)
        objects.append(
            {
                "id": object_id,
                "label": str(raw.get("label") or object_id),
                "kind": str(raw.get("kind") or "system"),
                "color": str(raw.get("color") or "#2563eb"),
            }
        )

    raw_conditions = spec.get("event_conditions") or []
    if not isinstance(raw_conditions, list) or len(raw_conditions) > 64:
        raise DomainSimulationError("event_conditions must be a list with at most 64 entries")
    conditions: List[Dict[str, Any]] = []
    condition_names = names | set(observables)
    condition_ids: set[str] = set()
    for index, raw in enumerate(raw_conditions):
        if not isinstance(raw, Mapping):
            raise DomainSimulationError(f"event_conditions[{index}] must be an object")
        condition_id = _safe_id(raw.get("id"), f"event_conditions[{index}].id")
        if condition_id in condition_ids:
            raise DomainSimulationError(f"duplicate event condition id {condition_id!r}")
        condition_ids.add(condition_id)
        conditions.append(
            {
                "id": condition_id,
                "expression": _SafeExpression(
                    raw.get("expression"), condition_names,
                    f"event_conditions[{index}].expression",
                ),
                "terminal": bool(raw.get("terminal", False)),
            }
        )
    raw_actions = spec.get("actions") or []
    if not isinstance(raw_actions, list) or len(raw_actions) > 64:
        raise DomainSimulationError("actions must be a list with at most 64 entries")
    actions: List[Dict[str, Any]] = []
    for index, raw in enumerate(raw_actions):
        if not isinstance(raw, Mapping):
            raise DomainSimulationError(f"actions[{index}] must be an object")
        target = _expression_id(raw.get("target"), f"actions[{index}].target")
        if target not in parameters:
            raise DomainSimulationError(f"action target {target!r} is not a parameter")
        actions.append(
            {
                "time": _bounded_number(raw.get("time"), f"actions[{index}].time", 0, duration),
                "target": target,
                "value": _finite_number(raw.get("value"), f"actions[{index}].value"),
            }
        )
    actions.sort(key=lambda item: item["time"])

    def environment(t: float, values: Mapping[str, float]) -> Dict[str, float]:
        return {**parameters, **values, "t": t}

    def derivative(t: float, values: Mapping[str, float]) -> Dict[str, float]:
        env = environment(t, values)
        return {
            variable_id: _finite_number(expression.evaluate(env), f"d{variable_id}/dt")
            for variable_id, expression in derivatives.items()
        }

    def add_scaled(values: Mapping[str, float], delta: Mapping[str, float], scale: float) -> Dict[str, float]:
        return {key: values[key] + scale * delta[key] for key in values}

    def snapshot(t: float) -> Dict[str, Any]:
        env = environment(t, state)
        observed = {
            key: _round(_finite_number(expression.evaluate(env), f"observables.{key}"))
            for key, expression in observables.items()
        }
        return {
            "t": _round(t),
            "state": {key: _round(value) for key, value in state.items()},
            "observables": observed,
            "parameters": {key: _round(value) for key, value in parameters.items()},
        }

    time_series: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    active_conditions = {condition["id"]: False for condition in conditions}
    action_index = 0
    terminal = False
    for step in range(steps + 1):
        current_time = step * dt
        while action_index < len(actions) and actions[action_index]["time"] <= current_time + 1e-12:
            action = actions[action_index]
            parameters[action["target"]] = action["value"]
            events.append({"type": "parameter_change", "t": _round(current_time), **action})
            action_index += 1
        frame = snapshot(current_time)
        time_series.append(frame)
        condition_env = {
            **environment(current_time, state),
            **frame["observables"],
        }
        for condition in conditions:
            now_active = bool(condition["expression"].evaluate(condition_env))
            if now_active and not active_conditions[condition["id"]]:
                events.append(
                    {"type": "condition", "id": condition["id"], "t": _round(current_time)}
                )
                terminal = terminal or condition["terminal"]
            active_conditions[condition["id"]] = now_active
        if terminal or step == steps:
            break
        k1 = derivative(current_time, state)
        k2 = derivative(current_time + dt / 2, add_scaled(state, k1, dt / 2))
        k3 = derivative(current_time + dt / 2, add_scaled(state, k2, dt / 2))
        k4 = derivative(current_time + dt, add_scaled(state, k3, dt))
        for variable_id in state:
            state[variable_id] += dt / 6 * (
                k1[variable_id] + 2 * k2[variable_id] + 2 * k3[variable_id] + k4[variable_id]
            )
            if not math.isfinite(state[variable_id]):
                raise DomainSimulationError(
                    f"state {variable_id!r} overflow; use a smaller dt or normalized units"
                )

    channels = [*state, *observables]
    requested_channels = spec.get("plot_channels") or channels[:4]
    if not isinstance(requested_channels, list) or not requested_channels:
        raise DomainSimulationError("plot_channels must be a non-empty list")
    plot_channels = [
        _expression_id(value, f"plot_channels[{index}]")
        for index, value in enumerate(requested_channels)
    ]
    unknown_channels = set(plot_channels) - set(channels)
    if unknown_channels:
        raise DomainSimulationError(f"unknown plot channels: {sorted(unknown_channels)}")

    raw_bindings = spec.get("visual_bindings") or []
    if not isinstance(raw_bindings, list) or len(raw_bindings) > 12:
        raise DomainSimulationError("visual_bindings must be a list with at most 12 entries")
    visual_bindings: List[Dict[str, Any]] = []
    for index, raw in enumerate(raw_bindings):
        if not isinstance(raw, Mapping):
            raise DomainSimulationError(f"visual_bindings[{index}] must be an object")
        object_id = _safe_id(raw.get("object_id"), f"visual_bindings[{index}].object_id")
        if object_id not in object_ids:
            raise DomainSimulationError(f"visual binding object {object_id!r} is not declared")
        channel = _expression_id(raw.get("channel"), f"visual_bindings[{index}].channel")
        if channel not in channels:
            raise DomainSimulationError(f"visual binding channel {channel!r} is unknown")
        kind = str(raw.get("type") or "gauge").strip().lower()
        if kind not in {"slider", "gauge", "rotor", "lamp"}:
            raise DomainSimulationError(f"unsupported visual binding type {kind!r}")
        binding: Dict[str, Any] = {
            "object_id": object_id,
            "channel": channel,
            "type": kind,
            "label": str(raw.get("label") or channel),
        }
        if raw.get("minimum") is not None:
            binding["minimum"] = _finite_number(raw["minimum"], f"visual_bindings[{index}].minimum")
        if raw.get("maximum") is not None:
            binding["maximum"] = _finite_number(raw["maximum"], f"visual_bindings[{index}].maximum")
        if "minimum" in binding and "maximum" in binding and binding["minimum"] >= binding["maximum"]:
            raise DomainSimulationError(f"visual_bindings[{index}] minimum must be below maximum")
        visual_bindings.append(binding)

    def channel_value(frame: Mapping[str, Any], channel: str) -> float:
        source = frame["state"] if channel in frame["state"] else frame["observables"]
        return float(source[channel])

    ranges = {
        channel: {
            "initial": channel_value(time_series[0], channel),
            "final": channel_value(time_series[-1], channel),
            "minimum": min(channel_value(frame, channel) for frame in time_series),
            "maximum": max(channel_value(frame, channel) for frame in time_series),
        }
        for channel in channels
    }
    return {
        "schema_version": "1.0",
        "engine": "equation-solver",
        "domain_model": "ode_system",
        "duration": _round(duration),
        "playback_duration": _round(playback_duration),
        "dt": _round(dt),
        "variables": variables,
        "observable_ids": list(observables),
        "plot_channels": plot_channels,
        "objects": objects,
        "visual_bindings": visual_bindings,
        "time_series": time_series,
        "events": events,
        "summary": {"channel_ranges": ranges, "final_time": time_series[-1]["t"]},
    }


def _finish_charged_particles(
    *,
    spec: Mapping[str, Any],
    duration: float,
    playback_duration: float,
    steps: int,
    dt: float,
    electric: Tuple[float, float],
    magnetic: Tuple[float, float, float],
    particles: List[Dict[str, Any]],
    bounds: Dict[str, float],
) -> Dict[str, Any]:
    if bounds["x_min"] >= bounds["x_max"] or bounds["y_min"] >= bounds["y_max"]:
        raise DomainSimulationError("bounds minima must be below maxima")

    raw_regions = spec.get("field_regions") or []
    if not isinstance(raw_regions, list) or len(raw_regions) > 32:
        raise DomainSimulationError("field_regions must be a list with at most 32 entries")
    field_regions: List[Dict[str, Any]] = []
    region_ids: set[str] = set()
    for index, raw in enumerate(raw_regions):
        if not isinstance(raw, Mapping):
            raise DomainSimulationError(f"field_regions[{index}] must be an object")
        region_id = _safe_id(raw.get("id"), f"field_regions[{index}].id")
        if region_id in region_ids:
            raise DomainSimulationError(f"duplicate field region {region_id!r}")
        region_ids.add(region_id)
        region_bounds_raw = raw.get("bounds") or {}
        if not isinstance(region_bounds_raw, Mapping):
            raise DomainSimulationError(f"{region_id}.bounds must be an object")
        region_bounds = {
            "x_min": _finite_number(region_bounds_raw.get("x_min"), f"{region_id}.bounds.x_min"),
            "x_max": _finite_number(region_bounds_raw.get("x_max"), f"{region_id}.bounds.x_max"),
            "y_min": _finite_number(region_bounds_raw.get("y_min"), f"{region_id}.bounds.y_min"),
            "y_max": _finite_number(region_bounds_raw.get("y_max"), f"{region_id}.bounds.y_max"),
        }
        if region_bounds["x_min"] >= region_bounds["x_max"] or region_bounds["y_min"] >= region_bounds["y_max"]:
            raise DomainSimulationError(f"{region_id} bounds minima must be below maxima")
        region_electric = _vec2(raw.get("electric_field", [0, 0]), f"{region_id}.electric_field")
        region_magnetic = _vec3(raw.get("magnetic_field", [0, 0, 0]), f"{region_id}.magnetic_field")
        if abs(region_magnetic[0]) > 1e-12 or abs(region_magnetic[1]) > 1e-12:
            raise DomainSimulationError(f"{region_id} supports only an out-of-plane Bz field")
        mode = str(raw.get("mode") or "add").lower()
        if mode not in {"add", "override"}:
            raise DomainSimulationError(f"{region_id}.mode must be add or override")
        field_regions.append(
            {
                "id": region_id,
                "bounds": region_bounds,
                "electric_field": list(region_electric),
                "magnetic_field": list(region_magnetic),
                "mode": mode,
                "label": str(raw.get("label") or region_id),
                "color": str(raw.get("color") or "#dbeafe"),
            }
        )

    def fields_at(position: Sequence[float]) -> Tuple[Tuple[float, float], float, List[str]]:
        ex, ey, bz_value = electric[0], electric[1], magnetic[2]
        active: List[str] = []
        for region in field_regions:
            region_bounds = region["bounds"]
            if not (
                region_bounds["x_min"] <= position[0] <= region_bounds["x_max"]
                and region_bounds["y_min"] <= position[1] <= region_bounds["y_max"]
            ):
                continue
            active.append(region["id"])
            rex, rey = region["electric_field"]
            rbz = region["magnetic_field"][2]
            if region["mode"] == "override":
                ex, ey, bz_value = rex, rey, rbz
            else:
                ex, ey, bz_value = ex + rex, ey + rey, bz_value + rbz
        return (ex, ey), bz_value, active

    events: List[Dict[str, Any]] = []
    exited: set[str] = set()

    def snapshot(t: float) -> Dict[str, Any]:
        states: Dict[str, Any] = {}
        for particle in particles:
            vx, vy = particle["velocity"]
            speed = math.hypot(vx, vy)
            local_electric, local_bz, active_regions = fields_at(particle["position"])
            states[particle["id"]] = {
                "position": [_round(particle["position"][0]), _round(particle["position"][1])],
                "velocity": [_round(vx), _round(vy)],
                "speed": _round(speed),
                "kinetic_energy": _round(0.5 * particle["mass"] * speed * speed),
                "electric_field": [_round(value) for value in local_electric],
                "magnetic_field_bz": _round(local_bz),
                "active_regions": active_regions,
            }
        return {"t": _round(t), "objects": states}

    time_series = [snapshot(0.0)]
    active_regions_by_particle = {
        particle["id"]: set(fields_at(particle["position"])[2]) for particle in particles
    }
    for step in range(1, steps + 1):
        for particle in particles:
            local_electric, bz, _ = fields_at(particle["position"])
            q_over_m_half_dt = particle["charge"] / particle["mass"] * dt * 0.5
            vx, vy = particle["velocity"]
            # Boris electric half-kick, magnetic rotation, electric half-kick.
            vmx = vx + q_over_m_half_dt * local_electric[0]
            vmy = vy + q_over_m_half_dt * local_electric[1]
            t_b = q_over_m_half_dt * bz
            s_b = 2.0 * t_b / (1.0 + t_b * t_b)
            vpx = vmx + vmy * t_b
            vpy = vmy - vmx * t_b
            vplus_x = vmx + vpy * s_b
            vplus_y = vmy - vpx * s_b
            vx_new = vplus_x + q_over_m_half_dt * local_electric[0]
            vy_new = vplus_y + q_over_m_half_dt * local_electric[1]
            if not all(math.isfinite(value) for value in (vx_new, vy_new)):
                raise DomainSimulationError(
                    f"particle {particle['id']!r} state overflow; use a smaller dt or normalized units"
                )
            particle["velocity"] = [vx_new, vy_new]
            particle["position"][0] += vx_new * dt
            particle["position"][1] += vy_new * dt
            if not all(math.isfinite(value) for value in particle["position"]):
                raise DomainSimulationError(
                    f"particle {particle['id']!r} position overflow; use normalized units"
                )
            x, y = particle["position"]
            new_regions = set(fields_at(particle["position"])[2])
            old_regions = active_regions_by_particle[particle["id"]]
            for region_id in sorted(new_regions - old_regions):
                events.append(
                    {"type": "field_region_entry", "t": _round(step * dt), "participants": [particle["id"], region_id]}
                )
            for region_id in sorted(old_regions - new_regions):
                events.append(
                    {"type": "field_region_exit", "t": _round(step * dt), "participants": [particle["id"], region_id]}
                )
            active_regions_by_particle[particle["id"]] = new_regions
            if particle["id"] not in exited and not (
                bounds["x_min"] <= x <= bounds["x_max"]
                and bounds["y_min"] <= y <= bounds["y_max"]
            ):
                exited.add(particle["id"])
                events.append(
                    {
                        "type": "boundary_exit",
                        "t": _round(step * dt),
                        "participants": [particle["id"]],
                    }
                )
        time_series.append(snapshot(step * dt))

    initial = time_series[0]["objects"]
    final = time_series[-1]["objects"]
    return {
        "schema_version": "1.0",
        "engine": "equation-solver",
        "domain_model": "charged_particle_2d",
        "units": dict(spec.get("units") or {"position": "m", "time": "s"}),
        "dt": _round(dt),
        "duration": _round(duration),
        "playback_duration": _round(playback_duration),
        "fields": {
            "electric": [_round(value) for value in electric],
            "magnetic": [_round(value) for value in magnetic],
        },
        "field_regions": field_regions,
        "bounds": bounds,
        "particles": [
            {key: particle[key] for key in ("id", "mass", "charge", "color", "label")}
            for particle in particles
        ],
        "time_series": time_series,
        "events": events,
        "summary": {
            particle_id: {
                "initial_speed": initial[particle_id]["speed"],
                "final_speed": final[particle_id]["speed"],
                "minimum_speed": min(frame["objects"][particle_id]["speed"] for frame in time_series),
                "maximum_speed": max(frame["objects"][particle_id]["speed"] for frame in time_series),
                "final_position": final[particle_id]["position"],
                "x_range": [
                    min(frame["objects"][particle_id]["position"][0] for frame in time_series),
                    max(frame["objects"][particle_id]["position"][0] for frame in time_series),
                ],
                "y_range": [
                    min(frame["objects"][particle_id]["position"][1] for frame in time_series),
                    max(frame["objects"][particle_id]["position"][1] for frame in time_series),
                ],
            }
            for particle_id in sorted(initial)
        },
    }


def _solve_linear(matrix: List[List[float]], vector: List[float]) -> List[float]:
    size = len(vector)
    augmented = [list(matrix[row]) + [vector[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            raise DomainSimulationError("circuit equations are singular or under-specified")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if abs(factor) <= 1e-18:
                continue
            augmented[row] = [
                augmented[row][item] - factor * augmented[column][item]
                for item in range(size + 1)
            ]
    return [augmented[row][-1] for row in range(size)]


_RESISTIVE_TYPES = {"resistor", "lamp", "ammeter", "voltmeter", "switch"}
_CIRCUIT_TYPES = _RESISTIVE_TYPES | {"voltage_source", "current_source", "wire"}


def _component_resistance(component: Mapping[str, Any]) -> float:
    kind = component["type"]
    if kind == "switch":
        default = 1e-6 if bool(component.get("closed", False)) else 1e12
        key = "closed_resistance" if bool(component.get("closed", False)) else "open_resistance"
        return _bounded_number(component.get(key, default), f"{component['id']}.{key}", 1e-12, 1e15)
    if kind == "wire":
        return _bounded_number(component.get("resistance", 1e-6), f"{component['id']}.resistance", 1e-12, 1e15)
    defaults = {"ammeter": 1e-6, "voltmeter": 1e9}
    return _bounded_number(
        component.get("resistance", defaults.get(kind)),
        f"{component['id']}.resistance",
        1e-12,
        1e15,
    )


def _normalise_circuit(spec: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], str, List[Dict[str, Any]]]:
    raw_nodes = spec.get("nodes")
    if not isinstance(raw_nodes, list) or len(raw_nodes) < 2 or len(raw_nodes) > 64:
        raise DomainSimulationError("dc_circuit requires 2 to 64 nodes")
    nodes: List[Dict[str, Any]] = []
    node_ids: set[str] = set()
    for index, raw in enumerate(raw_nodes):
        value = {"id": raw} if isinstance(raw, str) else dict(raw) if isinstance(raw, Mapping) else None
        if value is None:
            raise DomainSimulationError(f"nodes[{index}] must be a string or object")
        node_id = _safe_id(value.get("id"), f"nodes[{index}].id")
        if node_id in node_ids:
            raise DomainSimulationError(f"duplicate circuit node {node_id!r}")
        node_ids.add(node_id)
        nodes.append(
            {
                "id": node_id,
                "x": _finite_number(value.get("x", 0.0), f"nodes[{index}].x"),
                "y": _finite_number(value.get("y", 0.0), f"nodes[{index}].y"),
                "label": str(value.get("label") or node_id),
            }
        )
    ground = _safe_id(spec.get("ground") or nodes[0]["id"], "ground")
    if ground not in node_ids:
        raise DomainSimulationError("ground must reference a declared node")

    raw_components = spec.get("components")
    if not isinstance(raw_components, list) or not raw_components or len(raw_components) > 128:
        raise DomainSimulationError("dc_circuit requires 1 to 128 components")
    components: List[Dict[str, Any]] = []
    component_ids: set[str] = set()
    for index, raw in enumerate(raw_components):
        if not isinstance(raw, Mapping):
            raise DomainSimulationError(f"components[{index}] must be an object")
        component = dict(raw)
        component_id = _safe_id(component.get("id"), f"components[{index}].id")
        if component_id in component_ids:
            raise DomainSimulationError(f"duplicate circuit component {component_id!r}")
        component_ids.add(component_id)
        kind = str(component.get("type") or "").strip().lower()
        if kind not in _CIRCUIT_TYPES:
            raise DomainSimulationError(f"unsupported circuit component type {kind!r}")
        node_a = _safe_id(component.get("node_a"), f"{component_id}.node_a")
        node_b = _safe_id(component.get("node_b"), f"{component_id}.node_b")
        if node_a not in node_ids or node_b not in node_ids or node_a == node_b:
            raise DomainSimulationError(f"{component_id} must connect two different declared nodes")
        component.update(
            {
                "id": component_id,
                "type": kind,
                "node_a": node_a,
                "node_b": node_b,
                "label": str(component.get("label") or component_id),
            }
        )
        if kind == "switch" and "closed" in component and not isinstance(component["closed"], bool):
            raise DomainSimulationError(f"{component_id}.closed must be boolean")
        if kind == "voltage_source":
            component["voltage"] = _finite_number(component.get("voltage"), f"{component_id}.voltage")
        elif kind == "current_source":
            component["current"] = _finite_number(component.get("current"), f"{component_id}.current")
        else:
            component["resistance"] = _component_resistance(component)
        if kind == "lamp":
            component["rated_power"] = _bounded_number(
                component.get("rated_power", 1.0), f"{component_id}.rated_power", 1e-12, 1e15
            )
        components.append(component)
    return nodes, ground, components


def solve_dc_operating_point(
    nodes: Sequence[Mapping[str, Any]], ground: str, components: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    non_ground = [str(node["id"]) for node in nodes if str(node["id"]) != ground]
    node_index = {node_id: index for index, node_id in enumerate(non_ground)}
    voltage_sources = [component for component in components if component["type"] == "voltage_source"]
    size = len(non_ground) + len(voltage_sources)
    if size == 0:
        raise DomainSimulationError("circuit has no solvable unknowns")
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    rhs = [0.0 for _ in range(size)]
    # A tiny shunt makes genuinely floating nodes numerically explicit rather
    # than crashing; voltmeter/open-switch values remain effectively ideal.
    for index in range(len(non_ground)):
        matrix[index][index] += 1e-12

    def stamp_conductance(node_a: str, node_b: str, conductance: float) -> None:
        ia, ib = node_index.get(node_a), node_index.get(node_b)
        if ia is not None:
            matrix[ia][ia] += conductance
        if ib is not None:
            matrix[ib][ib] += conductance
        if ia is not None and ib is not None:
            matrix[ia][ib] -= conductance
            matrix[ib][ia] -= conductance

    for component in components:
        kind = str(component["type"])
        node_a, node_b = str(component["node_a"]), str(component["node_b"])
        if kind in _RESISTIVE_TYPES or kind == "wire":
            stamp_conductance(node_a, node_b, 1.0 / _component_resistance(component))
        elif kind == "current_source":
            current = _finite_number(component["current"], f"{component['id']}.current")
            ia, ib = node_index.get(node_a), node_index.get(node_b)
            if ia is not None:
                rhs[ia] -= current
            if ib is not None:
                rhs[ib] += current

    for source_number, component in enumerate(voltage_sources):
        row = len(non_ground) + source_number
        ia = node_index.get(str(component["node_a"]))
        ib = node_index.get(str(component["node_b"]))
        if ia is not None:
            matrix[ia][row] += 1.0
            matrix[row][ia] += 1.0
        if ib is not None:
            matrix[ib][row] -= 1.0
            matrix[row][ib] -= 1.0
        rhs[row] = _finite_number(component["voltage"], f"{component['id']}.voltage")

    solution = _solve_linear(matrix, rhs)
    voltages = {ground: 0.0, **{node: solution[index] for node, index in node_index.items()}}
    source_currents = {
        str(component["id"]): solution[len(non_ground) + index]
        for index, component in enumerate(voltage_sources)
    }
    states: Dict[str, Any] = {}
    for component in components:
        component_id = str(component["id"])
        kind = str(component["type"])
        voltage = voltages[str(component["node_a"])] - voltages[str(component["node_b"])]
        if kind in _RESISTIVE_TYPES or kind == "wire":
            current = voltage / _component_resistance(component)
        elif kind == "current_source":
            current = _finite_number(component["current"], f"{component_id}.current")
        else:
            current = source_currents[component_id]
        power = voltage * current
        state: Dict[str, Any] = {
            "voltage": _round(voltage),
            "current": _round(current),
            "power": _round(power),
        }
        if kind == "ammeter":
            state["reading"] = _round(current)
            state["reading_unit"] = "A"
        elif kind == "voltmeter":
            state["reading"] = _round(voltage)
            state["reading_unit"] = "V"
        elif kind == "lamp":
            rated = _finite_number(component["rated_power"], f"{component_id}.rated_power")
            state["brightness"] = _round(min(1.0, max(0.0, abs(power) / rated)))
            state["on"] = abs(power) > rated * 1e-6
        elif kind == "switch":
            state["closed"] = bool(component.get("closed", False))
        states[component_id] = state
    return {
        "node_voltages": {key: _round(value) for key, value in sorted(voltages.items())},
        "components": states,
    }


def _apply_circuit_action(
    components: MutableMapping[str, Dict[str, Any]], action: Mapping[str, Any]
) -> Dict[str, Any]:
    target = _safe_id(action.get("target"), "action.target")
    component = components.get(target)
    if component is None:
        raise DomainSimulationError(f"circuit action references unknown component {target!r}")
    prop = str(action.get("property") or "").strip().lower()
    value = action.get("value")
    allowed = {
        "closed": "closed",
        "resistance": "resistance",
        "voltage": "voltage",
        "current": "current",
        "rated_power": "rated_power",
    }
    if prop not in allowed:
        raise DomainSimulationError(f"unsupported circuit action property {prop!r}")
    allowed_by_type = {
        "switch": {"closed", "resistance"},
        "resistor": {"resistance"},
        "lamp": {"resistance", "rated_power"},
        "ammeter": {"resistance"},
        "voltmeter": {"resistance"},
        "wire": {"resistance"},
        "voltage_source": {"voltage"},
        "current_source": {"current"},
    }
    if prop not in allowed_by_type[str(component["type"])]:
        raise DomainSimulationError(
            f"property {prop!r} is not valid for {component['type']!r} component {target!r}"
        )
    if prop == "closed":
        if not isinstance(value, bool):
            raise DomainSimulationError(f"action.{target}.closed must be boolean")
        component[prop] = value
        component["resistance"] = _component_resistance(component)
    else:
        component[prop] = _finite_number(value, f"action.{target}.{prop}")
        if prop in {"resistance", "rated_power"} and component[prop] <= 0:
            raise DomainSimulationError(f"action {prop} must be positive")
    return {"target": target, "property": prop, "value": component[prop]}


def simulate_dc_circuit(spec: Mapping[str, Any]) -> Dict[str, Any]:
    nodes, ground, component_list = _normalise_circuit(spec)
    duration = _bounded_number(spec.get("duration", 8.0), "duration", 0.01, 30.0)
    requested_dt = _bounded_number(spec.get("dt", 0.1), "dt", 0.01, 1.0)
    steps = int(math.ceil(duration / requested_dt))
    if steps > 3000:
        raise DomainSimulationError("circuit trace exceeds 3000 time steps")
    dt = duration / steps
    raw_actions = spec.get("actions") or []
    if not isinstance(raw_actions, list) or len(raw_actions) > 128:
        raise DomainSimulationError("actions must be a list with at most 128 entries")
    actions: List[Dict[str, Any]] = []
    for index, raw in enumerate(raw_actions):
        if not isinstance(raw, Mapping):
            raise DomainSimulationError(f"actions[{index}] must be an object")
        action = dict(raw)
        action["time"] = _bounded_number(action.get("time"), f"actions[{index}].time", 0.0, duration)
        actions.append(action)
    actions.sort(key=lambda item: item["time"])

    components = {component["id"]: copy.deepcopy(component) for component in component_list}
    time_series: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    action_index = 0
    operating_point: Dict[str, Any] | None = None
    for step in range(steps + 1):
        current_time = step * dt
        dirty = operating_point is None
        while action_index < len(actions) and actions[action_index]["time"] <= current_time + 1e-12:
            applied = _apply_circuit_action(components, actions[action_index])
            events.append({"type": "parameter_change", "t": _round(current_time), **applied})
            action_index += 1
            dirty = True
        if dirty:
            operating_point = solve_dc_operating_point(nodes, ground, list(components.values()))
        time_series.append({"t": _round(current_time), **copy.deepcopy(operating_point)})
    assert operating_point is not None
    final_state = copy.deepcopy(time_series[-1])
    component_ranges: Dict[str, Any] = {}
    for component_id in components:
        values = [frame["components"][component_id] for frame in time_series]
        component_ranges[component_id] = {}
        for key in ("voltage", "current", "power", "reading", "brightness"):
            numeric = [value[key] for value in values if isinstance(value.get(key), (int, float))]
            if numeric:
                component_ranges[component_id][key] = {
                    "initial": numeric[0], "final": numeric[-1],
                    "minimum": min(numeric), "maximum": max(numeric),
                }
    final_state["component_ranges"] = component_ranges
    return {
        "schema_version": "1.0",
        "engine": "circuit-solver",
        "domain_model": "dc_circuit",
        "units": {"voltage": "V", "current": "A", "resistance": "ohm", "power": "W"},
        "duration": _round(duration),
        "dt": _round(dt),
        "ground": ground,
        "nodes": nodes,
        "components": list(components.values()),
        "time_series": time_series,
        "events": events,
        "summary": final_state,
    }


def _cross(left: Tuple[float, float], right: Tuple[float, float]) -> float:
    return left[0] * right[1] - left[1] * right[0]


def _dot(left: Tuple[float, float], right: Tuple[float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1]


def _ray_segment_intersection(
    origin: Tuple[float, float],
    direction: Tuple[float, float],
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
) -> Tuple[float, float, Tuple[float, float]] | None:
    segment = point_b[0] - point_a[0], point_b[1] - point_a[1]
    denominator = _cross(direction, segment)
    if abs(denominator) < 1e-12:
        return None
    delta = point_a[0] - origin[0], point_a[1] - origin[1]
    distance = _cross(delta, segment) / denominator
    fraction = _cross(delta, direction) / denominator
    if distance <= 1e-8 or fraction < -1e-10 or fraction > 1.0 + 1e-10:
        return None
    point = origin[0] + distance * direction[0], origin[1] + distance * direction[1]
    return distance, fraction, point


def _element_segment(element: Mapping[str, Any]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if element["type"] == "thin_lens":
        x = _finite_number(element.get("x"), f"{element['id']}.x")
        center_y = _finite_number(element.get("optical_axis_y", 0.0), f"{element['id']}.optical_axis_y")
        aperture = _bounded_number(element.get("aperture", 10.0), f"{element['id']}.aperture", 1e-9, 1e9)
        return (x, center_y - aperture / 2), (x, center_y + aperture / 2)
    return _vec2(element.get("p1"), f"{element['id']}.p1"), _vec2(
        element.get("p2"), f"{element['id']}.p2"
    )


def _reflect(direction: Tuple[float, float], normal: Tuple[float, float]) -> Tuple[float, float]:
    normal = _normalise(normal, "surface normal")
    if _dot(direction, normal) > 0:
        normal = -normal[0], -normal[1]
    projection = 2.0 * _dot(direction, normal)
    return _normalise((direction[0] - projection * normal[0], direction[1] - projection * normal[1]))


def _refract(
    direction: Tuple[float, float], normal: Tuple[float, float], n_from: float, n_to: float
) -> Tuple[Tuple[float, float], bool]:
    normal = _normalise(normal, "surface normal")
    if _dot(direction, normal) > 0:
        normal = -normal[0], -normal[1]
    cosine = -_dot(normal, direction)
    eta = n_from / n_to
    discriminant = 1.0 - eta * eta * (1.0 - cosine * cosine)
    if discriminant < 0:
        return _reflect(direction, normal), True
    result = (
        eta * direction[0] + (eta * cosine - math.sqrt(discriminant)) * normal[0],
        eta * direction[1] + (eta * cosine - math.sqrt(discriminant)) * normal[1],
    )
    return _normalise(result), False


def trace_geometric_rays(spec: Mapping[str, Any]) -> Dict[str, Any]:
    raw_sources = spec.get("sources")
    raw_elements = spec.get("elements")
    if not isinstance(raw_sources, list) or not raw_sources or len(raw_sources) > 32:
        raise DomainSimulationError("geometric_ray_2d requires 1 to 32 sources")
    if not isinstance(raw_elements, list) or not raw_elements or len(raw_elements) > 64:
        raise DomainSimulationError("geometric_ray_2d requires 1 to 64 elements")
    max_interactions = int(_bounded_number(spec.get("max_interactions", 12), "max_interactions", 1, 32))
    max_distance = _bounded_number(spec.get("max_distance", 30.0), "max_distance", 0.01, 1e6)

    elements: List[Dict[str, Any]] = []
    element_ids: set[str] = set()
    allowed_elements = {"mirror", "refractive_interface", "thin_lens", "screen", "absorber"}
    for index, raw in enumerate(raw_elements):
        if not isinstance(raw, Mapping):
            raise DomainSimulationError(f"elements[{index}] must be an object")
        element = dict(raw)
        element_id = _safe_id(element.get("id"), f"elements[{index}].id")
        if element_id in element_ids:
            raise DomainSimulationError(f"duplicate optical element {element_id!r}")
        element_ids.add(element_id)
        kind = str(element.get("type") or "").strip().lower()
        if kind not in allowed_elements:
            raise DomainSimulationError(f"unsupported optical element type {kind!r}")
        element.update({"id": element_id, "type": kind, "label": str(element.get("label") or element_id)})
        point_a, point_b = _element_segment(element)
        if math.dist(point_a, point_b) <= 1e-12:
            raise DomainSimulationError(f"optical element {element_id!r} has zero length")
        element["p1"], element["p2"] = list(point_a), list(point_b)
        if kind == "thin_lens":
            element["focal_length"] = _finite_number(element.get("focal_length"), f"{element_id}.focal_length")
            if abs(element["focal_length"]) <= 1e-12:
                raise DomainSimulationError("thin lens focal length must be non-zero")
        elif kind == "refractive_interface":
            element["n1"] = _bounded_number(element.get("n1", 1.0), f"{element_id}.n1", 1e-6, 100.0)
            element["n2"] = _bounded_number(element.get("n2"), f"{element_id}.n2", 1e-6, 100.0)
        elements.append(element)

    paths: List[Dict[str, Any]] = []
    source_ids: set[str] = set()
    all_events: List[Dict[str, Any]] = []
    for index, raw in enumerate(raw_sources):
        if not isinstance(raw, Mapping):
            raise DomainSimulationError(f"sources[{index}] must be an object")
        source_id = _safe_id(raw.get("id"), f"sources[{index}].id")
        if source_id in source_ids:
            raise DomainSimulationError(f"duplicate ray source {source_id!r}")
        source_ids.add(source_id)
        origin = _vec2(raw.get("origin"), f"{source_id}.origin")
        direction = _normalise(_vec2(raw.get("direction"), f"{source_id}.direction"))
        medium = _bounded_number(raw.get("refractive_index", 1.0), f"{source_id}.refractive_index", 1e-6, 100.0)
        points = [list(origin)]
        interactions: List[Dict[str, Any]] = []
        last_element = ""
        current_origin = origin
        current_direction = direction
        for _ in range(max_interactions):
            candidates: List[Tuple[float, Dict[str, Any], Tuple[float, float]]] = []
            for element in elements:
                hit = _ray_segment_intersection(
                    current_origin, current_direction, tuple(element["p1"]), tuple(element["p2"])
                )
                if hit is None:
                    continue
                distance, _, point = hit
                if element["id"] == last_element and distance < 1e-5:
                    continue
                candidates.append((distance, element, point))
            if not candidates:
                endpoint = (
                    current_origin[0] + current_direction[0] * max_distance,
                    current_origin[1] + current_direction[1] * max_distance,
                )
                points.append([_round(endpoint[0]), _round(endpoint[1])])
                break
            distance, element, point = min(candidates, key=lambda item: item[0])
            if distance > max_distance:
                endpoint = (
                    current_origin[0] + current_direction[0] * max_distance,
                    current_origin[1] + current_direction[1] * max_distance,
                )
                points.append([_round(endpoint[0]), _round(endpoint[1])])
                break
            points.append([_round(point[0]), _round(point[1])])
            kind = element["type"]
            event: Dict[str, Any] = {
                "type": kind,
                "element": element["id"],
                "point": [_round(point[0]), _round(point[1])],
                "incoming_direction": [_round(value) for value in current_direction],
            }
            segment = (
                element["p2"][0] - element["p1"][0],
                element["p2"][1] - element["p1"][1],
            )
            normal = _normalise((-segment[1], segment[0]), "surface normal")
            stop = False
            if kind == "mirror":
                current_direction = _reflect(current_direction, normal)
            elif kind == "refractive_interface":
                n1, n2 = float(element["n1"]), float(element["n2"])
                tolerance = 1e-7 * max(1.0, n1, n2)
                if min(abs(medium - n1), abs(medium - n2)) > tolerance:
                    raise DomainSimulationError(
                        f"ray {source_id!r} medium {medium} does not match either side of "
                        f"interface {element['id']!r}"
                    )
                target_medium = n2 if abs(medium - n1) <= abs(medium - n2) else n1
                current_direction, total_internal_reflection = _refract(
                    current_direction, normal, medium, target_medium
                )
                event["total_internal_reflection"] = total_internal_reflection
                if not total_internal_reflection:
                    medium = target_medium
                event["refractive_index"] = _round(medium)
            elif kind == "thin_lens":
                sign = 1.0 if current_direction[0] >= 0 else -1.0
                if abs(current_direction[0]) <= 1e-9:
                    raise DomainSimulationError("thin-lens paraxial model cannot accept a vertical ray")
                slope = current_direction[1] / current_direction[0]
                axis_y = _finite_number(element.get("optical_axis_y", 0.0), f"{element['id']}.optical_axis_y")
                outgoing_slope = slope - sign * (point[1] - axis_y) / float(element["focal_length"])
                current_direction = _normalise((sign, sign * outgoing_slope))
            else:
                stop = True
            event["outgoing_direction"] = [_round(value) for value in current_direction]
            interactions.append(event)
            all_events.append({"source": source_id, **event})
            if stop:
                break
            last_element = element["id"]
            current_origin = (
                point[0] + current_direction[0] * 1e-7,
                point[1] + current_direction[1] * 1e-7,
            )
        paths.append(
            {
                "source_id": source_id,
                "label": str(raw.get("label") or source_id),
                "color": str(raw.get("color") or "#dc2626"),
                "points": points,
                "interactions": interactions,
            }
        )
    return {
        "schema_version": "1.0",
        "engine": "ray-optics",
        "domain_model": "geometric_ray_2d",
        "units": dict(spec.get("units") or {"position": "m"}),
        "elements": elements,
        "paths": paths,
        "events": all_events,
        "summary": {
            "ray_count": len(paths),
            "interaction_count": len(all_events),
            "screen_hits": sum(event["type"] == "screen" for event in all_events),
        },
    }
