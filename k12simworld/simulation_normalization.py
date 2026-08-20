"""Safe normalization for common declarative simulation aliases."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Mapping, Tuple

from .domain_solvers import DomainSimulationError


_TIME_TRIGGER_ALIASES = {"time", "at_time", "time_reached"}
_MISSING = object()


def normalize_domain_simulation_spec(
    engine: str, value: Mapping[str, Any]
) -> Tuple[Dict[str, Any], List[str]]:
    """Return a canonical deep copy and an audit list of deterministic changes.

    Only unambiguous representation aliases are changed. Semantic errors and
    unsupported triggers remain strict solver errors.
    """
    payload = copy.deepcopy(dict(value))
    changes: List[str] = []
    if str(engine).strip().lower() != "mechanics-2d":
        return payload, changes

    # Accept the natural model alias ``curve_constraints`` but persist and
    # validate one canonical collection throughout the compiler and trace.
    curve_constraints = payload.pop("curve_constraints", None)
    if curve_constraints is not None:
        if not isinstance(curve_constraints, list):
            raise DomainSimulationError("curve_constraints must be an array")
        path_constraints = payload.get("path_constraints")
        if path_constraints is None:
            path_constraints = []
        if not isinstance(path_constraints, list):
            raise DomainSimulationError("path_constraints must be an array")
        payload["path_constraints"] = [*path_constraints, *curve_constraints]
        changes.append(
            "curve_constraints: merged into canonical path_constraints"
        )

    geometry = payload.get("static_geometry")
    if isinstance(geometry, list):
        retained_geometry = []
        for index, item in enumerate(geometry):
            if not isinstance(item, Mapping) or str(item.get("type") or "segment") != "segment":
                retained_geometry.append(item)
                continue
            p1, p2 = item.get("p1"), item.get("p2")
            try:
                zero_length = (
                    isinstance(p1, (list, tuple))
                    and isinstance(p2, (list, tuple))
                    and len(p1) == 2
                    and len(p2) == 2
                    and all(math.isfinite(float(value)) for value in (*p1, *p2))
                    and math.dist(tuple(map(float, p1)), tuple(map(float, p2))) <= 1e-12
                )
            except (TypeError, ValueError):
                zero_length = False
            if zero_length:
                item_id = str(item.get("id") or index)
                changes.append(
                    f"static_geometry[{index}]: removed zero-length segment {item_id!r}"
                )
            else:
                retained_geometry.append(item)
        payload["static_geometry"] = retained_geometry

    actions = payload.get("actions")
    if not isinstance(actions, list):
        return payload, changes
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        trigger = action.get("trigger")
        if not isinstance(trigger, Mapping):
            continue
        trigger_type = str(trigger.get("type") or "").strip().lower()
        if trigger_type not in _TIME_TRIGGER_ALIASES:
            continue

        trigger_time: Any = _MISSING
        for key in ("time", "at_time", "value"):
            if key in trigger:
                trigger_time = trigger[key]
                break
        location = f"actions[{index}].trigger"
        if trigger_time is _MISSING:
            raise DomainSimulationError(
                f"{location} time alias requires one of time, at_time, or value"
            )
        if "time" in action:
            try:
                top_time = float(action["time"])
                nested_time = float(trigger_time)
            except (TypeError, ValueError) as exc:
                raise DomainSimulationError(
                    f"actions[{index}] contains a non-numeric time trigger"
                ) from exc
            if (
                not math.isfinite(top_time)
                or not math.isfinite(nested_time)
                or abs(top_time - nested_time) > 1e-12
            ):
                raise DomainSimulationError(
                    f"actions[{index}].time conflicts with nested time trigger"
                )
        else:
            action["time"] = trigger_time
        action.pop("trigger", None)
        changes.append(
            f"actions[{index}]: converted trigger.type={trigger_type} to top-level time"
        )
    return payload, changes
