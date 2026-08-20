"""Safe, semantics-preserving normalization for EduWorldSpec formulas.

Models often put arithmetic in a lookup binding or place dotted lookup paths
directly in an expression.  The runtime deliberately rejects both forms.  This
module translates those common representations into the small, auditable
formula DSL before the strict public contract is constructed.
"""

from __future__ import annotations

import ast
import copy
import math
import re
from typing import Any, Dict, List, Mapping, MutableMapping, Tuple

from .domain_solvers import DomainSimulationError, _SafeExpression


_LOOKUP_PATH_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_-]*(?:\.(?:[A-Za-z_][A-Za-z0-9_-]*|[0-9]+))*$"
)
_DOTTED_PATH_RE = re.compile(
    # A leading '-' is an arithmetic unary/binary operator, not part of the
    # lookup token. Keeping '-' in this look-behind made expressions such as
    # ``-objects.ball.velocity.1`` survive normalization as unsafe Attribute
    # syntax. Hyphens remain valid inside lookup path segments.
    r"(?<![A-Za-z0-9_.])"
    r"([A-Za-z_][A-Za-z0-9_-]*(?:\.(?:[A-Za-z_][A-Za-z0-9_-]*|[0-9]+))+)"
    r"(?![A-Za-z0-9_.-])"
)
_FORMULA_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_FUNCTION_NAMES = {
    "abs", "min", "max", "sqrt", "sin", "cos", "tan", "exp", "log",
    "floor", "ceil",
}
_CONSTANT_NAMES = {"pi", "e"}


def normalize_world_spec_formulas(
    value: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """Return a deep-copied payload and a list of deterministic changes.

    The function is deliberately best-effort.  It never weakens the downstream
    validator: any expression that remains unsafe or ambiguous is left for the
    normal EduWorldSpec validation error and one targeted model repair.
    """
    payload = copy.deepcopy(dict(value))
    changes: List[str] = []
    parameters = _numeric_parameters(payload.get("parameters"))
    for collection_name in ("target_observables", "invariants"):
        collection = payload.get(collection_name)
        if not isinstance(collection, list):
            continue
        for index, item in enumerate(collection):
            if not isinstance(item, MutableMapping):
                continue
            location = f"{collection_name}[{index}]"
            _normalize_formula_item(item, location, parameters, changes)
    return payload, changes


def _numeric_parameters(raw: Any) -> Dict[str, float | int]:
    result: Dict[str, float | int] = {}
    if not isinstance(raw, list):
        return result
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        identifier = item.get("id")
        value = item.get("value")
        if (
            isinstance(identifier, str)
            and _FORMULA_NAME_RE.fullmatch(identifier)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            result[identifier] = value
    return result


def _normalize_formula_item(
    item: MutableMapping[str, Any],
    location: str,
    parameters: Mapping[str, float | int],
    changes: List[str],
) -> None:
    raw_path = item.get("path")
    raw_expression = item.get("expression")
    path = raw_path.strip() if isinstance(raw_path, str) else ""
    expression = raw_expression.strip() if isinstance(raw_expression, str) else ""

    if path and _LOOKUP_PATH_RE.fullmatch(path):
        if path != raw_path:
            item["path"] = path
            changes.append(f"{location}.path: stripped surrounding whitespace")
        return

    # A frequent model error is to put a derived formula in `path`.  It is safe
    # to reinterpret only when no separate expression competes with it.
    if path and not expression:
        expression = path
        item.pop("path", None)
        item["expression"] = expression
        changes.append(f"{location}: moved arithmetic path to expression")
    if not expression:
        return

    normalized_expression = _normalize_operators(expression)
    if normalized_expression != expression:
        changes.append(f"{location}.expression: normalized arithmetic operators")
    expression = normalized_expression

    raw_bindings = item.get("bindings")
    bindings: Dict[str, str] = {}
    invalid_bindings: List[Tuple[str, Any]] = []
    if isinstance(raw_bindings, Mapping):
        for raw_alias, raw_value in raw_bindings.items():
            alias = str(raw_alias)
            if (
                _FORMULA_NAME_RE.fullmatch(alias)
                and isinstance(raw_value, str)
                and _LOOKUP_PATH_RE.fullmatch(raw_value.strip())
            ):
                bindings[alias] = raw_value.strip()
                if raw_value != raw_value.strip():
                    changes.append(f"{location}.bindings.{alias}: stripped whitespace")
            else:
                invalid_bindings.append((alias, raw_value))
    elif raw_bindings not in (None, {}):
        # Keep the invalid value so strict validation can report it precisely.
        return

    path_aliases = {path_value: alias for alias, path_value in bindings.items()}

    # Arithmetic accidentally placed in a binding is expanded into the parent
    # expression.  Its dotted paths become ordinary lookup bindings.
    for alias, raw_value in invalid_bindings:
        if not _FORMULA_NAME_RE.fullmatch(alias):
            continue
        replacement: str | None = None
        if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
            if math.isfinite(float(raw_value)):
                replacement = repr(raw_value)
        elif isinstance(raw_value, str) and raw_value.strip():
            replacement = _normalize_operators(raw_value.strip())
            replacement = _replace_dotted_paths(
                replacement, bindings, path_aliases, preferred_prefix=alias
            )
            replacement = _inline_numeric_parameters(replacement, parameters, bindings)
            if not _is_safe_expression(replacement, bindings):
                replacement = None
        if replacement is None:
            continue
        updated = re.sub(rf"\b{re.escape(alias)}\b", f"({replacement})", expression)
        if updated == expression:
            continue
        expression = updated
        changes.append(f"{location}.bindings.{alias}: inlined derived expression")

    expression = _replace_dotted_paths(expression, bindings, path_aliases)
    expression = _inline_numeric_parameters(expression, parameters, bindings)

    item["expression"] = expression
    item["bindings"] = bindings
    if expression != normalized_expression:
        changes.append(f"{location}.expression: converted lookups to bound aliases")


def _normalize_operators(source: str) -> str:
    source = source.replace("×", "*").replace("÷", "/").replace("−", "-")
    return re.sub(r"(?<!\*)\^(?!\*)", "**", source)


def _replace_dotted_paths(
    source: str,
    bindings: Dict[str, str],
    path_aliases: Dict[str, str],
    *,
    preferred_prefix: str = "value",
) -> str:
    def replace(match: re.Match[str]) -> str:
        path = match.group(1)
        alias = path_aliases.get(path)
        if alias is None:
            alias = _new_alias(path, bindings, preferred_prefix)
            bindings[alias] = path
            path_aliases[path] = alias
        return alias

    return _DOTTED_PATH_RE.sub(replace, source)


def _new_alias(path: str, bindings: Mapping[str, str], preferred_prefix: str) -> str:
    leaf = re.sub(r"[^A-Za-z0-9_]", "_", path.rsplit(".", 1)[-1]).strip("_")
    prefix = re.sub(r"[^A-Za-z0-9_]", "_", preferred_prefix).strip("_")
    base = leaf or prefix or "value"
    if not base[0].isalpha() and base[0] != "_":
        base = f"value_{base}"
    base = base[:56]
    candidate = base
    suffix = 2
    while candidate in bindings or candidate in _FUNCTION_NAMES or candidate in _CONSTANT_NAMES:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _inline_numeric_parameters(
    source: str,
    parameters: Mapping[str, float | int],
    bindings: Mapping[str, str],
) -> str:
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError:
        return source
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    for name in sorted(names - set(bindings) - _FUNCTION_NAMES - _CONSTANT_NAMES):
        if name in parameters:
            source = re.sub(rf"\b{re.escape(name)}\b", repr(parameters[name]), source)
    return source


def _is_safe_expression(source: str, bindings: Mapping[str, str]) -> bool:
    try:
        _SafeExpression(source, bindings.keys(), "normalized expression")
    except DomainSimulationError:
        return False
    return True
