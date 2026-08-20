"""Shared validation and safe-expression helpers for declarative domain solvers."""

from __future__ import annotations

import ast
import keyword
import math
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


class DomainSimulationError(ValueError):
    """Raised when a declarative simulation cannot be executed safely."""


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise DomainSimulationError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DomainSimulationError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise DomainSimulationError(f"{name} must be finite")
    return result


def _bounded_number(
    value: Any, name: str, minimum: float, maximum: float, *, inclusive_minimum: bool = True
) -> float:
    result = _finite_number(value, name)
    lower_ok = result >= minimum if inclusive_minimum else result > minimum
    if not lower_ok or result > maximum:
        bracket = "[" if inclusive_minimum else "("
        raise DomainSimulationError(f"{name} must be in {bracket}{minimum}, {maximum}]")
    return result


def _vec2(value: Any, name: str) -> Tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise DomainSimulationError(f"{name} must be a two-element vector")
    return _finite_number(value[0], f"{name}[0]"), _finite_number(value[1], f"{name}[1]")


def _vec3(value: Any, name: str) -> Tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise DomainSimulationError(f"{name} must be a three-element vector")
    return (
        _finite_number(value[0], f"{name}[0]"),
        _finite_number(value[1], f"{name}[1]"),
        _finite_number(value[2], f"{name}[2]"),
    )


def _safe_id(value: Any, name: str) -> str:
    result = str(value or "").strip()
    if not result or len(result) > 64 or not result[0].isalpha() or not all(
        char.isalnum() or char in "_-" for char in result
    ):
        raise DomainSimulationError(f"{name} must be a safe identifier")
    return result


def _expression_id(value: Any, name: str) -> str:
    result = _safe_id(value, name)
    if (
        not result.isascii()
        or not result.isidentifier()
        or keyword.iskeyword(result)
        or "-" in result
    ):
        raise DomainSimulationError(f"{name} must contain only ASCII letters, digits, and underscores")
    return result


def _normalise(vector: Tuple[float, float], name: str = "direction") -> Tuple[float, float]:
    length = math.hypot(*vector)
    if length <= 1e-12:
        raise DomainSimulationError(f"{name} must be non-zero")
    return vector[0] / length, vector[1] / length


def _round(value: float) -> float:
    """Keep traces stable and compact without discarding meaningful precision."""
    return float(f"{value:.12g}")


_EXPRESSION_FUNCTIONS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "floor": math.floor,
    "ceil": math.ceil,
}
_EXPRESSION_CONSTANTS = {"pi": math.pi, "e": math.e}


class _SafeExpression:
    """Small arithmetic interpreter; deliberately does not use Python eval."""

    def __init__(self, source: Any, allowed_names: Iterable[str], name: str) -> None:
        self.source = str(source or "").strip()
        self.name = name
        if not self.source or len(self.source) > 512:
            raise DomainSimulationError(f"{name} must be a non-empty expression up to 512 characters")
        try:
            self.tree = ast.parse(self.source, mode="eval")
        except SyntaxError as exc:
            raise DomainSimulationError(f"invalid {name}: {exc.msg}") from exc
        nodes = list(ast.walk(self.tree))
        if len(nodes) > 128:
            raise DomainSimulationError(f"{name} is too complex")
        self.allowed_names = set(allowed_names) | set(_EXPRESSION_CONSTANTS)
        for node in nodes:
            if isinstance(node, ast.Name):
                if node.id not in self.allowed_names and node.id not in _EXPRESSION_FUNCTIONS:
                    raise DomainSimulationError(f"{name} references unknown name {node.id!r}")
            elif isinstance(
                node,
                (
                    ast.Expression, ast.Load, ast.Constant, ast.BinOp, ast.UnaryOp,
                    ast.BoolOp, ast.Compare, ast.IfExp, ast.Call, ast.Name,
                    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
                    ast.USub, ast.UAdd, ast.And, ast.Or, ast.Not,
                    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
                ),
            ):
                continue
            else:
                raise DomainSimulationError(
                    f"{name} contains blocked syntax {type(node).__name__}"
                )
        for node in nodes:
            if isinstance(node, ast.Call) and (
                not isinstance(node.func, ast.Name)
                or node.func.id not in _EXPRESSION_FUNCTIONS
                or node.keywords
            ):
                raise DomainSimulationError(f"{name} contains a blocked function call")

    def evaluate(self, values: Mapping[str, float]) -> float | bool:
        environment: Dict[str, Any] = {**_EXPRESSION_CONSTANTS, **values}

        def visit(node: ast.AST) -> Any:
            if isinstance(node, ast.Expression):
                return visit(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, bool):
                return node.value
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return _finite_number(node.value, self.name)
            if isinstance(node, ast.Name):
                return environment[node.id]
            if isinstance(node, ast.UnaryOp):
                operand = visit(node.operand)
                if isinstance(node.op, ast.USub):
                    return -operand
                if isinstance(node.op, ast.UAdd):
                    return +operand
                if isinstance(node.op, ast.Not):
                    return not operand
            if isinstance(node, ast.BinOp):
                left, right = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                if isinstance(node.op, ast.Pow):
                    if abs(right) > 100:
                        raise DomainSimulationError(
                            f"exponent in {self.name} exceeds the safe limit of 100"
                        )
                    return left ** right
                if isinstance(node.op, ast.Mod):
                    return left % right
            if isinstance(node, ast.BoolOp):
                values_inner = [bool(visit(value)) for value in node.values]
                return all(values_inner) if isinstance(node.op, ast.And) else any(values_inner)
            if isinstance(node, ast.Compare):
                left = visit(node.left)
                for operator, comparator in zip(node.ops, node.comparators):
                    right = visit(comparator)
                    if isinstance(operator, ast.Lt):
                        result = left < right
                    elif isinstance(operator, ast.LtE):
                        result = left <= right
                    elif isinstance(operator, ast.Gt):
                        result = left > right
                    elif isinstance(operator, ast.GtE):
                        result = left >= right
                    elif isinstance(operator, ast.Eq):
                        result = left == right
                    else:
                        result = left != right
                    if not result:
                        return False
                    left = right
                return True
            if isinstance(node, ast.IfExp):
                return visit(node.body) if bool(visit(node.test)) else visit(node.orelse)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                return _EXPRESSION_FUNCTIONS[node.func.id](*(visit(arg) for arg in node.args))
            raise DomainSimulationError(f"cannot execute blocked syntax in {self.name}")

        try:
            result = visit(self.tree)
        except (ArithmeticError, OverflowError, ValueError) as exc:
            raise DomainSimulationError(f"failed to evaluate {self.name}: {exc}") from exc
        if isinstance(result, bool):
            return result
        return _finite_number(result, self.name)
