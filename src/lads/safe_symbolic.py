"""Non-evaluating parser for the small symbolic grammar used by Ansatz templates."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from typing import Any

import sympy as sp

_BUILTIN_FUNCTIONS: Mapping[str, Any] = {
    "Abs": sp.Abs,
    "atan": sp.atan,
    "cos": sp.cos,
    "cosh": sp.cosh,
    "exp": sp.exp,
    "erf": sp.erf,
    "log": sp.log,
    "sech": sp.sech,
    "sin": sp.sin,
    "sinh": sp.sinh,
    "sqrt": sp.sqrt,
    "tan": sp.tan,
    "tanh": sp.tanh,
}
SUPPORTED_ELEMENTARY_FUNCTIONS = tuple(_BUILTIN_FUNCTIONS)
MAX_SYMBOLIC_EXPRESSION_LENGTH = 65_536
MAX_SYMBOLIC_AST_NODES = 8_192
MAX_SYMBOLIC_AST_DEPTH = 256
_COEFFICIENT_NAME = re.compile(r"(?:a|alpha|b|beta|c|d|gamma|k|kappa|lambda|mu|nu|omega)(?:_?\d+)?")


class SafeSymbolicError(ValueError):
    """The expression is outside the deliberately small symbolic grammar."""


def is_abstract_coefficient_name(name: str) -> bool:
    """Return whether an undeclared scalar uses the neutral coefficient convention."""

    return _COEFFICIENT_NAME.fullmatch(name) is not None


def _check_tree(tree: ast.AST) -> None:
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_SYMBOLIC_AST_NODES:
        raise SafeSymbolicError("symbolic expression is too complex")

    def visit(node: ast.AST, depth: int) -> None:
        if depth > MAX_SYMBOLIC_AST_DEPTH:
            raise SafeSymbolicError("symbolic expression is too deep")
        for child in ast.iter_child_nodes(node):
            visit(child, depth + 1)

    visit(tree, 0)


def parse_symbolic_expression(
    source: str,
    *,
    names: Mapping[str, object] | None = None,
    functions: Mapping[str, object] | None = None,
) -> sp.Expr:
    """Parse arithmetic without ``eval``, attributes, indexing, or arbitrary calls.

    Unknown identifiers become inert SymPy symbols. Unknown function names are
    accepted only when they begin with an uppercase letter, which is the repository's
    abstract-profile convention.
    """

    if (
        not isinstance(source, str)
        or not source.strip()
        or len(source) > MAX_SYMBOLIC_EXPRESSION_LENGTH
    ):
        raise SafeSymbolicError("symbolic expression text is invalid")
    try:
        tree = ast.parse(source.replace("^", "**"), mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise SafeSymbolicError("symbolic expression syntax is invalid") from exc
    _check_tree(tree)
    name_map = dict(names or {})
    function_map = dict(functions or {})

    def convert(node: ast.AST) -> sp.Expr:
        if isinstance(node, ast.Expression):
            return convert(node.body)
        if isinstance(node, ast.Constant):
            constant = node.value
            if isinstance(constant, bool) or not isinstance(constant, (int, float)):
                raise SafeSymbolicError("only real numeric literals are allowed")
            if not (-(10**100) <= constant <= 10**100):
                raise SafeSymbolicError("numeric literal is too large")
            return sp.Integer(constant) if isinstance(constant, int) else sp.Rational(str(constant))
        if isinstance(node, ast.Name):
            if "__" in node.id:
                raise SafeSymbolicError("dunder identifiers are forbidden")
            symbolic_value = name_map.get(node.id)
            if symbolic_value is None:
                return sp.Symbol(node.id)
            if not isinstance(symbolic_value, sp.Expr):
                raise SafeSymbolicError("identifier is not a symbolic value")
            return symbolic_value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = convert(node.operand)
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.BinOp):
            left = convert(node.left)
            right = convert(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                small_rational = bool(
                    right.is_Rational and abs(int(right.p)) <= 64 and abs(int(right.q)) <= 64
                )
                signed_symbol = bool(right.is_Symbol or (-right).is_Symbol)
                if not (small_rational or signed_symbol):
                    raise SafeSymbolicError(
                        "only small rational or scalar-parameter powers are allowed"
                    )
                return left**right
            raise SafeSymbolicError("symbolic operator is forbidden")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise SafeSymbolicError("only direct positional function calls are allowed")
            function_name = node.func.id
            if "__" in function_name:
                raise SafeSymbolicError("dunder function names are forbidden")
            if not 1 <= len(node.args) <= 16:
                raise SafeSymbolicError("symbolic function arity is invalid")
            function: Any = function_map.get(function_name, _BUILTIN_FUNCTIONS.get(function_name))
            if function is None:
                if not function_name[0].isupper():
                    raise SafeSymbolicError("unknown lowercase function is forbidden")
                function = sp.Function(function_name)
            try:
                return function(*(convert(argument) for argument in node.args))
            except (TypeError, ValueError) as exc:
                raise SafeSymbolicError("symbolic function application is invalid") from exc
        raise SafeSymbolicError(f"forbidden symbolic syntax: {type(node).__name__}")

    return convert(tree)


__all__ = [
    "MAX_SYMBOLIC_AST_DEPTH",
    "MAX_SYMBOLIC_AST_NODES",
    "MAX_SYMBOLIC_EXPRESSION_LENGTH",
    "SUPPORTED_ELEMENTARY_FUNCTIONS",
    "SafeSymbolicError",
    "is_abstract_coefficient_name",
    "parse_symbolic_expression",
]
