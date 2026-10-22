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
    "log": sp.log,
    "sech": sp.sech,
    "sin": sp.sin,
    "sinh": sp.sinh,
    "sqrt": sp.sqrt,
    "tan": sp.tan,
    "tanh": sp.tanh,
}
SUPPORTED_ELEMENTARY_FUNCTIONS = tuple(_BUILTIN_FUNCTIONS)
_MAX_EXPRESSION_LENGTH = 4096
_MAX_AST_NODES = 512
_MAX_AST_DEPTH = 64
_COEFFICIENT_NAME = re.compile(r"(?:a|alpha|b|beta|c|d|gamma|k|kappa|lambda|mu|nu|omega)(?:_?\d+)?")


class SafeSymbolicError(ValueError):
    """The expression is outside the deliberately small symbolic grammar."""


def is_abstract_coefficient_name(name: str) -> bool:
    """Return whether an undeclared scalar uses the neutral coefficient convention."""

    return _COEFFICIENT_NAME.fullmatch(name) is not None


def _check_tree(tree: ast.AST) -> None:
    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_AST_NODES:
        raise SafeSymbolicError("symbolic expression is too complex")

    def visit(node: ast.AST, depth: int) -> None:
        if depth > _MAX_AST_DEPTH:
            raise SafeSymbolicError("symbolic expression is too deep")
        for child in ast.iter_child_nodes(node):
            visit(child, depth + 1)

    visit(tree, 0)


def parse_symbolic_expression(
    source: str,
    *,
    names: Mapping[str, object] | None = None,
    functions: Mapping[str, object] | None = None,
    allow_derivatives: bool = False,
    derivative_variables: tuple[str, ...] | list[str] = (),
) -> sp.Expr:
    """Parse arithmetic without ``eval``, attributes, indexing, or arbitrary calls.

    Unknown identifiers become inert SymPy symbols. Unknown function names are
    accepted only when they begin with an uppercase letter, which is the repository's
    abstract-profile convention. Derivative operators require explicit opt-in and declared
    coordinates; they remain unevaluated. Single-argument D remains an open profile, while
    D with derivative arguments and Derivative/diff are reserved operators.
    """

    if not isinstance(source, str) or not source.strip() or len(source) > _MAX_EXPRESSION_LENGTH:
        raise SafeSymbolicError("symbolic expression text is invalid")
    try:
        tree = ast.parse(source.replace("^", "**"), mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise SafeSymbolicError("symbolic expression syntax is invalid") from exc
    _check_tree(tree)
    name_map = dict(names or {})
    function_map = dict(functions or {})

    def derivative_call(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and (
                node.func.id in {"Derivative", "diff"}
                or node.func.id == "D"
                and len(node.args) >= 2
            )
        )

    def orders(node: ast.Call) -> list[tuple[sp.Symbol, int]]:
        result = []
        args = list(node.args[1:])
        while args:
            variable = args.pop(0)
            if isinstance(variable, ast.Tuple):
                if len(variable.elts) != 2:
                    raise SafeSymbolicError("derivative tuple must have two elements")
                variable, count = variable.elts
            elif args and isinstance(args[0], (ast.Constant, ast.UnaryOp)):
                count = args.pop(0)
            else:
                count = ast.Constant(value=1)
            if not isinstance(variable, ast.Name) or variable.id not in derivative_variables:
                raise SafeSymbolicError("derivative variable must be declared")
            if (
                not isinstance(count, ast.Constant)
                or type(count.value) is not int
                or count.value <= 0
            ):
                raise SafeSymbolicError("derivative order must be positive integer literal")
            symbol = name_map.get(variable.id, sp.Symbol(variable.id))
            if not isinstance(symbol, sp.Symbol):
                raise SafeSymbolicError("derivative coordinate binding must be a symbol")
            result.append((symbol, count.value))
        return result

    def cumulative_order(node: ast.AST) -> int:
        if derivative_call(node):
            assert isinstance(node, ast.Call)
            if len(node.args) < 2:
                raise SafeSymbolicError("derivative requires variable")
            return sum(count for _, count in orders(node)) + cumulative_order(node.args[0])
        return max((cumulative_order(child) for child in ast.iter_child_nodes(node)), default=0)

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
            if derivative_call(node):
                if not allow_derivatives or len(node.args) < 2:
                    raise SafeSymbolicError("derivatives require explicit opt-in")
                derivative_orders = orders(node)
                total = cumulative_order(node)
                if total > 8:
                    raise SafeSymbolicError("cumulative derivative order exceeds eight")
                return sp.Derivative(convert(node.args[0]), *derivative_orders, evaluate=False)
            function: Any = function_map.get(function_name, _BUILTIN_FUNCTIONS.get(function_name))
            if function is None:
                if not function_name[0].isupper():
                    raise SafeSymbolicError("unknown lowercase function is forbidden")
                function = sp.Function(function_name)
            if (
                not isinstance(function, sp.FunctionClass)
                and function not in _BUILTIN_FUNCTIONS.values()
            ):
                raise SafeSymbolicError("function binding is not a symbolic function")
            try:
                return function(*(convert(argument) for argument in node.args))
            except (TypeError, ValueError) as exc:
                raise SafeSymbolicError("symbolic function application is invalid") from exc
        raise SafeSymbolicError(f"forbidden symbolic syntax: {type(node).__name__}")

    return convert(tree)


__all__ = [
    "SUPPORTED_ELEMENTARY_FUNCTIONS",
    "SafeSymbolicError",
    "is_abstract_coefficient_name",
    "parse_symbolic_expression",
]
