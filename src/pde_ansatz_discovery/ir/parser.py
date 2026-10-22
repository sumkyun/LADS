"""Allowlisted Python-AST parser that produces project expression IR without evaluation."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from ..models import DiscoveryContractError, DiscoveryProblemInputV1
from ..serialization import canonical_sha256
from .expressions import (
    ExpressionIRV1,
    ExpressionKindV1,
    SymbolRoleV1,
    derivative,
    function,
    jet,
    rational,
    symbol,
)
from .normalizer import add, divide, mul, negate, normalize, power, subtract

SAFE_FUNCTIONS_V1 = frozenset(
    {
        "Abs",
        "atan",
        "cos",
        "cosh",
        "exp",
        "log",
        "sech",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    }
)
DERIVATIVE_CALLS_V1 = frozenset({"D", "Derivative", "diff"})


class PDEParseError(DiscoveryContractError):
    pass


@dataclass(frozen=True)
class ParserLimitsV1:
    max_characters: int = 10_000
    max_ast_nodes: int = 5_000
    max_ast_depth: int = 64
    max_literal_digits: int = 128
    max_derivative_order: int = 8
    max_power_magnitude: int = 16

    schema_version: ClassVar[str] = "ParserLimitsV1"

    def __post_init__(self) -> None:
        for name in (
            "max_characters",
            "max_ast_nodes",
            "max_ast_depth",
            "max_literal_digits",
            "max_derivative_order",
            "max_power_magnitude",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise PDEParseError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class ParserContextV1:
    dependent_variables: tuple[str, ...]
    independent_variables: tuple[str, ...]
    parameters: tuple[str, ...] = ()
    declared_functions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        groups = tuple(tuple(group) for group in (self.dependent_variables, self.independent_variables, self.parameters, self.declared_functions))
        for group in groups:
            if any(not isinstance(item, str) or not item for item in group):
                raise PDEParseError("parser context names must be nonempty strings")
            if len(set(group)) != len(group):
                raise PDEParseError("parser context names must be unique within each role")
        if set(self.dependent_variables) & set(self.independent_variables):
            raise PDEParseError("dependent and independent variables overlap")

    @classmethod
    def from_problem(cls, problem: DiscoveryProblemInputV1) -> "ParserContextV1":
        functions = tuple(
            name
            for name, description in problem.parameters.items()
            if isinstance(description, Mapping)
            and description.get("kind") in {"function", "coefficient_function"}
        )
        return cls(
            problem.dependent_variables,
            problem.independent_variables,
            tuple(problem.parameters),
            functions,
        )


@dataclass(frozen=True)
class ParsedEquationV1:
    source: str
    residual: ExpressionIRV1
    schema_version: ClassVar[str] = "ParsedEquationV1"

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source:
            raise PDEParseError("equation source must be nonempty")
        if not isinstance(self.residual, ExpressionIRV1):
            raise PDEParseError("equation residual must be expression IR")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "residual": self.residual.to_mapping(),
        }

    @property
    def canonical_hash(self) -> str:
        return canonical_sha256({"schema_version": self.schema_version, "residual": self.residual.to_mapping()})

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ParsedEquationV1":
        if set(value) != {"schema_version", "source", "residual"}:
            raise PDEParseError("parsed equation fields differ")
        if value["schema_version"] != cls.schema_version:
            raise PDEParseError("unsupported parsed equation schema")
        return cls(value["source"], ExpressionIRV1.from_mapping(value["residual"]))


@dataclass(frozen=True)
class ParsedPDEProblemV1:
    equations: tuple[ParsedEquationV1, ...]
    context: ParserContextV1
    input_hash: str
    schema_version: ClassVar[str] = "ParsedPDEProblemV1"

    def __post_init__(self) -> None:
        if not self.equations or any(not isinstance(item, ParsedEquationV1) for item in self.equations):
            raise PDEParseError("parsed PDE requires typed equations")
        if not isinstance(self.context, ParserContextV1):
            raise PDEParseError("parsed PDE context must be typed")


def _exceeds_depth(node: ast.AST, maximum: int) -> bool:
    pending = [(node, 1)]
    while pending:
        current, depth = pending.pop()
        if depth > maximum:
            return True
        pending.extend(
            (child, depth + 1) for child in ast.iter_child_nodes(current)
        )
    return False


def _split_equation(source: str) -> tuple[str, str | None]:
    depth = 0
    positions: list[int] = []
    for index, character in enumerate(source):
        if character in "([":
            depth += 1
        elif character in ")]":
            depth -= 1
            if depth < 0:
                raise PDEParseError("unbalanced delimiters")
        elif character == "=" and depth == 0:
            before = source[index - 1] if index else ""
            after = source[index + 1] if index + 1 < len(source) else ""
            if before in "<>!:=" or after == "=":
                raise PDEParseError("comparisons are outside the PDE grammar")
            positions.append(index)
    if depth != 0:
        raise PDEParseError("unbalanced delimiters")
    if len(positions) > 1:
        raise PDEParseError("an equation must contain at most one top-level equals")
    if not positions:
        return source, None
    index = positions[0]
    if not source[:index].strip() or not source[index + 1 :].strip():
        raise PDEParseError("equation sides must be nonempty")
    return source[:index], source[index + 1 :]


def _decode_derivative_suffix(suffix: str, variables: tuple[str, ...]) -> tuple[str, ...] | None:
    if not suffix:
        return None
    ordered_variables = tuple(sorted(variables, key=lambda item: (-len(item), item)))
    solution_counts = [0] * (len(suffix) + 1)
    solution_counts[-1] = 1

    # Count suffix decodings bottom-up and saturate at two. Enumerating every
    # segmentation is exponential when names overlap (for example ``x`` and
    # ``xx``), while callers only need to distinguish zero, one, or many.
    for start in range(len(suffix) - 1, -1, -1):
        count = 0
        for variable in ordered_variables:
            end = start + len(variable)
            if suffix.startswith(variable, start):
                count += solution_counts[end]
                if count > 1:
                    count = 2
                    break
        solution_counts[start] = count

    if solution_counts[0] != 1:
        return None

    parts: list[str] = []
    start = 0
    while start < len(suffix):
        match = next(
            (
                variable
                for variable in ordered_variables
                if suffix.startswith(variable, start)
                and solution_counts[start + len(variable)] == 1
            ),
            None,
        )
        if match is None:
            return None
        parts.append(match)
        start += len(match)
    return tuple(parts)


class _IRBuilder(ast.NodeVisitor):
    def __init__(self, context: ParserContextV1, limits: ParserLimitsV1) -> None:
        self.context = context
        self.limits = limits

    def generic_visit(self, node: ast.AST):
        raise PDEParseError(f"unsupported syntax: {type(node).__name__}")

    def visit_Constant(self, node: ast.Constant) -> ExpressionIRV1:
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise PDEParseError("only exact integer literals are allowed")
        if len(str(abs(node.value))) > self.limits.max_literal_digits:
            raise PDEParseError("integer literal exceeds digit limit")
        return rational(node.value)

    def visit_Name(self, node: ast.Name) -> ExpressionIRV1:
        name = node.id
        if name in self.context.dependent_variables:
            return jet(name)
        if name in self.context.independent_variables:
            return symbol(name, SymbolRoleV1.INDEPENDENT)
        if name in self.context.parameters:
            return symbol(name, SymbolRoleV1.PARAMETER)
        # Exact named constants use existing safe function IR, never float literals.
        if name == "pi":
            return mul(rational(4), function("atan", rational(1)))
        if name == "E":
            return function("exp", rational(1))
        if "_" in name:
            field, suffix = name.split("_", 1)
            if field in self.context.dependent_variables:
                variables = _decode_derivative_suffix(suffix, self.context.independent_variables)
                if variables is None:
                    raise PDEParseError(f"ambiguous or invalid derivative shorthand: {name}")
                if len(variables) > self.limits.max_derivative_order:
                    raise PDEParseError("derivative order exceeds limit")
                return jet(field, variables)
        raise PDEParseError(f"unknown identifier: {name}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ExpressionIRV1:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return negate(operand)
        raise PDEParseError("unsupported unary operator")

    def visit_BinOp(self, node: ast.BinOp) -> ExpressionIRV1:
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return add(lhs, rhs)
        if isinstance(node.op, ast.Sub):
            return subtract(lhs, rhs)
        if isinstance(node.op, ast.Mult):
            return mul(lhs, rhs)
        if isinstance(node.op, ast.Div):
            return divide(lhs, rhs)
        if isinstance(node.op, ast.Pow):
            exponent = rhs
            if exponent.kind is ExpressionKindV1.RATIONAL:
                if (
                    abs(exponent.numerator) > self.limits.max_power_magnitude
                    or exponent.denominator > self.limits.max_power_magnitude
                ):
                    raise PDEParseError("power magnitude exceeds limit")
            elif not (
                exponent.kind is ExpressionKindV1.SYMBOL
                and exponent.role is SymbolRoleV1.PARAMETER
            ):
                raise PDEParseError("powers require a bounded rational or declared parameter exponent")
            return power(lhs, rhs)
        raise PDEParseError("unsupported binary operator")

    def visit_Call(self, node: ast.Call) -> ExpressionIRV1:
        if node.keywords or not isinstance(node.func, ast.Name):
            raise PDEParseError("only positional calls to declared functions are allowed")
        name = node.func.id
        if name in DERIVATIVE_CALLS_V1:
            return self._derivative_call(node)
        if name in self.context.dependent_variables:
            expected = tuple(self.context.independent_variables)
            actual: list[str] = []
            for argument in node.args:
                if not isinstance(argument, ast.Name):
                    raise PDEParseError("dependent field arguments must be independent variables")
                actual.append(argument.id)
            if tuple(actual) != expected:
                raise PDEParseError("dependent field arguments must match declared independent variables")
            return jet(name)
        if name not in SAFE_FUNCTIONS_V1 and name not in self.context.declared_functions:
            raise PDEParseError(f"call to undeclared function: {name}")
        if not node.args:
            raise PDEParseError("function calls require arguments")
        return function(name, *(self.visit(argument) for argument in node.args))

    def _derivative_call(self, node: ast.Call) -> ExpressionIRV1:
        if len(node.args) < 2:
            raise PDEParseError("derivative call requires expression and variables")
        expression = self.visit(node.args[0])
        specifications: list[tuple[str, int]] = []
        local_order = 0
        arguments = node.args[1:]
        index = 0
        while index < len(arguments):
            argument = arguments[index]
            if isinstance(argument, ast.Tuple):
                variable, count = self._tuple_derivative_specification(argument)
                index += 1
            elif isinstance(argument, ast.Name):
                variable = self._declared_derivative_variable(argument)
                count = 1
                if index + 1 < len(arguments) and isinstance(
                    arguments[index + 1], ast.Constant
                ):
                    count = self._derivative_multiplicity(arguments[index + 1])
                    index += 1
                index += 1
            else:
                raise PDEParseError("invalid derivative variable specification")

            # Bound multiplicities before materializing repeated variables.
            if count > self.limits.max_derivative_order - local_order:
                raise PDEParseError("derivative order exceeds limit")
            local_order += count
            specifications.append((variable, count))

        if self._maximum_derivative_order(expression) > (
            self.limits.max_derivative_order - local_order
        ):
            raise PDEParseError("derivative order exceeds limit")

        variables = tuple(
            variable
            for variable, count in specifications
            for _ in range(count)
        )
        if expression.kind is ExpressionKindV1.DERIVATIVE:
            return derivative(
                expression.operands[0],
                expression.derivative_variables + variables,
            )
        return derivative(expression, variables)

    def _declared_derivative_variable(self, node: ast.Name) -> str:
        if node.id not in self.context.independent_variables:
            raise PDEParseError("invalid derivative variable specification")
        return node.id

    def _derivative_multiplicity(self, node: ast.AST) -> int:
        if (
            not isinstance(node, ast.Constant)
            or isinstance(node.value, bool)
            or not isinstance(node.value, int)
        ):
            raise PDEParseError("invalid derivative multiplicity")
        if node.value < 1:
            raise PDEParseError("derivative multiplicity must be positive")
        return node.value

    def _tuple_derivative_specification(self, node: ast.Tuple) -> tuple[str, int]:
        if len(node.elts) != 2 or not isinstance(node.elts[0], ast.Name):
            raise PDEParseError("invalid derivative variable specification")
        variable = self._declared_derivative_variable(node.elts[0])
        try:
            count = self._derivative_multiplicity(node.elts[1])
        except PDEParseError as exc:
            raise PDEParseError("invalid derivative variable specification") from exc
        return variable, count

    def _maximum_derivative_order(self, expression: ExpressionIRV1) -> int:
        if expression.kind is ExpressionKindV1.JET and expression.jet is not None:
            return len(expression.jet.multi_index.variables)
        nested = max(
            (self._maximum_derivative_order(operand) for operand in expression.operands),
            default=0,
        )
        if expression.kind is ExpressionKindV1.DERIVATIVE:
            return len(expression.derivative_variables) + nested
        return nested


def parse_expression(
    source: str,
    context: ParserContextV1,
    limits: ParserLimitsV1 | None = None,
) -> ExpressionIRV1:
    limits = limits or ParserLimitsV1()
    if not isinstance(source, str) or not source.strip():
        raise PDEParseError("expression source must be nonempty")
    if len(source) > limits.max_characters:
        raise PDEParseError("expression exceeds character limit")
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except RecursionError as exc:
        raise PDEParseError("expression exceeds AST depth limit") from exc
    except (SyntaxError, ValueError) as exc:
        raise PDEParseError(f"invalid expression syntax: {exc.msg if isinstance(exc, SyntaxError) else exc}") from exc
    nodes = tuple(ast.walk(tree))
    if len(nodes) > limits.max_ast_nodes:
        raise PDEParseError("expression exceeds AST node limit")
    if _exceeds_depth(tree, limits.max_ast_depth):
        raise PDEParseError("expression exceeds AST depth limit")
    return normalize(_IRBuilder(context, limits).visit(tree.body))


def parse_equation(
    source: str,
    context: ParserContextV1,
    limits: ParserLimitsV1 | None = None,
) -> ParsedEquationV1:
    lhs_source, rhs_source = _split_equation(source)
    lhs = parse_expression(lhs_source, context, limits)
    residual = lhs if rhs_source is None else subtract(lhs, parse_expression(rhs_source, context, limits))
    return ParsedEquationV1(source.strip(), normalize(residual))


def parse_problem(
    problem: DiscoveryProblemInputV1,
    limits: ParserLimitsV1 | None = None,
) -> ParsedPDEProblemV1:
    if not isinstance(problem, DiscoveryProblemInputV1):
        raise PDEParseError("problem must be DiscoveryProblemInputV1")
    context = ParserContextV1.from_problem(problem)
    return ParsedPDEProblemV1(
        tuple(parse_equation(source, context, limits) for source in problem.pde),
        context,
        problem.input_hash,
    )


__all__ = [
    "DERIVATIVE_CALLS_V1",
    "PDEParseError",
    "ParsedEquationV1",
    "ParsedPDEProblemV1",
    "ParserContextV1",
    "ParserLimitsV1",
    "SAFE_FUNCTIONS_V1",
    "parse_equation",
    "parse_expression",
    "parse_problem",
]
