"""Canonical, dimension-agnostic public benchmark input/output contract.

This module defines serialization and public-input validity only.  A supported
problem is not thereby claimed to be representable by LADS or any other method.
"""

from __future__ import annotations

import ast
import json
import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp  # type: ignore[import-untyped]

from pde_ansatz_discovery.ir import ParserLimitsV1
from tools.pilot_public import PUBLIC_KEYS, math_expr

PROBLEM_SCHEMA = "cdr.pde.problem.v1"
SOLUTION_SCHEMA = "cdr.pde.solution.v1"
PROBLEM_KEYS = PUBLIC_KEYS | {"schema_version"}
CONDITION_GROUPS = ("initial_conditions", "boundary_conditions")
DERIVATIVE_NAMES = frozenset({"D", "Derivative", "diff"})
FORBIDDEN_CALLS = frozenset(
    {"compile", "eval", "exec", "getattr", "globals", "locals", "open", "setattr"}
)


def _fail(code: str, location: str, message: str) -> dict[str, Any]:
    return {
        "status": "invalid_public_input",
        "checks": {},
        "errors": [{"code": code, "location": location, "message": message}],
    }


def _positive_integer(node: ast.AST) -> int:
    if (
        not isinstance(node, ast.Constant)
        or isinstance(node.value, bool)
        or not isinstance(node.value, int)
        or node.value < 1
    ):
        raise ValueError("derivative multiplicity must be a positive integer")
    return node.value


def _derivative_specs(nodes: Sequence[ast.AST]) -> list[tuple[str, int]]:
    if not nodes:
        raise ValueError("derivative requires at least one variable")
    result: list[tuple[str, int]] = []
    index = 0
    while index < len(nodes):
        node = nodes[index]
        if isinstance(node, ast.Tuple):
            if len(node.elts) != 2 or not isinstance(node.elts[0], ast.Name):
                raise ValueError("derivative tuple must be (variable, positive integer)")
            name = node.elts[0].id
            count = _positive_integer(node.elts[1])
            index += 1
        elif isinstance(node, ast.Name):
            name = node.id
            count = 1
            if index + 1 < len(nodes) and isinstance(nodes[index + 1], ast.Constant):
                count = _positive_integer(nodes[index + 1])
                index += 1
            index += 1
        else:
            raise ValueError("derivative variable must be an identifier")
        if result and result[-1][0] == name:
            result[-1] = (name, result[-1][1] + count)
        else:
            result.append((name, count))
    return result


class _Canonicalizer(ast.NodeTransformer):
    """Reject non-mathematical Python AST and normalize derivative calls."""

    def generic_visit(self, node: ast.AST) -> ast.AST:
        allowed = (
            ast.Expression,
            ast.Constant,
            ast.Name,
            ast.Load,
            ast.Tuple,
            ast.UnaryOp,
            ast.UAdd,
            ast.USub,
            ast.BinOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.Call,
        )
        if not isinstance(node, allowed):
            raise ValueError(f"forbidden mathematical AST: {type(node).__name__}")
        return super().generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric literals are permitted")
        if isinstance(node.value, float) and not math.isfinite(node.value):
            raise ValueError("numeric literals must be finite")
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ValueError("calls must use an identifier and positional arguments")
        name = node.func.id
        if name.startswith("_") or name in FORBIDDEN_CALLS:
            raise ValueError("forbidden function name")
        if name in DERIVATIVE_NAMES:
            if len(node.args) < 2:
                raise ValueError("derivative requires an expression and variables")
            expression = self.visit(node.args[0])
            specs = _derivative_specs(node.args[1:])
            if (
                isinstance(expression, ast.Call)
                and isinstance(expression.func, ast.Name)
                and expression.func.id == "Derivative"
            ):
                specs = _derivative_specs(expression.args[1:]) + specs
                expression = expression.args[0]
                merged: list[tuple[str, int]] = []
                for axis, count in specs:
                    if merged and merged[-1][0] == axis:
                        merged[-1] = (axis, merged[-1][1] + count)
                    else:
                        merged.append((axis, count))
                specs = merged
            arguments: list[ast.expr] = [expression]  # type: ignore[list-item]
            for axis, count in specs:
                arguments.append(ast.Name(id=axis, ctx=ast.Load()))
                if count != 1:
                    arguments.append(ast.Constant(value=count))
            return ast.Call(
                func=ast.Name(id="Derivative", ctx=ast.Load()),
                args=arguments,
                keywords=[],
            )
        node.args = [self.visit(argument) for argument in node.args]  # type: ignore[list-item]
        return node


def _compact(source: str) -> str:
    """Remove insignificant whitespace produced by ``ast.unparse``."""

    tokens: list[str] = []
    previous = ""
    for character in source:
        if character.isspace():
            if previous and (previous.isalnum() or previous == "_"):
                tokens.append(" ")
            continue
        if tokens and tokens[-1] == " ":
            if not (character.isalnum() or character == "_"):
                tokens.pop()
            elif previous and not (previous.isalnum() or previous == "_"):
                tokens.pop()
        tokens.append(character)
        previous = character
    return "".join(tokens)


def _maximum_derivative_order(node: ast.AST) -> int:
    children = max(
        (_maximum_derivative_order(child) for child in ast.iter_child_nodes(node)),
        default=0,
    )
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Derivative"
    ):
        local = sum(count for _, count in _derivative_specs(node.args[1:]))
        inner = _maximum_derivative_order(node.args[0])
        return inner + local
    return children


def canonical_expression(text: str, *, fitted_output: bool = False) -> str:
    """Return a safe AST canonical form; no expression is executed."""

    limits = ParserLimitsV1()
    if not isinstance(text, str) or not text.strip():
        raise ValueError("expression must be a nonempty string")
    if len(text) > limits.max_characters:
        raise ValueError("expression exceeds character limit")
    if "^" in text:
        raise ValueError("powers must use **")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except RecursionError as exc:
        raise ValueError("expression exceeds AST depth limit") from exc
    except (SyntaxError, ValueError) as exc:
        raise ValueError("invalid arithmetic expression") from exc
    nodes = tuple(ast.walk(tree))
    if len(nodes) > limits.max_ast_nodes:
        raise ValueError("expression exceeds AST node limit")

    def depth(node: ast.AST) -> int:
        children = tuple(ast.iter_child_nodes(node))
        return 1 + max((depth(child) for child in children), default=0)

    if depth(tree) > limits.max_ast_depth:
        raise ValueError("expression exceeds AST depth limit")
    for node in nodes:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if isinstance(node.value, float) and not math.isfinite(node.value):
                raise ValueError("numeric literals must be finite")
            if len(str(abs(node.value))) > limits.max_literal_digits:
                raise ValueError("numeric literal exceeds digit limit")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):

            def direct_float(value: ast.AST) -> float | None:
                if isinstance(value, ast.Constant) and type(value.value) is float:
                    return value.value
                if (
                    isinstance(value, ast.UnaryOp)
                    and isinstance(value.op, (ast.UAdd, ast.USub))
                    and isinstance(value.operand, ast.Constant)
                    and type(value.operand.value) is float
                ):
                    return (
                        -value.operand.value
                        if isinstance(value.op, ast.USub)
                        else value.operand.value
                    )
                return None

            def rational(value: ast.AST) -> Fraction | None:
                if (
                    isinstance(value, ast.Constant)
                    and not isinstance(value.value, bool)
                    and isinstance(value.value, (int, float))
                ):
                    return Fraction(str(value.value))
                if isinstance(value, ast.UnaryOp) and isinstance(value.op, (ast.UAdd, ast.USub)):
                    operand = rational(value.operand)
                    if operand is None:
                        return None
                    return -operand if isinstance(value.op, ast.USub) else operand
                if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Div):
                    numerator, denominator = rational(value.left), rational(value.right)
                    if numerator is None or denominator in {None, 0}:
                        return None
                    return numerator / denominator
                return None

            fitted_float = direct_float(node.right)
            if fitted_float is not None:
                if not fitted_output and abs(fitted_float) > limits.max_power_magnitude:
                    raise ValueError("power magnitude exceeds limit")
            else:
                exponent = rational(node.right)
                if exponent is not None and (
                    abs(exponent.numerator) > limits.max_power_magnitude
                    or exponent.denominator > limits.max_power_magnitude
                ):
                    raise ValueError("power magnitude exceeds limit")
    normalized = _Canonicalizer().visit(tree)
    ast.fix_missing_locations(normalized)
    if _maximum_derivative_order(normalized) > limits.max_derivative_order:
        raise ValueError(f"derivative order exceeds {limits.max_derivative_order}")
    return _compact(ast.unparse(normalized))


def canonical_equation(text: str) -> str:
    """Canonicalize both sides of an equation containing exactly one ``=``."""

    if not isinstance(text, str) or text.count("=") != 1:
        raise ValueError("equation must contain exactly one equals")
    lhs, rhs = text.split("=", 1)
    return f"{canonical_expression(lhs)} = {canonical_expression(rhs)}"


def read_json_strict(path: str | Path) -> object:
    """Read JSON while rejecting duplicate keys and non-finite numbers."""

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def nonfinite(value: str) -> object:
        raise ValueError(f"non-finite JSON number: {value}")

    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid benchmark JSON: {exc}") from exc


def _numbers(value: object, *, location: str, minimum: int = 1) -> list[float]:
    if not isinstance(value, list) or len(value) < minimum:
        raise ValueError(f"{location} must contain at least {minimum} numbers")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{location} values must be finite numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{location} values must be finite numbers")
        result.append(number)
    return result


def _value_shape(value: object, shape: Sequence[int], *, location: str) -> None:
    if not shape:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{location} must contain finite numeric values")
        if not math.isfinite(float(value)):
            raise ValueError(f"{location} must contain finite numeric values")
        return
    if not isinstance(value, list) or len(value) != shape[0]:
        raise ValueError(f"{location} shape must match axis_order and axes")
    for item in value:
        _value_shape(item, shape[1:], location=location)


def _domain_numbers(row: Mapping[str, Any], axis: str) -> tuple[float, float]:
    parameters = {"pi": sp.pi, "E": sp.E}
    for name, value in row["parameters"].items():
        parameters[name] = math_expr(value, parameters)
    bounds = row["domain"][axis]
    return tuple(float(sp.N(math_expr(value, parameters))) for value in bounds)  # type: ignore[return-value]


def _sampled_condition(
    condition: object,
    row: Mapping[str, Any],
    *,
    location: str,
) -> str:
    expected = {"kind", "lhs", "axis_order", "axes", "values", "interpolation"}
    if not isinstance(condition, dict) or set(condition) != expected:
        raise ValueError(f"{location} sampled condition keys differ from the contract")
    lhs = canonical_expression(condition["lhs"])
    axis_order = condition["axis_order"]
    independent = row["independent_variables"]
    if (
        not isinstance(axis_order, list)
        or not axis_order
        or any(not isinstance(axis, str) or axis not in independent for axis in axis_order)
        or len(set(axis_order)) != len(axis_order)
    ):
        raise ValueError(f"{location}.axis_order must contain unique declared axes")
    axes = condition["axes"]
    if not isinstance(axes, dict) or set(axes) != set(axis_order):
        raise ValueError(f"{location}.axes must define exactly the axis_order axes")
    shape: list[int] = []
    for axis in axis_order:
        points = _numbers(axes[axis], location=f"{location}.axes.{axis}", minimum=2)
        if any(left >= right for left, right in zip(points, points[1:], strict=False)):
            raise ValueError(f"{location}.axes.{axis} must be strictly ordered")
        lower, upper = _domain_numbers(row, axis)
        if not (math.isclose(points[0], lower) and math.isclose(points[-1], upper)):
            raise ValueError(f"{location}.axes.{axis} must cover the entire declared domain")
        shape.append(len(points))
    _value_shape(condition["values"], shape, location=f"{location}.values")
    interpolation = condition["interpolation"]
    if interpolation not in {"linear", "natural_cubic"}:
        raise ValueError(f"{location}.interpolation is unsupported")
    if interpolation == "natural_cubic" and len(axis_order) != 1:
        raise ValueError("natural_cubic interpolation is one-dimensional only")

    names = {name: sp.Symbol(name, real=True) for name in independent}
    names.update({"pi": sp.pi, "E": sp.E})
    for name, value in row["parameters"].items():
        names[name] = math_expr(value, names)
    formal_fields: dict[str, Any] = {}
    arity = len(independent)
    for field in row["dependent_variables"]:
        function = sp.Function(field)

        def apply(*arguments: sp.Expr, _function: Any = function) -> sp.Expr:
            if len(arguments) != arity:
                raise ValueError("field argument arity differs")
            return _function(*arguments)

        formal_fields[field] = apply
    parsed = math_expr(lhs, names, formal_fields)
    free_axes = {str(symbol) for symbol in parsed.free_symbols} & set(independent)
    if free_axes != set(axis_order):
        raise ValueError(f"{location}.lhs varying axes must agree with axis_order")
    tree = ast.parse(lhs, mode="eval")
    called_fields = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in set(row["dependent_variables"])
    }
    if not called_fields:
        raise ValueError(f"{location}.lhs must contain a declared field")
    return f"{lhs} = 0"


def validate_problem(row: object) -> dict[str, Any]:
    """Validate the common public format without asserting method support."""

    if not isinstance(row, dict) or set(row) != PROBLEM_KEYS:
        return _fail("SCHEMA_ERROR", "$", "problem must contain exactly the common keys")
    if row.get("schema_version") != PROBLEM_SCHEMA:
        return _fail("SCHEMA_ERROR", "schema_version", "unsupported problem schema_version")
    try:
        independent = row["independent_variables"]
        parameters = row["parameters"]
        domain = row["domain"]
        if not isinstance(independent, list):
            raise ValueError("independent_variables must be a list")
        if not isinstance(parameters, dict) or any(
            not isinstance(value, str) for value in parameters.values()
        ):
            raise ValueError("parameter values must be strings")
        if not isinstance(domain, dict) or set(domain) != set(independent):
            raise ValueError("domain must define every and only declared axis")
        if any(
            not isinstance(bounds, list)
            or len(bounds) != 2
            or any(not isinstance(bound, str) for bound in bounds)
            for bounds in domain.values()
        ):
            raise ValueError("domain bounds must be two-item string lists")
    except (AttributeError, KeyError, TypeError, ValueError, RecursionError) as exc:
        return _fail("SCHEMA_ERROR", "$", str(exc))
    legacy = {key: row[key] for key in PUBLIC_KEYS}
    sampled_count = 0
    try:
        legacy["pde"] = [canonical_equation(item) for item in row["pde"]]
        for group in CONDITION_GROUPS:
            conditions = row[group]
            if not isinstance(conditions, list):
                raise ValueError(f"{group} must be a list")
            equations: list[str] = []
            for index, condition in enumerate(conditions):
                location = f"{group}[{index}]"
                if not isinstance(condition, dict) or "kind" not in condition:
                    raise ValueError(f"{location} must be a condition object")
                if condition["kind"] == "equation":
                    if set(condition) != {"kind", "equation"}:
                        raise ValueError(f"{location} equation condition keys differ")
                    equations.append(canonical_equation(condition["equation"]))
                elif condition["kind"] == "sampled":
                    equations.append(_sampled_condition(condition, row, location=location))
                    sampled_count += 1
                else:
                    raise ValueError(f"{location}.kind is unsupported")
            legacy[group] = equations
    except (KeyError, TypeError, ValueError) as exc:
        return _fail("CONDITION_PARSE_ERROR", "$", str(exc))

    # Lazy import prevents benchmark-contract users from creating an import cycle.
    from tools.pilot_preflight import validate_public_row

    try:
        result = validate_public_row(legacy)
    except (AttributeError, KeyError, TypeError, ValueError, RecursionError) as exc:
        return _fail("SCHEMA_ERROR", "$", str(exc))
    if result["status"] == "supported":
        result["checks"]["schema_version"] = PROBLEM_SCHEMA
        result["checks"]["sampled_conditions"] = sampled_count
        result["checks"]["method_representation"] = "not_checked"
    return result


def equation_problem(row: object) -> dict[str, Any]:
    """Project an equation-only common problem to the legacy seven-key adapter."""

    result = validate_problem(row)
    if result["status"] != "supported":
        raise ValueError(f"invalid public problem: {result['errors']}")
    assert isinstance(row, dict)
    legacy = {key: row[key] for key in PUBLIC_KEYS}
    legacy["pde"] = [canonical_equation(equation) for equation in row["pde"]]
    for group in CONDITION_GROUPS:
        conditions = row[group]
        if any(condition["kind"] == "sampled" for condition in conditions):
            raise ValueError("sampled-condition runner is unsupported by equation_problem")
        legacy[group] = [canonical_equation(condition["equation"]) for condition in conditions]
    return legacy


def load_problem(path: str | Path) -> dict[str, Any]:
    """Load and validate one versioned problem without legacy projection."""

    value = read_json_strict(path)
    result = validate_problem(value)
    if result["status"] != "supported" or not isinstance(value, dict):
        raise ValueError(f"invalid public problem: {result['errors']}")
    return value


def load_solution(
    path: str | Path,
    problem: object,
    *,
    expected_case_id: str,
) -> dict[str, Any]:
    """Load and validate one versioned solution against its public problem."""

    value = read_json_strict(path)
    validate_solution(value, problem, expected_case_id=expected_case_id)
    assert isinstance(value, dict)
    return value


def _exact_mapping(value: object, keys: set[str], *, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{location} keys differ from the solution contract")
    return value


def validate_solution(value: object, problem: object, *, expected_case_id: str) -> None:
    """Raise ``ValueError`` unless *value* satisfies the common output schema."""

    problem_result = validate_problem(problem)
    if problem_result["status"] != "supported" or not isinstance(problem, dict):
        raise ValueError("solution cannot be validated against an invalid problem")
    output = _exact_mapping(
        value,
        {"schema_version", "case_id", "status", "solution", "diagnostics", "validation"},
        location="$",
    )
    if output["schema_version"] != SOLUTION_SCHEMA:
        raise ValueError("unsupported solution schema_version")
    if not isinstance(expected_case_id, str) or not expected_case_id:
        raise ValueError("expected_case_id must be a nonempty string")
    if not isinstance(output["case_id"], str) or not output["case_id"]:
        raise ValueError("case_id must be a nonempty string")
    if output["case_id"] != expected_case_id:
        raise ValueError("case_id does not match expected_case_id")
    status = output["status"]
    if status not in {"candidate", "no_candidate", "failed"}:
        raise ValueError("solution status is unsupported")
    diagnostics = _exact_mapping(
        output["diagnostics"], {"category", "message"}, location="diagnostics"
    )
    if diagnostics["category"] not in {"none", "method", "infrastructure", "budget"}:
        raise ValueError("diagnostics.category is unsupported")
    if not isinstance(diagnostics["message"], str):
        raise ValueError("diagnostics.message must be a string")
    validation = _exact_mapping(output["validation"], {"status", "residual"}, location="validation")
    if validation["status"] not in {
        "not_checked",
        "passed",
        "residual_above_threshold",
        "unscorable",
    }:
        raise ValueError("validation.status is unsupported")
    residual = validation["residual"]
    if validation["status"] in {"not_checked", "unscorable"} and residual is not None:
        raise ValueError("not_checked and unscorable validation require a null residual")
    if validation["status"] in {"passed", "residual_above_threshold"} and residual is None:
        raise ValueError("checked validation requires a numeric residual")
    if residual is not None and (
        isinstance(residual, bool)
        or not isinstance(residual, (int, float))
        or not math.isfinite(float(residual))
        or float(residual) < 0
    ):
        raise ValueError("validation.residual must be finite, nonnegative, or null")

    solution = output["solution"]
    if status != "candidate":
        if solution is not None:
            raise ValueError("no_candidate and failed outputs must not contain a solution")
        if validation["status"] == "passed":
            raise ValueError("no_candidate and failed outputs cannot have passed validation")
        return
    if solution is None:
        raise ValueError("candidate output requires a solution")
    solution = _exact_mapping(
        solution,
        {"kind", "fields"}
        if isinstance(solution, dict) and solution.get("kind") == "expressions"
        else {"kind", "axis_order", "axes", "fields"},
        location="solution",
    )
    fields = solution["fields"]
    declared_fields = problem["dependent_variables"]
    if not isinstance(fields, dict) or set(fields) != set(declared_fields):
        raise ValueError("solution field coverage must exactly match the problem")
    if solution["kind"] == "expressions":
        names = {name: sp.Symbol(name, real=True) for name in problem["independent_variables"]}
        names.update({"pi": sp.pi, "E": sp.E})
        for name, raw in problem["parameters"].items():
            names[name] = math_expr(raw, names)
        for expression in fields.values():
            canonical = canonical_expression(expression, fitted_output=True)
            expression_tree = ast.parse(canonical, mode="eval")
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in DERIVATIVE_NAMES | {"Integral", "Sum"}
                for node in ast.walk(expression_tree)
            ):
                raise ValueError("solution expressions must not contain unresolved operators")
            # Validate serialized arithmetic without evaluating potentially large powers.
            math_expr(canonical, names, evaluate=False)
        return
    if solution["kind"] != "grid":
        raise ValueError("solution.kind is unsupported")
    axis_order = solution["axis_order"]
    if axis_order != problem["independent_variables"]:
        raise ValueError("grid axis_order must equal the declared coordinate order")
    axes = solution["axes"]
    if not isinstance(axes, dict) or set(axes) != set(axis_order):
        raise ValueError("grid axes must cover every declared coordinate")
    shape: list[int] = []
    for axis in axis_order:
        points = _numbers(axes[axis], location=f"solution.axes.{axis}", minimum=2)
        if any(left >= right for left, right in zip(points, points[1:], strict=False)):
            raise ValueError(f"solution.axes.{axis} must be strictly ordered")
        lower, upper = _domain_numbers(problem, axis)
        if not (math.isclose(points[0], lower) and math.isclose(points[-1], upper)):
            raise ValueError(f"solution.axes.{axis} must cover the entire declared domain")
        shape.append(len(points))
    for field, nested in fields.items():
        _value_shape(nested, shape, location=f"solution.fields.{field}")
