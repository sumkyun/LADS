"""Public-only collocation adapter, shared by LADS and upstream SIGS pilot.

Restricted AST interpretation supports mathematical equations, never Python eval.
No private reference or scoring artifact is imported or read.
"""

from __future__ import annotations

import ast
import itertools
from pathlib import Path

import numpy as np
import sympy as sp

from combined_solution_v1.coefficient_fitting import CollocationResidualV1
from combined_solution_v1.condition_aware_solution import _evaluate_terms
from combined_solution_v1.safe_symbolic import parse_symbolic_expression

PUBLIC_KEYS = {
    "boundary_conditions",
    "dependent_variables",
    "domain",
    "independent_variables",
    "initial_conditions",
    "parameters",
    "pde",
}


def row_from_public(path):
    from tools.benchmark_contract import load_problem, read_json_strict

    row = read_json_strict(path)
    if not isinstance(row, dict):
        raise ValueError("public input must be a JSON object")
    if "schema_version" in row:
        row = load_problem(path)
    elif set(row) != PUBLIC_KEYS:
        raise ValueError("public schema mismatch")
    return {**row, "benchmark_id": Path(path).stem}


def math_expr(source, names, functions=None, *, evaluate=True):
    """Finite arithmetic AST; no eval, attributes, indexing, imports, or comprehensions."""
    tree = ast.parse(str(source).strip().replace("^", "**"), mode="eval")
    if len(list(ast.walk(tree))) > 4096:
        raise ValueError("expression too large")
    funcs = {
        n: getattr(sp, n)
        for n in (
            "sin",
            "cos",
            "tan",
            "exp",
            "log",
            "sqrt",
            "sinh",
            "cosh",
            "tanh",
            "sech",
            "Abs",
            "atan",
            "asin",
            "acos",
            "erf",
            "sinc",
        )
    }
    funcs.update(
        {"D": sp.Derivative, "Derivative": sp.Derivative, "diff": sp.diff, "Subs": sp.Subs}
    )
    funcs.update(functions or {})

    def convert(node):
        if isinstance(node, ast.Expression):
            return convert(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return sp.Rational(str(node.value))
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise ValueError("undeclared symbol " + node.id)
            return names[node.id]
        if isinstance(node, ast.Tuple):
            return tuple(convert(x) for x in node.elts)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            x = convert(node.operand)
            return sp.Mul(-1, x, evaluate=evaluate) if isinstance(node.op, ast.USub) else x
        if isinstance(node, ast.BinOp):
            a, b = convert(node.left), convert(node.right)
            if isinstance(node.op, ast.Add):
                return sp.Add(a, b, evaluate=evaluate)
            if isinstance(node.op, ast.Sub):
                return sp.Add(a, sp.Mul(-1, b, evaluate=evaluate), evaluate=evaluate)
            if isinstance(node.op, ast.Mult):
                return sp.Mul(a, b, evaluate=evaluate)
            if isinstance(node.op, ast.Div):
                return sp.Mul(a, sp.Pow(b, -1, evaluate=evaluate), evaluate=evaluate)
            if isinstance(node.op, ast.Pow):
                return sp.Pow(a, b, evaluate=evaluate)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            if node.func.id not in funcs:
                raise ValueError("unsupported mathematical function")
            arguments = tuple(convert(a) for a in node.args)
            if not evaluate:
                return funcs[node.func.id](*arguments, evaluate=False)
            return funcs[node.func.id](*arguments)
        raise ValueError("forbidden mathematical AST")

    return convert(tree)


def validation_spec(row):
    # Full public row retained so derivative ICs, periodic BCs and coupled fields are exact.
    return {"public_row": dict(row), "training_count": 3, "validation_count": 4}


def field_residual_terms(expressions, coefficients, spec, *, split):
    del coefficients
    row = spec["public_row"]
    if "schema_version" in row:
        from tools.benchmark_contract import validate_problem

        check = validate_problem({k: v for k, v in row.items() if k != "benchmark_id"})
        if check["status"] != "supported":
            raise ValueError(f"invalid public input: {check['errors']}")
    coordinates = {n: sp.Symbol(n, real=True) for n in row["independent_variables"]}
    variables = tuple(coordinates.values())
    names = {**coordinates, "pi": sp.pi, "E": sp.E}
    for n, v in row.get("parameters", {}).items():
        names[n] = math_expr(v, names)
    fields = {}
    for name, expression in expressions.items():
        expression = expression.xreplace(
            {s: coordinates[str(s)] for s in expression.free_symbols if str(s) in coordinates}
        )

        def field(*args, expr=expression):
            if len(args) != len(variables):
                raise ValueError("field argument arity differs")
            return expr.subs(dict(zip(variables, args, strict=True)), simultaneous=True)

        fields[name] = field
    if split not in {"training", "validation"}:
        raise ValueError("unknown collocation split")
    bounds = {n: tuple(float(math_expr(v, names)) for v in row["domain"][n]) for n in coordinates}

    def axis_points(variable, kind):
        lo, hi = bounds[str(variable)]
        policy = spec.get("collocation_grid")
        if policy is None:
            count = spec["training_count" if split == "training" else "validation_count"]
            return [lo + (hi - lo) * (k + 0.5) / count for k in range(count)]
        grid = policy[split]
        count = int(grid[str(variable) if kind == "pde" else "condition"])
        if grid["rule"] == "uniform_endpoints":
            return np.linspace(lo, hi, count).tolist()
        if grid["rule"] == "uniform_midpoints":
            return [lo + (hi - lo) * (k + 0.5) / count for k in range(count)]
        raise ValueError("unsupported collocation rule")

    def weight(kind):
        policy = spec.get("collocation_grid")
        return 1.0 if policy is None else float(policy["weights"][kind.removeprefix("condition_")])

    terms = []
    for kind, equations in [
        ("pde", row["pde"]),
        ("condition_initial", row["initial_conditions"]),
        ("condition_boundary", row["boundary_conditions"]),
    ]:
        for i, equation in enumerate(equations):
            if isinstance(equation, dict):
                if equation["kind"] == "sampled":
                    from scipy.interpolate import CubicSpline, RegularGridInterpolator

                    sample = tuple(coordinates[v] for v in equation["axis_order"])
                    sample_axes = [axis_points(v, kind) for v in sample]
                    locations = tuple(itertools.product(*sample_axes))
                    source_axes = [equation["axes"][v] for v in equation["axis_order"]]
                    if equation["interpolation"] == "natural_cubic":
                        interpolator = CubicSpline(
                            source_axes[0], equation["values"], bc_type="natural", extrapolate=False
                        )
                        observed = interpolator([x[0] for x in locations])
                    else:
                        interpolator = RegularGridInterpolator(
                            source_axes, equation["values"], method="linear", bounds_error=True
                        )
                        observed = interpolator(np.asarray(locations))
                    target = sp.Dummy("observed_condition", real=True)
                    residual = math_expr(equation["lhs"], names, fields).doit() - target
                    points = tuple(
                        (*point, float(value))
                        for point, value in zip(locations, observed, strict=True)
                    )
                    terms.append(
                        CollocationResidualV1(
                            f"{kind}_{i}", residual, (*sample, target), points, weight(kind)
                        )
                    )
                    continue
                equation = equation["equation"]
            if not isinstance(equation, str) or equation.count("=") != 1:
                raise ValueError("unsupported public condition representation")
            lhs, rhs = equation.split("=")
            residual = (math_expr(lhs, names, fields) - math_expr(rhs, names, fields)).doit()
            sample = (
                variables
                if kind == "pde"
                else tuple(x for x in variables if x in residual.free_symbols)
            )
            axes = [axis_points(x, kind) for x in sample]
            points = tuple(itertools.product(*axes)) if axes else ((),)
            terms.append(
                CollocationResidualV1(f"{kind}_{i}", residual, sample, points, weight(kind))
            )
    return tuple(terms)


def score_candidate(case, expr):
    row = case
    if isinstance(expr, str):
        if len(row["dependent_variables"]) != 1:
            raise ValueError("coupled fields must be supplied separately")
        expr = {row["dependent_variables"][0]: expr}
    if set(expr) != set(row["dependent_variables"]):
        raise ValueError("field census differs")
    names = {n: sp.Symbol(n, real=True) for n in row["independent_variables"]}
    names["pi"] = sp.pi
    expressions = {n: parse_symbolic_expression(str(e), names=names) for n, e in expr.items()}
    terms = field_residual_terms(expressions, (), validation_spec(row), split="validation")
    loss, details = _evaluate_terms(terms, sp.Add(*expressions.values()))
    return {"loss": float(loss), "term_losses": dict(details)}
