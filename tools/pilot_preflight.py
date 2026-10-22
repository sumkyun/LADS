"""Public-only input validation gate for the fixed Ansatz pilot selection.

This module validates input syntax and frozen hashes before any provider is
constructed. It never imports or reads candidates, predictions, scores, or
private reference artifacts.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import sympy as sp  # type: ignore[import-untyped]

from pde_ansatz_discovery.ir import (  # type: ignore[import-untyped]
    ParserContextV1,
    ParserLimitsV1,
    parse_equation,
)
from tools.pilot_public import PUBLIC_KEYS, math_expr

MAX_DERIVATIVE_ORDER = 8
DERIVATIVE_NAMES = frozenset({"D", "Derivative", "diff"})
RESERVED_NAMES = frozenset(
    {
        "Abs",
        "D",
        "Derivative",
        "E",
        "Subs",
        "acos",
        "asin",
        "atan",
        "cos",
        "cosh",
        "diff",
        "erf",
        "exp",
        "log",
        "pi",
        "sech",
        "sin",
        "sinc",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    }
)


class PreflightInputError(ValueError):
    """A classified public-input contract failure."""

    def __init__(self, code: str, message: str, location: str) -> None:
        super().__init__(message)
        self.code = code
        self.location = location

    def to_mapping(self) -> dict[str, str]:
        return {"code": self.code, "location": self.location, "message": str(self)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _name_list(value: object, *, location: str, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise PreflightInputError("DECLARATION_ERROR", "declaration must be a list", location)
    names = tuple(value)
    if any(not isinstance(name, str) or not name.isidentifier() for name in names):
        raise PreflightInputError(
            "DECLARATION_ERROR", "declarations must be valid identifiers", location
        )
    if len(set(names)) != len(names):
        raise PreflightInputError("DECLARATION_ERROR", "declarations must be unique", location)
    return names


def _positive_order(node: ast.AST, *, location: str) -> int:
    if (
        not isinstance(node, ast.Constant)
        or isinstance(node.value, bool)
        or not isinstance(node.value, int)
        or node.value < 1
    ):
        raise PreflightInputError(
            "DERIVATIVE_SYNTAX_ERROR",
            "derivative multiplicity must be a positive integer",
            location,
        )
    return node.value


def _derivative_specs(
    arguments: Sequence[ast.AST],
    independent: frozenset[str],
    *,
    location: str,
) -> int:
    if not arguments:
        raise PreflightInputError(
            "DERIVATIVE_SYNTAX_ERROR", "derivative call requires variables", location
        )
    order = 0
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if isinstance(argument, ast.Tuple):
            if (
                len(argument.elts) != 2
                or not isinstance(argument.elts[0], ast.Name)
                or argument.elts[0].id not in independent
            ):
                raise PreflightInputError(
                    "DERIVATIVE_SYNTAX_ERROR",
                    "derivative tuple must be (declared_variable, positive_integer)",
                    location,
                )
            count = _positive_order(argument.elts[1], location=location)
            index += 1
        elif isinstance(argument, ast.Name) and argument.id in independent:
            count = 1
            if index + 1 < len(arguments) and isinstance(arguments[index + 1], ast.Constant):
                count = _positive_order(arguments[index + 1], location=location)
                index += 1
            index += 1
        else:
            raise PreflightInputError(
                "DERIVATIVE_SYNTAX_ERROR",
                "derivative variable must be a declared independent variable",
                location,
            )
        if count > MAX_DERIVATIVE_ORDER - order:
            raise PreflightInputError(
                "DERIVATIVE_SYNTAX_ERROR",
                f"derivative order exceeds {MAX_DERIVATIVE_ORDER}",
                location,
            )
        order += count
    return order


def _validate_derivatives(
    source: str,
    independent: tuple[str, ...],
    *,
    location: str,
) -> int:
    try:
        tree = ast.parse(source.strip().replace("^", "**"), mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise PreflightInputError(
            "EXPRESSION_SYNTAX_ERROR", "expression is not valid arithmetic syntax", location
        ) from exc
    independent_set = frozenset(independent)

    def maximum(node: ast.AST) -> int:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in DERIVATIVE_NAMES
        ):
            if node.keywords or len(node.args) < 2:
                raise PreflightInputError(
                    "DERIVATIVE_SYNTAX_ERROR",
                    "derivatives require positional expression and variable specifications",
                    location,
                )
            inner = maximum(node.args[0])
            local = _derivative_specs(
                node.args[1:], independent_set, location=location
            )
            if inner > MAX_DERIVATIVE_ORDER - local:
                raise PreflightInputError(
                    "DERIVATIVE_SYNTAX_ERROR",
                    f"nested derivative order exceeds {MAX_DERIVATIVE_ORDER}",
                    location,
                )
            return inner + local
        return max((maximum(child) for child in ast.iter_child_nodes(node)), default=0)

    return maximum(tree)


def _equation_sides(
    source: object,
    *,
    location: str,
    error_code: str = "CONDITION_PARSE_ERROR",
) -> tuple[str, str]:
    if not isinstance(source, str) or source.count("=") != 1:
        raise PreflightInputError(
            error_code, "equation must contain exactly one equals", location
        )
    lhs, rhs = (part.strip() for part in source.split("=", 1))
    if not lhs or not rhs:
        raise PreflightInputError(
            error_code, "equation sides must be nonempty", location
        )
    return lhs, rhs


def _field_calls(source: str, declared: frozenset[str]) -> set[str]:
    tree = ast.parse(source.strip().replace("^", "**"), mode="eval")
    return {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in declared
    }


def _formal_fields(fields: tuple[str, ...], arity: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in fields:
        formal = sp.Function(field)

        def apply(*arguments: sp.Expr, _formal: Any = formal, _field: str = field) -> sp.Expr:
            if len(arguments) != arity:
                raise ValueError(f"field {_field} argument arity differs")
            return _formal(*arguments)

        result[field] = apply
    return result


def _numeric_parameters(value: object) -> tuple[dict[str, sp.Expr], dict[str, str]]:
    if not isinstance(value, dict):
        raise PreflightInputError(
            "PARAMETER_ERROR", "parameters must be an object", "parameters"
        )
    names: dict[str, sp.Expr] = {"pi": sp.pi, "E": sp.E}
    normalized: dict[str, str] = {}
    for name, raw in value.items():
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or name in RESERVED_NAMES
        ):
            raise PreflightInputError(
                "PARAMETER_ERROR", "parameter name is invalid or reserved", f"parameters.{name}"
            )
        try:
            parsed = math_expr(raw, {"pi": sp.pi, "E": sp.E})
            numeric = float(sp.N(parsed))
        except Exception as exc:
            raise PreflightInputError(
                "PARAMETER_ERROR", "parameter value must be numeric", f"parameters.{name}"
            ) from exc
        if parsed.free_symbols or not math.isfinite(numeric) or parsed.is_real is False:
            raise PreflightInputError(
                "PARAMETER_ERROR", "parameter value must be finite and real", f"parameters.{name}"
            )
        names[name] = parsed
        normalized[name] = str(parsed)
    return names, normalized


def validate_public_row(row: object) -> dict[str, Any]:
    """Validate one seven-key public row without constructing any model/provider."""

    if isinstance(row, dict) and "schema_version" in row:
        from tools.benchmark_contract import validate_problem

        return validate_problem(row)

    checks: dict[str, Any] = {}
    errors: list[dict[str, str]] = []
    maximum_derivative_order = 0
    pde_fields: set[str] = set()
    condition_fields: set[str] = set()

    try:
        if not isinstance(row, dict) or set(row) != PUBLIC_KEYS:
            raise PreflightInputError(
                "SCHEMA_ERROR", "public row must contain exactly the seven public keys", "$"
            )
        dependent = _name_list(row["dependent_variables"], location="dependent_variables")
        independent = _name_list(
            row["independent_variables"], location="independent_variables"
        )
        if set(dependent) & set(independent):
            raise PreflightInputError(
                "DECLARATION_ERROR", "dependent and independent names overlap", "$"
            )
        if (set(dependent) | set(independent)) & RESERVED_NAMES:
            raise PreflightInputError(
                "DECLARATION_ERROR", "field or axis name is reserved", "$"
            )
        parameter_names, normalized_parameters = _numeric_parameters(row["parameters"])
        if set(parameter_names) - {"pi", "E"} & (set(dependent) | set(independent)):
            raise PreflightInputError(
                "DECLARATION_ERROR", "parameter and variable names overlap", "$"
            )
        checks["declarations"] = {
            "dependent_variables": list(dependent),
            "independent_variables": list(independent),
        }
        checks["parameters"] = normalized_parameters

        domain = row["domain"]
        if not isinstance(domain, dict) or set(domain) != set(independent):
            raise PreflightInputError(
                "DOMAIN_ERROR", "domain must define every and only declared axis", "domain"
            )
        normalized_domain: dict[str, list[str]] = {}
        for axis in independent:
            bounds = domain[axis]
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise PreflightInputError(
                    "DOMAIN_ERROR", "axis bounds must be a two-item list", f"domain.{axis}"
                )
            try:
                parsed_bounds = [math_expr(bound, parameter_names) for bound in bounds]
                numeric_bounds = [float(sp.N(bound)) for bound in parsed_bounds]
            except Exception as exc:
                raise PreflightInputError(
                    "DOMAIN_ERROR", "axis bounds must be numeric", f"domain.{axis}"
                ) from exc
            if (
                any(bound.free_symbols for bound in parsed_bounds)
                or not all(math.isfinite(bound) for bound in numeric_bounds)
                or numeric_bounds[0] >= numeric_bounds[1]
            ):
                raise PreflightInputError(
                    "DOMAIN_ERROR",
                    "axis bounds must be finite and strictly ordered",
                    f"domain.{axis}",
                )
            normalized_domain[axis] = [str(bound) for bound in parsed_bounds]
        checks["domain"] = normalized_domain

        pde = row["pde"]
        if not isinstance(pde, list) or not pde or any(not isinstance(item, str) for item in pde):
            raise PreflightInputError(
                "PDE_PARSE_ERROR", "pde must be a nonempty string list", "pde"
            )
        context = ParserContextV1(dependent, independent, tuple(normalized_parameters))
        declared_fields = frozenset(dependent)
        for index, equation in enumerate(pde):
            location = f"pde[{index}]"
            lhs, rhs = _equation_sides(
                equation, location=location, error_code="PDE_PARSE_ERROR"
            )
            maximum_derivative_order = max(
                maximum_derivative_order,
                _validate_derivatives(lhs, independent, location=f"{location}.lhs"),
                _validate_derivatives(rhs, independent, location=f"{location}.rhs"),
            )
            try:
                parse_equation(
                    equation,
                    context,
                    ParserLimitsV1(max_derivative_order=MAX_DERIVATIVE_ORDER),
                )
                pde_fields.update(_field_calls(lhs, declared_fields))
                pde_fields.update(_field_calls(rhs, declared_fields))
            except PreflightInputError:
                raise
            except Exception as exc:
                raise PreflightInputError("PDE_PARSE_ERROR", str(exc), location) from exc
        if pde_fields != set(dependent):
            raise PreflightInputError(
                "FIELD_CENSUS_ERROR",
                "PDE field calls must cover every declared dependent variable",
                "pde",
            )

        names = {
            **{name: sp.Symbol(name, real=True) for name in independent},
            **parameter_names,
        }
        formal_fields = _formal_fields(dependent, len(independent))
        for group in ("initial_conditions", "boundary_conditions"):
            equations = row[group]
            if not isinstance(equations, list) or any(
                not isinstance(item, str) for item in equations
            ):
                raise PreflightInputError(
                    "CONDITION_PARSE_ERROR", "conditions must be a string list", group
                )
            for index, equation in enumerate(equations):
                location = f"{group}[{index}]"
                lhs, rhs = _equation_sides(equation, location=location)
                maximum_derivative_order = max(
                    maximum_derivative_order,
                    _validate_derivatives(lhs, independent, location=f"{location}.lhs"),
                    _validate_derivatives(rhs, independent, location=f"{location}.rhs"),
                )
                try:
                    math_expr(lhs, names, formal_fields)
                    math_expr(rhs, names, formal_fields)
                    condition_fields.update(_field_calls(lhs, declared_fields))
                    condition_fields.update(_field_calls(rhs, declared_fields))
                except Exception as exc:
                    raise PreflightInputError(
                        "CONDITION_PARSE_ERROR", str(exc), location
                    ) from exc
        checks["field_census"] = {
            "declared": list(dependent),
            "pde": sorted(pde_fields),
            "conditions": sorted(condition_fields),
        }
        checks["maximum_derivative_order"] = maximum_derivative_order
    except PreflightInputError as exc:
        errors.append(exc.to_mapping())

    return {
        "status": "supported" if not errors else "invalid_public_input",
        "checks": checks,
        "errors": errors,
    }


def _safe_snapshot(selection: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise PreflightInputError("MANIFEST_ERROR", "snapshot path is invalid", "snapshot_path")
    candidate = (selection / relative).resolve()
    if not candidate.is_relative_to(selection.resolve()):
        raise PreflightInputError(
            "MANIFEST_ERROR", "snapshot path escapes selection directory", "snapshot_path"
        )
    return candidate


def run_preflight(selection: Path, output: Path) -> tuple[dict[str, Any], int]:
    """Validate all selected cases, write one detailed artifact, and return its exit code."""

    selection = Path(selection).resolve()
    output = Path(output)
    manifest_path = selection / "selection_manifest.json"
    frozen_path = selection / "FROZEN_ARTIFACTS.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    selected = manifest.get("selected_cases")
    frozen_files = frozen.get("files")
    if not isinstance(selected, list) or not isinstance(frozen_files, dict):
        raise PreflightInputError("MANIFEST_ERROR", "selection manifests are malformed", "$")

    manifest_sha = _sha256(manifest_path)
    global_errors: list[dict[str, str]] = []
    if frozen_files.get("selection_manifest.json") != manifest_sha:
        global_errors.append(
            PreflightInputError(
                "HASH_MISMATCH",
                "selection manifest differs from frozen hash",
                "selection_manifest.json",
            ).to_mapping()
        )

    cases: list[dict[str, Any]] = []
    for index, entry in enumerate(selected):
        case_id = entry.get("case_id") if isinstance(entry, Mapping) else None
        relative = entry.get("snapshot_path") if isinstance(entry, Mapping) else None
        expected = entry.get("sha256") if isinstance(entry, Mapping) else None
        base: dict[str, Any] = {
            "case_id": case_id,
            "selection_index": index,
            "status": "invalid_public_input",
            "checks": {},
            "errors": [],
        }
        try:
            path = _safe_snapshot(selection, relative)
            actual = _sha256(path)
            frozen_expected = frozen_files.get(relative)
            hash_matches = actual == expected == frozen_expected
            base["source"] = {
                "snapshot_path": relative,
                "expected_sha256": expected,
                "frozen_sha256": frozen_expected,
                "actual_sha256": actual,
                "hash_matches": hash_matches,
                "bytes": path.stat().st_size,
            }
            row = json.loads(path.read_text(encoding="utf-8"))
            validation = validate_public_row(row)
            base.update(validation)
            if not hash_matches:
                base["status"] = "invalid_public_input"
                base["errors"].insert(
                    0,
                    PreflightInputError(
                        "HASH_MISMATCH", "public snapshot differs from frozen hash", str(relative)
                    ).to_mapping(),
                )
        except Exception as exc:
            classified = (
                exc
                if isinstance(exc, PreflightInputError)
                else PreflightInputError("INPUT_READ_ERROR", str(exc), f"selected_cases[{index}]")
            )
            base["errors"].append(classified.to_mapping())
        cases.append(base)

    supported = sum(case["status"] == "supported" for case in cases)
    artifact = {
        "schema_version": 1,
        "purpose": "public-input contract validation before provider construction",
        "provider_calls": 0,
        "selection_directory": str(selection),
        "input_source_hashes": {
            "selection_manifest.json": manifest_sha,
            "FROZEN_ARTIFACTS.json": _sha256(frozen_path),
        },
        "selected_count": len(selected),
        "supported_count": supported,
        "invalid_count": len(cases) - supported,
        "all_supported": supported == len(cases) and not global_errors,
        "global_errors": global_errors,
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact, 0 if artifact["all_supported"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    _, exit_code = run_preflight(arguments.selection, arguments.output)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
