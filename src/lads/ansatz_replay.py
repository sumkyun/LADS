"""Independent symbolic replay for atom-free Ansatz proposals.

This module does not generate candidates and never opens evaluator data.  It compiles a
host-validated structural proposal and substitutes it into the public PDE. Strong
certificates prove a reduction; a separately labeled weak certificate proves only that a
reduced, separated, coupled, or full-dimensional open-slot skeleton survives symbolic
substitution for later Atom completion.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import sympy as sp
from pde_ansatz_discovery.cas.sympy_backend import SymPyBackendV1
from pde_ansatz_discovery.ir import parse_problem
from sympy.core.function import AppliedUndef

from .generalized_ansatz import _problem_from_public_input
from .llm_ansatz import (
    LLMAnsatzError,
    sanitize_public_pde_v1,
    validate_llm_ansatz_candidate,
)
from .safe_symbolic import parse_symbolic_expression

SLOT_COMPATIBILITY_OPERATION_LIMIT = 64_000
FULL_DIMENSIONAL_OPERATIONS_PER_PROFILE = 8_000

_IDENTIFIER = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\b")
_FUNCTION = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s*(?=\()")


@dataclass(frozen=True)
class NeutralAnsatzASTV1:
    """A concrete syntax tree containing only fields, profiles, and symbolic maps."""

    source: str
    field_expressions: tuple[tuple[str, sp.Expr], ...]
    profile_names: tuple[str, ...]
    symbolic_coefficients: tuple[str, ...]


@dataclass(frozen=True)
class AnsatzReplayCertificateV1:
    """Fail-closed result of independent symbolic Ansatz replay."""

    passed: bool
    validation_kind: str
    ansatz: str
    reduced_equations: tuple[str, ...]
    invariant_coordinates: tuple[str, ...]
    eliminated_variables: tuple[str, ...]
    side_conditions: tuple[str, ...]
    evidence: tuple[str, ...]
    failure_reason: str | None = None


def _real_symbols(names: Sequence[str]) -> dict[str, sp.Symbol]:
    return {name: sp.Symbol(name, real=True) for name in names}


def compile_neutral_ansatz(
    row: Mapping[str, Any], candidate: Mapping[str, Any]
) -> NeutralAnsatzASTV1:
    """Compile a validated atom-free proposal into the common symbolic representation."""

    dependent = tuple(map(str, row["dependent_variables"]))
    independent = tuple(map(str, row["independent_variables"]))
    raw_parameters = row.get("parameters", {})
    parameters = (
        tuple(map(str, raw_parameters))
        if isinstance(raw_parameters, Mapping)
        else tuple(map(str, raw_parameters))
    )
    ansatz, _reason = validate_llm_ansatz_candidate(
        candidate,
        dependent_variables=dependent,
        independent_variables=independent,
        parameters=parameters,
    )
    expressions: list[tuple[str, sp.Expr]] = []
    profile_names: set[str] = set()
    coefficients: set[str] = set()
    declared = {*dependent, *independent, *parameters}
    for equation in ansatz.split(";"):
        lhs, rhs = equation.split("=", maxsplit=1)
        field = lhs.split("(", maxsplit=1)[0]
        function_names = set(_FUNCTION.findall(rhs))
        identifiers = set(_IDENTIFIER.findall(rhs))
        names = _real_symbols(sorted(identifiers - function_names))
        expression = parse_symbolic_expression(rhs, names=names)
        expressions.append((field, expression))
        profile_names.update(
            str(function.func)
            for function in expression.atoms(AppliedUndef)
            if str(function.func)[0].isupper()
        )
        coefficients.update(
            str(symbol) for symbol in expression.free_symbols if str(symbol) not in declared
        )
    return NeutralAnsatzASTV1(
        source=ansatz,
        field_expressions=tuple(expressions),
        profile_names=tuple(sorted(profile_names)),
        symbolic_coefficients=tuple(sorted(coefficients)),
    )


def parse_public_residuals(
    row: Mapping[str, Any],
) -> tuple[tuple[sp.Symbol, ...], tuple[AppliedUndef, ...], tuple[sp.Expr, ...]]:
    replay_row = dict(row)
    replay_row["pde"] = sanitize_public_pde_v1(tuple(map(str, row["pde"])))
    problem = _problem_from_public_input(replay_row)
    parsed = parse_problem(problem)
    backend = SymPyBackendV1.from_parser_context(parsed.context)
    coordinates = tuple(backend._symbol(name) for name in problem.independent_variables)
    fields = tuple(sp.Function(name)(*coordinates) for name in problem.dependent_variables)
    residuals = tuple(backend.to_sympy(equation.residual) for equation in parsed.equations)
    return coordinates, fields, residuals


def _remove_multiplicative_orbit_factor(
    expression: sp.Expr, orbit: sp.Symbol, invariants: set[sp.Symbol]
) -> sp.Expr | None:
    if sp.count_ops(expression) > 2_000:
        return None
    value = sp.factor_terms(expression)
    if orbit not in value.free_symbols or value == 0:
        return value
    independent, dependent = value.as_independent(orbit, as_Add=False)
    power = dependent.as_powers_dict().get(orbit)
    if power is None or power.has(orbit) or power.free_symbols & invariants:
        return None
    stripped = dependent / orbit**power
    if stripped.has(orbit):
        return None
    return sp.factor_terms(independent * stripped)


def _single_coordinate_replay(
    ast: NeutralAnsatzASTV1,
    coordinates: tuple[sp.Symbol, ...],
    fields: tuple[AppliedUndef, ...],
    residuals: tuple[sp.Expr, ...],
) -> AnsatzReplayCertificateV1 | None:
    profile_calls = {
        function
        for _field, expression in ast.field_expressions
        for function in expression.atoms(AppliedUndef)
    }
    if not profile_calls or any(len(function.args) != 1 for function in profile_calls):
        return None
    arguments = {sp.cancel(function.args[0]) for function in profile_calls}
    if len(arguments) != 1:
        return None
    coordinate_map = next(iter(arguments))
    if coordinate_map.atoms(AppliedUndef):
        return None
    active_coordinates = tuple(item for item in coordinates if coordinate_map.has(item))
    if not active_coordinates or len(coordinates) < 2:
        return None
    eta = sp.Symbol("eta_1", real=True)
    pivot: sp.Symbol | None = None
    inverse: sp.Expr | None = None
    for trial in active_coordinates:
        try:
            solutions = sp.solve(sp.Eq(coordinate_map, eta), trial, dict=False)
        except (NotImplementedError, TypeError, ValueError):
            continue
        if solutions:
            pivot, inverse = trial, solutions[0]
            break
    if pivot is None or inverse is None or inverse.has(pivot):
        return None

    field_map = {
        field: expression
        for field, (_name, expression) in zip(fields, ast.field_expressions, strict=True)
    }
    orbit_coordinates = tuple(item for item in coordinates if item != pivot)
    reduced_equations: list[str] = []
    for residual in residuals:
        transformed = (
            residual.xreplace(field_map).doit().subs({pivot: inverse}, simultaneous=True).doit()
        )
        if sp.count_ops(transformed) > 2_000:
            return None
        transformed = sp.factor_terms(transformed)
        reduced = transformed
        for orbit in orbit_coordinates:
            reduced = _remove_multiplicative_orbit_factor(reduced, orbit, {eta})
            if reduced is None:
                return None
        reduced = sp.factor_terms(reduced)
        if reduced.free_symbols & set(coordinates):
            return None
        reduced_equations.append(sp.sstr(reduced))
    return AnsatzReplayCertificateV1(
        passed=True,
        validation_kind="single_coordinate_symbolic_replay_v1",
        ansatz=ast.source,
        reduced_equations=tuple(reduced_equations),
        invariant_coordinates=(f"eta_1={sp.sstr(coordinate_map)}",),
        eliminated_variables=tuple(map(str, orbit_coordinates)),
        side_conditions=(),
        evidence=(
            "ATOM_FREE_NEUTRAL_AST",
            "PDE_SUBSTITUTION_REPLAYED",
            "CHAIN_RULE_EXPANDED",
            "ORBIT_DEPENDENCE_ELIMINATED",
            "REDUCED_EQUATIONS_CLOSED",
        ),
    )


def _separation_replay(
    ast: NeutralAnsatzASTV1,
    coordinates: tuple[sp.Symbol, ...],
    fields: tuple[AppliedUndef, ...],
    residuals: tuple[sp.Expr, ...],
) -> AnsatzReplayCertificateV1 | None:
    if len(fields) != 1 or len(ast.field_expressions) != 1 or len(coordinates) < 2:
        return None
    expression = ast.field_expressions[0][1]
    profile_calls = expression.atoms(AppliedUndef)
    if len(profile_calls) < 2:
        return None
    profile_blocks: list[frozenset[sp.Symbol]] = []
    for function in profile_calls:
        if (
            not function.args
            or any(argument not in coordinates for argument in function.args)
            or len(set(function.args)) != len(function.args)
        ):
            return None
        profile_blocks.append(frozenset(function.args))
    if len(set(profile_blocks)) < 2:
        return None
    for left_index, left_block in enumerate(profile_blocks):
        if any(left_block & right_block for right_block in profile_blocks[left_index + 1 :]):
            return None
    if set().union(*profile_blocks) != set(coordinates):
        return None
    expanded = sp.expand(expression)
    if expanded.is_Add:
        mode = "additive"
    elif expression.is_Mul:
        mode = "multiplicative"
    else:
        return None
    field_map = {fields[0]: expression}
    reduced_equations: list[str] = []
    for residual in residuals:
        transformed = sp.factor_terms(residual.xreplace(field_map).doit())
        if mode == "multiplicative":
            transformed = sp.cancel(transformed / expression)
        transformed = sp.factor_terms(sp.cancel(sp.together(transformed)))
        for left_index, left_block in enumerate(profile_blocks):
            for right_block in profile_blocks[left_index + 1 :]:
                for left in left_block:
                    for right in right_block:
                        mixed = sp.simplify(sp.diff(transformed, left, right).doit())
                        if mixed != 0:
                            return None
        reduced_equations.append(sp.sstr(transformed))
    return AnsatzReplayCertificateV1(
        passed=True,
        validation_kind=f"{mode}_separation_symbolic_replay_v1",
        ansatz=ast.source,
        reduced_equations=tuple(reduced_equations),
        invariant_coordinates=tuple(
            f"profile_block_{index + 1}={','.join(map(str, block))}"
            for index, block in enumerate(
                sorted(
                    (tuple(sorted(group, key=str)) for group in set(profile_blocks)),
                    key=lambda group: tuple(map(str, group)),
                )
            )
        ),
        eliminated_variables=(),
        side_conditions=("profiles_nonzero",) if mode == "multiplicative" else (),
        evidence=(
            "ATOM_FREE_NEUTRAL_AST",
            "PDE_SUBSTITUTION_REPLAYED",
            "CHAIN_RULE_EXPANDED",
            "PAIRWISE_MIXED_DERIVATIVES_PROVED_ZERO",
            "SEPARATED_EQUATION_CLOSED",
        ),
    )


def _separation_slot_compatibility_replay(
    ast: NeutralAnsatzASTV1,
    coordinates: tuple[sp.Symbol, ...],
    fields: tuple[AppliedUndef, ...],
    residuals: tuple[sp.Expr, ...],
) -> AnsatzReplayCertificateV1 | None:
    """Validate an atom-free separated slot skeleton without claiming generic reduction.

    Forced and nonlinear PDEs can require a specific Atom before cross-coordinate terms
    cancel.  The stronger separation replay correctly refuses to prove a reduction for
    arbitrary profiles in that situation, but that proof obligation is too strong for an
    Ansatz whose declared purpose is subsequent Atom completion.  This fallback therefore
    proves only that the separated profile blocks are well formed and that symbolic PDE
    substitution closes over those slots and the public operator vocabulary.
    """

    if len(fields) != 1 or len(ast.field_expressions) != 1 or len(coordinates) < 2:
        return None
    expression = ast.field_expressions[0][1]
    profile_calls = expression.atoms(AppliedUndef)
    if len(profile_calls) < 2:
        return None
    profile_blocks: list[frozenset[sp.Symbol]] = []
    for function in profile_calls:
        if (
            not function.args
            or any(argument not in coordinates for argument in function.args)
            or len(set(function.args)) != len(function.args)
        ):
            return None
        profile_blocks.append(frozenset(function.args))
    if len(set(profile_blocks)) < 2:
        return None
    for left_index, left_block in enumerate(profile_blocks):
        if any(left_block & right_block for right_block in profile_blocks[left_index + 1 :]):
            return None
    if set().union(*profile_blocks) != set(coordinates):
        return None
    expanded = sp.expand(expression)
    if expanded.is_Add:
        mode = "additive"
    elif expression.is_Mul:
        mode = "multiplicative"
    else:
        return None

    field_map = {fields[0]: expression}
    public_symbols = set(coordinates) | expression.free_symbols
    allowed_function_names = set(ast.profile_names)
    for residual in residuals:
        public_symbols.update(residual.free_symbols)
        allowed_function_names.update(
            str(call.func) for call in residual.atoms(AppliedUndef) if call not in fields
        )

    transformed_equations: list[str] = []
    used_profile_names: set[str] = set()
    for residual in residuals:
        transformed = sp.factor_terms(residual.xreplace(field_map).doit())
        if sp.count_ops(transformed) > SLOT_COMPATIBILITY_OPERATION_LIMIT or any(
            transformed.has(field) for field in fields
        ):
            return None
        transformed_calls = transformed.atoms(AppliedUndef)
        transformed_function_names = {str(call.func) for call in transformed_calls}
        if not transformed_function_names.issubset(allowed_function_names):
            return None
        if not transformed.free_symbols.issubset(public_symbols):
            return None
        used_profile_names.update(transformed_function_names & set(ast.profile_names))
        transformed_equations.append(sp.sstr(transformed))
    if used_profile_names != set(ast.profile_names):
        return None

    return AnsatzReplayCertificateV1(
        passed=True,
        validation_kind=f"{mode}_separation_slot_compatibility_symbolic_replay_v2",
        ansatz=ast.source,
        reduced_equations=tuple(transformed_equations),
        invariant_coordinates=tuple(
            f"slot_block_{index + 1}={','.join(map(str, block))}"
            for index, block in enumerate(
                sorted(
                    (tuple(sorted(group, key=str)) for group in set(profile_blocks)),
                    key=lambda group: tuple(map(str, group)),
                )
            )
        ),
        eliminated_variables=(),
        side_conditions=(
            "generic_separation_reduction_not_proved",
            "atom_completion_required_before_final_pde_validation",
        ),
        evidence=(
            "ATOM_FREE_NEUTRAL_AST",
            "DISJOINT_PROFILE_BLOCKS_COVER_PUBLIC_COORDINATES",
            "PDE_SUBSTITUTION_REPLAYED",
            "CHAIN_RULE_EXPANDED",
            "NO_ORIGINAL_DEPENDENT_FIELD_REMAINS",
            "TRANSFORMED_RESIDUAL_CLOSES_OVER_DECLARED_SLOTS",
        ),
    )


def _multi_field_slot_compatibility_replay(
    ast: NeutralAnsatzASTV1,
    coordinates: tuple[sp.Symbol, ...],
    fields: tuple[AppliedUndef, ...],
    residuals: tuple[sp.Expr, ...],
) -> AnsatzReplayCertificateV1 | None:
    """Prove coupled open-slot substitution closure without claiming a reduction."""

    if len(fields) < 2 or len(ast.field_expressions) != len(fields):
        return None
    expressions_by_name = dict(ast.field_expressions)
    field_names = tuple(str(field.func) for field in fields)
    if set(expressions_by_name) != set(field_names):
        return None
    profile_names_by_field: dict[str, set[str]] = {}
    for field_name in field_names:
        expression = expressions_by_name[field_name]
        calls = expression.atoms(AppliedUndef)
        if not calls:
            return None
        if any(
            not call.args or any(argument.free_symbols - set(coordinates) for argument in call.args)
            for call in calls
        ):
            return None
        profile_names_by_field[field_name] = {str(call.func) for call in calls}

    field_map = {field: expressions_by_name[str(field.func)] for field in fields}
    public_symbols = set(coordinates)
    allowed_function_names = set(ast.profile_names)
    for expression in expressions_by_name.values():
        public_symbols.update(expression.free_symbols)
    for residual in residuals:
        public_symbols.update(residual.free_symbols)
        allowed_function_names.update(
            str(call.func) for call in residual.atoms(AppliedUndef) if call not in fields
        )

    transformed_equations: list[str] = []
    used_profile_names: set[str] = set()
    for residual in residuals:
        transformed = sp.factor_terms(residual.xreplace(field_map).doit())
        if sp.count_ops(transformed) > SLOT_COMPATIBILITY_OPERATION_LIMIT or any(
            transformed.has(field) for field in fields
        ):
            return None
        transformed_names = {str(call.func) for call in transformed.atoms(AppliedUndef)}
        if not transformed_names.issubset(allowed_function_names):
            return None
        if not transformed.free_symbols.issubset(public_symbols):
            return None
        used_profile_names.update(transformed_names & set(ast.profile_names))
        transformed_equations.append(sp.sstr(transformed))
    if used_profile_names != set(ast.profile_names):
        return None

    return AnsatzReplayCertificateV1(
        passed=True,
        validation_kind="multi_field_slot_compatibility_symbolic_replay_v1",
        ansatz=ast.source,
        reduced_equations=tuple(transformed_equations),
        invariant_coordinates=tuple(
            f"{field_name}_slots={','.join(sorted(profile_names_by_field[field_name]))}"
            for field_name in field_names
        ),
        eliminated_variables=(),
        side_conditions=(
            "coupled_reduction_not_proved",
            "atom_completion_required_before_final_pde_validation",
        ),
        evidence=(
            "ATOM_FREE_NEUTRAL_AST",
            "ALL_DEPENDENT_FIELDS_HAVE_OPEN_SLOTS",
            "PDE_SUBSTITUTION_REPLAYED",
            "CHAIN_RULE_EXPANDED",
            "NO_ORIGINAL_DEPENDENT_FIELD_REMAINS",
            "TRANSFORMED_RESIDUAL_CLOSES_OVER_DECLARED_SLOTS",
        ),
    )


def _full_dimensional_slot_compatibility_replay(
    ast: NeutralAnsatzASTV1,
    coordinates: tuple[sp.Symbol, ...],
    fields: tuple[AppliedUndef, ...],
    residuals: tuple[sp.Expr, ...],
) -> AnsatzReplayCertificateV1 | None:
    """Retain an unreduced full-coordinate slot without pretending it is a reduction.

    A full-dimensional arbitrary profile such as ``F(x, y)`` is often the least biased
    structure for nonseparable forcing.  Earlier protocols rejected it because it does
    not reduce the PDE dimension.  That conflated *reduction proof* with *safe staged
    completion*.  This verifier proves only that the atom-free structure is genuinely
    full-dimensional and that substitution closes over declared slots and public data.
    """

    if len(fields) != 1 or len(ast.field_expressions) != 1 or not coordinates:
        return None
    expression = ast.field_expressions[0][1]
    profile_calls = tuple(sorted(expression.atoms(AppliedUndef), key=sp.default_sort_key))
    if not profile_calls:
        return None
    coordinate_set = set(coordinates)
    full_dimensional_calls: list[AppliedUndef] = []
    for call in profile_calls:
        if not call.args or any(argument.atoms(AppliedUndef) for argument in call.args):
            return None
        covered = set().union(*(argument.free_symbols & coordinate_set for argument in call.args))
        # Require enough independent argument slots as well as full coordinate coverage.
        # This keeps one-dimensional similarity maps on the reduced-coordinate path.
        if len(call.args) >= len(coordinates) and covered == coordinate_set:
            full_dimensional_calls.append(call)
    if not full_dimensional_calls:
        return None

    field_map = {fields[0]: expression}
    public_symbols = set(coordinates) | expression.free_symbols
    allowed_function_names = set(ast.profile_names)
    for residual in residuals:
        public_symbols.update(residual.free_symbols)
        allowed_function_names.update(
            str(call.func) for call in residual.atoms(AppliedUndef) if call not in fields
        )

    transformed_equations: list[str] = []
    used_profile_names: set[str] = set()
    operation_limit = max(
        SLOT_COMPATIBILITY_OPERATION_LIMIT,
        FULL_DIMENSIONAL_OPERATIONS_PER_PROFILE * len(profile_calls),
    )
    for residual in residuals:
        transformed = sp.factor_terms(residual.xreplace(field_map).doit())
        if sp.count_ops(transformed) > operation_limit or transformed.has(fields[0]):
            return None
        transformed_names = {str(call.func) for call in transformed.atoms(AppliedUndef)}
        if not transformed_names.issubset(allowed_function_names):
            return None
        if not transformed.free_symbols.issubset(public_symbols):
            return None
        used_profile_names.update(transformed_names & set(ast.profile_names))
        transformed_equations.append(sp.sstr(transformed))
    if used_profile_names != set(ast.profile_names):
        return None

    return AnsatzReplayCertificateV1(
        passed=True,
        validation_kind="full_dimensional_slot_compatibility_symbolic_replay_v1",
        ansatz=ast.source,
        reduced_equations=tuple(transformed_equations),
        invariant_coordinates=tuple(
            f"full_slot_{index + 1}={sp.sstr(call)}"
            for index, call in enumerate(full_dimensional_calls)
        ),
        eliminated_variables=(),
        side_conditions=(
            "full_dimensional_reduction_not_claimed",
            "atom_completion_required_before_final_pde_validation",
        ),
        evidence=(
            "ATOM_FREE_NEUTRAL_AST",
            "FULL_DIMENSIONAL_OPEN_SLOT_RETAINED",
            "PDE_SUBSTITUTION_REPLAYED",
            "CHAIN_RULE_EXPANDED",
            "NO_ORIGINAL_DEPENDENT_FIELD_REMAINS",
            "TRANSFORMED_RESIDUAL_CLOSES_OVER_DECLARED_SLOTS",
        ),
    )


def _multi_coordinate_replay(
    ast: NeutralAnsatzASTV1,
    coordinates: tuple[sp.Symbol, ...],
    fields: tuple[AppliedUndef, ...],
    residuals: tuple[sp.Expr, ...],
) -> AnsatzReplayCertificateV1 | None:
    leaf_coordinate_maps: set[sp.Expr] = set()
    coordinate_set = set(coordinates)
    for _field, expression in ast.field_expressions:
        for function in expression.atoms(AppliedUndef):
            if any(argument.atoms(AppliedUndef) for argument in function.args):
                continue
            for argument in function.args:
                reduced_argument = sp.cancel(argument)
                if reduced_argument.free_symbols & coordinate_set:
                    leaf_coordinate_maps.add(reduced_argument)
    coordinate_maps = tuple(sorted(leaf_coordinate_maps, key=sp.default_sort_key))
    if not 2 <= len(coordinate_maps) < len(coordinates):
        return None
    jacobian = sp.Matrix(coordinate_maps).jacobian(coordinates)
    if jacobian.rank() != len(coordinate_maps):
        return None

    eta_symbols = tuple(
        sp.Symbol(f"eta_{index + 1}", real=True) for index in range(len(coordinate_maps))
    )
    inverse_coordinates: dict[sp.Symbol, sp.Expr] | None = None
    orbit_symbols: tuple[sp.Symbol, ...] = ()
    eliminated_coordinates: tuple[sp.Symbol, ...] = ()
    chart_side_conditions: tuple[str, ...] = ()
    for pivots in combinations(coordinates, len(coordinate_maps)):
        orbit_coordinates = tuple(item for item in coordinates if item not in pivots)
        pivot_indices = tuple(coordinates.index(item) for item in pivots)
        chart_minor = sp.factor(jacobian[:, pivot_indices].det())
        if chart_minor == 0:
            continue
        trial_orbits = tuple(
            sp.Symbol(f"orbit_{index + 1}", real=True) for index in range(len(orbit_coordinates))
        )
        orbit_map = dict(zip(orbit_coordinates, trial_orbits, strict=True))
        try:
            solutions = sp.solve(
                [
                    sp.Eq(value, eta)
                    for value, eta in zip(coordinate_maps, eta_symbols, strict=True)
                ],
                pivots,
                dict=True,
            )
        except (NotImplementedError, TypeError, ValueError):
            continue
        if not solutions or any(pivot not in solutions[0] for pivot in pivots):
            continue
        candidate_inverse = {
            pivot: sp.cancel(solutions[0][pivot].subs(orbit_map, simultaneous=True))
            for pivot in pivots
        }
        if any(value.has(*pivots) for value in candidate_inverse.values()):
            continue
        candidate_inverse.update(orbit_map)
        inverse_coordinates = candidate_inverse
        orbit_symbols = trial_orbits
        eliminated_coordinates = orbit_coordinates
        chart_side_conditions = () if chart_minor.is_number else (f"{sp.sstr(chart_minor)}!=0",)
        break
    if inverse_coordinates is None:
        return None

    field_map = {
        field: expression
        for field, (_name, expression) in zip(fields, ast.field_expressions, strict=True)
    }
    reduced_equations: list[str] = []
    for residual in residuals:
        transformed = (
            residual.xreplace(field_map).doit().subs(inverse_coordinates, simultaneous=True).doit()
        )
        if sp.count_ops(transformed) > 4_000:
            return None
        reduced = sp.factor_terms(transformed)
        for orbit in orbit_symbols:
            reduced = _remove_multiplicative_orbit_factor(reduced, orbit, set(eta_symbols))
            if reduced is None:
                return None
        reduced = sp.factor_terms(reduced)
        if reduced.free_symbols & {*coordinates, *orbit_symbols}:
            return None
        reduced_equations.append(sp.sstr(reduced))
    return AnsatzReplayCertificateV1(
        passed=True,
        validation_kind="multi_coordinate_symbolic_replay_v1",
        ansatz=ast.source,
        reduced_equations=tuple(reduced_equations),
        invariant_coordinates=tuple(
            f"eta_{index + 1}={sp.sstr(value)}" for index, value in enumerate(coordinate_maps)
        ),
        eliminated_variables=tuple(map(str, eliminated_coordinates)),
        side_conditions=chart_side_conditions,
        evidence=(
            "ATOM_FREE_NEUTRAL_AST",
            "INVARIANT_MAP_JACOBIAN_FULL_ROW_RANK",
            "PDE_SUBSTITUTION_REPLAYED",
            "CHAIN_RULE_EXPANDED",
            "ORBIT_DEPENDENCE_ELIMINATED",
            "REDUCED_EQUATIONS_CLOSED",
        ),
    )


def _reduced_coordinate_slot_compatibility_replay(
    ast: NeutralAnsatzASTV1,
    coordinates: tuple[sp.Symbol, ...],
    fields: tuple[AppliedUndef, ...],
    residuals: tuple[sp.Expr, ...],
) -> AnsatzReplayCertificateV1 | None:
    """Certify a lower-dimensional coordinate slot without claiming a PDE reduction.

    A forced PDE can retain explicit orbit dependence until the Atom is supplied, so the
    strong replay may correctly decline a generic reduction proof.  This bounded fallback
    only proves that the atom-free profile uses a full-rank, lower-dimensional coordinate
    map and that public-PDE substitution closes over that slot and declared public symbols.
    """

    if len(fields) != 1 or len(ast.field_expressions) != 1 or len(coordinates) < 2:
        return None
    expression = ast.field_expressions[0][1]
    profile_calls = expression.atoms(AppliedUndef)
    if not profile_calls:
        return None
    coordinate_set = set(coordinates)
    coordinate_maps = tuple(
        sorted(
            {
                sp.cancel(argument)
                for function in profile_calls
                for argument in function.args
                if not argument.atoms(AppliedUndef) and argument.free_symbols & coordinate_set
            },
            key=sp.default_sort_key,
        )
    )
    if not 1 <= len(coordinate_maps) < len(coordinates):
        return None
    if sp.Matrix(coordinate_maps).jacobian(coordinates).rank() != len(coordinate_maps):
        return None

    field_map = {fields[0]: expression}
    public_symbols = set(coordinates) | expression.free_symbols
    allowed_function_names = set(ast.profile_names)
    for residual in residuals:
        public_symbols.update(residual.free_symbols)
        allowed_function_names.update(
            str(call.func) for call in residual.atoms(AppliedUndef) if call not in fields
        )
    transformed_equations: list[str] = []
    used_profile_names: set[str] = set()
    for residual in residuals:
        transformed = sp.factor_terms(residual.xreplace(field_map).doit())
        if sp.count_ops(transformed) > SLOT_COMPATIBILITY_OPERATION_LIMIT or transformed.has(
            fields[0]
        ):
            return None
        transformed_names = {str(call.func) for call in transformed.atoms(AppliedUndef)}
        if not transformed_names.issubset(allowed_function_names):
            return None
        if not transformed.free_symbols.issubset(public_symbols):
            return None
        used_profile_names.update(transformed_names & set(ast.profile_names))
        transformed_equations.append(sp.sstr(transformed))
    if used_profile_names != set(ast.profile_names):
        return None
    return AnsatzReplayCertificateV1(
        passed=True,
        validation_kind="reduced_coordinate_slot_compatibility_symbolic_replay_v1",
        ansatz=ast.source,
        reduced_equations=tuple(transformed_equations),
        invariant_coordinates=tuple(
            f"slot_coordinate_{index + 1}={sp.sstr(value)}"
            for index, value in enumerate(coordinate_maps)
        ),
        eliminated_variables=(),
        side_conditions=(
            "generic_coordinate_reduction_not_proved",
            "atom_completion_required_before_final_pde_validation",
        ),
        evidence=(
            "ATOM_FREE_NEUTRAL_AST",
            "LOWER_DIMENSIONAL_COORDINATE_MAP",
            "COORDINATE_MAP_JACOBIAN_FULL_ROW_RANK",
            "PDE_SUBSTITUTION_REPLAYED",
            "NO_ORIGINAL_DEPENDENT_FIELD_REMAINS",
            "TRANSFORMED_RESIDUAL_CLOSES_OVER_DECLARED_SLOT",
        ),
    )


def _failed_replay(
    candidate: Mapping[str, Any], validation_kind: str, reason: str
) -> AnsatzReplayCertificateV1:
    return AnsatzReplayCertificateV1(
        False,
        validation_kind,
        str(candidate.get("ansatz", "")),
        (),
        (),
        (),
        (),
        (),
        reason,
    )


def _replay_compiled(
    ast: NeutralAnsatzASTV1,
    coordinates: tuple[sp.Symbol, ...],
    fields: tuple[AppliedUndef, ...],
    residuals: tuple[sp.Expr, ...],
) -> AnsatzReplayCertificateV1:
    for verifier in (
        _single_coordinate_replay,
        _multi_coordinate_replay,
        _full_dimensional_slot_compatibility_replay,
        _reduced_coordinate_slot_compatibility_replay,
        _separation_replay,
        _separation_slot_compatibility_replay,
        _multi_field_slot_compatibility_replay,
    ):
        try:
            certificate = verifier(ast, coordinates, fields, residuals)
        except (NotImplementedError, TypeError, ValueError, ZeroDivisionError):
            certificate = None
        if certificate is not None:
            return certificate
    return _failed_replay(
        {"ansatz": ast.source},
        "unsupported_or_unproved_reduction_v1",
        "candidate did not satisfy an implemented symbolic reduction proof",
    )


def replay_ansatz_candidates(
    row: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> tuple[AnsatzReplayCertificateV1, ...]:
    """Replay a proposal batch while parsing the public PDE exactly once."""

    if not candidates:
        return ()
    try:
        coordinates, fields, residuals = parse_public_residuals(row)
    except (KeyError, TypeError, ValueError, LLMAnsatzError) as exc:
        return tuple(
            _failed_replay(candidate, "rejected_before_symbolic_replay_v1", str(exc))
            for candidate in candidates
        )
    certificates: list[AnsatzReplayCertificateV1] = []
    for candidate in candidates:
        try:
            ast = compile_neutral_ansatz(row, candidate)
        except (KeyError, TypeError, ValueError, LLMAnsatzError) as exc:
            certificates.append(
                _failed_replay(candidate, "rejected_before_symbolic_replay_v1", str(exc))
            )
            continue
        certificates.append(_replay_compiled(ast, coordinates, fields, residuals))
    return tuple(certificates)


def replay_ansatz_candidate(
    row: Mapping[str, Any], candidate: Mapping[str, Any]
) -> AnsatzReplayCertificateV1:
    """Certify one proposal or return an explicit fail-closed result."""

    return replay_ansatz_candidates(row, (candidate,))[0]


__all__ = [
    "AnsatzReplayCertificateV1",
    "FULL_DIMENSIONAL_OPERATIONS_PER_PROFILE",
    "NeutralAnsatzASTV1",
    "SLOT_COMPATIBILITY_OPERATION_LIMIT",
    "compile_neutral_ansatz",
    "parse_public_residuals",
    "replay_ansatz_candidate",
    "replay_ansatz_candidates",
]
