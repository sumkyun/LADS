"""SymPy 1.14 backend. This is the sole discovery module allowed to import SymPy."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import sympy
from sympy.core.function import AppliedUndef
from sympy.polys.polyerrors import PolynomialError

from ..ir import (
    ExpressionIRV1,
    ExpressionKindV1,
    ParserContextV1,
    SymbolRoleV1,
    derivative,
    function,
    jet,
    parse_problem,
    rational,
    symbol,
)
from ..ir.parser import SAFE_FUNCTIONS_V1
from ..models import BranchCoverageV1, DiscoveryProblemInputV1, ProofStatusV1
from .backend import (
    BranchKindV1,
    BranchResultV1,
    CASOperationResultV1,
    ProofResultV1,
)


class SymPyBackendV1:
    backend_name = "sympy"
    normalization_version = "SymPyNamedPipelineV1"

    def __init__(
        self,
        independent_variables: Sequence[str] = ("x", "t"),
        symbol_roles: Mapping[str, SymbolRoleV1 | str] | None = None,
        declared_functions: Sequence[str] = (),
    ) -> None:
        self.independent_variables = tuple(independent_variables)
        self.symbol_roles = {
            name: SymbolRoleV1(role) for name, role in (symbol_roles or {}).items()
        }
        for name in self.independent_variables:
            self.symbol_roles.setdefault(name, SymbolRoleV1.INDEPENDENT)
        self.declared_functions = frozenset(declared_functions)
        self._symbols = {
            name: sympy.Symbol(name, real=True) for name in sorted(self.symbol_roles)
        }

    @property
    def backend_version(self) -> str:
        return sympy.__version__

    @classmethod
    def from_parser_context(cls, context: ParserContextV1) -> "SymPyBackendV1":
        roles: dict[str, SymbolRoleV1] = {
            name: SymbolRoleV1.INDEPENDENT for name in context.independent_variables
        }
        roles.update({name: SymbolRoleV1.PARAMETER for name in context.parameters})
        return cls(context.independent_variables, roles, context.declared_functions)

    def _symbol(self, name: str):
        if name not in self._symbols:
            role = (
                self.symbol_roles[name]
                if name in self.symbol_roles
                else SymbolRoleV1.PARAMETER
            )
            self.symbol_roles[name] = role
            self._symbols[name] = sympy.Symbol(name, real=True)
        return self._symbols[name]

    @staticmethod
    def _derivative_function_name(name: str, order: int) -> str:
        base, marker, suffix = name.rpartition("__d")
        if marker and base and suffix.isdigit():
            return f"{base}__d{int(suffix) + order}"
        return f"{name}__d{order}"

    def _is_declared_function_or_derivative(self, name: str) -> bool:
        if name in self.declared_functions:
            return True
        base, marker, suffix = name.rpartition("__d")
        return bool(marker and suffix.isdigit() and base in self.declared_functions)

    def to_sympy(self, expression: ExpressionIRV1):
        kind = expression.kind
        if kind is ExpressionKindV1.RATIONAL:
            return sympy.Rational(expression.numerator, expression.denominator)
        if kind is ExpressionKindV1.SYMBOL:
            return self._symbol(expression.name)
        if kind is ExpressionKindV1.JET:
            base = sympy.Function(expression.jet.field)(
                *(self._symbol(name) for name in self.independent_variables)
            )
            if expression.jet.multi_index.variables:
                return sympy.Derivative(
                    base,
                    *(self._symbol(name) for name in expression.jet.multi_index.variables),
                )
            return base
        if kind is ExpressionKindV1.ADD:
            return sympy.Add(*(self.to_sympy(item) for item in expression.operands), evaluate=False)
        if kind is ExpressionKindV1.MUL:
            return sympy.Mul(*(self.to_sympy(item) for item in expression.operands), evaluate=False)
        if kind is ExpressionKindV1.POW:
            return sympy.Pow(
                self.to_sympy(expression.operands[0]),
                self.to_sympy(expression.operands[1]),
                evaluate=False,
            )
        if kind is ExpressionKindV1.FUNCTION:
            arguments = tuple(self.to_sympy(item) for item in expression.operands)
            if expression.name in SAFE_FUNCTIONS_V1:
                return getattr(sympy, expression.name)(*arguments)
            return sympy.Function(expression.name)(*arguments)
        if kind is ExpressionKindV1.DERIVATIVE:
            return sympy.Derivative(
                self.to_sympy(expression.operands[0]),
                *(self._symbol(name) for name in expression.derivative_variables),
            )
        raise TypeError(f"unsupported expression kind: {kind}")

    def from_sympy(self, expression) -> ExpressionIRV1:
        if expression.is_Integer:
            return rational(int(expression))
        if expression.is_Rational:
            return rational(int(expression.p), int(expression.q))
        if expression.is_Symbol:
            name = str(expression)
            role = (
                self.symbol_roles[name]
                if name in self.symbol_roles
                else SymbolRoleV1.PARAMETER
            )
            return symbol(name, role)
        if isinstance(expression, sympy.Subs):
            inner = expression.expr
            if isinstance(inner, sympy.Derivative) and isinstance(inner.expr, AppliedUndef):
                function_name = str(inner.expr.func)
                order = len(inner.variables)
                points = tuple(expression.point)
                if len(points) == 1:
                    return function(
                        self._derivative_function_name(function_name, order),
                        self.from_sympy(points[0]),
                    )
            raise TypeError("unsupported SymPy Subs node")
        if isinstance(expression, sympy.Derivative):
            base = expression.expr
            variables = tuple(str(variable) for variable in expression.variables)
            if isinstance(base, AppliedUndef) and self._is_declared_function_or_derivative(str(base.func)):
                if len(base.args) == 1 and all(variable == str(base.args[0]) for variable in variables):
                    return function(
                        self._derivative_function_name(str(base.func), len(variables)),
                        self.from_sympy(base.args[0]),
                    )
            if isinstance(base, AppliedUndef) and str(base.func) not in self.declared_functions:
                return jet(str(base.func), variables)
            return derivative(self.from_sympy(base), variables)
        if isinstance(expression, AppliedUndef):
            name = str(expression.func)
            arguments = tuple(self.from_sympy(item) for item in expression.args)
            if name not in self.declared_functions and tuple(map(str, expression.args)) == self.independent_variables:
                return jet(name)
            return function(name, *arguments)
        if expression.is_Add:
            from ..ir import add

            return add(*(self.from_sympy(item) for item in expression.args))
        if expression.is_Mul:
            from ..ir import mul

            return mul(*(self.from_sympy(item) for item in expression.args))
        if expression.is_Pow:
            from ..ir import power

            return power(self.from_sympy(expression.base), self.from_sympy(expression.exp))
        if expression.is_Function:
            return function(str(expression.func), *(self.from_sympy(item) for item in expression.args))
        raise TypeError(f"SymPy node outside project IR: {type(expression).__name__}")

    def round_trip(self, expression: ExpressionIRV1) -> ExpressionIRV1:
        return self.from_sympy(self.to_sympy(expression))

    def render(self, expression: ExpressionIRV1) -> str:
        return str(self.to_sympy(expression))

    def _apply(self, operation, expression: ExpressionIRV1) -> ExpressionIRV1:
        return self.from_sympy(operation(self.to_sympy(expression)))

    def differentiate(self, expression: ExpressionIRV1, variable: str, order: int = 1) -> ExpressionIRV1:
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError("derivative order must be positive")
        return self.from_sympy(sympy.diff(self.to_sympy(expression), self._symbol(variable), order))

    def substitute(
        self,
        expression: ExpressionIRV1,
        replacements: Mapping[ExpressionIRV1, ExpressionIRV1],
    ) -> ExpressionIRV1:
        mapping = {self.to_sympy(key): self.to_sympy(value) for key, value in replacements.items()}
        return self.from_sympy(self.to_sympy(expression).subs(mapping, simultaneous=True))

    def expand(self, expression: ExpressionIRV1) -> ExpressionIRV1:
        return self._apply(sympy.expand, expression)

    def collect(self, expression: ExpressionIRV1, generators: Sequence[ExpressionIRV1]) -> ExpressionIRV1:
        sympy_expression = self.to_sympy(expression)
        for generator in generators:
            sympy_expression = sympy.collect(sympy_expression, self.to_sympy(generator), exact=True)
        return self.from_sympy(sympy_expression)

    def factor(self, expression: ExpressionIRV1) -> ExpressionIRV1:
        return self._apply(sympy.factor, expression)

    def cancel(self, expression: ExpressionIRV1) -> CASOperationResultV1:
        value = self.to_sympy(expression)
        _, denominator = sympy.fraction(value, exact=True)
        result = self.from_sympy(sympy.cancel(value))
        conditions = () if denominator == 1 else (self.from_sympy(denominator),)
        return CASOperationResultV1(result, conditions)

    def together(self, expression: ExpressionIRV1) -> ExpressionIRV1:
        return self._apply(sympy.together, expression)

    def factor_terms(self, expression: ExpressionIRV1) -> ExpressionIRV1:
        return self._apply(sympy.factor_terms, expression)

    def prove_zero(self, expression: ExpressionIRV1) -> ProofResultV1:
        value = sympy.cancel(sympy.together(sympy.expand(self.to_sympy(expression))))
        normalized = self.from_sympy(value)
        if value == 0:
            return ProofResultV1(ProofStatusV1.PROVED_ZERO, normalized, "named_pipeline_exact_zero")
        if value.is_zero is False:
            return ProofResultV1(ProofStatusV1.PROVED_NONZERO, normalized, "sympy_is_zero_false")
        return ProofResultV1(ProofStatusV1.UNKNOWN, normalized, "zero_status_undetermined")

    def solve_parameters(
        self,
        equations: Sequence[ExpressionIRV1],
        parameters: Sequence[str],
        branch_cap: int,
    ) -> BranchResultV1:
        if isinstance(branch_cap, bool) or not isinstance(branch_cap, int) or branch_cap < 1:
            raise ValueError("branch_cap must be positive")
        equations = tuple(equations)
        zero_statuses = tuple(self.prove_zero(equation).status for equation in equations)
        if not equations or all(
            status is ProofStatusV1.PROVED_ZERO for status in zero_statuses
        ):
            return BranchResultV1(
                BranchKindV1.FINITE_BRANCHES,
                ({},),
                BranchCoverageV1.EXHAUSTIVE,
                None,
                1,
                branch_cap,
            )
        symbols = tuple(self._symbol(name) for name in parameters)
        try:
            raw = sympy.solve(
                [self.to_sympy(equation) for equation in equations],
                symbols,
                dict=True,
                simplify=False,
                check=True,
            )
        except Exception:
            return BranchResultV1(
                BranchKindV1.UNKNOWN,
                (),
                BranchCoverageV1.UNKNOWN,
                "solver_exception",
                0,
                branch_cap,
            )
        if raw is None:
            return BranchResultV1(
                BranchKindV1.UNKNOWN,
                (),
                BranchCoverageV1.UNKNOWN,
                "solver_returned_none",
                0,
                branch_cap,
            )
        if raw == []:
            return BranchResultV1(
                BranchKindV1.INCONSISTENT,
                (),
                BranchCoverageV1.EXHAUSTIVE,
                None,
                0,
                branch_cap,
            )
        capped = len(raw) > branch_cap
        converted = tuple(
            {
                str(key): self.from_sympy(value)
                for key, value in sorted(branch.items(), key=lambda item: str(item[0]))
            }
            for branch in raw[:branch_cap]
        )
        validated: list[dict[str, ExpressionIRV1]] = []
        unknown_revalidations = 0
        for branch in converted:
            replacements = {
                symbol(name, SymbolRoleV1.PARAMETER): value
                for name, value in branch.items()
            }
            statuses = tuple(
                self.prove_zero(self.substitute(equation, replacements)).status
                for equation in equations
            )
            if all(status is ProofStatusV1.PROVED_ZERO for status in statuses):
                validated.append(branch)
            elif not any(
                status is ProofStatusV1.PROVED_NONZERO for status in statuses
            ):
                unknown_revalidations += 1
        conditional = any(
            any(self.to_sympy(item).free_symbols - set(symbols) for item in branch.values())
            for branch in validated
        )
        if capped:
            return BranchResultV1(
                BranchKindV1.FINITE_BRANCHES,
                tuple(validated),
                BranchCoverageV1.PARTIAL,
                "branch_cap_reached",
                len(converted),
                branch_cap,
            )
        if unknown_revalidations:
            return BranchResultV1(
                BranchKindV1.UNKNOWN,
                tuple(validated),
                BranchCoverageV1.UNKNOWN,
                "branch_revalidation_unknown",
                len(converted),
                branch_cap,
            )
        if not validated:
            return BranchResultV1(
                BranchKindV1.INCONSISTENT,
                (),
                BranchCoverageV1.EXHAUSTIVE,
                None,
                len(converted),
                branch_cap,
            )
        return BranchResultV1(
            BranchKindV1.CONDITIONAL if conditional else BranchKindV1.FINITE_BRANCHES,
            tuple(validated),
            BranchCoverageV1.UNKNOWN if conditional else BranchCoverageV1.EXHAUSTIVE,
            "free_symbols_remain" if conditional else None,
            len(converted),
            branch_cap,
        )


@dataclass(frozen=True)
class AffineLieReductionV2:
    """A fully replayed reduction obtained from an affine Lie generator."""

    candidate_id: str
    ansatz: str
    generator: str
    invariants: tuple[str, ...]
    determining_equations: tuple[str, ...]
    reduced_equations: tuple[str, ...]
    free_parameters: tuple[str, ...]
    side_conditions: tuple[str, ...]
    symmetry_dimension: int
    basis_degree: int
    validation_passed: bool


@dataclass(frozen=True)
class _AffineSystem:
    coordinates: tuple[sympy.Symbol, ...]
    fields: tuple[AppliedUndef, ...]
    residuals: tuple[sympy.Expr, ...]
    problem_parameters: tuple[sympy.Symbol, ...]
    generator_unknowns: tuple[sympy.Symbol, ...]
    multiplier_unknowns: tuple[sympy.Symbol, ...]
    coordinate_terms: tuple[tuple[sympy.Expr, ...], ...]
    field_terms: tuple[tuple[sympy.Expr, ...], ...]
    coordinate_unknowns: tuple[tuple[sympy.Symbol, ...], ...]
    field_unknowns: tuple[tuple[sympy.Symbol, ...], ...]
    determining_matrix: sympy.Matrix
    determining_equations: tuple[sympy.Expr, ...]


def _safe_parameter_names(
    count: int, forbidden: set[str], *, prefix: str = "lambda"
) -> tuple[str, ...]:
    names: list[str] = []
    index = 1
    while len(names) < count:
        candidate = f"{prefix}_{index}"
        index += 1
        if candidate not in forbidden:
            names.append(candidate)
            forbidden.add(candidate)
    return tuple(names)


def _jet_atoms(expression: sympy.Expr, fields: Sequence[AppliedUndef]) -> tuple[sympy.Expr, ...]:
    field_set = set(fields)
    found: set[sympy.Expr] = set()
    for item in expression.atoms(sympy.Derivative):
        if item.expr in field_set:
            found.add(item)
    for field in fields:
        if expression.has(field):
            found.add(field)
    return tuple(sorted(found, key=sympy.default_sort_key))


def _held_partial(
    expression: sympy.Expr,
    target: sympy.Expr,
    fields: Sequence[AppliedUndef],
) -> sympy.Expr:
    atoms = _jet_atoms(expression, fields)
    held = {atom: sympy.Dummy(f"held_{index}") for index, atom in enumerate(atoms)}
    reverse = {value: key for key, value in held.items()}
    replaced = expression.xreplace(held)
    derivative_target = held[target] if target in held else target  # noqa: SIM401
    return sympy.diff(replaced, derivative_target).xreplace(reverse)


def _prolonged_action(
    residual: sympy.Expr,
    coordinates: Sequence[sympy.Symbol],
    fields: Sequence[AppliedUndef],
    coordinate_components: Sequence[sympy.Expr],
    field_components: Sequence[sympy.Expr],
) -> sympy.Expr:
    action = sympy.S.Zero
    for coordinate, component in zip(coordinates, coordinate_components):
        action += component * _held_partial(residual, coordinate, fields)
    for field, field_component in zip(fields, field_components):
        characteristic = field_component - sum(
            component * sympy.diff(field, coordinate)
            for coordinate, component in zip(coordinates, coordinate_components)
        )
        for atom in _jet_atoms(residual, (field,)):
            if atom == field:
                coefficient = field_component
            else:
                coefficient = sympy.diff(characteristic, *atom.variables) + sum(
                    component * sympy.Derivative(field, *atom.variables, coordinate)
                    for coordinate, component in zip(coordinates, coordinate_components)
                )
            action += coefficient * _held_partial(residual, atom, fields)
    return sympy.expand(action.doit())


def _non_polynomial_state_atoms(
    expression: sympy.Expr,
    state_symbols: set[sympy.Symbol],
) -> tuple[sympy.Expr, ...]:
    atoms: set[sympy.Expr] = set(expression.atoms(sympy.Derivative))
    atoms.update(expression.atoms(sympy.Function))
    for power in expression.atoms(sympy.Pow):
        exponent = power.exp
        if power.free_symbols & state_symbols and not (
            exponent.is_Integer and exponent.is_nonnegative
        ):
            atoms.add(power)
    # Outer expressions must be replaced before their contained functions.
    return tuple(sorted(atoms, key=lambda item: (-sympy.count_ops(item), sympy.srepr(item))))


def _split_identity(
    expression: sympy.Expr,
    *,
    coordinates: Sequence[sympy.Symbol],
    fields: Sequence[AppliedUndef],
    unknowns: Sequence[sympy.Symbol],
    problem_parameters: Sequence[sympy.Symbol],
) -> tuple[sympy.Expr, ...] | None:
    numerator = sympy.fraction(sympy.cancel(sympy.together(expression)))[0]
    numerator = sympy.expand(numerator)
    state_symbols = set(coordinates)
    replacement_atoms = _non_polynomial_state_atoms(numerator, state_symbols)
    replacements = {
        atom: sympy.Dummy(f"state_atom_{index}")
        for index, atom in enumerate(replacement_atoms)
    }
    polynomial_expression = numerator.xreplace(replacements)
    excluded = {*unknowns, *problem_parameters}
    generators = tuple(
        sorted(
            (polynomial_expression.free_symbols - excluded),
            key=sympy.default_sort_key,
        )
    )
    try:
        coefficients = (
            sympy.Poly(polynomial_expression, *generators, domain="EX").coeffs()
            if generators
            else [polynomial_expression]
        )
    except PolynomialError:
        return None
    equations: list[sympy.Expr] = []
    for coefficient in coefficients:
        simplified = sympy.cancel(sympy.together(coefficient))
        if simplified != 0:
            equations.append(simplified)
    return tuple(equations)


def _affine_system(problem: DiscoveryProblemInputV1) -> _AffineSystem | None:
    if "..." in problem.independent_variables:
        return None
    parsed = parse_problem(problem)
    backend = SymPyBackendV1.from_parser_context(parsed.context)
    coordinates = tuple(backend._symbol(name) for name in problem.independent_variables)
    fields = tuple(
        sympy.Function(name)(*coordinates) for name in problem.dependent_variables
    )
    residuals = tuple(backend.to_sympy(equation.residual) for equation in parsed.equations)
    problem_parameters = tuple(backend._symbol(name) for name in problem.parameters)
    point_terms: tuple[sympy.Expr, ...] = (sympy.S.One, *coordinates, *fields)

    coordinate_unknowns: list[tuple[sympy.Symbol, ...]] = []
    coordinate_terms: list[tuple[sympy.Expr, ...]] = []
    coordinate_components: list[sympy.Expr] = []
    for component_index, _coordinate in enumerate(coordinates):
        unknown_row = tuple(
            sympy.Symbol(f"__g_x{component_index}_{term_index}")
            for term_index in range(len(point_terms))
        )
        coordinate_unknowns.append(unknown_row)
        coordinate_terms.append(point_terms)
        coordinate_components.append(sum(map(sympy.Mul, unknown_row, point_terms)))

    field_unknowns: list[tuple[sympy.Symbol, ...]] = []
    field_terms: list[tuple[sympy.Expr, ...]] = []
    field_components: list[sympy.Expr] = []
    for component_index, _field in enumerate(fields):
        unknown_row = tuple(
            sympy.Symbol(f"__g_u{component_index}_{term_index}")
            for term_index in range(len(point_terms))
        )
        field_unknowns.append(unknown_row)
        field_terms.append(point_terms)
        field_components.append(sum(map(sympy.Mul, unknown_row, point_terms)))

    generator_unknowns = tuple(
        item for row in (*coordinate_unknowns, *field_unknowns) for item in row
    )
    multiplier_unknowns = tuple(
        sympy.Symbol(f"__mu_{row}_{column}")
        for row in range(len(residuals))
        for column in range(len(residuals))
    )
    all_unknowns = (*generator_unknowns, *multiplier_unknowns)
    actions = tuple(
        _prolonged_action(
            residual,
            coordinates,
            fields,
            coordinate_components,
            field_components,
        )
        for residual in residuals
    )
    identities = []
    for row, action in enumerate(actions):
        multiplier_row = multiplier_unknowns[
            row * len(residuals) : (row + 1) * len(residuals)
        ]
        identities.append(
            sympy.expand(
                action
                - sum(
                    multiplier * residual
                    for multiplier, residual in zip(multiplier_row, residuals)
                )
            )
        )
    determining_equations: list[sympy.Expr] = []
    for identity in identities:
        split = _split_identity(
            identity,
            coordinates=coordinates,
            fields=fields,
            unknowns=all_unknowns,
            problem_parameters=problem_parameters,
        )
        if split is None:
            return None
        determining_equations.extend(split)
    try:
        matrix, right_hand_side = sympy.linear_eq_to_matrix(
            determining_equations, all_unknowns
        )
    except Exception:  # noqa: BLE001 - unsupported CAS identity fails closed.
        return None
    if any(item != 0 for item in right_hand_side):
        return None
    return _AffineSystem(
        coordinates,
        fields,
        residuals,
        problem_parameters,
        generator_unknowns,
        multiplier_unknowns,
        tuple(coordinate_terms),
        tuple(field_terms),
        tuple(coordinate_unknowns),
        tuple(field_unknowns),
        matrix,
        tuple(determining_equations),
    )


def _restricted_kernel(
    system: _AffineSystem,
    allowed_generator_unknowns: Iterable[sympy.Symbol],
) -> tuple[Mapping[sympy.Symbol, sympy.Expr], ...]:
    allowed_generator = tuple(allowed_generator_unknowns)
    allowed = (*allowed_generator, *system.multiplier_unknowns)
    all_unknowns = (*system.generator_unknowns, *system.multiplier_unknowns)
    index = {name: position for position, name in enumerate(all_unknowns)}
    submatrix = system.determining_matrix[:, [index[name] for name in allowed]]
    try:
        raw_basis = submatrix.nullspace(simplify=False)
    except Exception:  # noqa: BLE001 - unsupported CAS nullspace fails closed.
        return ()
    results: list[Mapping[sympy.Symbol, sympy.Expr]] = []
    seen: set[str] = set()
    for vector in raw_basis:
        mapping = {name: sympy.cancel(value) for name, value in zip(allowed, vector)}
        if all(
            (mapping[name] if name in mapping else sympy.S.Zero) == 0  # noqa: SIM401
            for name in allowed_generator
        ):
            continue
        first = next(
            mapping[name] for name in allowed_generator if mapping[name] != 0
        )
        normalized = {name: sympy.cancel(value / first) for name, value in mapping.items()}
        key = sympy.srepr(sympy.Tuple(*(normalized[name] for name in allowed_generator)))
        if key not in seen:
            seen.add(key)
            results.append(normalized)
    return tuple(results)


def _linear_combination(
    basis: Sequence[Mapping[sympy.Symbol, sympy.Expr]],
    all_unknowns: Sequence[sympy.Symbol],
    forbidden_names: set[str],
) -> tuple[Mapping[sympy.Symbol, sympy.Expr], tuple[str, ...]]:
    if len(basis) == 1:
        return basis[0], ()
    parameter_names = _safe_parameter_names(len(basis) - 1, forbidden_names)
    parameters = tuple(sympy.Symbol(name) for name in parameter_names)
    coefficients = (sympy.S.One, *parameters)
    vector = {
        name: sympy.cancel(
            sum(
                coefficient
                * (item[name] if name in item else sympy.S.Zero)  # noqa: SIM401
                for coefficient, item in zip(coefficients, basis)
            )
        )
        for name in all_unknowns
    }
    return vector, parameter_names


def _generator_components(
    system: _AffineSystem,
    vector: Mapping[sympy.Symbol, sympy.Expr],
) -> tuple[tuple[sympy.Expr, ...], tuple[sympy.Expr, ...]]:
    coordinates = tuple(
        sympy.cancel(
            sum(
                (vector[unknown] if unknown in vector else sympy.S.Zero)  # noqa: SIM401
                * term
                for unknown, term in zip(unknowns, terms)
            )
        )
        for unknowns, terms in zip(system.coordinate_unknowns, system.coordinate_terms)
    )
    fields = tuple(
        sympy.cancel(
            sum(
                (vector[unknown] if unknown in vector else sympy.S.Zero)  # noqa: SIM401
                * term
                for unknown, term in zip(unknowns, terms)
            )
        )
        for unknowns, terms in zip(system.field_unknowns, system.field_terms)
    )
    return coordinates, fields


def _coordinate_anchor_first(
    system: _AffineSystem,
    basis: Sequence[Mapping[sympy.Symbol, sympy.Expr]],
) -> tuple[Mapping[sympy.Symbol, sympy.Expr], ...]:
    """Choose a projective anchor that moves the last reducible coordinate."""

    components = [_generator_components(system, item)[0] for item in basis]
    pivot = next(
        (
            index
            for index in range(len(system.coordinates) - 1, -1, -1)
            if any(row[index] != 0 for row in components)
        ),
        None,
    )
    if pivot is None:
        return tuple(basis)
    anchor = next(index for index, row in enumerate(components) if row[pivot] != 0)
    return (
        basis[anchor],
        *(item for index, item in enumerate(basis) if index != anchor),
    )


def _orbit_factor_removed(
    expression: sympy.Expr,
    orbit: sympy.Symbol,
    invariants: Sequence[sympy.Symbol],
) -> sympy.Expr | None:
    expression = sympy.factor_terms(sympy.cancel(sympy.together(expression)))
    if orbit not in expression.free_symbols:
        return expression
    if expression == 0:
        return expression
    logarithmic_weight = sympy.cancel(sympy.diff(expression, orbit) / expression)
    weight = sympy.cancel(logarithmic_weight * orbit)
    forbidden = {orbit, *invariants}
    if weight.free_symbols & forbidden:
        return None
    reduced = sympy.simplify(
        sympy.powsimp(sympy.cancel(expression / orbit**weight), force=True)
    )
    return reduced if orbit not in reduced.free_symbols else None


def _replay_characteristic_reduction(
    system: _AffineSystem,
    coordinate_components: Sequence[sympy.Expr],
    field_components: Sequence[sympy.Expr],
    *,
    scaling: bool,
) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...], bool] | None:
    pivot_index = next(
        (
            index
            for index in range(len(system.coordinates) - 1, -1, -1)
            if coordinate_components[index] != 0
        ),
        None,
    )
    if pivot_index is None:
        return None
    pivot = system.coordinates[pivot_index]
    pivot_component = coordinate_components[pivot_index]
    invariant_coordinates = tuple(
        coordinate
        for index, coordinate in enumerate(system.coordinates)
        if index != pivot_index
    )
    eta_symbols = tuple(
        sympy.Symbol(f"eta_{index + 1}", real=True)
        for index in range(len(invariant_coordinates))
    )
    profile_functions = tuple(
        sympy.Function(f"F_{index + 1}") for index in range(len(system.fields))
    )
    orbit = sympy.Symbol("orbit", positive=True)
    side_conditions: set[str] = set()

    def require_nonzero(expression: sympy.Expr) -> None:
        simplified = sympy.factor(expression)
        if simplified.is_number and simplified != 0:
            return
        coefficient, primitive = simplified.as_coeff_Mul()
        if coefficient != 0:
            simplified = primitive
        if simplified not in {sympy.S.One, -sympy.S.One}:
            side_conditions.add(f"{sympy.sstr(simplified)}!=0")

    def record_denominator(expression: sympy.Expr) -> None:
        require_nonzero(sympy.denom(sympy.cancel(expression)))

    invariants_original: list[sympy.Expr] = []
    inverse_coordinates: dict[sympy.Symbol, sympy.Expr] = {pivot: orbit}
    if scaling:
        pivot_weight = sympy.cancel(pivot_component / pivot)
        if pivot_weight == 0 or pivot_weight.has(*system.coordinates, *system.fields):
            return None
        require_nonzero(pivot_weight)
        side_conditions.add(f"{pivot}>0")
        for eta, coordinate in zip(eta_symbols, invariant_coordinates):
            index = system.coordinates.index(coordinate)
            weight = sympy.cancel(coordinate_components[index] / coordinate)
            if weight.has(*system.coordinates, *system.fields):
                return None
            ratio = sympy.cancel(weight / pivot_weight)
            record_denominator(ratio)
            invariants_original.append(sympy.cancel(coordinate / pivot**ratio))
            inverse_coordinates[coordinate] = eta * orbit**ratio
        profiles = tuple(
            function(*invariants_original) for function in profile_functions
        )
        ansatz_values = []
        for field, component, profile in zip(system.fields, field_components, profiles):
            field_weight = sympy.cancel(component / field)
            if field_weight.has(*system.coordinates, *system.fields):
                return None
            exponent = sympy.cancel(field_weight / pivot_weight)
            record_denominator(exponent)
            ansatz_values.append(sympy.cancel(pivot**exponent * profile))
    else:
        require_nonzero(pivot_component)
        for eta, coordinate in zip(eta_symbols, invariant_coordinates):
            index = system.coordinates.index(coordinate)
            ratio = sympy.cancel(coordinate_components[index] / pivot_component)
            record_denominator(ratio)
            invariants_original.append(sympy.cancel(coordinate - ratio * pivot))
            inverse_coordinates[coordinate] = eta + ratio * orbit
        profiles = tuple(
            function(*invariants_original) for function in profile_functions
        )
        ansatz_values = [
            sympy.cancel(
                component / pivot_component * pivot
                + profile
            )
            for component, profile in zip(field_components, profiles)
        ]

    invariant_surface = tuple(
        sympy.cancel(
            component.xreplace(
                {item: value for item, value in zip(system.fields, ansatz_values)}
            )
            - sum(
                coordinate_component * sympy.diff(ansatz, coordinate)
                for coordinate, coordinate_component in zip(
                    system.coordinates, coordinate_components
                )
            )
        )
        for component, ansatz in zip(field_components, ansatz_values)
    )
    if any(sympy.simplify(item.doit()) != 0 for item in invariant_surface):
        return None

    reduced_residuals: list[str] = []
    for residual in system.residuals:
        substituted = residual.xreplace(
            {field: ansatz for field, ansatz in zip(system.fields, ansatz_values)}
        ).doit()
        transformed = sympy.simplify(
            substituted.subs(inverse_coordinates, simultaneous=True).doit()
        )
        reduced = _orbit_factor_removed(transformed, orbit, eta_symbols)
        if reduced is None:
            return None
        reduced_residuals.append(sympy.sstr(sympy.factor_terms(reduced)))

    lhs_arguments = ",".join(map(str, system.coordinates))
    ansatz = ";".join(
        f"{field.func}({lhs_arguments})={sympy.sstr(value)}"
        for field, value in zip(system.fields, ansatz_values)
    )
    invariant_strings = tuple(
        f"eta_{index + 1}={sympy.sstr(value)}"
        for index, value in enumerate(invariants_original)
    )
    return (
        ansatz,
        invariant_strings,
        tuple(reduced_residuals),
        tuple(sorted(side_conditions)),
        True,
    )


def _generator_text(
    system: _AffineSystem,
    coordinate_components: Sequence[sympy.Expr],
    field_components: Sequence[sympy.Expr],
) -> str:
    terms = [
        f"({sympy.sstr(component)})*d/d{coordinate}"
        for coordinate, component in zip(system.coordinates, coordinate_components)
        if component != 0
    ]
    terms.extend(
        f"({sympy.sstr(component)})*d/d{field.func}"
        for field, component in zip(system.fields, field_components)
        if component != 0
    )
    return "X=" + "+".join(terms)


def _semantic_problem_hash(problem: DiscoveryProblemInputV1) -> str:
    payload = {
        "dependent_variables": list(problem.dependent_variables),
        "independent_variables": list(problem.independent_variables),
        "pde": list(problem.pde),
        "parameter_names": sorted(problem.parameters),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _make_reduction(
    problem: DiscoveryProblemInputV1,
    system: _AffineSystem,
    vector: Mapping[sympy.Symbol, sympy.Expr],
    *,
    scaling: bool,
    free_parameters: tuple[str, ...],
    symmetry_dimension: int,
) -> AffineLieReductionV2 | None:
    ordered_unknowns = (*system.generator_unknowns, *system.multiplier_unknowns)
    kernel_remainder = system.determining_matrix * sympy.Matrix(
        [
            vector[name] if name in vector else sympy.S.Zero  # noqa: SIM401
            for name in ordered_unknowns
        ]
    )
    if any(sympy.simplify(item) != 0 for item in kernel_remainder):
        return None
    coordinate_components, field_components = _generator_components(system, vector)
    replay = _replay_characteristic_reduction(
        system,
        coordinate_components,
        field_components,
        scaling=scaling,
    )
    if replay is None:
        return None
    ansatz, invariants, reduced_equations, side_conditions, passed = replay
    generator = _generator_text(
        system, coordinate_components, field_components
    )
    determining = tuple(
        sympy.sstr(item) for item in system.determining_equations
    )
    identity_payload = {
        "problem": _semantic_problem_hash(problem),
        "generator": generator,
        "invariants": list(invariants),
        "ansatz": ansatz,
        "reduced_equations": list(reduced_equations),
        "side_conditions": list(side_conditions),
    }
    candidate_id = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return AffineLieReductionV2(
        candidate_id,
        ansatz,
        generator,
        invariants,
        determining,
        reduced_equations,
        free_parameters,
        side_conditions,
        symmetry_dimension,
        1,
        passed,
    )


def _translation_unknowns(system: _AffineSystem) -> tuple[sympy.Symbol, ...]:
    # point_terms[0] is the constant monomial; fields are deliberately fixed.
    return tuple(row[0] for row in system.coordinate_unknowns)


def _diagonal_scaling_unknowns(system: _AffineSystem) -> tuple[sympy.Symbol, ...]:
    coordinate_count = len(system.coordinates)
    result = [
        row[1 + index] for index, row in enumerate(system.coordinate_unknowns)
    ]
    result.extend(
        row[1 + coordinate_count + index]
        for index, row in enumerate(system.field_unknowns)
    )
    return tuple(result)


def _projective_generator_key(
    system: _AffineSystem,
    reduction: AffineLieReductionV2,
) -> str:
    del system
    return reduction.generator


def discover_affine_lie_reductions(
    problem: DiscoveryProblemInputV1,
    *,
    top_k: int = 5,
) -> tuple[AffineLieReductionV2, ...]:
    """Infer up to ``top_k`` certified reductions from affine determining equations."""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    try:
        system = _affine_system(problem)
    except Exception:  # noqa: BLE001 - public discovery must fail closed.
        return ()
    if system is None:
        return ()
    forbidden_names = {
        *problem.dependent_variables,
        *problem.independent_variables,
        *problem.parameters,
    }
    all_unknowns = (*system.generator_unknowns, *system.multiplier_unknowns)
    reductions: list[AffineLieReductionV2] = []

    searches = (
        (_translation_unknowns(system), False),
        (_diagonal_scaling_unknowns(system), True),
    )
    for allowed, scaling in searches:
        basis = _restricted_kernel(system, allowed)
        if not basis:
            continue
        basis = _coordinate_anchor_first(system, basis)
        generic_vector, generic_parameters = _linear_combination(
            basis, all_unknowns, set(forbidden_names)
        )
        generic = _make_reduction(
            problem,
            system,
            generic_vector,
            scaling=scaling,
            free_parameters=generic_parameters,
            symmetry_dimension=len(basis),
        )
        if generic is not None:
            reductions.append(generic)
        if len(basis) > 1:
            for vector in basis:
                item = _make_reduction(
                    problem,
                    system,
                    vector,
                    scaling=scaling,
                    free_parameters=(),
                    symmetry_dimension=1,
                )
                if item is not None:
                    reductions.append(item)

    unique: dict[str, AffineLieReductionV2] = {}
    for item in reductions:
        unique.setdefault(_projective_generator_key(system, item), item)
    ranked = sorted(
        unique.values(),
        key=lambda item: (
            -item.symmetry_dimension,
            len(item.generator),
            item.candidate_id,
        ),
    )
    return tuple(ranked[:top_k])


__all__ = [
    "AffineLieReductionV2",
    "SymPyBackendV1",
    "discover_affine_lie_reductions",
]
