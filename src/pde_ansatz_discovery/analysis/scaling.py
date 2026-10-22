"""Exact homogeneous term-weight analysis for 1+1 scalar PDEs."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ..ir import ExpressionIRV1, ExpressionKindV1, SymbolRoleV1


@dataclass(frozen=True)
class ScalingWeightV1:
    spatial: Fraction
    time: Fraction
    field: Fraction

    def __add__(self, other: "ScalingWeightV1") -> "ScalingWeightV1":
        return ScalingWeightV1(
            self.spatial + other.spatial,
            self.time + other.time,
            self.field + other.field,
        )

    def __sub__(self, other: "ScalingWeightV1") -> "ScalingWeightV1":
        return ScalingWeightV1(
            self.spatial - other.spatial,
            self.time - other.time,
            self.field - other.field,
        )

    def scaled(self, factor: Fraction) -> "ScalingWeightV1":
        return ScalingWeightV1(
            self.spatial * factor, self.time * factor, self.field * factor
        )


@dataclass(frozen=True)
class ScalingAnalysisV1:
    spatial_variable: str
    time_variable: str
    field: str
    term_weights: tuple[ScalingWeightV1, ...]
    balance_equations: tuple[ScalingWeightV1, ...]
    spatial_over_time: Fraction | None
    field_over_time: Fraction | None
    field_weight_free: bool
    reason_code: str

    @property
    def admits_similarity_generator(self) -> bool:
        return self.spatial_over_time is not None and (
            self.field_over_time is not None or self.field_weight_free
        )


_ZERO = ScalingWeightV1(Fraction(0), Fraction(0), Fraction(0))


def _term_weight(
    expression: ExpressionIRV1,
    *,
    spatial: str,
    time: str,
    field: str,
) -> ScalingWeightV1 | None:
    kind = expression.kind
    if kind is ExpressionKindV1.RATIONAL:
        return _ZERO
    if kind is ExpressionKindV1.SYMBOL:
        if expression.role is SymbolRoleV1.PARAMETER:
            return _ZERO
        if expression.name == spatial:
            return ScalingWeightV1(Fraction(1), Fraction(0), Fraction(0))
        if expression.name == time:
            return ScalingWeightV1(Fraction(0), Fraction(1), Fraction(0))
        return None
    if kind is ExpressionKindV1.JET and expression.jet.field == field:
        variables = expression.jet.multi_index.variables
        return ScalingWeightV1(
            Fraction(-variables.count(spatial)),
            Fraction(-variables.count(time)),
            Fraction(1),
        )
    if kind is ExpressionKindV1.MUL:
        total = _ZERO
        for operand in expression.operands:
            weight = _term_weight(operand, spatial=spatial, time=time, field=field)
            if weight is None:
                return None
            total += weight
        return total
    if kind is ExpressionKindV1.POW:
        base, exponent = expression.operands
        if exponent.kind is not ExpressionKindV1.RATIONAL:
            return None
        weight = _term_weight(base, spatial=spatial, time=time, field=field)
        if weight is None:
            return None
        return weight.scaled(Fraction(exponent.numerator, exponent.denominator))
    if kind is ExpressionKindV1.ADD:
        weights = tuple(
            _term_weight(item, spatial=spatial, time=time, field=field)
            for item in expression.operands
        )
        if not weights or any(item is None for item in weights) or len(set(weights)) != 1:
            return None
        return weights[0]
    return None


def _solve_time_normalized(
    equations: tuple[ScalingWeightV1, ...]
) -> tuple[Fraction | None, Fraction | None, bool]:
    # Fix b=1 and solve A*a + C*c = -B with exact two-variable RREF.
    rows = [
        [equation.spatial, equation.field, -equation.time]
        for equation in equations
        if equation != _ZERO
    ]
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in (0, 1):
        selected = next(
            (index for index in range(pivot_row, len(rows)) if rows[index][column]),
            None,
        )
        if selected is None:
            continue
        rows[pivot_row], rows[selected] = rows[selected], rows[pivot_row]
        pivot = rows[pivot_row][column]
        rows[pivot_row] = [item / pivot for item in rows[pivot_row]]
        for index, row in enumerate(rows):
            if index != pivot_row and row[column]:
                factor = row[column]
                rows[index] = [
                    item - factor * pivot_item
                    for item, pivot_item in zip(row, rows[pivot_row])
                ]
        pivot_columns.append(column)
        pivot_row += 1
    if any(not row[0] and not row[1] and row[2] for row in rows):
        return None, None, False
    values: list[Fraction | None] = [None, None]
    for row, column in zip(rows, pivot_columns):
        if all(not row[other] for other in (0, 1) if other != column):
            values[column] = row[2]
    field_free = 1 not in pivot_columns
    return values[0], values[1], field_free


def analyze_scaling(
    equation: ExpressionIRV1,
    *,
    independent_variables: tuple[str, ...],
    field: str,
) -> ScalingAnalysisV1:
    if len(independent_variables) != 2:
        raise ValueError("scaling V1 supports exactly two independent variables")
    spatial, time = independent_variables
    terms = equation.operands if equation.kind is ExpressionKindV1.ADD else (equation,)
    weights = tuple(
        _term_weight(term, spatial=spatial, time=time, field=field) for term in terms
    )
    if len(terms) < 2 or any(weight is None for weight in weights):
        return ScalingAnalysisV1(
            spatial, time, field, (), (), None, None, False, "SCALING_WEIGHT_UNSUPPORTED"
        )
    typed_weights = tuple(weight for weight in weights if weight is not None)
    equations = tuple(weight - typed_weights[0] for weight in typed_weights[1:])
    alpha, gamma, gamma_free = _solve_time_normalized(equations)
    reason = "SCALING_BALANCE_SOLVED" if alpha is not None else "SCALING_BALANCE_UNDERDETERMINED"
    return ScalingAnalysisV1(
        spatial,
        time,
        field,
        typed_weights,
        equations,
        alpha,
        gamma,
        gamma_free,
        reason,
    )


__all__ = ["ScalingAnalysisV1", "ScalingWeightV1", "analyze_scaling"]
