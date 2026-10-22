"""Bounded explicit reduction-chart construction."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..cas.backend import CASBackendV1
from ..generation.direct import DirectAnsatzV1
from ..ir import (
    ExpressionIRV1,
    ExpressionKindV1,
    ParserContextV1,
    SymbolRoleV1,
    mul,
    divide,
    parse_expression,
    subtract,
    symbol,
)
from ..models import ProofStatusV1, ReductionChartV1

_NONZERO_CONDITION = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*(?:!=\s*0|>\s*0|<\s*0)\s*$")
_NONZERO_EXPRESSION = re.compile(r"^\s*\(?(.+?)\)?\s*(?:!=\s*0|>\s*0|<\s*0)\s*$")


@dataclass(frozen=True)
class ChartConstructionResultV1:
    chart: ReductionChartV1 | None
    inverse_expressions: tuple[tuple[str, ExpressionIRV1], ...]
    reason_code: str


def _nonzero_names(conditions: tuple[str, ...]) -> frozenset[str]:
    return frozenset(
        match.group(1)
        for condition in conditions
        if (match := _NONZERO_CONDITION.fullmatch(condition)) is not None
    )


def _proved_nonzero_from_domain(
    expression: ExpressionIRV1,
    nonzero_names: frozenset[str],
    nonzero_expressions: tuple[ExpressionIRV1, ...],
    backend: CASBackendV1,
) -> bool:
    if any(
        backend.prove_zero(subtract(expression, item)).status is ProofStatusV1.PROVED_ZERO
        for item in nonzero_expressions
    ):
        return True
    for item in nonzero_expressions:
        ratio = backend.cancel(divide(expression, item)).expression
        if ratio.kind is ExpressionKindV1.RATIONAL and ratio.numerator != 0:
            return True
    if expression.kind is ExpressionKindV1.RATIONAL:
        return expression.numerator != 0
    if expression.kind is ExpressionKindV1.SYMBOL:
        return expression.name in nonzero_names
    if expression.kind is ExpressionKindV1.MUL:
        return all(
            _proved_nonzero_from_domain(item, nonzero_names, nonzero_expressions, backend)
            for item in expression.operands
        )
    if expression.kind is ExpressionKindV1.POW:
        return _proved_nonzero_from_domain(
            expression.operands[0], nonzero_names, nonzero_expressions, backend
        )
    if expression.kind is ExpressionKindV1.FUNCTION and expression.name == "exp":
        return True
    if expression.kind is ExpressionKindV1.FUNCTION and expression.name in {"sqrt", "Abs"}:
        return _proved_nonzero_from_domain(
            expression.operands[0], nonzero_names, nonzero_expressions, backend
        )
    return False


def prove_nonzero_on_conditions(
    expression: ExpressionIRV1,
    conditions: tuple[str, ...],
    *,
    context: ParserContextV1,
    backend: CASBackendV1,
) -> bool:
    if backend.prove_zero(expression).status is ProofStatusV1.PROVED_NONZERO:
        return True
    domain_expressions: list[ExpressionIRV1] = []
    for condition in conditions:
        match = _NONZERO_EXPRESSION.fullmatch(condition)
        if match is None:
            continue
        try:
            domain_expressions.append(parse_expression(match.group(1).strip(), context))
        except Exception:
            return False
    return _proved_nonzero_from_domain(
        expression,
        _nonzero_names(conditions),
        tuple(domain_expressions),
        backend,
    )


def construct_reduction_chart(
    ansatz: DirectAnsatzV1,
    *,
    independent_variables: tuple[str, ...],
    context: ParserContextV1,
    backend: CASBackendV1,
) -> ChartConstructionResultV1:
    if len(independent_variables) != 2:
        return ChartConstructionResultV1(None, (), "UNSUPPORTED_CHART_DIMENSION")
    if set(ansatz.inverse_map) != set(independent_variables):
        return ChartConstructionResultV1(None, (), "CHART_INVERSE_MISSING")
    if ansatz.complementary_coordinate not in independent_variables:
        return ChartConstructionResultV1(None, (), "CHART_COMPLEMENT_INVALID")

    try:
        eta = symbol(ansatz.coordinate_name, SymbolRoleV1.PROFILE_COORDINATE)
        coordinate = parse_expression(ansatz.coordinate_expression, context)
        inverse = tuple(
            (name, parse_expression(ansatz.inverse_map[name], context))
            for name in independent_variables
        )
        inverse_substitutions = {
            parse_expression(name, context): expression for name, expression in inverse
        }
        forward_after_inverse = backend.substitute(coordinate, inverse_substitutions)
        if backend.prove_zero(subtract(forward_after_inverse, eta)).status is not ProofStatusV1.PROVED_ZERO:
            return ChartConstructionResultV1(None, inverse, "CHART_FORWARD_REPLAY_FAILED")

        eta_from_old = parse_expression(ansatz.coordinate_expression, context)
        eta_key = parse_expression(ansatz.coordinate_name, context)
        for name, expression in inverse:
            composed = backend.substitute(expression, {eta_key: eta_from_old})
            original = parse_expression(name, context)
            if backend.prove_zero(subtract(composed, original)).status is not ProofStatusV1.PROVED_ZERO:
                return ChartConstructionResultV1(None, inverse, "CHART_INVERSE_REPLAY_FAILED")

        first, second = independent_variables
        s = ansatz.complementary_coordinate
        jacobian = subtract(
            mul(
                backend.differentiate(dict(inverse)[first], ansatz.coordinate_name),
                backend.differentiate(dict(inverse)[second], s),
            ),
            mul(
                backend.differentiate(dict(inverse)[first], s),
                backend.differentiate(dict(inverse)[second], ansatz.coordinate_name),
            ),
        )
        jacobian = backend.factor_terms(jacobian)
        proof = backend.prove_zero(jacobian).status
        domain_expressions: list[ExpressionIRV1] = []
        singular_expressions: list[str] = []
        for condition in ansatz.domain_conditions:
            match = _NONZERO_EXPRESSION.fullmatch(condition)
            if match is None:
                continue
            expression_text = match.group(1).strip()
            try:
                domain_expressions.append(parse_expression(expression_text, context))
                singular_expressions.append(f"{expression_text}=0")
            except Exception:
                return ChartConstructionResultV1(
                    None, inverse, "CHART_DOMAIN_PARSE_FAILED"
                )
        if proof is not ProofStatusV1.PROVED_NONZERO and not prove_nonzero_on_conditions(
            jacobian,
            ansatz.domain_conditions,
            context=context,
            backend=backend,
        ):
            return ChartConstructionResultV1(None, inverse, "CHART_REGULARITY_UNPROVED")

        singular_names = sorted(set(singular_expressions))
        chart = ReductionChartV1(
            eta=ansatz.coordinate_expression,
            complementary_coordinate=s,
            forward_map={ansatz.coordinate_name: ansatz.coordinate_expression, s: s},
            inverse_map=dict(ansatz.inverse_map),
            domain_conditions=ansatz.domain_conditions,
            jacobian=backend.render(jacobian),
            jacobian_proof=ProofStatusV1.PROVED_NONZERO,
            singular_set=tuple(singular_names),
        )
        return ChartConstructionResultV1(chart, inverse, "REGULAR_CHART")
    except Exception:
        return ChartConstructionResultV1(None, (), "CHART_CONSTRUCTION_FAILED")


__all__ = [
    "ChartConstructionResultV1",
    "construct_reduction_chart",
    "prove_nonzero_on_conditions",
]
