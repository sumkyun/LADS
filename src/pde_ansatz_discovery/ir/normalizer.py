"""Deterministic bounded normalization for expression IR."""

from __future__ import annotations

from fractions import Fraction

from .expressions import (
    ExpressionIRV1,
    ExpressionKindV1,
    as_fraction,
    is_rational,
    rational,
)


def _key(expression: ExpressionIRV1) -> bytes:
    return expression.canonical_bytes


def add(*expressions: ExpressionIRV1) -> ExpressionIRV1:
    terms: list[ExpressionIRV1] = []
    numeric = Fraction(0)
    for expression in expressions:
        expression = normalize(expression)
        nested = expression.operands if expression.kind is ExpressionKindV1.ADD else (expression,)
        for term in nested:
            if is_rational(term):
                numeric += as_fraction(term)
            else:
                terms.append(term)
    if numeric:
        terms.append(rational(numeric))
    if not terms:
        return rational(0)
    terms.sort(key=_key)
    if len(terms) == 1:
        return terms[0]
    return ExpressionIRV1(ExpressionKindV1.ADD, tuple(terms))


def mul(*expressions: ExpressionIRV1) -> ExpressionIRV1:
    factors: list[ExpressionIRV1] = []
    numeric = Fraction(1)
    for expression in expressions:
        expression = normalize(expression)
        nested = expression.operands if expression.kind is ExpressionKindV1.MUL else (expression,)
        for factor in nested:
            if is_rational(factor):
                numeric *= as_fraction(factor)
            else:
                factors.append(factor)
    if numeric == 0:
        return rational(0)
    if numeric != 1 or not factors:
        factors.append(rational(numeric))
    factors.sort(key=_key)
    if len(factors) == 1:
        return factors[0]
    return ExpressionIRV1(ExpressionKindV1.MUL, tuple(factors))


def power(base: ExpressionIRV1, exponent: ExpressionIRV1) -> ExpressionIRV1:
    base = normalize(base)
    exponent = normalize(exponent)
    if is_rational(exponent):
        value = as_fraction(exponent)
        if value == 0:
            return rational(1)
        if value == 1:
            return base
        if is_rational(base) and value.denominator == 1:
            number = as_fraction(base)
            if value < 0 and number == 0:
                return ExpressionIRV1(ExpressionKindV1.POW, (base, exponent))
            return rational(number ** value.numerator)
    return ExpressionIRV1(ExpressionKindV1.POW, (base, exponent))


def negate(expression: ExpressionIRV1) -> ExpressionIRV1:
    return mul(rational(-1), expression)


def subtract(lhs: ExpressionIRV1, rhs: ExpressionIRV1) -> ExpressionIRV1:
    return add(lhs, negate(rhs))


def divide(lhs: ExpressionIRV1, rhs: ExpressionIRV1) -> ExpressionIRV1:
    if is_rational(lhs) and is_rational(rhs) and as_fraction(rhs) != 0:
        return rational(as_fraction(lhs) / as_fraction(rhs))
    return mul(lhs, power(rhs, rational(-1)))


def normalize(expression: ExpressionIRV1) -> ExpressionIRV1:
    if expression.kind is ExpressionKindV1.ADD:
        return add(*expression.operands)
    if expression.kind is ExpressionKindV1.MUL:
        return mul(*expression.operands)
    if expression.kind is ExpressionKindV1.POW:
        return power(*expression.operands)
    if expression.kind in {ExpressionKindV1.FUNCTION, ExpressionKindV1.DERIVATIVE}:
        operands = tuple(normalize(item) for item in expression.operands)
        if operands == expression.operands:
            return expression
        return ExpressionIRV1(
            expression.kind,
            operands,
            expression.name,
            expression.role,
            expression.numerator,
            expression.denominator,
            expression.jet,
            expression.derivative_variables,
        )
    return expression


__all__ = ["add", "divide", "mul", "negate", "normalize", "power", "subtract"]
