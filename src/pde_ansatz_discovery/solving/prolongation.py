"""Bounded jet prolongation for scalar 1+1 point generators."""

from __future__ import annotations

from dataclasses import dataclass

from ..cas.backend import CASBackendV1
from ..ir import (
    ExpressionIRV1,
    ExpressionKindV1,
    add,
    jet,
    mul,
    power,
    rational,
    subtract,
)


@dataclass(frozen=True)
class PointGeneratorV1:
    time_component: ExpressionIRV1
    spatial_component: ExpressionIRV1
    field_component: ExpressionIRV1
    field: str
    spatial_variable: str
    time_variable: str


@dataclass(frozen=True)
class ProlongationV1:
    generator: PointGeneratorV1
    jet_coefficients: tuple[tuple[tuple[str, ...], ExpressionIRV1], ...]

    def coefficient(self, variables: tuple[str, ...]) -> ExpressionIRV1:
        return dict(self.jet_coefficients)[variables]


def expression_polynomial_degree(expression: ExpressionIRV1) -> int | None:
    """Return the total point-variable degree, or ``None`` outside the basis."""

    if expression.kind is ExpressionKindV1.RATIONAL:
        return 0
    if expression.kind is ExpressionKindV1.SYMBOL:
        return 0 if expression.role.value == "parameter" else 1
    if expression.kind is ExpressionKindV1.JET:
        return 1 if not expression.jet.multi_index.variables else None
    if expression.kind is ExpressionKindV1.ADD:
        degrees = tuple(expression_polynomial_degree(item) for item in expression.operands)
        return None if any(item is None for item in degrees) else max(degrees, default=0)
    if expression.kind is ExpressionKindV1.MUL:
        degrees = tuple(expression_polynomial_degree(item) for item in expression.operands)
        return None if any(item is None for item in degrees) else sum(degrees)
    if expression.kind is ExpressionKindV1.POW:
        base, exponent = expression.operands
        base_degree = expression_polynomial_degree(base)
        if (
            base_degree is None
            or exponent.kind is not ExpressionKindV1.RATIONAL
            or exponent.denominator != 1
            or exponent.numerator < 0
        ):
            return None
        return base_degree * exponent.numerator
    return None


def point_generator_basis_degree(generator: PointGeneratorV1) -> int | None:
    """Return the maximum total polynomial degree of a point generator."""

    degrees = tuple(
        expression_polynomial_degree(component)
        for component in (
            generator.time_component,
            generator.spatial_component,
            generator.field_component,
        )
    )
    return None if any(item is None for item in degrees) else max(degrees, default=0)


def formal_partial(
    expression: ExpressionIRV1, target: ExpressionIRV1
) -> ExpressionIRV1:
    if expression == target:
        return rational(1)
    if expression.kind in {
        ExpressionKindV1.RATIONAL,
        ExpressionKindV1.SYMBOL,
        ExpressionKindV1.JET,
    }:
        return rational(0)
    if expression.kind is ExpressionKindV1.ADD:
        return add(*(formal_partial(item, target) for item in expression.operands))
    if expression.kind is ExpressionKindV1.MUL:
        summands = []
        for index, operand in enumerate(expression.operands):
            derivative = formal_partial(operand, target)
            factors = tuple(
                derivative if position == index else item
                for position, item in enumerate(expression.operands)
            )
            summands.append(mul(*factors))
        return add(*summands)
    if expression.kind is ExpressionKindV1.POW:
        base, exponent = expression.operands
        if target in exponent.operands or exponent == target:
            raise ValueError("formal differentiation of symbolic exponents is unsupported")
        return mul(
            exponent,
            power(base, subtract(exponent, rational(1))),
            formal_partial(base, target),
        )
    if expression.kind is ExpressionKindV1.FUNCTION:
        raise ValueError("formal differentiation through coefficient functions is unsupported")
    if expression.kind is ExpressionKindV1.DERIVATIVE:
        return rational(0)
    raise ValueError(f"unsupported formal derivative node: {expression.kind.value}")


def prolong_generator(
    generator: PointGeneratorV1,
    required_multi_indices: tuple[tuple[str, ...], ...],
    *,
    backend: CASBackendV1,
) -> ProlongationV1:
    coefficients: dict[tuple[str, ...], ExpressionIRV1] = {
        (): generator.field_component
    }
    coordinate_components = {
        generator.spatial_variable: generator.spatial_component,
        generator.time_variable: generator.time_component,
    }
    needed: set[tuple[str, ...]] = {()}
    for multi_index in required_multi_indices:
        for length in range(1, len(multi_index) + 1):
            needed.add(multi_index[:length])
    for multi_index in sorted(needed, key=lambda item: (len(item), item)):
        if not multi_index:
            continue
        prefix, derivative_variable = multi_index[:-1], multi_index[-1]
        coefficient = backend.differentiate(
            coefficients[prefix], derivative_variable
        )
        for coordinate, component in coordinate_components.items():
            correction = mul(
                jet(generator.field, prefix + (coordinate,)),
                backend.differentiate(component, derivative_variable),
            )
            coefficient = subtract(coefficient, correction)
        coefficients[multi_index] = backend.factor_terms(backend.expand(coefficient))
    return ProlongationV1(
        generator,
        tuple((index, coefficients[index]) for index in sorted(needed, key=lambda item: (len(item), item))),
    )


__all__ = [
    "PointGeneratorV1",
    "ProlongationV1",
    "formal_partial",
    "expression_polynomial_degree",
    "point_generator_basis_degree",
    "prolong_generator",
]
