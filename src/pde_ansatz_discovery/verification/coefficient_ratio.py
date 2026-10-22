"""Bounded prefactor extraction and old-coordinate audit."""

from __future__ import annotations

from dataclasses import dataclass

from ..cas.backend import CASBackendV1
from ..ir import ExpressionIRV1, ExpressionKindV1, divide, mul, rational


@dataclass(frozen=True)
class ReductionFactorizationV1:
    transformed_residual: ExpressionIRV1
    prefactor: ExpressionIRV1
    reduced_equation: ExpressionIRV1
    remaining_old_variables: tuple[str, ...]
    cancellation_conditions: tuple[ExpressionIRV1, ...]


def _contains_profile(node: ExpressionIRV1, profile: str) -> bool:
    if node.kind is ExpressionKindV1.FUNCTION and node.name is not None:
        return node.name == profile or node.name.startswith(profile + "__d")
    return any(_contains_profile(operand, profile) for operand in node.operands)


def factor_reduction(
    expression: ExpressionIRV1,
    *,
    complementary_coordinate: str,
    old_variables: tuple[str, ...],
    profile: str,
    backend: CASBackendV1,
) -> ReductionFactorizationV1:
    transformed = backend.factor_terms(expression)
    factors = (
        transformed.operands
        if transformed.kind is ExpressionKindV1.MUL
        else (transformed,)
    )
    prefactor_factors = tuple(
        factor
        for factor in factors
        if complementary_coordinate in factor.free_names and not _contains_profile(factor, profile)
    )
    prefactor = mul(*prefactor_factors) if prefactor_factors else rational(1)
    cancellation = backend.cancel(divide(transformed, prefactor))
    reduced = cancellation.expression
    remaining = tuple(sorted(set(old_variables) & set(reduced.free_names)))
    return ReductionFactorizationV1(
        transformed,
        prefactor,
        reduced,
        remaining,
        cancellation.side_conditions,
    )


__all__ = ["ReductionFactorizationV1", "factor_reduction"]
