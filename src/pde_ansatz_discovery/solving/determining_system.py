"""Classical and conditional determining remainders."""

from __future__ import annotations

from dataclasses import dataclass

from ..cas.backend import CASBackendV1
from ..ir import (
    ExpressionIRV1,
    ExpressionKindV1,
    SymbolRoleV1,
    add,
    divide,
    jet,
    mul,
    negate,
    rational,
    symbol,
    subtract,
)
from ..models import ProofStatusV1
from .prolongation import PointGeneratorV1, formal_partial, prolong_generator


@dataclass(frozen=True)
class GeneratorVerificationV1:
    mode: str
    invariant: bool | None
    raw_determining_expression: ExpressionIRV1
    reduced_determining_expression: ExpressionIRV1
    proof_status: ProofStatusV1
    reason_code: str
    invariant_surface_condition: ExpressionIRV1 | None = None


def _formal_polynomial_nonzero(expression: ExpressionIRV1) -> bool:
    if expression.kind in {
        ExpressionKindV1.RATIONAL,
        ExpressionKindV1.SYMBOL,
        ExpressionKindV1.JET,
    }:
        return not (
            expression.kind is ExpressionKindV1.RATIONAL and expression.numerator == 0
        )
    if expression.kind in {ExpressionKindV1.ADD, ExpressionKindV1.MUL}:
        return all(_formal_polynomial_nonzero(item) for item in expression.operands)
    if expression.kind is ExpressionKindV1.POW:
        return (
            expression.operands[1].kind is ExpressionKindV1.RATIONAL
            and _formal_polynomial_nonzero(expression.operands[0])
        )
    return False


def _closed_proof_status(proof) -> ProofStatusV1:
    if proof.status is ProofStatusV1.UNKNOWN and _formal_polynomial_nonzero(proof.normalized):
        return ProofStatusV1.PROVED_NONZERO
    return proof.status


def _field_jets(expression: ExpressionIRV1, field: str) -> tuple[tuple[str, ...], ...]:
    found: set[tuple[str, ...]] = set()

    def visit(node: ExpressionIRV1) -> None:
        if node.kind is ExpressionKindV1.JET and node.jet.field == field:
            found.add(node.jet.multi_index.variables)
        for operand in node.operands:
            visit(operand)

    visit(expression)
    return tuple(sorted(found, key=lambda item: (len(item), item)))


def prolonged_action(
    residual: ExpressionIRV1,
    generator: PointGeneratorV1,
    *,
    backend: CASBackendV1,
) -> ExpressionIRV1:
    indices = _field_jets(residual, generator.field)
    prolongation = prolong_generator(generator, indices, backend=backend)
    action = rational(0)
    coordinates = (
        (generator.spatial_variable, generator.spatial_component),
        (generator.time_variable, generator.time_component),
    )
    for name, component in coordinates:
        action = add(
            action,
            mul(component, formal_partial(residual, symbol(name, SymbolRoleV1.INDEPENDENT))),
        )
    for index in indices:
        action = add(
            action,
            mul(prolongation.coefficient(index), formal_partial(residual, jet(generator.field, index))),
        )
    return backend.factor_terms(backend.expand(action))


def _reduce_on_evolution_equation(
    expression: ExpressionIRV1,
    residual: ExpressionIRV1,
    generator: PointGeneratorV1,
    backend: CASBackendV1,
) -> ExpressionIRV1:
    leading = jet(generator.field, (generator.time_variable,))
    coefficient = backend.factor_terms(formal_partial(residual, leading))
    if backend.prove_zero(coefficient).status is ProofStatusV1.PROVED_ZERO:
        return expression
    remainder = subtract(residual, mul(coefficient, leading))
    solution = backend.cancel(negate(divide(remainder, coefficient))).expression
    return backend.factor_terms(backend.expand(backend.substitute(expression, {leading: solution})))


def verify_classical_generator(
    residual: ExpressionIRV1,
    generator: PointGeneratorV1,
    *,
    backend: CASBackendV1,
) -> GeneratorVerificationV1:
    raw = prolonged_action(residual, generator, backend=backend)
    reduced = _reduce_on_evolution_equation(raw, residual, generator, backend)
    proof = backend.prove_zero(reduced)
    status = _closed_proof_status(proof)
    invariant = True if status is ProofStatusV1.PROVED_ZERO else (
        False if status is ProofStatusV1.PROVED_NONZERO else None
    )
    return GeneratorVerificationV1(
        "classical",
        invariant,
        raw,
        reduced,
        status,
        "CLASSICAL_INVARIANT" if invariant is True else (
            "CLASSICAL_NOT_INVARIANT" if invariant is False else "CLASSICAL_INVARIANCE_UNKNOWN"
        ),
    )


def verify_nonclassical_generator(
    residual: ExpressionIRV1,
    generator: PointGeneratorV1,
    *,
    backend: CASBackendV1,
) -> GeneratorVerificationV1:
    classical = verify_classical_generator(residual, generator, backend=backend)
    u_x = jet(generator.field, (generator.spatial_variable,))
    u_t = jet(generator.field, (generator.time_variable,))
    condition = subtract(
        generator.field_component,
        add(
            mul(generator.time_component, u_t),
            mul(generator.spatial_component, u_x),
        ),
    )
    # Bounded V1 conditional reduction handles the normalized Q = scalar*u_x case.
    q_without_ux = backend.cancel(divide(condition, u_x)).expression
    q_replay = backend.prove_zero(subtract(condition, mul(q_without_ux, u_x)))
    if q_replay.status is not ProofStatusV1.PROVED_ZERO:
        return GeneratorVerificationV1(
            "nonclassical", None, classical.raw_determining_expression,
            classical.reduced_determining_expression, ProofStatusV1.UNKNOWN,
            "INVARIANT_SURFACE_UNSUPPORTED", condition,
        )
    replacements = {
        jet(generator.field, index): rational(0)
        for index in _field_jets(classical.reduced_determining_expression, generator.field)
        if generator.spatial_variable in index
    }
    reduced = backend.factor_terms(
        backend.expand(backend.substitute(classical.reduced_determining_expression, replacements))
    )
    proof = backend.prove_zero(reduced)
    status = _closed_proof_status(proof)
    invariant = True if status is ProofStatusV1.PROVED_ZERO else (
        False if status is ProofStatusV1.PROVED_NONZERO else None
    )
    return GeneratorVerificationV1(
        "nonclassical",
        invariant,
        classical.raw_determining_expression,
        reduced,
        status,
        "NONCLASSICAL_INVARIANT" if invariant is True else (
            "NONCLASSICAL_NOT_INVARIANT" if invariant is False else "NONCLASSICAL_INVARIANCE_UNKNOWN"
        ),
        condition,
    )


__all__ = [
    "GeneratorVerificationV1",
    "prolonged_action",
    "verify_classical_generator",
    "verify_nonclassical_generator",
]
