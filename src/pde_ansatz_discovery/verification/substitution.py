"""Substitute an Ansatz for field jets after differentiating the full field expression."""

from __future__ import annotations

from ..cas.backend import CASBackendV1
from ..ir import ExpressionIRV1, ExpressionKindV1


def field_jets(expression: ExpressionIRV1, field: str) -> tuple[ExpressionIRV1, ...]:
    found: dict[bytes, ExpressionIRV1] = {}

    def visit(node: ExpressionIRV1) -> None:
        if node.kind is ExpressionKindV1.JET and node.jet.field == field:
            found[node.canonical_bytes] = node
        for operand in node.operands:
            visit(operand)

    visit(expression)
    return tuple(found[key] for key in sorted(found))


def substitute_field_ansatz(
    residual: ExpressionIRV1,
    *,
    field: str,
    field_expression: ExpressionIRV1,
    backend: CASBackendV1,
) -> ExpressionIRV1:
    replacements: dict[ExpressionIRV1, ExpressionIRV1] = {}
    for field_jet in field_jets(residual, field):
        replacement = field_expression
        for variable in field_jet.jet.multi_index.variables:
            replacement = backend.differentiate(replacement, variable)
        replacements[field_jet] = replacement
    return backend.expand(backend.substitute(residual, replacements))


__all__ = ["field_jets", "substitute_field_ansatz"]
