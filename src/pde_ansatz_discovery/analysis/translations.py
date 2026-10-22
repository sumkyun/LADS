"""Detect coordinate-translation terminals from PDE IR, not family labels."""

from __future__ import annotations

from dataclasses import dataclass

from ..ir import ExpressionIRV1, ExpressionKindV1, SymbolRoleV1


@dataclass(frozen=True)
class TranslationAnalysisV1:
    spatial_variable: str
    time_variable: str
    spatial_invariant: bool
    time_invariant: bool

    @property
    def admits_traveling_generator(self) -> bool:
        return self.spatial_invariant and self.time_invariant


def _explicit_independent_symbols(expression: ExpressionIRV1) -> frozenset[str]:
    found: set[str] = set()

    def visit(node: ExpressionIRV1) -> None:
        if (
            node.kind is ExpressionKindV1.SYMBOL
            and node.role is SymbolRoleV1.INDEPENDENT
            and node.name is not None
        ):
            found.add(node.name)
        for operand in node.operands:
            visit(operand)

    visit(expression)
    return frozenset(found)


def analyze_translations(
    equations: tuple[ExpressionIRV1, ...], independent_variables: tuple[str, ...]
) -> TranslationAnalysisV1:
    if len(independent_variables) != 2:
        raise ValueError("translation V1 supports exactly two independent variables")
    explicit = frozenset().union(*(_explicit_independent_symbols(item) for item in equations))
    spatial, time = independent_variables
    return TranslationAnalysisV1(
        spatial,
        time,
        spatial not in explicit,
        time not in explicit,
    )


__all__ = ["TranslationAnalysisV1", "analyze_translations"]
