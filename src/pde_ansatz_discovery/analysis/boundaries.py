"""Boundary-preservation checks and bounded affine Dirichlet lifts."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..cas.backend import CASBackendV1
from ..ir import ParserContextV1, mul, parse_expression
from ..models import ProofStatusV1

_DIRICHLET = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_]*)\(\s*([^,]+)\s*,\s*([^)]+)\)\s*=\s*(.+?)\s*$"
)


@dataclass(frozen=True)
class BoundaryPreservationResultV1:
    preserved: bool | None
    reason_code: str
    tangency_expression: str


@dataclass(frozen=True)
class BoundaryLiftV1:
    expression: str
    homogeneous_factor: str
    left_condition: str
    right_condition: str
    method: str = "affine_dirichlet_lift_v1"


def check_boundary_preservation(
    boundary_expression: str,
    generator_component: str,
    *,
    spatial_variable: str,
    boundary_value: str,
    context: ParserContextV1,
    backend: CASBackendV1,
) -> BoundaryPreservationResultV1:
    try:
        boundary = parse_expression(boundary_expression, context)
        component = parse_expression(generator_component, context)
        tangency = backend.factor_terms(
            mul(component, backend.differentiate(boundary, spatial_variable))
        )
        restricted = backend.substitute(
            tangency,
            {
                parse_expression(spatial_variable, context): parse_expression(
                    boundary_value, context
                )
            },
        )
        proof = backend.prove_zero(restricted)
        if proof.status is ProofStatusV1.PROVED_ZERO:
            return BoundaryPreservationResultV1(
                True, "BOUNDARY_PRESERVED", backend.render(restricted)
            )
        if proof.status is ProofStatusV1.PROVED_NONZERO:
            return BoundaryPreservationResultV1(
                False, "BOUNDARY_VIOLATED", backend.render(restricted)
            )
        return BoundaryPreservationResultV1(
            None, "BOUNDARY_PRESERVATION_UNKNOWN", backend.render(restricted)
        )
    except Exception:
        return BoundaryPreservationResultV1(
            None, "BOUNDARY_PRESERVATION_FAILED", "unavailable"
        )


def build_affine_dirichlet_lift(
    boundary_conditions: tuple[str, ...],
    *,
    field: str,
    spatial_variable: str,
) -> BoundaryLiftV1 | None:
    parsed = []
    for condition in boundary_conditions:
        match = _DIRICHLET.fullmatch(condition)
        if match is not None and match.group(1) == field:
            parsed.append((match.group(2).strip(), match.group(4).strip(), condition))
    if len(parsed) != 2:
        return None
    (left_x, left_value, left_source), (right_x, right_value, right_source) = parsed
    if left_x == right_x:
        return None
    denominator = f"({right_x})-({left_x})"
    expression = (
        f"({right_x}-{spatial_variable})/({denominator})*({left_value})"
        f"+({spatial_variable}-{left_x})/({denominator})*({right_value})"
    )
    homogeneous_factor = (
        f"({spatial_variable}-({left_x}))*(({right_x})-{spatial_variable})"
    )
    return BoundaryLiftV1(
        expression, homogeneous_factor, left_source, right_source
    )


def check_boundary_preservation_for_conditions(
    boundary_conditions: tuple[str, ...],
    generator_component: str,
    *,
    spatial_variable: str,
    context: ParserContextV1,
    backend: CASBackendV1,
) -> tuple[BoundaryPreservationResultV1, ...]:
    results: list[BoundaryPreservationResultV1] = []
    for condition in boundary_conditions:
        match = _DIRICHLET.fullmatch(condition)
        if match is None:
            results.append(
                BoundaryPreservationResultV1(
                    None, "BOUNDARY_SHAPE_UNSUPPORTED", "unavailable"
                )
            )
            continue
        results.append(
            check_boundary_preservation(
                spatial_variable,
                generator_component,
                spatial_variable=spatial_variable,
                boundary_value=match.group(2).strip(),
                context=context,
                backend=backend,
            )
        )
    return tuple(results)


__all__ = [
    "BoundaryLiftV1",
    "BoundaryPreservationResultV1",
    "build_affine_dirichlet_lift",
    "check_boundary_preservation",
    "check_boundary_preservation_for_conditions",
]
