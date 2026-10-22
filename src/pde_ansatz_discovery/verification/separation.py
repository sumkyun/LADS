"""Replayable product/additive separation for linear scalar evolution PDEs."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

from ..cas.backend import CASBackendV1
from ..cas.sympy_backend import SymPyBackendV1
from ..generation.separation import SeparationAnsatzV1, SeparationFamilyV1
from ..ir import (
    ExpressionIRV1,
    ExpressionKindV1,
    ParserContextV1,
    SymbolRoleV1,
    add,
    divide,
    jet,
    mul,
    negate,
    parse_expression,
    parse_problem,
    rational,
    subtract,
)
from ..models import (
    BranchCoverageV1,
    CandidateVerificationV1,
    ConditionCompatibilityV1,
    DiscoveryCandidateV1,
    DiscoveryProblemInputV1,
    ParameterBranchV1,
    ProofStatusV1,
    ReductionCertificateV1,
    ReductionChartV1,
    VerificationStatusV1,
)
from ..serialization import canonical_sha256, thaw_json
from .charts import prove_nonzero_on_conditions
from .replay import certificate_replay_digest, replay_candidate_certificate
from .substitution import substitute_field_ansatz


def separation_candidate_id(
    problem: DiscoveryProblemInputV1,
    request: SeparationAnsatzV1,
    branch: ParameterBranchV1,
) -> str:
    return canonical_sha256(
        {
            "kind": "SeparationCandidateV1",
            "problem": problem.to_mapping(),
            "ansatz": request.to_mapping(),
            "branch": branch.to_mapping(),
        }
    )


def _term_coefficient(
    term: ExpressionIRV1, target: ExpressionIRV1
) -> ExpressionIRV1 | None:
    if term == target:
        return rational(1)
    if term.kind is not ExpressionKindV1.MUL:
        return None
    matches = [index for index, factor in enumerate(term.operands) if factor == target]
    if len(matches) != 1:
        return None
    remaining = tuple(
        factor for index, factor in enumerate(term.operands) if index != matches[0]
    )
    if any(factor.kind is ExpressionKindV1.JET for factor in remaining):
        return None
    return mul(*remaining)


def _linear_heat_coefficients(
    residual: ExpressionIRV1,
    *,
    field: str,
    spatial: str,
    time: str,
    backend: CASBackendV1,
) -> tuple[ExpressionIRV1, ExpressionIRV1] | None:
    expanded = backend.expand(residual)
    terms = expanded.operands if expanded.kind is ExpressionKindV1.ADD else (expanded,)
    targets = (jet(field, (time,)), jet(field, (spatial, spatial)))
    coefficients: list[ExpressionIRV1] = []
    for target in targets:
        found = [
            coefficient
            for term in terms
            if (coefficient := _term_coefficient(term, target)) is not None
        ]
        if len(found) != 1:
            return None
        coefficients.append(found[0])
    reconstructed = add(
        mul(coefficients[0], targets[0]), mul(coefficients[1], targets[1])
    )
    if backend.prove_zero(subtract(expanded, reconstructed)).status is not ProofStatusV1.PROVED_ZERO:
        return None
    return coefficients[0], coefficients[1]


def _problem_domain_conditions(problem: DiscoveryProblemInputV1) -> tuple[str, ...]:
    """Translate the closed V1 interval domain schema into replayable conditions."""

    conditions: list[str] = []
    for name, raw in thaw_json(problem.domain).items():
        if not isinstance(raw, dict):
            continue
        component = raw.get("component")
        if component == "strictly_positive":
            conditions.append(f"{name}>0")
        elif component == "strictly_negative":
            conditions.append(f"{name}<0")
        lower = raw.get("lower")
        if isinstance(lower, (int, float)) and not isinstance(lower, bool):
            operator = ">" if raw.get("lower_open") is True else ">="
            conditions.append(f"{name}{operator}{lower}")
        upper = raw.get("upper")
        if isinstance(upper, (int, float)) and not isinstance(upper, bool):
            operator = "<" if raw.get("upper_open") is True else "<="
            conditions.append(f"{name}{operator}{upper}")
    return tuple(dict.fromkeys((*conditions, *problem.additional_constraints)))


def _equation_uses_only(
    expression: ExpressionIRV1,
    *,
    allowed: set[str],
) -> bool:
    return set(expression.free_names) <= allowed


def _terminal_candidate(
    problem: DiscoveryProblemInputV1,
    request: SeparationAnsatzV1,
    reason: str,
    status: VerificationStatusV1,
) -> DiscoveryCandidateV1:
    branch = ParameterBranchV1("generic", {}, (), (), BranchCoverageV1.EXHAUSTIVE)
    return DiscoveryCandidateV1(
        separation_candidate_id(problem, request, branch),
        request.family.value,
        "x|t",
        (request.spatial_profile, request.temporal_profile),
        (request.separation_constant, *problem.parameters),
        branch,
        (f"separation:{request.family.value}", "verification:separation_v1"),
        (reason,),
        CandidateVerificationV1(status, reason, None),
        ConditionCompatibilityV1(()),
        f"separation:{request.family.value}",
        f"separation:{request.family.value}",
        f"invalid:{reason}",
        None,
        {"pipeline_version": "SeparationVerifierV1"},
    )


def _unresolved_candidate(
    problem: DiscoveryProblemInputV1,
    request: SeparationAnsatzV1,
    reason: str,
) -> DiscoveryCandidateV1:
    return _terminal_candidate(problem, request, reason, VerificationStatusV1.UNRESOLVED)


def _verify_separation_candidate(
    problem: DiscoveryProblemInputV1,
    request: SeparationAnsatzV1,
    *,
    backend: CASBackendV1 | None = None,
) -> DiscoveryCandidateV1:
    if (
        len(problem.dependent_variables) != 1
        or len(problem.independent_variables) != 2
        or len(problem.pde) != 1
        or request.field != problem.dependent_variables[0]
    ):
        return _unresolved_candidate(problem, request, "SEPARATION_PROBLEM_UNSUPPORTED")
    if request.family not in {
        SeparationFamilyV1.PRODUCT,
        SeparationFamilyV1.ADDITIVE,
    }:
        return _unresolved_candidate(problem, request, "SEPARATION_TRANSFORM_UNSUPPORTED")

    spatial, time = problem.independent_variables
    field = problem.dependent_variables[0]
    roles = {spatial: SymbolRoleV1.INDEPENDENT, time: SymbolRoleV1.INDEPENDENT}
    roles.update({name: SymbolRoleV1.PARAMETER for name in problem.parameters})
    roles[request.separation_constant] = SymbolRoleV1.SEPARATION_CONSTANT
    functions = (
        request.spatial_profile,
        request.temporal_profile,
    )
    if backend is None:
        backend = cast(
            CASBackendV1,
            SymPyBackendV1((spatial, time), roles, functions),
        )
    coefficients = _linear_heat_coefficients(
        parse_problem(problem).equations[0].residual,
        field=field,
        spatial=spatial,
        time=time,
        backend=backend,
    )
    if coefficients is None:
        return _unresolved_candidate(problem, request, "SEPARATION_EXTRACTION_UNSUPPORTED")
    time_coefficient, spatial_coefficient = coefficients
    diffusivity_reduction = backend.cancel(
        negate(divide(spatial_coefficient, time_coefficient))
    )
    diffusivity = diffusivity_reduction.expression

    context = ParserContextV1(
        (),
        (spatial, time),
        tuple(dict.fromkeys((*problem.parameters, request.separation_constant))),
        functions,
    )
    x_profile = parse_expression(f"{request.spatial_profile}({spatial})", context)
    t_profile = parse_expression(f"{request.temporal_profile}({time})", context)
    x_second = backend.differentiate(x_profile, spatial, 2)
    t_first = backend.differentiate(t_profile, time)
    separation_constant = parse_expression(request.separation_constant, context)
    if request.family is SeparationFamilyV1.PRODUCT:
        ansatz_expression = mul(x_profile, t_profile)
        time_equation = add(t_first, mul(separation_constant, t_profile))
        space_equation = add(
            mul(diffusivity, x_second), mul(separation_constant, x_profile)
        )
        identity_sum = add(
            mul(x_profile, time_equation), negate(mul(t_profile, space_equation))
        )
        ansatz_display = f"{request.spatial_profile}({spatial})*{request.temporal_profile}({time})"
    else:
        ansatz_expression = add(x_profile, t_profile)
        time_equation = subtract(t_first, separation_constant)
        space_equation = subtract(mul(diffusivity, x_second), separation_constant)
        identity_sum = subtract(time_equation, space_equation)
        prefix = "Identity(" if request.family is SeparationFamilyV1.TRANSFORMED_IDENTITY else ""
        suffix = ")" if prefix else ""
        ansatz_display = f"{prefix}{field}{suffix}={request.spatial_profile}({spatial})+{request.temporal_profile}({time})"

    allowed_time = {
        time,
        request.temporal_profile,
        f"{request.temporal_profile}__d1",
        request.separation_constant,
        *problem.parameters,
    }
    allowed_space = {
        spatial,
        request.spatial_profile,
        f"{request.spatial_profile}__d2",
        request.separation_constant,
        *problem.parameters,
    }
    if not _equation_uses_only(time_equation, allowed=allowed_time) or not _equation_uses_only(
        space_equation, allowed=allowed_space
    ):
        return _unresolved_candidate(
            problem, request, "SEPARATION_VARIABLE_LEAK"
        )

    domain_conditions = _problem_domain_conditions(problem)
    if not prove_nonzero_on_conditions(
        time_coefficient,
        domain_conditions,
        context=context,
        backend=backend,
    ):
        return _unresolved_candidate(
            problem, request, "SEPARATION_PREFACTOR_NONZERO_UNPROVED"
        )
    for condition in diffusivity_reduction.side_conditions:
        if not prove_nonzero_on_conditions(
            condition,
            domain_conditions,
            context=context,
            backend=backend,
        ):
            return _unresolved_candidate(
                problem,
                request,
                "SEPARATION_CANCELLATION_CONDITION_UNPROVED",
            )
    cancellation_conditions = tuple(
        dict.fromkeys(
            f"{backend.render(condition)}!=0"
            for condition in diffusivity_reduction.side_conditions
        )
    )
    side_conditions = tuple(
        dict.fromkeys((*domain_conditions, *cancellation_conditions))
    )

    substituted = substitute_field_ansatz(
        parse_problem(problem).equations[0].residual,
        field=field,
        field_expression=ansatz_expression,
        backend=backend,
    )
    identity = backend.prove_zero(
        subtract(substituted, mul(time_coefficient, identity_sum))
    )
    if identity.status is ProofStatusV1.UNKNOWN:
        return _unresolved_candidate(problem, request, "SEPARATION_IDENTITY_UNKNOWN")
    if identity.status is not ProofStatusV1.PROVED_ZERO:
        return _unresolved_candidate(problem, request, "SEPARATION_IDENTITY_FAILED")

    branch = ParameterBranchV1("generic", {}, (), (), BranchCoverageV1.EXHAUSTIVE)
    chart = ReductionChartV1(
        spatial,
        time,
        {spatial: spatial, time: time},
        {spatial: spatial, time: time},
        domain_conditions,
        "1",
        ProofStatusV1.PROVED_NONZERO,
        (),
    )
    exact_ir = {
        "kind": f"separation_{request.family.value}",
        "transformed_residual": substituted.to_mapping(),
        "prefactor": time_coefficient.to_mapping(),
        "reduced_equations": [time_equation.to_mapping(), space_equation.to_mapping()],
        "identity_sum": identity_sum.to_mapping(),
        "coordinate": parse_expression(spatial, context).to_mapping(),
        "coordinate_name": spatial,
        "old_variables": [],
        "inverse_map": {
            spatial: parse_expression(spatial, context).to_mapping(),
            time: parse_expression(time, context).to_mapping(),
        },
        "coefficient_dependence": [],
        "cancellation_conditions": [
            condition.to_mapping()
            for condition in diffusivity_reduction.side_conditions
        ],
        "domain_conditions": list(domain_conditions),
        "problem_input_hash": problem.input_hash,
    }
    provisional = ReductionCertificateV1(
        backend.render(substituted),
        backend.render(time_coefficient),
        (backend.render(time_equation), backend.render(space_equation)),
        True,
        tuple(
            dict.fromkeys(
                (
                    spatial,
                    time,
                    request.separation_constant,
                    *functions,
                    f"{request.spatial_profile}__d2",
                    f"{request.temporal_profile}__d1",
                    *problem.parameters,
                )
            )
        ),
        (),
        side_conditions,
        chart,
        (),
        "0" * 64,
        {
            "backend": backend.backend_name,
            "backend_version": backend.backend_version,
            "normalization": getattr(backend, "normalization_version", "unknown"),
            "expression_ir": ExpressionIRV1.schema_version,
            "pipeline": "SeparationVerifierV1",
        },
    )
    certificate = replace(
        provisional,
        replay_digest=certificate_replay_digest(
            provisional, exact_ir, branch.to_mapping()
        ),
    )
    candidate = DiscoveryCandidateV1(
        separation_candidate_id(problem, request, branch),
        ansatz_display,
        f"{spatial}|{time}",
        (request.spatial_profile, request.temporal_profile),
        tuple(dict.fromkeys((request.separation_constant, *problem.parameters))),
        branch,
        (f"separation:{request.family.value}", "verification:separation_v1"),
        (),
        CandidateVerificationV1(
            VerificationStatusV1.CERTIFIED_VALID, "EXACT_SEPARATION", certificate
        ),
        ConditionCompatibilityV1(()),
        f"separation:{request.family.value}",
        f"{spatial}|{time}",
        canonical_sha256(
            {"family": request.family.value, "equations": certificate.reduced_equations}
        ),
        None,
        {
            "certificate_ir": exact_ir,
            "problem_input": problem.to_mapping(),
            "problem_input_hash": problem.input_hash,
            "pipeline_version": "SeparationVerifierV1",
            "separation_constant": request.separation_constant,
            "separation_definition": request.to_mapping(),
        },
    )
    replay = replay_candidate_certificate(candidate, problem, backend)
    if not replay.valid:
        return _unresolved_candidate(problem, request, replay.reason_code)
    return candidate


def verify_separation_candidate(
    problem: DiscoveryProblemInputV1,
    request: SeparationAnsatzV1,
    *,
    backend: CASBackendV1 | None = None,
) -> DiscoveryCandidateV1:
    """Verify one separation request and fail closed on operational uncertainty."""

    try:
        return _verify_separation_candidate(problem, request, backend=backend)
    except Exception:  # noqa: BLE001 - backend/parser failures are unresolved
        return _unresolved_candidate(problem, request, "CAS_OPERATION_FAILED")


__all__ = ["separation_candidate_id", "verify_separation_candidate"]
