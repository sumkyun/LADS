"""Certificate-first direct single-profile PDE reduction verifier."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..cas.backend import CASBackendV1
from ..cas.sympy_backend import SymPyBackendV1
from ..generation.direct import DirectAnsatzV1
from ..ir import (
    ExpressionIRV1,
    ParserContextV1,
    SymbolRoleV1,
    divide,
    mul,
    parse_expression,
    parse_problem,
    subtract,
    symbol,
)
from ..models import (
    BranchCoverageV1,
    CandidateVerificationV1,
    CoefficientDependenceResultV1,
    ConditionCompatibilityV1,
    DependenceStatusV1,
    DiscoveryCandidateV1,
    DiscoveryProblemInputV1,
    ProofStatusV1,
    ReductionCertificateV1,
    VerificationStatusV1,
)
from ..serialization import canonical_sha256
from .charts import construct_reduction_chart, prove_nonzero_on_conditions
from .coefficient_ratio import factor_reduction
from .jet_collection import (
    UnsupportedProfileDependenceV1,
    collect_profile_polynomial,
    reconstruct_profile_polynomial,
)
from .replay import (
    _replay_candidate_certificate,
    certificate_replay_digest,
    dependence_replay_digest,
)
from .substitution import substitute_field_ansatz

_PIPELINE_VERSION = "DirectReductionVerifierV1"


def _candidate_id(problem: DiscoveryProblemInputV1, ansatz: DirectAnsatzV1) -> str:
    return canonical_sha256(
        {
            "kind": "DirectReductionCandidateV1",
            "problem": problem.to_mapping(),
            "ansatz": ansatz.to_mapping(),
            "branch": ansatz.branch.to_mapping(),
        }
    )


def _candidate_context(
    problem: DiscoveryProblemInputV1, ansatz: DirectAnsatzV1
) -> ParserContextV1:
    parameters = tuple(dict.fromkeys((*problem.parameters, *ansatz.free_parameters)))
    return ParserContextV1(
        (),
        tuple(dict.fromkeys((*problem.independent_variables, ansatz.coordinate_name))),
        parameters,
        (ansatz.profile_function,),
    )


def _default_backend(
    problem: DiscoveryProblemInputV1, ansatz: DirectAnsatzV1
) -> SymPyBackendV1:
    roles: dict[str, SymbolRoleV1] = {
        name: SymbolRoleV1.INDEPENDENT for name in problem.independent_variables
    }
    roles[ansatz.coordinate_name] = SymbolRoleV1.PROFILE_COORDINATE
    roles.update(
        {
            name: SymbolRoleV1.PARAMETER
            for name in dict.fromkeys((*problem.parameters, *ansatz.free_parameters))
        }
    )
    return SymPyBackendV1(
        problem.independent_variables,
        roles,
        (ansatz.profile_function,),
    )


def _base_candidate(
    problem: DiscoveryProblemInputV1,
    ansatz: DirectAnsatzV1,
    verification: CandidateVerificationV1,
    *,
    reduction_key: str,
    metadata: dict[str, Any],
) -> DiscoveryCandidateV1:
    return DiscoveryCandidateV1(
        candidate_id=_candidate_id(problem, ansatz),
        ansatz=ansatz.expression,
        coordinate=ansatz.coordinate_expression,
        profile_functions=(ansatz.profile_function,),
        free_parameters=ansatz.free_parameters,
        branch=ansatz.branch,
        production_trace=(f"direct:{ansatz.family}", "verification:direct_v1"),
        failure_history=()
        if verification.status is VerificationStatusV1.CERTIFIED_VALID
        else (verification.reason_code,),
        verification=verification,
        condition_compatibility=ConditionCompatibilityV1(()),
        generator_key=f"direct:{ansatz.family}",
        invariant_key=canonical_sha256(
            {"coordinate": ansatz.coordinate_expression, "domain": ansatz.domain_conditions}
        ),
        reduction_key=reduction_key,
        equivalence_witness=None,
        metadata=metadata,
    )


def _unresolved(
    problem: DiscoveryProblemInputV1,
    ansatz: DirectAnsatzV1,
    reason: str,
    *,
    detail: str | None = None,
) -> DiscoveryCandidateV1:
    return _base_candidate(
        problem,
        ansatz,
        CandidateVerificationV1(VerificationStatusV1.UNRESOLVED, reason, None),
        reduction_key=f"unresolved:{reason}",
        metadata={
            "ansatz_definition": ansatz.to_mapping(),
            "pipeline_version": _PIPELINE_VERSION,
            "detail": detail,
        },
    )


def _coefficient_dependence(
    polynomial,
    *,
    ansatz: DirectAnsatzV1,
    context: ParserContextV1,
    inverse: tuple[tuple[str, ExpressionIRV1], ...],
    coordinate: ExpressionIRV1,
    backend: CASBackendV1,
) -> tuple[tuple[CoefficientDependenceResultV1, ...], list[dict[str, Any]], bool, bool]:
    if not polynomial.terms:
        return (), [], False, False
    pivot_term = min(
        polynomial.terms,
        key=lambda term: (
            term.coefficient.node_count,
            term.coefficient.canonical_bytes,
            term.monomial,
        ),
    )
    inverse_substitutions = {
        parse_expression(name, context): expression for name, expression in inverse
    }
    eta = symbol(ansatz.coordinate_name, SymbolRoleV1.PROFILE_COORDINATE)
    results: list[CoefficientDependenceResultV1] = []
    exact_ir: list[dict[str, Any]] = []
    found_not_dependence = False
    found_unknown = False
    for term in polynomial.terms:
        cancelled = backend.cancel(divide(term.coefficient, pivot_term.coefficient))
        ratio = backend.factor_terms(cancelled.expression)
        rho = backend.factor_terms(backend.substitute(ratio, inverse_substitutions))
        side_conditions = tuple(
            dict.fromkeys(
                (
                    *ansatz.domain_conditions,
                    *(f"{backend.render(item)}!=0" for item in cancelled.side_conditions),
                )
            )
        )
        if ansatz.complementary_coordinate in rho.free_names:
            results.append(
                CoefficientDependenceResultV1(
                    DependenceStatusV1.PROVED_NOT_DEPENDENCE,
                    backend.render(ratio),
                    None,
                    None,
                    side_conditions,
                    None,
                )
            )
            exact_ir.append(
                {
                    "ratio": ratio.to_mapping(),
                    "rho": rho.to_mapping(),
                    "cancellation_conditions": [
                        item.to_mapping() for item in cancelled.side_conditions
                    ],
                }
            )
            found_not_dependence = True
            continue
        reconstructed = backend.substitute(rho, {eta: coordinate})
        proof = backend.prove_zero(subtract(ratio, reconstructed))
        if proof.status is not ProofStatusV1.PROVED_ZERO:
            results.append(
                CoefficientDependenceResultV1(
                    DependenceStatusV1.UNKNOWN,
                    backend.render(ratio),
                    backend.render(rho),
                    None,
                    side_conditions,
                    None,
                )
            )
            exact_ir.append(
                {
                    "ratio": ratio.to_mapping(),
                    "rho": rho.to_mapping(),
                    "cancellation_conditions": [
                        item.to_mapping() for item in cancelled.side_conditions
                    ],
                }
            )
            found_unknown = True
            continue
        identity = (
            f"{backend.render(ratio)}="
            f"{backend.render(rho)}|{ansatz.coordinate_name}={ansatz.coordinate_expression}"
        )
        record_ir = {
            "ratio": ratio.to_mapping(),
            "rho": rho.to_mapping(),
            "cancellation_conditions": [
                item.to_mapping() for item in cancelled.side_conditions
            ],
        }
        provisional = CoefficientDependenceResultV1(
            DependenceStatusV1.PROVED_DEPENDENCE,
            backend.render(ratio),
            backend.render(rho),
            identity,
            side_conditions,
            "0" * 64,
        )
        digest = dependence_replay_digest(provisional.to_mapping(), record_ir)
        results.append(replace(provisional, replay_digest=digest))
        exact_ir.append(record_ir)
    return tuple(results), exact_ir, found_not_dependence, found_unknown


def verify_direct_candidate(
    problem: DiscoveryProblemInputV1,
    ansatz: DirectAnsatzV1,
    *,
    backend: CASBackendV1 | None = None,
) -> DiscoveryCandidateV1:
    """Verify one explicit direct candidate and return a three-way typed verdict."""

    if ansatz.branch.coverage is not BranchCoverageV1.EXHAUSTIVE:
        return _unresolved(problem, ansatz, "BRANCH_COVERAGE_INCOMPLETE")
    if (
        len(problem.dependent_variables) != 1
        or len(problem.independent_variables) != 2
        or len(problem.pde) != 1
        or ansatz.field != problem.dependent_variables[0]
    ):
        return _unresolved(problem, ansatz, "UNSUPPORTED_DIRECT_PROBLEM")

    context = _candidate_context(problem, ansatz)
    cas = backend or _default_backend(problem, ansatz)
    try:
        parsed_problem = parse_problem(problem)
        field_expression = parse_expression(ansatz.expression, context)
        coordinate = parse_expression(ansatz.coordinate_expression, context)
        substituted = substitute_field_ansatz(
            parsed_problem.equations[0].residual,
            field=ansatz.field,
            field_expression=field_expression,
            backend=cas,
        )
        chart_result = construct_reduction_chart(
            ansatz,
            independent_variables=problem.independent_variables,
            context=context,
            backend=cas,
        )
        if chart_result.chart is None:
            return _unresolved(problem, ansatz, chart_result.reason_code)

        polynomial = collect_profile_polynomial(
            substituted, profile=ansatz.profile_function, backend=cas
        )
        reconstruction = cas.prove_zero(
            subtract(substituted, reconstruct_profile_polynomial(polynomial))
        )
        if reconstruction.status is not ProofStatusV1.PROVED_ZERO:
            return _unresolved(problem, ansatz, "PROFILE_COLLECTION_REPLAY_FAILED")
        for _, generator in polynomial.generators:
            if len(generator.operands) != 1:
                return _unresolved(problem, ansatz, "PROFILE_ARGUMENT_UNSUPPORTED")
            if cas.prove_zero(subtract(generator.operands[0], coordinate)).status is not ProofStatusV1.PROVED_ZERO:
                return _unresolved(problem, ansatz, "PROFILE_ARGUMENT_MISMATCH")

        dependence, dependence_ir, not_dependence, unknown_dependence = _coefficient_dependence(
            polynomial,
            ansatz=ansatz,
            context=context,
            inverse=chart_result.inverse_expressions,
            coordinate=coordinate,
            backend=cas,
        )
        if unknown_dependence:
            return _unresolved(problem, ansatz, "COEFFICIENT_DESCENT_UNKNOWN")

        inverse_substitutions = {
            parse_expression(name, context): expression
            for name, expression in chart_result.inverse_expressions
        }
        chart_residual = cas.factor(
            cas.cancel(
                cas.expand(cas.substitute(substituted, inverse_substitutions))
            ).expression
        )
        factorization = factor_reduction(
            chart_residual,
            complementary_coordinate=ansatz.complementary_coordinate,
            old_variables=problem.independent_variables,
            profile=ansatz.profile_function,
            backend=cas,
        )
        base_nonzero_conditions = tuple(
            dict.fromkeys(
                (
                    *ansatz.domain_conditions,
                    *ansatz.branch.assumptions,
                    *ansatz.branch.side_conditions,
                )
            )
        )
        if not prove_nonzero_on_conditions(
            factorization.prefactor,
            base_nonzero_conditions,
            context=context,
            backend=cas,
        ):
            return _unresolved(problem, ansatz, "PREFACTOR_NONZERO_UNPROVED")
        cancellation_conditions = tuple(
            dict.fromkeys(
                f"{cas.render(expression)}!=0"
                for expression in factorization.cancellation_conditions
            )
        )
        identity = cas.prove_zero(
            subtract(
                factorization.transformed_residual,
                mul(factorization.prefactor, factorization.reduced_equation),
            )
        )
        if identity.status is ProofStatusV1.UNKNOWN:
            return _unresolved(problem, ansatz, "REDUCTION_IDENTITY_UNKNOWN")
        if identity.status is not ProofStatusV1.PROVED_ZERO:
            return _unresolved(problem, ansatz, "REDUCTION_IDENTITY_FAILED")

        old_variables_removed = not factorization.remaining_old_variables
        valid = old_variables_removed and not not_dependence
        status = (
            VerificationStatusV1.CERTIFIED_VALID
            if valid
            else VerificationStatusV1.CERTIFIED_INVALID
        )
        reason = "EXACT_REDUCTION" if valid else "COORDINATE_MISMATCH"
        exact_ir = {
            "transformed_residual": factorization.transformed_residual.to_mapping(),
            "prefactor": factorization.prefactor.to_mapping(),
            "reduced_equation": factorization.reduced_equation.to_mapping(),
            "coordinate": coordinate.to_mapping(),
            "coordinate_name": ansatz.coordinate_name,
            "old_variables": list(problem.independent_variables),
            "inverse_map": {
                name: expression.to_mapping()
                for name, expression in chart_result.inverse_expressions
            },
            "coefficient_dependence": dependence_ir,
            "cancellation_conditions": [
                expression.to_mapping()
                for expression in factorization.cancellation_conditions
            ],
            "problem_input_hash": problem.input_hash,
        }
        versions = {
            "backend": cas.backend_name,
            "backend_version": cas.backend_version,
            "normalization": getattr(cas, "normalization_version", "unknown"),
            "expression_ir": ExpressionIRV1.schema_version,
            "pipeline": _PIPELINE_VERSION,
        }
        provisional = ReductionCertificateV1(
            substituted_residual=cas.render(factorization.transformed_residual),
            prefactor=cas.render(factorization.prefactor),
            reduced_equations=(cas.render(factorization.reduced_equation),),
            old_variables_removed=old_variables_removed,
            allowed_symbols=tuple(
                dict.fromkeys(
                    (
                        ansatz.coordinate_name,
                        *ansatz.free_parameters,
                        ansatz.profile_function,
                        *(
                            signature.generator_name
                            for signature, _ in polynomial.generators
                            if signature.order > 0
                        ),
                    )
                )
            ),
            parameter_conditions=ansatz.branch.assumptions,
            side_conditions=tuple(
                dict.fromkeys(
                    (
                        *ansatz.domain_conditions,
                        *ansatz.branch.side_conditions,
                        *cancellation_conditions,
                    )
                )
            ),
            chart=chart_result.chart,
            coefficient_dependence=dependence,
            replay_digest="0" * 64,
            versions=versions,
        )
        digest = certificate_replay_digest(
            provisional, exact_ir, ansatz.branch.to_mapping()
        )
        certificate = replace(provisional, replay_digest=digest)
        candidate = _base_candidate(
            problem,
            ansatz,
            CandidateVerificationV1(status, reason, certificate),
            reduction_key=canonical_sha256(
                {
                    "equation": factorization.reduced_equation.to_mapping(),
                    "conditions": certificate.side_conditions,
                }
            ),
            metadata={
                "ansatz_definition": ansatz.to_mapping(),
                "certificate_ir": exact_ir,
                "problem_input": problem.to_mapping(),
                "problem_input_hash": problem.input_hash,
                "pipeline_version": _PIPELINE_VERSION,
            },
        )
        replay = _replay_candidate_certificate(
            candidate, problem, cas, require_generator_provenance=False
        )
        if not replay.valid:
            return _unresolved(problem, ansatz, replay.reason_code)
        return candidate
    except UnsupportedProfileDependenceV1 as exc:
        return _unresolved(
            problem, ansatz, "UNSUPPORTED_PROFILE_DEPENDENCE", detail=str(exc)
        )
    except Exception as exc:
        return _unresolved(problem, ansatz, "CAS_OPERATION_FAILED", detail=type(exc).__name__)


__all__ = ["verify_direct_candidate"]
