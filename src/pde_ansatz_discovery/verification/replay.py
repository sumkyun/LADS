"""Independent mathematical replay of reduction certificates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..cas.backend import CASBackendV1
from ..generation.direct import DirectAnsatzV1
from ..generation.separation import SeparationAnsatzV1, SeparationFamilyV1
from ..ir import (
    ExpressionIRV1,
    ParserContextV1,
    SymbolRoleV1,
    add,
    divide,
    mul,
    negate,
    parse_expression,
    parse_problem,
    rational,
    subtract,
    symbol,
)
from ..ir.parser import SAFE_FUNCTIONS_V1
from ..models import (
    BranchCoverageV1,
    ConditionCompatibilityV1,
    DependenceStatusV1,
    DiscoveryCandidateV1,
    DiscoveryProblemInputV1,
    ParameterBranchV1,
    ProofStatusV1,
    ReductionCertificateV1,
    ReductionChartV1,
    VerificationStatusV1,
)
from ..serialization import canonical_sha256, thaw_json
from .charts import construct_reduction_chart, prove_nonzero_on_conditions
from .coefficient_ratio import factor_reduction
from .jet_collection import collect_profile_polynomial, reconstruct_profile_polynomial
from .substitution import substitute_field_ansatz


@dataclass(frozen=True)
class CertificateReplayResultV1:
    valid: bool
    reason_code: str
    proof_status: ProofStatusV1


def dependence_replay_digest(
    dependence_mapping: dict[str, Any], exact_ir: dict[str, Any]
) -> str:
    payload = dict(dependence_mapping)
    payload.pop("replay_digest", None)
    return canonical_sha256(
        {"kind": "CoefficientDependenceReplayV1", "record": payload, "exact_ir": exact_ir}
    )


def certificate_replay_digest(
    certificate: ReductionCertificateV1,
    exact_ir: dict[str, Any],
    branch_mapping: dict[str, Any],
) -> str:
    payload = certificate.to_mapping()
    payload.pop("replay_digest")
    return canonical_sha256(
        {
            "kind": "ReductionCertificateReplayV1",
            "certificate": payload,
            "exact_ir": exact_ir,
            "branch": branch_mapping,
        }
    )


def _expression(value: Any) -> ExpressionIRV1:
    if not isinstance(value, dict):
        raise TypeError("certificate IR node is not a mapping")
    return ExpressionIRV1.from_mapping(value)


def _failure(reason: str, status: ProofStatusV1 = ProofStatusV1.PROVED_NONZERO):
    return CertificateReplayResultV1(False, reason, status)


def _direct_context(
    candidate: DiscoveryCandidateV1, ansatz: DirectAnsatzV1
) -> ParserContextV1:
    metadata = thaw_json(candidate.metadata)
    problem_variables = tuple(metadata["certificate_ir"]["old_variables"])
    return ParserContextV1(
        (),
        tuple(dict.fromkeys((*problem_variables, ansatz.coordinate_name))),
        tuple(dict.fromkeys((*candidate.free_parameters, *candidate.branch.assignments))),
        candidate.profile_functions,
    )


def _same_expression(
    left: ExpressionIRV1, right: ExpressionIRV1, backend: CASBackendV1
) -> bool:
    return backend.prove_zero(subtract(left, right)).status is ProofStatusV1.PROVED_ZERO


def _mapped_expression_matches(
    value: Any, expected: ExpressionIRV1, backend: CASBackendV1
) -> bool:
    return isinstance(value, dict) and _same_expression(
        ExpressionIRV1.from_mapping(value), expected, backend
    )


def _classical_generator_key(
    problem: DiscoveryProblemInputV1,
    candidate: DiscoveryCandidateV1,
    ansatz: DirectAnsatzV1,
    metadata: dict[str, Any],
) -> str | None:
    from ..cas.sympy_backend import SymPyBackendV1
    from ..generation.direct import materialized_parameter_branch
    from ..generation.lie import generate_classical_lie_seeds
    from ..solving.determining_system import verify_classical_generator
    from ..solving.prolongation import PointGeneratorV1, point_generator_basis_degree

    lie_metadata = metadata.get("classical_lie")
    if not isinstance(lie_metadata, dict):
        return None
    for seed in generate_classical_lie_seeds(problem):
        if seed.ansatz.family != ansatz.family:
            continue
        parameters = tuple(
            dict.fromkeys((*problem.parameters, *seed.ansatz.free_parameters))
        )
        context = ParserContextV1(
            (),
            tuple(
                dict.fromkeys(
                    (*problem.independent_variables, seed.ansatz.coordinate_name)
                )
            ),
            parameters,
            (seed.ansatz.profile_function,),
        )
        roles = {
            name: SymbolRoleV1.INDEPENDENT for name in problem.independent_variables
        }
        roles[seed.ansatz.coordinate_name] = SymbolRoleV1.PROFILE_COORDINATE
        roles.update({name: SymbolRoleV1.PARAMETER for name in parameters})
        seed_backend = SymPyBackendV1(
            problem.independent_variables, roles, (seed.ansatz.profile_function,)
        )
        assignments = thaw_json(candidate.branch.assignments)
        if any(
            name not in seed.ansatz.free_parameters or not isinstance(value, str)
            for name, value in assignments.items()
        ):
            continue
        replacements = {
            parse_expression(name, context): parse_expression(value, context)
            for name, value in assignments.items()
        }

        def substitute_text(
            value: str,
            seed_backend: CASBackendV1 = seed_backend,
            context: ParserContextV1 = context,
            replacements: dict[ExpressionIRV1, ExpressionIRV1] = replacements,
        ) -> str:
            return seed_backend.render(
                seed_backend.substitute(parse_expression(value, context), replacements)
            )

        if assignments:
            assignment_text = {
                name: seed_backend.render(value)
                for name, value in (
                    (name, parse_expression(text, context))
                    for name, text in assignments.items()
                )
            }
            branch = materialized_parameter_branch(
                assignment_text, seed.ansatz.domain_conditions
            )
            materialized_ansatz = replace(
                seed.ansatz,
                expression=substitute_text(seed.ansatz.expression),
                coordinate_expression=substitute_text(
                    seed.ansatz.coordinate_expression
                ),
                inverse_map={
                    name: substitute_text(value)
                    for name, value in seed.ansatz.inverse_map.items()
                },
                free_parameters=tuple(
                    name
                    for name in seed.ansatz.free_parameters
                    if name not in assignments
                ),
                branch=branch,
            )
        else:
            materialized_ansatz = seed.ansatz
        if materialized_ansatz != ansatz or materialized_ansatz.branch != candidate.branch:
            continue
        generator = PointGeneratorV1(
            seed_backend.substitute(seed.generator.time_component, replacements),
            seed_backend.substitute(seed.generator.spatial_component, replacements),
            seed_backend.substitute(seed.generator.field_component, replacements),
            seed.generator.field,
            seed.generator.spatial_variable,
            seed.generator.time_variable,
        )
        verification = verify_classical_generator(
            parse_problem(problem).equations[0].residual,
            generator,
            backend=seed_backend,
        )
        solver_metadata = lie_metadata.get("parameter_solver")
        if not isinstance(solver_metadata, dict):
            continue
        branch_cap = solver_metadata.get("branch_cap")
        if isinstance(branch_cap, bool) or not isinstance(branch_cap, int):
            continue
        derived_solver = seed_backend.solve_parameters(
            (seed.determining.reduced_determining_expression,),
            seed.solver_parameters,
            branch_cap,
        )
        if (
            verification.invariant is not True
            or verification.proof_status is not ProofStatusV1.PROVED_ZERO
            or derived_solver.to_mapping() != solver_metadata
            or lie_metadata.get("generator_label") != seed.generator_label
            or lie_metadata.get("proof_status") != verification.proof_status.value
            or lie_metadata.get("reason_code") != verification.reason_code
            or lie_metadata.get("basis_degree")
            != point_generator_basis_degree(seed.generator)
            or not _mapped_expression_matches(
                lie_metadata.get("time_component"),
                generator.time_component,
                seed_backend,
            )
            or not _mapped_expression_matches(
                lie_metadata.get("spatial_component"),
                generator.spatial_component,
                seed_backend,
            )
            or not _mapped_expression_matches(
                lie_metadata.get("field_component"),
                generator.field_component,
                seed_backend,
            )
            or not _mapped_expression_matches(
                lie_metadata.get("determining_raw"),
                verification.raw_determining_expression,
                seed_backend,
            )
            or not _mapped_expression_matches(
                lie_metadata.get("determining_reduced"),
                verification.reduced_determining_expression,
                seed_backend,
            )
        ):
            continue
        return f"classical_lie:{seed.generator_label}"
    return None


def _nonclassical_generator_key(
    problem: DiscoveryProblemInputV1,
    ansatz: DirectAnsatzV1,
    metadata: dict[str, Any],
    backend: CASBackendV1,
) -> str | None:
    from ..solving.determining_system import (
        verify_classical_generator,
        verify_nonclassical_generator,
    )
    from ..solving.prolongation import PointGeneratorV1

    field = problem.dependent_variables[0]
    spatial, time = problem.independent_variables
    expected_ansatz = DirectAnsatzV1(
        field,
        f"F({time})",
        "eta",
        time,
        "F",
        tuple(problem.parameters),
        {spatial: spatial, time: "eta"},
        spatial,
        family="nonclassical:x_invariant",
    )
    if ansatz != expected_ansatz:
        return None
    generator = PointGeneratorV1(
        rational(0), rational(1), rational(0), field, spatial, time
    )
    residual = parse_problem(problem).equations[0].residual
    classical = verify_classical_generator(residual, generator, backend=backend)
    conditional = verify_nonclassical_generator(residual, generator, backend=backend)
    classical_metadata = metadata.get("classical")
    conditional_metadata = metadata.get("conditional")
    if (
        conditional.invariant is not True
        or conditional.proof_status is not ProofStatusV1.PROVED_ZERO
        or metadata.get("operator") != f"d_{spatial}"
        or not isinstance(classical_metadata, dict)
        or not isinstance(conditional_metadata, dict)
        or classical_metadata.get("invariant") is not classical.invariant
        or classical_metadata.get("reason_code") != classical.reason_code
        or conditional_metadata.get("invariant") is not conditional.invariant
        or conditional_metadata.get("reason_code") != conditional.reason_code
        or not _mapped_expression_matches(
            classical_metadata.get("determining"),
            classical.reduced_determining_expression,
            backend,
        )
        or not _mapped_expression_matches(
            conditional_metadata.get("determining"),
            conditional.reduced_determining_expression,
            backend,
        )
        or conditional.invariant_surface_condition is None
        or not _mapped_expression_matches(
            conditional_metadata.get("invariant_surface"),
            conditional.invariant_surface_condition,
            backend,
        )
    ):
        return None
    return "nonclassical:d_x"


def _expected_generator_key(
    problem: DiscoveryProblemInputV1,
    candidate: DiscoveryCandidateV1,
    ansatz: DirectAnsatzV1,
    metadata: dict[str, Any],
    backend: CASBackendV1,
) -> str | None:
    if ansatz.family.startswith("classical_lie:"):
        if metadata.get("pipeline_version") != "DirectReductionVerifierV1":
            return None
        return _classical_generator_key(problem, candidate, ansatz, metadata)
    if ansatz.family == "nonclassical:x_invariant":
        if metadata.get("pipeline_version") != "NonclassicalVerifierV1":
            return None
        return _nonclassical_generator_key(problem, ansatz, metadata, backend)
    if (
        metadata.get("pipeline_version") != "DirectReductionVerifierV1"
        or "classical_lie" in metadata
    ):
        return None
    return f"direct:{ansatz.family}"


def _replay_direct_source(
    problem: DiscoveryProblemInputV1,
    candidate: DiscoveryCandidateV1,
    ansatz: DirectAnsatzV1,
    exact_ir: dict[str, Any],
    backend: CASBackendV1,
    *,
    require_generator_provenance: bool,
) -> CertificateReplayResultV1 | None:
    expected_id = canonical_sha256(
        {
            "kind": "DirectReductionCandidateV1",
            "problem": problem.to_mapping(),
            "ansatz": ansatz.to_mapping(),
            "branch": ansatz.branch.to_mapping(),
        }
    )
    if candidate.candidate_id != expected_id:
        return _failure("CANDIDATE_PROBLEM_BINDING_MISMATCH")
    expected_invariant_key = canonical_sha256(
        {
            "coordinate": ansatz.coordinate_expression,
            "domain": ansatz.domain_conditions,
        }
    )
    metadata = thaw_json(candidate.metadata)
    expected_generator_key = (
        _expected_generator_key(problem, candidate, ansatz, metadata, backend)
        if require_generator_provenance
        else f"direct:{ansatz.family}"
    )
    if (
        expected_generator_key is None
        or
        candidate.ansatz != ansatz.expression
        or candidate.coordinate != ansatz.coordinate_expression
        or candidate.profile_functions != (ansatz.profile_function,)
        or candidate.free_parameters != ansatz.free_parameters
        or candidate.branch != ansatz.branch
        or candidate.invariant_key != expected_invariant_key
        or candidate.generator_key != expected_generator_key
    ):
        return _failure("PUBLIC_CANDIDATE_BINDING_MISMATCH")
    context = _direct_context(candidate, ansatz)
    original_residual = parse_problem(problem).equations[0].residual
    field_expression = parse_expression(ansatz.expression, context)
    substituted = substitute_field_ansatz(
        original_residual,
        field=ansatz.field,
        field_expression=field_expression,
        backend=backend,
    )
    coordinate = parse_expression(ansatz.coordinate_expression, context)
    inverse = tuple(
        (name, parse_expression(ansatz.inverse_map[name], context))
        for name in problem.independent_variables
    )
    inverse_substitutions = {
        parse_expression(name, context): expression for name, expression in inverse
    }
    polynomial = collect_profile_polynomial(
        substituted, profile=ansatz.profile_function, backend=backend
    )
    reconstruction = backend.prove_zero(
        subtract(substituted, reconstruct_profile_polynomial(polynomial))
    )
    if reconstruction.status is not ProofStatusV1.PROVED_ZERO:
        return _failure("SOURCE_PROFILE_COLLECTION_REPLAY_MISMATCH")
    for _, generator in polynomial.generators:
        if len(generator.operands) != 1 or not _same_expression(
            generator.operands[0], coordinate, backend
        ):
            return _failure("SOURCE_PROFILE_ARGUMENT_REPLAY_MISMATCH")

    # Lazy import avoids the reduction<->replay module cycle.  This re-runs the
    # authoritative coefficient extraction over the freshly substituted source
    # PDE; stored certificate IR is compared only after that derivation.
    from .reduction import _coefficient_dependence

    (
        derived_dependence,
        derived_dependence_ir,
        found_not_dependence,
        found_unknown_dependence,
    ) = _coefficient_dependence(
        polynomial,
        ansatz=ansatz,
        context=context,
        inverse=inverse,
        coordinate=coordinate,
        backend=backend,
    )
    certificate = candidate.verification.certificate
    assert certificate is not None
    if (
        found_unknown_dependence
        or derived_dependence != certificate.coefficient_dependence
        or derived_dependence_ir != exact_ir.get("coefficient_dependence")
    ):
        return _failure("SOURCE_COEFFICIENT_DEPENDENCE_REPLAY_MISMATCH")
    chart_residual = backend.factor(
        backend.cancel(
            backend.expand(backend.substitute(substituted, inverse_substitutions))
        ).expression
    )
    factorization = factor_reduction(
        chart_residual,
        complementary_coordinate=ansatz.complementary_coordinate,
        old_variables=problem.independent_variables,
        profile=ansatz.profile_function,
        backend=backend,
    )
    comparisons = (
        (factorization.transformed_residual, _expression(exact_ir["transformed_residual"])),
        (factorization.prefactor, _expression(exact_ir["prefactor"])),
        (factorization.reduced_equation, _expression(exact_ir["reduced_equation"])),
        (coordinate, _expression(exact_ir["coordinate"])),
    )
    if not all(_same_expression(left, right, backend) for left, right in comparisons):
        return _failure("SOURCE_SUBSTITUTION_REPLAY_MISMATCH")
    exact_cancellations = tuple(
        _expression(item) for item in exact_ir.get("cancellation_conditions", ())
    )
    if factorization.cancellation_conditions != exact_cancellations:
        return _failure("SOURCE_CANCELLATION_REPLAY_MISMATCH")
    expected_side_conditions = tuple(
        dict.fromkeys(
            (
                *ansatz.domain_conditions,
                *candidate.branch.side_conditions,
                *(
                    f"{backend.render(item)}!=0"
                    for item in factorization.cancellation_conditions
                ),
            )
        )
    )
    expected_reduction_key = canonical_sha256(
        {
            "equation": factorization.reduced_equation.to_mapping(),
            "conditions": expected_side_conditions,
        }
    )
    if candidate.reduction_key != expected_reduction_key:
        return _failure("PUBLIC_REDUCTION_KEY_MISMATCH")
    expected_valid = (
        not factorization.remaining_old_variables and not found_not_dependence
    )
    expected_status = (
        VerificationStatusV1.CERTIFIED_VALID
        if expected_valid
        else VerificationStatusV1.CERTIFIED_INVALID
    )
    expected_reason = "EXACT_REDUCTION" if expected_valid else "COORDINATE_MISMATCH"
    if (
        candidate.verification.status is not expected_status
        or candidate.verification.reason_code != expected_reason
    ):
        return _failure("PUBLIC_VERIFICATION_BINDING_MISMATCH")
    return None


def _replay_separation_source(
    problem: DiscoveryProblemInputV1,
    candidate: DiscoveryCandidateV1,
    request: SeparationAnsatzV1,
    exact_ir: dict[str, Any],
    backend: CASBackendV1,
) -> CertificateReplayResultV1 | None:
    # Lazy import avoids the replay<->separation module cycle while keeping one
    # authoritative extraction routine for certification and independent replay.
    from .separation import _linear_heat_coefficients, separation_candidate_id

    if request.family not in {
        SeparationFamilyV1.PRODUCT,
        SeparationFamilyV1.ADDITIVE,
    }:
        return _failure("SEPARATION_TRANSFORM_UNSUPPORTED", ProofStatusV1.UNKNOWN)
    expected_branch = ParameterBranchV1(
        "generic", {}, (), (), BranchCoverageV1.EXHAUSTIVE
    )
    if candidate.branch != expected_branch:
        return _failure("BRANCH_BINDING_MISMATCH")
    expected_id = separation_candidate_id(problem, request, candidate.branch)
    if candidate.candidate_id != expected_id:
        return _failure("CANDIDATE_PROBLEM_BINDING_MISMATCH")
    spatial, time = problem.independent_variables
    context = ParserContextV1(
        (),
        (spatial, time),
        tuple(dict.fromkeys((*problem.parameters, request.separation_constant))),
        (request.spatial_profile, request.temporal_profile),
    )
    x_profile = parse_expression(f"{request.spatial_profile}({spatial})", context)
    t_profile = parse_expression(f"{request.temporal_profile}({time})", context)
    ansatz_expression = (
        mul(x_profile, t_profile)
        if request.family is SeparationFamilyV1.PRODUCT
        else add(x_profile, t_profile)
    )
    residual = parse_problem(problem).equations[0].residual
    substituted = substitute_field_ansatz(
        residual,
        field=request.field,
        field_expression=ansatz_expression,
        backend=backend,
    )
    if not _same_expression(substituted, _expression(exact_ir["transformed_residual"]), backend):
        return _failure("SOURCE_SUBSTITUTION_REPLAY_MISMATCH")
    coefficients = _linear_heat_coefficients(
        residual,
        field=request.field,
        spatial=spatial,
        time=time,
        backend=backend,
    )
    if coefficients is None:
        return _failure("SOURCE_SEPARATION_EXTRACTION_FAILED")
    time_coefficient, spatial_coefficient = coefficients
    diffusivity_reduction = backend.cancel(
        negate(divide(spatial_coefficient, time_coefficient))
    )
    diffusivity = diffusivity_reduction.expression
    separation_constant = parse_expression(request.separation_constant, context)
    x_second = backend.differentiate(x_profile, spatial, 2)
    t_first = backend.differentiate(t_profile, time)
    if request.family is SeparationFamilyV1.PRODUCT:
        expected_equations = (
            add(t_first, mul(separation_constant, t_profile)),
            add(mul(diffusivity, x_second), mul(separation_constant, x_profile)),
        )
        expected_identity = add(
            mul(x_profile, expected_equations[0]),
            negate(mul(t_profile, expected_equations[1])),
        )
        expected_ansatz = (
            f"{request.spatial_profile}({spatial})*"
            f"{request.temporal_profile}({time})"
        )
    else:
        expected_equations = (
            subtract(t_first, separation_constant),
            subtract(mul(diffusivity, x_second), separation_constant),
        )
        expected_identity = subtract(expected_equations[0], expected_equations[1])
        prefix = (
            "Identity("
            if request.family is SeparationFamilyV1.TRANSFORMED_IDENTITY
            else ""
        )
        suffix = ")" if prefix else ""
        expected_ansatz = (
            f"{prefix}{request.field}{suffix}="
            f"{request.spatial_profile}({spatial})+"
            f"{request.temporal_profile}({time})"
        )
    certificate = candidate.verification.certificate
    assert certificate is not None
    expected_reduction_key = canonical_sha256(
        {
            "family": request.family.value,
            "equations": certificate.reduced_equations,
        }
    )
    metadata = thaw_json(candidate.metadata)
    if (
        candidate.ansatz != expected_ansatz
        or candidate.coordinate != f"{spatial}|{time}"
        or candidate.profile_functions
        != (request.spatial_profile, request.temporal_profile)
        or candidate.free_parameters
        != tuple(dict.fromkeys((request.separation_constant, *problem.parameters)))
        or candidate.generator_key != f"separation:{request.family.value}"
        or candidate.invariant_key != f"{spatial}|{time}"
        or candidate.reduction_key != expected_reduction_key
        or candidate.production_trace
        != (
            f"separation:{request.family.value}",
            "verification:separation_v1",
        )
        or candidate.failure_history
        or candidate.condition_compatibility != ConditionCompatibilityV1(())
        or metadata.get("pipeline_version") != "SeparationVerifierV1"
        or metadata.get("separation_constant") != request.separation_constant
    ):
        return _failure("PUBLIC_CANDIDATE_BINDING_MISMATCH")
    if (
        candidate.verification.status is not VerificationStatusV1.CERTIFIED_VALID
        or candidate.verification.reason_code != "EXACT_SEPARATION"
    ):
        return _failure("PUBLIC_VERIFICATION_BINDING_MISMATCH")
    expected_coordinate = parse_expression(spatial, context).to_mapping()
    expected_inverse = {
        spatial: parse_expression(spatial, context).to_mapping(),
        time: parse_expression(time, context).to_mapping(),
    }
    if (
        exact_ir.get("kind") != f"separation_{request.family.value}"
        or exact_ir.get("coordinate") != expected_coordinate
        or exact_ir.get("coordinate_name") != spatial
        or exact_ir.get("old_variables") != []
        or exact_ir.get("inverse_map") != expected_inverse
        or exact_ir.get("coefficient_dependence") != []
        or exact_ir.get("problem_input_hash") != problem.input_hash
    ):
        return _failure("SOURCE_SEPARATION_METADATA_MISMATCH")
    stored_equations = tuple(
        _expression(item) for item in exact_ir["reduced_equations"]
    )
    if len(stored_equations) != 2 or not all(
        _same_expression(left, right, backend)
        for left, right in zip(expected_equations, stored_equations)
    ):
        return _failure("SOURCE_SEPARATION_EQUATION_MISMATCH")
    if not _same_expression(time_coefficient, _expression(exact_ir["prefactor"]), backend):
        return _failure("SOURCE_PREFACTOR_REPLAY_MISMATCH")
    if not _same_expression(
        expected_identity, _expression(exact_ir["identity_sum"]), backend
    ):
        return _failure("SOURCE_SEPARATION_IDENTITY_MISMATCH")
    stored_cancellations = tuple(
        _expression(item) for item in exact_ir.get("cancellation_conditions", ())
    )
    if diffusivity_reduction.side_conditions != stored_cancellations:
        return _failure("SOURCE_CANCELLATION_REPLAY_MISMATCH")
    return None


def _reconstruct_separation_chart(
    problem: DiscoveryProblemInputV1,
) -> ReductionChartV1 | None:
    """Rebuild the V1 identity chart only from the public problem contract."""

    if len(problem.independent_variables) != 2:
        return None
    conditions: list[str] = []
    for name, raw in thaw_json(problem.domain).items():
        if not isinstance(raw, dict):
            return None
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
    domain_conditions = tuple(
        dict.fromkeys((*conditions, *problem.additional_constraints))
    )
    spatial, time = problem.independent_variables
    identity_map = {spatial: spatial, time: time}
    return ReductionChartV1(
        spatial,
        time,
        identity_map,
        identity_map,
        domain_conditions,
        "1",
        ProofStatusV1.PROVED_NONZERO,
        (),
    )


def _replay_candidate_certificate(
    candidate: DiscoveryCandidateV1,
    problem: DiscoveryProblemInputV1,
    backend: CASBackendV1,
    *,
    require_generator_provenance: bool,
) -> CertificateReplayResultV1:
    certificate = candidate.verification.certificate
    if certificate is None:
        return _failure("CERTIFICATE_MISSING", ProofStatusV1.UNKNOWN)
    if candidate.branch.coverage is not BranchCoverageV1.EXHAUSTIVE:
        return _failure("BRANCH_COVERAGE_INCOMPLETE", ProofStatusV1.UNKNOWN)
    try:
        metadata = thaw_json(candidate.metadata)
        exact_ir = metadata["certificate_ir"]
        if (
            metadata.get("problem_input_hash") != problem.input_hash
            or metadata.get("problem_input") != problem.to_mapping()
        ):
            return _failure("PROBLEM_INPUT_HASH_MISMATCH")
        if certificate_replay_digest(
            certificate, exact_ir, candidate.branch.to_mapping()
        ) != certificate.replay_digest:
            return _failure("CERTIFICATE_DIGEST_MISMATCH")

        separation = str(exact_ir.get("kind", "")).startswith("separation_")
        expected_pipeline = (
            "SeparationVerifierV1" if separation else "DirectReductionVerifierV1"
        )
        expected_versions = {
            "backend": backend.backend_name,
            "backend_version": backend.backend_version,
            "normalization": getattr(backend, "normalization_version", "unknown"),
            "expression_ir": ExpressionIRV1.schema_version,
            "pipeline": expected_pipeline,
        }
        if thaw_json(certificate.versions) != expected_versions:
            return _failure("CERTIFICATE_VERSION_MISMATCH")

        transformed = _expression(exact_ir["transformed_residual"])
        prefactor = _expression(exact_ir["prefactor"])
        reduced_expressions = (
            tuple(_expression(item) for item in exact_ir["reduced_equations"])
            if separation
            else (_expression(exact_ir["reduced_equation"]),)
        )
        if backend.render(transformed) != certificate.substituted_residual:
            return _failure("SUBSTITUTED_RESIDUAL_MISMATCH")
        if backend.render(prefactor) != certificate.prefactor:
            return _failure("PREFACTOR_DISPLAY_MISMATCH")
        if (
            tuple(backend.render(item) for item in reduced_expressions)
            != certificate.reduced_equations
        ):
            return _failure("REDUCED_EQUATION_MISMATCH")

        request: SeparationAnsatzV1 | None = None
        ansatz: DirectAnsatzV1 | None = None
        if separation:
            definition = metadata.get("separation_definition")
            if not isinstance(definition, dict):
                return _failure("SEPARATION_DEFINITION_MISSING")
            request = SeparationAnsatzV1.from_mapping(definition)
            source_failure = _replay_separation_source(
                problem, candidate, request, exact_ir, backend
            )
        else:
            ansatz = DirectAnsatzV1.from_mapping(metadata["ansatz_definition"])
            if ansatz.branch != candidate.branch:
                return _failure("BRANCH_BINDING_MISMATCH")
            source_failure = _replay_direct_source(
                problem,
                candidate,
                ansatz,
                exact_ir,
                backend,
                require_generator_provenance=require_generator_provenance,
            )
        if source_failure is not None:
            return source_failure

        identity_expression = (
            _expression(exact_ir["identity_sum"])
            if separation
            else reduced_expressions[0]
        )
        identity = backend.prove_zero(
            subtract(transformed, mul(prefactor, identity_expression))
        )
        if identity.status is not ProofStatusV1.PROVED_ZERO:
            return _failure("REDUCTION_IDENTITY_FAILED", identity.status)

        old_variables = set(exact_ir["old_variables"])
        removed = not any(
            old_variables & set(expression.free_names)
            for expression in reduced_expressions
        )
        if removed != certificate.old_variables_removed or (
            candidate.verification.status is VerificationStatusV1.CERTIFIED_VALID
            and not removed
        ):
            return _failure("OLD_VARIABLE_AUDIT_MISMATCH")

        coordinate = _expression(exact_ir["coordinate"])
        inverse_expressions = tuple(
            _expression(item) for item in exact_ir["inverse_map"].values()
        )
        cancellation_conditions = tuple(
            _expression(item) for item in exact_ir.get("cancellation_conditions", ())
        )
        used_names = set().union(
            *(expression.free_names for expression in (
                transformed,
                prefactor,
                coordinate,
                *reduced_expressions,
                *inverse_expressions,
                *cancellation_conditions,
            ))
        )
        expected_allowed = (
            used_names
            - set(SAFE_FUNCTIONS_V1)
            - old_variables
            | set(candidate.free_parameters)
            | set(candidate.profile_functions)
        )
        if set(certificate.allowed_symbols) != expected_allowed:
            return _failure("ALLOWED_SYMBOL_AUDIT_MISMATCH")

        if separation:
            assert request is not None
            expected_chart = _reconstruct_separation_chart(problem)
            if expected_chart is None:
                return _failure("SEPARATION_CHART_RECONSTRUCTION_UNSUPPORTED")
            base_conditions = expected_chart.domain_conditions
            expected_side_conditions = tuple(
                dict.fromkeys(
                    (
                        *base_conditions,
                        *(
                            f"{backend.render(item)}!=0"
                            for item in cancellation_conditions
                        ),
                    )
                )
            )
            if (
                tuple(exact_ir.get("domain_conditions", ()))
                != base_conditions
                or certificate.parameter_conditions
            ):
                return _failure("SEPARATION_CONDITION_MISMATCH")
            if certificate.chart != expected_chart:
                return _failure("SEPARATION_CHART_MISMATCH")
            spatial, time = problem.independent_variables
            separation_context = ParserContextV1(
                (),
                (spatial, time),
                tuple(
                    dict.fromkeys(
                        (*problem.parameters, request.separation_constant)
                    )
                ),
                (request.spatial_profile, request.temporal_profile),
            )
            if not prove_nonzero_on_conditions(
                prefactor,
                base_conditions,
                context=separation_context,
                backend=backend,
            ):
                return _failure("SEPARATION_PREFACTOR_NONZERO_UNPROVED")
            for condition in cancellation_conditions:
                if not prove_nonzero_on_conditions(
                    condition,
                    base_conditions,
                    context=separation_context,
                    backend=backend,
                ):
                    return _failure(
                        "SEPARATION_CANCELLATION_CONDITION_UNPROVED"
                    )
            if certificate.side_conditions != expected_side_conditions:
                return _failure("SEPARATION_CONDITION_MISMATCH")
        else:
            assert ansatz is not None
            context = _direct_context(candidate, ansatz)
            reconstructed_chart = construct_reduction_chart(
                ansatz,
                independent_variables=tuple(exact_ir["old_variables"]),
                context=context,
                backend=backend,
            )
            if reconstructed_chart.chart != certificate.chart:
                return _failure("CHART_REPLAY_MISMATCH")
            if certificate.parameter_conditions != candidate.branch.assumptions:
                return _failure("PARAMETER_CONDITION_MISMATCH")
            base_conditions = tuple(
                dict.fromkeys(
                    (
                        *ansatz.domain_conditions,
                        *candidate.branch.assumptions,
                        *candidate.branch.side_conditions,
                    )
                )
            )
            if not prove_nonzero_on_conditions(
                prefactor,
                base_conditions,
                context=context,
                backend=backend,
            ):
                return _failure("PREFACTOR_NONZERO_UNPROVED")
            for condition in cancellation_conditions:
                if not prove_nonzero_on_conditions(
                    condition,
                    base_conditions,
                    context=context,
                    backend=backend,
                ):
                    return _failure("CANCELLATION_CONDITION_UNPROVED")
            expected_side_conditions = tuple(
                dict.fromkeys(
                    (
                        *ansatz.domain_conditions,
                        *candidate.branch.side_conditions,
                        *(f"{backend.render(item)}!=0" for item in cancellation_conditions),
                    )
                )
            )
            if certificate.side_conditions != expected_side_conditions:
                return _failure("SIDE_CONDITION_MISMATCH")

        eta = symbol(exact_ir["coordinate_name"], SymbolRoleV1.PROFILE_COORDINATE)
        dependence_ir = exact_ir["coefficient_dependence"]
        if len(dependence_ir) != len(certificate.coefficient_dependence):
            return _failure("DEPENDENCE_COUNT_MISMATCH")
        for record, record_ir in zip(
            certificate.coefficient_dependence, dependence_ir
        ):
            ratio = _expression(record_ir["ratio"])
            rho = _expression(record_ir["rho"])
            reconstructed = backend.substitute(rho, {eta: coordinate})
            proof = backend.prove_zero(subtract(ratio, reconstructed))
            complementary = certificate.chart.complementary_coordinate
            expected_dependence = complementary not in rho.free_names
            if expected_dependence:
                expected_identity = (
                    f"{backend.render(ratio)}={backend.render(rho)}|"
                    f"{exact_ir['coordinate_name']}={candidate.coordinate}"
                )
                if (
                    record.status is not DependenceStatusV1.PROVED_DEPENDENCE
                    or record.ratio_expression != backend.render(ratio)
                    or record.rho_expression != backend.render(rho)
                    or record.identity != expected_identity
                    or record.replay_digest is None
                ):
                    return _failure("DEPENDENCE_RECORD_MISMATCH")
                if dependence_replay_digest(
                    record.to_mapping(), record_ir
                ) != record.replay_digest:
                    return _failure("DEPENDENCE_DIGEST_MISMATCH")
                if proof.status is not ProofStatusV1.PROVED_ZERO:
                    return _failure("DEPENDENCE_REPLAY_FAILED", proof.status)
            elif (
                record.status is not DependenceStatusV1.PROVED_NOT_DEPENDENCE
                or record.ratio_expression != backend.render(ratio)
                or record.rho_expression is not None
                or record.identity is not None
                or record.replay_digest is not None
            ):
                return _failure("DEPENDENCE_RECORD_MISMATCH")
            dependence_cancellations = tuple(
                _expression(item)
                for item in record_ir.get("cancellation_conditions", ())
            )
            expected_conditions = tuple(
                dict.fromkeys(
                    (
                        *certificate.chart.domain_conditions,
                        *(f"{backend.render(item)}!=0" for item in dependence_cancellations),
                    )
                )
            )
            if record.side_conditions != expected_conditions:
                return _failure("DEPENDENCE_SIDE_CONDITION_MISMATCH")
        if candidate.equivalence_witness is not None:
            from ..canonicalization.equivalence import (
                replay_equivalence_witness_mapping,
            )

            witness = thaw_json(candidate.equivalence_witness)
            if (
                witness.get("representative_id") != candidate.candidate_id
                or not replay_equivalence_witness_mapping(witness)
            ):
                return _failure("EQUIVALENCE_WITNESS_BINDING_MISMATCH")
        return CertificateReplayResultV1(
            True, "EXACT_REPLAY", ProofStatusV1.PROVED_ZERO
        )
    except Exception:  # noqa: BLE001 - replay must convert all failures to invalid
        return _failure("CERTIFICATE_REPLAY_ERROR", ProofStatusV1.UNKNOWN)


def replay_candidate_certificate(
    candidate: DiscoveryCandidateV1,
    problem: DiscoveryProblemInputV1,
    backend: CASBackendV1,
) -> CertificateReplayResultV1:
    """Replay a durable candidate with mandatory generator provenance binding."""

    return _replay_candidate_certificate(
        candidate,
        problem,
        backend,
        require_generator_provenance=True,
    )


__all__ = [
    "CertificateReplayResultV1",
    "certificate_replay_digest",
    "dependence_replay_digest",
    "replay_candidate_certificate",
]
