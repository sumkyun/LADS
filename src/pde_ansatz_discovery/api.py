"""Public dynamic discovery API for certified PDE reductions."""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import replace

from .cas.worker import (
    WorkerOperationV1,
    WorkerRequestV1,
    WorkerResultV1,
    run_worker_job,
)
from .models import (
    CONDITION_ELIGIBLE_V1,
    CONDITION_REJECTED_V1,
    BranchCoverageV1,
    CandidateVerificationV1,
    ConditionCompatibilityV1,
    DiscoveryBudgetV1,
    DiscoveryCandidateV1,
    DiscoveryContractError,
    DiscoveryProblemInputV1,
    DiscoveryResultV1,
    ParameterBranchV1,
    ProofStatusV1,
    VerificationStatusV1,
    WorkerStatusV1,
)
from .serialization import canonical_sha256


def _validate_inputs(
    problem: DiscoveryProblemInputV1,
    budget: DiscoveryBudgetV1,
    top_k: int,
) -> None:
    if not isinstance(problem, DiscoveryProblemInputV1):
        raise DiscoveryContractError("problem must be DiscoveryProblemInputV1")
    if not isinstance(budget, DiscoveryBudgetV1):
        raise DiscoveryContractError("budget must be DiscoveryBudgetV1")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise DiscoveryContractError("top_k must be a positive integer")
    if len(problem.dependent_variables) != 1 or len(problem.independent_variables) != 2:
        raise DiscoveryContractError(
            "dynamic discovery V1 supports one field in 1+1 dimensions"
        )
    if len(problem.pde) != 1:
        raise DiscoveryContractError("dynamic discovery V1 supports one PDE residual")


def _terminal_unresolved_candidate(
    problem: DiscoveryProblemInputV1,
    budget: DiscoveryBudgetV1,
    reason: str,
    *,
    stage: str,
) -> DiscoveryCandidateV1:
    branch = ParameterBranchV1(
        "worker_terminal",
        {},
        (),
        (f"unexplored_reason:{reason}",),
        BranchCoverageV1.UNKNOWN,
    )
    candidate = DiscoveryCandidateV1(
        canonical_sha256(
            {
                "kind": "DiscoveryTerminalCandidateV1",
                "problem": problem.to_mapping(),
                "budget": budget.to_mapping(),
                "reason": reason,
                "stage": stage,
            }
        ),
        "unresolved",
        problem.independent_variables[0],
        ("F",),
        tuple(problem.parameters),
        branch,
        (stage,),
        (reason,),
        CandidateVerificationV1(VerificationStatusV1.UNRESOLVED, reason, None),
        ConditionCompatibilityV1(()),
        f"{stage}:terminal",
        f"{stage}:{reason}",
        f"unresolved:{reason}",
        None,
        {"reason_code": reason},
    )
    return candidate


def _terminal_unresolved_result(
    problem: DiscoveryProblemInputV1,
    budget: DiscoveryBudgetV1,
    top_k: int,
    reason: str,
) -> DiscoveryResultV1:
    candidate = _terminal_unresolved_candidate(
        problem,
        budget,
        reason,
        stage="worker:isolated_discovery_v1",
    )
    return DiscoveryResultV1(
        problem.input_hash,
        budget.config_hash,
        "sympy-1.14.0",
        top_k,
        (),
        (),
        (),
        (candidate,),
        {
            "reason_counts": {reason: 1},
            "failure_feedback": {},
            "failure_expansion_materialization": [],
            "search_counters": {
                "generated": 1,
                "verified": 1,
                "certified_valid": 0,
                "certified_invalid": 0,
                "unresolved": 1,
                "skipped_by_cap": 0,
                "total_candidate_cap": budget.total_candidates,
            },
            "budget_enforcement": {
                **budget.to_mapping(),
                "worker_isolation": True,
            },
        },
    )


def _discover_in_process(
    problem: DiscoveryProblemInputV1,
    budget: DiscoveryBudgetV1,
    top_k: int,
) -> DiscoveryResultV1:
    """Orchestrate bounded named jobs without importing CAS objects in the caller."""

    from .analysis.boundaries import build_affine_dirichlet_lift
    from .analysis.translations import analyze_translations
    from .cas.backend import BranchResultV1
    from .generation.direct import DirectAnsatzV1
    from .generation.lie import ClassicalLieSeedV1
    from .generation.operator_grammar import generate_translation_scaling_candidates
    from .generation.separation import SeparationAnsatzV1, SeparationFamilyV1
    from .ir import (
        ExpressionIRV1,
        ParserLimitsV1,
        parse_problem,
    )
    from .search import (
        FailureFeedbackStateV1,
        SearchBudgetLedgerV1,
        expand_failure,
        rank_valid_candidates,
    )
    from .serialization import thaw_json
    from .solving.determining_system import GeneratorVerificationV1
    from .solving.prolongation import PointGeneratorV1

    _validate_inputs(problem, budget, top_k)
    search_started = time.monotonic()

    parser_limits = ParserLimitsV1(
        max_characters=max(256, min(10_000, budget.expression_node_cap * 32)),
        max_ast_nodes=max(64, min(5_000, budget.expression_node_cap * 8)),
    )
    parsed = parse_problem(problem, parser_limits)
    if any(
        equation.residual.node_count > budget.expression_node_cap
        for equation in parsed.equations
    ):
        return _terminal_unresolved_result(
            problem, budget, top_k, "EXPRESSION_NODE_CAP_REACHED"
        )
    generated = generate_translation_scaling_candidates(problem, parsed)
    if time.monotonic() - search_started >= budget.total_wall_clock_seconds:
        return _terminal_unresolved_result(
            problem, budget, top_k, "TOTAL_WALL_CLOCK_TIMEOUT"
        )
    ledger = SearchBudgetLedgerV1(budget)
    verified: list[DiscoveryCandidateV1] = []
    reasons = Counter(generated.failures)
    feedback_states: dict[str, FailureFeedbackStateV1] = {}
    feedback_details: dict[str, dict[str, object]] = {}
    requested_expansions: set[str] = set()
    materialization: list[dict[str, str]] = []
    parameter_solver_calls = 0
    isolated_job_statuses: Counter[str] = Counter()

    def isolated_job(
        operation: WorkerOperationV1,
        options: dict[str, object],
        *,
        deadline: int,
    ):
        remaining = budget.total_wall_clock_seconds - (
            time.monotonic() - search_started
        )
        if remaining <= 0:
            result = WorkerResultV1(
                WorkerStatusV1.TIMEOUT,
                None,
                "TOTAL_WALL_CLOCK_EXHAUSTED",
                time.monotonic() - search_started,
                None,
            )
            isolated_job_statuses[f"{operation.value}:{result.status.value}"] += 1
            return result
        job_deadline = min(float(deadline), remaining)
        result = run_worker_job(
            WorkerRequestV1(
                operation,
                None,
                independent_variables=problem.independent_variables,
                options=options,
            ),
            deadline_seconds=job_deadline,
            cpu_seconds=max(1, math.ceil(job_deadline)),
            result_size_limit=max(
                1_048_576,
                min(16 * 1024 * 1024, budget.expression_node_cap * 256),
            ),
        )
        isolated_job_statuses[f"{operation.value}:{result.status.value}"] += 1
        return result

    def worker_reason(result, *, characteristic: bool = False) -> str:
        if result.error_code == "TOTAL_WALL_CLOCK_EXHAUSTED":
            return "TOTAL_WALL_CLOCK_TIMEOUT"
        prefix = "CHARACTERISTIC" if characteristic else "CAS"
        status_reasons = {
            WorkerStatusV1.TIMEOUT: f"{prefix}_TIMEOUT",
            WorkerStatusV1.RESOURCE_EXHAUSTED: f"{prefix}_RESOURCE_EXHAUSTED",
            WorkerStatusV1.CRASHED: f"{prefix}_WORKER_CRASHED",
        }
        return (
            status_reasons[result.status]  # noqa: SIM401 - scanner-allowlisted internal state
            if result.status in status_reasons
            else f"{prefix}_WORKER_FAILED"
        )

    def isolated_direct(ansatz: DirectAnsatzV1) -> DiscoveryCandidateV1:
        result = isolated_job(
            WorkerOperationV1.VERIFY_DIRECT,
            {"problem": problem.to_mapping(), "ansatz": ansatz.to_mapping()},
            deadline=budget.cas_call_deadline_seconds,
        )
        if result.status is not WorkerStatusV1.COMPLETED or result.payload is None:
            return _terminal_unresolved_candidate(
                problem,
                budget,
                worker_reason(result),
                stage=f"verify_direct:{ansatz.family}",
            )
        return DiscoveryCandidateV1.from_mapping(result.payload["candidate"])

    def isolated_conditions(candidate: DiscoveryCandidateV1) -> DiscoveryCandidateV1:
        if not (
            problem.initial_conditions
            or problem.boundary_conditions
            or problem.additional_constraints
        ):
            return candidate
        result = isolated_job(
            WorkerOperationV1.APPLY_CONDITIONS,
            {"problem": problem.to_mapping(), "candidate": candidate.to_mapping()},
            deadline=budget.cas_call_deadline_seconds,
        )
        if result.status is not WorkerStatusV1.COMPLETED or result.payload is None:
            reason = worker_reason(result)
            return replace(
                candidate,
                failure_history=(*candidate.failure_history, reason),
                verification=CandidateVerificationV1(
                    VerificationStatusV1.UNRESOLVED, reason, None
                ),
                reduction_key=f"unresolved:{reason}",
            )
        return DiscoveryCandidateV1.from_mapping(result.payload["candidate"])

    def enforce_expression_cap(
        candidate: DiscoveryCandidateV1,
    ) -> DiscoveryCandidateV1:
        metadata = thaw_json(candidate.metadata)
        maximum = 0

        def visit(value) -> None:
            nonlocal maximum
            if isinstance(value, dict):
                if value.get("schema_version") == ExpressionIRV1.schema_version:
                    maximum = max(
                        maximum, ExpressionIRV1.from_mapping(value).node_count
                    )
                    return
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(metadata.get("certificate_ir"))
        if maximum <= budget.expression_node_cap:
            return candidate
        return replace(
            candidate,
            failure_history=(*candidate.failure_history, "EXPRESSION_NODE_CAP_REACHED"),
            verification=CandidateVerificationV1(
                VerificationStatusV1.UNRESOLVED,
                "EXPRESSION_NODE_CAP_REACHED",
                None,
            ),
            reduction_key="unresolved:EXPRESSION_NODE_CAP_REACHED",
            equivalence_witness=None,
            metadata={
                "reason_code": "EXPRESSION_NODE_CAP_REACHED",
                "maximum_expression_nodes": maximum,
            },
        )

    def record_feedback(reason: str) -> None:
        state = (
            feedback_states[reason]
            if reason in feedback_states
            else FailureFeedbackStateV1()
        )
        outcome = expand_failure(reason, state, budget)
        feedback_states[reason] = outcome.state
        details = feedback_details.setdefault(
            reason, {"expansions": [], "terminal_reason": None}
        )
        additions = details["expansions"]
        assert isinstance(additions, list)
        for expansion in outcome.expansions:
            if expansion not in additions:
                additions.append(expansion)
            requested_expansions.add(expansion)
        if outcome.terminal_reason is not None:
            details["terminal_reason"] = outcome.terminal_reason

    for reason in generated.failures:
        record_feedback(reason)

    direct_pool = [
        ansatz
        for ansatz in generated.candidates
        if budget.grammar_depth >= (2 if ansatz.family.startswith("singular:") else 1)
    ]
    lift = (
        build_affine_dirichlet_lift(
            problem.boundary_conditions,
            field=problem.dependent_variables[0],
            spatial_variable=problem.independent_variables[0],
        )
        if budget.grammar_depth >= 2
        else None
    )
    if lift is not None:
        lifted = []
        for ansatz in direct_pool[: budget.beam_width]:
            right = ansatz.expression.split("=", 1)[-1]
            lifted.append(
                replace(
                    ansatz,
                    expression=(
                        f"({lift.expression})+({lift.homogeneous_factor})*({right})"
                    ),
                    family=f"boundary_lift:{ansatz.family}",
                )
            )
        direct_pool.extend(lifted)

    attempted_definitions: set[str] = set()
    admitted_definitions: set[str] = set()

    def verify_direct(ansatz) -> bool:
        if ansatz.definition_hash in attempted_definitions:
            return False
        attempted_definitions.add(ansatz.definition_hash)
        if not ledger.admit_candidate():
            reasons["TOTAL_CANDIDATE_CAP_REACHED"] += 1
            return False
        admitted_definitions.add(ansatz.definition_hash)
        candidate = isolated_conditions(enforce_expression_cap(isolated_direct(ansatz)))
        if any(
            record.kind.startswith("initial") and record.status.value == "incompatible"
            for record in candidate.condition_compatibility.records
        ):
            record_feedback("IC_INCOMPATIBLE")
        ledger.record_verdict(candidate.verification.status.value)
        verified.append(candidate)
        if candidate.verification.status is not VerificationStatusV1.CERTIFIED_VALID:
            reasons[candidate.verification.reason_code] += 1
            record_feedback(candidate.verification.reason_code)
        return True

    for ansatz in direct_pool[: budget.beam_width]:
        verify_direct(ansatz)

    def expansion_matches(expansion: str, family: str) -> bool:
        return (
            (expansion == "traveling" and "translation" in family)
            or (
                expansion == "scaling"
                and "scaling" in family
                and "singular" not in family
            )
            or (expansion == "singular" and "singular" in family)
            or expansion in {"direct_grammar", "alternate_coordinate"}
        )

    # A failure transition is an actual bounded queue expansion, not a report-only hint.
    direct_expansions = {
        "traveling",
        "scaling",
        "singular",
        "direct_grammar",
        "alternate_coordinate",
    }
    for expansion in sorted(requested_expansions & direct_expansions):
        matches = [
            ansatz
            for ansatz in direct_pool
            if ansatz.definition_hash not in attempted_definitions
            and expansion_matches(expansion, ansatz.family)
        ][: budget.beam_width]
        admitted = sum(verify_direct(ansatz) for ansatz in matches)
        materialization.append(
            {
                "expansion": expansion,
                "outcome": (
                    "candidate_enqueued"
                    if admitted
                    else "blocked_by_total_candidate_cap"
                    if matches and ledger.generated >= budget.total_candidates
                    else "no_new_allowlisted_candidate"
                ),
            }
        )

    for expansion, parameter in (("amplitude", "A"), ("offset", "B")):
        if expansion not in requested_expansions:
            continue
        variants = []
        for ansatz in direct_pool[: budget.beam_width]:
            right = ansatz.expression.split("=", 1)[-1]
            expression = (
                f"{parameter}*({right})"
                if expansion == "amplitude"
                else f"({right})+{parameter}"
            )
            variant = replace(
                ansatz,
                expression=expression,
                free_parameters=tuple(
                    dict.fromkeys((*ansatz.free_parameters, parameter))
                ),
                family=f"feedback_{expansion}:{ansatz.family}",
                branch=None,
            )
            variants.append(variant)
        admitted = sum(verify_direct(variant) for variant in variants)
        materialization.append(
            {
                "expansion": expansion,
                "outcome": (
                    "candidate_enqueued"
                    if admitted
                    else "blocked_by_total_candidate_cap"
                    if variants and ledger.generated >= budget.total_candidates
                    else "no_base_candidate"
                ),
            }
        )

    separation_families = (
        SeparationFamilyV1.PRODUCT,
        SeparationFamilyV1.ADDITIVE,
        SeparationFamilyV1.TRANSFORMED_IDENTITY,
    )[: budget.beam_width]
    separation_enabled = budget.grammar_depth >= 2
    separation_admitted = False
    for family in separation_families if separation_enabled else ():
        if not ledger.admit_candidate():
            reasons["TOTAL_CANDIDATE_CAP_REACHED"] += 1
            continue
        separation_admitted = True
        request = SeparationAnsatzV1(family, problem.dependent_variables[0])
        separation_worker = isolated_job(
            WorkerOperationV1.VERIFY_SEPARATION,
            {
                "problem": problem.to_mapping(),
                "request": request.to_mapping(),
            },
            deadline=budget.cas_call_deadline_seconds,
        )
        if (
            separation_worker.status is WorkerStatusV1.COMPLETED
            and separation_worker.payload is not None
        ):
            candidate = DiscoveryCandidateV1.from_mapping(
                separation_worker.payload["candidate"]
            )
        else:
            candidate = _terminal_unresolved_candidate(
                problem,
                budget,
                worker_reason(separation_worker),
                stage=f"verify_separation:{family.value}",
            )
        candidate = isolated_conditions(enforce_expression_cap(candidate))
        ledger.record_verdict(candidate.verification.status.value)
        verified.append(candidate)
        if candidate.verification.status is not VerificationStatusV1.CERTIFIED_VALID:
            reasons[candidate.verification.reason_code] += 1
            record_feedback(candidate.verification.reason_code)
    if "separation" in requested_expansions:
        materialization.append(
            {
                "expansion": "separation",
                "outcome": (
                    "candidate_enqueued"
                    if separation_admitted
                    else "blocked_by_total_candidate_cap"
                    if separation_enabled
                    and separation_families
                    and ledger.generated >= budget.total_candidates
                    else "blocked_by_grammar_or_beam_budget"
                ),
            }
        )

    maximum_observed_generator_basis_degree = 0

    def deserialize_generator(value) -> PointGeneratorV1:
        return PointGeneratorV1(
            ExpressionIRV1.from_mapping(value["time_component"]),
            ExpressionIRV1.from_mapping(value["spatial_component"]),
            ExpressionIRV1.from_mapping(value["field_component"]),
            value["field"],
            value["spatial_variable"],
            value["time_variable"],
        )

    def deserialize_determining(value) -> GeneratorVerificationV1:
        return GeneratorVerificationV1(
            value["mode"],
            value["invariant"],
            ExpressionIRV1.from_mapping(value["raw"]),
            ExpressionIRV1.from_mapping(value["reduced"]),
            ProofStatusV1(value["proof_status"]),
            value["reason_code"],
            None
            if value["invariant_surface"] is None
            else ExpressionIRV1.from_mapping(value["invariant_surface"]),
        )

    if budget.grammar_depth >= 3:
        lie_worker = isolated_job(
            WorkerOperationV1.CLASSICAL_LIE_SEEDS,
            {"problem": problem.to_mapping()},
            deadline=budget.characteristic_deadline_seconds,
        )
        if (
            lie_worker.status is WorkerStatusV1.COMPLETED
            and lie_worker.payload is not None
        ):
            seed_items = tuple(lie_worker.payload["seeds"])
            finite_degrees = tuple(
                item["basis_degree"]
                for item in seed_items
                if item["basis_degree"] is not None
            )
            maximum_observed_generator_basis_degree = max(finite_degrees, default=0)
            eligible_seed_items = tuple(
                item
                for item in seed_items
                if item["basis_degree"] is not None
                and item["basis_degree"] <= budget.generator_basis_degree
            )[: budget.beam_width]
            lie_seeds = tuple(
                ClassicalLieSeedV1(
                    DirectAnsatzV1.from_mapping(item["ansatz"]),
                    deserialize_generator(item["generator"]),
                    deserialize_determining(item["determining"]),
                    item["generator_label"],
                    tuple(item["solver_parameters"]),
                    item["basis_degree"],
                )
                for item in eligible_seed_items
            )
        else:
            lie_seeds = ()
            reason = worker_reason(lie_worker, characteristic=True)
            reasons[reason] += 1
            record_feedback(reason)
    else:
        lie_seeds = ()

    maximum_materialized_branches = 0

    for seed in lie_seeds:
        boundary_checks: tuple[dict[str, object], ...] = ()
        if problem.boundary_conditions:
            boundary_worker = isolated_job(
                WorkerOperationV1.CHECK_BOUNDARY_PRESERVATION,
                {
                    "problem": problem.to_mapping(),
                    "spatial_component": seed.generator.spatial_component.to_mapping(),
                },
                deadline=budget.cas_call_deadline_seconds,
            )
            if (
                boundary_worker.status is WorkerStatusV1.COMPLETED
                and boundary_worker.payload is not None
            ):
                boundary_checks = tuple(boundary_worker.payload["checks"])
                boundary_preserved = bool(boundary_checks) and all(
                    item["preserved"] is True for item in boundary_checks
                )
                boundary_reason = "BOUNDARY_NOT_PRESERVED"
            else:
                boundary_preserved = False
                boundary_reason = worker_reason(boundary_worker)
            if not boundary_preserved:
                if ledger.admit_candidate():
                    candidate = _terminal_unresolved_candidate(
                        problem,
                        budget,
                        boundary_reason,
                        stage=f"classical_lie:{seed.generator_label}:boundary",
                    )
                    metadata = thaw_json(candidate.metadata)
                    metadata["boundary_preservation"] = list(boundary_checks)
                    candidate = replace(candidate, metadata=metadata)
                    ledger.record_verdict(VerificationStatusV1.UNRESOLVED.value)
                    verified.append(candidate)
                    reasons[boundary_reason] += 1
                    record_feedback(boundary_reason)
                continue
        solver_worker = isolated_job(
            WorkerOperationV1.SOLVE_PARAMETER_BRANCHES,
            {
                "ansatz": seed.ansatz.to_mapping(),
                "equations": [
                    seed.determining.reduced_determining_expression.to_mapping()
                ],
                "parameters": list(seed.solver_parameters),
                "branch_cap": budget.parameter_branch_cap,
                "seed_label": seed.generator_label,
                "problem": problem.to_mapping(),
                "generator": {
                    "time_component": seed.generator.time_component.to_mapping(),
                    "spatial_component": seed.generator.spatial_component.to_mapping(),
                    "field_component": seed.generator.field_component.to_mapping(),
                    "field": seed.generator.field,
                    "spatial_variable": seed.generator.spatial_variable,
                    "time_variable": seed.generator.time_variable,
                },
            },
            deadline=budget.cas_call_deadline_seconds,
        )
        parameter_solver_calls += 1
        if (
            solver_worker.status is not WorkerStatusV1.COMPLETED
            or solver_worker.payload is None
        ):
            reason = worker_reason(solver_worker)
            if ledger.admit_candidate():
                candidate = _terminal_unresolved_candidate(
                    problem,
                    budget,
                    reason,
                    stage=f"classical_lie:{seed.generator_label}:parameter_solver",
                )
                ledger.record_verdict(VerificationStatusV1.UNRESOLVED.value)
                verified.append(candidate)
                reasons[reason] += 1
                record_feedback(reason)
            continue
        branch_result = BranchResultV1.from_mapping(
            solver_worker.payload["branch_result"]
        )
        branch_ansatze = tuple(
            DirectAnsatzV1.from_mapping(item)
            for item in solver_worker.payload["materialized_ansatze"]
        )
        branch_verifications = {
            item["branch_id"]: item
            for item in solver_worker.payload["branch_verifications"]
        }
        base_lie_metadata = {
            "generator_label": seed.generator_label,
            "time_component": seed.generator.time_component.to_mapping(),
            "spatial_component": seed.generator.spatial_component.to_mapping(),
            "field_component": seed.generator.field_component.to_mapping(),
            "determining_raw": seed.determining.raw_determining_expression.to_mapping(),
            "determining_reduced": seed.determining.reduced_determining_expression.to_mapping(),
            "proof_status": seed.determining.proof_status.value,
            "reason_code": seed.determining.reason_code,
            "parameter_solver": branch_result.to_mapping(),
            "boundary_preservation": list(boundary_checks),
            "basis_degree": seed.basis_degree,
        }
        maximum_materialized_branches = max(
            maximum_materialized_branches, len(branch_ansatze)
        )
        for branch_ansatz in branch_ansatze:
            branch_verification = (
                branch_verifications[  # noqa: SIM401 - scanner-allowlisted internal state
                    branch_ansatz.branch.branch_id
                ]
                if branch_ansatz.branch.branch_id in branch_verifications
                else None
            )
            if (
                branch_verification is None
                or branch_verification["determining"]["invariant"] is not True
                or branch_verification["determining"]["proof_status"] != "proved_zero"
            ):
                reason = "PARAMETER_BRANCH_INVARIANCE_UNPROVED"
                if ledger.admit_candidate():
                    candidate = _terminal_unresolved_candidate(
                        problem,
                        budget,
                        reason,
                        stage=f"classical_lie:{seed.generator_label}:branch_replay",
                    )
                    ledger.record_verdict(VerificationStatusV1.UNRESOLVED.value)
                    verified.append(candidate)
                    reasons[reason] += 1
                    record_feedback(reason)
                continue
            if not ledger.admit_candidate():
                reasons["TOTAL_CANDIDATE_CAP_REACHED"] += 1
                continue
            candidate = enforce_expression_cap(isolated_direct(branch_ansatz))
            metadata = thaw_json(candidate.metadata)
            lie_metadata = {
                **base_lie_metadata,
                "time_component": branch_verification["generator"]["time_component"],
                "spatial_component": branch_verification["generator"][
                    "spatial_component"
                ],
                "field_component": branch_verification["generator"]["field_component"],
                "determining_raw": branch_verification["determining"]["raw"],
                "determining_reduced": branch_verification["determining"]["reduced"],
                "proof_status": branch_verification["determining"]["proof_status"],
                "reason_code": branch_verification["determining"]["reason_code"],
            }
            metadata["classical_lie"] = lie_metadata
            candidate = isolated_conditions(
                replace(
                    candidate,
                    production_trace=(
                        f"classical_lie:{seed.generator_label}",
                        "determining_system:proved_zero",
                        f"parameter_branch:{branch_ansatz.branch.branch_id}",
                        "characteristic:explicit",
                        "verification:direct_v1",
                    ),
                    generator_key=f"classical_lie:{seed.generator_label}",
                    metadata=metadata,
                )
            )
            ledger.record_verdict(candidate.verification.status.value)
            verified.append(candidate)
            if (
                candidate.verification.status
                is not VerificationStatusV1.CERTIFIED_VALID
            ):
                reasons[candidate.verification.reason_code] += 1
                record_feedback(candidate.verification.reason_code)
        if branch_result.coverage is not BranchCoverageV1.EXHAUSTIVE:
            reason = "PARAMETER_BRANCH_COVERAGE_INCOMPLETE"
            if ledger.admit_candidate():
                remainder = _terminal_unresolved_candidate(
                    problem,
                    budget,
                    reason,
                    stage=f"classical_lie:{seed.generator_label}:unexplored_remainder",
                )
                remainder_metadata = thaw_json(remainder.metadata)
                remainder_metadata["classical_lie"] = base_lie_metadata
                remainder = replace(
                    remainder,
                    branch=ParameterBranchV1(
                        f"unexplored:{canonical_sha256(base_lie_metadata)}",
                        {},
                        (),
                        (f"unexplored_reason:{branch_result.unexplored_reason}",),
                        branch_result.coverage,
                    ),
                    metadata=remainder_metadata,
                )
                ledger.record_verdict(VerificationStatusV1.UNRESOLVED.value)
                verified.append(remainder)
                reasons[reason] += 1
                record_feedback(reason)
            else:
                reasons["TOTAL_CANDIDATE_CAP_REACHED"] += 1

    translation_analysis = analyze_translations(
        tuple(item.residual for item in parsed.equations), problem.independent_variables
    )
    if budget.grammar_depth >= 4 and not translation_analysis.spatial_invariant:
        if ledger.admit_candidate():
            nonclassical_worker = isolated_job(
                WorkerOperationV1.VERIFY_NONCLASSICAL,
                {"problem": problem.to_mapping()},
                deadline=budget.characteristic_deadline_seconds,
            )
            if (
                nonclassical_worker.status is not WorkerStatusV1.COMPLETED
                or nonclassical_worker.payload is None
            ):
                reason = worker_reason(nonclassical_worker, characteristic=True)
                candidate = _terminal_unresolved_candidate(
                    problem,
                    budget,
                    reason,
                    stage="nonclassical:bounded_q",
                )
            else:
                candidate = DiscoveryCandidateV1.from_mapping(
                    nonclassical_worker.payload["candidate"]
                )
            candidate = isolated_conditions(enforce_expression_cap(candidate))
            ledger.record_verdict(candidate.verification.status.value)
            verified.append(candidate)
            if (
                candidate.verification.status
                is not VerificationStatusV1.CERTIFIED_VALID
            ):
                reasons[candidate.verification.reason_code] += 1
                record_feedback(candidate.verification.reason_code)
        else:
            reasons["TOTAL_CANDIDATE_CAP_REACHED"] += 1

    raw_valid = tuple(
        item
        for item in verified
        if item.verification.status is VerificationStatusV1.CERTIFIED_VALID
    )
    if raw_valid:
        dedupe_worker = isolated_job(
            WorkerOperationV1.DEDUPLICATE_CANDIDATES,
            {"candidates": [item.to_mapping() for item in raw_valid]},
            deadline=budget.cas_call_deadline_seconds,
        )
    else:
        dedupe_worker = None
    if (
        dedupe_worker is not None
        and dedupe_worker.status is WorkerStatusV1.COMPLETED
        and dedupe_worker.payload is not None
    ):
        deduplicated = tuple(
            DiscoveryCandidateV1.from_mapping(item)
            for item in dedupe_worker.payload["candidates"]
        )
    else:
        # Failure to prove an equivalence must preserve distinct certified candidates.
        deduplicated = raw_valid
        if dedupe_worker is not None:
            reason = worker_reason(dedupe_worker)
            reasons[f"DEDUPE_{reason}"] += 1
            record_feedback(f"DEDUPE_{reason}")
    valid = rank_valid_candidates(deduplicated)
    unresolved = tuple(
        sorted(
            (
                item
                for item in verified
                if item.verification.status is VerificationStatusV1.UNRESOLVED
            ),
            key=lambda item: item.candidate_id,
        )
    )
    eligible = tuple(
        item
        for item in valid
        if item.condition_compatibility.overall in CONDITION_ELIGIBLE_V1
    )
    rejected = tuple(
        item
        for item in valid
        if item.condition_compatibility.overall in CONDITION_REJECTED_V1
    )
    for reason in sorted(reasons):
        if reason not in feedback_details:
            record_feedback(reason)
    materialized_names = {item["expansion"] for item in materialization}
    for expansion in sorted(requested_expansions - materialized_names):
        if expansion == "boundary_lift":
            lift_admitted = any(
                ansatz.definition_hash in admitted_definitions
                for ansatz in direct_pool
                if ansatz.family.startswith("boundary_lift:")
            )
            outcome = (
                "candidate_enqueued"
                if lift_admitted
                else "blocked_by_search_budget"
                if lift is not None
                else "no_affine_dirichlet_lift"
            )
        elif expansion == "boundary_preserving_generator":
            outcome = "generator_filter_applied"
        elif expansion in {"multi_profile", "implicit_invariant"}:
            outcome = "closed_v1_verifier_unavailable_unresolved"
        else:
            outcome = "no_new_allowlisted_candidate"
        materialization.append({"expansion": expansion, "outcome": outcome})
    feedback = {reason: feedback_details[reason] for reason in sorted(feedback_details)}
    return DiscoveryResultV1(
        input_hash=problem.input_hash,
        config_hash=budget.config_hash,
        backend_version="sympy-1.14.0",
        top_k=top_k,
        valid_candidates=valid,
        recommended_candidates=eligible[:top_k],
        valid_condition_rejected=rejected,
        unresolved_candidates=unresolved,
        failure_summaries={
            "reason_counts": dict(sorted(reasons.items())),
            "failure_feedback": feedback,
            "failure_expansion_materialization": materialization,
            "search_counters": ledger.to_mapping(),
            "budget_enforcement": {
                "grammar_depth": budget.grammar_depth,
                "generator_basis_degree": budget.generator_basis_degree,
                "maximum_observed_generator_basis_degree": maximum_observed_generator_basis_degree,
                "beam_width": budget.beam_width,
                "parameter_branch_cap": budget.parameter_branch_cap,
                "maximum_branches_per_generated_candidate": maximum_materialized_branches,
                "parameter_solver_calls": parameter_solver_calls,
                "expression_node_cap": budget.expression_node_cap,
                "parser_max_characters": parser_limits.max_characters,
                "parser_max_ast_nodes": parser_limits.max_ast_nodes,
                "cas_call_deadline_seconds": budget.cas_call_deadline_seconds,
                "characteristic_deadline_seconds": budget.characteristic_deadline_seconds,
                "total_wall_clock_seconds": budget.total_wall_clock_seconds,
                "worker_isolation": True,
                "isolation_granularity": "fresh_process_per_named_cas_or_characteristic_job",
                "isolated_job_statuses": dict(sorted(isolated_job_statuses.items())),
            },
        },
    )


def discover_ansatz_reductions(
    problem: DiscoveryProblemInputV1,
    budget: DiscoveryBudgetV1,
    top_k: int,
) -> DiscoveryResultV1:
    """Orchestrate fresh named workers under one cumulative wall-clock budget."""

    _validate_inputs(problem, budget, top_k)
    return _discover_in_process(problem, budget, top_k)


__all__ = ["discover_ansatz_reductions"]
