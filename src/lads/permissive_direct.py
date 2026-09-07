"""Permissive direct Ansatz -> Atom execution lane for the V11 clone.

The lane removes tree-coverage and parameterization gates that discarded useful LLM
structures.  Stage I still has the required atom-free host boundary and every Stage-II
request is bound to one open profile slot.  After that, any Atom that parses and evaluates
with the declared slot symbols is retained for coefficient fitting.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import sympy as sp
from sympy.core.function import AppliedUndef

from .ansatz_replay import AnsatzReplayCertificateV1, NeutralAnsatzASTV1
from .llm_ansatz import (
    LLMAnsatzCandidateV1,
    LLMAnsatzError,
    canonicalize_llm_ansatz_notation,
    propose_llm_ansatz_batch,
    public_problem_payload,
    validate_llm_ansatz_candidate,
)
from .llm_provider import LLMBatchProviderV1, LLMResponseError
from .safe_symbolic import SafeSymbolicError, parse_symbolic_expression
from .staged_examples import (
    MAX_ATOM_COMBINATIONS,
    MAX_CERTIFIED_ANSATZ_RUNS_PER_CASE,
    MAX_FITTED_COEFFICIENTS,
    MAX_PROFILE_ARITY,
    MAX_REGENERATION_ATTEMPTS,
    MAX_SEMANTIC_FITS_PER_ANSATZ,
    MAX_SEMANTIC_FITS_PER_CASE,
    SEMANTIC_ACCEPTANCE_LOSS,
    STAGE_I_CANDIDATE_COUNT,
    STAGE_II_CANDIDATE_COUNT,
    AnsatzRichnessV1,
    RecordingProvider,
    RuntimeSlot,
    StagedExperimentError,
    _atom_retry_feedback,
    _provider_evidence,
    _safe_feedback,
    assemble_candidate_beam,
    assess_atom_pool_semantics,
    atom_response_format,
    canonical_sha256,
    render_atom_messages,
    screen_semantic_candidate_beam,
    semantic_validate_candidates,
    validate_atom_response,
)


def _syntax_only_certificate(
    row: Mapping[str, Any], candidate: LLMAnsatzCandidateV1
) -> AnsatzReplayCertificateV1:
    """Keep only the mandatory atom-free parse/slot boundary before Atom generation."""

    ast = _compile_direct_ansatz(row, candidate.template)
    if not ast.profile_names:
        raise StagedExperimentError("Ansatz has no open profile slot")
    return AnsatzReplayCertificateV1(
        passed=True,
        validation_kind="atom_free_syntax_slot_validation_v1",
        ansatz=ast.source,
        reduced_equations=(),
        invariant_coordinates=(),
        eliminated_variables=(),
        side_conditions=("atom_completion_required_before_pde_validation",),
        evidence=(
            "ATOM_FREE_NEUTRAL_AST",
            "PUBLIC_FIELD_CENSUS_PARSED",
            "OPEN_PROFILE_SLOTS_EXTRACTED",
        ),
    )


def _compile_direct_ansatz(row: Mapping[str, Any], source: str) -> NeutralAnsatzASTV1:
    """Compile an LLM Ansatz using syntax only, without PDE replay or topology tests."""

    dependent = tuple(map(str, row["dependent_variables"]))
    independent = tuple(map(str, row["independent_variables"]))
    raw_parameters = row.get("parameters", {})
    parameters = (
        tuple(map(str, raw_parameters))
        if isinstance(raw_parameters, Mapping)
        else tuple(map(str, raw_parameters))
        if isinstance(raw_parameters, Sequence) and not isinstance(raw_parameters, (str, bytes))
        else ()
    )
    names: dict[str, object] = {
        **{name: sp.Symbol(name, real=True) for name in independent},
        **{name: sp.Symbol(name, real=True) for name in dependent},
        **{name: sp.Symbol(name, real=True) for name in parameters},
        "pi": sp.pi,
    }
    source = canonicalize_llm_ansatz_notation(str(source))
    expressions: list[tuple[str, sp.Expr]] = []
    profile_names: set[str] = set()
    declared = {*dependent, *independent, *parameters}
    coefficients: set[str] = set()
    equations = tuple(part.strip() for part in source.split(";") if part.strip())
    if not equations:
        raise StagedExperimentError("Ansatz has no executable equations")
    for equation in equations:
        if equation.count("=") != 1:
            raise StagedExperimentError("Ansatz equation syntax is invalid")
        lhs, rhs = (item.strip() for item in equation.split("=", maxsplit=1))
        if not lhs or "__" in lhs or "__" in rhs:
            raise StagedExperimentError("Ansatz field syntax is invalid")
        field = lhs.split("(", maxsplit=1)[0].strip()
        if not field:
            raise StagedExperimentError("Ansatz field name is missing")
        try:
            expression = parse_symbolic_expression(rhs, names=names)
        except SafeSymbolicError as exc:
            raise StagedExperimentError("Ansatz RHS is not executable") from exc
        expressions.append((field, expression))
        profile_names.update(str(call.func) for call in expression.atoms(AppliedUndef))
        coefficients.update(
            str(symbol) for symbol in expression.free_symbols if str(symbol) not in declared
        )
    if not profile_names:
        raise StagedExperimentError("Ansatz has no open profile slot")
    return NeutralAnsatzASTV1(
        source=source,
        field_expressions=tuple(expressions),
        profile_names=tuple(sorted(profile_names)),
        symbolic_coefficients=tuple(sorted(coefficients)),
    )


def _render_direct_ansatz_with_slot(
    ast: NeutralAnsatzASTV1, call: AppliedUndef, slot_id: str
) -> str:
    marker = sp.Symbol(f"OPEN_{slot_id.upper()}")
    equations = [
        f"{field}={sp.sstr(expression.xreplace({call: marker}))}"
        for field, expression in ast.field_expressions
    ]
    rendered = ";".join(equations)
    if str(marker) not in rendered:
        raise StagedExperimentError("slot marker was not inserted")
    return rendered


def _extract_direct_runtime_slots(
    row: Mapping[str, Any], ansatz: str
) -> tuple[NeutralAnsatzASTV1, tuple[RuntimeSlot, ...]]:
    ast = _compile_direct_ansatz(row, ansatz)
    calls = sorted(
        {
            call
            for _field, expression in ast.field_expressions
            for call in expression.atoms(AppliedUndef)
        },
        key=sp.default_sort_key,
    )
    if not calls:
        raise StagedExperimentError("Ansatz has no open profile slot")
    slots: list[RuntimeSlot] = []
    for index, call in enumerate(calls, 1):
        if not call.args or len(call.args) > MAX_PROFILE_ARITY:
            raise StagedExperimentError("profile arity is outside the Atom slot contract")
        if any(argument.atoms(AppliedUndef) for argument in call.args):
            raise StagedExperimentError("nested profile arguments are unsupported")
        slot_id = f"slot_{index:03d}"
        roles = tuple(f"ARG{position}" for position in range(1, len(call.args) + 1))
        slots.append(
            RuntimeSlot(slot_id, call, roles, _render_direct_ansatz_with_slot(ast, call, slot_id))
        )
    return ast, tuple(slots)


def _direct_richness(
    row: Mapping[str, Any], source: str
) -> AnsatzRichnessV1:
    """Retain the existing richness ordering without re-entering strict validation."""

    ast = _compile_direct_ansatz(row, source)
    coordinates = {sp.Symbol(str(name), real=True) for name in row["independent_variables"]}
    calls = tuple(
        sorted(
            {
                call
                for _field, expression in ast.field_expressions
                for call in expression.atoms(AppliedUndef)
            },
            key=sp.default_sort_key,
        )
    )
    coverage = tuple(
        set().union(*(argument.free_symbols & coordinates for argument in call.args))
        for call in calls
    )
    covered = set().union(*coverage) if coverage else set()
    return AnsatzRichnessV1(
        full_arity_profile_count=sum(
            value == coordinates and len(call.args) >= len(coordinates)
            for call, value in zip(calls, coverage, strict=True)
        ),
        full_coordinate_profile_count=sum(value == coordinates for value in coverage),
        covered_coordinate_count=len(covered),
        profile_call_count=len(calls),
        profile_name_count=len(ast.profile_names),
        symbolic_coefficient_count=len(ast.symbolic_coefficients),
        operation_count=sum(
            sp.count_ops(expression) for _field, expression in ast.field_expressions
        ),
        source_length=len(re.sub(r"\s+", "", ast.source)),
    )


def _rank_direct_ansatz_candidates(
    row: Mapping[str, Any],
    certified: Sequence[tuple[LLMAnsatzCandidateV1, AnsatzReplayCertificateV1]],
) -> tuple[tuple[LLMAnsatzCandidateV1, AnsatzReplayCertificateV1, AnsatzRichnessV1], ...]:
    ranked = [
        (candidate, certificate, _direct_richness(row, candidate.template))
        for candidate, certificate in certified
    ]
    ranked.sort(key=lambda item: item[2].priority_key(provider_rank=item[0].rank))
    return tuple(ranked)


def _direct_atom_pool(
    row: Mapping[str, Any],
    candidate: LLMAnsatzCandidateV1,
    slot: RuntimeSlot,
    *,
    provider_factory: Callable[[int], LLMBatchProviderV1],
    counters: dict[str, int],
) -> tuple[tuple[dict[str, Any], ...], list[dict[str, Any]]]:
    """Ask the LLM for direct slot Atoms and retain every parseable response."""

    retained: dict[str, dict[str, Any]] = {}
    attempts: list[dict[str, Any]] = []
    feedback: tuple[str, ...] = ()
    for attempt in range(1, MAX_REGENERATION_ATTEMPTS + 1):
        counters["stage_ii_requests"] += 1
        messages = render_atom_messages(
            row,
            certified_ansatz=candidate.template,
            slot_id=slot.slot_id,
            current_ansatz=slot.current_ansatz,
            profile_call=sp.sstr(slot.call),
            profile_arguments=tuple(sp.sstr(item) for item in slot.call.args),
            validation_feedback=feedback,
        )
        recorder = RecordingProvider(provider_factory(4_096))
        attempt_record: dict[str, Any] = {
            "attempt": attempt,
            "host_feedback_sent": list(feedback),
            "rendered_messages_sha256": canonical_sha256(messages),
        }
        try:
            completion = recorder.complete(
                tokens=(slot.slot_id, "direct_atom"),
                messages=messages,
                response_format=atom_response_format(
                    slot.argument_roles, permissive=True
                ),
            )
        except (LLMResponseError, TypeError, ValueError) as exc:
            counters["stage_ii_provider_errors"] += 1
            attempt_record.update(status="provider_error", error=f"{type(exc).__name__}: {exc}")
            attempts.append(attempt_record)
            continue
        counters["prompt_tokens_ii"] += completion.prompt_tokens
        counters["completion_tokens_ii"] += completion.completion_tokens
        try:
            accepted, rejected = validate_atom_response(
                completion.response, slot, permissive=True
            )
        except (SafeSymbolicError, StagedExperimentError, TypeError, ValueError) as exc:
            counters["stage_ii_host_errors"] += 1
            attempt_record.update(
                status="host_response_error",
                raw_response=completion.response,
                provider_evidence=_provider_evidence(completion),
                error=f"{type(exc).__name__}: {exc}",
            )
            attempts.append(attempt_record)
            feedback = (_safe_feedback(f"Host Atom parse error: {exc}"),)
            continue
        counters["stage_ii_host_errors"] += len(rejected)
        for item in accepted:
            key = str(item["canonical_template"])
            if key not in retained:
                normalized = dict(item)
                normalized["rank"] = len(retained) + 1
                retained[key] = normalized
        attempt_record.update(
            status="completed",
            raw_response=completion.response,
            provider_evidence=_provider_evidence(completion),
            accepted_candidates=list(accepted),
            rejected_candidates=list(rejected),
            accepted_count_after_attempt=len(retained),
        )
        attempts.append(attempt_record)
        if retained:
            break
        feedback = _atom_retry_feedback(rejected, accepted)
    return tuple(retained.values()), attempts


def run_staged_experiment_direct(
    rows: Sequence[Mapping[str, Any]],
    validation_specs: Mapping[str, Mapping[str, Any]],
    *,
    provider_factory: Callable[[int], LLMBatchProviderV1],
    requested_model: str,
    case_observer: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the permissive direct Atom lane used by new V11 runs."""

    counters = {
        "stage_i_requests": 0,
        "stage_i_provider_errors": 0,
        "stage_i_syntax_errors": 0,
        "stage_i_exhausted": 0,
        "stage_ii_requests": 0,
        "stage_ii_provider_errors": 0,
        "stage_ii_host_errors": 0,
        "stage_ii_exhausted": 0,
        "semantic_validation_runs": 0,
        "prompt_tokens_i": 0,
        "completion_tokens_i": 0,
        "prompt_tokens_ii": 0,
        "completion_tokens_ii": 0,
    }
    cases: list[dict[str, Any]] = []

    for row in rows:
        case_id = str(row["benchmark_id"])
        case_record: dict[str, Any] = {
            "case_id": case_id,
            "claim_type": (
                "in_sample_full_problem_condition_aware_"
                "permissive_direct_atom_diagnostic"
            ),
            "atom_ansatz_policy": "prompt_ordered_syntax_only_atom_generation",
            "stage_i": {},
            "certified_ansatz_runs": [],
            "assembled_candidates": [],
            "fitted_candidates": [],
            "selected": None,
        }

        certified: list[tuple[LLMAnsatzCandidateV1, AnsatzReplayCertificateV1]] = []
        stage_i_attempts: list[dict[str, Any]] = []
        seen: set[str] = set()
        feedback: tuple[str, ...] = ()
        for attempt in range(1, MAX_REGENERATION_ATTEMPTS + 1):
            counters["stage_i_requests"] += 1
            recorder = RecordingProvider(provider_factory(8_192))
            attempt_record: dict[str, Any] = {
                "attempt": attempt,
                "host_feedback_sent": list(feedback),
            }
            try:
                pools, evidence = propose_llm_ansatz_batch(
                    (row,),
                    provider=recorder,
                    allow_empty_cases=True,
                    validation_feedback=feedback,
                    permissive=True,
                )
            except (LLMAnsatzError, LLMResponseError, TypeError, ValueError) as exc:
                counters["stage_i_provider_errors"] += 1
                attempt_record.update(
                    status="provider_or_schema_error",
                    error=f"{type(exc).__name__}: {exc}",
                )
                stage_i_attempts.append(attempt_record)
                continue
            if recorder.completion is None:
                raise StagedExperimentError("Stage I provider returned no completion evidence")
            counters["prompt_tokens_i"] += recorder.completion.prompt_tokens
            counters["completion_tokens_i"] += recorder.completion.completion_tokens
            raw_response = recorder.completion.response
            host_rows: list[dict[str, Any]] = []
            for rank, raw in enumerate(raw_response["cases"][0]["candidates"], 1):
                try:
                    ansatz, reason = validate_llm_ansatz_candidate(
                        raw,
                        dependent_variables=row["dependent_variables"],
                        independent_variables=row["independent_variables"],
                        parameters=public_problem_payload(row)["parameters"],
                        permissive=True,
                    )
                    candidate = LLMAnsatzCandidateV1(
                        rank=rank,
                        candidate_id=f"{case_id}:stage_i:{rank}",
                        family="condition_aware_high_dimensional_structure",
                        template=ansatz,
                        reason=reason,
                        model=requested_model,
                    )
                    certificate = _syntax_only_certificate(row, candidate)
                    key = re.sub(r"\s+", "", ansatz)
                    if key not in seen and len(certified) < STAGE_I_CANDIDATE_COUNT:
                        certified.append((candidate, certificate))
                        seen.add(key)
                    host_rows.append(
                        {
                            "rank": rank,
                            "ansatz": ansatz,
                            "reason": reason,
                            "host_syntax_validation_passed": True,
                            "validation_kind": certificate.validation_kind,
                        }
                    )
                except (LLMAnsatzError, StagedExperimentError, TypeError, ValueError) as exc:
                    counters["stage_i_syntax_errors"] += 1
                    host_rows.append(
                        {
                            "rank": rank,
                            "raw_candidate": raw,
                            "host_syntax_validation_passed": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            attempt_record.update(
                status="completed",
                raw_response=raw_response,
                provider_evidence=_provider_evidence(recorder.completion),
                generation_evidence=evidence,
                host_syntax_results=host_rows,
                symbolic_replay_results=[],
                symbolic_replay_policy="disabled_in_permissive_direct_lane",
                certified_count_after_attempt=len(certified),
            )
            stage_i_attempts.append(attempt_record)
            if len(certified) >= STAGE_I_CANDIDATE_COUNT:
                break
            feedback = (
                _safe_feedback(
                    f"Retain every atom-free parseable Ansatz; "
                    f"{STAGE_I_CANDIDATE_COUNT - len(certified)} more distinct candidate(s) needed."
                ),
            )
        else:
            counters["stage_i_exhausted"] += 1

        ranked_certified = _rank_direct_ansatz_candidates(row, certified)
        case_record["stage_i"] = {
            "status": "completed" if certified else "no_certified_ansatz",
            "attempt_count": len(stage_i_attempts),
            "max_attempts": MAX_REGENERATION_ATTEMPTS,
            "stop_reason": (
                "syntax_only_atom_free_candidates" if certified else "max_attempts_exhausted"
            ),
            "attempts": stage_i_attempts,
            "certified_count": len(certified),
            "certified_candidates": [
                {
                    "provider_rank": candidate.rank,
                    "execution_rank": execution_rank,
                    "ansatz": candidate.template,
                    "reason": candidate.reason,
                    "validation_kind": certificate.validation_kind,
                    "richness": richness.to_mapping(),
                }
                for execution_rank, (candidate, certificate, richness) in enumerate(
                    ranked_certified, 1
                )
            ],
        }

        spec = validation_specs.get(case_id)
        assembled_all: list[dict[str, Any]] = []
        for execution_rank, (candidate, certificate, richness) in enumerate(
            ranked_certified[:MAX_CERTIFIED_ANSATZ_RUNS_PER_CASE], 1
        ):
            ansatz_record: dict[str, Any] = {
                "ansatz_rank": execution_rank,
                "execution_rank": execution_rank,
                "provider_rank": candidate.rank,
                "ansatz": candidate.template,
                "symbolic_validation_kind": certificate.validation_kind,
                "richness": richness.to_mapping(),
                "slots": [],
                "status": "pending",
            }
            try:
                ast, slots = _extract_direct_runtime_slots(row, candidate.template)
            except (LLMAnsatzError, StagedExperimentError, TypeError, ValueError) as exc:
                ansatz_record.update(
                    status="slot_extraction_error",
                    error=f"{type(exc).__name__}: {exc}",
                )
                case_record["certified_ansatz_runs"].append(ansatz_record)
                continue
            pools_for_ansatz: list[tuple[dict[str, Any], ...]] = []
            ansatz_blocked = False
            for slot in slots:
                pool, attempts = _direct_atom_pool(
                    row,
                    candidate,
                    slot,
                    provider_factory=provider_factory,
                    counters=counters,
                )
                if not pool:
                    ansatz_blocked = True
                pools_for_ansatz.append(pool)
                ansatz_record["slots"].append(
                    {
                        **slot.public_mapping(),
                        "status": "completed" if pool else "max_attempts_exhausted",
                        "attempt_count": len(attempts),
                        "attempts": attempts,
                        "accepted_candidates": list(pool),
                        "validation_policy": "syntax_only_executable_expression",
                    }
                )
            if ansatz_blocked:
                counters["stage_ii_exhausted"] += 1
                ansatz_record["status"] = "blocked_before_assembly"
                case_record["certified_ansatz_runs"].append(ansatz_record)
                continue
            assembled = assemble_candidate_beam(
                ast,
                slots,
                pools_for_ansatz,
                ansatz_rank=execution_rank,
                max_combinations=MAX_ATOM_COMBINATIONS,
            )
            assembled_all.extend(assembled)
            ansatz_record["assembled_candidate_count"] = len(assembled)
            if spec is not None and assembled:
                screened = screen_semantic_candidate_beam(
                    row,
                    assembled,
                    spec,
                    max_candidates=MAX_SEMANTIC_FITS_PER_ANSATZ,
                )
                counters["semantic_validation_runs"] += 1
                fitted, selected, diagnostics = semantic_validate_candidates(
                    row, screened, spec
                )
                ansatz_record["semantic_rounds"] = [
                    {
                        "round": 0,
                        "candidate_count": len(assembled),
                        "screened_count": len(screened),
                        "fitted_count": len(fitted),
                        "best_validation_loss": (
                            None if selected is None else selected.get("validation_loss")
                        ),
                        "best_validation_max_component_nmse": (
                            None
                            if selected is None
                            else selected.get("validation_max_component_nmse")
                        ),
                        "passed": bool(
                            selected and selected.get("semantic_validation_passed")
                        ),
                        "diagnostics": list(diagnostics),
                        "atom_assessments": {
                            slot_id: list(items)
                            for slot_id, items in assess_atom_pool_semantics(
                                fitted, slots, pools_for_ansatz
                            ).items()
                        },
                    }
                ]
            ansatz_record["status"] = (
                "assembled_generation_only" if spec is None else "semantic_evaluated"
            )
            case_record["certified_ansatz_runs"].append(ansatz_record)

        case_record["assembled_candidates"] = assembled_all
        if spec is None:
            case_record["status"] = "staged_generation_only_unscorable_public_forcing"
            case_record["optimization_blocker"] = (
                "no executable public-only residual/condition specification; optimization not run"
            )
        elif not assembled_all:
            case_record["status"] = "no_fully_staged_candidate"
        else:
            semantic_beam = screen_semantic_candidate_beam(
                row,
                assembled_all,
                spec,
                max_candidates=MAX_SEMANTIC_FITS_PER_CASE,
            )
            counters["semantic_validation_runs"] += 1
            fitted, selected, diagnostics = semantic_validate_candidates(
                row, semantic_beam, spec
            )
            case_record["fitted_candidates"] = list(fitted)
            case_record["selected"] = selected
            case_record["semantic_diagnostics"] = list(diagnostics)
            if selected is None:
                case_record["status"] = "no_valid_fit"
            elif selected["semantic_validation_passed"]:
                case_record["status"] = "semantic_validation_passed"
            else:
                case_record["status"] = "semantic_validation_failed"
        cases.append(case_record)
        if case_observer is not None:
            case_observer(case_record)

    return {
        "schema_version": "StagedAnsatzDirectPermissiveAtomNMSEDiagnosticV12",
        "claim_type": (
            "in_sample_full_problem_condition_aware_"
            "permissive_direct_atom_diagnostic"
        ),
        "gold_accessed": False,
        "reference_solution_accessed": False,
        "generation_input_policy": (
            "public PDE/forcing/IC/BC/domain/parameters to Stage I and to each "
            "slot-bound direct Atom request; no offline gold/reference during generation"
        ),
        "provider": {
            "name": "openrouter",
            "model": requested_model,
            "sampling_policy": {"temperature": 0.0, "top_p": 1.0},
            "transport_retry_policy": {
                "max_retries": 1,
                "backoff_seconds": [1],
                "fallbacks": False,
            },
            "stage_i": {
                "request_count": counters["stage_i_requests"],
                "provider_error_count": counters["stage_i_provider_errors"],
                "host_syntax_candidate_error_count": counters["stage_i_syntax_errors"],
                "symbolic_replay_error_count": 0,
                "max_attempts_exhausted_case_count": counters["stage_i_exhausted"],
                "prompt_tokens": counters["prompt_tokens_i"],
                "completion_tokens": counters["completion_tokens_i"],
            },
            "stage_ii": {
                "request_count": counters["stage_ii_requests"],
                "provider_error_count": counters["stage_ii_provider_errors"],
                "host_candidate_or_response_error_count": counters["stage_ii_host_errors"],
                "max_attempts_exhausted_slot_count": counters["stage_ii_exhausted"],
                "semantic_regeneration_request_count": 0,
                "host_semantic_validation_run_count": counters["semantic_validation_runs"],
                "prompt_tokens": counters["prompt_tokens_ii"],
                "completion_tokens": counters["completion_tokens_ii"],
            },
        },
        "policies": {
            "ansatz_symbolic_validation": (
                "atom-free syntax/slot validation only; PDE replay disabled"
            ),
            "atom_generation": (
                "direct LLM output per open slot; no tree seed or host grammar expansion"
            ),
            "atom_host_acceptance": (
                "parseable executable expression using declared ARG symbols and fitted p symbols"
            ),
            "parameterization": (
                "not enforced; coefficients and fixed/variable shape are left to L-BFGS-B"
            ),
            "complexity_and_topology_filters": (
                "disabled except finite parser/runtime safeguards"
            ),
            "stage_i_candidates_per_response": STAGE_I_CANDIDATE_COUNT,
            "stage_ii_candidates_per_response": STAGE_II_CANDIDATE_COUNT,
            "max_fitted_coefficients": MAX_FITTED_COEFFICIENTS,
            "semantic_acceptance_max_component_nmse": SEMANTIC_ACCEPTANCE_LOSS,
            "selection": (
                "minimum public validation residual across every assembled candidate "
                "and every Ansatz"
            ),
        },
        "case_count": len(cases),
        "cases": cases,
    }


__all__ = ["run_staged_experiment_direct"]
