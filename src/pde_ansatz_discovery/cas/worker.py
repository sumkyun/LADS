"""Fresh-process CAS execution using canonical project-IR JSON bytes only."""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import signal
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from ..ir import ExpressionIRV1, SymbolRoleV1
from ..models import DiscoveryContractError, WorkerStatusV1
from ..serialization import canonical_json_bytes, freeze_json, thaw_json


class WorkerOperationV1(str, Enum):
    ROUND_TRIP = "round_trip"
    DIFFERENTIATE = "differentiate"
    EXPAND = "expand"
    FACTOR = "factor"
    TOGETHER = "together"
    FACTOR_TERMS = "factor_terms"
    PROVE_ZERO = "prove_zero"
    VERIFY_DIRECT = "verify_direct"
    VERIFY_SEPARATION = "verify_separation"
    APPLY_CONDITIONS = "apply_conditions"
    CLASSICAL_LIE_SEEDS = "classical_lie_seeds"
    SOLVE_PARAMETER_BRANCHES = "solve_parameter_branches"
    VERIFY_NONCLASSICAL = "verify_nonclassical"
    DEDUPLICATE_CANDIDATES = "deduplicate_candidates"
    REPLAY_EQUIVALENCE = "replay_equivalence"
    CHECK_BOUNDARY_PRESERVATION = "check_boundary_preservation"
    SLEEP = "sleep"
    CRASH = "crash"
    OVERSIZE = "oversize"


@dataclass(frozen=True)
class WorkerRequestV1:
    operation: WorkerOperationV1
    expression: ExpressionIRV1 | None
    independent_variables: tuple[str, ...] = ("x", "t")
    symbol_roles: Mapping[str, str] = None  # type: ignore[assignment]
    options: Mapping[str, Any] = None  # type: ignore[assignment]

    schema_version = "WorkerRequestV1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", WorkerOperationV1(self.operation))
        if self.expression is not None and not isinstance(self.expression, ExpressionIRV1):
            raise DiscoveryContractError("worker expression must be project IR")
        variables = tuple(self.independent_variables)
        if any(not isinstance(item, str) or not item for item in variables):
            raise DiscoveryContractError("worker independent variables must be strings")
        object.__setattr__(self, "independent_variables", variables)
        roles = dict(self.symbol_roles or {})
        for name, role in roles.items():
            if not isinstance(name, str):
                raise DiscoveryContractError("worker role names must be strings")
            SymbolRoleV1(role)
        object.__setattr__(self, "symbol_roles", freeze_json(roles))
        object.__setattr__(self, "options", freeze_json(self.options or {}))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "operation": self.operation.value,
            "expression": None if self.expression is None else self.expression.to_mapping(),
            "independent_variables": list(self.independent_variables),
            "symbol_roles": thaw_json(self.symbol_roles),
            "options": thaw_json(self.options),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> WorkerRequestV1:
        expected = {"schema_version", "operation", "expression", "independent_variables", "symbol_roles", "options"}
        if set(value) != expected or value["schema_version"] != cls.schema_version:
            raise DiscoveryContractError("invalid worker request schema")
        expression = None if value["expression"] is None else ExpressionIRV1.from_mapping(value["expression"])
        return cls(value["operation"], expression, tuple(value["independent_variables"]), value["symbol_roles"], value["options"])


@dataclass(frozen=True)
class WorkerResultV1:
    status: WorkerStatusV1
    payload: Mapping[str, Any] | None
    error_code: str | None
    elapsed_seconds: float
    exit_code: int | None

    schema_version = "WorkerResultV1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", WorkerStatusV1(self.status))
        object.__setattr__(self, "payload", None if self.payload is None else freeze_json(self.payload))
        if not isinstance(self.elapsed_seconds, (int, float)) or self.elapsed_seconds < 0:
            raise DiscoveryContractError("worker elapsed time must be nonnegative")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "payload": None if self.payload is None else thaw_json(self.payload),
            "error_code": self.error_code,
            "elapsed_seconds": self.elapsed_seconds,
            "exit_code": self.exit_code,
        }


def _resource_limits(memory_limit_bytes: int, cpu_seconds: int) -> None:
    try:
        import resource
    except ImportError as exc:
        raise DiscoveryContractError("CAS_RESOURCE_LIMIT_UNAVAILABLE") from exc
    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    except (OSError, ValueError) as exc:
        raise DiscoveryContractError("CAS_RESOURCE_LIMIT_REJECTED") from exc


def _child(send_connection, request_bytes: bytes, memory_limit_bytes: int, cpu_seconds: int) -> None:
    started = time.monotonic()
    try:
        _resource_limits(memory_limit_bytes, cpu_seconds)
        request = WorkerRequestV1.from_mapping(json.loads(request_bytes))
        if request.operation is WorkerOperationV1.CRASH:
            os._exit(97)
        if request.operation is WorkerOperationV1.SLEEP:
            time.sleep(float(request.options.get("seconds", 10)))
            payload: dict[str, Any] = {"slept": True}
        elif request.operation is WorkerOperationV1.OVERSIZE:
            payload = {"data": "x" * int(request.options.get("size", 1_000_000))}
        elif request.operation is WorkerOperationV1.VERIFY_DIRECT:
            from ..generation.direct import DirectAnsatzV1
            from ..models import DiscoveryProblemInputV1
            from ..verification.reduction import verify_direct_candidate

            problem = DiscoveryProblemInputV1.from_mapping(request.options["problem"])
            ansatz = DirectAnsatzV1.from_mapping(request.options["ansatz"])
            payload = {"candidate": verify_direct_candidate(problem, ansatz).to_mapping()}
        elif request.operation is WorkerOperationV1.VERIFY_SEPARATION:
            from ..generation.separation import SeparationAnsatzV1
            from ..models import DiscoveryProblemInputV1
            from ..verification.separation import verify_separation_candidate

            problem = DiscoveryProblemInputV1.from_mapping(request.options["problem"])
            separation = SeparationAnsatzV1.from_mapping(request.options["request"])
            payload = {
                "candidate": verify_separation_candidate(problem, separation).to_mapping()
            }
        elif request.operation is WorkerOperationV1.APPLY_CONDITIONS:
            from ..models import DiscoveryCandidateV1, DiscoveryProblemInputV1
            from ..verification.conditions import apply_condition_compatibility

            problem = DiscoveryProblemInputV1.from_mapping(request.options["problem"])
            candidate = DiscoveryCandidateV1.from_mapping(request.options["candidate"])
            payload = {
                "candidate": apply_condition_compatibility(problem, candidate).to_mapping()
            }
        elif request.operation is WorkerOperationV1.CLASSICAL_LIE_SEEDS:
            from ..generation.lie import generate_classical_lie_seeds
            from ..models import DiscoveryProblemInputV1

            problem = DiscoveryProblemInputV1.from_mapping(request.options["problem"])
            seeds = generate_classical_lie_seeds(problem)
            payload = {
                "seeds": [
                    {
                        "ansatz": seed.ansatz.to_mapping(),
                        "generator": {
                            "time_component": seed.generator.time_component.to_mapping(),
                            "spatial_component": seed.generator.spatial_component.to_mapping(),
                            "field_component": seed.generator.field_component.to_mapping(),
                            "field": seed.generator.field,
                            "spatial_variable": seed.generator.spatial_variable,
                            "time_variable": seed.generator.time_variable,
                        },
                        "determining": {
                            "mode": seed.determining.mode,
                            "invariant": seed.determining.invariant,
                            "raw": seed.determining.raw_determining_expression.to_mapping(),
                            "reduced": seed.determining.reduced_determining_expression.to_mapping(),
                            "proof_status": seed.determining.proof_status.value,
                            "reason_code": seed.determining.reason_code,
                            "invariant_surface": None
                            if seed.determining.invariant_surface_condition is None
                            else seed.determining.invariant_surface_condition.to_mapping(),
                        },
                        "generator_label": seed.generator_label,
                        "solver_parameters": list(seed.solver_parameters),
                        "basis_degree": seed.basis_degree,
                    }
                    for seed in seeds
                ]
            }
        elif request.operation is WorkerOperationV1.SOLVE_PARAMETER_BRANCHES:
            from ..generation.direct import (
                DirectAnsatzV1,
                materialized_parameter_branch,
            )
            from ..ir import ParserContextV1, parse_expression, parse_problem
            from .sympy_backend import SymPyBackendV1

            ansatz = DirectAnsatzV1.from_mapping(request.options["ansatz"])
            equations = tuple(
                ExpressionIRV1.from_mapping(item)
                for item in request.options["equations"]
            )
            parameters = tuple(request.options["parameters"])
            branch_cap = int(request.options["branch_cap"])
            roles = {
                name: SymbolRoleV1.INDEPENDENT
                for name in request.independent_variables
            }
            roles.update(
                {name: SymbolRoleV1.PARAMETER for name in ansatz.free_parameters}
            )
            backend = SymPyBackendV1(
                request.independent_variables,
                roles,
                (ansatz.profile_function,),
            )
            branch_result = backend.solve_parameters(equations, parameters, branch_cap)
            context = ParserContextV1(
                (),
                tuple(
                    dict.fromkeys(
                        (*request.independent_variables, ansatz.coordinate_name)
                    )
                ),
                ansatz.free_parameters,
                (ansatz.profile_function,),
            )

            def substitute_text(
                value: str,
                replacements: Mapping[ExpressionIRV1, ExpressionIRV1],
            ) -> str:
                return backend.render(
                    backend.substitute(parse_expression(value, context), replacements)
                )

            materialized = []
            branch_verifications = []
            for assignments in branch_result.branches:
                replacements = {
                    parse_expression(name, context): value
                    for name, value in assignments.items()
                }

                assignment_text = {
                    name: backend.render(value) for name, value in assignments.items()
                }
                branch = (
                    ansatz.branch
                    if not assignments
                    else materialized_parameter_branch(
                        assignment_text, ansatz.domain_conditions
                    )
                )
                assert branch is not None
                materialized_ansatz = (
                    ansatz
                    if not assignments
                    else replace(
                        ansatz,
                        expression=substitute_text(ansatz.expression, replacements),
                        coordinate_expression=substitute_text(
                            ansatz.coordinate_expression, replacements
                        ),
                        inverse_map={
                            name: substitute_text(value, replacements)
                            for name, value in ansatz.inverse_map.items()
                        },
                        free_parameters=tuple(
                            name
                            for name in ansatz.free_parameters
                            if name not in assignments
                        ),
                        branch=branch,
                    )
                )
                materialized.append(materialized_ansatz)
                if "problem" in request.options and "generator" in request.options:
                    from ..models import DiscoveryProblemInputV1
                    from ..solving.determining_system import verify_classical_generator
                    from ..solving.prolongation import PointGeneratorV1

                    problem = DiscoveryProblemInputV1.from_mapping(
                        request.options["problem"]
                    )
                    generator_mapping = request.options["generator"]
                    generator = PointGeneratorV1(
                        backend.substitute(
                            ExpressionIRV1.from_mapping(
                                generator_mapping["time_component"]
                            ),
                            replacements,
                        ),
                        backend.substitute(
                            ExpressionIRV1.from_mapping(
                                generator_mapping["spatial_component"]
                            ),
                            replacements,
                        ),
                        backend.substitute(
                            ExpressionIRV1.from_mapping(
                                generator_mapping["field_component"]
                            ),
                            replacements,
                        ),
                        generator_mapping["field"],
                        generator_mapping["spatial_variable"],
                        generator_mapping["time_variable"],
                    )
                    residual = parse_problem(problem).equations[0].residual
                    verification = verify_classical_generator(
                        residual, generator, backend=backend
                    )
                    branch_verifications.append(
                        {
                            "branch_id": branch.branch_id,
                            "generator": {
                                "time_component": generator.time_component.to_mapping(),
                                "spatial_component": generator.spatial_component.to_mapping(),
                                "field_component": generator.field_component.to_mapping(),
                                "field": generator.field,
                                "spatial_variable": generator.spatial_variable,
                                "time_variable": generator.time_variable,
                            },
                            "determining": {
                                "mode": verification.mode,
                                "invariant": verification.invariant,
                                "raw": verification.raw_determining_expression.to_mapping(),
                                "reduced": verification.reduced_determining_expression.to_mapping(),
                                "proof_status": verification.proof_status.value,
                                "reason_code": verification.reason_code,
                                "invariant_surface": None,
                            },
                        }
                    )
            payload = {
                "branch_result": branch_result.to_mapping(),
                "materialized_ansatze": [item.to_mapping() for item in materialized],
                "branch_verifications": branch_verifications,
            }
        elif request.operation is WorkerOperationV1.VERIFY_NONCLASSICAL:
            from ..generation.nonclassical import verify_x_invariant_nonclassical
            from ..models import DiscoveryProblemInputV1

            problem = DiscoveryProblemInputV1.from_mapping(request.options["problem"])
            payload = {
                "candidate": verify_x_invariant_nonclassical(problem).to_mapping()
            }
        elif request.operation is WorkerOperationV1.DEDUPLICATE_CANDIDATES:
            from ..canonicalization import deduplicate_reduction_candidates
            from ..models import DiscoveryCandidateV1

            candidates = tuple(
                DiscoveryCandidateV1.from_mapping(item)
                for item in request.options["candidates"]
            )
            payload = {
                "candidates": [
                    item.to_mapping()
                    for item in deduplicate_reduction_candidates(candidates)
                ]
            }
        elif request.operation is WorkerOperationV1.REPLAY_EQUIVALENCE:
            from ..canonicalization.equivalence import (
                _replay_equivalence_witness_mapping_in_process,
            )

            payload = {
                "valid": _replay_equivalence_witness_mapping_in_process(
                    request.options["witness"]
                )
            }
        elif request.operation is WorkerOperationV1.CHECK_BOUNDARY_PRESERVATION:
            from ..analysis.boundaries import check_boundary_preservation_for_conditions
            from ..ir import ParserContextV1
            from ..models import DiscoveryProblemInputV1
            from .sympy_backend import SymPyBackendV1

            problem = DiscoveryProblemInputV1.from_mapping(request.options["problem"])
            component = ExpressionIRV1.from_mapping(request.options["spatial_component"])
            roles = {
                name: SymbolRoleV1.INDEPENDENT
                for name in problem.independent_variables
            }
            roles.update(
                {name: SymbolRoleV1.PARAMETER for name in problem.parameters}
            )
            backend = SymPyBackendV1(problem.independent_variables, roles)
            context = ParserContextV1(
                (), problem.independent_variables, tuple(problem.parameters)
            )
            checks = check_boundary_preservation_for_conditions(
                problem.boundary_conditions,
                backend.render(component),
                spatial_variable=problem.independent_variables[0],
                context=context,
                backend=backend,
            )
            payload = {
                "checks": [
                    {
                        "preserved": item.preserved,
                        "reason_code": item.reason_code,
                        "tangency_expression": item.tangency_expression,
                    }
                    for item in checks
                ]
            }
        else:
            from .sympy_backend import SymPyBackendV1

            backend = SymPyBackendV1(
                request.independent_variables,
                request.symbol_roles,
                tuple(request.options.get("declared_functions", ())),
            )
            if request.expression is None:
                raise DiscoveryContractError("CAS operation requires an expression")
            if request.operation is WorkerOperationV1.ROUND_TRIP:
                payload = {"expression": backend.round_trip(request.expression).to_mapping()}
            elif request.operation is WorkerOperationV1.DIFFERENTIATE:
                payload = {
                    "expression": backend.differentiate(
                        request.expression,
                        str(request.options["variable"]),
                        int(request.options.get("order", 1)),
                    ).to_mapping()
                }
            elif request.operation is WorkerOperationV1.EXPAND:
                payload = {"expression": backend.expand(request.expression).to_mapping()}
            elif request.operation is WorkerOperationV1.FACTOR:
                payload = {"expression": backend.factor(request.expression).to_mapping()}
            elif request.operation is WorkerOperationV1.TOGETHER:
                payload = {"expression": backend.together(request.expression).to_mapping()}
            elif request.operation is WorkerOperationV1.FACTOR_TERMS:
                payload = {"expression": backend.factor_terms(request.expression).to_mapping()}
            elif request.operation is WorkerOperationV1.PROVE_ZERO:
                payload = {"proof": backend.prove_zero(request.expression).to_mapping()}
            else:
                raise DiscoveryContractError("unsupported worker operation")
        response = {
            "ok": True,
            "payload": payload,
            "elapsed_seconds": time.monotonic() - started,
        }
    except BaseException as exc:  # noqa: BLE001 - child boundary serializes fatal failures
        response = {
            "ok": False,
            "error_code": f"{type(exc).__name__}:{exc}",
            "elapsed_seconds": time.monotonic() - started,
        }
    try:
        send_connection.send_bytes(canonical_json_bytes(response))
    finally:
        send_connection.close()


def _classify_child_failure(error_code: str) -> tuple[WorkerStatusV1, str]:
    resource_setup_prefix = "DiscoveryContractError:CAS_RESOURCE_LIMIT_"
    if error_code.startswith(resource_setup_prefix):
        return WorkerStatusV1.RESOURCE_EXHAUSTED, error_code.removeprefix(
            "DiscoveryContractError:"
        )
    if error_code.startswith("MemoryError:") or "Cannot allocate memory" in error_code:
        return WorkerStatusV1.RESOURCE_EXHAUSTED, "CAS_MEMORY_RESOURCE_EXHAUSTED"
    return WorkerStatusV1.CRASHED, error_code


def run_worker_job(
    request: WorkerRequestV1,
    *,
    deadline_seconds: float,
    memory_limit_bytes: int = 768 * 1024 * 1024,
    cpu_seconds: int = 30,
    result_size_limit: int = 4 * 1024 * 1024,
) -> WorkerResultV1:
    """Run exactly one request in a fresh spawn child and discard its channel afterward."""

    if (
        isinstance(deadline_seconds, bool)
        or not isinstance(deadline_seconds, (int, float))
        or not math.isfinite(float(deadline_seconds))
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (memory_limit_bytes, cpu_seconds, result_size_limit)
        )
        or deadline_seconds <= 0
        or memory_limit_bytes < 1
        or cpu_seconds < 1
        or result_size_limit < 1
    ):
        raise ValueError("worker resource limits must be finite and positive")
    started = time.monotonic()
    context = multiprocessing.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_child,
        args=(
            send_connection,
            canonical_json_bytes(request.to_mapping()),
            memory_limit_bytes,
            cpu_seconds,
        ),
    )
    try:
        process.start()
    except BaseException:
        receive_connection.close()
        send_connection.close()
        raise
    send_connection.close()

    def stop_process() -> None:
        if process.is_alive():
            process.terminate()
            process.join(1)
        if process.is_alive():
            process.kill()
        process.join()

    def exited_from_resource_limit(exit_code: int | None) -> bool:
        return exit_code in {
            -getattr(signal, "SIGXCPU", 24),
            -getattr(signal, "SIGKILL", 9),
        }

    # Read before join: a child sending an allowed multi-page response can fill the pipe
    # and block until the parent drains it. Joining first would misclassify that as timeout.
    if not receive_connection.poll(deadline_seconds):
        stop_process()
        receive_connection.close()
        return WorkerResultV1(
            WorkerStatusV1.TIMEOUT,
            None,
            "CAS_TIMEOUT",
            time.monotonic() - started,
            process.exitcode,
        )
    try:
        response_bytes = receive_connection.recv_bytes(result_size_limit + 1)
    except OSError:
        stop_process()
        return WorkerResultV1(
            WorkerStatusV1.RESOURCE_EXHAUSTED,
            None,
            "CAS_RESULT_SIZE_EXCEEDED",
            time.monotonic() - started,
            process.exitcode,
        )
    except EOFError:
        process.join(max(0.0, deadline_seconds - (time.monotonic() - started)))
        if process.is_alive():
            stop_process()
        resource_limited = exited_from_resource_limit(process.exitcode)
        return WorkerResultV1(
            WorkerStatusV1.RESOURCE_EXHAUSTED
            if resource_limited
            else WorkerStatusV1.CRASHED,
            None,
            "CAS_RESOURCE_LIMIT_EXIT" if resource_limited else "CAS_CRASHED",
            time.monotonic() - started,
            process.exitcode,
        )
    finally:
        receive_connection.close()
    process.join(max(0.0, deadline_seconds - (time.monotonic() - started)))
    if process.is_alive():
        stop_process()
        return WorkerResultV1(
            WorkerStatusV1.TIMEOUT,
            None,
            "CAS_TIMEOUT",
            time.monotonic() - started,
            process.exitcode,
        )
    exit_code = process.exitcode
    if len(response_bytes) > result_size_limit:
        return WorkerResultV1(
            WorkerStatusV1.RESOURCE_EXHAUSTED,
            None,
            "CAS_RESULT_SIZE_EXCEEDED",
            time.monotonic() - started,
            exit_code,
        )
    try:
        response = json.loads(response_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return WorkerResultV1(
            WorkerStatusV1.CRASHED,
            None,
            "CAS_INVALID_RESPONSE",
            time.monotonic() - started,
            exit_code,
        )
    if not response.get("ok"):
        child_error = str(response.get("error_code", "CAS_CHILD_ERROR"))
        status, error_code = _classify_child_failure(child_error)
        return WorkerResultV1(
            status,
            None,
            error_code,
            time.monotonic() - started,
            exit_code,
        )
    return WorkerResultV1(
        WorkerStatusV1.COMPLETED,
        response["payload"],
        None,
        time.monotonic() - started,
        exit_code,
    )


__all__ = [
    "WorkerOperationV1",
    "WorkerRequestV1",
    "WorkerResultV1",
    "run_worker_job",
]
