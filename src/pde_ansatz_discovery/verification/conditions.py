"""Fail-closed symbolic condition compatibility, independent from PDE validity."""

from __future__ import annotations

import re
from dataclasses import replace
from fractions import Fraction

from ..cas.sympy_backend import SymPyBackendV1
from ..ir import (
    ExpressionIRV1,
    ParserContextV1,
    SymbolRoleV1,
    parse_expression,
    rational,
    subtract,
)
from ..models import (
    ConditionCompatibilityV1,
    ConditionRecordV1,
    ConditionStatusV1,
    DiscoveryCandidateV1,
    DiscoveryProblemInputV1,
    ProofStatusV1,
)

_SPACE = re.compile(r"\s+")
_FIELD_CALL = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)\((.*)\)$")


def _condition_parts(source: str) -> tuple[str, str] | None:
    if source.count("=") != 1:
        return None
    left, right = (_SPACE.sub("", item) for item in source.split("=", 1))
    return (left, right) if left and right else None


def _arguments(source: str) -> tuple[str, ...] | None:
    values: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(source):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                return None
        elif character == "," and depth == 0:
            values.append(source[start:index])
            start = index + 1
    if depth != 0:
        return None
    values.append(source[start:])
    return tuple(value for value in values if value)


def _is_singular_initial_condition(
    source: str,
    problem: DiscoveryProblemInputV1,
    candidate: DiscoveryCandidateV1,
) -> bool | None:
    certificate = candidate.verification.certificate
    if certificate is None:
        return False
    parts = _condition_parts(source)
    if parts is None:
        return False
    match = _FIELD_CALL.fullmatch(parts[0])
    if match is None:
        return False
    arguments = _arguments(match.group(2))
    if arguments is None or len(arguments) != len(problem.independent_variables):
        return False
    context, backend = _condition_context(problem, candidate)
    replacements = {
        parse_expression(variable, context): parse_expression(value, context)
        for variable, value in zip(problem.independent_variables, arguments)
    }
    unresolved_entry = False
    for singular in certificate.chart.singular_set:
        singular_parts = _condition_parts(singular)
        if singular_parts is None:
            unresolved_entry = True
            continue
        try:
            singular_expression = subtract(
                parse_expression(singular_parts[0], context),
                parse_expression(singular_parts[1], context),
            )
            evaluated = backend.substitute(singular_expression, replacements)
        except Exception:  # noqa: BLE001 - failures are propagated as unresolved
            unresolved_entry = True
            continue
        if backend.prove_zero(evaluated).status is ProofStatusV1.PROVED_ZERO:
            return True
    return None if unresolved_entry else False


def _display_expression(candidate: DiscoveryCandidateV1) -> str:
    return candidate.ansatz.rsplit("=", 1)[-1]


def _condition_context(
    problem: DiscoveryProblemInputV1,
    candidate: DiscoveryCandidateV1,
) -> tuple[ParserContextV1, SymPyBackendV1]:
    parameters = tuple(
        dict.fromkeys((*problem.parameters, *candidate.free_parameters))
    )
    context = ParserContextV1(
        (),
        tuple(dict.fromkeys((*problem.independent_variables, "eta"))),
        parameters,
        candidate.profile_functions,
    )
    roles = {
        name: SymbolRoleV1.INDEPENDENT for name in problem.independent_variables
    }
    roles["eta"] = SymbolRoleV1.PROFILE_COORDINATE
    roles.update({name: SymbolRoleV1.PARAMETER for name in parameters})
    return context, SymPyBackendV1(
        problem.independent_variables, roles, candidate.profile_functions
    )


def _condition_derivatives(
    name: str,
    *,
    field: str,
    independent_variables: tuple[str, ...],
) -> tuple[str, ...] | None:
    if name == field:
        return ()
    prefix = f"{field}_"
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix) :]
    if not suffix:
        return None
    if "_" in suffix:
        variables = tuple(suffix.split("_"))
        return variables if all(item in independent_variables for item in variables) else None
    if all(len(item) == 1 for item in independent_variables):
        variables = tuple(suffix)
        return variables if all(item in independent_variables for item in variables) else None
    return (suffix,) if suffix in independent_variables else None


def _symbolic_constraint(
    source: str,
    problem: DiscoveryProblemInputV1,
    candidate: DiscoveryCandidateV1,
) -> tuple[ExpressionIRV1, ExpressionIRV1, ProofStatusV1, bool] | None:
    parts = _condition_parts(source)
    if parts is None:
        return None
    match = _FIELD_CALL.fullmatch(parts[0])
    field = problem.dependent_variables[0]
    if match is None:
        return None
    derivatives = _condition_derivatives(
        match.group(1),
        field=field,
        independent_variables=problem.independent_variables,
    )
    if derivatives is None:
        return None
    arguments = _arguments(match.group(2))
    if arguments is None or len(arguments) != len(problem.independent_variables):
        return None
    context, backend = _condition_context(problem, candidate)
    expression = parse_expression(_display_expression(candidate), context)
    for variable in derivatives:
        expression = backend.differentiate(expression, variable)
    replacements = {
        parse_expression(variable, context): parse_expression(value, context)
        for variable, value in zip(problem.independent_variables, arguments)
    }
    constrained = backend.factor_terms(backend.substitute(expression, replacements))
    right = backend.factor_terms(parse_expression(parts[1], context))
    proof = backend.prove_zero(subtract(constrained, right)).status
    adjustable = (
        set(candidate.profile_functions)
        | set(candidate.free_parameters)
        | set(problem.parameters)
    )
    direct_contradiction = (
        proof is ProofStatusV1.PROVED_NONZERO
        and not adjustable.intersection(constrained.free_names | right.free_names)
    )
    return (
        constrained,
        right,
        proof,
        direct_contradiction,
    )


def _condition_records(
    kind: str,
    sources: tuple[str, ...],
    problem: DiscoveryProblemInputV1,
    candidate: DiscoveryCandidateV1,
) -> list[ConditionRecordV1]:
    parsed: list[
        tuple[str, ExpressionIRV1, ExpressionIRV1, ProofStatusV1, bool]
    ] = []
    records: list[ConditionRecordV1] = []
    for source in sources:
        singular_status = (
            _is_singular_initial_condition(source, problem, candidate)
            if kind == "initial"
            else False
        )
        if singular_status is True:
            records.append(
                ConditionRecordV1(
                    kind,
                    ConditionStatusV1.SINGULAR,
                    source,
                    {"reason_code": "INITIAL_TIME_ON_SINGULAR_SET"},
                )
            )
            continue
        if singular_status is None:
            records.append(
                ConditionRecordV1(
                    kind,
                    ConditionStatusV1.UNRESOLVED,
                    source,
                    {"reason_code": "SINGULAR_SET_REPLAY_UNRESOLVED"},
                )
            )
            continue
        try:
            constraint = _symbolic_constraint(source, problem, candidate)
        except Exception as exc:  # noqa: BLE001 - failures become explicit UNRESOLVED records
            constraint = None
            error = type(exc).__name__
        else:
            error = "unsupported_condition_shape"
        if constraint is None:
            records.append(
                ConditionRecordV1(
                    kind,
                    ConditionStatusV1.UNRESOLVED,
                    source,
                    {
                        "reason_code": f"{kind.upper()}_CONDITION_UNRESOLVED",
                        "detail": error,
                    },
                )
            )
            continue
        parsed.append((source, *constraint))

    context, backend = _condition_context(problem, candidate)
    values_by_left: dict[ExpressionIRV1, list[ExpressionIRV1]] = {}
    for _, left, right, _, _ in parsed:
        values_by_left.setdefault(left, []).append(right)
    conflicts: set[ExpressionIRV1] = set()
    unresolved_collisions: set[ExpressionIRV1] = set()
    adjustable = (
        set(candidate.profile_functions)
        | set(candidate.free_parameters)
        | set(problem.parameters)
    )
    for left, rights in values_by_left.items():
        for index, first in enumerate(rights):
            for second in rights[index + 1 :]:
                difference = subtract(first, second)
                proof = backend.prove_zero(difference).status
                if proof is ProofStatusV1.PROVED_ZERO:
                    continue
                has_adjustable_names = bool(adjustable & difference.free_names)
                witness_proof = ProofStatusV1.UNKNOWN
                if not has_adjustable_names:
                    zero_witness = {
                        parse_expression(variable, context): rational(0)
                        for variable in problem.independent_variables
                    }
                    witness_proof = backend.prove_zero(
                        backend.substitute(difference, zero_witness)
                    ).status
                if not has_adjustable_names and (
                    proof is ProofStatusV1.PROVED_NONZERO
                    or witness_proof is ProofStatusV1.PROVED_NONZERO
                ):
                    conflicts.add(left)
                else:
                    unresolved_collisions.add(left)
    for source, left, right, proof, direct_contradiction in parsed:
        if left in conflicts or direct_contradiction:
            status = ConditionStatusV1.INCOMPATIBLE
            reason_code = (
                "CONDITION_PROFILE_COLLISION"
                if left in conflicts
                else "CONDITION_DIRECT_CONTRADICTION"
            )
        elif left in unresolved_collisions:
            status = ConditionStatusV1.UNRESOLVED
            reason_code = "CONDITION_COLLISION_UNRESOLVED"
        elif proof is ProofStatusV1.PROVED_ZERO:
            status = ConditionStatusV1.COMPATIBLE
            reason_code = None
        else:
            status = ConditionStatusV1.COMPATIBLE_WITH_CONSTRAINTS
            reason_code = None
        records.append(
            ConditionRecordV1(
                f"{kind}:profile_collision" if left in conflicts else kind,
                status,
                f"{backend.render(left)}={backend.render(right)}",
                {
                    "method": "ansatz_substitution",
                    "reason_code": reason_code,
                    "source": source,
                },
            )
        )
    return records


def _spatial_scaling_exponent(
    problem: DiscoveryProblemInputV1, candidate: DiscoveryCandidateV1
) -> Fraction | None:
    spatial, time = problem.independent_variables
    compact = _SPACE.sub("", candidate.coordinate)
    if compact == f"{spatial}/sqrt({time})":
        return Fraction(1, 2)
    pattern = re.compile(
        rf"^{re.escape(spatial)}/{re.escape(time)}\*\*\(?(-?\d+)(?:/(\d+))?\)?$"
    )
    match = pattern.fullmatch(compact)
    if match is None:
        return None
    return Fraction(int(match.group(1)), int(match.group(2) or 1))


def _mass_record(
    source: str,
    problem: DiscoveryProblemInputV1,
    candidate: DiscoveryCandidateV1,
) -> ConditionRecordV1:
    if "scaling:pde_weight_balance" not in candidate.generator_key:
        return ConditionRecordV1(
            "additional",
            ConditionStatusV1.UNRESOLVED,
            source,
            {"reason_code": "MASS_BALANCE_NOT_A_SCALING_CANDIDATE"},
        )
    exponent = _spatial_scaling_exponent(problem, candidate)
    if exponent is None or "beta" not in candidate.free_parameters:
        return ConditionRecordV1(
            "additional",
            ConditionStatusV1.UNRESOLVED,
            source,
            {"reason_code": "MASS_BALANCE_EXPONENT_UNRESOLVED"},
        )
    beta = -exponent
    rendered = (
        str(beta.numerator)
        if beta.denominator == 1
        else f"{beta.numerator}/{beta.denominator}"
    )
    return ConditionRecordV1(
        "conservation:mass",
        ConditionStatusV1.COMPATIBLE_WITH_CONSTRAINTS,
        f"beta={rendered}",
        {
            "method": "amplitude_plus_spatial_jacobian_balance",
            "profile_fitted": False,
            "spatial_jacobian_exponent": str(exponent),
        },
    )


def analyze_candidate_conditions(
    problem: DiscoveryProblemInputV1,
    candidate: DiscoveryCandidateV1,
) -> ConditionCompatibilityV1:
    records = _condition_records(
        "initial", problem.initial_conditions, problem, candidate
    )
    records.extend(
        _condition_records(
            "boundary", problem.boundary_conditions, problem, candidate
        )
    )
    for source in problem.additional_constraints:
        normalized = _SPACE.sub("_", source.strip().lower())
        is_mass = normalized in {"mass_conservation", "conserve_mass"} or (
            "integral" in normalized and "constant" in normalized
        )
        records.append(
            _mass_record(source, problem, candidate)
            if is_mass
            else ConditionRecordV1(
                "additional",
                ConditionStatusV1.UNRESOLVED,
                source,
                {"reason_code": "ADDITIONAL_CONSTRAINT_UNSUPPORTED"},
            )
        )
    return ConditionCompatibilityV1(tuple(records))


def apply_condition_compatibility(
    problem: DiscoveryProblemInputV1,
    candidate: DiscoveryCandidateV1,
) -> DiscoveryCandidateV1:
    return replace(
        candidate,
        condition_compatibility=analyze_candidate_conditions(problem, candidate),
    )


__all__ = ["analyze_candidate_conditions", "apply_condition_compatibility"]
