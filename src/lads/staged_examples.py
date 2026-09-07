"""High-dimensional Ansatz -> parameter-hole tree -> slot-bound Atom diagnostic.

Stage I receives the complete declared public problem so it can preserve boundary masks,
coordinate interactions, and nonseparable forcing structure. Stage II remains bound to one
certified Ansatz, one slot contract, and the target-neutral PDE operator. Numeric coefficient
values are introduced only by the host optimizer after every open slot has an LLM Atom.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

import sympy as sp
from sympy.core.function import AppliedUndef

from .ansatz_replay import compile_neutral_ansatz, replay_ansatz_candidates
from .atom_tree_feedback import (
    EXPANDED_ATOM_VALIDATION_BATCH_SIZE,
    TARGET_EXPANDED_ATOM_COUNT,
    TREE_CANDIDATE_COUNT,
    AtomTreeError,
    request_bound_atoms,
    request_tree_batch,
)
from .coefficient_fitting import BFGSConfigV1, fit_expression_coefficients_bfgs
from .condition_aware_solution import _residual_terms
from .llm_ansatz import (
    LLMAnsatzBatchCompletionV1,
    LLMAnsatzCandidateV1,
    LLMAnsatzError,
    propose_llm_ansatz_batch,
    public_problem_payload,
    validate_llm_ansatz_candidate,
)
from .llm_provider import LLMBatchCompletionV1, LLMBatchProviderV1, LLMResponseError
from .residual_nmse import (
    NORMALIZATION_ENERGY_FLOOR,
    evaluate_residual_nmse,
    normalize_residual_terms,
    public_normalization_energies,
)
from .safe_symbolic import (
    SUPPORTED_ELEMENTARY_FUNCTIONS,
    SafeSymbolicError,
    parse_symbolic_expression,
)

STAGE_I_CANDIDATE_COUNT = 3
STAGE_II_CANDIDATE_COUNT = 8
MAX_ATOM_TEMPLATE_LENGTH = 65_536
MAX_ATOM_REASON_LENGTH = 4_096
TREE_STAGE_INITIAL_ATOM_TARGET = TARGET_EXPANDED_ATOM_COUNT
MAX_CERTIFIED_ANSATZ_RUNS_PER_CASE = STAGE_I_CANDIDATE_COUNT
MAX_REGENERATION_ATTEMPTS = 10
MAX_STAGE_II_REQUESTS = 1_440
MAX_FITTED_COEFFICIENTS = 96
MAX_PROFILE_ARITY = 16
MAX_ATOM_POOL_SIZE = TARGET_EXPANDED_ATOM_COUNT * 3
MAX_ATOM_COMBINATIONS = TARGET_EXPANDED_ATOM_COUNT
MAX_SEMANTIC_FITS_PER_ANSATZ = 24
MAX_MULTISLOT_SEMANTIC_FITS_PER_ANSATZ = 24
MAX_SEMANTIC_FITS_PER_CASE = 72
MAX_SEMANTIC_REGENERATION_ROUNDS = 2
# The paper's development structure+optimizer table reports public validation losses in the
# 1e-2--5e-2 range for the nontrivial PG/shallow-water cases.  V11 deliberately uses this
# development-only paper-calibrated diagnostic band instead of treating 1e-8 as a universal
# solution boundary.  It must not be presented as a holdout or generalization threshold.
SEMANTIC_ACCEPTANCE_LOSS = 5.0e-2
SEMANTIC_ACCEPTANCE_POLICY = "paper_development_calibrated_max_component_nmse_5e-2"
ATOM_TYPES = ("PRIMITIVE", "ENVELOPE", "COORDINATE", "TERM")
_PARAMETER = re.compile(r"p([1-9][0-9]*)")
_INSTANTIATED_PARAMETER = re.compile(r"z[1-9][0-9]*_([1-9][0-9]*)")
_RATIONALE = re.compile(r"[A-Z][A-Z0-9_]*")

STAGE_II_SYSTEM_PROMPT = f"""You are a mathematical Atom generator, not an Ansatz generator.

An Atom is an executable expression that replaces exactly one arbitrary profile function
inside the supplied host-certified atom-free Ansatz. The request is bound to one Ansatz,
one slot marker, the original profile arguments, and the declared public PDE problem.
Do not change the Ansatz, fill another slot, or output a complete field solution.

Use the supplied formal slot arguments ARG1, ARG2, ..., real numeric literals, pi, and
free scalar parameters p1, p2, ... . Public forcing, IC/BC, domain, and parameter values
are available as mathematical context. Reference solutions, benchmark identities, and
evaluator labels are absent. Do not force a particular topology or parameterization: the host
keeps every expression that is syntactically parseable and executable.

Format-only synthetic examples: if a fictitious certified Ansatz contains q(r)=a1*H(r),
then the slot H(r) may be represented by formal argument ARG1. Both H(r) -> p1*ARG1+p2
and H(r) -> sin(ARG1) are valid; H(r) -> p1*tanh(p2*ARG1+p3)+p4 is also valid.
Coefficients are fitted only when present.

The complete supported elementary-function allowlist is:
{", ".join(SUPPORTED_ELEMENTARY_FUNCTIONS)}.
Do not call any lowercase function outside this list. Integer powers and arithmetic products
or compositions of supported functions are allowed.

Return strict JSON only. Rank exactly eight executable Atom expressions. Shape and
parameterization are hypotheses for the optimizer, not host rejection criteria.
"""


class StagedExperimentError(RuntimeError):
    """A fail-closed staged-generation or artifact error."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _exact(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise StagedExperimentError(f"{label} fields differ")


def _bullets(values: Sequence[object]) -> str:
    items = tuple(values)
    return "\n".join(f"- {item}" for item in items) if items else "- (none)"


def atom_response_format(
    argument_roles: Sequence[str], *, permissive: bool = False
) -> dict[str, Any]:
    roles = tuple(argument_roles)
    proposal_required = (
        ["canonical_template"]
        if permissive
        else [
            "canonical_template",
            "atom_type",
            "variable_roles",
            "free_parameters",
            "rationale_codes",
            "reason",
        ]
    )
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "pde_parameterized_atom_slot_batch_v5",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["proposals"],
                "properties": {
                    "proposals": {
                        "type": "array",
                        "minItems": STAGE_II_CANDIDATE_COUNT,
                        "maxItems": STAGE_II_CANDIDATE_COUNT,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": proposal_required,
                            "properties": {
                                "canonical_template": {"type": "string"},
                                "atom_type": (
                                    {"type": "string"}
                                    if permissive
                                    else {"type": "string", "enum": list(ATOM_TYPES)}
                                ),
                                "variable_roles": (
                                    {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    }
                                    if permissive
                                    else {
                                        "type": "array",
                                        "minItems": 1,
                                        "maxItems": len(roles),
                                        "uniqueItems": True,
                                        "items": {"type": "string", "enum": list(roles)},
                                    }
                                ),
                                "free_parameters": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "rationale_codes": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "reason": {"type": "string"},
                            },
                        },
                    }
                },
            },
        },
    }


def render_atom_messages(
    row: Mapping[str, Any],
    *,
    certified_ansatz: str,
    slot_id: str,
    current_ansatz: str,
    profile_call: str,
    profile_arguments: Sequence[str],
    validation_feedback: Sequence[str] = (),
) -> tuple[dict[str, str], ...]:
    """Render one public-problem Atom request bound to one certified Ansatz slot."""

    payload = public_problem_payload(row)
    roles = tuple(f"ARG{index}" for index in range(1, len(profile_arguments) + 1))
    role_bindings = tuple(
        f"{role} -> {argument}" for role, argument in zip(roles, profile_arguments, strict=True)
    )
    feedback = tuple(validation_feedback)
    if any(not isinstance(item, str) or not item.strip() or len(item) > 4_096 for item in feedback):
        raise ValueError("Stage II host validation feedback is invalid")
    repair_block = ""
    if feedback:
        repair_block = f"""

Previous response: host validation or semantic feedback
{_bullets(feedback)}

Correct the reported host-contract or residual failure class. Generate a new batch, do not
repeat an already accepted Atom, do not change the certified Ansatz, and do not infer condition
targets, target expressions, benchmark identity, or reference solutions. Host feedback is not
a solution-expression hint.
"""
    user = f"""Generate exactly eight diverse Atom candidates for
this single open slot.

Dependent variables:
{_bullets(payload["dependent_variables"])}

Independent variables:
{_bullets(payload["independent_variables"])}

Public parameter names:
{_bullets(payload["parameters"])}

Public parameter values:
{_bullets(
    tuple(
        f"{name} = {json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"
        for name, value in payload["parameter_values"].items()
    )
)}

PDE operator:
{_bullets(payload["pde"])}

Public forcing definitions:
{_bullets(payload["forcing_definitions"])}

Initial conditions:
{_bullets(payload["initial_conditions"])}

Boundary conditions:
{_bullets(payload["boundary_conditions"])}

Domain:
{json.dumps(payload["domain"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))}

Host-certified atom-free Ansatz:
{certified_ansatz}

Current Ansatz with exactly one open slot:
{current_ansatz}

Open slot ID:
{slot_id}

Original arbitrary profile call:
{profile_call}

Formal slot arguments:
{_bullets(role_bindings)}

Slot contract:
- replace only the declared profile call;
- use the declared ARG roles (do not use problem-coordinate names directly);
- use supported elementary functions ({", ".join(SUPPORTED_ELEMENTARY_FUNCTIONS)}), arithmetic,
  powers, products, sums, and p1, p2, ...;
- coefficients are optional and are fitted only when present;
- return an Atom expression, never a field equation or full solution;
- return exactly eight structurally diverse proposals; do not optimize for compactness.
{repair_block}

Return one JSON object with exactly the key proposals. Every proposal must contain exactly
canonical_template, atom_type, variable_roles, free_parameters, rationale_codes, and
reason. Do not add prose outside JSON."""
    return (
        {"role": "system", "content": STAGE_II_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    )


@dataclass(frozen=True)
class RuntimeSlot:
    slot_id: str
    call: AppliedUndef
    argument_roles: tuple[str, ...]
    current_ansatz: str

    def public_mapping(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "profile_call": sp.sstr(self.call),
            "profile_name": str(self.call.func),
            "profile_arguments": [sp.sstr(item) for item in self.call.args],
            "argument_roles": list(self.argument_roles),
            "current_ansatz": self.current_ansatz,
        }


@dataclass(frozen=True)
class AnsatzRichnessV1:
    """Target-free structural capacity used to schedule certified Ansatz candidates."""

    full_arity_profile_count: int
    full_coordinate_profile_count: int
    covered_coordinate_count: int
    profile_call_count: int
    profile_name_count: int
    symbolic_coefficient_count: int
    operation_count: int
    source_length: int

    def priority_key(self, *, provider_rank: int) -> tuple[int, ...]:
        """Sort richer/full-dimensional structures first; provider rank breaks ties."""

        return (
            -self.full_arity_profile_count,
            -self.full_coordinate_profile_count,
            -self.covered_coordinate_count,
            -self.profile_call_count,
            -self.profile_name_count,
            -self.symbolic_coefficient_count,
            -self.operation_count,
            -self.source_length,
            int(provider_rank),
        )

    def to_mapping(self) -> dict[str, int]:
        return {
            "full_arity_profile_count": self.full_arity_profile_count,
            "full_coordinate_profile_count": self.full_coordinate_profile_count,
            "covered_coordinate_count": self.covered_coordinate_count,
            "profile_call_count": self.profile_call_count,
            "profile_name_count": self.profile_name_count,
            "symbolic_coefficient_count": self.symbolic_coefficient_count,
            "operation_count": self.operation_count,
            "source_length": self.source_length,
        }


def measure_ansatz_richness(
    row: Mapping[str, Any], ansatz: str, reason: str = "structural richness measurement"
) -> AnsatzRichnessV1:
    """Measure structure only; never inspect scores, gold, or validation targets."""

    ast = compile_neutral_ansatz(row, {"ansatz": ansatz, "reason": reason})
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
    coverage_by_call = tuple(
        set().union(*(argument.free_symbols & coordinates for argument in call.args))
        for call in calls
    )
    covered = set().union(*coverage_by_call) if coverage_by_call else set()
    full_coordinate_count = sum(coverage == coordinates for coverage in coverage_by_call)
    full_arity_count = sum(
        coverage == coordinates and len(call.args) >= len(coordinates)
        for call, coverage in zip(calls, coverage_by_call, strict=True)
    )
    return AnsatzRichnessV1(
        full_arity_profile_count=full_arity_count,
        full_coordinate_profile_count=full_coordinate_count,
        covered_coordinate_count=len(covered),
        profile_call_count=len(calls),
        profile_name_count=len(ast.profile_names),
        symbolic_coefficient_count=len(ast.symbolic_coefficients),
        operation_count=sum(
            sp.count_ops(expression) for _field, expression in ast.field_expressions
        ),
        source_length=len(re.sub(r"\s+", "", ast.source)),
    )


def rank_certified_ansatz_candidates(
    row: Mapping[str, Any],
    certified: Sequence[tuple[LLMAnsatzCandidateV1, Any]],
) -> tuple[tuple[LLMAnsatzCandidateV1, Any, AnsatzRichnessV1], ...]:
    """Retain every certified candidate and schedule richer structures first."""

    ranked = [
        (
            candidate,
            certificate,
            measure_ansatz_richness(row, candidate.template, candidate.reason),
        )
        for candidate, certificate in certified
    ]
    ranked.sort(key=lambda item: item[2].priority_key(provider_rank=item[0].rank))
    return tuple(ranked)


class RecordingProvider:
    """Capture credential-free completion metadata and parsed response."""

    def __init__(self, wrapped: LLMBatchProviderV1) -> None:
        self.wrapped = wrapped
        self.completion: LLMAnsatzBatchCompletionV1 | None = None

    def complete(
        self,
        *,
        tokens: Sequence[str],
        messages: Sequence[Mapping[str, str]],
        response_format: Mapping[str, Any],
    ) -> LLMAnsatzBatchCompletionV1:
        raw = self.wrapped.complete(
            tokens=tokens,
            messages=messages,
            response_format=response_format,
        )
        self.completion = LLMAnsatzBatchCompletionV1(
            raw.response,
            raw.provider_name,
            raw.requested_model,
            raw.returned_model,
            raw.provider_response_id,
            raw.prompt_tokens,
            raw.completion_tokens,
            raw.request_payload_sha256,
            raw.raw_response_sha256,
            raw.response_mode,
        )
        return self.completion


class GenericRecordingProvider:
    """Capture generic Stage-II tree/binding completion evidence without recasting it."""

    def __init__(self, wrapped: LLMBatchProviderV1) -> None:
        self.wrapped = wrapped
        self.completion: LLMBatchCompletionV1 | None = None

    def complete(
        self,
        *,
        tokens: Sequence[str],
        messages: Sequence[Mapping[str, str]],
        response_format: Mapping[str, Any],
    ) -> LLMBatchCompletionV1:
        self.completion = self.wrapped.complete(
            tokens=tokens,
            messages=messages,
            response_format=response_format,
        )
        return self.completion


def _provider_evidence(
    completion: LLMBatchCompletionV1 | LLMAnsatzBatchCompletionV1,
) -> dict[str, Any]:
    return {
        "provider_name": completion.provider_name,
        "requested_model": completion.requested_model,
        "returned_model": completion.returned_model,
        "provider_response_id": completion.provider_response_id,
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
        "request_payload_sha256": completion.request_payload_sha256,
        "raw_response_sha256": completion.raw_response_sha256,
        "response_mode": completion.response_mode,
    }


def _render_ansatz_with_slot(ast: Any, call: AppliedUndef, slot_id: str) -> str:
    marker = sp.Symbol(f"OPEN_{slot_id.upper()}")
    equations = []
    for field, expression in ast.field_expressions:
        equations.append(f"{field}={sp.sstr(expression.xreplace({call: marker}))}")
    rendered = ";".join(equations)
    if rendered.count(str(marker)) < 1:
        raise StagedExperimentError("slot marker was not inserted")
    return rendered


def extract_runtime_slots(
    row: Mapping[str, Any], ansatz: str, reason: str
) -> tuple[Any, tuple[RuntimeSlot, ...]]:
    ast = compile_neutral_ansatz(row, {"ansatz": ansatz, "reason": reason})
    calls = sorted(
        {
            call
            for _field, expression in ast.field_expressions
            for call in expression.atoms(AppliedUndef)
        },
        key=sp.default_sort_key,
    )
    if not calls:
        raise StagedExperimentError("certified Ansatz has no open profile slot")
    slots = []
    for index, call in enumerate(calls, 1):
        if not call.args or len(call.args) > MAX_PROFILE_ARITY:
            raise StagedExperimentError("profile arity is outside the Atom slot contract")
        if any(argument.atoms(AppliedUndef) for argument in call.args):
            raise StagedExperimentError("nested profile arguments are unsupported")
        slot_id = f"slot_{index:03d}"
        roles = tuple(f"ARG{position}" for position in range(1, len(call.args) + 1))
        slots.append(
            RuntimeSlot(slot_id, call, roles, _render_ansatz_with_slot(ast, call, slot_id))
        )
    return ast, tuple(slots)


def _parameter_names(expression: sp.Expr) -> tuple[str, ...]:
    names = sorted(
        {str(symbol) for symbol in expression.free_symbols if _PARAMETER.fullmatch(str(symbol))},
        key=lambda name: int(_PARAMETER.fullmatch(name).group(1)),  # type: ignore[union-attr]
    )
    expected = tuple(f"p{index}" for index in range(1, len(names) + 1))
    if tuple(names) != expected:
        raise StagedExperimentError("Atom parameters are not consecutive p1..pN")
    return tuple(names)


def _parameterization_certificate(
    expression: sp.Expr,
    role_symbols: Sequence[sp.Symbol],
    parameter_names: Sequence[str],
) -> dict[str, Any]:
    """Require optimizable amplitude and nonlinear argument-scale coefficients."""

    parameters = tuple(sp.Symbol(name) for name in parameter_names)
    roles = tuple(role_symbols)
    if not parameters:
        raise StagedExperimentError(
            "Atom must contain at least one free coefficient; fixed-shape Atoms are rejected"
        )

    amplitude_parameters: list[str] = []
    for parameter in parameters:
        slope = sp.diff(expression, parameter)
        if not any(slope.has(role) for role in roles):
            continue
        if sp.diff(expression, parameter, 2) == 0:
            amplitude_parameters.append(str(parameter))
    if not amplitude_parameters:
        raise StagedExperimentError(
            "role-dependent Atom requires a free outer amplitude coefficient"
        )

    nonlinear_scale_parameters: set[str] = set()
    for function in expression.atoms(sp.Function):
        if isinstance(function, AppliedUndef) or not any(function.has(role) for role in roles):
            continue
        function_amplitudes = {
            str(parameter)
            for parameter in parameters
            if sp.diff(expression, parameter).has(function)
            and sp.diff(expression, parameter, 2) == 0
        }
        if not function_amplitudes:
            raise StagedExperimentError(
                f"nonlinear function {function.func} requires its own free outer amplitude "
                "coefficient"
            )
        for role in roles:
            if not function.has(role):
                continue
            derivatives = tuple(sp.diff(argument, role) for argument in function.args)
            scale_parameters = {
                str(parameter)
                for parameter in parameters
                if any(derivative.has(parameter) for derivative in derivatives)
            }
            if not scale_parameters:
                raise StagedExperimentError(
                    f"nonlinear function {function.func} requires a free argument scale "
                    f"coefficient for {role}"
                )
            nonlinear_scale_parameters.update(scale_parameters)

    return {
        "amplitude_parameters": amplitude_parameters,
        "nonlinear_argument_scale_parameters": sorted(nonlinear_scale_parameters),
    }


def validate_atom_response(
    response: Mapping[str, Any],
    slot: RuntimeSlot,
    *,
    expected_count: int = STAGE_II_CANDIDATE_COUNT,
    accepted_limit: int | None = None,
    permissive: bool = False,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Return executable proposals and explicit rejected proposal records.

    permissive=True is used only by the isolated direct Atom lane. It keeps the
    transport/schema and safe-symbolic execution boundary, but skips parameterization,
    role-census, duplicate-shape, and rationale-quality gates.
    """

    _exact(response, {"proposals"}, "Atom response")
    proposals = response["proposals"]
    if (
        not isinstance(proposals, Sequence)
        or isinstance(proposals, (str, bytes, bytearray, Mapping))
        or len(proposals) != expected_count
    ):
        raise StagedExperimentError(
            f"Atom response must contain exactly {expected_count} proposals"
        )
    if accepted_limit is not None and accepted_limit < 1:
        raise ValueError("accepted_limit must be positive")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, raw in enumerate(proposals, 1):
        try:
            if not isinstance(raw, Mapping):
                raise StagedExperimentError("Atom proposal is not an object")
            if permissive:
                if not isinstance(raw, Mapping):
                    raise StagedExperimentError("Atom proposal is not an object")
                if "canonical_template" not in raw:
                    raise StagedExperimentError("Atom proposal lacks canonical_template")
            else:
                _exact(
                    raw,
                    {
                        "canonical_template",
                        "atom_type",
                        "variable_roles",
                        "free_parameters",
                        "rationale_codes",
                        "reason",
                    },
                    "Atom proposal",
                )
            template = raw["canonical_template"]
            reason = raw.get("reason", "LLM direct Atom; host syntax check only")
            atom_type = raw.get("atom_type", "PRIMITIVE")
            roles = raw.get("variable_roles", ())
            parameters = raw.get("free_parameters", ())
            rationale_codes = raw.get("rationale_codes", ["SYNTAX_ONLY"])
            if (
                not isinstance(template, str)
                or not template.strip()
                or len(template) > MAX_ATOM_TEMPLATE_LENGTH
            ):
                raise StagedExperimentError("Atom template text is invalid")
            if (
                not isinstance(reason, str)
                or not reason.strip()
                or len(reason) > MAX_ATOM_REASON_LENGTH
            ):
                raise StagedExperimentError("Atom reason is invalid")
            # The isolated direct lane deliberately treats Atom metadata as
            # advisory.  Providers commonly return labels such as
            # ``profile_function`` or ``profile_atom`` even though the
            # canonical template is executable.  Keep the historical enum
            # gate for the strict tree/binding lane, but do not discard a
            # syntactically valid expression in permissive mode.
            if not isinstance(atom_type, str):
                if not permissive:
                    raise StagedExperimentError("Atom type is invalid")
                atom_type = "PRIMITIVE"
            elif not permissive and atom_type not in ATOM_TYPES:
                raise StagedExperimentError("Atom type is invalid")
            if not permissive and (
                not isinstance(roles, Sequence)
                or isinstance(roles, (str, bytes, Mapping))
                or not roles
                or len(roles) != len(set(roles))
                or not set(roles).issubset(slot.argument_roles)
            ):
                raise StagedExperimentError("Atom variable roles violate the slot contract")
            if not permissive and (
                not isinstance(parameters, Sequence)
                or isinstance(parameters, (str, bytes, Mapping))
                or len(parameters) != len(set(parameters))
                or any(not isinstance(item, str) for item in parameters)
            ):
                raise StagedExperimentError("Atom free parameter declaration is invalid")
            if not permissive and (
                not isinstance(rationale_codes, Sequence)
                or isinstance(rationale_codes, (str, bytes, Mapping))
                or not rationale_codes
                or any(
                    not isinstance(item, str) or _RATIONALE.fullmatch(item) is None
                    for item in rationale_codes
                )
            ):
                raise StagedExperimentError("Atom rationale codes are invalid")
            names = {name: sp.Symbol(name, real=True) for name in slot.argument_roles}
            names["pi"] = sp.pi
            expression = parse_symbolic_expression(template, names=names)
            if expression.atoms(AppliedUndef):
                raise StagedExperimentError("Atom contains an unknown profile function")
            parsed_parameters = _parameter_names(expression)
            if not permissive and tuple(parameters) != parsed_parameters:
                raise StagedExperimentError("Atom free parameters differ from the expression")
            used_roles = tuple(
                role for role in slot.argument_roles if names[role] in expression.free_symbols
            )
            if not permissive and tuple(roles) != used_roles:
                raise StagedExperimentError("Atom variable_roles differ from the expression")
            allowed = {names[role] for role in used_roles} | {
                sp.Symbol(name) for name in parsed_parameters
            }
            unresolved = expression.free_symbols - allowed
            if unresolved:
                raise StagedExperimentError("Atom contains symbols outside the slot contract")
            parameterization = (
                {
                    "amplitude_parameters": [],
                    "nonlinear_argument_scale_parameters": [],
                }
                if permissive
                else _parameterization_certificate(
                    expression,
                    tuple(names[role] for role in used_roles),
                    parsed_parameters,
                )
            )
            canonical = sp.sstr(expression)
            parse_symbolic_expression(canonical, names=names)
            if canonical in seen and not permissive:
                raise StagedExperimentError("Atom duplicates an earlier template")
            seen.add(canonical)
            accepted.append(
                {
                    "rank": rank,
                    "canonical_template": canonical,
                    "atom_type": atom_type,
                    "variable_roles": list(used_roles),
                    "free_parameters": list(parsed_parameters),
                    "rationale_codes": (
                        [str(item) for item in rationale_codes]
                        if isinstance(rationale_codes, Sequence)
                        and not isinstance(rationale_codes, (str, bytes, bytearray))
                        else ["SYNTAX_ONLY"]
                    ),
                    "reason": reason.strip(),
                    "host_validation_passed": True,
                    "parameterization_validation_passed": not permissive,
                    "validation_policy": (
                        "syntax_only_executable_expression"
                        if permissive
                        else "strict_parameterized_atom"
                    ),
                    "parameterization_certificate": parameterization,
                    "complexity": {
                        "source_length": len(template),
                        "canonical_length": len(canonical),
                        "operation_count": int(sp.count_ops(expression)),
                        "parameter_count": len(parsed_parameters),
                        "role_count": len(used_roles),
                        "function_count": len(expression.atoms(sp.Function)),
                    },
                }
            )
            if accepted_limit is not None and len(accepted) >= accepted_limit:
                break
        except (SafeSymbolicError, StagedExperimentError, TypeError, ValueError) as exc:
            rejected.append(
                {
                    "rank": rank,
                    "raw_proposal": raw,
                    "host_validation_passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return tuple(accepted), tuple(rejected)


def _instantiate_atom(slot: RuntimeSlot, candidate: Mapping[str, Any], slot_index: int) -> sp.Expr:
    roles = {name: sp.Symbol(name, real=True) for name in slot.argument_roles}
    roles["pi"] = sp.pi
    expression = parse_symbolic_expression(str(candidate["canonical_template"]), names=roles)
    parameter_substitutions = {
        sp.Symbol(name): sp.Symbol(f"z{slot_index}_{position}", real=True)
        for position, name in enumerate(candidate["free_parameters"], 1)
    }
    argument_substitutions = {
        roles[f"ARG{position}"]: argument for position, argument in enumerate(slot.call.args, 1)
    }
    return expression.subs(parameter_substitutions).subs(argument_substitutions)


def assemble_candidate_beam(
    ast: Any,
    slots: Sequence[RuntimeSlot],
    pools: Sequence[Sequence[Mapping[str, Any]]],
    *,
    ansatz_rank: int,
    max_combinations: int = MAX_ATOM_COMBINATIONS,
) -> tuple[dict[str, Any], ...]:
    """Build a bounded Cartesian beam; ranks from different slots are not aligned."""

    if len(slots) != len(pools) or any(not pool for pool in pools):
        return ()
    if isinstance(max_combinations, bool) or not isinstance(max_combinations, int):
        raise ValueError("max_combinations must be an integer")
    if max_combinations < 1:
        raise ValueError("max_combinations must be positive")

    ordered_pools = tuple(
        tuple(
            sorted(
                pool,
                key=lambda atom: (int(atom["rank"]), str(atom["canonical_template"])),
            )
        )
        for pool in pools
    )

    def index_key(indices: tuple[int, ...]) -> tuple[Any, ...]:
        combination = tuple(
            ordered_pools[slot_index][atom_index] for slot_index, atom_index in enumerate(indices)
        )
        ranks = tuple(int(atom["rank"]) for atom in combination)
        return (
            sum(ranks),
            ranks,
            tuple(str(atom["canonical_template"]) for atom in combination),
        )

    zero = tuple(0 for _ in ordered_pools)
    chosen_indices: set[tuple[int, ...]] = {zero}
    combination_indices: list[tuple[int, ...]] = [zero]
    anchors: list[tuple[int, ...]] = []
    for slot_index, pool in enumerate(ordered_pools):
        for atom_index in range(1, len(pool)):
            anchor_indices = list(zero)
            anchor_indices[slot_index] = atom_index
            anchors.append(tuple(anchor_indices))

    def anchor_priority(indices: tuple[int, ...]) -> tuple[Any, ...]:
        slot_index = next(index for index, value in enumerate(indices) if value != 0)
        atom = ordered_pools[slot_index][indices[slot_index]]
        rank = int(atom["rank"])
        regenerated = rank > TREE_STAGE_INITIAL_ATOM_TARGET
        return (
            0 if regenerated else 1,
            -rank if regenerated else rank,
            slot_index,
            str(atom["canonical_template"]),
        )

    anchors.sort(key=anchor_priority)
    for anchor in anchors:
        if len(combination_indices) >= max_combinations:
            break
        chosen_indices.add(anchor)
        combination_indices.append(anchor)

    heap: list[tuple[tuple[Any, ...], tuple[int, ...]]] = [(index_key(zero), zero)]
    visited = {zero}
    while heap and len(combination_indices) < max_combinations:
        _key, current_indices = heapq.heappop(heap)
        if current_indices not in chosen_indices:
            chosen_indices.add(current_indices)
            combination_indices.append(current_indices)
        for slot_index, atom_index in enumerate(current_indices):
            next_index = atom_index + 1
            if next_index >= len(ordered_pools[slot_index]):
                continue
            neighbor = list(current_indices)
            neighbor[slot_index] = next_index
            neighbor_tuple = tuple(neighbor)
            if neighbor_tuple in visited:
                continue
            visited.add(neighbor_tuple)
            heapq.heappush(heap, (index_key(neighbor_tuple), neighbor_tuple))

    outputs = []
    for combination_rank, selected_indices in enumerate(combination_indices, 1):
        combination = tuple(
            ordered_pools[slot_index][atom_index]
            for slot_index, atom_index in enumerate(selected_indices)
        )
        replacements = {
            slot.call: _instantiate_atom(slot, combination[index], index + 1)
            for index, slot in enumerate(slots)
        }
        fields = {
            field: expression.xreplace(replacements) for field, expression in ast.field_expressions
        }
        if any(expression.atoms(AppliedUndef) for expression in fields.values()):
            raise StagedExperimentError("assembled candidate retains an open profile")
        outputs.append(
            {
                "ansatz_rank": ansatz_rank,
                "atom_combination_rank": combination_rank,
                "field_expressions": {name: sp.sstr(expr) for name, expr in fields.items()},
                "slot_atom_ranks": {
                    slot.slot_id: int(combination[index]["rank"])
                    for index, slot in enumerate(slots)
                },
                "slot_atom_templates": {
                    slot.slot_id: str(combination[index]["canonical_template"])
                    for index, slot in enumerate(slots)
                },
                "assembly_policy": "cartesian rank-sum beam",
            }
        )
    return tuple(outputs)


def assemble_aligned_candidates(
    ast: Any,
    slots: Sequence[RuntimeSlot],
    pools: Sequence[Sequence[Mapping[str, Any]]],
    *,
    ansatz_rank: int,
) -> tuple[dict[str, Any], ...]:
    """Compatibility wrapper for callers migrating from V4."""

    return assemble_candidate_beam(ast, slots, pools, ansatz_rank=ansatz_rank)


def select_semantic_candidate_beam(
    candidates: Sequence[Mapping[str, Any]],
    *,
    max_candidates: int,
) -> tuple[Mapping[str, Any], ...]:
    """Keep low-rank combinations plus pairwise neighborhoods of regenerated Atoms."""

    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    ranked = sorted(
        candidates,
        key=lambda item: (
            int(item["atom_combination_rank"]),
            int(item["ansatz_rank"]),
        ),
    )
    selected: list[Mapping[str, Any]] = []
    selected_ids: set[int] = set()

    def retain(item: Mapping[str, Any]) -> None:
        identity = id(item)
        if identity not in selected_ids and len(selected) < max_candidates:
            selected.append(item)
            selected_ids.add(identity)

    for item in ranked[: min(4, max_candidates)]:
        retain(item)

    features: dict[tuple[int, str, int], list[Mapping[str, Any]]] = {}
    for item in ranked:
        for slot_id, atom_rank in item["slot_atom_ranks"].items():
            feature = (int(item["ansatz_rank"]), str(slot_id), int(atom_rank))
            features.setdefault(feature, []).append(item)

    by_slot: dict[tuple[int, str], list[int]] = {}
    for ansatz_rank, slot_id, atom_rank in features:
        by_slot.setdefault((ansatz_rank, slot_id), []).append(atom_rank)
    prioritized_by_slot: dict[tuple[int, str], tuple[int, ...]] = {}
    for slot, atom_ranks in by_slot.items():
        unique = sorted(set(atom_ranks))
        regenerated = [rank for rank in unique if rank > STAGE_II_CANDIDATE_COUNT]
        initial = [rank for rank in unique if rank <= STAGE_II_CANDIDATE_COUNT]
        priority = regenerated[:1] + list(reversed(regenerated[1:])) + list(reversed(initial))
        prioritized_by_slot[slot] = tuple(dict.fromkeys(priority))

    depth = 0
    while len(selected) < max_candidates:
        retained_before = len(selected)
        for (ansatz_rank, slot_id), prioritized_ranks in sorted(prioritized_by_slot.items()):
            if depth >= len(prioritized_ranks):
                continue
            feature = (ansatz_rank, slot_id, prioritized_ranks[depth])
            for item in features[feature][:2]:
                retain(item)
        if len(selected) == retained_before:
            break
        depth += 1
    for item in ranked:
        retain(item)
    return tuple(selected)


def screen_semantic_candidate_beam(
    row: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    validation_spec: Mapping[str, Any],
    *,
    max_candidates: int,
) -> tuple[Mapping[str, Any], ...]:
    """Evaluate every assembled structure once, then send only the best NMSE rows to BFGS."""

    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    training_energies = public_normalization_energies(validation_spec, split="training")
    screened: list[dict[str, Any]] = []
    for screening_input_rank, candidate in enumerate(candidates, 1):
        record = {**candidate, "screening_input_rank": screening_input_rank}
        try:
            fields = candidate["field_expressions"]
            if not isinstance(fields, Mapping) or set(fields) != set(row["dependent_variables"]):
                raise StagedExperimentError("assembled scalar field census differs")
            field = str(row["dependent_variables"][0])
            expression, coefficients = _real_expression(str(fields[field]), row)
            if len(coefficients) > MAX_FITTED_COEFFICIENTS:
                raise StagedExperimentError("candidate exceeds the frozen coefficient budget")
            initial = coefficient_initial_guesses(expression, coefficients)[0]
            initialized_expression = expression.subs(dict(zip(coefficients, initial, strict=True)))
            training = _residual_terms(
                initialized_expression, (), validation_spec, split="training"
            )
            metrics = evaluate_residual_nmse(training, initialized_expression, training_energies)
            total = float(metrics["total_weighted_nmse"])
            maximum = float(metrics["max_component_nmse"])
            if not math.isfinite(total) or not math.isfinite(maximum):
                raise ValueError("screening residual NMSE is nonfinite")
            record.update(
                screening_initial_coefficients={
                    str(symbol): float(value)
                    for symbol, value in zip(coefficients, initial, strict=True)
                },
                screening_total_weighted_nmse=total,
                screening_max_component_nmse=maximum,
                screening_residual_nmse=metrics,
            )
        except (
            KeyError,
            NotImplementedError,
            OverflowError,
            SafeSymbolicError,
            StagedExperimentError,
            TypeError,
            ValueError,
            ZeroDivisionError,
        ) as exc:
            record["screening_error"] = f"{type(exc).__name__}: {exc}"
        screened.append(record)

    valid = [item for item in screened if "screening_max_component_nmse" in item]
    valid.sort(
        key=lambda item: (
            float(item["screening_max_component_nmse"]),
            float(item["screening_total_weighted_nmse"]),
            int(item["screening_input_rank"]),
        )
    )
    for screening_rank, item in enumerate(valid, 1):
        item["screening_rank"] = screening_rank
        item["screening_population_count"] = len(screened)
        item["screening_valid_count"] = len(valid)
    return tuple(valid[:max_candidates])


def _real_expression(source: str, row: Mapping[str, Any]) -> tuple[sp.Expr, tuple[sp.Symbol, ...]]:
    coordinates = {name: sp.Symbol(str(name), real=True) for name in row["independent_variables"]}
    names: dict[str, object] = {**coordinates, "pi": sp.pi}
    expression = parse_symbolic_expression(source, names=names)
    numeric_parameters = {
        str(name): float(value)
        for name, value in row.get("parameters", {}).items()
        if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    }
    expression = expression.subs(
        {
            symbol: numeric_parameters[str(symbol)]
            for symbol in expression.free_symbols
            if str(symbol) in numeric_parameters
        }
    )
    coordinate_symbols = set(coordinates.values())
    expression = expression.xreplace(
        {
            symbol: sp.Symbol(str(symbol), real=True)
            for symbol in expression.free_symbols - coordinate_symbols
        }
    )
    coefficients = tuple(
        sorted(expression.free_symbols - coordinate_symbols, key=lambda item: str(item))
    )
    return expression, coefficients


def coefficient_initial_guesses(
    expression: sp.Expr,
    coefficients: Sequence[sp.Symbol],
) -> tuple[tuple[float, ...], ...]:
    """Give Atom coefficients deterministic role-aware, mixed-sign starting points."""

    coefficient_tuple = tuple(coefficients)
    coefficient_set = set(coefficient_tuple)
    coordinates = expression.free_symbols - coefficient_set
    scale_coefficients: set[sp.Symbol] = set()
    for function in expression.atoms(sp.Function):
        if isinstance(function, AppliedUndef):
            continue
        for argument in function.args:
            for coordinate in coordinates:
                coordinate_slope = sp.diff(argument, coordinate)
                scale_coefficients.update(
                    coefficient
                    for coefficient in coefficient_tuple
                    if coordinate_slope.has(coefficient)
                )

    def vector(scale_values: Mapping[sp.Symbol, float]) -> tuple[float, ...]:
        values = []
        for coefficient in coefficient_tuple:
            if coefficient in scale_values:
                values.append(scale_values[coefficient])
                continue
            match = _INSTANTIATED_PARAMETER.fullmatch(str(coefficient))
            values.append(1.0 if match is None or match.group(1) == "1" else 0.0)
        return tuple(values)

    base = vector({coefficient: 1.0 for coefficient in scale_coefficients})
    if not scale_coefficients:
        return (base,)
    ordered_scales = tuple(
        coefficient for coefficient in coefficient_tuple if coefficient in scale_coefficients
    )
    if len(ordered_scales) == 1:
        coefficient = ordered_scales[0]
        return tuple(vector({coefficient: scale}) for scale in (1.0, -1.0, 4.0, -4.0, 16.0, -16.0))

    guesses = [base]
    guesses.extend(
        vector({candidate: (-1.0 if candidate == flipped else 1.0) for candidate in ordered_scales})
        for flipped in ordered_scales
    )
    for scale in (-1.0, 4.0, 16.0, -4.0, -16.0):
        guesses.append(vector({coefficient: scale for coefficient in ordered_scales}))
    return tuple(guesses)


def optimizer_config_for_slot_count(slot_count: int) -> BFGSConfigV1:
    """Scale the optimizer budget using the paper-calibrated diagnostic threshold.

    ``acceptance_loss`` is a reporting/selection flag in this development lane, not
    a machine-precision proof obligation.  The old ``1e-8`` gate marked useful
    paper-level fits as rejected even when their component NMSE passed the semantic
    diagnostic tolerance, so keep the two decisions aligned.
    """

    if isinstance(slot_count, bool) or not isinstance(slot_count, int) or slot_count < 1:
        raise ValueError("slot_count must be a positive integer")
    return BFGSConfigV1(
        max_iterations=min(900, 300 + 75 * (slot_count - 1)),
        gradient_tolerance=1.0e-8,
        restarts=min(10, 6 + (slot_count - 1) // 2),
        random_seed=20260904,
        restart_scale=1.0,
        acceptance_loss=SEMANTIC_ACCEPTANCE_LOSS,
    )


def fit_scalar_candidates(
    row: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    validation_spec: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], Mapping[str, Any] | None]:
    fitted = []
    for generation_rank, candidate in enumerate(candidates, 1):
        record = {**candidate, "generation_rank": generation_rank}
        try:
            fields = candidate["field_expressions"]
            if not isinstance(fields, Mapping) or set(fields) != set(row["dependent_variables"]):
                raise StagedExperimentError("assembled scalar field census differs")
            field = str(row["dependent_variables"][0])
            expression, coefficients = _real_expression(str(fields[field]), row)
            if len(coefficients) > MAX_FITTED_COEFFICIENTS:
                raise StagedExperimentError("candidate exceeds the frozen coefficient budget")
            slot_ranks = candidate.get("slot_atom_ranks", {})
            slot_count = (
                len(slot_ranks) if isinstance(slot_ranks, Mapping) and len(slot_ranks) > 0 else 1
            )
            config = optimizer_config_for_slot_count(slot_count)
            training_energies = public_normalization_energies(validation_spec, split="training")
            validation_energies = public_normalization_energies(validation_spec, split="validation")
            training = _residual_terms(expression, coefficients, validation_spec, split="training")
            normalized_training = normalize_residual_terms(training, training_energies)
            if coefficients:
                fit = fit_expression_coefficients_bfgs(
                    expression,
                    coefficients,
                    normalized_training,
                    config=config,
                    initial_guesses=coefficient_initial_guesses(expression, coefficients),
                )
                fitted_expression = fit.expression
                fit_mapping: Mapping[str, Any] = fit.to_mapping()
            else:
                training_metrics = evaluate_residual_nmse(training, expression, training_energies)
                loss = float(training_metrics["total_weighted_nmse"])
                fitted_expression = expression
                fit_mapping = {
                    "schema_version": "NoCoefficientFitV1",
                    "accepted": loss <= config.acceptance_loss,
                    "optimizer_converged": True,
                    "expression": sp.sstr(expression),
                    "coefficient_assignments": {},
                    "final_loss": loss,
                    "residual_nmse": training_metrics,
                    "restart_count": 0,
                }
            validation = _residual_terms(fitted_expression, (), validation_spec, split="validation")
            validation_metrics = evaluate_residual_nmse(
                validation, fitted_expression, validation_energies
            )
            validation_loss = float(validation_metrics["total_weighted_nmse"])
            record.update(
                {
                    "fit": dict(fit_mapping),
                    "fitted_expression": sp.sstr(fitted_expression),
                    "training_normalization_energies": training_energies,
                    "validation_normalization_energies": validation_energies,
                    "validation_loss": validation_loss,
                    "validation_max_component_nmse": float(
                        validation_metrics["max_component_nmse"]
                    ),
                    "validation_residual_nmse": validation_metrics,
                }
            )
        except (
            KeyError,
            NotImplementedError,
            OverflowError,
            SafeSymbolicError,
            StagedExperimentError,
            TypeError,
            ValueError,
            ZeroDivisionError,
        ) as exc:
            record["fit_error"] = f"{type(exc).__name__}: {exc}"
        fitted.append(record)
    valid = [record for record in fitted if "validation_loss" in record]
    valid.sort(
        key=lambda item: (
            float(item["validation_max_component_nmse"]),
            float(item["validation_loss"]),
            int(item["generation_rank"]),
        )
    )
    return tuple(fitted), (valid[0] if valid else None)


def _semantic_diagnostics(
    selected: Mapping[str, Any] | None,
    *,
    tolerance: float,
) -> tuple[str, ...]:
    """Return target-free host diagnostics suitable for a Stage-II retry."""

    if selected is None:
        return (
            "NO_EXECUTABLE_SEMANTIC_CANDIDATE: every assembled Atom combination failed "
            "symbolic compilation or numerical validation",
        )
    total_loss = float(selected["validation_loss"])
    max_component_nmse = float(selected.get("validation_max_component_nmse", total_loss))
    diagnostics = [
        f"SEMANTIC_VALIDATION_FAILED: best total weighted residual NMSE {total_loss:.6g}; "
        f"worst component NMSE {max_component_nmse:.6g} "
        f"exceeds {tolerance:.6g}"
    ]
    metrics = selected.get("validation_residual_nmse", {})
    components = metrics.get("components", {}) if isinstance(metrics, Mapping) else {}
    if isinstance(components, Mapping):
        pde = components.get("pde", {})
        pde_nmse = pde.get("nmse") if isinstance(pde, Mapping) else None
        if isinstance(pde_nmse, (int, float)) and float(pde_nmse) > tolerance:
            diagnostics.append(
                f"PDE_RESIDUAL_NMSE_HIGH: operator residual NMSE {float(pde_nmse):.6g}; "
                "propose a different tree topology, slot binding, or scaling"
            )
        condition_nmse = [
            (str(name), float(value["nmse"]))
            for name, value in components.items()
            if str(name).startswith("condition_")
            and isinstance(value, Mapping)
            and isinstance(value.get("nmse"), (int, float))
        ]
        for name, nmse in condition_nmse:
            if nmse <= tolerance:
                continue
            diagnostics.append(
                f"PUBLIC_CONDITION_RESIDUAL_NMSE_HIGH: {name} NMSE {nmse:.6g}; its target "
                "expression is intentionally not disclosed in feedback"
            )
    diagnostics.append(
        "Generate structurally new, fully parameterized Atoms; do not repeat any retained template"
    )
    return tuple(diagnostics)


def semantic_validate_candidates(
    row: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    validation_spec: Mapping[str, Any],
    *,
    tolerance: float = SEMANTIC_ACCEPTANCE_LOSS,
) -> tuple[tuple[dict[str, Any], ...], Mapping[str, Any] | None, tuple[str, ...]]:
    """Fit assembled candidates and certify both PDE and public-condition residuals."""

    fitted, _selected = fit_scalar_candidates(row, candidates, validation_spec)
    annotated: list[dict[str, Any]] = []
    for candidate in fitted:
        record = dict(candidate)
        loss = record.get("validation_max_component_nmse")
        passed = (
            isinstance(loss, (int, float))
            and math.isfinite(float(loss))
            and float(loss) <= tolerance
        )
        record["semantic_validation_passed"] = passed
        record["semantic_validation_tolerance"] = tolerance
        annotated.append(record)
    valid = [record for record in annotated if "validation_loss" in record]
    valid.sort(
        key=lambda item: (
            float(item["validation_max_component_nmse"]),
            float(item["validation_loss"]),
            int(item["generation_rank"]),
        )
    )
    selected: Mapping[str, Any] | None = valid[0] if valid else None
    diagnostics = (
        ()
        if selected is not None and bool(selected["semantic_validation_passed"])
        else _semantic_diagnostics(selected, tolerance=tolerance)
    )
    return tuple(annotated), selected, diagnostics


def _semantic_evaluation_key(
    evaluation: tuple[tuple[dict[str, Any], ...], Mapping[str, Any] | None, tuple[str, ...]],
) -> tuple[float, float, int]:
    selected = evaluation[1]
    if selected is None:
        return (math.inf, math.inf, 2**31 - 1)
    return (
        float(selected.get("validation_max_component_nmse", selected["validation_loss"])),
        float(selected["validation_loss"]),
        int(selected.get("generation_rank", 2**31 - 1)),
    )


def _retain_better_semantic_evaluation(
    best: tuple[tuple[dict[str, Any], ...], Mapping[str, Any] | None, tuple[str, ...]] | None,
    candidate: tuple[tuple[dict[str, Any], ...], Mapping[str, Any] | None, tuple[str, ...]],
) -> tuple[tuple[dict[str, Any], ...], Mapping[str, Any] | None, tuple[str, ...]]:
    if best is None or _semantic_evaluation_key(candidate) < _semantic_evaluation_key(best):
        return candidate
    return best


def assess_atom_pool_semantics(
    fitted_candidates: Sequence[Mapping[str, Any]],
    slots: Sequence[RuntimeSlot],
    pools: Sequence[Sequence[Mapping[str, Any]]],
    *,
    tolerance: float = SEMANTIC_ACCEPTANCE_LOSS,
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Attribute each Atom's best assembled validation result without overclaiming causality."""

    if len(slots) != len(pools):
        raise ValueError("slot and Atom-pool counts differ")
    assessments: dict[str, tuple[dict[str, Any], ...]] = {}
    for slot, pool in zip(slots, pools, strict=True):
        rows: list[dict[str, Any]] = []
        for atom in pool:
            atom_rank = int(atom["rank"])
            containing = [
                candidate
                for candidate in fitted_candidates
                if isinstance(candidate.get("slot_atom_ranks"), Mapping)
                and int(candidate["slot_atom_ranks"].get(slot.slot_id, -1)) == atom_rank
                and isinstance(candidate.get("validation_max_component_nmse"), (int, float))
                and math.isfinite(float(candidate["validation_max_component_nmse"]))
            ]
            containing.sort(
                key=lambda candidate: (
                    float(candidate["validation_max_component_nmse"]),
                    float(candidate["validation_loss"]),
                    int(candidate["generation_rank"]),
                )
            )
            best = containing[0] if containing else None
            best_loss = None if best is None else float(best["validation_max_component_nmse"])
            rows.append(
                {
                    "rank": atom_rank,
                    "canonical_template": str(atom["canonical_template"]),
                    "tree_id": atom.get("tree_id"),
                    "tree_template": atom.get("tree_template"),
                    "variable_bindings": atom.get("variable_bindings"),
                    "evaluated_combination_count": len(containing),
                    "best_validation_max_component_nmse": best_loss,
                    "best_validation_residual_nmse": (
                        None if best is None else best.get("validation_residual_nmse")
                    ),
                    "semantic_status": (
                        "not_evaluated"
                        if best_loss is None
                        else "survived"
                        if best_loss <= tolerance
                        else "no_passing_combination_observed"
                    ),
                    "causality_claimed": False,
                }
            )
        assessments[slot.slot_id] = tuple(rows)
    return assessments


def _safe_feedback(value: object) -> str:
    """Keep retry feedback mechanical and free of protected evaluation vocabulary."""

    text = str(value).replace("`", "'")
    text = re.sub(r"\bbenchmark\b", "dataset", text, flags=re.IGNORECASE)
    text = re.sub(r"\bgold\b", "withheld", text, flags=re.IGNORECASE)
    text = re.sub(r"\breference[_ ]solution\b", "withheld solution", text, flags=re.IGNORECASE)
    return text[:4_000]


def _ansatz_retry_feedback(
    host_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    certified: Sequence[tuple[LLMAnsatzCandidateV1, Any]],
) -> tuple[str, ...]:
    feedback: list[str] = []
    for item in host_rows:
        if item.get("host_syntax_validation_passed"):
            continue
        raw = item.get("raw_candidate", {})
        ansatz = raw.get("ansatz", "<missing>") if isinstance(raw, Mapping) else raw
        feedback.append(
            _safe_feedback(f"Rejected Ansatz {ansatz}: {item.get('error', 'host syntax error')}")
        )
    for item in replay_rows:
        if item.get("passed"):
            continue
        feedback.append(
            _safe_feedback(
                f"Rejected Ansatz {item.get('ansatz', '<missing>')}: "
                f"{item.get('failure_reason', 'symbolic replay failed')}"
            )
        )
    if certified:
        retained = "; ".join(candidate.template for candidate, _certificate in certified)
        feedback.append(_safe_feedback(f"Already certified; do not repeat: {retained}"))
    feedback.append(
        "Return new candidates that pass the stated host contract; three distinct certified "
        "Ansatz candidates are required in total."
    )
    return tuple(dict.fromkeys(feedback))


def _atom_retry_feedback(
    rejected: Sequence[Mapping[str, Any]],
    accepted: Sequence[Mapping[str, Any]],
    *,
    duplicate_templates: Sequence[str] = (),
) -> tuple[str, ...]:
    feedback: list[str] = []
    for item in rejected:
        raw = item.get("raw_proposal", {})
        template = raw.get("canonical_template", "<missing>") if isinstance(raw, Mapping) else raw
        feedback.append(
            _safe_feedback(f"Rejected Atom {template}: {item.get('error', 'host Atom error')}")
        )
    for template in duplicate_templates:
        feedback.append(_safe_feedback(f"Duplicate Atom rejected: {template}"))
    if accepted:
        retained_items = tuple(accepted)[-32:]
        retained = "; ".join(str(item["canonical_template"]) for item in retained_items)
        feedback.append(
            _safe_feedback(
                f"Already accepted; do not repeat ({len(accepted)} total; latest 32 shown): "
                f"{retained}"
            )
        )
    feedback.append(
        "Return new proposals that exactly match expression-used ARG roles, consecutive p1..pN "
        "parameters, fitted outer amplitudes, nonlinear argument scales, and uppercase rationale "
        f"codes. {TREE_STAGE_INITIAL_ATOM_TARGET} distinct expanded Atoms are required in the "
        "initial pool."
    )
    return tuple(dict.fromkeys(feedback))


def _tree_bound_atom_validation(
    proposals: Sequence[Mapping[str, Any]], slot: RuntimeSlot
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Reuse the strict Atom validator and reattach immutable tree/binding provenance."""

    atom_fields = {
        "canonical_template",
        "atom_type",
        "variable_roles",
        "free_parameters",
        "rationale_codes",
        "reason",
    }
    pure = [{key: item[key] for key in atom_fields} for item in proposals]
    serialized = json.dumps(pure, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    accepted, rejected = _cached_tree_bound_atom_validation(
        serialized,
        slot.argument_roles,
        TREE_STAGE_INITIAL_ATOM_TARGET,
    )
    enriched = []
    for item in accepted:
        source = proposals[int(item["rank"]) - 1]
        enriched.append(
            {
                **item,
                "tree_id": source["tree_id"],
                "seed_tree_id": source.get("seed_tree_id", source["tree_id"]),
                "seed_tree_template": source.get("seed_tree_template", source["tree_template"]),
                "seed_structural_key": source.get("seed_structural_key", source["structural_key"]),
                "coverage_lane": source["coverage_lane"],
                "tree_template": source["tree_template"],
                "structural_key": source["structural_key"],
                "variable_bindings": source["variable_bindings"],
            }
        )
    return tuple(enriched), rejected


@lru_cache(maxsize=32)
def _cached_tree_bound_atom_validation(
    serialized_proposals: str,
    argument_roles: tuple[str, ...],
    accepted_limit: int,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Cache slot-shape-identical host checks without caching any provider generation."""

    role_symbols = tuple(sp.Symbol(role, real=True) for role in argument_roles)
    call = sp.Function("CachedProfile")(*role_symbols)
    slot = RuntimeSlot("cached_slot", call, argument_roles, "cached host-validation slot")
    proposals = json.loads(serialized_proposals)
    return validate_atom_response(
        {"proposals": proposals},
        slot,
        expected_count=len(proposals),
        accepted_limit=accepted_limit,
    )


def _tree_stage_feedback(
    tree_rejected: Sequence[Mapping[str, Any]],
    binding_rejected: Sequence[Mapping[str, Any]],
    atom_rejected: Sequence[Mapping[str, Any]],
    accepted: Sequence[Mapping[str, Any]],
    *,
    duplicate_templates: Sequence[str] = (),
) -> tuple[str, ...]:
    feedback: list[str] = []
    for item in tree_rejected:
        feedback.append(_safe_feedback(f"Rejected tree: {item.get('error', 'tree error')}"))
    for item in binding_rejected:
        feedback.append(
            _safe_feedback(f"Rejected tree binding: {item.get('error', 'binding error')}")
        )
    feedback.extend(
        _atom_retry_feedback(
            atom_rejected,
            accepted,
            duplicate_templates=duplicate_templates,
        )
    )
    feedback.append(
        "The next attempt must first propose new parameter-hole trees, then bind their V holes "
        "to this certified slot. Preserve P holes for host fitting."
    )
    return tuple(dict.fromkeys(feedback))


def _semantic_retry_feedback(
    diagnostics: Sequence[str],
    accepted: Sequence[Mapping[str, Any]],
    assessments: Sequence[Mapping[str, Any]] = (),
) -> tuple[str, ...]:
    def component_summary(item: Mapping[str, Any]) -> str:
        metrics = item.get("best_validation_residual_nmse")
        components = metrics.get("components") if isinstance(metrics, Mapping) else None
        if not isinstance(components, Mapping):
            return "component NMSE unavailable"
        values = []
        for name, component in sorted(components.items(), key=lambda pair: str(pair[0])):
            nmse = component.get("nmse") if isinstance(component, Mapping) else None
            if isinstance(nmse, (int, float)) and math.isfinite(float(nmse)):
                values.append(f"{name} NMSE {float(nmse):.6g}")
        return ", ".join(values) if values else "component NMSE unavailable"

    feedback = [_safe_feedback(item) for item in diagnostics]
    if accepted:
        retained_items = tuple(accepted)[-32:]
        retained = "; ".join(
            (
                f"{item.get('tree_template', '<legacy direct Atom tree>')} -> "
                f"{item['canonical_template']}"
            )
            for item in retained_items
        )
        feedback.append(
            _safe_feedback(
                f"Already retained; do not repeat ({len(accepted)} total; latest 32 shown): "
                f"{retained}"
            )
        )
    failed = [
        item
        for item in assessments
        if item.get("semantic_status") == "no_passing_combination_observed"
    ]
    if failed:
        summary = "; ".join(
            f"{item.get('tree_template', '<tree unavailable>')} -> "
            f"{item['canonical_template']} (best max component NMSE "
            f"{float(item['best_validation_max_component_nmse']):.6g}; "
            f"{component_summary(item)})"
            for item in failed[:8]
        )
        feedback.append(
            _safe_feedback(
                "No passing assembled combination was observed for these slot Atoms: " + summary
            )
        )
    feedback.append(
        "The next generation round must propose eight new parameter-hole tree seed programs from "
        "structurally different families, then bind them to this slot. Since retained structures "
        "did not pass, change integer-power, product, rational, sum, or composition topology "
        "rather than merely renaming it. Every nonlinear component still needs a fitted outer "
        "and inner argument scale. Host diagnostics may report PDE or public-condition failure "
        "classes, but never disclose a condition target, withheld solution, or preferred target "
        "family."
    )
    return tuple(dict.fromkeys(feedback))


def run_staged_experiment(
    rows: Sequence[Mapping[str, Any]],
    validation_specs: Mapping[str, Mapping[str, Any]],
    *,
    provider_factory: Callable[[int], LLMBatchProviderV1],
    requested_model: str,
    case_observer: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run condition-aware Stage I and target-neutral Stage II without offline gold."""

    stage_i_requests = 0
    stage_i_provider_errors = 0
    stage_i_syntax_errors = 0
    stage_i_replay_errors = 0
    stage_i_exhausted = 0
    stage_ii_requests = 0
    stage_ii_tree_requests = 0
    stage_ii_binding_requests = 0
    stage_ii_provider_errors = 0
    stage_ii_host_errors = 0
    stage_ii_tree_host_errors = 0
    stage_ii_binding_host_errors = 0
    stage_ii_exhausted = 0
    stage_ii_semantic_requests = 0
    stage_ii_semantic_provider_errors = 0
    stage_ii_semantic_host_errors = 0
    stage_ii_expanded_candidates = 0
    stage_ii_host_valid_candidates = 0
    stage_ii_screened_candidates = 0
    stage_ii_screening_valid_candidates = 0
    semantic_validation_runs = 0
    prompt_tokens_i = 0
    completion_tokens_i = 0
    prompt_tokens_ii = 0
    completion_tokens_ii = 0
    cases: list[dict[str, Any]] = []

    for row in rows:
        case_id = str(row["benchmark_id"])
        case_record: dict[str, Any] = {
            "case_id": case_id,
            "claim_type": "in_sample_full_problem_condition_aware_development_diagnostic",
            "atom_ansatz_policy": "richness_ranked_all_host_certified_ansatz",
            "stage_i": {},
            "certified_ansatz_runs": [],
            "assembled_candidates": [],
            "fitted_candidates": [],
            "selected": None,
        }
        best_semantic_evaluation: (
            tuple[tuple[dict[str, Any], ...], Mapping[str, Any] | None, tuple[str, ...]] | None
        ) = None

        stage_i_attempts: list[dict[str, Any]] = []
        certified_by_key: dict[str, tuple[LLMAnsatzCandidateV1, Any]] = {}
        feedback: tuple[str, ...] = ()
        stage_i_stop_reason = "max_attempts_exhausted"
        for attempt_index in range(1, MAX_REGENERATION_ATTEMPTS + 1):
            stage_i_requests += 1
            recorder = RecordingProvider(provider_factory(8_192))
            attempt_record: dict[str, Any] = {
                "attempt": attempt_index,
                "host_feedback_sent": list(feedback),
            }
            try:
                pools, stage_i_evidence = propose_llm_ansatz_batch(
                    (row,),
                    provider=recorder,
                    allow_empty_cases=True,
                    validation_feedback=feedback,
                )
            except (LLMAnsatzError, LLMResponseError, TypeError, ValueError) as exc:
                stage_i_provider_errors += 1
                attempt_record.update(
                    status="provider_or_schema_error", error=f"{type(exc).__name__}: {exc}"
                )
                stage_i_attempts.append(attempt_record)
                continue
            if recorder.completion is None:
                raise StagedExperimentError("Stage I provider returned no completion evidence")
            prompt_tokens_i += recorder.completion.prompt_tokens
            completion_tokens_i += recorder.completion.completion_tokens
            raw_response = recorder.completion.response
            raw_candidates = raw_response["cases"][0]["candidates"]
            host_rows: list[dict[str, Any]] = []
            for rank, raw_candidate in enumerate(raw_candidates, 1):
                try:
                    ansatz, reason = validate_llm_ansatz_candidate(
                        raw_candidate,
                        dependent_variables=row["dependent_variables"],
                        independent_variables=row["independent_variables"],
                        parameters=public_problem_payload(row)["parameters"],
                    )
                    host_rows.append(
                        {
                            "rank": rank,
                            "raw_candidate": raw_candidate,
                            "ansatz": ansatz,
                            "reason": reason,
                            "host_syntax_validation_passed": True,
                        }
                    )
                except (LLMAnsatzError, TypeError, ValueError) as exc:
                    stage_i_syntax_errors += 1
                    host_rows.append(
                        {
                            "rank": rank,
                            "raw_candidate": raw_candidate,
                            "host_syntax_validation_passed": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            validated = pools[0]
            certificates = replay_ansatz_candidates(
                row,
                tuple({"ansatz": item.template, "reason": item.reason} for item in validated),
            )
            replay_rows: list[dict[str, Any]] = []
            for candidate, certificate in zip(validated, certificates, strict=True):
                mapping = {
                    "rank": candidate.rank,
                    "ansatz": candidate.template,
                    "reason": candidate.reason,
                    "passed": certificate.passed,
                    "validation_kind": certificate.validation_kind,
                    "reduced_equations": list(certificate.reduced_equations),
                    "invariant_coordinates": list(certificate.invariant_coordinates),
                    "side_conditions": list(certificate.side_conditions),
                    "evidence": list(certificate.evidence),
                    "failure_reason": certificate.failure_reason,
                }
                replay_rows.append(mapping)
                if not certificate.passed:
                    stage_i_replay_errors += 1
                    continue
                key = re.sub(r"\s+", "", candidate.template)
                if key not in certified_by_key and len(certified_by_key) < STAGE_I_CANDIDATE_COUNT:
                    ranked = replace(candidate, rank=len(certified_by_key) + 1)
                    certified_by_key[key] = (ranked, certificate)
            attempt_record.update(
                status="completed",
                raw_response=raw_response,
                provider_evidence=_provider_evidence(recorder.completion),
                generation_evidence=stage_i_evidence,
                host_syntax_results=host_rows,
                symbolic_replay_results=replay_rows,
                certified_count_after_attempt=len(certified_by_key),
            )
            stage_i_attempts.append(attempt_record)
            if len(certified_by_key) == STAGE_I_CANDIDATE_COUNT:
                stage_i_stop_reason = f"{STAGE_I_CANDIDATE_COUNT}_host_certified_ansatz_candidates"
                break
            if certificates and all(
                not certificate.passed
                and certificate.validation_kind == "rejected_before_symbolic_replay_v1"
                for certificate in certificates
            ):
                stage_i_stop_reason = "deterministic_public_operator_parse_blocker"
                break
            feedback = _ansatz_retry_feedback(
                host_rows, replay_rows, tuple(certified_by_key.values())
            )
        else:
            stage_i_exhausted += 1

        certified = list(certified_by_key.values())
        ranked_certified = rank_certified_ansatz_candidates(row, certified)
        case_record["stage_i"] = {
            "status": "completed" if certified else "no_certified_ansatz",
            "attempt_count": len(stage_i_attempts),
            "max_attempts": MAX_REGENERATION_ATTEMPTS,
            "stop_reason": stage_i_stop_reason,
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
                ast, slots = extract_runtime_slots(row, candidate.template, candidate.reason)
            except (LLMAnsatzError, StagedExperimentError, TypeError, ValueError) as exc:
                ansatz_record.update(
                    status="slot_extraction_error", error=f"{type(exc).__name__}: {exc}"
                )
                case_record["certified_ansatz_runs"].append(ansatz_record)
                continue
            pools_for_ansatz: list[tuple[dict[str, Any], ...]] = []
            slot_states: list[
                tuple[
                    RuntimeSlot,
                    dict[str, dict[str, Any]],
                    list[dict[str, Any]],
                    dict[str, Any],
                ]
            ] = []
            ansatz_blocked = False
            for slot in slots:
                slot_attempts: list[dict[str, Any]] = []
                accepted_by_template: dict[str, dict[str, Any]] = {}
                slot_feedback: tuple[str, ...] = ()
                for attempt_index in range(1, MAX_REGENERATION_ATTEMPTS + 1):
                    if stage_ii_requests >= MAX_STAGE_II_REQUESTS:
                        raise StagedExperimentError("frozen Stage II request budget exhausted")
                    stage_ii_requests += 1
                    stage_ii_tree_requests += 1
                    tree_recorder = GenericRecordingProvider(provider_factory(4_096))
                    attempt_record = {
                        "attempt": attempt_index,
                        "phase": "tree_then_slot_binding",
                        "host_feedback_sent": list(slot_feedback),
                    }
                    try:
                        trees, tree_rejected, tree_completion, tree_prompt_sha = request_tree_batch(
                            tree_recorder,
                            row,
                            certified_ansatz=candidate.template,
                            slot_id=slot.slot_id,
                            profile_arity=len(slot.call.args),
                            validation_feedback=slot_feedback,
                            forbidden_structural_keys=tuple(
                                str(item["structural_key"])
                                for item in accepted_by_template.values()
                            ),
                            exploration_pass=attempt_index - 1,
                        )
                    except (LLMResponseError, TypeError) as exc:
                        stage_ii_provider_errors += 1
                        attempt_record.update(
                            status="tree_provider_error", error=f"{type(exc).__name__}: {exc}"
                        )
                        slot_attempts.append(attempt_record)
                        continue
                    except (AtomTreeError, ValueError) as exc:
                        stage_ii_host_errors += 1
                        stage_ii_tree_host_errors += 1
                        error = f"{type(exc).__name__}: {exc}"
                        attempt_record.update(
                            status="tree_host_response_error",
                            raw_tree_response=(
                                None
                                if tree_recorder.completion is None
                                else tree_recorder.completion.response
                            ),
                            error=error,
                        )
                        slot_attempts.append(attempt_record)
                        slot_feedback = (_safe_feedback(f"Host tree response error: {error}"),)
                        continue
                    prompt_tokens_ii += tree_completion.prompt_tokens
                    completion_tokens_ii += tree_completion.completion_tokens
                    stage_ii_host_errors += len(tree_rejected)
                    stage_ii_tree_host_errors += len(tree_rejected)
                    attempt_record.update(
                        tree_prompt_sha256=tree_prompt_sha,
                        raw_tree_response=tree_completion.response,
                        tree_provider_evidence=_provider_evidence(tree_completion),
                        accepted_trees=[tree.to_mapping() for tree in trees],
                        rejected_trees=list(tree_rejected),
                    )
                    if not trees:
                        attempt_record["status"] = "tree_host_validation_failed"
                        slot_attempts.append(attempt_record)
                        slot_feedback = _tree_stage_feedback(
                            tree_rejected, (), (), tuple(accepted_by_template.values())
                        )
                        continue

                    if stage_ii_requests >= MAX_STAGE_II_REQUESTS:
                        raise StagedExperimentError("frozen Stage II request budget exhausted")
                    stage_ii_requests += 1
                    stage_ii_binding_requests += 1
                    binding_recorder = GenericRecordingProvider(provider_factory(4_096))
                    try:
                        bound, binding_rejected, binding_completion, binding_prompt_sha = (
                            request_bound_atoms(
                                binding_recorder,
                                row,
                                certified_ansatz=candidate.template,
                                slot_id=slot.slot_id,
                                current_ansatz=slot.current_ansatz,
                                profile_call=sp.sstr(slot.call),
                                profile_arguments=tuple(sp.sstr(item) for item in slot.call.args),
                                trees=trees,
                                validation_feedback=slot_feedback,
                            )
                        )
                    except (LLMResponseError, TypeError) as exc:
                        stage_ii_provider_errors += 1
                        attempt_record.update(
                            status="binding_provider_error",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        slot_attempts.append(attempt_record)
                        continue
                    except (AtomTreeError, ValueError) as exc:
                        stage_ii_host_errors += 1
                        stage_ii_binding_host_errors += 1
                        error = f"{type(exc).__name__}: {exc}"
                        attempt_record.update(
                            status="binding_host_response_error",
                            raw_binding_response=(
                                None
                                if binding_recorder.completion is None
                                else binding_recorder.completion.response
                            ),
                            error=error,
                        )
                        slot_attempts.append(attempt_record)
                        slot_feedback = (_safe_feedback(f"Host binding response error: {error}"),)
                        continue
                    prompt_tokens_ii += binding_completion.prompt_tokens
                    completion_tokens_ii += binding_completion.completion_tokens
                    stage_ii_expanded_candidates += len(bound)
                    stage_ii_host_errors += len(binding_rejected)
                    stage_ii_binding_host_errors += len(binding_rejected)
                    attempt_record.update(
                        binding_prompt_sha256=binding_prompt_sha,
                        raw_binding_response=binding_completion.response,
                        binding_provider_evidence=_provider_evidence(binding_completion),
                        rejected_bindings=list(binding_rejected),
                        expanded_atom_candidate_count=len(bound),
                    )
                    if not bound:
                        attempt_record.update(
                            status="binding_host_validation_failed",
                            bound_tree_candidates=list(bound),
                        )
                        slot_attempts.append(attempt_record)
                        slot_feedback = _tree_stage_feedback(
                            tree_rejected,
                            binding_rejected,
                            (),
                            tuple(accepted_by_template.values()),
                        )
                        continue
                    try:
                        accepted, rejected = _tree_bound_atom_validation(bound, slot)
                    except (StagedExperimentError, TypeError, ValueError) as exc:
                        stage_ii_host_errors += 1
                        error = f"{type(exc).__name__}: {exc}"
                        attempt_record.update(status="atom_host_response_error", error=error)
                        slot_attempts.append(attempt_record)
                        slot_feedback = (_safe_feedback(f"Host Atom response error: {error}"),)
                        continue
                    stage_ii_host_errors += len(rejected)
                    stage_ii_host_valid_candidates += len(accepted)
                    duplicate_templates: list[str] = []
                    duplicate_structural_keys: list[str] = []
                    retained_structural_keys = {
                        str(item["structural_key"]) for item in accepted_by_template.values()
                    }
                    for item in accepted:
                        key = str(item["canonical_template"])
                        if key in accepted_by_template:
                            duplicate_templates.append(key)
                            continue
                        structural_key = str(item["structural_key"])
                        if structural_key in retained_structural_keys:
                            duplicate_structural_keys.append(structural_key)
                            continue
                        if len(accepted_by_template) < TREE_STAGE_INITIAL_ATOM_TARGET:
                            ranked_atom = dict(item)
                            ranked_atom["rank"] = len(accepted_by_template) + 1
                            accepted_by_template[key] = ranked_atom
                            retained_structural_keys.add(structural_key)
                    attempt_record.update(
                        status="completed",
                        bound_tree_candidates=list(bound),
                        accepted_candidates=list(accepted),
                        host_valid_expanded_atom_count=len(accepted),
                        rejected_candidates=list(rejected),
                        duplicate_templates=duplicate_templates,
                        duplicate_structural_keys=duplicate_structural_keys,
                        accepted_count_after_attempt=len(accepted_by_template),
                    )
                    slot_attempts.append(attempt_record)
                    if len(accepted_by_template) == TREE_STAGE_INITIAL_ATOM_TARGET:
                        break
                    slot_feedback = _tree_stage_feedback(
                        tree_rejected,
                        binding_rejected,
                        rejected,
                        tuple(accepted_by_template.values()),
                        duplicate_templates=duplicate_templates,
                    )
                accepted_pool = tuple(accepted_by_template.values())
                if len(accepted_pool) != TREE_STAGE_INITIAL_ATOM_TARGET:
                    stage_ii_exhausted += 1
                    ansatz_blocked = True
                pools_for_ansatz.append(accepted_pool)
                slot_record = {
                    **slot.public_mapping(),
                    "status": (
                        "completed"
                        if len(accepted_pool) == TREE_STAGE_INITIAL_ATOM_TARGET
                        else "max_attempts_exhausted"
                    ),
                    "attempt_count": len(slot_attempts),
                    "max_initial_attempts": MAX_REGENERATION_ATTEMPTS,
                    "max_semantic_regeneration_rounds": MAX_SEMANTIC_REGENERATION_ROUNDS,
                    "attempts": slot_attempts,
                    "accepted_candidates": list(accepted_pool),
                }
                ansatz_record["slots"].append(slot_record)
                slot_states.append((slot, accepted_by_template, slot_attempts, slot_record))
            if ansatz_blocked:
                ansatz_record["status"] = "blocked_before_assembly"
            else:
                assembled = assemble_candidate_beam(
                    ast, slots, pools_for_ansatz, ansatz_rank=execution_rank
                )
                semantic_rounds: list[dict[str, Any]] = []
                if spec is not None:
                    reuse_semantic_evaluation = False
                    for semantic_round in range(MAX_SEMANTIC_REGENERATION_ROUNDS + 1):
                        if not reuse_semantic_evaluation:
                            semantic_validation_runs += 1
                            screened_round = screen_semantic_candidate_beam(
                                row,
                                assembled,
                                spec,
                                max_candidates=(
                                    MAX_SEMANTIC_FITS_PER_ANSATZ
                                    if len(slots) == 1
                                    else MAX_MULTISLOT_SEMANTIC_FITS_PER_ANSATZ
                                ),
                            )
                            stage_ii_screened_candidates += len(assembled)
                            stage_ii_screening_valid_candidates += (
                                0
                                if not screened_round
                                else int(
                                    screened_round[0].get(
                                        "screening_valid_count", len(screened_round)
                                    )
                                )
                            )
                            fitted_round, selected_round, diagnostics = (
                                semantic_validate_candidates(
                                    row,
                                    screened_round,
                                    spec,
                                )
                            )
                            current_semantic_evaluation = (
                                fitted_round,
                                selected_round,
                                diagnostics,
                            )
                            best_semantic_evaluation = _retain_better_semantic_evaluation(
                                best_semantic_evaluation,
                                current_semantic_evaluation,
                            )
                            passed = bool(
                                selected_round is not None
                                and selected_round["semantic_validation_passed"]
                            )
                            atom_assessments = assess_atom_pool_semantics(
                                fitted_round,
                                slots,
                                pools_for_ansatz,
                            )
                            semantic_rounds.append(
                                {
                                    "round": semantic_round,
                                    "candidate_count": len(assembled),
                                    "screened_count": len(assembled),
                                    "screening_valid_count": (
                                        0
                                        if not screened_round
                                        else int(
                                            screened_round[0].get(
                                                "screening_valid_count", len(screened_round)
                                            )
                                        )
                                    ),
                                    "bfgs_candidate_count": len(screened_round),
                                    "fitted_count": len(fitted_round),
                                    "best_validation_loss": (
                                        None
                                        if selected_round is None
                                        else selected_round.get("validation_loss")
                                    ),
                                    "best_validation_max_component_nmse": (
                                        None
                                        if selected_round is None
                                        else selected_round.get("validation_max_component_nmse")
                                    ),
                                    "best_validation_residual_nmse": (
                                        None
                                        if selected_round is None
                                        else selected_round.get("validation_residual_nmse")
                                    ),
                                    "passed": passed,
                                    "diagnostics": list(diagnostics),
                                    "atom_assessments": {
                                        slot_id: list(items)
                                        for slot_id, items in atom_assessments.items()
                                    },
                                    "regeneration_attempts": [],
                                }
                            )
                        else:
                            semantic_rounds[-1]["duplicate_revalidation_skipped"] = True
                        if passed or semantic_round == MAX_SEMANTIC_REGENERATION_ROUNDS:
                            break

                        new_atom_count = 0
                        for slot, accepted_by_template, slot_attempts, slot_record in slot_states:
                            if len(accepted_by_template) >= MAX_ATOM_POOL_SIZE:
                                continue
                            if stage_ii_requests >= MAX_STAGE_II_REQUESTS:
                                raise StagedExperimentError(
                                    "frozen Stage II request budget exhausted"
                                )
                            semantic_feedback = _semantic_retry_feedback(
                                diagnostics,
                                tuple(accepted_by_template.values()),
                                atom_assessments.get(slot.slot_id, ()),
                            )
                            stage_ii_requests += 1
                            stage_ii_tree_requests += 1
                            stage_ii_semantic_requests += 1
                            tree_recorder = GenericRecordingProvider(provider_factory(4_096))
                            semantic_attempt_record: dict[str, Any] = {
                                "attempt": len(slot_attempts) + 1,
                                "phase": "semantic_tree_then_slot_binding_regeneration",
                                "semantic_round": semantic_round + 1,
                                "host_feedback_sent": list(semantic_feedback),
                            }
                            try:
                                trees, tree_rejected, tree_completion, tree_prompt_sha = (
                                    request_tree_batch(
                                        tree_recorder,
                                        row,
                                        certified_ansatz=candidate.template,
                                        slot_id=slot.slot_id,
                                        profile_arity=len(slot.call.args),
                                        validation_feedback=semantic_feedback,
                                        forbidden_structural_keys=tuple(
                                            str(item["structural_key"])
                                            for item in accepted_by_template.values()
                                        ),
                                        exploration_pass=semantic_round + 1,
                                    )
                                )
                            except (LLMResponseError, TypeError) as exc:
                                stage_ii_provider_errors += 1
                                stage_ii_semantic_provider_errors += 1
                                semantic_attempt_record.update(
                                    status="tree_provider_error",
                                    error=f"{type(exc).__name__}: {exc}",
                                )
                                slot_attempts.append(semantic_attempt_record)
                                continue
                            except (AtomTreeError, ValueError) as exc:
                                stage_ii_host_errors += 1
                                stage_ii_tree_host_errors += 1
                                stage_ii_semantic_host_errors += 1
                                semantic_attempt_record.update(
                                    status="tree_host_response_error",
                                    raw_tree_response=(
                                        None
                                        if tree_recorder.completion is None
                                        else tree_recorder.completion.response
                                    ),
                                    error=f"{type(exc).__name__}: {exc}",
                                )
                                slot_attempts.append(semantic_attempt_record)
                                continue
                            prompt_tokens_ii += tree_completion.prompt_tokens
                            completion_tokens_ii += tree_completion.completion_tokens
                            stage_ii_host_errors += len(tree_rejected)
                            stage_ii_tree_host_errors += len(tree_rejected)
                            stage_ii_semantic_host_errors += len(tree_rejected)
                            semantic_attempt_record.update(
                                tree_prompt_sha256=tree_prompt_sha,
                                raw_tree_response=tree_completion.response,
                                tree_provider_evidence=_provider_evidence(tree_completion),
                                accepted_trees=[tree.to_mapping() for tree in trees],
                                rejected_trees=list(tree_rejected),
                            )
                            if not trees:
                                semantic_attempt_record.update(
                                    status="tree_host_validation_failed",
                                )
                                slot_attempts.append(semantic_attempt_record)
                                continue

                            if stage_ii_requests >= MAX_STAGE_II_REQUESTS:
                                raise StagedExperimentError(
                                    "frozen Stage II request budget exhausted"
                                )
                            stage_ii_requests += 1
                            stage_ii_binding_requests += 1
                            stage_ii_semantic_requests += 1
                            binding_recorder = GenericRecordingProvider(provider_factory(4_096))
                            try:
                                bound, binding_rejected, binding_completion, binding_prompt_sha = (
                                    request_bound_atoms(
                                        binding_recorder,
                                        row,
                                        certified_ansatz=candidate.template,
                                        slot_id=slot.slot_id,
                                        current_ansatz=slot.current_ansatz,
                                        profile_call=sp.sstr(slot.call),
                                        profile_arguments=tuple(
                                            sp.sstr(item) for item in slot.call.args
                                        ),
                                        trees=trees,
                                        validation_feedback=semantic_feedback,
                                    )
                                )
                            except (LLMResponseError, TypeError) as exc:
                                stage_ii_provider_errors += 1
                                stage_ii_semantic_provider_errors += 1
                                semantic_attempt_record.update(
                                    status="binding_provider_error",
                                    error=f"{type(exc).__name__}: {exc}",
                                )
                                slot_attempts.append(semantic_attempt_record)
                                continue
                            except (AtomTreeError, ValueError) as exc:
                                stage_ii_host_errors += 1
                                stage_ii_binding_host_errors += 1
                                stage_ii_semantic_host_errors += 1
                                semantic_attempt_record.update(
                                    status="binding_host_response_error",
                                    raw_binding_response=(
                                        None
                                        if binding_recorder.completion is None
                                        else binding_recorder.completion.response
                                    ),
                                    error=f"{type(exc).__name__}: {exc}",
                                )
                                slot_attempts.append(semantic_attempt_record)
                                continue
                            prompt_tokens_ii += binding_completion.prompt_tokens
                            completion_tokens_ii += binding_completion.completion_tokens
                            stage_ii_expanded_candidates += len(bound)
                            stage_ii_host_errors += len(binding_rejected)
                            stage_ii_binding_host_errors += len(binding_rejected)
                            stage_ii_semantic_host_errors += len(binding_rejected)
                            semantic_attempt_record.update(
                                binding_prompt_sha256=binding_prompt_sha,
                                raw_binding_response=binding_completion.response,
                                binding_provider_evidence=_provider_evidence(binding_completion),
                                rejected_bindings=list(binding_rejected),
                                expanded_atom_candidate_count=len(bound),
                                bound_tree_candidates=list(bound),
                            )
                            if not bound:
                                semantic_attempt_record["status"] = "binding_host_validation_failed"
                                slot_attempts.append(semantic_attempt_record)
                                continue
                            try:
                                accepted, rejected = _tree_bound_atom_validation(bound, slot)
                            except (StagedExperimentError, TypeError, ValueError) as exc:
                                stage_ii_host_errors += 1
                                stage_ii_semantic_host_errors += 1
                                semantic_attempt_record.update(
                                    status="atom_host_response_error",
                                    error=f"{type(exc).__name__}: {exc}",
                                )
                                slot_attempts.append(semantic_attempt_record)
                                continue
                            stage_ii_host_errors += len(rejected)
                            stage_ii_semantic_host_errors += len(rejected)
                            stage_ii_host_valid_candidates += len(accepted)
                            semantic_duplicate_templates: list[str] = []
                            semantic_duplicate_structural_keys: list[str] = []
                            retained_structural_keys = {
                                str(item["structural_key"])
                                for item in accepted_by_template.values()
                            }
                            accepted_before = len(accepted_by_template)
                            for item in accepted:
                                key = str(item["canonical_template"])
                                if key in accepted_by_template:
                                    semantic_duplicate_templates.append(key)
                                    continue
                                structural_key = str(item["structural_key"])
                                if structural_key in retained_structural_keys:
                                    semantic_duplicate_structural_keys.append(structural_key)
                                    continue
                                if len(accepted_by_template) >= MAX_ATOM_POOL_SIZE:
                                    break
                                ranked_atom = dict(item)
                                ranked_atom["rank"] = len(accepted_by_template) + 1
                                accepted_by_template[key] = ranked_atom
                                retained_structural_keys.add(structural_key)
                            added = len(accepted_by_template) - accepted_before
                            new_atom_count += added
                            semantic_attempt_record.update(
                                status="completed",
                                accepted_candidates=list(accepted),
                                host_valid_expanded_atom_count=len(accepted),
                                rejected_candidates=list(rejected),
                                duplicate_templates=semantic_duplicate_templates,
                                duplicate_structural_keys=semantic_duplicate_structural_keys,
                                newly_retained_count=added,
                                accepted_count_after_attempt=len(accepted_by_template),
                            )
                            slot_attempts.append(semantic_attempt_record)
                            slot_record["attempt_count"] = len(slot_attempts)
                            slot_record["accepted_candidates"] = list(accepted_by_template.values())
                        if new_atom_count == 0:
                            semantic_rounds[-1]["regeneration_attempts"].append(
                                {
                                    "attempt": semantic_round + 1,
                                    "status": "semantic_regeneration_added_no_new_atoms",
                                }
                            )
                            reuse_semantic_evaluation = True
                        else:
                            semantic_rounds[-1]["regeneration_attempts"].append(
                                {
                                    "attempt": semantic_round + 1,
                                    "status": "semantic_regeneration_added_new_atoms",
                                    "new_atom_count": new_atom_count,
                                }
                            )
                            reuse_semantic_evaluation = False
                            pools_for_ansatz = [tuple(state[1].values()) for state in slot_states]
                            assembled = assemble_candidate_beam(
                                ast,
                                slots,
                                pools_for_ansatz,
                                ansatz_rank=execution_rank,
                            )
                assembled_all.extend(assembled)
                ansatz_record["semantic_rounds"] = semantic_rounds
                if not semantic_rounds:
                    ansatz_record["status"] = "assembled_generation_only"
                elif semantic_rounds[-1]["passed"]:
                    ansatz_record["status"] = "semantic_validation_passed"
                else:
                    ansatz_record["status"] = "semantic_validation_failed"
                ansatz_record["assembled_candidate_count"] = len(assembled)
            case_record["certified_ansatz_runs"].append(ansatz_record)

        case_record["assembled_candidates"] = assembled_all
        if spec is None:
            case_record["status"] = "staged_generation_only_unscorable_public_forcing"
            case_record["optimization_blocker"] = (
                "no executable public-only residual/condition specification; optimization not run"
            )
        elif not assembled_all:
            case_record["status"] = "no_fully_staged_candidate"
        elif best_semantic_evaluation is not None:
            fitted, selected, diagnostics = best_semantic_evaluation
            case_record["fitted_candidates"] = list(fitted)
            case_record["selected"] = selected
            case_record["semantic_diagnostics"] = list(diagnostics)
            if selected is None:
                case_record["status"] = "no_valid_fit"
            elif selected["semantic_validation_passed"]:
                case_record["status"] = "semantic_validation_passed"
            else:
                case_record["status"] = "semantic_validation_failed"
        else:
            ranked_assembled = sorted(
                assembled_all,
                key=lambda item: (
                    int(item["atom_combination_rank"]),
                    int(item["ansatz_rank"]),
                ),
            )
            semantic_beam = screen_semantic_candidate_beam(
                row,
                ranked_assembled,
                spec,
                max_candidates=MAX_SEMANTIC_FITS_PER_CASE,
            )
            stage_ii_screened_candidates += len(ranked_assembled)
            stage_ii_screening_valid_candidates += (
                0
                if not semantic_beam
                else int(semantic_beam[0].get("screening_valid_count", len(semantic_beam)))
            )
            fitted, selected, diagnostics = semantic_validate_candidates(row, semantic_beam, spec)
            semantic_validation_runs += 1
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
        "schema_version": "StagedAnsatzTreeAtomNMSEDiagnosticV11",
        "claim_type": "in_sample_full_problem_condition_aware_development_diagnostic",
        "gold_accessed": False,
        "reference_solution_accessed": False,
        "generation_input_policy": (
            "declared public PDE/forcing/IC/BC/domain/parameters to Stage I; only the public PDE "
            "operator, host-certified Ansatz, and slot contract to Stage II; offline gold and "
            "reference solutions are never opened"
        ),
        "condition_use_policy": (
            "declared public conditions inform atom-free Stage I structure and post-assembly host "
            "validation; condition target expressions are excluded from Stage II and feedback"
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
            "host_feedback_regeneration_policy": {
                "max_attempts_per_problem_or_slot": MAX_REGENERATION_ATTEMPTS,
                "max_semantic_regeneration_rounds": MAX_SEMANTIC_REGENERATION_ROUNDS,
                "feedback": (
                    "previous generated trees and bound expressions, host rejection reasons, "
                    "total weighted residual NMSE, per-component residual NMSE, and target-free "
                    "PDE/public-condition failure classes"
                ),
                "structural_diversity_instruction": (
                    "require eight fixed topology-coverage seed lanes, expand every accepted "
                    "LLM seed over the complete target-neutral operator grammar, reject reused "
                    "structural keys, and after semantic failure request new seed topology"
                ),
                "duplicate_only_round_policy": (
                    "record the zero-novelty round and continue to the next bounded regeneration "
                    "round instead of terminating the loop early"
                ),
                "retain_valid_candidates": True,
                "condition_targets_or_gold_in_feedback": False,
                "public_condition_failure_class_in_feedback": True,
            },
            "stage_i": {
                "request_count": stage_i_requests,
                "provider_or_schema_error_count": stage_i_provider_errors,
                "host_syntax_candidate_error_count": stage_i_syntax_errors,
                "symbolic_replay_error_count": stage_i_replay_errors,
                "max_attempts_exhausted_case_count": stage_i_exhausted,
                "prompt_tokens": prompt_tokens_i,
                "completion_tokens": completion_tokens_i,
            },
            "stage_ii": {
                "request_count": stage_ii_requests,
                "tree_request_count": stage_ii_tree_requests,
                "binding_request_count": stage_ii_binding_requests,
                "provider_error_count": stage_ii_provider_errors,
                "host_candidate_or_response_error_count": stage_ii_host_errors,
                "tree_host_error_count": stage_ii_tree_host_errors,
                "binding_host_error_count": stage_ii_binding_host_errors,
                "max_attempts_exhausted_slot_count": stage_ii_exhausted,
                "semantic_regeneration_request_count": stage_ii_semantic_requests,
                "semantic_regeneration_provider_error_count": (stage_ii_semantic_provider_errors),
                "semantic_regeneration_host_error_count": stage_ii_semantic_host_errors,
                "host_semantic_validation_run_count": semantic_validation_runs,
                "expanded_atom_candidate_count": stage_ii_expanded_candidates,
                "host_valid_atom_candidate_count": stage_ii_host_valid_candidates,
                "residual_screened_assembled_candidate_count": stage_ii_screened_candidates,
                "residual_screening_valid_candidate_count": (stage_ii_screening_valid_candidates),
                "prompt_tokens": prompt_tokens_ii,
                "completion_tokens": completion_tokens_ii,
            },
        },
        "policies": {
            "ansatz_symbolic_validation": {
                "strong": (
                    "single-coordinate, multi-coordinate, or generic separation reduction proof"
                ),
                "weak": (
                    "reduced, disjoint-separation, coupled, or full-dimensional slot closure "
                    "under public-PDE substitution; explicitly requires Atom completion"
                ),
            },
            "stage_i_candidates_per_response": STAGE_I_CANDIDATE_COUNT,
            "stage_i_certified_target_per_problem": STAGE_I_CANDIDATE_COUNT,
            "atom_ansatz_policy": "richness_ranked_all_host_certified_ansatz",
            "ansatz_richness_priority": (
                "full-arity profiles, full-coordinate coverage, covered coordinates, profile/"
                "coefficient count, symbolic operation count, source length, then provider rank"
            ),
            "max_certified_ansatz_runs_per_case": MAX_CERTIFIED_ANSATZ_RUNS_PER_CASE,
            "max_profile_arity": MAX_PROFILE_ARITY,
            "stage_ii_candidates_per_response": STAGE_II_CANDIDATE_COUNT,
            "llm_tree_seed_programs_per_response": TREE_CANDIDATE_COUNT,
            "expanded_atom_validation_batch_size": EXPANDED_ATOM_VALIDATION_BATCH_SIZE,
            "stage_ii_accepted_target_per_slot": TREE_STAGE_INITIAL_ATOM_TARGET,
            "max_retained_atoms_per_slot": MAX_ATOM_POOL_SIZE,
            "semantic_regeneration_rounds": MAX_SEMANTIC_REGENERATION_ROUNDS,
            "stage_ii_request_budget": MAX_STAGE_II_REQUESTS,
            "parameterization": (
                "every role-dependent Atom requires a fitted outer amplitude; nonlinear "
                "functions also require fitted argument scales"
            ),
            "tree_then_binding": (
                "LLM-generated prefix seed programs contain only consecutive V/P holes; a "
                "second LLM call binds V holes to certified slot ARG roles; the host expands "
                "every seed over the complete frozen unary/power grammar and deterministically "
                "instantiates P holes as fitted p coefficients"
            ),
            "supported_elementary_functions": list(SUPPORTED_ELEMENTARY_FUNCTIONS),
            "semantic_regeneration_structural_diversity": (
                "feedback-guided target-neutral mutation across unused unary operators, powers, "
                "products, sums, rational forms, and composed variants"
            ),
            "partial_tree_batch_policy": (
                "bind and retain every host-valid tree even when another tree in the same LLM "
                "response is rejected"
            ),
            "duplicate_revalidation_policy": (
                "when regeneration adds no Atom, reuse the previous NMSE assessment for the next "
                "regeneration request instead of repeating deterministic fitting"
            ),
            "assembly": "bounded 1024-candidate Cartesian beam across slot pools",
            "semantic_fit_beam": (
                "evaluate fixed-initialization residual NMSE for every assembled candidate, "
                "then send only the best bounded subset to multi-start BFGS"
            ),
            "max_atom_combinations_per_ansatz": MAX_ATOM_COMBINATIONS,
            "residual_screening": {
                "population": "every assembled candidate, up to 1024 per semantic round",
                "coefficient_policy": (
                    "first frozen role-aware deterministic coefficient initialization"
                ),
                "ranking": (
                    "training max component residual NMSE, total weighted residual NMSE, "
                    "then generation order"
                ),
                "bfgs_only_after_screening": True,
                "complex_value_policy": (
                    "reject before real casting, optimization, or candidate ranking"
                ),
            },
            "max_semantic_fits_per_ansatz": MAX_SEMANTIC_FITS_PER_ANSATZ,
            "max_multislot_semantic_fits_per_ansatz": (MAX_MULTISLOT_SEMANTIC_FITS_PER_ANSATZ),
            "max_semantic_fits_per_case": MAX_SEMANTIC_FITS_PER_CASE,
            "semantic_acceptance_max_component_nmse": SEMANTIC_ACCEPTANCE_LOSS,
            "semantic_acceptance_policy": SEMANTIC_ACCEPTANCE_POLICY,
            "normalization": {
                "policy": (
                    "candidate-independent public zero-field PDE forcing energy and public "
                    "condition-target energy"
                ),
                "energy_floor": NORMALIZATION_ENERGY_FLOOR,
            },
            "selection": (
                "minimum disjoint public validation max component NMSE, then total weighted "
                "NMSE, then generation rank across every computed semantic round and every "
                "host-certified Ansatz; no first-pass Ansatz early stop"
            ),
            "max_fitted_coefficients": MAX_FITTED_COEFFICIENTS,
            "optimizer": {
                "name": "BFGS",
                "max_iterations": "min(900, 300 + 75*(slot_count-1))",
                "gradient_tolerance": 1.0e-8,
                "restarts": "min(10, 6 + floor((slot_count-1)/2))",
                "random_seed": 20260904,
                "restart_scale": 1.0,
                "acceptance_loss": SEMANTIC_ACCEPTANCE_LOSS,
                "nonlinear_argument_scale_starts": [1.0, -1.0, 4.0, -4.0, 16.0, -16.0],
            },
        },
        "case_count": len(cases),
        "cases": cases,
    }


__all__ = [
    "ATOM_TYPES",
    "AnsatzRichnessV1",
    "MAX_ATOM_REASON_LENGTH",
    "MAX_ATOM_TEMPLATE_LENGTH",
    "MAX_ATOM_COMBINATIONS",
    "MAX_ATOM_POOL_SIZE",
    "MAX_CERTIFIED_ANSATZ_RUNS_PER_CASE",
    "MAX_FITTED_COEFFICIENTS",
    "MAX_PROFILE_ARITY",
    "MAX_REGENERATION_ATTEMPTS",
    "MAX_MULTISLOT_SEMANTIC_FITS_PER_ANSATZ",
    "MAX_SEMANTIC_FITS_PER_ANSATZ",
    "MAX_SEMANTIC_FITS_PER_CASE",
    "MAX_SEMANTIC_REGENERATION_ROUNDS",
    "MAX_STAGE_II_REQUESTS",
    "SEMANTIC_ACCEPTANCE_LOSS",
    "SEMANTIC_ACCEPTANCE_POLICY",
    "STAGE_I_CANDIDATE_COUNT",
    "STAGE_II_CANDIDATE_COUNT",
    "TREE_STAGE_INITIAL_ATOM_TARGET",
    "STAGE_II_SYSTEM_PROMPT",
    "StagedExperimentError",
    "assess_atom_pool_semantics",
    "assemble_candidate_beam",
    "assemble_aligned_candidates",
    "atom_response_format",
    "canonical_sha256",
    "coefficient_initial_guesses",
    "extract_runtime_slots",
    "fit_scalar_candidates",
    "measure_ansatz_richness",
    "optimizer_config_for_slot_count",
    "rank_certified_ansatz_candidates",
    "render_atom_messages",
    "run_staged_experiment",
    "screen_semantic_candidate_beam",
    "select_semantic_candidate_beam",
    "semantic_validate_candidates",
    "validate_atom_response",
]
