"""Strict condition-aware Stage-I Ansatz -> Stage-II Atom development diagnostic.

Both LLM stages receive the public PDE, IC/BC, domain, and parameter values.  Stage II is
reached only for an atom-free Ansatz that passes the host parser and independent symbolic
reduction replay.  Gold and reference solutions remain unavailable to public generation.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import sympy as sp
from sympy.core.function import AppliedUndef

from .ansatz_replay import compile_neutral_ansatz, replay_ansatz_candidates
from .coefficient_fitting import BFGSConfigV1, fit_expression_coefficients_bfgs
from .condition_aware_solution import _evaluate_terms, _field_residual_terms
from .llm_ansatz import (
    LLMAnsatzBatchCompletionV1,
    LLMAnsatzCandidateV1,
    LLMAnsatzError,
    common_condition_protocol_instructions,
    propose_llm_ansatz_batch,
    public_problem_payload,
    validate_llm_ansatz_candidate,
)
from .llm_provider import LLMBatchCompletionV1, LLMBatchProviderV1, LLMResponseError
from .safe_symbolic import (
    SUPPORTED_ELEMENTARY_FUNCTIONS,
    SafeSymbolicError,
    parse_symbolic_expression,
)

STAGE_I_CANDIDATE_COUNT = 3
STAGE_II_CANDIDATE_COUNT = 8
MAX_REGENERATION_ATTEMPTS = 10
MAX_STAGE_II_REQUESTS = 1_440
MAX_FITTED_COEFFICIENTS = 24
MAX_ATOM_POOL_SIZE = 24
MAX_ATOM_COMBINATIONS = 512
MAX_SEMANTIC_FITS_PER_ANSATZ = 16
MAX_MULTISLOT_SEMANTIC_FITS_PER_ANSATZ = 8
MAX_SEMANTIC_FITS_PER_CASE = 48
MAX_SEMANTIC_REGENERATION_ROUNDS = 2
SEMANTIC_ACCEPTANCE_LOSS = 1.0e-8
ATOM_TYPES = ("PRIMITIVE", "ENVELOPE", "COORDINATE", "TERM")
_PARAMETER = re.compile(r"p([1-9][0-9]*)")
_INSTANTIATED_PARAMETER = re.compile(r"z[1-9][0-9]*_([1-9][0-9]*)")
_RATIONALE = re.compile(r"[A-Z][A-Z0-9_]*")

ATOM_COEFFICIENT_POLICY = "fixed_or_free_v7"

STAGE_II_SYSTEM_PROMPT = f"""You are a mathematical Atom generator, not an Ansatz generator.

An Atom is an executable expression that replaces exactly one arbitrary profile function
inside the supplied host-certified atom-free Ansatz. The request is bound to one Ansatz,
one slot marker, the original profile arguments, and the disclosed public problem.
Do not change the Ansatz, fill another slot, or output a complete field solution.

Use only the supplied formal slot arguments ARG1, ARG2, ..., real numeric literals, pi,
and consecutive free parameters p1, p2, ... . Public initial/boundary conditions, domain
bounds, and parameter values are supplied so that Atom families can be ranked against the
same disclosed problem information. Reference solutions, benchmark identities, evaluator
labels, and precomputed final expressions remain absent. The same frozen, development-informed
format-only parameterization examples are used for every case, never selected by case identity.

Coefficient policy: {ATOM_COEFFICIENT_POLICY}. Fixed, partially parameterized, and fully
parameterized Atoms are all eligible. Inspect canonical_template itself: use
free_parameters=[] only when it contains no p1, p2, ... symbols; otherwise declare
exactly the consecutive p1, p2, ... symbols used anywhere in the expression.
List parameters in ascending p-number and roles in slot order; equivalent declaration
permutations are normalized after exact membership and uniqueness checks.
Do not introduce free coefficients merely to make a fixed expression pass the schema.
A profile constant in ARG has variable_roles=[] but can still contain free parameters.
Format-only contrast: canonical_template="p1" needs variable_roles=[] and
free_parameters=["p1"]; canonical_template="1" needs both arrays []. Check these
two symbol sets independently; neither declaration may omit or invent a symbol.
Before returning, check every proposal has the six required fields, a permitted atom_type,
exact declared ARG roles and p-symbols, and consecutive parameter names. This is
proposal eligibility, not a claim that the Atom satisfies the PDE: every assembled
field is validated later.

Format-only synthetic example: H(r) -> p1*ARG1+p2 is a parameterized affine profile.
Fixed coefficients are also allowed. Do not prefer an expression solely because it has
more free parameters. The host must not repair or add coefficients to your proposal.

The complete supported elementary-function allowlist is:
{", ".join(SUPPORTED_ELEMENTARY_FUNCTIONS)}.
Do not call any lowercase function outside this list. Integer powers and arithmetic products
or compositions of supported functions are allowed with fixed or free coefficients.

Return strict JSON only. Rank exactly eight structurally distinct Atom expressions by
compatibility derived from the supplied operator, conditions, domain, and slot contract.
"""


class StagedExperimentError(RuntimeError):
    """A fail-closed staged-generation or artifact error."""


class AtomAssemblyRejection(StagedExperimentError):
    """Expected single-combination domain or completion-representation failure."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _assembly_rejection_recorder(
    records: list[Mapping[str, Any]], phase: str, *, round_number: int = 0
) -> Callable[[Mapping[str, Any]], None]:
    def record(item: Mapping[str, Any]) -> None:
        records.append({**item, "assembly_phase": phase, "semantic_round": round_number})

    return record


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


def atom_response_format(argument_roles: Sequence[str]) -> dict[str, Any]:
    roles = tuple(argument_roles)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "pde_atom_slot_batch_v7",
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
                            "required": [
                                "canonical_template",
                                "atom_type",
                                "variable_roles",
                                "free_parameters",
                                "rationale_codes",
                                "reason",
                            ],
                            "properties": {
                                "canonical_template": {"type": "string"},
                                "atom_type": {"type": "string", "enum": list(ATOM_TYPES)},
                                "variable_roles": {
                                    "type": "array",
                                    "minItems": 0,
                                    "maxItems": len(roles),
                                    "uniqueItems": True,
                                    "items": {"type": "string", "enum": list(roles)},
                                },
                                "free_parameters": {
                                    "type": "array",
                                    "uniqueItems": True,
                                    "items": {"type": "string", "pattern": "^p[1-9][0-9]*$"},
                                },
                                "rationale_codes": {
                                    "type": "array",
                                    "minItems": 1,
                                    "uniqueItems": True,
                                    "items": {"type": "string", "pattern": "^[A-Z][A-Z0-9_]*$"},
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
    feedback_mode: str = "validation",
    compositional_structure: bool = False,
    profile_occurrences: Sequence[str] = (),
) -> tuple[dict[str, str], ...]:
    """Render one condition-aware Atom request bound to one certified Ansatz slot."""

    payload = public_problem_payload(row)
    roles = tuple(f"ARG{index}" for index in range(1, len(profile_arguments) + 1))
    role_bindings = tuple(
        f"{role} -> {argument}" for role, argument in zip(roles, profile_arguments, strict=True)
    )
    feedback = tuple(validation_feedback)
    if any(not isinstance(item, str) or not item.strip() or len(item) > 4_096 for item in feedback):
        raise ValueError("Stage II host validation feedback is invalid")
    if feedback_mode not in {"validation", "candidate_revision"}:
        raise ValueError("unsupported Stage II feedback mode")
    if feedback_mode == "candidate_revision" and not feedback:
        raise ValueError("candidate revision requires prior-candidate feedback")
    repair_block = ""
    if feedback:
        if feedback_mode == "candidate_revision":
            repair_block = f"""

Previous model-generated candidate and public evaluator observations
{_bullets(feedback)}

Generate a new batch for this same certified Ansatz and declared slot only. Prefer valid
structures different from the listed prior Atom pool when possible, but a repeated prior
Atom is still valid and eight novel Atoms are not required. The observations describe the
whole fitted candidate; they do not prove that this slot caused a residual. Do not infer
undisclosed condition targets, benchmark identity, or reference solutions, and do not
return a full field or fitted numeric coefficients.
"""
        else:
            repair_block = f"""

Previous response: host validation or semantic feedback
{_bullets(feedback)}

Correct the reported host-contract or residual failure class. Generate a new batch, do not
repeat an already accepted Atom, do not change the certified Ansatz, and do not invent
undisclosed condition targets, benchmark identity, or reference solutions. Host feedback is
not a solution-expression hint.
"""
    parameter_values = tuple(
        f"{name} = {json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"
        for name, value in payload["parameter_values"].items()
    )
    domain = json.dumps(
        payload["domain"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    condition_protocol = common_condition_protocol_instructions(payload)
    identity_contract = "replace only the declared profile call;"
    current_label = "Current Ansatz with exactly one open slot:"
    occurrence_block = ""
    if compositional_structure:
        identity_contract = (
            "replace ALL occurrences of the declared arbitrary function using the SAME Atom "
            "template and SAME free coefficients, at each occurrence's own argument map; "
            "nested profiles have independent slots; preserve structural derivatives/transforms;"
        )
        current_label = "Current Ansatz with one open function-identity slot (all its occurrences):"
        occurrence_block = "\nAll profile occurrences in the certified structure:\n" + _bullets(
            tuple(profile_occurrences) or (profile_call,)
        )
    user = f"""Generate exactly eight ranked Atom candidates for
this single open slot.

Dependent variables:
{_bullets(payload["dependent_variables"])}

Independent variables:
{_bullets(payload["independent_variables"])}

Public parameter names:
{_bullets(payload["parameters"])}

Public parameter values:
{_bullets(parameter_values)}

PDE operator:
{_bullets(payload["pde"])}{condition_protocol}

Initial conditions:
{_bullets(payload["initial_conditions"])}

Boundary conditions:
{_bullets(payload["boundary_conditions"])}

Domain:
{domain}

Host-certified atom-free Ansatz:
{certified_ansatz}

{current_label}
{current_ansatz}

Open slot ID:
{slot_id}

Original arbitrary profile call:
{profile_call}{occurrence_block}

Formal slot arguments:
{_bullets(role_bindings)}

Slot contract:
- {identity_contract}
- declare exactly the ARG roles used, or [] for a constant; never use problem coordinates directly;
- use only these supported elementary functions and p1, p2, ... coefficients:
  {", ".join(SUPPORTED_ELEMENTARY_FUNCTIONS)};
- fixed, partially parameterized, and fully parameterized expressions are all allowed;
- use free_parameters=[] for fixed expressions; otherwise list exactly their p1..pN symbols;
- rationale_codes must be nonempty uppercase identifiers matching [A-Z][A-Z0-9_]*;
- Before returning, silently count the exact p1..pN symbols in each template and list
  all and only those symbols in free_parameters; remove unused declarations;
- do not add unnecessary free scales or amplitudes solely to satisfy a format requirement;
- integer-powered, product, and composed profiles remain valid within the supported grammar;
- return an Atom expression, never a field equation or full solution;
- return exactly eight structurally distinct proposals spanning multiple function families.
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
    occurrences: tuple[AppliedUndef, ...] = ()
    dependency_profiles: tuple[str, ...] = ()
    function_identity: bool = False
    occurrence_contexts: tuple[tuple[str, str, tuple[int, ...]], ...] = ()

    @property
    def calls(self) -> tuple[AppliedUndef, ...]:
        return self.occurrences or (self.call,)

    def public_mapping(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "profile_call": sp.sstr(self.call),
            "profile_name": str(self.call.func),
            "profile_arguments": [sp.sstr(item) for item in self.call.args],
            "argument_roles": list(self.argument_roles),
            "current_ansatz": self.current_ansatz,
            "profile_occurrences": [sp.sstr(call) for call in self.calls],
            "dependency_profiles": list(self.dependency_profiles),
            "dependency_semantics": "informational function names; finite occurrence AST controls order",
            "occurrence_contexts": [
                {"field": field, "profile_call": call, "compiled_AST_argument_path": list(path)}
                for field, call, path in self.occurrence_contexts
            ],
            "coefficient_scope": (
                "shared across all occurrences of this function"
                if self.function_identity
                else "legacy per-call slot"
            ),
            "function_identity": self.function_identity,
        }


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


def _render_ansatz_with_slot(
    ast: Any, calls: Sequence[AppliedUndef], slot_id: str, *, function_identity: bool = False
) -> str:
    if not function_identity:
        marker = sp.Symbol(f"OPEN_{slot_id.upper()}")
        return ";".join(
            f"{field}={sp.sstr(expression.xreplace({calls[0]: marker}))}"
            for field, expression in ast.field_expressions
        )
    marker = sp.Function(f"OPEN_{slot_id.upper()}")
    function = calls[0].func

    def mark(expression: sp.Expr) -> sp.Expr:
        arguments = tuple(mark(arg) for arg in expression.args)
        if isinstance(expression, AppliedUndef) and expression.func == function:
            return marker(*arguments)
        return expression.func(*arguments) if arguments else expression

    rendered = ";".join(
        f"{field}={sp.sstr(mark(expression))}" for field, expression in ast.field_expressions
    )
    if str(marker) not in rendered:
        raise StagedExperimentError("slot marker was not inserted")
    return rendered


def extract_runtime_slots(
    row: Mapping[str, Any],
    ansatz: str,
    reason: str,
    *,
    compositional_structure: bool = False,
    structural_carriers: bool = False,
) -> tuple[Any, tuple[RuntimeSlot, ...]]:
    kwargs = {"compositional_structure": True} if compositional_structure else {}
    if structural_carriers:
        kwargs["structural_carriers"] = True
    ast = compile_neutral_ansatz(row, {"ansatz": ansatz, "reason": reason}, **kwargs)
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
    if not compositional_structure:
        slots = []
        for index, call in enumerate(calls, 1):
            if not 1 <= len(call.args) <= 4:
                raise StagedExperimentError("profile arity is outside the Atom slot contract")
            if any(argument.atoms(AppliedUndef) for argument in call.args):
                raise StagedExperimentError("nested profile arguments are unsupported")
            slot_id = f"slot_{index:03d}"
            roles = tuple(f"ARG{position}" for position in range(1, len(call.args) + 1))
            slots.append(
                RuntimeSlot(slot_id, call, roles, _render_ansatz_with_slot(ast, (call,), slot_id))
            )
        return ast, tuple(slots)
    grouped: dict[str, list[AppliedUndef]] = {}
    for call in calls:
        grouped.setdefault(str(call.func), []).append(call)
    dependencies: dict[str, set[str]] = {}
    for name, occurrences in grouped.items():
        arities = {len(call.args) for call in occurrences}
        if len(arities) != 1 or not 1 <= next(iter(arities)) <= 4:
            raise StagedExperimentError("profile arity is outside or conflicts with Atom contract")
        dependencies[name] = {
            str(nested.func)
            for call in occurrences
            for argument in call.args
            for nested in argument.atoms(AppliedUndef)
        }
    # Function-name dependencies may contain cycles in a finite composition such as
    # F(G(x)) + G(F(t)); these are not recursively defined Atom templates. Actual
    # invocation nodes form a finite expression AST and are instantiated bottom-up.
    contexts: dict[str, list[tuple[str, str, tuple[int, ...]]]] = {name: [] for name in grouped}

    def record_context(field: str, expression: sp.Expr, path: tuple[int, ...]) -> None:
        if isinstance(expression, AppliedUndef):
            contexts[str(expression.func)].append((field, sp.sstr(expression), path))
        for index, argument in enumerate(expression.args):
            record_context(field, argument, (*path, index))

    for field, expression in ast.field_expressions:
        record_context(field, expression, ())
    slots = []
    for index, (name, occurrences) in enumerate(grouped.items(), 1):
        call = occurrences[0]
        slot_id = f"slot_{index:03d}"
        roles = tuple(f"ARG{position}" for position in range(1, len(call.args) + 1))
        slots.append(
            RuntimeSlot(
                slot_id,
                call,
                roles,
                _render_ansatz_with_slot(ast, occurrences, slot_id, function_identity=True),
                tuple(occurrences),
                tuple(sorted(dependencies[name])),
                True,
                tuple(contexts[name]),
            )
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
    """Describe free coefficients, without requiring artificial amplitudes or scales.

    Symbol/role/parameter census is checked by the caller. This certificate is a
    representation check, not a symbolic or numerical proof that a slot solves a PDE.
    """

    parameters = tuple(sp.Symbol(name) for name in parameter_names)
    roles = tuple(role_symbols)
    amplitude_parameters = [
        str(parameter)
        for parameter in parameters
        if any(sp.diff(expression, parameter).has(role) for role in roles)
        and sp.diff(expression, parameter, 2) == 0
    ]
    nonlinear_scale_parameters: set[str] = set()
    for function in expression.atoms(sp.Function):
        for role in roles:
            if function.has(role):
                derivatives = tuple(sp.diff(argument, role) for argument in function.args)
                nonlinear_scale_parameters.update(
                    str(parameter)
                    for parameter in parameters
                    if any(derivative.has(parameter) for derivative in derivatives)
                )
    return {
        "coefficient_policy": ATOM_COEFFICIENT_POLICY,
        "coefficient_mode": "free" if parameters else "fixed",
        "amplitude_parameters": amplitude_parameters,
        "nonlinear_argument_scale_parameters": sorted(nonlinear_scale_parameters),
        "assembled_pde_validation_required": True,
    }


def validate_atom_response(
    response: Mapping[str, Any],
    slot: RuntimeSlot,
    *,
    expected_count: int = STAGE_II_CANDIDATE_COUNT,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Return host-valid proposals and explicit rejected proposal records."""

    if expected_count not in (1, STAGE_II_CANDIDATE_COUNT):
        raise ValueError("Atom response expected_count must be one or the original batch size")
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
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, raw in enumerate(proposals, 1):
        try:
            if not isinstance(raw, Mapping):
                raise StagedExperimentError("Atom proposal is not an object")
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
            reason = raw["reason"]
            atom_type = raw["atom_type"]
            roles = raw["variable_roles"]
            parameters = raw["free_parameters"]
            rationale_codes = raw["rationale_codes"]
            if not isinstance(template, str) or not template.strip() or len(template) > 4096:
                raise StagedExperimentError("Atom template text is invalid")
            if not isinstance(reason, str) or not reason.strip() or len(reason) > 1024:
                raise StagedExperimentError("Atom reason is invalid")
            if atom_type not in ATOM_TYPES:
                raise StagedExperimentError("Atom type is invalid")
            if (
                not isinstance(roles, Sequence)
                or isinstance(roles, (str, bytes, Mapping))
                or len(roles) != len(set(roles))
                or not set(roles).issubset(slot.argument_roles)
            ):
                raise StagedExperimentError("Atom variable roles violate the slot contract")
            if (
                not isinstance(parameters, Sequence)
                or isinstance(parameters, (str, bytes, Mapping))
                or len(parameters) != len(set(parameters))
                or any(not isinstance(item, str) for item in parameters)
            ):
                raise StagedExperimentError("Atom free parameter declaration is invalid")
            if (
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
            names["E"] = sp.E
            expression = parse_symbolic_expression(template, names=names)
            if expression.atoms(AppliedUndef):
                raise StagedExperimentError("Atom contains an unknown profile function")
            parsed_parameters = _parameter_names(expression)
            if set(parameters) != set(parsed_parameters):
                raise StagedExperimentError("Atom free parameters differ from the expression")
            used_roles = tuple(
                role for role in slot.argument_roles if names[role] in expression.free_symbols
            )
            if set(roles) != set(used_roles):
                raise StagedExperimentError("Atom variable_roles differ from the expression")
            allowed = {names[role] for role in used_roles} | {
                sp.Symbol(name) for name in parsed_parameters
            }
            unresolved = expression.free_symbols - allowed
            if unresolved:
                raise StagedExperimentError("Atom contains symbols outside the slot contract")
            parameterization = _parameterization_certificate(
                expression,
                tuple(names[role] for role in used_roles),
                parsed_parameters,
            )
            canonical = sp.sstr(expression)
            if canonical in seen:
                raise StagedExperimentError("Atom duplicates an earlier template")
            seen.add(canonical)
            accepted.append(
                {
                    "rank": rank,
                    "canonical_template": canonical,
                    "atom_type": atom_type,
                    "variable_roles": list(used_roles),
                    "free_parameters": list(parsed_parameters),
                    "rationale_codes": list(rationale_codes),
                    "reason": reason.strip(),
                    "host_validation_passed": True,
                    "parameterization_validation_passed": True,
                    "parameterization_certificate": parameterization,
                }
            )
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


def _atom_formal_template(
    slot: RuntimeSlot, candidate: Mapping[str, Any], slot_index: int
) -> tuple[sp.Expr, tuple[sp.Dummy, ...]]:
    roles = {name: sp.Symbol(name, real=True) for name in slot.argument_roles}
    roles.update(pi=sp.pi, E=sp.E)
    expression = parse_symbolic_expression(str(candidate["canonical_template"]), names=roles)
    if expression.atoms(AppliedUndef, sp.Derivative, sp.Subs):
        raise StagedExperimentError(
            "Atom template must not contain undefined profiles or derivatives"
        )
    parameters = tuple(candidate["free_parameters"])
    actual = _parameter_names(expression)
    if parameters != actual:
        raise StagedExperimentError("Atom coefficient declaration does not match template")
    dummies = tuple(sp.Dummy(role, real=True) for role in slot.argument_roles)
    substitutions = {
        roles[role]: dummy for role, dummy in zip(slot.argument_roles, dummies, strict=True)
    }
    substitutions.update(
        {
            symbol: sp.Symbol(f"z{slot_index}_{position}", real=True)
            for position, name in enumerate(parameters, 1)
            for symbol in expression.free_symbols
            if str(symbol) == name
        }
    )
    return expression.xreplace(substitutions), dummies


def _instantiate_atom(slot: RuntimeSlot, candidate: Mapping[str, Any], slot_index: int) -> sp.Expr:
    template, roles = _atom_formal_template(slot, candidate, slot_index)
    return template.xreplace(dict(zip(roles, slot.call.args, strict=True)))


def _complete_profile_fields(
    ast: Any, slots: Sequence[RuntimeSlot], combination: Sequence[Mapping[str, Any]]
) -> dict[str, sp.Expr]:
    functions = {str(slot.call.func): index for index, slot in enumerate(slots)}
    required = {
        str(call.func)
        for _field, expression in ast.field_expressions
        for call in expression.atoms(AppliedUndef)
    }
    if len(functions) != len(slots) or set(functions) != required:
        raise StagedExperimentError("function slot coverage does not match certified fields")
    reserved_names = {
        str(symbol)
        for _field, expression in ast.field_expressions
        for symbol in expression.free_symbols
    }
    templates = []
    scope = 1
    for index, slot in enumerate(slots):
        parameters = combination[index]["free_parameters"]
        while any(
            f"z{scope}_{position}" in reserved_names for position in range(1, len(parameters) + 1)
        ):
            scope += 1
        templates.append(_atom_formal_template(slot, combination[index], scope))
        scope += 1

    def complete(expression: sp.Expr) -> sp.Expr:
        arguments = tuple(complete(argument) for argument in expression.args)
        if isinstance(expression, AppliedUndef):
            index = functions[str(expression.func)]
            template, roles = templates[index]
            if len(arguments) != len(roles):
                raise StagedExperimentError("profile arity differs from function slot")
            completed = template.xreplace(dict(zip(roles, arguments, strict=True)))
        else:
            completed = expression.func(*arguments) if arguments else expression
        # Check before differentiating: d/dx(log(0)) must not simplify invalid
        # structural input to the apparently finite constant zero.
        if completed.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
            raise AtomAssemblyRejection(
                "nonfinite structural completion", reason_code="NONFINITE_STRUCTURAL_COMPLETION"
            )
        if completed.is_real is False:
            raise AtomAssemblyRejection(
                "structural intermediate is provably nonreal for this real-valued method",
                reason_code="NONREAL_STRUCTURAL_COMPLETION",
            )
        return completed

    fields = {field: complete(expression).doit() for field, expression in ast.field_expressions}
    for expression in fields.values():
        if expression.atoms(AppliedUndef, sp.Derivative, sp.Subs, sp.Dummy):
            raise AtomAssemblyRejection(
                "assembled candidate retains unresolved structural terms",
                reason_code="UNRESOLVED_STRUCTURAL_COMPLETION",
            )
        if expression.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
            raise AtomAssemblyRejection(
                "assembled candidate contains nonfinite symbolic terms",
                reason_code="NONFINITE_STRUCTURAL_COMPLETION",
            )
        if expression.is_real is False:
            raise AtomAssemblyRejection(
                "completed expression is provably nonreal for this real-valued method",
                reason_code="NONREAL_STRUCTURAL_COMPLETION",
            )
    return fields


def assemble_candidate_beam(
    ast: Any,
    slots: Sequence[RuntimeSlot],
    pools: Sequence[Sequence[Mapping[str, Any]]],
    *,
    ansatz_rank: int,
    max_combinations: int | None = None,
    rejection_recorder: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Build a bounded Cartesian beam; ranks from different slots are not aligned."""

    if len(slots) != len(pools) or any(not pool for pool in pools):
        return ()
    if max_combinations is None:
        max_combinations = MAX_ATOM_COMBINATIONS
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
    anchors: list[tuple[int, ...]] = []
    for slot_index, pool in enumerate(ordered_pools):
        for atom_index in range(1, len(pool)):
            anchor_indices = list(zero)
            anchor_indices[slot_index] = atom_index
            anchors.append(tuple(anchor_indices))
    anchors.sort(key=index_key)
    for anchor in anchors:
        if len(chosen_indices) >= max_combinations:
            break
        chosen_indices.add(anchor)

    heap: list[tuple[tuple[Any, ...], tuple[int, ...]]] = [(index_key(zero), zero)]
    visited = {zero}
    while heap and len(chosen_indices) < max_combinations:
        _key, current_indices = heapq.heappop(heap)
        chosen_indices.add(current_indices)
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

    combination_indices = sorted(chosen_indices, key=index_key)[:max_combinations]
    outputs = []
    for combination_rank, selected_indices in enumerate(combination_indices, 1):
        combination = tuple(
            ordered_pools[slot_index][atom_index]
            for slot_index, atom_index in enumerate(selected_indices)
        )
        if all(slot.function_identity for slot in slots):
            try:
                fields = _complete_profile_fields(ast, slots, combination)
            except AtomAssemblyRejection as error:
                if rejection_recorder is not None:
                    rejection_recorder(
                        {
                            "schema_version": "CompositionalAtomAssemblyRejectionV2",
                            "ansatz_rank": ansatz_rank,
                            "atom_combination_rank": combination_rank,
                            "slot_atom_ranks": {
                                slot.slot_id: int(combination[index]["rank"])
                                for index, slot in enumerate(slots)
                            },
                            "slot_atom_templates": {
                                slot.slot_id: str(combination[index]["canonical_template"])
                                for index, slot in enumerate(slots)
                            },
                            "reason_code": error.reason_code,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                continue
        elif any(slot.function_identity for slot in slots):
            raise StagedExperimentError("mixed legacy/function-identity slots are unsupported")
        else:
            replacements = {
                slot.call: _instantiate_atom(slot, combination[index], index + 1)
                for index, slot in enumerate(slots)
            }
            fields = {
                field: expression.xreplace(replacements)
                for field, expression in ast.field_expressions
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
                "assembly_policy": (
                    "cartesian rank-sum function-identity compositional beam v2"
                    if all(slot.function_identity for slot in slots)
                    else "cartesian rank-sum beam"
                ),
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


def _real_expression(source: str, row: Mapping[str, Any]) -> tuple[sp.Expr, tuple[sp.Symbol, ...]]:
    coordinates = {name: sp.Symbol(str(name), real=True) for name in row["independent_variables"]}
    names: dict[str, object] = {**coordinates, "pi": sp.pi, "E": sp.E}
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


def fit_scalar_candidates(
    row: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    validation_spec: Mapping[str, Any],
    *,
    coefficient_backend: str = "cpu_bfgs",
    cuda_device_index: int = 0,
) -> tuple[tuple[dict[str, Any], ...], Mapping[str, Any] | None]:
    if coefficient_backend not in {"cpu_bfgs", "cuda_lbfgs"}:
        raise ValueError(f"unsupported coefficient backend: {coefficient_backend}")
    cuda_runtime: Mapping[str, Any] | None = None
    if coefficient_backend == "cuda_lbfgs":
        from care_pde.fitting.cuda_lbfgs import cuda_runtime_info

        cuda_runtime = cuda_runtime_info(device_index=cuda_device_index)

    fitted = []
    for generation_rank, candidate in enumerate(candidates, 1):
        record = {**candidate, "generation_rank": generation_rank}
        try:
            fields = candidate["field_expressions"]
            if not isinstance(fields, Mapping) or set(fields) != set(row["dependent_variables"]):
                raise StagedExperimentError("assembled field census differs")
            expressions: dict[str, sp.Expr] = {}
            coefficient_set: set[sp.Symbol] = set()
            for field in map(str, row["dependent_variables"]):
                expression, field_coefficients = _real_expression(str(fields[field]), row)
                expressions[field] = expression
                coefficient_set.update(field_coefficients)
            coefficients = tuple(sorted(coefficient_set, key=str))
            if len(coefficients) > MAX_FITTED_COEFFICIENTS:
                raise StagedExperimentError("candidate exceeds the frozen coefficient budget")
            slot_ranks = candidate.get("slot_atom_ranks", {})
            multislot = isinstance(slot_ranks, Mapping) and len(slot_ranks) > 1
            config = BFGSConfigV1(
                max_iterations=120 if multislot else 300,
                gradient_tolerance=1.0e-8,
                restarts=4 if multislot else 6,
                random_seed=20260904,
                restart_scale=1.0,
                acceptance_loss=1.0e-8,
            )
            training = _field_residual_terms(
                expressions, coefficients, validation_spec, split="training"
            )
            guess_expression = sp.Add(*expressions.values(), evaluate=False)
            if coefficients:
                fit: Any
                if coefficient_backend == "cuda_lbfgs":
                    from care_pde.fitting.cuda_lbfgs import (
                        fit_expression_coefficients_cuda_lbfgs,
                    )

                    fit = fit_expression_coefficients_cuda_lbfgs(
                        next(iter(expressions.values())),
                        coefficients,
                        training,
                        config=config,
                        initial_guesses=coefficient_initial_guesses(guess_expression, coefficients),
                        device_index=cuda_device_index,
                    )
                else:
                    fit = fit_expression_coefficients_bfgs(
                        next(iter(expressions.values())),
                        coefficients,
                        training,
                        config=config,
                        initial_guesses=coefficient_initial_guesses(guess_expression, coefficients),
                    )
                fit_mapping: Mapping[str, Any] = fit.to_mapping()
                assignments = {
                    sp.Symbol(name, real=True): sp.Float(value)
                    for name, value in fit.coefficient_assignments
                }
                fitted_expressions = {
                    field: expression.xreplace(assignments)
                    for field, expression in expressions.items()
                }
            else:
                terms: Mapping[str, float] | Sequence[tuple[str, float]]
                if coefficient_backend == "cuda_lbfgs":
                    from care_pde.fitting.cuda_lbfgs import evaluate_residual_terms_cuda

                    loss, terms = evaluate_residual_terms_cuda(
                        training, device_index=cuda_device_index
                    )
                else:
                    loss, terms = _evaluate_terms(training, guess_expression)
                fitted_expressions = expressions
                fit_mapping = {
                    "schema_version": (
                        "NoCoefficientCudaFitV1"
                        if coefficient_backend == "cuda_lbfgs"
                        else "NoCoefficientFitV1"
                    ),
                    "accepted": loss <= config.acceptance_loss,
                    "optimizer_converged": True,
                    "expression": sp.sstr(next(iter(expressions.values()))),
                    "coefficient_assignments": {},
                    "final_loss": loss,
                    "term_losses": dict(terms),
                    "restart_count": 0,
                }
                if coefficient_backend == "cuda_lbfgs":
                    fit_mapping = {
                        **fit_mapping,
                        "coefficient_backend": coefficient_backend,
                        "cuda_device_index": cuda_device_index,
                        "cuda_runtime": dict(cuda_runtime or {}),
                    }
            validation = _field_residual_terms(
                fitted_expressions, (), validation_spec, split="validation"
            )
            validation_terms: Mapping[str, float] | Sequence[tuple[str, float]]
            if coefficient_backend == "cuda_lbfgs":
                from care_pde.fitting.cuda_lbfgs import evaluate_residual_terms_cuda

                validation_loss, validation_terms = evaluate_residual_terms_cuda(
                    validation, device_index=cuda_device_index
                )
            else:
                validation_loss, validation_terms = _evaluate_terms(
                    validation, sp.Add(*fitted_expressions.values(), evaluate=False)
                )
            update = {
                "fit": dict(fit_mapping),
                "fitted_field_expressions": {
                    field: sp.sstr(expression) for field, expression in fitted_expressions.items()
                },
                "validation_loss": validation_loss,
                "validation_term_losses": dict(validation_terms),
            }
            if len(fitted_expressions) == 1:
                update["fitted_expression"] = sp.sstr(next(iter(fitted_expressions.values())))
            record.update(update)
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
    valid.sort(key=lambda item: (float(item["validation_loss"]), int(item["generation_rank"])))
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
    diagnostics = [
        f"SEMANTIC_VALIDATION_FAILED: best weighted validation loss {total_loss:.6g} "
        f"exceeds {tolerance:.6g}"
    ]
    term_losses = selected.get("validation_term_losses", {})
    if isinstance(term_losses, Mapping):
        operator_losses = [
            float(value)
            for name, value in term_losses.items()
            if not str(name).startswith("condition_") and isinstance(value, (int, float))
        ]
        if operator_losses and max(operator_losses) > tolerance:
            diagnostics.append(
                f"PDE_RESIDUAL_HIGH: assembled Atoms leave operator residual "
                f"{max(operator_losses):.6g}; propose a different function family or scaling"
            )
        condition_losses = [
            float(value)
            for name, value in term_losses.items()
            if str(name).startswith("condition_") and isinstance(value, (int, float))
        ]
        if condition_losses and max(condition_losses) > tolerance:
            diagnostics.append(
                "PUBLIC_CONDITION_RESIDUAL_HIGH: at least one public condition remains "
                "incompatible; its target expression is intentionally not disclosed in feedback"
            )
    diagnostics.append(
        "Generate structurally new fixed or parameterized Atoms; do not repeat any retained template"
    )
    return tuple(diagnostics)


def semantic_validate_candidates(
    row: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    validation_spec: Mapping[str, Any],
    *,
    tolerance: float = SEMANTIC_ACCEPTANCE_LOSS,
    coefficient_backend: str = "cpu_bfgs",
    cuda_device_index: int = 0,
) -> tuple[tuple[dict[str, Any], ...], Mapping[str, Any] | None, tuple[str, ...]]:
    """Fit assembled candidates and certify both PDE and public-condition residuals."""

    fitted, _selected = fit_scalar_candidates(
        row,
        candidates,
        validation_spec,
        coefficient_backend=coefficient_backend,
        cuda_device_index=cuda_device_index,
    )
    annotated: list[dict[str, Any]] = []
    for candidate in fitted:
        record = dict(candidate)
        loss = record.get("validation_loss")
        passed = (
            isinstance(loss, (int, float))
            and math.isfinite(float(loss))
            and float(loss) <= tolerance
        )
        record["semantic_validation_passed"] = passed
        record["semantic_validation_tolerance"] = tolerance
        annotated.append(record)
    valid = [record for record in annotated if "validation_loss" in record]
    valid.sort(key=lambda item: (float(item["validation_loss"]), int(item["generation_rank"])))
    selected: Mapping[str, Any] | None = valid[0] if valid else None
    diagnostics = (
        ()
        if selected is not None and bool(selected["semantic_validation_passed"])
        else _semantic_diagnostics(selected, tolerance=tolerance)
    )
    return tuple(annotated), selected, diagnostics


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
                and isinstance(candidate.get("validation_loss"), (int, float))
                and math.isfinite(float(candidate["validation_loss"]))
            ]
            containing.sort(
                key=lambda candidate: (
                    float(candidate["validation_loss"]),
                    int(candidate["generation_rank"]),
                )
            )
            best = containing[0] if containing else None
            best_loss = None if best is None else float(best["validation_loss"])
            rows.append(
                {
                    "rank": atom_rank,
                    "canonical_template": str(atom["canonical_template"]),
                    "evaluated_combination_count": len(containing),
                    "best_validation_loss": best_loss,
                    "best_validation_term_losses": (
                        None if best is None else best.get("validation_term_losses")
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
        retained = "; ".join(str(item["canonical_template"]) for item in accepted)
        feedback.append(_safe_feedback(f"Already accepted; do not repeat: {retained}"))
    feedback.append(
        "Return new proposals that exactly match expression-used ARG roles, consecutive p1..pN "
        "parameters if any (otherwise []), and uppercase rationale "
        f"codes. {STAGE_II_CANDIDATE_COUNT} distinct accepted Atoms are required in the initial "
        "pool."
    )
    return tuple(dict.fromkeys(feedback))


def _semantic_retry_feedback(
    diagnostics: Sequence[str],
    accepted: Sequence[Mapping[str, Any]],
    assessments: Sequence[Mapping[str, Any]] = (),
) -> tuple[str, ...]:
    feedback = [_safe_feedback(item) for item in diagnostics]
    if accepted:
        retained = "; ".join(str(item["canonical_template"]) for item in accepted)
        feedback.append(_safe_feedback(f"Already retained; do not repeat: {retained}"))
    failed = [
        item
        for item in assessments
        if item.get("semantic_status") == "no_passing_combination_observed"
    ]
    if failed:
        summary = "; ".join(
            f"{item['canonical_template']} (best loss {float(item['best_validation_loss']):.6g})"
            for item in failed[:8]
        )
        feedback.append(
            _safe_feedback(
                "No passing assembled combination was observed for these slot Atoms: " + summary
            )
        )
    feedback.append(
        "Return eight new fixed or parameterized Atom proposals from structurally different function "
        "families. Since the retained simple families did not pass, include new integer-powered, "
        "product, or composed variants of supported nonlinear functions rather than merely "
        "renaming or repeating them. Free amplitude and scale coefficients are optional. "
        "Host diagnostics may report PDE or public-condition "
        "failure "
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
    coefficient_backend: str = "cpu_bfgs",
    cuda_device_index: int = 0,
    compositional_structure: bool = False,
    transport_retry_policy: object | None = None,
) -> dict[str, Any]:
    """Run condition-aware host-feedback regeneration without opening offline gold."""

    if coefficient_backend not in {"cpu_bfgs", "cuda_lbfgs"}:
        raise ValueError(f"unsupported coefficient backend: {coefficient_backend}")
    cuda_runtime: Mapping[str, Any] | None = None
    if coefficient_backend == "cuda_lbfgs":
        from care_pde.fitting.cuda_lbfgs import cuda_runtime_info

        # Fail before constructing or invoking any provider when CUDA was requested.
        cuda_runtime = cuda_runtime_info(device_index=cuda_device_index)

    stage_i_requests = 0
    stage_i_provider_errors = 0
    stage_i_syntax_errors = 0
    stage_i_replay_errors = 0
    stage_i_exhausted = 0
    stage_ii_requests = 0
    stage_ii_provider_errors = 0
    stage_ii_host_errors = 0
    stage_ii_exhausted = 0
    stage_ii_semantic_requests = 0
    stage_ii_semantic_provider_errors = 0
    stage_ii_semantic_host_errors = 0
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
            "claim_type": "in_sample_development_diagnostic",
            "stage_i": {},
            "certified_ansatz_runs": [],
            "assembled_candidates": [],
            "fitted_candidates": [],
            "selected": None,
        }

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
                    compositional_structure=compositional_structure,
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
                        compositional_structure=compositional_structure,
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
                compositional_structure=compositional_structure,
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
                stage_i_stop_reason = "three_host_certified_ansatz_candidates"
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
        case_record["stage_i"] = {
            "status": "completed" if certified else "no_certified_ansatz",
            "attempt_count": len(stage_i_attempts),
            "max_attempts": MAX_REGENERATION_ATTEMPTS,
            "stop_reason": stage_i_stop_reason,
            "attempts": stage_i_attempts,
            "certified_count": len(certified),
            "certified_candidates": [
                {
                    "rank": candidate.rank,
                    "ansatz": candidate.template,
                    "reason": candidate.reason,
                    "validation_kind": certificate.validation_kind,
                }
                for candidate, certificate in certified
            ],
        }

        spec = validation_specs.get(case_id)
        assembled_all: list[dict[str, Any]] = []
        fitted_all: list[dict[str, Any]] = []
        passing_selection: Mapping[str, Any] | None = None
        for candidate, certificate in certified:
            ansatz_record: dict[str, Any] = {
                "ansatz_rank": candidate.rank,
                "ansatz": candidate.template,
                "symbolic_validation_kind": certificate.validation_kind,
                "slots": [],
                "status": "pending",
            }
            try:
                ast, slots = extract_runtime_slots(
                    row,
                    candidate.template,
                    candidate.reason,
                    compositional_structure=compositional_structure,
                )
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
                    messages = render_atom_messages(
                        row,
                        certified_ansatz=candidate.template,
                        slot_id=slot.slot_id,
                        current_ansatz=slot.current_ansatz,
                        profile_call=sp.sstr(slot.call),
                        profile_arguments=tuple(sp.sstr(item) for item in slot.call.args),
                        validation_feedback=slot_feedback,
                        compositional_structure=compositional_structure,
                        profile_occurrences=tuple(sp.sstr(call) for call in slot.calls),
                    )
                    atom_recorder = RecordingProvider(provider_factory(4_096))
                    attempt_record = {
                        "attempt": attempt_index,
                        "host_feedback_sent": list(slot_feedback),
                        "rendered_messages_sha256": canonical_sha256(messages),
                    }
                    try:
                        completion = atom_recorder.complete(
                            tokens=(slot.slot_id,),
                            messages=messages,
                            response_format=atom_response_format(slot.argument_roles),
                        )
                    except (LLMResponseError, TypeError, ValueError) as exc:
                        stage_ii_provider_errors += 1
                        attempt_record.update(
                            status="provider_error", error=f"{type(exc).__name__}: {exc}"
                        )
                        slot_attempts.append(attempt_record)
                        continue
                    prompt_tokens_ii += completion.prompt_tokens
                    completion_tokens_ii += completion.completion_tokens
                    try:
                        accepted, rejected = validate_atom_response(completion.response, slot)
                    except (StagedExperimentError, TypeError, ValueError) as exc:
                        stage_ii_host_errors += 1
                        error = f"{type(exc).__name__}: {exc}"
                        attempt_record.update(
                            status="host_response_error",
                            raw_response=completion.response,
                            provider_evidence=_provider_evidence(completion),
                            error=error,
                        )
                        slot_attempts.append(attempt_record)
                        slot_feedback = (_safe_feedback(f"Host response error: {error}"),)
                        continue
                    stage_ii_host_errors += len(rejected)
                    duplicate_templates: list[str] = []
                    for item in accepted:
                        key = str(item["canonical_template"])
                        if key in accepted_by_template:
                            duplicate_templates.append(key)
                            continue
                        if len(accepted_by_template) < STAGE_II_CANDIDATE_COUNT:
                            ranked_atom = dict(item)
                            ranked_atom["rank"] = len(accepted_by_template) + 1
                            accepted_by_template[key] = ranked_atom
                    attempt_record.update(
                        status="completed",
                        raw_response=completion.response,
                        provider_evidence=_provider_evidence(completion),
                        accepted_candidates=list(accepted),
                        rejected_candidates=list(rejected),
                        duplicate_templates=duplicate_templates,
                        accepted_count_after_attempt=len(accepted_by_template),
                    )
                    slot_attempts.append(attempt_record)
                    if len(accepted_by_template) == STAGE_II_CANDIDATE_COUNT:
                        break
                    slot_feedback = _atom_retry_feedback(
                        rejected,
                        tuple(accepted_by_template.values()),
                        duplicate_templates=duplicate_templates,
                    )
                accepted_pool = tuple(accepted_by_template.values())
                if len(accepted_pool) != STAGE_II_CANDIDATE_COUNT:
                    stage_ii_exhausted += 1
                if not accepted_pool:
                    ansatz_blocked = True
                pools_for_ansatz.append(accepted_pool)
                slot_record = {
                    **slot.public_mapping(),
                    "status": (
                        "completed"
                        if len(accepted_pool) == STAGE_II_CANDIDATE_COUNT
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
                assembly_rejections: list[Mapping[str, Any]] = []
                if compositional_structure:
                    ansatz_record["assembly_rejections"] = assembly_rejections
                assembled = assemble_candidate_beam(
                    ast,
                    slots,
                    pools_for_ansatz,
                    ansatz_rank=candidate.rank,
                    rejection_recorder=_assembly_rejection_recorder(assembly_rejections, "initial")
                    if compositional_structure
                    else None,
                )
                semantic_rounds: list[dict[str, Any]] = []
                if spec is not None:
                    for semantic_round in range(MAX_SEMANTIC_REGENERATION_ROUNDS + 1):
                        semantic_validation_runs += 1
                        semantic_candidates = select_semantic_candidate_beam(
                            assembled,
                            max_candidates=(
                                MAX_SEMANTIC_FITS_PER_ANSATZ
                                if len(slots) == 1
                                else MAX_MULTISLOT_SEMANTIC_FITS_PER_ANSATZ
                            ),
                        )
                        if coefficient_backend == "cuda_lbfgs":
                            fitted_round, selected_round, diagnostics = (
                                semantic_validate_candidates(
                                    row,
                                    semantic_candidates,
                                    spec,
                                    coefficient_backend=coefficient_backend,
                                    cuda_device_index=cuda_device_index,
                                )
                            )
                        else:
                            fitted_round, selected_round, diagnostics = (
                                semantic_validate_candidates(row, semantic_candidates, spec)
                            )
                        fitted_all.extend(fitted_round)
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
                                "fitted_count": len(fitted_round),
                                "best_validation_loss": (
                                    None
                                    if selected_round is None
                                    else selected_round.get("validation_loss")
                                ),
                                "passed": passed,
                                "diagnostics": list(diagnostics),
                                "atom_assessments": {
                                    slot_id: list(items)
                                    for slot_id, items in atom_assessments.items()
                                },
                            }
                        )
                        if passed or semantic_round == MAX_SEMANTIC_REGENERATION_ROUNDS:
                            if passed:
                                passing_selection = selected_round
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
                            stage_ii_semantic_requests += 1
                            messages = render_atom_messages(
                                row,
                                certified_ansatz=candidate.template,
                                slot_id=slot.slot_id,
                                current_ansatz=slot.current_ansatz,
                                profile_call=sp.sstr(slot.call),
                                profile_arguments=tuple(sp.sstr(item) for item in slot.call.args),
                                validation_feedback=semantic_feedback,
                                compositional_structure=compositional_structure,
                                profile_occurrences=tuple(sp.sstr(call) for call in slot.calls),
                            )
                            atom_recorder = RecordingProvider(provider_factory(4_096))
                            semantic_attempt_record: dict[str, Any] = {
                                "attempt": len(slot_attempts) + 1,
                                "phase": "semantic_regeneration",
                                "semantic_round": semantic_round + 1,
                                "host_feedback_sent": list(semantic_feedback),
                                "rendered_messages_sha256": canonical_sha256(messages),
                            }
                            try:
                                completion = atom_recorder.complete(
                                    tokens=(slot.slot_id,),
                                    messages=messages,
                                    response_format=atom_response_format(slot.argument_roles),
                                )
                            except (LLMResponseError, TypeError, ValueError) as exc:
                                stage_ii_provider_errors += 1
                                stage_ii_semantic_provider_errors += 1
                                semantic_attempt_record.update(
                                    status="provider_error",
                                    error=f"{type(exc).__name__}: {exc}",
                                )
                                slot_attempts.append(semantic_attempt_record)
                                continue
                            prompt_tokens_ii += completion.prompt_tokens
                            completion_tokens_ii += completion.completion_tokens
                            try:
                                accepted, rejected = validate_atom_response(
                                    completion.response, slot
                                )
                            except (StagedExperimentError, TypeError, ValueError) as exc:
                                stage_ii_host_errors += 1
                                stage_ii_semantic_host_errors += 1
                                semantic_attempt_record.update(
                                    status="host_response_error",
                                    raw_response=completion.response,
                                    provider_evidence=_provider_evidence(completion),
                                    error=f"{type(exc).__name__}: {exc}",
                                )
                                slot_attempts.append(semantic_attempt_record)
                                continue
                            stage_ii_host_errors += len(rejected)
                            stage_ii_semantic_host_errors += len(rejected)
                            semantic_duplicate_templates: list[str] = []
                            accepted_before = len(accepted_by_template)
                            for item in accepted:
                                key = str(item["canonical_template"])
                                if key in accepted_by_template:
                                    semantic_duplicate_templates.append(key)
                                    continue
                                if len(accepted_by_template) >= MAX_ATOM_POOL_SIZE:
                                    break
                                ranked_atom = dict(item)
                                ranked_atom["rank"] = len(accepted_by_template) + 1
                                accepted_by_template[key] = ranked_atom
                            added = len(accepted_by_template) - accepted_before
                            new_atom_count += added
                            semantic_attempt_record.update(
                                status="completed",
                                raw_response=completion.response,
                                provider_evidence=_provider_evidence(completion),
                                accepted_candidates=list(accepted),
                                rejected_candidates=list(rejected),
                                duplicate_templates=semantic_duplicate_templates,
                                newly_retained_count=added,
                                accepted_count_after_attempt=len(accepted_by_template),
                            )
                            slot_attempts.append(semantic_attempt_record)
                            slot_record["attempt_count"] = len(slot_attempts)
                            slot_record["accepted_candidates"] = list(accepted_by_template.values())
                        if new_atom_count == 0:
                            semantic_rounds[-1]["stop_reason"] = (
                                "semantic_regeneration_added_no_new_atoms"
                            )
                            break
                        pools_for_ansatz = [tuple(state[1].values()) for state in slot_states]
                        assembled = assemble_candidate_beam(
                            ast,
                            slots,
                            pools_for_ansatz,
                            ansatz_rank=candidate.rank,
                            rejection_recorder=_assembly_rejection_recorder(
                                assembly_rejections,
                                "semantic_regeneration",
                                round_number=len(semantic_rounds),
                            )
                            if compositional_structure
                            else None,
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
            if ansatz_record.get("status") == "semantic_validation_passed":
                case_record["ansatz_early_stop_reason"] = (
                    "first host-semantic pass in certified Ansatz rank order"
                )
                break

        case_record["assembled_candidates"] = assembled_all
        if spec is None:
            case_record["status"] = "staged_generation_only_unscorable_public_forcing"
            case_record["optimization_blocker"] = (
                "no executable public-only residual/condition specification; optimization not run"
            )
        elif not assembled_all:
            case_record["status"] = "no_fully_staged_candidate"
        elif passing_selection is not None:
            case_record["selected"] = passing_selection
            case_record["semantic_diagnostics"] = []
            case_record["status"] = "semantic_validation_passed"
        else:
            ranked_assembled = sorted(
                assembled_all,
                key=lambda item: (
                    int(item["atom_combination_rank"]),
                    int(item["ansatz_rank"]),
                ),
            )
            semantic_beam = select_semantic_candidate_beam(
                ranked_assembled,
                max_candidates=MAX_SEMANTIC_FITS_PER_CASE,
            )
            if coefficient_backend == "cuda_lbfgs":
                fitted, selected, diagnostics = semantic_validate_candidates(
                    row,
                    semantic_beam,
                    spec,
                    coefficient_backend=coefficient_backend,
                    cuda_device_index=cuda_device_index,
                )
            else:
                fitted, selected, diagnostics = semantic_validate_candidates(
                    row, semantic_beam, spec
                )
            semantic_validation_runs += 1
            fitted_all.extend(fitted)
            # The final rank beam may omit a candidate already evaluated in an earlier
            # Ansatz or regeneration round. Select over the accumulated observations,
            # retaining generation-rank ties and stable observation order.
            valid_fitted = [
                item
                for item in fitted_all
                if isinstance(item.get("validation_loss"), (int, float))
                and math.isfinite(float(item["validation_loss"]))
            ]
            selected = min(
                valid_fitted,
                key=lambda item: (float(item["validation_loss"]), int(item["generation_rank"])),
                default=None,
            )
            diagnostics = (
                ()
                if selected is not None and selected["semantic_validation_passed"]
                else _semantic_diagnostics(selected, tolerance=SEMANTIC_ACCEPTANCE_LOSS)
            )
            case_record["selected"] = selected
            case_record["semantic_diagnostics"] = list(diagnostics)
            if selected is None:
                case_record["status"] = "no_valid_fit"
            elif selected["semantic_validation_passed"]:
                case_record["status"] = "semantic_validation_passed"
            else:
                case_record["status"] = "semantic_validation_failed"
        case_record["fitted_candidates"] = fitted_all
        cases.append(case_record)
        if case_observer is not None:
            case_observer(case_record)

    result: dict[str, Any] = {
        "schema_version": (
            "StagedConditionAwareAnsatzAtomDiagnosticCudaLBFGSV2"
            if coefficient_backend == "cuda_lbfgs"
            else "StagedConditionAwareAnsatzAtomDiagnosticV7"
        ),
        "claim_type": "in_sample_development_diagnostic",
        "gold_accessed": False,
        "reference_solution_accessed": False,
        "generation_input_policy": (
            "public PDE, IC/BC, domain, parameter values, a fixed target-informed development "
            "parameterization example, host contract errors, and public PDE/condition failure "
            "classes; no gold, reference solution, benchmark identity, or reviewed Ansatz"
        ),
        "condition_use_policy": (
            "public IC/BC targets and domain are sent to both LLM stages and are also used by "
            "host semantic fitting and validation"
        ),
        "provider": {
            "name": "openrouter",
            "model": requested_model,
            "sampling_policy": {"temperature": 0.0, "top_p": 1.0},
            "transport_retry_policy": (
                transport_retry_policy
                if transport_retry_policy is not None
                else "unspecified by caller; provider_factory owns transport retries"
            ),
            "host_feedback_regeneration_policy": {
                "max_attempts_per_problem_or_slot": MAX_REGENERATION_ATTEMPTS,
                "max_semantic_regeneration_rounds": MAX_SEMANTIC_REGENERATION_ROUNDS,
                "feedback": (
                    "previous generated expressions, host rejection reasons, numeric best loss, "
                    "and PDE/public-condition residual failure classes; base prompts already "
                    "contain the disclosed public IC/BC"
                ),
                "structural_diversity_instruction": (
                    "after semantic failure, request new integer-powered, product, or composed "
                    "variants of supported functions without naming a preferred target family"
                ),
                "retain_valid_candidates": True,
                "condition_targets_repeated_in_base_prompt": True,
                "gold_or_reference_in_feedback": False,
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
                "provider_error_count": stage_ii_provider_errors,
                "host_candidate_or_response_error_count": stage_ii_host_errors,
                "max_attempts_exhausted_slot_count": stage_ii_exhausted,
                "semantic_regeneration_request_count": stage_ii_semantic_requests,
                "semantic_regeneration_provider_error_count": (stage_ii_semantic_provider_errors),
                "semantic_regeneration_host_error_count": stage_ii_semantic_host_errors,
                "host_semantic_validation_run_count": semantic_validation_runs,
                "prompt_tokens": prompt_tokens_ii,
                "completion_tokens": completion_tokens_ii,
            },
        },
        "policies": {
            "ansatz_symbolic_validation": {
                "strong": (
                    "single-coordinate, multi-coordinate, generic separation reduction proof, "
                    "or exact operator identity"
                ),
                "weak": (
                    "disjoint separation or proper lower-dimensional slot closure, plus "
                    "coupled all-field slot closure under public-PDE symbolic substitution; "
                    "not a solution proof; explicitly requires final Atom validation"
                ),
            },
            "stage_i_candidates_per_response": STAGE_I_CANDIDATE_COUNT,
            "stage_i_certified_target_per_problem": STAGE_I_CANDIDATE_COUNT,
            "stage_ii_candidates_per_response": STAGE_II_CANDIDATE_COUNT,
            "stage_ii_accepted_target_per_slot": STAGE_II_CANDIDATE_COUNT,
            "max_retained_atoms_per_slot": MAX_ATOM_POOL_SIZE,
            "semantic_regeneration_rounds": MAX_SEMANTIC_REGENERATION_ROUNDS,
            "stage_ii_request_budget": MAX_STAGE_II_REQUESTS,
            "parameterization": (
                "fixed_or_free_v7: fixed or partial/full free coefficients; exact symbol census; "
                "no host coefficient injection; assembled PDE/condition validation required"
            ),
            "supported_elementary_functions": list(SUPPORTED_ELEMENTARY_FUNCTIONS),
            "semantic_regeneration_structural_diversity": (
                "integer-powered, product, and composed variants of supported functions"
            ),
            "assembly": "bounded Cartesian rank-sum beam across slot pools",
            "semantic_fit_beam": (
                "lowest-rank candidates plus two low-rank Cartesian neighbors for prioritized "
                "regenerated Atom features, interleaved across slots"
            ),
            "max_atom_combinations_per_ansatz": MAX_ATOM_COMBINATIONS,
            "max_semantic_fits_per_ansatz": MAX_SEMANTIC_FITS_PER_ANSATZ,
            "max_multislot_semantic_fits_per_ansatz": (MAX_MULTISLOT_SEMANTIC_FITS_PER_ANSATZ),
            "max_semantic_fits_per_case": MAX_SEMANTIC_FITS_PER_CASE,
            "semantic_acceptance_loss": SEMANTIC_ACCEPTANCE_LOSS,
            "selection": (
                "minimum disjoint public PDE-and-condition validation weighted MSE; "
                "generation rank tie-break; stop after the first semantic pass in certified "
                "Ansatz rank order"
            ),
            "max_fitted_coefficients": MAX_FITTED_COEFFICIENTS,
            "optimizer": {
                "name": "BFGS",
                "max_iterations": 300,
                "multislot_max_iterations": 120,
                "gradient_tolerance": 1.0e-8,
                "restarts": 6,
                "multislot_restarts": 4,
                "random_seed": 20260904,
                "restart_scale": 1.0,
                "acceptance_loss": 1.0e-8,
                "nonlinear_argument_scale_starts": [1.0, -1.0, 4.0, -4.0, 16.0, -16.0],
            },
        },
        "case_count": len(cases),
        "cases": cases,
    }
    if compositional_structure:
        result["structure_protocol"] = "FUNCTION-IDENTITY-COMPOSITIONAL-ANSATZ-v2"
        result["compositional_structure"] = True
        result["policies"]["assembly"] = (
            "function-identity pools; shared coefficients across argument maps; nested DAG; "
            "capture-avoiding bottom-up simultaneous binding; structural differentiation "
            "after complete substitution; no unresolved derivative/profile/dummy terms"
        )
        result["policies"]["ansatz_symbolic_validation"]["weak"] += (
            "; opt-in bounded compositional substitution closure, not existence/regularity proof"
        )
    if coefficient_backend == "cuda_lbfgs":
        result["coefficient_backend"] = coefficient_backend
        result["cuda_device_index"] = cuda_device_index
        result["cuda_runtime"] = dict(cuda_runtime or {})
        result["policies"]["optimizer"] = {
            **result["policies"]["optimizer"],
            "name": "CUDA L-BFGS",
            "coefficient_backend": coefficient_backend,
            "cuda_device_index": cuda_device_index,
        }
    return result


__all__ = [
    "ATOM_TYPES",
    "AtomAssemblyRejection",
    "MAX_ATOM_COMBINATIONS",
    "MAX_ATOM_POOL_SIZE",
    "MAX_REGENERATION_ATTEMPTS",
    "MAX_MULTISLOT_SEMANTIC_FITS_PER_ANSATZ",
    "MAX_SEMANTIC_FITS_PER_ANSATZ",
    "MAX_SEMANTIC_FITS_PER_CASE",
    "MAX_SEMANTIC_REGENERATION_ROUNDS",
    "MAX_STAGE_II_REQUESTS",
    "SEMANTIC_ACCEPTANCE_LOSS",
    "STAGE_I_CANDIDATE_COUNT",
    "STAGE_II_CANDIDATE_COUNT",
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
    "render_atom_messages",
    "run_staged_experiment",
    "select_semantic_candidate_beam",
    "semantic_validate_candidates",
    "validate_atom_response",
]
