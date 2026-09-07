"""Condition-aware LLM Ansatz proposals with a strict public-input boundary.

The LLM is a proposal generator, not a proof engine.  Benchmark identifiers and
offline gold are never rendered into a provider request.  Returned templates are
accepted only after exact-schema and host-side structural validation, and they remain
explicitly non-certified until a separate symbolic replay proves them. Public IC/BC,
domain bounds, parameter values, and declared public forcing definitions are intentionally
included in this isolated V11 protocol.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import sympy as sp
from sympy.core.function import AppliedUndef

from .safe_symbolic import (
    SafeSymbolicError,
    is_abstract_coefficient_name,
    parse_symbolic_expression,
)

MAX_ANSATZ_LENGTH = 32_768
MAX_ANSATZ_REASON_LENGTH = 4_096

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MAX_PROPOSALS_PER_CASE = 3
_ANSATZ_INPUT_FIELDS = (
    "dependent_variables",
    "independent_variables",
    "pde",
    "initial_conditions",
    "boundary_conditions",
    "domain",
    "parameters",
    "forcing_definitions",
)
_FORBIDDEN_PROVIDER_KEYS = frozenset(
    {
        "benchmark_id",
        "gold",
        "reference_solution",
        "acceptable_families",
        "acceptable_patterns",
        "ansatz",
    }
)
_FORBIDDEN_RESPONSE_TEXT = re.compile(
    r"(?:\bbenchmark\b|\bgold\b|\breference[_ ]solution\b|\bPA-[A-Z0-9-]+\b)",
    re.IGNORECASE,
)
_CONCRETE_ATOM = re.compile(
    r"\b(?:Abs|sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|sech|exp|log|sqrt|erf|"
    r"Airy[A-Za-z]*|Bessel[A-Za-z]*)\s*\(",
    re.IGNORECASE,
)
_CONCRETE_FUNCTION_NAMES = frozenset(
    {
        "Abs",
        "Airy",
        "Bessel",
        "acos",
        "asin",
        "atan",
        "cos",
        "cosh",
        "erf",
        "exp",
        "log",
        "sech",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    }
)
_REFERENCE_IDENTIFIER = re.compile(
    r"\b[A-Za-z][A-Za-z0-9_]*_(?:star|exact)\b",
    re.IGNORECASE,
)
_REFERENCE_PHRASE = re.compile(
    r"\breference(?:[-_ ]solution)?[-_ ]trace\b|\breference[-_ ]solution\b",
    re.IGNORECASE,
)
_IDENTIFIER = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\b")
_FUNCTION_NAME = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s*(?=\()")
_PROFILE_CALL = re.compile(r"\b[A-Z][A-Za-z0-9_]*\s*\(")
_PROFILE_SYMBOL = re.compile(r"\b[A-Z][A-Za-z0-9_]*\b")
_IMPLICIT_PROFILE_PRODUCT = re.compile(r"(?<=\))(?=(?:[A-Z][A-Za-z0-9_]*\s*\(|\())")
_FIELD_LHS = re.compile(r"(?P<field>[A-Za-z][A-Za-z0-9_]*)(?:\((?P<arguments>[A-Za-z0-9_,\s]*)\))?")
_PROPOSAL_FAMILY = "condition_aware_high_dimensional_structure"

SYSTEM_PROMPT = """You are a mathematical PDE Ansatz analyst, not an Atom generator.

Definition: an Ansatz is an abstract structural hypothesis for unknown PDE fields. It
specifies dependencies, invariant or similarity coordinates, separation, scaling, or
coupling by using arbitrary unknown profile functions and symbolic coefficients. An
Ansatz is not a concrete basis function, initial/boundary trace, fitted expression,
closed-form solution, or Atom. Never output any fixed named mathematical function,
polynomial, special function, or other closed form as a field profile.

Use the supplied dependent variables, independent variables, sanitized PDE operator,
declared public forcing definitions, public initial and boundary conditions, domain bounds,
and parameter values. Public forcing or conditions may contain concrete functions or traces.
Use them as mathematical constraints when choosing dependencies, coordinates, separation,
scaling, and coupling, but keep every Stage-I output atom-free. A fixed elementary function may
appear inside an arbitrary profile's arguments when it defines a coordinate transformation
(for example a radial coordinate); it may not appear as a fixed field factor or field branch.
Final reference fields,
benchmark metadata, evaluator labels, and precomputed structural skeletons are intentionally
absent. Do not identify a benchmark or reconstruct a remembered solution.

Dimension reduction is optional, not the objective. For nonseparable forcing, interacting
coordinates, or coupled operators, preserve a full-coordinate arbitrary profile such as
F1(x,y) or F1(x,y,t) in at least one candidate instead of forcing an unjustified separated
or one-coordinate form. A longer multi-profile rank sum or product may be structurally
better than a short Ansatz; do not reject, truncate, or demote it merely because it has more
slots or symbolic operations. Every coefficient must remain a symbolic scalar a1, a2, ... .
Never guess numeric coefficient values: the host optimizer fits them after Atom completion.

Format-only synthetic example (not derived from any evaluated PDE): for a fictitious
field q and fictitious coordinates r,s, `q(r,s)=a1*F1(r)*F2(s)` is an atom-free Ansatz
because F1 and F2 remain arbitrary unknown profiles. This example demonstrates notation
only. It does not make multiplicative separation a preferred candidate and must not be
copied unless the supplied operator independently supports it.

For every case, derive candidates from the differential operator and disclosed conditions:
1. Classify order, linearity, autonomy, coupling, forcing, and distinguished variables.
2. Derive admissible transformations, reduced coordinates, profile dependencies, and
   coefficient constraints from operator identities, determining equations, IC/BC, and domain.
3. Show in the short reason which operator or condition calculation produced each structure.
   Do not select or copy a structure from a prewritten Ansatz catalogue.
4. Rank genuinely distinct derived candidates; prioritize retained coordinate interactions,
   justified profile multiplicity, and boundary-compatible structure before concision. Do not
   fill the list with coefficient variants or generic guesses unsupported by the supplied input.

Return strict JSON only. Every field equation must contain at least one arbitrary unknown
profile function. Do not assign a predefined structural family. The short reason must
state the operator or disclosed-condition calculation supporting the proposal, without
claiming knowledge of a hidden or remembered solution. The output schema_version must be exactly
LLMAnsatzBatchResponseV1. A proposal is not a proof.
"""


class LLMAnsatzError(RuntimeError):
    """Fail-closed provider or response-boundary error."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LLMAnsatzError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json_content(content: str) -> Any:
    stripped = content.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        stripped = stripped[len("```json") : -len("```")].strip()
    elif stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped[len("```") : -len("```")].strip()
    return json.loads(stripped, object_pairs_hook=_reject_duplicate_pairs)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _exact(mapping: Mapping[str, Any], fields: set[str], label: str) -> None:
    if set(mapping) != fields:
        raise LLMAnsatzError(f"{label} fields differ: {sorted(mapping)}")


def _normalized_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")


def _assert_no_nested_private_data(value: object, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = _normalized_key(key)
            if (
                normalized in _FORBIDDEN_PROVIDER_KEYS
                or "gold" in normalized
                or normalized.startswith("reference_")
                or normalized
                in {
                    "api_key",
                    "authorization",
                    "credential",
                    "password",
                    "secret",
                    "token",
                }
            ):
                raise ValueError(f"private provider key at {path}.{key}")
            _assert_no_nested_private_data(nested, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _assert_no_nested_private_data(nested, path=f"{path}[{index}]")


def _public_parameter_names(value: object) -> list[str]:
    if isinstance(value, Mapping):
        names = list(value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        names = list(value)
    else:
        raise ValueError("LLM Ansatz parameters must be a mapping or sequence")
    if any(
        not isinstance(name, str)
        or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name)
        or _normalized_key(name) in _FORBIDDEN_PROVIDER_KEYS
        for name in names
    ):
        raise ValueError("LLM Ansatz parameter names are invalid")
    return sorted(set(names))


def _sanitize_public_text_v1(value: str) -> str:
    """Remove reference placeholders while retaining legitimate problem data."""

    sanitized = _REFERENCE_PHRASE.sub("prescribed_trace", value)
    sanitized = _REFERENCE_IDENTIFIER.sub("external_forcing", sanitized)
    return re.sub(r"\bmanufactured\b", "prescribed", sanitized, flags=re.IGNORECASE)


def sanitize_public_pde_v1(equations: Sequence[str]) -> tuple[str, ...]:
    """Neutralize reference-derived forcing definitions without dropping the case."""

    sanitized = tuple(
        _sanitize_public_text_v1(equation.strip())
        for equation in equations
        if isinstance(equation, str) and equation.strip() and ":=" not in equation
    )
    if not sanitized:
        raise ValueError("LLM Ansatz input has no usable public PDE equation")
    return sanitized


def _sanitize_public_condition_lines(values: object, *, label: str) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"LLM Ansatz {label} must be a sequence")
    sanitized = [
        _sanitize_public_text_v1(value.strip())
        for value in values
        if isinstance(value, str) and value.strip()
    ]
    if len(sanitized) != len(values):
        raise ValueError(f"LLM Ansatz {label} contains an invalid condition")
    return sanitized


def _public_parameter_values(value: object, names: Sequence[str]) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {name: value[name] for name in names}
    return {name: "unspecified" for name in names}


def public_problem_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the public full-problem condition-aware V11 boundary."""

    if _FORBIDDEN_PROVIDER_KEYS.intersection(row):
        # Public benchmark rows carry an opaque ID locally; it must be dropped rather
        # than making the whole row unusable.
        row = {key: value for key, value in row.items() if key not in _FORBIDDEN_PROVIDER_KEYS}
    required = {"dependent_variables", "independent_variables", "pde"}
    if not required.issubset(row):
        raise ValueError("LLM Ansatz input lacks required public fields")
    selected = {
        key: row.get(
            key,
            {} if key in {"parameters", "domain"} else [],
        )
        for key in _ANSATZ_INPUT_FIELDS
    }
    _assert_no_nested_private_data(selected, path="public_problem")
    pde = list(sanitize_public_pde_v1(selected["pde"]))
    parameter_names = _public_parameter_names(selected["parameters"])
    if not isinstance(selected["domain"], Mapping):
        raise ValueError("LLM Ansatz domain must be a mapping")
    payload = {
        **selected,
        "parameters": parameter_names,
        "parameter_values": _public_parameter_values(selected["parameters"], parameter_names),
        "pde": pde,
        "initial_conditions": _sanitize_public_condition_lines(
            selected["initial_conditions"], label="initial_conditions"
        ),
        "boundary_conditions": _sanitize_public_condition_lines(
            selected["boundary_conditions"], label="boundary_conditions"
        ),
        "forcing_definitions": _sanitize_public_condition_lines(
            selected["forcing_definitions"], label="forcing_definitions"
        ),
        "domain": dict(selected["domain"]),
    }
    sequence_fields = (
        "dependent_variables",
        "independent_variables",
        "pde",
        "initial_conditions",
        "boundary_conditions",
        "forcing_definitions",
    )
    if not all(
        isinstance(payload[key], Sequence) and not isinstance(payload[key], str)
        for key in sequence_fields
    ):
        raise ValueError("LLM Ansatz public sequence fields are invalid")
    if any(not isinstance(equation, str) or not equation.strip() for equation in payload["pde"]):
        raise ValueError("LLM Ansatz PDE text is invalid")
    encoded = _canonical_bytes(payload)
    if len(encoded) > 131_072:
        raise ValueError("LLM Ansatz public payload is oversized")
    return payload


def _response_format(case_tokens: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "pde_ansatz_batch_v1",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["schema_version", "cases"],
                "properties": {
                    "schema_version": {
                        "type": "string",
                        "const": "LLMAnsatzBatchResponseV1",
                    },
                    "cases": {
                        "type": "array",
                        "minItems": len(case_tokens),
                        "maxItems": len(case_tokens),
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["case_token", "candidates"],
                            "properties": {
                                "case_token": {
                                    "type": "string",
                                    "enum": list(case_tokens),
                                },
                                "candidates": {
                                    "type": "array",
                                    "minItems": MAX_PROPOSALS_PER_CASE,
                                    "maxItems": MAX_PROPOSALS_PER_CASE,
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": ["ansatz", "reason"],
                                        "properties": {
                                            "ansatz": {"type": "string"},
                                            "reason": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def render_batch_messages(
    public_rows: Sequence[Mapping[str, Any]],
    *,
    validation_feedback: Sequence[str] = (),
) -> tuple[tuple[str, ...], tuple[dict[str, str], ...], tuple[dict[str, Any], ...]]:
    """Bind opaque request tokens to sanitized condition-aware public cases."""

    if len(public_rows) != 1:
        raise ValueError("Stage I requires exactly one PDE per provider request")
    payloads = tuple(public_problem_payload(row) for row in public_rows)
    tokens = tuple(f"case_{index:03d}" for index in range(1, len(payloads) + 1))
    token = tokens[0]
    payload = payloads[0]
    feedback = tuple(validation_feedback)
    if any(
        not isinstance(item, str)
        or not item.strip()
        or len(item) > 4_096
        or _FORBIDDEN_RESPONSE_TEXT.search(item)
        for item in feedback
    ):
        raise ValueError("Stage I host validation feedback is invalid")

    def bullets(values: Sequence[object]) -> str:
        items = tuple(values)
        return "\n".join(f"- {item}" for item in items) if items else "- (none)"

    repair_block = ""
    if feedback:
        repair_block = f"""

Previous response: host validation feedback
{bullets(feedback)}

Correct only these mechanical or symbolic-validation failures. Generate a new batch of
three candidates, do not repeat an already certified Ansatz, and do not invent undisclosed
conditions, traces, or reference solutions. Host feedback is not a solution hint.
"""

    parameter_values = tuple(
        f"{name} = {json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"
        for name, value in payload["parameter_values"].items()
    )
    domain = json.dumps(
        payload["domain"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )

    user = f"""Generate exactly {MAX_PROPOSALS_PER_CASE} ranked Ansatz candidates for this problem.

Use semicolon-separated field equations and retain the declared variable names.
Do not include explanations outside the required reason fields.

Problem token:
{token}

Dependent variables:
{bullets(payload["dependent_variables"])}

Independent variables:
{bullets(payload["independent_variables"])}

Public parameter names:
{bullets(payload["parameters"])}

Public parameter values:
{bullets(parameter_values)}

PDE operator:
{bullets(payload["pde"])}

Public forcing definitions:
{bullets(payload["forcing_definitions"])}

Initial conditions:
{bullets(payload["initial_conditions"])}

Boundary conditions:
{bullets(payload["boundary_conditions"])}

Domain:
{domain}
{repair_block}

Return only one JSON object with exactly schema_version and cases. The schema_version
must be LLMAnsatzBatchResponseV1. The single case must contain exactly case_token and
candidates. Each of the exactly {MAX_PROPOSALS_PER_CASE} candidates must contain exactly
ansatz and reason. Do not add any other keys."""
    messages = (
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    )
    return tokens, messages, payloads


@dataclass(frozen=True)
class LLMAnsatzCandidateV1:
    rank: int
    candidate_id: str
    family: str
    template: str
    reason: str
    model: str
    host_validation_passed: bool = True

    def to_mapping(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "candidate_id": self.candidate_id,
            "atom_free_family": self.family,
            "ansatz": self.template,
            "score": 6 - self.rank,
            "evidence": ["LLM_PUBLIC_PROPOSAL", "HOST_STRUCTURE_VALIDATED"],
            "reason": self.reason,
            "model": self.model,
            "validation_kind": "safe_ast_host_structure_validation_v2",
            "host_validation_passed": self.host_validation_passed,
            "independent_structure_replay": False,
        }


@dataclass(frozen=True)
class LLMAnsatzBatchCompletionV1:
    response: Mapping[str, Any]
    provider_name: str
    requested_model: str
    returned_model: str
    provider_response_id: str
    prompt_tokens: int
    completion_tokens: int
    request_payload_sha256: str
    raw_response_sha256: str
    response_mode: str = "json_schema"


class LLMAnsatzBatchProviderV1(Protocol):
    def complete(
        self,
        *,
        tokens: Sequence[str],
        messages: Sequence[Mapping[str, str]],
        response_format: Mapping[str, Any],
    ) -> LLMAnsatzBatchCompletionV1: ...


class InMemoryLLMAnsatzProviderV1:
    """Deterministic in-memory provider for boundary tests."""

    def __init__(
        self, response: Mapping[str, Any], *, model: str = "in-memory-ansatz-test-v1"
    ) -> None:
        self.response = response
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        tokens: Sequence[str],
        messages: Sequence[Mapping[str, str]],
        response_format: Mapping[str, Any],
    ) -> LLMAnsatzBatchCompletionV1:
        request = {
            "tokens": list(tokens),
            "messages": list(messages),
            "response_format": response_format,
        }
        self.calls.append(request)
        return LLMAnsatzBatchCompletionV1(
            self.response,
            "in-memory-test",
            self.model,
            self.model,
            "in-memory-response-1",
            1,
            1,
            _sha256(request),
            _sha256(self.response),
        )


class OpenRouterLLMAnsatzProviderV1:
    """Bounded stdlib-only OpenRouter structured-output transport."""

    def __init__(
        self,
        credential: Callable[[], str],
        *,
        model: str,
        timeout_seconds: float = 120.0,
        max_retries: int = 1,
        use_json_schema: bool = True,
        max_output_tokens: int = 24_000,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> None:
        if not callable(credential):
            raise ValueError("credential must be callable")
        if not model or len(model) > 128:
            raise ValueError("model is invalid")
        if timeout_seconds <= 0 or timeout_seconds > 300:
            raise ValueError("timeout_seconds is invalid")
        if max_retries not in {0, 1, 2}:
            raise ValueError("max_retries is invalid")
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or not 512 <= max_output_tokens <= 24_000
        ):
            raise ValueError("max_output_tokens is invalid")
        if temperature is not None and (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0.0 <= float(temperature) <= 2.0
        ):
            raise ValueError("temperature is invalid")
        if top_p is not None and (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not 0.0 < float(top_p) <= 1.0
        ):
            raise ValueError("top_p is invalid")
        self._credential = credential
        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max_retries
        self.use_json_schema = bool(use_json_schema)
        self.max_output_tokens = max_output_tokens
        self.temperature = None if temperature is None else float(temperature)
        self.top_p = None if top_p is None else float(top_p)
        self.request_count = 0

    def complete(
        self,
        *,
        tokens: Sequence[str],
        messages: Sequence[Mapping[str, str]],
        response_format: Mapping[str, Any],
    ) -> LLMAnsatzBatchCompletionV1:
        if self.request_count:
            raise LLMAnsatzError("provider instance allows exactly one request")
        body_mapping: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "provider": {
                "require_parameters": self.use_json_schema,
                "allow_fallbacks": False,
            },
        }
        if self.use_json_schema:
            body_mapping["response_format"] = response_format
        if self.temperature is not None:
            body_mapping["temperature"] = self.temperature
        if self.top_p is not None:
            body_mapping["top_p"] = self.top_p
        completion_limit_key = (
            "max_tokens" if self.model.startswith("google/") else "max_completion_tokens"
        )
        body_mapping[completion_limit_key] = self.max_output_tokens
        body = _canonical_bytes(body_mapping)
        if len(body) > 1_048_576:
            raise LLMAnsatzError("provider request is oversized")
        key = self._credential()
        if not isinstance(key, str) or not key:
            raise LLMAnsatzError("provider credential is unavailable")
        request = urllib.request.Request(
            OPENROUTER_ENDPOINT,
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        self.request_count += 1
        raw: bytes | None = None
        last_error: Exception | None = None
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    with urllib.request.urlopen(  # nosec B310
                        request, timeout=self.timeout_seconds
                    ) as response:
                        raw = response.read(1_048_577)
                    break
                except urllib.error.HTTPError as exc:
                    detail = exc.read(8_192).decode("utf-8", errors="replace")
                    last_error = LLMAnsatzError(f"provider HTTP {exc.code}: {detail[:1000]}")
                    if exc.code < 500 and exc.code != 429:
                        break
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
        finally:
            key = ""
        if raw is None:
            raise LLMAnsatzError(f"provider request failed: {last_error}") from last_error
        if len(raw) > 1_048_576:
            raise LLMAnsatzError("provider response is oversized")
        raw_sha256 = hashlib.sha256(raw).hexdigest()
        try:
            outer = json.loads(raw, object_pairs_hook=_reject_duplicate_pairs)
            choice = outer["choices"][0]
            content = choice["message"]["content"]
            returned_model = outer["model"]
            response_id = outer["id"]
            usage = outer.get("usage") or {}
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LLMAnsatzError("provider outer response is malformed") from exc
        if returned_model != self.model:
            raise LLMAnsatzError(
                f"provider model mismatch: requested {self.model}, got {returned_model}"
            )
        if not isinstance(content, str) or not isinstance(response_id, str):
            raise LLMAnsatzError("provider content or identity is invalid")
        try:
            parsed = _parse_json_content(content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise LLMAnsatzError("provider structured body is malformed") from exc
        return LLMAnsatzBatchCompletionV1(
            parsed,
            "openrouter",
            self.model,
            returned_model,
            response_id,
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
            hashlib.sha256(body).hexdigest(),
            raw_sha256,
            "json_schema" if self.use_json_schema else "prompt_json_host_validated",
        )


def canonicalize_llm_ansatz_notation(ansatz: str) -> str:
    """Normalize only unambiguous adjacent profile/group multiplication."""

    rendered = []
    for part in (item.strip() for item in ansatz.split(";") if item.strip()):
        if part.count("=") != 1:
            rendered.append(part)
            continue
        lhs, rhs = (item.strip() for item in part.split("=", maxsplit=1))
        rendered.append(f"{lhs}={_IMPLICIT_PROFILE_PRODUCT.sub('*', rhs)}")
    return ";".join(rendered)


def _every_additive_term_has_abstract_profile(source: str) -> bool:
    """Reject fixed closed-form branches mixed beside an abstract profile."""

    try:
        root = ast.parse(source.replace("^", "**"), mode="eval").body
    except (SyntaxError, ValueError):
        return False

    def terms(node: ast.AST) -> list[ast.AST]:
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            return [*terms(node.left), *terms(node.right)]
        return [node]

    def has_profile(node: ast.AST) -> bool:
        return any(
            isinstance(nested, ast.Call)
            and isinstance(nested.func, ast.Name)
            and nested.func.id[0].isupper()
            and nested.func.id != "Abs"
            for nested in ast.walk(node)
        )

    return all(has_profile(term) for term in terms(root))


def _inline_auxiliary_coordinates(
    ansatz: str,
    dependent_variables: Sequence[str],
    independent_variables: Sequence[str],
    parameters: Sequence[str],
) -> str:
    """Strictly parse field equations and inline explicit scalar coordinates."""

    dependent = set(dependent_variables)
    independent = set(independent_variables)
    parameter_names = set(parameters)
    field_equations: dict[str, str] = {}
    auxiliaries: dict[str, str] = {}
    for part in (item.strip() for item in ansatz.split(";") if item.strip()):
        if part.count("=") != 1:
            raise LLMAnsatzError("LLM candidate equation syntax is invalid")
        lhs, rhs = (item.strip() for item in part.split("=", maxsplit=1))
        lhs_match = _FIELD_LHS.fullmatch(lhs)
        if lhs_match is None:
            raise LLMAnsatzError("LLM candidate LHS is malformed")
        field = lhs_match.group("field")
        raw_arguments = lhs_match.group("arguments")
        arguments = tuple(item.strip() for item in (raw_arguments or "").split(",") if item.strip())
        if field in dependent:
            if field in field_equations:
                raise LLMAnsatzError("LLM candidate duplicates a dependent field")
            if raw_arguments is not None and arguments != tuple(independent_variables):
                raise LLMAnsatzError("LLM candidate field arguments differ from public variables")
            field_equations[field] = rhs
            continue
        if raw_arguments is None:
            if field in independent or field in parameter_names or field in auxiliaries:
                raise LLMAnsatzError("LLM candidate auxiliary name is invalid or duplicate")
            if _PROFILE_CALL.search(rhs):
                raise LLMAnsatzError("LLM auxiliary coordinate may not call profiles")
            auxiliaries[lhs] = rhs
            continue
        raise LLMAnsatzError("LLM candidate LHS is not a declared field or coordinate")
    if set(field_equations) != dependent:
        raise LLMAnsatzError("LLM candidate must define every dependent field exactly once")

    resolved: dict[str, str] = {}

    def resolve(name: str, active: frozenset[str]) -> str:
        if name in resolved:
            return resolved[name]
        if name in active:
            raise LLMAnsatzError("LLM candidate auxiliary coordinates are cyclic")
        source = auxiliaries[name]
        try:
            parse_symbolic_expression(source)
        except SafeSymbolicError as exc:
            raise LLMAnsatzError("LLM auxiliary coordinate syntax is invalid") from exc
        rendered = _IDENTIFIER.sub(
            lambda match: (
                f"({resolve(match.group(0), active | {name})})"
                if match.group(0) in auxiliaries
                else match.group(0)
            ),
            source,
        )
        resolved[name] = rendered
        return rendered

    for name in auxiliaries:
        resolve(name, frozenset())

    used_auxiliaries: set[str] = set()
    rendered = []
    declared_arguments = ",".join(independent_variables)
    for field in dependent_variables:
        rhs = field_equations[field]
        used_auxiliaries.update(set(_IDENTIFIER.findall(rhs)) & set(auxiliaries))
        inlined = _IDENTIFIER.sub(
            lambda match: (
                f"({resolved[match.group(0)]})" if match.group(0) in resolved else match.group(0)
            ),
            rhs,
        )
        rendered.append(f"{field}({declared_arguments})={inlined}")
    if set(auxiliaries) != used_auxiliaries:
        raise LLMAnsatzError("LLM candidate contains an unused auxiliary coordinate")
    return ";".join(rendered)


def _has_concrete_atom_outside_profile_arguments(source: str) -> bool:
    """Distinguish fixed field Atoms from fixed coordinate constructors.

    Stage I may use a disclosed map such as ``sqrt((x-x0)**2+(y-y0)**2)`` only as an
    argument to an otherwise arbitrary profile.  The same function outside that profile
    call would already instantiate part of the field and therefore belongs in Stage II.
    """

    try:
        tree = ast.parse(source.replace("^", "**"), mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise LLMAnsatzError("LLM candidate RHS syntax is invalid") from exc

    def concrete(node: ast.AST) -> bool:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name and name[0].isupper() and name != "Abs":
                return False
            if name in _CONCRETE_FUNCTION_NAMES or name.startswith(("Airy", "Bessel")):
                return True
        return any(concrete(child) for child in ast.iter_child_nodes(node))

    return concrete(tree)


def _validate_candidate(
    raw: Mapping[str, Any],
    *,
    dependent_variables: Sequence[str],
    independent_variables: Sequence[str],
    parameters: Sequence[str],
) -> tuple[str, str]:
    _exact(raw, {"ansatz", "reason"}, "LLM candidate")
    values = tuple(raw[key] for key in ("ansatz", "reason"))
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise LLMAnsatzError("LLM candidate text is invalid")
    ansatz, reason = (value.strip() for value in values)
    ansatz = _inline_auxiliary_coordinates(
        ansatz,
        dependent_variables,
        independent_variables,
        parameters,
    )
    ansatz = canonicalize_llm_ansatz_notation(ansatz)
    if len(ansatz) > MAX_ANSATZ_LENGTH or len(reason) > MAX_ANSATZ_REASON_LENGTH:
        raise LLMAnsatzError("LLM candidate text is oversized")
    if _FORBIDDEN_RESPONSE_TEXT.search(f"{ansatz}\n{reason}"):
        raise LLMAnsatzError("LLM candidate contains forbidden benchmark language")
    equations = tuple(part.strip() for part in ansatz.split(";") if part.strip())
    if not equations or any(part.count("=") != 1 for part in equations):
        raise LLMAnsatzError("LLM candidate equation syntax is invalid")
    canonical_equations: list[str] = []
    independent_symbols = {sp.Symbol(name) for name in independent_variables}
    dependent_symbols = {sp.Symbol(name) for name in dependent_variables}
    parameter_symbols = {sp.Symbol(name) for name in parameters}
    abstract_profile_found = False
    for part in equations:
        lhs, rhs = (item.strip() for item in part.split("=", maxsplit=1))
        try:
            expression = parse_symbolic_expression(rhs)
        except SafeSymbolicError as exc:
            raise LLMAnsatzError("LLM candidate RHS syntax is invalid") from exc
        if _has_concrete_atom_outside_profile_arguments(rhs):
            raise LLMAnsatzError(
                "LLM candidate contains a concrete atom outside a profile coordinate"
            )
        known_symbols = independent_symbols | dependent_symbols | parameter_symbols
        unknown_scalars = {
            symbol
            for symbol in expression.free_symbols - known_symbols
            if not str(symbol)[0].isupper()
        }
        if any("__" in str(symbol) for symbol in expression.free_symbols):
            raise LLMAnsatzError("LLM candidate contains a forbidden identifier")
        if any(not is_abstract_coefficient_name(str(symbol)) for symbol in unknown_scalars):
            raise LLMAnsatzError("LLM candidate uses an undeclared coordinate")
        has_axis = bool(expression.free_symbols & independent_symbols)
        has_dependent = bool(expression.free_symbols & dependent_symbols)
        called_profiles = {
            name for name in _FUNCTION_NAME.findall(rhs) if name[0].isupper() and name != "Abs"
        }
        if not called_profiles:
            raise LLMAnsatzError("every dependent field must use an abstract profile function")
        if not _every_additive_term_has_abstract_profile(rhs):
            raise LLMAnsatzError("LLM candidate mixes a fixed field branch with abstract profiles")
        abstract_profile_found = abstract_profile_found or bool(called_profiles)
        if unknown_scalars and not has_axis:
            raise LLMAnsatzError("LLM candidate uses an undeclared coordinate")
        if not has_axis and not has_dependent:
            raise LLMAnsatzError("LLM candidate profile has no public coordinate dependence")
        canonical_equations.append(f"{lhs}={rhs}")
    if not abstract_profile_found:
        raise LLMAnsatzError("LLM candidate lacks an abstract profile function")
    ansatz = ";".join(canonical_equations)
    return ansatz, reason


def _validate_candidate_permissive(
    raw: Mapping[str, Any],
    *,
    dependent_variables: Sequence[str],
    independent_variables: Sequence[str],
    parameters: Sequence[str],
) -> tuple[str, str]:
    """Parse-only Ansatz gate for the isolated direct Atom lane.

    This intentionally does not enforce a field census, profile topology, concrete-Atom
    ban, coordinate coverage, or parameter naming policy.  It still requires an equation
    that the safe symbolic parser can execute and at least one open (uppercase) profile,
    because the next stage must have a slot to bind.  The strict validator above remains
    the default for the original tree/binding implementation.
    """

    _exact(raw, {"ansatz", "reason"}, "LLM candidate")
    values = tuple(raw[key] for key in ("ansatz", "reason"))
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise LLMAnsatzError("LLM candidate text is invalid")
    ansatz, reason = (value.strip() for value in values)
    ansatz = canonicalize_llm_ansatz_notation(ansatz)
    if len(ansatz) > MAX_ANSATZ_LENGTH or len(reason) > MAX_ANSATZ_REASON_LENGTH:
        raise LLMAnsatzError("LLM candidate text is oversized")
    if _FORBIDDEN_RESPONSE_TEXT.search(f"{ansatz}\n{reason}"):
        raise LLMAnsatzError("LLM candidate contains forbidden benchmark language")
    equations = tuple(part.strip() for part in ansatz.split(";") if part.strip())
    if not equations or any(part.count("=") != 1 for part in equations):
        raise LLMAnsatzError("LLM candidate equation syntax is invalid")
    # Public names are supplied as symbols only to avoid mistaking them for open profiles;
    # unknown identifiers remain inert symbolic coefficients for the later optimizer.
    names: dict[str, object] = {
        **{str(name): sp.Symbol(str(name), real=True) for name in independent_variables},
        **{str(name): sp.Symbol(str(name), real=True) for name in dependent_variables},
        **{str(name): sp.Symbol(str(name), real=True) for name in parameters},
        "pi": sp.pi,
    }
    canonical_equations: list[str] = []
    profile_count = 0
    for part in equations:
        lhs, rhs = (item.strip() for item in part.split("=", maxsplit=1))
        if _FIELD_LHS.fullmatch(lhs) is None:
            raise LLMAnsatzError("LLM candidate LHS is malformed")
        if "__" in lhs or "__" in rhs:
            raise LLMAnsatzError("LLM candidate contains a forbidden identifier")
        try:
            expression = parse_symbolic_expression(rhs, names=names)
        except SafeSymbolicError as exc:
            raise LLMAnsatzError("LLM candidate RHS syntax is invalid") from exc
        profiles = tuple(expression.atoms(AppliedUndef))
        profile_count += len(profiles)
        canonical_equations.append(f"{lhs}={rhs}")
    if profile_count == 0:
        raise LLMAnsatzError("LLM candidate lacks an open profile function")
    return ";".join(canonical_equations), reason


def validate_llm_ansatz_candidate(
    candidate: Mapping[str, Any],
    *,
    dependent_variables: Sequence[str],
    independent_variables: Sequence[str],
    parameters: Sequence[str],
    permissive: bool = False,
) -> tuple[str, str]:
    """Apply the strict or direct-lane parse-only host validator."""

    raw = {
        "ansatz": candidate.get("ansatz"),
        "reason": candidate.get("reason"),
    }
    validator = _validate_candidate_permissive if permissive else _validate_candidate
    return validator(
        raw,
        dependent_variables=dependent_variables,
        independent_variables=independent_variables,
        parameters=parameters,
    )


def propose_llm_ansatz_batch(
    public_rows: Sequence[Mapping[str, Any]],
    *,
    provider: LLMAnsatzBatchProviderV1,
    allow_empty_cases: bool = False,
    validation_feedback: Sequence[str] = (),
    permissive: bool = False,
) -> tuple[tuple[tuple[LLMAnsatzCandidateV1, ...], ...], dict[str, Any]]:
    """Generate and validate a broad proposal pool for every public row.

    ``permissive=True`` is reserved for the V11 direct Atom clone and only relaxes
    structural checks before the mandatory syntax/slot certificate.
    """

    if not public_rows:
        raise ValueError("public_rows must be nonempty")
    tokens, messages, payloads = render_batch_messages(
        public_rows, validation_feedback=validation_feedback
    )
    completion = provider.complete(
        tokens=tokens,
        messages=messages,
        response_format=_response_format(tokens),
    )
    response = completion.response
    if not isinstance(response, Mapping):
        raise LLMAnsatzError("LLM response must be an object")
    _exact(response, {"schema_version", "cases"}, "LLM response")
    if response["schema_version"] != "LLMAnsatzBatchResponseV1":
        raise LLMAnsatzError("LLM response schema version differs")
    raw_cases = response["cases"]
    if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, (str, bytes)):
        raise LLMAnsatzError("LLM response cases are invalid")
    by_token: dict[str, Mapping[str, Any]] = {}
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping):
            raise LLMAnsatzError("LLM response case is invalid")
        _exact(raw_case, {"case_token", "candidates"}, "LLM response case")
        token = raw_case["case_token"]
        if not isinstance(token, str) or token in by_token:
            raise LLMAnsatzError("LLM response case token is invalid or duplicate")
        by_token[token] = raw_case
    if set(by_token) != set(tokens):
        raise LLMAnsatzError("LLM response case token census differs")

    all_candidates: list[tuple[LLMAnsatzCandidateV1, ...]] = []
    rejected: Counter[str] = Counter()
    for token, payload in zip(tokens, payloads, strict=True):
        raw_candidates = by_token[token]["candidates"]
        if (
            not isinstance(raw_candidates, Sequence)
            or isinstance(raw_candidates, (str, bytes))
            or len(raw_candidates) != MAX_PROPOSALS_PER_CASE
        ):
            raise LLMAnsatzError("LLM candidate count is invalid")
        unique: dict[str, LLMAnsatzCandidateV1] = {}
        for raw in raw_candidates:
            if not isinstance(raw, Mapping):
                rejected["candidate_not_object"] += 1
                continue
            try:
                validator = _validate_candidate_permissive if permissive else _validate_candidate
                ansatz, reason = validator(
                    raw,
                    dependent_variables=payload["dependent_variables"],
                    independent_variables=payload["independent_variables"],
                    parameters=payload["parameters"],
                )
            except LLMAnsatzError as exc:
                rejected[str(exc)] += 1
                continue
            key = re.sub(r"\s+", "", ansatz)
            if key in unique:
                continue
            rank = len(unique) + 1
            unique[key] = LLMAnsatzCandidateV1(
                rank,
                _sha256(
                    {
                        "public_problem_sha256": _sha256(payload),
                        "model": completion.requested_model,
                        "ansatz": ansatz,
                        "family": _PROPOSAL_FAMILY,
                    }
                ),
                _PROPOSAL_FAMILY,
                ansatz,
                reason,
                completion.requested_model,
            )
        if not unique:
            if not allow_empty_cases:
                raise LLMAnsatzError(
                    f"LLM produced no valid candidates for {token}; rejections={dict(rejected)}"
                )
        all_candidates.append(tuple(unique.values()))

    evidence = {
        "schema_version": "LLMAnsatzProviderEvidenceV1",
        "provider_name": completion.provider_name,
        "requested_model": completion.requested_model,
        "returned_model": completion.returned_model,
        "provider_response_id": completion.provider_response_id,
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
        "request_payload_sha256": completion.request_payload_sha256,
        "raw_response_sha256": completion.raw_response_sha256,
        "response_mode": completion.response_mode,
        "provider_received_benchmark_ids": False,
        "provider_received_gold": False,
        "public_payloads_sha256": _sha256(payloads),
        "provider_parameter_policy": "public_names_and_values_v11",
        "provider_received_conditions": True,
        "provider_received_domain": True,
        "provider_received_forcing_definitions": bool(
            any(payload["forcing_definitions"] for payload in payloads)
        ),
        "provider_received_concrete_atoms": any(
            _CONCRETE_ATOM.search(equation) for payload in payloads for equation in payload["pde"]
        ),
        "provider_received_concrete_condition_expressions": any(
            _CONCRETE_ATOM.search(condition)
            for payload in payloads
            for condition in (
                *payload["initial_conditions"],
                *payload["boundary_conditions"],
            )
        ),
        "provider_received_concrete_forcing_expressions": any(
            _CONCRETE_ATOM.search(definition)
            for payload in payloads
            for definition in payload["forcing_definitions"]
        ),
        "input_policy": "public_pde_forcing_ic_bc_domain_full_problem_v11",
        "prompt_revision": "full_problem_high_dimensional_atom_free_v11",
        "host_validation_revision": (
            "atom_free_syntax_slot_only_v12"
            if permissive
            else "atom_free_ansatz_full_dimensional_v11"
        ),
        "case_count": len(public_rows),
        "rejected_candidate_count": sum(rejected.values()),
        "rejection_reason_counts": dict(sorted(rejected.items())),
    }
    return tuple(all_candidates), evidence


def propose_llm_ansatz_chunked(
    public_rows: Sequence[Mapping[str, Any]],
    *,
    provider_factory: Callable[[], LLMAnsatzBatchProviderV1],
    batch_size: int = 4,
) -> tuple[tuple[tuple[LLMAnsatzCandidateV1, ...], ...], dict[str, Any]]:
    """Run the same mathematical prompt in bounded batches for provider portability."""

    if not public_rows:
        raise ValueError("public_rows must be nonempty")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    all_candidates: list[tuple[LLMAnsatzCandidateV1, ...]] = []
    source_evidence: list[dict[str, Any]] = []
    for start in range(0, len(public_rows), batch_size):
        candidates, evidence = propose_llm_ansatz_batch(
            public_rows[start : start + batch_size],
            provider=provider_factory(),
            allow_empty_cases=True,
        )
        all_candidates.extend(candidates)
        source_evidence.append(evidence)
    models = {item["requested_model"] for item in source_evidence}
    returned_models = {item["returned_model"] for item in source_evidence}
    response_modes = {item["response_mode"] for item in source_evidence}
    if len(models) != 1 or returned_models != models or len(response_modes) != 1:
        raise LLMAnsatzError("chunked provider model identities differ")
    rejection_counts: Counter[str] = Counter()
    for item in source_evidence:
        rejection_counts.update(item["rejection_reason_counts"])
    evidence = {
        "schema_version": "LLMAnsatzChunkedProviderEvidenceV1",
        "provider_name": source_evidence[0]["provider_name"],
        "requested_model": source_evidence[0]["requested_model"],
        "returned_model": source_evidence[0]["returned_model"],
        "response_mode": source_evidence[0]["response_mode"],
        "request_count": len(source_evidence),
        "batch_size": batch_size,
        "prompt_tokens": sum(item["prompt_tokens"] for item in source_evidence),
        "completion_tokens": sum(item["completion_tokens"] for item in source_evidence),
        "provider_received_benchmark_ids": False,
        "provider_received_gold": False,
        "provider_parameter_policy": "public_names_and_values_v11",
        "provider_received_conditions": True,
        "provider_received_domain": True,
        "provider_received_forcing_definitions": any(
            item["provider_received_forcing_definitions"] for item in source_evidence
        ),
        "provider_received_concrete_atoms": any(
            item["provider_received_concrete_atoms"] for item in source_evidence
        ),
        "provider_received_concrete_condition_expressions": any(
            item["provider_received_concrete_condition_expressions"] for item in source_evidence
        ),
        "provider_received_concrete_forcing_expressions": any(
            item["provider_received_concrete_forcing_expressions"] for item in source_evidence
        ),
        "input_policy": "public_pde_forcing_ic_bc_domain_full_problem_v11",
        "prompt_revision": "full_problem_high_dimensional_pool_v11",
        "host_validation_revision": "atom_free_ansatz_full_dimensional_v11",
        "case_count": len(public_rows),
        "rejected_candidate_count": sum(
            item["rejected_candidate_count"] for item in source_evidence
        ),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "source_provider_evidence": source_evidence,
    }
    return tuple(all_candidates), evidence


__all__ = [
    "InMemoryLLMAnsatzProviderV1",
    "LLMAnsatzBatchCompletionV1",
    "LLMAnsatzCandidateV1",
    "LLMAnsatzError",
    "MAX_ANSATZ_LENGTH",
    "MAX_ANSATZ_REASON_LENGTH",
    "OpenRouterLLMAnsatzProviderV1",
    "canonicalize_llm_ansatz_notation",
    "public_problem_payload",
    "sanitize_public_pde_v1",
    "propose_llm_ansatz_batch",
    "propose_llm_ansatz_chunked",
    "render_batch_messages",
    "validate_llm_ansatz_candidate",
]
