"""Mandatory-LLM concrete candidates with public-only coefficient fitting.

This module is for condition-aware development diagnostics.  Public conditions may
reveal the target family, so its outputs are not blind-discovery evidence.  The LLM
proposes both an Ansatz/Atom composition and numeric starting values; the host then
fits and selects candidates using only explicitly declared public physics residuals.
Reference solutions are deliberately outside this module.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import sympy as sp

from .coefficient_fitting import (
    BFGSConfigV1,
    CollocationResidualV1,
    fit_expression_coefficients_bfgs,
)
from .llm_provider import LLMBatchProviderV1, LLMResponseError
from .safe_symbolic import SafeSymbolicError, parse_symbolic_expression

SCHEMA_VERSION = "ConditionAwareSolutionBatchV1"
MAX_CANDIDATES = 8
_PARAMETER = re.compile(r"p([1-9][0-9]*)")
_REFERENCE_PLACEHOLDER = re.compile(
    r"\breference(?:[-_ ]solution)?[-_ ]trace\b|\breference[-_ ]solution\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You are generating executable PDE solution candidates for a
condition-aware development diagnostic. This is not blind discovery: initial and
boundary conditions may directly reveal the target family.

For every supplied public problem, derive several concrete parameterized expressions.
Each candidate must combine an Ansatz structure with named mathematical Atom functions
and must be executable using only the declared coordinates, real numeric literals,
pi, and consecutive scalar parameters p1, p2, ... . Do not use unknown functions,
unknown spatial profiles, reference traces, external forcing, benchmark memory, stored
answers, or prose inside expressions. Use explicit * and ** Python/SymPy syntax.

The expression_template is the right-hand side for the declared scalar field. Provide
one finite numeric initial value per free parameter in exact p1..pN order. Initial
values must be calculated or reasonably initialized from the supplied public PDE and
conditions, never from an external answer. Rank candidates by direct public-condition
compatibility and public PDE residual plausibility. Return exactly the requested JSON
and no prose outside it.
"""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def public_condition_problem_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep public mathematical evidence while dropping IDs and source metadata."""

    required = {
        "dependent_variables",
        "independent_variables",
        "pde",
        "initial_conditions",
        "boundary_conditions",
        "domain",
        "parameters",
    }
    if not required.issubset(row):
        raise ValueError("condition-aware solution input lacks required public fields")
    dependent = list(map(str, row["dependent_variables"]))
    independent = list(map(str, row["independent_variables"]))
    if len(dependent) != 1 or not independent:
        raise ValueError("condition-aware solution generation supports one scalar field")

    def clean(items: object) -> list[str]:
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            raise ValueError("public equation channel must be a sequence")
        return [
            _REFERENCE_PLACEHOLDER.sub("prescribed_trace", str(item).strip())
            for item in items
            if str(item).strip()
        ]

    payload = {
        "dependent_variables": dependent,
        "independent_variables": independent,
        "pde": clean(row["pde"]),
        "initial_conditions": clean(row["initial_conditions"]),
        "boundary_conditions": clean(row["boundary_conditions"]),
        "domain": row["domain"],
        "parameters": row["parameters"],
    }
    encoded = _canonical_bytes(payload)
    if len(encoded) > 131_072:
        raise ValueError("condition-aware solution input is oversized")
    return payload


def render_condition_aware_solution_messages(
    rows: Sequence[Mapping[str, Any]], *, candidate_count: int
) -> tuple[tuple[str, ...], tuple[dict[str, str], ...], tuple[dict[str, Any], ...]]:
    """Render a target-neutral batched request with opaque case tokens."""

    if not rows:
        raise ValueError("at least one public problem is required")
    if not 2 <= candidate_count <= MAX_CANDIDATES:
        raise ValueError("candidate_count must be between 2 and 8")
    payloads = tuple(public_condition_problem_payload(row) for row in rows)
    tokens = tuple(f"case_{index:03d}" for index in range(1, len(rows) + 1))
    request_cases = [
        {"case_token": token, "public_problem": payload}
        for token, payload in zip(tokens, payloads, strict=True)
    ]
    shape = {
        "schema_version": SCHEMA_VERSION,
        "cases": [
            {
                "case_token": tokens[0],
                "candidates": [
                    {
                        "ansatz": "u(...)=parameterized named-Atom composition",
                        "atom": "short Atom-family description",
                        "expression_template": "executable right-hand side",
                        "initial_values": [1.0],
                        "reason": "calculation from supplied public fields",
                    }
                ],
            }
        ],
    }
    user = (
        f"Generate exactly {candidate_count} ranked candidates for every case token. "
        "Every case item must have exactly case_token and candidates. Every candidate "
        "must have exactly ansatz, atom, expression_template, initial_values, and reason. "
        "Do not include a benchmark ID or any additional key. Set schema_version exactly "
        f"to {SCHEMA_VERSION}.\n\n"
        + json.dumps(
            {"input_schema": "ConditionAwarePublicProblemBatchV1", "cases": request_cases},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n\nRequired output shape (placeholder values only):\n"
        + json.dumps(shape, ensure_ascii=False, separators=(",", ":"))
    )
    messages = (
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    )
    return tokens, messages, payloads


def _parameter_names(expression: sp.Expr) -> tuple[str, ...]:
    indexed: list[tuple[int, str]] = []
    for symbol in expression.free_symbols:
        match = _PARAMETER.fullmatch(str(symbol))
        if match:
            indexed.append((int(match.group(1)), str(symbol)))
    indexed.sort()
    names = tuple(name for _, name in indexed)
    if names != tuple(f"p{index}" for index in range(1, len(names) + 1)):
        raise LLMResponseError("candidate parameters must be consecutive p1..pN")
    return names


def _validate_candidate(raw: object, payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != {
        "ansatz",
        "atom",
        "expression_template",
        "initial_values",
        "reason",
    }:
        raise LLMResponseError("concrete candidate fields differ")
    text_fields = ("ansatz", "atom", "expression_template", "reason")
    if any(not isinstance(raw[key], str) or not raw[key].strip() for key in text_fields):
        raise LLMResponseError("concrete candidate text is empty")
    coordinates = tuple(map(str, payload["independent_variables"]))
    names = {name: sp.Symbol(name, real=True) for name in coordinates}
    names["pi"] = sp.pi
    try:
        expression = parse_symbolic_expression(str(raw["expression_template"]), names=names)
    except SafeSymbolicError as exc:
        raise LLMResponseError(f"concrete candidate syntax invalid: {exc}") from exc
    parameters = _parameter_names(expression)
    allowed = {names[name] for name in coordinates} | {sp.Symbol(name) for name in parameters}
    unresolved = expression.free_symbols - allowed
    if unresolved:
        raise LLMResponseError(
            "concrete candidate has unresolved symbols: " + ", ".join(sorted(map(str, unresolved)))
        )
    initial = raw["initial_values"]
    if (
        not isinstance(initial, Sequence)
        or isinstance(initial, (str, bytes))
        or len(initial) != len(parameters)
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in initial)
        or any(not math.isfinite(float(value)) for value in initial)
    ):
        raise LLMResponseError("candidate initial_values differ from its p1..pN parameters")
    if not parameters:
        initial = []
    return {
        "ansatz": raw["ansatz"].strip(),
        "atom": raw["atom"].strip(),
        "expression_template": sp.sstr(expression),
        "parameters": list(parameters),
        "initial_values": [float(value) for value in initial],
        "reason": raw["reason"].strip(),
    }


def generate_condition_aware_solution_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    provider: LLMBatchProviderV1,
    candidate_count: int = 4,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Call the LLM once, then fail closed on invalid executable candidates."""

    tokens, messages, payloads = render_condition_aware_solution_messages(
        rows, candidate_count=candidate_count
    )
    completion = provider.complete(tokens=tokens, messages=messages, response_format={})
    response = completion.response
    if not isinstance(response, Mapping) or set(response) != {"schema_version", "cases"}:
        raise LLMResponseError("condition-aware solution response fields differ")
    if response["schema_version"] != SCHEMA_VERSION:
        raise LLMResponseError("condition-aware solution schema differs")
    cases = response["cases"]
    if (
        not isinstance(cases, Sequence)
        or isinstance(cases, (str, bytes))
        or len(cases) != len(tokens)
    ):
        raise LLMResponseError("condition-aware solution case count differs")
    by_token: dict[str, Mapping[str, Any]] = {}
    for case in cases:
        if not isinstance(case, Mapping) or set(case) != {"case_token", "candidates"}:
            raise LLMResponseError("condition-aware solution case fields differ")
        token = case["case_token"]
        if not isinstance(token, str) or token not in tokens or token in by_token:
            raise LLMResponseError("condition-aware solution token invalid")
        by_token[token] = case
    if set(by_token) != set(tokens):
        raise LLMResponseError("condition-aware solution token set differs")

    results: list[dict[str, Any]] = []
    for token, payload in zip(tokens, payloads, strict=True):
        raw_candidates = by_token[token]["candidates"]
        if (
            not isinstance(raw_candidates, Sequence)
            or isinstance(raw_candidates, (str, bytes))
            or len(raw_candidates) != candidate_count
        ):
            raise LLMResponseError("condition-aware solution candidate count differs")
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for ordinal, raw in enumerate(raw_candidates, start=1):
            try:
                candidate = _validate_candidate(raw, payload)
                key = candidate["expression_template"]
                if key in seen:
                    raise LLMResponseError("duplicate concrete expression")
                seen.add(key)
                candidate["generation_rank"] = ordinal
                accepted.append(candidate)
            except LLMResponseError as exc:
                rejected.append({"generation_rank": ordinal, "reason": str(exc), "raw": raw})
        results.append(
            {
                "case_token": token,
                "accepted_candidates": accepted,
                "rejected_candidates": rejected,
                "raw_candidate_count": len(raw_candidates),
            }
        )
    evidence = {
        "provider_name": completion.provider_name,
        "requested_model": completion.requested_model,
        "returned_model": completion.returned_model,
        "provider_response_id": completion.provider_response_id,
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
        "request_payload_sha256": completion.request_payload_sha256,
        "raw_response_sha256": completion.raw_response_sha256,
        "response_mode": completion.response_mode,
        "logical_provider_requests": 1,
        "provider_errors": 0,
        "non_llm_fallback_used": False,
        "public_payload_sha256": _sha256(payloads),
    }
    return tuple(results), evidence


def _grid_points(
    variables: Sequence[str],
    bounds: Mapping[str, Sequence[float]],
    counts: Mapping[str, int],
    *,
    midpoint: bool,
) -> tuple[tuple[float, ...], ...]:
    import numpy as np

    axes: list[tuple[float, ...]] = []
    for variable in variables:
        lower, upper = map(float, bounds[variable])
        count = int(counts[variable])
        if count < 2 or not lower < upper:
            raise ValueError("invalid public validation grid")
        if midpoint:
            step = (upper - lower) / count
            values = lower + (np.arange(count, dtype=np.float64) + 0.5) * step
        else:
            values = np.linspace(lower, upper, count, dtype=np.float64)
        axes.append(tuple(map(float, values)))
    return tuple(itertools.product(*axes))


def _operator_residual(expression: sp.Expr, spec: Mapping[str, Any]) -> sp.Expr:
    coordinates = {name: sp.Symbol(name, real=True) for name in spec["variables"]}
    names: dict[str, object] = {**coordinates, "u": expression, "pi": sp.pi}
    for alias, derivative in spec["derivatives"].items():
        variable = coordinates[str(derivative["variable"])]
        names[str(alias)] = sp.diff(expression, variable, int(derivative["order"]))
    return parse_symbolic_expression(str(spec["operator_residual"]), names=names)


def _residual_terms(
    expression: sp.Expr, coefficients: tuple[sp.Symbol, ...], spec: Mapping[str, Any], *, split: str
) -> tuple[CollocationResidualV1, ...]:
    del coefficients  # Residual compilation infers them later; retained for call-site clarity.
    coordinates = {name: sp.Symbol(name, real=True) for name in spec["variables"]}
    grid = spec[split]
    terms: list[CollocationResidualV1] = []
    operator_variables = tuple(str(name) for name in spec["variables"])
    terms.append(
        CollocationResidualV1(
            "pde",
            _operator_residual(expression, spec),
            tuple(coordinates[name] for name in operator_variables),
            _grid_points(
                operator_variables,
                spec["bounds"],
                grid["operator_counts"],
                midpoint=bool(grid["midpoint"]),
            ),
            float(spec.get("operator_weight", 1.0)),
        )
    )
    for index, condition in enumerate(spec["conditions"], start=1):
        assignments = {
            coordinates[str(name)]: float(value)
            for name, value in condition.get("assign", {}).items()
        }
        residual = expression.subs(assignments) - parse_symbolic_expression(
            str(condition["target"]), names={**coordinates, "pi": sp.pi}
        ).subs(assignments)
        sample_variables = tuple(map(str, condition["sample_variables"]))
        terms.append(
            CollocationResidualV1(
                f"condition_{index}",
                residual,
                tuple(coordinates[name] for name in sample_variables),
                _grid_points(
                    sample_variables,
                    spec["bounds"],
                    {name: int(grid["condition_count"]) for name in sample_variables},
                    midpoint=bool(grid["midpoint"]),
                ),
                float(condition.get("weight", 1.0)),
            )
        )
    return tuple(terms)


def _evaluate_terms(
    terms: Sequence[CollocationResidualV1], expression: sp.Expr
) -> tuple[float, tuple[tuple[str, float], ...]]:
    import numpy as np

    losses: list[tuple[str, float]] = []
    for term in terms:
        unresolved = term.residual.free_symbols - set(term.variables)
        if unresolved:
            raise ValueError("validation residual retains fitted parameters")
        if term.residual.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            raise ValueError("validation residual is symbolically nonfinite")
        points = np.asarray(term.points, dtype=np.float64)
        columns = tuple(points[:, index] for index in range(len(term.variables)))
        try:
            evaluator = sp.lambdify(term.variables, term.residual, modules="numpy")
            with np.errstate(all="ignore"):
                raw_values = np.asarray(evaluator(*columns))
        except (KeyError, NameError, NotImplementedError, TypeError, ValueError) as exc:
            raise ValueError("validation residual is not numerically evaluable") from exc
        if np.iscomplexobj(raw_values):
            raise ValueError("validation residual is complex")
        values = np.asarray(raw_values, dtype=np.float64)
        if values.ndim == 0:
            values = np.full(len(term.points), float(values), dtype=np.float64)
        values = np.asarray(np.broadcast_to(values, (len(term.points),)), dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("validation residual is nonfinite")
        with np.errstate(over="ignore", invalid="ignore"):
            mean_square = float(np.mean(np.square(values)))
        if not np.isfinite(mean_square):
            raise ValueError("validation residual mean square is nonfinite")
        losses.append((term.name, term.weight * mean_square))
    return sum(value for _, value in losses), tuple(losses)


def fit_and_select_public_candidates(
    generated_cases: Sequence[Mapping[str, Any]],
    case_ids: Sequence[str],
    validation_specs: Mapping[str, Mapping[str, Any]],
    *,
    config: BFGSConfigV1 | None = None,
) -> tuple[dict[str, Any], ...]:
    """Fit every LLM candidate and select solely by disjoint public validation loss."""

    if len(generated_cases) != len(case_ids):
        raise ValueError("generated case and case-id counts differ")
    selected_config = config or BFGSConfigV1(
        max_iterations=800,
        gradient_tolerance=1.0e-9,
        restarts=6,
        random_seed=20260903,
        restart_scale=0.5,
        acceptance_loss=1.0e-8,
    )
    outputs: list[dict[str, Any]] = []
    for generated, case_id in zip(generated_cases, case_ids, strict=True):
        spec = validation_specs[case_id]
        coordinates = {name: sp.Symbol(name, real=True) for name in spec["variables"]}
        fitted_rows: list[dict[str, Any]] = []
        for candidate in generated["accepted_candidates"]:
            try:
                expression = parse_symbolic_expression(
                    str(candidate["expression_template"]),
                    names={**coordinates, "pi": sp.pi},
                )
                coefficients = tuple(sp.Symbol(name) for name in candidate["parameters"])
                if coefficients:
                    training = _residual_terms(expression, coefficients, spec, split="training")
                    fit = fit_expression_coefficients_bfgs(
                        expression,
                        coefficients,
                        training,
                        config=selected_config,
                        initial_guesses=(tuple(candidate["initial_values"]),),
                    )
                    fitted_expression = fit.expression
                    fit_mapping: dict[str, Any] = fit.to_mapping()
                else:
                    fitted_expression = expression
                    training = _residual_terms(expression, (), spec, split="training")
                    training_loss, training_terms = _evaluate_terms(training, expression)
                    fit_mapping = {
                        "schema_version": "NoCoefficientFitV1",
                        "accepted": training_loss <= selected_config.acceptance_loss,
                        "expression": sp.sstr(expression),
                        "coefficient_assignments": {},
                        "final_loss": training_loss,
                        "term_losses": dict(training_terms),
                        "restart_count": 0,
                    }
                validation = _residual_terms(fitted_expression, (), spec, split="validation")
                validation_loss, validation_terms = _evaluate_terms(validation, fitted_expression)
                fitted_rows.append(
                    {
                        **candidate,
                        "fit": fit_mapping,
                        "fitted_expression": sp.sstr(fitted_expression),
                        "validation_loss": validation_loss,
                        "validation_term_losses": dict(validation_terms),
                    }
                )
            except (ValueError, TypeError, SafeSymbolicError, OverflowError) as exc:
                fitted_rows.append({**candidate, "fit_error": f"{type(exc).__name__}: {exc}"})
        valid = [row for row in fitted_rows if "validation_loss" in row]
        valid.sort(key=lambda row: (float(row["validation_loss"]), int(row["generation_rank"])))
        outputs.append(
            {
                "case_id": case_id,
                "status": "selected" if valid else "no_valid_fit",
                "selection_policy": (
                    "minimum disjoint public validation weighted MSE; generation rank tie-break"
                ),
                "selected": valid[0] if valid else None,
                "all_fits": fitted_rows,
                "generated_candidate_count": generated["raw_candidate_count"],
                "host_accepted_candidate_count": len(generated["accepted_candidates"]),
                "host_rejected_candidate_count": len(generated["rejected_candidates"]),
                "fit_error_count": sum("fit_error" in row for row in fitted_rows),
            }
        )
    return tuple(outputs)


__all__ = [
    "SCHEMA_VERSION",
    "fit_and_select_public_candidates",
    "generate_condition_aware_solution_candidates",
    "public_condition_problem_payload",
    "render_condition_aware_solution_messages",
]
