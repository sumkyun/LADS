"""One shared CUDA FP64 L-BFGS policy for LADS and adapted upstream SIGS.

This module fits already-generated expressions only. It never proposes solutions,
loads a reference, or provides a CPU optimizer fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict
from pathlib import Path

import sympy as sp

from care_pde.fitting.cuda_lbfgs import (
    CudaInfrastructureError,
    CudaLBFGSConfig,
    cuda_runtime_info,
    evaluate_residual_terms_cuda,
    fit_expression_coefficients_cuda_lbfgs,
)
from combined_solution_v1.safe_symbolic import is_abstract_coefficient_name
from combined_solution_v1.staged_examples import _real_expression
from tools.pilot_public import field_residual_terms, math_expr, validation_spec

CONFIG = asdict(
    CudaLBFGSConfig(
        max_iterations=300,
        gradient_tolerance=1e-8,
        restarts=6,
        random_seed=20260904,
        restart_scale=1.0,
        acceptance_loss=1e-8,
    )
)
OPTIMIZER_POLICY = {
    "id": "SHARED-CUDA-FP64-LBFGS-v1",
    "implementation": "care_pde.fitting.cuda_lbfgs.fit_expression_coefficients_cuda_lbfgs",
    "config": CONFIG,
    "initialization": "all ones, then identical seeded Gaussian restarts from shared fitter",
    "training_midpoints_per_axis": 3,
    "validation_midpoints_per_axis": 4,
    "max_coefficients": 24,
    "max_distinct_fits_per_case": 4,
    "max_field_top_level_additive_terms": 4,
    "max_field_AST_nodes": 128,
    "cpu_fallback": False,
    "environment": ".venv (same Python/JAX/Optax environment for both methods)",
}


def finite_metrics(value):
    """JSON null denotes nonfinite diagnostic metrics, never a successful loss."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_metrics(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_metrics(item) for item in value]
    return value


def parse_fields(row, field_expressions):
    if set(field_expressions) != set(row["dependent_variables"]):
        raise ValueError("dependent field census mismatch")
    public_parameters = {"pi": sp.pi, "E": sp.E}
    for name, value in row.get("parameters", {}).items():
        # Use the same public arithmetic interpretation as the PDE/IC/BC compiler.
        # Declared constants must never become unknown optimizer coefficients.
        resolved = math_expr(value, public_parameters)
        if resolved.free_symbols or resolved.is_real is not True or resolved.is_finite is not True:
            raise ValueError("public parameter must be a finite real constant: " + name)
        public_parameters[name] = resolved
    expressions = {}
    parameters = set()
    for field, source in field_expressions.items():
        expression, coefficients = _real_expression(str(source), row)
        expression = expression.xreplace(
            {
                symbol: public_parameters[str(symbol)]
                for symbol in expression.free_symbols
                if str(symbol) in row.get("parameters", {})
            }
        )
        coefficients = tuple(symbol for symbol in coefficients if symbol in expression.free_symbols)
        if len(sp.Add.make_args(expression)) > 4:
            raise ValueError("candidate exceeds four top-level additive terms")
        if sum(1 for _ in sp.preorder_traversal(expression)) > 128:
            raise ValueError("candidate exceeds 128 expression AST nodes")
        for coefficient in coefficients:
            name = str(coefficient)
            if not (
                is_abstract_coefficient_name(name)
                or re.fullmatch(r"p[1-9]\d*|z[1-9]\d*_[1-9]\d*", name)
            ):
                raise ValueError("undeclared coordinate or unapproved coefficient name: " + name)
        expressions[field] = expression
        parameters.update(coefficients)
    coefficients = tuple(sorted(parameters, key=str))
    if len(coefficients) > 24:
        raise ValueError("candidate exceeds 24 coefficients")
    return expressions, coefficients


def fit_fields(row, field_expressions, config=None):
    if config is not None and dict(config) != CONFIG:
        raise ValueError("optimizer configuration must equal the shared frozen policy")
    expressions, coefficients = parse_fields(row, field_expressions)
    runtime = cuda_runtime_info()
    spec = validation_spec(row)
    training = field_residual_terms(expressions, coefficients, spec, split="training")
    if coefficients:
        fit = fit_expression_coefficients_cuda_lbfgs(
            next(iter(expressions.values())),
            coefficients,
            training,
            config=CudaLBFGSConfig(**CONFIG),
            initial_guesses=None,
            device_index=0,
        )
        fit_record = finite_metrics(fit.to_mapping())
        assignments = {
            sp.Symbol(name, real=True): sp.Float(value)
            for name, value in fit.coefficient_assignments
        }
        fitted = {field: expr.xreplace(assignments) for field, expr in expressions.items()}
    else:
        loss, terms = evaluate_residual_terms_cuda(training)
        fit_record = {
            "optimizer_updates_executed": False,
            "reason": "no free coefficients",
            "final_loss": loss,
            "term_losses": terms,
            "cuda": runtime,
        }
        fitted = expressions
    validation = field_residual_terms(fitted, (), spec, split="validation")
    loss, terms = evaluate_residual_terms_cuda(validation)
    if not math.isfinite(loss):
        raise ValueError("nonfinite fitted validation loss")
    result = {
        "fit": fit_record,
        "fitted_field_expressions": {field: sp.sstr(expr) for field, expr in fitted.items()},
        "validation_loss": loss,
        "validation_term_losses": terms,
        "optimizer_policy": OPTIMIZER_POLICY,
        "semantic_validation_passed": loss <= CONFIG["acceptance_loss"],
    }
    if len(fitted) == 1:
        result["fitted_expression"] = sp.sstr(next(iter(fitted.values())))
    return result


class FitSession:
    """Per-case fit cache with one global budget, identical for both methods."""

    def __init__(self, row, artifact_directory=None):
        self.row = row
        self.cache = {}
        self.artifact_directory = Path(artifact_directory) if artifact_directory else None

    def fit(self, fields):
        key = tuple(sorted((str(k), str(v)) for k, v in fields.items()))
        if key in self.cache:
            return {**self.cache[key], "fit_cache_hit": True}
        if len(self.cache) >= OPTIMIZER_POLICY["max_distinct_fits_per_case"]:
            return {
                "fit_error": "shared global distinct-fit budget exhausted",
                "budget_exhausted": True,
            }
        from tools.pilot_transport import write_json

        ordinal = len(self.cache) + 1
        if self.artifact_directory:
            write_json(
                self.artifact_directory / f"fit_{ordinal:02d}_input.json",
                {
                    "row": self.row,
                    "field_expressions": fields,
                    "optimizer_policy": OPTIMIZER_POLICY,
                },
            )
        try:
            result = fit_fields(self.row, fields)
        except CudaInfrastructureError:
            raise
        except (
            ValueError,
            TypeError,
            KeyError,
            NotImplementedError,
            OverflowError,
            ZeroDivisionError,
        ) as error:
            result = {"fit_error": type(error).__name__ + ": " + str(error)}
        self.cache[key] = result
        if self.artifact_directory:
            write_json(self.artifact_directory / f"fit_{ordinal:02d}_output.json", result)
        return {**result, "fit_cache_hit": False}

    def candidates(self, candidates):
        fitted = []
        for rank, candidate in enumerate(candidates, 1):
            record = {**candidate, "generation_rank": rank}
            record.update(self.fit(candidate["field_expressions"]))
            fitted.append(record)
        valid = [item for item in fitted if "validation_loss" in item]
        selected = (
            min(valid, key=lambda x: (x["validation_loss"], x["generation_rank"]))
            if valid
            else None
        )
        return tuple(fitted), selected


def main():
    from tools.pilot_transport import write_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text())
    if payload.get("config") != CONFIG:
        raise ValueError("worker optimizer config mismatch")
    session = FitSession(payload["row"], Path(args.output).with_suffix(".fits"))
    fitted, selected = session.candidates(payload["candidates"])
    write_json(
        args.output,
        {
            "fitted_candidates": fitted,
            "selected": selected,
            "optimizer_policy": OPTIMIZER_POLICY,
            "distinct_fits": len(session.cache),
        },
    )


if __name__ == "__main__":
    main()
