from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lads.coefficient_fitting import (  # noqa: E402
    CollocationResidualV1,
    LBFGSBConfigV1,
    fit_expression_coefficients_lbfgsb,
)
from lads.llm_provider import InMemoryLLMBatchProviderV1  # noqa: E402
from lads.permissive_direct import run_staged_experiment_direct  # noqa: E402


def test_public_inputs_are_complete() -> None:
    problems = json.loads((ROOT / "data/problems.json").read_text(encoding="utf-8"))
    validation = json.loads((ROOT / "data/validation.json").read_text(encoding="utf-8"))
    ids = {str(row["benchmark_id"]) for row in problems["cases"]}
    assert ids == set(validation["cases"])
    assert len(ids) == 9


def test_cli_dry_run() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--dry-run", "--case-ids", "burgers"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {"case_count": 1, "cases": ["burgers"]}


def test_coefficient_fit_uses_limited_memory_bfgs() -> None:
    coefficient, x = sp.symbols("coefficient x")
    fit = fit_expression_coefficients_lbfgsb(
        coefficient * x,
        (coefficient,),
        (
            CollocationResidualV1(
                "target",
                coefficient * x - 2.0,
                (x,),
                ((1.0,),),
            ),
        ),
        config=LBFGSBConfigV1(max_iterations=50, restarts=1),
    )
    assert fit.optimizer_converged
    assert fit.accepted
    assert abs(dict(fit.coefficient_assignments)["coefficient"] - 2.0) < 1.0e-6


def test_model_path_runs_without_network() -> None:
    row = {
        "benchmark_id": "smoke",
        "dependent_variables": ["u"],
        "independent_variables": ["x", "t"],
        "pde": ["u_t-u_xx=0"],
        "initial_conditions": [],
        "boundary_conditions": [],
        "domain": {},
        "parameters": {},
    }
    stage_i = {
        "schema_version": "LLMAnsatzBatchResponseV1",
        "cases": [{
            "case_token": "case_001",
            "candidates": [
                {"ansatz": "u(x,t)=F1(x,t)", "reason": "full profile"},
                {"ansatz": "u(x,t)=F1(x-t)", "reason": "traveling profile"},
                {"ansatz": "u(x,t)=F1(x)*F2(t)", "reason": "separated profile"},
            ],
        }],
    }
    atoms = {"proposals": [{"canonical_template": template} for template in (
        "sin(ARG1)",
        "p1*cos(ARG1)+p2",
        "p1*tanh(p2*ARG1+p3)+p4",
        "p1*exp(p2*ARG1)+p3",
        "p1*ARG1+p2",
        "p1*ARG1**2+p2",
        "p1/(1+p2*ARG1**2)+p3",
        "p1*sinh(ARG1)+p2",
    )]}

    def provider_factory(max_tokens: int) -> InMemoryLLMBatchProviderV1:
        return InMemoryLLMBatchProviderV1(
            stage_i if max_tokens == 8192 else atoms,
            model="lads-smoke",
        )

    result = run_staged_experiment_direct(
        [row], {}, provider_factory=provider_factory, requested_model="lads-smoke"
    )
    assert result["cases"][0]["assembled_candidates"]
