"""Explicit bounded pilot configuration over the original staged LADS runtime."""

from __future__ import annotations

from contextlib import contextmanager

import combined_solution_v1.staged_examples as staged
from care_pde.fitting.cuda_lbfgs import CudaInfrastructureError
from tools.pilot_optimizer import OPTIMIZER_POLICY, FitSession
from tools.pilot_public import field_residual_terms, validation_spec
from tools.pilot_transport import RETRY_POLICY

METHOD = "LADS-ANSATZ-ATOM-COMPOSITIONAL-SHAREDOPT-DEV-v6"
CONFIG = {
    "MAX_REGENERATION_ATTEMPTS": 2,
    "MAX_STAGE_II_REQUESTS": 32,
    "MAX_ATOM_COMBINATIONS": 32,
    "MAX_SEMANTIC_FITS_PER_ANSATZ": 2,
    "MAX_MULTISLOT_SEMANTIC_FITS_PER_ANSATZ": 2,
    "MAX_SEMANTIC_FITS_PER_CASE": 4,
    "MAX_SEMANTIC_REGENERATION_ROUNDS": 0,
}


def classify_stage(system_prompt):
    from combined_solution_v1.llm_ansatz import COMPOSITIONAL_SYSTEM_PROMPT, SYSTEM_PROMPT

    if system_prompt in (SYSTEM_PROMPT, COMPOSITIONAL_SYSTEM_PROMPT):
        return "stage_i"
    if system_prompt == staged.STAGE_II_SYSTEM_PROMPT:
        return "stage_ii"
    raise ValueError("unknown stage prompt; refuse misleading audit label")


@contextmanager
def pilot_configuration(session=None):
    previous = {name: getattr(staged, name) for name in CONFIG}
    old_terms = staged._field_residual_terms
    old_fit = staged.fit_scalar_candidates

    def shared_fit(row, candidates, validation_spec, *, coefficient_backend, cuda_device_index):
        if coefficient_backend != "cuda_lbfgs" or cuda_device_index != 0:
            raise CudaInfrastructureError("shared optimizer requires CUDA device zero")
        if row != session.row:
            raise ValueError("fit session is bound to another public case")
        return session.candidates(candidates)

    try:
        for name, value in CONFIG.items():
            setattr(staged, name, value)
        # This explicit public-condition adapter extends derivative/coupled IC/BC compilation.
        # Generation and proof gates remain unchanged; fitting has one common policy.
        staged._field_residual_terms = field_residual_terms
        if session is not None:
            staged.fit_scalar_candidates = shared_fit
        yield
    finally:
        for name, value in previous.items():
            setattr(staged, name, value)
        staged._field_residual_terms = old_terms
        staged.fit_scalar_candidates = old_fit


def run_case(row, provider_factory, requested_model, case_observer=None, optimizer_artifacts=None):
    with pilot_configuration(FitSession(row, optimizer_artifacts)):
        return staged.run_staged_experiment(
            (row,),
            {str(row["benchmark_id"]): validation_spec(row)},
            provider_factory=provider_factory,
            requested_model=requested_model,
            case_observer=case_observer,
            coefficient_backend="cuda_lbfgs",
            compositional_structure=True,
            transport_retry_policy=RETRY_POLICY,
        )


def main():
    import argparse
    import hashlib
    import json
    import traceback
    from pathlib import Path

    from tools.benchmark_runtime import (
        require_versioned_runtime_manifest,
        unified_problem_from_path,
        write_common_solution,
    )
    from tools.pilot_public import row_from_public
    from tools.pilot_transport import (
        MODEL,
        AuditedProvider,
        ProviderInfrastructureError,
        write_json,
    )

    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    unified_problem = unified_problem_from_path(a.case)
    row = row_from_public(a.case)
    method_manifest = json.loads((out.parent / "manifest.json").read_text())
    assert method_manifest["method_id"] == METHOD and method_manifest["config"] == CONFIG
    assert method_manifest["optimizer_policy"] == OPTIMIZER_POLICY
    require_versioned_runtime_manifest(
        method_manifest,
        unified_problem,
        executing_code_paths=frozenset({"tools/pilot_lads.py"}),
    )
    root = Path.cwd()
    for path, digest in method_manifest["code_hashes"].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, path
    counts = {"stage_i": 0, "stage_ii": 0}
    common_record = {
        "runtime_status": "method_runtime_error",
        "prediction": None,
        "validation_loss": None,
        "error": "runtime ended without a terminal record",
    }

    class StageProvider:
        def __init__(self, max_tokens):
            self.max_tokens = max_tokens

        def complete(self, *, tokens, messages, response_format):
            stage = classify_stage(messages[0]["content"])
            counts[stage] += 1
            return AuditedProvider(
                out / stage / f"request_{counts[stage]:03}", max_output_tokens=self.max_tokens
            ).complete(tokens=tokens, messages=messages, response_format=response_format)

    try:
        result = run_case(
            row,
            StageProvider,
            MODEL,
            case_observer=lambda c: write_json(out / "case_record.json", c),
            optimizer_artifacts=out / "optimizer_io",
        )
        write_json(out / "runtime_result.json", result)
        case = result["cases"][0]
        selected = case.get("selected")
        prediction = selected.get("fitted_field_expressions") if selected else None
        common_record = {
            "runtime_status": case["status"],
            "prediction": prediction,
            "validation_loss": selected.get("validation_loss") if selected else None,
            "error": None,
        }
        write_json(
            out / "prediction.json",
            {
                "case_id": row["benchmark_id"],
                "method_id": METHOD,
                "status": case["status"],
                "prediction": prediction,
                "request_counts": counts,
            },
        )
    except (ProviderInfrastructureError, CudaInfrastructureError) as e:
        runtime_status = (
            "provider_infrastructure_error"
            if isinstance(e, ProviderInfrastructureError)
            else "cuda_infrastructure_error"
        )
        common_record = {
            "runtime_status": runtime_status,
            "prediction": None,
            "validation_loss": None,
            "error": str(e),
        }
        write_json(
            out / "prediction.json",
            {
                "case_id": row["benchmark_id"],
                "method_id": METHOD,
                "status": runtime_status,
                "prediction": None,
                "error": str(e),
                "request_counts": counts,
            },
        )
    except Exception as e:
        traceback.print_exc()
        common_record = {
            "runtime_status": "method_runtime_error",
            "prediction": None,
            "validation_loss": None,
            "error": type(e).__name__ + ": " + str(e),
        }
        write_json(
            out / "prediction.json",
            {
                "case_id": row["benchmark_id"],
                "method_id": METHOD,
                "status": "method_runtime_error",
                "prediction": None,
                "error": type(e).__name__ + ": " + str(e),
                "request_counts": counts,
            },
        )
    if unified_problem is not None:
        try:
            write_common_solution(
                out / "solution.json",
                unified_problem,
                case_id=row["benchmark_id"],
                **common_record,
            )
        except ValueError as exc:
            # Preserve raw candidate artifacts; malformed output is not a valid candidate.
            write_json(
                out / "output_contract_error.json",
                {
                    "stage": "solution_serialization",
                    "error": str(exc),
                    "original_candidate_artifact": "prediction.json",
                },
            )
            write_common_solution(
                out / "solution.json",
                unified_problem,
                case_id=row["benchmark_id"],
                runtime_status="method_runtime_error",
                prediction=None,
                error=f"output contract: {exc}",
            )


if __name__ == "__main__":
    main()
