"""Run LADS on one PDE problem with at most eight LLM calls and four coefficient fits.

Usage:
  export OPENROUTER_API_KEY=...
  python run_lads.py problems/case_001.json --output runs/case_001
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from tools import pilot_lads  # noqa: E402
from tools.pilot_public import row_from_public  # noqa: E402
from tools.pilot_transport import MODEL, AuditedProvider  # noqa: E402

REQUEST_CAP = 8  # LLM calls per problem, shared by Ansatz generation and slot completion
FIT_CAP = 4  # distinct coefficient fits per problem
CONFIG = {**pilot_lads.CONFIG, "MAX_SEMANTIC_REGENERATION_ROUNDS": 1,
          "MAX_SEMANTIC_FITS_PER_CASE": FIT_CAP}


class RequestCapReached(BaseException):
    """Stops the search without being caught by the method's own error handling."""


def run(problem: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    row = row_from_public(problem)
    counts = {"stage_i": 0, "stage_ii": 0}

    def provider_factory(max_tokens: int):
        class Provider:
            def complete(self, *, tokens, messages, response_format):
                if sum(counts.values()) >= REQUEST_CAP:
                    raise RequestCapReached()
                stage = pilot_lads.classify_stage(messages[0]["content"])
                counts[stage] += 1
                path = output / stage / f"request_{counts[stage]:03d}"
                return AuditedProvider(path, max_output_tokens=max_tokens).complete(
                    tokens=tokens, messages=messages, response_format=response_format)
        return Provider()

    previous = dict(pilot_lads.CONFIG)
    pilot_lads.CONFIG.clear()
    pilot_lads.CONFIG.update(CONFIG)
    try:
        case = pilot_lads.run_case(row, provider_factory, MODEL,
                                   optimizer_artifacts=output / "fits")["cases"][0]
        status, selected = case.get("status"), case.get("selected")
    except RequestCapReached:
        # Return the fitted candidate with the lowest validation loss seen so far.
        fitted = []
        for path in sorted((output / "fits").glob("fit_*_output.json")):
            record = json.loads(path.read_text())
            loss = record.get("validation_loss")
            if isinstance(loss, (int, float)) and math.isfinite(loss) \
                    and record.get("fitted_field_expressions"):
                fitted.append(record)
        selected = min(fitted, key=lambda r: r["validation_loss"], default=None)
        status = "request_cap_reached" + ("" if selected else "_without_fit")
    finally:
        pilot_lads.CONFIG.clear()
        pilot_lads.CONFIG.update(previous)

    result = {
        "problem": problem.stem,
        "status": status,
        "llm_calls": counts,
        "prediction": selected.get("fitted_field_expressions") if selected else None,
        "validation_loss": selected.get("validation_loss") if selected else None,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("problem", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or ROOT / "runs" / args.problem.stem
    print(json.dumps(run(args.problem, output), indent=2))
