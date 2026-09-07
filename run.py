#!/usr/bin/env python3
"""Run the LADS Ansatz-to-Atom model on the released public problems."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from lads.llm_provider import OpenRouterLLMBatchProviderV1  # noqa: E402
from lads.permissive_direct import run_staged_experiment_direct  # noqa: E402


def load_inputs() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    problems = json.loads((ROOT / "data/problems.json").read_text(encoding="utf-8"))
    validation = json.loads((ROOT / "data/validation.json").read_text(encoding="utf-8"))
    rows = problems["cases"]
    specs = validation["cases"]
    if not isinstance(rows, list) or not isinstance(specs, dict):
        raise ValueError("invalid release input files")
    return rows, specs


def select_cases(
    rows: list[dict[str, Any]],
    specs: dict[str, dict[str, Any]],
    requested: list[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    by_id = {str(row["benchmark_id"]): row for row in rows}
    # The direct Atom runner is a scalar host-validation path.  Keep opaque coupled/system
    # rows in the public catalog for reference, but do not silently run them without a spec.
    case_ids = (
        [case_id for case_id in by_id if case_id in specs]
        if requested is None
        else requested
    )
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case IDs must be unique")
    missing = [case_id for case_id in case_ids if case_id not in by_id]
    if missing:
        raise ValueError(f"unknown case(s): {', '.join(missing)}")
    missing_specs = [case_id for case_id in case_ids if case_id not in specs]
    if missing_specs:
        raise ValueError(f"no scalar validation spec for case(s): {', '.join(missing_specs)}")
    selected_rows = [by_id[case_id] for case_id in case_ids]
    selected_specs = {case_id: specs[case_id] for case_id in case_ids}
    return selected_rows, selected_specs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-ids", nargs="+", help="cases to run (default: all public cases)")
    parser.add_argument("--model", default="google/gemini-2.5-flash")
    parser.add_argument("--credential-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--output", type=Path, default=ROOT / "results/predictions.json")
    parser.add_argument(
        "--dry-run", action="store_true", help="validate inputs without an API call"
    )
    args = parser.parse_args()

    rows, specs = load_inputs()
    rows, specs = select_cases(rows, specs, args.case_ids)
    if args.dry_run:
        print(json.dumps({"case_count": len(rows), "cases": [row["benchmark_id"] for row in rows]}))
        return 0

    if not os.environ.get(args.credential_env):
        parser.error(f"missing provider credential: {args.credential_env}")

    def provider_factory(max_output_tokens: int) -> OpenRouterLLMBatchProviderV1:
        return OpenRouterLLMBatchProviderV1(
            lambda: os.environ[args.credential_env],
            model=args.model,
            max_output_tokens=max_output_tokens,
            max_retries=1,
            temperature=0.0,
            top_p=1.0,
        )

    result = run_staged_experiment_direct(
        rows,
        specs,
        provider_factory=provider_factory,
        requested_model=args.model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output} ({len(rows)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
