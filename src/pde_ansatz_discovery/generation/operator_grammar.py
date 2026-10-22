"""PDE-derived Phase-3 operator terminals and direct candidates."""

from __future__ import annotations

from dataclasses import dataclass

from ..analysis import analyze_scaling, analyze_translations
from ..generation.direct import DirectAnsatzV1
from ..ir import ParsedPDEProblemV1
from ..models import DiscoveryProblemInputV1
from ..solving.characteristics import scaling_ansatz, traveling_ansatz
from .singular import singular_scaling_ansatz


@dataclass(frozen=True)
class GeneratedDirectCandidatesV1:
    candidates: tuple[DirectAnsatzV1, ...]
    failures: tuple[str, ...]


def generate_translation_scaling_candidates(
    problem: DiscoveryProblemInputV1,
    parsed: ParsedPDEProblemV1,
) -> GeneratedDirectCandidatesV1:
    failures: list[str] = []
    candidates: list[DirectAnsatzV1] = []
    parameters = tuple(problem.parameters)

    scaling = analyze_scaling(
        parsed.equations[0].residual,
        independent_variables=problem.independent_variables,
        field=problem.dependent_variables[0],
    )
    scaling_candidate = scaling_ansatz(scaling, problem_parameters=parameters)
    if scaling_candidate is None:
        failures.append(scaling.reason_code)
    else:
        candidates.append(scaling_candidate)
        singular_candidate = singular_scaling_ansatz(
            scaling, problem_parameters=parameters
        )
        if singular_candidate is not None:
            candidates.append(singular_candidate)

    translation = analyze_translations(
        tuple(item.residual for item in parsed.equations), problem.independent_variables
    )
    traveling_candidate = traveling_ansatz(
        translation,
        field=problem.dependent_variables[0],
        problem_parameters=parameters,
    )
    if traveling_candidate is None:
        failures.append("TRANSLATION_EXPLICIT_COORDINATE_DEPENDENCE")
    else:
        candidates.append(traveling_candidate)

    seen: set[str] = set()
    unique: list[DirectAnsatzV1] = []
    for candidate in candidates:
        if candidate.definition_hash not in seen:
            seen.add(candidate.definition_hash)
            unique.append(candidate)
    return GeneratedDirectCandidatesV1(tuple(unique), tuple(sorted(failures)))


__all__ = ["GeneratedDirectCandidatesV1", "generate_translation_scaling_candidates"]
