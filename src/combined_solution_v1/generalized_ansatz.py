"""Benchmark-blind Ansatz discovery through affine Lie determining equations.

The production path does not contain completed Ansatz templates or a PDE-family
classifier. Candidates are emitted only after the copied symbolic engine proves a
point symmetry, integrates its characteristics, and replays the coordinate reduction.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pde_ansatz_discovery import (
    DiscoveryProblemInputV1,
    discover_affine_lie_reductions,
)

from .structure_adapter import normalized_pde_equations

_DISCOVERY_TRACE = (
    "AFFINE_GENERATOR_BASIS",
    "JET_PROLONGATION",
    "DETERMINING_IDENTITIES_PROVED",
    "CHARACTERISTICS_SOLVED",
    "REDUCTION_REPLAY_PROVED",
)
_IDENTIFIER = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\b")
_FUNCTION = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s*(?=\()")
_OPERATOR_NAMES = frozenset(
    {
        "Abs",
        "Delta",
        "abs",
        "cos",
        "d_t",
        "d_x",
        "d_y",
        "d_z",
        "exp",
        "log",
        "max",
        "min",
        "sin",
        "sqrt",
        "tan",
    }
)


@dataclass(frozen=True)
class GeneralizedAnsatzCandidateV1:
    rank: int
    candidate_id: str
    family: str
    template: str
    score: int
    evidence: tuple[str, ...]
    validation_kind: str
    validation_passed: bool
    generator: str
    invariants: tuple[str, ...]
    determining_equations: tuple[str, ...]
    reduced_equations: tuple[str, ...]
    free_parameters: tuple[str, ...]
    side_conditions: tuple[str, ...]
    symmetry_dimension: int
    generator_basis_degree: int

    def to_mapping(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "candidate_id": self.candidate_id,
            "atom_free_family": self.family,
            "ansatz": self.template,
            "score": self.score,
            "evidence": list(self.evidence),
            "validation_kind": self.validation_kind,
            "independent_structure_replay": self.validation_passed,
            "generator": self.generator,
            "characteristic_invariants": list(self.invariants),
            "determining_equations": list(self.determining_equations),
            "reduced_equations": list(self.reduced_equations),
            "free_parameters": list(self.free_parameters),
            "side_conditions": list(self.side_conditions),
            "symmetry_dimension": self.symmetry_dimension,
            "generator_basis_degree": self.generator_basis_degree,
        }


def _symbolic_parameter_contract(row: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Keep symbolic declarations while discarding all supplied parameter values."""

    raw = row.get("parameters", {})
    if not isinstance(raw, Mapping):
        raise ValueError("parameters must be a mapping")
    result: dict[str, dict[str, str]] = {}
    for name, description in raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError("parameter names must be nonempty strings")
        kind = description.get("kind") if isinstance(description, Mapping) else None
        result[name] = {"kind": str(kind)} if kind in {"function", "coefficient_function"} else {}
    declared = {
        *map(str, row.get("dependent_variables", ())),
        *map(str, row.get("independent_variables", ())),
    }
    for source in normalized_pde_equations(dict(row)):
        function_names = set(_FUNCTION.findall(source))
        for name in function_names:
            if name not in declared and name not in _OPERATOR_NAMES | {"D", "Derivative"}:
                result[name] = {"kind": "function"}
        for name in _IDENTIFIER.findall(source):
            if name in declared or name in _OPERATOR_NAMES | {"D", "Derivative"}:
                continue
            if any(
                name == field or name.startswith(f"{field}_")
                for field in row.get("dependent_variables", ())
            ):
                continue
            result.setdefault(name, {})
    return result


def _problem_from_public_input(row: Mapping[str, Any]) -> DiscoveryProblemInputV1:
    required = {"dependent_variables", "independent_variables", "pde"}
    if not required.issubset(row):
        raise ValueError("generalized Ansatz input is missing required public fields")
    return DiscoveryProblemInputV1(
        dependent_variables=tuple(row["dependent_variables"]),
        independent_variables=tuple(row["independent_variables"]),
        pde=normalized_pde_equations(dict(row)),
        domain={},
        parameters=_symbolic_parameter_contract(row),
        initial_conditions=(),
        boundary_conditions=(),
        additional_constraints=(),
    )


def recommend_generalized_ansatz(
    row: Mapping[str, Any], *, top_k: int = 5
) -> tuple[GeneralizedAnsatzCandidateV1, ...]:
    """Return only symbolically certified Ansatz reductions, with no result padding."""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if "..." in row.get("independent_variables", ()):
        return ()
    problem = _problem_from_public_input(row)
    reductions = discover_affine_lie_reductions(problem, top_k=top_k)
    return tuple(
        GeneralizedAnsatzCandidateV1(
            rank=rank,
            candidate_id=item.candidate_id,
            family="lie_point_symmetry",
            template=item.ansatz,
            score=item.symmetry_dimension,
            evidence=_DISCOVERY_TRACE,
            validation_kind="desolv_affine_lie_reduction_replay",
            validation_passed=item.validation_passed,
            generator=item.generator,
            invariants=item.invariants,
            determining_equations=item.determining_equations,
            reduced_equations=item.reduced_equations,
            free_parameters=item.free_parameters,
            side_conditions=item.side_conditions,
            symmetry_dimension=item.symmetry_dimension,
            generator_basis_degree=item.basis_degree,
        )
        for rank, item in enumerate(reductions, start=1)
    )


__all__ = ["GeneralizedAnsatzCandidateV1", "recommend_generalized_ansatz"]
