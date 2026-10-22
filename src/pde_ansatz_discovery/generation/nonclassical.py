"""Bounded nonclassical x-invariant operator with genuine Q reduction."""

from __future__ import annotations

from dataclasses import replace

from ..cas.sympy_backend import SymPyBackendV1
from ..generation.direct import DirectAnsatzV1
from ..ir import SymbolRoleV1, parse_problem, rational
from ..models import (
    BranchCoverageV1,
    CandidateVerificationV1,
    ConditionCompatibilityV1,
    DiscoveryCandidateV1,
    DiscoveryProblemInputV1,
    ParameterBranchV1,
    VerificationStatusV1,
)
from ..serialization import canonical_sha256, thaw_json
from ..solving.determining_system import (
    verify_classical_generator,
    verify_nonclassical_generator,
)
from ..solving.prolongation import PointGeneratorV1
from ..verification.reduction import verify_direct_candidate


def _terminal_candidate(
    problem: DiscoveryProblemInputV1,
    status: VerificationStatusV1,
    reason: str,
    metadata: dict,
) -> DiscoveryCandidateV1:
    branch = ParameterBranchV1("generic", {}, (), (), BranchCoverageV1.EXHAUSTIVE)
    return DiscoveryCandidateV1(
        canonical_sha256({"kind": "NonclassicalXInvariantV1", "problem": problem.to_mapping()}),
        "u=F(t)",
        "t",
        ("F",),
        tuple(problem.parameters),
        branch,
        ("nonclassical:X=d_x", "invariant_surface:Q=-u_x", "verification:direct_v1"),
        () if status is VerificationStatusV1.CERTIFIED_VALID else (reason,),
        CandidateVerificationV1(status, reason, None),
        ConditionCompatibilityV1(()),
        "nonclassical:d_x",
        "t",
        f"nonclassical:{reason}",
        None,
        metadata,
    )


def verify_x_invariant_nonclassical(
    problem: DiscoveryProblemInputV1,
) -> DiscoveryCandidateV1:
    field = problem.dependent_variables[0]
    spatial, time = problem.independent_variables
    roles = {name: SymbolRoleV1.INDEPENDENT for name in problem.independent_variables}
    roles.update({name: SymbolRoleV1.PARAMETER for name in problem.parameters})
    backend = SymPyBackendV1(problem.independent_variables, roles)
    generator = PointGeneratorV1(
        rational(0), rational(1), rational(0), field, spatial, time
    )
    residual = parse_problem(problem).equations[0].residual
    classical = verify_classical_generator(residual, generator, backend=backend)
    conditional = verify_nonclassical_generator(residual, generator, backend=backend)
    proof_metadata = {
        "pipeline_version": "NonclassicalVerifierV1",
        "operator": f"d_{spatial}",
        "classical": {
            "invariant": classical.invariant,
            "reason_code": classical.reason_code,
            "determining": classical.reduced_determining_expression.to_mapping(),
        },
        "conditional": {
            "invariant": conditional.invariant,
            "reason_code": conditional.reason_code,
            "determining": conditional.reduced_determining_expression.to_mapping(),
            "invariant_surface": conditional.invariant_surface_condition.to_mapping(),
        },
    }
    if conditional.invariant is False:
        return _terminal_candidate(
            problem,
            VerificationStatusV1.CERTIFIED_INVALID,
            "NONCLASSICAL_NOT_INVARIANT",
            proof_metadata,
        )
    if conditional.invariant is None:
        return _terminal_candidate(
            problem,
            VerificationStatusV1.UNRESOLVED,
            "NONCLASSICAL_INVARIANCE_UNKNOWN",
            proof_metadata,
        )

    ansatz = DirectAnsatzV1(
        field,
        f"F({time})",
        "eta",
        time,
        "F",
        tuple(problem.parameters),
        {spatial: spatial, time: "eta"},
        spatial,
        family="nonclassical:x_invariant",
    )
    candidate = verify_direct_candidate(problem, ansatz)
    metadata = thaw_json(candidate.metadata)
    metadata.update(proof_metadata)
    return replace(
        candidate,
        production_trace=(
            "nonclassical:X=d_x",
            "invariant_surface:Q=-u_x",
            "conditional_determining:proved_zero",
            "verification:direct_v1",
        ),
        generator_key="nonclassical:d_x",
        metadata=metadata,
    )


__all__ = ["verify_x_invariant_nonclassical"]
