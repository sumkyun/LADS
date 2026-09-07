"""PDE-derived, certificate-first Ansatz discovery."""
from .models import (
    BranchCoverageV1,
    CandidateVerificationV1,
    CoefficientDependenceResultV1,
    ConditionCompatibilityV1,
    ConditionRecordV1,
    ConditionStatusV1,
    DependenceStatusV1,
    DiscoveryBudgetV1,
    DiscoveryCandidateV1,
    DiscoveryContractError,
    DiscoveryProblemInputV1,
    DiscoveryResultV1,
    ParameterBranchV1,
    ProofStatusV1,
    ReductionCertificateV1,
    ReductionChartV1,
    VerificationStatusV1,
    WorkerStatusV1,
    aggregate_condition_status,
)
from .serialization import (
    CanonicalDataError,
    canonical_json_bytes,
    canonical_sha256,
    freeze_json,
    thaw_json,
)


def discover_affine_lie_reductions(
    problem: DiscoveryProblemInputV1,
    *,
    top_k: int = 5,
):
    """Lazily load benchmark-blind affine Lie reduction discovery."""

    from .cas.affine_lie import discover_affine_lie_reductions as _discover

    return _discover(problem, top_k=top_k)


def discover_ansatz_reductions(
    problem: DiscoveryProblemInputV1,
    budget: DiscoveryBudgetV1,
    top_k: int,
) -> DiscoveryResultV1:
    """Lazily load the CAS-backed API while keeping contract-only imports lightweight."""

    from .api import discover_ansatz_reductions as _discover

    return _discover(problem, budget, top_k)

__all__ = [
    "BranchCoverageV1",
    "CandidateVerificationV1",
    "CanonicalDataError",
    "CoefficientDependenceResultV1",
    "ConditionCompatibilityV1",
    "ConditionRecordV1",
    "ConditionStatusV1",
    "DependenceStatusV1",
    "DiscoveryBudgetV1",
    "DiscoveryCandidateV1",
    "DiscoveryContractError",
    "DiscoveryProblemInputV1",
    "DiscoveryResultV1",
    "ParameterBranchV1",
    "ProofStatusV1",
    "ReductionCertificateV1",
    "ReductionChartV1",
    "VerificationStatusV1",
    "WorkerStatusV1",
    "aggregate_condition_status",
    "canonical_json_bytes",
    "canonical_sha256",
    "discover_affine_lie_reductions",
    "discover_ansatz_reductions",
    "freeze_json",
    "thaw_json",
]
