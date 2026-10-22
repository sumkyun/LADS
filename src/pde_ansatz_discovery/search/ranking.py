"""Weight-free lexicographic candidate ranking."""

from __future__ import annotations

from collections.abc import Mapping

from ..ir import ExpressionIRV1
from ..models import (
    ConditionStatusV1,
    DiscoveryCandidateV1,
    DiscoveryContractError,
    VerificationStatusV1,
)
from ..serialization import thaw_json

_VERIFICATION_ORDER = {
    VerificationStatusV1.CERTIFIED_VALID: 0,
    VerificationStatusV1.UNRESOLVED: 1,
    VerificationStatusV1.CERTIFIED_INVALID: 2,
}
_CONDITION_ORDER = {
    ConditionStatusV1.COMPATIBLE: 0,
    ConditionStatusV1.COMPATIBLE_WITH_CONSTRAINTS: 1,
    ConditionStatusV1.NO_CONDITION: 2,
    ConditionStatusV1.UNRESOLVED: 3,
    ConditionStatusV1.SINGULAR: 4,
    ConditionStatusV1.INCOMPATIBLE: 5,
}


def _equation_complexity(candidate: DiscoveryCandidateV1) -> tuple[int, int]:
    exact = thaw_json(candidate.metadata).get("certificate_ir")
    if not isinstance(exact, Mapping):
        raise DiscoveryContractError("ranked candidate is missing certificate_ir metadata")
    mappings = exact.get("reduced_equations") or [exact["reduced_equation"]]
    expressions = tuple(ExpressionIRV1.from_mapping(item) for item in mappings)
    return sum(item.node_count for item in expressions), len(expressions)


def _reduction_strength(candidate: DiscoveryCandidateV1) -> int:
    certificate = candidate.verification.certificate
    if certificate is None:
        return 0
    return 2 if certificate.old_variables_removed else 1


def _remaining_freedom(candidate: DiscoveryCandidateV1) -> int:
    assigned = set(candidate.branch.assignments)
    return len(set(candidate.free_parameters) - assigned) + (
        0 if candidate.branch.coverage.value == "exhaustive" else 1
    )


def reduction_family(candidate: DiscoveryCandidateV1) -> str:
    if "scaling" in candidate.generator_key:
        return "scaling"
    if "translation" in candidate.generator_key or "d_t+c*d_x" in candidate.generator_key:
        return "traveling"
    if "separation" in candidate.generator_key:
        return "separation"
    if "nonclassical" in candidate.generator_key:
        return "nonclassical"
    if "singular" in candidate.generator_key:
        return "singular"
    return candidate.generator_key.split(":", 1)[0]


def candidate_order_key(
    candidate: DiscoveryCandidateV1,
    *,
    seen_families: frozenset[str] = frozenset(),
) -> tuple[int, int, int, int, int, int, int, bytes, str]:
    nodes, equations = _equation_complexity(candidate)
    family = reduction_family(candidate)
    return (
        _VERIFICATION_ORDER[candidate.verification.status],
        _CONDITION_ORDER[candidate.condition_compatibility.overall],
        -_reduction_strength(candidate),
        equations,
        nodes,
        _remaining_freedom(candidate),
        1 if family in seen_families else 0,
        candidate.canonical_bytes,
        candidate.candidate_id,
    )


def rank_valid_candidates(
    candidates: tuple[DiscoveryCandidateV1, ...],
) -> tuple[DiscoveryCandidateV1, ...]:
    remaining = list(candidates)
    ranked: list[DiscoveryCandidateV1] = []
    seen: set[str] = set()
    while remaining:
        selected = min(
            remaining,
            key=lambda item: candidate_order_key(item, seen_families=frozenset(seen)),
        )
        remaining.remove(selected)
        ranked.append(selected)
        seen.add(reduction_family(selected))
    return tuple(ranked)


__all__ = ["candidate_order_key", "rank_valid_candidates", "reduction_family"]
