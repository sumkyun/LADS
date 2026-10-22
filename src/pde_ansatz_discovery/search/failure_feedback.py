"""Closed bounded failure-to-production transitions."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import DiscoveryBudgetV1

FAILURE_EXPANSIONS_V1: dict[str, tuple[str, ...]] = {
    "NO_SCALING_BALANCE": ("traveling", "separation"),
    "SCALING_BALANCE_UNDERDETERMINED": ("traveling", "separation"),
    "OLD_VARIABLE_REMAINS": ("amplitude", "offset"),
    "AMPLITUDE_MISMATCH": ("amplitude",),
    "COORDINATE_MISMATCH": ("traveling", "scaling", "singular"),
    "BOUNDARY_NOT_PRESERVED": ("boundary_preserving_generator", "boundary_lift"),
    "IC_INCOMPATIBLE": ("offset",),
    "CHARACTERISTIC_FAILED": ("direct_grammar",),
    "CAS_TIMEOUT": (),
}


@dataclass(frozen=True)
class FailureFeedbackStateV1:
    attempted_transitions: tuple[str, ...] = ()
    depth: int = 0


@dataclass(frozen=True)
class FailureFeedbackResultV1:
    expansions: tuple[str, ...]
    state: FailureFeedbackStateV1
    terminal_reason: str | None


def expand_failure(
    reason_code: str,
    state: FailureFeedbackStateV1,
    budget: DiscoveryBudgetV1,
) -> FailureFeedbackResultV1:
    if reason_code not in FAILURE_EXPANSIONS_V1:
        return FailureFeedbackResultV1((), state, "UNLISTED_FAILURE_NO_EXPANSION")
    if reason_code == "CAS_TIMEOUT":
        return FailureFeedbackResultV1((), state, "CAS_TIMEOUT_REMAINS_UNRESOLVED")
    if state.depth >= budget.grammar_depth:
        return FailureFeedbackResultV1((), state, "GRAMMAR_DEPTH_REACHED")
    attempted = set(state.attempted_transitions)
    available = tuple(
        item
        for item in FAILURE_EXPANSIONS_V1[reason_code]
        if f"{reason_code}->{item}" not in attempted
    )[: budget.beam_width]
    if not available:
        return FailureFeedbackResultV1((), state, "NO_NEW_TRANSITION")
    transitions = tuple(f"{reason_code}->{item}" for item in available)
    next_state = FailureFeedbackStateV1(
        tuple(sorted((*state.attempted_transitions, *transitions))), state.depth + 1
    )
    return FailureFeedbackResultV1(available, next_state, None)


__all__ = [
    "FAILURE_EXPANSIONS_V1",
    "FailureFeedbackResultV1",
    "FailureFeedbackStateV1",
    "expand_failure",
]
