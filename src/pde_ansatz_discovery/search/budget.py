"""Monotone deterministic Phase-3 search accounting."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import DiscoveryBudgetV1


@dataclass
class SearchBudgetLedgerV1:
    budget: DiscoveryBudgetV1
    generated: int = 0
    verified: int = 0
    certified_valid: int = 0
    certified_invalid: int = 0
    unresolved: int = 0
    skipped_by_cap: int = 0

    def admit_candidate(self) -> bool:
        if self.generated >= self.budget.total_candidates:
            self.skipped_by_cap += 1
            return False
        self.generated += 1
        return True

    def record_verdict(self, status: str) -> None:
        if status not in {"certified_valid", "certified_invalid", "unresolved"}:
            raise ValueError(f"unknown verification status: {status}")
        if self.verified >= self.generated:
            raise RuntimeError(
                "search verification counter would exceed generated counter"
            )
        self.verified += 1
        if status == "certified_valid":
            self.certified_valid += 1
        elif status == "certified_invalid":
            self.certified_invalid += 1
        else:
            self.unresolved += 1

    def to_mapping(self) -> dict[str, int]:
        return {
            "generated": self.generated,
            "verified": self.verified,
            "certified_valid": self.certified_valid,
            "certified_invalid": self.certified_invalid,
            "unresolved": self.unresolved,
            "skipped_by_cap": self.skipped_by_cap,
            "total_candidate_cap": self.budget.total_candidates,
        }


__all__ = ["SearchBudgetLedgerV1"]
