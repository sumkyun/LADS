"""CAS-independent protocols and explicit proof/branch outcomes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from ..ir import ExpressionIRV1
from ..models import BranchCoverageV1, DiscoveryContractError, ProofStatusV1


class BranchKindV1(str, Enum):
    FINITE_BRANCHES = "finite_branches"
    CONDITIONAL = "conditional"
    INCONSISTENT = "inconsistent"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProofResultV1:
    status: ProofStatusV1
    normalized: ExpressionIRV1
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ProofStatusV1(self.status))
        if not isinstance(self.normalized, ExpressionIRV1):
            raise DiscoveryContractError("proof normalized expression must be typed")
        if not isinstance(self.reason, str) or not self.reason:
            raise DiscoveryContractError("proof reason must be nonempty")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "normalized": self.normalized.to_mapping(),
            "reason": self.reason,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProofResultV1":
        if set(value) != {"status", "normalized", "reason"}:
            raise DiscoveryContractError("proof result fields differ")
        return cls(
            value["status"], ExpressionIRV1.from_mapping(value["normalized"]), value["reason"]
        )


@dataclass(frozen=True)
class CASOperationResultV1:
    expression: ExpressionIRV1
    side_conditions: tuple[ExpressionIRV1, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.expression, ExpressionIRV1):
            raise DiscoveryContractError("CAS operation expression must be typed")
        conditions = tuple(self.side_conditions)
        if any(not isinstance(item, ExpressionIRV1) for item in conditions):
            raise DiscoveryContractError("CAS side conditions must be expression IR")
        object.__setattr__(self, "side_conditions", conditions)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "expression": self.expression.to_mapping(),
            "side_conditions": [item.to_mapping() for item in self.side_conditions],
        }


@dataclass(frozen=True)
class BranchResultV1:
    kind: BranchKindV1
    branches: tuple[Mapping[str, ExpressionIRV1], ...]
    coverage: BranchCoverageV1
    unexplored_reason: str | None
    explored_count: int
    branch_cap: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", BranchKindV1(self.kind))
        object.__setattr__(self, "coverage", BranchCoverageV1(self.coverage))
        if isinstance(self.explored_count, bool) or self.explored_count < 0:
            raise DiscoveryContractError("explored_count must be nonnegative")
        if isinstance(self.branch_cap, bool) or self.branch_cap < 1:
            raise DiscoveryContractError("branch_cap must be positive")
        copied: list[Mapping[str, ExpressionIRV1]] = []
        for branch in self.branches:
            if not isinstance(branch, Mapping) or any(
                not isinstance(key, str) or not isinstance(item, ExpressionIRV1)
                for key, item in branch.items()
            ):
                raise DiscoveryContractError("solver branches must map names to expression IR")
            copied.append(dict(sorted(branch.items())))
        object.__setattr__(self, "branches", tuple(copied))
        if self.coverage is not BranchCoverageV1.EXHAUSTIVE and not self.unexplored_reason:
            raise DiscoveryContractError("non-exhaustive coverage requires a reason")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "branches": [
                {key: expression.to_mapping() for key, expression in branch.items()}
                for branch in self.branches
            ],
            "coverage": self.coverage.value,
            "unexplored_reason": self.unexplored_reason,
            "explored_count": self.explored_count,
            "branch_cap": self.branch_cap,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BranchResultV1":
        expected = {
            "kind",
            "branches",
            "coverage",
            "unexplored_reason",
            "explored_count",
            "branch_cap",
        }
        if set(value) != expected:
            raise DiscoveryContractError("branch result fields differ")
        return cls(
            value["kind"],
            tuple(
                {
                    name: ExpressionIRV1.from_mapping(expression)
                    for name, expression in branch.items()
                }
                for branch in value["branches"]
            ),
            value["coverage"],
            value["unexplored_reason"],
            value["explored_count"],
            value["branch_cap"],
        )


class CASBackendV1(Protocol):
    backend_name: str

    @property
    def backend_version(self) -> str: ...

    def round_trip(self, expression: ExpressionIRV1) -> ExpressionIRV1: ...

    def render(self, expression: ExpressionIRV1) -> str: ...

    def differentiate(
        self, expression: ExpressionIRV1, variable: str, order: int = 1
    ) -> ExpressionIRV1: ...

    def substitute(
        self,
        expression: ExpressionIRV1,
        replacements: Mapping[ExpressionIRV1, ExpressionIRV1],
    ) -> ExpressionIRV1: ...

    def expand(self, expression: ExpressionIRV1) -> ExpressionIRV1: ...

    def factor(self, expression: ExpressionIRV1) -> ExpressionIRV1: ...

    def cancel(self, expression: ExpressionIRV1) -> CASOperationResultV1: ...

    def together(self, expression: ExpressionIRV1) -> ExpressionIRV1: ...

    def factor_terms(self, expression: ExpressionIRV1) -> ExpressionIRV1: ...

    def prove_zero(self, expression: ExpressionIRV1) -> ProofResultV1: ...

    def solve_parameters(
        self, equations: Sequence[ExpressionIRV1], parameters: Sequence[str], branch_cap: int
    ) -> BranchResultV1: ...


__all__ = [
    "BranchKindV1",
    "BranchResultV1",
    "CASBackendV1",
    "CASOperationResultV1",
    "ProofResultV1",
]
