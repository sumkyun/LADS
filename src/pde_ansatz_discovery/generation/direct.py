"""Caller-supplied single-profile Ansatz contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from ..models import BranchCoverageV1, DiscoveryContractError, ParameterBranchV1
from ..serialization import canonical_sha256, freeze_json, thaw_json


def materialized_parameter_branch(
    assignments: Mapping[str, Any],
    side_conditions: tuple[str, ...],
) -> ParameterBranchV1:
    assignment_mapping = dict(sorted(assignments.items()))
    return ParameterBranchV1(
        canonical_sha256(
            {
                "kind": "MaterializedParameterBranchV1",
                "assignments": assignment_mapping,
                "side_conditions": list(side_conditions),
            }
        ),
        assignment_mapping,
        (),
        side_conditions,
        BranchCoverageV1.EXHAUSTIVE,
    )


@dataclass(frozen=True)
class DirectAnsatzV1:
    field: str
    expression: str
    coordinate_name: str
    coordinate_expression: str
    profile_function: str
    free_parameters: tuple[str, ...]
    inverse_map: Mapping[str, str]
    complementary_coordinate: str
    domain_conditions: tuple[str, ...] = ()
    family: str = "direct"
    branch: ParameterBranchV1 | None = None

    schema_version: ClassVar[str] = "DirectAnsatzV1"

    def __post_init__(self) -> None:
        for name in (
            "field",
            "expression",
            "coordinate_name",
            "coordinate_expression",
            "profile_function",
            "complementary_coordinate",
            "family",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise DiscoveryContractError(f"{name} must be nonempty")
        parameters = tuple(self.free_parameters)
        if any(not isinstance(item, str) or not item for item in parameters):
            raise DiscoveryContractError("free_parameters must contain strings")
        if len(set(parameters)) != len(parameters):
            raise DiscoveryContractError("free_parameters must be unique")
        object.__setattr__(self, "free_parameters", parameters)
        inverse = dict(self.inverse_map)
        if any(not isinstance(key, str) or not isinstance(value, str) or not value for key, value in inverse.items()):
            raise DiscoveryContractError("inverse_map must map names to expression strings")
        object.__setattr__(self, "inverse_map", freeze_json(inverse))
        conditions = tuple(self.domain_conditions)
        if any(not isinstance(item, str) or not item for item in conditions):
            raise DiscoveryContractError("domain_conditions must contain strings")
        object.__setattr__(self, "domain_conditions", conditions)
        if self.branch is None:
            object.__setattr__(
                self,
                "branch",
                ParameterBranchV1(
                    "generic", {}, (), conditions, BranchCoverageV1.EXHAUSTIVE
                ),
            )
        elif not isinstance(self.branch, ParameterBranchV1):
            raise DiscoveryContractError("branch must be ParameterBranchV1")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "field": self.field,
            "expression": self.expression,
            "coordinate_name": self.coordinate_name,
            "coordinate_expression": self.coordinate_expression,
            "profile_function": self.profile_function,
            "free_parameters": list(self.free_parameters),
            "inverse_map": thaw_json(self.inverse_map),
            "complementary_coordinate": self.complementary_coordinate,
            "domain_conditions": list(self.domain_conditions),
            "family": self.family,
            "branch": self.branch.to_mapping(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DirectAnsatzV1":
        expected = {
            "schema_version",
            "field",
            "expression",
            "coordinate_name",
            "coordinate_expression",
            "profile_function",
            "free_parameters",
            "inverse_map",
            "complementary_coordinate",
            "domain_conditions",
            "family",
            "branch",
        }
        if set(value) != expected or value["schema_version"] != cls.schema_version:
            raise DiscoveryContractError("invalid direct Ansatz schema")
        return cls(
            value["field"],
            value["expression"],
            value["coordinate_name"],
            value["coordinate_expression"],
            value["profile_function"],
            tuple(value["free_parameters"]),
            value["inverse_map"],
            value["complementary_coordinate"],
            tuple(value["domain_conditions"]),
            value["family"],
            ParameterBranchV1.from_mapping(value["branch"]),
        )

    @property
    def definition_hash(self) -> str:
        return canonical_sha256(self.to_mapping())


__all__ = ["DirectAnsatzV1", "materialized_parameter_branch"]
