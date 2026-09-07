"""Typed field-jet identifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from ..models import DiscoveryContractError


@dataclass(frozen=True, order=True)
class JetMultiIndexV1:
    """An ordered derivative multi-index, preserving mixed derivative order."""

    variables: tuple[str, ...] = ()
    schema_version: ClassVar[str] = "JetMultiIndexV1"

    def __post_init__(self) -> None:
        variables = tuple(self.variables)
        if any(not isinstance(item, str) or not item for item in variables):
            raise DiscoveryContractError("jet derivative variables must be nonempty strings")
        object.__setattr__(self, "variables", variables)

    @property
    def order(self) -> int:
        return len(self.variables)

    def append(self, *variables: str) -> "JetMultiIndexV1":
        return JetMultiIndexV1(self.variables + tuple(variables))

    def to_mapping(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "variables": list(self.variables)}

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "JetMultiIndexV1":
        if set(value) != {"schema_version", "variables"}:
            raise DiscoveryContractError("jet multi-index fields differ")
        if value["schema_version"] != cls.schema_version:
            raise DiscoveryContractError("unsupported jet multi-index schema")
        return cls(tuple(value["variables"]))


@dataclass(frozen=True, order=True)
class JetVariableV1:
    field: str
    multi_index: JetMultiIndexV1 = JetMultiIndexV1()
    schema_version: ClassVar[str] = "JetVariableV1"

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field:
            raise DiscoveryContractError("jet field must be nonempty")
        if not isinstance(self.multi_index, JetMultiIndexV1):
            raise DiscoveryContractError("jet multi_index must be typed")

    @property
    def order(self) -> int:
        return self.multi_index.order

    def differentiate(self, *variables: str) -> "JetVariableV1":
        return JetVariableV1(self.field, self.multi_index.append(*variables))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "field": self.field,
            "multi_index": self.multi_index.to_mapping(),
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "JetVariableV1":
        if set(value) != {"schema_version", "field", "multi_index"}:
            raise DiscoveryContractError("jet variable fields differ")
        if value["schema_version"] != cls.schema_version:
            raise DiscoveryContractError("unsupported jet variable schema")
        return cls(value["field"], JetMultiIndexV1.from_mapping(value["multi_index"]))


__all__ = ["JetMultiIndexV1", "JetVariableV1"]
