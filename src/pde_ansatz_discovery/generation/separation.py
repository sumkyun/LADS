"""Closed separation-family requests for the Phase-4 verifier."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar


class SeparationFamilyV1(str, Enum):
    PRODUCT = "product"
    ADDITIVE = "additive"
    TRANSFORMED_IDENTITY = "transformed_identity"


@dataclass(frozen=True)
class SeparationAnsatzV1:
    family: SeparationFamilyV1
    field: str
    spatial_profile: str = "X"
    temporal_profile: str = "T"
    separation_constant: str = "kappa"

    schema_version: ClassVar[str] = "SeparationAnsatzV1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", SeparationFamilyV1(self.family))
        for name in (
            "field",
            "spatial_profile",
            "temporal_profile",
            "separation_constant",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "family": self.family.value,
            "field": self.field,
            "spatial_profile": self.spatial_profile,
            "temporal_profile": self.temporal_profile,
            "separation_constant": self.separation_constant,
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "SeparationAnsatzV1":
        expected = {
            "schema_version",
            "family",
            "field",
            "spatial_profile",
            "temporal_profile",
            "separation_constant",
        }
        if set(value) != expected or value["schema_version"] != cls.schema_version:
            raise ValueError("invalid separation Ansatz schema")
        return cls(
            value["family"],
            value["field"],
            value["spatial_profile"],
            value["temporal_profile"],
            value["separation_constant"],
        )


__all__ = ["SeparationAnsatzV1", "SeparationFamilyV1"]
