"""Exact, data-only expression IR used at every trust boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any, ClassVar

from ..models import DiscoveryContractError
from ..serialization import canonical_json_bytes, canonical_sha256
from .jets import JetVariableV1


class ExpressionKindV1(str, Enum):
    RATIONAL = "rational"
    SYMBOL = "symbol"
    JET = "jet"
    ADD = "add"
    MUL = "mul"
    POW = "pow"
    FUNCTION = "function"
    DERIVATIVE = "derivative"


class SymbolRoleV1(str, Enum):
    INDEPENDENT = "independent"
    PARAMETER = "parameter"
    PROFILE_COORDINATE = "profile_coordinate"
    SEPARATION_CONSTANT = "separation_constant"


@dataclass(frozen=True)
class ExpressionIRV1:
    kind: ExpressionKindV1
    operands: tuple["ExpressionIRV1", ...] = ()
    name: str | None = None
    role: SymbolRoleV1 | None = None
    numerator: int | None = None
    denominator: int | None = None
    jet: JetVariableV1 | None = None
    derivative_variables: tuple[str, ...] = ()

    schema_version: ClassVar[str] = "ExpressionIRV1"

    def __post_init__(self) -> None:
        try:
            kind = ExpressionKindV1(self.kind)
        except (TypeError, ValueError) as exc:
            raise DiscoveryContractError("invalid expression kind") from exc
        object.__setattr__(self, "kind", kind)
        operands = tuple(self.operands)
        if any(not isinstance(item, ExpressionIRV1) for item in operands):
            raise DiscoveryContractError("expression operands must be typed")
        object.__setattr__(self, "operands", operands)
        derivative_variables = tuple(self.derivative_variables)
        if any(not isinstance(item, str) or not item for item in derivative_variables):
            raise DiscoveryContractError("derivative variables must be nonempty strings")
        object.__setattr__(self, "derivative_variables", derivative_variables)

        if kind is ExpressionKindV1.RATIONAL:
            if isinstance(self.numerator, bool) or not isinstance(self.numerator, int):
                raise DiscoveryContractError("rational numerator must be an integer")
            if isinstance(self.denominator, bool) or not isinstance(self.denominator, int) or self.denominator <= 0:
                raise DiscoveryContractError("rational denominator must be positive")
            reduced = Fraction(self.numerator, self.denominator)
            if (reduced.numerator, reduced.denominator) != (self.numerator, self.denominator):
                raise DiscoveryContractError("rational must be in lowest terms")
            self._require_empty(name=True, role=True, jet=True, derivatives=True, operands=True)
        elif kind is ExpressionKindV1.SYMBOL:
            if not self.name:
                raise DiscoveryContractError("symbol name must be nonempty")
            try:
                object.__setattr__(self, "role", SymbolRoleV1(self.role))
            except (TypeError, ValueError) as exc:
                raise DiscoveryContractError("symbol role is invalid") from exc
            self._require_empty(numbers=True, jet=True, derivatives=True, operands=True)
        elif kind is ExpressionKindV1.JET:
            if not isinstance(self.jet, JetVariableV1):
                raise DiscoveryContractError("jet expression requires a typed jet")
            self._require_empty(name=True, role=True, numbers=True, derivatives=True, operands=True)
        elif kind in {ExpressionKindV1.ADD, ExpressionKindV1.MUL}:
            if len(operands) < 2:
                raise DiscoveryContractError(f"{kind.value} requires at least two operands")
            self._require_empty(name=True, role=True, numbers=True, jet=True, derivatives=True)
        elif kind is ExpressionKindV1.POW:
            if len(operands) != 2:
                raise DiscoveryContractError("pow requires base and exponent")
            self._require_empty(name=True, role=True, numbers=True, jet=True, derivatives=True)
        elif kind is ExpressionKindV1.FUNCTION:
            if not self.name:
                raise DiscoveryContractError("function name must be nonempty")
            self._require_empty(role=True, numbers=True, jet=True, derivatives=True)
        elif kind is ExpressionKindV1.DERIVATIVE:
            if len(operands) != 1 or not derivative_variables:
                raise DiscoveryContractError("derivative requires one expression and variables")
            self._require_empty(name=True, role=True, numbers=True, jet=True)

    def _require_empty(
        self,
        *,
        name: bool = False,
        role: bool = False,
        numbers: bool = False,
        jet: bool = False,
        derivatives: bool = False,
        operands: bool = False,
    ) -> None:
        if name and self.name is not None:
            raise DiscoveryContractError("unexpected expression name")
        if role and self.role is not None:
            raise DiscoveryContractError("unexpected expression role")
        if numbers and (self.numerator is not None or self.denominator is not None):
            raise DiscoveryContractError("unexpected expression rational value")
        if jet and self.jet is not None:
            raise DiscoveryContractError("unexpected expression jet")
        if derivatives and self.derivative_variables:
            raise DiscoveryContractError("unexpected derivative variables")
        if operands and self.operands:
            raise DiscoveryContractError("unexpected expression operands")

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def canonical_hash(self) -> str:
        return canonical_sha256(self.to_mapping())

    @property
    def node_count(self) -> int:
        return 1 + sum(item.node_count for item in self.operands)

    @property
    def free_names(self) -> frozenset[str]:
        names: set[str] = set()
        if self.kind is ExpressionKindV1.SYMBOL and self.name is not None:
            names.add(self.name)
        if self.kind is ExpressionKindV1.JET and self.jet is not None:
            names.add(self.jet.field)
        if self.kind is ExpressionKindV1.FUNCTION and self.name is not None:
            names.add(self.name)
        for operand in self.operands:
            names.update(operand.free_names)
        return frozenset(names)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "operands": [item.to_mapping() for item in self.operands],
            "name": self.name,
            "role": None if self.role is None else self.role.value,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "jet": None if self.jet is None else self.jet.to_mapping(),
            "derivative_variables": list(self.derivative_variables),
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ExpressionIRV1":
        expected = {"schema_version", "kind", "operands", "name", "role", "numerator", "denominator", "jet", "derivative_variables"}
        if set(value) != expected:
            raise DiscoveryContractError("expression IR fields differ")
        if value["schema_version"] != cls.schema_version:
            raise DiscoveryContractError("unsupported expression IR schema")
        return cls(
            value["kind"],
            tuple(cls.from_mapping(item) for item in value["operands"]),
            value["name"],
            value["role"],
            value["numerator"],
            value["denominator"],
            None if value["jet"] is None else JetVariableV1.from_mapping(value["jet"]),
            tuple(value["derivative_variables"]),
        )


def rational(value: int | Fraction, denominator: int = 1) -> ExpressionIRV1:
    fraction = value if isinstance(value, Fraction) else Fraction(value, denominator)
    return ExpressionIRV1(
        ExpressionKindV1.RATIONAL,
        numerator=fraction.numerator,
        denominator=fraction.denominator,
    )


def symbol(name: str, role: SymbolRoleV1) -> ExpressionIRV1:
    return ExpressionIRV1(ExpressionKindV1.SYMBOL, name=name, role=role)


def jet(field: str, variables: tuple[str, ...] = ()) -> ExpressionIRV1:
    from .jets import JetMultiIndexV1

    return ExpressionIRV1(
        ExpressionKindV1.JET,
        jet=JetVariableV1(field, JetMultiIndexV1(variables)),
    )


def function(name: str, *arguments: ExpressionIRV1) -> ExpressionIRV1:
    return ExpressionIRV1(ExpressionKindV1.FUNCTION, operands=tuple(arguments), name=name)


def derivative(expression: ExpressionIRV1, variables: tuple[str, ...]) -> ExpressionIRV1:
    if expression.kind is ExpressionKindV1.JET and expression.jet is not None:
        return jet(expression.jet.field, expression.jet.multi_index.variables + variables)
    return ExpressionIRV1(
        ExpressionKindV1.DERIVATIVE,
        operands=(expression,),
        derivative_variables=variables,
    )


def is_rational(expression: ExpressionIRV1) -> bool:
    return expression.kind is ExpressionKindV1.RATIONAL


def as_fraction(expression: ExpressionIRV1) -> Fraction:
    if not is_rational(expression):
        raise TypeError("expression is not rational")
    return Fraction(expression.numerator, expression.denominator)


__all__ = [
    "ExpressionIRV1",
    "ExpressionKindV1",
    "SymbolRoleV1",
    "as_fraction",
    "derivative",
    "function",
    "is_rational",
    "jet",
    "rational",
    "symbol",
]
