"""Exact profile-jet polynomial collection for reduction verification."""

from __future__ import annotations

from dataclasses import dataclass

from ..cas.backend import CASBackendV1
from ..ir import (
    ExpressionIRV1,
    ExpressionKindV1,
    add,
    function,
    mul,
    power,
    rational,
)


class UnsupportedProfileDependenceV1(ValueError):
    """Raised when a residual is not polynomial in the bounded profile jets."""


@dataclass(frozen=True, order=True)
class ProfileJetSignatureV1:
    profile: str
    order: int

    @property
    def generator_name(self) -> str:
        return self.profile if self.order == 0 else f"{self.profile}__d{self.order}"


@dataclass(frozen=True, order=True)
class ProfileJetMonomialV1:
    powers: tuple[tuple[ProfileJetSignatureV1, int], ...] = ()

    def __post_init__(self) -> None:
        powers = tuple(self.powers)
        if any(exponent < 1 for _, exponent in powers):
            raise ValueError("profile-jet monomial exponents must be positive")
        if tuple(sorted(powers)) != powers or len({item for item, _ in powers}) != len(powers):
            raise ValueError("profile-jet monomial powers must be unique and sorted")
        object.__setattr__(self, "powers", powers)

    def multiplied(self, other: "ProfileJetMonomialV1") -> "ProfileJetMonomialV1":
        combined: dict[ProfileJetSignatureV1, int] = dict(self.powers)
        for signature, exponent in other.powers:
            combined[signature] = (
                combined[signature] if signature in combined else 0
            ) + exponent
        return ProfileJetMonomialV1(tuple(sorted(combined.items())))


@dataclass(frozen=True)
class ProfileCoefficientTermV1:
    monomial: ProfileJetMonomialV1
    coefficient: ExpressionIRV1


@dataclass(frozen=True)
class CollectedProfilePolynomialV1:
    terms: tuple[ProfileCoefficientTermV1, ...]
    generators: tuple[tuple[ProfileJetSignatureV1, ExpressionIRV1], ...]


def _profile_signature(node: ExpressionIRV1, profile: str) -> ProfileJetSignatureV1 | None:
    if node.kind is not ExpressionKindV1.FUNCTION or node.name is None:
        return None
    if node.name == profile:
        return ProfileJetSignatureV1(profile, 0)
    prefix = profile + "__d"
    if node.name.startswith(prefix) and node.name[len(prefix) :].isdigit():
        return ProfileJetSignatureV1(profile, int(node.name[len(prefix) :]))
    return None


def _contains_profile(node: ExpressionIRV1, profile: str) -> bool:
    return _profile_signature(node, profile) is not None or any(
        _contains_profile(operand, profile) for operand in node.operands
    )


def profile_jet_signatures(
    expression: ExpressionIRV1, profile: str
) -> tuple[ProfileJetSignatureV1, ...]:
    found: set[ProfileJetSignatureV1] = set()

    def visit(node: ExpressionIRV1) -> None:
        signature = _profile_signature(node, profile)
        if signature is not None:
            found.add(signature)
            return
        for operand in node.operands:
            visit(operand)

    visit(expression)
    return tuple(sorted(found))


def _add_polynomials(
    left: dict[ProfileJetMonomialV1, ExpressionIRV1],
    right: dict[ProfileJetMonomialV1, ExpressionIRV1],
) -> dict[ProfileJetMonomialV1, ExpressionIRV1]:
    result = dict(left)
    for monomial, coefficient in right.items():
        result[monomial] = (
            add(result[monomial], coefficient) if monomial in result else coefficient
        )
    return result


def _multiply_polynomials(
    left: dict[ProfileJetMonomialV1, ExpressionIRV1],
    right: dict[ProfileJetMonomialV1, ExpressionIRV1],
) -> dict[ProfileJetMonomialV1, ExpressionIRV1]:
    result: dict[ProfileJetMonomialV1, ExpressionIRV1] = {}
    for left_monomial, left_coefficient in left.items():
        for right_monomial, right_coefficient in right.items():
            monomial = left_monomial.multiplied(right_monomial)
            coefficient = mul(left_coefficient, right_coefficient)
            result[monomial] = (
                add(result[monomial], coefficient) if monomial in result else coefficient
            )
    return result


def collect_profile_polynomial(
    expression: ExpressionIRV1,
    *,
    profile: str,
    backend: CASBackendV1,
    max_profile_degree: int = 8,
) -> CollectedProfilePolynomialV1:
    """Collect a bounded exact polynomial in ``F, F__d1, ...``."""

    generators: dict[ProfileJetSignatureV1, ExpressionIRV1] = {}

    def visit(node: ExpressionIRV1) -> dict[ProfileJetMonomialV1, ExpressionIRV1]:
        signature = _profile_signature(node, profile)
        if signature is not None:
            previous = generators[signature] if signature in generators else None
            if previous is not None and previous != node:
                raise UnsupportedProfileDependenceV1(
                    "one profile jet is evaluated at multiple arguments"
                )
            generators[signature] = node
            monomial = ProfileJetMonomialV1(((signature, 1),))
            return {monomial: rational(1)}

        if not _contains_profile(node, profile):
            return {ProfileJetMonomialV1(): node}
        if node.kind is ExpressionKindV1.ADD:
            result: dict[ProfileJetMonomialV1, ExpressionIRV1] = {}
            for operand in node.operands:
                result = _add_polynomials(result, visit(operand))
            return result
        if node.kind is ExpressionKindV1.MUL:
            result = {ProfileJetMonomialV1(): rational(1)}
            for operand in node.operands:
                result = _multiply_polynomials(result, visit(operand))
            return result
        if node.kind is ExpressionKindV1.POW:
            base, exponent = node.operands
            if (
                exponent.kind is not ExpressionKindV1.RATIONAL
                or exponent.denominator != 1
                or exponent.numerator < 0
                or exponent.numerator > max_profile_degree
            ):
                raise UnsupportedProfileDependenceV1(
                    "profile jets require a bounded nonnegative integer power"
                )
            result = {ProfileJetMonomialV1(): rational(1)}
            base_polynomial = visit(base)
            for _ in range(exponent.numerator):
                result = _multiply_polynomials(result, base_polynomial)
            return result
        raise UnsupportedProfileDependenceV1(
            f"profile jet occurs below unsupported {node.kind.value} node"
        )

    raw = visit(backend.expand(expression))
    terms: list[ProfileCoefficientTermV1] = []
    for monomial in sorted(raw):
        coefficient = backend.factor_terms(backend.expand(raw[monomial]))
        if backend.prove_zero(coefficient).status.value != "proved_zero":
            terms.append(ProfileCoefficientTermV1(monomial, coefficient))
    return CollectedProfilePolynomialV1(
        tuple(terms), tuple(sorted(generators.items(), key=lambda item: item[0]))
    )


def reconstruct_profile_polynomial(
    polynomial: CollectedProfilePolynomialV1,
) -> ExpressionIRV1:
    generators = dict(polynomial.generators)
    summands: list[ExpressionIRV1] = []
    for term in polynomial.terms:
        factors = [term.coefficient]
        for signature, exponent in term.monomial.powers:
            generator = generators[signature] if signature in generators else None
            if generator is None:
                generator = function(signature.generator_name, rational(0))
            factors.append(generator if exponent == 1 else power(generator, rational(exponent)))
        summands.append(mul(*factors))
    return rational(0) if not summands else add(*summands)


__all__ = [
    "CollectedProfilePolynomialV1",
    "ProfileCoefficientTermV1",
    "ProfileJetMonomialV1",
    "ProfileJetSignatureV1",
    "UnsupportedProfileDependenceV1",
    "collect_profile_polynomial",
    "profile_jet_signatures",
    "reconstruct_profile_polynomial",
]
