"""Closed, replayable V1 reduction-equivalence grammar."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from fractions import Fraction
from typing import Any

from ..cas.backend import CASBackendV1
from ..ir import (
    ExpressionIRV1,
    ExpressionKindV1,
    SymbolRoleV1,
    divide,
    mul,
    rational,
    subtract,
)
from ..models import DiscoveryCandidateV1, DiscoveryProblemInputV1, ProofStatusV1
from ..serialization import canonical_sha256, freeze_json, thaw_json

_BOUND = 2**31 - 1


@dataclass(frozen=True, order=True)
class BoundedRationalV1:
    numerator: int
    denominator: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.numerator, bool) or isinstance(self.denominator, bool):
            raise ValueError(  # noqa: TRY004 - V1 contract exposes ValueError uniformly
                "bounded rational parts must be integers"
            )
        if not isinstance(self.numerator, int) or not isinstance(self.denominator, int):
            raise ValueError(  # noqa: TRY004 - V1 contract exposes ValueError uniformly
                "bounded rational parts must be integers"
            )
        if self.denominator < 1 or abs(self.numerator) > _BOUND or self.denominator > _BOUND:
            raise ValueError("bounded rational is outside the V1 closed bound")
        if math.gcd(abs(self.numerator), self.denominator) != 1:
            raise ValueError("bounded rational must be in lowest terms")

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    @property
    def nonzero(self) -> bool:
        return self.numerator != 0

    def to_mapping(self) -> dict[str, int]:
        return {"numerator": self.numerator, "denominator": self.denominator}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> BoundedRationalV1:
        if set(value) != {"numerator", "denominator"}:
            raise ValueError("bounded rational fields differ")
        return cls(value["numerator"], value["denominator"])


class DomainComponentKindV1(str, Enum):
    ALL_REAL = "all_real"
    STRICTLY_POSITIVE = "strictly_positive"
    STRICTLY_NEGATIVE = "strictly_negative"
    INTERVAL = "interval"


@dataclass(frozen=True)
class DomainComponentV1:
    kind: DomainComponentKindV1
    lower: BoundedRationalV1 | None = None
    upper: BoundedRationalV1 | None = None
    lower_open: bool = True
    upper_open: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", DomainComponentKindV1(self.kind))
        if self.kind is DomainComponentKindV1.INTERVAL:
            if self.lower is None or self.upper is None or self.lower.fraction >= self.upper.fraction:
                raise ValueError("interval requires ordered exact rational endpoints")
        elif self.lower is not None or self.upper is not None:
            raise ValueError("only interval components carry endpoints")

    def contains(self, value: Fraction) -> bool:
        if self.kind is DomainComponentKindV1.ALL_REAL:
            return True
        if self.kind is DomainComponentKindV1.STRICTLY_POSITIVE:
            return value > 0
        if self.kind is DomainComponentKindV1.STRICTLY_NEGATIVE:
            return value < 0
        lower_ok = value > self.lower.fraction if self.lower_open else value >= self.lower.fraction
        upper_ok = value < self.upper.fraction if self.upper_open else value <= self.upper.fraction
        return lower_ok and upper_ok

    def to_mapping(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "lower": None if self.lower is None else self.lower.to_mapping(),
            "upper": None if self.upper is None else self.upper.to_mapping(),
            "lower_open": self.lower_open,
            "upper_open": self.upper_open,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DomainComponentV1:
        expected = {"kind", "lower", "upper", "lower_open", "upper_open"}
        if set(value) != expected:
            raise ValueError("domain component fields differ")
        if not isinstance(value["lower_open"], bool) or not isinstance(
            value["upper_open"], bool
        ):
            raise ValueError(  # noqa: TRY004 - V1 contract exposes ValueError uniformly
                "domain openness flags must be booleans"
            )
        return cls(
            value["kind"],
            None
            if value["lower"] is None
            else BoundedRationalV1.from_mapping(value["lower"]),
            None
            if value["upper"] is None
            else BoundedRationalV1.from_mapping(value["upper"]),
            value["lower_open"],
            value["upper_open"],
        )


class EquivalenceGrammarV1(str, Enum):
    PROJECTIVE_GENERATOR_Q = "PROJECTIVE_GENERATOR_Q"
    AFFINE_INVARIANT_Q = "AFFINE_INVARIANT_Q"
    MOBIUS_INVARIANT_Z = "MOBIUS_INVARIANT_Z"
    MONOMIAL_INVARIANT_QZ = "MONOMIAL_INVARIANT_QZ"
    PROFILE_AFFINE_Q = "PROFILE_AFFINE_Q"
    REDUCED_EQUATION_FACTOR_Q = "REDUCED_EQUATION_FACTOR_Q"


@dataclass(frozen=True)
class EquivalenceWitnessV1:
    transform: EquivalenceGrammarV1
    parameters: Any
    forward: str
    inverse: str
    source_domain: DomainComponentV1
    target_domain: DomainComponentV1
    jacobian: str
    obligations: tuple[str, ...]
    identities: tuple[str, ...]
    representative_key: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "transform", EquivalenceGrammarV1(self.transform))
        object.__setattr__(self, "parameters", freeze_json(self.parameters))
        object.__setattr__(self, "obligations", tuple(self.obligations))
        object.__setattr__(self, "identities", tuple(self.identities))

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": "EquivalenceWitnessV1",
            "transform": self.transform.value,
            "parameters": thaw_json(self.parameters),
            "forward": self.forward,
            "inverse": self.inverse,
            "source_domain": self.source_domain.to_mapping(),
            "target_domain": self.target_domain.to_mapping(),
            "jacobian": self.jacobian,
            "obligations": list(self.obligations),
            "identities": list(self.identities),
            "representative_key": self.representative_key,
        }

    @property
    def replay_digest(self) -> str:
        return canonical_sha256(self._payload())

    def to_mapping(self) -> dict[str, Any]:
        return {**self._payload(), "replay_digest": self.replay_digest}


def _all_real() -> DomainComponentV1:
    return DomainComponentV1(DomainComponentKindV1.ALL_REAL)


def _bounded(value: Fraction) -> BoundedRationalV1 | None:
    try:
        return BoundedRationalV1(value.numerator, value.denominator)
    except ValueError:
        return None


def _affine_image(
    domain: DomainComponentV1, scale: Fraction, shift: Fraction
) -> DomainComponentV1 | None:
    if domain.kind is DomainComponentKindV1.ALL_REAL:
        return _all_real()
    if domain.kind in {
        DomainComponentKindV1.STRICTLY_POSITIVE,
        DomainComponentKindV1.STRICTLY_NEGATIVE,
    }:
        # V1 has no shifted half-line component, so only a zero shift is closed.
        if shift != 0:
            return None
        source_sign = (
            1
            if domain.kind is DomainComponentKindV1.STRICTLY_POSITIVE
            else -1
        )
        target_sign = source_sign * (1 if scale > 0 else -1)
        return DomainComponentV1(
            DomainComponentKindV1.STRICTLY_POSITIVE
            if target_sign > 0
            else DomainComponentKindV1.STRICTLY_NEGATIVE
        )
    lower_value = scale * domain.lower.fraction + shift
    upper_value = scale * domain.upper.fraction + shift
    lower = _bounded(min(lower_value, upper_value))
    upper = _bounded(max(lower_value, upper_value))
    if lower is None or upper is None:
        return None
    if scale > 0:
        lower_open, upper_open = domain.lower_open, domain.upper_open
    else:
        lower_open, upper_open = domain.upper_open, domain.lower_open
    return DomainComponentV1(
        DomainComponentKindV1.INTERVAL,
        lower,
        upper,
        lower_open,
        upper_open,
    )


def projective_generator_witness(
    scale: BoundedRationalV1, representative_key: str = "projective"
) -> EquivalenceWitnessV1 | None:
    if not scale.nonzero:
        return None
    return EquivalenceWitnessV1(
        EquivalenceGrammarV1.PROJECTIVE_GENERATOR_Q,
        {"scale": scale.to_mapping()},
        f"X2=({scale.numerator}/{scale.denominator})*X1",
        f"X1=({scale.denominator}/{scale.numerator})*X2",
        _all_real(),
        _all_real(),
        f"{scale.numerator}/{scale.denominator}",
        ("scale!=0",),
        ("inverse(forward(X1))=X1",),
        representative_key,
    )


def affine_invariant_witness(
    scale: BoundedRationalV1,
    shift: BoundedRationalV1,
    domain: DomainComponentV1,
    representative_key: str = "affine",
) -> EquivalenceWitnessV1 | None:
    if not scale.nonzero:
        return None
    a, b = scale.fraction, shift.fraction
    target = _affine_image(domain, a, b)
    if target is None:
        return None
    return EquivalenceWitnessV1(
        EquivalenceGrammarV1.AFFINE_INVARIANT_Q,
        {"scale": scale.to_mapping(), "shift": shift.to_mapping()},
        f"z=({a})*eta+({b})",
        f"eta=(z-({b}))/({a})",
        domain,
        target,
        str(a),
        ("scale!=0", "domain_bijection"),
        ("inverse(forward(eta))=eta", "forward(inverse(z))=z"),
        representative_key,
    )


def mobius_invariant_witness(
    coefficients: tuple[int, int, int, int],
    domain: DomainComponentV1,
    representative_key: str = "mobius",
) -> EquivalenceWitnessV1 | None:
    if len(coefficients) != 4 or any(abs(value) > _BOUND for value in coefficients):
        return None
    a, b, c, d = coefficients
    if math.gcd(math.gcd(abs(a), abs(b)), math.gcd(abs(c), abs(d))) != 1:
        return None
    first = next((value for value in coefficients if value), 0)
    determinant = a * d - b * c
    if first <= 0 or determinant == 0:
        return None
    if c:
        pole = Fraction(-d, c)
        if domain.kind is not DomainComponentKindV1.INTERVAL or domain.contains(pole):
            return None
        lower_denominator = c * domain.lower.numerator + d * domain.lower.denominator
        upper_denominator = c * domain.upper.numerator + d * domain.upper.denominator
        if lower_denominator == 0 or upper_denominator == 0:
            return None
        endpoint_values = (
            Fraction(a * domain.lower.numerator + b * domain.lower.denominator,
                     lower_denominator),
            Fraction(a * domain.upper.numerator + b * domain.upper.denominator,
                     upper_denominator),
        )
        lower = _bounded(min(endpoint_values))
        upper = _bounded(max(endpoint_values))
        if lower is None or upper is None:
            return None
        increasing = determinant > 0
        target_domain = DomainComponentV1(
            DomainComponentKindV1.INTERVAL,
            lower,
            upper,
            domain.lower_open if increasing else domain.upper_open,
            domain.upper_open if increasing else domain.lower_open,
        )
    else:
        scale = Fraction(a, d)
        shift = Fraction(b, d)
        target_domain = _affine_image(domain, scale, shift)
        if target_domain is None:
            return None
    return EquivalenceWitnessV1(
        EquivalenceGrammarV1.MOBIUS_INVARIANT_Z,
        {"a": a, "b": b, "c": c, "d": d},
        "z=(a*eta+b)/(c*eta+d)",
        "eta=(b-d*z)/(c*z-a)",
        domain,
        target_domain,
        "(a*d-b*c)/(c*eta+d)**2",
        ("determinant!=0", "source_excludes_pole", "target_excludes_pole"),
        ("inverse(forward(eta))=eta", "forward(inverse(z))=z"),
        representative_key,
    )


def monomial_invariant_witness(
    scale: BoundedRationalV1,
    exponent: int,
    source_domain: DomainComponentV1,
    representative_key: str = "monomial",
) -> EquivalenceWitnessV1 | None:
    if not scale.nonzero or exponent == 0 or abs(exponent) > 8:
        return None
    if exponent != 1 and source_domain.kind not in {
        DomainComponentKindV1.STRICTLY_POSITIVE,
        DomainComponentKindV1.STRICTLY_NEGATIVE,
    }:
        return None
    sign = -1 if source_domain.kind is DomainComponentKindV1.STRICTLY_NEGATIVE else 1
    magnitude = abs(exponent)
    if exponent == 1:
        inverse = "eta=z/s"
    elif exponent > 0:
        inverse = f"eta={sign}*Abs(z/s)**(1/{magnitude})"
    else:
        inverse = f"eta={sign}*Abs(s/z)**(1/{magnitude})"
    image_rule = "z/s>0" if magnitude % 2 == 0 else f"sign(z/s)={sign}"
    obligations = [image_rule, "selected_real_root_unique", "jacobian!=0"]
    if exponent < 0:
        obligations.extend(("eta!=0", "z!=0"))
    if exponent == 1 and source_domain.kind is DomainComponentKindV1.ALL_REAL:
        target_domain = _all_real()
    else:
        image_sign = (1 if scale.numerator > 0 else -1) * (
            1 if magnitude % 2 == 0 else sign
        )
        target_domain = DomainComponentV1(
            DomainComponentKindV1.STRICTLY_POSITIVE
            if image_sign > 0
            else DomainComponentKindV1.STRICTLY_NEGATIVE
        )
    return EquivalenceWitnessV1(
        EquivalenceGrammarV1.MONOMIAL_INVARIANT_QZ,
        {"scale": scale.to_mapping(), "exponent": exponent, "source_sign": sign},
        f"z=({scale.fraction})*eta**{exponent}",
        inverse,
        source_domain,
        target_domain,
        f"({scale.fraction})*{exponent}*eta**{exponent-1}",
        tuple(obligations),
        ("inverse(forward(eta))=eta", "forward(inverse(z))=z"),
        representative_key,
    )


def profile_affine_witness(
    scale: BoundedRationalV1,
    shift: BoundedRationalV1,
    *,
    absorbable: bool,
    representative_key: str = "profile",
) -> EquivalenceWitnessV1 | None:
    if not scale.nonzero or not absorbable:
        return None
    return EquivalenceWitnessV1(
        EquivalenceGrammarV1.PROFILE_AFFINE_Q,
        {"scale": scale.to_mapping(), "shift": shift.to_mapping()},
        "G=a*F+b",
        "F=(G-b)/a",
        _all_real(),
        _all_real(),
        str(scale.fraction),
        ("scale!=0", "profile_amplitude_offset_absorbable"),
        ("inverse(forward(F))=F", "transformed_reduced_equation_replays"),
        representative_key,
    )


def _as_bounded_rational(expression: ExpressionIRV1) -> BoundedRationalV1 | None:
    if expression.kind is not ExpressionKindV1.RATIONAL:
        return None
    try:
        return BoundedRationalV1(expression.numerator, expression.denominator)
    except ValueError:
        return None


def reduced_equation_factor_witness(
    source: tuple[ExpressionIRV1, ...],
    target: tuple[ExpressionIRV1, ...],
    *,
    backend: CASBackendV1,
    same_domain: bool = True,
    domain_conditions: tuple[str, ...] = (),
    semantic_binding: Mapping[str, Any] | None = None,
    source_candidate: DiscoveryCandidateV1 | None = None,
    target_candidate: DiscoveryCandidateV1 | None = None,
    representative_key: str = "reduced",
) -> EquivalenceWitnessV1 | None:
    if len(source) != len(target) or not source or not same_domain:
        return None
    factors: list[BoundedRationalV1] = []
    identities: list[str] = []
    for index, (left, right) in enumerate(zip(source, target)):
        ratio = backend.cancel(divide(right, left)).expression
        factor = _as_bounded_rational(ratio)
        if factor is None or not factor.nonzero:
            return None
        if backend.prove_zero(subtract(right, mul(rational(factor.fraction), left))).status is not ProofStatusV1.PROVED_ZERO:
            return None
        factors.append(factor)
        identities.append(f"R2_{index}=({factor.fraction})*R1_{index}")
    if (source_candidate is None) is not (target_candidate is None):
        return None
    parameters = {
        "factors": [item.to_mapping() for item in factors],
        "source_equations": [item.to_mapping() for item in source],
        "target_equations": [item.to_mapping() for item in target],
        "same_domain": True,
        "domain_conditions": list(domain_conditions),
        "semantic_binding": dict(semantic_binding or {}),
    }
    if source_candidate is not None and target_candidate is not None:
        parameters["source_candidate"] = source_candidate.to_mapping()
        parameters["target_candidate"] = target_candidate.to_mapping()
    return EquivalenceWitnessV1(
        EquivalenceGrammarV1.REDUCED_EQUATION_FACTOR_Q,
        parameters,
        "R2_i=a_i*R1_i",
        "R1_i=R2_i/a_i",
        _all_real(),
        _all_real(),
        "product(a_i)",
        (
            "all_factors_bounded_rational_nonzero",
            "equation_order_preserved",
            "certificate_side_conditions_exact",
            "ansatz_coordinate_profile_branch_exact",
        ),
        tuple(identities),
        representative_key,
    )


def _representative_key(candidate: DiscoveryCandidateV1) -> tuple[int, int, bytes, str]:
    tier = 0 if candidate.generator_key.startswith("direct:") else 1
    return (tier, len(candidate.ansatz) + len(candidate.coordinate), candidate.canonical_bytes, candidate.candidate_id)


def _candidate_reduced_equations(
    candidate: DiscoveryCandidateV1,
) -> tuple[ExpressionIRV1, ...] | None:
    certificate = candidate.verification.certificate
    metadata = thaw_json(candidate.metadata)
    exact = metadata.get("certificate_ir")
    if certificate is None or not isinstance(exact, dict):
        return None
    reduced = exact.get("reduced_equation")
    try:
        return (ExpressionIRV1.from_mapping(reduced),)
    except (TypeError, ValueError):
        return None


def _candidate_semantic_binding(
    candidate: DiscoveryCandidateV1,
) -> dict[str, Any] | None:
    metadata = thaw_json(candidate.metadata)
    exact = metadata.get("certificate_ir")
    definition = metadata.get("ansatz_definition")
    certificate = candidate.verification.certificate
    if (
        not isinstance(exact, dict)
        or not isinstance(definition, dict)
        or certificate is None
    ):
        return None
    try:
        from ..ir import ParserContextV1, parse_expression

        coordinate_name = definition["coordinate_name"]
        context = ParserContextV1(
            (),
            tuple(dict.fromkeys((*exact["old_variables"], coordinate_name))),
            tuple(
                dict.fromkeys(
                    (*candidate.free_parameters, *candidate.branch.assignments)
                )
            ),
            candidate.profile_functions,
        )
        ansatz_ir = parse_expression(definition["expression"], context)
        coordinate_ir = ExpressionIRV1.from_mapping(exact["coordinate"])
        if (
            parse_expression(candidate.ansatz.rsplit("=", 1)[-1], context)
            != ansatz_ir
            or parse_expression(candidate.coordinate, context) != coordinate_ir
        ):
            return None
        return {
            "field": definition["field"],
            "ansatz_ir": ansatz_ir.to_mapping(),
            "coordinate_ir": coordinate_ir.to_mapping(),
            "coordinate_name": coordinate_name,
            "profile_functions": list(candidate.profile_functions),
            "free_parameters": list(candidate.free_parameters),
            "branch": candidate.branch.to_mapping(),
            "inverse_map_ir": exact["inverse_map"],
            "chart": {
                "eta_ir": coordinate_ir.to_mapping(),
                "complementary_coordinate": certificate.chart.complementary_coordinate,
                "domain_conditions": list(certificate.chart.domain_conditions),
                "jacobian_ir": parse_expression(
                    certificate.chart.jacobian, context
                ).to_mapping(),
                "jacobian_proof": certificate.chart.jacobian_proof.value,
                "singular_set": list(certificate.chart.singular_set),
            },
        }
    except (KeyError, TypeError, ValueError):
        return None


def _semantic_factor_witness(
    source: DiscoveryCandidateV1,
    target: DiscoveryCandidateV1,
) -> EquivalenceWitnessV1 | None:
    left = _candidate_reduced_equations(source)
    right = _candidate_reduced_equations(target)
    left_certificate = source.verification.certificate
    right_certificate = target.verification.certificate
    if (
        left is None
        or right is None
        or left_certificate is None
        or right_certificate is None
        or source.condition_compatibility.overall
        is not target.condition_compatibility.overall
    ):
        return None
    semantic_binding = _candidate_semantic_binding(source)
    target_binding = _candidate_semantic_binding(target)
    if semantic_binding is None or semantic_binding != target_binding:
        return None
    same_domain = (
        left_certificate.side_conditions == right_certificate.side_conditions
        and left_certificate.parameter_conditions
        == right_certificate.parameter_conditions
    )
    backend = _backend_for_expressions((*left, *right))
    return reduced_equation_factor_witness(
        left,
        right,
        backend=backend,
        same_domain=same_domain,
        domain_conditions=left_certificate.side_conditions,
        semantic_binding=semantic_binding,
        source_candidate=source,
        target_candidate=target,
        representative_key=source.candidate_id,
    )


def _embedded_candidate_source_replays(candidate: DiscoveryCandidateV1) -> bool:
    """Authenticate an embedded candidate against its source problem and certificate."""

    from ..cas.sympy_backend import SymPyBackendV1
    from ..generation.direct import DirectAnsatzV1
    from ..verification.replay import replay_candidate_certificate

    metadata = thaw_json(candidate.metadata)
    problem_mapping = metadata.get("problem_input")
    definition = metadata.get("ansatz_definition")
    if not isinstance(problem_mapping, dict) or not isinstance(definition, dict):
        return False
    problem = DiscoveryProblemInputV1.from_mapping(problem_mapping)
    ansatz = DirectAnsatzV1.from_mapping(definition)
    expected_id = canonical_sha256(
        {
            "kind": "DirectReductionCandidateV1",
            "problem": problem.to_mapping(),
            "ansatz": ansatz.to_mapping(),
            "branch": ansatz.branch.to_mapping(),
        }
    )
    if candidate.candidate_id != expected_id or candidate.branch != ansatz.branch:
        return False
    roles = {
        name: SymbolRoleV1.INDEPENDENT for name in problem.independent_variables
    }
    roles[ansatz.coordinate_name] = SymbolRoleV1.PROFILE_COORDINATE
    roles.update(
        {
            name: SymbolRoleV1.PARAMETER
            for name in dict.fromkeys(
                (
                    *problem.parameters,
                    *candidate.free_parameters,
                    *candidate.branch.assignments,
                )
            )
        }
    )
    backend = SymPyBackendV1(
        problem.independent_variables, roles, candidate.profile_functions
    )
    return replay_candidate_certificate(candidate, problem, backend).valid


def deduplicate_reduction_candidates(
    candidates: tuple[DiscoveryCandidateV1, ...],
) -> tuple[DiscoveryCandidateV1, ...]:
    groups: list[list[DiscoveryCandidateV1]] = []
    for candidate in sorted(candidates, key=_representative_key):
        for group in groups:
            representative = group[0]
            if (
                representative.condition_compatibility.overall
                is candidate.condition_compatibility.overall
                and _semantic_factor_witness(representative, candidate) is not None
            ):
                group.append(candidate)
                break
        else:
            groups.append([candidate])
    representatives: list[DiscoveryCandidateV1] = []
    for group in groups:
        representative = group[0]
        if len(group) > 1:
            member_witnesses = []
            for item in group[1:]:
                semantic = _semantic_factor_witness(representative, item)
                if semantic is None:
                    raise RuntimeError("semantic equivalence group lost its replay witness")
                member_witnesses.append(
                    {"candidate_id": item.candidate_id, "witness": semantic.to_mapping()}
                )
            witness = {
                "schema_version": "ReductionClassMergeWitnessV1",
                "transform": EquivalenceGrammarV1.REDUCED_EQUATION_FACTOR_Q.value,
                "representative_id": representative.candidate_id,
                "merged_candidate_ids": [item.candidate_id for item in group[1:]],
                "member_witnesses": member_witnesses,
                "proof": "replayed_nonzero_rational_factors_same_domain",
            }
            witness["replay_digest"] = canonical_sha256(witness)
            metadata = thaw_json(representative.metadata)
            metadata["equivalent_derivations"] = [
                {
                    "candidate_id": item.candidate_id,
                    "generator_key": item.generator_key,
                    "production_trace": list(item.production_trace),
                    "classical_lie": thaw_json(item.metadata).get("classical_lie"),
                }
                for item in group
            ]
            traces = tuple(
                dict.fromkeys(
                    trace for item in group for trace in item.production_trace
                )
            )
            representative = replace(
                representative,
                production_trace=traces,
                equivalence_witness=witness,
                metadata=metadata,
            )
        representatives.append(representative)
    return tuple(representatives)


def _backend_for_expressions(expressions: tuple[ExpressionIRV1, ...]) -> CASBackendV1:
    # Kept lazy so importing public contracts never imports SymPy in the caller process.
    from ..cas.sympy_backend import SymPyBackendV1

    roles: dict[str, SymbolRoleV1] = {}
    functions: set[str] = set()

    def visit(expression: ExpressionIRV1) -> None:
        if expression.kind is ExpressionKindV1.SYMBOL:
            roles[expression.name] = expression.role
        elif expression.kind is ExpressionKindV1.FUNCTION:
            functions.add(expression.name)
        for operand in expression.operands:
            visit(operand)

    for expression in expressions:
        visit(expression)
    independent = tuple(
        sorted(name for name, role in roles.items() if role is SymbolRoleV1.INDEPENDENT)
    )
    return SymPyBackendV1(independent, roles, tuple(sorted(functions)))


def _rebuild_witness(payload: dict[str, Any]) -> EquivalenceWitnessV1 | None:
    transform = EquivalenceGrammarV1(payload["transform"])
    parameters = payload["parameters"]
    source = DomainComponentV1.from_mapping(payload["source_domain"])
    representative = payload["representative_key"]
    if transform is EquivalenceGrammarV1.PROJECTIVE_GENERATOR_Q:
        return projective_generator_witness(
            BoundedRationalV1.from_mapping(parameters["scale"]), representative
        )
    if transform is EquivalenceGrammarV1.AFFINE_INVARIANT_Q:
        return affine_invariant_witness(
            BoundedRationalV1.from_mapping(parameters["scale"]),
            BoundedRationalV1.from_mapping(parameters["shift"]),
            source,
            representative,
        )
    if transform is EquivalenceGrammarV1.MOBIUS_INVARIANT_Z:
        return mobius_invariant_witness(
            (
                parameters["a"],
                parameters["b"],
                parameters["c"],
                parameters["d"],
            ),
            source,
            representative,
        )
    if transform is EquivalenceGrammarV1.MONOMIAL_INVARIANT_QZ:
        return monomial_invariant_witness(
            BoundedRationalV1.from_mapping(parameters["scale"]),
            parameters["exponent"],
            source,
            representative,
        )
    if transform is EquivalenceGrammarV1.PROFILE_AFFINE_Q:
        return profile_affine_witness(
            BoundedRationalV1.from_mapping(parameters["scale"]),
            BoundedRationalV1.from_mapping(parameters["shift"]),
            absorbable="profile_amplitude_offset_absorbable" in payload["obligations"],
            representative_key=representative,
        )
    if transform is EquivalenceGrammarV1.REDUCED_EQUATION_FACTOR_Q:
        if "source_candidate" in parameters or "target_candidate" in parameters:
            if not (
                "source_candidate" in parameters
                and "target_candidate" in parameters
            ):
                return None
            source_candidate = DiscoveryCandidateV1.from_mapping(
                parameters["source_candidate"]
            )
            target_candidate = DiscoveryCandidateV1.from_mapping(
                parameters["target_candidate"]
            )
            if not (
                _embedded_candidate_source_replays(source_candidate)
                and _embedded_candidate_source_replays(target_candidate)
            ):
                return None
            return _semantic_factor_witness(source_candidate, target_candidate)
        source_equations = tuple(
            ExpressionIRV1.from_mapping(item) for item in parameters["source_equations"]
        )
        target_equations = tuple(
            ExpressionIRV1.from_mapping(item) for item in parameters["target_equations"]
        )
        return reduced_equation_factor_witness(
            source_equations,
            target_equations,
            backend=_backend_for_expressions((*source_equations, *target_equations)),
            same_domain=parameters["same_domain"] is True,
            domain_conditions=tuple(parameters.get("domain_conditions", ())),
            semantic_binding=parameters.get("semantic_binding", {}),
            representative_key=representative,
        )
    return None


def _replay_equivalence_witness_mapping_in_process(value: Mapping[str, Any]) -> bool:
    """Rebuild and verify a witness inside an already resource-bounded worker."""

    if not isinstance(value, Mapping) or "replay_digest" not in value:
        return False
    try:
        payload = thaw_json(freeze_json(value))
        digest = payload.pop("replay_digest")
        if not isinstance(digest, str) or canonical_sha256(payload) != digest:
            return False
        if payload.get("schema_version") == "ReductionClassMergeWitnessV1":
            expected = {
                "schema_version",
                "transform",
                "representative_id",
                "merged_candidate_ids",
                "member_witnesses",
                "proof",
            }
            if (
                set(payload) != expected
                or payload["proof"]
                != "replayed_nonzero_rational_factors_same_domain"
            ):
                return False
            members = payload["member_witnesses"]
            if not isinstance(members, (list, tuple)) or [
                item["candidate_id"] for item in members
            ] != list(payload["merged_candidate_ids"]):
                return False
            for item in members:
                if (
                    not isinstance(item, Mapping)
                    or set(item) != {"candidate_id", "witness"}
                    or not _replay_equivalence_witness_mapping_in_process(
                        item["witness"]
                    )
                ):
                    return False
                witness = item["witness"]
                parameters = witness["parameters"]
                source = parameters.get("source_candidate")
                target = parameters.get("target_candidate")
                if (
                    not isinstance(source, Mapping)
                    or not isinstance(target, Mapping)
                    or source.get("candidate_id") != payload["representative_id"]
                    or target.get("candidate_id") != item["candidate_id"]
                    or witness.get("representative_key")
                    != payload["representative_id"]
                ):
                    return False
            return True
        expected = {
            "schema_version",
            "transform",
            "parameters",
            "forward",
            "inverse",
            "source_domain",
            "target_domain",
            "jacobian",
            "obligations",
            "identities",
            "representative_key",
        }
        if set(payload) != expected or payload["schema_version"] != "EquivalenceWitnessV1":
            return False
        rebuilt = _rebuild_witness(payload)
        return rebuilt is not None and rebuilt._payload() == payload
    except Exception:  # noqa: BLE001 - untrusted witness replay fails closed
        return False


def replay_equivalence_witness_mapping(value: Mapping[str, Any]) -> bool:
    """Fail closed after replaying the witness in a fresh resource-bounded worker."""

    try:
        from ..cas.worker import (
            WorkerOperationV1,
            WorkerRequestV1,
            run_worker_job,
        )
        from ..models import WorkerStatusV1

        result = run_worker_job(
            WorkerRequestV1(
                WorkerOperationV1.REPLAY_EQUIVALENCE,
                None,
                options={"witness": thaw_json(freeze_json(value))},
            ),
            deadline_seconds=5,
            cpu_seconds=5,
            result_size_limit=64 * 1024,
        )
    except Exception:  # noqa: BLE001 - worker/replay failures fail closed
        return False
    return (
        result.status is WorkerStatusV1.COMPLETED
        and result.payload is not None
        and result.payload.get("valid") is True
    )


__all__ = [
    "BoundedRationalV1",
    "DomainComponentKindV1",
    "DomainComponentV1",
    "EquivalenceGrammarV1",
    "EquivalenceWitnessV1",
    "affine_invariant_witness",
    "deduplicate_reduction_candidates",
    "mobius_invariant_witness",
    "monomial_invariant_witness",
    "profile_affine_witness",
    "projective_generator_witness",
    "reduced_equation_factor_witness",
    "replay_equivalence_witness_mapping",
]
