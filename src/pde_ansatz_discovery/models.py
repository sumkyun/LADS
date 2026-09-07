"""Immutable public contracts for PDE Ansatz discovery."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, TypeVar

from .serialization import canonical_json_bytes, canonical_sha256, freeze_json, thaw_json

_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_T = TypeVar("_T")


class DiscoveryContractError(ValueError):
    """Raised when a versioned discovery record is invalid."""


class _ClosedEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class VerificationStatusV1(_ClosedEnum):
    CERTIFIED_VALID = "certified_valid"
    CERTIFIED_INVALID = "certified_invalid"
    UNRESOLVED = "unresolved"


class ProofStatusV1(_ClosedEnum):
    PROVED_ZERO = "proved_zero"
    PROVED_NONZERO = "proved_nonzero"
    UNKNOWN = "unknown"


class BranchCoverageV1(_ClosedEnum):
    EXHAUSTIVE = "exhaustive"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class WorkerStatusV1(_ClosedEnum):
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    CRASHED = "crashed"


class ConditionStatusV1(_ClosedEnum):
    COMPATIBLE = "compatible"
    COMPATIBLE_WITH_CONSTRAINTS = "compatible_with_constraints"
    NO_CONDITION = "no_condition"
    UNRESOLVED = "unresolved"
    SINGULAR = "singular"
    INCOMPATIBLE = "incompatible"


class DependenceStatusV1(_ClosedEnum):
    PROVED_DEPENDENCE = "proved_dependence"
    PROVED_NOT_DEPENDENCE = "proved_not_dependence"
    UNKNOWN = "unknown"


CONDITION_ELIGIBLE_V1 = frozenset(
    {
        ConditionStatusV1.COMPATIBLE,
        ConditionStatusV1.COMPATIBLE_WITH_CONSTRAINTS,
        ConditionStatusV1.NO_CONDITION,
        ConditionStatusV1.UNRESOLVED,
    }
)
CONDITION_REJECTED_V1 = frozenset(
    {ConditionStatusV1.SINGULAR, ConditionStatusV1.INCOMPATIBLE}
)


def _exact_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise DiscoveryContractError(f"{label} must be a mapping")
    actual = set(value)
    if actual != expected:
        raise DiscoveryContractError(
            f"{label} fields differ: missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )


def _strings(
    value: Sequence[str],
    *,
    field: str,
    identifiers: bool = False,
    nonempty: bool = False,
    unique: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DiscoveryContractError(f"{field} must be a sequence of strings")
    result = tuple(value)
    if nonempty and not result:
        raise DiscoveryContractError(f"{field} must not be empty")
    if any(not isinstance(item, str) or not item for item in result):
        raise DiscoveryContractError(f"{field} must contain nonempty strings")
    if identifiers and any(not _IDENTIFIER.fullmatch(item) for item in result):
        raise DiscoveryContractError(f"{field} contains an invalid identifier")
    if unique and len(set(result)) != len(result):
        raise DiscoveryContractError(f"{field} must be unique")
    return result


def _positive_int(value: Any, *, field: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise DiscoveryContractError(f"{field} must be an integer in [1,{maximum}]")
    return value


def _enum(enum_type: type[_T], value: Any, *, field: str) -> _T:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise DiscoveryContractError(f"invalid {field}: {value!r}") from exc


def _hash(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise DiscoveryContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


class _CanonicalRecord:
    def to_mapping(self) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def canonical_hash(self) -> str:
        return canonical_sha256(self.to_mapping())


@dataclass(frozen=True)
class DiscoveryProblemInputV1(_CanonicalRecord):
    dependent_variables: tuple[str, ...]
    independent_variables: tuple[str, ...]
    pde: tuple[str, ...]
    domain: Mapping[str, Any]
    parameters: Mapping[str, Any]
    initial_conditions: tuple[str, ...]
    boundary_conditions: tuple[str, ...]
    additional_constraints: tuple[str, ...]

    schema_version: ClassVar[str] = "DiscoveryProblemInputV1"

    def __post_init__(self) -> None:
        dependent = _strings(
            self.dependent_variables,
            field="dependent_variables",
            identifiers=True,
            nonempty=True,
            unique=True,
        )
        independent = _strings(
            self.independent_variables,
            field="independent_variables",
            identifiers=True,
            nonempty=True,
            unique=True,
        )
        if set(dependent) & set(independent):
            raise DiscoveryContractError("dependent and independent variables must be disjoint")
        object.__setattr__(self, "dependent_variables", dependent)
        object.__setattr__(self, "independent_variables", independent)
        object.__setattr__(self, "pde", _strings(self.pde, field="pde", nonempty=True))
        object.__setattr__(self, "domain", freeze_json(self.domain))
        object.__setattr__(self, "parameters", freeze_json(self.parameters))
        object.__setattr__(
            self, "initial_conditions", _strings(self.initial_conditions, field="initial_conditions")
        )
        object.__setattr__(
            self, "boundary_conditions", _strings(self.boundary_conditions, field="boundary_conditions")
        )
        object.__setattr__(
            self,
            "additional_constraints",
            _strings(self.additional_constraints, field="additional_constraints"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dependent_variables": list(self.dependent_variables),
            "independent_variables": list(self.independent_variables),
            "pde": list(self.pde),
            "domain": thaw_json(self.domain),
            "parameters": thaw_json(self.parameters),
            "initial_conditions": list(self.initial_conditions),
            "boundary_conditions": list(self.boundary_conditions),
            "additional_constraints": list(self.additional_constraints),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DiscoveryProblemInputV1":
        expected = {
            "schema_version",
            "dependent_variables",
            "independent_variables",
            "pde",
            "domain",
            "parameters",
            "initial_conditions",
            "boundary_conditions",
            "additional_constraints",
        }
        _exact_fields(value, expected, "discovery problem")
        if value["schema_version"] != cls.schema_version:
            raise DiscoveryContractError("unsupported discovery problem schema")
        return cls(
            dependent_variables=value["dependent_variables"],
            independent_variables=value["independent_variables"],
            pde=value["pde"],
            domain=value["domain"],
            parameters=value["parameters"],
            initial_conditions=value["initial_conditions"],
            boundary_conditions=value["boundary_conditions"],
            additional_constraints=value["additional_constraints"],
        )

    @property
    def input_hash(self) -> str:
        return self.canonical_hash


@dataclass(frozen=True)
class DiscoveryBudgetV1(_CanonicalRecord):
    grammar_depth: int
    generator_basis_degree: int
    beam_width: int
    total_candidates: int
    parameter_branch_cap: int
    expression_node_cap: int
    cas_call_deadline_seconds: int
    characteristic_deadline_seconds: int
    total_wall_clock_seconds: int

    schema_version: ClassVar[str] = "DiscoveryBudgetV1"
    _LIMITS: ClassVar[dict[str, int]] = {
        "grammar_depth": 16,
        "generator_basis_degree": 8,
        "beam_width": 4096,
        "total_candidates": 100_000,
        "parameter_branch_cap": 4096,
        "expression_node_cap": 1_000_000,
        "cas_call_deadline_seconds": 3600,
        "characteristic_deadline_seconds": 3600,
        "total_wall_clock_seconds": 86_400,
    }

    def __post_init__(self) -> None:
        for name, maximum in self._LIMITS.items():
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), field=name, maximum=maximum)
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            **{name: getattr(self, name) for name in self._LIMITS},
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DiscoveryBudgetV1":
        expected = {"schema_version", *cls._LIMITS}
        _exact_fields(value, expected, "discovery budget")
        if value["schema_version"] != cls.schema_version:
            raise DiscoveryContractError("unsupported discovery budget schema")
        return cls(
            grammar_depth=value["grammar_depth"],
            generator_basis_degree=value["generator_basis_degree"],
            beam_width=value["beam_width"],
            total_candidates=value["total_candidates"],
            parameter_branch_cap=value["parameter_branch_cap"],
            expression_node_cap=value["expression_node_cap"],
            cas_call_deadline_seconds=value["cas_call_deadline_seconds"],
            characteristic_deadline_seconds=value[
                "characteristic_deadline_seconds"
            ],
            total_wall_clock_seconds=value["total_wall_clock_seconds"],
        )

    @property
    def config_hash(self) -> str:
        return self.canonical_hash


@dataclass(frozen=True)
class ConditionRecordV1(_CanonicalRecord):
    kind: str
    status: ConditionStatusV1
    expression: str | None = None
    details: Mapping[str, Any] = None  # type: ignore[assignment]

    schema_version: ClassVar[str] = "ConditionRecordV1"

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise DiscoveryContractError("condition kind must be nonempty")
        object.__setattr__(self, "status", _enum(ConditionStatusV1, self.status, field="condition status"))
        if self.expression is not None and not isinstance(self.expression, str):
            raise DiscoveryContractError("condition expression must be a string or null")
        object.__setattr__(self, "details", freeze_json(self.details or {}))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "status": self.status.value,
            "expression": self.expression,
            "details": thaw_json(self.details),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ConditionRecordV1":
        expected = {"schema_version", "kind", "status", "expression", "details"}
        _exact_fields(value, expected, "condition record")
        if value["schema_version"] != cls.schema_version:
            raise DiscoveryContractError("unsupported condition record schema")
        return cls(value["kind"], value["status"], value["expression"], value["details"])


def aggregate_condition_status(
    records: Sequence[ConditionRecordV1],
) -> ConditionStatusV1:
    statuses = {record.status for record in records}
    for status in (
        ConditionStatusV1.INCOMPATIBLE,
        ConditionStatusV1.SINGULAR,
        ConditionStatusV1.UNRESOLVED,
        ConditionStatusV1.COMPATIBLE_WITH_CONSTRAINTS,
    ):
        if status in statuses:
            return status
    if not records:
        return ConditionStatusV1.NO_CONDITION
    return ConditionStatusV1.COMPATIBLE


@dataclass(frozen=True)
class ConditionCompatibilityV1(_CanonicalRecord):
    records: tuple[ConditionRecordV1, ...]
    overall: ConditionStatusV1 | None = None

    schema_version: ClassVar[str] = "ConditionCompatibilityV1"

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if any(not isinstance(record, ConditionRecordV1) for record in records):
            raise DiscoveryContractError("condition records must be typed records")
        expected = aggregate_condition_status(records)
        supplied = expected if self.overall is None else _enum(
            ConditionStatusV1, self.overall, field="overall condition status"
        )
        if supplied is not expected:
            raise DiscoveryContractError("overall condition status disagrees with records")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "overall", expected)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "records": [record.to_mapping() for record in self.records],
            "overall": self.overall.value,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ConditionCompatibilityV1":
        expected = {"schema_version", "records", "overall"}
        _exact_fields(value, expected, "condition compatibility")
        if value["schema_version"] != cls.schema_version:
            raise DiscoveryContractError("unsupported condition compatibility schema")
        return cls(
            tuple(ConditionRecordV1.from_mapping(item) for item in value["records"]),
            value["overall"],
        )


@dataclass(frozen=True)
class ParameterBranchV1(_CanonicalRecord):
    branch_id: str
    assignments: Mapping[str, Any]
    assumptions: tuple[str, ...]
    side_conditions: tuple[str, ...]
    coverage: BranchCoverageV1

    schema_version: ClassVar[str] = "ParameterBranchV1"

    def __post_init__(self) -> None:
        if not isinstance(self.branch_id, str) or not self.branch_id:
            raise DiscoveryContractError("branch_id must be nonempty")
        object.__setattr__(self, "assignments", freeze_json(self.assignments))
        object.__setattr__(self, "assumptions", _strings(self.assumptions, field="assumptions"))
        object.__setattr__(self, "side_conditions", _strings(self.side_conditions, field="side_conditions"))
        object.__setattr__(self, "coverage", _enum(BranchCoverageV1, self.coverage, field="branch coverage"))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "branch_id": self.branch_id,
            "assignments": thaw_json(self.assignments),
            "assumptions": list(self.assumptions),
            "side_conditions": list(self.side_conditions),
            "coverage": self.coverage.value,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ParameterBranchV1":
        expected = {"schema_version", "branch_id", "assignments", "assumptions", "side_conditions", "coverage"}
        _exact_fields(value, expected, "parameter branch")
        if value["schema_version"] != cls.schema_version:
            raise DiscoveryContractError("unsupported parameter branch schema")
        return cls(value["branch_id"], value["assignments"], tuple(value["assumptions"]), tuple(value["side_conditions"]), value["coverage"])


@dataclass(frozen=True)
class ReductionChartV1(_CanonicalRecord):
    eta: str
    complementary_coordinate: str
    forward_map: Mapping[str, Any]
    inverse_map: Mapping[str, Any]
    domain_conditions: tuple[str, ...]
    jacobian: str
    jacobian_proof: ProofStatusV1
    singular_set: tuple[str, ...]

    schema_version: ClassVar[str] = "ReductionChartV1"

    def __post_init__(self) -> None:
        for field_name in ("eta", "complementary_coordinate", "jacobian"):
            if not isinstance(getattr(self, field_name), str) or not getattr(self, field_name):
                raise DiscoveryContractError(f"{field_name} must be nonempty")
        object.__setattr__(self, "forward_map", freeze_json(self.forward_map))
        object.__setattr__(self, "inverse_map", freeze_json(self.inverse_map))
        object.__setattr__(self, "domain_conditions", _strings(self.domain_conditions, field="domain_conditions"))
        object.__setattr__(self, "singular_set", _strings(self.singular_set, field="singular_set"))
        proof = _enum(ProofStatusV1, self.jacobian_proof, field="Jacobian proof")
        if proof is not ProofStatusV1.PROVED_NONZERO:
            raise DiscoveryContractError("a certified reduction chart requires a nonzero Jacobian proof")
        object.__setattr__(self, "jacobian_proof", proof)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "eta": self.eta,
            "complementary_coordinate": self.complementary_coordinate,
            "forward_map": thaw_json(self.forward_map),
            "inverse_map": thaw_json(self.inverse_map),
            "domain_conditions": list(self.domain_conditions),
            "jacobian": self.jacobian,
            "jacobian_proof": self.jacobian_proof.value,
            "singular_set": list(self.singular_set),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReductionChartV1":
        expected = {"schema_version", "eta", "complementary_coordinate", "forward_map", "inverse_map", "domain_conditions", "jacobian", "jacobian_proof", "singular_set"}
        _exact_fields(value, expected, "reduction chart")
        if value["schema_version"] != cls.schema_version:
            raise DiscoveryContractError("unsupported reduction chart schema")
        return cls(value["eta"], value["complementary_coordinate"], value["forward_map"], value["inverse_map"], tuple(value["domain_conditions"]), value["jacobian"], value["jacobian_proof"], tuple(value["singular_set"]))


@dataclass(frozen=True)
class CoefficientDependenceResultV1(_CanonicalRecord):
    status: DependenceStatusV1
    ratio_expression: str
    rho_expression: str | None
    identity: str | None
    side_conditions: tuple[str, ...]
    replay_digest: str | None

    schema_version: ClassVar[str] = "CoefficientDependenceResultV1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _enum(DependenceStatusV1, self.status, field="dependence status"))
        if not isinstance(self.ratio_expression, str) or not self.ratio_expression:
            raise DiscoveryContractError("ratio_expression must be nonempty")
        if self.status is DependenceStatusV1.PROVED_DEPENDENCE:
            if not self.rho_expression or not self.identity or self.replay_digest is None:
                raise DiscoveryContractError("proved dependence requires rho, identity, and replay digest")
        if self.replay_digest is not None:
            _hash(self.replay_digest, field="replay_digest")
        object.__setattr__(self, "side_conditions", _strings(self.side_conditions, field="dependence side_conditions"))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "ratio_expression": self.ratio_expression,
            "rho_expression": self.rho_expression,
            "identity": self.identity,
            "side_conditions": list(self.side_conditions),
            "replay_digest": self.replay_digest,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CoefficientDependenceResultV1":
        expected = {"schema_version", "status", "ratio_expression", "rho_expression", "identity", "side_conditions", "replay_digest"}
        _exact_fields(value, expected, "coefficient dependence")
        if value["schema_version"] != cls.schema_version:
            raise DiscoveryContractError("unsupported dependence schema")
        return cls(value["status"], value["ratio_expression"], value["rho_expression"], value["identity"], tuple(value["side_conditions"]), value["replay_digest"])


@dataclass(frozen=True)
class ReductionCertificateV1(_CanonicalRecord):
    substituted_residual: str
    prefactor: str
    reduced_equations: tuple[str, ...]
    old_variables_removed: bool
    allowed_symbols: tuple[str, ...]
    parameter_conditions: tuple[str, ...]
    side_conditions: tuple[str, ...]
    chart: ReductionChartV1
    coefficient_dependence: tuple[CoefficientDependenceResultV1, ...]
    replay_digest: str
    versions: Mapping[str, Any]

    schema_version: ClassVar[str] = "ReductionCertificateV1"

    def __post_init__(self) -> None:
        for name in ("substituted_residual", "prefactor"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise DiscoveryContractError(f"{name} must be nonempty")
        object.__setattr__(self, "reduced_equations", _strings(self.reduced_equations, field="reduced_equations", nonempty=True))
        if not isinstance(self.old_variables_removed, bool):
            raise DiscoveryContractError("old_variables_removed must be boolean")
        object.__setattr__(self, "allowed_symbols", _strings(self.allowed_symbols, field="allowed_symbols", identifiers=True, unique=True))
        object.__setattr__(self, "parameter_conditions", _strings(self.parameter_conditions, field="parameter_conditions"))
        object.__setattr__(self, "side_conditions", _strings(self.side_conditions, field="certificate side_conditions"))
        if not isinstance(self.chart, ReductionChartV1):
            raise DiscoveryContractError("certificate chart must be typed")
        dependence = tuple(self.coefficient_dependence)
        if any(not isinstance(item, CoefficientDependenceResultV1) for item in dependence):
            raise DiscoveryContractError("coefficient_dependence must be typed")
        object.__setattr__(self, "coefficient_dependence", dependence)
        _hash(self.replay_digest, field="replay_digest")
        object.__setattr__(self, "versions", freeze_json(self.versions))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "substituted_residual": self.substituted_residual,
            "prefactor": self.prefactor,
            "reduced_equations": list(self.reduced_equations),
            "old_variables_removed": self.old_variables_removed,
            "allowed_symbols": list(self.allowed_symbols),
            "parameter_conditions": list(self.parameter_conditions),
            "side_conditions": list(self.side_conditions),
            "chart": self.chart.to_mapping(),
            "coefficient_dependence": [item.to_mapping() for item in self.coefficient_dependence],
            "replay_digest": self.replay_digest,
            "versions": thaw_json(self.versions),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReductionCertificateV1":
        expected = {"schema_version", "substituted_residual", "prefactor", "reduced_equations", "old_variables_removed", "allowed_symbols", "parameter_conditions", "side_conditions", "chart", "coefficient_dependence", "replay_digest", "versions"}
        _exact_fields(value, expected, "reduction certificate")
        if value["schema_version"] != cls.schema_version:
            raise DiscoveryContractError("unsupported certificate schema")
        return cls(value["substituted_residual"], value["prefactor"], tuple(value["reduced_equations"]), value["old_variables_removed"], tuple(value["allowed_symbols"]), tuple(value["parameter_conditions"]), tuple(value["side_conditions"]), ReductionChartV1.from_mapping(value["chart"]), tuple(CoefficientDependenceResultV1.from_mapping(item) for item in value["coefficient_dependence"]), value["replay_digest"], value["versions"])


@dataclass(frozen=True)
class CandidateVerificationV1(_CanonicalRecord):
    status: VerificationStatusV1
    reason_code: str
    certificate: ReductionCertificateV1 | None

    schema_version: ClassVar[str] = "CandidateVerificationV1"

    def __post_init__(self) -> None:
        status = _enum(VerificationStatusV1, self.status, field="verification status")
        if not isinstance(self.reason_code, str) or not self.reason_code:
            raise DiscoveryContractError("reason_code must be nonempty")
        if status is VerificationStatusV1.CERTIFIED_VALID and not isinstance(self.certificate, ReductionCertificateV1):
            raise DiscoveryContractError("certified_valid requires a certificate")
        if status is VerificationStatusV1.UNRESOLVED and self.certificate is not None:
            raise DiscoveryContractError("unresolved cannot carry a partial certificate")
        object.__setattr__(self, "status", status)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "certificate": None if self.certificate is None else self.certificate.to_mapping(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CandidateVerificationV1":
        expected = {"schema_version", "status", "reason_code", "certificate"}
        _exact_fields(value, expected, "candidate verification")
        if value["schema_version"] != cls.schema_version:
            raise DiscoveryContractError("unsupported verification schema")
        certificate = None if value["certificate"] is None else ReductionCertificateV1.from_mapping(value["certificate"])
        return cls(value["status"], value["reason_code"], certificate)


@dataclass(frozen=True)
class DiscoveryCandidateV1(_CanonicalRecord):
    candidate_id: str
    ansatz: str
    coordinate: str
    profile_functions: tuple[str, ...]
    free_parameters: tuple[str, ...]
    branch: ParameterBranchV1
    production_trace: tuple[str, ...]
    failure_history: tuple[str, ...]
    verification: CandidateVerificationV1
    condition_compatibility: ConditionCompatibilityV1
    generator_key: str
    invariant_key: str
    reduction_key: str
    equivalence_witness: Mapping[str, Any] | None
    metadata: Mapping[str, Any]

    schema_version: ClassVar[str] = "DiscoveryCandidateV1"

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise DiscoveryContractError("candidate_id must be nonempty")
        for name in ("ansatz", "coordinate", "generator_key", "invariant_key", "reduction_key"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise DiscoveryContractError(f"{name} must be nonempty")
        object.__setattr__(self, "profile_functions", _strings(self.profile_functions, field="profile_functions", identifiers=True, nonempty=True, unique=True))
        object.__setattr__(self, "free_parameters", _strings(self.free_parameters, field="free_parameters", identifiers=True, unique=True))
        object.__setattr__(self, "production_trace", _strings(self.production_trace, field="production_trace", nonempty=True))
        object.__setattr__(self, "failure_history", _strings(self.failure_history, field="failure_history"))
        if not isinstance(self.branch, ParameterBranchV1):
            raise DiscoveryContractError("branch must be typed")
        if not isinstance(self.verification, CandidateVerificationV1):
            raise DiscoveryContractError("verification must be typed")
        if not isinstance(self.condition_compatibility, ConditionCompatibilityV1):
            raise DiscoveryContractError("condition compatibility must be typed")
        witness = None if self.equivalence_witness is None else freeze_json(self.equivalence_witness)
        object.__setattr__(self, "equivalence_witness", witness)
        object.__setattr__(self, "metadata", freeze_json(self.metadata))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "ansatz": self.ansatz,
            "coordinate": self.coordinate,
            "profile_functions": list(self.profile_functions),
            "free_parameters": list(self.free_parameters),
            "branch": self.branch.to_mapping(),
            "production_trace": list(self.production_trace),
            "failure_history": list(self.failure_history),
            "verification": self.verification.to_mapping(),
            "condition_compatibility": self.condition_compatibility.to_mapping(),
            "generator_key": self.generator_key,
            "invariant_key": self.invariant_key,
            "reduction_key": self.reduction_key,
            "equivalence_witness": None if self.equivalence_witness is None else thaw_json(self.equivalence_witness),
            "metadata": thaw_json(self.metadata),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DiscoveryCandidateV1":
        expected = {"schema_version", "candidate_id", "ansatz", "coordinate", "profile_functions", "free_parameters", "branch", "production_trace", "failure_history", "verification", "condition_compatibility", "generator_key", "invariant_key", "reduction_key", "equivalence_witness", "metadata"}
        _exact_fields(value, expected, "discovery candidate")
        if value["schema_version"] != cls.schema_version:
            raise DiscoveryContractError("unsupported candidate schema")
        return cls(value["candidate_id"], value["ansatz"], value["coordinate"], tuple(value["profile_functions"]), tuple(value["free_parameters"]), ParameterBranchV1.from_mapping(value["branch"]), tuple(value["production_trace"]), tuple(value["failure_history"]), CandidateVerificationV1.from_mapping(value["verification"]), ConditionCompatibilityV1.from_mapping(value["condition_compatibility"]), value["generator_key"], value["invariant_key"], value["reduction_key"], value["equivalence_witness"], value["metadata"])


@dataclass(frozen=True)
class DiscoveryResultV1(_CanonicalRecord):
    input_hash: str
    config_hash: str
    backend_version: str
    top_k: int
    valid_candidates: tuple[DiscoveryCandidateV1, ...]
    recommended_candidates: tuple[DiscoveryCandidateV1, ...]
    valid_condition_rejected: tuple[DiscoveryCandidateV1, ...]
    unresolved_candidates: tuple[DiscoveryCandidateV1, ...]
    failure_summaries: Mapping[str, Any]

    schema_version: ClassVar[str] = "DiscoveryResultV1"

    def __post_init__(self) -> None:
        _hash(self.input_hash, field="input_hash")
        _hash(self.config_hash, field="config_hash")
        if not isinstance(self.backend_version, str) or not self.backend_version:
            raise DiscoveryContractError("backend_version must be nonempty")
        _positive_int(self.top_k, field="top_k", maximum=100_000)
        valid = tuple(self.valid_candidates)
        recommended = tuple(self.recommended_candidates)
        rejected = tuple(self.valid_condition_rejected)
        unresolved = tuple(self.unresolved_candidates)
        for name, items in (("valid", valid), ("recommended", recommended), ("rejected", rejected), ("unresolved", unresolved)):
            if any(not isinstance(item, DiscoveryCandidateV1) for item in items):
                raise DiscoveryContractError(f"{name} candidates must be typed")
            ids = [item.candidate_id for item in items]
            if len(set(ids)) != len(ids):
                raise DiscoveryContractError(f"{name} candidate IDs must be unique")
        if any(item.verification.status is not VerificationStatusV1.CERTIFIED_VALID for item in valid):
            raise DiscoveryContractError("valid_candidates must all be certified_valid")
        if any(item.verification.status is not VerificationStatusV1.UNRESOLVED for item in unresolved):
            raise DiscoveryContractError("unresolved_candidates must all be unresolved")
        by_id = {item.candidate_id: item for item in valid}
        for projection in (*recommended, *rejected):
            if projection.candidate_id not in by_id or projection != by_id[projection.candidate_id]:
                raise DiscoveryContractError("result projections must reference exact valid records")
        eligible = tuple(
            item for item in valid if item.condition_compatibility.overall in CONDITION_ELIGIBLE_V1
        )
        expected_recommended = eligible[: self.top_k]
        expected_rejected = tuple(
            item for item in valid if item.condition_compatibility.overall in CONDITION_REJECTED_V1
        )
        if recommended != expected_recommended:
            raise DiscoveryContractError("recommended_candidates must be the eligible Top-K projection")
        if rejected != expected_rejected:
            raise DiscoveryContractError("valid_condition_rejected must be the complete rejected projection")
        valid_ids = set(by_id)
        if valid_ids & {item.candidate_id for item in unresolved}:
            raise DiscoveryContractError("valid and unresolved candidate IDs must be disjoint")
        object.__setattr__(self, "valid_candidates", valid)
        object.__setattr__(self, "recommended_candidates", recommended)
        object.__setattr__(self, "valid_condition_rejected", rejected)
        object.__setattr__(self, "unresolved_candidates", unresolved)
        object.__setattr__(self, "failure_summaries", freeze_json(self.failure_summaries))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "input_hash": self.input_hash,
            "config_hash": self.config_hash,
            "backend_version": self.backend_version,
            "top_k": self.top_k,
            "valid_candidates": [item.to_mapping() for item in self.valid_candidates],
            "recommended_candidates": [item.to_mapping() for item in self.recommended_candidates],
            "valid_condition_rejected": [item.to_mapping() for item in self.valid_condition_rejected],
            "unresolved_candidates": [item.to_mapping() for item in self.unresolved_candidates],
            "failure_summaries": thaw_json(self.failure_summaries),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DiscoveryResultV1":
        expected = {"schema_version", "input_hash", "config_hash", "backend_version", "top_k", "valid_candidates", "recommended_candidates", "valid_condition_rejected", "unresolved_candidates", "failure_summaries"}
        _exact_fields(value, expected, "discovery result")
        if value["schema_version"] != cls.schema_version:
            raise DiscoveryContractError("unsupported result schema")
        convert = lambda rows: tuple(DiscoveryCandidateV1.from_mapping(item) for item in rows)
        return cls(value["input_hash"], value["config_hash"], value["backend_version"], value["top_k"], convert(value["valid_candidates"]), convert(value["recommended_candidates"]), convert(value["valid_condition_rejected"]), convert(value["unresolved_candidates"]), value["failure_summaries"])


__all__ = [
    "BranchCoverageV1",
    "CandidateVerificationV1",
    "CoefficientDependenceResultV1",
    "CONDITION_ELIGIBLE_V1",
    "CONDITION_REJECTED_V1",
    "ConditionCompatibilityV1",
    "ConditionRecordV1",
    "ConditionStatusV1",
    "DependenceStatusV1",
    "DiscoveryBudgetV1",
    "DiscoveryCandidateV1",
    "DiscoveryContractError",
    "DiscoveryProblemInputV1",
    "DiscoveryResultV1",
    "ParameterBranchV1",
    "ProofStatusV1",
    "ReductionCertificateV1",
    "ReductionChartV1",
    "VerificationStatusV1",
    "WorkerStatusV1",
    "aggregate_condition_status",
]
