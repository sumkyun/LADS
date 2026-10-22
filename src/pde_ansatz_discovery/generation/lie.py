"""Bounded classical Lie seeds derived from PDE determining equations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..analysis import analyze_scaling, analyze_translations
from ..generation.direct import DirectAnsatzV1
from ..ir import ParserContextV1, SymbolRoleV1, mul, parse_expression, parse_problem, rational, subtract
from ..models import ProofStatusV1
from ..models import DiscoveryProblemInputV1
from ..solving.characteristics import scaling_ansatz, traveling_ansatz
from ..solving.determining_system import GeneratorVerificationV1, verify_classical_generator
from ..solving.prolongation import PointGeneratorV1, point_generator_basis_degree

if TYPE_CHECKING:
    from ..cas.sympy_backend import SymPyBackendV1


@dataclass(frozen=True)
class ClassicalLieSeedV1:
    ansatz: DirectAnsatzV1
    generator: PointGeneratorV1
    determining: GeneratorVerificationV1
    generator_label: str
    solver_parameters: tuple[str, ...]
    basis_degree: int | None


def projective_generators_equivalent(
    left: PointGeneratorV1,
    right: PointGeneratorV1,
    *,
    backend: "SymPyBackendV1",
) -> bool | None:
    left_components = (
        left.time_component, left.spatial_component, left.field_component
    )
    right_components = (
        right.time_component, right.spatial_component, right.field_component
    )
    reference = next(
        (
            index
            for index, (lhs, rhs) in enumerate(zip(left_components, right_components))
            if backend.prove_zero(lhs).status is not ProofStatusV1.PROVED_ZERO
            and backend.prove_zero(rhs).status is not ProofStatusV1.PROVED_ZERO
        ),
        None,
    )
    if reference is None:
        return None
    statuses = []
    for lhs, rhs in zip(left_components, right_components):
        cross = subtract(
            mul(lhs, right_components[reference]),
            mul(rhs, left_components[reference]),
        )
        statuses.append(backend.prove_zero(cross).status)
    if all(status is ProofStatusV1.PROVED_ZERO for status in statuses):
        return True
    if any(status is ProofStatusV1.PROVED_NONZERO for status in statuses):
        return False
    return None


def _backend_and_context(problem: DiscoveryProblemInputV1, extra: tuple[str, ...]):
    from ..cas.sympy_backend import SymPyBackendV1

    parameters = tuple(dict.fromkeys((*problem.parameters, *extra)))
    context = ParserContextV1(
        problem.dependent_variables, problem.independent_variables, parameters
    )
    roles = {name: SymbolRoleV1.INDEPENDENT for name in problem.independent_variables}
    roles.update({name: SymbolRoleV1.PARAMETER for name in parameters})
    return context, SymPyBackendV1(problem.independent_variables, roles)


def generate_classical_lie_seeds(
    problem: DiscoveryProblemInputV1,
) -> tuple[ClassicalLieSeedV1, ...]:
    if len(problem.dependent_variables) != 1 or len(problem.independent_variables) != 2:
        return ()
    parsed = parse_problem(problem)
    residual = parsed.equations[0].residual
    field = problem.dependent_variables[0]
    spatial, time = problem.independent_variables
    seeds: list[ClassicalLieSeedV1] = []

    translation = analyze_translations((residual,), problem.independent_variables)
    translation_ansatz = traveling_ansatz(
        translation, field=field, problem_parameters=tuple(problem.parameters)
    )
    if translation_ansatz is not None:
        context, backend = _backend_and_context(problem, ("c",))
        generator = PointGeneratorV1(
            rational(1),
            parse_expression("c", context),
            rational(0),
            field,
            spatial,
            time,
        )
        determining = verify_classical_generator(residual, generator, backend=backend)
        if determining.invariant is True or "c" in determining.reduced_determining_expression.free_names:
            seeds.append(
                ClassicalLieSeedV1(
                    replace(translation_ansatz, family="classical_lie:translation"),
                    generator,
                    determining,
                    f"d_{time}+c*d_{spatial}",
                    ("c",),
                    point_generator_basis_degree(generator),
                )
            )

    scaling = analyze_scaling(
        residual, independent_variables=problem.independent_variables, field=field
    )
    scaling_candidate = scaling_ansatz(
        scaling, problem_parameters=tuple(problem.parameters)
    )
    if scaling_candidate is not None and scaling.spatial_over_time is not None:
        context, backend = _backend_and_context(problem, ("beta",))
        field_weight = parse_expression("beta", context)
        generator = PointGeneratorV1(
            parse_expression(time, context),
            mul(rational(scaling.spatial_over_time), parse_expression(spatial, context)),
            mul(field_weight, parse_expression(field, context)),
            field,
            spatial,
            time,
        )
        determining = verify_classical_generator(residual, generator, backend=backend)
        if determining.invariant is True or "beta" in determining.reduced_determining_expression.free_names:
            parameterized_ansatz = replace(
                scaling_candidate,
                expression=(
                    f"{time}**beta*{scaling_candidate.profile_function}"
                    f"({scaling_candidate.coordinate_expression})"
                ),
                free_parameters=tuple(
                    dict.fromkeys((*scaling_candidate.free_parameters, "beta"))
                ),
                branch=None,
            )
            seeds.append(
                ClassicalLieSeedV1(
                    replace(parameterized_ansatz, family="classical_lie:scaling"),
                    generator,
                    determining,
                    f"{time}*d_{time}+{scaling.spatial_over_time}*{spatial}*d_{spatial}+field_weight*{field}*d_{field}",
                    ("beta",),
                    point_generator_basis_degree(generator),
                )
            )
    return tuple(seeds)


__all__ = [
    "ClassicalLieSeedV1",
    "generate_classical_lie_seeds",
    "projective_generators_equivalent",
]
