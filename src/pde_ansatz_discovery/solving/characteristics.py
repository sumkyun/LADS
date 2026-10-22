"""Bounded explicit characteristic constructions for Phase-3 generators."""

from __future__ import annotations

from fractions import Fraction
from dataclasses import dataclass

from ..analysis.scaling import ScalingAnalysisV1
from ..analysis.translations import TranslationAnalysisV1
from ..generation.direct import DirectAnsatzV1, materialized_parameter_branch
from ..cas.backend import CASBackendV1
from ..ir import SymbolRoleV1, divide, symbol
from .prolongation import PointGeneratorV1


@dataclass(frozen=True)
class CharacteristicOutcomeV1:
    kind: str
    invariant: str | None
    reason_code: str


def solve_characteristic(
    generator: PointGeneratorV1, *, backend: CASBackendV1
) -> CharacteristicOutcomeV1:
    tau = backend.render(generator.time_component)
    xi = backend.render(generator.spatial_component)
    spatial, time = generator.spatial_variable, generator.time_variable
    if tau == "0" and xi == "0":
        return CharacteristicOutcomeV1("failed", None, "ZERO_GENERATOR")
    if tau == "0" and xi in {"1", "-1"}:
        return CharacteristicOutcomeV1("explicit", time, "X_INVARIANT")
    if tau == "1" and not ({spatial, time} & set(generator.spatial_component.free_names)):
        return CharacteristicOutcomeV1(
            "explicit", f"{spatial}-({xi})*{time}", "AFFINE_CHARACTERISTIC"
        )
    if tau == time and spatial in generator.spatial_component.free_names:
        ratio = backend.cancel(
            divide(
                generator.spatial_component,
                symbol(spatial, SymbolRoleV1.INDEPENDENT),
            )
        ).expression
        if not ({spatial, time} & set(ratio.free_names)):
            rendered = backend.render(ratio)
            invariant = (
                f"{spatial}/sqrt({time})"
                if rendered in {"1/2", "0.5"}
                else f"{spatial}/{time}**({rendered})"
            )
            return CharacteristicOutcomeV1(
                "explicit", invariant, "SCALING_CHARACTERISTIC"
            )
        return CharacteristicOutcomeV1(
            "implicit", None, "SCALING_CHARACTERISTIC_REQUIRES_WEIGHT_EXTRACTION"
        )
    return CharacteristicOutcomeV1(
        "implicit", None, "CHARACTERISTIC_EXPLICIT_FORM_UNAVAILABLE"
    )


def _fraction_power(variable: str, exponent: Fraction) -> str:
    if exponent == 1:
        return variable
    if exponent == Fraction(1, 2):
        return f"sqrt({variable})"
    if exponent.denominator == 1:
        return f"{variable}**{exponent.numerator}"
    return f"{variable}**({exponent.numerator}/{exponent.denominator})"


def _scaled_coordinate(spatial: str, time: str, alpha: Fraction) -> tuple[str, str]:
    if alpha > 0:
        scale = _fraction_power(time, alpha)
        return f"{spatial}/{scale}", f"eta*{scale}"
    if alpha < 0:
        scale = _fraction_power(time, -alpha)
        return f"{spatial}*{scale}", f"eta/{scale}"
    return spatial, "eta"


def _scaled_field(profile_call: str, time: str, gamma: Fraction | None, parameter: str) -> str:
    if gamma is None:
        return f"{time}**{parameter}*{profile_call}"
    if gamma == 0:
        return profile_call
    if gamma > 0:
        return f"{_fraction_power(time, gamma)}*{profile_call}"
    return f"{profile_call}/{_fraction_power(time, -gamma)}"


def traveling_ansatz(
    analysis: TranslationAnalysisV1,
    *,
    field: str,
    problem_parameters: tuple[str, ...],
    profile: str = "F",
) -> DirectAnsatzV1 | None:
    if not analysis.admits_traveling_generator:
        return None
    speed = "c"
    spatial, time = analysis.spatial_variable, analysis.time_variable
    coordinate = f"{spatial}-{speed}*{time}"
    return DirectAnsatzV1(
        field,
        f"{profile}({coordinate})",
        "eta",
        coordinate,
        profile,
        tuple(dict.fromkeys((speed, *problem_parameters))),
        {spatial: f"eta+{speed}*{time}", time: time},
        time,
        family="translation:pde_coordinate_absence",
    )


def scaling_ansatz(
    analysis: ScalingAnalysisV1,
    *,
    problem_parameters: tuple[str, ...],
    profile: str = "F",
) -> DirectAnsatzV1 | None:
    if not analysis.admits_similarity_generator or analysis.spatial_over_time is None:
        return None
    coordinate, inverse_spatial = _scaled_coordinate(
        analysis.spatial_variable, analysis.time_variable, analysis.spatial_over_time
    )
    amplitude_parameter = "beta"
    profile_call = f"{profile}({coordinate})"
    expression = _scaled_field(
        profile_call,
        analysis.time_variable,
        None if analysis.field_weight_free else analysis.field_over_time,
        amplitude_parameter,
    )
    parameters = list(problem_parameters)
    if analysis.field_weight_free and amplitude_parameter not in parameters:
        parameters.append(amplitude_parameter)
    conditions = (f"{analysis.time_variable}>0",)
    branch = None
    if not analysis.field_weight_free and analysis.field_over_time is not None:
        exponent = analysis.field_over_time
        rendered_exponent = (
            str(exponent.numerator)
            if exponent.denominator == 1
            else f"{exponent.numerator}/{exponent.denominator}"
        )
        branch = materialized_parameter_branch(
            {amplitude_parameter: rendered_exponent}, conditions
        )
    return DirectAnsatzV1(
        analysis.field,
        expression,
        "eta",
        coordinate,
        profile,
        tuple(parameters),
        {
            analysis.spatial_variable: inverse_spatial,
            analysis.time_variable: analysis.time_variable,
        },
        analysis.time_variable,
        conditions,
        family="scaling:pde_weight_balance",
        branch=branch,
    )


__all__ = [
    "CharacteristicOutcomeV1",
    "scaling_ansatz",
    "solve_characteristic",
    "traveling_ansatz",
]
