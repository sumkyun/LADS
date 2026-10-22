"""Bounded finite-time singular-scaling grammar."""

from __future__ import annotations

from fractions import Fraction

from ..analysis.scaling import ScalingAnalysisV1
from .direct import DirectAnsatzV1


def singular_scaling_ansatz(
    analysis: ScalingAnalysisV1,
    *,
    problem_parameters: tuple[str, ...],
    profile: str = "F",
) -> DirectAnsatzV1 | None:
    if analysis.spatial_over_time != Fraction(1, 2):
        return None
    spatial, time = analysis.spatial_variable, analysis.time_variable
    horizon, center, exponent = "T", "x0", "beta"
    distance = f"{horizon}-{time}"
    coordinate = f"({spatial}-{center})/sqrt({distance})"
    parameters = tuple(
        dict.fromkeys((*problem_parameters, horizon, center, exponent))
    )
    return DirectAnsatzV1(
        analysis.field,
        f"{profile}({coordinate})/({distance})**{exponent}",
        "eta",
        coordinate,
        profile,
        parameters,
        {
            spatial: f"{center}+eta*sqrt({distance})",
            time: time,
        },
        time,
        (f"({distance})>0",),
        family="singular:finite_time_scaling",
    )


__all__ = ["singular_scaling_ansatz"]
