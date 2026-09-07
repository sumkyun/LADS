"""Candidate-independent residual NMSE for public PDE/condition validation.

Each component denominator is derived only from the frozen public validation
specification: the zero-field PDE forcing energy or the public condition-target
energy.  A unit floor handles homogeneous PDEs and zero boundary conditions without
introducing candidate-specific normalization into cross-candidate ranking.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import sympy as sp

from .coefficient_fitting import CollocationResidualV1
from .condition_aware_solution import _evaluate_terms, _residual_terms

NORMALIZATION_ENERGY_FLOOR = 1.0


def public_normalization_energies(
    spec: Mapping[str, Any],
    *,
    split: str,
    floor: float = NORMALIZATION_ENERGY_FLOOR,
) -> dict[str, float]:
    """Return fixed per-component MSE denominators from the public zero-field problem."""

    if floor <= 0.0 or not math.isfinite(floor):
        raise ValueError("normalization energy floor must be finite and positive")
    zero = sp.Integer(0)
    baseline = _residual_terms(zero, (), spec, split=split)
    energies: dict[str, float] = {}
    for term in baseline:
        _weighted_total, items = _evaluate_terms((term,), zero)
        weighted_mse = float(dict(items)[term.name])
        raw_mse = weighted_mse / float(term.weight)
        energies[term.name] = max(raw_mse, floor)
    if len(energies) != len(baseline):
        raise ValueError("public residual component names must be unique")
    return energies


def normalize_residual_terms(
    terms: Sequence[CollocationResidualV1],
    energies: Mapping[str, float],
) -> tuple[CollocationResidualV1, ...]:
    """Scale optimizer weights so its objective is total weighted component NMSE."""

    normalized = []
    for term in terms:
        energy = float(energies[term.name])
        if energy <= 0.0 or not math.isfinite(energy):
            raise ValueError(f"invalid normalization energy for {term.name}")
        normalized.append(replace(term, weight=float(term.weight) / energy))
    return tuple(normalized)


def evaluate_residual_nmse(
    terms: Sequence[CollocationResidualV1],
    expression: Any,
    energies: Mapping[str, float],
) -> dict[str, Any]:
    """Evaluate raw MSE, NMSE, and weighted NMSE for every fixed residual component."""

    components: dict[str, dict[str, float]] = {}
    total_weighted_nmse = 0.0
    max_component_nmse = 0.0
    for term in terms:
        _total, rows = _evaluate_terms((term,), expression)
        weighted_mse = float(dict(rows)[term.name])
        raw_mse = weighted_mse / float(term.weight)
        energy = float(energies[term.name])
        nmse = raw_mse / energy
        weighted_nmse = float(term.weight) * nmse
        components[term.name] = {
            "mse": raw_mse,
            "normalization_energy": energy,
            "nmse": nmse,
            "weighted_nmse": weighted_nmse,
            "rmse": math.sqrt(max(raw_mse, 0.0)),
            "nrmse": math.sqrt(max(nmse, 0.0)),
        }
        total_weighted_nmse += weighted_nmse
        max_component_nmse = max(max_component_nmse, nmse)
    return {
        "components": components,
        "total_weighted_nmse": total_weighted_nmse,
        "max_component_nmse": max_component_nmse,
    }


__all__ = [
    "NORMALIZATION_ENERGY_FLOOR",
    "evaluate_residual_nmse",
    "normalize_residual_terms",
    "public_normalization_energies",
]
