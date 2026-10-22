"""BFGS fitting for coefficients in a symbolic expression.

The fitter is intentionally independent of a particular PDE.  Callers provide
one or more residual expressions and the collocation points at which each
residual is minimized.  This keeps the numerical optimizer reusable for PDE,
initial-condition, boundary-condition, and ordinary observation losses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import sympy as sp

FloatArray = Any
_NONFINITE_PENALTY = 1.0e100


@dataclass(frozen=True)
class BFGSConfigV1:
    """Deterministic multi-start BFGS budget and acceptance rule."""

    max_iterations: int = 500
    gradient_tolerance: float = 1.0e-8
    restarts: int = 8
    random_seed: int = 0
    restart_scale: float = 2.0
    acceptance_loss: float = 1.0e-10

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.gradient_tolerance <= 0 or not math.isfinite(self.gradient_tolerance):
            raise ValueError("gradient_tolerance must be finite and positive")
        if self.restarts <= 0:
            raise ValueError("restarts must be positive")
        if self.restart_scale <= 0 or not math.isfinite(self.restart_scale):
            raise ValueError("restart_scale must be finite and positive")
        if self.acceptance_loss < 0 or not math.isfinite(self.acceptance_loss):
            raise ValueError("acceptance_loss must be finite and nonnegative")


@dataclass(frozen=True)
class CollocationResidualV1:
    """One weighted residual evaluated on an explicit point set."""

    name: str
    residual: sp.Expr
    variables: tuple[sp.Symbol, ...]
    points: tuple[tuple[float, ...], ...]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("residual name must be nonempty")
        if len(set(self.variables)) != len(self.variables):
            raise ValueError(f"duplicate variables in residual {self.name}")
        if not self.points:
            raise ValueError(f"residual {self.name} needs at least one collocation point")
        if any(len(point) != len(self.variables) for point in self.points):
            raise ValueError(f"point width differs from variables in residual {self.name}")
        if any(not math.isfinite(value) for point in self.points for value in point):
            raise ValueError(f"residual {self.name} has a nonfinite collocation point")
        if self.weight <= 0 or not math.isfinite(self.weight):
            raise ValueError(f"residual {self.name} weight must be finite and positive")


@dataclass(frozen=True)
class BFGSCoefficientFitV1:
    """Best fit selected from every deterministic BFGS restart."""

    accepted: bool
    optimizer_converged: bool
    message: str
    expression: sp.Expr
    coefficient_assignments: tuple[tuple[str, float], ...]
    final_loss: float
    gradient_norm: float
    term_losses: tuple[tuple[str, float], ...]
    iterations: int
    function_evaluations: int
    gradient_evaluations: int
    selected_restart: int
    restart_count: int

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": "BFGSCoefficientFitV1",
            "accepted": self.accepted,
            "optimizer_converged": self.optimizer_converged,
            "message": self.message,
            "expression": sp.sstr(self.expression),
            "coefficient_assignments": dict(self.coefficient_assignments),
            "final_loss": self.final_loss,
            "gradient_norm": self.gradient_norm,
            "term_losses": dict(self.term_losses),
            "iterations": self.iterations,
            "function_evaluations": self.function_evaluations,
            "gradient_evaluations": self.gradient_evaluations,
            "selected_restart": self.selected_restart,
            "restart_count": self.restart_count,
        }


@dataclass(frozen=True)
class _CompiledResidual:
    name: str
    weight: float
    point_columns: tuple[FloatArray, ...]
    point_count: int
    residual_function: Any
    derivative_functions: tuple[Any, ...]

    def values_and_jacobian(self, parameters: FloatArray) -> tuple[FloatArray, FloatArray]:
        import numpy as np

        arguments = (*self.point_columns, *parameters)
        values = _as_point_vector(self.residual_function(*arguments), self.point_count)
        columns = tuple(
            _as_point_vector(function(*arguments), self.point_count)
            for function in self.derivative_functions
        )
        jacobian = np.column_stack(columns)
        return values, jacobian


def _as_point_vector(value: Any, point_count: int) -> FloatArray:
    import numpy as np

    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return np.full(point_count, float(array), dtype=np.float64)
    try:
        return np.asarray(np.broadcast_to(array, (point_count,)), dtype=np.float64)
    except ValueError as exc:
        raise ValueError("lambdified residual did not return one value per point") from exc


def _compile_residual(
    term: CollocationResidualV1,
    coefficients: tuple[sp.Symbol, ...],
) -> _CompiledResidual:
    import numpy as np

    allowed = set(term.variables).union(coefficients)
    unknown = term.residual.free_symbols.difference(allowed)
    if unknown:
        names = ", ".join(sorted(str(symbol) for symbol in unknown))
        raise ValueError(f"residual {term.name} has unresolved symbols: {names}")
    arguments = (*term.variables, *coefficients)
    points = np.asarray(term.points, dtype=np.float64)
    point_columns = tuple(points[:, index] for index in range(len(term.variables)))

    def zero_dirac_delta(value: Any, *_: Any) -> Any:
        """A collocation grid samples the distributional derivative as zero a.e."""

        return np.zeros_like(value, dtype=np.float64)

    modules: list[Any] = [{"DiracDelta": zero_dirac_delta}, "numpy"]
    return _CompiledResidual(
        name=term.name,
        weight=float(term.weight),
        point_columns=point_columns,
        point_count=len(term.points),
        residual_function=sp.lambdify(arguments, term.residual, modules=modules),
        derivative_functions=tuple(
            sp.lambdify(arguments, sp.diff(term.residual, coefficient), modules=modules)
            for coefficient in coefficients
        ),
    )


def _loss_and_gradient(
    parameters: FloatArray,
    residuals: tuple[_CompiledResidual, ...],
) -> tuple[float, FloatArray, tuple[tuple[str, float], ...]]:
    import numpy as np

    loss = 0.0
    gradient = np.zeros_like(parameters)
    term_losses: list[tuple[str, float]] = []
    try:
        with np.errstate(all="ignore"):
            for residual in residuals:
                values, jacobian = residual.values_and_jacobian(parameters)
                if not np.all(np.isfinite(values)) or not np.all(np.isfinite(jacobian)):
                    raise FloatingPointError
                item_loss = residual.weight * float(np.mean(np.square(values)))
                item_gradient = (
                    residual.weight
                    * 2.0
                    * np.mean(
                        values[:, np.newaxis] * jacobian,
                        axis=0,
                    )
                )
                if not np.isfinite(item_loss) or not np.all(np.isfinite(item_gradient)):
                    raise FloatingPointError
                loss += item_loss
                gradient += item_gradient
                term_losses.append((residual.name, item_loss))
    except (FloatingPointError, OverflowError, ValueError):
        penalty = _NONFINITE_PENALTY + float(np.dot(parameters, parameters))
        return (
            penalty,
            2.0 * parameters,
            tuple((residual.name, _NONFINITE_PENALTY) for residual in residuals),
        )
    return loss, gradient, tuple(term_losses)


def _initial_vectors(
    coefficient_count: int,
    config: BFGSConfigV1,
    initial_guesses: Sequence[Sequence[float]] | None,
) -> tuple[FloatArray, ...]:
    import numpy as np

    supplied: list[FloatArray] = []
    for guess in initial_guesses or ():
        vector = np.asarray(tuple(guess), dtype=np.float64)
        if vector.shape != (coefficient_count,) or not np.all(np.isfinite(vector)):
            raise ValueError("each initial guess must be a finite coefficient-width vector")
        supplied.append(vector)
    if not supplied:
        supplied.append(np.ones(coefficient_count, dtype=np.float64))
    supplied = supplied[: config.restarts]
    generator = np.random.default_rng(config.random_seed)
    base = supplied[0]
    while len(supplied) < config.restarts:
        supplied.append(base + generator.normal(0.0, config.restart_scale, coefficient_count))
    return tuple(supplied)


def fit_expression_coefficients_bfgs(
    expression: sp.Expr,
    coefficients: Sequence[sp.Symbol],
    residual_terms: Sequence[CollocationResidualV1],
    *,
    config: BFGSConfigV1 | None = None,
    initial_guesses: Sequence[Sequence[float]] | None = None,
) -> BFGSCoefficientFitV1:
    """Fit all free coefficients by minimizing weighted collocation MSE with BFGS."""

    import numpy as np
    from scipy.optimize import minimize

    selected_config = config or BFGSConfigV1()
    coefficient_tuple = tuple(coefficients)
    term_tuple = tuple(residual_terms)
    if not coefficient_tuple:
        raise ValueError("at least one coefficient is required")
    if any(not isinstance(item, sp.Symbol) for item in coefficient_tuple):
        raise TypeError("coefficients must be SymPy symbols")
    if len(set(coefficient_tuple)) != len(coefficient_tuple):
        raise ValueError("coefficients must be unique")
    if not term_tuple:
        raise ValueError("at least one residual term is required")
    compiled = tuple(_compile_residual(term, coefficient_tuple) for term in term_tuple)

    def objective(vector: FloatArray) -> float:
        return _loss_and_gradient(np.asarray(vector, dtype=np.float64), compiled)[0]

    def jacobian(vector: FloatArray) -> FloatArray:
        return _loss_and_gradient(np.asarray(vector, dtype=np.float64), compiled)[1]

    outcomes: list[tuple[int, Any]] = []
    starts = _initial_vectors(len(coefficient_tuple), selected_config, initial_guesses)
    for restart, initial in enumerate(starts):
        outcome = minimize(
            objective,
            initial,
            method="BFGS",
            jac=jacobian,
            options={
                "maxiter": selected_config.max_iterations,
                "gtol": selected_config.gradient_tolerance,
                "disp": False,
            },
        )
        outcomes.append((restart, outcome))
    selected_restart, best = min(
        outcomes,
        key=lambda item: (
            float(item[1].fun) if np.isfinite(item[1].fun) else _NONFINITE_PENALTY,
            item[0],
        ),
    )
    best_vector = np.asarray(best.x, dtype=np.float64)
    final_loss, final_gradient, term_losses = _loss_and_gradient(best_vector, compiled)
    assignments = dict(zip(coefficient_tuple, best_vector, strict=True))
    fitted_expression = expression.xreplace(
        {coefficient: sp.Float(float(value)) for coefficient, value in assignments.items()}
    )
    return BFGSCoefficientFitV1(
        accepted=bool(np.isfinite(final_loss) and final_loss <= selected_config.acceptance_loss),
        optimizer_converged=bool(best.success),
        message=str(best.message),
        expression=fitted_expression,
        coefficient_assignments=tuple(
            (str(coefficient), float(value))
            for coefficient, value in zip(coefficient_tuple, best_vector, strict=True)
        ),
        final_loss=float(final_loss),
        gradient_norm=float(np.linalg.norm(final_gradient)),
        term_losses=term_losses,
        iterations=int(best.nit),
        function_evaluations=int(best.nfev),
        gradient_evaluations=int(best.njev),
        selected_restart=selected_restart,
        restart_count=len(starts),
    )


__all__ = [
    "BFGSCoefficientFitV1",
    "BFGSConfigV1",
    "CollocationResidualV1",
    "fit_expression_coefficients_bfgs",
]
