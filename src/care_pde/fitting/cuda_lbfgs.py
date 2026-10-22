"""CUDA-only FP64 collocation fitting with JAX/Optax L-BFGS.

Symbolic construction and reproducible restart seeds are host-side. Residuals,
gradients, line search, and all L-BFGS updates execute on an explicit CUDA device.
This is an optimizer, never a model candidate generator or an LLM fallback.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import sympy as sp

OPTIMIZER_ID = "jax_cuda_optax_lbfgs_v1"


class CudaInfrastructureError(RuntimeError):
    """CUDA execution is unavailable; do not fall back to CPU or another method."""


@dataclass(frozen=True)
class CudaLBFGSConfig:
    max_iterations: int = 300
    gradient_tolerance: float = 1.0e-8
    restarts: int = 6
    random_seed: int = 20260904
    restart_scale: float = 1.0
    acceptance_loss: float = 1.0e-8


def _runtime(device_index: int = 0):
    try:
        import jax
        import jax.numpy as jnp
        import optax

        jax.config.update("jax_enable_x64", True)
        devices = jax.devices("gpu")
        if (
            isinstance(device_index, bool)
            or not isinstance(device_index, int)
            or not 0 <= device_index < len(devices)
        ):
            raise CudaInfrastructureError("requested CUDA device index is unavailable")
        device = devices[device_index]
        if (
            device.platform != "gpu"
            or "cuda" not in device.client.platform_version.lower()
        ):
            raise CudaInfrastructureError("an NVIDIA CUDA backend is required")
        return jax, jnp, optax, device
    except CudaInfrastructureError:
        raise
    except Exception as exc:
        raise CudaInfrastructureError(
            f"CUDA initialization failed: {type(exc).__name__}: {exc}"
        ) from exc


def cuda_runtime_info(device_index: int = 0) -> dict[str, Any]:
    jax, jnp, optax, device = _runtime(device_index)
    try:
        probe = jax.device_put(jnp.asarray([1.0, 2.0], dtype=jnp.float64), device)
        result = jax.jit(lambda x: x * x)(probe)
        result.block_until_ready()
        if result.devices() != {device} or str(result.dtype) != "float64":
            raise CudaInfrastructureError("CUDA/FP64 placement assertion failed")
    except CudaInfrastructureError:
        raise
    except Exception as exc:
        raise CudaInfrastructureError(f"CUDA arithmetic failed: {exc}") from exc
    return {
        "optimizer_id": OPTIMIZER_ID,
        "backend": "cuda",
        "device_index": device_index,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "device_kind": device.device_kind,
        "device_platform": device.platform,
        "platform_version": device.client.platform_version,
        "dtype": "float64",
        "jax_version": jax.__version__,
        "optax_version": optax.__version__,
        "residuals_on_cuda": True,
        "gradients_on_cuda": True,
        "optimizer_updates_on_cuda": True,
        "cpu_fallback": False,
    }


@dataclass(frozen=True)
class CudaCoefficientFit:
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
    cuda: dict[str, Any]
    restart_diagnostics: tuple[dict[str, Any], ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "schema_version": "CudaLBFGSCoefficientFitV1",
            "expression": sp.sstr(self.expression),
            "coefficient_assignments": dict(self.coefficient_assignments),
            "term_losses": dict(self.term_losses),
            "optimizer_id": OPTIMIZER_ID,
            "selected_restart_function_evaluations": self.restart_diagnostics[
                self.selected_restart
            ]["value_gradient_evaluations"],
            "selected_restart_gradient_evaluations": self.restart_diagnostics[
                self.selected_restart
            ]["value_gradient_evaluations"],
            "post_fit_residual_evaluations": 1,
        }


def _compile(terms, coefficients, jax, jnp, device):
    if not terms:
        raise ValueError("at least one residual term is required")
    names = [term.name for term in terms]
    if len(set(names)) != len(names):
        raise ValueError("residual names must be unique")
    compiled = []
    for term in terms:
        if term.residual.free_symbols - set(term.variables) - set(coefficients):
            raise ValueError(f"residual {term.name} has undeclared symbols")
        if not term.points or any(len(p) != len(term.variables) for p in term.points):
            raise ValueError("invalid collocation points")
        if any(not math.isfinite(float(v)) for p in term.points for v in p):
            raise ValueError("nonfinite collocation point")
        if not math.isfinite(float(term.weight)) or term.weight <= 0:
            raise ValueError("residual weight must be finite and positive")
        points = jax.device_put(jnp.asarray(term.points, dtype=jnp.float64), device)
        columns = tuple(points[:, i] for i in range(len(term.variables)))
        if term.residual.has(sp.DiracDelta, sp.SingularityFunction):
            raise ValueError("distributional residuals are unsupported on CUDA")
        # SymPy's default erf binding is scalar math.erf, not a batched JAX primitive.
        from jax.scipy.special import erf

        modules = [{"erf": erf}, "jax"]
        function = sp.lambdify(
            (*term.variables, *coefficients),
            term.residual,
            modules=modules,
            cse=True,
            docstring_limit=0,
        )
        compiled.append(
            (term.name, float(term.weight), len(term.points), columns, function)
        )

    def term_losses(parameters):
        values = []
        for _, weight, count, columns, function in compiled:
            raw = jnp.asarray(function(*columns, *parameters), dtype=jnp.float64)
            vector = jnp.broadcast_to(raw, (count,))
            values.append(weight * jnp.mean(jnp.square(vector)))
        return jnp.stack(values)

    def raw_loss(parameters):
        return jnp.sum(term_losses(parameters))

    def loss(parameters):
        # Inactive invalid expressions must not poison the penalty gradient.
        return jax.lax.cond(
            jnp.isfinite(raw_loss(parameters)),
            raw_loss,
            lambda p: jnp.asarray(jnp.inf, dtype=jnp.float64),
            parameters,
        )

    return loss, term_losses, names


def evaluate_residual_terms_cuda(
    terms: Sequence[Any], *, device_index: int = 0
) -> tuple[float, dict[str, float]]:
    jax, jnp, _, device = _runtime(device_index)
    _, terms_fn, names = _compile(tuple(terms), (), jax, jnp, device)
    try:
        vector = jax.device_put(jnp.empty((0,), dtype=jnp.float64), device)
        result = jax.jit(terms_fn)(vector)
        result.block_until_ready()
        if result.devices() != {device}:
            raise CudaInfrastructureError("validation moved off CUDA")
        values = list(map(float, jax.device_get(result)))
    except (ValueError, TypeError, NotImplementedError):
        raise
    except CudaInfrastructureError:
        raise
    except Exception as exc:
        raise CudaInfrastructureError(
            f"CUDA validation execution failed: {exc}"
        ) from exc
    if not all(map(math.isfinite, values)):
        raise ValueError("nonfinite CUDA validation residual")
    return sum(values), dict(zip(names, values, strict=True))


def fit_expression_coefficients_cuda_lbfgs(
    expression: sp.Expr,
    coefficients: Sequence[sp.Symbol],
    residual_terms: Sequence[Any],
    *,
    config: Any = None,
    initial_guesses: Sequence[Sequence[float]] | None = None,
    device_index: int = 0,
) -> CudaCoefficientFit:
    """Independently optimize every deterministic restart, entirely on CUDA."""
    import numpy as np

    selected = config or CudaLBFGSConfig()
    if selected.max_iterations <= 0 or selected.restarts <= 0:
        raise ValueError("positive iteration/restart budgets required")
    for name in ("gradient_tolerance", "restart_scale"):
        if not math.isfinite(getattr(selected, name)) or getattr(selected, name) <= 0:
            raise ValueError(f"invalid {name}")
    if not math.isfinite(selected.acceptance_loss) or selected.acceptance_loss < 0:
        raise ValueError("invalid acceptance loss")
    coeffs = tuple(coefficients)
    if (
        not coeffs
        or any(not isinstance(c, sp.Symbol) for c in coeffs)
        or len(set(coeffs)) != len(coeffs)
    ):
        raise ValueError("unique, nonempty SymPy coefficients required")
    jax, jnp, optax, device = _runtime(device_index)
    loss, terms_fn, names = _compile(tuple(residual_terms), coeffs, jax, jnp, device)
    starts = []
    for guess in initial_guesses or ():
        vector = np.asarray(tuple(guess), dtype=np.float64)
        if vector.shape != (len(coeffs),) or not np.all(np.isfinite(vector)):
            raise ValueError("initial guess must be finite and coefficient-width")
        starts.append(vector)
    if not starts:
        starts = [np.ones(len(coeffs), dtype=np.float64)]
    starts = starts[: selected.restarts]
    generator = np.random.default_rng(selected.random_seed)
    while len(starts) < selected.restarts:
        starts.append(
            starts[0] + generator.normal(0.0, selected.restart_scale, len(coeffs))
        )
    solver = optax.lbfgs(
        memory_size=10,
        scale_init_precond=True,
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=20, initial_guess_strategy="one"
        ),
    )
    cached_value_grad = optax.value_and_grad_from_state(loss)

    def solve_one(initial):
        state = solver.init(initial)
        value, grad = jax.value_and_grad(loss)(initial)
        carry = (
            initial,
            state,
            value,
            grad,
            jnp.int32(0),
            jnp.int64(1),
            jnp.bool_(False),
        )

        def condition(c):
            _, _, v, g, iteration, _, failed = c
            return (
                (iteration < selected.max_iterations)
                & ~failed
                & jnp.isfinite(v)
                & jnp.all(jnp.isfinite(g))
                & (jnp.max(jnp.abs(g)) > selected.gradient_tolerance)
            )

        def step(c):
            params, state, value, grad, iteration, evaluations, _ = c
            updates, next_state = solver.update(
                grad, state, params, value=value, grad=grad, value_fn=loss
            )
            next_params = optax.apply_updates(params, updates)
            next_value, next_grad = cached_value_grad(next_params, state=next_state)
            decrease_error = next_state[-1].info.decrease_error
            safe = (
                jnp.isfinite(next_value)
                & jnp.all(jnp.isfinite(next_grad))
                & (next_value <= value)
                & jnp.isfinite(decrease_error)
                & (decrease_error <= 0)
            )
            # Failed zoom searches can return unsafe steps: preserve the last
            # safe iterate and terminate this restart with an explicit diagnostic.
            return (
                jnp.where(safe, next_params, params),
                next_state,
                jnp.where(safe, next_value, value),
                jnp.where(safe, next_grad, grad),
                iteration + 1,
                evaluations + next_state[-1].info.num_linesearch_steps,
                ~safe,
            )

        params, state, value, grad, iterations, evaluations, failed = (
            jax.lax.while_loop(condition, step, carry)
        )
        return (
            params,
            value,
            grad,
            iterations,
            evaluations,
            state[-1].info.decrease_error,
            state[-1].info.curvature_error,
            failed,
        )

    try:
        initial = jax.device_put(np.stack(starts), device)
        # vmap wraps COMPLETE optimizers: each restart has its own L-BFGS history,
        # line search state, convergence predicate and iteration count.
        output = jax.jit(jax.vmap(solve_one))(initial)
        jax.block_until_ready(output)
        if any(array.devices() != {device} for array in output):
            raise CudaInfrastructureError("optimizer outputs moved off CUDA")
        (
            parameters,
            losses,
            gradients,
            iterations,
            evaluations,
            decrease,
            curvature,
            failed,
        ) = map(np.asarray, jax.device_get(output))
    except (ValueError, TypeError, NotImplementedError):
        raise
    except CudaInfrastructureError:
        raise
    except Exception as exc:
        raise CudaInfrastructureError(f"CUDA L-BFGS execution failed: {exc}") from exc
    valid = np.isfinite(losses) & np.all(np.isfinite(gradients), axis=1)
    if not valid.any():
        raise ValueError("all CUDA restarts have nonfinite residual/gradient")
    best = min(
        range(len(starts)),
        key=lambda i: (float(losses[i]) if valid[i] else float("inf"), i),
    )
    best_vector = parameters[best]
    per_term = jax.jit(terms_fn)(jax.device_put(best_vector, device))
    per_term.block_until_ready()
    term_values = np.asarray(jax.device_get(per_term))
    final_loss = float(term_values.sum())
    converged = bool(np.max(np.abs(gradients[best])) <= selected.gradient_tolerance)
    diagnostics = tuple(
        {
            "restart": i,
            "finite": bool(valid[i]),
            "line_search_failed": bool(failed[i]),
            "loss": float(losses[i]) if np.isfinite(losses[i]) else None,
            "iterations": int(iterations[i]),
            "value_gradient_evaluations": int(evaluations[i]),
            "line_search_decrease_error": float(decrease[i])
            if np.isfinite(decrease[i])
            else None,
            "line_search_curvature_error": float(curvature[i])
            if np.isfinite(curvature[i])
            else None,
        }
        for i in range(len(starts))
    )
    assignments = tuple(
        (str(c), float(v)) for c, v in zip(coeffs, best_vector, strict=True)
    )
    fitted = expression.xreplace(
        {c: sp.Float(float(v)) for c, v in zip(coeffs, best_vector, strict=True)}
    )
    return CudaCoefficientFit(
        bool(math.isfinite(final_loss) and final_loss <= selected.acceptance_loss),
        converged,
        "gradient tolerance reached"
        if converged
        else "line search failed; retained last safe iterate"
        if failed[best]
        else "iteration budget exhausted or nonfinite step",
        fitted,
        assignments,
        final_loss,
        float(np.linalg.norm(gradients[best])),
        tuple(zip(names, map(float, term_values), strict=True)),
        int(iterations[best]),
        int(evaluations.sum()) + 1,
        int(evaluations.sum()),
        best,
        len(starts),
        cuda_runtime_info(device_index),
        diagnostics,
    )
