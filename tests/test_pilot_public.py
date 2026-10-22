import pytest

from tools.pilot_public import math_expr, score_candidate


def row(fields=("u",), variables=("x", "t")):
    return {
        "dependent_variables": list(fields),
        "independent_variables": list(variables),
        "parameters": {},
        "domain": {v: ["0", "1"] for v in variables},
        "pde": [],
        "initial_conditions": [],
        "boundary_conditions": [],
        "benchmark_id": "synthetic",
    }


def test_scalar_pde_and_derivative_initial_condition():
    r = row()
    r["pde"] = ["Derivative(u(x,t),t)-Derivative(u(x,t),x)=0"]
    r["initial_conditions"] = ["u(x,0)=x", "Subs(Derivative(u(x,t),t),t,0)=1"]
    assert score_candidate(r, "x+t")["loss"] == 0
    assert score_candidate(r, "x")["loss"] > 0


def test_coupled_and_periodic_conditions():
    r = row(("u", "v"))
    r["pde"] = ["diff(u(x,t),t)-v(x,t)=0", "diff(v(x,t),t)=0"]
    r["initial_conditions"] = ["u(x,0)=0"]
    r["boundary_conditions"] = ["v(0,t)=v(1,t)"]
    assert score_candidate(r, {"u": "t", "v": "1"})["loss"] == 0


def test_radial_pi_domain():
    r = row(variables=("r", "theta", "t"))
    r["domain"]["theta"] = ["0", "2*pi"]
    r["pde"] = ["diff(u(r,theta,t),t)=0"]
    r["initial_conditions"] = ["u(r,theta,0)=r**2"]
    assert score_candidate(r, "r**2")["loss"] == 0


def test_parser_rejects_code():
    with pytest.raises(ValueError):
        math_expr("__import__(1)", {})


def test_equation_whitespace_regression():
    r = row()
    r["pde"] = [" Derivative(u(x,t),t) - Derivative(u(x,t),x) = 0 "]
    r["initial_conditions"] = [" u(x,0) = x "]
    assert score_candidate(r, "x+t")["loss"] == 0
