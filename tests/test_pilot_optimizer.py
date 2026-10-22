"""Shared optimizer contract tests; mocked fitter, no model or real provider run."""

import pytest

from tools import pilot_optimizer as shared


def public_row():
    return {
        "dependent_variables": ["q"],
        "independent_variables": ["r", "s"],
        "parameters": {},
        "domain": {"r": ["0", "1"], "s": ["0", "1"]},
        "pde": ["diff(q(r,s),s)=0"],
        "initial_conditions": ["q(r,0)=r"],
        "boundary_conditions": [],
        "benchmark_id": "synthetic",
    }


def test_same_policy_and_field_census():
    expressions, coefficients = shared.parse_fields(public_row(), {"q": "p1*r+p2"})
    assert len(coefficients) == 2 and set(expressions) == {"q"}
    assert shared.CONFIG["max_iterations"] == 300
    assert shared.CONFIG["restarts"] == 6
    assert shared.OPTIMIZER_POLICY["cpu_fallback"] is False
    with pytest.raises(ValueError, match="census"):
        shared.parse_fields(public_row(), {"u": "p1*r"})


def test_no_missing_coordinate_silently_fit_as_parameter():
    with pytest.raises(ValueError, match="undeclared coordinate"):
        shared.parse_fields(public_row(), {"q": "p1*x"})


def test_global_fit_budget_and_cache(monkeypatch):
    calls = []

    def fake(row, fields):
        calls.append(fields)
        return {"validation_loss": float(len(calls))}

    monkeypatch.setattr(shared, "fit_fields", fake)
    session = shared.FitSession(public_row())
    for i in range(4):
        session.fit({"q": f"p1*r+{i}"})
    assert session.fit({"q": "p1*r+0"})["fit_cache_hit"] is True
    assert session.fit({"q": "p1*r+5"})["budget_exhausted"] is True
    assert len(calls) == 4


def test_configuration_mismatch_rejected_before_cuda():
    with pytest.raises(ValueError, match="shared frozen policy"):
        shared.fit_fields(public_row(), {"q": "p1*r"}, config={})


def test_nonfinite_diagnostics_are_strict_json_null():
    import json

    record = shared.finite_metrics({"restarts": [float("inf"), float("nan"), 0.0]})
    assert record == {"restarts": [None, None, 0.0]}
    json.dumps(record, allow_nan=False)
