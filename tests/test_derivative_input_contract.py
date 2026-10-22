"""Regression tests for the allowlisted PDE derivative input contract."""

from __future__ import annotations

import pytest

from pde_ansatz_discovery.ir import (
    ParserContextV1,
    ParserLimitsV1,
    PDEParseError,
    parse_expression,
)


@pytest.fixture
def u_context() -> ParserContextV1:
    return ParserContextV1(("u",), ("x", "t"))


@pytest.mark.parametrize(
    "source",
    (
        "D(u(x, t), x, x)",
        "Derivative(u(x, t), x, x)",
        "diff(u(x, t), x, x)",
        "D(u(x, t), x, 2)",
        "Derivative(u(x, t), (x, 2))",
        "diff(u(x, t), (x, 2))",
        "D(D(u(x, t), x), x)",
    ),
)
def test_second_derivative_spellings_normalize_to_same_ir(
    source: str,
    u_context: ParserContextV1,
) -> None:
    expected = parse_expression("D(u(x,t),x,x)", u_context)

    assert parse_expression(source, u_context) == expected


@pytest.mark.parametrize(
    "source",
    (
        "Derivative(u(x,t), x, t)",
        "Derivative(u(x,t), (x,1), (t,1))",
        "diff(u(x,t), x, 1, t, 1)",
        "D(D(u(x,t), x), t)",
    ),
)
def test_mixed_partial_spellings_normalize_to_same_ir(
    source: str,
    u_context: ParserContextV1,
) -> None:
    expected = parse_expression("D(u(x,t),x,t)", u_context)

    assert parse_expression(source, u_context) == expected


def test_derivative_contract_supports_arbitrary_declared_variable_names() -> None:
    context = ParserContextV1(("q",), ("r", "s", "z"))

    assert parse_expression("Derivative(q(r,s,z), (r,2), z)", context) == parse_expression(
        "diff(diff(q(r,s,z),r,2),z)",
        context,
    )


def test_nested_derivative_nodes_are_flattened_to_the_same_ir() -> None:
    context = ParserContextV1(("q",), ("r", "s", "z"))

    assert parse_expression("D(D(sin(r),r),z)", context) == parse_expression(
        "Derivative(sin(r),r,z)",
        context,
    )


@pytest.mark.parametrize(
    "name",
    (
        "Abs",
        "atan",
        "cos",
        "cosh",
        "exp",
        "log",
        "sech",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    ),
)
def test_elementary_function_allowlist_matches_staged_symbolic_grammar(name: str) -> None:
    context = ParserContextV1(("q",), ("r", "s", "z"))

    parsed = parse_expression(f"{name}(r)", context)

    assert parsed.name == name


@pytest.mark.parametrize(
    "source",
    (
        "Derivative(u(x,t), (x,))",
        "Derivative(u(x,t), (x,2,3))",
        "Derivative(u(x,t), (x,t))",
        "Derivative(u(x,t), (1,2))",
        "Derivative(u(x,t), (x,(2,)))",
        "Derivative(u(x,t), ((x,2),))",
    ),
)
def test_malformed_derivative_tuples_are_rejected(
    source: str,
    u_context: ParserContextV1,
) -> None:
    with pytest.raises(PDEParseError, match="derivative variable specification"):
        parse_expression(source, u_context)


@pytest.mark.parametrize(
    "source",
    (
        "diff(u(x,t),x,True)",
        "diff(u(x,t),x,0)",
        "diff(u(x,t),x,-1)",
        "diff(u(x,t),x,1.5)",
        "diff(u(x,t),(x,True))",
        "diff(u(x,t),(x,0))",
        "diff(u(x,t),(x,-1))",
        "diff(u(x,t),(x,1.5))",
    ),
)
def test_nonpositive_boolean_and_noninteger_orders_are_rejected(
    source: str,
    u_context: ParserContextV1,
) -> None:
    with pytest.raises(PDEParseError, match="derivative"):
        parse_expression(source, u_context)


@pytest.mark.parametrize(
    "source",
    (
        "diff(u(x,t),y)",
        "diff(u(x,t),(y,2))",
        "diff(u(x,t),y,2)",
    ),
)
def test_undeclared_derivative_variables_are_rejected(
    source: str,
    u_context: ParserContextV1,
) -> None:
    with pytest.raises(PDEParseError, match="derivative variable specification"):
        parse_expression(source, u_context)


@pytest.mark.parametrize(
    "source",
    (
        "diff(u(x,t),x,9)",
        "diff(u(x,t),(x,9))",
        "diff(u(x,t),x,999999999999999999999999999999999999999999)",
        "diff(diff(u(x,t),x,5),t,4)",
        "diff(sin(diff(u(x,t),x,5)),t,4)",
    ),
)
def test_derivative_order_limit_applies_before_expansion_and_across_nesting(
    source: str,
    u_context: ParserContextV1,
) -> None:
    with pytest.raises(PDEParseError, match="derivative order exceeds limit"):
        parse_expression(source, u_context, ParserLimitsV1(max_derivative_order=8))


def test_expression_entrypoint_ignores_surrounding_whitespace(
    u_context: ParserContextV1,
) -> None:
    assert parse_expression("  \n Derivative(u(x,t), (x,2)) \t ", u_context) == parse_expression(
        "diff(u(x,t),x,2)",
        u_context,
    )
