"""Public-input normalization and dimension-agnostic structural proposals."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from fractions import Fraction
from typing import Any

from pde_ansatz_discovery import DiscoveryProblemInputV1

_DECIMAL = re.compile(r"(?<![A-Za-z0-9_.])(\d+\.\d+)(?![A-Za-z0-9_.])")
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_TIME_AXIS_NAMES = frozenset({"t", "time"})
_DERIVATIVE_CALL = re.compile(r"\bd_(?P<axis>[A-Za-z][A-Za-z0-9_]*)\s*\(")
_FINITE_SUM = re.compile(
    r"\bsum\s*\(\s*(?P<index>[A-Za-z][A-Za-z0-9_]*)\s*=\s*"
    r"(?P<start>[0-9]+)\s*\.\.\s*(?P<stop>[0-9]+)\s*,"
)
_METADATA = re.compile(
    r"^(?P<names>.+?)\s+(?:is|are)\s+(?:specified\s+only|prescribed|unavailable)",
    re.IGNORECASE,
)
_VECTOR = re.compile(r"^\s*\((?P<body>.*)\)\s*\*\*\s*T\s*$")


def rationalize_decimal_literals(expression: str) -> str:
    """Convert finite decimal literals to exact rational syntax accepted by the parser."""

    def replace(match: re.Match[str]) -> str:
        fraction = Fraction(Decimal(match.group(1)))
        return f"({fraction.numerator}/{fraction.denominator})"

    return _DECIMAL.sub(replace, expression).replace("^", "**")


def _matching_parenthesis(source: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(source)):
        character = source[index]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("unbalanced public PDE function call")


def _rewrite_calls(
    source: str,
    pattern: re.Pattern[str],
    replacer: Callable[[re.Match[str], str], str],
) -> str:
    """Rewrite balanced function calls matched by a prefix ending in ``(``."""

    result = source
    cursor = 0
    while match := pattern.search(result, cursor):
        opening = result.find("(", match.start(), match.end())
        closing = _matching_parenthesis(result, opening)
        body_start = match.end()
        replacement = replacer(match, result[body_start:closing])
        result = result[: match.start()] + replacement + result[closing + 1 :]
        cursor = match.start() + len(replacement)
    return result


def _split_top_level(source: str) -> tuple[str, ...]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, character in enumerate(source):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced public PDE vector definition")
        elif character == "," and depth == 0:
            parts.append(source[start:index].strip())
            start = index + 1
    if depth != 0:
        raise ValueError("unbalanced public PDE vector definition")
    parts.append(source[start:].strip())
    if any(not part for part in parts):
        raise ValueError("empty public PDE vector component")
    return tuple(parts)


def _metadata_function_names(sources: Sequence[str]) -> set[str]:
    names: set[str] = set()
    for source in sources:
        if "=" in source:
            continue
        match = _METADATA.match(source.strip())
        if match is None:
            continue
        names.update(
            name
            for name in _IDENTIFIER.findall(match.group("names"))
            if name.lower() not in {"and"}
        )
    return names


def _inline_scalar_definitions(source: str, definitions: Mapping[str, str]) -> str:
    result = source
    for _ in range(len(definitions) + 1):
        previous = result
        for name, value in definitions.items():
            result = re.sub(rf"\b{re.escape(name)}\b", f"({value})", result)
        if result == previous:
            return result
    raise ValueError("cyclic public PDE scalar definitions")


def _expand_vector_system(
    equations: Sequence[str], opaque_names: set[str], dependent_names: set[str]
) -> tuple[tuple[str, ...], set[str]]:
    """Expand a displayed divergence of named vector fluxes into scalar equations."""

    vector_definitions: dict[str, tuple[str, ...]] = {}
    scalar_definitions: dict[str, str] = {}
    ordinary: list[str] = []
    for source in equations:
        if "=" not in source:
            ordinary.append(source)
            continue
        lhs, rhs = (part.strip() for part in source.split("=", maxsplit=1))
        if not _IDENTIFIER.fullmatch(lhs) or lhs in dependent_names:
            ordinary.append(source)
            continue
        vector = _VECTOR.match(rationalize_decimal_literals(rhs))
        if vector is None:
            scalar_definitions[lhs] = rhs
        else:
            vector_definitions[lhs] = _split_top_level(vector.group("body"))

    referenced = {
        name
        for name in vector_definitions
        if any(re.search(rf"\b{re.escape(name)}\s*\(", equation) for equation in ordinary)
    }
    if not referenced:
        return tuple(ordinary), opaque_names
    widths = {len(vector_definitions[name]) for name in referenced}
    if len(widths) != 1:
        raise ValueError("public PDE vector flux widths differ")
    width = widths.pop()
    expanded: list[str] = []
    generated_opaque = set(opaque_names)
    for equation in ordinary:
        if not any(re.search(rf"\b{re.escape(name)}\s*\(", equation) for name in referenced):
            expanded.append(equation)
            continue
        for component_index in range(width):
            component_equation = equation
            for name in referenced:
                component = _inline_scalar_definitions(
                    vector_definitions[name][component_index], scalar_definitions
                )
                call = re.compile(
                    rf"\bd_(?P<axis>[A-Za-z][A-Za-z0-9_]*)\s*\(\s*"
                    rf"{re.escape(name)}\s*\([^()]*\)\s*\)"
                )

                def replace_flux_call(
                    match: re.Match[str], replacement_component: str = component
                ) -> str:
                    return f"d_{match.group('axis')}({replacement_component})"

                component_equation = call.sub(
                    replace_flux_call,
                    component_equation,
                )
            for name in opaque_names:
                component_name = f"{name}_{component_index + 1}"
                component_equation = re.sub(
                    rf"\b{re.escape(name)}\b", component_name, component_equation
                )
                generated_opaque.add(component_name)
                generated_opaque.discard(name)
            expanded.append(component_equation)
    return tuple(expanded), generated_opaque


def _expand_finite_sums(source: str, row: Mapping[str, Any]) -> str:
    parameters = row.get("parameters", {})
    centers = parameters.get("centers") if isinstance(parameters, Mapping) else None
    spatial_axes = tuple(
        str(axis)
        for axis in row["independent_variables"]
        if str(axis) not in _TIME_AXIS_NAMES | {"..."}
    )

    def expand(match: re.Match[str], body: str) -> str:
        start = int(match.group("start"))
        stop = int(match.group("stop"))
        if stop < start or stop - start > 128:
            raise ValueError("invalid or oversized finite public PDE sum")
        index_name = match.group("index")
        terms: list[str] = []
        for public_index in range(start, stop + 1):
            term = body
            center_index = public_index - start

            def replace_center(
                center_match: re.Match[str], center_index: int = center_index
            ) -> str:
                axis = center_match.group("axis")
                if (
                    not isinstance(centers, Sequence)
                    or isinstance(centers, (str, bytes))
                    or center_index >= len(centers)
                    or axis not in spatial_axes
                ):
                    raise ValueError("finite public PDE sum lacks matching center data")
                center = centers[center_index]
                if not isinstance(center, Sequence) or isinstance(center, (str, bytes)):
                    raise ValueError("public PDE center must be a coordinate sequence")
                axis_index = spatial_axes.index(axis)
                if axis_index >= len(center):
                    raise ValueError("public PDE center dimension differs")
                return str(center[axis_index])

            term = re.sub(
                rf"\bmu_(?P<axis>[A-Za-z][A-Za-z0-9_]*)_{re.escape(index_name)}\b",
                replace_center,
                term,
            )
            term = re.sub(rf"\b{re.escape(index_name)}\b", str(public_index), term)
            terms.append(f"({term})")
        return "(" + "+".join(terms) + ")"

    return _rewrite_calls(source, _FINITE_SUM, expand)


def _expand_derivative_calls(source: str) -> str:
    result = source
    while _DERIVATIVE_CALL.search(result):
        result = _rewrite_calls(
            result,
            _DERIVATIVE_CALL,
            lambda match, body: f"D(({body}),{match.group('axis')})",
        )
    return result


def _declare_opaque_functions(source: str, names: set[str], coordinates: Sequence[str]) -> str:
    arguments = ",".join(map(str, coordinates))
    result = source
    for name in sorted(names, key=lambda item: (-len(item), item)):
        result = re.sub(
            rf"\b{re.escape(name)}\b(?!\s*\()",
            f"{name}({arguments})",
            result,
        )
    return result


def normalized_pde_equations(row: dict[str, Any]) -> tuple[str, ...]:
    """Normalize public PDE notation without changing the represented equation."""

    independent = tuple(row["independent_variables"])
    spatial = tuple(axis for axis in independent if axis not in _TIME_AXIS_NAMES | {"..."})
    dependent = tuple(row["dependent_variables"])
    laplacian = re.compile(
        r"\bDelta\s*\(\s*(?P<field>" + "|".join(map(re.escape, dependent)) + r")\s*\)"
    )
    raw_sources = tuple(map(str, row["pde"]))
    opaque_names = _metadata_function_names(raw_sources)
    formal_sources = tuple(
        source for source in raw_sources if "=" in source or _METADATA.match(source.strip()) is None
    )
    formal_sources, opaque_names = _expand_vector_system(
        formal_sources, opaque_names, set(map(str, dependent))
    )
    equations = []
    for source in formal_sources:
        equation = source
        if spatial and dependent:
            equation = laplacian.sub(
                lambda match: (
                    "(" + "+".join(f"{match.group('field')}_{axis}{axis}" for axis in spatial) + ")"
                ),
                equation,
            )
        equation = _expand_finite_sums(equation, row)
        equation = _expand_derivative_calls(equation)
        equation = _declare_opaque_functions(equation, opaque_names, independent)
        equations.append(rationalize_decimal_literals(equation))
    if not equations:
        raise ValueError("public PDE contains no formal equations")
    return tuple(equations)


def normalized_public_problem(
    row: dict[str, Any], *, without_conditions: bool = False
) -> DiscoveryProblemInputV1:
    """Build a normalized copied-contract problem for parsing and exact solvers."""

    return DiscoveryProblemInputV1(
        tuple(row["dependent_variables"]),
        tuple(row["independent_variables"]),
        normalized_pde_equations(row),
        {} if without_conditions else row["domain"],
        row["parameters"],
        () if without_conditions else tuple(row["initial_conditions"]),
        () if without_conditions else tuple(row["boundary_conditions"]),
        tuple(row["additional_constraints"]),
    )


def normalized_structure_problem(row: dict[str, Any]) -> DiscoveryProblemInputV1:
    """Build the primary-equation projection used by the copied two-axis generator."""

    problem = normalized_public_problem(row, without_conditions=True)
    return DiscoveryProblemInputV1(
        problem.dependent_variables,
        problem.independent_variables,
        (problem.pde[0],),
        {},
        problem.parameters,
        (),
        (),
        problem.additional_constraints,
    )


def referenced_auxiliary_definitions(row: dict[str, Any]) -> tuple[str, ...]:
    """Find named fields defined by later equations and used in the primary PDE."""

    if len(row["pde"]) < 2:
        return ()
    primary = row["pde"][0]
    referenced = []
    for equation in row["pde"][1:]:
        left = equation.split("=", maxsplit=1)[0].strip()
        if _IDENTIFIER.fullmatch(left) and re.search(rf"\b{re.escape(left)}\b", primary):
            referenced.append(left)
    return tuple(referenced)


def _proposal(kind: str, tag: str, evidence: str) -> dict[str, str]:
    return {"root_kind": kind, "root_tag": tag, "evidence": evidence}


def _deduplicate(proposals: list[dict[str, str]]) -> list[dict[str, str]]:
    found: dict[tuple[str, str], dict[str, str]] = {}
    for proposal in proposals:
        found.setdefault((proposal["root_kind"], proposal["root_tag"]), proposal)
    return list(found.values())


def infer_public_structure(row: dict[str, Any]) -> list[dict[str, str]]:
    """Infer coarse Ansatz roots from public IC/BC/PDE syntax, never from gold labels."""

    if len(row["dependent_variables"]) != 1:
        return []
    axes = tuple(axis for axis in row["independent_variables"] if axis != "...")
    initials = " ".join(row["initial_conditions"])
    boundaries = " ".join(row["boundary_conditions"])
    pde = " ".join(row["pde"])
    channel = f"{initials} {boundaries} {pde}"
    lower = channel.lower()
    proposals: list[dict[str, str]] = []

    if "..." in row["independent_variables"]:
        if "sin(" in lower and "sum(i=1..d" in lower:
            proposals.append(
                _proposal("compose", "function_composition", "generic sine of axis sum")
            )
        elif "exp(" in lower and "sum(i=1..d" in lower:
            proposals.append(
                _proposal("compose", "function_composition", "generic exponential composition")
            )
        else:
            proposals.append(_proposal("sum", "generic_sum", "generic repeated axes"))

    if "max(" in lower:
        proposals.append(_proposal("envelope", "max", "public max envelope"))
    if "u_t" in pde and "sqrt(u_x" in pde and "sqrt(" in initials:
        proposals.append(
            _proposal(
                "envelope",
                "max",
                "outward Eikonal front with nonnegative radial initial profile",
            )
        )
    if "min(" in lower:
        proposals.append(_proposal("envelope", "min", "public min envelope"))
    if "abs(" in lower and "max(" not in lower and "min(" not in lower:
        proposals.append(_proposal("sum", "additive", "public absolute-value sum"))

    if (
        "u = 0 on the spatial boundary" in boundaries
        and "exp(" in pde
        and ("delta(u)" in pde.lower() or "u_xx" in pde)
    ):
        proposals.append(
            _proposal("boundary_mask", "dirichlet_zero", "homogeneous Dirichlet boundary")
        )

    rhs = " ".join(
        item.split("=", maxsplit=1)[1] for item in row["initial_conditions"] if "=" in item
    )
    trig_count = sum(rhs.count(name) for name in ("sin(", "cos("))
    if trig_count >= 2 and (" + " in rhs or " - " in rhs):
        proposals.append(
            _proposal("sum", "finite_superposition", "multiple public trigonometric modes")
        )
    if "*" in rhs and (
        ("sin(" in rhs and "cos(" in rhs)
        or ("exp(" in rhs and ("sin(" in rhs or "cos(" in rhs))
        or rhs.count("sin(") >= 2
    ):
        proposals.append(
            _proposal("product", "separated_product", "multiplicative axis factors in IC")
        )
    if rhs.count("exp(") >= 2 and "*" in rhs:
        proposals.extend(
            (
                _proposal("product", "separated_product", "product of axis exponentials"),
                _proposal("compose", "function_composition", "equivalent exponential of a sum"),
            )
        )
    if "u_tt" in pde and len(row["initial_conditions"]) >= 2:
        proposals.append(
            _proposal("product", "separated_product", "value and time-derivative IC channels")
        )
    if "tanh(" in lower or "sech(" in lower:
        proposals.append(
            _proposal("compose", "function_composition", "public traveling profile function")
        )
    if "exp(-(" in lower and len(axes) >= 2:
        proposals.append(
            _proposal("compose", "function_composition", "public Gaussian composition")
        )
    if trig_count == 1:
        trig_rhs = rhs.replace(" ", "")
        mentioned_axes = sum(axis in trig_rhs for axis in axes)
        if mentioned_axes >= 2 or "sum(i=1..d" in trig_rhs:
            proposals.append(
                _proposal("compose", "function_composition", "trigonometric affine coordinate")
            )

    if rhs and len(axes) >= 2:
        axis_terms = sum(axis in rhs for axis in axes)
        if axis_terms >= 2 and ("+" in rhs or "-" in rhs[1:]):
            proposals.append(_proposal("sum", "additive", "additive public initial data"))
    if not rhs and len(axes) >= 2 and ("+" in pde or "-" in pde):
        if "exp(" in pde or re.search(r"\b[A-Za-z][A-Za-z0-9_]*\^", pde):
            proposals.append(_proposal("sum", "additive", "axis-separated public forcing"))

    return _deduplicate(proposals)


def proposals_from_certified_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, str]]:
    proposals: list[dict[str, str]] = []
    for candidate in candidates:
        key = candidate["generator_key"]
        if key == "separation:product":
            proposals.append(
                _proposal("product", "separated_product", "certified product separation")
            )
        elif key == "separation:additive":
            proposals.append(_proposal("sum", "additive", "certified additive separation"))
        elif key.startswith(("direct:", "classical_lie:", "nonclassical:")):
            proposals.append(
                _proposal("compose", "function_composition", "certified invariant profile")
            )
    return _deduplicate(proposals)
