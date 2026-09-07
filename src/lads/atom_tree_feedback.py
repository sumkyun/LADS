"""LLM-generated parameter-hole trees and slot-bound Atom instantiation.

The first Atom substage proposes only typed expression trees.  The trees contain
numbered variable holes (``V1``...) and coefficient holes (``P1``...) but no PDE
coordinates, slot arguments, or fitted values.  A second LLM call sees the certified
Ansatz slot and binds every variable hole to one of that slot's formal ``ARG`` roles.
The host then instantiates the final symbolic Atom and reuses the strict Atom validator.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .llm_ansatz import public_problem_payload
from .llm_provider import LLMBatchCompletionV1, LLMBatchProviderV1

TREE_CANDIDATE_COUNT = 8
TARGET_EXPANDED_ATOM_COUNT = 1_024
EXPANDED_ATOM_VALIDATION_BATCH_SIZE = 1_280
MAX_EXPANSIONS_PER_SEED = 512
MAX_PROMPT_FEEDBACK_ITEMS = 32
MAX_PROMPT_STRUCTURAL_KEYS = 128
MAX_TREE_TEMPLATE_LENGTH = 16_384
MAX_TREE_NODES = 512
MAX_TREE_DEPTH = 48
MAX_TREE_SUM_ARITY = 16
TREE_COVERAGE_LANES = (
    "AFFINE_BASE",
    "AFFINE_UNARY",
    "OUTPUT_POWER",
    "INPUT_POWER",
    "PRODUCT_COMPOSITION",
    "SUM_COMPOSITION",
    "RATIONAL_COMPOSITION",
    "OPERATOR_FREE",
)
TREE_BINARY_OPERATORS = frozenset({"add", "sub", "mul", "div"})
TREE_VARIADIC_OPERATORS = frozenset({"sum"})
TREE_UNARY_OPERATORS = frozenset(
    {
        "pow2",
        "pow3",
        "sin",
        "cos",
        "tan",
        "sinh",
        "cosh",
        "tanh",
        "exp",
        "log",
        "sqrt",
        "abs",
        "erf",
        "sech",
    }
)
TREE_ELEMENTARY_UNARY_OPERATORS = (
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "exp",
    "log",
    "sqrt",
    "abs",
    "erf",
    "sech",
)
TREE_POWER_OPERATORS = ("pow2", "pow3")
TREE_OPERATORS = TREE_BINARY_OPERATORS | TREE_UNARY_OPERATORS | TREE_VARIADIC_OPERATORS
_TREE_ID = re.compile(r"T([1-9][0-9]*)")
_VARIABLE_HOLE = re.compile(r"V([1-9][0-9]*)")
_PARAMETER_HOLE = re.compile(r"P([1-9][0-9]*)")
_RATIONALE = re.compile(r"[A-Z][A-Z0-9_]*")


class AtomTreeError(ValueError):
    """A tree or binding response violated the frozen host contract."""


@dataclass(frozen=True)
class ValidatedAtomTreeV1:
    tree_id: str
    coverage_lane: str
    tree_template: str
    variable_holes: tuple[str, ...]
    parameter_holes: tuple[str, ...]
    rationale_codes: tuple[str, ...]
    reason: str
    structural_key: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "tree_id": self.tree_id,
            "coverage_lane": self.coverage_lane,
            "tree_template": self.tree_template,
            "variable_holes": list(self.variable_holes),
            "parameter_holes": list(self.parameter_holes),
            "rationale_codes": list(self.rationale_codes),
            "reason": self.reason,
            "structural_key": self.structural_key,
        }


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _bullets(values: Sequence[object]) -> str:
    return "\n".join(f"- {value}" for value in values) if values else "- (none)"


def tree_response_format() -> dict[str, Any]:
    """Strict response schema for the variable-free tree proposal substage."""

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "pde_atom_parameter_hole_trees_v2",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["trees"],
                "properties": {
                    "trees": {
                        "type": "array",
                        "minItems": TREE_CANDIDATE_COUNT,
                        "maxItems": TREE_CANDIDATE_COUNT,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "tree_id",
                                "coverage_lane",
                                "tree_template",
                                "rationale_codes",
                                "reason",
                            ],
                            "properties": {
                                "tree_id": {"type": "string"},
                                "coverage_lane": {
                                    "type": "string",
                                    "enum": list(TREE_COVERAGE_LANES),
                                },
                                "tree_template": {"type": "string"},
                                "rationale_codes": {
                                    "type": "array",
                                    "minItems": 1,
                                    "uniqueItems": True,
                                    "items": {
                                        "type": "string",
                                        "pattern": "^[A-Z][A-Z0-9_]*$",
                                    },
                                },
                                "reason": {"type": "string"},
                            },
                        },
                    }
                },
            },
        },
    }


def binding_response_format(
    trees: Sequence[ValidatedAtomTreeV1], argument_roles: Sequence[str]
) -> dict[str, Any]:
    """Strict response schema for binding tree holes to certified slot roles."""

    tree_ids = [tree.tree_id for tree in trees]
    roles = list(argument_roles)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "pde_atom_tree_slot_bindings_v1",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["bindings"],
                "properties": {
                    "bindings": {
                        "type": "array",
                        "minItems": len(tree_ids),
                        "maxItems": len(tree_ids),
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "tree_id",
                                "variable_bindings",
                                "atom_type",
                                "rationale_codes",
                                "reason",
                            ],
                            "properties": {
                                "tree_id": {"type": "string", "enum": tree_ids},
                                "variable_bindings": {
                                    "type": "array",
                                    "minItems": 1,
                                    "uniqueItems": True,
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": ["hole", "slot_argument"],
                                        "properties": {
                                            "hole": {"type": "string"},
                                            "slot_argument": {"type": "string", "enum": roles},
                                        },
                                    },
                                },
                                "atom_type": {
                                    "type": "string",
                                    "enum": ["PRIMITIVE", "ENVELOPE", "COORDINATE", "TERM"],
                                },
                                "rationale_codes": {
                                    "type": "array",
                                    "minItems": 1,
                                    "uniqueItems": True,
                                    "items": {
                                        "type": "string",
                                        "pattern": "^[A-Z][A-Z0-9_]*$",
                                    },
                                },
                                "reason": {"type": "string"},
                            },
                        },
                    }
                },
            },
        },
    }


def render_tree_messages(
    row: Mapping[str, Any],
    *,
    certified_ansatz: str,
    slot_id: str,
    profile_arity: int,
    validation_feedback: Sequence[str] = (),
    forbidden_structural_keys: Sequence[str] = (),
    exploration_pass: int = 0,
) -> tuple[dict[str, str], ...]:
    """Ask for function-shape trees without exposing coordinates or condition targets."""

    payload = public_problem_payload(row)
    feedback = tuple(validation_feedback)[-MAX_PROMPT_FEEDBACK_ITEMS:]
    forbidden = tuple(dict.fromkeys(str(item) for item in forbidden_structural_keys))[
        -MAX_PROMPT_STRUCTURAL_KEYS:
    ]
    repair = ""
    if feedback:
        repair = f"""

Previous tree/Atom outcomes from host validation:
{_bullets(feedback)}

Propose structurally new trees. Do not repeat a retained or failed tree merely by renumbering
its holes. Treat each recorded Atom and its component NMSE as the outcome of a previous
experiment. Mutate high-NMSE structures rather than restarting from the same default forms.
Residual NMSE values are diagnostics, not hidden target expressions.
"""
    exclusions = ""
    if forbidden:
        exclusions = f"""

Previously used structural keys that this batch must not reuse:
{_bullets(forbidden)}
"""
    system = """You generate bounded parameter-hole expression-tree seed programs for one open
Ansatz slot. Return seed programs, not executable Atoms and not complete field solutions. The
host expands every accepted seed over the complete frozen elementary-function grammar before
residual screening. Tree leaves are only
numbered variable holes V1,V2,... and fitted coefficient holes P1,P2,... . Do not use x, y, t,
ARG names, numeric fitted values, conditions, benchmark identities, or reference solutions.
Allowed binary nodes are add, sub, mul, div. The variadic sum node accepts two to sixteen explicit
terms. Allowed unary nodes are pow2, pow3, sin, cos, tan, sinh, cosh, tanh, exp, log, sqrt, abs,
erf, sech. Every tree must contain a variable hole and a
coefficient hole. Every role-dependent tree needs an outer multiplicative P amplitude, and every
nonlinear function argument that depends on a V hole needs a multiplicative P scale inside that
argument. Use consecutive hole numbering and keep the tree bounded and executable. A long
multi-branch or composite tree is valid and must not be shortened merely to look compact.
Return strict JSON only."""
    coverage_rows = (
        'T1 has coverage_lane="AFFINE_BASE": affine/polynomial base; no elementary unary',
        'T2 has coverage_lane="AFFINE_UNARY": unary with fitted input shift and output bias',
        'T3 has coverage_lane="OUTPUT_POWER": integer power applied to a unary-function output',
        'T4 has coverage_lane="INPUT_POWER": unary applied to a powered variable expression',
        'T5 has coverage_lane="PRODUCT_COMPOSITION": product of variable-dependent branches',
        'T6 has coverage_lane="SUM_COMPOSITION": explicit sum or add/sub of variable branches',
        'T7 has coverage_lane="RATIONAL_COMPOSITION": division with a variable branch',
        'T8 has coverage_lane="OPERATOR_FREE": operator-derived topology distinct from T1-T7',
    )
    user = f"""Generate exactly {TREE_CANDIDATE_COUNT} structurally distinct parameter-hole tree
seed programs. The host expands these seeds to {TARGET_EXPANDED_ATOM_COUNT} concrete Atom
structures; these eight seeds are not the final candidate count.

PDE operator:
{_bullets(payload["pde"])}

Host-certified atom-free Ansatz:
{certified_ansatz}

Open slot ID: {slot_id}
Slot profile arity: {profile_arity}
Exploration pass: {exploration_pass}

Required ID-bound coverage lanes:
{_bullets(coverage_rows)}

Serialized tree grammar examples are notation only:
- mul(P1,V1)
- add(mul(P1,tanh(sin(add(mul(P2,V1),P3)))),P4)
- mul(P1,pow2(cos(sin(mul(P2,V1)))))
- sum(mul(P1,sin(mul(P2,V1))),mul(P3,cos(mul(P4,V1))))

Symbol rules:
- add(a,b), sub(a,b), mul(a,b), and div(a,b) take exactly two arguments.
- sum(a,b) through sum(a1,...,a16) adds two to sixteen explicit branches.
- Never emit Sigma, summation indices, ellipses, or forms such as sum(i=1..N,...).
- pow2(a) and pow3(a) are the only power nodes; do not emit caret notation.
- V1,V2,... are variable holes and P1,P2,... are fitted scalar parameter holes.

Metadata rules:
- coverage_lane must be the exact quoted value assigned to that tree ID above.
- rationale_codes must contain only labels like OPERATOR_DERIVED or DISPERSION_BALANCE.
- Every rationale code matches ^[A-Z][A-Z0-9_]*$: uppercase letters, digits, underscores only.
- Do not put labels such as T1/AFFINE_BASE in rationale_codes; `/` is invalid there.

Cross-pass diversity rules:
- Inspect the previous structural keys and Atom/NMSE outcomes before choosing new operators.
- At an analogous tree position, replace a previously used unary operator with a different
  allowed unary operator; do not default repeatedly to sin or cos.
- Across this batch, maximize distinct primary unary operators when the lane permits it.
- Preserve operator-supported long rank sums and composite forms; complexity alone is not a
  reason to demote a seed, and every P hole remains a host-fitted coefficient.
- For OUTPUT_POWER and INPUT_POWER, vary both pow2 versus pow3 and the nested unary operator.
- For AFFINE_UNARY, preserve the fitted input shift and output bias while changing the unary.
- These are target-neutral mutations. Do not infer or name a condition target.

SIGS-scale expansion contract:
- T1 is the elementary-free baseline.
- Every seed T2 through T8 must contain at least one elementary unary call site selected from
  sin, cos, tan, sinh, cosh, tanh, exp, log, sqrt, abs, erf, sech.
- Nesting two unary sites or placing one unary site on each of two branches is encouraged.
- At every elementary site, the host enumerates identity, all 12 single unary operators, and all
  12x12 depth-two unary compositions. It also substitutes pow2/pow3 choices, rejects invalid lane
  variants, deduplicates topology, and retains {TARGET_EXPANDED_ATOM_COUNT} concrete structures.
- This expansion is target-neutral: no operator is preferred because of a dataset case.

Do not bind V holes to coordinates or ARG roles yet. P holes remain unfitted coefficients.
Use tree IDs T1 through T{TREE_CANDIDATE_COUNT}. The host infers the exact V and P holes from
each tree_template; do not declare separate hole lists. Explain the operator-derived reason with
uppercase rationale codes.{repair}
Every tree must satisfy the coverage lane bound to its tree ID. On later exploration passes,
change operators or nesting rather than repeating a previous structural key. The listed keys
are host-derived topology summaries and must not be reused.{exclusions}
"""
    return ({"role": "system", "content": system}, {"role": "user", "content": user})


def render_binding_messages(
    row: Mapping[str, Any],
    *,
    certified_ansatz: str,
    slot_id: str,
    current_ansatz: str,
    profile_call: str,
    profile_arguments: Sequence[str],
    trees: Sequence[ValidatedAtomTreeV1],
    validation_feedback: Sequence[str] = (),
) -> tuple[dict[str, str], ...]:
    """Ask the LLM to bind each validated tree to the certified slot's formal arguments."""

    payload = public_problem_payload(row)
    roles = tuple(f"ARG{index}" for index in range(1, len(profile_arguments) + 1))
    role_bindings = tuple(
        f"{role} -> {argument}" for role, argument in zip(roles, profile_arguments, strict=True)
    )
    tree_rows = tuple(
        f"{tree.tree_id}/{tree.coverage_lane}: {tree.tree_template}; "
        f"V holes={','.join(tree.variable_holes)}"
        for tree in trees
    )
    feedback = tuple(validation_feedback)[-MAX_PROMPT_FEEDBACK_ITEMS:]
    repair = ""
    if feedback:
        repair = f"""

Previous host outcomes:
{_bullets(feedback)}

Use these outcomes to choose better slot-role bindings. Do not infer or emit a condition target.
"""
    system = """You bind validated function-tree variable holes to the formal arguments of one
host-certified Ansatz slot. Do not change a tree, fit its P holes, change the Ansatz, or output a
complete solution. P1,P2,... are fitted later by the host. Return strict JSON only."""
    user = f"""Bind every supplied tree to this single slot.

PDE operator:
{_bullets(payload["pde"])}

Certified Ansatz: {certified_ansatz}
Current one-slot view: {current_ansatz}
Slot ID: {slot_id}
Profile call: {profile_call}

Formal slot arguments:
{_bullets(role_bindings)}

Validated trees:
{_bullets(tree_rows)}

For each tree, bind every declared V hole exactly once to one available ARG role. Multiple V
holes may bind to the same ARG when the operator supports it. The host will deterministically
replace P1,P2,... with p1,p2,... and instantiate the final Atom. Return one binding for every
tree, preserving tree IDs.{repair}
"""
    return ({"role": "system", "content": system}, {"role": "user", "content": user})


def _tree_details(template: str) -> tuple[ast.Expression, tuple[str, ...], tuple[str, ...], str]:
    if (
        not isinstance(template, str)
        or not template.strip()
        or len(template) > MAX_TREE_TEMPLATE_LENGTH
    ):
        raise AtomTreeError("tree_template is empty or too long")
    try:
        parsed = ast.parse(template, mode="eval")
    except SyntaxError as exc:
        raise AtomTreeError("tree_template is not valid prefix-call syntax") from exc
    variable_holes: set[str] = set()
    parameter_holes: set[str] = set()
    node_count = 0

    def visit(node: ast.AST, depth: int) -> str:
        nonlocal node_count
        node_count += 1
        if node_count > MAX_TREE_NODES or depth > MAX_TREE_DEPTH:
            raise AtomTreeError("tree exceeds the frozen node/depth budget")
        if isinstance(node, ast.Name):
            if _VARIABLE_HOLE.fullmatch(node.id):
                variable_holes.add(node.id)
                return "V"
            if _PARAMETER_HOLE.fullmatch(node.id):
                parameter_holes.add(node.id)
                return "P"
            raise AtomTreeError(f"unsupported tree leaf: {node.id}")
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise AtomTreeError("tree nodes must be allowed prefix calls or V/P holes")
        if node.keywords:
            raise AtomTreeError("tree calls cannot contain keywords")
        operator = node.func.id
        if operator not in TREE_OPERATORS:
            raise AtomTreeError(f"unsupported tree operator: {operator}")
        if operator in TREE_VARIADIC_OPERATORS:
            if not 2 <= len(node.args) <= MAX_TREE_SUM_ARITY:
                raise AtomTreeError(
                    f"tree operator {operator} requires arity 2 through {MAX_TREE_SUM_ARITY}"
                )
        else:
            expected_arity = 2 if operator in TREE_BINARY_OPERATORS else 1
            if len(node.args) != expected_arity:
                raise AtomTreeError(f"tree operator {operator} requires arity {expected_arity}")
        children = ",".join(visit(child, depth + 1) for child in node.args)
        return f"{operator}({children})"

    structural_key = visit(parsed.body, 1)
    if not variable_holes or not parameter_holes:
        raise AtomTreeError("every tree needs at least one V hole and one P hole")
    variables = tuple(sorted(variable_holes, key=lambda item: int(item[1:])))
    parameters = tuple(sorted(parameter_holes, key=lambda item: int(item[1:])))
    if variables != tuple(f"V{index}" for index in range(1, len(variables) + 1)):
        raise AtomTreeError("V holes must be consecutive from V1")
    if parameters != tuple(f"P{index}" for index in range(1, len(parameters) + 1)):
        raise AtomTreeError("P holes must be consecutive from P1")
    return parsed, variables, parameters, structural_key


def _depends_on_variable(node: ast.AST) -> bool:
    return any(
        isinstance(item, ast.Name) and _VARIABLE_HOLE.fullmatch(item.id) for item in ast.walk(node)
    )


def _contains_operator(node: ast.AST, operators: frozenset[str]) -> bool:
    return any(
        isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id in operators
        for item in ast.walk(node)
    )


def _elementary_operator_count(node: ast.AST) -> int:
    allowed = set(TREE_ELEMENTARY_UNARY_OPERATORS)
    return sum(
        isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id in allowed
        for item in ast.walk(node)
    )


def _has_binary_with_two_variable_branches(node: ast.AST, operator: str) -> bool:
    return any(
        isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == operator
        and len(item.args) == 2
        and _depends_on_variable(item.args[0])
        and _depends_on_variable(item.args[1])
        for item in ast.walk(node)
    )


def _has_sum_with_two_variable_branches(node: ast.AST) -> bool:
    return any(
        isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == "sum"
        and sum(_depends_on_variable(argument) for argument in item.args) >= 2
        for item in ast.walk(node)
    )


def _has_affine_unary(node: ast.AST) -> bool:
    unary_nodes = [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id in TREE_UNARY_OPERATORS
    ]
    input_shift = any(
        any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id in {"add", "sub"}
            and len(child.args) == 2
            and _depends_on_variable(child.args[0]) != _depends_on_variable(child.args[1])
            for child in ast.walk(unary.args[0])
        )
        for unary in unary_nodes
    )
    output_bias = (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"add", "sub"}
        and len(node.args) == 2
        and _depends_on_variable(node.args[0]) != _depends_on_variable(node.args[1])
    )
    return input_shift and output_bias


def _matches_coverage_lane(lane: str, node: ast.AST) -> bool:
    if lane == "AFFINE_BASE":
        elementary = TREE_UNARY_OPERATORS - {"pow2", "pow3"}
        return not _contains_operator(node, elementary) and _contains_operator(
            node, frozenset({"add", "sub"})
        )
    if lane == "AFFINE_UNARY":
        return _has_affine_unary(node)
    if lane == "OUTPUT_POWER":
        return any(
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Name)
            and item.func.id in {"pow2", "pow3"}
            and _contains_operator(item.args[0], TREE_UNARY_OPERATORS)
            for item in ast.walk(node)
        )
    if lane == "INPUT_POWER":
        return any(
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Name)
            and item.func.id in TREE_UNARY_OPERATORS
            and _contains_operator(item.args[0], frozenset({"pow2", "pow3"}))
            for item in ast.walk(node)
        )
    if lane == "PRODUCT_COMPOSITION":
        return _has_binary_with_two_variable_branches(node, "mul")
    if lane == "SUM_COMPOSITION":
        return (
            _has_binary_with_two_variable_branches(node, "add")
            or _has_binary_with_two_variable_branches(node, "sub")
            or _has_sum_with_two_variable_branches(node)
        )
    if lane == "RATIONAL_COMPOSITION":
        return _contains_operator(node, frozenset({"div"}))
    return lane == "OPERATOR_FREE"


def validate_tree_response(
    response: Mapping[str, Any],
) -> tuple[tuple[ValidatedAtomTreeV1, ...], tuple[dict[str, Any], ...]]:
    """Validate every proposed serialized tree and retain structurally unique trees."""

    if not isinstance(response, Mapping) or set(response) != {"trees"}:
        raise AtomTreeError("tree response fields differ")
    raw_trees = response["trees"]
    if not isinstance(raw_trees, Sequence) or isinstance(raw_trees, (str, bytes)):
        raise AtomTreeError("trees must be an array")
    if len(raw_trees) != TREE_CANDIDATE_COUNT:
        raise AtomTreeError(f"tree response must contain exactly {TREE_CANDIDATE_COUNT} trees")
    raw_ids = [
        str(item.get("tree_id", "")) if isinstance(item, Mapping) else "" for item in raw_trees
    ]
    if set(raw_ids) != {f"T{index}" for index in range(1, TREE_CANDIDATE_COUNT + 1)}:
        raise AtomTreeError("tree IDs must be exactly T1 through T8")
    accepted: list[ValidatedAtomTreeV1] = []
    rejected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_shapes: set[str] = set()
    expected_fields = {
        "tree_id",
        "coverage_lane",
        "tree_template",
        "rationale_codes",
        "reason",
    }
    for raw in raw_trees:
        try:
            if not isinstance(raw, Mapping) or set(raw) != expected_fields:
                raise AtomTreeError("tree fields differ")
            tree_id = str(raw["tree_id"])
            if not _TREE_ID.fullmatch(tree_id) or tree_id in seen_ids:
                raise AtomTreeError("tree_id is invalid or duplicated")
            parsed, variables, parameters, structural_key = _tree_details(str(raw["tree_template"]))
            coverage_lane = str(raw["coverage_lane"])
            expected_lane = TREE_COVERAGE_LANES[int(tree_id[1:]) - 1]
            if coverage_lane != expected_lane:
                raise AtomTreeError(f"{tree_id} must declare coverage lane {expected_lane}")
            if not _matches_coverage_lane(coverage_lane, parsed.body):
                raise AtomTreeError(f"tree does not satisfy coverage lane {coverage_lane}")
            if tree_id != "T1" and _elementary_operator_count(parsed.body) < 1:
                raise AtomTreeError("SIGS-scale seed requires at least one elementary unary site")
            rationales = tuple(raw["rationale_codes"])
            if not rationales or any(
                not isinstance(item, str) or not _RATIONALE.fullmatch(item) for item in rationales
            ):
                raise AtomTreeError("rationale_codes are invalid")
            reason = str(raw["reason"]).strip()
            if not reason:
                raise AtomTreeError("tree reason is empty")
            if structural_key in seen_shapes:
                raise AtomTreeError("tree shape duplicates an earlier proposal")
            seen_ids.add(tree_id)
            seen_shapes.add(structural_key)
            accepted.append(
                ValidatedAtomTreeV1(
                    tree_id,
                    coverage_lane,
                    str(raw["tree_template"]),
                    variables,
                    parameters,
                    rationales,
                    reason,
                    structural_key,
                )
            )
        except (AtomTreeError, TypeError, ValueError) as exc:
            rejected.append({"raw_tree": raw, "error": f"{type(exc).__name__}: {exc}"})
    return tuple(accepted), tuple(rejected)


def _operator_choices(original: str, universe: Sequence[str], *, identity: bool) -> tuple[str, ...]:
    choices = [original]
    if identity:
        choices.append("identity")
    choices.extend(operator for operator in universe if operator != original)
    if identity:
        choices.extend(f"{outer}>{inner}" for outer in universe for inner in universe)
    return tuple(choices)


def _render_expanded_tree(node: ast.AST, replacements: Mapping[int, str]) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise AtomTreeError("invalid validated tree node during grammar expansion")
    operator = replacements.get(id(node), node.func.id)
    arguments = tuple(_render_expanded_tree(child, replacements) for child in node.args)
    if operator == "identity":
        if len(arguments) != 1:
            raise AtomTreeError("identity expansion requires one argument")
        return arguments[0]
    if ">" in operator:
        if len(arguments) != 1:
            raise AtomTreeError("unary composition expansion requires one argument")
        outer, inner = operator.split(">", maxsplit=1)
        return f"{outer}({inner}({arguments[0]}))"
    return f"{operator}({','.join(arguments)})"


def _call_operator(node: ast.Call) -> str:
    if not isinstance(node.func, ast.Name):
        raise AtomTreeError("expanded tree operator must be a direct name")
    return node.func.id


def _expanded_tree_variants(
    seed: ValidatedAtomTreeV1,
) -> tuple[ValidatedAtomTreeV1, ...]:
    """Expand one mandatory LLM seed over the frozen target-neutral operator grammar."""

    parsed, _variables, _parameters, _structural_key = _tree_details(seed.tree_template)
    mutable_sites: list[ast.Call] = []
    mutable_operators = set(TREE_ELEMENTARY_UNARY_OPERATORS) | set(TREE_POWER_OPERATORS)
    for item in ast.walk(parsed.body):
        if (
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Name)
            and item.func.id in mutable_operators
        ):
            mutable_sites.append(item)
    choices_by_site = tuple(
        _operator_choices(
            _call_operator(site),
            (
                TREE_ELEMENTARY_UNARY_OPERATORS
                if _call_operator(site) in TREE_ELEMENTARY_UNARY_OPERATORS
                else TREE_POWER_OPERATORS
            ),
            identity=_call_operator(site) in TREE_ELEMENTARY_UNARY_OPERATORS,
        )
        for site in mutable_sites
    )
    combinations = itertools.product(*choices_by_site) if choices_by_site else ((),)
    variants: list[ValidatedAtomTreeV1] = []
    seen_structures: set[str] = set()
    for operators in combinations:
        replacements = {
            id(site): operator for site, operator in zip(mutable_sites, operators, strict=True)
        }
        template = _render_expanded_tree(parsed.body, replacements)
        try:
            concrete, variables, parameters, structural_key = _tree_details(template)
        except AtomTreeError:
            continue
        if not _matches_coverage_lane(seed.coverage_lane, concrete.body):
            continue
        if structural_key in seen_structures:
            continue
        seen_structures.add(structural_key)
        variants.append(
            ValidatedAtomTreeV1(
                tree_id=f"{seed.tree_id}:E{len(variants) + 1:04d}",
                coverage_lane=seed.coverage_lane,
                tree_template=template,
                variable_holes=variables,
                parameter_holes=parameters,
                rationale_codes=seed.rationale_codes,
                reason=seed.reason,
                structural_key=structural_key,
            )
        )
        if len(variants) >= MAX_EXPANSIONS_PER_SEED:
            break
    return tuple(variants)


def _instantiate_tree(tree: ValidatedAtomTreeV1, bindings: Mapping[str, str]) -> str:
    parsed, _variables, _parameters, _key = _tree_details(tree.tree_template)

    def render(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            if node.id in bindings:
                return bindings[node.id]
            match = _PARAMETER_HOLE.fullmatch(node.id)
            if match:
                return f"p{match.group(1)}"
            raise AtomTreeError(f"unbound tree leaf: {node.id}")
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise AtomTreeError("invalid validated tree node")
        operator = node.func.id
        arguments = [render(child) for child in node.args]
        if operator == "add":
            return f"({arguments[0]}+{arguments[1]})"
        if operator == "sub":
            return f"({arguments[0]}-{arguments[1]})"
        if operator == "mul":
            return f"({arguments[0]}*{arguments[1]})"
        if operator == "div":
            return f"({arguments[0]}/{arguments[1]})"
        if operator == "sum":
            return f"({'+'.join(arguments)})"
        if operator == "pow2":
            return f"({arguments[0]}**2)"
        if operator == "pow3":
            return f"({arguments[0]}**3)"
        if operator == "abs":
            return f"Abs({arguments[0]})"
        return f"{operator}({arguments[0]})"

    return render(parsed.body)


def validate_binding_response(
    response: Mapping[str, Any],
    trees: Sequence[ValidatedAtomTreeV1],
    argument_roles: Sequence[str],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Bind LLM seeds, then expand them into a SIGS-scale concrete Atom population."""

    if not isinstance(response, Mapping) or set(response) != {"bindings"}:
        raise AtomTreeError("binding response fields differ")
    raw_bindings = response["bindings"]
    if not isinstance(raw_bindings, Sequence) or isinstance(raw_bindings, (str, bytes)):
        raise AtomTreeError("bindings must be an array")
    tree_by_id = {tree.tree_id: tree for tree in trees}
    if len(raw_bindings) != len(tree_by_id):
        raise AtomTreeError("binding count differs from validated tree count")
    allowed_roles = set(argument_roles)
    bound_seeds: list[
        tuple[
            ValidatedAtomTreeV1,
            Mapping[str, str],
            str,
            tuple[str, ...],
            str,
        ]
    ] = []
    rejected: list[dict[str, Any]] = []
    seen_tree_ids: set[str] = set()
    expected_fields = {
        "tree_id",
        "variable_bindings",
        "atom_type",
        "rationale_codes",
        "reason",
    }
    for raw in raw_bindings:
        try:
            if not isinstance(raw, Mapping) or set(raw) != expected_fields:
                raise AtomTreeError("binding fields differ")
            tree_id = str(raw["tree_id"])
            if tree_id not in tree_by_id or tree_id in seen_tree_ids:
                raise AtomTreeError("binding tree_id is unknown or duplicated")
            tree = tree_by_id[tree_id]
            items = raw["variable_bindings"]
            if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
                raise AtomTreeError("variable_bindings must be an array")
            mapping: dict[str, str] = {}
            for item in items:
                if not isinstance(item, Mapping) or set(item) != {"hole", "slot_argument"}:
                    raise AtomTreeError("variable binding fields differ")
                hole = str(item["hole"])
                role = str(item["slot_argument"])
                if hole in mapping or role not in allowed_roles:
                    raise AtomTreeError("variable binding is duplicate or uses an unknown role")
                mapping[hole] = role
            if tuple(sorted(mapping, key=lambda item: int(item[1:]))) != tree.variable_holes:
                raise AtomTreeError("binding does not cover the tree's exact V holes")
            rationales = tuple(raw["rationale_codes"])
            if not rationales or any(
                not isinstance(item, str) or not _RATIONALE.fullmatch(item) for item in rationales
            ):
                raise AtomTreeError("binding rationale_codes are invalid")
            reason = str(raw["reason"]).strip()
            if not reason:
                raise AtomTreeError("binding reason is empty")
            bound_seeds.append((tree, mapping, str(raw["atom_type"]), rationales, reason))
            seen_tree_ids.add(tree_id)
        except (AtomTreeError, TypeError, ValueError) as exc:
            rejected.append({"raw_binding": raw, "error": f"{type(exc).__name__}: {exc}"})

    groups: list[list[dict[str, Any]]] = []
    for seed, variable_mapping, atom_type, rationales, reason in bound_seeds:
        group: list[dict[str, Any]] = []
        for concrete in _expanded_tree_variants(seed):
            group.append(
                {
                    "tree_id": concrete.tree_id,
                    "seed_tree_id": seed.tree_id,
                    "seed_tree_template": seed.tree_template,
                    "seed_structural_key": seed.structural_key,
                    "coverage_lane": concrete.coverage_lane,
                    "tree_template": concrete.tree_template,
                    "structural_key": concrete.structural_key,
                    "canonical_template": _instantiate_tree(concrete, variable_mapping),
                    "atom_type": atom_type,
                    "variable_roles": sorted(set(variable_mapping.values())),
                    "free_parameters": [
                        f"p{index}" for index in range(1, len(concrete.parameter_holes) + 1)
                    ],
                    "rationale_codes": list(
                        dict.fromkeys((*seed.rationale_codes, *rationales, "GRAMMAR_EXPANDED"))
                    ),
                    "reason": f"{reason} Host-expanded over the frozen target-neutral grammar.",
                    "variable_bindings": dict(variable_mapping),
                }
            )
        groups.append(group)

    accepted: list[dict[str, Any]] = []
    seen_templates: set[str] = set()
    seen_structures: set[str] = set()
    for expansion_index in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if expansion_index >= len(group):
                continue
            candidate = group[expansion_index]
            template = str(candidate["canonical_template"])
            structure = str(candidate["structural_key"])
            if template in seen_templates or structure in seen_structures:
                continue
            accepted.append(candidate)
            seen_templates.add(template)
            seen_structures.add(structure)
            if len(accepted) == EXPANDED_ATOM_VALIDATION_BATCH_SIZE:
                return tuple(accepted), tuple(rejected)
    return tuple(accepted), tuple(rejected)


def request_tree_batch(
    provider: LLMBatchProviderV1,
    row: Mapping[str, Any],
    *,
    certified_ansatz: str,
    slot_id: str,
    profile_arity: int,
    validation_feedback: Sequence[str] = (),
    forbidden_structural_keys: Sequence[str] = (),
    exploration_pass: int = 0,
) -> tuple[
    tuple[ValidatedAtomTreeV1, ...],
    tuple[dict[str, Any], ...],
    LLMBatchCompletionV1,
    str,
]:
    messages = render_tree_messages(
        row,
        certified_ansatz=certified_ansatz,
        slot_id=slot_id,
        profile_arity=profile_arity,
        validation_feedback=validation_feedback,
        forbidden_structural_keys=forbidden_structural_keys,
        exploration_pass=exploration_pass,
    )
    completion = provider.complete(
        tokens=(slot_id, "tree"), messages=messages, response_format=tree_response_format()
    )
    accepted, rejected = validate_tree_response(completion.response)
    return accepted, rejected, completion, _canonical_sha256(messages)


def request_bound_atoms(
    provider: LLMBatchProviderV1,
    row: Mapping[str, Any],
    *,
    certified_ansatz: str,
    slot_id: str,
    current_ansatz: str,
    profile_call: str,
    profile_arguments: Sequence[str],
    trees: Sequence[ValidatedAtomTreeV1],
    validation_feedback: Sequence[str] = (),
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], LLMBatchCompletionV1, str]:
    roles = tuple(f"ARG{index}" for index in range(1, len(profile_arguments) + 1))
    messages = render_binding_messages(
        row,
        certified_ansatz=certified_ansatz,
        slot_id=slot_id,
        current_ansatz=current_ansatz,
        profile_call=profile_call,
        profile_arguments=profile_arguments,
        trees=trees,
        validation_feedback=validation_feedback,
    )
    completion = provider.complete(
        tokens=(slot_id, "binding"),
        messages=messages,
        response_format=binding_response_format(trees, roles),
    )
    accepted, rejected = validate_binding_response(completion.response, trees, roles)
    return accepted, rejected, completion, _canonical_sha256(messages)


__all__ = [
    "AtomTreeError",
    "EXPANDED_ATOM_VALIDATION_BATCH_SIZE",
    "MAX_EXPANSIONS_PER_SEED",
    "MAX_PROMPT_FEEDBACK_ITEMS",
    "MAX_PROMPT_STRUCTURAL_KEYS",
    "MAX_TREE_DEPTH",
    "MAX_TREE_NODES",
    "MAX_TREE_SUM_ARITY",
    "MAX_TREE_TEMPLATE_LENGTH",
    "TARGET_EXPANDED_ATOM_COUNT",
    "TREE_CANDIDATE_COUNT",
    "TREE_COVERAGE_LANES",
    "TREE_ELEMENTARY_UNARY_OPERATORS",
    "TREE_VARIADIC_OPERATORS",
    "ValidatedAtomTreeV1",
    "binding_response_format",
    "render_binding_messages",
    "render_tree_messages",
    "request_bound_atoms",
    "request_tree_batch",
    "tree_response_format",
    "validate_binding_response",
    "validate_tree_response",
]
