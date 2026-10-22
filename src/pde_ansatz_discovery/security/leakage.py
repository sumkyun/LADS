"""Static discovery-core leakage scanner."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

_LEGACY_PREFIX = "pde_ansatz_" + "recommender_v1"
_FORBIDDEN_IMPORT_PARTS = {
    _LEGACY_PREFIX + ".bank",
    _LEGACY_PREFIX + ".scoring",
    _LEGACY_PREFIX + ".selector",
    "benchmark_gold",
    "benchmark_evaluator",
    "holdout_gold",
    "pde_ansatz_discovery.evaluation.disclosed_reference",
}
_FORBIDDEN_CALLS = {"eval", "exec", "sympify", "parse_expr"}
_SENSITIVE_LITERALS = {
    "benchmark_id",
    "source_key",
    "pde_family",
    "reference_solution",
    "expected_ansatz",
    "catalog_row",
    "gold",
    "title",
    "notes",
    "ansatz_gold",
}
_CATALOG_FORMULAS = {
    "u_t=u_xx",
    "u_t+u*u_x=nu*u_xx",
}
_FULL_ROW_FINGERPRINTS = {
    "49c28fc0c5f07a54d8b9bb91ddd530316e83c9045e9e44c748fd14e731e585e6",
}
_REFERENCE_LITERAL_FILES = {
    "evaluation/disclosed_reference.py",
}
_REFERENCE_FIELD_ACCESS_FILES = {
    "evaluation/metrics.py",
}
_REFERENCE_DYNAMIC_MAPPING_FILES = {
    "evaluation/manifests.py",
    "evaluation/metrics.py",
    "evaluation/opaque.py",
    "evaluation/splits.py",
}

# Dynamic subscripting is denied unless the receiver is a narrowly identified,
# discovery-owned collection.  The allowlist deliberately names receivers rather
# than whole modules so adding ``row[key]`` (or an equivalent external-record
# lookup) anywhere in the discovery core remains a scan failure.
_DYNAMIC_SUBSCRIPT_RECEIVERS = {
    "analysis/scaling.py": {"row", "rows", "rows[index]", "rows[pivot_row]"},
    "api.py": {
        "branch_verifications",
        "feedback_details",
        "feedback_states",
        "status_reasons",
    },
    "cas/sympy_backend.py": {
        "basis",
        "coordinate_components",
        "held",
        "index",
        "item",
        "mapping",
        "normalized",
        "row",
        "self._symbols",
        "self.symbol_roles",
        "system.coordinates",
        "system.determining_matrix",
        "vector",
    },
    "generation/lie.py": {"left_components", "right_components"},
    "ir/parser.py": {"solution_counts"},
    "models.py": {"by_id"},
    "search/failure_feedback.py": {"FAILURE_EXPANSIONS_V1"},
    "search/ranking.py": {"_CONDITION_ORDER", "_VERIFICATION_ORDER"},
    "solving/prolongation.py": {
        "coefficients",
        "dict(self.jet_coefficients)",
    },
    "verification/charts.py": {"ansatz.inverse_map", "dict(inverse)"},
    "verification/jet_collection.py": {
        "combined",
        "generators",
        "raw",
        "result",
    },
    "verification/replay.py": {"ansatz.inverse_map"},
    "verification/substitution.py": {"found"},
}


def _constant_text(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _constant_text(node.left)
        right = _constant_text(node.right)
        return None if left is None or right is None else left + right
    return None


def _allowed_dynamic_getattr(relative: str, node: ast.Call) -> bool:
    if relative == "cas/sympy_backend.py":
        return True
    if len(node.args) < 2 or not isinstance(node.args[0], ast.Name):
        return False
    owner = node.args[0].id
    attribute = node.args[1]
    if owner in {"self", "cas"} and isinstance(attribute, ast.Name):
        return attribute.id in {"name", "field_name"}
    return (
        relative == "benchmark_adapter.py"
        and owner == "source"
        and isinstance(attribute, ast.Name)
        and attribute.id == "name"
    )


def _annotation_node_ids(tree: ast.AST) -> set[int]:
    """Return nodes that are syntax inside type annotations, not data access."""

    annotated: set[int] = set()
    for node in ast.walk(tree):
        annotations: tuple[ast.AST, ...] = ()
        if isinstance(node, (ast.AnnAssign, ast.arg)) and node.annotation is not None:
            annotations = (node.annotation,)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is not None:
            annotations = (node.returns,)
        for annotation in annotations:
            annotated.update(id(item) for item in ast.walk(annotation))
    return annotated


def _statically_non_text_subscript(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return not isinstance(node.value, str)
    if isinstance(node, ast.Slice):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
        return isinstance(node.operand.value, (int, float, complex))
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
    ):
        return any(
            isinstance(item, ast.Constant)
            and isinstance(item.value, (int, float, complex))
            for item in (node.left, node.right)
        )
    return False


def _allowed_dynamic_subscript(relative: str, node: ast.Subscript) -> bool:
    try:
        receiver = ast.unparse(node.value)
    except (AttributeError, ValueError):
        return False
    return receiver in _DYNAMIC_SUBSCRIPT_RECEIVERS.get(relative, set())


def _is_forbidden_import(name: str) -> bool:
    return any(
        name == item or name.startswith(item + ".")
        for item in _FORBIDDEN_IMPORT_PARTS
    )


def _import_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    """Resolve common aliases for importlib and its dynamic import entry point."""

    importlib_names = {"importlib", "builtins", "__builtins__"}
    import_call_names = {"__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"importlib", "builtins"}:
                    importlib_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    import_call_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "builtins":
            for alias in node.names:
                if alias.name == "__import__":
                    import_call_names.add(alias.asname or alias.name)
    # Alias assignments are intentionally narrow, but close the easy
    # ``loader = import_module`` / ``loader = importlib.import_module`` bypass.
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            value: ast.expr | None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                target = node.target
                value = node.value
            else:
                continue
            if not isinstance(target, ast.Name) or value is None:
                continue
            source_is_import = (
                isinstance(value, ast.Name)
                and value.id in import_call_names
            ) or (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id in importlib_names
                and value.attr in {"import_module", "__import__"}
            ) or (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "getattr"
                and len(value.args) >= 2
                and isinstance(value.args[0], ast.Name)
                and value.args[0].id in importlib_names
                and _constant_text(value.args[1]) in {"import_module", "__import__"}
            ) or (
                isinstance(value, ast.Subscript)
                and isinstance(value.value, ast.Name)
                and value.value.id == "__builtins__"
                and _constant_text(value.slice) == "__import__"
            )
            if source_is_import and target.id not in import_call_names:
                import_call_names.add(target.id)
                changed = True
    return importlib_names, import_call_names


def _introspection_aliases(
    tree: ast.AST,
) -> tuple[set[str], set[str], set[str]]:
    """Resolve aliases for whole-object mapping introspection entry points."""

    dataclasses_names = {"dataclasses"}
    builtins_names = {"builtins"}
    introspection_calls = {"vars"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "dataclasses":
                    dataclasses_names.add(alias.asname or alias.name)
                elif alias.name == "builtins":
                    builtins_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "dataclasses":
            for alias in node.names:
                if alias.name == "asdict":
                    introspection_calls.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "builtins":
            for alias in node.names:
                if alias.name == "vars":
                    introspection_calls.add(alias.asname or alias.name)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            value: ast.expr | None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                target = node.target
                value = node.value
            else:
                continue
            if not isinstance(target, ast.Name) or value is None:
                continue
            source_is_introspection = (
                isinstance(value, ast.Name)
                and value.id in introspection_calls
            ) or (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id in dataclasses_names
                and value.attr == "asdict"
            ) or (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id in builtins_names
                and value.attr == "vars"
            ) or (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "getattr"
                and len(value.args) >= 2
                and isinstance(value.args[0], ast.Name)
                and (
                    (
                        value.args[0].id in dataclasses_names
                        and _constant_text(value.args[1]) == "asdict"
                    )
                    or (
                        value.args[0].id in builtins_names
                        and _constant_text(value.args[1]) == "vars"
                    )
                )
            )
            if source_is_introspection and target.id not in introspection_calls:
                introspection_calls.add(target.id)
                changed = True
    return dataclasses_names, builtins_names, introspection_calls


def _dynamic_import_target(
    node: ast.Call,
    importlib_names: set[str],
    import_call_names: set[str],
) -> tuple[bool, str | None]:
    func = node.func
    recognized = isinstance(func, ast.Name) and func.id in import_call_names
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id in importlib_names
        and func.attr in {"import_module", "__import__"}
    ):
        recognized = True
    if (
        isinstance(func, ast.Call)
        and isinstance(func.func, ast.Name)
        and func.func.id == "getattr"
        and len(func.args) >= 2
        and isinstance(func.args[0], ast.Name)
        and func.args[0].id in importlib_names
        and _constant_text(func.args[1]) in {"import_module", "__import__"}
    ):
        recognized = True
    if (
        isinstance(func, ast.Subscript)
        and isinstance(func.value, ast.Name)
        and func.value.id == "__builtins__"
        and _constant_text(func.slice) == "__import__"
    ):
        recognized = True
    return recognized, (_constant_text(node.args[0]) if node.args else None)


@dataclass(frozen=True, order=True)
class LeakageFindingV1:
    path: str
    line: int
    code: str
    detail: str


def scan_discovery_source(root: Path) -> tuple[LeakageFindingV1, ...]:
    root = root.resolve()
    findings: list[LeakageFindingV1] = []
    backend = root / "cas" / "sympy_backend.py"
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text("utf-8"), filename=str(path))
        importlib_names, import_call_names = _import_aliases(tree)
        dataclasses_names, builtins_names, introspection_calls = _introspection_aliases(tree)
        annotation_nodes = _annotation_node_ids(tree)
        relative = str(path.relative_to(root))
        reference_literal_scope = relative in _REFERENCE_LITERAL_FILES
        reference_field_scope = relative in _REFERENCE_FIELD_ACCESS_FILES
        reference_dynamic_mapping_scope = relative in _REFERENCE_DYNAMIC_MAPPING_FILES
        scanner_scope = relative == "security/leakage.py"
        for node in ast.walk(tree):
            line = getattr(node, "lineno", 0)
            imports: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                imports = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports = (module,) + tuple(
                    ".".join(part for part in (module, alias.name) if part)
                    for alias in node.names
                )
            for name in imports:
                if _is_forbidden_import(name):
                    findings.append(LeakageFindingV1(relative, line, "FORBIDDEN_IMPORT", name))
                if (name == "sympy" or name.startswith("sympy.")) and path != backend:
                    findings.append(LeakageFindingV1(relative, line, "SYMPY_OUTSIDE_BACKEND", name))
            if isinstance(node, ast.Call):
                is_dynamic_import, import_target = _dynamic_import_target(
                    node, importlib_names, import_call_names
                )
                if is_dynamic_import:
                    if import_target is None:
                        findings.append(
                            LeakageFindingV1(
                                relative,
                                node.lineno,
                                "DYNAMIC_IMPORT",
                                "import target is not statically known",
                            )
                        )
                    elif _is_forbidden_import(import_target):
                        findings.append(
                            LeakageFindingV1(
                                relative,
                                node.lineno,
                                "FORBIDDEN_IMPORT",
                                import_target,
                            )
                        )
                called = node.func.id if isinstance(node.func, ast.Name) else (
                    node.func.attr if isinstance(node.func, ast.Attribute) else ""
                )
                full_row_introspection = (
                    isinstance(node.func, ast.Name)
                    and node.func.id in introspection_calls
                ) or (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in dataclasses_names
                    and node.func.attr == "asdict"
                ) or (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in builtins_names
                    and node.func.attr == "vars"
                ) or (
                    isinstance(node.func, ast.Call)
                    and isinstance(node.func.func, ast.Name)
                    and node.func.func.id == "getattr"
                    and len(node.func.args) >= 2
                    and isinstance(node.func.args[0], ast.Name)
                    and node.func.args[0].id in dataclasses_names
                    and _constant_text(node.func.args[1]) == "asdict"
                ) or (
                    isinstance(node.func, ast.Call)
                    and isinstance(node.func.func, ast.Name)
                    and node.func.func.id == "getattr"
                    and len(node.func.args) >= 2
                    and isinstance(node.func.args[0], ast.Name)
                    and node.func.args[0].id in builtins_names
                    and _constant_text(node.func.args[1]) == "vars"
                ) or (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "dict"
                    and len(node.args) == 1
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id in {"row", "record"}
                ) or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "__getattribute__"
                    and any(_constant_text(argument) == "__dict__" for argument in node.args)
                ) or (
                    isinstance(node.func, ast.Call)
                    and isinstance(node.func.func, ast.Name)
                    and node.func.func.id == "getattr"
                    and len(node.func.args) >= 2
                    and _constant_text(node.func.args[1]) == "__getattribute__"
                    and any(_constant_text(argument) == "__dict__" for argument in node.args)
                )
                if full_row_introspection and not scanner_scope:
                    findings.append(
                        LeakageFindingV1(
                            relative,
                            node.lineno,
                            "FULL_ROW_INTROSPECTION",
                            called,
                        )
                    )
                if called in _FORBIDDEN_CALLS:
                    findings.append(LeakageFindingV1(relative, node.lineno, "RAW_EVALUATION", called))
                if called == "getattr" and len(node.args) >= 2:
                    attribute = _constant_text(node.args[1])
                    if not reference_field_scope and not reference_literal_scope and not scanner_scope:
                        if attribute == "__dict__":
                            findings.append(
                                LeakageFindingV1(
                                    relative,
                                    node.lineno,
                                    "FULL_ROW_INTROSPECTION",
                                    "__dict__",
                                )
                            )
                        elif attribute in _SENSITIVE_LITERALS:
                            findings.append(LeakageFindingV1(relative, node.lineno, "SENSITIVE_ACCESS", attribute))
                        elif attribute is None and not _allowed_dynamic_getattr(relative, node):
                            findings.append(
                                LeakageFindingV1(
                                    relative,
                                    node.lineno,
                                    "DYNAMIC_ATTRIBUTE_ACCESS",
                                    "getattr attribute is not statically allowlisted",
                                )
                            )
                if (
                    called == "get"
                    and isinstance(node.func, ast.Attribute)
                    and node.args
                    and not reference_field_scope
                    and not reference_literal_scope
                    and not scanner_scope
                ):
                    key = _constant_text(node.args[0])
                    if key in _SENSITIVE_LITERALS:
                        findings.append(
                            LeakageFindingV1(
                                relative, node.lineno, "SENSITIVE_ACCESS", key
                            )
                        )
                    elif key is None and not reference_dynamic_mapping_scope:
                        findings.append(
                            LeakageFindingV1(
                                relative,
                                node.lineno,
                                "DYNAMIC_MAPPING_ACCESS",
                                "mapping key is not statically allowlisted",
                            )
                        )
            if (
                isinstance(node, ast.Attribute)
                and not reference_field_scope
                and not reference_literal_scope
                and not scanner_scope
                and node.attr == "__dict__"
            ):
                findings.append(
                    LeakageFindingV1(
                        relative,
                        node.lineno,
                        "FULL_ROW_INTROSPECTION",
                        "__dict__",
                    )
                )
            if (
                isinstance(node, ast.Attribute)
                and not reference_field_scope
                and not reference_literal_scope
                and not scanner_scope
                and node.attr in _SENSITIVE_LITERALS
            ):
                findings.append(
                    LeakageFindingV1(
                        relative, node.lineno, "SENSITIVE_ACCESS", node.attr
                    )
                )
            if isinstance(node, ast.Subscript):
                key = _constant_text(node.slice)
                if not reference_field_scope and not reference_literal_scope and not scanner_scope and key in _SENSITIVE_LITERALS:
                    findings.append(LeakageFindingV1(relative, node.lineno, "SENSITIVE_ACCESS", key))
                elif (
                    not reference_field_scope
                    and not reference_literal_scope
                    and not scanner_scope
                    and not reference_dynamic_mapping_scope
                    and key is None
                    and id(node) not in annotation_nodes
                    and isinstance(node.ctx, ast.Load)
                    and not _statically_non_text_subscript(node.slice)
                    and not _allowed_dynamic_subscript(relative, node)
                ):
                    findings.append(
                        LeakageFindingV1(
                            relative,
                            node.lineno,
                            "DYNAMIC_SUBSCRIPT_ACCESS",
                            "subscript key is not statically allowlisted",
                        )
                    )
            literal = _constant_text(node)
            if literal is not None:
                compact = literal.replace(" ", "")
                if not reference_field_scope and not reference_literal_scope and not scanner_scope and literal in _SENSITIVE_LITERALS:
                    findings.append(LeakageFindingV1(relative, line, "SENSITIVE_LITERAL", literal))
                if not scanner_scope and compact in _CATALOG_FORMULAS:
                    findings.append(LeakageFindingV1(relative, line, "CATALOG_FORMULA_LITERAL", compact))
                looks_like_sha256 = len(compact) == 64 and all(
                    character in "0123456789abcdefABCDEF" for character in compact
                )
                if not scanner_scope and (
                    looks_like_sha256
                    or compact.lower() in _FULL_ROW_FINGERPRINTS
                ):
                    findings.append(
                        LeakageFindingV1(
                            relative,
                            line,
                            "FULL_ROW_FINGERPRINT",
                            compact.lower(),
                        )
                    )
    return tuple(sorted(set(findings)))


__all__ = ["LeakageFindingV1", "scan_discovery_source"]
