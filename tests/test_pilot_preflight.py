"""Public-input preflight tests; no provider, candidate, reference, or scoring data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.pilot_preflight import run_preflight, validate_public_row


def _row() -> dict[str, object]:
    return {
        "boundary_conditions": [
            "q(0,s,z)=0",
            "Subs(Derivative(q(r,s,z),(r,2),z),r,1)=0",
        ],
        "dependent_variables": ["q"],
        "domain": {"r": ["0", "1"], "s": ["-1", "1"], "z": ["0", "2*pi"]},
        "independent_variables": ["r", "s", "z"],
        "initial_conditions": [],
        "parameters": {"alpha": "1/2"},
        "pde": [
            "diff(q(r,s,z),r,2)+Derivative(q(r,s,z),(s,2))+D(q(r,s,z),z)=0"
        ],
    }


def _write_selection(root: Path, rows: list[tuple[str, dict[str, object]]]) -> Path:
    selection = root / "selection"
    selected_cases = []
    frozen_files: dict[str, str] = {}
    for case_id, row in rows:
        relative = f"public/synthetic/cases/{case_id}.json"
        path = selection / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        frozen_files[relative] = digest
        selected_cases.append(
            {
                "case_id": case_id,
                "snapshot_path": relative,
                "sha256": digest,
                "stratum": "synthetic",
            }
        )
    manifest = {"schema_version": 1, "selected_cases": selected_cases}
    manifest_path = selection / "selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    frozen_files[manifest_path.name] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (selection / "FROZEN_ARTIFACTS.json").write_text(
        json.dumps({"schema_version": 1, "algorithm": "sha256", "files": frozen_files})
        + "\n",
        encoding="utf-8",
    )
    return selection


def test_valid_row_passes_real_ir_and_formal_condition_checks() -> None:
    result = validate_public_row(_row())

    assert result["status"] == "supported"
    assert result["errors"] == []
    assert result["checks"]["field_census"]["pde"] == ["q"]
    assert result["checks"]["maximum_derivative_order"] == 3


@pytest.mark.parametrize(
    ("condition", "code"),
    (
        ("q(0,s)=0", "CONDITION_PARSE_ERROR"),
        ("v(0,s,z)=0", "CONDITION_PARSE_ERROR"),
        ("Derivative(q(r,s,z),(r,0))=0", "DERIVATIVE_SYNTAX_ERROR"),
        ("Derivative(q(r,s,z),(r,2,3))=0", "DERIVATIVE_SYNTAX_ERROR"),
        ("diff(diff(q(r,s,z),r,5),s,4)=0", "DERIVATIVE_SYNTAX_ERROR"),
    ),
)
def test_condition_contract_rejects_bad_fields_and_derivatives(
    condition: str,
    code: str,
) -> None:
    row = _row()
    row["boundary_conditions"] = [condition]

    result = validate_public_row(row)

    assert result["status"] == "invalid_public_input"
    assert code in {error["code"] for error in result["errors"]}


@pytest.mark.parametrize(
    "domain",
    (
        {"r": ["0", "1"], "s": ["0", "inf"], "z": ["0", "1"]},
        {"r": ["1", "0"], "s": ["0", "1"], "z": ["0", "1"]},
        {"r": ["0", "1"], "s": ["0", "1"]},
    ),
)
def test_domain_requires_finite_ordered_bounds_for_every_axis(
    domain: dict[str, list[str]],
) -> None:
    row = _row()
    row["domain"] = domain

    result = validate_public_row(row)

    assert "DOMAIN_ERROR" in {error["code"] for error in result["errors"]}


def test_parameters_must_be_numeric() -> None:
    row = _row()
    row["parameters"] = {"alpha": "r + 1"}

    result = validate_public_row(row)

    assert "PARAMETER_ERROR" in {error["code"] for error in result["errors"]}


def test_malformed_pde_equation_has_pde_specific_classification() -> None:
    row = _row()
    row["pde"] = ["q(r,s,z)"]

    result = validate_public_row(row)

    assert "PDE_PARSE_ERROR" in {error["code"] for error in result["errors"]}


def test_preflight_retains_every_case_and_reports_hash_mismatch(tmp_path: Path) -> None:
    invalid = _row()
    invalid["pde"] = ["Derivative(q(r,s,z),(r,9))=0"]
    selection = _write_selection(tmp_path, [("case_good", _row()), ("case_bad", invalid)])
    bad_path = selection / "public/synthetic/cases/case_bad.json"
    bad_path.write_text(bad_path.read_text() + " ", encoding="utf-8")
    output = tmp_path / "preflight.json"

    artifact, exit_code = run_preflight(selection, output)

    assert exit_code == 1
    assert artifact["selected_count"] == 2
    assert len(artifact["cases"]) == 2
    assert artifact["supported_count"] == 1
    assert artifact["invalid_count"] == 1
    assert artifact["cases"][1]["source"]["hash_matches"] is False
    assert output.exists()
