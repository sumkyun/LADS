from __future__ import annotations

from combined_solution_v1 import staged_examples
from combined_solution_v1.llm_ansatz import (
    MAX_PROPOSALS_PER_CASE,
    SYSTEM_PROMPT,
    render_batch_messages,
)
from combined_solution_v1.llm_provider import InMemoryLLMBatchProviderV1
from combined_solution_v1.staged_examples import (
    MAX_REGENERATION_ATTEMPTS,
    STAGE_II_SYSTEM_PROMPT,
    extract_runtime_slots,
    render_atom_messages,
    run_staged_experiment,
    validate_atom_response,
)


def row():
    return {
        "benchmark_id": "local_synthetic",
        "dependent_variables": ["u"],
        "independent_variables": ["x", "t"],
        "pde": ["u_t-u_xx=0"],
        "initial_conditions": ["u(x,0)=sin(x)"],
        "boundary_conditions": ["u(0,t)=0", "u(1,t)=0"],
        "domain": {
            "status": "specified",
            "bounds": {"x": [0.0, 1.0], "t": [0.0, 2.0]},
        },
        "parameters": {"nu": 0.125},
    }


def atom_response():
    return {
        "proposals": [
            {
                "canonical_template": "p1*ARG1+p2",
                "atom_type": "TERM",
                "variable_roles": ["ARG1"],
                "free_parameters": ["p1", "p2"],
                "rationale_codes": ["OPERATOR_COMPATIBLE"],
                "reason": "affine test proposal",
            },
            {
                "canonical_template": "p1*sin(p2*ARG1+p3)+p4",
                "atom_type": "PRIMITIVE",
                "variable_roles": ["ARG1"],
                "free_parameters": ["p1", "p2", "p3", "p4"],
                "rationale_codes": ["OPERATOR_COMPATIBLE"],
                "reason": "oscillatory test proposal",
            },
            {
                "canonical_template": "p1*exp(p2*ARG1)+p3",
                "atom_type": "ENVELOPE",
                "variable_roles": ["ARG1"],
                "free_parameters": ["p1", "p2", "p3"],
                "rationale_codes": ["OPERATOR_COMPATIBLE"],
                "reason": "exponential test proposal",
            },
            {
                "canonical_template": "p1*cos(p2*ARG1+p3)+p4",
                "atom_type": "PRIMITIVE",
                "variable_roles": ["ARG1"],
                "free_parameters": ["p1", "p2", "p3", "p4"],
                "rationale_codes": ["OPERATOR_COMPATIBLE"],
                "reason": "cosine test proposal",
            },
            {
                "canonical_template": "p1*tanh(p2*ARG1+p3)+p4",
                "atom_type": "PRIMITIVE",
                "variable_roles": ["ARG1"],
                "free_parameters": ["p1", "p2", "p3", "p4"],
                "rationale_codes": ["OPERATOR_COMPATIBLE"],
                "reason": "hyperbolic test proposal",
            },
            {
                "canonical_template": "p1*cosh(p2*ARG1+p3)+p4",
                "atom_type": "ENVELOPE",
                "variable_roles": ["ARG1"],
                "free_parameters": ["p1", "p2", "p3", "p4"],
                "rationale_codes": ["OPERATOR_COMPATIBLE"],
                "reason": "hyperbolic envelope test proposal",
            },
            {
                "canonical_template": "p1/(1+(p2*ARG1+p3)**2)+p4",
                "atom_type": "TERM",
                "variable_roles": ["ARG1"],
                "free_parameters": ["p1", "p2", "p3", "p4"],
                "rationale_codes": ["OPERATOR_COMPATIBLE"],
                "reason": "rational test proposal",
            },
            {
                "canonical_template": "p1*(p2*ARG1+p3)**2+p4",
                "atom_type": "TERM",
                "variable_roles": ["ARG1"],
                "free_parameters": ["p1", "p2", "p3", "p4"],
                "rationale_codes": ["OPERATOR_COMPATIBLE"],
                "reason": "quadratic test proposal",
            },
        ]
    }


def stage_i_response():
    return {
        "schema_version": "LLMAnsatzBatchResponseV1",
        "cases": [
            {
                "case_token": "case_001",
                "candidates": [
                    {
                        "ansatz": "u(x,t)=a1*F1(x)*F2(t)",
                        "reason": "separation under the linear operator",
                    },
                    {
                        "ansatz": "u(x,t)=F1(x)+F2(t)",
                        "reason": "additive decomposition under the linear operator",
                    },
                    {
                        "ansatz": "u(x,t)=F1(a1*x+a2*t)",
                        "reason": "translation invariance gives a joint coordinate",
                    },
                ],
            }
        ],
    }


def invalid_stage_i_response():
    return {
        "schema_version": "LLMAnsatzBatchResponseV1",
        "cases": [
            {
                "case_token": "case_001",
                "candidates": [
                    {"ansatz": "u(x,t)=sin(x)", "reason": "invalid fixed profile"},
                    {"ansatz": "u(x,t)=cos(t)", "reason": "invalid fixed profile"},
                    {"ansatz": "u(x,t)=exp(x-t)", "reason": "invalid fixed profile"},
                ],
            }
        ],
    }


def partially_invalid_atom_response():
    value = atom_response()
    value["proposals"][0] = {
        "canonical_template": "p1",
        "atom_type": "PRIMITIVE",
        "variable_roles": ["ARG1"],
        "free_parameters": ["p1"],
        "rationale_codes": ["CONSTANT"],
        "reason": "invalid role declaration for a constant",
    }
    return value


def semantically_regenerated_atom_response():
    return {
        "proposals": [
            {
                "canonical_template": f"p1*(p2*ARG1+p3)**{power}+p4",
                "atom_type": "TERM",
                "variable_roles": ["ARG1"],
                "free_parameters": ["p1", "p2", "p3", "p4"],
                "rationale_codes": ["SEMANTIC_REGENERATION"],
                "reason": "new synthetic family after host semantic feedback",
            }
            for power in range(3, 11)
        ]
    }


def coupled_stage_i_response():
    return {
        "schema_version": "LLMAnsatzBatchResponseV1",
        "cases": [
            {
                "case_token": "case_001",
                "candidates": [
                    {
                        "ansatz": (
                            "rho(x,y,t)=R1(x,y,t);u(x,y,t)=U1(x,y,t);"
                            "v(x,y,t)=V1(x,y,t)"
                        ),
                        "reason": "three coupled open profiles preserve all operator coordinates",
                    },
                    {
                        "ansatz": (
                            "rho(x,y,t)=R2(x,y,t);u(x,y,t)=U2(x,y,t);"
                            "v(x,y,t)=V2(x,y,t)"
                        ),
                        "reason": "a distinct coupled profile basis preserves all fields",
                    },
                    {
                        "ansatz": (
                            "rho(x,y,t)=R3(x,y,t);u(x,y,t)=U3(x,y,t);"
                            "v(x,y,t)=V3(x,y,t)"
                        ),
                        "reason": "an alternate coupled profile basis preserves all fields",
                    },
                ],
            }
        ],
    }


def three_argument_atom_response():
    return {
        "proposals": [
            {
                "canonical_template": f"p1*ARG1**{power}+p2*ARG2+p3*ARG3+p4",
                "atom_type": "TERM",
                "variable_roles": ["ARG1", "ARG2", "ARG3"],
                "free_parameters": ["p1", "p2", "p3", "p4"],
                "rationale_codes": ["COUPLED_OPERATOR_COMPATIBLE"],
                "reason": "synthetic coupled Atom transport test",
            }
            for power in range(1, 9)
        ]
    }


def test_stage_i_prompt_includes_public_conditions_domain_and_exact_three():
    tokens, messages, payloads = render_batch_messages((row(),))
    assert MAX_PROPOSALS_PER_CASE == 3
    assert tokens == ("case_001",)
    assert len(payloads) == 1
    assert "q(r,s)=a1*F1(r)*F2(s)" in SYSTEM_PROMPT
    assert "sin(" not in SYSTEM_PROMPT
    assert "Generate exactly 3" in messages[1]["content"]
    assert "Initial conditions:" in messages[1]["content"]
    assert "u(x,0)=sin(x)" in messages[1]["content"]
    assert "Boundary conditions:" in messages[1]["content"]
    assert "u(0,t)=0" in messages[1]["content"]
    assert "Domain:" in messages[1]["content"]
    assert '"x":[0.0,1.0]' in messages[1]["content"]
    assert "nu = 0.125" in messages[1]["content"]
    assert payloads[0]["initial_conditions"] == ["u(x,0)=sin(x)"]
    assert payloads[0]["boundary_conditions"] == ["u(0,t)=0", "u(1,t)=0"]
    assert payloads[0]["parameter_values"] == {"nu": 0.125}
    assert "benchmark_id" not in messages[1]["content"]


def test_atom_prompt_is_bound_to_one_certified_slot_and_has_transform_example():
    ansatz = "u(x,t)=a1*F1(x)*F2(t)"
    ast, slots = extract_runtime_slots(row(), ansatz, "synthetic")
    assert ast.profile_names == ("F1", "F2")
    assert len(slots) == 2
    messages = render_atom_messages(
        row(),
        certified_ansatz=ansatz,
        slot_id=slots[0].slot_id,
        current_ansatz=slots[0].current_ansatz,
        profile_call=str(slots[0].call),
        profile_arguments=tuple(map(str, slots[0].call.args)),
    )
    assert "H(r) -> p1*ARG1+p2" in STAGE_II_SYSTEM_PROMPT
    assert "exactly one open slot" in messages[1]["content"]
    assert "Initial conditions:" in messages[1]["content"]
    assert "u(x,0)=sin(x)" in messages[1]["content"]
    assert "Boundary conditions:" in messages[1]["content"]
    assert "u(1,t)=0" in messages[1]["content"]
    assert "Domain:" in messages[1]["content"]
    assert "nu = 0.125" in messages[1]["content"]
    assert "benchmark_id" not in messages[1]["content"]
    accepted, rejected = validate_atom_response(atom_response(), slots[0])
    assert len(accepted) == 8
    assert rejected == ()


def test_full_staged_generation_never_uses_a_non_llm_fallback():
    created = []
    observed = []

    def factory(max_tokens: int):
        response = stage_i_response() if max_tokens == 8192 else atom_response()
        provider = InMemoryLLMBatchProviderV1(response, model="test-live-shape")
        created.append(provider)
        return provider

    result = run_staged_experiment(
        (row(),),
        {},
        provider_factory=factory,
        requested_model="test-live-shape",
        case_observer=observed.append,
        transport_retry_policy="synthetic outer transport policy",
    )
    case = result["cases"][0]
    assert result["provider"]["stage_i"]["request_count"] == 1
    assert result["provider"]["stage_ii"]["request_count"] >= 1
    assert result["provider"]["transport_retry_policy"] == (
        "synthetic outer transport policy"
    )
    assert case["stage_i"]["certified_count"] >= 1
    assert case["assembled_candidates"]
    assert case["status"] == "staged_generation_only_unscorable_public_forcing"
    assert all(provider.calls for provider in created)
    assert observed == [case]


def test_coupled_public_operator_reaches_nonempty_staged_generation_result():
    coupled_row = {
        "benchmark_id": "local_coupled",
        "dependent_variables": ["rho", "u", "v"],
        "independent_variables": ["x", "y", "t"],
        "pde": [
            "rho_t+d_x(rho*u)+d_y(rho*v)-f_rho=0",
            "d_t(rho*u)+d_x(rho*u^2)+d_y(rho*u*v)-f_u=0",
            "d_t(rho*v)+d_x(rho*u*v)+d_y(rho*v^2)-f_v=0",
            "f_rho, f_u, and f_v are specified only by unavailable source fields",
        ],
        "initial_conditions": [],
        "boundary_conditions": ["periodic in x and y"],
        "domain": {
            "status": "specified",
            "bounds": {"x": [-1.0, 1.0], "y": [-1.0, 1.0], "t": [0.0, 1.0]},
        },
        "parameters": {},
    }

    def factory(max_tokens: int):
        response = (
            coupled_stage_i_response()
            if max_tokens == 8192
            else three_argument_atom_response()
        )
        return InMemoryLLMBatchProviderV1(response, model="test-live-shape")

    result = run_staged_experiment(
        (coupled_row,),
        {},
        provider_factory=factory,
        requested_model="test-live-shape",
    )

    case = result["cases"][0]
    assert case["stage_i"]["certified_count"] == 3
    assert case["assembled_candidates"]
    assert case["status"] == "staged_generation_only_unscorable_public_forcing"
    assert result["provider"]["stage_i"]["request_count"] == 1
    assert result["provider"]["stage_ii"]["request_count"] == 9


def test_stage_i_retries_with_exact_host_errors_until_three_ansatz_candidates_pass():
    created = []
    stage_i_call_count = 0

    def factory(max_tokens: int):
        nonlocal stage_i_call_count
        if max_tokens == 8192:
            response = invalid_stage_i_response() if stage_i_call_count == 0 else stage_i_response()
            stage_i_call_count += 1
        else:
            response = atom_response()
        provider = InMemoryLLMBatchProviderV1(response, model="test-live-shape")
        created.append((max_tokens, provider))
        return provider

    result = run_staged_experiment(
        (row(),), {}, provider_factory=factory, requested_model="test-live-shape"
    )

    stage_i = result["cases"][0]["stage_i"]
    assert stage_i["attempt_count"] == 2
    assert stage_i["certified_count"] == 3
    second_stage_i_provider = [provider for size, provider in created if size == 8192][1]
    repair_prompt = second_stage_i_provider.calls[0]["messages"][1]["content"]
    assert "Previous response: host validation feedback" in repair_prompt
    assert "Rejected Ansatz u(x,t)=sin(x)" in repair_prompt
    assert "contains a concrete atom" in repair_prompt


def test_stage_i_host_feedback_loop_stops_after_ten_attempts():
    created = []

    def factory(max_tokens: int):
        assert max_tokens == 8192
        provider = InMemoryLLMBatchProviderV1(invalid_stage_i_response(), model="test-live-shape")
        created.append(provider)
        return provider

    result = run_staged_experiment(
        (row(),), {}, provider_factory=factory, requested_model="test-live-shape"
    )

    stage_i = result["cases"][0]["stage_i"]
    assert MAX_REGENERATION_ATTEMPTS == 10
    assert stage_i["attempt_count"] == 10
    assert stage_i["certified_count"] == 0
    assert result["provider"]["stage_i"]["max_attempts_exhausted_case_count"] == 1
    assert len(created) == 10


def test_stage_ii_retains_valid_atoms_and_retries_only_until_pool_has_eight():
    created = []
    atom_call_count = 0

    def factory(max_tokens: int):
        nonlocal atom_call_count
        if max_tokens == 8192:
            response = stage_i_response()
        else:
            response = (
                partially_invalid_atom_response() if atom_call_count % 2 == 0 else atom_response()
            )
            atom_call_count += 1
        provider = InMemoryLLMBatchProviderV1(response, model="test-live-shape")
        created.append((max_tokens, provider))
        return provider

    result = run_staged_experiment(
        (row(),), {}, provider_factory=factory, requested_model="test-live-shape"
    )

    slots = [
        slot for ansatz in result["cases"][0]["certified_ansatz_runs"] for slot in ansatz["slots"]
    ]
    assert slots
    assert all(slot["attempt_count"] == 2 for slot in slots)
    assert all(len(slot["accepted_candidates"]) == 8 for slot in slots)
    repair_providers = [provider for size, provider in created if size == 4096][1::2]
    assert repair_providers
    for provider in repair_providers:
        repair_prompt = provider.calls[0]["messages"][1]["content"]
        assert "Rejected Atom p1" in repair_prompt
        assert "variable_roles differ from the expression" in repair_prompt


def test_semantic_failure_regenerates_each_slot_and_revalidates(monkeypatch):
    created = []

    class AdaptiveAtomProvider:
        def __init__(self):
            self.calls = []

        def complete(self, *, tokens, messages, response_format):
            self.calls.append({"tokens": tokens, "messages": messages})
            prompt = messages[1]["content"]
            response = (
                semantically_regenerated_atom_response()
                if "SEMANTIC_VALIDATION_FAILED" in prompt
                else atom_response()
            )
            return InMemoryLLMBatchProviderV1(response, model="test-live-shape").complete(
                tokens=tokens,
                messages=messages,
                response_format=response_format,
            )

    def factory(max_tokens: int):
        if max_tokens == 8192:
            provider = InMemoryLLMBatchProviderV1(stage_i_response(), model="test-live-shape")
        else:
            provider = AdaptiveAtomProvider()
        created.append((max_tokens, provider))
        return provider

    def fake_semantic_validator(_row, candidates, _spec, *, tolerance=1.0e-8):
        regenerated = any(
            any(int(rank) > 8 for rank in item["slot_atom_ranks"].values()) for item in candidates
        )
        selected = {
            "validation_loss": 0.0 if regenerated else 1.0,
            "semantic_validation_passed": regenerated,
            "semantic_validation_tolerance": tolerance,
        }
        diagnostics = (
            ()
            if regenerated
            else (
                "SEMANTIC_VALIDATION_FAILED: synthetic residual remains high",
                "PDE_RESIDUAL_HIGH: synthetic operator mismatch",
            )
        )
        return (), selected, diagnostics

    monkeypatch.setattr(staged_examples, "semantic_validate_candidates", fake_semantic_validator)
    result = run_staged_experiment(
        (row(),),
        {"local_synthetic": {"synthetic": True}},
        provider_factory=factory,
        requested_model="test-live-shape",
    )

    case = result["cases"][0]
    assert case["status"] == "semantic_validation_passed"
    assert all(
        run["semantic_rounds"][-1]["passed"]
        for run in case["certified_ansatz_runs"]
        if run["status"] != "blocked_before_assembly"
    )
    atom_prompts = [
        call["messages"][1]["content"]
        for size, provider in created
        if size == 4096
        for call in provider.calls
    ]
    assert any("PDE_RESIDUAL_HIGH" in prompt for prompt in atom_prompts)
    assert any("Already retained; do not repeat" in prompt for prompt in atom_prompts)
