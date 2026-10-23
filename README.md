# LADS: LLM-guided Ansatz Discovery for Symbolic PDE Solutions

Code for the paper *LADS: LLM-guided Ansatz Discovery for Symbolic PDE Solutions*
(Sum Kyun Song, Jinkyu Jeong, Jae Yong Lee; Chung-Ang University).

LADS finds a closed-form solution of a PDE from the equation and its initial and
boundary conditions:

1. **Ansatz generation.** An LLM proposes atom-free templates with open slots,
   e.g. `u(x,t) = F1(x) + F2(t)`, which are checked symbolically.
2. **Slot completion.** For a certified Ansatz, the LLM proposes parameterized atoms
   for each slot (eight per request), e.g. `p1*ARG1 + p2`.
3. **Fit and select.** L-BFGS fits the free coefficients to the PDE and condition
   residuals. A candidate is returned once its validation loss is at most `1e-8`.
   Otherwise the outcome is sent back to the next slot-completion request: either the
   expression could not be used for computation, or it was computable but not accurate
   enough.

The LLM is called at most **eight times per problem** in total, and at most four
candidates are fitted. If the cap is reached first, the fitted candidate with the
lowest validation loss is returned.

## Setup

Requirements: Linux, Python 3.13, [uv](https://docs.astral.sh/uv/), an NVIDIA GPU with
CUDA 12 (fitting runs in float64 on the GPU), and an
[OpenRouter](https://openrouter.ai) API key. The paper uses `google/gemini-2.5-flash`
with temperature 0.

```bash
uv sync --frozen
# Let JAX find the CUDA libraries installed in the virtual environment.
export LD_LIBRARY_PATH=$(ls -d "$PWD"/.venv/lib/python3.13/site-packages/nvidia/*/lib | tr '\n' ':')
uv run --frozen pytest -q
```

## Run

```bash
export OPENROUTER_API_KEY=...
uv run --frozen python run_lads.py problems/case_001.json
```

The result (`status`, `prediction`, `validation_loss`, number of LLM calls) is printed
and written to `runs/case_001/result.json`, together with every request, raw response,
and fit.

`problems/` contains the 27 problems of the paper: `case_001`–`case_020` are the
manufactured problems and `case_101`–`case_107` are public problems following
*Neuro-Symbolic AI for Analytical Solutions of Differential Equations*
(arXiv:2502.01476). A problem file states only the PDE, the initial and boundary
conditions, the domain, and the parameters; no reference solution is given to the method.

## Layout

```
run_lads.py   run LADS on one problem (8-call cap, 4-fit cap)
src/          Ansatz and atom generation, symbolic checks, CUDA L-BFGS fitting
tools/        problem loading, residuals, shared optimizer, OpenRouter transport
problems/     27 problem definitions
tests/        unit tests
```
