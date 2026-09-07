# LADS

Minimal release code for the paper's **Ansatz → direct Atom → host L-BFGS-B fitting** model.
The repository contains only the model, its small symbolic replay engine, and the released
public problem/validation files. No CDR research logs, paper sources, private data, checkpoints,
or API keys are included.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Run

The model uses OpenRouter. Coefficients are fitted with deterministic multi-start CPU L-BFGS-B. Set a key, then run all nine executable public cases:

```bash
export OPENROUTER_API_KEY=...
python3 run.py --model google/gemini-2.5-flash
```

Run one or more scalar cases:

```bash
python3 run.py --case-ids burgers poisson1
```

Use another environment variable for the key or another output path when needed:

```bash
python3 run.py --credential-env MY_OPENROUTER_KEY \
  --output results/poisson.json --case-ids poisson1
```

Check the bundled inputs without making an API request:

```bash
python3 run.py --dry-run
```

Results are written to `results/predictions.json` by default. The bundled nine cases are scalar host-validation inputs. They are condition-aware development
inputs, not a blind or held-out benchmark.

## Structure

```text
LADS/
├── run.py                       # single execution entry point
├── data/                        # released public inputs
├── src/lads/                    # model and host validation
├── src/pde_ansatz_discovery/    # minimal vendored symbolic replay engine
└── tests/                       # release smoke tests
```
