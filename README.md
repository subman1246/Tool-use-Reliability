# tool-use-reliability

Invocation-level reliability of tool use in LLM agents. This repository holds
the experimental pipeline for the study: measuring how reliably models issue
correct function calls (correct tool AND correct arguments) on multi-step tasks,
separating context-length degradation from error propagation, and estimating a
poisoning-and-recovery model per model.

The methodology and its formal model live in `docs/`. This README covers the
code.

## What is here

A working pipeline that runs offline against a mock model and against real
providers through LiteLLM:

- `src/tur/tasks/` contamination-free synthetic DAG tasks with controllable
  dependency depth, distractors, and semantic noise.
- `src/tur/harness/` the run loop (free and teacher-forced), a tool executor
  that distinguishes syntactic from semantic failures and returns structured or
  opaque feedback, native and uniform calling modes, and a response cache.
- `src/tur/eval/` scoring: tool selection, argument correctness (strict and
  soft), and syntactic-versus-semantic error classification.
- `src/tur/model/` the three-state propagation recurrence, a forward simulator,
  and the PyMC hierarchical estimator with identifiability diagnostics.
- `src/tur/data/` loaders that map BFCL and tau-bench into a neutral task format
  for the real, primary evaluation. These are scaffolds; wire the exact fields
  for the benchmark version in use.

## How the code maps to the model

- Teacher-forced runs present the correct history at its true growing length and
  measure the depth-varying clean baseline `p_t`. Context-length degradation
  lives entirely in `p_t`.
- Free runs let the model thread its own outputs and measure the global rate
  `g_t`.
- Net propagation loss `L_t = 1 - g_t / p_t` is the fit-free headline.
- The hierarchical model fits severity `pi` and the two recovery rates
  `r_syn`, `r_sem`, with partial pooling across families and scales. When a
  parameter is not separately identified for a flatlined model, report `L_t`.

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

# offline, no keys needed
PYTHONPATH=src python scripts/run_experiment.py --mock

# a real model (reads keys from .env)
cp .env.example .env   # then fill in
PYTHONPATH=src python scripts/run_experiment.py --model groq/qwen2.5-7b-instruct

# tests
PYTHONPATH=src python tests/test_smoke.py       # pipeline, no heavy deps
PYTHONPATH=src python tests/test_regression.py # scoring/state-tracking fixes
PYTHONPATH=src python tests/test_fit.py        # Bayesian parameter recovery
PYTHONPATH=src python scripts/stress_test.py   # all policies x task kinds
```

## Status and known items

- **Full pipeline validated end to end** against six simulated model policies
  with known ground truth, run through the real harness (see
  `docs/RESULTS_validation_run.md` for the full write-up, and
  `data/results/figures/` for all seven generated figures).
- Confirmed working: task generation (linear and routing-DAG), the harness
  (free and teacher-forced, both retry-aligned), syntactic/semantic error
  classification, aggregation with NaN/edge-case guards, and the Bayesian
  hierarchical fit (R-hat 1.001, ESS ~2900, zero divergences after fixes).
- **Key finding from validation:** the fit-free net-propagation-loss metric
  `L_t = 1 - g_t/p_t` correctly recovers the configured severity ordering; the
  parametric `pi`/recovery fit does not, because `pi` and the recovery rates
  are structurally correlated under exact-match scoring (confirmed via the
  posterior-correlation diagnostic, not just asserted). This is the exact
  failure mode the methodology's fallback plan anticipated, and it now has
  empirical evidence behind it. Report `L_t` as primary; report `pi`/recovery
  as secondary, always alongside the identifiability diagnostic.
- **Seven correctness issues** were found and fixed across two audit rounds:
  retry-budget asymmetry, a stuck-poisoned simulated state, selection errors
  mis-attributed under poisoning (routing tasks scored against gold rather
  than against what was correct given the ref actually held), invisible
  stalled chains, `f_syn` contaminated by propagation, missing `L_t`
  uncertainty, and sampler divergences. See `docs/RESULTS_validation_run.md`
  Section 2. Four regression tests in `tests/test_regression.py` lock in the
  scoring and state-tracking behaviour.
- Selection-error propagation (not just argument-value propagation) is now
  exercised via `RoutingTask` (`tur.tasks.dag.generate_routing_task`), where
  the correct next tool is a function of the incoming value rather than
  pre-announced.
- BFCL and tau-bench loaders (`src/tur/data/loaders.py`) are scaffolds with
  the parsing structure and explicit TODOs; field mapping against the exact
  benchmark versions and tau-bench reference trajectories are the next real
  work items, and need network access this environment doesn't have.
- **Not yet done:** real model runs. Swap `SimPolicy`/`MockBackend` for
  `LiteLLMBackend` with real model names — everything else in the pipeline
  (harness, scoring, aggregation, fitting, figures) is unchanged.

## Reproducibility

Greedy decoding for main runs, fixed seeds for task generation and sampling,
every response cached by prompt, model, and calling mode, and provider and date
recorded per run. Secrets stay in `.env`, which is gitignored and never
committed.
