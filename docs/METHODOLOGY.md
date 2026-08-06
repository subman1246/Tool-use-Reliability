# Methodology and Implementation Plan

## 1. Research questions and hypotheses

The study measures how reliably tool-augmented language models issue correct function calls, and how that reliability degrades as calls depend on one another. A correct invocation means selecting the correct tool and supplying valid, correct arguments. We measure it at two levels: locally, given clean inputs, and globally, inside the model's own multi-step run where inputs may already be wrong. Crucially, we separate two distinct forces that both lower reliability with depth: the growth of the context window itself, and the propagation of earlier errors.

Research questions:

- RQ1: What is each model's per-step correct-invocation rate, and how does it split between selection and argument errors?
- RQ2: How much of the decline in reliability with depth is due to context-length degradation of the clean baseline, and how much is due to error propagation?
- RQ3: Once a context is corrupted, how likely is a model to recover, and does recovery differ between syntactic failures and silent semantic errors?
- RQ4: How do poisoning severity and recovery vary with model scale, model family, and function-calling-specific tuning?

Hypotheses:

- H1: For weaker models, global per-step reliability falls with depth faster than the measured clean baseline alone can explain, indicating propagation.
- H2: Within a single family, poisoning severity shrinks as parameter scale grows.
- H3: Function-calling-tuned models raise the clean rate but do not necessarily reduce poisoning severity.
- H4: The error mix shifts with depth, from selection-dominated at the first step to argument-dominated deeper in the chain.
- H5a: Recovery from syntactic failures rises when the tool returns a structured exception rather than an opaque failure.
- H5b: Recovery from semantic errors stays low regardless of feedback format, because no exception is raised.
- H6: The clean baseline itself declines with depth (context-length degradation), independent of any propagation.

## 2. Formal framework

The earlier version of this plan treated the clean per-step reliability as a single constant `p`. That is unsafe. As depth grows the prompt grows, the model must ingest a longer observation history, and attention degrades with context length. If the clean baseline is forced to be constant, a nonlinear fit will absorb that natural degradation into the poisoning parameter and inflate the estimated severity. The framework below fixes this by measuring the clean baseline at each depth and defining poisoning as a penalty relative to it, and it splits recovery into two channels because syntactic and semantic errors behave differently.

### Definitions

A task is a sequence of calls `c_1, ..., c_n`, where the arguments of `c_t` may depend on the outputs of earlier calls.

- Clean baseline `p_t`: the probability step `t` is correct given a correct upstream history of length `t - 1`. This is measured, not fitted, by teacher-forcing the clean history at its true length so `p_t` carries the full context-length effect. `p_t` is expected to fall with `t`.
- Poisoning severity `pi`: a single constant fractional penalty. Correctness under a poisoned context at depth `t` is `rho_t = (1 - pi) * p_t`. Because the context-length effect is already inside `p_t`, `pi` captures only the extra damage from corruption, which is exactly the separation the critique demands.

### Independence and baseline-only references

Two reference curves bound the interpretation. If errors neither propagated nor compounded, the global per-step rate would equal the measured clean baseline `p_t`. If errors were independent and multiplicative, chain-correctness would be the running product of `p_t`. Any global per-step rate below `p_t` is the propagation signal.

### Error types and two recovery channels

When a step errs, the error is one of two kinds:

- Syntactic: a parse or schema failure. The executor catches it and can return a structured exception.
- Semantic: a well-formed call that executes successfully but is wrong, either a valid but incorrect tool or a correctly formatted but factually wrong argument. No exception is raised; the tool returns plausible, irrelevant data.

These recover differently, so the poisoned state is split by origin, with separate recovery probabilities `r_syn` and `r_sem`. The share of errors that are syntactic at depth `t`, written `f_syn(t)`, is measured from the logs.

### State model

Track three occupancies entering step `t`: clean `c_t`, poisoned-by-syntax `s_t`, poisoned-by-semantics `m_t`, summing to 1, starting at `c_1 = 1`. Using the measured `p_t` and `f_syn(t)`:

```
c_(t+1) = c_t * p_t + s_t * r_syn + m_t * r_sem
s_(t+1) = c_t * (1 - p_t) * f_syn(t) + s_t * (1 - r_syn)
m_(t+1) = c_t * (1 - p_t) * (1 - f_syn(t)) + m_t * (1 - r_sem)
```

Total poisoned mass is `x_t = s_t + m_t = 1 - c_t`. The global per-step correctness is a mixture of clean and poisoned reliability:

```
g_t = c_t * p_t + x_t * (1 - pi) * p_t = p_t * (1 - pi * x_t)
```

The structure is clean: the observed global curve is the measured baseline `p_t` discounted by the severity `pi` weighted by how poisoned the context is likely to be, `x_t`, where `x_t` is driven by the two recovery rates. At `t = 1`, `g_1 = p_1`. Setting `r_syn = r_sem = 0` reproduces the absorbing model; a single common recovery reproduces the previous two-state model; a constant `p` reproduces the original decay fit. Each earlier version is a special case, which is the right way to present the progression to reviewers.

### Reported quantities

The headline number is directly observable and needs no fit:

```
Net propagation loss at depth t:   L_t = 1 - g_t / p_t = pi * x_t
```

Because both `g_t` and `p_t` are measured, `L_t` is always available even when the underlying parameters are hard to separate. On top of it we report the fitted structural parameters `pi`, `r_syn`, `r_sem`, and the context-degradation slope of `p_t`. Reporting `L_t` as primary and the parameters as mechanism protects the central claim against the identifiability problem discussed below.

### Why most parameters are measured, not just fit

The synthetic tasks have exact ground truth, so at every step we can label the true state (clean, poisoned-syntax, poisoned-semantic) and observe the transitions directly. This means `p_t`, `f_syn(t)`, and both recovery rates are measured from labelled transitions, not inferred from the shape of a curve. `pi` is pinned by the `g_t / p_t` ratio. The structural model mainly supplies a principled way to pool these measurements and to carry them to the real data, where latent states are recovered against gold trajectories.

## 3. Estimation: Bayesian hierarchical model

Nonlinear least squares on the depth grid is fragile exactly where it matters. A weak 7B or 8B model can collapse to its floor by depth 2, so `g_4`, `g_6`, and `g_8` all sit at the same level, the Jacobian becomes near-singular, and the optimizer cannot tell high severity from low recovery. We therefore estimate with a Bayesian hierarchical model (PyMC, NUTS sampler) rather than a point fit.

- Parameters `pi`, `r_syn`, `r_sem` per model, drawn from family-level and scale-level hyperpriors so that scales within a family and families within the suite share strength through partial pooling. A model that flatlines early borrows structure from its relatives instead of diverging.
- Weakly informative priors: Beta priors on `pi`, `r_syn`, `r_sem` centred on modest values with enough spread to be dominated by data when data exist. The directly measured recovery rates from the labelled synthetic transitions enter as informative priors, tightening the real-data fits.
- The likelihood combines the per-call Bernoulli outcomes with the measured `p_t` supplied as data, so the fit is of `g_t = p_t (1 - pi x_t)` against a dynamic baseline, never a static one.
- Diagnostics reported for every model: R-hat, effective sample size, divergences, and the posterior correlation between `pi` and each recovery rate. When that correlation approaches plus or minus one, the two are not separately identified for that model, and we say so and fall back to reporting the identifiable `L_t` and the steady-state floor rather than pretending to a precise `r`.

This is the difference between a fit that quietly explodes on the weakest and most interesting models and one that reports honest, regularized uncertainty.

## 4. Model suite

The suite varies one factor at a time so differences in `p_t`, `pi`, and the recovery rates can be attributed to a cause. Versions are pinned by release date at run time and logged.

| Axis | What it isolates | Models |
|---|---|---|
| Scale within a family | Effect of parameter count, architecture held constant | Qwen2.5-Instruct at 7B, 14B, 32B, 72B |
| Family at matched scale | Effect of architecture and training data | Qwen2.5-7B vs Llama-3.1-8B vs Gemma-2-9B vs Mistral-8B-class |
| Function-calling tuning | Effect of tool-specific fine-tuning | An FC-tuned model (xLAM-2 or Hammer2.1, 7B) vs its general instruct base |
| Proprietary reference | A frontier ceiling for context | One or two of GPT-class and Claude-class, on a reduced subset |

Serving is through a single client (LiteLLM): open-weight models on a fast low-cost host (Groq, with Together, Fireworks, or OpenRouter as fallback), proprietary models on their own APIs. Decoding is greedy for the main measurement, with a few seeds for reliability variance.

## 5. Task design and data

### Primary: synthetic, contamination-free, with semantic noise

Each task is a hidden dependency graph over deterministic, executable functions; solving it means traversing the graph and feeding each function the outputs of its predecessors. This gives exact ground truth per call and direct control of dependency depth, at depths `n` in {1, 2, 4, 6, 8} and two or three distractor levels. To avoid sterility we inject semantic noise: natural-language argument values, paraphrased and overlapping tool descriptions, distractor tools with similar names, and free-text fields that must be extracted rather than copied. The exact ground truth is what lets us detect semantic errors that execute without raising an exception, which is essential for the semantic recovery channel.

### Error-feedback manipulation

Tool failures are returned in one of two randomised forms: a structured exception naming the offending parameter and expected type, or an opaque generic failure. This manipulation acts on the syntactic channel and is the direct test of H5a and H5b.

### Secondary: real benchmark, sized to matter

A large multi-step real set, on the order of 300 to 500 tasks, is drawn from BFCL's multi-turn and multi-step splits and tau-bench, with StableToolBench's simulator where live APIs are needed, sized so each depth bin can fit its own curve. Semantic-error labelling on real tasks uses the gold trajectories. Scoring is identical to the synthetic set so the two are directly comparable.

## 6. Evaluation protocol

For each task and model:

1. Present schemas and task; collect calls; record raw output, parsed call, and any parse failure.
2. Score selection against the gold tool.
3. Score arguments in two stages: schema validity, then value correctness (exact for identifiers and enums, numeric tolerance, normalised or soft match for free text), with strict and soft value scores reported side by side.
4. Classify each error as syntactic (parse or schema failure) or semantic (executed but wrong tool or value), using the exact ground truth on synthetic tasks and gold trajectories on real ones.
5. Clean baseline `p_t`: teacher-force the correct upstream history at its true length and measure step-`t` correctness at each depth bin. This yields the dynamic `p_t`, capturing context-length degradation.
6. Global score: score each step inside the model's own unmodified run to get `g_t`.
7. Recovery: from the free runs, measure `r_syn` and `r_sem` separately as the rates at which a syntactically or semantically poisoned context returns to a correct, on-track chain.
8. Repeat over seeds for a pass-across-trials reliability figure and variance bands.

### Controlling the function-calling confound

Parse failures are bucketed separately from semantic argument errors, so provider grammar differences show up as parse-failure rate, not as inflated argument error. The headline comparison runs under one uniform calling protocol with a single shared parser across all models. A native-versus-uniform ablation on a fixed subset quantifies the residual and enters the regression as a covariate. Each provider's tool-calling implementation and date is logged.

## 7. Metrics

- Clean baseline curve `p_t` and its slope with depth (H6).
- Global per-step curve `g_t`.
- Net propagation loss `L_t = 1 - g_t / p_t`, the fit-free headline.
- Posterior severity `pi`, syntactic recovery `r_syn`, semantic recovery `r_sem`, with credible intervals.
- Parse-failure rate per model and per calling mode.
- Error-type breakdown by depth (H4).
- End-to-end task success, as context only.

## 8. Analysis plan

Supply the measured `p_t`, `f_syn(t)`, and labelled recovery events as data, and fit `pi`, `r_syn`, `r_sem` with the hierarchical model of Section 3. Compare `pi`, `r_syn`, and `r_sem` across the scale axis (H2), across families at matched scale, and between FC-tuned and base models (H3). Test H5a and H5b by contrasting `r_syn` and `r_sem` between the structured-exception and opaque-failure conditions. Regress teacher-forced `p_t` on depth to quantify context degradation (H6). Fit a mixed-effects logistic model of raw per-call outcomes with fixed effects for depth, model, calling mode, feedback condition, and upstream state, and a task random effect, to confirm the depth and recovery effects survive controls. For H4, test whether the selection-to-argument error ratio declines with depth.

### Synthetic versus real, decided in advance

Fit `pi` and the recovery rates on both sets independently. Agreement means overlapping credible intervals and the same model ordering. On disagreement, the real set is primary for every headline claim, the synthetic set explains mechanism only, and the divergence is analysed rather than hidden. The real set is sized to carry this weight.

## 9. Implementation

Python throughout. LiteLLM for the multi-provider client. Pydantic for schemas and validation, with the two error-feedback modes built into the executor. PyMC and ArviZ for the hierarchical fit and diagnostics. NumPy, SciPy, statsmodels for the regression; Matplotlib for the `p_t` and `g_t` curves, the `L_t` bars, and the posterior plots.

Modules on the existing `tool-use-reliability` repository:

- `src/tasks/` graph generator with depth, distractor, semantic-noise, and error-feedback controls.
- `src/harness/` run loop across native and uniform calling modes, logging every call to JSONL.
- `src/eval/` selection and argument scorers, the syntactic-versus-semantic classifier, the parse-failure bucket, the teacher-forced `p_t` estimator, and the global scorer.
- `src/model/` the state recurrence, the PyMC hierarchical model, and identifiability diagnostics.
- `src/analysis/` regression and figures.
- `data/raw/` tasks; `data/results/` runs, out of version control and regenerable from a seed.

Reproducibility: greedy decoding for main runs, fixed seeds for generation and sampling, every response cached by prompt, calling mode, and model, and provider and date logged. One config file fixes the depth grid, distractor and noise levels, feedback condition, calling mode, task counts, and model list.

## 10. Feasibility, cost, and timeline

Cost stays bounded by open-weight hosting, a reduced proprietary subset, aggressive caching, and capped tasks per depth. The added load is the calling-mode ablation, the larger real set, and the MCMC fitting, all budgeted explicitly. Six weeks:

- Week 1: task generator (semantic noise, both feedback modes) and harness end to end on one model in both calling modes.
- Week 2: full suite wired; scoring pipeline including the error classifier and the teacher-forced `p_t` estimator.
- Week 3: run the synthetic depth, distractor, and feedback grid across models and seeds.
- Week 4: calling-mode ablation and the real-benchmark subset.
- Week 5: hierarchical fit, `pi` and recovery posteriors, identifiability diagnostics, regression, figures.
- Week 6: reconcile synthetic and real under the pre-registered rule, then write up.

## 11. Threats to validity

- Context-length degradation: separated from propagation by measuring a dynamic baseline `p_t` at each depth and defining `pi` as a penalty relative to it, so context growth can no longer inflate the severity estimate.
- Recovery confound: split into syntactic and semantic channels with separate rates, since only syntactic failures raise exceptions; the feedback manipulation is expected to move `r_syn` and leave `r_sem` low, and both are measured directly.
- Statistical identifiability: handled by a Bayesian hierarchical model with partial pooling and weakly informative priors, with explicit diagnostics; where `pi` and a recovery rate are not separable for a flatlined model, the identifiable `L_t` and floor are reported instead of an overconfident point estimate.
- Function-calling confound: parse failures separated from semantic errors, headline under a uniform protocol, residual measured by ablation and controlled as a covariate.
- External validity: the real subset is large enough to stand alone, with a pre-registered rule making it primary on disagreement.
- Model drift: provider and date pinned and logged, re-runs served from cache.
