# Reviewer Feedback Status

This document reconciles the harsh external review received on 2026-08-24 against the
current state of `paper/main.tex` and this repository, and records what was verified,
what was fixed, and what genuinely still requires new work. Written 2026-08-26.

The review is not reproduced in full here; it is preserved in the conversation history
that produced this repo update. This document is the durable, checkable record of what
was actually done about it.

---

## Part 1: The 12 "fatal technical attacks" — reconciled against current `paper/main.tex`

| # | Attack | Status | Where |
|---|---|---|---|
| 1 | Estimand misdefined: $p_t$/$g_t$ defined as per-step but Table 3 reports pooled | **Fixed** | `§3.1 Definitions` now uses $\bar p_d$/$\bar g_d$ with explicit pooled-sum equation, and states directly: "$L_d$ is descriptive and invocation-pooled; it is not the stepwise quantity in the Markov recurrence below." |
| 2 | "Widens monotonically" contradicted by $L_6=0.686$ vs $L_8=0.680$ | **Fixed** | `§5.1` now says "growth followed by a plateau, not strict monotonicity or significant adjacent-depth differences" |
| 3 | Baseline vs. free-running conflates more than propagation (prompt distribution, retry feedback, etc.) | **Open — needs new experiments.** Disclosed as a named limitation ("material threats, not cosmetic extensions"), but only running the causal-injection ablation actually resolves it. See Part 3. | `§7 Limitations` |
| 4 | Joint scorer inconsistent; gold-agreement headlined despite being invalid | **Fixed** | Gold-agreement is now framed as a "scorer-imposed boundary" throughout; conditional-on-state is the interpretive result. Abstract leads with the construct-validity finding, not a gold-agreement number. |
| 5 | Hidden constants insufficient to prove unpredictability (constant reuse, non-coprime $a$, collisions) | **Fixed as a scoping fix, not a proof.** The claim is now stated as an explicit assumption ("Under the generator assumption that this target is conditionally uniform over $M=100{,}000$..."), with a "Counterexample and scope" paragraph naming exactly the cases where it wouldn't hold (reconstructible targets, reset/re-query, milestone/final-state evaluators). This does not empirically verify the assumption — see Part 3 for what would. | `§5.3`, "Counterexample and scope" paragraph |
| 6 | "Cannot be estimated" is the wrong diagnosis; should be construct validity, not identifiability | **Fixed, essentially verbatim.** Section is titled "When canonical gold agreement ceases to measure behavior" and states explicitly: "It does not say the operational statistic is numerically unidentified; rather, the statistic is pinned near a boundary that no longer represents the intended behavioral severity." | `§5.3` |
| 7 | MCMC section looks like a straw man; no priors/PPC shown | **Partially fixed, partially open.** Priors ARE fully specified in code (see Part 2 below) — the review's claim of "no prior specification" is factually wrong about the codebase, though the paper text still doesn't display them. Posterior predictive checks are genuinely not implemented anywhere in the repo. See Part 3. | `src/tur/model/hierarchical.py`; not yet in paper |
| 8 | Propagation result effectively supported by one model | **Fixed via honest framing**, not by adding data. Abstract now says "the two 7–8B models lose about 68% of baseline capability... stronger models mostly remain at ceiling," not "across five models." Per-model caveats throughout §5.2 ("neither supports a robustness claim," "the sample is thin"). | Abstract, `§5.2` |
| 9 | Argument errors untested (389/389 clean errors are selection errors) | **Disclosed, not resolved.** This is a real data gap requiring the transform_h4 ablation. See Part 3. | `§5.6`, `§7` |
| 10 | CIs unreproducible / may treat dependent calls as independent | **Already correctly implemented in code — the review's concern does not apply.** See Part 2 below for the exact verification. | `src/tur/analysis/aggregate.py::bootstrap_L_ci` |
| 11 | $L_1=0$ oversold as strong validation ("tautology") | **Fixed.** Now just "an implementation invariant," no longer called "the cheapest available check that the two arms are wired correctly." | `§5.1` |
| 12 | Recovery poorly operationalized (conflates several distinct notions) | **Fixed.** "Canonical reconvergence" is now used consistently as the one specific measured quantity, distinguished from the general behavioral construct "recovery" throughout. | `§5.3` |

---

## Part 2: Two things the review got wrong about the actual codebase

These were checked directly against the source, not just the paper text, because the
review's claims were checkable and consequential.

### The confidence intervals ARE task-clustered (review's #10 is incorrect)

`src/tur/analysis/aggregate.py::bootstrap_L_ci` groups records by `task_id` before
resampling (`by_task_map`), and every bootstrap draw resamples whole tasks with
replacement, not individual calls. It goes further than the minimum fix: when the two
arms share task ids (the standard case), it resamples **paired** — the same task-id
draw is used to read both arms, preserving the fact that every task was run under both
protocols. The docstring states the reasoning directly: "steps in the same chain are
not independent once propagation is in play." **No correction was needed here.**

### The hierarchical model DOES specify priors (review's #7 is partially incorrect)

`src/tur/model/hierarchical.py` uses a non-centered hierarchical parameterization with
explicit priors: `mu_pi ~ Normal(0, 1.5)`, `mu_rs`/`mu_rm ~ Normal(centre, 1.0–1.5)`,
`sd_pi`/`sd_rs`/`sd_rm ~ HalfNormal(1.0)`, all on the logit scale, with an optional
"informed" mode that centers the recovery hyperpriors on measured per-family transition
rates. This is a real, documented prior specification. What is genuinely missing is a
posterior predictive check — `sample_posterior_predictive` does not appear anywhere in
`src/` or `scripts/`. The review conflated these two things; only the second is a real gap.

---

## Part 3: What's genuinely still open (requires new work, not just text)

### Already built, just never run — the fast path

Three of the ablations the review calls for already have complete, tested configs and
cost estimates in this repo. They were never executed, not because they weren't
designed, but because of the six-day scheduling gap documented in `§7 Limitations` and
`docs/METHOD_NOTES_real_run.md`.

| Ablation | Config | Test | Addresses |
|---|---|---|---|
| Randomized presentation order | `config/shuffle_control.yaml` | `tests/test_shuffle_control.py` | Review's parity/order confound concern; whether allam-2-7b's anti-correlation is genuine or a presentation artifact |
| Transformed arguments | `config/transform_h4.yaml` | `tests/test_transform_variant.py` | Review's #9 (argument errors untested); whether conditional-severity estimates (0.149, 0.316) survive harder argument construction |
| Native vs. uniform calling mode | `config/ablation_native.yaml` | (native mode probe-verified, per config header) | Whether measured unreliability is partly a calling-interface artifact |

To run any of these: `python scripts/run_real_suite.py --config config/<name>.yaml --tag <name>`,
then `python scripts/report_real.py --tag <name>`. Each config file documents its own
cost estimate and design rationale in its header comments.

**Correction to earlier guidance:** a prior version of the plan for closing these gaps
drafted new task-generation logic for these three ablations from scratch. That was
redundant — the logic already exists, tested, and ready to run. The actual remaining
work is execution (API budget and wall-clock time), not design.

### Not yet built

- **Controlled error injection** (review's Priority 1) — **RUN, see results.**
  `run_injection_pair()` in `src/tur/harness/runner.py`, `config/error_injection.yaml`,
  `tests/test_error_injection.py`, `scripts/report_injection.py`. Both members of a pair
  are built from a byte-identical gold prefix and differ in exactly one substituted
  number, so treatment is assigned externally rather than by the model. `allam-2-7b`
  only — a single-model causal check, not suite-wide validation.

  Depth 6, positions $j\in\{1,3,5\}$, two corruption modes. Cap-stopped at **52 of 60
  tasks** (316 complete pairs, 632 records); nested prefix, used as such.

  **Recovering the data required a workaround.** The runner cap-stopped, then exited 2
  because it excludes single-depth data from the hierarchical fit — correct behaviour,
  since this arm is single-depth by design and has no depth trend to fit — but it exited
  before flushing its jsonl, so no records reached disk despite the log saying they had.
  All completed calls were in the response cache, so they were recovered by replaying
  the suite through a cache-only backend that raises on a miss
  (`scripts/replay_injection_from_cache.py`). Zero additional API calls. **This is a real
  runner defect: a cap-stop on a single-depth config loses every record it collected.**

  **Corrected-branch validity check — PASSES EXACTLY.** The injection suite draws the
  same `base_seed=1000` tasks as the routing run, so the clean branch is comparable to
  the teacher-forced baseline on literally the same tasks. On all 40 overlapping tasks:

  | position | corrected branch | baseline $p_j$ | diff |
  |---|---|---|---|
  | $j=1$ | 0.400 | 0.400 | +0.000 |
  | $j=3$ | 0.500 | 0.500 | +0.000 |
  | $j=5$ | 0.425 | 0.425 | +0.000 |

  Exact agreement at every position. The contrast is not contaminated by any harness
  difference between the two constructions.

  **Result 1 — the causal severity is much smaller than the observational one.**
  `parity_flip` (substituted value has opposite parity, so the correct tool changes):
  $\hat\pi = +0.108$ $[+0.006, +0.215]$ pooled, 65 pairs degrading against 48
  improving. The interval only barely excludes zero.

  **Result 2 — the observational estimate over-states severity by ~2.9x.** The routing
  arm's conditional severity for this model is $+0.316$, far outside the causal interval
  $[+0.006, +0.215]$. Expected in direction — in the free arm the model chooses its own
  corruption, so whatever made it err upstream persists downstream — but it means the
  paper's severity figures are confounded, not merely uncertain. **This is the most
  consequential finding of the phase.**

  **Result 3 — there is no argument-channel effect.** `parity_preserving` (correct tool
  unchanged, only the argument differs): $\hat\pi = -0.032$ $[-0.057, -0.013]$, with
  **zero** pairs degrading at every position and 5 improving. Corrupting a value without
  changing which tool is correct does not hurt this model. The interval excludes zero on
  the wrong side; at 5 discordant pairs this should not be over-read as a real
  improvement, but it is certainly not a degradation.

  This is why the main-text mechanism claim was withdrawn: line 297 and Appendix D
  briefly asserted that linear's residual loss WAS argument-value propagation, which our
  own injection arm does not support.

  **Result 4 — severity is NOT constant along the chain.** $j=1$: $+0.132$
  $[-0.038,+0.302]$; $j=3$: $+0.226$ $[+0.038,+0.415]$; $j=5$: $-0.038$
  $[-0.231,+0.154]$. Spread 0.265. Only $j=3$ excludes zero, and the effect at $j=5$ is
  negative. The step model assumes $\pi$ is constant along the chain; these data do not
  support that, and the non-monotone profile is not what a simple positional-decay story
  would predict either. Per-position intervals are wide at n=52.

  **Caveat on power.** The pooled flip interval barely excludes zero and one of three
  positions carries the effect, so this establishes that the observational estimate is
  biased upward more firmly than it establishes the causal severity's magnitude. The
  remaining 8 tasks would not change that.

  Data: `data/results/inject_groq_allam-2-7b.jsonl`, `inject_injection_summary.json`.

- **Full linear-task null control** at the main suite's depths and sample sizes —
  **RUN, see results.** `config/linear_null.yaml`, tag `linear`, `allam-2-7b` ONLY (both
  Llama models were retired from the Groq free tier on 2026-08-27, mid-project). Depths
  {1,2,4,6}; a depth-8 bin was specified and dropped, being the most expensive bin and the
  only depth with no allam routing counterpart to compare against. Cap-stopped at depth 6,
  achieving n = {1:60, 2:60, 4:65, 6:31} of a planned {1:60, 2:60, 4:65, 6:40}; the
  achieved counts are a nested prefix and are used as such.

  **Result: this is NOT a clean null.** $L_d$ is small but its 89% CI excludes zero at
  depths 4 and 6:

  | depth | routing $L_d$ (published) | linear $L_d$ (new) |
  |---|---|---|
  | 1 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
  | 2 | +0.087 [+0.023, +0.160] | +0.008 [+0.000, +0.026] |
  | 4 | +0.514 [+0.431, +0.596] | **+0.056 [+0.028, +0.086]** |
  | 6 | +0.684 [+0.624, +0.745] | **+0.156 [+0.089, +0.229]** |

  Same estimator and convention as the routing numbers (`bootstrap_L_ci`, paired,
  task-level resampling, alpha=0.11), so the two columns are directly comparable.

  **Interpretation.** The linear variant was described as a null, but it cannot be a null
  for propagation in general, and this result makes that concrete. Its arguments are
  verbatim copies of the previous observation, so a wrong value still carries forward —
  what it removes is only the SELECTION dependency, i.e. a wrong value changing which tool
  is correct next. So a nonzero $L_d$ here is mechanistically expected, and the arm should
  be described as isolating the selection-dependency contribution rather than as a null.

  What it does establish is a decisive separation: the routing and linear intervals do not
  overlap at depth 4 or depth 6, and routing's loss at depth 6 is ~4.4x linear's
  (0.684 vs 0.156). Propagation is therefore not an artifact of the harness, the scorer, or
  context length — all of which are shared between the two arms — but roughly three
  quarters of it at depth 6 is attributable to the selection dependency, not all of it.

  Data: `data/results/linear_groq_allam-2-7b.jsonl`, `linear_meta.json`, `linear_idata.pkl`.

  **Two caveats on this run.** (1) The hierarchical fit on the linear data is not usable:
  every one of the 69 free-arm errors is in the `semantic` bucket, so there are zero
  syntactic errors, `f_syn` is degenerate and the measured `r_syn` is `nan`. The runner
  flagged this and the fit was still written; do not read it. The same degeneracy is
  present in the published routing data (all errors `semantic` there too), which is a
  known property of `error_type` being a syntactic/semantic axis rather than a
  selection/argument one — it is not a new defect and no published analysis uses
  `error_type` as a selection/argument decomposition. (2) `delta_1` on linear reproduces
  the routing arm's boundary exactly: P(next correct | this one wrong) = 0.000 over 43
  transitions, so semantic recovery is again zero as a measurement, not an estimate.
- **Posterior predictive check** for the hierarchical model — **RUN, see results.**
  Implemented in `scripts/posterior_predictive_check.py`, which rebuilds the fitted model
  with one extra observed node and calls `pm.sample_posterior_predictive()` against the
  existing trace. No model API calls. Run on both the real fit and, as a control, the
  simulated one.

  **Result: 0 of 6,000 replicates reach the observed value on the real fit.** The model
  predicts 67.7 canonical matches on average among the 869 corrupted-context calls (89%
  interval [39, 98], minimum across all replicates 10); the observed count is 0. The
  standard check on the same fit — replicated total free-arm successes — *passes*
  comfortably at Bayesian p = 0.420, so the conventional PPC has no power against this
  failure mode. The simulated fit fails the same targeted check in the same direction
  (observed 1,648 of 9,624, replicate interval [2,899, 3,427]) while also passing the
  standard one at p = 0.282, which shows the over-prediction of poisoned-call success is a
  property of the fit rather than an artifact of the real data's boundary.

  Data: `data/results/real_ppc.json`, `data/results/official_ppc.json`.
  Figure: `paper/fig_ppc.pdf`, reproducible via `posterior_predictive()` in
  `paper/make_aggregate_figures.py`.

---

## Part 4: Citation verification

Two new citations were added to address the review's novelty concerns
(`§Related Work`). Both were verified against arXiv directly before inclusion:

- **AgentProcessBench** (`fan2026agentprocessbench`, arXiv:2603.14465) — confirmed real.
  Author list (Fan, Ye, Huo, Chen, Guo, Yang, Yang, Ye, Chen, Chen, Cong, Lin) matches
  the arXiv listing exactly. 1,000 trajectories, 8,509 human-labeled steps, explicit
  error-propagation rule, as cited.
- **FALAT** (`rafi2026falat`, arXiv:2606.00765) — confirmed real. Author list (Rafi,
  Ahasanuzzaman, Kim, Wang, Chen) matches exactly. Dependency-guided failure
  attribution, as cited.

Several other titles named in the review's novelty section (e.g. "AgentProp-Bench,"
"TRAJECT-Bench," "TrajDebug," "Causal Agent Replay," an unnamed "air-traffic-control
study") were **not** added to the bibliography. Their citation URLs in the review
pointed to generic pages rather than the specific papers described, which is a
signature of fabricated or unverifiable citations. They should not be added without
independent verification.

AppWorld (`trivedi2024appworld`) and ToolSandbox (`lu2024toolsandbox`), also newly
cited, are well-established papers (ACL 2024 and arXiv:2408.04682 respectively) and
were not re-verified beyond confirming the arXiv ID and venue against training
knowledge.

---

## Part 5: Bugs fixed in this pass (not scientific content)

- `main.tex` referenced `fig_severity_comparison.pdf` and `fig_metric_sensitivity.pdf`,
  neither of which existed in the submission bundle — a compile-breaking bug (or,
  depending on the LaTeX engine's error tolerance, silently rendered as broken image
  boxes). Both are now generated by `paper/make_aggregate_figures.py`, which already
  existed in the submission bundle but had never been run.
- `paper/make_aggregate_figures.py` was extended with a fifth function,
  `lt_all_models()`, generating `fig_Lt_all_models.pdf` (the five-model $L_d$-vs-depth
  comparison with CIs) in the same script, using the same color palette as
  `discrimination()`, so all five figures now come from one reproducible source instead
  of two.
- 16 em-dashes (`---` in LaTeX source, rendering as `—` in the compiled PDF) were
  restructured into commas, colons, or split sentences, per the project's no-em-dash
  style rule. Verified zero remain via direct grep.
- The abstract had a stray paragraph break; merged into one paragraph per the project's
  abstract style rule.
- Page count: main content (Introduction through Conclusion, including the full
  19-entry reference list) fits exactly within 9 pages. The appendix is a separate
  10th page. The Verify-Agents workshop's own stated page limit ("4 to 9 pages,
  excluding references and appendices") is satisfied as written; no content was cut
  to force this.

---

## Part 6: Status of `docs/PAPER_DRAFT.md`

This file is the original ~16,000-word markdown draft from an earlier stage of the
project. It predates the construct-validity reframing, the formal proposition in
`§5.3`, the corrected notation ($\bar p_d$/$\bar g_d$), and the current citation list.
**It has been marked superseded** (see header note added to the file) rather than
deleted or rewritten in place, since it has historical value as a record of the
project's earlier state. `paper/main.tex` is the current source of truth for the
paper's content.
