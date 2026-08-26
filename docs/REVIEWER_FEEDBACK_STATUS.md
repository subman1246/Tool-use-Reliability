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

- **Controlled error injection** (review's Priority 1, the most informative single
  experiment per the review: inject a known error at position $j$, branch into
  corrupted-continuation vs.\ corrected-continuation from the identical prefix, compare
  downstream outcomes). No config or task-generator support exists for this yet. This
  directly addresses review point #3 (the baseline/free-running contrast conflates more
  than propagation) by establishing propagation causally rather than observationally.
- **Full linear-task null control** at the main suite's depths and sample sizes. Piloted
  only (`§3.3`: "Pilot runs were degenerate... the full null arm was not executed").
- **Posterior predictive check** for the hierarchical model (Part 2 above). This is a
  code addition to `src/tur/model/hierarchical.py` or a new analysis script calling
  `pm.sample_posterior_predictive()` against the existing `data/results/official_idata.pkl`
  trace, comparing replicated zero-count structure against the observed zero. No new
  model API calls needed — this reuses the already-fit posterior.

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
