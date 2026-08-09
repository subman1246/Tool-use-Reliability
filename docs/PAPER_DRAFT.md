# Invocation-Level Reliability of Tool Use in Language Model Agents

*Working draft. Sections marked **[PENDING REAL DATA]** contain the exact
structure to fill in once `scripts/run_real_suite.py` has been run against
real models; everything else is complete and ready for review/editing.*

---

## Abstract

**[PENDING REAL DATA — draft shape below, fill in after real results]**

Tool-augmented language model agents are increasingly deployed to act on real
systems, where a single incorrect function call can silently derail an entire
task. Existing benchmarks largely report a single end-to-end or aggregate
success score, which conflates two distinct failure channels — choosing the
wrong tool and forming wrong arguments — and obscures how errors made early in
a multi-step task propagate to corrupt everything downstream. We introduce a
correct-invocation rate that disaggregates tool selection from argument
correctness, measured both under a clean, teacher-forced context and under a
model's own free-running context, across [N] models spanning [scale/family/
tuning axes] on a fixed, contamination-free suite of multi-step tasks. We find
that [headline finding 1 — e.g., propagation loss grows with dependency depth
and separates models more sharply than single-step accuracy does], and that
[headline finding 2 — e.g., a parametric severity/recovery model is not fully
identifiable under exact-match scoring, while a simple fit-free propagation
metric reliably orders models by severity]. Our pipeline, task suite, and
analysis code are released to support reproducible evaluation of tool-use
reliability at the invocation level.

---

## 1. Introduction

Large language models are increasingly deployed as agents that call external
tools and APIs to act in the world, and the point at which they do so has
become a common source of failure. A model can reason correctly about what
needs to happen and still produce a call that fails, either by choosing the
wrong tool or by filling its arguments incorrectly. Benchmarks that report a
single aggregate success number do not say where a call went wrong, and they
typically reflect clean, single-shot conditions rather than the multi-step,
state-carrying tasks real agents are asked to perform.

This gap matters because the two failure modes have different causes and
different fixes. A model that reliably picks the right tool but frequently
mis-fills arguments needs different intervention than one that hallucinates
tools outright. And a model whose per-call accuracy looks acceptable in
isolation can still fail badly in a multi-step chain if an early mistake
corrupts everything that depends on it — a failure mode that a single
end-to-end score cannot localize.

We study tool-use reliability at the level of the individual invocation. We
define a call as correct when it both selects the right tool and supplies
valid, correct arguments, and we measure this rate two ways: once under a
clean, teacher-forced context that isolates how reliability itself degrades
as more context accumulates, and once under the model's own free-running
context, where an earlier mistake can poison everything after it. The gap
between these two measurements is a direct, interpretable signature of error
propagation, distinct from the ordinary decline in accuracy that comes from
longer context alone.

**Contributions.**

1. A disaggregated correct-invocation metric and an evaluation protocol
  (matched clean/free-running scoring, syntactic-vs-semantic error
  classification, and a routing-task design that lets *selection* errors, not
  only argument errors, propagate) for measuring tool-use reliability at the
  invocation level rather than only end-to-end.
2. A formal propagation model with an interpretable severity parameter and
  origin-specific recovery rates, together with a fit-free propagation-loss
  metric that remains informative when the parametric model is not fully
  identifiable.
3. An empirical demonstration — first on a fully-controlled simulated suite
  with known ground truth, then on real models **[PENDING REAL DATA]** —
  showing where the parametric model succeeds, where it is structurally
  limited by exact-match scoring, and which metric to trust in each regime.
4. A released, tested pipeline (task generation, harness, scoring,
  aggregation, Bayesian fitting, and figure generation) intended to make this
  kind of evaluation reproducible and extensible to new models and
  benchmarks.

---

## 2. Related Work

Large language models are increasingly deployed as agents that call external
tools and APIs to act in the world, and the point at which they do so has
become a common source of failure. A model can reason correctly about what
needs to happen and still produce a call that fails, either by choosing the
wrong tool or by filling its arguments incorrectly. This section surveys how
the field has studied that failure, what it has measured, and where the
measurement remains incomplete.

### 2.1 Foundations

The idea of teaching a model when and how to call an API goes back to
Toolformer (Schick et al., 2023), which learned in a self-supervised way which
tool to use, at what point, and with what arguments. ReAct (Yao et al., 2023)
established the reasoning-action-observation loop that most agent frameworks
still follow, and Reflexion (Shinn et al., 2023) added a self-critique step
that later reliability work would build on. Gorilla (Patil et al., 2023) was
among the first to measure API-call correctness at scale and to treat
hallucinated calls as a distinct error to be counted rather than an
occasional nuisance. Together these define the pipeline the rest of the
literature examines: pick a tool, then form its arguments.

### 2.2 Benchmarks and what they measure

Most large tool-use benchmarks evaluate whether a task ultimately succeeds
rather than whether each individual call was correct. ToolLLM (Qin et al.,
2024) introduced ToolBench, built on thousands of real APIs, but the
instability of those live endpoints made results hard to reproduce, which
StableToolBench (Guo et al., 2024) addressed by simulating the APIs and
caching responses. API-Bank (Li et al., 2023) scored calls by execution match
across a smaller, controlled set of tools. The Berkeley Function Calling
Leaderboard (Patil, Yan et al., 2025) has become the de facto standard,
combining abstract-syntax-tree matching with executable checks and expanding
over successive versions from single calls to parallel, multi-turn, and
multi-step settings. tau-bench (Yao et al., 2024) took a different route,
grading the final state of a simulated environment after an agent-user
dialogue and reporting reliability across repeated trials, where even strong
models succeed on fewer than half of tasks and behave inconsistently from run
to run.

The common thread is that these benchmarks report an aggregate number. That
number is useful for ranking models, but it does not say where a call went
wrong, and it usually reflects clean conditions in which the tools behave as
expected.

### 2.3 Localizing the failure

A smaller group of benchmarks breaks the call apart. RoTBench (Ye et al.,
2024) evaluates tool selection, parameter identification, and content filling
as separate stages, and does so under increasing levels of noise, showing
that accuracy that looks solid in clean conditions can fall sharply once the
environment is perturbed. MTU-Bench (Wang et al., 2025) reports
tool-selection and parameter accuracy directly and computes them without
relying on a language model as judge, which makes the scores cheaper and more
consistent. FuncBenchGen (Maekawa et al., 2025) generates contamination-free
multi-step tasks as dependency graphs and finds that capable models often
produce syntactically valid calls while carrying incorrect or stale argument
values from one step to the next, with performance degrading as the chain of
dependencies grows. This work is the closest in spirit to an invocation-level
view, though it still reports task success rather than a single disaggregated
measure of call correctness.

### 2.4 How the field categorizes errors

Across these studies the failures sort into two families:

- **Selection errors:** choosing the wrong tool, inventing a tool that does
  not exist, calling a tool when none was needed, or skipping a call that was
  required.
- **Argument errors:** malformed or unparseable output, missing required
  parameters, wrong types, wrong values, and fabricated arguments supplied
  instead of asking the user for the missing information.

Relign (Xu et al., 2025) formalizes this split into tool-selection and
tool-usage hallucination and adds the option for a model to defer or ask for
clarification rather than guess.

### 2.5 Error propagation in multi-step chains

Selection and argument errors are stateless: they describe a single call in
isolation. In a multi-step task, where the arguments of one call are drawn
from the outputs of earlier calls, a further structure appears that the two
families do not capture. A single mistake at step $t$ does not simply lower
the score for that step; it changes the inputs every downstream step
receives, so later calls can be locally well-formed and still globally wrong.
This is the behaviour FuncBenchGen (Maekawa et al., 2025) isolates when it
reports models carrying stale or incorrect values forward.

The distinction that makes this tractable is between two notions of a
correct invocation. A call is *locally correct* if, given the correct inputs,
the model selects the right tool and forms valid, correct arguments. A call
is *globally correct* if it is correct inside the model's own run, where its
inputs may already be poisoned by an upstream error. A model can be strong
locally and still lose accuracy across a chain if it cannot recover once the
context is corrupted.

### 2.6 Methods for improving reliability

Proposed fixes sit at different points in the pipeline. Reliability alignment
(Relign) trains models to abstain or clarify when they are unsure.
Data-centric approaches such as ToolACE (Liu et al., 2025) and Hammer (Lin et
al., 2024) improve function-calling accuracy through synthesized training
data and by masking function and parameter names so models generalize beyond
surface patterns rather than memorizing them. Constrained or schema-guided
decoding removes formatting errors by construction, but it does not correct a
well-formed call that carries the wrong value. Self-reflection and retry
loops attempt to catch errors after execution, and a more recent line of
work, including the paper anchoring this project (Healy et al., 2026),
detects bad tool selection from a model's internal representations at
inference time rather than trying to prevent it during training.

### 2.7 Metrics and the gap this study addresses

Evaluation across this literature uses exact match on the full call,
tool-selection accuracy, parameter-level accuracy, abstract-syntax-tree
matching, and success across repeated trials. A recurring observation is that
tool-selection accuracy alone is necessary but not sufficient: a model that
picks the right tool but passes a wrong argument scores well on selection and
still fails the user.

Three gaps follow from this review. First, headline benchmarks aggregate
their results and rarely localize where invocation fails, and the studies
that do disaggregate tend to cover few models or non-fixed task sets. Second,
cross-model comparisons that separate selection errors from argument errors
are uncommon, and they are reported on different tasks with different
metrics, so no clean and comparable picture exists. Third, most evaluation
reflects a happy path, while the work that stresses models with noise,
distractors, and dependency depth shows reliability dropping in ways a clean
score hides. This work addresses all three: a disaggregated,
depth-controlled, multi-model measurement with a fixed task suite and a
single comparable metric.

---

## 3. Method

### 3.1 Definitions

A task is a sequence of calls $c_1, \dots, c_n$, where the arguments of $c_t$
may depend on the outputs of earlier calls. Dependency depth is the length of
the longest such chain. We define:

- **Correct invocation:** a call that both selects the correct tool and
  supplies valid, correct arguments.
- **Local correctness ($p_t$):** the probability step $t$ is correct given a
  correct upstream history of length $t-1$, measured by teacher-forcing the
  clean history at its true length. Because this baseline is measured at
  every depth rather than assumed constant, it captures context-length
  degradation (a model's accuracy naturally declining as the prompt grows) on
  its own, separately from propagation.
- **Global correctness ($g_t$):** the probability step $t$ is correct inside
  the model's own free-running trajectory, where upstream inputs may already
  be poisoned.
- **Net propagation loss:** $L_t = 1 - g_t / p_t$, the fit-free, always
  computable signature of propagation: the fraction of reliability lost at
  depth $t$ that is not explained by the context-length baseline alone.

### 3.2 A propagation model with recovery

We additionally model propagation as a three-state Markov process over a
task's context: clean, poisoned-by-syntax, or poisoned-by-semantics, with
transition probabilities driven by the measured baseline $p_t$, the measured
syntactic error share $f_{\text{syn}}(t)$, and two free parameters — recovery
rates $r_{\text{syn}}$ and $r_{\text{sem}}$ specific to each corruption
origin, since syntactic failures are caught by the tool executor (which can
return a structured exception) while semantic failures execute silently and
have no natural signal to prompt recovery. A single severity parameter $\pi$
scales the correctness probability while poisoned relative to the clean
baseline: $\rho_t = (1-\pi) p_t$. Occupancies evolve as

$$
\begin{aligned}
c_{t+1} &= c_t p_t + s_t r_{\text{syn}} + m_t r_{\text{sem}} \\
s_{t+1} &= c_t (1-p_t) f_{\text{syn}}(t) + s_t (1 - r_{\text{syn}}) \\
m_{t+1} &= c_t (1-p_t) (1-f_{\text{syn}}(t)) + m_t (1 - r_{\text{sem}})
\end{aligned}
$$

with observed global correctness $g_t = p_t (1 - \pi \, x_t)$, where $x_t =
s_t + m_t$ is the poisoned occupancy. Setting $r_{\text{syn}} =
r_{\text{sem}} = 0$ recovers a simple absorbing-poison model; each is a
special case of the general model, which we present to make the reviewer-facing
progression of the framework explicit rather than only stating the final
form.

### 3.3 Estimation

We fit $\pi$, $r_{\text{syn}}$, $r_{\text{sem}}$ with a Bayesian hierarchical
model (PyMC, NUTS), pooling partially across model families and scales so
that a model which collapses to its floor at shallow depth borrows statistical
strength from related models rather than producing an unidentified point
estimate. We report standard MCMC diagnostics (R-hat, effective sample size,
divergence count) for every fit, and — critically — the posterior correlation
between $\pi$ and each recovery rate per model, which is the direct
diagnostic for whether severity and recovery are separably identified from
the data at hand (Section 5).

### 3.4 Task design

Two task sources are used, for internal control and external validity
respectively.

**Synthetic, contamination-free tasks.** Each task is a hidden dependency
graph over deterministic, executable functions; solving it means traversing
the graph and feeding each function the outputs of its predecessors. This
gives exact ground truth for every call and direct control of dependency
depth (we use depths $\{1, 2, 4, 6, 8\}$) and distractor density. Two task
variants are used:

- *Linear tasks*, where the tool order is announced up front, so only
  argument-value errors can propagate.
- *Routing tasks*, where the correct next tool is a deterministic function of
  the incoming value rather than pre-announced, so a wrong tool choice also
  changes what is correct downstream. This is necessary because scoring tool
  selection against a fixed gold trajectory is only valid when the trajectory
  does not depend on the (possibly corrupted) value the model is holding; on
  routing tasks we score selection *conditionally* — against the tool that is
  correct given the ref the model actually holds — while separately recording
  divergence from the gold trajectory, so that an agent applying its own rule
  correctly to a poisoned input is not misclassified as a selection failure.

**Real benchmark tasks. [PENDING REAL DATA]** A held-out multi-step subset of
[BFCL multi-turn/multi-step splits and/or tau-bench], scored under the
identical local/global protocol, to check whether findings from the
controlled synthetic suite generalize to real-world tool schemas and
argument types.

### 3.5 Evaluation protocol

For each task and model, we: (1) present tool schemas and the task; (2) score
selection (conditionally, per Section 3.4) and arguments strictly (exact match,
after coercing digit-only JSON strings to integers so that a provider's number
formatting is not scored as a value error). A soft score is computed but is
identical to the strict score on this suite, since every argument is a single
integer — see Section 7; (3) classify each error as syntactic (parse or schema
failure, caught by the executor) or semantic (executes but is wrong). The
executor can return either a structured exception or an opaque failure; only the
structured condition was run (Section 7); (4) measure $p_t$ via teacher-forcing;
(5) measure $g_t$ via the model's own free-running trajectory; (6) measure
recovery directly from the logs, separately by origin, as the rate at which a
context poisoned syntactically or semantically returns to an on-track chain on
the following step, where on-track means the value carried in equals the value
the gold trajectory expects. The within-step retry-recovery rate is a distinct
quantity and is reported separately rather than folded into $r_{syn}$. These
measured rates centre the recovery priors in the fit. Both the
teacher-forced and free-running conditions use the same within-step retry
budget, so that $p_t$ and $g_t$ are scored under identical rules and their
ratio is not an artifact of asymmetric retrying.

### 3.6 Model suite [PENDING REAL DATA]

The suite is designed to vary one factor at a time:

| Axis | Isolates | Models |
|---|---|---|
| Scale within a family | Effect of parameter count | [e.g., Qwen2.5-Instruct 7B/14B/32B/72B] |
| Family at matched scale | Effect of architecture/training | [e.g., Qwen2.5-7B vs Llama-3.1-8B vs ...] |
| Function-calling tuning | Effect of tool-specific fine-tuning | [FC-tuned model vs its general instruct base] |
| Proprietary reference | An absolute-scale ceiling | [e.g., GPT-class, Claude-class, reduced subset] |

Exact model identifiers, provider, and access date are pinned and logged per
run for reproducibility (Section 3.7).

### 3.7 Reproducibility

All decoding is greedy (temperature 0) for the primary measurement. Every
model response is cached by (model, calling mode, exact prompt), so repeated
analysis costs nothing. Task generation uses fixed seeds. All code, the exact
task suite, and analysis scripts are released; see Appendix A.

---

## 4. Pipeline Validation on a Controlled Simulated Suite

Before spending API budget on real models, we validated the full pipeline —
task generation, harness, scoring, aggregation, and the hierarchical fit — on
six simulated model policies with known, configured ground truth (severity,
recovery rates, and clean-baseline decay), run through the actual harness
rather than an analytic shortcut. This section reports that validation in
full, both because it is a legitimate methodological check that a reviewer
will ask for, and because it produced a substantive empirical finding
(Section 4.3) that directly shapes how we report results on real models.

### 4.1 Setup

Six policies across two families ("A", "B"), each at three severity tiers
(weak/mid/strong), with configured clean baseline, severity, syntactic error
share, and origin-specific recovery rates (full configuration in Appendix B).
Each policy ran through the harness at depths $\{1,2,4,6,8\}$, 400 tasks per
depth (200 $\times$ 2 seeds), on the **linear** task variant. (The routing
variant is exercised by the stress-test suite but not by this validation run;
an earlier draft of this section incorrectly claimed both. Section 4.4 explains
why the distinction matters.)

### 4.2 Hardening found during validation

Seven correctness issues were identified and fixed across two audit passes,
each locked in by a regression test:

1. A retry-budget asymmetry between teacher-forced and free runs, which made
  $p_t$ and $g_t$ not directly comparable.
2. No mechanism for selection errors to propagate (the original linear tasks
  pre-announce tool order); addressed by the routing task variant.
3. A stuck-poisoned state in the simulated policy with no recovery path once
  a syntax failure's retry budget was exhausted.
4. Selection errors mis-attributed under poisoning: on routing tasks, an
  agent applying its own routing rule correctly to an already-poisoned value
  was scored as a selection failure against the fixed gold trajectory. Fixed
  by conditional selection scoring (Section 3.4).
5. Invisible stalled chains: a step that never executed left the carried
  value stale, and downstream steps were indistinguishable from ordinary
  semantic corruption. Fixed with an explicit stalled-state flag.
6. The syntactic error share was estimated from *all* errors, including ones
  that were wrong purely because of upstream poisoning, inflating the
  semantic share. Fixed by restricting the estimator to fresh errors on clean
  contexts; validated against the known configuration (recovered $f_{\text{syn}}$
  closely tracked the configured syntax share once fixed, whereas it did not
  before).
7. $L_t$ initially had no uncertainty estimate; added task-level bootstrap
  confidence intervals.

### 4.3 Findings

**The fit-free metric $L_t$ correctly recovers the configured severity
ordering.** Averaged and at the deepest measured depth, $L_t$ orders
weak $>$ mid $>$ strong within both simulated families, and — after adding
bootstrap confidence intervals — the intervals do not overlap between tiers
within either family at the deepest depth. A model-free lag check
($\Delta_1 = P(\text{step } t{+}1 \text{ correct} \mid t \text{ correct}) -
P(\text{step } t{+}1 \text{ correct} \mid t \text{ wrong})$) is large and
positive for every model (0.55–0.70), independently confirming that
propagation is real and detectable without any parametric assumption.

**The parametric severity/recovery fit is not fully identifiable under
exact-match scoring.** MCMC diagnostics are healthy (max R-hat 1.002, min ESS
$\approx$ 4350, zero divergences), yet the fitted $\pi$ does not track the
configured severity ordering as cleanly as $L_t$ does. The posterior
correlation between $\pi$ and each recovery rate is substantial for every
model (0.18–0.74), which is the direct diagnostic signature of
non-identifiability. This is not an artifact of the measurement bugs found in
Section 4.2: the correlation structure persisted after every one of those
fixes, which rules out the simplest alternative explanation and supports a
structural account instead. The mechanism is that under exact-match scoring,
a step can only be judged globally correct while its context is poisoned if
it exactly recovers the true value — a merely locally-consistent continuation
of an already-wrong input essentially never coincidentally matches the fixed
gold trajectory — which collapses much of the distinction the model draws
between "how bad is poisoning" and "how fast do you recover," since both
manifest through the same observable event. A secondary contributor is likely
task design: the synthetic tasks use exact integer arithmetic, which has no
graded notion of "close but not quite correct" that soft matching could give
independent empirical content to $\pi$.

**Implication for reporting on real models.** We report $L_t$ as the primary,
always-trustworthy quantity throughout this paper. We report the fitted
$\pi$/recovery posteriors as secondary and diagnostic, always alongside their
identifiability correlation, following exactly the fallback the method was
designed with. Section 3.4's real-benchmark arguments (which, unlike our
integer chains, have genuine graded closeness) let us test directly whether
identifiability improves outside the synthetic setting **[PENDING REAL
DATA]**.

### 4.4 What this validation can and cannot establish

Validating on simulated policies has a boundary that we state explicitly,
because we crossed it and were caught by it.

The simulated policy is invoked as `policy(task, step, ref, attempt)`: it is
*handed* the reference value carried into the step as an argument, delivered
out-of-band rather than read from the conversation. A real model receives only
the message list. Those interfaces are not equivalent, and the difference is
precisely the dependency this study measures.

Because of that, the validation reported above could not detect that the
free-running loop was appending neither the model's own call nor the tool's
result to the conversation. Every simulated policy behaved identically whether
the history was complete or empty, since none of them read it. The validation
suite reported a healthy pipeline while the run loop could not have worked on
any real model; the defect surfaced in the first minute of real API traffic and
had survived 24 stress-test configurations.

We therefore scope this section's claim precisely: **it establishes that the
estimation and aggregation pipeline recovers configured parameters from labelled
transitions. It does not establish that the prompt-construction and
observation-passing path is correct**, and no simulation with an out-of-band
state interface can establish that. Two generalisable points follow: a mock that
receives state out-of-band cannot validate the channel that state travels
through, and tests should assert on the artifact sent to the provider rather than
on backend behaviour, since behavioural assertions inherit the mock's blind spots
by construction.

After the defect was fixed, this entire validation run was regenerated and
compared element-by-element with the pre-fix artifacts. The clean baseline,
syntactic share, success counts, trial counts, $\Delta_1$ and every bootstrap
$L_t$ interval were **identical**, for the structural reason above, so the
findings in Sections 4.2–4.3 stand as reported. The identifiability finding also
survives: posterior correlations moved from 0.18–0.74 to 0.15–0.76, which is
sampling noise across a change of environment, not a change in structure.

A separate caveat, not about the defect: this validation used the linear task
variant, and the real-model pilot subsequently established that linear tasks
carry no propagation signal on actual models ($p_t = g_t = 1.000$, $L_t = 0$),
because the tool order is announced and the argument is a verbatim copy of the
prior observation. The simulated policies still err on linear tasks by
construction, so the estimator check remains valid, but it was performed on a
task family whose real-model counterpart is degenerate. The real experiment's
primary suite is therefore the routing variant, in which a wrong value also sends
the agent down a wrong branch and selection errors can propagate.

---

## 5. Results [PENDING REAL DATA]

*This section is a template. Replace each subsection with the corresponding
output of `scripts/run_real_suite.py` followed by
`python -m tur.analysis.plots --tag real`.*

### 5.1 Per-depth clean baseline and global correctness

*Table: $p_t$, $g_t$ per model per depth (from `real_meta.json`). Figure 1
($p_t$ vs $g_t$, propagation gap).*

### 5.2 Net propagation loss

*Table: $L_t$ with bootstrap CIs, per model, averaged and at max depth.
Figure 2. State explicitly whether CIs separate model tiers, as they did in
the validation run.*

### 5.3 Model-free propagation check

*Table: $\Delta_1$ per model. Figure 7. Confirms propagation independent of
any parametric assumption.*

### 5.4 Severity and recovery

*Report fitted $\pi$, $r_{\text{syn}}$, $r_{\text{sem}}$ with credible
intervals (Figures 3a–c), the pi-vs-scale check for H2 (Figure 4), and the
identifiability diagnostic (posterior correlation) per model, exactly as in
Section 4.3. State plainly whether real-benchmark arguments (Section 3.4)
improved identifiability relative to the synthetic-only result, since that
was flagged as an open question.*

### 5.5 Error-type composition and depth (H4)

*Figure 5. Test whether the selection-to-argument error ratio shifts with
depth using real per-depth data (the simulated suite could not test this
since syntax share was configured flat).*

### 5.6 Function-calling mode ablation

*Native vs. uniform calling mode on a fixed subset, quantifying how much of
the argument-error rate is a calling-mode artifact rather than a genuine
model limitation. Report as a covariate-adjusted comparison per Methodology
Section 6.*

### 5.7 Synthetic vs. real agreement

*Per the pre-registered rule: report whether $L_t$ and the identifiability
finding from Section 4 replicate on real-benchmark tasks. If they disagree,
real data is primary and the divergence itself is analyzed (schema
complexity, argument realism, distractor density).*

---

## 6. Discussion

**[PENDING REAL DATA for model-specific claims; the framing below is ready.]**

The central methodological result of this work, independent of which real
models are measured, is that a fit-free propagation-loss metric and a
parametric severity/recovery model answer different questions and should be
reported together, not as substitutes for one another. $L_t$ answers "how
much reliability is lost to propagation, and does that ordering hold up
statistically" — a question exact-match scoring can answer cleanly. The
parametric model answers "why," in terms of an interpretable severity-versus-
recovery decomposition — a question that, on integer-chain synthetic tasks
under exact-match scoring, current evidence suggests is not fully separable
from the data alone. Whether this limitation is intrinsic to the *scoring
regime* (exact match) or the *task design* (no graded closeness) is
answerable directly from the real-benchmark comparison in Section 5.7,
because BFCL/tau-bench arguments (city names, dates, quantities) do have
graded closeness that our integer chains do not.

*[Once real results are in: discuss (a) which models show the largest
propagation gap and whether it tracks parameter count, family, or
function-calling tuning; (b) whether the calling-mode ablation reveals a
material fairness confound; (c) whether real-benchmark identifiability
differs from the synthetic result, and what that implies for future
tool-reliability evaluation design.]*

---

## 7. Limitations

- **Simulation-based validation cannot test channels the mock bypasses.** Three
  separate defects in this work were invisible to a validation suite that
  reported a healthy pipeline, and all three share one cause: the simulated
  policy received information through a channel the real interface does not
  have. It is called as `policy(task, step, ref, attempt)` and is *handed* the
  reference value it needs, whereas a real model is handed only a list of
  messages. The three: (i) the free-running loop appended neither the model's
  call nor the tool's result to the conversation, so a real model was asked for
  a value it had never seen, and measured $g_t$ was exactly $1/\text{depth}$;
  (ii) the linear task family is degenerate on real models ($p_t = g_t = 1$,
  $L_t = 0$) because the tool order is announced and the argument is a verbatim
  copy, which no simulated policy revealed because policies err by
  construction; (iii) the simulated clean baseline was contaminated by
  cross-run-mode state, driving $L_1$ to $-0.11$ where it is $0$ by
  construction. Two generalisable rules follow. A mock that receives state
  out-of-band cannot validate the channel that state is supposed to travel
  through. And tests must assert on the artifact sent to the provider rather
  than on the backend's behaviour, because behavioural assertions inherit the
  mock's blind spots by construction. We report this as a limitation of the
  methodology rather than as three incidents, because the pattern — not any
  individual bug — is what a replicator needs to guard against.
- **Withdrawn hypotheses and unimplemented analyses.** H5a and H5b concerned
  the error-feedback manipulation. Both feedback modes are implemented, but only
  the structured condition was run: exercising the manipulation requires
  repeating the full sweep, which the free-tier token budget cannot absorb. They
  are withdrawn rather than left listed as though tested. Likewise, an earlier
  plan specified a mixed-effects logistic model of per-call outcomes as a
  robustness check and as the vehicle for the calling-mode covariate; it was
  never built, so the calling-mode residual is reported as a direct contrast and
  the claim is removed. Release-date version pinning was also specified but is
  not possible: the providers expose no version field, and every model
  identifier in the original configuration had been retired by run time, so
  reproduction rests on the recorded identifier, the run date and the response
  cache.
- **Vacuous soft scoring on the synthetic suite.** Every synthetic argument is a
  single integer, so the soft (tolerance/normalised) score cannot differ from the
  strict score: a relative tolerance of $10^{-9}$ never separates two distinct
  integers, and there are no free-text or enum arguments for a normalised match
  to act on. The synthetic results therefore report the strict score alone. We
  did not add graded arguments to make the metric non-degenerate, since that
  would change the task design for the sake of a measurement; it is deferred to
  the real-benchmark stage, where graded closeness exists naturally and where it
  is also the direct test of whether $\pi$ becomes better identified.
- **Distractor pressure differs between task arms.** The linear generator adds
  never-correct distractor tools; the routing generator exposes only branch
  tools, all reachable on some branch, and silently ignores the distractor
  parameter. No distractor sweep was run. Cross-arm inferences that depend on
  distractor pressure are therefore not drawn. Since linear serves only as a
  null control, and a control carrying *more* distraction that still yields
  $L_t = 0$ is the conservative direction for that argument, we tolerate the
  asymmetry rather than remove it.
- **Exact-match identifiability.** Section 4.3 establishes, on a fully
  controlled suite where ground truth is known, that the parametric severity/
  recovery model is not cleanly identifiable under exact-match scoring. This
  should be treated as a property of the measurement regime unless and until
  the real-benchmark comparison (Section 5.7) shows otherwise.
- **Synthetic task realism.** The integer/modulo synthetic tasks give exact
  ground truth and contamination control, but have no graded notion of
  "almost correct," and their tool descriptions, while designed with
  paraphrase and distractor variation, remain more abstract than real-world
  API schemas. The real-benchmark comparison is the direct test of how much
  this matters.
- **Cost-bounded model coverage.** The model suite (Section 3.6) is chosen to
  isolate specific factors (scale, family, tuning) rather than to be
  exhaustive; conclusions about e.g. scale trends should be read as evidence
  from the sampled models, not as a claim about all models at that scale.
- **Provider and calling-mode heterogeneity.** Despite the uniform-protocol
  control and native-vs-uniform ablation (Section 5.6), some residual
  provider-specific effects on argument formatting may remain uncontrolled.
- **Recovery-model persistence assumption.** The three-state model assumes a
  poisoned context stays poisoned (of the same origin) until an explicit
  recovery event; because the synthetic tasks have full state labels, the
  rate at which real trajectories violate this assumption is measurable and
  should be reported as a check on the model's applicability.
- **Single-turn user simulation.** [If applicable given final real-benchmark
  choice: note whether tasks involve simulated multi-turn user interaction or
  only agent-tool interaction, and what that excludes.]

---

## 8. Conclusion

**[PENDING REAL DATA — draft shape below]**

We presented an invocation-level framework for measuring tool-use reliability
in language model agents, disaggregating tool selection from argument
correctness and separating context-length degradation from error propagation
via matched clean and free-running measurement. A fully validated pipeline,
tested first on a controlled simulated suite with known ground truth,
surfaced both real implementation bugs and a genuine, load-bearing
methodological finding about the identifiability of parametric propagation
models under exact-match scoring — a finding that survived every fix we made
and that we carry forward as an explicit reporting discipline for real
models. On [N] real models spanning [axes], we find [headline results]. We
release the full pipeline, task suite, and analysis code to support
reproducible, invocation-level evaluation of tool-use reliability going
forward.

---

## Appendix A: Reproducibility artifact

Code: `github.com/[org]/tool-use-reliability`. Key entry points:
`scripts/estimate_cost.py` (budget before running), `scripts/run_real_suite.py`
(the real experiment), `scripts/run_full_analysis.py` (the simulated
validation in Section 4), `tests/test_regression.py` (locks in the scoring
fixes from Section 4.2). Model list, task parameters, and feedback/calling
modes are set in `config/default.yaml`.

## Appendix B: Simulated-suite configuration

| model | p0 | pi | syntax_share | r_syn | r_sem |
|---|---|---|---|---|---|
| fam0-weak | 0.78 | 0.70 | 0.50 | 0.35 | 0.05 |
| fam0-mid | 0.88 | 0.45 | 0.50 | 0.55 | 0.15 |
| fam0-strong | 0.95 | 0.22 | 0.50 | 0.75 | 0.30 |
| fam1-weak | 0.75 | 0.65 | 0.35 | 0.30 | 0.05 |
| fam1-mid | 0.86 | 0.40 | 0.35 | 0.50 | 0.12 |
| fam1-strong | 0.94 | 0.18 | 0.35 | 0.70 | 0.25 |

`p0` is the clean baseline at depth 1; each policy's baseline declines
linearly with depth (`p_slope`, see `src/tur/harness/sim_policy.py`). All
other parameters are as defined in Section 3.2.

## References

Chen et al. (2025). ACEBench: A comprehensive evaluation of tool usage.
Findings of EMNLP.

Guo et al. (2024). StableToolBench: Towards stable large-scale benchmarking
of tool learning. Findings of ACL.

Healy et al. (2026). Internal representations as indicators of hallucinations
in agent tool selection.

Li et al. (2023). API-Bank: A comprehensive benchmark for tool-augmented
LLMs. EMNLP.

Lin et al. (2024). Hammer: Robust function-calling for on-device language
models via function masking.

Liu et al. (2025). ToolACE: Winning the points of LLM function calling. ICLR.

Maekawa et al. (2025). Towards reliable benchmarking: A contamination-free,
controllable evaluation framework for multi-step LLM function calling
(FuncBenchGen).

Patil et al. (2023). Gorilla: Large language model connected with massive
APIs.

Patil, Yan et al. (2025). The Berkeley Function Calling Leaderboard (BFCL):
From tool use to agentic evaluation. ICML.

Qin et al. (2024). ToolLLM: Facilitating large language models to master
16000+ real-world APIs. ICLR.

Schick et al. (2023). Toolformer: Language models can teach themselves to
use tools. NeurIPS.

Shinn et al. (2023). Reflexion: Language agents with verbal reinforcement
learning. NeurIPS.

Wang et al. (2025). MTU-Bench: A multi-granularity tool-use benchmark for
large language models. ICLR.

Xu et al. (2025). Reducing tool hallucination via reliability alignment.
ICML.

Yao et al. (2023). ReAct: Synergizing reasoning and acting in language
models. ICLR.

Yao et al. (2024). tau-bench: A benchmark for tool-agent-user interaction in
real-world domains. ICLR 2025.

Ye et al. (2024). RoTBench: A multi-level benchmark for evaluating the
robustness of large language models in tool learning. EMNLP.

**⚠ Citation verification still pending** (carried over from the literature
review): confirm every citation above — exact author lists, venues, and years
— before this reference list is submission-ready. Several are recent
preprints (2025–2026) that may not have final published forms yet.
