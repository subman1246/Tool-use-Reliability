# Invocation-Level Reliability of Tool Use in Language Model Agents

*Working draft. Sections marked **[PENDING REAL DATA]** contain the exact
structure to fill in once `scripts/run_real_suite.py` has been run against
real models; everything else is complete and ready for review/editing.*

---

## Abstract

**[INTERIM — numbers below are from a partial sweep; see §5.0 for exactly what is
and is not yet collected. Every figure quoted is measured, none is projected.]**

Tool-augmented language model agents are increasingly deployed to act on real
systems, where a single incorrect function call can silently derail an entire
task. Existing benchmarks largely report a single end-to-end or aggregate
success score, which conflates two distinct failure channels — choosing the
wrong tool and forming wrong arguments — and obscures how errors made early in
a multi-step task propagate to corrupt everything downstream. We introduce a
correct-invocation rate that disaggregates tool selection from argument
correctness, measured both under a clean, teacher-forced context and under a
model's own free-running context, across six open-weight models on a fixed,
contamination-free suite of multi-step tasks with dependency depths from 1 to 8.

Separating the two arms lets us split degradation into a context-length component
and a propagation component. On the 7–8B models the split is stark: the
teacher-forced baseline declines gently with depth (0.700 → 0.549) while the
free-running rate collapses (0.700 → 0.167), so that by depth 6 roughly **70% of a
model's own clean-context capability is lost to its own earlier mistakes**
(net propagation loss $L_t$ = 0.696 and 0.700, 89% CI widths under 0.20). We
also observe **no recovery at all**:
$P(\text{next call correct} \mid \text{current call wrong})$ is 0.000, and 0 of 284
poisoned steps return to an on-track trajectory. We argue this should be read as a
property of the *measurement*, not of the models. Returning to on-track under
exact-match scoring means emitting exactly the gold argument, which after divergence
is information the model has never received and cannot derive, so a model that
self-corrects often and one that never does produce identical data. The recovery
parameters are therefore **not identified** rather than measured as zero, the
three-state model reduces to its absorbing case as a consequence of the scoring
regime, and — since the simulated policies "recovered" only by reading gold values
out of band — the recovery channel was never estimable from observable data in any
version of this study. That names the cause of a non-identifiability we had
previously only documented.

Two negative results are reported rather than omitted. A hypothesised shift in
error composition along the chain is **untestable on this task family**, not for
lack of statistical power but because 302 of 302 clean-context errors are tool
*selection* errors and the argument channel is empty where the composition is
defined. And the larger models in the suite sit at ceiling on these tasks
($p_t \approx 1.000$), so scale contrasts remain unresolved. Our pipeline, task
suite, and analysis code are released, together with the measured provider rate
limits that constrain replication and are documented nowhere else.

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

**$L_t$ measures net propagation loss, and that is not the same thing as a
severity ranking.** Both facts belong here, next to the numbers, because
reporting either alone misleads. On the routing suite at depth 8, $L_t$ is
0.355 [0.322, 0.387] for the policy configured at $\pi = 0.70$, 0.197
[0.170, 0.226] at $\pi = 0.40$, 0.154 [0.128, 0.182] at $\pi = 0.45$, and
0.025 [0.010, 0.038] at $\pi = 0.22$ (89% paired-bootstrap intervals). Five of
the six pairwise comparisons separate with non-overlapping intervals, and every
pair whose configured $\pi$ differs by at least 0.18 separates. The sixth pair —
$\pi = 0.45$ against $\pi = 0.40$ — overlaps, and its point estimates are ordered
against the configured $\pi$.

This is expected rather than anomalous, and it is the single most important thing
to understand about the metric. $L_t = \pi \cdot x_t$ is a **product**: the
severity of poisoning multiplied by the poisoned mass accumulated by depth $t$.
Two models with similar $\pi$ but different error-mix dynamics have different
$x_t$, so their $L_t$ can order against their $\pi$. The linear suite reproduces
this independently across families, where the policy configured at $\pi = 0.65$
has a *higher* $L_8$ (0.383) than the one configured at $\pi = 0.70$ (0.330),
the two families differing in syntactic share and recovery rates.

So $L_t$ answers "how much reliability is actually lost along a chain of this
depth" — which is the operationally meaningful question, and which it answers
without any fitted parameter. It does not answer "which model is intrinsically
more fragile per unit of corruption." Where models differ mainly in severity,
the two coincide; where they differ in error composition, they need not. We
report $L_t$ as the headline for the first question and never read a ranking off
it for the second.

A model-free lag check
($\Delta_1 = P(\text{step } t{+}1 \text{ correct} \mid t \text{ correct}) -
P(\text{step } t{+}1 \text{ correct} \mid t \text{ wrong})$) is large and
positive for every policy (0.59–0.68 on routing), independently confirming that
propagation is real and detectable without any parametric assumption.

**The parametric severity/recovery fit is not fully identifiable under
exact-match scoring.** MCMC diagnostics are healthy (max R-hat 1.002, min ESS
$\approx$ 3400, zero divergences), yet the fitted $\pi$ does not recover the
configured values: absolute error is 0.064–0.313 on routing (mean 0.189) and
0.023–0.385 on linear (mean 0.167). The posterior correlation between $\pi$ and
at least one recovery rate is substantial for every model — 0.84–0.89 on routing,
0.22–0.74 on linear — which is the direct diagnostic signature of
non-identifiability.

This finding was re-examined specifically because it might have been an artifact.
Section 4.2's fixes made the recovery rates measurable from labelled transitions
for the first time, and those measured rates now centre the recovery hyperpriors;
the prior centres land within 0.02 of the configured family means. The natural
hypothesis was that correct, informative priors would resolve the
non-identifiability. **They did not.** Fixing the inputs did not change the
correlation structure, which is what a structural rather than
mis-specification account predicts, and it rules out the most attractive
alternative explanation. The mechanism is that under exact-match scoring,
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
state interface can establish that.

**A simulator validates the channels it exercises, and silently certifies the
ones it does not.** We state this as a design principle rather than as a list of
incidents, because it is the transferable contribution of this section and it is
predictive: it says in advance where to look. Nine defects were found in this
pipeline, and every one of them is an instance of it — a channel the real
interface has, which the simulator bypassed, and which therefore reported healthy
no matter what it did:

| Bypassed channel | What the simulator did instead | What it certified falsely |
|---|---|---|
| the conversation itself | received `ref` as a call argument | a free-running loop that appended neither its own call nor the tool result |
| task difficulty | erred by construction at a configured rate | a linear suite on which real models never err, so nothing propagates |
| run-mode separation | one policy object served both arms | a teacher-forced baseline poisoned by the free run, giving $L_1 = -0.113$ where it is 0 by construction |
| the tool namespace | emitted a tool name no task defines | a selection-error share of exactly 0.00 at every step index, making H4 unfalsifiable |
| branch selection | recovered the value but branched on the corrupted one | a configured recovery of 0.60 observable as 0.179 |
| the observable error label | kept a hidden corruption origin the logs cannot see | recovery rates biased ~2× toward each other, and hence biased priors |
| task identity | reused ids across generator seeds | task-grouped statistics merging unrelated trajectories |
| the definition of a clean context | compared the value carried IN against the expected ARGUMENT | identical under a copy-argument task, silently wrong under any transform: every clean step would be marked poisoned and $L_t$ made meaningless |
| recovery itself | recovered by reading the gold output out of band | a recovery channel that cannot exist for a real model, whose priors then informed a fit whose non-identifiability we attributed to statistics |

The last two are the sharpest illustrations of the principle, because neither is a
coding error. Comparing the carried value against the expected argument is *correct*
on a copy-argument task and only becomes wrong on a task that can distinguish them:
the copy variant was masking a definitional error, and it took building a task that
could tell them apart to expose it. And the simulator's recovery was not a bug at
all -- it did exactly what it was configured to do -- but it did so through a channel
no real model has, which means the recovery parameters were never estimable from
observable data in any version of this study. We had documented that
non-identifiability across three separate reworkings before identifying its cause.

Three of the nine were invisible to *every* assertion the suite made, because the
suite asserted on rates the simulator itself produced. Two practical rules
follow. First, tests should assert on the artifact sent to the provider — the
message list — rather than on backend behaviour, since behavioural assertions
inherit the mock's blind spots by construction. Second, any quantity the analysis
derives should be checked against a configured ground truth *through the full
pipeline*, not against the generator that produced it: four of the seven above
were found only once a measured value was compared with the value it was
configured from, and the last one only because the same configuration measured
0.34 at one seed count and 0.10 at another.

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

## 5. Results

Full tables, per-model achieved sample sizes, and the diagnostic checks behind
every claim here are in `docs/RESULTS_real_run.md`. This section states the
findings; that document shows the working.

### 5.0 What is collected, and what is not

The sweep is **incomplete** and the results below are read accordingly. Cost on
this task family is roughly quadratic in dependency depth — the tool schema grows
with depth and the whole transcript is re-sent at every step — and the binding
constraint is an undocumented per-model daily token allowance (§3.7). One day of
the eight-day window has been spent. Depths 1, 2 and 4 are complete for five of
six models; depth 6 is partial; **depth 8 has no data on any model**, and one
model (`gpt-oss-20b`) has none at all because pilot runs earlier the same day had
consumed 199,814 of its 200,000 daily tokens before the sweep began.

Since $L_t$ grows with depth, and since the larger models only begin to make
errors at depth, the missing bins are disproportionately the informative ones.
Nothing below should be read as final, and two of the hypotheses are explicitly
unresolved for this reason rather than answered weakly.

Models run **nested prefixes** of one task suite, sized to their individual token
allowances, so $n$ differs across models by design; cross-model comparisons are
made on the common prefix. §3.7 gives the argument and `tests/test_nesting.py`
asserts the property it depends on.

### 5.1 Per-depth clean baseline and global correctness

The two arms separate cleanly on the two small models. On
`llama-3.1-8b-instant`, the teacher-forced baseline $p_t$ falls from 0.700 at
depth 1 to 0.549 at depth 6 — a 22% relative decline, which is the
context-length effect the design exists to isolate — while the free-running rate
$g_t$ falls from 0.700 to 0.167, a 76% decline. `allam-2-7b` behaves the same way
(0.533 → 0.556 baseline, 0.533 → 0.167 free-running). Figure 1.

$L_1 = 0.000$ **exactly** on every model, not approximately. At depth 1 both arms
construct an identical prompt and share one cached response, so the
baseline-purity property holds by construction rather than within tolerance. It is
the cheapest available check that the two arms are wired correctly.

The clearest evidence is not the depth curve but the **step-index profile**, which
requires no model at all. On `llama-3.1-8b-instant` at depth 6, error counts per
step index on the *same tasks* were 21, 32, 35, 37, 40, 40 in the free arm against
21, 17, 15, 18, 20, 20 teacher-forced. The arms are identical at step 0, as they
must be since the context is the same there, then the free arm diverges upward and
stays up while the teacher-forced arm stays flat. A flat teacher-forced profile is
what rules out context growth as the explanation; 94 of 246 shared (task, step)
cells disagree between arms.

**The three larger models are at ceiling** ($p_t$ = 0.865–1.000 at every depth
reached), so $L_t = 0$ there is a statement about the tasks, not about robustness.
Their $p_t = g_t$ identity was flagged by an automated structural check and then
diagnosed rather than trusted: on `llama-3.3-70b-versatile` every one of its seven
errors falls at step 2 or 3 of a depth-4 chain, and step 3 is the *final* step,
where no downstream call exists to poison. Propagation is impossible there by
construction, not absent by failure, so the correct reading is **inconclusive**
pending the deeper bins.

### 5.2 Net propagation loss

| depth | `llama-3.1-8b` $L_t$ [89% CI] | `allam-2-7b` $L_t$ [89% CI] |
|---|---|---|
| 1 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 2 | 0.145 [0.082, 0.222] | 0.087 [0.022, 0.163] |
| 4 | 0.391 [0.311, 0.474] | 0.514 [0.429, 0.600] |
| 6 | **0.696** [0.598, 0.795] | **0.700** [0.619, 0.778] |

$L_t$ rises monotonically with depth on both models, and the intervals at adjacent
depths do not overlap — the depth trend is resolved, not merely suggested.
Figure 2.

The two models are **not** separated from each other: their intervals overlap at
depths 4 and 6. That is consistent with §4.3's finding that $L_t$ is a product
$\pi \cdot x_t$ and therefore not a severity ranking; the fitted $\pi$ values do
differ (0.91 against 0.68, §5.4), while the net losses coincide. This is exactly
the case the validation run predicted would arise, and it is reported here as a
property of the metric rather than as a failure to distinguish the models.

Against the ceiling-level models the separation is total: $L_6 \approx 0.70$
against $L_t = 0.000$. But that contrast conflates severity with task difficulty
and is not offered as a model comparison.

### 5.3 Model-free propagation check

$\Delta_1$ is positive for every model that makes any errors at all: +0.556
(`llama-3.1-8b-instant`), +0.322 (`allam-2-7b`), +0.880
(`llama-3.3-70b-versatile`), +1.000 (`gpt-oss-120b`). `qwen3.6-27b` produced no
errors, so $\Delta_1$ is undefined for it. Figure 7. Propagation is therefore
confirmed with no parametric assumption whatsoever.

But the check is weaker than it looks here, and §5.4 explains why: the second term
of $\Delta_1$ is **exactly zero for every model**, so $\Delta_1$ reduces to
$P(\text{ok} \mid \text{ok})$ and stops being an independent quantity. We report it
because a positive $\Delta_1$ was pre-registered as the model-free confirmation of
propagation and it delivers that; we do not treat its magnitude as an independent
severity measure, because on this task family it is not one.

### 5.4 Severity and recovery

**Recovery is not observable under exact-match scoring, and that is a scoping
statement about the method rather than a finding about the models.**
$P(\text{next call correct} \mid \text{current call wrong})$ is 0.000, and 0 of 284
poisoned-context steps with a following step returned to an on-track context.

We initially read this as "these models never recover". That reading is not
supported by the data, and the reason is decisive. Returning to on-track under
exact-match scoring requires emitting exactly the gold argument. The gold value at
step $t$ is the output of a tool whose constants are never exposed to the model, so
once a chain diverges the gold value is information the model has never received and
cannot derive. The only route back is coincidence, at $pprox 1/100{,}000$ per
opportunity, giving an expected 0.0028 coincidental returns over the 284 observed.
**A model that self-corrects 20% of the time and one that never self-corrects produce
identical observable data**, because the self-correcting model still cannot emit a
number it has never seen.

So $r_{\text{syn}}$ and $r_{\text{sem}}$ are **not identified** here; they are not
measured as zero. The three-state recurrence of §3.2 reduces to its absorbing special
case -- $x_t = 1 - \prod_{j<t} p_j$ and $L_t = \pi \cdot x_t$ exactly, with no
recovery term -- as a consequence of the *scoring regime*, not of model behaviour.

**This explains the identifiability failure across every version of this pipeline,
including the simulated ones.** The simulated policies did recover, but they
recovered by reading the gold output directly, i.e. by being handed the true value
out of band -- exactly the mock-interface channel §4.4 is about. The recovery channel
was therefore never estimable from observable data in any version of this study, and
the measured-transition priors of §4.3 were derived from transitions only a simulator
can produce. §4.3's finding that correct priors still could not resolve the
$\pi$/recovery confound is not a statistical accident; it follows from the scoring
regime, and we now name that as the cause.

Two consequences stand. First, $\Delta_1$ degenerates: its second term is
structurally zero, so it equals $P(\text{ok} \mid \text{ok})$ and confirms
propagation without being an independent severity measure. Second, the *apparent*
improvement in identifiability on real data -- largest $\pi$/recovery correlation
0.54 against 0.84--0.89 simulated -- must not be read as the fit working better.
There is no recovery variation left for $\pi$ to trade against, because recovery
cannot be observed at all. A parameter that cannot move is not identified; it is
pinned.

**What would make recovery estimable.** Scoring would have to credit a return to a
*self-consistent* trajectory rather than the *gold* one -- a model that detects the
inconsistency, restarts from the seed value stated in the prompt, and proceeds
consistently. That is a scoring-regime change and it is the prerequisite for the
recovery half of the state model to be estimable at all.

**Zero syntactic errors.** Every one of roughly 1,900 recorded invocations executed,
and no retry ever fired. A syntactic error is by definition a call that fails to
execute, so there were none to classify: $f_{\text{syn}} = 0$. The fitted
$r_{\text{syn}}$ values in Figure 3b are therefore prior, not posterior, and the
figure is labelled as such. This was flagged by the same automated check that catches
bucketing defects, and verified against raw rows before being accepted, because an
identical symptom in the simulated suite *was* a defect.

**Only two of five $\pi$ estimates are informative**: 0.91 [0.82, 0.98] for
`llama-3.1-8b-instant` and 0.68 [0.55, 0.82] for `allam-2-7b`. The three
ceiling-level models produce too few errors to update the prior, and their
posteriors span almost the entire unit interval; reporting 0.49 for
`gpt-oss-120b` would be reporting the prior mean back. MCMC diagnostics are
healthy throughout (max R-hat 1.0019, min ESS 4,044, zero divergences), which says
the posterior was explored properly and says nothing about whether the data
constrained it.

### 5.5 Error-type composition along the chain (H4): withdrawn

**H4 is untestable on this task family, and the reason is more informative than a
power limitation.** Of 302 clean-context errors, **302 are tool-selection errors
and 0 are argument errors**. The selection-to-argument ratio the hypothesis is
about is undefined at every step index: there is no mix, so there is no shift in
the mix to detect. This was verified directly against raw rows for both models
rather than inferred from the aggregate.

The mechanism is structural. On a routing task a clean-context failure means the
parity rule was mis-applied — the wrong tool was named. The argument, by contrast,
is copied verbatim from the previous result stated in the observation, and these
models essentially never get that wrong *while holding a correct value*. Argument
errors do occur — 145 of 289 poisoned-context errors name the gold tool while
carrying a wrong value, because the rule applied to a corrupted value can
coincidentally land on the right branch — but those are propagated errors, which
the composition deliberately excludes as not being fresh evidence about what kind
of error the model makes.

**Power was not the binding constraint, and we checked.** Projected against the
*measured* per-model accuracies, the allocation yields four contiguous usable step
bins, enough for a rank correlation to reach $p < 0.05$ at $\rho = -1$. (An earlier
projection of six bins assumed a uniform 0.50 accuracy across models; correcting it
for the three ceiling-level models, which contribute almost no errors however many
tasks they run, drops it to four.) H4 would have been marginally testable on power
grounds. It fails because one of its two categories does not occur where the
composition is defined.

**What would be needed.** A task variant whose arguments can be wrong
*independently of tool choice* — arguments requiring a transformation of the
carried value rather than a verbatim copy, or multi-argument calls where one field
can be wrong while the tool is right. That is a task-design change, not a budget
change, and we state it as the concrete prerequisite rather than as future work.

*The per-step aggregation machinery, and the finding that per-depth pooling
recovers only 26% of a configured composition span where per-step resolution
recovers 75%, are retained from the validation run (§4) and remain the right way
to test H4 on a task family where both categories occur.*

### 5.6 Function-calling mode ablation

*Native vs. uniform calling mode on a fixed subset, quantifying how much of
the argument-error rate is a calling-mode artifact rather than a genuine
model limitation. Report as a direct contrast per Methodology Section 6.*

**Scope note: this ablation covers five of the six models.**
`groq/allam-2-7b` rejects native tool calling outright ("tool calling is not
supported with this model"), verified before the ablation was run rather than
discovered during it. It is retained in the main uniform-mode sweep, where it is
the 7B leg of the family-at-matched-scale axis against `llama-3.1-8b-instant`;
dropping it would cost that axis entirely. Its absence from the calling-mode
comparison means the ablation's five models span the two scale-within-family
axes but not the smallest scale point.

### 5.7 Simulated vs. real: three divergences, real primary

The pre-registered rule makes real data primary on disagreement and treats the
divergence itself as the object of analysis. There are three, and they are
substantive rather than quantitative.

**1. Recovery. Simulated 0.05–0.30 by configuration; real exactly 0.** The
simulated policies were built to have the recovery structure the state model
assumes, so the validation exercised a channel that does not exist on real models
here. This is the clearest instance of the general lesson in §4.4 — a simulator
validates the channels it exercises — showing up in the *parameters* rather than in
the code.

**2. Identifiability, and it moved in the direction the validation did not
predict.** Largest $\pi$/recovery posterior correlation: 0.84–0.89 simulated,
**0.54 real**. §4.3 established that even correctly measured priors could not
resolve the confound on simulated data; on real data the confound is largely
absent, because consequence 1 removes the quantity $\pi$ was trading against. The
method's headline weakness is milder in the setting it was built for than in the
setting built to test it.

**3. Error composition. Simulated policies produced both error types by
construction; real models produce one.** At depth 4 the simulated $L_t$ spanned
[+0.042, +0.169] against a real span of [+0.000, +0.514] — the real suite is far
more dispersed, because it contains both near-ceiling and heavily-degrading models,
which no simulated tier was configured to be.

What replicates: the **shape** of the result. $L_t$ rises monotonically with depth,
$L_1 = 0$ holds exactly, the teacher-forced arm stays flat where the free arm
degrades, and $L_t$ fails to separate two models whose $\pi$ differ while
separating those whose $\pi$ differ widely — all four were established on simulated
data first and all four hold on real models. The estimator transferred; the
parameter regime did not.

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
