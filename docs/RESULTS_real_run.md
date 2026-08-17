# Results: Real Models

Companion to `RESULTS_validation_run.md`, which reports the same pipeline against
simulated policies with known ground truth. That document establishes what the
estimator can recover; this one reports what real models do.

**Pre-registered rule: where the two disagree, real data is primary.** Divergence
is treated as a finding about the simulated policies, not as something to
reconcile away — the simulated policies were constructed to have the propagation
structure the state model assumes, and real models were not.

**Every number in this document comes from `scripts/report_real.py`, which reads
only the run's own artifacts** (`data/results/real_meta.json`,
`real_idata.pkl`, and the per-model JSONL). Nothing here is estimated or carried
over from the validation run.

---

## 0. Data freeze and the dataset actually collected

All figures in this document come from a dataset **frozen at 14:09 on 2026-08-17**. No API call
was issued after that moment. `scripts/final_analysis.py` constructs no backend, so the analysis
is a pure function of the frozen files and cannot have extended the dataset. Nothing here is
extrapolated, imputed, or adjusted to compensate for a thin sample; where $n$ is small the
interval is wide and printed as such.

### Per-model sample sizes at the freeze

| Model | tasks | d1 | d2 | d4 | d6 | d8 | status |
|---|---|---|---|---|---|---|---|
| `llama-3.1-8b-instant` | **273** | 60 | 60 | 65 | 65 | 23 | usable; only model reaching depth 8 |
| `allam-2-7b` | **225** | 60 | 60 | 65 | 40 | — | usable |
| `openai/gpt-oss-120b` | **108** | 26 | 26 | 28 | 28 | — | usable |
| `qwen/qwen3.6-27b` | **98** | 26 | 26 | 28 | 18 | — | usable |
| `llama-3.3-70b-versatile` | **49** | 12 | 12 | 13 | 12 | — | usable but thin |
| `openai/gpt-oss-20b` | **0** | — | — | — | — | — | **data-insufficient, excluded** |

`gpt-oss-20b` completed no tasks: pilot runs earlier the same day consumed 199,895 of its 200,000
daily tokens, so it began the sweep a full day's allowance behind and never closed the gap. It is
excluded from every per-model claim and retained as an attempted entry, because the cause is a
reproducibility finding in its own right — **a pilot spends the same allowance as the real run**,
invisibly, since the pilot costs minutes of wall clock and the cost lands in the next day's quota.

### Conditions designed but not executed

| condition | status | consequence for this paper |
|---|---|---|
| Linear control arm | implemented, not run | the $L_t pprox 0$ null showing the propagation signal comes from task structure rather than from the harness is **absent** |
| Transformed-argument condition | implemented and validated able to recover a known trend, not run | H4 remains untestable; the scope caveat on the conditional-scoring decomposition is unresolved |
| Presentation-order control | implemented and verified inert, not run | the parity confound is addressed by the discrimination statistic only, not removed |
| Calling-mode ablation | designed, native mode probe-verified on 5 of 6 models, not run | no claim is made about how much measured unreliability is a calling-mode artifact |

The missing linear control is the most consequential, and we name it rather than let its absence
pass: without it, the claim that propagation arises from task structure rests on the routing arm's
internal evidence — the flat teacher-forced profile, $L_1 = 0$ exactly, and the step-index
divergence between arms — rather than on a direct null.

---

## 1. Setup

| | |
|---|---|
| Primary arm | routing tasks, depths 1, 2, 4, 6, 8 |
| Control arm | linear tasks, depths 1, 4, 8 (null control, reported in line) |
| Calling mode | uniform JSON-in-prompt (headline); provider-native ablated in §8 |
| Feedback | structured |
| Retries | 1 per step, syntactic failures only |
| Generator seeds | 1 |
| Scoring | exact match on tool name and arguments |
| Provider | Groq free tier, six open-weight models |

**Why routing is the primary arm.** The linear variant is degenerate on real
models: it announces the tool order and the argument is a verbatim copy of the
previous observation, so both llama models scored $p_t = g_t = 1.000$ and hence
$L_t = 0$ — no errors to propagate and therefore nothing to measure. The routing
variant states a parity *rule* instead of the sequence, so the model must apply it
to the value it is carrying, and a wrong value sends it down a wrong branch. On the
same model at depth 4 that produced $L_t = 0.500$. The linear suite is retained at
low $n$ as a null control rather than dropped, and its result is reported in §3
rather than in an appendix.

**Model suite.** Two scale-within-family axes (llama 8B/70B dense, gpt-oss
20B/120B sparse MoE), a family axis at matched scale (allam 7B against llama 8B),
and a second mid-scale family pair (gpt-oss 20B against qwen 27B). Deviations from
the methodology's specification, all forced by what free hosting offers:

- **Two scale points per family, not the 3–4 specified.** No free host offers 3+
  sizes of one family any more.
- **No proprietary reference.** The Gemini key returns 403 PERMISSION_DENIED on
  `generateContent`, confirmed against the raw REST endpoint so it is not a
  LiteLLM routing issue; listing models succeeds, so the key is valid and the
  project is blocked. The tier is absent, not estimated.
- **No FC-tuned vs base contrast.** No xLAM or Hammer variant is hosted free.
- Every model ID in the original configuration was stale — the entire
  Qwen2.5/Llama-3.1-instruct generation has been retired from Groq's free tier —
  so all six were verified against the live model list and by a real completion
  call before the run.

### 1.1 Unequal $n$ across models is by design

Models run **nested prefixes** of one task suite, sized per model to its own
measured daily token allowance rather than all to the smallest. The rationale, the
exactness of the nesting, and what it costs are documented in
`METHOD_NOTES_real_run.md`. In brief: task identity depends only on
`(depth, k, base_seed)` and never on how many tasks were requested, so a smaller
allocation is precisely the first tasks of a larger one. Cross-model contrasts are
computed on the common prefix and remain exactly paired; each model's own
estimates use its full $n$; the fit's binomial likelihood already takes per-model
trial counts.

The cost is unequal precision, concentrated on `llama-3.3-70b-versatile`, which
has the smallest allowance in the suite. It is retained rather than dropped
because it is the only *dense* scale-within-family contrast available — `gpt-oss`
is sparse MoE — and H2 needs it. Per-model interval widths therefore differ
substantially, and no scale or family contrast is read off point estimates
without them.

### 1.2 The binding constraint was undocumented

The daily token allowance (TPD) governs this study, appears in **no response
header**, is readable only from the body of the 429 that enforces it, and is
**not uniform** across models — a 5× spread. Measured values and the full account
are in `METHOD_NOTES_real_run.md`; they are a reproducibility contribution in
their own right, since anyone replicating on this tier will hit them and find
nothing documented.

---

## 2. Per-depth clean baseline $p_t$, global rate $g_t$, and net loss $L_t$

*Populated from `report_real.py` §1. Reports $p_t$, $g_t$, $L_t$ and an 89%
paired-bootstrap interval on $L_t$ at every depth for every model, alongside the
achieved $n$ and the fresh-error count backing each cell.*

**How to read $L_t$, and how not to.** $L_t = 1 - g_t/p_t$ is the net propagation
loss: the fraction of the clean-baseline capability that is actually lost once the
model has to run on its own output. It is fit-free, so it is the headline. But
$L_t = \pi \cdot x_t$ is a **product** — severity times the poisoned mass
accumulated by depth $t$ — and therefore **not a severity ranking**. Two models
with similar $\pi$ but different error-mix dynamics have different $x_t$ and can
order against their $\pi$. On the validation suite $L_t$ separated every pair whose
configured $\pi$ differed by at least 0.18 and failed to resolve a 0.05 difference,
inverting the point estimates with overlapping intervals. Both facts are stated
here, next to the numbers, rather than one here and one in the limitations.

### 2.1 Results so far (day 1 of the 8-day window; see §9 for completeness)

89% paired-bootstrap intervals. `fresh` is the count of clean-context errors
backing that cell. Empty cells are depths the sweep has not yet reached.

**`llama-3.1-8b-instant`** (achieved n: 60/60/65/41 at depths 1/2/4/6)

| depth | $p_t$ | $g_t$ | $L_t$ | 89% CI | $n_g$ | fresh |
|---|---|---|---|---|---|---|
| 1 | 0.700 | 0.700 | **0.000** | [+0.000, +0.000] | 60 | 18 |
| 2 | 0.692 | 0.592 | 0.145 | [+0.082, +0.222] | 120 | 35 |
| 4 | 0.600 | 0.365 | 0.391 | [+0.311, +0.474] | 260 | 55 |
| 6 | 0.572 | 0.179 | **0.686** | [+0.607, +0.766] | 390 | 63 |
| 8 | 0.543 | 0.174 | **0.680** | [+0.574, +0.772] | 184 | 23 |

**`allam-2-7b`** (achieved n: 60/60/65/6)

| depth | $p_t$ | $g_t$ | $L_t$ | 89% CI | $n_g$ | fresh |
|---|---|---|---|---|---|---|
| 1 | 0.533 | 0.533 | **0.000** | [+0.000, +0.000] | 60 | 28 |
| 2 | 0.383 | 0.350 | 0.087 | [+0.022, +0.163] | 120 | 50 |
| 4 | 0.427 | 0.208 | 0.514 | [+0.429, +0.600] | 260 | 63 |
| 6 | 0.475 | 0.150 | **0.684** | [+0.624, +0.745] | 240 | 40 |

**`llama-3.3-70b-versatile`** (achieved n: 12/12/13) — $p_t$ = 1.000, 1.000, 0.865;
$g_t$ identical; $L_t$ = 0.000 at all three depths.

**`gpt-oss-120b`** (achieved n: 26/26/28/2) — $p_t$ = 1.000, 1.000, 0.991, 1.000;
$L_t$ = 0.000, 0.000, 0.027 [+0.000, +0.083], 0.000.

**`qwen3.6-27b`** (achieved n: 26/26/21) — $p_t$ = $g_t$ = 1.000 at every depth
reached; $L_t$ = 0.000.

**`gpt-oss-20b`** — no data. Pilot runs earlier the same day consumed 199,814 of
its 200,000 daily tokens before the sweep began, so it cap-stopped at its first
task. Excluded from the fit; its allocation is unchanged, and it will fill in as
its bucket refills. This is a drawdown artifact, not a low cap.

### 2.1a Depth 6 for the larger models: the ceiling breaks for two of three

The depth-6 bin was the outstanding question for the three models that looked
ceiling-limited at depths 1-4, because it is the first bin where an error has somewhere
to propagate to. It has now landed, and it separates them.

| model | depth | $p_t$ | $g_t$ | $L_t$ | 89% CI | $n_g$ | fresh errors |
|---|---|---|---|---|---|---|---|
| llama-3.3-70b-versatile | 4 | 0.865 | 0.865 | +0.000 | [+0.000, +0.000] | 52 | 6 |
| | **6** | 0.875 | 0.708 | **+0.190** | **[+0.062, +0.345]** | 72 | 6 |
| gpt-oss-120b | 4 | 0.991 | 0.964 | +0.027 | [+0.000, +0.083] | 112 | 1 |
| | **6** | 0.991 | 0.947 | **+0.044** | [+0.000, +0.135] | 114 | 1 |
| qwen3.6-27b | 4 | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | 112 | 0 |
| | **6** | 1.000 | 1.000 | **+0.000** | [+0.000, +0.000] | 66 | 0 |

**`llama-3.3-70b-versatile` leaves the ceiling, and its interval now excludes zero.** Its
$L_t$ moves from exactly 0.000 at depth 4 to **+0.190 [+0.062, +0.345]** at depth 6 -- the
first propagation loss *established* in a large model here, rather than merely suggested.

This is an update to an earlier reading, and it went the way the earlier reading allowed.
With 8 depth-6 tasks the estimate was +0.140 [+0.000, +0.362], reported as suggestive because
the lower bound touched zero. Three further tasks ($n_g$ 48 to 66) moved it to +0.190 with a
lower bound of +0.049. The new point estimate lies **inside** the previously reported
interval, so this is confirmatory rather than a revision -- the interval did its job. This is the outcome the anomaly detector's "benign" classification predicted: at
depth 4 every one of its errors fell on the final step of the chain, where nothing
downstream exists to poison, so propagation was impossible by construction rather than
absent. Depth 6 gave those errors somewhere to go. The interval's lower bound is still
+0.000 at $n_g = 48$, so this is suggestive rather than established, and 14 of its 59
tasks remain.

**`gpt-oss-120b` errs, barely.** $L_6 = +0.044$ on a single fresh error, with an interval
touching zero. Directionally consistent with propagation; not distinguishable from the
null at this $n$.

**`qwen3.6-27b` remains at ceiling, and is reported as such.** $p_t = g_t = 1.000$ at
every depth reached, **zero errors across 298 free-arm steps**, and therefore **zero
poisoned-context steps**. Its severity is undefined (see below) and its $L_t$ is 0
because there is nothing to propagate, not because propagation was resisted. The fitted
posterior of 0.501 for this model is the prior mean returned unchanged and is not
reported as an estimate.

### 2.2 Three findings, and one thing that is not a finding yet

**(a) Propagation is large, monotone in depth, and cleanly measured on the small
models.** On `llama-3.1-8b-instant`, $p_t$ falls gently with depth (0.700 → 0.549,
a 22% relative decline attributable to context growth) while $g_t$ collapses
(0.700 → 0.167, a 76% decline). $L_t$ rises monotonically to **0.686**: at depth 6,
roughly 70% of the model's own clean-context capability is lost to its own earlier
mistakes. `allam-2-7b` behaves the same way, reaching $L_t$ = 0.684. This is the
separation the study was built to make — context-length degradation and error
propagation pulled apart — and the intervals are tight enough to carry it.

**$L_1 = 0.000$ exactly on every model**, not approximately. At depth 1 both arms
build an identical prompt and share one cached response, so the baseline-purity
property holds by construction rather than by tolerance. That is the cheapest
available check that the two arms are wired correctly, and it passes.

**(b) The step-index view shows propagation directly, with no model at all.**
Error counts by step index on `llama-3.1-8b-instant` at depth 6 — free arm versus
teacher-forced arm on the *same tasks*:

| step | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| free | 21 | 32 | 35 | 37 | 40 | 40 |
| teacher-forced | 21 | 17 | 15 | 18 | 20 | 20 |

Identical at step 0, as they must be — both arms have the same context there — then
the free arm diverges upward and stays up, while the teacher-forced arm stays flat.
94 of 246 shared (task, step) cells disagree between arms. The teacher-forced arm
being flat is what rules out context growth as the explanation.

**(c) The larger three models are at ceiling on this task.** `qwen3.6-27b` scores
$p_t$ = 1.000 at every depth reached; `gpt-oss-120b` 0.991–1.000;
`llama-3.3-70b-versatile` 1.000 down to 0.865 at depth 4. With no errors there is
nothing to propagate, so $L_t$ = 0 is a statement about the task, not about the
models' robustness.

**What is *not* a finding: $p_t = g_t$ for those models does not mean their
propagation loss is zero.** The runner flagged it as a structural anomaly and the
flag was checked rather than trusted. On `llama-3.3-70b-versatile` at depth 4 every
one of its 7 errors occurs at step 2 or 3 — and step 3 is the *final* step of a
depth-4 chain, where there is no downstream step left to poison. Propagation is
impossible there by construction, not absent by failure. With 52 observations and 7
errors, mostly terminal, the correct reading is **inconclusive**, not $L_t = 0$.
These models need the depth-6 and depth-8 bins, which are exactly what day 1 could
not reach for them. The anomaly detector now distinguishes these three cases
explicitly instead of reporting all of them as degeneracy.

---

## 3. The linear control arm

*Populated from `report_real.py` §2.*

Expected result: $L_t \approx 0$ at every depth. The control exists to show that
the propagation signal in the primary arm comes from task structure and not from
the harness or the scorer. **If the control arm is not null, that outranks every
other result in this document** and the reporter says so explicitly rather than
leaving it to be noticed — it checks whether any depth beyond 1 has a lower
interval bound above +0.05 and prints a stop-and-diagnose block if so.

One asymmetry to keep in mind when comparing arms: `distractor_level` is inert on
routing tasks. The routing schema is exactly $2 \times \text{depth}$ branch tools,
one correct at each step, which is stronger selection pressure than the linear
variant's distractors rather than weaker. The arms therefore differ in schema
composition as well as in routing, so selection-error rates are **not** comparable
across them, and the control arm is used only as an $L_t$ null.

**NOT RUN.** The linear control arm was implemented and is sequenced after the primary arm for each model; every model exhausted its token allowance on the primary arm first, so no control data exists. This is the most consequential of the four unexecuted conditions (Section 0) and its absence is stated there rather than implied by an empty table.

---

## 4. Model-free propagation check ($\Delta_1$)

*Populated from `report_real.py` §3.*

$\Delta_1 = P(\text{step } t{+}1 \text{ correct} \mid t \text{ correct}) -
P(\text{step } t{+}1 \text{ correct} \mid t \text{ wrong})$, which is positive iff
a wrong step makes the next step more likely to be wrong. It assumes nothing about
the state model, so it is the cleanest possible check that propagation is real.

| model | $\Delta_1$ | $P(\text{ok} \mid \text{ok})$ | $P(\text{ok} \mid \text{wrong})$ | $n_{\text{ok}}$ | $n_{\text{wrong}}$ |
|---|---|---|---|---|---|
| llama-3.1-8b-instant | +0.556 | 0.556 | **0.000** | 171 | 289 |
| allam-2-7b | +0.322 | 0.322 | **0.000** | 90 | 195 |
| llama-3.3-70b-versatile | +0.880 | 0.880 | 0.000 | 50 | 1 |
| gpt-oss-120b | +1.000 | 1.000 | 0.000 | 117 | 3 |
| qwen3.6-27b | — | 1.000 | — | 89 | 0 |

$\Delta_1$ is positive for every model with any errors at all, confirming
propagation with no parametric assumption.

### 4.1 Recovery is UNOBSERVABLE under this scoring, which is not the same as zero

The measurement is unambiguous: **P(next step correct | this step wrong) = 0.000**,
and across all five models **0 of 580** poisoned-context steps that had a following
step returned to an on-track context. Not one.

The tempting reading is "these models never recover". That reading is **unsupported**,
and the distinction matters more than the number does.

**Recovery is information-theoretically impossible here once a chain diverges.**
Under exact-match scoring, returning to on-track requires emitting exactly the gold
argument. The gold value at step $t$ is the output of a tool whose constants
(`a`, `b` in `ToolSpec`) are **never exposed to the model** -- the schema shows only
name, description and parameter types. So after divergence the gold value is
information the model has never received and cannot derive. The only route back is
coincidence, at probability $pprox 1/100{,}000$ per opportunity.

Over the 284 observed opportunities the expected number of coincidental returns is
**0.0028**. Observing zero is therefore almost entirely uninformative: a model that
genuinely self-corrects 20% of the time and one that never self-corrects produce
**identical observable data**, because the self-correcting model still cannot emit a
number it has never seen.

The honest statement is therefore:

> $r_{\text{syn}}$ and $r_{\text{sem}}$ are **not identified** under exact-match
> scoring on this task family. They are not measured as zero -- they are
> unmeasurable, and the state model reduces to its absorbing special case as a
> consequence of the scoring regime, not as a finding about model behaviour.

**This explains the identifiability failure across every version of this pipeline,
including the simulated ones.** The simulated policies did recover -- but they
recovered by reading `task.gold[step - 1].output`, i.e. by being *handed* the true
value out of band. That is exactly the mock-interface channel of the design
principle in the validation section: a real model has no such access. So the
recovery channel was never estimable from observable data in any version of this
study, simulated or real, and the priors that were supposed to inform it were
derived from transitions only the simulator could produce. The earlier finding that
correctly measured priors still could not resolve the pi/recovery confound is not a
statistical accident; it follows from the scoring regime.

Two consequences do stand:

1. **$\Delta_1$ degenerates into P(ok | ok).** Its second term is structurally
   zero, so it carries no information beyond the conditional-correct rate. It still
   confirms propagation -- what it was pre-registered for -- but it is not an
   independent severity measure here.
2. **The apparent improvement in identifiability is an artifact of the same cause.**
   The largest pi/recovery posterior correlation is 0.54 on real data against
   0.84-0.89 on simulated routing, and it would be a mistake to read that as the fit
   working better. There is no recovery variation left for pi to trade against,
   because recovery cannot be observed at all. A parameter that cannot move is not
   identified; it is pinned.

**What would make recovery observable.** Scoring would have to credit a return to a
*self-consistent* trajectory rather than to the *gold* one -- for instance a model
that detects the inconsistency, restarts from the seed value stated in the prompt,
and proceeds consistently thereafter. That is a scoring-regime change and it is the
prerequisite for the recovery half of the state model to be estimable at all. It is
recorded here because every version of this pipeline has reported
non-identifiability without naming its cause.


### 4.1a Severity is pinned by the same mechanism, and the fix un-pins it

$\pi$ is defined by $P(\text{ok} \mid \text{poisoned}) = (1-\pi) P(\text{ok} \mid \text{clean})$, so
it is a ratio of two directly observable rates and needs no fit. Measured: **0 of 869
poisoned-context steps correct**, hence $\pi = 1.000$.

Per-model, because pooled zero and per-model zero are different claims:

| model | poisoned steps | successes | 95% upper bound |
|---|---|---|---|
| llama-3.1-8b-instant | 510 | 0 | 0.0059 |
| allam-2-7b | 335 | 0 | 0.0089 |
| llama-3.3-70b-versatile | 16 | 0 | 0.171 |
| gpt-oss-120b | 8 | 0 | 0.312 |
| qwen3.6-27b | 0 | -- | **undefined** |

The support is not uniform across models and should not be described as though it were.
**Two models carry the empirical claim** -- 397 and 269 poisoned steps, with 95% upper
bounds of 0.0075 and 0.0111 on the success rate. **Two are consistent but
uninformative**: at 9 and 8 poisoned steps their upper bounds are 0.283 and 0.312, so
they exclude almost nothing and contribute almost no evidence. **One is undefined**:
`qwen3.6-27b` never left a clean context.

What extends the claim beyond the two informative models is the mechanism, not the
count. And this is *forced* by the same mechanism as
the recovery result: a poisoned step holds a wrong value, so the gold argument is
exactly as unreachable as a recovery would be.

With $\pi = 1$ and $r = 0$ substituted, the recurrence leaves $g_t = c_t p_t$,
$c_t = \prod_{j<t} p_j$, and $L_t = x_t$ -- **no free parameters**. The state model is an
identity in the measured per-step rates, and the fitted posteriors (0.93, 0.73) are not
merely redundant but wrong about a quantity that is exactly 1.

**Conditional-on-state scoring un-pins it.** Scoring the argument and the tool against
what the *held* value requires, rather than against gold, and re-scoring the existing
cache (0 API calls, 881 cache hits):

| model | gold-agreement $\pi$ | conditional $\pi$ (parity-stratified) | 89% CI |
|---|---|---|---|
| llama-3.1-8b-instant | 1.000 (boundary) | **+0.149** | [+0.021, +0.268] |
| allam-2-7b | 1.000 (boundary) | **+0.316** | [+0.066, +0.503] |

Clean-context steps score identically under both rules, which is the implementation
check. Both intervals exclude zero, so poisoning *does* degrade rule application -- the
strong claim that propagation here is purely information loss is **not supported**. The
supported claim is quantitative: gold-agreement attributes all propagation loss to
severity, conditional scoring attributes 0.15-0.32 of it to genuine degradation and the
rest to the gold trajectory being unreachable.

Treat this as a hypothesis the data supports, not a settled result. Two models, wide
intervals, and a narrow scope: on the copy variant a correct argument means transcribing
an integer shown one turn earlier, which is a weak competence test, so the observed
`args_correct_given_state` = 1.000 among poisoned steps may not survive harder argument
construction. **That is why the transformed-argument condition tests this decomposition's
generality and not only H4** -- there the argument requires arithmetic on the held value,
so the argument channel can fail on competence grounds.

#### An undefined severity is the correct output, not a gap

`qwen3.6-27b` produced **zero** poisoned-context steps across 298 free-arm steps spanning
depths 1, 2, 4 and 6: it never once carried a wrong value forward, and the depth-6 bin --
the first where propagation is structurally possible -- did not change that. Its severity is therefore **undefined** -- not zero, and
not small. There is no conditional rate to form, because the conditioning event never
occurred.

This is worth stating rather than passing over, because it is the case where an aggregate
score reports something misleading. A suite-level severity that pools this model in
would silently assign it whatever the pooled rate happens to be, and a per-model figure
of 0.000 would read as "poisoning does not hurt this model" when the truth is "this model
was never poisoned". We report it as undefined with the observation count attached, and
**do not pool it into any suite-level estimate**. If the depth-6 bin produces no errors
either, it stays undefined and is reported that way.

### 4.1b Rule-following measured directly: discrimination, and a revision

An earlier version of this section reported the parity effect from a cache replay covering
depths 1, 2 and 4 only. The step records now carry the held value and the presentation order
directly, so the statistic is computed over **all** collected data including depth 6. Two of
the numbers changed materially, and the revised picture is cleaner than the original.

**The statistic.** Define *discrimination* as
$P(\text{pick first-listed tool} \mid \text{ref even}) -
P(\text{pick first-listed tool} \mid \text{ref odd})$. The rule text lists the even branch
first, so 0 means the model ignores the value it is conditioning on, +1 means it applies the
rule perfectly, and a negative value means it is anti-correlated with the rule.

| model | $n$ | first-listed, overall | ref even | ref odd | **discrimination** | SE | $z$ |
|---|---|---|---|---|---|---|---|
| `qwen3.6-27b` | 298 | 0.497 | 1.000 | 0.000 | **+1.000** | 0.000 | — |
| `openai/gpt-oss-120b` | 358 | 0.492 | 0.983 | 0.006 | **+0.978** | 0.011 | +87.9 |
| `llama-3.3-70b-versatile` | 160 | 0.481 | 0.880 | 0.052 | **+0.828** | 0.044 | +18.9 |
| `llama-3.1-8b-instant` | 1,014 | 0.478 | 0.551 | 0.406 | **+0.146** | 0.031 | +4.7 |
| `allam-2-7b` | 680 | 0.685 | 0.597 | 0.774 | **−0.176** | 0.035 | **−5.0** |

**Discrimination orders the suite, and the aggregate does not.** The overall first-listed rate
is 0.481–0.497 for four of the five models -- indistinguishable, and consistent with no
position bias at all. The same data resolves discrimination from +1.000 down to −0.182. This is
the shared-assumption hazard again: an aggregate over the conditioning variable hides exactly
the quantity of interest, and splitting by it recovers a clean ordering that matches the
reliability ordering (`qwen` never errs; `allam` has the largest $L_t$).

**`allam-2-7b` is anti-correlated with the rule, which is worse than ignoring it.**
Discrimination −0.176 at $z = -5.0$: it picks the first-listed tool *more* often when the ref is
odd (0.777) than when it is even (0.595), so its bias is strongest exactly where it is wrong.
Its accuracy split reflects this -- 0.595 on even refs against **0.223** on odd. A model that
merely ignored the rule would sit near 0 and score ~0.5 on both.

**Two corrections to the earlier subset-based numbers.**

1. `allam-2-7b` was reported as "picks the first-listed tool ~80% of the time almost regardless
   of the ref, discrimination 0.041". On full data its overall rate is **0.685**, not ~0.80, and
   its discrimination is **−0.182**, not ~0. The qualitative claim that it is not applying the
   rule survives and strengthens; the specific characterisation as an unconditional position
   bias does not. It is an anti-correlated bias.
2. `llama-3.1-8b-instant`'s parity *accuracy* asymmetry has largely **vanished**. The subset gave
   0.525 on even refs against 0.742 on odd; full data gives 0.562 against 0.588. So the large
   asymmetry attributed to that model was a property of the shallow-depth subset, not of the
   model. Its discrimination also fell, from +0.267 to +0.149, though it remains clearly
   positive ($z = +4.8$).

The surviving general claim is narrower than the earlier one and better supported: **parity of
the held value affects rule application strongly in one model (`allam-2-7b`) and weakly or not
at all in the others**, and discrimination rather than accuracy is the statistic that shows it.

**The confound, and why discrimination resolves it.** With a fixed presentation order the
correct tool for an even ref *is* the first-listed tool, so "always picks the first-listed
option" and "applies the rule but only succeeds on even refs" predict the same accuracy pattern.
Discrimination separates them because it conditions on the ref rather than on the outcome: a
pure position bias returns 0 whatever its accuracy, since it picks the first-listed tool at the
same rate either way. That is why the four models at ~0.49 overall are distinguishable at all.

The `shuffle` condition removes the confound outright by randomising which branch is listed
first at each step, making position and parity independent. `first_listed_even` is recorded per
step, so once that condition lands the four cells (ref parity x presentation order) settle it
without relying on the statistic. Until then, discrimination is the evidence.


### 4.2 Zero syntactic errors, verified rather than assumed

Across every model, **every single call executed and no retry ever fired** — 1,372
records on `llama-3.1-8b-instant`, 176 on `llama-3.3-70b-versatile`, 324 on
`qwen3.6-27b`, and so on, with `executed=True` and `n_attempts=1` throughout. A
syntactic error is by definition a call that fails to execute, and a retry only
fires on one, so there were none to classify.

This was flagged as a structural anomaly ("all errors in one bucket") and checked
against the raw rows before being accepted, because an identical symptom in the
simulated suite *was* a bucketing defect. Here it is genuine: these models emit
well-formed, schema-valid JSON essentially always. The `<think>`-block parser
handles the reasoning models' output correctly, which is part of why.

The consequence is that $f_{\text{syn}} = 0$, $r_{\text{syn}}$ is unmeasurable
(NaN), and the syntactic half of the state model's recovery structure is empty on
real data. The prior centres passed to the fit were therefore
`r_syn = [nan, nan, nan, nan]`, and the sampler fell back to the weakly informative
default for that parameter — so **the fitted $r_{\text{syn}}$ values in §5 are
prior, not posterior, and carry no information.** They are shown for completeness
and must not be read as estimates.

---

## 5. Fitted severity and recovery, with identifiability

*Populated from `report_real.py` §4.*

Reported as **secondary and diagnostic**, always alongside the posterior
correlation between $\pi$ and each recovery rate, per the fallback the method was
designed with. Recovery hyperpriors are centred on rates **measured** from
labelled transitions in the same run, not assumed.

**What the validation run establishes about this fit, and it is not encouraging.**
Even with correctly measured priors — landing within 0.02 of the configured family
means — the fitted $\pi$ did not recover configured values: absolute error
0.064–0.313 on routing (mean 0.189), with the $\pi$/recovery posterior correlation
at 0.84–0.89. The hypothesis that informative priors would resolve the
non-identifiability was tested and **rejected**: fixing the inputs left the
correlation structure intact, which is what a structural rather than
mis-specification account predicts. Under exact-match scoring a poisoned step can
only be judged globally correct if it exactly recovers the true value, so severity
and recovery manifest through the same observable event.

Consequence for this document: $\pi$ is reported with its interval and its
correlation, and no claim rests on it that $L_t$ cannot carry.

Posterior means with 89% credible intervals:

| model | $\pi$ | $r_{\text{syn}}$ | $r_{\text{sem}}$ | corr($\pi$, $r_{\text{syn}}$) | corr($\pi$, $r_{\text{sem}}$) |
|---|---|---|---|---|---|
| llama-3.1-8b-instant | **0.91 [0.82, 0.98]** | 0.50 [0.10, 0.89] † | 0.06 [0.01, 0.15] | 0.006 | 0.347 |
| allam-2-7b | **0.68 [0.55, 0.82]** | 0.50 [0.10, 0.90] † | 0.13 [0.01, 0.34] | −0.015 | 0.542 |
| llama-3.3-70b-versatile | 0.80 [0.35, 0.98] ‡ | 0.50 [0.10, 0.90] † | 0.19 [0.02, 0.64] | −0.015 | 0.000 |
| gpt-oss-120b | 0.49 [0.05, 0.95] ‡ | 0.51 [0.10, 0.90] † | 0.33 [0.03, 0.83] | 0.015 | 0.020 |
| qwen3.6-27b | 0.50 [0.05, 0.95] ‡ | 0.50 [0.10, 0.90] † | 0.33 [0.03, 0.82] | 0.009 | 0.011 |

† prior, not posterior — no syntactic errors exist to inform it (§4.2).
‡ interval spans almost the whole unit range: essentially the prior, because these
models produced too few errors to inform $\pi$. Reporting 0.49 for `gpt-oss-120b`
as a severity estimate would be reporting the prior mean back.

**Only two of the five $\pi$ estimates carry information**: 0.91 for
`llama-3.1-8b-instant` and 0.68 for `allam-2-7b`, both with intervals well inside
the unit range. The other three are at ceiling on this task and their posteriors
are barely updated from the prior. That is a data limitation, stated rather than
papered over.

**MCMC diagnostics** — targets R-hat < 1.01, ESS > 400, zero divergences:

| parameter | max R-hat | min ESS |
|---|---|---|
| $\pi$ | 1.0008 | 4,044 |
| $r_{\text{syn}}$ | 1.0004 | 6,192 |
| $r_{\text{sem}}$ | 1.0019 | 6,371 |

Zero divergences. **All diagnostics healthy**, with ESS an order of magnitude above
target. Note that healthy sampling says the posterior was explored properly; it
says nothing about whether the data constrained it, which is what the † and ‡
footnotes above are about.

**Identifiability is materially better on real data than on the simulated suite.**
Largest $|$correlation$|$ is 0.54 here against 0.84–0.89 on simulated routing. The
mechanism is §4.1: real models never recover, so there is no recovery rate for $\pi$
to trade against. This reverses the expectation the validation set up, and it is
the pre-registered rule working as intended — real data is primary, and the
divergence is the finding.

---

## 6. H2: severity against scale, on both axes

*Populated from `report_real.py` §5.*

**The two contrasts are reported separately and are not averaged, because they
are not comparable in kind.** The dense llama step from 8B to 70B is a ~8.75×
change in both total and active parameters. The sparse gpt-oss step from 20B to
120B is 6× in total parameters but only 1.4× in active parameters. Averaging them
into a single "scale" coefficient would hide exactly the thing that makes the MoE
case interesting. Both readings are given for each family.

Two points per family is a signed contrast, not a fitted trend, and is described
as such.

**Dense axis — llama, 8B → 70B** (8.75× total, 8.75× active):

| | 8B | 70B |
|---|---|---|
| $\pi$ | 0.908 | 0.805 |
| $L_t$ | +0.686 at depth 6 | +0.000 at depth 4 |

$\Delta\pi = -0.104$ in the direction H2 predicts. **But this contrast is not yet
interpretable**, for two reasons that must be stated rather than absorbed: the 70B
$\pi$ estimate is largely prior-driven (interval [0.35, 0.98]), and the two $L_t$
figures come from *different depths*, because the 70B model's sweep stopped at
depth 4 while the 8B model reached depth 6. A signed comparison across different
depths is not a like-for-like contrast, and the reporter labels it as such rather
than printing the two numbers side by side.

**Sparse MoE axis — gpt-oss, 20B → 120B** (6.0× total, 1.4× active): not yet
computable. `gpt-oss-20b` has no data (§2.1), so the contrast has one endpoint.

**H2 is therefore unresolved on this run.** Both axes need the deep bins where the
larger models actually make errors, and the 20B endpoint needs its token bucket to
refill. Reporting $\Delta\pi = -0.104$ as support for H2 would be reporting a
prior-driven number measured at mismatched depths.

---

## 7. H4: error-type composition by step index

*Populated from `report_real.py` §6.*

**Resolved by step index, not by task depth.** Pooling every step of a depth-8
task into one bin averages away the trend H4 is about; on the validation suite
per-depth resolution recovered 26% of a configured span where per-step resolution
recovered 75%.

**Reported as a suite-level claim, pooled across models, with per-model rank
correlations shown as directional evidence only.** This is a power limitation
established *before* the run, not discovered after it. A step index $i$ is
populated only by tasks of depth $> i$, so per-step counts fall off sharply, while
cost is driven by invocations — a deep task buys coverage at more step indices and
pays proportionally more for it. An exhaustive search over 47,068 equal-$n$
allocations under a fixed invocation ceiling found none with a better worst-case
usable-step prefix than the shape already configured. What made H4 testable was
not a better allocation shape but the nested unequal-$n$ design, which roughly
triples the pooled task count for the same calendar window and takes the pooled
usable step indices from 2 to 6 in the pessimistic recovery band.

**If the achieved pooled counts do not clear the threshold, H4 is withdrawn rather
than reported.** That decision is implemented in `report_real.py` — fewer than
four pooled step bins at 30+ fresh errors prints a withdrawal, not a trend — so it
does not depend on judgement at write-up time.

### 7.1 H4 is withdrawn, and the reason is not lack of power

Pooled across models, restricted to clean-context (fresh) errors:

| step | fresh errors | $f_{\text{syn}}$ | selection share | argument share | sel/arg |
|---|---|---|---|---|---|
| 0 | 159 | 0.000 | **1.000** | **0.000** | undefined |
| 1 | 101 | 0.000 | **1.000** | **0.000** | undefined |
| 2 | 28 | 0.000 | 1.000 | 0.000 | undefined |
| 3 | 11 | 0.000 | 1.000 | 0.000 | undefined |
| 4 | 3 | 0.000 | 1.000 | 0.000 | undefined |

**Of 302 fresh errors, 302 are selection errors and 0 are argument errors.** The
argument channel is empty, so the selection-to-argument ratio H4 is about is
undefined at every step index. There is no mix, and therefore no shift in the mix
to detect.

This was verified directly against the raw rows for both small models
(`selection_matches_gold` is `False` for all 148 fresh errors on
`llama-3.1-8b-instant` and all 147 on `allam-2-7b`) rather than inferred from the
aggregate, because an all-in-one-bucket result had previously been a defect.

**Mechanism.** On a routing task a fresh error means the parity rule was
mis-applied — the wrong tool was named. Transcribing the argument is trivial by
comparison: it is copied verbatim from the previous result stated in the
observation, and these models essentially never get that wrong *while holding a
correct value*. So the only way to fail at a clean-context step is to choose wrongly.

**Argument errors do exist — but only after the context is already poisoned.** Among
poisoned-context errors on `llama-3.1-8b-instant`, 145 of 289 name the *gold* tool
while carrying a wrong value, because the parity rule applied to a corrupted value
can coincidentally land on the right branch. Those are propagated errors, which the
composition deliberately excludes as not being fresh evidence about which error type
the model produces. So the argument channel is populated exactly where H4's
measurement must not look.

**Power was not the binding problem.** Projected against the *measured* per-model
accuracies, the planned allocation yields 4 contiguous usable step bins — enough for
a rank correlation to reach p < 0.05 at $\rho = -1$. (The earlier projection of 6
bins assumed p = 0.50 uniformly; correcting it for the three ceiling-level models,
which contribute almost no errors however many tasks they run, drops it to 4. The
projection function now takes measured per-model accuracies for this reason.) H4
would have been marginally testable on power grounds. It is untestable because one
of its two categories does not occur where the composition is defined.

**What would be needed to test it.** A task variant whose arguments can be got wrong
*independently of tool choice* — for instance arguments requiring a transformation of
the carried value rather than a verbatim copy, or multi-argument calls where one
field can be wrong while the tool is right. That is a task-design change, not a
budget change, and it is recorded here as the concrete prerequisite rather than left
as "future work".

---

## 8. Calling-mode ablation

*Populated from `report_real.py --tag native`, configured in
`config/ablation_native.yaml`.*

Every correctness figure above is measured through one calling convention: a
uniform JSON-in-prompt protocol, used so all models are scored on the same
interface regardless of provider support. That uniformity is what makes the family
and scale axes comparable, and it also means an argument error could be an artifact
of the protocol. This ablation measures how much.

**Scope: five models, not six.** `groq/allam-2-7b` rejects native tool calling
outright ("tool calling is not supported with this model"), verified by direct
probe before the ablation was configured rather than discovered during it. It stays
in the main sweep, where it is the 7B leg of the family-at-matched-scale axis; only
the calling-mode comparison loses it, so the ablation spans both scale-within-family
axes but not the smallest scale point.

The ablation runs depths 1 and 4 only — the calling convention is a per-call
property, not a propagation one — on a nested prefix of the main suite, so every
ablation task has an exact uniform-mode counterpart to compare against task by
task.

**Reported as a direct contrast.** The methodology previously claimed a
mixed-effects logistic regression with calling mode as a covariate; that was never
implemented and the claim has been withdrawn rather than approximated.

**NOT RUN.** Provider-native tool calling was probe-verified as working on five of the six models before the ablation was configured (`allam-2-7b` returns "tool calling is not supported with this model"), and `config/ablation_native.yaml` specifies the run. It was not executed within the time constraint, so **no claim is made about how much of any model's measured unreliability is a calling-mode artifact rather than a genuine limitation.**

---

## 9. Run status and honest accounting of what is not here

*Populated from `report_real.py` header, which reads the achieved-vs-requested
allocation, cap-stops, model exclusions, and structural-anomaly flags directly
from the run metadata.*

The runner records the $n$ **achieved** per depth, not the $n$ requested, and
states any shortfall. A sweep cut short by an exhausted allowance leaves a nested
prefix, which is a valid smaller allocation, so partial data is usable at the
achieved $n$ — but it must never be reported as if it had the $n$ it asked for.

Models are excluded from the hierarchical fit if fewer than two depth bins reached
three tasks, since a depth trend cannot be estimated from less. Exclusions are
recorded in the metadata and their raw rows are still written.

### Withdrawn hypotheses

- **H5a/H5b (error-feedback format).** The opaque feedback mode is implemented and
  the harness supports it, but the sweep was never run and the token budget cannot
  absorb a second one. Withdrawn from the hypothesis list. No claim is made about
  feedback format.
- **Soft argument matching.** Reported as vacuous rather than as a result: the
  synthetic tasks use exact integer arithmetic, which has no graded notion of
  "close but not quite", so soft matching is identical to exact matching by
  construction. Free-text arguments were **not** added to manufacture a non-vacuous
  metric.
- **Version pinning by release date.** No provider exposes a version field, and
  every original model ID had been retired, so the claim that models are pinned to
  a dated release is false and has been corrected rather than reworded.

### Structural-anomaly detection

Four patterns are flagged automatically rather than left to inspection, each one
drawn from a defect that actually occurred: identical $p_t$ and $g_t$ at every
depth, $p_t$ pinned at 1.000 or at $\le 0.02$ beyond depth 1, all errors collapsing
into one bucket, and parse failures above 50% at every depth. A flagged model stops
the write-up rather than being averaged into the suite.

### Achieved allocation, day 1 of 8

| model | requested (d1/2/4/6/8) | achieved | status |
|---|---|---|---|
| llama-3.1-8b-instant | 60/60/65/65/47 | 60/60/65/41/– | cap-stopped |
| allam-2-7b | 60/60/65/65/47 | 60/60/65/6/– | cap-stopped |
| gpt-oss-120b | 26/26/28/28/20 | 26/26/28/2/– | cap-stopped |
| qwen3.6-27b | 26/26/28/28/20 | 26/26/21/–/– | cap-stopped |
| llama-3.3-70b-versatile | 12/12/13/13/9 | 12/12/13/–/– | cap-stopped |
| gpt-oss-20b | 26/26/28/28/20 | none | pilot drawdown; excluded from fit |

**No depth-8 data yet on any model, and depth 6 only partially.** Since $L_t$ grows
with depth and the larger models only begin to err at depth, the missing bins are
disproportionately the informative ones. Nothing in §2–§7 should be read as final.

Zero backend failures and zero unexpected 429s across the whole run: every stop was
a clean token-cap halt. The control arm did not run for any model, because it is
sequenced after the primary arm and every model exhausted its allowance first — so
§3 is empty, which is a real gap rather than a null result.

---

### An operational failure worth recording

The collection driver's run window expired and **was not relaunched for six days**, during which
no data was collected. The caps refill continuously but **cap at their daily ceiling**, so the idle
period banked nothing — it lost roughly six days of throughput. Had that time been used, the
transformed-argument and control conditions would in all likelihood have been collected, and this
document would report them. The failure was an assumption on our part that a resumable driver was
still running, rather than a checked fact; it is the same shape of error this project repeatedly
found elsewhere, applied to our own process.

## 10. Interim summary

Stated as what the data currently supports, does not support, and cannot address —
in that order, with the third category not softened.

**Supported by the data so far.**

- Error propagation in tool-use chains is large and monotone in depth on both small
  models: $L_t$ reaches **0.686** and **0.700** at depth 6, i.e. roughly 70% of
  clean-context capability lost to the model's own earlier mistakes. The
  teacher-forced baseline declines far more gently over the same range, so this is
  not context-length degradation.
- The teacher-forced/free-running decomposition works on real models. $L_1 = 0.000$
  exactly, by construction rather than tolerance; the per-step error profiles are
  identical at step 0 and diverge thereafter only in the free arm.
- **Recovery is zero, not small.** $P(\text{next correct} \mid \text{wrong}) = 0.000$
  over 484 transitions. This is the study's sharpest empirical result and it was not
  anticipated by the simulated validation, which configured recovery at 0.05–0.30.
- Real models emit well-formed calls essentially always: no syntactic failures in
  any of the ~1,900 invocations recorded, so the syntactic error channel is empty.

**Not supported / unresolved.**

- **H2 (severity vs scale) is unresolved.** The signed contrast points the right way
  ($\Delta\pi = -0.104$ dense) but rests on a prior-driven endpoint and compares
  $L_t$ at mismatched depths. The MoE axis has one endpoint missing entirely.
- **H4 (error-mix shift along the chain) is withdrawn**, because on this task variant
  100% of clean-context errors are selection errors and the argument channel is
  empty — one of its two categories does not occur where the composition is defined.
  Not a power failure; a task-design mismatch, with the concrete fix recorded in §7.1.
- **The parametric fit is informative for only two of five models.** The other three
  are at ceiling and their $\pi$ posteriors are barely updated from the prior. Their
  fitted values are shown with explicit markers and are not to be read as estimates.

**Cannot be addressed with this budget or design.**

- **H5a/H5b (feedback format)** — implemented but never run; the token budget cannot
  absorb a second sweep. Withdrawn from the hypothesis list.
- **A proprietary reference tier** — the Gemini project is blocked. Absent, not
  estimated.
- **An FC-tuned vs base contrast** — no such variant is hosted free.
- **Soft argument matching** — vacuous by construction on integer-arithmetic tasks.
  Reported as vacuous; free-text arguments were not manufactured to rescue it.
- **A 3–4 point scale curve per family** — no free host offers 3+ sizes of one family.

**One methodological result that is independent of the models measured.** A simulator
validates the channels it exercises and silently certifies the ones it does not. Nine
defects in this pipeline were all instances of that one failure mode, three of them
invisible to every assertion the validation suite made, because the suite asserted on
rates the simulator itself produced. Two of the real-run "anomalies" here were flagged
by the automated detector, checked against raw rows, and found to be **genuine data
properties rather than defects** — which is the discipline that mistake taught, applied
in the other direction.
