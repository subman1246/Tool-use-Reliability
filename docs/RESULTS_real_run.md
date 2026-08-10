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

*[PENDING: table. Requires a completed sweep; see §9 for current run status.]*

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

*[PENDING: table.]*

---

## 4. Model-free propagation check ($\Delta_1$)

*Populated from `report_real.py` §3.*

$\Delta_1 = P(\text{step } t{+}1 \text{ correct} \mid t \text{ correct}) -
P(\text{step } t{+}1 \text{ correct} \mid t \text{ wrong})$, which is positive iff
a wrong step makes the next step more likely to be wrong. It assumes nothing about
the state model, so it is the cleanest possible check that propagation is real.

*[PENDING: table.]*

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

*[PENDING: table and MCMC diagnostics against their targets — R-hat < 1.01,
ESS > 400, zero divergences.]*

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

*[PENDING: table.]*

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

*[PENDING: pooled table, per-model directional table, and the trend test.]*

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

*[PENDING.]*

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

*[PENDING: achieved allocation table.]*

---

## 10. Summary

*[PENDING: to be written from the sections above once the sweep completes. It
will state which hypotheses the real data supports, which it does not, and which
this budget could not test — in that order, and without softening the third
category.]*
