# Method notes for the real run

Working notes captured during execution, for folding into
`docs/RESULTS_real_run.md` and the paper's reproducibility section. Recorded as
they happened rather than reconstructed afterwards.

## Environment and invocation (Windows)

The `python` on PATH was the Microsoft Store stub, which exits without running
anything. The real interpreter is reached through the `py` launcher, so every
command in this run was:

```
py -m pip install -r requirements.txt
py -m pip install -e .
py scripts/estimate_cost.py
py scripts/run_real_suite.py [--pilot] [--tag TAG] [--call-mode native]
py -m tur.analysis.plots --tag real
py tests/test_smoke.py            # likewise test_regression, test_parser,
                                  # test_limiter, test_observation_loop
py scripts/stress_test.py
```

`PYTHONPATH=src` is unnecessary after `pip install -e .` but harmless. There is
no `source` on Windows, so `.env` is loaded in-process by `run_real_suite.py`
via `python-dotenv`, anchored to the repository root rather than the working
directory.

Versions used: Python 3.13.7, pymc 6.2.0, arviz 1.2.0. `pymc` and `arviz` are
pinned exactly in `requirements.txt`, not floored. Both are a major version
ahead of what the analysis code was originally written against, and arviz's
classic `plot_forest` signature has already broken once across a release (there
is a fallback in `analysis/plots.py` for it). A silent upgrade would take the
figures with it.

## Context adequacy was verified empirically, not assumed

`allam-2-7b` has a 4,096-token context window, an order of magnitude smaller
than every other model in the suite (131,072). The initial concern was that
depth-8 chains would be truncated for that model only, which would confound
model identity with context overflow and make its `p_t` decline uninterpretable.

This was measured rather than argued about. The largest prompt the harness ever
builds is the deepest teacher-forced step of a depth-8 task, since that carries
the full correct history. Measured with `tiktoken`/`o200k_base` over the actual
prompt builders, the worst case across sampled depth-8 tasks was **1,042
tokens** — about 25% of allam's window, and no risk of truncation.

The concern was therefore unfounded and `allam-2-7b` stays in the suite at full
depth. The measurement is reported because "we checked" is a materially
different claim from "we assumed it was fine", and because the headroom figure
tells a replicator how much larger the task suite could grow before that model
would need dropping.

## Tokens per DAY is the binding constraint, it is undocumented, and it is not uniform

This is the single most consequential operational fact about running this study
on Groq's free tier, and it is documented nowhere we could find. Anyone
replicating will hit it.

**TPD appears in no response header.** Groq returns
`x-ratelimit-limit-requests`, `x-ratelimit-limit-tokens` (per minute), and their
`remaining`/`reset` counterparts. For tokens per day, every model in the suite
returns `-`. The only place the number appears is the **body of the 429 that
enforces it**:

```
Rate limit reached for model `llama-3.3-70b-versatile` in organization
`org_...` service tier `on_demand` on tokens per day (TPD):
Limit 100000, Used 99442, Requested 679.
```

So the limit is discoverable only by being refused by it. The harness therefore
parses the figure out of the 429 body, records it, and persists it to
`data/results/discovered_tpd.json` so that later days plan against a measured
value instead of rediscovering it by being blocked again.

**The values are not uniform, so one observation does not generalise.** Measured
2026-08-10, each from the 429 that enforced it:

| Model | TPD (measured) | RPD | TPM |
|---|---|---|---|
| `groq/llama-3.1-8b-instant` | > 500,000 (not reached) | 14,400 | 6,000 |
| `groq/llama-3.3-70b-versatile` | **100,000** | 1,000 | 12,000 |
| `groq/openai/gpt-oss-20b` | 200,000 | 1,000 | 8,000 |
| `groq/openai/gpt-oss-120b` | 200,000 | 1,000 | 8,000 |
| `groq/qwen/qwen3.6-27b` | 200,000 | 1,000 | 8,000 |
| `groq/allam-2-7b` | **500,000** | 7,000 | 6,000 |

A 5× spread, and the *lowest* TPD belongs to the model with the *highest* TPM —
so TPD cannot be inferred from any other published limit. Our own first estimate
extrapolated the one pilot observation (200,000) to all six models and was wrong
in both directions: too generous for `llama-3.3-70b` and far too stingy for the
two high-RPD models.

**The caps refill continuously rather than resetting at midnight.** The 429 for a
daily cap quotes a retry delay of one to three minutes, not hours, and the
observed `x-ratelimit-reset-requests` for a 1,000/day model is 1m26.4s, which is
exactly 86,400/1,000 seconds. So these behave as token buckets refilling at
`TPD/86400` per second. A run can burst to the bucket size and then proceeds at
the sustained rate, which is why spreading a sweep across days works at all, and
why a cap-stop can be resumed within minutes rather than waiting for a reset.

**Scale of the problem for this study.** Routing cost is roughly quadratic in
depth — the schema grows with depth *and* the whole prompt is re-sent at every
step — measured at 640 tokens for a depth-1 task and 24,447 for a depth-8 task,
both arms, under an erring policy. The originally planned sweep came to 2.9M
tokens per model, i.e. 37 days against the binding model's 100,000/day. That is
what forced the nested unequal-n design in the next section.

One artifact worth flagging so it is not misread: `gpt-oss-20b` cap-stopped after
21 tasks on the first day of the real run, far earlier than its siblings. That
was not a lower cap. Pilot runs earlier the same day had already consumed 199,718
of its 200,000 tokens. **Pilots draw on the same daily allowance as the real run**,
which is easy to forget when the pilot is cheap in wall-clock terms.

## Tokens per minute is the binding constraint within a day

Provider documentation leads with requests-per-day, and Groq's docs quote an
org-wide 14,400/day. Both framings are misleading for this workload, in two
separate ways.

First, the enforced request cap is **per model**, not per organisation, and is
far lower than the headline for most models. Read from the
`x-ratelimit-limit-requests` response header on 2026-08-09:

| Model | requests/day | tokens/min |
|---|---|---|
| `groq/llama-3.1-8b-instant` | 14,400 | 6,000 |
| `groq/llama-3.3-70b-versatile` | 1,000 | 12,000 |
| `groq/qwen/qwen3.6-27b` | 1,000 | 8,000 |
| `groq/openai/gpt-oss-20b` | 1,000 | 8,000 |
| `groq/openai/gpt-oss-120b` | 1,000 | 8,000 |
| `groq/allam-2-7b` | 7,000 | 6,000 |

Only one model in the suite gets the documented 14,400. The rest get 1,000,
which is what actually sizes the experiment.

Second: **within** a day the sweep is throughput-bound rather than request-bound.
The daily request cap is cleared comfortably, but the tokens-per-minute ceiling
dictates several hours of wall clock per model regardless. Observed on the real
run: `llama-3.1-8b-instant` accumulated 5,361 seconds of deliberate pacing sleep
across 675 calls, i.e. the run spends most of its wall clock waiting on TPM by
design. So RPD decides nothing here, TPM decides how long a day's work takes, and
TPD (previous section) decides how many days there are. Three different ceilings,
and the run has to clear all three.

Three models are swept **concurrently** — the caps are per model, so concurrent
models are not drawing down a shared allowance, and calendar time becomes the
slowest model rather than the sum over models.

A separate correction to an earlier note in this file: the daily allowance was
described as a request bucket, and it is really a *token* bucket, refilling at
`TPD/86400` per second. Both refill continuously; it is the token one that binds.

The harness paces against TPM at 80% of the ceiling (`--headroom`) rather than
riding it, because the token estimate is approximate and overshooting converts
directly into provider 429s and wasted retries. Per-model `rpd`/`tpm` live in
`config/default.yaml` and are read by both the runner and `estimate_cost.py`, so
there is one source of truth.

## Why n differs across models: the nested unequal-n design

A reader will notice that the six models do not run the same number of tasks. That
is deliberate and it is not a compromise on comparability. The short version: the
models share a common *prefix* of one task suite, and comparisons are made on that
prefix, while each model's own estimates use everything it ran.

**The problem.** The binding constraint is the per-model daily token allowance,
and it spans 5× across the suite (100,000 for `llama-3.3-70b` against 500,000 for
`allam-2-7b`; see the previous section). Sizing every model to the smallest
allowance — the obvious way to keep the suite comparable — wastes almost all the
capacity of the others. Measured over the same 8-day window it gives 240 usable
tasks across the suite against 817 for the nested design, a factor of 3.4.

**The design.** `per_depth` in `config/default.yaml` is a *reference* allocation:
the most any model runs. Each model runs a nested prefix of it, scaled to its own
measured TPD over `budget_days`. Both arms scale by the same factor, so the
control arm shrinks alongside the primary rather than crowding it out on the
poorest model.

**Why nesting is exact rather than conventional.** The suite generators emit tasks
depth by depth with `k` ascending, and a task's id and content depend only on
`(depth, k, base_seed)` — never on how many tasks were requested. So asking for 12
tasks at depth 4 returns precisely the first 12 of what asking for 65 returns, same
ids, same gold trajectories. This is asserted in `tests/test_nesting.py` rather
than assumed, because if it ever broke, every cross-model contrast would silently
stop being paired and nothing would raise.

**What it buys, and what it costs.**

* Cross-model contrasts — the scale and family axes, and the paired bootstrap —
  are computed on the common prefix and remain exactly paired, as they were under
  equal n.
* Each model's own $p_t$, $g_t$ and $L_t$ use its full n, so the high-allowance
  models get tighter intervals instead of being truncated to the poorest model's.
* The hierarchical fit needs no change at all: its likelihood is binomial with
  per-model trial counts, which already accommodates unequal n. The partial
  pooling then does what it is for — the low-n models are shrunk further toward
  their family mean, which is the correct behaviour rather than a workaround.
* The cost is unequal precision across models, and it is not evenly distributed:
  `llama-3.3-70b` runs 59 tasks against `allam-2-7b`'s 297, so its per-model
  intervals are materially wider. That model is retained rather than dropped
  because it is the only *dense* scale-within-family contrast in the suite —
  `gpt-oss` is sparse MoE — and H2 needs it. Lower n with wider intervals is
  exactly what nesting is for.
* Pooled per-step counts for H4 draw their mass from the high-allowance models,
  which is the only reason H4 remains testable here at all: pooled usable step
  indices go from 2 under equal n to 6 under nesting, in the pessimistic recovery
  band.

**A consequence for interrupted runs.** Because a sweep generates tasks in a fixed
order, a run cut short by an exhausted allowance leaves exactly a prefix of its
allocation — which is itself a valid, smaller nested allocation. So partial sweeps
are usable on the same terms as any other model's data, provided the n *achieved*
is recorded rather than the n requested. The runner records the achieved per-depth
counts, states any shortfall loudly, and carries both into the run metadata. Under
the earlier equal-n design a truncated sweep genuinely was unusable, because it
broke the assumption that every model ran the same tasks; that assumption is gone.

Models are excluded from the hierarchical fit only if fewer than two depth bins
reached three tasks, since a depth trend cannot be estimated from less. Exclusions
are recorded in the metadata, and the raw rows are still written.

## MoE scale is reported on both axes, because they disagree

Two models in the suite are sparse mixture-of-experts, and for them "parameter
scale" has two defensible readings that diverge sharply:

| Model | total params | active params/token |
|---|---|---|
| `groq/openai/gpt-oss-20b` | 20B | 3.6B |
| `groq/openai/gpt-oss-120b` | 120B | 5.1B |

The gpt-oss contrast spans **6× on total parameters but only 1.4× on active
parameters**. H2 (severity shrinks as scale grows) can therefore come out
differently depending on which axis is used, and that difference is
interpretable rather than cosmetic:

- If `pi` tracks **total** parameters, propagation robustness is associated with
  stored knowledge/capacity.
- If `pi` tracks **active** parameters, it is associated with per-token compute.

Both are reported. `scale` in the config is total parameters (the conventional
headline figure) and `active_scale` records the active count, so the pi-vs-scale
figure can be regenerated against either.

This also makes the two scale contrasts **non-comparable in kind**, which must
be stated rather than averaged away:

- `llama` 8B → 70B is a **dense** 8.75× jump.
- `gpt-oss` 20B → 120B is **sparse**: 6× total, 1.4× active.

If both contrasts point the same way despite that structural difference, the
evidence for H2 is stronger than either contrast alone would support. If they
diverge, the dense/sparse distinction is the obvious first hypothesis and should
be named as such. Collapsing them into a single pooled scale claim would hide
exactly the comparison that makes the result informative.

## Model suite is smaller than the methodology specifies

Verified against each provider's live model list on 2026-08-09. Every Groq model
ID in the original config was stale: the entire Qwen2.5 / Llama-3.1-Instruct
generation has been retired from the free tier. Consequences for the design:

- **Scale within family**: no free host offers 3–4 sizes of a single family any
  more. Reduced to two 2-point contrasts (above) instead of the 4-point Qwen
  ladder the methodology assumes.
- **Family at matched scale**: only `allam-2-7b` (7B) and `llama-3.1-8b-instant`
  (8B) sit in the 7–9B band, so two families rather than 3+. `gpt-oss-20b` vs
  `qwen3.6-27b` provides a second, better-populated mid-scale bin.
- **Function-calling-tuned vs base**: **not testable**. No xLAM or Hammer
  variant is available on any accessible free host.
- **Proprietary reference**: not obtained. The Gemini key returns
  `403 PERMISSION_DENIED` ("Your project has been denied access") on
  `generateContent`, reproduced against the raw REST endpoint so it is not a
  LiteLLM routing artifact. Listing models succeeds, so the key is valid and the
  project is blocked. One line in `config/default.yaml` re-enables it once a key
  from a working project is available; the response cache is per model, so
  adding it later costs nothing already spent.

## The free run was not an agent loop until this run fixed it

Found at pilot scale on real models, and worth recording because of how it
evaded the entire simulated validation.

`run_free` appended neither the model's own call nor the tool's result to the
conversation. At step `t` the model was asked for a `ref` it had never seen —
the only reference value anywhere in its context was the seed stated in the task
intro. Measured `g_t` was exactly `1/depth` on both pilot models (0.250 at depth
4, 0.000 at the deepest step), because step 0 is the only answerable step.
`L_t` computed from that was measuring the harness, not propagation.

No simulated test could have detected it: `MockBackend` reads the carried `ref`
from the runner's stashed `_ctx` rather than from the message history, so a
history missing every observation was indistinguishable from a correct one. The
full validation suite passed against a run loop that could not work on a real
model. The lesson generalises — a mock that receives state out-of-band cannot
validate the channel that state is supposed to travel through.

Both run modes now build structurally identical histories, so `p_t` and `g_t`
differ only in whether that history is *correct*; with a perfect policy the two
are byte-identical. `tests/test_observation_loop.py` asserts against the message
history directly, which is the thing the mock bypasses.

### Limitation: simulation-based validation cannot test the channel the mock bypasses

This is the generalisable lesson, and it is a limitation of the validation
methodology rather than an implementation slip.

The mock backend's interface was **too generous**. `SimPolicy` is called as
`policy(task, step, ref, attempt)` — it is *handed* the reference value it needs
as a positional argument, delivered out-of-band through a `_ctx` dictionary the
runner stashes on the last message. A real backend receives one thing only: the
list of messages. The two interfaces are not equivalent, and the difference is
exactly the dependency the experiment exists to measure.

Consequently the simulated suite could not observe that the message history was
missing every observation. Every simulated policy behaved identically whether
the history was complete, truncated, or empty, because none of them read it. The
validation suite reported a healthy pipeline — correct `p_t` decline, plausible
`g_t`, recovered `pi` within 0.15, good MCMC diagnostics — while the run loop it
was validating could not have worked on any real model.

Two corollaries worth stating for anyone building a similar harness:

1. **A mock that receives state out-of-band cannot validate the channel that
   state is supposed to travel through.** If the quantity under study is "does
   information reach the model", the mock must be forced to obtain that
   information the same way the model does. A stricter mock here would have
   parsed its `ref` out of the message history and failed loudly when it was
   absent — which is what a real model effectively did the moment one was
   attached.

2. **Tests must assert on the artifact sent to the provider, not on the
   backend's behaviour.** Behavioural assertions inherit the mock's blind spots
   by construction. `tests/test_observation_loop.py` therefore captures the
   `messages` list at each backend call and asserts on its structure and
   contents directly: that the ref a step needs appears somewhere in its
   history, that observations and assistant turns are present, and that the free
   and teacher-forced histories are byte-identical under a perfect policy. Those
   assertions hold regardless of what any backend chooses to read.

The practical implication for this study is scoping: the simulated validation
run demonstrates that the **fitting and aggregation** pipeline recovers known
parameters from labelled transitions. It does not, and structurally cannot,
validate the **prompt-construction and observation-passing** path. Only a real
model exercises that, which is why this bug surfaced in the first pilot minute
of real API traffic and not in any of the 24 stress-test configurations.
