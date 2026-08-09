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

## Tokens per minute is the binding constraint, not requests per day

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

Second, and more importantly for anyone replicating: the sweep is **throughput-
bound, not request-bound**. At 2,780 calls and ~2.0M tokens per model, the daily
request cap is cleared comfortably when spread over three days (927/day), but
the tokens-per-minute ceiling dictates 3.7–7.4 hours of wall clock *per model*
regardless — roughly 35 hours for the six-model suite run sequentially. The
request count is the constraint that decides whether the run is *possible*; TPM
is the constraint that decides how *long* it takes, and it is the one that
actually hurts.

A practical consequence: the per-model daily allowance behaves as a
continuously-refilling bucket rather than a calendar-day counter. The observed
`x-ratelimit-reset-requests` for a 1,000/day model was 1m26.4s, which is exactly
86,400/1,000 seconds — one day's worth of refill per request. So a run can burst
up to the bucket size and then proceeds at the sustained rate, which is why
spreading across days works at all.

The harness paces against TPM at 80% of the ceiling (`--headroom`) rather than
riding it, because the token estimate is approximate and overshooting converts
directly into provider 429s and wasted retries. Per-model `rpd`/`tpm` live in
`config/default.yaml` and are read by both the runner and `estimate_cost.py`, so
there is one source of truth.

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
