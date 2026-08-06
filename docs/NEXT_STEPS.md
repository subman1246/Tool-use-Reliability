# Next Steps

Ordered checklist for moving from the simulated-suite validation to a real
model comparison and a submittable paper. Do these roughly in order; each
builds on the last.

## 1. Environment setup (Claude Code / any networked machine)

- [ ] Clone the repo, `pip install -r requirements.txt && pip install -e .`
- [ ] `cp .env.example .env`, fill in whichever provider keys you'll use
      (Groq for open-weight models is cheap and fast; add OpenAI/Anthropic
      keys for the proprietary reference tier).
- [ ] Confirm `tests/test_smoke.py` and `tests/test_regression.py` still pass
      in the new environment before touching real APIs — this confirms the
      environment itself is fine before spending money.

## 2. Finalize the model suite

- [ ] Open `config/default.yaml` and replace the placeholder model list with
      real, currently-available model identifiers. Check each provider's
      current model list directly — names and availability drift, and the
      methodology's suggested models (Qwen2.5, Llama-3.1, etc.) may have
      newer versions available by the time you run this.
- [ ] Fill in `family` and `scale` for each model entry — the pi-vs-scale
      figure (H2) reads these directly.
- [ ] Decide the function-calling-tuned vs. base-model pair for the tuning
      axis (e.g. an xLAM or Hammer variant vs. its instruct base) and confirm
      both are actually available on your chosen host.

## 3. Budget before running anything real

- [ ] `PYTHONPATH=src python scripts/estimate_cost.py` — gives a worst-case
      and typical-case dollar estimate for the configured sweep. Update the
      `PRICE_PER_1M` table in that script with current provider pricing
      first; it's a rough placeholder.
- [ ] If the estimate is larger than expected, reduce `per_depth` or `seeds`
      in `config/default.yaml` rather than cutting depths or models — depth
      coverage is what the propagation analysis needs most.

## 4. Pilot run (small, cheap, catches real-world surprises)

- [ ] `PYTHONPATH=src python scripts/run_real_suite.py --pilot` — 2 depths,
      5 tasks/depth, 1 seed, no fitting. This is cheap and will surface the
      things that only show up with real models:
  - Does the model actually follow the uniform JSON-call format, or does it
    wrap it in markdown/prose that the parser needs to handle better?
  - Are backend failures (rate limits, timeouts) showing up in the printed
    stats? If `n_failures` is high, increase `max_retries`/`base_delay` in
    `LiteLLMBackend` or slow down the request rate.
  - Do selection/argument scores look sane (not near-zero from a parsing
    mismatch, not suspiciously perfect from a scoring bug)?
- [ ] Spot-check a handful of raw JSONL rows by hand. This is the single best
      use of five minutes before a full run.

## 5. Harden anything the pilot surfaces

Likely candidates, roughly in order of likelihood:

- [ ] **Prompt format robustness.** Real models often wrap JSON in code
      fences or add a sentence before/after it. `_extract_json` in
      `runner.py` does a naive brace-matching extraction — if a pilot shows
      real parse failures that are clearly formatting, not genuine model
      mistakes, improve this before it contaminates `f_syn`.
- [ ] **Native calling-mode field names.** `LiteLLMBackend`'s native-mode
      parsing assumes the LiteLLM-normalized `tool_calls` shape; confirm this
      actually works per-provider in the pilot (Groq, OpenAI, Anthropic can
      all differ slightly even through LiteLLM).
- [ ] **Rate limiting.** If running many models/depths back to back hits
      provider rate limits, consider adding a request-rate cap or running
      models sequentially with a cooldown, rather than relying on retry/backoff
      alone to absorb it.

## 6. Full real run

- [ ] `PYTHONPATH=src python scripts/run_real_suite.py` — this is the real
      experiment. It writes `data/results/real_<model>.jsonl`,
      `real_idata.pkl`, `real_meta.json`.
- [ ] `PYTHONPATH=src python -m tur.analysis.plots --tag real` — regenerates
      all seven figures against the real data.
- [ ] Check MCMC diagnostics first (printed during the fit, and re-checkable
      via the same `az.rhat`/`az.ess` snippet used in
      `docs/RESULTS_validation_run.md`). If divergences appear, raise
      `target_accept` in `run_real_suite.py` the same way we did for the
      validation run.

## 7. The calling-mode ablation

- [ ] Rerun a fixed subset (a handful of tasks per model, not the full sweep)
      with `--call-mode native` and compare argument-error rates against the
      `uniform` run. This quantifies how much of any model's apparent
      unreliability is a calling-mode artifact rather than a genuine
      limitation — Methodology Section 6's fairness control, and reviewers
      will ask for it if it's missing.

## 8. Real benchmark tasks (BFCL / tau-bench)

- [ ] Download the benchmark files you're targeting.
- [ ] Finish `src/tur/data/loaders.py` — it's a scaffold with the parsing
      structure and explicit `TODO`s for field-name mapping against the
      exact benchmark version you're using.
- [ ] For tau-bench specifically: it grades end-state, not per-call gold, so
      you need a reference trajectory (either from the benchmark's own
      annotations or a strong-model rollout used as a stand-in gold) to
      support invocation-level scoring rather than only end-to-end success.
- [ ] Run the same local/global protocol on this real subset and follow the
      pre-registered synthetic-vs-real agreement rule from the methodology
      (Section 8 of `docs/METHODOLOGY.md`): if `L_t`/identifiability
      disagree between synthetic and real, real data becomes primary for
      every headline claim.
- [ ] This is also the direct test of whether real-benchmark arguments (which
      have graded closeness, unlike our integer chains) give `pi` better
      identifiability than the synthetic suite did — flagged as an open
      question in the validation results and worth checking early, since it
      changes how confidently you can report the parametric fit.

## 9. Writing the paper

- [ ] `docs/PAPER_DRAFT.md` has Introduction, Related Work, Method, and the
      full pipeline-validation section (Section 4) already written. Sections
      5 (Results), 6 (Discussion), and 8 (Conclusion) are templated with the
      exact structure to fill in once step 6 is done.
- [ ] Before anything else, **verify every citation** in the Related Work /
      References section — this was flagged when the literature review was
      first drafted and is still outstanding. Pull up each paper and confirm
      author list, venue, and year.
- [ ] Once real results are in, fill Section 5 using the same tables/figures
      pattern as `docs/RESULTS_validation_run.md` — that document is
      essentially a worked example of exactly the prose and table structure
      the real results section needs.
- [ ] Convert from Markdown to the target venue's LaTeX template last, not
      first — much easier to edit prose in Markdown and do a mechanical
      conversion once the content is stable than to fight LaTeX while still
      figuring out what the results say.
- [ ] For TMLR: no page limit, but the certification-focused review process
      cares a lot about the claims being precisely as strong as the evidence
      supports — the identifiability-limitation framing in Section 4.3/7 is
      exactly the kind of honest scoping that process rewards. For IEEE
      venues, check the specific conference/journal's page limit and template
      before finalizing figure sizing.

## 10. Optional but valuable if time allows

- [ ] A negative-control baseline (e.g., a model that ignores context and
      answers from the tool description alone, or simple random valid-schema
      guessing) to calibrate what "zero skill" looks like on this task suite.
- [ ] A determinism check: rerun one model's pilot with caching disabled and
      confirm results are stable at temperature 0 (or characterize the
      variance if not — some providers are not perfectly deterministic even
      at temperature 0).
- [ ] Parallelizing `run_real_suite.py` (currently sequential) with a thread
      pool or LiteLLM's async client if the full sweep's wall-clock time is
      inconvenient — not necessary for correctness, only for speed.
