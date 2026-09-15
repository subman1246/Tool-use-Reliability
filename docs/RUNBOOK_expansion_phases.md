# Expansion phases: state, and what to run when each lands

Written so the in-flight work is recoverable by someone who was not here. Both sweeps are
detached (`nohup`), so they survive the session that launched them; what does not survive
is the context for reading their output, which is what this file is.

Rate limits, not compute, are the binding constraint throughout. Groq's daily bucket
refills continuously at `TPD/86400` tokens per second, so a stalled-looking run is usually
a run waiting for its bucket, not a broken one. Check `(cap-stopped)` in the log before
assuming otherwise.

---

## Phase 1 — assigned-repair arm (`allam-2-7b`)

**Running.** Driver: `scripts/drive_bfcl.sh`'s counterpart for this arm is
`scripts/resume_until_done.py --hours 60 --interval 1200 --tag repair --extra --config
config/repair.yaml`.

**Report with:**
```
py scripts/report_repair.py data/results/repair_groq_allam-2-7b.jsonl
```

**Headline is `completed_original`, repaired vs unrepaired**, 89% paired bootstrap over
pairs. Paired is structural, not stylistic: the two branches share a pre-repair segment
that was run once and inherited, so the pair is the independent unit.

**State at last check:** depth 6 complete at n=37 and **written up as Appendix F**
(commit `5270341`); depth 8 in progress, ~17h remaining.

**Decision rule when depth 8 lands — do not improvise this:**

- Add the depth-8 row to Appendix F's table.
- If `completed_original`'s interval **still includes zero**: keep the agreed framing —
  a null on the headline measure, a positive on downstream gold agreement, and *not*
  written up as a positive result overall. No check-in needed; this is the expected path.
- If `completed_original`'s interval **excludes zero at depth 8**: STOP. That strengthens
  the finding from a dissociation-with-one-supported-half into a positive result, and the
  wording must not be silently upgraded. Report it and wait.

**Depth-6 result, already in hand:**

Depth-6 ONLY (n=37). These are the figures in Appendix F. An earlier report of this arm
pooled depths 6 and 8 and gave 0.109 / 0.322 / 0.293 for the three measures; those are
**not** the depth-6 numbers and must not be quoted as such. `report_repair.py` prints the
pooled row first, so re-filter to depth 6 before copying anything into the paper.

| measure | repaired | unrepaired | delta | 89% interval | b/c |
|---|---|---|---|---|---|
| completed_original | 0.054 | 0.000 | +0.054 | [+0.000, +0.108] | 2/0 |
| downstream gold agreement | 0.126 | 0.000 | +0.126 | [+0.063, +0.198] | 9/0 |
| downstream conditional | 0.342 | 0.297 | +0.045 | [−0.018, +0.108] | 6/3 |

Read together: the repair restores per-call agreement with gold (interval excludes zero)
but does not restore completion of the original task (2 events in 37, lower bound at
zero), while conditional competence is unchanged either way. Returning to the canonical
state and recovering the goal come apart. **The completion delta rests on 2 successes and
does not exclude zero — do not report it as an effect.** If depth 8 also lands near the
floor, the honest framing is a null on the headline with a positive on the supporting
measure, not a positive result.

Allocation note: the nested allocator trimmed 40→37 per depth to fit `budget_days: 3`, so
the achieved n is 37, not the 40 the cost estimate assumed. State that rather than
presenting 37 as the design.

---

## Phase 2 — BFCL L_d arm (`gpt-oss-120b`)

**SINGLE-MODEL CHECK, NOT COMPARATIVE.** Three of the paper's five models are retired from
Groq; `gpt-oss-20b` has no synthetic baseline to compare against. This framing must travel
with every number the arm produces — it is recorded in the run's meta file and in the
report banner for that reason.

**Running:** pilot at n=8, depth 6, verbose schema, via
`bash scripts/drive_bfcl.sh bfclpilot 8 60`.

**Report with:**
```
py scripts/report_bfcl_ld.py data/results/bfclpilot_groq_openai_gpt-oss-120b.jsonl
```

**The suppression rule is enforced in the script, not left to the reader.** If
`p_d < 0.30`, `L_d` and its interval are withheld from both the console and the saved
JSON, and only `p_d` and `g_d` are reported. This is deliberate: the smoke task showed
`p_d = 0.333`, close enough to the threshold that the full pilot may fall below it, and a
ratio published with a caveat gets quoted without the caveat.

**Decision rule when the pilot lands — do not improvise this either:**

- If `p_6 >= 0.30`: the suppression rule does not fire, `L_6` is reportable, and the
  n=28 scale-up proceeds **automatically**. It is pre-approved; no separate go-ahead.
  ```
  bash scripts/drive_bfcl.sh bfcl28 28 200
  ```
- If `p_6 < 0.30`: `L_6` is withheld by the script. Write up `p_6` and `g_6` alone as the
  finding. **Do not scale to n=28 on a suppressed pilot** — stop and report instead. A
  4.9-day run whose ratio is already known to be unreportable is spend without a result.
- Either way, once final numbers exist: **Appendix G**, same register as F — single-model,
  non-comparative, explicitly not BFCL leaderboard numbers, with the scorer-structure
  point referenced from Related Work rather than re-argued. Plus one main-text sentence
  carrying the actual number, not a bare pointer.

Keep `verbose` schema. `compact` is ~30% cheaper but the synthetic arms used `verbose`,
and comparing BFCL L_d against synthetic L_d is the point — changing the rendering would
confound exactly that comparison.

**Cost reference** (depth 6, verbose, 200,000 TPD × 0.8):

| n | calls | tokens | TPD-days |
|---|---|---|---|
| 8 | 96 | 257,714 | 1.61 |
| 28 | 336 | 781,953 | 4.89 |
| 38 (all) | 456 | 1,110,768 | 6.94 |

The full curated subset across depths 6–10 is 24 days and is not viable on this model.

---

## Phase 0 — equivalence gate: **CLOSED, FAILED**

Local-quantized `meta-llama/Llama-3.1-8B-Instruct` is **not** equivalent to the frozen
`groq/llama-3.1-8b-instant` arm. Fails at depths 6 and 8 in teacher-forced, and 4, 6 and 8
in free. Local exceeds frozen in all ten cells with the gap widening by depth.

**Consequence, already decided:** local results are their own explicitly-labelled
condition. Nothing merges into the existing tables. No local number is compared against a
frozen one as though they measured the same thing. Pooling would have moved L_6 from 0.687
to 0.518.

Verdict at `data/results/localval_equivalence.json`.

---

## Explicitly NOT to be started in this pass

Shuffle control, calling-mode ablation, transformed-argument condition, full-scale linear
control, and local-quantized Phases 3/4. All are built or designed and idle. This pass
writes up what exists; it does not open new experimental fronts. Do not start any of them
on the reasoning that the harness is already there — that it runs is why the line has to
be held deliberately.

## Phases 3/4 on local-quantized llama-3.1-8b — **HELD**

Do not start. They were conditional on the equivalence gate passing, and it failed. They
could still run self-contained *within* the local condition, but that is a different claim
than the one the gate was meant to license. Revisit after Phase 1's full arm and Phase 2's
pilot both report.

---

## Closing checklist, when both arms are in

1. Compile; confirm zero errors; report page count and where main content, references and
   appendices fall.
2. Update `docs/PAPER_FLAGS.md` and this file so nothing still reads "pending".
3. Commit.
4. The standalone 9-page arXiv version (real names, no conference template) predates all
   of this and is **stale**. Do not regenerate it — confirm it needs a fresh pass once
   `main.tex` is final; the author handles that separately.

## Paper follow-ups

`docs/PAPER_FLAGS.md` — items 1 (BFCL Related Work misclassification) and 2
(equivalence-gate failure) are **APPLIED** in commit `4b9a7fe` and compile clean. Item 3
(elected repair chosen by prompt position) is **APPLIED** as a paragraph of Appendix F in
commit `5270341`. The flags file is now a provenance record of what was written and why —
do not re-draft from it.
