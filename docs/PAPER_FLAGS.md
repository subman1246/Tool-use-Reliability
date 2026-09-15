# Flagged for a later paper pass

Things found during the expansion phases that the paper should account for. Each is
recorded with enough detail and provenance to write from, rather than drafted early:
drafting before the shaping evidence arrives tends to freeze a claim prematurely.

**Status: items 1 and 2 are APPLIED** in commit `4b9a7fe` and compile clean (0 errors, 0
overfull boxes, 14 pages, 9-page main-content boundary held). They are kept here as the
provenance record for what was written and why — do not re-draft them. Item 3 remains
outstanding, pending the repair arm's full result.

---

## 1. Related Work mischaracterises BFCL (factual correction) — APPLIED `4b9a7fe`

**Where:** Section 2, the paragraph beginning *"Not all agent evaluation uses one canonical
trajectory."* It lists τ-bench (final state), AppWorld (state-based unit tests) and
ToolSandbox (milestone DAGs) as evaluators that accept alternative trajectories. BFCL is
currently grouped with the fixed-reference benchmarks instead.

**The correction:** BFCL multi-turn belongs in the alternative-trajectory group. Read from
`bfcl_eval/eval_checker/multi_turn_eval/multi_turn_checker.py` (gorilla repo at `main`),
not inferred from its documentation or its data format:

- `state_checker` compares **final environment state** — object attributes of the model's
  API instances against the ground truth's — not the call trajectory.
- `response_checker` uses `_is_subsequence_unordered(ground_truth, model)`, documented as
  *"Checks if all elements of list1 are present in list2, regardless of order."* Reordering
  passes; **extra model calls pass**, since it is containment rather than equality; and the
  model's results accumulate across turns, so a call made early satisfies a later turn.
- **The citable detail:** `method_invoke_order_checker` — the one check that would enforce
  a trajectory — is **commented out at the call site**. Its own docstring concedes the
  design anyway: *"model_instance can call additional methods, but not skip any method that
  the ground_truth_instance called"*, and it *"only checks for the method names and not the
  arguments."*

**Why it matters beyond tidiness:** the `possible_answer` file supplies one reference
sequence, which is what makes BFCL *look* like a fixed-reference benchmark from the outside.
It is used as a state-and-response target, not as a trajectory to match call-for-call. The
paper's scope condition (Section 5.3) turns on exactly this distinction, so getting BFCL's
side of it wrong weakens the boundary the central result depends on.

**Supporting material:** `docs/NOTE_bfcl_scorer_structure.md` has the full reading with
quoted source.

**As shipped:** BFCL was removed from the fixed-reference sentence and added to the
alternative-trajectory paragraph, with the clause *"notwithstanding that it distributes one
reference call sequence per task"* — the `possible_answer` file is what makes BFCL look
fixed-reference from outside, so the objection is pre-empted rather than invited.

---

## 2. Limitations: the local-quantized equivalence gate failed, and by how much — APPLIED `4b9a7fe`

**Where:** Limitations, alongside the existing model-retirement count.

**The finding:** local-quantized `meta-llama/Llama-3.1-8B-Instruct` (4-bit nf4) was tested
for equivalence against the frozen `groq/llama-3.1-8b-instant` arm on matched tasks
(`base_seed=1000`, paired at task level). Under the verdict rule fixed before any data
existed — within 0.05 **and** an 89% paired interval containing zero, at every shared
depth, in both run modes — it **fails**:

- `p_d` (teacher-forced): depths **6** and **8** differ (+0.062 [+0.021, +0.103]; +0.082
  [+0.043, +0.125]).
- `g_d` (free): depths **4**, **6**, **8** differ (+0.062; +0.126 [+0.077, +0.177]; +0.082
  [+0.005, +0.163]). Depth 4's interval contains zero but its point difference exceeds the
  0.05 tolerance — a fail under the conjunctive rule, recorded as such rather than softened.

Local exceeds frozen in **all ten cells**, with the gap widening by depth: a systematic
offset, not sampling noise.

**The magnitude worth stating:** had the two been pooled, L_6 would have moved from 0.687
to 0.518 — roughly **a sixth of the headline propagation claim** ("the two 7–8B models lose
about 68% of baseline capability by depth 6"). The gate prevented a material
misstatement, which is itself the argument for pre-registering the rule.

**The honest caveat:** these were probably never the same artifact — Groq's served
`llama-3.1-8b-instant` (its own serving and quantization stack) against local
`meta-llama/Llama-3.1-8B-Instruct` at 4-bit nf4. Since that Groq model is now retired
(404), this is **unfalsifiable**: the frozen arm cannot be re-probed to isolate which
difference is responsible. The limitation is therefore about reproducibility under
provider retirement as much as about quantization.

**Provenance:** `data/results/localval_equivalence.json`,
`scripts/compare_local_vs_groq.py`.

**As shipped, plus two corrections this pulled in.** The retirement count went from two
models to three (`qwen3.6-27b`, verified by live probe) and the instance count from third
to fourth. That tally cited stale model identifiers as a prior instance, which appeared
nowhere in the paper — so Data Collected (`sec:data`) now states plainly that the
originally specified Qwen2.5/Llama-3.1-instruct generation had been retired before the
first run. All four counted instances now have a referent; previously a reader could not
reconstruct the number.

---

## 3. Elected repair is chosen by prompt position, not by state — OUTSTANDING

Already written up in full at `docs/FINDING_elected_repair.md`. Flagged here so the paper
pass does not miss it: 14/16 tasks called the repair tool when the rule was stated last
before step 0, 0/16 when the same rule sat inside the task description, with tasks, seeds,
gold trajectories and model held fixed. Zero repairs landed on a diverged step under either
prompt. Belongs in the recovery-arm write-up regardless of what the assigned-repair arm
returns.
