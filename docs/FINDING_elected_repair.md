# Elected repair is chosen by prompt position, not by state

A standalone result from two pilots that were run to validate an arm we then redesigned.
The arm changed; the observation stands on its own and should not disappear with it.

## What was offered

The recovery-enabled routing task (`src/tur/tasks/recovery.py`) adds one tool,
`check_state`, which returns the true canonical ref for the step the agent is on. It costs
a backend call, does not advance the task, and is usable once per task. The design intent
was that an agent which suspects its carried value is wrong can buy its way back onto the
gold trajectory — making "genuine recovery" separable from "continuing competently from a
wrong state", which Section 7 says the project cannot currently distinguish.

## The two pilots

Both: `allam-2-7b`, 16 tasks (8 at depth 4, 8 at depth 6), `base_seed=1000`, uniform call
mode, structured feedback, `max_retries=1`, one-shot budget. The tasks are identical
between pilots — same generator, same seed, same gold trajectories. The **only** change
that could affect election was where the repair rule was stated.

| | rule stated as the last turn before `[step 0]` | rule stated inside the task description |
|---|---|---|
| tasks calling `check_state` at all | **14 / 16** | **0 / 16** |
| — depth 4 | 8 / 8 | 0 / 8 |
| — depth 6 | 6 / 8 | 0 / 8 |
| calls issued at step 0 | 14 of 20 | — |
| calls landing on a diverged step | 6 (all refused, budget already spent) | 0 |
| repairs granted while actually diverged | **0** | **0** |

Every granted repair in the first pilot was issued at **step 0**, where the held value
equals the canonical value by construction — so each one returned the number already in
the prompt. The six calls that did land on a diverged step were all at step 3 of tasks
`r4_0` and `r4_3`, after those tasks had already spent their budget at step 0; each was
refused and re-issued until the call guard truncated the trajectory at 3 of 4 steps.

## What it shows

Election is driven by the salience of the instruction, not by any inference about state.
Moving one paragraph moved usage from near-universal to zero, with the task, the seeds,
the gold trajectories and the model held fixed. Neither setting produced a single repair
at a step where a repair would have helped.

The mechanism was not at fault. `tests/test_recovery_mechanism.py` establishes before any
spend that an always-repair oracle recovers the original trajectory in 100% of tasks, that
a never-repair policy is step-for-step identical to the free-running arm, that the budget
is enforced, and that resync restores the canonical value.

The likely reason is structural: **the task gives the agent no divergence signal.** Tool
feedback returns a number with nothing to check it against, so the model cannot form a
belief that it is off-track. Under a one-shot budget an agent with no such signal has only
two coherent policies — always ask, or never ask — and prompt salience selects between
them. That also means raising the budget does not fix it: a larger budget selects "always
ask", which resyncs before every step and collapses the free-running arm into the
teacher-forced arm.

## Caveat on the zero

"Repairs granted while diverged: 0" is robust in the second pilot (no calls were made at
all) but partly mechanical in the first. That pilot ran under pricing in which a step-0
resync consumed the budget; under the revised pricing — a resync issued while already
canonical does not charge (`run_recovery`, commit `48205ba`) — those tasks would have
retained their repair and might have spent it later. The claim that survives both pilots is
the **election** result, 14/16 against 0/16. The downstream zero is reported as what was
observed, not as an estimate of what an agent with budget remaining would do.

## Consequence

Repair was reassigned rather than elected: `run_repair_pair` corrupts at a fixed position,
runs free, hands the canonical value back at a fixed later position, and then scores the
trajectory to completion. See `config/repair.yaml` and `tests/test_repair_arm.py`.

## Provenance

- `data/results/recpilot_groq_allam-2-7b.jsonl` — first pilot (rule as last turn)
- `data/results/recpilot2_groq_allam-2-7b.jsonl` — second pilot (rule in task description)
- `scripts/report_recovery.py` — regenerates every number in the table above
- both pilots also record a teacher-forced arm in the same file, distinguished by
  `run_mode`; reading without that filter doubles the step count per task
