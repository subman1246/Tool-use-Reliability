# BFCL's multi-turn scorer does not require a single fixed reference trajectory

Read from BFCL's scoring source, not inferred from its documentation or its data format.
This matters because it determines whether a BFCL L_d result falls inside or outside the
paper's own scope condition (Section 5.3), and the answer is "outside" — which constrains
what the Phase 2 replication can be claimed to show.

**Source:** `bfcl_eval/eval_checker/multi_turn_eval/multi_turn_checker.py`, gorilla repo
at `main` (the same ref the execution environment is vendored from).

## What the checker actually does

`multi_turn_checker` runs two checks per turn, and a third that is **commented out**.

**1. `state_checker` — final environment state, not trajectory.**

> "Checks if, after executing the function calls, the model_instance has the same state
> (defined by the attributes) as the ground_truth_instance."

It compares object attributes of the model's API instances against the ground truth's. Any
call sequence that lands the environment in the same state passes, regardless of how it
got there.

**2. `response_checker` — unordered containment, not equality.**

> "Checks if the model_response is a subsequence of the ground_truth_response."

It calls `_is_subsequence_unordered(ground_truth_response_list, model_response_list)`,
whose docstring is:

> "Checks if all elements of list1 are present in list2, regardless of order."

Two consequences, both explicit in the code. Order is not enforced — the in-source comment
says *"We don't need to enforce the order of the responses, because many entries have
parallel operations, and so the model can execute them in any order."* And the model may
make **additional** calls beyond the ground truth: the check is containment of the ground
truth within the model's results, not equality. The model's list is also accumulated
across all previous turns (`all_turn_model_execution_results`), so a call made early
satisfies a later turn's requirement.

**3. `method_invoke_order_checker` — written, but disabled.**

The one check that would constrain the trajectory is commented out at the call site. Its
own docstring concedes the point anyway: *"model_instance can call additional methods, but
not skip any method that the ground_truth_instance called"*, and *"this function only
checks for the method names and not the arguments."*

## The answer

**BFCL's multi-turn scorer permits alternative valid call sequences.** It accepts
reordering, extra calls, and any path reaching the same environment state. The
`possible_answer` file supplies one reference sequence, but the scorer uses it as a
*state-and-response target*, not as a trajectory to match call-for-call.

## Why this connects to Section 5.3

The paper already carves this class out. Section 5.3:

> "Final-state unit tests and milestone-DAG evaluators can also accept alternative valid
> trajectories. Our claim is therefore restricted to canonical scoring under an
> empirically or structurally small $\epsilon_k$; it is not a blanket claim about
> execution match, AST match, or fixed references."

And Section 2 lists τ-bench (final state), AppWorld (state-based unit tests), and
ToolSandbox (milestone DAGs) as evaluators that "avoid the unreachable-canonical-target
failure". **BFCL multi-turn belongs on that list and is not currently on it.** That is a
factual gap in the related-work characterisation, independent of anything Phase 2 measures.

## What this does and does not license

Phase 2 scores BFCL trajectories with *our* fixed-gold per-call scorer, which is not
BFCL's. That is legitimate and is the point — L_d is defined in terms of canonical
per-call agreement under two histories — but it means:

- The L_d numbers are **not** BFCL leaderboard numbers and must never be presented as
  comparable to published BFCL scores. They are our scorer applied to BFCL's tasks.
- A BFCL L_d result does **not** extend the paper's central proposition to BFCL. The
  proposition is about canonical scoring under small $\epsilon_k$; BFCL's own evaluator
  sits in the excluded class, so the tasks are being used as a realistic task *source*,
  not as a case where the theorem's antecedent has been shown to hold.
- What it can show is whether the **depth-dependent propagation loss** that the synthetic
  suite exhibits also appears on realistic multi-step tool-use tasks — a generalisation of
  the empirical L_d trend, not of the theorem.

A stronger claim would require establishing that $\epsilon_k$ is small on these tasks,
which needs the intent oracle that Phase 2 was explicitly scoped not to build.
