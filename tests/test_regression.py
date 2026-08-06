"""Regression tests for the scoring and state-tracking fixes.

These lock in behaviour that was previously wrong and would silently corrupt
the error decomposition if it regressed.
"""

import numpy as np

from tur.tasks.dag import generate_task, generate_routing_task
from tur.harness.runner import MockBackend, run_free
from tur.harness.executor import FeedbackMode
from tur.analysis.aggregate import aggregate_by_depth, bootstrap_L_ci


def test_conditional_selection_not_blamed_for_propagation():
    """An agent that applies the routing rule correctly to the (poisoned) ref
    it actually holds must NOT be scored as a selection error. Only its
    divergence from gold should be recorded, via selection_matches_gold."""
    task = generate_routing_task("rt", depth=4, distractor_level=0, seed=2)

    def faithful(task, step, ref, attempt):
        even_t, odd_t = task.branches[step]
        correct = (even_t if ref % 2 == 0 else odd_t).name
        if step == 1:
            return correct, {"ref": ref + 1}, True  # inject value corruption
        return correct, {"ref": ref}, True

    recs = run_free(task, MockBackend(faithful), "uniform",
                    FeedbackMode.STRUCTURED, max_retries=0)

    # every step: the agent was faithful to the rule given its own ref
    assert all(r.selection_correct for r in recs), \
        "faithful routing must not be scored as a selection error"
    # but later steps did diverge from the gold trajectory
    assert any(not r.selection_matches_gold for r in recs[2:]), \
        "divergence from gold should still be recorded"
    print("conditional selection: OK")


def test_stalled_chain_is_flagged():
    """If a step exhausts retries without ever executing, the chain stalls and
    downstream steps must be marked stalled_in rather than silently treated as
    ordinary poisoned steps."""
    task = generate_task("t", depth=4, distractor_level=0, seed=2)

    def always_bad_at_1(task, step, ref, attempt):
        tool = task.gold[step].tool
        if step == 1:
            return tool, {}, True   # schema-invalid, never executes
        return tool, {"ref": ref}, True

    recs = run_free(task, MockBackend(always_bad_at_1), "uniform",
                    FeedbackMode.STRUCTURED, max_retries=1)

    assert not recs[1].executed, "step 1 should never execute"
    assert recs[2].stalled_in, "step after a stall must be flagged stalled_in"
    print("stalled chain flagging: OK")


def test_f_syn_uses_fresh_errors_only():
    """f_syn must be estimated from fresh errors on clean contexts, so that
    propagation-induced failures don't inflate the semantic share."""
    task_recs = []
    # one clean-context syntactic error, plus several poisoned downstream steps
    rows = [
        dict(task_id="a", depth=4, step=0, run_mode="free", call_mode="uniform",
             tool="x", selection_correct=True, selection_matches_gold=True,
             args_correct_strict=False, args_correct_soft=False,
             error_type="syntactic", n_attempts=1, executed=False,
             context_clean_in=True, recovered=False, stalled_in=False),
    ] + [
        dict(task_id="a", depth=4, step=s, run_mode="free", call_mode="uniform",
             tool="x", selection_correct=True, selection_matches_gold=False,
             args_correct_strict=False, args_correct_soft=False,
             error_type="semantic", n_attempts=1, executed=True,
             context_clean_in=False, recovered=False, stalled_in=False)
        for s in (1, 2, 3)
    ]
    # a teacher-forced row so p_t is defined
    rows.append(dict(task_id="a", depth=4, step=0, run_mode="teacher_forced",
                     call_mode="uniform", tool="x", selection_correct=True,
                     selection_matches_gold=True, args_correct_strict=True,
                     args_correct_soft=True, error_type="none", n_attempts=1,
                     executed=True, context_clean_in=True, recovered=False,
                     stalled_in=False))
    task_recs.extend(rows)

    stats = aggregate_by_depth(task_recs, [4])
    s = stats[0]
    # only the one clean-context error counts as fresh, and it was syntactic
    assert s.n_fresh_errors == 1, f"expected 1 fresh error, got {s.n_fresh_errors}"
    assert s.f_syn == 1.0, f"expected f_syn=1.0, got {s.f_syn}"
    print("fresh-error f_syn: OK")


def test_bootstrap_ci_sane():
    """Bootstrap CI should bracket the point estimate and be finite."""
    rows = []
    for i in range(60):
        rows.append(dict(task_id=f"t{i}", depth=4, step=0, run_mode="teacher_forced",
                         call_mode="uniform", tool="x", selection_correct=True,
                         selection_matches_gold=True,
                         args_correct_strict=(i % 10 != 0), args_correct_soft=True,
                         error_type="none", n_attempts=1, executed=True,
                         context_clean_in=True, recovered=False, stalled_in=False))
        rows.append(dict(task_id=f"t{i}", depth=4, step=0, run_mode="free",
                         call_mode="uniform", tool="x", selection_correct=True,
                         selection_matches_gold=True,
                         args_correct_strict=(i % 4 != 0), args_correct_soft=True,
                         error_type="none", n_attempts=1, executed=True,
                         context_clean_in=True, recovered=False, stalled_in=False))
    L, lo, hi = bootstrap_L_ci(rows, 4, n_boot=300, seed=0)
    assert np.isfinite(L) and np.isfinite(lo) and np.isfinite(hi)
    assert lo <= L <= hi, f"CI [{lo},{hi}] must bracket point estimate {L}"
    print(f"bootstrap CI: OK (L={L:.3f}, CI=[{lo:.3f}, {hi:.3f}])")


if __name__ == "__main__":
    test_conditional_selection_not_blamed_for_propagation()
    test_stalled_chain_is_flagged()
    test_f_syn_uses_fresh_errors_only()
    test_bootstrap_ci_sane()
    print("\nALL REGRESSION TESTS PASSED")
