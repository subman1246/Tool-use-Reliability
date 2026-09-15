"""Assigned repair: the arm must measure the trajectory, not the next call.

Why it exists. This arm replaced the elected-repair arm after two pilots showed the model
was choosing when to repair on prompt salience rather than on evidence. Assigning the
repair removes that, but it introduces three ways the arm could be quietly worthless, all
of which would still produce a publishable-looking table:

  it is Appendix E again      Appendix E already establishes that handing back the true
                              value restores the NEXT call. If the arm's verdict is
                              decided by the call at repair_at, it reports a known result
                              with more machinery. The verdict must depend on the steps
                              after that one.

  the branches are not paired if the repaired and unrepaired branches do not share an
                              identical history up to the fork, the contrast includes
                              whatever else drifted, exactly the failure the injection
                              arm's one-message test exists to catch.

  the repair does not repair  if handing back the canonical value does not actually let a
                              competent agent rejoin the gold trajectory, then a low
                              recovery rate measures the mechanism, not the model. This is
                              the same hazard test_perfect_policy_is_inert pins down for
                              injection, and the same one the no-op-resync test pins down
                              for the elected arm.

Asserted here:

  1. a perfect conditional-on-state policy completes the ORIGINAL task on the repaired
     branch and cannot on the unrepaired one -- the arm separates what it claims to
  2. the two branches differ in exactly one message, the handed-back value at repair_at
  3. the shared prefix is byte-identical across branches, because it was run once
  4. the verdict is not decided at repair_at: a policy that is correct at repair_at and
     then drifts must score as NOT recovered, which is what distinguishes this arm from
     Appendix E's corrected branch
  5. the guard rails reject repair_at too late to leave two downstream steps, repair_at
     at or before inject_at, and linear tasks
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tur.harness.executor import FeedbackMode                          # noqa: E402
from tur.harness.runner import (MockBackend, run_repair_pair,          # noqa: E402
                                repair_outcome,
                                _REP_REPAIRED, _REP_UNREPAIRED,
                                _expected_args_given_held,
                                _expected_tool_given_ref)
from tur.tasks.dag import generate_routing_suite                       # noqa: E402

DEPTH = 6
N = 12
SEED = 90210
INJECT_AT = 1
REPAIR_AT = 3          # leaves steps 3,4,5 -- three downstream, above the minimum of two


def _tasks(n=N, depth=DEPTH):
    return generate_routing_suite([depth], n, 1, base_seed=SEED)


def _perfect(task, step, ref, attempt):
    """Applies the routing rule correctly to whatever value it is actually holding."""
    return (_expected_tool_given_ref(task, step, ref),
            _expected_args_given_held(task, step, ref), True)


def test_perfect_policy_completes_only_the_repaired_branch() -> None:
    tasks = _tasks()
    rep_done = unrep_done = 0
    for task in tasks:
        recs = run_repair_pair(task, MockBackend(_perfect), INJECT_AT, REPAIR_AT,
                               "parity_flip", "uniform", FeedbackMode.STRUCTURED, 1)
        out = repair_outcome(task, recs, INJECT_AT, REPAIR_AT)
        rep_done += out[_REP_REPAIRED]["completed_original"]
        unrep_done += out[_REP_UNREPAIRED]["completed_original"]

    assert rep_done == len(tasks), (
        "a perfect policy finished the ORIGINAL task on only %d of %d repaired branches. "
        "Handing back the canonical value does not let a competent agent rejoin the gold "
        "trajectory, so any recovery rate measured here would report the mechanism's "
        "brokenness rather than the model's behaviour." % (rep_done, len(tasks)))
    assert unrep_done == 0, (
        "a perfect policy finished the original task on %d unrepaired branches. The "
        "corruption is recoverable without the repair, so the contrast measures nothing."
        % unrep_done)
    print("  ok  perfect policy completes %d/%d repaired and 0/%d unrepaired branches"
          % (rep_done, len(tasks), len(tasks)))


def _branch_messages(task, policy):
    """Capture the message list each branch was run on, by recording backend calls."""
    seen = []

    class Recording(MockBackend):
        def complete(self, messages, tools, mode):
            seen.append([{k: v for k, v in m.items() if not k.startswith("_")}
                         for m in messages])
            return super().complete(messages, tools, mode)

    run_repair_pair(task, Recording(policy), INJECT_AT, REPAIR_AT, "parity_flip",
                    "uniform", FeedbackMode.STRUCTURED, 1)
    return seen


def test_branches_differ_in_exactly_one_message() -> None:
    task = _tasks(n=1)[0]
    seen = _branch_messages(task, _perfect)
    # The call made AT repair_at, identified by it being the LAST message: matching
    # anywhere in the history would also pick up every later call on the same branch,
    # and then both "branches" compared would come from the repaired one.
    marker = "[step %d]" % REPAIR_AT
    forks = [m for m in seen if m and m[-1].get("content") == marker]
    assert len(forks) == 2, ("expected exactly one call at repair_at per branch, got %d"
                             % len(forks))
    a, b = forks
    n = min(len(a), len(b))
    diffs = [i for i in range(n) if a[i] != b[i]]
    assert len(diffs) == 1, (
        "the two branches differ in %d messages, expected exactly 1 (the handed-back "
        "value). Differing indices: %s" % (len(diffs), diffs[:5]))
    d = diffs[0]
    assert a[d]["content"].startswith("result: ") and b[d]["role"] == "user", (
        "the differing message is %r / %r, expected the repair result line"
        % (a[d], b[d]))
    print("  ok  branches differ in exactly one message, the handed-back value")


def test_shared_prefix_is_run_once_and_identical() -> None:
    task = _tasks(n=1)[0]
    recs = run_repair_pair(task, MockBackend(_perfect), INJECT_AT, REPAIR_AT,
                           "parity_flip", "uniform", FeedbackMode.STRUCTURED, 1)
    a = sorted((r for r in recs if r.run_mode == _REP_REPAIRED), key=lambda r: r.step)
    b = sorted((r for r in recs if r.run_mode == _REP_UNREPAIRED), key=lambda r: r.step)
    # steps 0..inject_at-1 are teacher-forced into the prefix and never scored
    n_scored = task.depth - INJECT_AT
    assert len(a) == len(b) == n_scored, (len(a), len(b), n_scored)
    for x, y in zip(a, b):
        if x.step >= REPAIR_AT:
            continue
        key = lambda r: (r.step, r.tool, r.selection_matches_gold,
                         r.args_correct_strict, r.held_ref, r.held_out, r.diverged_in)
        assert key(x) == key(y), (
            "pre-repair step %d differs between branches (%r vs %r); the shared segment "
            "was not actually shared" % (x.step, key(x), key(y)))
    print("  ok  the pre-repair segment is identical across branches (%d steps)"
          % (REPAIR_AT - INJECT_AT))


def test_verdict_is_not_decided_at_the_repair_step() -> None:
    """Correct at repair_at, then drifting, must NOT count as recovered.

    This is the test that separates the arm from Appendix E. Appendix E's corrected branch
    would score this policy as a full success, because it only looks at the call right
    after the value is handed back.
    """
    task = _tasks(n=1)[0]

    def correct_then_drift(t, step, ref, attempt):
        if step == REPAIR_AT:                      # the Appendix E call: do it right
            return (_expected_tool_given_ref(t, step, ref),
                    _expected_args_given_held(t, step, ref), True)
        if step > REPAIR_AT:                       # then wander off
            bad = (ref + 1) % 100000
            return _expected_tool_given_ref(t, step, bad), {"ref": bad}, True
        return _perfect(t, step, ref, attempt)

    recs = run_repair_pair(task, MockBackend(correct_then_drift), INJECT_AT, REPAIR_AT,
                           "parity_flip", "uniform", FeedbackMode.STRUCTURED, 1)
    out = repair_outcome(task, recs, INJECT_AT, REPAIR_AT)[_REP_REPAIRED]
    at_repair = [r for r in recs
                 if r.run_mode == _REP_REPAIRED and r.step == REPAIR_AT][0]

    assert at_repair.selection_matches_gold and at_repair.args_correct_strict, (
        "the policy was supposed to make the repair-step call correctly")
    assert not out["completed_original"], (
        "a trajectory that was correct at the repair step and then drifted scored as "
        "having completed the original task. The verdict is being decided by the call at "
        "repair_at, which is Appendix E's corrected-branch result, not a downstream "
        "recovery measure.")
    assert out["n_downstream"] >= 2, out["n_downstream"]
    print("  ok  correct-at-repair-then-drift scores as NOT recovered "
          "(%d downstream steps scored)" % out["n_downstream"])


def test_guard_rails() -> None:
    task = _tasks(n=1)[0]
    bad = [
        (INJECT_AT, task.depth - 1, "repair_at leaves only one downstream step"),
        (INJECT_AT, task.depth, "repair_at at depth"),
        (2, 2, "repair_at equal to inject_at"),
        (3, 2, "repair_at before inject_at"),
        (0, 3, "inject_at at step 0"),
    ]
    for inj, rep, why in bad:
        try:
            run_repair_pair(task, MockBackend(_perfect), inj, rep, "parity_flip",
                            "uniform", FeedbackMode.STRUCTURED, 1)
        except ValueError:
            continue
        raise AssertionError("guard rail did not reject: %s (inject_at=%d repair_at=%d)"
                             % (why, inj, rep))

    linear = generate_routing_suite([DEPTH], 1, 1, base_seed=SEED)[0]
    del linear.branches
    try:
        run_repair_pair(linear, MockBackend(_perfect), INJECT_AT, REPAIR_AT)
    except ValueError:
        pass
    else:
        raise AssertionError("guard rail did not reject a task without branches")
    print("  ok  guard rails reject every invalid (inject_at, repair_at) and linear tasks")


def main() -> None:
    test_perfect_policy_completes_only_the_repaired_branch()
    test_branches_differ_in_exactly_one_message()
    test_shared_prefix_is_run_once_and_identical()
    test_verdict_is_not_decided_at_the_repair_step()
    test_guard_rails()
    print("\nassigned repair is paired, is genuinely repairing, and is scored downstream "
          "to completion rather than at the repaired call")


if __name__ == "__main__":
    main()
