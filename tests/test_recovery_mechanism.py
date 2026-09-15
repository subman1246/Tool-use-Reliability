"""The repair mechanism must actually repair, and must be inert when unused.

Why it exists. This arm's entire claim is that it can finally distinguish "continuing
competently from an off-gold state" from "actually recovering". That distinction is only
real if the repair tool genuinely puts the agent back on the canonical trajectory. Two
ways it could be fake, both of which would produce publishable-looking numbers:

  the tool doesn't work   an always-repair policy would fail to reach the original goal,
                          and "recovery rate" would measure the mechanism's brokenness
                          rather than any model's behaviour.
  the arm isn't a control a never-repair policy must behave EXACTLY like the existing
                          free-running arm. If merely offering the tool changes the
                          trajectory, then recovery-arm numbers cannot be compared
                          against the free-running numbers already in the paper, and the
                          comparison is the whole point.

So, per the instruction that the validity test comes first:

  1. always-repair recovers the original trajectory with high probability
  2. never-repair is identical, step for step, to run_free on the same task
  3. the one-shot budget is enforced (a second resync is refused, not silently granted)
  4. resync restores the canonical value, verified directly against the gold trajectory
  5. recovery_outcome's three-way split disagrees where it should: a perfectly-continuing
     but never-repaired trajectory scores high on conditional and zero on recovered
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tur.harness.executor import FeedbackMode                        # noqa: E402
from tur.harness.runner import (MockBackend, run_free, run_recovery,  # noqa: E402
                                recovery_outcome,
                                _expected_args_given_held,
                                _expected_tool_given_ref)
from tur.tasks.dag import generate_routing_suite                     # noqa: E402
from tur.tasks.recovery import (RESYNC_TOOL, canonical_ref_at,       # noqa: E402
                                to_recovery_task)

DEPTH = 6
N = 15
SEED = 31337


def _tasks(n=N, depth=DEPTH):
    return [to_recovery_task(t) for t in
            generate_routing_suite([depth], n, 1, base_seed=SEED)]


def _perfect(task, step, ref, attempt):
    return (_expected_tool_given_ref(task, step, ref),
            _expected_args_given_held(task, step, ref), True)


def test_never_repair_matches_free_running() -> None:
    """Offering the tool must not change behaviour when it is not used."""
    plain = generate_routing_suite([DEPTH], N, 1, base_seed=SEED)
    diffs = 0
    for pt in plain:
        rt = to_recovery_task(pt)
        a = run_free(pt, MockBackend(_perfect), "uniform", FeedbackMode.STRUCTURED, 1)
        b = [r for r in run_recovery(rt, MockBackend(_perfect), "uniform",
                                     FeedbackMode.STRUCTURED, 1)
             if not r.used_resync]
        assert len(a) == len(b), ("%s: free produced %d steps, recovery produced %d"
                                  % (pt.task_id, len(a), len(b)))
        for x, y in zip(a, b):
            if (x.tool, x.selection_matches_gold, x.args_correct_strict,
                    x.held_ref) != (y.tool, y.selection_matches_gold,
                                    y.args_correct_strict, y.held_ref):
                diffs += 1
    assert diffs == 0, (
        "%d steps differ between the free-running arm and an unused-repair recovery "
        "arm. The recovery arm would not be comparable to the paper's existing "
        "free-running numbers." % diffs)
    print("  ok  never-repair is step-for-step identical to run_free (%d tasks)"
          % len(plain))


def test_always_repair_recovers_the_original_trajectory() -> None:
    """An oracle policy that errs once then repairs must finish the ORIGINAL task."""
    recovered = 0
    tasks = _tasks()
    for task in tasks:
        state = {"erred": False, "repaired": False}

        def policy(t, step, ref, attempt, _s=state):
            # err exactly once, at step 1, by sending a parity-flipped ref
            if step == 1 and not _s["erred"]:
                _s["erred"] = True
                bad = (ref + 1) % 100000
                return _expected_tool_given_ref(t, step, bad), {"ref": bad}, True
            # having erred, repair at the next opportunity
            if _s["erred"] and not _s["repaired"]:
                _s["repaired"] = True
                return RESYNC_TOOL, {}, True
            return (_expected_tool_given_ref(t, step, ref),
                    _expected_args_given_held(t, step, ref), True)

        recs = run_recovery(task, MockBackend(policy), "uniform",
                            FeedbackMode.STRUCTURED, 1)
        out = recovery_outcome(task, recs)
        assert out["used_resync"], "%s: policy never issued a resync" % task.task_id
        if out["recovered"]:
            recovered += 1

    rate = recovered / len(tasks)
    assert rate >= 0.9, (
        "always-repair recovered the original trajectory in only %.0f%% of tasks. The "
        "repair tool does not restore the canonical state, so any 'recovery rate' "
        "measured with it would be reporting a broken mechanism." % (100 * rate))
    print("  ok  always-repair recovers the original trajectory in %.0f%% of tasks"
          % (100 * rate))


def test_resync_restores_the_canonical_value() -> None:
    task = _tasks(n=1)[0]
    seen = {}

    def policy(t, step, ref, attempt, _s=seen):
        if step == 2 and "pre" not in _s:
            _s["pre"] = ref
            return RESYNC_TOOL, {}, True
        if step == 2 and "post" not in _s:
            _s["post"] = ref
        return (_expected_tool_given_ref(t, step, ref),
                _expected_args_given_held(t, step, ref), True)

    # divert the chain first so the held value at step 2 is genuinely wrong
    def diverting(t, step, ref, attempt, _s=seen):
        if step == 0:
            bad = (ref + 1) % 100000
            return _expected_tool_given_ref(t, step, bad), {"ref": bad}, True
        return policy(t, step, ref, attempt)

    run_recovery(task, MockBackend(diverting), "uniform", FeedbackMode.STRUCTURED, 1)
    gold2 = canonical_ref_at(task, 2)
    assert seen.get("post") == gold2, (
        "after resync the held value was %r, canonical is %r" % (seen.get("post"), gold2))
    assert seen.get("pre") != gold2, "the test did not actually diverge before resyncing"
    print("  ok  resync restores the canonical value (%r -> %r)"
          % (seen["pre"], seen["post"]))


def test_budget_is_enforced() -> None:
    task = _tasks(n=1)[0]
    calls = {"n": 0}

    def spammer(t, step, ref, attempt, _c=calls):
        _c["n"] += 1
        return RESYNC_TOOL, {}, True

    recs = run_recovery(task, MockBackend(spammer), "uniform",
                        FeedbackMode.STRUCTURED, 1)
    granted = [r for r in recs if r.used_resync and r.resync_remaining == 0]
    n_resync = sum(1 for r in recs if r.used_resync)
    assert n_resync >= 1
    assert task.resync_budget == 1
    assert all(r.resync_remaining >= 0 for r in recs if r.used_resync)
    assert len(granted) >= 1, "budget never reached zero"
    # and the loop terminates rather than spinning forever. The bound grew when no-op
    # resyncs stopped costing the budget: depth steps, one free no-op per step, the
    # charged repairs, plus retry slack.
    assert len(recs) <= 2 * task.depth + task.resync_budget + 2, len(recs)
    # the free allowance must not have become a second repair
    charged = [r for r in recs if r.used_resync and r.resync_charged]
    assert len(charged) <= task.resync_budget, (
        "%d resyncs consumed the budget under a spammer, budget is %d"
        % (len(charged), task.resync_budget))
    print("  ok  one-shot budget enforced and the loop terminates under a resync spammer")


def test_three_way_split_disagrees_where_it_should() -> None:
    """A competent-but-never-repaired trajectory: high conditional, zero recovery."""
    task = _tasks(n=1)[0]

    def diverge_then_continue(t, step, ref, attempt):
        if step == 0:
            bad = (ref + 1) % 100000
            return _expected_tool_given_ref(t, step, bad), {"ref": bad}, True
        return (_expected_tool_given_ref(t, step, ref),
                _expected_args_given_held(t, step, ref), True)

    recs = run_recovery(task, MockBackend(diverge_then_continue), "uniform",
                        FeedbackMode.STRUCTURED, 1)
    out = recovery_outcome(task, recs)
    assert out["diverged"], "the trajectory was supposed to diverge"
    assert not out["used_resync"]
    assert not out["recovered"], (
        "a trajectory that never repaired was scored as RECOVERED; the recovery "
        "measure is just re-reading conditional competence")
    assert out["conditional"] >= 0.8, out["conditional"]
    assert out["gold_agreement"] < out["conditional"], (
        "gold agreement (%.2f) should be below conditional (%.2f) after a divergence"
        % (out["gold_agreement"], out["conditional"]))
    print("  ok  three-way split separates: gold=%.2f conditional=%.2f recovered=%s"
          % (out["gold_agreement"], out["conditional"], out["recovered"]))


def test_free_no_op_resync_opens_no_extra_information() -> None:
    """Reflexive resyncing must buy nothing over asking once, at the moment it matters.

    Not charging the budget for a resync issued while the held value is already canonical
    fixed a real degeneracy -- the pilot model spent its single repair at step 0, where
    there is nothing to repair -- but it introduces a way to cheat: if an agent can call
    check_state before every step at no cost, it could use the free calls as a divergence
    detector, or accumulate more than one genuine repair, and the free-running arm would
    quietly become the teacher-forced arm.

    So the adversarial policy is run directly: err once, then emit check_state at the top
    of EVERY step. Its scored steps must come out identical, step for step, to the oracle
    that resyncs exactly once at the moment it is actually off-canonical, and it must not
    consume more than the budget in genuine repairs.
    """
    tasks = _tasks()
    mismatched = []
    for task in tasks:

        def oracle(t, step, ref, attempt, _s={}):
            if step == 1 and not _s.get("erred"):
                _s["erred"] = True
                bad = (ref + 1) % 100000
                return _expected_tool_given_ref(t, step, bad), {"ref": bad}, True
            if _s.get("erred") and not _s.get("repaired"):
                _s["repaired"] = True
                return RESYNC_TOOL, {}, True
            return (_expected_tool_given_ref(t, step, ref),
                    _expected_args_given_held(t, step, ref), True)

        def reflexive(t, step, ref, attempt, _s={"seen": set()}):
            # ask at the top of every step, whether or not anything is wrong
            if step not in _s["seen"]:
                _s["seen"].add(step)
                return RESYNC_TOOL, {}, True
            if step == 1 and not _s.get("erred"):
                _s["erred"] = True
                bad = (ref + 1) % 100000
                return _expected_tool_given_ref(t, step, bad), {"ref": bad}, True
            return (_expected_tool_given_ref(t, step, ref),
                    _expected_args_given_held(t, step, ref), True)

        a = run_recovery(task, MockBackend(oracle), "uniform",
                         FeedbackMode.STRUCTURED, 1)
        b = run_recovery(task, MockBackend(reflexive), "uniform",
                         FeedbackMode.STRUCTURED, 1)

        charged = [r for r in b if r.used_resync and r.resync_charged]
        assert len(charged) <= task.resync_budget, (
            "%s: reflexive resyncing consumed %d repairs, budget is %d -- the free "
            "no-op allowance became a second repair"
            % (task.task_id, len(charged), task.resync_budget))

        sa = [r for r in a if not r.used_resync]
        sb = [r for r in b if not r.used_resync]
        if len(sa) != len(sb):
            mismatched.append((task.task_id, "step count %d vs %d" % (len(sa), len(sb))))
            continue
        for x, y in zip(sa, sb):
            key = lambda r: (r.step, r.tool, r.selection_matches_gold,
                             r.args_correct_strict, r.held_ref, r.held_out,
                             r.diverged_in)
            if key(x) != key(y):
                mismatched.append((task.task_id, "step %d: %r vs %r"
                                   % (x.step, key(x), key(y))))

        assert recovery_outcome(task, a)["recovered"] == \
            recovery_outcome(task, b)["recovered"], (
            "%s: the two policies disagree on whether the task was recovered"
            % task.task_id)

    assert not mismatched, (
        "reflexive resyncing changed %d scored steps, so free no-op resyncs are an "
        "information channel rather than a no-op: %s"
        % (len(mismatched), mismatched[:3]))
    print("  ok  free no-op resyncs buy nothing: reflexive == oracle on every scored "
          "step (%d tasks)" % len(tasks))


def main() -> None:
    test_never_repair_matches_free_running()
    test_always_repair_recovers_the_original_trajectory()
    test_resync_restores_the_canonical_value()
    test_budget_is_enforced()
    test_three_way_split_disagrees_where_it_should()
    test_free_no_op_resync_opens_no_extra_information()
    print("\nrecovery mechanism repairs state, is inert unused, and the three-way "
          "split genuinely separates")


if __name__ == "__main__":
    main()
