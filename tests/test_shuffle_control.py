"""Randomised branch presentation order: a control, so it must change nothing but wording.

Why it exists. The primary arm found rule-application accuracy depending strongly on
the parity of the held value, and attributed it to a position bias toward the
first-listed branch. In that arm the two accounts are CONFOUNDED: the rule text always
lists the even branch first, so the correct tool for an even ref *is* the first-listed
tool, and "always picks the first-listed option" predicts the same data as "follows the
rule but only succeeds on even refs". Randomising the order per step decouples them.

A control is only useful if it is inert with respect to everything else, so this asserts:

  1. the gold trajectory is bit-identical to the unshuffled task with the same seed --
     shuffling changes the WORDING of the rule, never which tool is correct
  2. a perfect policy still scores 1.000, i.e. the task remains solvable and the prompt
     still agrees with the gold trajectory
  3. the rule text actually varies, and both orders occur (a "control" that silently
     always emitted the same order would pass 1 and 2 while testing nothing)
  4. `first_listed_even` is recorded on the step records, since the analysis needs it
     per step rather than per task
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tur.analysis.aggregate import aggregate_by_depth                 # noqa: E402
from tur.harness.executor import FeedbackMode                         # noqa: E402
from tur.harness.runner import MockBackend, run_free                  # noqa: E402
from tur.harness.sim_policy import _gold_arg_for                      # noqa: E402
from tur.tasks.dag import generate_routing_suite                      # noqa: E402

DEPTHS = [1, 2, 4]
N = 20


def perfect(task, step, ref, attempt):
    return task.gold[step].tool, {"ref": _gold_arg_for(task, step, ref)}, True


def test_gold_unchanged() -> None:
    plain = generate_routing_suite(DEPTHS, N, 1, base_seed=1000)
    shuf = generate_routing_suite(DEPTHS, N, 1, base_seed=1000,
                                  shuffle_branch_order=True)
    assert len(plain) == len(shuf)
    for a, b in zip(plain, shuf):
        assert a.task_id == b.task_id
        assert a.seed_value == b.seed_value, f"{a.task_id}: seed value changed"
        assert [s.tool for s in a.gold] == [s.tool for s in b.gold], (
            f"{a.task_id}: shuffling the presentation order changed WHICH tool is "
            f"correct. It must only change the wording of the rule.")
        assert [s.args for s in a.gold] == [s.args for s in b.gold]
        assert [s.output for s in a.gold] == [s.output for s in b.gold]
    print(f"  ok  gold trajectories identical across {len(plain)} tasks")


def test_still_solvable() -> None:
    records = []
    for task in generate_routing_suite(DEPTHS, N, 1, base_seed=1000,
                                       shuffle_branch_order=True):
        records += [r.__dict__ for r in
                    run_free(task, MockBackend(perfect), "uniform",
                             FeedbackMode.STRUCTURED, 1)]
    for s in aggregate_by_depth(records, DEPTHS):
        assert s.g_t == 1.0, (f"depth {s.depth}: a perfect policy scored "
                              f"{s.g_t:.3f} under shuffled presentation, so the "
                              f"prompt and the gold trajectory disagree")
    orders = {r["first_listed_even"] for r in records}
    assert orders == {True, False}, (
        f"presentation order took only {orders} across {len(records)} steps -- a "
        f"control that never varies tests nothing")
    n_even_first = sum(1 for r in records if r["first_listed_even"])
    frac = n_even_first / len(records)
    assert 0.25 < frac < 0.75, f"order is lopsided: {frac:.2f} even-first"
    assert all(r["held_ref"] is not None for r in records), \
        "held_ref must be recorded; the parity analysis reads it per step"
    print(f"  ok  perfect policy scores 1.000 at every depth; order varies "
          f"({frac:.2f} even-first over {len(records)} steps); held_ref recorded")


def test_wording_varies() -> None:
    task = next(t for t in generate_routing_suite([6], 5, 1, base_seed=1000,
                                                  shuffle_branch_order=True)
                if len(set(t.present_odd_first)) > 1)
    text = task.routing_rule_text()
    assert "if the incoming ref is odd call" in text
    assert "if the incoming ref is even call" in text
    print("  ok  rule text emits both orders within a single task")


def main() -> None:
    test_gold_unchanged()
    test_still_solvable()
    test_wording_varies()
    print("\nshuffle control is inert with respect to task semantics")


if __name__ == "__main__":
    main()
