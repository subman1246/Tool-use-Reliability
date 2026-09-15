"""The BFCL L_d arm must not penalise correct behaviour, in either history.

Why it exists. L_d = 1 - g_d/p_d is a ratio of two gold-agreement rates. Anything that
depresses gold agreement for reasons unrelated to the model -- an argument normalisation,
a calling convention, an environment that refuses the gold call -- enters BOTH rates, but
not equally, and moves L_d without anything having gone wrong in the model. So the
load-bearing test is the same one the injection arm uses: an oracle must score 1.000.

It caught two real defects before any spend:

  int coercion      parse_response turns digit-strings into ints, which is correct on the
                    synthetic suite (every argument there is an integer reference) and
                    wrong here: BFCL's echo(content='9') writes the STRING "9", so an
                    oracle emitting the gold call scored as wrong.

  positional args   33 gold calls in the curated subset are written positionally
                    (sort('x')). The scorer compared an empty argument dict against a
                    filled one, and the free arm could not execute the model's
                    keyword-only output at all.

Both would have produced a plausible-looking L_d built partly on harness artefacts.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tur.harness.bfcl_runner import (curated_subset, run_bfcl_free,       # noqa: E402
                                     run_bfcl_teacher_forced)
from tur.harness.runner import MockBackend, parse_response               # noqa: E402

MIN_DEPTH, MAX_DEPTH = 6, 10


def _oracle(task, step, ref, attempt):
    g = task.gold[step]
    return g["tool"], g["args"], True


def test_oracle_scores_perfectly_in_both_arms() -> None:
    tasks = curated_subset(MIN_DEPTH, MAX_DEPTH)
    assert tasks, "curated subset is empty"
    bad = []
    for t in tasks:
        for name, fn in (("teacher_forced", run_bfcl_teacher_forced),
                         ("free", run_bfcl_free)):
            recs = fn(t, MockBackend(_oracle))
            assert len(recs) == t.depth, (t.task_id, name, len(recs), t.depth)
            rate = sum(r.args_correct_strict for r in recs) / len(recs)
            unexecuted = sum(1 for r in recs if not r.executed)
            if rate < 1.0 or unexecuted:
                bad.append((t.task_id, name, round(rate, 3), unexecuted))
    assert not bad, (
        "an oracle emitting the gold call was not scored 1.000 (or failed to execute) on "
        "%d of %d task-arms. Any L_d measured under this scorer would be partly a harness "
        "artefact. First few: %s" % (len(bad), 2 * len(tasks), bad[:5]))
    print("  ok  oracle scores 1.000 and executes cleanly on all %d tasks, both arms"
          % len(tasks))


def test_digit_strings_are_not_coerced_on_bfcl() -> None:
    """The synthetic coercion must stay off here, and stay ON by default elsewhere."""
    import json
    resp = {"text": json.dumps({"tool": "echo", "args": {"content": "9"}})}
    assert parse_response(resp, "uniform").args == {"content": 9}, (
        "default coercion changed; the synthetic arms depend on it")
    assert parse_response(resp, "uniform", coerce_ints=False).args == {"content": "9"}, (
        "BFCL path must preserve the string, or echo(content='9') scores as wrong")
    print("  ok  coercion is off for BFCL and unchanged by default")


def test_positional_gold_calls_are_normalised() -> None:
    tasks = curated_subset(MIN_DEPTH, MAX_DEPTH)
    leftover = [(t.task_id, g["call"]) for t in tasks for g in t.gold if g["pos"]]
    assert not leftover, (
        "%d gold calls still carry positional arguments after normalisation: %s"
        % (len(leftover), leftover[:3]))
    filled = [g for t in tasks for g in t.gold if g["args"]]
    assert filled, "no gold call carries arguments at all, which cannot be right"
    print("  ok  positional gold calls folded onto their parameter names")


def test_the_two_arms_differ_only_in_the_history() -> None:
    """A wrong policy must diverge in the free arm and not in the teacher-forced one."""
    t = curated_subset(MIN_DEPTH, MAX_DEPTH)[0]

    def wrong_first(task, step, ref, attempt):
        g = task.gold[step]
        if step == 0:
            return g["tool"], dict(g["args"], **{"__bogus": 1}), True
        return g["tool"], g["args"], True

    tf = run_bfcl_teacher_forced(t, MockBackend(wrong_first))
    fr = run_bfcl_free(t, MockBackend(wrong_first))
    assert not tf[0].args_correct_strict and not fr[0].args_correct_strict
    # after the bad call, teacher forcing puts the gold history back; free does not
    assert all(r.args_correct_strict for r in tf[1:]), (
        "the teacher-forced arm did not restore the gold history after a wrong call, so "
        "p_d is not a per-call measure")
    print("  ok  teacher forcing restores the gold history; free running does not")


def main() -> None:
    test_oracle_scores_perfectly_in_both_arms()
    test_digit_strings_are_not_coerced_on_bfcl()
    test_positional_gold_calls_are_normalised()
    test_the_two_arms_differ_only_in_the_history()
    print("\nBFCL L_d arm does not penalise correct behaviour in either history")


if __name__ == "__main__":
    main()
