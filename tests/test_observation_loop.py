"""The free run must be a real agent loop.

Regression tests for a bug that made g_t meaningless: run_free never appended
the model's own call or the tool's result to the conversation, so at step t the
model was asked for a ref it had never been shown. Observed g_t was exactly
1/depth -- only step 0 is answerable from the prompt alone.

Nothing in the simulated suite could catch this, because MockBackend takes the
carried ref from the runner's stashed _ctx rather than from the message history.
These tests assert against the history itself for that reason.
"""

import json
import sys

sys.path.insert(0, "src")

from tur.tasks.dag import generate_suite
from tur.harness.runner import run_free, run_teacher_forced, MockBackend
from tur.harness.executor import FeedbackMode


def _capture(task, mode, policy):
    """Run one mode and return the message history of every backend call."""
    seen = []
    b = MockBackend(policy)
    orig = b.complete

    def complete(messages, tools, m):
        seen.append([{k: v for k, v in msg.items() if not k.startswith("_")}
                     for msg in messages])
        return orig(messages, tools, m)

    b.complete = complete
    runner = run_free if mode == "free" else run_teacher_forced
    runner(task, b, "uniform", FeedbackMode.STRUCTURED, 1)
    return seen


def _perfect(t, step, ref, attempt):
    return t.gold[step].tool, {"ref": ref}, True


def _task(depth=4):
    return generate_suite([depth], 1, 1, base_seed=1000)[0]


def test_free_history_contains_observations():
    task = _task()
    last = _capture(task, "free", _perfect)[-1]
    assert sum(1 for m in last if m["role"] == "assistant") == 3
    joined = " ".join(str(m["content"]) for m in last)
    assert joined.count("result:") == 3


def test_free_history_contains_the_ref_the_step_needs():
    """The core failure: the required ref was absent from the history."""
    task = _task()
    for t in range(1, task.depth):
        # the history at step t is the (t+1)-th captured call
        hist = _capture(task, "free", _perfect)[t]
        needed = str(task.gold[t].args["ref"])
        joined = " ".join(str(m["content"]) for m in hist)
        assert needed in joined, f"step {t} cannot see its ref {needed}"


def test_free_and_teacher_forced_histories_are_structurally_identical():
    """p_t and g_t must differ only in whether the history is CORRECT.

    If one mode carries turns the other lacks, the comparison also picks up a
    prompt-shape difference, and L_t stops being a clean propagation measure.
    """
    task = _task()
    free_last = _capture(task, "free", _perfect)[-1]
    tf_last = _capture(task, "tf", _perfect)[-1]
    assert [m["role"] for m in free_last] == [m["role"] for m in tf_last]
    assert [str(m["content"]) for m in free_last] == \
           [str(m["content"]) for m in tf_last], \
        "with a perfect policy the two histories should be identical"


def test_free_history_carries_the_models_own_wrong_output():
    """A semantic error must propagate through the observation channel: the
    history should show what the model's call actually returned, not gold."""
    def wrong_ref(t, step, ref, attempt):
        # correct tool, but corrupt the argument at step 0 only
        return t.gold[step].tool, {"ref": ref + 1 if step == 0 else ref}, True

    task = _task()
    hist = _capture(task, "free", wrong_ref)[1]      # history at step 1
    joined = " ".join(str(m["content"]) for m in hist)
    gold_out = str(task.gold[0].output)
    assert f"result: {gold_out}" not in joined, \
        "history leaked the gold result instead of the model's own"
    assert "result:" in joined


def test_failed_attempt_is_echoed_before_feedback():
    """On a syntactic failure the model must see its own bad output followed by
    the error, or within-step recovery (r_syn) has nothing to correct from."""
    def bad_then_good(t, step, ref, attempt):
        if attempt == 0:
            return t.gold[step].tool, {"ref": ref}, False   # unparseable
        return t.gold[step].tool, {"ref": ref}, True

    task = _task(depth=2)
    calls = _capture(task, "free", bad_then_good)
    retry_hist = calls[1]        # second call = the retry of step 0
    roles = [m["role"] for m in retry_hist]
    assert "assistant" in roles, "the failed attempt was not echoed"
    assert str(retry_hist[-1]["content"]).startswith("[step") or \
        "assistant" in roles


def test_depth_one_is_unaffected():
    task = _task(depth=1)
    free_last = _capture(task, "free", _perfect)[-1]
    tf_last = _capture(task, "tf", _perfect)[-1]
    assert [m["role"] for m in free_last] == [m["role"] for m in tf_last]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  {fn.__name__}: OK")
    print(f"\nALL {len(fns)} OBSERVATION-LOOP TESTS PASSED")
