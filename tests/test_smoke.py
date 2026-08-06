"""Offline end-to-end check: no network, no heavy deps beyond numpy."""

import numpy as np

from tur.tasks.dag import generate_task, generate_suite
from tur.harness.runner import MockBackend, run_free, run_teacher_forced
from tur.harness.executor import FeedbackMode
from tur.model.state import g_curve, occupancies, simulate


def scripted_policy(task, step, ref, attempt):
    """A decent model that: fumbles step 1 once then recovers (syntactic),
    and passes a wrong value at step 2 (semantic, propagates)."""
    tool = task.gold[step].tool
    if step == 1 and attempt == 0:
        return tool, {}, True          # missing required ref -> syntactic
    if step == 2:
        return tool, {"ref": ref + 7}, True  # wrong value -> semantic
    return tool, {"ref": ref}, True


def test_runner_and_scoring():
    task = generate_task("t0", depth=5, distractor_level=1, seed=1)
    be = MockBackend(scripted_policy)

    free = run_free(task, be, call_mode="uniform",
                    feedback=FeedbackMode.STRUCTURED, max_retries=1)
    tf = run_teacher_forced(task, be, call_mode="uniform")

    assert len(free) == 5 and len(tf) == 5

    # step 1 should show a retry and a recovery
    s1 = free[1]
    assert s1.n_attempts == 2 and s1.recovered and s1.args_correct_strict

    # step 2 is an intrinsic semantic error that poisons step 3's context
    s2 = free[2]
    assert s2.error_type == "semantic" and not s2.args_correct_strict

    # step 3: the model is faithful, but in the free run its input is poisoned,
    # so it is globally wrong; under teacher forcing the input is clean, so the
    # same faithful behaviour is locally correct. This is the local/global gap.
    assert free[3].context_clean_in is False
    assert not free[3].args_correct_strict     # globally wrong (poisoned input)
    assert tf[3].args_correct_strict           # locally correct (clean input)

    print("free:", [(r.step, r.error_type, r.n_attempts, r.recovered,
                     r.context_clean_in) for r in free])
    print("teacher_forced correct:", [r.args_correct_strict for r in tf])


def test_state_model():
    T = 5
    p = np.array([0.95, 0.92, 0.88, 0.83, 0.78])  # baseline degrades with depth
    f_syn = np.full(T, 0.5)
    pi, r_syn, r_sem = 0.6, 0.7, 0.1
    g = g_curve(p, f_syn, pi, r_syn, r_sem)
    c, s, m = occupancies(p, f_syn, r_syn, r_sem)

    assert abs(g[0] - p[0]) < 1e-12                 # g_1 = p_1
    assert np.all(g <= p + 1e-9)                    # never above baseline
    L = 1 - g / p
    assert np.all(L >= -1e-9)                       # loss is nonnegative
    # fit-free identity L_t = pi * x_t
    x = s + m
    assert np.allclose(L, pi * x, atol=1e-9)

    succ, trials, gpop = simulate(p, f_syn, pi, r_syn, r_sem, 500,
                                  np.random.default_rng(0))
    assert np.all(succ <= trials)
    print("g:", np.round(g, 3), "L:", np.round(L, 3))


if __name__ == "__main__":
    test_runner_and_scoring()
    test_state_model()
    print("\nALL SMOKE TESTS PASSED")
