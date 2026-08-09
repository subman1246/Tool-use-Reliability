"""The clean baseline must actually be clean.

L_1 = 1 - g_1/p_1 is 0 by construction: at depth 1 there is no upstream history,
so the free and teacher-forced runs measure the same probability. A systematic
deviation means the two run modes are not measuring the same thing, and every
L_t is offset by whatever causes it.

This caught a real defect. SimPolicy keys its poisoned state by task_id, and one
instance served both run modes for a task, so the free run's corruption leaked
into the clean baseline: 31% of depth-1 teacher-forced calls entered "poisoned",
and L_1 measured -0.113.
"""

import sys

sys.path.insert(0, "src")

from tur.tasks.dag import generate_suite, generate_routing_suite
from tur.harness.executor import FeedbackMode
from tur.harness.runner import run_free, run_teacher_forced, MockBackend
from tur.harness.sim_policy import SimPolicy, SimPolicyConfig

# high pi on purpose: the leak's size scaled with pi, so a severe policy is the
# sensitive test. A benign one would hide it.
CFG = SimPolicyConfig("probe", p0=0.78, p_slope=0.020, pi=0.70,
                      syntax_share=0.5, r_syn=0.35, r_sem=0.05, seed=1)


def _run(gen, depth, n):
    policy = SimPolicy(CFG)
    backend = MockBackend(policy, model="probe")
    free_rows, tf_rows, leaked = [], [], 0
    for task in gen([depth], n, 1, base_seed=1000):
        policy.stateless = False
        free_rows += run_free(task, backend, "uniform",
                              FeedbackMode.STRUCTURED, 1)
        policy.stateless = True
        # any state visible here at depth 1 would be a leak
        if depth == 1 and policy._state.get(task.task_id, (False, None))[0]:
            leaked += 1
        tf_rows += run_teacher_forced(task, backend, "uniform",
                                      FeedbackMode.STRUCTURED, 1)
        policy.stateless = False
    g = sum(r.args_correct_strict for r in free_rows) / len(free_rows)
    p = sum(r.args_correct_strict for r in tf_rows) / len(tf_rows)
    return p, g, leaked


def test_L1_is_zero_linear():
    p, g, _ = _run(generate_suite, 1, 600)
    L1 = 1 - g / p
    # 600 tasks, rate ~0.8: binomial SE ~0.016 per arm, so ~0.05 covers 2 SE of
    # the ratio comfortably. The defect produced -0.113.
    assert abs(L1) < 0.05, f"L_1 = {L1:+.4f}, expected ~0 (p={p:.3f}, g={g:.3f})"


def test_L1_is_zero_routing():
    p, g, _ = _run(generate_routing_suite, 1, 600)
    L1 = 1 - g / p
    assert abs(L1) < 0.05, f"L_1 = {L1:+.4f}, expected ~0 (p={p:.3f}, g={g:.3f})"


def test_stateless_flag_blocks_cross_mode_leak():
    """With the flag honoured, a teacher-forced call never sees poisoning."""
    policy = SimPolicy(CFG)
    backend = MockBackend(policy, model="probe")
    seen_poisoned = []

    orig = policy.__class__.__call__

    def spy(self, task, step, ref, attempt):
        if self.stateless:
            seen_poisoned.append(self._state.get(task.task_id, (False, None))[0]
                                 and not self.stateless)
        return orig(self, task, step, ref, attempt)

    policy.__class__.__call__ = spy
    try:
        for task in generate_suite([4], 40, 1, base_seed=1000):
            policy.stateless = False
            run_free(task, backend, "uniform", FeedbackMode.STRUCTURED, 1)
            policy.stateless = True
            run_teacher_forced(task, backend, "uniform",
                               FeedbackMode.STRUCTURED, 1)
            policy.stateless = False
    finally:
        policy.__class__.__call__ = orig
    assert seen_poisoned and not any(seen_poisoned), \
        "a teacher-forced call observed poisoned state"


def test_teacher_forced_does_not_accumulate_corruption_across_steps():
    """Within a teacher-forced run every step is clean, so per-step correctness
    must not decay the way a free run's does."""
    policy = SimPolicy(CFG)
    backend = MockBackend(policy, model="probe")
    by_step = {}
    for task in generate_suite([8], 120, 1, base_seed=1000):
        policy.stateless = True
        for r in run_teacher_forced(task, backend, "uniform",
                                    FeedbackMode.STRUCTURED, 1):
            by_step.setdefault(r.step, []).append(r.args_correct_strict)
    rates = [sum(v) / len(v) for _, v in sorted(by_step.items())]
    # only the configured p_slope should reduce these (0.02/step over 8 steps
    # = 0.14 total); a leak compounds far faster than that.
    assert rates[0] - rates[-1] < 0.30, f"per-step rates decayed too fast: {rates}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  {fn.__name__}: OK")
    print(f"\nALL {len(fns)} BASELINE-PURITY TESTS PASSED")
