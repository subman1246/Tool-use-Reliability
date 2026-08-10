"""The transformed-argument routing variant, added because H4 was untestable.

On the copy-argument variant, real models produced 302 selection errors and 0
argument errors at clean-context steps: applying the routing rule is hard,
transcribing a number written verbatim in the previous observation is not. So the
selection-to-argument composition H4 is about had one empty category and the
hypothesis could not be tested at all -- not for lack of statistical power, but
because one of its two outcomes does not occur where the composition is defined.

`arg_shift` makes the required argument a stated transformation of the carried
value, so an argument can be wrong independently of tool choice. This test locks
the properties that make the variant usable:

  1. it is SOLVABLE -- a perfect policy scores 1.000 at every depth, which is the
     check that the gold trajectory and the prompt agree. If they disagreed, the
     variant would look impossibly hard rather than merely harder, and the error
     it produced would be indistinguishable from a real model failing.
  2. clean contexts are still recognised as clean. "Clean context" is a property
     of the value carried IN, not of the argument to send; those coincide only
     when the argument is a copy. Comparing the carried value against the expected
     ARGUMENT would mark every clean step as poisoned and make L_t meaningless.
  3. both error channels stay distinguishable in the logs.
  4. the copy variant (arg_shift=0) is byte-for-byte unaffected.

What this test does NOT establish: that REAL models produce argument errors here.
That needs API budget and is the open question the variant exists to answer. All
this shows is that the task now permits them.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tur.analysis.aggregate import aggregate_by_depth, aggregate_by_step  # noqa: E402
from tur.harness.executor import FeedbackMode                            # noqa: E402
from tur.harness.runner import (MockBackend, run_free,                   # noqa: E402
                                run_teacher_forced)
from tur.harness.sim_policy import (SimPolicy, SimPolicyConfig,          # noqa: E402
                                    _gold_arg_for)
from tur.tasks.dag import generate_routing_suite                         # noqa: E402

DEPTHS = [1, 2, 4, 6]
N = 25
SHIFT = 7


def perfect(task, step, ref, attempt):
    return task.gold[step].tool, {"ref": _gold_arg_for(task, step, ref)}, True


def collect(policy, shift: int) -> list[dict]:
    records: list[dict] = []
    for task in generate_routing_suite(DEPTHS, N, 1, base_seed=1000,
                                       arg_shift=shift):
        backend = MockBackend(policy)
        stateful = isinstance(policy, SimPolicy)
        if stateful:
            policy.stateless = False
        free = run_free(task, backend, "uniform", FeedbackMode.STRUCTURED, 1)
        if stateful:
            policy.stateless = True
        tf = run_teacher_forced(task, backend, "uniform",
                                FeedbackMode.STRUCTURED, 1)
        if stateful:
            policy.stateless = False
        records += [r.__dict__ for r in free] + [r.__dict__ for r in tf]
    return records


def test_solvable(shift: int) -> None:
    recs = collect(perfect, shift)
    for s in aggregate_by_depth(recs, DEPTHS):
        assert s.p_t == 1.0, (f"shift={shift} depth={s.depth}: a PERFECT policy "
                              f"scored p_t={s.p_t:.3f}, so the gold trajectory and "
                              f"the prompt disagree -- the variant is mis-built, "
                              f"not hard")
        assert s.g_t == 1.0, (f"shift={shift} depth={s.depth}: perfect policy "
                              f"g_t={s.g_t:.3f}")
    assert all(r["context_clean_in"] for r in recs), (
        f"shift={shift}: a perfect run has steps marked as entering a POISONED "
        f"context. context_clean must compare the value carried IN against the "
        f"previous gold output, not against the argument to be sent.")
    print(f"  ok  arg_shift={shift}: solvable, p_t = g_t = 1.000 at every depth, "
          f"every context clean")


def test_both_channels() -> None:
    """An erring policy must produce BOTH selection and argument fresh errors."""
    cfg = SimPolicyConfig("mix", p0=0.60, p_slope=0.01, pi=0.50,
                          syntax_share=0.0, r_syn=0.3, r_sem=0.1, seed=3,
                          selection_share=0.5)
    recs = collect(SimPolicy(cfg), SHIFT)
    steps = aggregate_by_step(recs)
    sel = sum(int((s.sel_err_share or 0) * s.n_fresh_errors) for s in steps
              if s.n_fresh_errors)
    arg = sum(int((s.arg_err_share or 0) * s.n_fresh_errors) for s in steps
              if s.n_fresh_errors)
    assert sel > 10 and arg > 10, (
        f"only sel={sel} arg={arg} fresh errors: the two channels are not both "
        f"populated, so the composition H4 is about still cannot be measured")
    ratio = sel / arg
    assert 0.3 < ratio < 3.0, (f"sel/arg = {ratio:.2f} at a configured selection "
                               f"share of 0.5 -- the channels are populated but "
                               f"badly skewed")
    print(f"  ok  both channels populated: {sel} selection, {arg} argument "
          f"fresh errors (ratio {ratio:.2f} at configured 0.5)")


def main() -> None:
    test_solvable(0)
    test_solvable(SHIFT)
    test_both_channels()
    print("\ntransformed-argument variant is solvable and exercises both error "
          "channels")


if __name__ == "__main__":
    main()
