"""Tests for per-step aggregation, measured recovery, and paired bootstrapping.

These back three claims the methodology makes in print, two of which the code
did not previously support:
  - the error mix is resolved by position in the chain (H4)
  - r_syn and r_sem are MEASURED from labelled transitions, not only fitted
  - L_t intervals resample tasks, and the two arms are paired by construction
"""

import sys

sys.path.insert(0, "src")

from tur.analysis.aggregate import (aggregate_by_step, measure_recovery,
                                    bootstrap_L_ci)


def _rec(task, step, mode="free", correct=True, etype="none", clean=True,
         gold_sel=True, attempts=1, recovered=False, depth=4):
    return {"task_id": task, "depth": depth, "step": step, "run_mode": mode,
            "args_correct_strict": correct, "error_type": etype,
            "context_clean_in": clean, "selection_matches_gold": gold_sel,
            "n_attempts": attempts, "recovered": recovered,
            "stalled_in": False, "backend_error": False}


def test_per_step_separates_a_trend_that_depth_pooling_hides():
    """Step 0 all selection errors, step 1 all argument errors.

    Pooled into one depth bin these average out; resolved by step index the
    trend is exact. This is the aggregation defect that made the H4 figure
    measure something other than what it claimed.
    """
    rows = []
    for i in range(20):
        rows.append(_rec(f"t{i}", 0, correct=False, etype="semantic",
                         gold_sel=False))
        rows.append(_rec(f"t{i}", 1, correct=False, etype="semantic",
                         gold_sel=True))
    steps = aggregate_by_step(rows)
    assert steps[0].sel_err_share == 1.0 and steps[0].arg_err_share == 0.0
    assert steps[1].sel_err_share == 0.0 and steps[1].arg_err_share == 1.0
    assert steps[0].sel_to_arg != steps[0].sel_to_arg  # NaN: no argument errors
    assert steps[1].sel_to_arg == 0.0


def test_per_step_f_syn_tracks_a_rising_syntactic_share():
    rows = []
    for i in range(20):
        rows.append(_rec(f"t{i}", 0, correct=False, etype="semantic"))
        rows.append(_rec(f"t{i}", 1, correct=False, etype="syntactic"))
    steps = aggregate_by_step(rows)
    assert steps[0].f_syn == 0.0
    assert steps[1].f_syn == 1.0


def test_per_step_excludes_already_poisoned_steps():
    """Only fresh errors count, same rule as f_syn by depth."""
    rows = [_rec("t0", 0, correct=False, etype="syntactic", clean=True),
            _rec("t0", 1, correct=False, etype="semantic", clean=False)]
    steps = aggregate_by_step(rows)
    assert steps[0].n_fresh_errors == 1
    assert steps[1].n_fresh_errors == 0
    assert steps[1].f_syn != steps[1].f_syn  # NaN, not 0


def test_measured_recovery_counts_return_to_track_by_origin():
    """One task poisoned syntactically at step 0 and recovering at step 2, one
    poisoned semantically at step 0 and never recovering."""
    rows = [
        # syntactic origin, recovers (step 2 enters clean)
        _rec("a", 0, correct=False, etype="syntactic", clean=True),
        _rec("a", 1, correct=False, etype="semantic", clean=False),
        _rec("a", 2, correct=True, clean=True),
        # semantic origin, stays poisoned
        _rec("b", 0, correct=False, etype="semantic", clean=True),
        _rec("b", 1, correct=False, etype="semantic", clean=False),
        _rec("b", 2, correct=False, etype="semantic", clean=False),
    ]
    r = measure_recovery(rows)
    assert r["n_syn_trials"] == 1 and r["r_syn_chain"] == 1.0
    assert r["n_sem_trials"] == 1 and r["r_sem_chain"] == 0.0


def test_measured_recovery_reports_retry_channel_separately():
    rows = [_rec("a", 0, correct=True, attempts=2, recovered=True),
            _rec("b", 0, correct=False, etype="syntactic", attempts=2,
                 recovered=False)]
    r = measure_recovery(rows)
    assert r["n_retry_trials"] == 2 and r["r_syn_retry"] == 0.5


def test_measured_recovery_ignores_teacher_forced_and_backend_errors():
    rows = [_rec("a", 0, mode="teacher_forced", correct=False,
                 etype="syntactic", clean=True),
            _rec("a", 1, mode="teacher_forced", correct=True, clean=True)]
    r = measure_recovery(rows)
    assert r["n_syn_trials"] == 0


def test_paired_bootstrap_is_narrower_than_independent():
    """Same task ids in both arms, so pairing removes a variance component."""
    rows = []
    for i in range(60):
        # correlated arms: task-level difficulty shared between run modes
        hard = i % 3 == 0
        for step in range(4):
            rows.append(_rec(f"t{i}", step, mode="teacher_forced",
                             correct=not hard))
            rows.append(_rec(f"t{i}", step, mode="free",
                             correct=not hard and step < 3))
    Lp, lo_p, hi_p = bootstrap_L_ci(rows, 4, n_boot=500, seed=3, paired=True)
    Li, lo_i, hi_i = bootstrap_L_ci(rows, 4, n_boot=500, seed=3, paired=False)
    assert abs(Lp - Li) < 1e-9, "point estimate must not depend on pairing"
    assert (hi_p - lo_p) <= (hi_i - lo_i) + 1e-9, \
        f"paired width {hi_p - lo_p:.4f} should not exceed independent {hi_i - lo_i:.4f}"


def test_bootstrap_falls_back_when_arms_share_no_ids():
    rows = []
    for i in range(20):
        rows.append(_rec(f"tf{i}", 0, mode="teacher_forced", correct=True, depth=1))
        rows.append(_rec(f"fr{i}", 0, mode="free", correct=i % 2 == 0, depth=1))
    L, lo, hi = bootstrap_L_ci(rows, 1, n_boot=200, seed=1, paired=True)
    assert L == L and lo == lo and hi == hi, "should still produce an interval"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  {fn.__name__}: OK")
    print(f"\nALL {len(fns)} AGGREGATE-EXTRAS TESTS PASSED")
