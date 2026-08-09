"""measure_recovery must recover the rates it was configured with.

These rates are not decoration: they are the prior centres for r_syn and r_sem
in the hierarchical fit, so a biased measurement biases pi. The measurement was
wrong twice, in ways that only showed up when the two configured rates were far
apart, so the wide-gap case below is the one that matters most:

  1. origin was labelled only on steps entering CLEAN, so it behaved as a
     property fixed at the first fresh error. The recurrence has two poisoned
     states with movement between them, so a poisoned step that errors again
     takes the OTHER rate from then on. Trials were attributed to the original
     type while the policy drew against the new one.

  2. a poisoned step whose accuracy draw SUCCEEDED emitted a well-formed call
     carrying the corrupted value -- an observably semantic error -- while the
     simulator left its hidden origin as "syntax".

Both pulled each measured rate toward the other: r_syn came out 0.12 against a
configured 0.35. Real models expose no hidden origin, so the observed error type
is the only definition available and both sides must use it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tur.analysis.aggregate import measure_recovery                  # noqa: E402
from tur.harness.executor import FeedbackMode                        # noqa: E402
from tur.harness.runner import (MockBackend, run_free,               # noqa: E402
                                run_teacher_forced)
from tur.harness.sim_policy import SimPolicy, SimPolicyConfig        # noqa: E402
from tur.tasks.dag import generate_routing_suite, generate_suite     # noqa: E402

# name, p0, p_slope, pi, syntax_share, r_syn, r_sem
CASES = [
    ("weak-low-recovery", 0.78, 0.020, 0.70, 0.50, 0.35, 0.05),
    ("mid", 0.88, 0.012, 0.45, 0.50, 0.55, 0.15),
    # the discriminating case: a 16x gap between the two rates, so any mixing
    # between them is immediately visible
    ("wide-gap", 0.85, 0.010, 0.50, 0.50, 0.80, 0.05),
]
TOL = 0.08          # absolute; n per case is a few hundred trials
MIN_TRIALS = 40


def collect(gen, cfg: SimPolicyConfig) -> list[dict]:
    policy = SimPolicy(cfg)
    backend = MockBackend(policy, model=cfg.name)
    records: list[dict] = []
    for task in gen([6, 8], 150, 1, base_seed=1000):
        policy.stateless = False
        free = run_free(task, backend, "uniform", FeedbackMode.STRUCTURED, 1)
        policy.stateless = True
        tf = run_teacher_forced(task, backend, "uniform",
                                FeedbackMode.STRUCTURED, 1)
        policy.stateless = False
        records += [r.__dict__ for r in free] + [r.__dict__ for r in tf]
    return records


def main() -> None:
    failures = []
    for label, gen in (("linear", generate_suite), ("routing", generate_routing_suite)):
        for name, p0, slope, pi, ss, r_syn, r_sem in CASES:
            cfg = SimPolicyConfig(name, p0=p0, p_slope=slope, pi=pi,
                                  syntax_share=ss, r_syn=r_syn, r_sem=r_sem,
                                  seed=7)
            m = measure_recovery(collect(gen, cfg))
            for key, want, n_key in (("r_syn_chain", r_syn, "n_syn_trials"),
                                     ("r_sem_chain", r_sem, "n_sem_trials")):
                got, n = m[key], m[n_key]
                if n < MIN_TRIALS:
                    print(f"  skip {label}/{name}/{key}: only {n} trials")
                    continue
                ok = abs(got - want) <= TOL
                print(f"  {'ok  ' if ok else 'FAIL'} {label}/{name:18} {key:12} "
                      f"measured {got:.3f} vs configured {want:.2f} (n={n})")
                if not ok:
                    failures.append(f"{label}/{name}/{key}: "
                                    f"{got:.3f} vs {want:.2f}")

    if failures:
        raise AssertionError("recovery measurement does not recover truth:\n  "
                             + "\n  ".join(failures))
    print("\nall recovery rates recovered within "
          f"{TOL} of the configured values")


if __name__ == "__main__":
    main()
