"""Assigned-repair arm: does handing the state back return the trajectory to the goal?

THE NUMBER THIS REPORTS

completed_original, repaired branch against unrepaired branch, as a paired difference with
an 89% percentile bootstrap interval resampling PAIRS -- the same convention as
report_injection.py and Table 2, and the right one here for a structural reason rather
than for consistency alone: the two branches of a pair share a single pre-repair segment
that was run once and inherited, so the pair is the independent unit and resampling
branches would understate the interval.

completed_original is strict on purpose: the branch ran every remaining step AND finished
holding the task's true final output. A branch that took the handed-back value, made one
correct call and then drifted does not count. That strictness is what keeps this from
restating Appendix E, whose corrected branch measures the call immediately after the value
is handed back and already reports +0.000 against baseline.

WHAT IS CHECKED BEFORE THE HEADLINE IS BELIEVED

The unrepaired branch is the denominator's honesty check. Under a perfect
conditional-on-state policy it completes the original task 0 times out of 12
(tests/test_repair_arm.py), because after a parity flip the gold value is unreachable by
rule-following. A real model is not rule-following, and a stray coincidence downstream
could put a trajectory back on gold by luck. If the unrepaired rate is materially above
zero, the corruption was recoverable without the repair and the contrast is weakened --
so it is printed rather than assumed.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tur.harness.runner import (StepRecord, repair_outcome,          # noqa: E402
                                _REP_REPAIRED, _REP_UNREPAIRED)
from tur.tasks.dag import generate_routing_suite                     # noqa: E402

N_BOOT = 10000
SEED = 20260915
POOL_PER_DEPTH = 96


def _tasks_for(depths, base_seed):
    suite = generate_routing_suite(list(depths), POOL_PER_DEPTH, 1, base_seed=base_seed)
    return {t.task_id: t for t in suite}


def load_pairs(path, base_seed):
    """One (repaired, unrepaired) outcome pair per pair_id."""
    from dataclasses import fields
    keep = {f.name for f in fields(StepRecord)}
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    rows = [r for r in rows if r.get("pair_id")]
    by_pair = collections.defaultdict(list)
    for r in rows:
        by_pair[r["pair_id"]].append(StepRecord(**{k: v for k, v in r.items()
                                                   if k in keep}))
    if not by_pair:
        raise SystemExit("no pair_id rows in %s -- is this a repair-arm file?" % path)

    depths = sorted({r.depth for rs in by_pair.values() for r in rs})
    tasks = _tasks_for(depths, base_seed)

    pairs, dropped = [], 0
    for pid, recs in by_pair.items():
        task_id = recs[0].task_id
        if task_id not in tasks:
            raise SystemExit("task id %s not reproducible from base_seed %d"
                             % (task_id, base_seed))
        inject_at = recs[0].inject_at
        # repair_at is the pair's fork point, recoverable from the id it was built with
        repair_at = int(pid.split("|r")[1].split("|")[0])
        out = repair_outcome(tasks[task_id], recs, inject_at, repair_at)
        if _REP_REPAIRED not in out or _REP_UNREPAIRED not in out:
            # A half pair is unusable: the design IS the matched contrast, and keeping
            # the surviving branch would mix paired and unpaired data.
            dropped += 1
            continue
        pairs.append((tasks[task_id].depth, out[_REP_REPAIRED], out[_REP_UNREPAIRED]))
    return pairs, dropped


def paired_boot(a: np.ndarray, b: np.ndarray, rng):
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(N_BOOT, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return float(np.percentile(diffs, 5.5)), float(np.percentile(diffs, 94.5))


def cell(pairs, field, rng):
    a = np.array([r[field] for _, r, _ in pairs], dtype=float)
    b = np.array([u[field] for _, _, u in pairs], dtype=float)
    lo, hi = paired_boot(a, b, rng)
    nb = int(sum(1 for _, r, u in pairs if r[field] and not u[field]))
    nc = int(sum(1 for _, r, u in pairs if u[field] and not r[field]))
    return {"n": len(pairs), "rep": float(a.mean()) if len(a) else float("nan"),
            "unrep": float(b.mean()) if len(b) else float("nan"),
            "delta": float(a.mean() - b.mean()) if len(a) else float("nan"),
            "lo": lo, "hi": hi, "b": nb, "c": nc}


def fmt(s):
    return ("n=%3d  repaired=%.3f  unrepaired=%.3f  delta=%+.3f  [%+.3f, %+.3f]  "
            "discordant b=%d c=%d"
            % (s["n"], s["rep"], s["unrep"], s["delta"], s["lo"], s["hi"],
               s["b"], s["c"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?",
                    default="data/results/repair_groq_allam-2-7b.jsonl")
    ap.add_argument("--base-seed", type=int, default=1000)
    args = ap.parse_args()

    pairs, dropped = load_pairs(args.path, args.base_seed)
    rng = np.random.default_rng(SEED)
    depths = sorted({d for d, _, _ in pairs})

    print(args.path)
    print("pairs: %d usable, %d dropped as half pairs" % (len(pairs), dropped))
    print("interval: 89%% percentile bootstrap over PAIRS, %d resamples\n" % N_BOOT)

    print("=== HEADLINE: completed_original (ran every step AND finished on gold) ===")
    print("  pooled   %s" % fmt(cell(pairs, "completed_original", rng)))
    for d in depths:
        sub = [p for p in pairs if p[0] == d]
        print("  depth %-2d %s" % (d, fmt(cell(sub, "completed_original", rng))))

    print("\n=== supporting: downstream per-call rates from the repair onward ===")
    for field in ("downstream_gold_agreement", "downstream_conditional"):
        print("  %-26s %s" % (field, fmt(cell(pairs, field, rng))))

    unrep = float(np.mean([u["completed_original"] for _, _, u in pairs]))
    print("\nunrepaired completion rate: %.3f" % unrep)
    if unrep > 0.05:
        print("  WARNING: the corruption is recoverable without the repair in %.1f%% of "
              "pairs, so the contrast understates what the repair contributes and the "
              "denominator is not clean." % (100 * unrep))
    else:
        print("  the corrupted trajectory essentially never returns to gold on its own, "
              "so the contrast is attributable to the repair")

    rows = {"pooled": cell(pairs, "completed_original", rng),
            "by_depth": {str(d): cell([p for p in pairs if p[0] == d],
                                      "completed_original", rng) for d in depths},
            "n_dropped": dropped}
    out = Path(args.path).with_name(Path(args.path).stem + "_repair_report.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
