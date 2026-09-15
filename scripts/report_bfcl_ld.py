"""L_d on the BFCL arm: pooled p_d, g_d, and the propagation loss, with intervals.

SINGLE-MODEL CHECK, NOT COMPARATIVE. And the scorer is OURS -- fixed-gold per-call --
not BFCL's, which compares final environment state and accepts the ground truth as an
unordered subset of the model's calls. These are not BFCL leaderboard numbers. See
docs/NOTE_bfcl_scorer_structure.md.

WHY THE INTERVAL IS PAIRED OVER TASKS

The same task is run under both histories, so the task is the independent unit and the two
arms are paired within it. Resampling tasks (not calls) also respects the fact that calls
within a task are not independent: one early error changes every later prompt in the free
arm, which is the entire phenomenon being measured. Resampling calls would treat those as
independent observations and report an interval far too narrow.

WHY L_d GETS ITS OWN STABILITY CHECK

L_d = 1 - g_d/p_d is a ratio, and its denominator is the model's own baseline competence.
On the synthetic suite p_d is comfortably above 0.5 and the ratio is well behaved. On a
hard real benchmark under strict per-call gold agreement, p_d can land low enough that L_d
swings on one or two calls. The report therefore prints p_d prominently and warns when it
is small, rather than presenting a ratio whose precision it cannot support.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

N_BOOT = 10000
SEED = 20260915
TF = "bfcl_teacher_forced"
FREE = "bfcl_free"
P_MIN = 0.30          # below this, the ratio is reported but flagged as unstable


def load(path):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    by_task = collections.defaultdict(lambda: {TF: [], FREE: []})
    for r in rows:
        if r["run_mode"] in (TF, FREE):
            by_task[r["task_id"]][r["run_mode"]].append(r)
    full, dropped = {}, 0
    for tid, d in by_task.items():
        if d[TF] and d[FREE] and len(d[TF]) == len(d[FREE]):
            full[tid] = d
        else:
            # A task present in one arm only cannot contribute to a paired ratio.
            dropped += 1
    return full, dropped


def _rates(tasks, ids, field):
    """Pooled rates over calls, for one resample of TASKS."""
    tf_hit = tf_n = fr_hit = fr_n = 0
    for tid in ids:
        d = tasks[tid]
        tf_hit += sum(bool(r[field]) for r in d[TF]);  tf_n += len(d[TF])
        fr_hit += sum(bool(r[field]) for r in d[FREE]); fr_n += len(d[FREE])
    p = tf_hit / tf_n if tf_n else float("nan")
    g = fr_hit / fr_n if fr_n else float("nan")
    return p, g


def analyse(tasks, field, rng):
    ids = list(tasks)
    p, g = _rates(tasks, ids, field)
    L = 1 - g / p if p else float("nan")

    n = len(ids)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    Ls, ps, gs = [], [], []
    for row in idx:
        sample = [ids[i] for i in row]
        pp, gg = _rates(tasks, sample, field)
        ps.append(pp); gs.append(gg)
        Ls.append(1 - gg / pp if pp else np.nan)
    Ls = np.array(Ls, dtype=float)
    Ls = Ls[np.isfinite(Ls)]
    lo, hi = (float(np.percentile(Ls, 5.5)), float(np.percentile(Ls, 94.5))) \
        if len(Ls) else (float("nan"), float("nan"))
    return {"n_tasks": n, "p": p, "g": g, "L": L, "lo": lo, "hi": hi,
            "p_lo": float(np.percentile(ps, 5.5)), "p_hi": float(np.percentile(ps, 94.5)),
            "n_calls_per_arm": sum(len(tasks[t][TF]) for t in ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?",
                    default="data/results/bfclpilot_groq_openai_gpt-oss-120b.jsonl")
    args = ap.parse_args()

    tasks, dropped = load(args.path)
    if not tasks:
        raise SystemExit("no paired tasks in %s" % args.path)
    rng = np.random.default_rng(SEED)

    depth = collections.Counter(len(d[TF]) for d in tasks.values())
    print("BFCL L_d -- SINGLE-MODEL CHECK (gpt-oss-120b), NOT comparative")
    print("scorer: fixed-gold per-call (ours, not BFCL's state-based checker)")
    print("tasks: %d paired, %d dropped unpaired | depth distribution: %s"
          % (len(tasks), dropped, dict(sorted(depth.items()))))
    print("interval: 89%% percentile bootstrap over TASKS, %d resamples\n" % N_BOOT)

    out = {}
    for field, label in (("args_correct_strict", "gold agreement (tool+args)"),
                         ("selection_matches_gold", "selection only")):
        a = analyse(tasks, field, rng)
        # The suppression rule, enforced here rather than left to whoever reads the
        # output: below P_MIN the ratio's denominator is too small for the point estimate
        # to mean anything, and a number printed with a caveat attached still gets quoted
        # without the caveat. So p_d and g_d are reported and L_d is WITHHELD.
        a["L_suppressed"] = a["p"] < P_MIN
        out[field] = a
        if a["L_suppressed"]:
            print("%-28s p_d=%.3f [%.3f, %.3f]  g_d=%.3f  L_d=WITHHELD  (%d calls/arm)"
                  % (label, a["p"], a["p_lo"], a["p_hi"], a["g"], a["n_calls_per_arm"]))
        else:
            print("%-28s p_d=%.3f [%.3f, %.3f]  g_d=%.3f  L_d=%.3f [%.3f, %.3f]  "
                  "(%d calls/arm)"
                  % (label, a["p"], a["p_lo"], a["p_hi"], a["g"], a["L"], a["lo"],
                     a["hi"], a["n_calls_per_arm"]))

    strict = out["args_correct_strict"]
    print()
    if len(tasks) < 10:
        # A task-level bootstrap over very few tasks resamples the same handful
        # repeatedly; at n=1 every resample is identical and the interval collapses to a
        # point, which looks like precision and is the opposite.
        print("WARNING: %d tasks. The task-level bootstrap has too few distinct units to "
              "give a meaningful interval -- at this n the bounds describe the resampling, "
              "not the uncertainty. Treat the point estimates as a smoke check only."
              % len(tasks))
    if strict["L_suppressed"]:
        print("L_d WITHHELD: p_d = %.3f is below %.2f. The propagation loss is a ratio "
              "with baseline competence as its denominator; at this p_d the point "
              "estimate is not supported and is deliberately not reported, rather than "
              "reported with a caveat. What the data supports at this n is p_d = %.3f "
              "[%.3f, %.3f] and g_d = %.3f."
              % (strict["p"], P_MIN, strict["p"], strict["p_lo"], strict["p_hi"],
                 strict["g"]))
    else:
        print("p_d = %.3f supports the ratio; L_d = %.3f [%.3f, %.3f]"
              % (strict["p"], strict["L"], strict["lo"], strict["hi"]))

    diag = collections.Counter()
    for d in tasks.values():
        for r in d[FREE]:
            diag[r["error_type"]] += 1
    print("\nfree-arm error types: %s" % dict(diag))
    print("free-arm unexecuted calls: %d"
          % sum(1 for d in tasks.values() for r in d[FREE] if not r["executed"]))

    saved = {}
    for k, v in out.items():
        v = dict(v)
        if v.get("L_suppressed"):
            for drop in ("L", "lo", "hi"):
                v[drop] = None
        saved[k] = v
    p = Path(args.path)
    p.with_name(p.stem + "_ld_report.json").write_text(json.dumps(saved, indent=2),
                                                       encoding="utf-8")
    print("\nwrote", p.with_name(p.stem + "_ld_report.json"))


if __name__ == "__main__":
    main()
