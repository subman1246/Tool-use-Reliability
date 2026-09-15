"""Read a recovery-arm jsonl and report the three-way split.

The point of the arm is that "continues competently from a wrong state" and "gets back
to the original goal" have been the same number wearing two names, because no task gave
the agent a route back. This script reports them side by side, per depth:

  gold agreement      call matched the canonical gold trajectory (pinned near zero after
                      a divergence, by construction -- that is the paper's Theorem)
  conditional         call was right GIVEN the value actually held (the existing remedy)
  recovery            the TRAJECTORY left the canonical path and still finished holding
                      the task's true final value

Severity is reported the same way the paper reports it -- clean-context rate minus
corrupted-context rate -- so the conditional number here is directly comparable to the
published +0.316 for allam-2-7b, and the recovery number is not a re-reading of it.

The outcome verdict is computed by runner.recovery_outcome rather than reimplemented,
so this report cannot silently disagree with the function the validity tests cover.
"""
from __future__ import annotations

import collections
import json
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tur.harness.runner import StepRecord, recovery_outcome   # noqa: E402
from tur.tasks.dag import generate_routing_suite              # noqa: E402
from tur.tasks.recovery import to_recovery_task               # noqa: E402

# Generated wide and indexed by id: generate_routing_suite seeds each task from its own
# index, so task k at depth d is the same task whatever n the run happened to use.
POOL_PER_DEPTH = 64


def _tasks_for(depths, base_seed):
    suite = generate_routing_suite(list(depths), POOL_PER_DEPTH, 1, base_seed=base_seed)
    return {t.task_id: to_recovery_task(t) for t in suite}


def _records(path, run_mode="recovery"):
    """Records for ONE arm, keyed by task.

    The suite writes the free-running arm and the teacher-forced arm into the same file,
    distinguished only by run_mode. Reading the file without that filter silently
    interleaves a depth-d recovery trajectory with a depth-d teacher-forced one under the
    same task id, which doubles the step count, makes every `completed` check wrong, and
    dilutes the clean/corrupted split with rows that have no gold_ref at all.
    """
    keep = {f.name for f in fields(StepRecord)}
    by = collections.defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("run_mode") != run_mode:
            continue
        by[d["task_id"]].append(StepRecord(**{k: v for k, v in d.items() if k in keep}))
    return by


def _rate(num, den):
    return float("nan") if den == 0 else num / den


def report(path, base_seed=1000):
    by_task = _records(path)
    depths = sorted({r[0].depth for r in by_task.values()})
    tasks = _tasks_for(depths, base_seed)
    missing = [t for t in by_task if t not in tasks]
    if missing:
        raise SystemExit("task ids not reproducible from base_seed %d: %s"
                         % (base_seed, missing[:3]))

    rows = []
    for d in depths:
        ids = [t for t in by_task if tasks[t].depth == d]
        outs = [recovery_outcome(tasks[t], by_task[t]) for t in ids]

        # per-step rates, split by whether the step was entered off-canonical
        cs = collections.Counter()
        for t in ids:
            for r in by_task[t]:
                if r.used_resync:
                    continue
                k = "corrupt" if r.diverged_in else "clean"
                cs[k + "_n"] += 1
                cs[k + "_cond"] += bool(r.correct_given_state)
                cs[k + "_gold"] += bool(r.selection_matches_gold and r.args_correct_strict)

        for t, o in zip(ids, outs):
            n = sum(1 for r in by_task[t] if not r.used_resync)
            assert n <= d, ("%s recorded %d scored steps at depth %d -- more steps than "
                            "the task has, so the arms are mixed" % (t, n, d))

        diverged = [o for o in outs if o["diverged"]]
        rows.append({
            "depth": d,
            "n_tasks": len(ids),
            "n_diverged": len(diverged),
            "n_used_resync": sum(1 for o in outs if o["used_resync"]),
            "recovery_rate": _rate(sum(1 for o in diverged if o["recovered"]),
                                   len(diverged)),
            "cond_clean": _rate(cs["clean_cond"], cs["clean_n"]),
            "cond_corrupt": _rate(cs["corrupt_cond"], cs["corrupt_n"]),
            "gold_corrupt": _rate(cs["corrupt_gold"], cs["corrupt_n"]),
            "n_clean_steps": cs["clean_n"],
            "n_corrupt_steps": cs["corrupt_n"],
        })
        rows[-1]["cond_severity"] = rows[-1]["cond_clean"] - rows[-1]["cond_corrupt"]
    return rows


def _fmt(x):
    return "  n/a" if x != x else "%.3f" % x


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "data/results/recpilot_groq_allam-2-7b.jsonl"
    base_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    rows = report(path, base_seed)
    print(path)
    print("%-6s %6s %9s %8s %9s %10s %12s %9s %9s"
          % ("depth", "tasks", "diverged", "resync", "RECOVERY", "cond|corr",
             "cond_sever", "gold|corr", "n_corr"))
    for r in rows:
        print("%-6d %6d %9d %8d %9s %10s %12s %9s %9d"
              % (r["depth"], r["n_tasks"], r["n_diverged"], r["n_used_resync"],
                 _fmt(r["recovery_rate"]), _fmt(r["cond_corrupt"]),
                 _fmt(r["cond_severity"]), _fmt(r["gold_corrupt"]),
                 r["n_corrupt_steps"]))
    Path("data/results").mkdir(parents=True, exist_ok=True)
    out = Path(path).with_name(Path(path).stem + "_recovery_report.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
