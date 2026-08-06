"""Run the synthetic experiment for one model and write results.

Defaults to the offline mock backend so the pipeline is runnable with no keys.
Pass --model to use a real provider through LiteLLM (reads keys from the env).

Outputs per-depth p_t (teacher-forced) and g_t (free), and the fit-free net
loss L_t = 1 - g_t / p_t, which is the headline that survives identifiability
problems. JSONL logs of every call go to the results directory.

Example:
  PYTHONPATH=src python scripts/run_experiment.py --mock
  PYTHONPATH=src python scripts/run_experiment.py --model groq/qwen2.5-7b-instruct
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from tur.tasks.dag import generate_suite
from tur.harness.executor import FeedbackMode
from tur.harness.runner import (MockBackend, LiteLLMBackend, run_free,
                                run_teacher_forced, dump_jsonl)


def demo_policy(task, step, ref, attempt):
    """A stand-in model: mostly faithful, occasionally drops the ref (syntactic)
    or perturbs the value (semantic). Only used with --mock."""
    tool = task.gold[step].tool
    rng = (hash((task.task_id, step)) % 100)
    if rng < 8 and attempt == 0:
        return tool, {}, True                       # syntactic slip, may recover
    if rng >= 92:
        return tool, {"ref": ref + 3}, True          # semantic slip, propagates
    return tool, {"ref": ref}, True


def summarize(free_recs, tf_recs, depths):
    by_depth_free = defaultdict(list)
    by_depth_tf = defaultdict(list)
    for r in free_recs:
        by_depth_free[r.depth].append(r.args_correct_strict)
    for r in tf_recs:
        by_depth_tf[r.depth].append(r.args_correct_strict)
    rows = []
    for d in depths:
        # per-step correctness averaged over the deepest step of each chain
        g = np.mean(by_depth_free[d]) if by_depth_free[d] else float("nan")
        p = np.mean(by_depth_tf[d]) if by_depth_tf[d] else float("nan")
        L = 1 - g / p if p else float("nan")
        rows.append((d, p, g, L))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    ap.add_argument("--per-depth", type=int, default=40)
    ap.add_argument("--distractors", type=int, default=1)
    ap.add_argument("--feedback", choices=["structured", "opaque"], default="structured")
    ap.add_argument("--call-mode", choices=["uniform", "native"], default="uniform")
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--out", default="data/results/run.jsonl")
    args = ap.parse_args()

    if args.mock or not args.model:
        backend = MockBackend(demo_policy, model="mock")
    else:
        backend = LiteLLMBackend(args.model)

    fb = FeedbackMode(args.feedback)
    suite = generate_suite(args.depths, args.per_depth, args.distractors)

    free_all, tf_all = [], []
    for task in suite:
        f = run_free(task, backend, args.call_mode, fb, args.max_retries)
        t = run_teacher_forced(task, backend, args.call_mode, fb, args.max_retries)
        free_all += f; tf_all += t
        dump_jsonl(f + t, args.out)

    print(f"model={backend.model}  tasks={len(suite)}  logged->{args.out}\n")
    print(f"{'depth':>6} {'p_t':>7} {'g_t':>7} {'L_t':>7}")
    for d, p, g, L in summarize(free_all, tf_all, args.depths):
        print(f"{d:>6} {p:>7.3f} {g:>7.3f} {L:>7.3f}")


if __name__ == "__main__":
    main()
