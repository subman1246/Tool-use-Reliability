"""Run every simulated policy through the actual harness (not the analytic
simulator) across both task types, several seeds, and report anything that
looks broken: crashes, NaNs propagating past aggregation guards, L_t outside
[0,1]-ish sanity range, empty bins, or degenerate f_syn/p_t. This is meant to
be run BEFORE trusting any numbers out of the pipeline.
"""

from __future__ import annotations

import numpy as np

from tur.tasks.dag import generate_suite, generate_routing_suite
from tur.harness.executor import FeedbackMode
from tur.harness.runner import run_free, run_teacher_forced, dump_jsonl
from tur.harness.sim_policy import SimPolicy, DEFAULT_SUITE
from tur.analysis.aggregate import load_records, aggregate_by_depth

DEPTHS = [1, 2, 4, 6, 8]
PER_DEPTH = 25
N_SEEDS = 2
OUT_DIR = "data/results"


def run_one(cfg, task_kind: str, run_seed: int):
    policy = SimPolicy(cfg.__class__(**{**cfg.__dict__, "seed": cfg.seed * 97 + run_seed}))
    backend_name = f"{cfg.name}"
    if task_kind == "linear":
        suite = generate_suite(DEPTHS, PER_DEPTH, distractor_level=1,
                               base_seed=run_seed * 31)
    else:
        suite = generate_routing_suite(DEPTHS, PER_DEPTH, distractor_level=1,
                                       base_seed=run_seed * 31)

    records = []
    for task in suite:
        f = run_free(task, _AsBackend(policy, backend_name), "uniform",
                     FeedbackMode.STRUCTURED, max_retries=1)
        t = run_teacher_forced(task, _AsBackend(policy, backend_name), "uniform",
                               FeedbackMode.STRUCTURED, max_retries=1)
        records += [r.__dict__ for r in f + t]
    return records


class _AsBackend:
    """Wraps a raw policy callable to satisfy the MockBackend interface used
    inside run_free/run_teacher_forced without importing MockBackend's own
    JSON-encoding assumptions twice."""
    def __init__(self, policy, model_name):
        from tur.harness.runner import MockBackend
        self._mb = MockBackend(policy, model=model_name)
        self.model = model_name

    def complete(self, messages, tools, mode):
        return self._mb.complete(messages, tools, mode)


def main():
    issues = []
    checked = 0
    for kind in ("linear", "routing"):
        for cfg in DEFAULT_SUITE:
            for seed in range(N_SEEDS):
                checked += 1
                try:
                    recs = run_one(cfg, kind, seed)
                except Exception as e:  # noqa
                    issues.append(f"[{kind}/{cfg.name}/seed{seed}] CRASH: {e}")
                    continue

                import json, os
                os.makedirs(OUT_DIR, exist_ok=True)
                path = f"{OUT_DIR}/stress_{kind}_{cfg.name}_s{seed}.jsonl"
                with open(path, "w") as fh:
                    for r in recs:
                        fh.write(json.dumps(r) + "\n")

                loaded = load_records(path)
                stats = aggregate_by_depth(loaded, DEPTHS)
                for s in stats:
                    tag = f"[{kind}/{cfg.name}/seed{seed}/depth{s.depth}]"
                    if s.n_p == 0 or s.n_g == 0:
                        issues.append(f"{tag} EMPTY BIN n_p={s.n_p} n_g={s.n_g}")
                    if not np.isnan(s.L_t) and not (-0.5 <= s.L_t <= 1.0):
                        issues.append(f"{tag} L_t out of sane range: {s.L_t:.3f}")
                    if not np.isnan(s.p_t) and not (0.0 <= s.p_t <= 1.0):
                        issues.append(f"{tag} p_t out of [0,1]: {s.p_t:.3f}")
                    if not np.isnan(s.g_t) and not (0.0 <= s.g_t <= 1.0):
                        issues.append(f"{tag} g_t out of [0,1]: {s.g_t:.3f}")
                    if s.n_p > 0 and np.isnan(s.p_t):
                        issues.append(f"{tag} p_t is NaN despite n_p={s.n_p}")

    print(f"checked {checked} (policy x task-kind x seed) runs")
    if issues:
        print(f"\n{len(issues)} ISSUES FOUND:")
        for i in issues[:60]:
            print(" -", i)
        if len(issues) > 60:
            print(f"   ... and {len(issues) - 60} more")
    else:
        print("no issues found")
    return issues


if __name__ == "__main__":
    main()
