"""The official pipeline-validation run: larger sample sizes, full aggregation,
Bayesian hierarchical fit on real harness output (not the analytic shortcut),
a model-free lag check (Delta_1), and export of everything the figures and the
results write-up need.

This validates the pipeline end to end on SIMULATED model tiers with known
ground truth. It is not a claim about real LLMs. Swap SimPolicy for
LiteLLMBackend + real model names to turn this into the real experiment.
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np

from tur.tasks.dag import generate_suite
from tur.harness.executor import FeedbackMode
from tur.harness.runner import run_free, run_teacher_forced, MockBackend
from tur.harness.sim_policy import SimPolicy, SimPolicyConfig, DEFAULT_SUITE, FAMILY_INDEX
from tur.analysis.aggregate import (load_records, aggregate_by_depth,
                                    stats_to_arrays, bootstrap_L_ci,
                                    aggregate_by_step, measure_recovery)
from tur.model.hierarchical import build_and_sample, identifiability

DEPTHS = [1, 2, 4, 6, 8]
PER_DEPTH = 200
SEEDS = 2
OUT_DIR = "data/results"
RUN_TAG = "official"


def run_policy(cfg: SimPolicyConfig) -> list[dict]:
    records = []
    for seed in range(SEEDS):
        policy = SimPolicy(SimPolicyConfig(**{**cfg.__dict__,
                                              "seed": cfg.seed * 97 + seed}))
        backend = MockBackend(policy, model=cfg.name)
        suite = generate_suite(DEPTHS, PER_DEPTH, distractor_level=1,
                               base_seed=seed * 31 + 1000)
        for task in suite:
            # The policy carries poisoned state between calls, so it must be
            # told which run mode is asking. Teacher forcing supplies a correct
            # upstream history by construction; without `stateless` the free
            # run's corruption leaked into the clean baseline and drove L_1
            # negative when it is 0 by construction.
            policy.stateless = False
            f = run_free(task, backend, "uniform", FeedbackMode.STRUCTURED,
                        max_retries=1)
            policy.stateless = True
            t = run_teacher_forced(task, backend, "uniform",
                                   FeedbackMode.STRUCTURED, max_retries=1)
            policy.stateless = False
            records += [r.__dict__ for r in f] + [r.__dict__ for r in t]
    return records


def delta_1(records: list[dict]) -> float:
    """Model-free propagation check: P(step t+1 correct | t correct)
    minus P(step t+1 correct | t wrong), pooled over all tasks and steps in
    the free run. Positive means an error at t measurably hurts t+1, which is
    the direct evidence for propagation the fitted pi is meant to summarise."""
    by_task = {}
    for r in records:
        if r["run_mode"] != "free":
            continue
        by_task.setdefault(r["task_id"], {})[r["step"]] = r["args_correct_strict"]

    given_correct, given_wrong = [], []
    for steps in by_task.values():
        for t in sorted(steps):
            if (t + 1) in steps:
                if steps[t]:
                    given_correct.append(steps[t + 1])
                else:
                    given_wrong.append(steps[t + 1])
    gc = np.mean(given_correct) if given_correct else float("nan")
    gw = np.mean(given_wrong) if given_wrong else float("nan")
    return float(gc - gw), float(gc), float(gw), len(given_correct), len(given_wrong)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_stats = {}
    p_rows, f_rows, succ_rows, tri_rows, group, names = [], [], [], [], [], []
    delta_results = {}
    L_ci = {}
    extra = {}
    recov, per_step = {}, {}

    for cfg in DEFAULT_SUITE:
        print(f"running {cfg.name} ...")
        records = run_policy(cfg)
        path = f"{OUT_DIR}/{RUN_TAG}_{cfg.name}.jsonl"
        with open(path, "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

        loaded = load_records(path)
        stats = aggregate_by_depth(loaded, DEPTHS)
        all_stats[cfg.name] = stats

        p, f_syn, filled_flags = stats_to_arrays(stats)
        filled = filled_flags["any"]
        # exact successes/trials from the free-run mean and count (mean is k/n)
        succ = np.array([round(s.g_t * s.n_g) if s.n_g else 0 for s in stats])
        tri = np.array([s.n_g for s in stats])

        p_rows.append(p); f_rows.append(f_syn)
        succ_rows.append(succ); tri_rows.append(tri)
        group.append(FAMILY_INDEX[cfg.name]); names.append(cfg.name)

        recov[cfg.name] = measure_recovery(loaded)
        per_step[cfg.name] = [st.__dict__ for st in aggregate_by_step(loaded)]
        d1, gc, gw, n_gc, n_gw = delta_1(loaded)
        delta_results[cfg.name] = {"delta_1": d1, "p_next_given_correct": gc,
                                   "p_next_given_wrong": gw,
                                   "n_given_correct": n_gc, "n_given_wrong": n_gw}

        # bootstrap CI for L_t at each depth (task-level resampling)
        L_ci[cfg.name] = {}
        for d in DEPTHS:
            Lp, lo, hi = bootstrap_L_ci(loaded, d, n_boot=1000, seed=13)
            L_ci[cfg.name][str(d)] = {"L": Lp, "lo": lo, "hi": hi}

        # extra per-depth diagnostics now available from the richer records
        extra[cfg.name] = [{"depth": s.depth, "f_syn": s.f_syn,
                            "n_fresh_errors": s.n_fresh_errors,
                            "selection_acc": s.selection_acc,
                            "selection_gold_acc": s.selection_gold_acc,
                            "parse_fail_rate": s.parse_fail_rate,
                            "stalled_rate": s.stalled_rate} for s in stats]

        if any(filled):
            print(f"  note: {cfg.name} had {int(sum(filled))} depth bin(s) "
                 f"needing NaN fill (low sample at some depth)")

    p = np.array(p_rows); f_syn = np.array(f_rows)
    successes = np.array(succ_rows); trials = np.array(tri_rows)
    group = np.array(group)

    # Recovery rates measured off labelled transitions, averaged within family
    # and fed in as informative prior centres. Methodology Section 3 states the
    # measured rates "enter as informative priors"; before this they did not.
    G = int(group.max()) + 1

    def fam_mean(key):
        out = []
        for g in range(G):
            vals = [recov[n][key] for i, n in enumerate(names)
                    if group[i] == g and not np.isnan(recov[n][key])]
            out.append(float(np.mean(vals)) if vals else float("nan"))
        return out

    prior_rs, prior_rm = fam_mean("r_syn_chain"), fam_mean("r_sem_chain")
    print(f"measured recovery -> prior centres: r_syn={prior_rs} r_sem={prior_rm}")

    print("\nfitting hierarchical model ...")
    idata = build_and_sample(p, f_syn, successes, trials, group,
                             draws=1500, tune=1500, chains=4, seed=7,
                             target_accept=0.97,
                             prior_r_syn=prior_rs, prior_r_sem=prior_rm)

    # persist everything the plotting script needs
    with open(f"{OUT_DIR}/{RUN_TAG}_idata.pkl", "wb") as fh:
        pickle.dump(idata, fh)
    with open(f"{OUT_DIR}/{RUN_TAG}_meta.json", "w") as fh:
        json.dump({"names": names, "group": group.tolist(), "depths": DEPTHS,
                  "p": p.tolist(), "f_syn": f_syn.tolist(),
                  "successes": successes.tolist(), "trials": trials.tolist(),
                  "delta_1": delta_results,
                  "L_ci": L_ci, "per_depth_extra": extra,
                  "measured_recovery": recov, "per_step": per_step,
                  "priors_used": {"r_syn": prior_rs, "r_sem": prior_rm},
                  "config": [c.__dict__ for c in DEFAULT_SUITE]}, fh, indent=2)

    print(f"\nsaved trace -> {OUT_DIR}/{RUN_TAG}_idata.pkl")
    print(f"saved meta  -> {OUT_DIR}/{RUN_TAG}_meta.json")
    print("\ndone.")


if __name__ == "__main__":
    main()
