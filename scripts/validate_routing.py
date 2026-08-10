"""Validate the estimator on ROUTING tasks against known ground truth.

Why this exists: the official validation run (run_full_analysis.py) used the
LINEAR task variant only, but the real experiment's primary suite is routing --
linear tasks turned out to carry no propagation signal on real models. So the
estimator has never been checked against known truth on the task family that
carries every headline claim. This closes that gap.

Two questions:

  1. Does L_t recover the configured severity ordering on routing, as it did on
     linear? Routing differs structurally: a wrong value sends the agent down a
     wrong BRANCH, so selection errors can propagate, which linear cannot
     exercise.

  2. Does the error-type decomposition detect an error mix that varies along the
     chain (H4)? The linear suite could not test H4: every policy held both the
     syntactic and the selection share constant, so there was nothing for the
     decomposition to find, and it could only ever confirm "flat in, flat out".
     Here one policy has a rising syntactic share and another a falling
     SELECTION share -- the latter being H4's actual claim, selection-dominated
     early and argument-dominated later. Two flat controls must show no trend.

     Note that this is measured by STEP INDEX, not by task depth: pooling all
     steps of a depth-8 task into one bin averages away the very trend H4 is
     about, and was measured to recover only 30-49% of a configured span where
     per-step resolution recovers 68-91%.
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np

import argparse

from tur.tasks.dag import generate_routing_suite
from tur.harness.executor import FeedbackMode
from tur.harness.runner import run_free, run_teacher_forced, MockBackend
from tur.harness.sim_policy import SimPolicy, SimPolicyConfig
from tur.analysis.aggregate import (load_records, aggregate_by_depth,
                                    stats_to_arrays, bootstrap_L_ci,
                                    aggregate_by_step, measure_recovery)
from tur.model.hierarchical import build_and_sample, identifiability

DEPTHS = [1, 2, 4, 6, 8]
PER_DEPTH = 200
SEEDS = 2
OUT_DIR = "data/results"
RUN_TAG = "routingval"
ARG_SHIFT = 0      # set by --arg-shift; 0 = copy-argument, non-zero = transformed


class DepthVaryingMixPolicy(SimPolicy):
    """SimPolicy whose error mix is a function of position in the chain.

    H4 predicts the mix shifts with depth: selection-dominated at the first step,
    argument-dominated deeper in. Every policy in the default suite holds both
    shares constant, so the H4 test in the linear validation run was vacuous by
    construction -- it could only ever confirm "flat in, flat out".

    Two shares can be given a trend, each moving linearly from its value at step
    0 to its value at step 7:

      syntax  : the syntactic share of all fresh errors
      select  : the selection share of the SEMANTIC (executing) errors, which is
                the quantity H4 actually names
    """

    def __init__(self, cfg: SimPolicyConfig, syntax_trend=None,
                 select_trend=None):
        super().__init__(cfg)
        self.syntax_trend = syntax_trend
        self.select_trend = select_trend

    @staticmethod
    def _at(trend, step: int) -> float:
        lo, hi = trend
        return lo + (hi - lo) * min(1.0, step / 7.0)

    def __call__(self, task, step: int, ref: int, attempt: int):
        # Mutate the configured shares for this step, then defer to the parent so
        # every other mechanism (recovery, poisoning, retry) stays identical.
        if self.syntax_trend:
            self.cfg.syntax_share = self._at(self.syntax_trend, step)
        if self.select_trend:
            self.cfg.selection_share = self._at(self.select_trend, step)
        return super().__call__(task, step, ref, attempt)


# Two flat controls, then two policies with known trends. `falling-sel` is the
# direct H4 shape: selection-dominated early, argument-dominated later, with the
# syntactic share held flat so the selection/argument movement is not confounded
# with a change in how many errors execute at all.
SUITE = [
    ("flat-mix", SimPolicyConfig("flat-mix", p0=0.78, p_slope=0.020, pi=0.70,
                                 syntax_share=0.50, r_syn=0.35, r_sem=0.05,
                                 seed=11, selection_share=0.40), None, None),
    ("flat-strong", SimPolicyConfig("flat-strong", p0=0.95, p_slope=0.006, pi=0.22,
                                    syntax_share=0.50, r_syn=0.75, r_sem=0.30,
                                    seed=12, selection_share=0.40), None, None),
    ("rising-syn", SimPolicyConfig("rising-syn", p0=0.88, p_slope=0.012, pi=0.45,
                                   syntax_share=0.50, r_syn=0.55, r_sem=0.15,
                                   seed=13, selection_share=0.40),
     (0.20, 0.80), None),
    ("falling-sel", SimPolicyConfig("falling-sel", p0=0.86, p_slope=0.013, pi=0.40,
                                    syntax_share=0.50, r_syn=0.50, r_sem=0.12,
                                    seed=14, selection_share=0.40),
     None, (0.80, 0.15)),
]
FAMILY = {"flat-mix": 0, "flat-strong": 0, "rising-syn": 1, "falling-sel": 1}


def run_policy(cfg: SimPolicyConfig, syn_trend, sel_trend) -> list[dict]:
    records = []
    for seed in range(SEEDS):
        c = SimPolicyConfig(**{**cfg.__dict__, "seed": cfg.seed * 97 + seed})
        policy = (DepthVaryingMixPolicy(c, syn_trend, sel_trend)
                  if (syn_trend or sel_trend) else SimPolicy(c))
        backend = MockBackend(policy, model=cfg.name)
        suite = generate_routing_suite(DEPTHS, PER_DEPTH, distractor_level=1,
                                       base_seed=seed * 31 + 1000,
                                       arg_shift=ARG_SHIFT)
        for task in suite:
            # see run_full_analysis for why the run mode must be signalled
            policy.stateless = False
            f = run_free(task, backend, "uniform", FeedbackMode.STRUCTURED, 1)
            policy.stateless = True
            t = run_teacher_forced(task, backend, "uniform",
                                   FeedbackMode.STRUCTURED, 1)
            policy.stateless = False
            records += [r.__dict__ for r in f] + [r.__dict__ for r in t]
    return records


def delta_1(records):
    by_task = {}
    for r in records:
        if r["run_mode"] != "free" or r.get("backend_error", False):
            continue
        by_task.setdefault(r["task_id"], {})[r["step"]] = r["args_correct_strict"]
    gc, gw = [], []
    for steps in by_task.values():
        for t in sorted(steps):
            if (t + 1) in steps:
                (gc if steps[t] else gw).append(steps[t + 1])
    a = float(np.mean(gc)) if gc else float("nan")
    b = float(np.mean(gw)) if gw else float("nan")
    return {"delta_1": a - b, "p_next_given_correct": a,
            "p_next_given_wrong": b, "n_given_correct": len(gc),
            "n_given_wrong": len(gw)}


def main():
    global ARG_SHIFT, RUN_TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--arg-shift", type=int, default=0,
                    help="run the validation on the TRANSFORMED-argument variant, "
                         "where the required argument is (held + shift) mod M rather "
                         "than a copy. Verifies that H4 is detectable there against "
                         "known ground truth before any real-model budget is spent "
                         "on it -- the copy variant could not test H4 at all, so the "
                         "transform variant must be shown able to before it is used.")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    ARG_SHIFT = args.arg_shift
    if args.tag:
        RUN_TAG = args.tag
    elif ARG_SHIFT:
        RUN_TAG = f"transformval{ARG_SHIFT}"
    print(f"variant: routing, arg_shift={ARG_SHIFT}, tag={RUN_TAG}")
    os.makedirs(OUT_DIR, exist_ok=True)
    p_rows, f_rows, s_rows, t_rows, group, names = [], [], [], [], [], []
    L_ci, d1, per_depth, truth, filled_note = {}, {}, {}, {}, {}
    per_step, recov = {}, {}

    for name, cfg, syn_trend, sel_trend in SUITE:
        label = (', varying syntax' if syn_trend else
                 ', varying selection' if sel_trend else '')
        print(f"running {name} (routing{label}) ...")
        records = run_policy(cfg, syn_trend, sel_trend)
        path = f"{OUT_DIR}/{RUN_TAG}_{name}.jsonl"
        with open(path, "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

        loaded = load_records(path)
        stats = aggregate_by_depth(loaded, DEPTHS)
        p, f_syn, fl_flags = stats_to_arrays(stats)
        fl = fl_flags["any"]
        if any(fl):
            filled_note[name] = [DEPTHS[i] for i, x in enumerate(fl) if x]
            print(f"  note: substituted inputs at depth(s) {filled_note[name]}")

        p_rows.append(p); f_rows.append(f_syn)
        s_rows.append(np.array([round(s.g_t * s.n_g) if s.n_g else 0 for s in stats]))
        t_rows.append(np.array([s.n_g for s in stats]))
        group.append(FAMILY[name]); names.append(name)

        d1[name] = delta_1(loaded)
        recov[name] = measure_recovery(loaded)
        per_step[name] = [st.__dict__ for st in aggregate_by_step(loaded)]
        L_ci[name] = {str(d): dict(zip(("L", "lo", "hi"),
                                       bootstrap_L_ci(loaded, d, 1000, seed=13)))
                      for d in DEPTHS}
        per_depth[name] = [{"depth": s.depth, "p_t": s.p_t, "g_t": s.g_t,
                            "L_t": s.L_t, "f_syn": s.f_syn,
                            "n_fresh_errors": s.n_fresh_errors,
                            "selection_acc": s.selection_acc,
                            "selection_gold_acc": s.selection_gold_acc,
                            "parse_fail_rate": s.parse_fail_rate,
                            "stalled_rate": s.stalled_rate} for s in stats]
        truth[name] = {**cfg.__dict__,
                       "syntax_trend": ({"at_step0": syn_trend[0],
                                         "at_step7": syn_trend[1]}
                                        if syn_trend else "flat"),
                       "selection_trend": ({"at_step0": sel_trend[0],
                                            "at_step7": sel_trend[1]}
                                           if sel_trend else "flat")}

    # measured recovery rates -> informative prior centres, per family
    G = max(group) + 1

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
    idata = build_and_sample(np.array(p_rows), np.array(f_rows),
                             np.array(s_rows), np.array(t_rows),
                             np.array(group), draws=1500, tune=1500, chains=4,
                             seed=7, target_accept=0.97,
                             prior_r_syn=prior_rs, prior_r_sem=prior_rm)

    ident = {n: identifiability(idata, i) for i, n in enumerate(names)}
    with open(f"{OUT_DIR}/{RUN_TAG}_idata.pkl", "wb") as fh:
        pickle.dump(idata, fh)
    with open(f"{OUT_DIR}/{RUN_TAG}_meta.json", "w") as fh:
        json.dump({"names": names, "group": [int(g) for g in group],
                   "depths": DEPTHS, "task_variant": "routing",
                   "p": np.array(p_rows).tolist(),
                   "f_syn": np.array(f_rows).tolist(),
                   "successes": np.array(s_rows).tolist(),
                   "trials": np.array(t_rows).tolist(),
                   "delta_1": d1, "L_ci": L_ci, "per_depth_extra": per_depth,
                   "truth": truth, "identifiability": ident,
                   "measured_recovery": recov, "per_step": per_step,
                   "priors_used": {"r_syn": prior_rs, "r_sem": prior_rm},
                   "substituted_input_depths": filled_note,
                   "config": [c.__dict__ for _, c, _, _ in SUITE]}, fh, indent=2)
    print(f"\nsaved -> {OUT_DIR}/{RUN_TAG}_idata.pkl, {RUN_TAG}_meta.json")


if __name__ == "__main__":
    main()
